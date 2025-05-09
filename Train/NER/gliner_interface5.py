import json
from gliner import GLiNER

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import os

from importlib.metadata import version
version('GLiNER')

# --- START OF ADDED PREPROCESSING FUNCTIONS ---

def preprocess_data_item_tokens(item):
    """
    Preprocesses the 'tokenized_text' field in a data item by lowercasing each token.
    Assumes 'tokenized_text' is a list of strings (tokens).
    """
    if "tokenized_text" in item and isinstance(item["tokenized_text"], list):
        # Create a new list of lowercased tokens
        item["tokenized_text"] = [token.lower() for token in item["tokenized_text"]]
    # Entity labels are already converted to lowercase later in the script.
    return item

def preprocess_raw_text(text_input):
    """
    Preprocesses a raw text string by converting to lowercase and normalizing whitespace.
    """
    if isinstance(text_input, str):
        # Lowercase
        processed_text = text_input.lower()
        # Normalize whitespace (remove extra spaces, strip leading/trailing)
        processed_text = " ".join(processed_text.split())
        return processed_text
    return text_input # Return as is if not a string (e.g., None)

# --- END OF ADDED PREPROCESSING FUNCTIONS ---

# Set the GLiNER model to be used (from HuggingFace)
model = GLiNER.from_pretrained("urchade/gliner_large_bio-v0.2")
model_name = "gliner_large_bio-v0.2"

# Define the confidence threshold to be used in evaluation
THRESHOLD = 0.9

# Define whether the code should be used for fine-tuning
finetune_model = True
# Define whether predictions should be generated (assuming this is defined elsewhere or set as needed)
generate_predictions = True # Placeholder: Ensure this is set according to your workflow

# Define the path to articles for which the final trained will generate predicted entities
PATH_ARTICLES = "../../Test_Data/articles_test.json"
PATH_OUTPUT_NER_PREDICTIONS = f"../../Predictions/NER/predicted_entities_{model_name}_T{str(THRESHOLD*100)}.json"


print('## LOADING TRAINING DATA ##')
PATH_PLATINUM_TRAIN = "data/train_platinum.json"
PATH_GOLD_TRAIN = "data/train_gold.json"
PATH_SILVER_TRAIN = "data/train_silver.json"
PATH_BRONZE_TRAIN = "data/train_bronze.json"
PATH_DEV = "data/dev.json"

with open(PATH_PLATINUM_TRAIN, 'r', encoding='utf-8') as file:
    train_platinum = json.load(file)

with open(PATH_GOLD_TRAIN, 'r', encoding='utf-8') as file:
    train_gold = json.load(file)

with open(PATH_SILVER_TRAIN, 'r', encoding='utf-8') as file:
    train_silver = json.load(file)

with open(PATH_BRONZE_TRAIN, 'r', encoding='utf-8') as file:
    train_bronze = json.load(file)

with open(PATH_DEV, 'r', encoding='utf-8') as file:
    eval_data_loaded_from_file = json.load(file) # Renamed to avoid immediate reassignment confusion

# --- APPLYING PREPROCESSING TO LOADED DATA ---
print('## PREPROCESSING LOADED TRAINING AND EVALUATION DATA ##')
train_platinum = [preprocess_data_item_tokens(d.copy()) for d in train_platinum]
train_gold = [preprocess_data_item_tokens(d.copy()) for d in train_gold]
train_silver = [preprocess_data_item_tokens(d.copy()) for d in train_silver]
train_bronze = [preprocess_data_item_tokens(d.copy()) for d in train_bronze] # Preprocess bronze data as well
eval_data_loaded_from_file = [preprocess_data_item_tokens(d.copy()) for d in eval_data_loaded_from_file]
# --- END OF PREPROCESSING LOADED DATA ---

# Set the data to be used for training
# Here we used the platinum, gold, and silver sets
train_data = train_platinum + train_gold + train_silver

# converting entities-level train data to token-level
new_data = []
for d in train_data: # train_data now contains preprocessed items
    new_ner = []
    for s, f, c in d["ner"]:
        for i in range(s, f + 1):
            # labels are intended to be lower-case
            new_ner.append((i, i, c.lower()))
    new_d = {
        "tokenized_text": d["tokenized_text"], # This is the preprocessed (lowercased) list of tokens
        "ner": new_ner,
    }
    new_data.append(new_d)
train_data = new_data

# converting entities-level eval data to token-level
new_data_eval = []
# Use the preprocessed eval_data_loaded_from_file for conversion
for d in eval_data_loaded_from_file:
    new_ner = []
    for s, f, c in d["ner"]:
        for i in range(s, f + 1):
            # labels are intended to be lower-case
            new_ner.append((i, i, c.lower()))
    new_d = {
        "tokenized_text": d["tokenized_text"], # This is the preprocessed (lowercased) list of tokens
        "ner": new_ner,
    }
    new_data_eval.append(new_d)
eval_data_samples = new_data_eval # This list of samples will be used in the eval_data dictionary

from types import SimpleNamespace

# Define the hyperparameters in a config variable
config = SimpleNamespace(
    num_steps=3000, # regulate number train, eval steps depending on the data size
    eval_every=200,

    train_batch_size=8, # regulate batch size depending on GPU memory available.

    max_len=384, # maximum sentence length. 2048 for NuNerZero_long_context, 384 for rest

    save_directory="logs", # log dir
    device='cuda' if torch.cuda.is_available() else 'cpu', #'cuda', # training device - cpu or cuda

    warmup_ratio=0.1, # Other parameters
    lr_encoder=1e-5,
    lr_others=5e-5,
    freeze_token_rep=False,

    max_types=15,
    shuffle_types=True,
    random_drop=True,
    max_neg_type_ratio=1,
)

# modify this to your own test data!
# don't forget to do the same preprocessing as for the train data:
# * converting entities-level data to token-level data
# * making entity_types lower-cased!!!
# * (Implicitly now) making tokenized_text in samples lower-cased
eval_data_dict_for_model = { # Renamed to distinguish from the list 'eval_data_samples'
    "entity_types": [
        "anatomical location",
        "animal",
        "biomedical technique",
        "bacteria",
        "chemical",
        "dietary supplement",
        "ddf",
        "drug",
        "food",
        "gene",
        "human",
        "microbiome",
        "statistical technique",
    ],
    "samples": eval_data_samples # Use the preprocessed and token-level converted samples
}

print('## DEFINING TRAINING FUNCTION ##')
def train(model, config, train_data, eval_data_param=None): # renamed eval_data to eval_data_param for clarity
    model = model.to(config.device)

    # Set sampling parameters from config
    model.set_sampling_params(
        max_types=config.max_types,
        shuffle_types=config.shuffle_types,
        random_drop=config.random_drop,
        max_neg_type_ratio=config.max_neg_type_ratio,
        max_len=config.max_len
    )

    model.train()

    # Initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=config.train_batch_size, shuffle=True)

    # Optimizer
    optimizer = model.get_optimizer(config.lr_encoder, config.lr_others, config.freeze_token_rep)

    pbar = tqdm(range(config.num_steps))

    if config.warmup_ratio < 1:
        num_warmup_steps = int(config.num_steps * config.warmup_ratio)
    else:
        num_warmup_steps = int(config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config.num_steps
    )

    iter_train_loader = iter(train_loader)

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(config.device)

        loss = model(x)  # Forward pass

        # Check if loss is nan
        if torch.isnan(loss):
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        pbar.set_description(description)

        if (step + 1) % config.eval_every == 0:
            model.eval()
            if eval_data_param is not None:
                results, f1 = model.evaluate(eval_data_param["samples"], flat_ner=True, threshold=THRESHOLD, batch_size=32,
                                             entity_types=eval_data_param["entity_types"])
                print(f"Step={step}\n{results}")

            if not os.path.exists(config.save_directory):
                os.makedirs(config.save_directory)

            model.save_pretrained(f"{config.save_directory}/finetuned_{step}")
            model.train()

if finetune_model:
    print('## LAUNCHING TRAINING ##')
    train(model, config, train_data, eval_data_dict_for_model) # Pass the correctly structured eval data

    print('## SAVING TRAINED MODEL ##')
    output_path = f"outputs/{model_name}_finetuned_T{str(THRESHOLD*100)}"
    model.save_pretrained(output_path)
    os.system(f'cp {output_path}/gliner_config.json {output_path}/config.json')
    md = GLiNER.from_pretrained(output_path, local_files_only=True)

if generate_predictions:
    output_path = f"outputs/{model_name}_finetuned_T{str(THRESHOLD*100)}"
    # Ensure the model is loaded if not fine-tuned in the same run
    if not finetune_model or 'md' not in locals():
        print(f"## LOADING PRE-TRAINED MODEL {output_path} (for predictions) ##")
        if os.path.exists(output_path):
            md = GLiNER.from_pretrained(output_path, local_files_only=True)
        else:
            print(f"Model path {output_path} not found. Using the initially loaded model for predictions.")
            md = model # Fallback to the model loaded at the script start (might be base or fine-tuned if finetune_model=True)
    else:
        print(f"## USING THE RECENTLY FINE-TUNED MODEL FROM {output_path} (for predictions) ##")


    print(f"## GENERATING NER PREDICTIONS FOR {PATH_ARTICLES}")
    with open(PATH_ARTICLES, 'r', encoding='utf-8') as file:
        articles = json.load(file)

    print(f"len(articles): {len(articles)}")
    # Use entity_types from the eval_data_dict_for_model for consistency
    entity_labels = eval_data_dict_for_model['entity_types']


    predictions = {}

    for pmid, content in tqdm(articles.items(), total=len(articles), desc="Predicting entities..."):
        # --- APPLYING PREPROCESSING TO TITLE AND ABSTRACT ---
        title_text = preprocess_raw_text(content.get('title', '')) # Use .get for safety
        abstract_text = preprocess_raw_text(content.get('abstract', '')) # Use .get for safety
        # --- END OF PREPROCESSING ---

        # Predict entities using preprocessed text
        title_entities = md.predict_entities(title_text, entity_labels, threshold=THRESHOLD, flat_ner=True, multi_label=False)
        abstract_entities = md.predict_entities(abstract_text, entity_labels, threshold=THRESHOLD, flat_ner=True, multi_label=False)

        # Adjust indices for predicted entities in the abstract
        # This must use the length of the *preprocessed* title_text
        # Concatenation usually assumes a space: "title. abstract"
        # If title_text is empty, len(title_text) is 0. If not empty, add 1 for the space.
        offset = len(title_text) + 1 if title_text else 0
        for entity in abstract_entities:
            entity['start'] += offset
            entity['end'] += offset

        unique_entities = []
        seen_entities = set()

        for entity in title_entities:
            # key uses entity['text'] which is from the preprocessed title_text
            key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
            if key not in seen_entities:
                tmp_entity = {
                    'start_idx': entity['start'],
                    'end_idx': entity['end'],
                    'tag': 't',
                    'text_span': entity['text'], # This text will be from the preprocessed input
                    'entity_label': entity['label'],
                    'score': entity['score']
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        for entity in abstract_entities:
            # key uses entity['text'] which is from the preprocessed abstract_text
            key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
            if key not in seen_entities:
                tmp_entity = {
                    'start_idx': entity['start'],
                    'end_idx': entity['end'],
                    'tag': 'a',
                    'text_span': entity['text'], # This text will be from the preprocessed input
                    'entity_label': entity['label'],
                    'score': entity['score']
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        predictions[pmid] = unique_entities
        articles[pmid]['pred_entities'] = unique_entities

    def default_serializer(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f'Type {type(obj)} not serializable')

    with open(PATH_OUTPUT_NER_PREDICTIONS, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"## Predictions have been exported in JSON format to '/{PATH_OUTPUT_NER_PREDICTIONS}' ##")