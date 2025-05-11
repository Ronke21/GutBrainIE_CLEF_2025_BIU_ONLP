python atlop_interface.py --data_dir ./data \
--transformer_type bert \
--model_name_or_path cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
--train_file train_annotated.json \
--save_path outputs2/ \
--load_path outputs2/ \
--load_checkpoint best.ckpt \
--dev_file dev.json \
--test_file predicted_entities_atlop_format.json \
--train_batch_size 4 \
--test_batch_size 4 \
--gradient_accumulation_steps 1 \
--num_labels 1 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 500.0 \
--seed 66 \
--num_class 18

mv outputs2/results.json ../../Predictions/RE/predicted_relations2.json
