python ./train.py \
--dataset_path "data/situationchat_original.json" \
--dataset_cache "situationchat_original_dataset_cache_GPT2Tokenizer" \
--model_checkpoint "microsoft/DialoGPT-small" \
--gradient_accumulation_steps=4 \
--max_history=3 \
--n_epochs=2 \
--num_candidates=4 \
--personality_permutations=4 \
--train_batch_size=8 \
--valid_batch_size=8

