python ./train.py \
--dataset_path "data/personachat_self_original.json" \
--dataset_cache "personachat_self_original_dataset_cache_GPT2Tokenizer" \
--model_checkpoint "microsoft/DialoGPT-small" \
--gradient_accumulation_steps=8 \
--max_history=2 \
--n_epochs=2 \
--num_candidates=4 \
--personality_permutations=1 \
--train_batch_size=4 \
--valid_batch_size=4
