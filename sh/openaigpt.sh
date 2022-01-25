python ./train.py \
--dataset_path "data/situationchat_original.json" \
--dataset_cache "" \
--model_checkpoint "openai-gpt" \
--gradient_accumulation_steps=4 \
--max_history=3 \
--n_epochs=2 \
--num_candidates=4 \
--personality_permutations=4 \
--train_batch_size=16 \
--valid_batch_size=16

