#%%
import json
import pandas as pd
from tqdm import tqdm
with open('data/personachat_self_original.json') as fp:
    input_json = json.load(fp)

# %%
dataset = input_json['train']
# %%


sample = dataset[0]
personalities = set(["%".join(sorted(dialog["personality"])) for dataset in input_json.values() for dialog in dataset])
personalities = [personality.split('%') for personality in personalities]

# %%
history = {}
for personality in tqdm(personalities):
    for dialogue in dataset:
        if sorted(dialogue['personality']) == sorted(personality):
            history["%".join(map(lambda x: x.strip(), personality))] = dialogue['utterances'][-1]['history']



# %%
with open('data/persona_history.json', 'w+') as fp:
    json.dump(history, fp, indent=4)
# %%
