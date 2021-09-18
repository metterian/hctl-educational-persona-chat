#%%
import json
from numba import jit
from easydict import EasyDict as edict
from tqdm import tqdm
# %%
with open('data/personachat_self_original.json', encoding='utf-8-sig') as fp:
    input_json = json.load(fp)
train = input_json['train']
# %%



def make_dict(dataset):
    data = {}
    for dataset in tqdm(dataset.values()):
        for dialog in dataset:
            persona = sorted(dialog["personality"])
            persona = map(lambda line: line.strip(), persona)
            persona = "%".join(persona)
            data[persona] = {
                'candidates' : dialog["utterances"][-1]['candidates'],
                'history' :dialog["utterances"][-1]["history"]
            }
    return data

data = make_dict(input_json)







# %%
