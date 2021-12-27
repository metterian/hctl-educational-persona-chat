#%%
import pandas as pd
import json
import sys
import os
import time
import re
import random
import string
import spacy
from tqdm import tqdm, tqdm_pandas
from setproctitle import setproctitle

#%%
# environment setting
tqdm.pandas()
setproctitle('joon_persona')
spacy.prefer_gpu(4) # set gpu setting
nlp = spacy.load("en_core_web_trf", disable = ["tok2vec", "parser", "attribute_ruler", "lemmatizer"])


# make absolute path from project path
def get_absolute_path(relative_path: str) -> str:
    return os.path.join(os.path.pardir, relative_path)

#%%
# load dataset
data_path = get_absolute_path('data/translation_eng_kor.xlsx')
trans = pd.read_excel(data_path, engine='openpyxl')
#%%
# preprocess the text
def space_before_eos(sentence: str, tokenizer = nlp):
    table = str.maketrans({key: " {0}".format(key) for key in string.punctuation})
    doc = tokenizer(sentence)
    tokens = [token.text.translate(table) if token.pos_ == 'PUNCT' else token.text for token in doc ]
    sentence = " ".join(tokens)

    return sentence

#%%
trans['번역문'] = trans['번역문'].progress_apply(space_before_eos)
# trans['번역문'] = trans['번역문'].apply(space_before_eos)
trans['번역문'] = trans['번역문'].str.lower()
#%%
data_path = get_absolute_path('data/translation_eng_kor_eos.xlsx')
trans = pd.read_excel(data_path, engine='openpyxl')
translations = trans['번역문'].to_list()
#%%
# get top 15 situations
situ_count = trans.groupby('상황').count().sort_values(by='소분류', ascending=False)
situ_count['대화수'] = situ_count['번역문'] / 4
situations = situ_count.head(15).index.tolist() # 대화수 상위 15개의 상황
# %%
with open('')

situ_persona =
#%%
def sample_candidate(candidates = translations, num = 18):
    return random.sample(candidates, num)
# %%
data = []

for situation, persona in situ_persona.items():
    filtered_situation = trans[trans['상황'].str.contains(situation)]
    situ_convs = filtered_situation.groupby('Set Nr.')['번역문'].apply(list).tolist()

    for conversation in situ_convs:
        utterances = []
        conversation.append(random.sample(situ_convs, 1)[0][-1])
        for i in range(0, len(conversation)-1, 2):
            candidates = sample_candidate() + [conversation[i+1]]
            utterances.append({
                "candidates" : candidates ,
                "history" : conversation[:i+1]
            })
        dialogue_entry = {
            "personality": persona,
            "utterances": utterances
                }
        data.append(dialogue_entry)

# %%
with open('test_dataset.json', 'w+') as fp:
    json.dump(data, fp, indent=4)
# %%
