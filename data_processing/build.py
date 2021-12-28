#%%
import pandas as pd
import json
import os
import random
import spacy
from tqdm import tqdm
from setproctitle import setproctitle

#%%
# environment setting
tqdm.pandas()
setproctitle("joon_persona")
spacy.prefer_gpu(4)  # set gpu setting

nlp = spacy.load(
    "en_core_web_sm",
)
#%%
# make absolute path from project path
def get_absolute_path(relative_path: str) -> str:
    return os.path.join(os.path.pardir, relative_path)


#%%
# load dataset
data_path = get_absolute_path("data/translation_eng_kor.xlsx")
trans = pd.read_excel(data_path, engine="openpyxl")
#%%

#%%
def check_punctuation(words):
    punctuation = [".", ","]
    return [word for word in words if word in punctuation]


# preprocess the text
def space_before_eos(sentence: str, tokenizer=nlp):
    table = str.maketrans({".": " .", ",": " ,"})
    sentence = sentence.lower().split()
    for i, word in enumerate(sentence):
        if check_punctuation(word):
            for token in tokenizer(word):
                if token.pos_ == "PUNCT":
                    sentence[i] = sentence[i].translate(table)
    return " ".join(sentence)


#%%
trans["번역문"] = trans["번역문"].progress_apply(space_before_eos)
# trans['번역문'] = trans['번역문'].apply(space_before_eos)
#%%
trans.to_excel(get_absolute_path("data/translation_eng_kor_eos.xlsx"))
#%%
# get top 15 situations
situ_count = trans.groupby("상황").count().sort_values(by="소분류", ascending=False)
situ_count["대화수"] = situ_count["번역문"] / 4
situations = situ_count.head(15).index.tolist()  # 대화수 상위 15개의 상황
#%%
# dataset reload
data_path = get_absolute_path("data/translation_eng_kor_eos.xlsx")
trans = pd.read_excel(data_path, engine="openpyxl")
translations = trans["번역문"].to_list()
# %%
# load situation labels
situation_label_path = get_absolute_path("data_processing/situation_label.json")
with open(situation_label_path) as fp:
    situation_labels = json.load(fp)

#%%
# space the punctuation in <eos>
for situation, description in situation_labels.items():
    description = list(map(space_before_eos, description))
    situation_labels[situation] = description

#%%
def sample_candidate(candidates=translations, num=18):
    return random.sample(candidates, num)


# %%
data = []

for situation_label, persona in situation_labels.items():
    situation = trans[trans["상황"].str.contains(situation_label)]
    top_situation = trans[trans["상황"].str.contains("마케팅/홍보")]["대분류"].iloc[0]
    candidates = trans[~trans["대분류"].str.contains(situation_label)]["번역문"].to_list()
    conversations = situation.groupby("Set Nr.")["번역문"].apply(list).tolist()

    for conversation in conversations:
        utterances = []
        for i in range(len(conversation) - 1):
            candidates = sample_candidate(candidates) + [conversation[i + 1]]
            utterances.append(
                {"candidates": candidates, "history": conversation[: i + 1]}
            )
        dialogue_entry = {"personality": persona, "utterances": utterances}
        data.append(dialogue_entry)

# %%
with open("test_dataset.json", "w+") as fp:
    json.dump(data, fp, indent=4)
# %%
