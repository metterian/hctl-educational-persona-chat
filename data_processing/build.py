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
#%%
data_path = os.path.join(os.path.pardir, 'data/translation_eng_kor.xlsx')
trans = pd.read_excel(data_path, engine='openpyxl')
nlp = spacy.load("en_core_web_trf")
#%%

def space_before_eos(sentence: str, tokenizer = nlp):
    table = str.maketrans({key: " {0} ".format(key) for key in string.punctuation})
    doc = tokenizer(sentence)
    tokens = [token.text.translate(table) if token.pos_ == 'PUNCT' else token.text for token in doc ]
    sentence = " ".join(tokens)

    return sentence

#%%
trans['번역문'] = trans['번역문'].apply(space_before_eos)
trans['번역문'] = trans['번역문'].str.lower()

translations = trans['번역문'].to_list()

#%%
situ_count = trans.groupby('상황').count().sort_values(by='소분류', ascending=False)
situ_count['대화수'] = situ_count['번역문'] / 4


situations = situ_count.head(15).index.tolist() # 대화수 상위 15개의 상황
# %%
situ_persona = {
    '직장에서의 일상 대화': [
        'I work at a company.',
        'I work with my boss, colleagues, and subordinates.',
        'I have a business conversation.'
        ],
    '찬성 및 반대': [
        "I'm in a meeting",
        "Pros and cons are divided.",
        'I have a business conversation.',
        ],
    '취직 면접 상황': [
        "I'm unemployed.",
        "I'm looking for a job.",
        "I'm having an interview.",
        'I have a business conversation.',
        ],
    # '회의 관련': [],
    # '의견 교환하기': [],
    '학교생활': [
        'i just took a quiz',
        'i have a test.'
        'i have a family road trip planned for the western u.s',
        'i attend the class.'
        ],
    '원하는 스타일에 대해 점원 or 친구와 대화 시술 전/시술 시/시술 후 대화': [
        'I came to the hair salon to do my hair.',
        'i have a friend.',
        'I want to dye my hair.'],
    '제안 및 협상하기': [
        "i am worried that our company's mobile phone sales are declining",
        "I have a business conversation.",
        "I'm negotiating with business customers.",
        "I propose a contract."
        ],
    'CS/고객 상담': [
        "I'm a customer counselor.",
        "I respond to customer requests.",
        "The phone number is 000-0000-0000.",
        "the custormer ordered product"
        ],
    '마케팅/홍보': [
        "The project proposal is due tomorrow,",
        "A new product was released last week.",
        "I want to negotiate the price.",
        "I want to promote the released product."
        ],
    '증상을 묻고 답하는 상황': [
        'i have a headache.'
        'I need medical consultation.',
        "I'm seeing a doctor."],
    '인사관리': [
        'I will only be able to work till the end of this week',
        'I will have to take an early leave',
        'About the sign up for voluntary resignation being conducted this time',
        'I am not confident with sales'],
    # '경영/사무': [],
    'IT/컴퓨터 (수리, 소프트웨어 설치 등)': [
        "I think the internet speed has slowed down lately.",
        "the IT department takes care of all things related to computers.",
        "Due to the low specs of the computer, our work efficiency decreased."
        ],
    '음식 먹고 맛 평가하는 상황' : [
        "I'm eating food.",
        "I am evaluating after eating food.",
        "let's order then. I am starving."

    ]
}#%%
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
