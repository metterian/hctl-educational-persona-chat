#%%
import json
import pandas as pd



# %%
df = pd.read_excel('../data/translation_eng_kor(fixed).xlsx')
# %%
# %%
dial_count = df.groupby(['Set Nr.','상황'],as_index=False).size()

# %%
not_fixed_dial = dial_count[dial_count['size'] != 4]
# %%
not_fixed_dial.to_excel('unfixed_dialogue.xlsx')
# %%
# %%
dial_count = df.groupby(['상황'],as_index=False).size()
dial_count = dial_count.sort_values(by='size', ascending=False)
# %%
import json
with open('../data_processing/situation_label.json') as fp:
    data = json.load(fp)
# %%
data.keys()
# %%
from pathlib import Path
import json
from dataclasses import dataclass


@dataclass
class Dataset:
    '''
    To get data statics of SITUATION CHAT
        - Dialogues
        - Average Turns
        - Utterances
        - word count (sentence length)
    Args:
        - dataset: train or valid
    '''
    path: str

    def __post_init__(self):
        with open(self.path) as fp:
            self.data = json.load(fp)
        self.train = self.data['train']
        self.valid = self.data['valid']
        self.name = Path(self.path).stem


    def num_of_dialogue(self):
        '''Get number of persona'''
        return f"Train: {len(self.train)}, Valid: {len(self.valid)}"

    def num_of_utterance(self):
        '''Get number of utterance'''
        train_utt = sum([len(dialogue['utterances']) for dialogue in self.train])
        valid_utt = sum([len(dialogue['utterances']) for dialogue in self.valid])
        return f"Train: {train_utt}, Valid: {valid_utt}"

    def average_turns(self):
        '''Get average turns'''
        train_turns = sum([len(dialogue['utterances']) for dialogue in self.train]) / len(self.train)
        valid_turns = sum([len(dialogue['utterances']) for dialogue in self.valid]) / len(self.valid)
        return f"Train: {train_turns}, Valid: {valid_turns}"

    def count_words(self, dataset):
        word_count = 0
        for dialogue in dataset:
            for history in dialogue['utterances'][-1]['history']:
                word_count += len(history.split())
        return word_count

    def num_of_word(self):
        '''Get number of word'''
        return f"Train: {self.count_words(self.train)}, Valid: {self.count_words(self.valid)}"



#%%
situation_chat = Dataset('../data/situationchat_original.json')
# situation_chat_augmented = Dataset('../data/situationchat_original(augmented).json')
persona_chat = Dataset('../data/personachat_self_original.json')


datasets = [situation_chat, persona_chat]
#%%

for dataset in datasets:
    print(f"{dataset.name}")
    print(f"num_of_dialogue : {dataset.num_of_dialogue()}",
          f"num_of_utterance : {dataset.num_of_utterance()}",
          f"average_turns : {dataset.average_turns()}",
          # TODO: augmented dataset can not work number of word because of number of dialogues
          f"num_of_word: {dataset.num_of_word()}",
          sep='\n')
    print('-'*20)

#%%
import json
SITUATION_CHAT = '../data/situationchat_original.json'
PERSONA_CHAT = '../data/personachat_original.json'

with open(f'{PERSONA_CHAT}') as fp:
    data = json.load(fp)

# %%
train = data['train']
# %%
entry_count = len(train) * 4
print(f"Entry count: {entry_count}")
# %%
len(train[0]['utterances'])
# %%

entry_cnt = 0
for entry in train:
    turn_cnt = len(entry['utterances'])
    entry_cnt += turn_cnt

print(f"Entry count: {entry_cnt}")



# %%
