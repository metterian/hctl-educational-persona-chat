#%%
import json

with open('data/en_book_conversational.json', encoding='utf-8-sig') as fp:
    input_json = json.load(fp)

# %%
train = input_json['train']
# %%
sample_data = train[:3]

with open('data/sample_data.json','w+', encoding='utf8') as fp:
    json.dump(sample_data, fp, indent=4)


# %%
