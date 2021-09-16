from AFL import AFL
from ChatBot import ChatBot
import grammer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from easydict import EasyDict as edict
import random
import torch
from transformers import (
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from train import add_special_tokens_
from utils import get_dataset, download_pretrained_model
import json
import pickle
import os


with open("data/persona_history.json") as fp:
    history_json = json.load(fp)

# pickle load
def pickle_load(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_save(path: str, data) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


args = edict(
    {
        "model": "gpt2",
        "dataset_path": "./data/personachat_self_original.json",
        "dataset_cache": "./cache.tar.gz_GPT2Tokenizer",
        "persona_cache": "cache/persona_cache",
        "history_cache": "cache/history_cache",
        "model_checkpoint": "./runs/train_6cans3",
        "temperature": 1.9,
        "top_k": 180,
        "top_p": 0.1,
        "max_history": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "no_sample": True,
        "max_length": 20,
        "min_length": 1,
        "seed": 0,
    }
)


if args.model_checkpoint == "":
    if args.model == "gpt2":
        raise ValueError(
            "Interacting with GPT2 requires passing a finetuned model_checkpoint"
        )
    else:
        args.model_checkpoint = download_pretrained_model()
#%%

if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # logger.info("Get pretrained model and tokenizer")
tokenizer_class, model_class = (
    (GPT2Tokenizer, GPT2LMHeadModel)
    if args.model == "gpt2"
    else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
)
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)
#%%
# logger.info("Sample a personality")
dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

if args.persona_cache and os.path.isfile(args.persona_cache):
    personalities = pickle_load(args.persona_cache)
else:
    personalities = [
        dialog["personality"] for dataset in dataset.values() for dialog in dataset
    ]
    pickle_save(path="./cache/persona_cache", data=personalities)


if args.history_cache and os.path.isfile(args.history_cache):
    history = pickle_load(args.history_cache)
else:
    history = [
        dialog["utterances"][-1]["history"]
        for dataset in dataset.values()
        for dialog in dataset
    ]
    pickle_save(path="./cache/history_cache", data=history)


#%%
# idx = int(np.random.choice(len(input_tokens['history']), 1, replace=False))
shuffle_idx = random.choice(range(len(personalities)))
# personality = random.choice(input_tokens['personality'])
personality = personalities[shuffle_idx]
history_original = history[shuffle_idx]
history_original = [tokenizer.decode(line) for line in history_original]


print(f"PERSONA:{[tokenizer.decode(line) for line in personality]}")


# SPELL_API_KEY="6e93cb58ed3b4ad594947f95c0c32600"
# SPELL_params = {
#         'mkt':'en-us',
#         'mode':'proof'
#              }
# SPELL_headers = {
#         'Content-Type': 'application/x-www-form-urlencoded',
#         'Ocp-Apim-Subscription-Key': SPELL_API_KEY,
#         }

mrpc_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
mrpc_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased-finetuned-mrpc"
)

cola_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-CoLA")
cola_model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-CoLA"
)

#%%


personality_decoded = [tokenizer.decode(line) for line in personality]

chatbot = ChatBot(args, tokenizer, model, personality)


MRPC = AFL(mrpc_model, mrpc_tokenizer, "MRPC")
CoLA = AFL(cola_model, cola_tokenizer, "CoLA")
Redundancy = AFL(mrpc_model, mrpc_tokenizer, "Redundancy")

# @app.route('/prediction', methods = ['POST'])
# def prediction():

# raw_text = input(">>> ")
raw_text = "hi"
sentence = raw_text.strip()  # json 데이터를 받아옴


history = edict({"chatbot": [], "human": []})


# sentence = request.get_json()#json 데이터를 받아옴
result_conv = chatbot.return_message(sample_json=sentence)

history.human.append(sentence)
history.chatbot.append(result_conv)

result_mrpc = MRPC.return_prediction(history, history_original)
result_cola = CoLA.return_prediction(history, history_original)

result_spell = grammer.correct(sentence)


result_redundancy = Redundancy.return_prediction(history, history_original)
print(result_redundancy)
AFL.count += 1
print(chatbot.history)

results = {
    "sentence": result_conv,
    "similarity": result_mrpc,
    "correct": result_cola,
    "contents": chatbot.contents,
    "count": AFL.count,
    "spell": result_spell if result_spell != sentence else ["nothing to change!"],
    "isChanged": AFL.changed_flag,
    "chapter": chatbot.chapter,
}

AFL.changed_flag = False

if AFL.count >= 4:  ## 나중에 5턴
    CoLA_avg = CoLA.average()
    MRPC_avg = MRPC.average()

    if CoLA_avg > 70 and MRPC_avg > 65 and AFL.changed_flag == False:
        shuffle_idx = random.choice(range(len(personalities)))
        personality = personalities[shuffle_idx]
        history_original = history[shuffle_idx]
        history_original = [tokenizer.decode(line) for line in history_original]
        chatbot.personality = personality
        AFL.count = 0
    elif result_redundancy > 70:
        print("너무 똑같아서 바꿈")
        shuffle_idx = random.choice(range(len(personalities)))
        personality = personalities[shuffle_idx]
        history_original = history[shuffle_idx]
        history_original = [tokenizer.decode(line) for line in history_original]
        chatbot.personality = personality
        Redundancy.answer = []
        AFL.count = 0

    chatbot.history = []


print(results)  # 받아온 데이터를 다시 전송


# @app.route('/first' , methods=['GET'])
# def first():
#     return jsonify({'chapter':chatbot.chapter})
# # def change_content():
# if __name__ == "__main__":
#     app.run(host='0.0.0.0')

# %%
