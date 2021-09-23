#%%
from AFL import AFL
from ChatBot import ChatBot
import grammer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from easydict import EasyDict as edict
import random
import torch
from transformers import GPT2LMHeadModel,GPT2Tokenizer
from train import add_special_tokens_
from utils import PERSONACHAT_URL, get_dataset, download_pretrained_model
import json
import pickle
import os
from pprint import pprint
from tqdm import tqdm
import numpy as np

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

def decode(tokens):
    return [tokenizer.decode(token) for token in tokens]


mrpc_models = [
    "bert-base-cased-finetuned-mrpc",
    "textattack/roberta-base-MRPC",
    "textattack/facebook-bart-large-MRPC",
    "textattack/xlnet-base-cased-MRPC"
    # "textattack/albert-base-v2-MRPC",
]

cola_models = [
    "textattack/facebook-bart-large-CoLA",
    "textattack/distilbert-base-uncased-CoLA",
    "textattack/bert-base-uncased-CoLA",
    "textattack/roberta-base-CoLA",
    # "textattack/albert-base-v2-CoLA"
]



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
        raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
    else:
        args.model_checkpoint = download_pretrained_model()


if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# Get pretrained model and tokenizer
tokenizer_class, model_class = GPT2Tokenizer, GPT2LMHeadModel
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

# mrpc_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
# mrpc_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

# cola_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
# cola_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")


# Get Dataset, Persoan, History
dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

if args.persona_cache and os.path.isfile(args.persona_cache):
    personalities = pickle_load(args.persona_cache)
else:
    personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    pickle_save(path="./cache/persona_cache", data=personalities)


if args.history_cache and os.path.isfile(args.history_cache):
    history = pickle_load(args.history_cache)
else:
    history = [ dialog["utterances"][-1]["history"] for dataset in dataset.values() for dialog in dataset ]
    pickle_save(path="./cache/history_cache", data=history)

utterances = [ dialog["utterances"] for dataset in dataset.values() for dialog in dataset ]


# Select the shuffled persona and history
shuffle_idx = random.choice(range(len(personalities)))
personality = personalities[shuffle_idx]
utterance = utterances[shuffle_idx]
gold_history = history[shuffle_idx]
gold_history = [tokenizer.decode(line) for line in gold_history]


personality_decoded = decode(personality)
print(f"PERSONA:{personality_decoded}")


chatbot = ChatBot(args, tokenizer, model)


MRPC = AFL(model_name = "bert-base-cased-finetuned-mrpc", task = "MRPC")
CoLA = AFL(model_name = "textattack/roberta-base-CoLA", task = "CoLA")
# Redundancy = AFL(mrpc_model, mrpc_tokenizer, "Redundancy")



# def

def shuffle_inputs(personalities: list, utterances: list, history: list):
    shuffle_idx = random.choice(range(len(personalities)))
    personality = personalities[shuffle_idx]
    utterance = utterances[shuffle_idx]
    gold_history = history[shuffle_idx]
    gold_history = [tokenizer.decode(line) for line in gold_history]

    return personality, utterance, gold_history

def pseudo_code(personalities, utterances, history, models):
    PERSONA_FLAG = False
    personality, utterance, gold_history = shuffle_inputs(personalities, utterances, history)
    history = edict({"chatbot": [], "human": []})
    turn = 0
    turning_point = random.randint(2, len(utterance)-1) # turn of inputting negative sample

    while True:
        sentence = "".join(decode(utterance[turn]['history'][-1]))
        next_answer = utterance[turn+1]['history'][-1]
        candidates = utterance[turn+1]['candidates']

        # pprint({"GOLD_ANSWER": decode(next_answer), "CANDIDATES": decode(candidates)})
        # predict next sentence
        result_conv = chatbot.return_message(sentence, personality)

        history.human.append(sentence)
        history.chatbot.append(result_conv)

        MRPC, CoLa = models
        result_mrpc = MRPC.return_prediction(history, gold_history)
        result_cola = CoLA.return_prediction(history, gold_history)
        turn += 1

        if turn >= turning_point:
            # set negative sample
            candidate = random.choice(candidates)
            while candidate == next_answer:
                candidate = random.shuffle(candidates)

            sentence = "".join(decode(candidate)) # false sentence
            result_conv = chatbot.return_message(sentence, personality)
            history.human.append(sentence)
            history.chatbot.append(result_conv)
            result_mrpc = MRPC.return_prediction(history, gold_history)
            result_cola = CoLA.return_prediction(history, gold_history)


            return result_mrpc, result_cola


def get_metrics(personalities, utterances, history, mrpc_model, cola_model):
    MRPC = AFL(model_name = mrpc_model, task = "MRPC")
    CoLA = AFL(model_name = cola_model, task = "CoLA")
    models = MRPC, CoLA
    model_names = mrpc_model, cola_model
    sim = []
    cor = []

    for _ in tqdm(range(1000)):
        similarity, correctness = pseudo_code(personalities, utterances, history, models)
        sim.append(similarity)
        cor.append(correctness)

    sim = np.array(sim)
    cor = np.array(cor)
    print("SIMLIARTY: ", np.mean(sim), "CORRECTNESS: ", np.mean(cor))

    mrpc_model = mrpc_model.replace('/', '-') if '/' in mrpc_model else mrpc_model
    cola_model = cola_model.replace('/', '-') if '/' in cola_model else mrpc_model

    model_names = mrpc_model, cola_model
    file_path = './afl_metrics_cola/'
    file_name = "_and_".join(model_names)
    with open(file_path+file_name+'.json', 'w+') as fp:
        json.dump({"SIMLIARTY": sim.tolist(), "CORRECTNESS": cor.tolist()},fp, indent=4)



#%%
mrpc_models = [
"bert-base-cased-finetuned-mrpc",
"textattack/roberta-base-MRPC",
# "textattack/facebook-bart-large-MRPC",
"textattack/xlnet-base-cased-MRPC",
"textattack/albert-base-v2-MRPC",
]

cola_models = [
    # "textattack/facebook-bart-large-CoLA",
    # "textattack/distilbert-base-uncased-CoLA",
    "textattack/bert-base-uncased-CoLA",
    "textattack/roberta-base-CoLA",
    "textattack/albert-base-v2-CoLA",
    "textattack/xlnet-base-cased-CoLA"
]

for mrpc_model in mrpc_models:
    for cola_model in cola_models:
        print("MODELS: ", mrpc_model, cola_model)
        get_metrics(personalities, utterances, history, mrpc_model, cola_model)



# %%
from datasets import load_dataset
from tqdm import tqdm
dataset = load_dataset('glue', 'cola')['test']


for cola_model in cola_models:
    CoLA = AFL(model_name = cola_model, task = "CoLA")
    results = [CoLA.return_prediction(line['sentence'], None )  for line in tqdm(dataset)]
    file_path = './afl_metrics_cola/'
    file_name = cola_model.replace('/','-') + '.json'

    with open(file_path+file_name, 'w+') as fp:
        json.dump({'corrrectness':results}, fp, indent=4)

# %%
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_metric, load_dataset
import numpy as np
from tqdm import tqdm
metric = load_metric("accuracy")
dataset = load_dataset('glue', 'cola', split='test')



model_name = "textattack/bert-base-uncased-CoLA"

def predict_cola(model_name, sentences):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    classes = [-1, 1]

    results = []
    for sentence in tqdm(sentences):
        paraphrase = tokenizer(sentence['sentence'], return_tensors="pt").to(device)
        paraphrase_classification_logits = model(**paraphrase)[0]
        paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1)[0].tolist()
        result = classes[np.argmax(paraphrase_results)]
        results.append(result)
    return results


for cola_model in cola_models:
    prediction = predict_cola(cola_model, dataset)
    metric_result = metric.compute(references=[1]*len(dataset), predictions=prediction)
    print(cola_model, ": ", metric_result)


# %%
