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

mrpc_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
mrpc_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

cola_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
cola_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")


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


MRPC = AFL(mrpc_model, mrpc_tokenizer, "MRPC")
CoLA = AFL(cola_model, cola_tokenizer, "CoLA")
# Redundancy = AFL(mrpc_model, mrpc_tokenizer, "Redundancy")



# def

def shuffle_inputs(personalities: list, utterances: list, history: list):
    shuffle_idx = random.choice(range(len(personalities)))
    personality = personalities[shuffle_idx]
    utterance = utterances[shuffle_idx]
    gold_history = history[shuffle_idx]
    gold_history = [tokenizer.decode(line) for line in gold_history]

    return personality, utterance, gold_history

def pseudo_code(personalities, utterances, history):
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

            # if threshold:
            #     PERSONA_FLAG = True

            return result_mrpc, result_cola




if __name__ == '__main__':
    sim = []
    cor = []

    for _ in tqdm(range(100)):
        similarity, correctness = pseudo_code(personalities, utterances, history)
        sim.append(similarity)
        cor.append(correctness)

    sim = np.array(sim)
    cor = np.array(cor)
    print("SIMLIARTY: ", np.mean(sim), "CORRECTNESS: ", np.mean(cor))

    with open('mrpc_cola_result.json') as fp:
        json.dump({"SIMLIARTY": sim, "CORRECTNESS": cor}, indent=4)
