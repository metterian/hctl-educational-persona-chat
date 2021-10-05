from models.AFL import *
from models.chatbot import Chatbot
from models.config import args
from models import grammar


import torch
import random
import torch
import json
import pickle
import os
from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict, namedtuple

with open("data/persona_history.json") as fp:
    history_json = json.load(fp)


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




if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)





chatbot = Chatbot(args)


MRPC = AFL(model_name = "bert-base-cased-finetuned-mrpc", task = "MRPC")
CoLA = AFL(model_name = "textattack/roberta-base-CoLA", task = "CoLA")
# Redundancy = AFL(mrpc_model, mrpc_tokenizer, "Redundancy")






turn_flag = False
turn = 0
threshold_sim = 25
threshold_cor = 75

while True:
    raw_text = input(">>> ")
    sentence = raw_text.strip()

    # predict next sentence
    result_conv = chatbot.message(sentence, personality)

    history.human.append(sentence)
    history.chatbot.append(result_conv)

    result_mrpc = MRPC.return_prediction(history, gold_history)
    result_cola = CoLA.return_prediction(history, gold_history)

    result_spell = grammer.correct(sentence)


    # When you got response from chatbot >> turn +1


    results = {
        "response": result_conv,
        "similarity": result_mrpc,
        "correct": result_cola,
        "persona": personality_decoded,
        "history" : [tokenizer.decode(line) for line in chatbot.history],
        "count": turn,
        "spell": result_spell if result_spell.lower() != sentence else ["nothing to change!"],
        "isChanged": AFL.changed_flag,
    }

    turn += 1

    if turn >= 2:
        if result_mrpc < threshold_sim or result_cola < threshold_cor:
            personality, utterance, gold_history = shuffle_inputs(personalities, utterances, history)
            chatbot.history = []
            turn = 0


    # if AFL.count >= 4:  ## 나중에 5턴
    #     CoLA_avg = CoLA.average()
    #     MRPC_avg = MRPC.average()

    #     if CoLA_avg > 70 and MRPC_avg > 65 and AFL.changed_flag == False:
    #         shuffle_idx = random.choice(range(len(personalities)))
    #         personality = personalities[shuffle_idx]
    #         history_original = history[shuffle_idx]
    #         history_original = [tokenizer.decode(line) for line in history_original]
    #         chatbot.personality = personality
    #         chatbot.hisory = []

    #         AFL.count = 0


    #     chatbot.history = []





    # pprint(results)  # 받아온 데이터를 다시 전송
    print(json.dumps(results, indent=4))




