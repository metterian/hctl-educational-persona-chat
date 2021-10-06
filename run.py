from models.context_detector import ContextSimilarity, LinguisticAcceptability
from models.chatbot import Chatbot
from models.config import args
from models import grammar

from typing import List
from dataclasses import dataclass, field
import torch
import json
import os
from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict, namedtuple


chatbot = Chatbot()
MRPC = ContextSimilarity()
CoLA = LinguisticAcceptability()




turn_flag = False
turn = 0
threshold_sim = 25
threshold_cor = 75


def divide_dialogue(history):
    human = history[::2]
    chatbot = history[1::2]
    return human, chatbot



while True:
    raw_text = input(">>> ")
    sentence = raw_text.strip()

    # predict next sentence
    message = chatbot.send_message(sentence)


    result_mrpc = MRPC.return_prediction(conv_history, gold_history)
    result_cola = CoLA.return_prediction(conv_history, gold_history)

    result_spell = grammar.correct(sentence)


    # When you got response from chatbot >> turn +1


    results = {
        "response": message,
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




