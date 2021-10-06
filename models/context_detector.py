from dataclasses import dataclass, field
from typing import List, ClassVar
import torch
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class AFL:
    count = 0
    changed_flag=False

    def __init__(self, model_name, task):
        self.score = []
        self.task = task
        self.answer = []
        self.changed_flag = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)


    def return_prediction(self, history: list, history_sentences: list) -> dict:
        if self.task == "MRPC":
            sentence = history.human[-1] # recent human input
            scores = []

            for history_sentence in history_sentences:
                paraphrase = self.tokenizer.encode_plus(sentence, history_sentence, return_tensors="pt").to(self.device)
                paraphrase_classification_logits = self.model(**paraphrase)[0]
                paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
                scores.append(float(paraphrase_results[1])) # classes = ["not paraphrase", "is paraphrase"]


            scores = np.array(scores)
            score_max = np.max(scores)
            self.score.append(score_max * 100)
            return score_max * 100

        # if self.task == 'MRPC':
        #     human_score, chatbot_score = [], []

        #     human_sequence = history.human[-1]
        #     chatbot_sequence = history.chatbot[-1]

        #     for sequence in history_original:
        #         human_paraphrase = self.tokenizer.encode_plus(human_sequence, sequence, return_tensors="pt")
        #         chatbot_paraphrase = self.tokenizer.encode_plus(chatbot_sequence, sequence, return_tensors="pt")

        #         human_classification_logits = self.model(**human_paraphrase)[0]
        #         chatbot_classification_logits = self.model(**chatbot_paraphrase)[0]

        #         human_result = torch.softmax(human_classification_logits, dim=1).tolist()[0]
        #         chatbot_result= torch.softmax(chatbot_classification_logits, dim=1).tolist()[0]

        #         human_score.append(human_result)
        #         chatbot_score.append(chatbot_result)

        #     return edict({'human': human_score, 'chatbot': chatbot_score})

        elif self.task == "CoLA":
            sentence = history.human[-1]
            sentence = history
            classes = ["wrong", "correct"]
            paraphrase = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            paraphrase_classification_logits = self.model(**paraphrase)[0]
            paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
            self.score.append(float(paraphrase_results[1] * 100))
            return float(paraphrase_results[1] * 100)



    def average(self):
        return sum(self.score) / AFL.count

    @classmethod
    def change_content(cls, chatbot, book):
        # # CoLA_avg = CoLA.average()
        # # MRPC_avg = MRPC.average()
        # # sent_redunancy = Redundancy.redundancy_rate(sentece['text'])
        # print(CoLA_avg,MRPC_avg)
        # personality =chatbot.personality
        # if CoLA_avg >80 and MRPC_avg > 70 and AFL.changed_flag ==False :
        originPersonality = chatbot.personality
        history = []
        NewChapter = ""
        NewPersonality = random.choice(chatbot.personalities)
        while NewPersonality[0] in originPersonality:
            NewPersonality = random.choice(chatbot.personalities)

        for i in NewPersonality:
            history.append(chatbot.tokenizer.decode(i))

        for unit, pers in book.items():
            if NewPersonality[0] in pers:
                NewChapter = unit

        AFL.changed_flag = True
        print(chatbot.history)
        print("_____________")
        print(history)
        chatbot.history = history
        chatbot.chapter = NewChapter
        return NewPersonality



class ContextDector:
    '''Parent class for Context Similarity and Lingustic Acceptability for initialzing '''

    flag: bool = False

    def __init__(self, model_name) -> None:
        self.score = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)


class ContextSimilarity(ContextDector):
    '''Get similarity between personality and history using fine-tuned model by MRPC dataset'''

    def __init__(self, model_name = "textattack/xlnet-base-cased-MRPC") -> None:
        super().__init__(model_name)

    def predict(self, history: list, history_sentences: list):
        sentence = history.human[-1] # recent human input
        scores = []

        for history_sentence in history_sentences:
            paraphrase = self.tokenizer.encode_plus(sentence, history_sentence, return_tensors="pt").to(self.device)
            paraphrase_classification_logits = self.model(**paraphrase)[0]
            paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
            scores.append(float(paraphrase_results[1])) # classes = ["not paraphrase", "is paraphrase"]

        scores = np.array(scores)
        score_max = np.max(scores)
        self.score.append(score_max * 100)
        return score_max * 100


class LinguisticAcceptability(ContextDector):
    '''Get a score on how linguistically acceptable a user's input sentence using fine-tuned by CoLA dataset'''

    def __init__(self, model_name = "textattack/xlnet-base-cased-CoLA") -> None:
        super().__init__(model_name)

    def predict(self, history):
        sentence = history.human[-1]
        sentence = history
        classes = ["wrong", "correct"]
        paraphrase = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        paraphrase_classification_logits = self.model(**paraphrase)[0]
        paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
        self.score.append(float(paraphrase_results[1] * 100))
        return float(paraphrase_results[1] * 100)
