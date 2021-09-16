from ChatBot import ChatBot
import torch
import torch.nn.functional as F
import numpy as np
import random
import requests
import json
from easydict import EasyDict as edict


class AFL:
    count=0
    changed_flag=False

    def __init__(self, model, tokenizer, task):
        self.model = model
        self.tokenizer= tokenizer
        self.score=[]
        # self.history =history
        self.task = task
        self.answer=[]

    def return_prediction(self, history:list, history_original) -> dict :
        if self.task == "MRPC":
            sen_1 = history.human[-1]
            out_put= []

            for sequence_persona in history_original:
                paraphrase = self.tokenizer.encode_plus(sen_1, sequence_persona, return_tensors="pt")
                paraphrase_classification_logits = self.model(**paraphrase)[0]
                paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
                out_put.append(paraphrase_results[1])
            out_put=np.array(out_put)
            out_put_max=np.max(out_put)
            self.score.append(round(out_put_max*100))
            results = {"max_value":round(out_put_max*100)}
            return round(out_put_max*100)

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
            sen_1 = history.human
            classes = ["wrong", "correct"]
            paraphrase = self.tokenizer(sen_1, return_tensors="pt")
            paraphrase_classification_logits = self.model(**paraphrase)[0]
            paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
            self.score.append(round(paraphrase_results[1] * 100))
            print(self.score)
            return round(paraphrase_results[1] * 100)

        elif self.task =='Redundancy':
            sen_1 =history.human
            out_put= []
            out_put_max=0
            classes = ["not paraphrase", "is paraphrase"]
            if len(self.answer) > 0 :
                for i in self.answer:
                    paraphrase = self.tokenizer.encode_plus(sen_1, i, return_tensors="pt")
                    paraphrase_classification_logits = self.model(**paraphrase)[0]
                    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
                    out_put.append(paraphrase_results[1])
                out_put=np.array(out_put)
                out_put_max=np.max(out_put)
                self.score.append(round(out_put_max*100))

            self.answer.append(sen_1)
            print(self.answer)
            return round(out_put_max*100)

      # if AFL.count==10:
      #   avg =average(self.score,AFL.count)
      #   return avg

            results = {"max_value":round(out_put_max*100)}

            return round(out_put_max*100)

    def spell_check(x,KEY,PARAM,HEADER):

        api_key = KEY
        example_text = x['text'] # the text to be spell-checked
        endpoint = "https://api.cognitive.microsoft.com/bing/v7.0/SpellCheck"
        data = {'text': example_text}
        params = PARAM
        headers = HEADER
        response = requests.post(endpoint, headers=headers, params=params, data=data)
        result=[]
        json_response = response.json()
        if json_response['flaggedTokens']:
            for i in range(0,len(json_response['flaggedTokens'])):
                max=0
                max_word=''
                for j in range(0,len(json_response['flaggedTokens'][i]['suggestions'])):

                    if max < json_response['flaggedTokens'][i]['suggestions'][j]['score']:
                        max = json_response['flaggedTokens'][i]['suggestions'][j]['score']
                        max_word=json_response['flaggedTokens'][i]['suggestions'][j]['suggestion']
                        result.append(f"{json_response['flaggedTokens'][i]['token']}-->{max_word}")
        print(f"AFL{result}")
        return result

    def average(self):
            return sum(self.score)/AFL.count

    @classmethod
    def change_content(cls,chatbot ,book):
        # # CoLA_avg = CoLA.average()
        # # MRPC_avg = MRPC.average()
        # # sent_redunancy = Redundancy.redundancy_rate(sentece['text'])
        # print(CoLA_avg,MRPC_avg)
        # personality =chatbot.personality
        # if CoLA_avg >80 and MRPC_avg > 70 and AFL.changed_flag ==False :
        originPersonality=chatbot.personality
        history=[]
        NewChapter=""
        NewPersonality = random.choice(chatbot.personalities)
        while NewPersonality[0] in originPersonality:
            NewPersonality = random.choice(chatbot.personalities)

        for i in NewPersonality:
            history.append(chatbot.tokenizer.decode(i))

        for unit, pers in book.items():
            if NewPersonality[0] in pers:
                NewChapter = unit





        AFL.changed_flag=True
        print(chatbot.history)
        print('_____________')
        print(history)
        chatbot.history = history
        chatbot.chapter = NewChapter
        return NewPersonality
