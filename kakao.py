from flask import Flask, request, jsonify
from werkzeug.wrappers import response
import json
import re
import requests
import regex
import random

from models.chatbot import Chatbot
from models.context_detector import ContextSimilarity, LinguisticAcceptability
from models import grammar
from pprint import pprint

app = Flask(__name__)
chatbot = Chatbot()
# similarity = ContextSimilarity()
linguistic = LinguisticAcceptability()



@app.route('/')
def hello():
    return 'Hello, World!'


def is_hangul(text: str):
    if regex.search(r'\p{IsHangul}', text):
        return True
    return False

def text_preprocess(text:str, output = False)-> str:
    if not output:
        text = text.lower()
        text = re.sub(r"([?.!,:;¿])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
    else:
        text = re.sub(r" ([?.!,:،؛؟¿])", r"\1", text)
        text = text.replace("▁"," ")
    return text

def correct_grammer(source: str) -> json:
    language = "en-en"
    URL = "http://nlplab.iptime.org:32293/translator/translate"
    headers = {"Content-Type": "application/json"}
    query = [{"src": source, "id": language}]
    response = requests.post(URL, json=query, headers=headers)
    correction = response.json()[0][0]['tgt']
    correction = text_preprocess(correction, output = True)
    return correction

turn = 0
@app.route('/grammer', methods=['POST'])
def grammer_check():
    global turn
    input_json = request.get_json()
    turn += 1
    print(f"TRUN: {turn}")
    user_sentence = input_json['userRequest']['utterance']
    # answer = correct_grammer(user_sentence)
    message = chatbot.send(user_sentence)
    human_history = chatbot.get_human_history()
    gold_history = chatbot.get_gold_history()
    print(gold_history)

    # print(f"GOLD HISTORY: {gold_history}")
    sim_score = random.uniform(0.5,1)
    lang_score = linguistic.predict(human_history)

    print(f"SIMILIARITY: {sim_score}")
    turn = 6
    if turn > 5:
        chatbot.shuffle()
        response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "How about talk about different topics?"
                        }
                    }
                ]
            }
        }

    else:
        response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": message
                        }
                    },
                    {
                        "simpleText": {
                            "text": f"상황유사도: {round(sim_score, 2)}\n언어적 허용도: {round(lang_score, 2)}"
                        }
                    },
                    {
                        "simpleText": {
                            "text": f"GEC\n'{grammar.correct(user_sentence)}'"
                        }
                    }
                ]
            }
        }


    return jsonify(response)

@app.route('/shuffe')
def shuffle():
    chatbot.shuffle()
    return "Success"

@app.route('/history')
def get_history():
    return jsonify(chatbot.get_gold_history())


@app.route('/persona')
def get_persona():
    response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": f"Personality: \n{chatbot.get_personality()}"
                        }
                    }
                ]
            }
        }

    return jsonify(response)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000 , threaded=True)

