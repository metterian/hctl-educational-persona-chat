from flask import Flask, request, jsonify
from werkzeug.wrappers import response
import json
import re
import requests
import regex

app = Flask(__name__)


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


@app.route('/grammer', methods=['POST'])
def grammer_check():
    input_json = request.get_json()
    user_sentence = input_json['userRequest']['utterance']
    answer = correct_grammer(user_sentence) if not is_hangul(user_sentence) else "영문으로 입력해주세요."
    # answer = correct_grammer(user_sentence)

    reponse = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }

    return jsonify(reponse)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=22 , threaded=True)

