from dataclasses import dataclass
from typing import Optional, List
from fastapi import FastAPI
import uvicorn
from fastapi.params import Query

from models.context_detector import ContextSimilarity, LinguisticAcceptability
from models.chatbot import Chatbot
from models.config import args
from models import grammar


@dataclass
class Response:
    message: str
    similarity: int
    acceptability: int
    personality: List[str]
    turn: Optional[int]
    correction: str
    changed: Optional[bool] = False


@dataclass
class Message:
    user_input: str


app = FastAPI()
chatbot = Chatbot()
similarity = ContextSimilarity()
linguistic = LinguisticAcceptability()


@app.post("/receive/")
async def receive(item: Message):
    raw_text = item.user_input
    sentence = raw_text.strip()

    message = chatbot.send_message(sentence)
    human_history = chatbot.get_human_history()
    gold_history = chatbot.get_gold_history()

    similarity_score = similarity.predict(human_history, gold_history)
    lang_score = linguistic.predict(human_history)
    correction = grammar.correct(sentence)

    response = Response(
        message=message,
        similarity=similarity_score,
        acceptability=lang_score,
        personality=chatbot.get_personality(),
        correction=correction,
    )

    return response


@app.get("/info")
async def persona_info():
    return chatbot.get_personality()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
