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
    human: str
    chatbot: Optional[str]


app = FastAPI()
chatbot = Chatbot()
context_sim = ContextSimilarity()
linguistic = LinguisticAcceptability()


@app.post("/receive/")
async def receive(item: Message):
    raw_text = item.human
    sentence = raw_text.strip()

    message = chatbot.send_message(sentence)
    human_history = chatbot.get_human_history()
    gold_history = chatbot.get_gold_history()

    similarity = context_sim.predict(human_history, gold_history)
    acceptability = linguistic.predict(human_history)

    correction = grammar.correct(sentence)

    response = Response(
        message=message,
        similarity=similarity,
        acceptability=acceptability,
        personality=chatbot.get_personality(),
        correction=correction,
    )

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
