from dataclasses import dataclass
from typing import Optional, List
from fastapi import FastAPI
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
    turn: int
    correction: str
    changed: bool = False


@dataclass
class Message:
    human: str
    chatbot: Optional[str]


app = FastAPI()
# chatbot = Chatbot()
# context_sim = ContextSimilarity()
# accept_score = LinguisticAcceptability()

@app.post('/receive/')
async def receive(item: Message):
    return item.human





