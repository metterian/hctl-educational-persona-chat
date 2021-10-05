from typing import List
from argparse import ArgumentParser
from itertools import chain
import warnings
import torch
import torch.nn.functional as F
from interact import top_filtering, sample_sequence

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_


class ChatBot:
    '''Conversation Agent model based on Hugging face, using GPT-2'''
    def __init__(self, args):
        '''Initialize tokenizer, model and datasets'''
        self.args = args
        tokenizer_class, model_class = GPT2Tokenizer, GPT2LMHeadModel

        # laod tokenizer and model
        self.tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
        self.model = model_class.from_pretrained(args.model_checkpoint)
        self.model.to(args.device)
        add_special_tokens_(self.model, self.tokenizer)

        # set history as empty list for recording the conversation
        self.history = []

    def message(self, sentence : str, personality: list) -> str:
        '''Receive user input with Persona and send the next utterance.'''
        self.personality = personality
        self.history.append(self.tokenizer.encode(sentence))
        with torch.no_grad():
            out_ids = sample_sequence(self.personality, self.history, self.tokenizer, self.model, self.args)
            self.history.append(out_ids)
            self.history = self.history[-(2*self.args.max_history+1):]
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return out_text

