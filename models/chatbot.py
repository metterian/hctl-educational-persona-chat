import os
import random
import pickle
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from interact import top_filtering, sample_sequence
from train import add_special_tokens_
from utils import get_dataset


# pickle load
def pickle_load(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_save(path: str, data) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class Chatbot:
    '''Conversation Agent model based on Hugging face, using GPT-2'''
    def __init__(self, args) -> None:
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


    def laod_dataset(self) -> None:
        '''Load Persona, History dataset as caches or json files'''
        dataset = get_dataset(self.tokenizer, self.args.dataset_path, self.args.dataset_cache)

        # load persona cache
        if self.args.persona_cache and os.path.isfile(self.args.persona_cache):
            personalities = pickle_load(self.args.persona_cache)
        else:
            personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
            pickle_save(path="./cache/persona_cache", data=personalities)

        # load history cache
        if self.args.history_cache and os.path.isfile(self.args.history_cache):
            history = pickle_load(self.args.history_cache)
        else:
            history = [ dialog["utterances"][-1]["history"] for dataset in dataset.values() for dialog in dataset ]
            pickle_save(path="./cache/history_cache", data=history)

        self.utterances = [ dialog["utterances"] for dataset in dataset.values() for dialog in dataset ]


    def shuffle_inputs(self, personalities: list, utterances: list, history: list) -> list:
        '''Shuffle the inputs which are persona, utterance and history by the persona index'''
        shuffle_idx = random.choice(range(len(personalities)))
        personality = personalities[shuffle_idx]
        utterance = utterances[shuffle_idx]
        gold_history = history[shuffle_idx]
        gold_history = [self.tokenizer.decode(line) for line in gold_history]

        return personality, utterance, gold_history

    def decode(self, tokens) -> list:
        'Decode the utterance by tokenizer'
        return [self.tokenizer.decode(token) for token in tokens]

    def get_personality(self):
        '''Return current personality'''
        personality_decoded = self.decode(self.personality)
        print(f"PERSONA:{personality_decoded}")
        return personality_decoded
