import json
from tqdm import tqdm
import random
import numpy as np
import torch


class DataGenerator:
    def __init__(self, path, tokenizer, batch_size=128, max_length=256, seed=2021, verbose=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed

        self.data = [self._tokenize(d) for d in tqdm(self._load_raw(path), desc="Tokenize", disable=not verbose)]

    @staticmethod
    def _load_raw(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def _tokenize(self, piece):
        text = piece["text"]
        mistakes = piece["mistakes"]

        text_char = list(text)
        for mistake in mistakes:
            index = int(mistake["loc"]) - 1
            text_char[index] = mistake["correct"]
        correct = ''.join(text_char)

        token_ids = self.tokenizer.encode(text=text, max_length=self.max_length)
        token_seg = [0] * len(token_ids)
        token_ids = token_ids[:-1][:self.max_length-1] + [token_ids[-1]] + [0] * (self.max_length - len(token_ids))
        token_seg = token_seg[:self.max_length] + [0] * (self.max_length - len(token_seg))

        token_mask = [1 if i > 0 else 0 for i in token_ids]

        label_ids = self.tokenizer.encode(text=correct, max_length=self.max_length)
        label_ids = label_ids[:-1][:self.max_length - 1] + [label_ids[-1]] + [0] * (self.max_length - len(label_ids))
        label_mistake = [0 if i == j else 1 for i, j in zip(token_ids, label_ids)]

        return token_ids, token_seg, token_mask, label_mistake, label_ids

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        random.seed(self.seed)
        while True:
            token_ids, token_seg, token_mask, label_mistake, label_ids = list(), list(), list(), list(), list()
            for d in self.data:
                token_ids.append(d[0])
                token_seg.append(d[1])
                token_mask.append(d[2])
                label_mistake.append(d[3])
                label_ids.append(d[4])
                if len(token_ids) == self.batch_size:
                    token_ids = torch.tensor(token_ids)
                    token_seg = torch.tensor(token_seg)
                    token_mask = torch.tensor(token_mask)
                    label_mistake = torch.tensor(label_mistake)
                    label_ids = torch.tensor(label_ids)
                    yield token_ids, token_seg, token_mask, label_mistake, label_ids
                    token_ids, token_seg, token_mask, label_mistake, label_ids = list(), list(), list(), list(), list()
