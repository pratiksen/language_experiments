import re
from typing import List
from collections import defaultdict

import numpy as np

class Index:
    def __init__(self):
        self.words = set()
        self.last_word_idx = 0
        self.index_word = {}
        self.word_index = {}

    def index(self, words:List[str]) -> List[int]:
        transformed = []
        for word in words:
            if word not in self.word_index:
                self.add_word(word)
            transformed.append(self.word_index[word])
        return transformed

    def add_word(self, word: str):
        self.last_word_idx += 1
        self.index_word[self.last_word_idx] = word
        self.word_index[word] = self.last_word_idx



def normalize_text(text):
    text = re.sub('[.!?]', ' ', text)
    text = re.sub('[^a-zA-Z.!?]+', " ", text)
    text = text.lower().strip().split(' ')
    return text

def to_array(source, target):
    array = defaultdict(list)

    for src, tgt in zip(source, target):
        src = [-1] + src + [0]
        tgt = [-1] + tgt + [0]
        array[(len(src), len(tgt))].append((src, tgt))
    return array

def get_data():
    with open('D:/data/fra-eng/fra.txt', encoding='utf-8') as f:
        frs = []
        eng = []
        french_index = Index()
        english_index = Index()
        for i in f.readlines():
            j =  i.split('	')
            french = normalize_text(j[1])
            english = normalize_text(j[0])

            frs.append(french_index.index(french))
            eng.append(english_index.index(english))

        data = to_array(eng, frs)
    return data

if __name__ == '__main__':
    data = get_data()
    print('done')


