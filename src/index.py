from typing import List

class Index:
    def __init__(self):
        self.words = set()
        self.last_word_idx = -1
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
