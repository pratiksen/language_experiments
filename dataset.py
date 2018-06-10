import random
import re

from keras.utils import to_categorical

from src.data import DataGenerator
from src.index import Index


def normalize_text(text):
    text = re.sub('[.!?]', ' ', text)
    text = re.sub('[^a-zA-Z.!?]+', " ", text)
    text = list(text.lower().strip())
    return text

def to_array(data, src_classes, tgt_classes):
    array = []

    for src, tgt in data:
        src = [-1] + src + [-1]
        tgt = [-1] + tgt + [-1]
        src = to_categorical(src, num_classes=src_classes+1)
        tgt = to_categorical(tgt, num_classes=tgt_classes+1)
        array.append((src, tgt))
    return array


def preprocess():
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

        data = list(zip(frs, eng))
    return data, (french_index, english_index)


def train_test_split(data, sample_frac=0.8):
    random.shuffle(data)
    n = len(data)
    n_train = int(n * sample_frac)
    train = data[:n_train]
    test = data[n_train:]
    return train, test


def etl():
    data, dictionary = preprocess()
    data = to_array(data, dictionary[0].last_word_idx+1, dictionary[0].last_word_idx+1)
    train, test = train_test_split(data)
    train = DataGenerator(*zip(*train))
    test = DataGenerator(*zip(*test))
    return train, test, dictionary

if __name__ == '__main__':
    a = etl()
    train = a[0]
    test = a[1]
    print(train[1])

