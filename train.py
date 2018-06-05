import logging
import hyperas

import model
import dataset

def train():
    french = dataset.french
    english = dataset.english
    frs, eng = dataset.frs, dataset.eng
    nmt = model.build_seq2seq()