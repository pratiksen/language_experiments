from typing import List, Dict, Optional, Tuple
import keras.backend as K
import keras.layers as kl
import keras.models as km


def build_seq2seq(max_sentence_len:int, vocabulary_size:Tuple(int, int), units:List(int), rnn_params:Dict()=None,
                  rnn:Optional(kl.LSTM, kl.GRU)=kl.LSTM)->km.Model:
    if rnn_params is None:
        rnn_params = {}

    encoder_input = kl.Input(shape=(max_sentence_len))
    encoder = kl.Embedding(vocabulary_size[0], 256)(encoder_input)
    encoder_states = []
    for unit in units:
        encoder, *states= rnn(unit, return_state=True, return_sequences=True, **rnn_params)(encoder)
        encoder_states.append(states)

    decoder_input = kl.Input(shape=(max_sentence_len))
    decoder = kl.Embedding(vocabulary_size[1], 256)(decoder_input)
    for nlayer, unit in enumerate(units):
        state = encoder_states[nlayer]
        decoder = rnn(unit, return_sequences=True, **rnn_params)(decoder, initial_state=state)

    decoder_out = kl.TimeDistributed(kl.Dense(vocabulary_size[1], activation='softmax'))

    model = km.Model([encoder_input, decoder_input], decoder_out)
    return model
