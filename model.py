from typing import List, Dict, Tuple, Union

import keras.layers as kl
import keras.models as km


def build_seq2seq(max_sentence_len:Union[int, None], vocabulary_size:Tuple[int, int], units:List[int], rnn_params:Dict=None,
                  rnn:Union[kl.LSTM, kl.GRU]=kl.LSTM)->km.Model:
    if rnn_params is None:
        rnn_params = {}

    encoder_input = kl.Input(shape=(max_sentence_len,vocabulary_size[0]))
    encoder_states = []
    encoder = encoder_input
    for nlayer, unit in enumerate(units):
        encoder, *states= rnn(unit, name=f'encoder_{nlayer}',
                              return_state=True,
                              return_sequences=True,
                              **rnn_params)(encoder)
        encoder_states.append(states)

    decoder_input = kl.Input(shape=(max_sentence_len, vocabulary_size[1]))
    decoder = decoder_input
    for nlayer, unit in enumerate(units):
        state = encoder_states[nlayer]
        decoder = rnn(unit, name=f'decoder_{nlayer}',
                      return_sequences=True,
                      **rnn_params)(decoder, initial_state=state)

    decoder_out = kl.TimeDistributed(kl.Dense(vocabulary_size[1], activation='softmax'))(decoder)

    model = km.Model([encoder_input, decoder_input], decoder_out)
    return model
