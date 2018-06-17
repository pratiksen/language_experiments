from sacred import Experiment

import dataset
import model

ex = Experiment('language')


@ex.config
def config():
    model_type = 'seq2seq'
    params = {'max_sentence_len':None,
              'units':[128],
              'rnn_params':{'recurrent_dropout':0.5}}
    optimizer = 'nadam'
    loss = 'categorical_crossentropy'
    verbose = 1


@ex.capture
def build_model(vocabulary_size, model_type, params, optimizer, loss):
    builder = getattr(model, f'build_{model_type}')
    params['vocabulary_size'] = vocabulary_size
    nmt = builder(**params)
    nmt.compile(optimizer=optimizer, loss=loss)
    return nmt


@ex.capture
def train_model(epochs, verbose):
    train, test, dictionary = dataset.etl()
    vocabulary = tuple(i.last_word_idx+2 for i in dictionary)
    language_model = build_model(vocabulary)
    language_model.fit_generator(train, epochs=epochs, verbose=verbose,
                                 validation_data=test)


@ex.main
def run():
    train_model()
