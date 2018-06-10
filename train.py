import dataset
import model


def train():
    train, test, dictionary = dataset.etl()
    vocabulary = tuple(i.last_word_idx+2 for i in dictionary)
    nmt = model.build_seq2seq(None, vocabulary_size=vocabulary, units=[1024]*2, rnn_params={'recurrent_dropout':0.5})
    nmt.compile(optimizer='nadam', loss='categorical_crossentropy')
    nmt.fit_generator(train, epochs=10, verbose=1, validation_data=test)


if __name__ == '__main__':
    train()
