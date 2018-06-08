import keras
import numpy as np
import random
from collections import defaultdict

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, x, y, batch_size=32):

        'Initialization'
        self._x = x
        self._y = y
        self.n = len(x)
        self.batch_size = batch_size
        bin_groups = defaultdict(list)
        for i in range(self.n):
            x = self._x[i]
            y = self._y[i]
            bin_groups[(len(x), len(y))].append((x, y))
        self.bin_groups = bin_groups
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.data[index]
        X = np.array(X)
        y = np.array(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.__data_generation()

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        data = []
        for bin in self.bin_groups:
            random.shuffle(self.bin_groups[bin])
            n_bin = len(self.bin_groups[bin])
            for i in range(0, n_bin, self.batch_size):
                batch = self.bin_groups[bin][i:i+self.batch_size]
                data.append(batch)
        random.shuffle(data)
        self.data = data
