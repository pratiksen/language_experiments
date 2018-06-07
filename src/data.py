import keras
import numpy as np
from collections import defaultdict

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, x, y, batch_size=32, shuffle=True):

        'Initialization'
        self._x = x
        self._y = y
        self.n = len(x)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        bin_groups = defaultdict(list)
        for i in range(self.n):
            x = self._x[i]
            y = self._y[i]
            bin_groups[(len(x), len(y))].append((x,y))

        data = []
        for bin in bin_groups:
            n_bin = len(bin_groups[bin])
            for i in range(0, n_bin, self.batch_size):
                batch = bin_groups[bin][i:i+self.batch_size]
                data.append(batch)