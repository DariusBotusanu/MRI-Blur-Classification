import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    ####Modify to read the csv
    'Generates data for Keras'
    #might need to add
    def __init__(self, list_IDs, labels, batch_size=32, dim=(176, 256, 256), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        ## reconstruct it
        #construct the batch that will be sent
        # x -> the cube of images 
        # y -> array of targets
        #randomize it 
       # return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)