import numpy as np
import pandas as pd
import keras

class DataGenerator(keras.utils.Sequence):
    ####Modify to read the csv
    'Generates data for Keras'
    #might need to add
    def __init__(self, partition='train', batch_size=10, dim=(176, 256, 256), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        
        
        df = pd.read_csv('../Data Splits/train_validation_test_split.csv')
        self.list_IDs = list(df[df['partition']==partition]['path'])
        self.labels = df[df['partition']==partition][['path','label']]
        
        
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # #sample/batch size
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
       
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
       
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(ID)
            
            # Store class
            y[i] = self.labels[self.labels['path']==ID]['label']

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)