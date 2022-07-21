import numpy as np
import pandas as pd
import keras

class DataGenerator(keras.utils.all_utils.Sequence):
    ####Modify to read the csv
    'Generates data for Keras'
    #might need to add
    def __init__(self, partition='train', batch_size=32, dim=(256, 256),
                 n_classes=2, shuffle=True):
        'Initialization'
        self.partition = partition
        self.dim = dim
        self.batch_size = batch_size
        
        
        df = pd.read_csv('Data Splits/train_validation_test_split.csv')
        self.list_IDs = list(df[df['partition']==partition]['path'])

        self.path_label_dict = dict() #this dictionary will associate the path to an iamge to its label
        for i in range(len(self.list_IDs)):
            self.path_label_dict[self.list_IDs[i]] = df[df['path']==self.list_IDs[i]]['label'].iloc[0]
   
        self.indexes = np.arange(len(self.list_IDs))
        
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
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
       
    def __data_generation(self, list_indexes_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_indexes_temp):
            # Store sample
            X[i,] = np.load(self.list_IDs[ID])
            # Store class
            y[i] = self.path_label_dict[self.list_IDs[ID]]
            
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class FoldGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, folds = [1,2,3,4], batch_size=32, dim=(256, 256),
                 n_classes=2, shuffle=True):
        'Initialization'
        self.folds = folds
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = []

        
        self.path_label_dict = dict() #this dictionary will associate the path to an iamge to its label
        for i in folds:
            df = pd.read_csv(f'Data Splits/{i}_fold.csv')
            aux_IDs = list(df['path'])

            for path in aux_IDs:
              self.path_label_dict[path] = df[df['path']==path]['label'].iloc[0]
              
            self.list_IDs += aux_IDs
        
        self.indexes = np.arange(len(self.list_IDs))
        
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
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
       
    def __data_generation(self, list_indexes_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_indexes_temp):
            # Store sample
            X[i,] = np.load(self.list_IDs[ID])
            # Store class
            y[i] = self.path_label_dict[self.list_IDs[ID]]
            
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)