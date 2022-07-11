import numpy as np
import pandas as pd
import keras

class DataGenerator(keras.utils.all_utils.Sequence):
    ####Modify to read the csv
    'Generates data for Keras'
    #might need to add
    def __init__(self, partition='train', trim_ends=20, batch_size=32, dim=(256, 256),
                 n_classes=2, shuffle=True):
        'Initialization'
        self.partition = partition
        self.trim_ends = trim_ends
        self.dim = dim
        self.batch_size = batch_size
        
        
        df = pd.read_csv('Data Splits/train_validation_test_split.csv')
        self.list_IDs = list(df[df['partition']==partition]['path'])

        for i in range(len(self.list_IDs)-1,-1, -1):
          #We check if the slice number is between the bounds [trim_ends:176-trim_ends]
          ID = self.list_IDs[i]
          first_underscore = ID.find('_')
          aux = ID[first_underscore+1:]
          second_underscore = aux.find('_')
          slice_num = int(aux[:second_underscore])
          #We check if we remove the path to the image
          if ((slice_num < trim_ends) or (slice_num >176 - trim_ends)):
            self.list_IDs.remove(self.list_IDs[i])

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