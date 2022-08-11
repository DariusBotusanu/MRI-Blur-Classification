import numpy as np
import pandas as pd
import keras
from scipy.signal import convolve2d

import imgaug as ia
import imgaug.augmenters as iaa
ia.seed(1)
class DataAugmenter():
  def __init__(self):
    self.seq = iaa.Sequential([
                              #iaa.Fliplr(0.5), # horizontal flips

                              # Apply affine transformations to each image.
                              # Scale/zoom them, translate/move them, rotate them and shear them.
                              iaa.Affine(
                                  scale={"x": (1.5, 1.5), "y": (1.5, 1.5)},
                                  translate_percent={"x": (-0.35, 0.2), "y": (-0.1, 0.1)},
                                  rotate=(-10, 10),
                                  )
                              ], random_order=True) # apply augmenters in random order
                              
  def augment(self, original_images):
    return self.seq(images=original_images)

    
class DataTransformer():
  def __init__(self, kernel=np.array([[1,0,-2],[1,0,-2],[1,0,-2]]), std_trimmer=1.5):
    self.kernel = kernel
    self.std_trimmer = std_trimmer
                      
  def transform(self, original_img):
    convolved_img = convolve2d(original_img, self.kernel, mode='same')
    std = pd.Series(convolved_img.reshape(convolved_img.shape[0]*convolved_img.shape[1])).describe()['std']
    for i in range(len(convolved_img)):
      for j in range(len(convolved_img[i])):
        if abs(convolved_img[i][j]) < std*self.std_trimmer:
          convolved_img[i][j] = 0
    return convolved_img



class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for training with Keras and applies a kernel on the images if specified'
    def __init__(self, partition='train', batch_size=16, dim=(256, 256),
                 n_classes=2, transform=False, kernel=np.array([[1,0,-2],[1,0,-2],[1,0,-2]]), shuffle=True):
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
        self.transform = transform
        self.kernel=kernel
        self.shuffle = shuffle
        self.on_epoch_end()

        ###Data Transformation
        if self.transform:
          self.transformer = DataTransformer()

        ###Data Augmentation
        if self.partition == 'train':
          self.augmentor = DataAugmenter()
        


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
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        if self.partition == 'train':
          X = np.empty((self.batch_size*2, *self.dim))
          y = np.empty((self.batch_size*2), dtype=int)
          for i, ID in enumerate(list_indexes_temp):
            # Store sample
            if self.transform:
              X[2*i] = convolve2d(np.load(self.list_IDs[ID]), self.kernel, mode='same')
              X[2*i+1,] = self.augmentor.augment(convolve2d(np.load(self.list_IDs[ID]), self.kernel, mode='same'))[0]
            else:
              X[2*i] = np.load(self.list_IDs[ID])
              X[2*i+1,] = self.augmentor.augment(np.load(self.list_IDs[ID]))[0]
            # Store class
            y[2*i] = self.path_label_dict[self.list_IDs[ID]]
            y[2*i+1] = self.path_label_dict[self.list_IDs[ID]]

        else:
          for i, ID in enumerate(list_indexes_temp):
            # Store sample
            if self.transform:
              X[i,] = convolve2d(np.load(self.list_IDs[ID]), self.kernel, mode='same')
            else:
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
    'Generates data for crossvalidation'
    def __init__(self, folds = [1,2,3,4], batch_size=32, dim=(256, 256),
                 n_classes=2, transform=True, kernel=np.array([[1,0,-2],[1,0,-2],[1,0,-2]]), shuffle=True):
        'Initialization'
        self.folds = folds
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = []

        
        self.path_label_dict = dict() #this dictionary will associate the path of an image to its label
        for i in folds:
            df = pd.read_csv(f'Data Splits/{i}_fold.csv')
            aux_IDs = list(df['path'])

            for path in aux_IDs:
              self.path_label_dict[path] = df[df['path']==path]['label'].iloc[0]
              
            self.list_IDs += aux_IDs
        
        self.indexes = np.arange(len(self.list_IDs))
        
        self.n_classes = n_classes
        self.transform = transform
        self.kernel=kernel
        self.shuffle = shuffle
        self.on_epoch_end()

        ###Data Transformation
        if self.transform:
          self.transformer = DataTransformer(kernel=kernel)

        ###Data Augmentation
        if len(self.folds) > 1:
          self.augmentor = DataAugmenter()

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

        for i, ID in enumerate(list_indexes_temp):
          # Store class
          y[i] = self.path_label_dict[self.list_IDs[ID]]
          # Store sample
          if self.transform:
            X[i,] = self.transformer.transform(np.load(self.list_IDs[ID]))
          else:
            X[i,] = np.load(self.list_IDs[ID])
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)