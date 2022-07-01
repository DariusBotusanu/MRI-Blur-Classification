import numpy as np

from keras.models import Sequential
from DataGenerator import DataGenerator
from project_scripts import train_validation_test_split

# Parameters
params = {'dim': (176, 256, 256),
          'batch_size': 32,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition, labels, test_labels = train_validation_test_split()

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)
testing_generator = DataGenerator(partition['test'], test_labels, **params)

# Design model

# Train model on dataset
