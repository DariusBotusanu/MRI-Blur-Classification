import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_uncompiled_model():
    inputs = keras.Input(shape=(256, 256,1,))

    initializer = keras.initializers.HeNormal()

    x = layers.Conv2D(1, 3, activation='relu', kernel_initializer=initializer)(inputs)

    x = layers.Flatten()(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model