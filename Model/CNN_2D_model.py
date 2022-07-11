from tensorflow import keras
from tensorflow.keras import layers

def get_uncompiled_model():
    inputs = keras.Input(shape=(256, 256,1))

    initializer = keras.initializers.HeNormal()

    x = layers.Conv2D(filters=6, kernel_size=5, stride=1, activation="tanh", kernel_initializer=initializer)(inputs) 
    x = layers.MeanPool2D(pool_size=2, stride=2)(x) 

    x = layers.Conv2D(filters=16, kernel_size=2, stride=2,activation="tanh", kernel_initializer=initializer)(inputs) 
    x = layers.MeanPool2D(pool_size=2, stride=2)(x)

    x = layers.Conv2D(filters=120, kernel_size=5, stride=1, activation="tanh", kernel_initializer=initializer)(inputs) 

    outputs = layers.Dense(units=1, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model