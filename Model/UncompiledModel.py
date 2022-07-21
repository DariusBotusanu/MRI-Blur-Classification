from tensorflow import keras
from tensorflow.keras import layers

def get_uncompiled_model():
    inputs = keras.Input(shape=(256, 256,1))   

    initializer = keras.initializers.HeNormal()

    x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(inputs)

    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model