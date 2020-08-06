import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def QModel(n_inputs, n_outputs):
  inputs = layers.Input(shape=(n_inputs, 1))

  layer1 = layers.Conv1D(filters=64, kernel_size=3, strides=1, activation="relu")(inputs)
  layer2 = layers.Flatten()(layer1)
  layer3 = layers.Dense(512, activation="relu")(layer2)
  action = layers.Dense(n_outputs, activation="linear")(layer3)

  return keras.Model(inputs=inputs, outputs=action)