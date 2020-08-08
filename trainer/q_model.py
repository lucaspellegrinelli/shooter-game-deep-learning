import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def QModel(n_inputs, n_outputs):
  inputs = layers.Input(shape=(n_inputs, 1))
  layer1 = layers.Conv1D(filters=64, kernel_size=3, strides=1, activation="relu")(inputs)
  layer2 = layers.Flatten()(layer1)
  layer3 = layers.Dense(512, activation="relu")(layer2)
  action = layers.Dense(n_outputs, activation="linear")(layer3)
  model = keras.Model(inputs=inputs, outputs=action)
  return model

def QModelMemoryConv(n_inputs, mem_size, n_outputs):
  inputs = layers.Input(shape=(mem_size, n_inputs))
  conv0 = layers.Conv1D(filters=96, kernel_size=5, strides=1, activation="relu")(inputs)
  conv1 = layers.Conv1D(filters=64, kernel_size=4, strides=1, activation="relu")(conv0)
  conv2 = layers.Conv1D(filters=32, kernel_size=3, strides=1, activation="relu")(conv1)
  flatten = layers.Flatten()(conv2)
  dense = layers.Dense(128, activation="relu")(flatten)
  action = layers.Dense(n_outputs, activation="linear")(dense)
  model = keras.Model(inputs=inputs, outputs=action)
  return model