import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop

from simpler_train import *

params = {
  "epsilon": 1,
  "epsilon_decay": 1.0 / 100.0,
  "min_epsilon": 0.1,
  "game_buffer": 100,
  "batch_size": 64,
  "memory": 120,
  "action_size": 8,
  "input_count": 15 * 2 + 4
}

def create_map():
  return {
    "agents": [ Agent([100, 400]), Agent([100, 100]) ],
    "obstacles": [ Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)]) ]
  }

def create_model(params):
  model = Sequential()
  model.add(Input(shape=(params["input_count"])))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(8, activation='linear'))
  model.compile(loss='mse', optimizer=Adam())
  return model

# def create_model(params):
#   model = Sequential()
#   model.add(Input(shape=(params["memory"], params["input_count"])))
#   model.add(LSTM(16))
#   model.add(Dense(64, activation='relu'))
#   model.add(Dense(7, activation='linear'))
#   model.compile(loss='mse', optimizer=Adam())
#   return model

agent_brains = [create_model(params), create_model(params)]

actions_buffer = None
for game_count in range(10000):
  print("Running simulation #" + str(game_count))
  actions_buffer = train_in_game(game_objs=create_map(),
                                 agent_models=agent_brains,
                                 params=params,
                                 actions_buffer=actions_buffer,
                                 log=(game_count % 10) == 0,
                                 save_file_name="games/game_{c}.json".format(c=game_count))