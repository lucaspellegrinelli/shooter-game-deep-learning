import sys
import json
import numpy as np
from tensorflow import keras

from shooter import ShooterEnv
from trainer import QModel

game = ShooterEnv()

model = QModel(33, 7)
model.load_weights(sys.argv[1])

epsilon = float(sys.argv[2])

all_actions = []
for i in range(1000):
  state = np.expand_dims(game._next_observation(0), 0)

  if epsilon > np.random.rand(1)[0]:
    actions_ = model(state, training=False)
    actions = [1 if x == np.argmax(actions_) else 0 for x in range(7)]
  else:
    r = np.random.choice(7)
    actions = [1 if x == r else 0 for x in range(7)]

  game.step([actions, None], , i / 1000)
  all_actions.append(actions)

model_name = sys.argv[1].split(".h5")[0].split("model_")[1]
with open("games/game_" + model_name + ".json", "w") as outfile:
  json.dump(all_actions, outfile)