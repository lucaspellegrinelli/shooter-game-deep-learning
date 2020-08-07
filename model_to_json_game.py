import sys
import json
import numpy as np
from tensorflow import keras

from shooter import ShooterEnv
from trainer import QModel

game = ShooterEnv()

model = QModel(34, 7)
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


  r1 = np.random.choice(7)
  a1 = [
    1 if r1 == 0 else 0,
    1 if (r1 == 1 or r1 == 4 or r1 == 5) else 0,
    1 if r1 == 2 else 0,
    1 if (r1 == 3 or r1 == 6) else 0,
    0, 0, 0
  ]

  game.step([actions, a1], i / 1000)
  all_actions.append([actions, a1])

model_name = sys.argv[1].split(".h5")[0].split("model_")[1]
with open("other_games/game_" + model_name + ".json", "w") as outfile:
  json.dump(all_actions, outfile)