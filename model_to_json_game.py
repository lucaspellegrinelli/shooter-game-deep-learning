import sys
import json
import os
import numpy as np
from tensorflow import keras

from shooter import ShooterEnv
from trainer import QModel

game_actions = {"players": [], "actions": [], "rewards": []}

game = ShooterEnv()

for agent in game.agents:
  game_actions["rewards"].append(0)
  game_actions["players"].append([
    agent.current_position[0],
    agent.current_position[1],
    agent.current_angle
  ])


model = QModel(37, 30, 7)
model.load_weights(sys.argv[1])

epsilon = float(sys.argv[2])

all_actions = []
all_states = []
for i in range(1000):
  state = np.expand_dims(game._next_observation(0), 0)
  all_states.append(state)

  if epsilon > np.random.rand(1)[0] and len(all_states) > 30:
    data = np.array(all_states[-30:])
    data = np.squeeze(data, 1)
    data = np.expand_dims(data, 0)
    actions_ = model(data, training=False)
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

  step_actions = [actions, a1]

  _, rewards, _, _ = game.step(step_actions, i / 1000)

  for i, r in enumerate(rewards): game_actions["rewards"][i] += r
  game_actions["actions"].append(step_actions)

model_name = sys.argv[1].split(".h5")[0].split("model_")[1]
os.makedirs(os.path.dirname("other_games/game_" + model_name + ".json"), exist_ok=True)
with open("other_games/game_" + model_name + ".json", "w") as outfile:
  json.dump(game_actions, outfile)