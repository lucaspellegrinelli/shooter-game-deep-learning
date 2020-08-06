import numpy as np
import random
random.seed(42)
import json
import collections
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop

from shooter import Game, Agent, Obstacle

params = {
  "epsilon": 1,
  "game_buffer": 100,
  "batch_size": 64,
  "memory": 120,
  "input_count": 45 * 2 + 4
}

# Creates the training data from mini batch
def process_mini_batch(mini_batch, model):
  mb_len = len(mini_batch)

  old_states = np.zeros(shape=(mb_len, params["input_count"]))
  actions = np.zeros(shape=(mb_len, 8))
  rewards = np.zeros(shape=(mb_len,))
  new_states = np.zeros(shape=(mb_len, params["input_count"]))

  def action_dict_to_array(action):
    action_arr = []
    for key, value in action.items():
      action_arr.append(1 if value else 0)
    return np.array(action_arr)

  for i, m in enumerate(mini_batch):
    old_states_m, action_m, reward_m, new_states_m = m
    old_states[i, :] = np.array(old_states_m)[...]
    new_states[i, :] = np.array(new_states_m)[...]
    actions[i, :] = action_dict_to_array(action_m)[...]
    rewards[i] = reward_m

  old_qvals = model.predict(old_states, batch_size=mb_len)
  new_qvals = model.predict(new_states, batch_size=mb_len)

  max_qs = np.max(new_qvals, axis=1)

  x_train = old_states
  y_train = old_qvals

  for s in range(len(y_train)):
    for a in range(len(y_train[i])):
      if actions[s][a] == 1:
        y_train[s][a] = rewards[s] + 0.9 * max_qs[s]

  return x_train, y_train

def create_model():
  model = Sequential()
  model.add(Input(shape=(params["input_count"])))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(8, activation='linear'))

  model.compile(loss='mse', optimizer=Adam())

  return model

def get_random_action():
  r = random.randint(0, 7)
  return {
    "left": r == 0,
    "right": r == 1,
    "up": r == 2,
    "down": r == 3,
    "rot_left": r == 4,
    "rot_right": r == 5,
    "fire": r == 6,
    "nothing": r == 7
  }

def create_game():
  agents = [
    Agent([100, 400]),
    Agent([100, 100])
  ]

  obstacles = [
    Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)]),
    # Obstacle([(250, 250), (300, 250), (500, 300), (400, 400), (250, 300)]),
    # Obstacle([(50, 50), (100, 50), (100, 100), (50, 100)]),
    # Obstacle([(150, 150), (200, 150), (200, 200), (150, 200)])
  ]

  return Game(agents, obstacles)

agent_models = [
  create_model(),
  create_model()
]

actions_buffer = [[], []]

log_freq = 10

game_count = 0
while True:
  # Game encapsulater
  game = create_game()

  game_count += 1

  # Get initial state
  _, _, states = game.step([get_random_action() for _ in range(len(game.agents))])

  # Decrement epsilon over time.
  if params["epsilon"] > 0.1:
    params["epsilon"] -= 1.0 / 100.0

  # Add new game to buffer
  for i in range(len(actions_buffer)):
    actions_buffer[i].append([])
    if len(actions_buffer[i]) > params["game_buffer"]:
      actions_buffer[i].pop(0)

  game_rewards_sum = [0 for _ in range(len(game.agents))]
  game_rewards_reasons = [[] for _ in range(len(game.agents))]

  while game.running:
    # If observing
    if random.random() < params["epsilon"]:
      # Take random actions
      actions = [get_random_action() for _ in range(len(game.agents))]
    else:
      # Take NN output actions
      actions = []
      for i in range(len(states)):
        prev_states = np.expand_dims(states[i], 0)
        pred_actions = agent_models[i].predict(prev_states)[0]

        actions.append({
          "left": np.argmax(pred_actions) == 0,
          "right": np.argmax(pred_actions) == 1,
          "up": np.argmax(pred_actions) == 2,
          "down": np.argmax(pred_actions) == 3,
          "rot_left": np.argmax(pred_actions) == 4,
          "rot_right": np.argmax(pred_actions) == 5,
          "fire": np.argmax(pred_actions) == 6,
          "nothing": np.argmax(pred_actions) == 7
        })

    # Take action, observe new state and get agents rewards
    rewards, rewards_reasons, new_states = game.step(actions)

    # Save relevant stuff
    for agent_i in range(len(game.agents)):
      game_rewards_sum[agent_i] += rewards[agent_i]
      game_rewards_reasons[agent_i].extend(rewards_reasons[agent_i])

      a = actions[agent_i].copy()
      for key, value in a.items():
        a[key] = bool(value)

    # Save the state, actions and reward
    for i in range(len(rewards)):
      actions_buffer[i][-1].append((states[i], actions[i], rewards[i], new_states[i]))

    # Update the starting states.
    states = new_states

  if ((game_count - 1) % log_freq) == 0:
    print("\nGame", game_count, "ended with", game.frame_count, "frames")
    for agent_i in range(len(game.agents)):
      print("Agent", agent_i, "Rewards:", game_rewards_sum[agent_i])

      reasons_counter = collections.Counter(game_rewards_reasons[agent_i])
      print("Reasons:", reasons_counter)

  # Training loop for each agent
  for agent_i in range(len(game.agents)):
    # Create the mini batches for each agent
    mini_batch = []

    # Populate mini batch
    for _ in range(params["batch_size"]):
      game_i = random.randint(0, len(actions_buffer[agent_i]) - 1)
      snap_i = random.randint(params["memory"], len(actions_buffer[agent_i][game_i]) - 1)
      replay_state = actions_buffer[agent_i][game_i][snap_i]
      mini_batch.append(replay_state)

    # Get training values
    x_train, y_train = process_mini_batch(mini_batch, agent_models[agent_i])

    # Train the model on this batch.
    history = agent_models[agent_i].fit(
        x_train, y_train, batch_size=params["batch_size"],
        epochs=1, verbose=0
    )

  if ((game_count - 1) % log_freq) == 0:
    game_actions = []
    for snap_i in range(len(actions_buffer[0][-1])):
      snap_actions = []
      for agent_i in range(len(game.agents)):
        a = actions_buffer[agent_i][-1][snap_i][1]
        for key, value in a.items():
          a[key] = bool(value)
        snap_actions.append(a)
      
      game_actions.append(snap_actions)
    
    with open("games/game_" + str(game_count) + ".json", "w") as game_json:
      json.dump(game_actions, game_json)