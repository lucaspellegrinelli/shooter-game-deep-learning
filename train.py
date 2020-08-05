import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM

from shooter import Game, Agent, Obstacle

params = {
  "observing_frames": 1000,
  "train_frames": 100000,
  "epsilon": 1,
  "game_buffer": 25,
  "batch_size": 64,
  "frame_back": 120,
  "input_count": 48
}

def create_model():
  model = Sequential()
  model.add(Input(shape=(params["frame_back"], params["input_count"])))
  model.add(LSTM(16))
  model.add(Dense(7, activation='relu'))

  return model

def get_random_action():
  return {
    "left": False,
    "right": False,
    "up": False,
    "down": False,
    "rot_left": False,
    "rot_right": False,
    "fire": False
  }

def create_game():
  agents = [
    Agent([100, 400]),
    Agent([500, 100])
  ]

  obstacles = [
    Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)]),
    Obstacle([(250, 250), (300, 250), (500, 300), (400, 400), (250, 300)]),
    Obstacle([(50, 50), (100, 50), (100, 100), (50, 100)]),
    Obstacle([(150, 150), (200, 150), (200, 200), (150, 200)])
  ]

  return Game(agents, obstacles)

agent_models = [
  create_model(),
  create_model()
]

actions_buffer = [[], []]
error_logger = [[], []]

while True:
  # Game encapsulater
  game = create_game()

  # Get initial state
  _, states = game.step([get_random_action() for _ in range(len(game.agents))])

  # Decrement epsilon over time.
  if params["epsilon"] > 0.1:
    params["epsilon"] -= 1.0 / 50.0

  # Add new game to buffer
  for i in range(len(actions_buffer)):
    actions_buffer[i].append([])
    if len(actions_buffer[i]) > params["game_buffer"]:
      actions_buffer[i].pop()

  while game.running:
    # If observing
    if random.random() < params["epsilon"] or len(actions_buffer[i][-1]) < params["frame_back"]:
      # Take random actions
      actions = [get_random_action() for _ in range(len(game.agents))]
    else:
      # Take NN output actions
      actions = []
      for i in range(len(states)):
        prev_states = [x[0] for x in actions_buffer[i][-1][-params["frame_back"]:]]
        prev_states = np.expand_dims(prev_states, 0)

        pred_actions = agent_models[i].predict(prev_states)[0]
        actions.append({
          "left": pred_actions[0] > 0.5,
          "right": pred_actions[1] > 0.5,
          "up": pred_actions[2] > 0.5,
          "down": pred_actions[3] > 0.5,
          "rot_left": pred_actions[4] > 0.5,
          "rot_right": pred_actions[5] > 0.5,
          "fire": pred_actions[6] > 0.5
        })

    # Take action, observe new state and get agents rewards
    rewards, new_states = game.step(actions)

    # Save the state, actions and reward
    for i in range(len(rewards)):
      actions_buffer[i][-1].append((states[i], actions[i], rewards[i], new_states[i]))

    # Update the starting states.
    states = new_states

  print("Game ended ", game.frame_count)
  print(len(actions_buffer[0]))

  # Training loop for each agent
  for agent_i in range(len(game.agents)):
    # Create the mini batches for each agent
    mini_batch = []

    # Populate mini batch
    for _ in range(params["batch_size"]):
      game_i = random.randint(0, len(actions_buffer[agent_i]) - 1)
      snap_i = random.randint(params["frame_back"], len(actions_buffer[agent_i][game_i]) - 1)

      series = actions_buffer[agent_i][game_i][snap_i - params["frame_back"] : snap_i]
      state_series = [x[0] for x in series]
      following_state_series = [x[3] for x in series]

      taken_action     = actions_buffer[agent_i][game_i][snap_i][1]
      following_reward = actions_buffer[agent_i][game_i][snap_i][2]

      mini_batch.append((state_series, taken_action, following_reward, following_state_series))

    # Get training values
    x_train, y_train = process_mini_batch(mini_batch, model)

    # Train the model on this batch.
    history = LossHistory()
    model.fit(
        x_train, y_train, batch_size=params["batch_size"],
        epochs=1, verbose=0, callbacks=[history]
    )

    error_logger[agent_i].append(history.losses)

  # Creates the training data from mini batch
  def process_mini_batch(mini_batch, model):
    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, params["frame_back"], params["input_count"]))
    actions = np.zeros(shape=(mb_len, 7))
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, params["frame_back"], params["input_count"]))

    # copy stuff

    old_qvals = model.predict(old_states, batch_size=mb_len)
    new_qvals = model.predict(new_states, batch_size=mb_len)

    x_train = old_states
    y_train = old_qvals

    for i in range(len(y_train)):
      for j in range(len(y_train[i])):
        if y_train[i][j] > 0:
          y_train[i][j] = rewards[i] + 0.9 * new_qvals[i][j]

    return x_train, y_train