import wandb
import json
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from trainer.q_model import QModel

# Limiting the number of cores used while running this code
cores = 8
tf.config.threading.set_intra_op_parallelism_threads(cores)
tf.config.threading.set_inter_op_parallelism_threads(cores)

class QTrainer:
  def __init__(self, env, params, logistic_params, init_params):
    # Learner params
    self.params = params

    # Logging parameters
    self.logistic_params = logistic_params

    # Game envirorment
    self.env = env

    # Makes the predictions for Q-values which are used to make a action.
    self.model = QModel(params["num_inputs"], params["agent_memory"], params["num_actions"])

    # Prediction of future rewards. Only updated when target Q-value is stable (after `update_target_network` steps)
    self.model_target = QModel(params["num_inputs"], params["agent_memory"], params["num_actions"])

    # The neural network optimizer
    self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    # The neural network loss function
    self.loss_function = keras.losses.Huber()

    # Experience replay buffers
    self.action_history = []
    self.state_history = []
    self.state_next_history = []
    self.rewards_history = []
    self.done_history = []
    self.episode_reward_history = []

    self.last_game_actions = []

    # Counters
    self.running_reward = 0
    self.running_reward_std = 0
    self.running_reward_median = 0
    self.episode_count = 0
    self.frame_count = 0

    if "episode_count" in init_params: self.episode_count = init_params["episode_count"]
    if "epsilon" in init_params: self.params["epsilon"] = init_params["epsilon"]
    if "modelpath" in init_params:
      self.model.load_weights(init_params["modelpath"])
      self.model_target.load_weights(init_params["modelpath"])

  def iterate(self):
    # Reset game to initial state
    current_game_actions = {"players": [], "actions": [], "rewards": []}

    state = np.array(self.env.reset())
    episode_reward = 0

    for agent in self.env.agents:
      current_game_actions["rewards"].append(0)
      current_game_actions["players"].append([
        agent.current_position[0],
        agent.current_position[1],
        agent.current_angle
      ])

    # For each step in game
    for timestep in range(1, self.params["max_steps_per_episode"]):
      self.frame_count += 1

      # Use epsilon-greedy for exploration
      if self.frame_count <= self.params["agent_memory"] and \
          self.frame_count < self.params["epsilon_random_frames"] or \
          self.params["epsilon"] > np.random.rand(1)[0]:
        # Take random action
        r0 = np.random.choice(self.params["num_actions"])
        a0 = [1 if x == r0 else 0 for x in range(self.params["num_actions"])]

        r1 = np.random.choice(4)
        a1 = [1 if x == r1 else 0 for x in range(self.params["num_actions"])]

        action = [a0, a1]
      else:
        # Predict action Q-values and take best action
        state_tensor = tf.convert_to_tensor(self.state_history[-self.params["agent_memory"]:])
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)

        action_i = tf.argmax(action_probs[0]).numpy()
        a_i = [1 if x == action_i else 0 for x in range(self.params["num_actions"])]

        r_n = np.random.choice(4)
        r_a = [1 if x == r_n else 0 for x in range(self.params["num_actions"])]

        action = [a_i, r_a]

      # Decay probability of taking random action
      self.decay_epsilon()

      # Apply the sampled action in our environment
      game_time = (timestep - 1) / self.params["max_steps_per_episode"]
      current_game_actions["actions"].append(action)
      state_next_, reward_, done, _ = self.env.step(action, game_time)
      for i, r in enumerate(reward_): current_game_actions["rewards"][i] += r
      reward = reward_[0]
      state_next = np.array(state_next_[0])
      episode_reward += reward

      # Save actions and states in replay buffer
      self.action_history.append(action[0])
      self.state_history.append(state)
      self.state_next_history.append(state_next)
      self.done_history.append(done)
      self.rewards_history.append(reward)
      state = state_next

      # Update every fourth frame and once batch size is over 32
      if self.frame_count % self.params["update_after_actions"] == 0:
        if len(self.done_history) > max(self.params["batch_size"], self.params["agent_memory"]):
          self.update_model()

      # Update every 10000th frame
      if self.frame_count % self.params["update_target_network"] == 0:
        self.update_target_model()

      # Limit the state and reward history
      if len(self.rewards_history) > self.params["max_memory_length"]:
        del self.rewards_history[:1]
        del self.state_history[:1]
        del self.state_next_history[:1]
        del self.action_history[:1]
        del self.done_history[:1]

      # If game has ended, break loop
      if done:
        break

    # Update last game actions
    self.last_game_actions = current_game_actions

    # Update running reward to check condition for solving
    self.update_running_reward(episode_reward)

    # Increate episode counter
    self.episode_count += 1

  def decay_epsilon(self):
    self.params["epsilon"] -= self.params["epsilon_interval"] / self.params["epsilon_greedy_frames"]
    self.params["epsilon"] = max(self.params["epsilon"], self.params["epsilon_min"])

  def update_running_reward(self, episode_reward):
    self.episode_reward_history.append(episode_reward)

    if len(self.episode_reward_history) > 100:
      del self.episode_reward_history[:1]

    self.running_reward = np.mean(self.episode_reward_history)
    self.running_reward_std = np.std(self.episode_reward_history)
    self.running_reward_median = np.median(self.episode_reward_history)

  def update_model(self):
    # Get indices of samples for replay buffers
    ind_min = self.params["agent_memory"]
    ind_max = len(self.done_history)
    indices = np.random.choice(range(ind_min, ind_max), size=self.params["batch_size"])

    # Using list comprehension to sample from replay buffer
    state_sample = np.array(
      [self.state_history[i - self.params["agent_memory"] + 1 : i + 1] for i in indices]
    )
    state_next_sample = np.array(
      [self.state_next_history[i - self.params["agent_memory"] + 1 : i + 1] for i in indices]
    )
    rewards_sample = [self.rewards_history[i] for i in indices]
    action_sample = [self.action_history[i] for i in indices]
    done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in indices])

    # Build the updated Q-values for the sampled future states
    # Use the target model for stability
    future_rewards = self.model_target.predict(state_next_sample)

    # Q value = reward + discount factor * expected future reward
    updated_q_values = rewards_sample + self.params["gamma"] * tf.reduce_max(future_rewards, axis=1)

    # If final frame set the last value to -1
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

    # Create a mask so we only calculate loss on the updated Q-values
    masks = np.array(action_sample) # tf.one_hot(action_sample, num_actions)

    with tf.GradientTape() as tape:
      # Train the model on the states and updated Q-values
      q_values = self.model(state_sample)

      # Apply the masks to the Q-values to get the Q-value for action taken
      q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

      # Calculate loss between new Q-value and old Q-value
      loss = self.loss_function(updated_q_values, q_action)

    # Backpropagation
    grads = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

  def update_target_model(self):
    # Update the the target network with new weights
    self.model_target.set_weights(self.model.get_weights())

    # Log details
    self.log_progress()
    
  def log_progress(self):
    # Print info
    template = "{} running reward: {:.2f} at episode {}, frame count {}, epsilon {}"
    print(template.format(str(datetime.now()), self.running_reward, self.episode_count,
                          self.frame_count, self.params["epsilon"]))

    # Saving / Wandb logging
    model_name = "models/model_{}_{:.3f}.h5".format(self.episode_count, self.running_reward)
    game_name = "games/game_{}_{:.3f}.json".format(self.episode_count, self.running_reward)

    if self.logistic_params["save_model"]:
      os.makedirs(os.path.dirname(model_name), exist_ok=True)
      self.model_target.save_weights(model_name)

    if self.logistic_params["save_replays"]:
      os.makedirs(os.path.dirname(game_name), exist_ok=True)
      with open(game_name, "w") as outfile:
        json.dump(self.last_game_actions, outfile)

    if self.logistic_params["use_wandb"]:
      wandb.log({
        "running_reward": self.running_reward,
        "running_reward_std": self.running_reward_std,
        "running_reward_median": self.running_reward_median,
        "episode_count": self.episode_count,
        "frame_count": self.frame_count,
        "epsilon": self.params["epsilon"]
      })

      if self.logistic_params["upload_model"] and self.logistic_params["save_model"]:
        wandb.save(model_name)