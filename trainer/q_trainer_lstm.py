import wandb

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from trainer.q_model import QModelLSTM

class QTrainerLSTM:
  def __init__(self, env, params, use_wandb=False, save_model=False):
    # Learner params
    self.params = params

    # Game envirorment
    self.env = env

    # If its going to log into wandb
    self.use_wandb = use_wandb
    self.save_model = save_model

    # Makes the predictions for Q-values which are used to make a action.
    self.model = QModelLSTM(params["num_inputs"], params["agent_memory"], params["num_actions"])

    # Prediction of future rewards. Only updated when target Q-value is stable (after `update_target_network` steps)
    self.model_target = QModelLSTM(params["num_inputs"], params["agent_memory"], params["num_actions"])

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

    # Counters
    self.running_reward = 0
    self.episode_count = 0
    self.frame_count = 0

  def iterate(self):
    # Reset game to initial state
    state = np.array(self.env.reset())
    episode_reward = 0

    # For each step in game
    for timestep in range(1, self.params["max_steps_per_episode"]):
      self.frame_count += 1

      # Use epsilon-greedy for exploration
      if self.frame_count <= self.params["agent_memory"] and \
          (self.frame_count < self.params["epsilon_random_frames"] or \
          self.params["epsilon"] > np.random.rand(1)[0]):
        # Take random action
        r = np.random.choice(self.params["num_actions"])
        action = [tf.one_hot(r, self.params["num_actions"]), None]
      else:
        # Predict action Q-values and take best action
        state_tensor = tf.convert_to_tensor(self.state_history[-self.params["agent_memory"]:])
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        action_i = tf.argmax(action_probs[0]).numpy()
        action = [tf.one_hot(action_i, self.params["num_actions"]), None]

      # Decay probability of taking random action
      self.decay_epsilon()

      # Apply the sampled action in our environment
      state_next_, reward_, done, _ = self.env.step(action)
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
    template = "running reward: {:.2f} at episode {}, frame count {}, epsilon {}"
    print(template.format(self.running_reward,
                          self.episode_count,
                          self.frame_count,
                          self.params["epsilon"]))

    if self.use_wandb:
      wandb.log({
        "running_reward": self.running_reward,
        "episode_count": self.episode_count,
        "frame_count": self.frame_count,
        "epsilon": self.params["epsilon"]
      })

      if self.save_model:
        self.model_target.save("model.h5")
        wandb.save("model.h5")