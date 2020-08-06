import wandb

from trainer import QTrainer, QTrainerLSTM
from shooter import ShooterEnv

params = {
  "num_inputs": 33, # Number of inputs the agent can take
  "num_actions": 7, # Number of actions the agent can take
  "agent_memory": 30, # Number of previous frames fed to the LSTM
  "seed": 42, # Random seed
  "gamma": 0.99, # Discount factor for past rewards
  "epsilon": 1.0, # Epsilon greedy parameter 
  "epsilon_max": 1.0, # Maximum epsilon greedy parameter
  "epsilon_min": 0.1, # Minimum epsilon greedy parameter
  "epsilon_interval": 0.9, # Rate at which to reduce chance of random action being taken
  "batch_size": 32, # Size of batch taken from replay buffers
  "max_steps_per_episode": 1000, # Max steps per episode
  "epsilon_random_frames": 50000, # Number of frames to take random action and observe output
  "epsilon_greedy_frames": 1000000.0, # Number of frames for exploration
  "max_memory_length": 100000, # Maximum replay length
  "update_after_actions": 4, # Train the model after 4 actions
  "update_target_network": 10000 # How often to update the target network
}

wandb.init(project="shooter-q-learning")

env = ShooterEnv()
trainer = QTrainerLSTM(env, params, use_wandb=True, save_model=False)

while True:
  trainer.iterate()