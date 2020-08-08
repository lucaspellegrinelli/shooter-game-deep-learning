import wandb
import argparse
import tensorflow as tf

from trainer import QTrainer
from shooter import ShooterEnv

parser = argparse.ArgumentParser(description='Agent Trainer')
parser.add_argument("-model", action="store", dest="modelpath", required=False)
parser.add_argument("-ep", action="store", dest="episode", type=int, required=False)
parser.add_argument("-frame", action="store", dest="frame", type=int, required=False)
parser.add_argument("-epsilon", action="store", dest="epsilon", type=float, required=False)
args = parser.parse_args()

params = {
  "num_inputs": 37, # Number of inputs the agent can take
  "num_actions": 7, # Number of actions the agent can take
  "agent_memory": 150, # Number of previous frames fed to the LSTM
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
  "update_target_network": 25000 # How often to update the target network
}

logistic_params = {
  "use_wandb": False,
  "save_model": False,
  "upload_model": False,
  "save_replays": False
}

if args.modelpath:
  init_params = {
    "modelpath": args.modelpath,
    "episode_count": args.episode,
    "frame_count": args.frame,
    "epsilon": args.epsilon
  }
else:
  init_params = {}

if logistic_params["use_wandb"]:
  wandb.init(project="shooter-q-learning")

env = ShooterEnv()
trainer = QTrainer(env, params, logistic_params, init_params)

with tf.device('/gpu:0'):
  while True:
    trainer.iterate()