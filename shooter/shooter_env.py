import numpy as np
import gym
import random
import collections
# import pygame

from shooter.agent import Agent
from shooter.obstacle import Obstacle

class ShooterEnv(gym.Env):
  def __init__(self):
    super(ShooterEnv, self).__init__()

    self.agents = [
      Agent([100, 400]),
      Agent([100, 100])
    ]

    self.obstacles = [
      Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)])
    ]

  # Steps the game frame
  def step(self, actions):
    rewards = []
    new_states = []

    for i, (action, agent) in enumerate(zip(actions, self.agents)):
      # Reset the agent rewards to keep track of current frame rewards
      agent.start_reward_record()

      # Report the game for the agent
      other_agents = [a for i_, a in enumerate(self.agents) if i != i_]
      agent.report_game(other_agents, self.obstacles)

      # Report the inputs the agent recieved
      if action is not None:
        agent.report_inputs(action)

      # Updates the agent
      agent.tick_time()

      # Saves the recorded reward
      agent_reward = agent.end_reward_record()
      rewards.append(agent_reward)

      # Saves the agent new state
      new_states.append(agent.get_state())

    return new_states, rewards, self._game_ended(), {}

  # Resets the game to its start configuration
  def reset(self):
    self.agents = [
      Agent([100, 400]),
      Agent([100, 100])
    ]

    self.obstacles = [
      Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)])
    ]

    return self._next_observation(0)

  # Sets the random seeds
  def seed(self, seed):
    random.seed(seed)
    for i in range(len(self.agents)):
      self.agents[i].set_seed(seed)

  # Render the game in a screen
  def render(self):
    pass

  def _next_observation(self, agent_id):
    return self.agents[agent_id].get_state()

  def _game_ended(self):
    ended = False
    for agent in self.agents:
      if agent.is_dead():
        ended = True
        break

    return ended
  