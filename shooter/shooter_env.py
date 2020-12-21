import numpy as np
import random
import math
import collections
import copy

import gym
from gym.utils import seeding

from shooter.agent import Agent
from shooter.obstacle import Obstacle

from gym.envs.classic_control import rendering

class ShooterEnv(gym.Env):
  def __init__(self, use_ui=False, action_lookup=None, init_agents=None):
    self.viewer = None

    def dist(a, b):
      return (a[0] - b[0])**2 + (a[1] - b[1])**2

    player_pos = [random.randint(50, 590), random.randint(50, 430)]
    other_pos = [random.randint(50, 590), random.randint(50, 430)]
    while dist(player_pos, other_pos) < 40:
      other_pos = [random.randint(50, 590), random.randint(50, 430)]
    
    self.agents = [
      Agent(player_pos, random.uniform(-math.pi, math.pi)),
      Agent(other_pos, random.uniform(-math.pi, math.pi))
    ]

    for agent in self.agents:
      agent.ui_transform = rendering.Transform(translation=(agent.current_position[0], agent.current_position[1]))

    self.obstacles = [
      Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)])
    ]

    self.frame = 0
    self.opp_move = 0

    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  # Resets the game to its start configuration
  def reset(self):
    for agent in self.agents:
      agent.set_position(random.randint(50, 590), random.randint(50, 430))
      agent.set_rotation(random.uniform(-math.pi, math.pi))
      agent.reset_values()

    self.frame = 0

    return self._next_observation(0)

  def _next_observation(self, agent_id):
    return self.agents[agent_id].get_state()

  # Steps the game frame
  def step(self, action, game_time=0):
    rewards = []
    new_states = []

    self.frame += 1

    self.agents[0].start_reward_record()

    # Report the game for the agent
    for i, agent in enumerate(self.agents):
      other_agents = [a for i_, a in enumerate(self.agents) if i != i_]
      agent.report_game(other_agents, self.obstacles, self.frame / 1000)

    # Report the inputs the agent recieved
    if action is not None:
      out = [False, False, False, False, False, False, False]
      if action < len(out):
        out[action] = True
      self.agents[0].report_inputs(out)

    if self.frame % 25 == 0:
      self.opp_move = random.randint(0, 4)

    out = [False, False, False, False, False, False, False]
    out[self.opp_move] = True
    self.agents[1].report_inputs(out)

    # Updates the agent
    for agent in self.agents:
      agent.tick_time()

    # Saves the recorded reward
    agent_reward = self.agents[0].end_reward_record()

    done = min([a.current_health for a in self.agents]) <= 0 or self.frame > 1000
    return self.agents[0].get_state(), agent_reward, done, {}
    
    # for i, (action, agent) in enumerate(zip(actions, self.agents)):
    #   # Reset the agent rewards to keep track of current frame rewards
    #   agent.start_reward_record()

    #   # Report the game for the agent
    #   other_agents = [a for i_, a in enumerate(self.agents) if i != i_]
    #   agent.report_game(other_agents, self.obstacles, game_time)

    #   # Report the inputs the agent recieved
    #   if action is not None:
    #     agent.report_inputs(action)

    #   # Updates the agent
    #   agent.tick_time()

    #   # Saves the recorded reward
    #   agent_reward = agent.end_reward_record()
    #   rewards.append(agent_reward)

    #   # Saves the agent new state
    #   new_states.append(agent.get_state())

    # return new_states[0], rewards[0], self._game_ended(), {}

  def render(self, mode='human'):
    if self.viewer is None:
      self.viewer = rendering.Viewer(640, 480)

      self.agent_transforms = []
      for agent in self.agents:
        a = rendering.make_circle(agent.agent_size)
        a.set_color(0, 0.8, 0.8)
        a.add_attr(agent.ui_transform)
        agent.ui_obj = a

      for obstacle in self.obstacles:
        o = self.viewer.draw_polygon(obstacle.points, filled=False)
        o.set_color(0.4, 0.4, 0.4)
        obstacle.ui_obj = o

    # Move agents and show them
    for agent in self.agents:
      agent.ui_transform.set_translation(agent.current_position[0], agent.current_position[1])
      self.viewer.add_onetime(agent.ui_obj)

    # Show obstacles
    for obstacle in self.obstacles:
      self.viewer.add_onetime(obstacle.ui_obj)

    # Get player agent
    player = self.agents[0]

    # Fov
    raycasts = player.calculate_raycasts()
    for r in raycasts:
      l = self.viewer.draw_line(player.current_position, r["pos"])
      l.set_color(0.4, 0.4, 0.4)
      self.viewer.add_onetime(l)

    # Aim
    pt = player.calculate_raycast_hit(player.current_angle)
    l = self.viewer.draw_line(player.current_position, pt["pos"])
    l.set_color(1, 0, 0)
    self.viewer.add_onetime(l)

    # Health
    for agent in self.agents:
      xl = agent.current_position[0] - 25
      xr = agent.current_position[0] + 25
      xcr = agent.current_position[0] - 25 + 50 * agent.current_health / 100
      yt = agent.current_position[1] + 45
      yb = agent.current_position[1] + 38

      bg_rect = [(xl, yt), (xl, yb), (xr, yb), (xr, yt)]
      overlay_rect = [(xl, yt), (xl, yb), (xcr, yb), (xcr, yt)]

      bg = self.viewer.draw_polygon(bg_rect)
      bg.set_color(1, 0, 0)

      overlay = self.viewer.draw_polygon(overlay_rect)
      overlay.set_color(0, 1, 0)

    # Accuracy
    for agent in self.agents:
      xl = agent.current_position[0] - 25
      xr = agent.current_position[0] + 25
      xcr = agent.current_position[0] - 25 + 50 * agent.current_accuracy
      yt = agent.current_position[1] + 35
      yb = agent.current_position[1] + 28

      bg_rect = [(xl, yt), (xl, yb), (xr, yb), (xr, yt)]
      overlay_rect = [(xl, yt), (xl, yb), (xcr, yb), (xcr, yt)]

      bg = self.viewer.draw_polygon(bg_rect)
      bg.set_color(1, 0, 0)

      overlay = self.viewer.draw_polygon(overlay_rect)
      overlay.set_color(0, 0, 1)

    # Hitbox
    for agent in self.agents:
      a = []
      for hb in agent.hitbox_vertex:
        a.append((hb[0] + agent.current_position[0], hb[1] + agent.current_position[1]))
      hitbox = self.viewer.draw_polygon(a, filled=False)

    return self.viewer.render(return_rgb_array = mode=='rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

