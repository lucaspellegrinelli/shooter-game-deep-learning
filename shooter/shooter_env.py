import numpy as np
import gym
import random
import math
import collections
import copy

try: import pygame
except: pass

from shooter.agent import Agent
from shooter.obstacle import Obstacle

class ShooterEnv(gym.Env):
  def __init__(self, use_ui=False, action_lookup=None, init_agents=None):
    super(ShooterEnv, self).__init__()
    self.use_ui = use_ui
    self.action_lookup = action_lookup

    # Create agents and obstacles
    self.reset()

    if init_agents is not None:
      self.agents = init_agents

    if self.use_ui:
      pygame.init()
      self.screen = pygame.display.set_mode([640, 480])
      self.clock = pygame.time.Clock()
      self.font = pygame.font.SysFont("Arial", 18)
      self.ui_running = True
      self._render_loop()

  # Steps the game frame
  def step(self, actions, game_time):
    rewards = []
    new_states = []

    for i, (action, agent) in enumerate(zip(actions, self.agents)):
      # Reset the agent rewards to keep track of current frame rewards
      agent.start_reward_record()

      # Report the game for the agent
      other_agents = [a for i_, a in enumerate(self.agents) if i != i_]
      agent.report_game(other_agents, self.obstacles, game_time)

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
    agent_positions = [
      [random.randint(50, 590), random.randint(50, 430)],
      [random.randint(50, 590), random.randint(50, 430)]
    ]

    self.agents = [
      Agent(agent_positions[0], random.uniform(-math.pi, math.pi)),
      Agent(agent_positions[1], random.uniform(-math.pi, math.pi))
    ]

    self.obstacles = [
      Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)])
    ]

    return self._next_observation(0)

  def _next_observation(self, agent_id):
    return self.agents[agent_id].get_state()

  def _game_ended(self):
    ended = False
    for agent in self.agents:
      if agent.is_dead():
        ended = True
        break

    return ended

  # Render the game in a screen
  def _render_loop(self):
    if not self.use_ui: return

    # Drawing the obstacles
    def draw_obstacle(obstacle):
      color = (100, 100, 100)
      for hb in obstacle.hitbox_lines:
        pygame.draw.line(self.screen, color, hb["from"], hb["to"])

    # Drawing agents
    def draw_agent(agent, hitbox=True, aim=True, fov=False):
      color = (255, 255, 255)
      pygame.draw.circle(self.screen, color, agent.current_position, agent.agent_size)

      if hitbox:
        last_vertex = agent.hitbox_vertex[-1]
        for v in agent.hitbox_vertex:
          last_vx = agent.current_position[0] + last_vertex[0]
          last_vy = agent.current_position[1] + last_vertex[1]
          vx = agent.current_position[0] + v[0]
          vy = agent.current_position[1] + v[1]

          pygame.draw.line(self.screen, (100, 100, 100), (last_vx, last_vy), (vx, vy))
          last_vertex = v

      if fov:
        raycasts = agent.calculate_raycasts()
        for r in raycasts:
          pygame.draw.line(self.screen, (100, 100, 100), agent.current_position, r["pos"])

          if isinstance(r["object"], Agent):
            color = (255, 0, 0)
            position = (int(r["pos"][0]), int(r["pos"][1]))
            pygame.draw.circle(self.screen, color, position, 1)

      if aim:
        pt = agent.calculate_raycast_hit(agent.current_angle)
        if agent.gun_fire_rate_counter > 1: line_color = (0, 0, 255)
        else: line_color = (255, 0, 0)
        pygame.draw.line(self.screen, line_color, agent.current_position, pt["pos"])

    # Drawing bars
    def draw_bars(agent, health=True, accuracy=True):
      if health:
        bg_color = (255, 0, 0)
        bg_rect = (agent.current_position[0] - 20, agent.current_position[1] - 20, 40, 5)
        overlay_color = (0, 255, 0)
        overlay_rect = (agent.current_position[0] - 20, agent.current_position[1] - 20, 40 * agent.current_health / 100, 5)
        
        pygame.draw.rect(self.screen, bg_color, bg_rect)

        if agent.current_health > 0:
          pygame.draw.rect(self.screen, overlay_color, overlay_rect)

      if accuracy:
        bg_color = (255, 0, 0)
        bg_rect = (agent.current_position[0] - 20, agent.current_position[1] - 30, 40, 5)
        overlay_color = (0, 0, 255)
        overlay_rect = (agent.current_position[0] - 20, agent.current_position[1] - 30, 40 * agent.current_accuracy, 5)

        pygame.draw.rect(self.screen, bg_color, bg_rect)

        if agent.current_accuracy > 0:
          pygame.draw.rect(self.screen, overlay_color, overlay_rect)
          
    # ----- END OF DRAWING FUNCTIONS ------

    rewards = [0 for _ in range(2)]
    frame = 0
    while self.ui_running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.ui_running = False

      action = self.action_lookup(frame)
      if action is None:
        break
      else:
        _, reward_, done, _ = self.step(action, frame / 1000)
        if done: self.ui_running = False
        for i, r in enumerate(reward_): rewards[i] += r

      self.screen.fill((50, 50, 50))
      for o in self.obstacles: draw_obstacle(o)
      for a in self.agents:
        draw_agent(a, hitbox=True, aim=True, fov=True)
        draw_bars(a, health=True, accuracy=True)

      pygame.display.flip()
      self.clock.tick(30)
      frame += 1

    print("Simulated rewards:", rewards)

