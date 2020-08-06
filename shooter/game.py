try:
  import pygame
except: pass

from shooter.agent import Agent
from shooter.obstacle import Obstacle
import random

class Game:
  def __init__(self, agents, obstacles, ui=False, player_control=False):
    self.running = True

    random.seed(42)

    self.fps = 30
    self.max_game_time = self.fps * 30 # 30 seconds
    self.frame_count = 0

    self.ui = ui
    self.player_control = player_control

    self.key_states = {
      "left": False,
      "right": False,
      "up": False,
      "down": False,
      "rot_left": False,
      "rot_right": False,
      "fire": False
    }

    self.agents = agents
    self.obstacles = obstacles

    if self.ui:
      pygame.init()
      self.screen = pygame.display.set_mode([640, 480])
      self.clock = pygame.time.Clock()
      self.font = pygame.font.SysFont("Arial", 18)

      for i in range(len(self.agents)): self.agents[i].set_canvas(self.screen)
      for i in range(len(self.obstacles)): self.obstacles[i].set_canvas(self.screen)

  def step(self, agent_actions):
    rewards = []
    rewards_reasons = []
    new_states = []

    self.frame_count += 1

    i = 0
    for action, agent in zip(agent_actions, self.agents):
      agent.reset_reward()
      other_agents = [a for i_, a in enumerate(self.agents) if i != i_]
      agent.report_game(other_agents, self.obstacles, self.frame_count / self.max_game_time)
      agent.report_inputs(action)
      agent.tick_time()

      rewards.append(agent.reward)
      rewards_reasons.append(agent.reward_reasons[:])

      new_states.append(agent.get_state())
      i += 1

      if agent.is_dead() or self.frame_count > self.max_game_time:
        self.running = False
    
    return rewards, rewards_reasons, new_states

  def play_game(self, agent_actions):
    rewards = []
    for _ in range(len(self.agents)): rewards.append(0)

    while self.running:
      if self.frame_count >= self.max_game_time:
        self.running = False

      self.frame_count += 1

      if self.ui:
        self.process_events(pygame.event.get())
        self.screen.fill((50, 50, 50))
        self.screen.blit(self.update_fps(), (10,0))

        for o in self.obstacles: o.draw()

      for i, agent in enumerate(self.agents):
        agent.reset_reward()
        other_agents = [a for i_, a in enumerate(self.agents) if i != i_]
        agent.report_game(other_agents, self.obstacles, self.frame_count / self.max_game_time)
        agent.report_inputs(agent_actions[min(self.frame_count - 1, len(agent_actions))][i])
        agent.tick_time()
  
        rewards[i] += agent.reward

        true_keys = [key for key, item in agent_actions[min(self.frame_count - 1, len(agent_actions))][i].items() if item]
        if len(true_keys) > 0:
          print("Agent", i, "Frame", self.frame_count - 1, "Keys", true_keys, "Reward", agent.reward)
        
        if self.ui:
          agent.draw_fov()
          agent.draw_agent()
          agent.draw_hitbox()
          agent.draw_health_bar()
          agent.draw_accuracy_bar()
          agent.draw_aim()

      if self.ui:
        pygame.display.flip()
        self.clock.tick(self.fps)

    if self.ui:
      pygame.quit()

    print("Total rewards:", rewards)

  def run(self):
    while self.running:
      if self.frame_count >= self.max_game_time:
        self.running = False

      self.frame_count += 1

      if self.ui:
        self.process_events(pygame.event.get())
        self.screen.fill((50, 50, 50))
        self.screen.blit(self.update_fps(), (10,0))

        for o in self.obstacles: o.draw()

      for i, agent in enumerate(self.agents):
        agent.tick_time()
        other_agents = [a for i_, a in enumerate(self.agents) if i != i_]
        agent.report_game(other_agents, self.obstacles, self.frame_count / self.max_game_time)

        if self.ui:
          if i == 0 and self.player_control:
            self.agents[0].report_inputs(self.key_states)
            self.agents[0].draw_fov()

          agent.draw_agent()
          agent.draw_hitbox()
          agent.draw_health_bar()
          agent.draw_accuracy_bar()
          agent.draw_aim()

      if self.ui:
        pygame.display.flip()
        self.clock.tick(self.fps)

    if self.ui:
      pygame.quit()

  def update_fps(self):
    fps = str(int(self.clock.get_fps()))
    return self.font.render(fps, 1, pygame.Color("coral"))

  def process_events(self, events):
    for event in events:
      if event.type == pygame.QUIT:
        self.running = False
      if self.player_control:
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_LEFT: self.key_states["left"] = True
          if event.key == pygame.K_RIGHT: self.key_states["right"] = True
          if event.key == pygame.K_UP: self.key_states["up"] = True
          if event.key == pygame.K_DOWN: self.key_states["down"] = True
          if event.key == pygame.K_a: self.key_states["rot_left"] = True
          if event.key == pygame.K_d: self.key_states["rot_right"] = True
          if event.key == pygame.K_SPACE: self.key_states["fire"] = True
        elif event.type == pygame.KEYUP:
          if event.key == pygame.K_LEFT: self.key_states["left"] = False
          if event.key == pygame.K_RIGHT: self.key_states["right"] = False
          if event.key == pygame.K_UP: self.key_states["up"] = False
          if event.key == pygame.K_DOWN: self.key_states["down"] = False
          if event.key == pygame.K_a: self.key_states["rot_left"] = False
          if event.key == pygame.K_d: self.key_states["rot_right"] = False
          if event.key == pygame.K_SPACE: self.key_states["fire"] = False