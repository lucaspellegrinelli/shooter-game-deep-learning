import pygame

from agent import Agent
from obstacle import Obstacle

class Game:
  def __init__(self, agents, obstacles, ui, player_control):
    self.running = True
    self.fps = 60

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

  def run(self):
    while self.running:
      if self.ui:
        self.process_events(pygame.event.get())
        self.screen.fill((50, 50, 50))
        self.screen.blit(self.update_fps(), (10,0))

        for o in self.obstacles: o.draw()

        if self.player_control:
          self.agents[0].report_inputs(self.key_states)
          self.agents[0].draw_fov()

      for i, agent in enumerate(self.agents):
        agent.tick_time()
        agent.report_game([a for i_, a in enumerate(self.agents) if i != i_], self.obstacles)

        if self.ui:
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