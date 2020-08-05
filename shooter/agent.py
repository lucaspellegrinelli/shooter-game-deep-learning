try:
  import pygame
except: pass

import math
import numpy as np
import random

class Agent:
  def __init__(self, position):
    self.canvas = None
    self.framerate = 30

    self.current_position = position
    self.current_speed = [0, 0]
    self.current_angle = 0

    self.current_health = 100

    self.current_bullets = 30

    self.current_walk_accuracy = 1.0
    self.current_accuracy = 1.0

    self.gun_fire_rate = self.framerate / 10 # 0.1 sec / shot
    self.gun_fire_rate_counter = 0
    self.gun_damage = 10

    self.gun_fire_accuracy_penalty = 0.125
    self.gun_frame_accuracy_recovery = 0.01 / (self.framerate / 60)

    self.agent_size = 10

    self.hitbox_vertex = [
      (-self.agent_size, -self.agent_size),
      (self.agent_size,-self.agent_size),
      (self.agent_size, self.agent_size),
      (-self.agent_size, self.agent_size)
    ]

    self.hitbox_lines = []

    self.fov_angle = 103 * math.pi / 180
    self.fov_points = 45

    self.angle_speed = 0.05
    self.speed = 2

    self.input_frame_delay = self.framerate / 10 # 0.1 sec
    self.input_cache = []

    self.vision_frame_delay = self.framerate / 10 # 0.1 sec
    self.vision_cache = []

    self.map_hitboxes = []

    self.time_progress = 0

    self.reward = 0

  def tick_time(self):
    self.calculate_hitbox_lines()
    self.calculate_raycasts()

    self.move()
    
    if self.gun_fire_rate_counter > 0:
      self.gun_fire_rate_counter -= 1

    self.current_accuracy += self.gun_frame_accuracy_recovery
    if self.current_accuracy > 1.0: self.current_accuracy = 1.0
    if self.current_accuracy < 0.0: self.current_accuracy = 0.0

  def move(self):
    new_pos_x = self.current_position[0] + self.current_speed[0] * self.speed
    new_pos_y = self.current_position[1] + self.current_speed[1] * self.speed

    delta_pos_x = new_pos_x - self.current_position[0]
    delta_pos_y = new_pos_y - self.current_position[1]

    clipped_x = False
    clipped_y = False

    for hb in self.map_hitboxes:
      if clipped_x and clipped_y: return

      for v in self.hitbox_lines:
        x1 = v["from"][0] + delta_pos_x
        y1 = v["from"][1] + delta_pos_y
        x2 = v["to"][0] + delta_pos_x
        y2 = v["to"][1] + delta_pos_y

        pt_x = self.calculate_line_interception(x1, v["from"][1], x2, v["to"][1],
          hb["from"][0], hb["from"][1], hb["to"][0], hb["to"][1])
        pt_y = self.calculate_line_interception(v["from"][0], y1, v["to"][0], y2,
          hb["from"][0], hb["from"][1], hb["to"][0], hb["to"][1])

        if pt_x["inbounds"]: clipped_x = True
        if pt_y["inbounds"]: clipped_y = True

        if clipped_x and clipped_y: return

    if not clipped_x: self.current_position[0] = new_pos_x
    if not clipped_y: self.current_position[1] = new_pos_y

    if (not clipped_x and self.current_speed[0] != 0) or (not clipped_y and self.current_speed[1] != 0):
      self.current_walk_accuracy = 0.5
    else:
      self.current_walk_accuracy = 1.0

  def fire_gun(self):
    if self.gun_fire_rate_counter == 0 and self.current_bullets > 0:
      self.gun_fire_rate_counter = self.gun_fire_rate
      self.current_accuracy -= self.gun_fire_accuracy_penalty
      self.current_bullets -= 1

      pt = self.calculate_raycast_hit(self.current_angle)
      if isinstance(pt["object"], Agent):
        self.give_reward("shoot_at_agent")
        if random.random() < self.current_accuracy * self.current_walk_accuracy:
          pt["object"].take_damage(self.gun_damage)
          self.give_reward("hit_agent")

  def calculate_hitbox_lines(self):
    self.hitbox_lines = []
    last_vertex = self.hitbox_vertex[-1]

    for v in self.hitbox_vertex:
      last_vx = self.current_position[0] + last_vertex[0]
      last_vy = self.current_position[1] + last_vertex[1]
      vx = self.current_position[0] + v[0]
      vy = self.current_position[1] + v[1]

      self.hitbox_lines.append({
        "from": (last_vx, last_vy),
        "to": (vx, vy)
      })

      last_vertex = v

  def calculate_raycasts(self):
    raycast_hits = []

    angle_start = self.current_angle - (self.fov_angle / 2) * (1 - 1 / self.fov_points)
    angle_end = self.current_angle + self.fov_angle / 2
    angle_step = self.fov_angle / self.fov_points

    closest_angle_to_enemy = None
    for a in np.arange(angle_start, angle_end, angle_step):
      pt = self.calculate_raycast_hit(a)
      raycast_hits.append(pt)

      if isinstance(pt["object"], Agent):
        a_dist = abs(a - self.current_angle)
        if (closest_angle_to_enemy is None) or (a_dist < closest_angle_to_enemy):
          closest_angle_to_enemy = a_dist

    if closest_angle_to_enemy is not None:
      self.give_reward("tracking", closest_angle_to_enemy)

    return raycast_hits

  def calculate_raycast_hit(self, angle):
    rc_x = self.current_position[0] + 1000 * math.cos(angle)
    rc_y = self.current_position[1] + 1000 * math.sin(angle)
    hit_object = None

    x1 = self.current_position[0]
    y1 = self.current_position[1]
    x2 = rc_x
    y2 = rc_y

    for line in self.map_hitboxes:
      x3 = line["from"][0]
      y3 = line["from"][1]
      x4 = line["to"][0]
      y4 = line["to"][1]

      try:
        pt = self.calculate_line_interception(x1, y1, x2, y2, x3, y3, x4, y4)
        if pt["inbounds"] and math.hypot(x1 - pt["pos"][0], y1 - pt["pos"][1]) < math.hypot(x1 - rc_x, y1 - rc_y):
          rc_x = pt["pos"][0]
          rc_y = pt["pos"][1]
          hit_object = line["object"]
      except: pass

    return {
      "pos": (rc_x, rc_y),
      "object": hit_object
    }

  def calculate_line_interception(self, x1, y1, x2, y2, x3, y3, x4, y4):
    try:
      t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
      u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
      px = x1 + t * (x2 - x1)
      py = y1 + t * (y2 - y1)

      return {
        "pos": (px, py),
        "inbounds": t >= 0 and t <= 1 and u >= 0 and u <= 1
      }
    except:
      return {
        "pos": (0, 0),
        "inbounds": False
      }

  def take_damage(self, damage):
    self.current_health -= damage
    self.give_reward("take_damage")
    if self.current_health < 0: self.current_health = 0

  def report_game(self, agents, obstacles, game_time):
    self.map_hitboxes = []
    self.time_progress = game_time

    for agent in agents:
      for line in agent.hitbox_lines:
        line["object"] = agent
        self.map_hitboxes.append(line)

    for obstacle in obstacles:
      for line in obstacle.hitbox_lines:
        line["object"] = obstacle
        self.map_hitboxes.append(line)

  def get_state(self):
    raycasts = self.calculate_raycasts()
    inputs = []
    rc_hit_agent = []

    for r in raycasts:
      d = math.hypot(self.current_position[0] - r["pos"][0], self.current_position[1] - r["pos"][1])
      inputs.append(d)
      inputs.append(1 if isinstance(r["object"], Agent) else 0)    

    inputs.append(self.current_health / 100)
    inputs.append(self.current_bullets / 30)
    inputs.append(self.gun_fire_rate_counter / self.gun_fire_rate)
    inputs.append(self.time_progress)

    return np.array(inputs)

  def report_inputs(self, inputs):
    self.input_cache.append(inputs)

    if len(self.input_cache) > self.input_frame_delay:
      inp = self.input_cache.pop()

      if inp["left"]: self.current_speed[0] = -self.speed
      if inp["right"]: self.current_speed[0] = self.speed
      if inp["up"]: self.current_speed[1] = -self.speed
      if inp["down"]: self.current_speed[1] = self.speed

      if (not inp["left"]) and (not inp["right"]): self.current_speed[0] = 0
      if inp["left"] and inp["right"]: self.current_speed[0] = 0
      if (not inp["up"]) and (not inp["down"]): self.current_speed[1] = 0
      if inp["up"] and inp["down"]: self.current_speed[1] = 0

      if inp["rot_left"]: self.current_angle -= self.angle_speed
      if inp["rot_right"]: self.current_angle += self.angle_speed
      if inp["fire"]: self.fire_gun()

  def draw_agent(self):
    color = (255, 255, 255)
    pygame.draw.circle(self.canvas, color, self.current_position, self.agent_size)

  def draw_aim(self):
    pt = self.calculate_raycast_hit(self.current_angle)

    if self.gun_fire_rate_counter > 1:
      line_color = (0, 0, 255)
    else:
      line_color = (255, 0, 0)

    pygame.draw.line(self.canvas, line_color, self.current_position, pt["pos"])

  def draw_hitbox(self):
    last_vertex = self.hitbox_vertex[-1]
    for v in self.hitbox_vertex:
      last_vx = self.current_position[0] + last_vertex[0]
      last_vy = self.current_position[1] + last_vertex[1]
      vx = self.current_position[0] + v[0]
      vy = self.current_position[1] + v[1]

      pygame.draw.line(self.canvas, (100, 100, 100), (last_vx, last_vy), (vx, vy))
      last_vertex = v
    
  def draw_fov(self):
    raycasts = self.calculate_raycasts()
    for r in raycasts:
      pygame.draw.line(self.canvas, (100, 100, 100), self.current_position, r["pos"])

      if isinstance(r["object"], Agent):
        color = (255, 0, 0)
        position = (int(r["pos"][0]), int(r["pos"][1]))
        pygame.draw.circle(self.canvas, color, position, 1)

  def draw_health_bar(self):
    bg_color = (255, 0, 0)
    bg_rect = (self.current_position[0] - 20, self.current_position[1] - 20, 40, 5)
    
    overlay_color = (0, 255, 0)
    overlay_rect = (self.current_position[0] - 20, self.current_position[1] - 20, 40 * self.current_health / 100, 5)
    
    pygame.draw.rect(self.canvas, bg_color, bg_rect)

    if self.current_health > 0:
      pygame.draw.rect(self.canvas, overlay_color, overlay_rect)

  def draw_accuracy_bar(self):
    bg_color = (255, 0, 0)
    bg_rect = (self.current_position[0] - 20, self.current_position[1] - 30, 40, 5)

    overlay_color = (0, 0, 255)
    overlay_rect = (self.current_position[0] - 20, self.current_position[1] - 30, 40 * self.current_accuracy, 5)

    pygame.draw.rect(self.canvas, bg_color, bg_rect)

    if self.current_accuracy > 0:
      pygame.draw.rect(self.canvas, overlay_color, overlay_rect)

  def set_canvas(self, canvas):
    self.canvas = canvas

  def is_dead(self):
    self.current_health <= 0

  def give_reward(self, label, info=0.0):
    if label == "shoot_at_agent":
      self.reward += 200
    elif label == "hit_agent":
      self.reward += 300
    elif label == "tracking":
      self.reward += 1.0 * (1 - abs(info))
    elif label == "take_damage":
      self.reward -= 100
    else:
      self.reward = 0

  def reset_reward(self):
    self.reward = 0