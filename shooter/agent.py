try:
  import pygame
except: pass

import math
import numpy as np
import random

class Agent:
  def __init__(self, position):
    self.current_position = position
    self.current_speed = [0, 0]
    self.current_angle = 0

    self.current_health = 100
    self.current_bullets = 30
    self.current_accuracy = 1.0

    self.gun_fire_rate = 3 # 3 frames / shot
    self.gun_fire_rate_counter = 0
    self.gun_damage = 10

    self.gun_fire_accuracy_penalty = 0.125 # penalty / shot
    self.gun_frame_accuracy_recovery = 0.02 # recovery / frame

    self.agent_size = 10

    self.hitbox_vertex = [
      (-self.agent_size, -self.agent_size),
      (self.agent_size,-self.agent_size),
      (self.agent_size, self.agent_size),
      (-self.agent_size, self.agent_size)
    ]

    self.hitbox_lines = []

    self.fov_angle = 103 * math.pi / 180
    self.fov_points = 15

    self.angle_speed = 0.05
    self.speed = 2

    self.input_frame_delay = 3 # 3 frames
    self.input_cache = []

    self.map_hitboxes = []

    self.reward = 0

  # Updates the angent positions and calculations for the next frame
  def tick_time(self):
    self.calculate_hitbox_lines()
    self.calculate_raycasts()

    self.move()
    
    if self.gun_fire_rate_counter > 0:
      self.gun_fire_rate_counter -= 1

    self.current_accuracy += self.gun_frame_accuracy_recovery
    if self.current_accuracy > 1.0: self.current_accuracy = 1.0
    if self.current_accuracy < 0.0: self.current_accuracy = 0.0

  # Calculates next position (avoiding clips) and moves the agent
  def move(self):
    # Theoretical next position
    new_pos_x = self.current_position[0] + self.current_speed[0] * self.speed
    new_pos_y = self.current_position[1] + self.current_speed[1] * self.speed

    # Theoretical position delta
    delta_pos_x = new_pos_x - self.current_position[0]
    delta_pos_y = new_pos_y - self.current_position[1]

    # Keeps track if the user has clipped into a wall/other agent
    clipped_x = False
    clipped_y = False

    # For each hitbox in the map
    for hb in self.map_hitboxes:
      if clipped_x and clipped_y: return

      # For each line in the agent hitbox
      for v in self.hitbox_lines:
        # Calculate new position of the hitbox line after moving
        x1 = v["from"][0] + delta_pos_x
        y1 = v["from"][1] + delta_pos_y
        x2 = v["to"][0] + delta_pos_x
        y2 = v["to"][1] + delta_pos_y

        # Calculate if the new lines intercept with the hitbox
        pt_x = self.calculate_line_interception(x1, v["from"][1], x2, v["to"][1],
          hb["from"][0], hb["from"][1], hb["to"][0], hb["to"][1])
        pt_y = self.calculate_line_interception(v["from"][0], y1, v["to"][0], y2,
          hb["from"][0], hb["from"][1], hb["to"][0], hb["to"][1])

        # If the new line position intercepts a hitbox, mark the movement as clipped
        if pt_x["inbounds"]: clipped_x = True
        if pt_y["inbounds"]: clipped_y = True

        if clipped_x and clipped_y: return

    # If the movement is not clipped into another hitbox, update the position
    if not clipped_x: self.current_position[0] = new_pos_x
    if not clipped_y: self.current_position[1] = new_pos_y

  # Logic for firing the agent's gun
  def fire_gun(self):
    # If fire rate has reseted and agent has bullets
    if self.gun_fire_rate_counter == 0 and self.current_bullets > 0:
      # Reset fire rate cooldown
      self.gun_fire_rate_counter = self.gun_fire_rate

      # Decreases accuracy
      self.current_accuracy -= self.gun_fire_accuracy_penalty

      # Decreases bullet count
      self.current_bullets -= 1

      # Hit coorinates of the agent aim raycast 
      pt = self.calculate_raycast_hit(self.current_angle)

      # If hit another agent
      if isinstance(pt["object"], Agent):
        # Updates reward
        self.give_reward("shoot_at_agent")

        # If agent hit shot (based on its accuracy)
        if random.random() < self.current_accuracy:
          # Deals damage to other agent
          pt["object"].take_damage(self.gun_damage)

          # Updates reward
          self.give_reward("hit_agent")

  # Calculate the coordinates of the lines composing the agents hitbox
  def calculate_hitbox_lines(self):
    self.hitbox_lines = []
    last_vertex = self.hitbox_vertex[-1]

    # For each vertex
    for v in self.hitbox_vertex:
      # Creates the from-to coordinates
      last_vx = self.current_position[0] + last_vertex[0]
      last_vy = self.current_position[1] + last_vertex[1]
      vx = self.current_position[0] + v[0]
      vy = self.current_position[1] + v[1]

      # Creates the line object
      self.hitbox_lines.append({
        "from": (last_vx, last_vy),
        "to": (vx, vy)
      })

      # Updates "from" vertex
      last_vertex = v

  # Calculates raycasts representing the agent's field of view
  def calculate_raycasts(self):
    raycast_hits = []

    # Defines the angles of the field of view
    angle_start = self.current_angle - (self.fov_angle / 2) * (1 - 1 / self.fov_points)
    angle_end = self.current_angle + self.fov_angle / 2
    angle_step = self.fov_angle / self.fov_points

    # Keeps track of the closest fov raycast to hit an agent
    closest_angle_to_enemy = None

    # For each raycast angle in the fov
    for a in np.arange(angle_start, angle_end, angle_step):
      # Calculates the point the raycast hits
      pt = self.calculate_raycast_hit(a)
      raycast_hits.append(pt)

      # If the raycast hits an agent
      if isinstance(pt["object"], Agent):
        # Calculate the distance in angles from the raycast to the center of the fov
        a_dist = abs(a - self.current_angle)

        # Updates the closest raycast angle to enemy
        if (closest_angle_to_enemy is None) or (a_dist < closest_angle_to_enemy):
          closest_angle_to_enemy = a_dist

    # If there was a raycast that hit an agent, give a reward based on how far
    # from the center of the fov it was
    if closest_angle_to_enemy is not None:
      self.give_reward("tracking", closest_angle_to_enemy)

    return raycast_hits

  # Calculates the hit position and object of a raycast
  def calculate_raycast_hit(self, angle):
    # Define initial values
    rc_x = self.current_position[0] + 1000 * math.cos(angle)
    rc_y = self.current_position[1] + 1000 * math.sin(angle)
    hit_object = None

    # Vertex of a line from the use to the initial value
    x1 = self.current_position[0]
    y1 = self.current_position[1]
    x2 = rc_x
    y2 = rc_y

    # For line in the map hitbox
    for line in self.map_hitboxes:
      # Define the vertexes of the line
      x3 = line["from"][0]
      y3 = line["from"][1]
      x4 = line["to"][0]
      y4 = line["to"][1]

      # Calculate interception of the hitbox line and the fov line
      pt = self.calculate_line_interception(x1, y1, x2, y2, x3, y3, x4, y4)

      # If the inteception was closer than the closest one yet, set it as the hit
      if pt["inbounds"] and math.hypot(x1 - pt["pos"][0], y1 - pt["pos"][1]) < math.hypot(x1 - rc_x, y1 - rc_y):
        rc_x = pt["pos"][0]
        rc_y = pt["pos"][1]
        hit_object = line["object"]

    # Returns the hit object
    return {
      "pos": (rc_x, rc_y),
      "object": hit_object
    }

  # Calculate the interception of two lines
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

  # Handles damage taken from the agent
  def take_damage(self, damage):
    # Decrease health
    self.current_health -= damage

    # Gives reward (negative)
    self.give_reward("take_damage")

    # Clamps the health to >= 0
    if self.current_health < 0: self.current_health = 0

  # Gets the state of the agent with the infomation he has
  def get_state(self):
    raycasts = self.calculate_raycasts()
    inputs = []
    rc_hit_agent = []

    # Gets the fov distances and if the raycasts hit another agent
    for r in raycasts:
      d = math.hypot(self.current_position[0] - r["pos"][0], self.current_position[1] - r["pos"][1])
      inputs.append(d)
      inputs.append(1 if isinstance(r["object"], Agent) else 0)    

    # Sets another informations he has
    inputs.append(self.current_health / 100)
    inputs.append(self.current_bullets / 30)
    inputs.append(self.gun_fire_rate_counter / self.gun_fire_rate)

    return np.array(inputs)

  # Reports the game to this agent, updating the hitboxes and gametime
  def report_game(self, agents, obstacles):
    self.map_hitboxes = []

    for agent in agents:
      for line in agent.hitbox_lines:
        line["object"] = agent
        self.map_hitboxes.append(line)

    for obstacle in obstacles:
      for line in obstacle.hitbox_lines:
        line["object"] = obstacle
        self.map_hitboxes.append(line)

  # Report the inputs this agent has to take
  def report_inputs(self, inputs):
    self.input_cache.append(inputs)

    if len(self.input_cache) > self.input_frame_delay:
      inp = (np.array(self.input_cache.pop()) == 1)

      if inp[0]: self.current_speed[0] = -self.speed # Left
      if inp[1]: self.current_speed[0] = self.speed # Right
      if inp[2]: self.current_speed[1] = -self.speed # Up
      if inp[3]: self.current_speed[1] = self.speed # Down

      if (not inp[0]) and (not inp[1]): self.current_speed[0] = 0
      if inp[0] and inp[1]: self.current_speed[0] = 0
      if (not inp[2]) and (not inp[3]): self.current_speed[1] = 0
      if inp[2] and inp[3]: self.current_speed[1] = 0

      if inp[4]: self.current_angle -= self.angle_speed # rot_left
      if inp[5]: self.current_angle += self.angle_speed # rot_right
      if inp[6]: self.fire_gun() # fire

  # Gives rewards
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

  def is_dead(self):
    self.current_health <= 0

  def start_reward_record(self):
    self.reward = 0

  def end_reward_record(self):
    return self.reward

  def set_seed(self, seed):
    random.seed(seed)