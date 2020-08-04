import pygame

class Obstacle:
  def __init__(self, points):
    self.canvas = None
    self.points = points
    self.hitbox_lines = []

    self.calculate_hitbox_lines()

  def calculate_hitbox_lines(self):
    last_vertex = self.points[-1]
    for p in self.points:
      self.hitbox_lines.append({
        "from": last_vertex,
        "to": p
      })

      last_vertex = p

  def draw(self):
    color = (100, 100, 100)

    for hb in self.hitbox_lines:
      pygame.draw.line(self.canvas, color, hb["from"], hb["to"])

  def set_canvas(self, canvas):
    self.canvas = canvas