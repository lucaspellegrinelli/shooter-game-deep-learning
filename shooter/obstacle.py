class Obstacle:
  def __init__(self, points):
    self.points = points
    self.hitbox_lines = []
    self.ui_obj = None

    self.calculate_hitbox_lines()

  def calculate_hitbox_lines(self):
    last_vertex = self.points[-1]
    for p in self.points:
      self.hitbox_lines.append({
        "from": last_vertex,
        "to": p
      })

      last_vertex = p