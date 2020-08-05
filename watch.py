from shooter import Game, Agent, Obstacle

import json
import sys

agents = [
  Agent([100, 400]),
  Agent([100, 100])
]

obstacles = [
  Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)]),
  # Obstacle([(250, 250), (300, 250), (500, 300), (400, 400), (250, 300)]),
  # Obstacle([(50, 50), (100, 50), (100, 100), (50, 100)]),
  # Obstacle([(150, 150), (200, 150), (200, 200), (150, 200)])
]

game = Game(agents, obstacles, ui=True, player_control=True)

with open(sys.argv[1]) as json_file:
  data = json.load(json_file)
  game.play_game(data)