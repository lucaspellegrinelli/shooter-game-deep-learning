from shooter import Game, Agent, Obstacle

agents = [
  Agent([100, 400]),
  Agent([500, 100])
]

obstacles = [
  Obstacle([(0, 0), (640, 0), (640, 480), (0, 480)]),
  Obstacle([(250, 250), (300, 250), (500, 300), (400, 400), (250, 300)]),
  Obstacle([(50, 50), (100, 50), (100, 100), (50, 100)]),
  Obstacle([(150, 150), (200, 150), (200, 200), (150, 200)])
]

game = Game(agents, obstacles, ui=True, player_control=True)
game.run()