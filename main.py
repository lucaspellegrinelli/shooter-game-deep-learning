import pygame

from game import Game
from agent import Agent
from obstacle import Obstacle

agents = [
  Agent([100, 200]),
  Agent([300, 200])
]

obstacles = [
  Obstacle([
    (250, 250),
    (300, 250),
    (500, 300),
    (400, 400),
    (250, 300)
  ]),
  Obstacle([
    (50, 50),
    (100, 50),
    (100, 100),
    (50, 100)
  ])
]

game = Game(agents, obstacles, True, True)
game.run()