import json
import time
import sys
import numpy as np

from shooter import ShooterEnv
from shooter import Agent

actions = json.load(open(sys.argv[1], "r"))

def action_lookup(frame):
  global actions
  if frame < len(actions["actions"]):
    return actions["actions"][frame]
  else:
    return None

agents = [
  Agent(actions["players"][0][:2], actions["players"][0][2]),
  Agent(actions["players"][1][:2], actions["players"][1][2])
]

print("Reported rewards:", actions["rewards"])
game = ShooterEnv(use_ui=True, action_lookup=action_lookup, init_agents=agents)