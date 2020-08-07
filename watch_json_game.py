import json
import time
import sys
import numpy as np

from shooter import ShooterEnv

actions = json.load(open(sys.argv[1], "r"))

def action_lookup(frame):
  global actions
  if frame < len(actions):
    return actions[frame]
  else:
    return None

game = ShooterEnv(use_ui=True, action_lookup=action_lookup)