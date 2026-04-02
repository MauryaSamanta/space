import numpy as np

MU = 3.986e14  # Earth gravitational parameter

DT = 10.0  # seconds
SIM_STEPS = 2000
LAMBERT_TOF = 5000  # seconds
# Orbit
A = 7250e3  # meters

# Constraints
MAX_LAMBERT_DV = 1000  # m/s
MAX_DISTANCE = 1500e3
CW_DISTANCE = 200e3

# Collision
COLLISION_RADIUS = 10.0
PC_THRESHOLD = 1e-4

# Scenario noise
POS_NOISE = 50.0       # meters
VEL_NOISE = 0.001      # m/s