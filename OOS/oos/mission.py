import numpy as np
from physics.lambert import lambert_transfer
from config import DELTA_T

def generate_oos_trajectory(start_pos, velocity, steps):
    traj = []
    pos = np.array(start_pos)
    vel = np.array(velocity)

    for _ in range(steps):
        pos = pos + vel * DELTA_T
        traj.append(tuple(pos))

    return traj