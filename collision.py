# collision.py

import math
import numpy as np
def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2 +
        (p1[2] - p2[2])**2
    )


def closest_approach(traj1, traj2):
    min_dist = float("inf")
    min_step = 0

    for i in range(min(len(traj1), len(traj2))):
        d = distance(traj1[i], traj2[i])

        if d < min_dist:
            min_dist = d
            min_step = i

    # assume each step = Δt seconds
    t_collision = min_step * 15

    return min_dist, min_step, t_collision


def collision_probability(d):
    if d > 5:
        return 0.0
    return np.exp(-d * 2)