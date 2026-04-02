import numpy as np
from config import COLLISION_RADIUS
from physics.chan_probability import chan_collision_probability

def compute_collision(obj1, obj2):

    rel_r = obj2.r - obj1.r
    rel_v = obj2.v - obj1.v

    cov = np.eye(3) * 100

    pc = chan_collision_probability(
        rel_r, rel_v, cov, COLLISION_RADIUS
    )

    dist = np.linalg.norm(rel_r)

    return pc, dist