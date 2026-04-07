import math
from config import DELTA_T

def distance(p1, p2):
    return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

def closest_approach(traj1, traj2):
    min_d = float("inf")
    min_idx = 0

    for i in range(min(len(traj1), len(traj2))):
        d = distance(traj1[i], traj2[i])
        if d < min_d:
            min_d = d
            min_idx = i

    return min_d, min_idx, min_idx * DELTA_T