import numpy as np
from physics.propagation import rk4_step
from config import DELTA_T

def predict_min_distance(obj1, obj2, steps=50):
    r1, v1 = obj1["r"].copy(), obj1["v"].copy()
    r2, v2 = obj2["r"].copy(), obj2["v"].copy()

    min_d = float("inf")

    for _ in range(steps):
        r1, v1 = rk4_step(r1, v1, DELTA_T)
        r2, v2 = rk4_step(r2, v2, DELTA_T)

        d = np.linalg.norm(r1 - r2)
        min_d = min(min_d, d)

    return min_d
def predict_min_distance_with_time(obj1, obj2, steps=200):
    r1, v1 = obj1["r"].copy(), obj1["v"].copy()
    r2, v2 = obj2["r"].copy(), obj2["v"].copy()

    min_d = float("inf")
    min_step = 0
    best_r2 = None

    for i in range(steps):
        r1, v1 = rk4_step(r1, v1, DELTA_T)
        r2, v2 = rk4_step(r2, v2, DELTA_T)

        d = np.linalg.norm(r1 - r2)

        if d < min_d:
            min_d = d
            min_step = i
            best_r2 = r2.copy()

    return min_d, min_step, best_r2
def collision_probability(d):
    if d <= 0:
        return 1.0
    return np.exp(-d * 10)