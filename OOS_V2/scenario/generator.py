import numpy as np
from config import A, MU, POS_NOISE, VEL_NOISE

def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def generate_scenario():

    v = np.sqrt(MU / A)

    r0 = np.array([A, 0, 0])
    v0 = np.array([0, v, 0])

    sats = []

    # Target
    sats.append({
        "id": "SAT_0",
        "r": r0.copy(),
        "v": v0.copy()
    })

    # Debris (collision pair)
    sats.append({
        "id": "DEBRIS",
        "r": r0 + np.random.normal(0, POS_NOISE, 3),
        "v": v0 + np.random.normal(0, VEL_NOISE, 3)
    })

    # OOS (phase shifted)
    phase = np.random.uniform(0.2, 0.5)
    R = rotation_z(phase)

    sats.append({
        "id": "OOS",
        "r": R @ r0,
        "v": R @ v0
    })

    return sats


def backpropagate(objects, steps=25, dt=-10):
    from physics.propagation import rk4_step

    for _ in range(steps):
        for obj in objects:
            obj["r"], obj["v"] = rk4_step(obj["r"], obj["v"], dt)

    return objects