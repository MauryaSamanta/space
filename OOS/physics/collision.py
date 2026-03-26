import numpy as np
from physics.propagation import rk4_step
from config import DELTA_T

from physics.chan_probability import chan_collision_probability


# -----------------------------
# FIND TCA + RELATIVE STATE
# -----------------------------
def predict_tca_state(obj1, obj2, steps=200):
    r1, v1 = obj1["r"].copy(), obj1["v"].copy()
    r2, v2 = obj2["r"].copy(), obj2["v"].copy()

    min_d = float("inf")
    best_state = None

    for _ in range(steps):
        r1, v1 = rk4_step(r1, v1, DELTA_T)
        r2, v2 = rk4_step(r2, v2, DELTA_T)

        rel_r = r1 - r2
        d = np.linalg.norm(rel_r)

        if d < min_d:
            min_d = d
            best_state = {
                "rel_r": rel_r.copy(),
                "rel_v": (v1 - v2).copy()
            }

    return min_d, best_state


# -----------------------------
# DEFAULT COVARIANCE MODEL
# -----------------------------
def get_default_covariance():
    """
    100 meter uncertainty → 0.1 km
    convert to meters²
    """
    sigma_m = 10.0  # meters
    return np.eye(3) * (sigma_m ** 2)


# -----------------------------
# MAIN COLLISION PROBABILITY
# -----------------------------
def collision_probability(obj1, obj2):
    d_km, state = predict_tca_state(obj1, obj2)

    rel_r_km = state["rel_r"]
    rel_v_km = state["rel_v"]

    # 🔥 convert to meters
    rel_r_m = rel_r_km * 1000.0
    rel_v_m = rel_v_km * 1000.0

    cov_rel = get_default_covariance()

    # collision radius (10m + 10m)
    collision_radius = 20.0  # meters

    Pc = chan_collision_probability(
        rel_r_m,
        rel_v_m,
        cov_rel,
        collision_radius
    )

    return Pc, d_km