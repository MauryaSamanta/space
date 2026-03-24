import numpy as np
from physics.propagation import rk4_step
from config import DELTA_T, MU


# -----------------------------
# CREATE GUARANTEED COLLISION PAIR
# -----------------------------
def generate_collision_pair(name, back_steps=200):
    """
    Create satellite + debris that WILL collide in future
    using backward propagation from collision point
    """

    # Step 1: exact collision point (TCA)
    r_collision = np.array([
        np.random.uniform(7000, 8000),
        0,
        0
    ])

    # Step 2: circular orbit velocity
    v_mag = np.sqrt(MU / np.linalg.norm(r_collision))

    # VERY IMPORTANT: tiny velocity difference (NOT 2%)
    v_sat = np.array([0, v_mag, 0])
    v_debris = v_sat + np.array([0, 0.001, 0])  # small perturbation

    sat = {"r": r_collision.copy(), "v": v_sat.copy()}
    debris = {"r": r_collision.copy(), "v": v_debris.copy()}

    # Step 3: propagate backward in time
    for _ in range(back_steps):
        sat["r"], sat["v"] = rk4_step(sat["r"], sat["v"], -DELTA_T)
        debris["r"], debris["v"] = rk4_step(debris["r"], debris["v"], -DELTA_T)

    return {
        name: sat,
        name + "_DEBRIS": debris
    }


# -----------------------------
# GENERATE FULL SCENARIO
# -----------------------------
def generate_scenario(n_satellites=3):
    state = {}

    for i in range(n_satellites):
        pair = generate_collision_pair(f"SAT_{i}")

        # Directly store state (NO trajectory conversion)
        state.update(pair)

    return state


# -----------------------------
# DEBUG
# -----------------------------
if __name__ == "__main__":
    s = generate_scenario(1)

    for k, v in s.items():
        print(k)
        print("Position:", np.round(v["r"], 2))
        print("Velocity:", np.round(v["v"], 4))
        print()