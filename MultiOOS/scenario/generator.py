import numpy as np
from physics.propagation import rk4_step
from config import DELTA_T, MU


# -----------------------------
# CREATE GUARANTEED COLLISION PAIR
# -----------------------------
def generate_collision_pair(name, back_steps=500):
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
# GENERATE FULL SCENARIO (LEGACY)
# -----------------------------
# def generate_scenario(n_satellites=3):
#     state = {}

#     for i in range(n_satellites):
#         pair = generate_collision_pair(f"SAT_{i}")

#         # Directly store state (NO trajectory conversion)
#         state.update(pair)

#     return state

def rotate_z(vec, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R @ vec


def generate_orbit_cluster(n):
    state = {}

    for i in range(n):
        name = f"SAT_{i}"

        # generate collision-consistent pair
        pair = generate_collision_pair(name)

        # apply SAME rotation to both objects
        theta = np.random.uniform(0, 2*np.pi)

        pair[name]["r"] = rotate_z(pair[name]["r"], theta)
        pair[name]["v"] = rotate_z(pair[name]["v"], theta)

        pair[name + "_DEBRIS"]["r"] = rotate_z(pair[name + "_DEBRIS"]["r"], theta)
        pair[name + "_DEBRIS"]["v"] = rotate_z(pair[name + "_DEBRIS"]["v"], theta)

        # ✅ IMPORTANT: add to state
        state.update(pair)

    return state

def generate_scenario(n_satellites=3):
    return generate_orbit_cluster(n_satellites)

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