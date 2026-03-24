import numpy as np
import math

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------
STEPS = 300
COLLISION_NOISE_KM = 0.05  # ~50 meters


# -----------------------------
# SATELLITE GENERATION
# -----------------------------
def generate_satellite(name):
    return {
        "name": name,
        "a": np.random.uniform(7000, 8000),   # km
        "e": np.random.uniform(0, 0.003),
        "i": np.random.uniform(0, 2*np.pi),
        "raan": np.random.uniform(0, 2*np.pi),
        "argp": np.random.uniform(0, 2*np.pi),
        "M": np.random.uniform(0, 2*np.pi)
    }


# -----------------------------
# SIMPLE ORBIT PROPAGATION
# (circular approximation)
# -----------------------------
def propagate_orbit(sat, steps=STEPS):
    traj = []

    r = sat["a"]  # radius
    omega = 0.001  # angular velocity (tunable)

    for t in range(steps):
        theta = sat["M"] + omega * t

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0  # keep simple (2D orbit)

        traj.append((x, y, z))

    return traj


# -----------------------------
# GENERATE DEBRIS FOR COLLISION
# -----------------------------
def generate_debris(traj_sat, tca_idx):
    traj_debris = [None] * len(traj_sat)

    # FORCE exact collision at TCA
    p_sat = np.array(traj_sat[tca_idx])
    noise = np.random.normal(0, 0.005, 3)
    traj_debris[tca_idx] = tuple(p_sat + noise)

    # small velocity
    v = np.random.normal(0, 0.001, 3)

    # propagate backward
    current = np.array(traj_debris[tca_idx])
    for i in range(tca_idx - 1, -1, -1):
        current = current - v
        traj_debris[i] = tuple(current)

    # propagate forward
    current = np.array(traj_debris[tca_idx])
    for i in range(tca_idx + 1, len(traj_sat)):
        current = current + v
        traj_debris[i] = tuple(current)

    return traj_debris


# -----------------------------
# GENERATE ONE COLLISION PAIR
# -----------------------------
def generate_collision_pair(name):
    sat = generate_satellite(name)
    traj_sat = propagate_orbit(sat)

    # pick random collision time
    tca_idx = np.random.randint(50, 200)

    traj_debris = generate_debris(traj_sat, tca_idx)

    return {
        "sat_name": name,
        "sat_traj": traj_sat,
        "debris_name": name + "_DEBRIS",
        "debris_traj": traj_debris,
        "tca_idx": tca_idx
    }


# -----------------------------
# GENERATE FULL SCENARIO
# -----------------------------
def generate_scenario(n_satellites=3):
    """
    Returns:
        trajectories: dict{name -> trajectory}
    """

    trajectories = {}

    for i in range(n_satellites):
        pair = generate_collision_pair(f"SAT_{i}")

        trajectories[pair["sat_name"]] = pair["sat_traj"]
        trajectories[pair["debris_name"]] = pair["debris_traj"]

    return trajectories


# -----------------------------
# DEBUG / TEST
# -----------------------------
if __name__ == "__main__":
    scenario = generate_scenario(2)

    for k, v in scenario.items():
        print(k, "trajectory length:", len(v))