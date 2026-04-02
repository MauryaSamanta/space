import numpy as np
from physics.propagation import rk4_step
from config import DELTA_T, MU

# -----------------------------------------
# KEPLERIAN → CARTESIAN (simplified)
# -----------------------------------------
def kepler_to_cartesian(a, e, i, raan, argp, M):
    """
    Convert Keplerian elements → Cartesian state
    Assumes near-circular orbits (as in paper)
    """

    # Mean motion
    n = np.sqrt(MU / a**3)

    # Approximate E = M (low eccentricity)
    E = M

    # Position in orbital plane
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1 - e**2) * np.sin(E)
    z_orb = 0

    # Velocity in orbital plane
    vx_orb = -a * n * np.sin(E)
    vy_orb =  a * n * np.sqrt(1 - e**2) * np.cos(E)
    vz_orb = 0

    r_orb = np.array([x_orb, y_orb, z_orb])
    v_orb = np.array([vx_orb, vy_orb, vz_orb])

    # Rotation matrices
    def Rz(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

    def Rx(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])

    # Apply rotations: RAAN → inclination → argument of perigee
    Q = Rz(raan) @ Rx(i) @ Rz(argp)

    r = Q @ r_orb
    v = Q @ v_orb

    return r, v


# -----------------------------------------
# SAMPLE ORBIT (TABLE 2 FROM PAPER)
# -----------------------------------------
base_a = 7250  # fixed shell
M_base = np.random.uniform(0, 2*np.pi)
def sample_orbit(i, raan,M_ref):
    a = base_a + np.random.uniform(-5, 5)  # max ±5 km
    e = np.random.uniform(0, 0.001)

    argp = 0
    M = M_ref + np.random.uniform(-0.05, 0.05)

    return kepler_to_cartesian(a, e, i, raan, argp, M)


# -----------------------------------------
# GENERATE COLLISION SCENARIO (PAPER STYLE)
# -----------------------------------------
def generate_collision_pair(name, base_i, base_raan,M_ref, back_steps=70):
    """
    EXACTLY following paper:
    - Generate target orbit randomly
    - Create debris at TCA near target
    - Add Gaussian noise (≈50m)
    - Perturb velocity direction
    - Back propagate
    """

    # --- Target satellite ---
    r_t, v_t = sample_orbit(base_i, base_raan, M_ref)

    # --- Debris at TCA (near target) ---
    r_d = r_t + np.random.normal(0, 0.05, size=3)  # 50m ~ 0.05 km

    # Rotate velocity by random angle (encounter geometry)
    angle = np.random.uniform(0.05, np.pi - 0.05)

    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    def rotate(vec, axis, theta):
        return (
            vec * np.cos(theta) +
            np.cross(axis, vec) * np.sin(theta) +
            axis * np.dot(axis, vec) * (1 - np.cos(theta))
        )
    # small velocity perturbation (REAL collision scenario)
    dv = np.random.normal(0, 0.0001, size=3)  # small relative velocity

    v_d = v_t + np.random.normal(0, 0.0001, size=3)

    sat = {"r": r_t.copy(), "v": v_t.copy()}
    debris = {"r": r_d.copy(), "v": v_d.copy()}

    # --- Back propagate ---
    for _ in range(back_steps):
        sat["r"], sat["v"] = rk4_step(sat["r"], sat["v"], -DELTA_T)
        debris["r"], debris["v"] = rk4_step(debris["r"], debris["v"], -DELTA_T)

    return {
        name: sat,
        name + "_DEBRIS": debris
    }


# -----------------------------------------
# FULL SCENARIO (MULTI-SATELLITE)
def generate_scenario_v2(n_satellites=3):

    state = {}

    # 🔥 FIX: shared orbital plane
    base_i = np.random.uniform(0, 2*np.pi)
    base_raan = np.random.uniform(0, 2*np.pi)
    M_base = np.random.uniform(0, 2*np.pi)
    for i in range(n_satellites):
        pair = generate_collision_pair(
            f"SAT_{i}",
            base_i,
            base_raan,
            M_base
        )
        state.update(pair)

    return state, base_i, base_raan


# -----------------------------------------
# DEBUG
# -----------------------------------------
if __name__ == "__main__":
    state = generate_scenario_v2(3)

    for k, v in state.items():
        print(k)
        print("r:", np.round(v["r"], 2))
        print("v:", np.round(v["v"], 4))
        print()