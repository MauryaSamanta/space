import numpy as np
from config import MU


# -----------------------------------------
# MEAN MOTION
# -----------------------------------------
def mean_motion(r_target):
    r_norm = np.linalg.norm(r_target)
    return np.sqrt(MU / r_norm**3)


# -----------------------------------------
# CW STATE TRANSITION MATRICES
# -----------------------------------------
def cw_matrices(n, t):
    ct = np.cos(n * t)
    st = np.sin(n * t)

    Phi_rr = np.array([
        [4 - 3 * ct, 0, 0],
        [6 * (st - n * t), 1, 0],
        [0, 0, ct]
    ])

    Phi_rv = np.array([
        [st / n, 2 * (1 - ct) / n, 0],
        [-2 * (1 - ct) / n, (4 * st - 3 * n * t) / n, 0],
        [0, 0, st / n]
    ])

    Phi_vr = np.array([
        [3 * n * st, 0, 0],
        [6 * n * (ct - 1), 0, 0],
        [0, 0, -n * st]
    ])

    Phi_vv = np.array([
        [ct, 2 * st, 0],
        [-2 * st, 4 * ct - 3, 0],
        [0, 0, ct]
    ])

    return Phi_rr, Phi_rv, Phi_vr, Phi_vv


# -----------------------------------------
# RELATIVE STATE (ECI)
# -----------------------------------------
def get_relative_state(oos, target):
    r_rel = oos["r"] - target["r"]
    v_rel = oos["v"] - target["v"]
    return r_rel, v_rel


# -----------------------------------------
# ECI → LVLH TRANSFORMATION
# -----------------------------------------
def eci_to_lvlh(r_target, v_target, r_rel, v_rel):
    """
    Convert relative state from ECI frame to LVLH frame
    """

    # Radial direction (x-axis)
    x_hat = r_target / np.linalg.norm(r_target)

    # Orbit normal (z-axis)
    h = np.cross(r_target, v_target)
    z_hat = h / np.linalg.norm(h)

    # Along-track direction (y-axis)
    y_hat = np.cross(z_hat, x_hat)

    # Rotation matrix (ECI → LVLH)
    R = np.vstack((x_hat, y_hat, z_hat)).T

    r_lvlh = R.T @ r_rel
    v_lvlh = R.T @ v_rel

    return r_lvlh, v_lvlh, R


# -----------------------------------------
# LVLH → ECI TRANSFORMATION
# -----------------------------------------
def lvlh_to_eci(vec_lvlh, R):
    return R @ vec_lvlh


# -----------------------------------------
# CW RENDEZVOUS SOLVER (LVLH FRAME)
# -----------------------------------------
def cw_rendezvous_delta_v(r_lvlh, v_lvlh, r_target_norm, t):

    n = np.sqrt(MU / r_target_norm**3)

    Phi_rr, Phi_rv, _, _ = cw_matrices(n, t)

    # Solve: 0 = Phi_rr * r + Phi_rv * v_required
    v_required = np.linalg.solve(Phi_rv, -Phi_rr @ r_lvlh)

    dv_lvlh = v_required - v_lvlh

    return dv_lvlh


# -----------------------------------------
# MAIN CW TRANSFER FUNCTION
# -----------------------------------------
def cw_transfer(oos, target, tof):

    # Step 1: Relative state in ECI
    r_rel_eci, v_rel_eci = get_relative_state(oos, target)

    # Step 2: Convert to LVLH
    r_lvlh, v_lvlh, R = eci_to_lvlh(
        target["r"],
        target["v"],
        r_rel_eci,
        v_rel_eci
    )

    r_target_norm = np.linalg.norm(target["r"])

    # Step 3: Solve CW in LVLH
    dv_lvlh = cw_rendezvous_delta_v(
        r_lvlh,
        v_lvlh,
        r_target_norm,
        tof
    )

    # Step 4: Convert ΔV back to ECI
    dv_eci = lvlh_to_eci(dv_lvlh, R)

    dv_mag = np.linalg.norm(dv_eci)

    return {
        "dv": dv_eci,
        "dv_mag": dv_mag
    }


# -----------------------------------------
# OPTIONAL: DEBUG PROPAGATION
# -----------------------------------------
def cw_propagate(r_lvlh, v_lvlh, r_target_norm, t):

    n = np.sqrt(MU / r_target_norm**3)

    Phi_rr, Phi_rv, Phi_vr, Phi_vv = cw_matrices(n, t)

    r_new = Phi_rr @ r_lvlh + Phi_rv @ v_lvlh
    v_new = Phi_vr @ r_lvlh + Phi_vv @ v_lvlh

    return r_new, v_new