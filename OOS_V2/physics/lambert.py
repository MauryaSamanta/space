import numpy as np
from poliastro.iod import lambert
from poliastro.bodies import Earth
from astropy import units as u

from physics.propagation import rk4_step


def propagate_future(r, v, tof, dt=20):
    steps = int(tof / dt)

    r_f = r.copy()
    v_f = v.copy()

    for _ in range(steps):
        r_f, v_f = rk4_step(r_f, v_f, dt)

    return r_f, v_f


def solve_lambert(r1, v1, r2, v2, tof):

    # predict target future
    r2_future, _ = propagate_future(r2, v2, tof)

    try:
        (v1_new, _), = lambert(
            Earth.k,
            r1 * u.m,
            r2_future * u.m,
            tof * u.s
        )

        v1_new = v1_new.to(u.m / u.s).value

        dv = np.linalg.norm(v1_new - v1)

        return v1_new, dv

    except Exception as e:
        print("[LAM ERROR]", e)
        return None, None