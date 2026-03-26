import numpy as np
from poliastro.iod import lambert
from poliastro.bodies import Earth
from astropy import units as u
from physics.propagation import rk4_step
from config import DELTA_T


def predict_future(obj, tof):
    """
    Propagate object forward in time to estimate future position & velocity
    """

    r = obj["r"].copy()
    v = obj["v"].copy()

    steps = int(tof / DELTA_T)

    for _ in range(steps):
        r, v = rk4_step(r, v, DELTA_T)

    return r, v
def find_best_transfer(oos, target):
    best = None
    best_dv = float("inf")

    for tof in [2000, 3000, 4000, 6000, 8000]:

        r2, v2_target = predict_future(target, tof)

        transfer = lambert_transfer(
            oos["r"],
            oos["v"],
            r2,
            v2_target,
            tof
        )

        if transfer is None:
            continue

        if transfer["total_dv"] < best_dv:
            best_dv = transfer["total_dv"]
            best = (transfer, tof)

    return best



def lambert_transfer(r1, v1, r2, v2_target, tof):
    try:
        r1 = np.array(r1) * u.km
        r2 = np.array(r2) * u.km

        (v1_new, v2), = lambert(Earth.k, r1, r2, tof * u.s)

        v1_new = v1_new.to(u.km/u.s).value
        v2 = v2.to(u.km/u.s).value

        dv_depart = np.linalg.norm(v1_new - np.array(v1))
        dv_arrive = np.linalg.norm(v2 - np.array(v2_target))

        total_dv = dv_depart + dv_arrive

        if np.isnan(v1_new).any() or total_dv > 2.0:
            return None

        return {
            "v1": v1_new,
            "total_dv": total_dv
        }

    except:
        return None