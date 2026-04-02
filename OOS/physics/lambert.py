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

    for tof in [1000, 2000, 3000, 4000]:

        r2, v2_target = predict_future(target, tof)

        transfer = lambert_transfer(
            oos["r"],
            oos["v"],
            r2,
            tof
        )

        if transfer is None:
            continue

        # arrival mismatch cost
        arrival_error = np.linalg.norm(transfer["v2"] - v2_target)

        total_cost = transfer["dv"] + arrival_error

        if total_cost < best_dv:
            best_dv = total_cost
            best = (transfer, tof)

    return best



def lambert_transfer(r1, v1, r2, tof):
    try:
        r1 = np.array(r1) * u.km
        r2 = np.array(r2) * u.km

        # solve Lambert
        (v1_new, v2_new), = lambert(Earth.k, r1, r2, tof * u.s)

        v1_new = v1_new.to(u.km/u.s).value
        v2_new = v2_new.to(u.km/u.s).value

        # ΔV at departure
        dv_depart = np.linalg.norm(v1_new - np.array(v1))
        print(f"Lambert candidate ΔV: {dv_depart:.4f}")
        # sanity check
        if np.isnan(v1_new).any() :
            print("❌ Rejected due to high ΔV")
            return None


        return {
            "v1": v1_new,
            "v2": v2_new,
            "dv": dv_depart
        }

    except Exception as e:
        print("Lambert error:", e)
        return None