import numpy as np
from poliastro.iod import lambert
from poliastro.bodies import Earth
from astropy import units as u

def find_best_transfer(oos, target_future_pos):
    best = None
    best_cost = float("inf")

    for tof in [500, 1000, 2000, 3000, 4000]:
        transfer = lambert_transfer(
            oos["r"],
            oos["v"],
            target_future_pos,
            tof
        )

        if transfer is None:
            continue

        dv = transfer["v1"] - oos["v"]
        dv_mag = np.linalg.norm(dv)

        if dv_mag < best_cost:
            best_cost = dv_mag
            best = (transfer, tof)

    return best

def lambert_transfer(r1, v1, r2, tof):
    try:
        r1 = np.array(r1) * u.km
        r2 = np.array(r2) * u.km

        (v1_new, v2), = lambert(Earth.k, r1, r2, tof * u.s)

        v1_new = v1_new.to(u.km/u.s).value

        if np.isnan(v1_new).any() or np.linalg.norm(v1_new) > 20:
            return None

        return {
            "v1": v1_new,
            "delta_v": np.linalg.norm(v1_new - np.array(v1))
        }

    except:
        return None