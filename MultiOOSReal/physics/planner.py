import numpy as np
from poliastro.iod import lambert
from poliastro.bodies import Earth
from astropy import units as u
from config import MU


# -----------------------------------------
# DISTANCE CHECK
# -----------------------------------------
def is_far(oos, target):
    r_rel = oos["r"] - target["r"]
    return np.linalg.norm(r_rel) > 200  # km


# -----------------------------------------
# SAFE LAMBERT TRANSFER
# -----------------------------------------
def lambert_transfer_safe(oos, target, predict_future, time_to_tca):

    best = None
    best_dv = float("inf")

    # try reasonable TOFs only
    for tof in range(2000, min(5000, int(time_to_tca - 300)), 500):

        r2, v2_target = predict_future(target, tof)

        try:
            r1 = oos["r"] * u.km
            r2_u = r2 * u.km

            (v1_new, v2_new), = lambert(Earth.k, r1, r2_u, tof * u.s)

            v1_new = v1_new.to(u.km/u.s).value
            v2_new = v2_new.to(u.km/u.s).value

            dv_depart = np.linalg.norm(v1_new - oos["v"])
            dv_arrival = np.linalg.norm(v2_new - v2_target)

            total_dv = dv_depart + dv_arrival

            print(f"[Lambert] TOF={tof} | ΔV={total_dv:.3f}")

            # # 🔥 CRITICAL FILTERS
            # if dv_depart > 1.0:
            #     continue
            # if dv_arrival > 1.0:
            #     continue

            if total_dv < best_dv:
                best_dv = total_dv
                best = {
                    "v1": v1_new,
                    "v2": v2_new,
                    "tof": tof,
                    "dv": dv_depart
                }

        except:
            continue

    return best