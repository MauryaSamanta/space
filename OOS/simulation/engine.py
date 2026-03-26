from scenario.generator import generate_scenario
from oos.state import init_oos
from physics.propagation import rk4_step
from physics.collision import collision_probability
from physics.lambert import find_best_transfer, lambert_transfer
from config import *

import numpy as np

def get_angle(r):
    return np.arctan2(r[1], r[0])
# -----------------------------
# 🔥 NEW: Predict TCA + future position
# -----------------------------
def predict_tca(obj1, obj2, steps=30):
    r1, v1 = obj1["r"].copy(), obj1["v"].copy()
    r2, v2 = obj2["r"].copy(), obj2["v"].copy()

    min_d = float("inf")
    best_step = 0
    best_r2 = None

    for i in range(steps):
        r1, v1 = rk4_step(r1, v1, DELTA_T)
        r2, v2 = rk4_step(r2, v2, DELTA_T)

        d = np.linalg.norm(r1 - r2)

        if d < min_d:
            min_d = d
            best_step = i
            best_r2 = r2.copy()

    return min_d, best_step, best_r2


# -----------------------------
# 🔥 NEW: Find valid transfer using FUTURE position
# -----------------------------
def find_valid_transfer(oos, target_future_pos, tof):
    try:
        transfer = lambert_transfer(
            oos["r"],
            oos["v"],
            target_future_pos,
            tof
        )

        if transfer is None:
            return None

        # reject insane ΔV
        dv = transfer["v1"] - oos["v"]
        if np.linalg.norm(dv) > 10:
            return None

        return transfer

    except:
        return None


# -----------------------------
# MAIN SIMULATION
# -----------------------------
def run_simulation():
    state = generate_scenario(3)

    first = list(state.keys())[0]
    oos = init_oos(state[first]["r"])
    oos["v"] = state[first]["v"].copy()
    current_time = 0

    for step in range(200):
        print(f"\n================ STEP {step} ================")

        # --- propagate ALL objects ---
        for name in state:
            r, v = state[name]["r"], state[name]["v"]
            r_new, v_new = rk4_step(r, v, DELTA_T)
            state[name]["r"], state[name]["v"] = r_new, v_new

        oos["r"], oos["v"] = rk4_step(oos["r"], oos["v"], DELTA_T)

        # ---------------- OOS STATE MACHINE ----------------

        if oos["state"] == "PHASING":
            target = oos["target"]

            theta_oos = get_angle(oos["r"])
            theta_target = get_angle(state[target]["r"])

            phase_error = theta_target - theta_oos

            print(f"→ Phasing... error: {phase_error:.4f}")

            # normalize angle
            phase_error = (phase_error + np.pi) % (2 * np.pi) - np.pi

            # small velocity tweak (phasing control)
            dv_mag = 0.002 * phase_error   # proportional control
            dv = np.array([0.0, dv_mag, 0.0])

            if phase_error > 0:
                oos["v"] += dv
            else:
                oos["v"] -= dv

            oos["fuel"] -= np.linalg.norm(dv)

            # check alignment
            if abs(phase_error) < 0.01:
                print("✅ Phase aligned → Docking")
                oos["state"] = "DOCKING"
                oos["dock_end_time"] = current_time + DOCKING_TIME

        elif oos["state"] == "DOCKING":
            print(f"→ Docking with {oos['target']}")

            if current_time >= oos["dock_end_time"]:
                oos["state"] = "MANEUVER"
                print("✅ DOCKED → Executing maneuver")

        elif oos["state"] == "MANEUVER":
            target = oos["target"]
            debris = target + "_DEBRIS"

            print("\n🚀 MANEUVER EXECUTION")

            # BEFORE
            pc_before, d_before = collision_probability(state[target], state[debris])

            print("\n--- BEFORE ---")
            print(f"Min Distance: {d_before:.4f} km")
            print(f"Pc: {pc_before:.6f}")

            # APPLY ΔV
            # push satellite slightly outward (radial burn)
            direction = state[target]["r"] / np.linalg.norm(state[target]["r"])
            dv = direction * 0.01   # 0.01 km/s (10 m/s)

            state[target]["v"] += dv

            dv_mag = np.linalg.norm(dv)
            oos["fuel"] -= dv_mag

            print("\n--- ΔV APPLIED ---")
            print(f"ΔV Magnitude: {dv_mag:.6f} km/s")

            # AFTER
            pc_after, d_after = collision_probability(state[target], state[debris])

            print("\n--- AFTER ---")
            print(f"Min Distance: {d_after:.4f} km")
            print(f"Pc: {pc_after:.6f}")

            print("\n--- RESULT ---")
            print(f"Risk Reduction: {pc_before - pc_after:.6f}")

            oos["state"] = "IDLE"
            oos["target"] = None
            oos["planned_dv"] = None

        # --------------------------------------------------

        if oos["state"] != "IDLE":
            current_time += DELTA_T
            continue

        # ---------------- COLLISION DETECTION ----------------

        for sat in state:
            if "_DEBRIS" in sat:
                continue

            debris = sat + "_DEBRIS"

            Pc, d = collision_probability(state[sat], state[debris])

            _, t_steps, target_future_pos = predict_tca(
                state[sat],
                state[debris]
            )

            print(f"{sat} | d={d:.4f} km | Pc={Pc:.4f}")

            if Pc < 0.05:
                continue

            print("\n⚠️ HIGH RISK DETECTED")

            tof = t_steps * DELTA_T

            oos["state"] = "PHASING"
            oos["target"] = sat

            print("\n🚀 MISSION STARTED (PHASING)")
            print(f"Target: {sat}")

            break

        current_time += DELTA_T