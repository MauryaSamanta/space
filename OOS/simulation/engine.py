from scenario.generator import generate_scenario
from oos.state import init_oos
from physics.propagation import rk4_step
from physics.collision import collision_probability
from physics.lambert import find_best_transfer, lambert_transfer
from config import *

import numpy as np


# -----------------------------
# 🔥 NEW: Predict TCA + future position
# -----------------------------
def predict_tca(obj1, obj2, steps=200):
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

        if oos["state"] == "TRANSIT":
            print(f"→ Traveling to {oos['target']}")

            if current_time >= oos["arrival_time"]:
                oos["state"] = "DOCKING"
                print("✅ ARRIVED → Docking...")

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
            d_before, _, _ = predict_tca(state[target], state[debris])
            pc_before = collision_probability(d_before)

            print("\n--- BEFORE ---")
            print(f"Min Distance: {d_before:.4f} km")
            print(f"Pc: {pc_before:.6f}")

            # APPLY ΔV
            if oos["planned_dv"] is not None:
                dv_mag = np.linalg.norm(oos["planned_dv"])
                state[target]["v"] += oos["planned_dv"]

                print("\n--- ΔV APPLIED ---")
                print(f"ΔV Magnitude: {dv_mag:.6f} km/s")

            # AFTER
            d_after, _, _ = predict_tca(state[target], state[debris])
            pc_after = collision_probability(d_after)

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

            d, t_steps, target_future_pos = predict_tca(
                state[sat],
                state[debris]
            )

            Pc = collision_probability(d)

            print(f"{sat} | d={d:.4f} km | Pc={Pc:.4f}")

            if Pc < 0.05:
                continue

            print("\n⚠️ HIGH RISK DETECTED")

            tof = t_steps * DELTA_T

            transfer,tof = find_best_transfer(oos, target_future_pos)

            if transfer is None:
                print("❌ No valid transfer")
                continue

            dv = transfer["v1"] - oos["v"]
            dv_mag = np.linalg.norm(dv)

            print("\n🧠 TRANSFER FOUND")
            print(f"TOF: {tof}")
            print(f"ΔV: {dv_mag:.4f} km/s")

            # schedule mission
            oos["state"] = "TRANSIT"
            oos["target"] = sat
            oos["arrival_time"] = current_time + tof
            oos["dock_end_time"] = oos["arrival_time"] + DOCKING_TIME
            oos["planned_dv"] = dv

            print("\n🚀 MISSION SCHEDULED")

            break

        current_time += DELTA_T