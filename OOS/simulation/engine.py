from scenario.generator import generate_scenario
from oos.state import init_oos
from oos.oos import OOS

from physics.propagation import rk4_step
from physics.collision import collision_probability
from physics.lambert import find_best_transfer, lambert_transfer, predict_future
from config import *

import numpy as np

from oos.oos_mission_manager import assign_mission, generate_missions, prioritize_missions

def get_angle(r):
    return np.arctan2(r[1], r[0])
# -----------------------------
# 🔥 NEW: Predict TCA + future position
# -----------------------------
def predict_tca(obj1, obj2, steps=500):
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
    # oos = init_oos(state[first]["r"])
    # oos["v"] = state[first]["v"].copy()
    oos = OOS(state[first]["r"], state[first]["v"])
    current_time = 0

    for step in range(1200):
        if step % 20==0 and oos.state!="PHASING":
            print(f"\n================ STEP {step} ================")

        # --- propagate ALL objects ---
        for name in state:
            r, v = state[name]["r"], state[name]["v"]
            r_new, v_new = rk4_step(r, v, DELTA_T)
            state[name]["r"], state[name]["v"] = r_new, v_new

        oos.r, oos.v = rk4_step(oos.r, oos.v, DELTA_T)

        # ---------------- OOS STATE MACHINE ----------------

        if oos.state == "PLANNING":

            target = oos.target

            theta_oos = np.arctan2(oos.r[1], oos.r[0])
            theta_target = np.arctan2(state[target]["r"][1], state[target]["r"][0])

            phase_error = theta_target - theta_oos
            phase_error = (phase_error + np.pi) % (2*np.pi) - np.pi

            print(f"→ Phasing... error: {phase_error:.4f}")

            # check alignment
            if abs(phase_error) < 0.02:
                print("✅ Phase aligned → restoring orbit")

                # reverse burn (circularize)
                tangent = oos.v / np.linalg.norm(oos.v)
                dv = -0.01 * oos.phase_direction * tangent

                oos.v += dv
                oos.fuel -= np.linalg.norm(dv)

                oos.state = "DOCKING"
                oos.dock_end_time = current_time + DOCKING_TIME
        # ---------------- PHASING ----------------

        elif oos.state == "PHASING":

                target = oos.target

                theta_oos = np.arctan2(oos.r[1], oos.r[0])
                theta_target = np.arctan2(state[target]["r"][1], state[target]["r"][0])

                phase_error = theta_target - theta_oos
                phase_error = (phase_error + np.pi) % (2*np.pi) - np.pi
                if step%20==0:
                    print(f"→ Phasing... error: {phase_error:.4f}")

                # 🔥 recompute direction every step (FIXES YOUR ISSUE)
                direction = -1 if phase_error > 0 else 1

                # 🔥 small feedback correction
                tangent = oos.v / np.linalg.norm(oos.v)
                dv_correction = 0.0005 * direction * tangent

                oos.v += dv_correction
                oos.fuel -= np.linalg.norm(dv_correction)

                # alignment check
                if abs(phase_error) < 0.02:
                    print("✅ Phase aligned → restoring orbit")

                    r_norm = np.linalg.norm(oos.r)
                    v_circular = np.sqrt(MU / r_norm)

                    current_speed = np.linalg.norm(oos.v)
                    dv_mag = v_circular - current_speed

                    tangent = oos.v / np.linalg.norm(oos.v)
                    dv_restore = dv_mag * tangent

                    oos.v += dv_restore
                    oos.fuel -= abs(dv_mag)

                    oos.state = "DOCKING"
                    oos.dock_end_time = current_time + DOCKING_TIME

        elif oos.state == "TRANSFER":

            print("→ Coasting (transfer phase)")

            if current_time >= oos.transfer_end_time:
                print("✅ Transfer complete → Docking")

                oos.state = "DOCKING"
                oos.dock_end_time = current_time + DOCKING_TIME

        elif oos.state == "DOCKING":
            print(f"→ Docking with {oos.target}")

            if current_time >= oos.dock_end_time:
                oos.state = "MANEUVER"
                print("✅ DOCKED → Executing maneuver")

        elif oos.state == "MANEUVER":
            target = oos.target
            debris = target + "_DEBRIS"

            print("\n🚀 MANEUVER EXECUTION")

            # BEFORE
            pc_before, d_before = collision_probability(state[target], state[debris])
            # ---------------- GO / NO-GO CHECK ----------------

            if pc_before < 0.01 and d_before > 1.0:
                print("⚠️ Skipping maneuver (not a real threat)")

                oos.state = "IDLE"
                oos.target = None

                if oos.current_mission:
                    oos.current_mission.completed = True
                    oos.current_mission = None

                continue
            print("\n--- BEFORE ---")
            print(f"Min Distance: {d_before:.4f} km")
            print(f"Pc: {pc_before:.6f}")

            # APPLY ΔV
            # push satellite slightly outward (radial burn)
            rel_v = state[target]["v"] - state[debris]["v"]

            # normalize
            rel_v_hat = rel_v / np.linalg.norm(rel_v)

            # find perpendicular direction
            # pick any vector not parallel
            temp = np.array([1, 0, 0])
            if abs(np.dot(temp, rel_v_hat)) > 0.9:
                temp = np.array([0, 1, 0])

            # perpendicular vector
            perp = np.cross(rel_v_hat, temp)
            perp = perp / np.linalg.norm(perp)

            # apply ΔV sideways
            dv = perp * 0.01
            state[target]["v"] += dv

            print("\n--- ΔV APPLIED ---")
            print(f"ΔV Magnitude: {np.linalg.norm(dv):.6f} km/s")

            # AFTER
            pc_after, d_after = collision_probability(state[target], state[debris])

            print("\n--- AFTER ---")
            print(f"Min Distance: {d_after:.4f} km")
            print(f"Pc: {pc_after:.6f}")

            print("\n--- RESULT ---")
            print(f"Risk Reduction: {pc_before - pc_after:.6f}")

            oos.state = "IDLE"
            oos.target = None
            if oos.current_mission:
                oos.current_mission.completed = True
                oos.current_mission = None
            # oos["planned_dv"] = None

        # --------------------------------------------------

        if oos.state != "IDLE":
            current_time += DELTA_T
            continue

        # ---------------- COLLISION DETECTION ----------------

        # ---------------- MISSION SCHEDULING ----------------

        if oos.state == "IDLE":

            missions = generate_missions(state, current_time, DELTA_T)

            if missions:
                missions = prioritize_missions(missions)

                print("\n📋 Mission Queue:")
                for m in missions:
                    print(m)

                assigned = assign_mission(oos, missions)

                if assigned:
                    print("🚀 Mission started")
                    target = oos.target

                    theta_oos = np.arctan2(oos.r[1], oos.r[0])
                    theta_target = np.arctan2(state[target]["r"][1], state[target]["r"][0])

                    phase_error = theta_target - theta_oos
                    phase_error = (phase_error + np.pi) % (2*np.pi) - np.pi

                    direction = -1 if phase_error > 0 else 1

                    tangent = oos.v / np.linalg.norm(oos.v)

                    dv = 0.01 * direction * tangent

                    oos.v += dv
                    oos.fuel -= np.linalg.norm(dv)

                    oos.phase_direction = direction
                    oos.state = "PHASING"

                    print(f"Initial phase error: {phase_error:.4f}")

        current_time += DELTA_T