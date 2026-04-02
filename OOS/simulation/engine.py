from scenario.generator import generate_scenario
from oos.state import init_oos
from oos.oos import OOS

from physics.propagation import rk4_step
from physics.collision import collision_probability
from physics.lambert import find_best_transfer, lambert_transfer, predict_future
from config import *

import numpy as np

from oos.oos_mission_manager import assign_mission, generate_missions, prioritize_missions
from scenario.generatorV2 import generate_scenario_v2

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
    state, base_i, base_raan = generate_scenario_v2(3)

    first = list(state.keys())[0]
    # oos = init_oos(state[first]["r"])
    # oos["v"] = state[first]["v"].copy()
    r = state["SAT_0"]["r"] + 0.01
    v = state["SAT_0"]["v"]
    oos=OOS(r,v)

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
        oos.step(state, current_time)
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
                    oos.start_mission(state, current_time)
                    

        current_time += DELTA_T