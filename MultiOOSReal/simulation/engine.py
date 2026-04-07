from scenario.generator import generate_scenario
from oos.state import init_oos
from oos.oos import OOS

from physics.propagation import rk4_step
from physics.collision import collision_probability
from physics.lambert import find_best_transfer, lambert_transfer, predict_future
from config import *

import numpy as np


from scenario.generatorV2 import generate_scenario_v2
from oos.network import Network

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

    fleet = [
        OOS(state["SAT_0"]["r"] + 0.01, state["SAT_0"]["v"], id=0),
        OOS(state["SAT_1"]["r"] + 0.01, state["SAT_1"]["v"], id=1)
    ]

    network = Network(delay=200)

    WAIT_TIME = 300
    current_time = 0

    for step in range(1200):

        print(f"\n================ STEP {step} ================")

        # -------- propagate --------
        for name in state:
            r, v = state[name]["r"], state[name]["v"]
            state[name]["r"], state[name]["v"] = rk4_step(r, v, DELTA_T)

        for oos in fleet:
            oos.r, oos.v = rk4_step(oos.r, oos.v, DELTA_T)

        # -------- deliver messages --------
        network.deliver(current_time, fleet)

        # -------- agent loop --------
        for oos in fleet:

            oos.step(state, current_time)

            # inbox update
            oos.process_inbox()

            # ---------------- CLAIM ----------------
            if oos.state == "IDLE" and oos.pending_claim is None:

                claim = oos.create_claim(state, current_time)

                if claim:
                    oos.pending_claim = claim
                    oos.claim_time = current_time

                    network.broadcast(claim, current_time)

                    print(f"OOS{oos.id} CLAIM → {claim['mission_id']}")

            # ---------------- WAIT + RESOLVE ----------------
            if oos.pending_claim:

                if current_time - oos.claim_time >= WAIT_TIME:

                    if oos.resolve_claim():
                        mission = oos.pending_claim["mission_id"]

                        print(f"OOS{oos.id} START → {mission}")

                        oos.target = mission
                        oos.state = "PLANNING"
                    else:
                        print(f"OOS{oos.id} LOST → {oos.pending_claim['mission_id']}")

                    oos.pending_claim = None

        current_time += DELTA_T