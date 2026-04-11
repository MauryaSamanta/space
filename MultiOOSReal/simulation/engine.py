"""
main.py
Multi-agent Orbital Servicing Simulation
-----------------------------------------
Architecture
  * Fully decentralised — no ground station, no central planner
  * Dominance-based coordination via INTENT broadcasts
  * Realistic comms: Earth occlusion + distance-based delay
  * Local sensing: each OOS perceives only objects within sensing_radius
"""

import numpy as np

from scenario.generatorV2 import generate_scenario_v2
from oos.oos import OOS
from oos.network import Network
from physics.propagation import rk4_step
from config import DELTA_T


# ── Simulation parameters ───────────────────────────────────────────────────

N_SATELLITES    = 24        # number of sat/debris pairs
N_AGENTS        = 6       # number of OOS spacecraft
N_STEPS         = 1000    # total simulation ticks
PROCESSING_DELAY = 0.1     # seconds of fixed processing delay in comms
SENSING_RADIUS  = 5_000e3  # 5 000 km local perception radius (m)
WAIT_DURATION   = 300.0    # seconds agents wait before committing to mission


# ── Helpers ─────────────────────────────────────────────────────────────────

def _names_summary(env_state: dict) -> str:
    """One-line summary of object positions for debug printing."""
    parts = []
    for name, obj in env_state.items():
        alt_km = (np.linalg.norm(obj["r"]) - 6_371_000) / 1000
        parts.append(f"{name}@{alt_km:.0f}km")
    return "  ".join(parts)


# ── Main ────────────────────────────────────────────────────────────────────

def run_simulation():
    # ── Initialise environment ───────────────────────────────────────────────
    env_state, base_inclination, base_raan = generate_scenario_v2(N_SATELLITES)

    print("=== SCENARIO GENERATED ===")
    print(_names_summary(env_state))

    # ── Initialise fleet ─────────────────────────────────────────────────────
    # Place each OOS just off the corresponding satellite to give distinct
    # starting positions; fall back to SAT_0 if fewer sats than agents.
    fleet: list[OOS] = []
    sat_keys = [k for k in env_state if "DEBRIS" not in k]

    for i in range(N_AGENTS):
        ref_key = sat_keys[i % len(sat_keys)]
        ref_obj = env_state[ref_key]
        offset  = np.array([10.0, 0.0, 0.0]) * (i + 1)   # small offset (m)

        agent = OOS(
            r=ref_obj["r"] + offset,
            v=ref_obj["v"].copy(),
            id=i,
            sensing_radius=SENSING_RADIUS,
            wait_duration=WAIT_DURATION,
        )
        fleet.append(agent)
        print(f"  OOS{i} spawned near {ref_key}")

    # ── Initialise network ────────────────────────────────────────────────────
    network = Network(processing_delay=PROCESSING_DELAY)

    # ── Main loop ─────────────────────────────────────────────────────────────
    current_time = 0.0

    

    for step in range(N_STEPS):
        # print(f"\n{'='*16} STEP {step:4d}  t={current_time:.0f}s {'='*16}")

        # ── 1. Propagate all objects ────────────────────────────────────────
        for name in env_state:
            r, v = env_state[name]["r"], env_state[name]["v"]
            env_state[name]["r"], env_state[name]["v"] = rk4_step(r, v, DELTA_T)

        for oos in fleet:
            oos.r, oos.v = rk4_step(oos.r, oos.v, DELTA_T)

        # ── 2. Deliver queued messages ──────────────────────────────────────
        delivered = network.deliver(current_time, fleet)
        # if delivered:
            # print(f"  [NET] delivered {delivered} msg(s)  "
                #   f"queue={network.queue_size}")

        # ── 3. Agent decision loop ──────────────────────────────────────────
        for oos in fleet:
            state_before = oos.state
            oos.step(env_state, current_time, network, fleet)
            if oos.state != state_before:
                print(
                    f"  OOS{oos.id}: {state_before} → {oos.state}"
                    + (f" [{oos.target}]" if oos.target else "")
                )

        current_time += DELTA_T

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n=== SIMULATION COMPLETE ===")
    for oos in fleet:
        print(f"  OOS{oos.id}: final state = {oos.state}")

    return fleet, env_state


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_simulation()