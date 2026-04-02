import numpy as np

from scenario.generator import generate_scenario, backpropagate
from core.satellite import Satellite
from core.oos import OOS
from core.mission_manager import generate_missions
from physics.propagation import rk4_step
from config import DT, SIM_STEPS


# ---------------------
# INIT
# ---------------------
raw = generate_scenario()
raw = backpropagate(raw)

sats = [Satellite(s["id"], s["r"], s["v"]) for s in raw]

# extract OOS
oos_sat = sats.pop(-1)
oos = OOS(oos_sat.r, oos_sat.v)


# ---------------------
# SIM LOOP
# ---------------------
for step in range(SIM_STEPS):

    # propagate sats
    for sat in sats:
        sat.r, sat.v = rk4_step(sat.r, sat.v, DT)

    oos.r, oos.v = rk4_step(oos.r, oos.v, DT)

    # missions
    missions = generate_missions(sats)

    # OOS agent step
    oos.step(missions)

    if step % 50 == 0:
        print(f"\nStep {step}")
        print(f"OOS State: {oos.state}")
        print(f"# Missions: {len(missions)}")

        if oos.target:
            dist = np.linalg.norm(oos.target.r - oos.r)
            print(f"Target distance: {dist:.2f}")