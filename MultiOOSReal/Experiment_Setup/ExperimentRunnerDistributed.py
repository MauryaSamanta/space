from MetricsManager import MetricsManager
import numpy as np
from physics.collision import collision_probability
from config import PC_THRESHOLD
class ExperimentRunnerDistributed:

    def __init__(self, config):
        self.config = config
        self.metrics = MetricsManager()

    def setup(self):
        from scenario.generatorV2 import generate_scenario_v2
        from oos.oos import OOS

        state, base_i, base_raan = generate_scenario_v2(
            self.config.n_satellites
        )

        fleet = []
        sat_keys = [k for k in state if "_DEBRIS" not in k]

        for i in range(self.config.n_oos):

            sat_name = sat_keys[i % len(sat_keys)]
            base = state[sat_name]

            r = base["r"] + 0.01
            v = base["v"]

            oos = OOS(
    r,
    v,
    id=i,
    sensing_radius=self.config.sensing_radius
)
            oos.metrics = self.metrics
            oos.fuel = self.config.fuel_per_oos

            fleet.append(oos)

        return state, fleet

    def run(self):
        from physics.propagation import rk4_step
        from oos.network import Network
        completed_targets = set()
        active_collisions = set()
        all_collisions = set()
        state, fleet = self.setup()

        network = Network(processing_delay=0.1)

        current_time = 0

        try:
            for step in range(self.config.steps):

                # ── 1. propagate environment ─────────────────────
                for name in state:
                    r, v = state[name]["r"], state[name]["v"]
                    state[name]["r"], state[name]["v"] = rk4_step(r, v, 120)

                current_collisions = set()

                for name in state:
                    if "_DEBRIS" in name:
                        sat = name.replace("_DEBRIS", "")

                        pc, _ = collision_probability(state[sat], state[name])

                        if pc >= PC_THRESHOLD:
                            current_collisions.add(sat)
                            all_collisions.add(sat)

                # ── 2. propagate OOS ─────────────────────────────
                for oos in fleet:
                    oos.r, oos.v = rk4_step(oos.r, oos.v, 120)

                # ── 3. deliver messages ──────────────────────────
                network.deliver(current_time, fleet)

                # ── 4. agent step ────────────────────────────────
                for oos in fleet:
                    oos.step(state, current_time, network, fleet, completed_targets, active_collisions)

                # ❌ REMOVE THIS (already handled inside step)
                # oos.process_inbox()

                current_time += 120

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted — returning partial metrics...")

        result = self.metrics.finalize()

        # 🔥 FIX TRUE NUMBER OF COLLISIONS
        result["missions_total"] = len(all_collisions)

        return result