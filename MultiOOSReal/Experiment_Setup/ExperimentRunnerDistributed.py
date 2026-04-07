from MetricsManager import MetricsManager
import numpy as np

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

            r = base["r"] + 0.01  # same as engine (NOT random noise)
            v = base["v"]

            oos = OOS(r, v, id=i)
            oos.metrics = self.metrics
            oos.fuel = self.config.fuel_per_oos

            fleet.append(oos)

        return state, fleet

    def run(self):
        from physics.propagation import rk4_step
        from oos.network import Network

        state, fleet = self.setup()

        network = Network(delay=200)
        WAIT_TIME = 300

        current_time = 0

        try:
            for step in range(self.config.steps):

                # propagate env
                for name in state:
                    r, v = state[name]["r"], state[name]["v"]
                    state[name]["r"], state[name]["v"] = rk4_step(r, v, 120)

                # propagate OOS
                for oos in fleet:
                    oos.r, oos.v = rk4_step(oos.r, oos.v, 120)

                # deliver messages
                network.deliver(current_time, fleet)

                # agent loop
                for oos in fleet:

                    oos.current_time = current_time
                    oos.step(state, current_time)

                    oos.process_inbox()

                    # CLAIM
                    if oos.state == "IDLE" and oos.pending_claim is None:

                        claim = oos.create_claim(state, current_time)

                        if claim:
                            oos.pending_claim = claim
                            oos.claim_time = current_time

                            network.broadcast(claim, current_time)

                    # RESOLVE
                    if oos.pending_claim:

                        if current_time - oos.claim_time >= WAIT_TIME:

                            if oos.resolve_claim():
                                oos.target = oos.pending_claim["mission_id"]
                                oos.state = "PLANNING"

                            oos.pending_claim = None

                current_time += 120

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted during simulation — returning partial metrics...")

        return self.metrics.finalize()