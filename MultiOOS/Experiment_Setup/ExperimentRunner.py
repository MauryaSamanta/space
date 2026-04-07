from MetricsManager import MetricsManager
import numpy as np
import sys
import os

# 🔥 Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
class ExperimentRunner:

    def __init__(self, config):
        self.config = config
        self.metrics = MetricsManager()

    def setup(self):
        # 🔥 generate scenario
        from scenario.generatorV2 import generate_scenario_v2

        state, base_i, base_raan = generate_scenario_v2(
            self.config.n_satellites
        )

        # 🔥 create OOS fleet
        from oos.oos import OOS

        oos_fleet = []
        for i in range(self.config.n_oos):
            r = state["SAT_0"]["r"] + np.random.randn(3) * 0.01
            v = state["SAT_0"]["v"]
            
            # oos.current_time_ref = lambda: current_time
            oos = OOS(r, v)
            oos.metrics = self.metrics
            oos.fuel = self.config.fuel_per_oos

            oos_fleet.append(oos)

        return state, oos_fleet

    def run(self):
        from physics.propagation import rk4_step
        from oos.oos_mission_manager import generate_missions, prioritize_missions, assign_mission

        state, oos_fleet = self.setup()
        failed_targets = set()
        current_time = 0
    
        for step in range(self.config.steps):
            print(f"Step {step}")
            # propagate all objects
            for name in state:
                r, v = state[name]["r"], state[name]["v"]
                state[name]["r"], state[name]["v"] = rk4_step(r, v, 120)

            # propagate all OOS
            for oos in oos_fleet:
                oos.r, oos.v = rk4_step(oos.r, oos.v, 120)
                
            # run each OOS
            for oos in oos_fleet:
                oos.current_time = current_time
                oos.step(state, current_time)

            if any(oos.state == "IDLE" for oos in oos_fleet):

                missions = generate_missions(state, current_time, 120)
                missions = prioritize_missions(missions)

                for oos in oos_fleet:
                    if oos.state == "IDLE" and missions:

                        m = missions[0]

                        # 🔥 skip failed targets
                        if m.sat in failed_targets:
                            continue

                        oos.target = m.sat
                        oos.current_mission = m
                        m.assigned = True

                        self.metrics.log_mission_start()
                        oos.mission_start_time = current_time

                        success = oos.start_mission(state, current_time)

                        # 🔥 track failure
                        if not success:
                            failed_targets.add(m.sat)
       
            current_time += 120

        return self.metrics.finalize()