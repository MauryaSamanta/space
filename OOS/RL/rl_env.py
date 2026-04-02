import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from oos.oos import OOS

from physics.propagation import rk4_step
from physics.collision import collision_probability

# from scenario.generator import generate_scenario_v2 as generate_scenario

from config import DELTA_T
from scenario.generatorV2 import generate_scenario_v2


class SatelliteEnv:
    def __init__(self):
        self.reset()

    # -----------------------------
    # TOTAL COLLISION RISK
    # -----------------------------
    def _compute_total_pc(self):
        total_pc = 0
        for sat in ["SAT_0", "SAT_1", "SAT_2"]:
            if sat not in self.state_data:
                continue

            debris = sat + "_DEBRIS"
            Pc, _ = collision_probability(
                self.state_data[sat],
                self.state_data[debris]
            )
            total_pc += Pc

        return total_pc

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self):
        self.state_data = generate_scenario_v2(3)

        # 🔥 pick random satellite for servicer start
        targets = ["SAT_0", "SAT_1", "SAT_2"]
        first = np.random.choice(targets)

        theta = np.random.uniform(0.5, 2.5)

        r0 = self.rotate_z(self.state_data[first]["r"], theta)
        v0 = self.rotate_z(self.state_data[first]["v"], theta)

        self.oos = OOS(r0, v0)

        self.time = 0
        self.prev_total_pc = self._compute_total_pc()
        self.prev_fuel = self.oos.fuel

        self.decision_timer = 0
        self.last_action = 0

        return self._get_state()

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action):
        reward = 0

        # --- propagate all objects ---
        for name in self.state_data:
            r, v = self.state_data[name]["r"], self.state_data[name]["v"]
            r, v = rk4_step(r, v, DELTA_T)
            self.state_data[name]["r"], self.state_data[name]["v"] = r, v

        self.oos.propagate(rk4_step, DELTA_T)

        # -----------------------------
        # RL DECISION
        # -----------------------------
        if self.oos.state == "IDLE" and self.decision_timer <= 0:

            targets = ["SAT_0", "SAT_1", "SAT_2"]

            if action > 0:
                target = targets[action - 1]

                if target in self.state_data:
                    self.oos.target = target
                    self.oos.start_phasing(self.state_data, self.time)
                    self.decision_timer = 5
                # -----------------------------
        # RUN OOS LOGIC
        # -----------------------------
        self.oos.step(self.state_data, self.time)
        # -----------------------------
# REMOVE COMPLETED MISSIONS (CRITICAL FIX)
# -----------------------------
        # -----------------------------
# REMOVE COMPLETED MISSIONS (SAFE)
# -----------------------------
        to_remove = []

        for sat in ["SAT_0", "SAT_1", "SAT_2"]:
            if sat not in self.state_data:
                continue

            debris = sat + "_DEBRIS"

            Pc, _ = collision_probability(
                self.state_data[sat],
                self.state_data[debris]
            )

            # mark for removal
            if Pc < 1e-4:
                # if OOS finished this target → clear it
                if self.oos.target == sat and self.oos.state == "IDLE":
                    self.oos.target = None

                # only remove if NOT actively being used
                if self.oos.target != sat:
                    to_remove.append(sat)

        for sat in to_remove:
            debris = sat + "_DEBRIS"

            print(f"🧹 Removing completed mission: {sat}")

            del self.state_data[sat]
            del self.state_data[debris]

        # -----------------------------
        # REWARD
        # -----------------------------
        reward += self._compute_reward()

        self.time += DELTA_T
        self.decision_timer = max(0, self.decision_timer - 1)

        current_pc = self._compute_total_pc()

        done = False

        # catastrophic collision
        if current_pc > 0.25:
            reward -= 200
            done = True

        # safe system
        elif current_pc < 0.001 and self.prev_total_pc > 0.01:
            reward += 100
            done = True

        # timeout
        elif self.time > 10000:
            done = True

        self.last_action = action

        return self._get_state(), reward, done

    # -----------------------------
    # STATE (FIXED PER SATELLITE)
    # -----------------------------
    def _get_state(self):
        state = []

        for sat in ["SAT_0", "SAT_1", "SAT_2"]:
            if sat in self.state_data:
                debris = sat + "_DEBRIS"

                Pc, _ = collision_probability(
                    self.state_data[sat],
                    self.state_data[debris]
                )

                d = np.linalg.norm(
                    self.state_data[sat]["r"] -
                    self.state_data[debris]["r"]
                )

                # 🔥 NORMALIZATION (important)
                Pc = Pc
                d = d / 1000.0  # km → scaled

                state += [Pc, d]
            else:
                state += [0, 10]

        # fuel normalized
        state.append(self.oos.fuel / 100.0)

        # busy flag
        state.append(1 if self.oos.state != "IDLE" else 0)

        return np.array(state, dtype=np.float32)

    # -----------------------------
    # REWARD (STABLE VERSION)
    # -----------------------------
    def _compute_reward(self):
        reward = 0

        current_pc = self._compute_total_pc()

        # --- main learning signal ---
        delta_pc = self.prev_total_pc - current_pc
        reward += delta_pc * 1000

        # --- penalize remaining risk ---
        reward -= current_pc * 2

        # --- penalize missed opportunities ---
        if current_pc > 0.05 and self.oos.state == "IDLE":
            reward -= 20

        # --- fuel penalty (small) ---
        delta_fuel = self.prev_fuel - self.oos.fuel
        reward -= delta_fuel * 10

        self.prev_total_pc = current_pc
        self.prev_fuel = self.oos.fuel

        return reward

    # -----------------------------
    # UTILS
    # -----------------------------
    @staticmethod
    def rotate_z(vec, theta):
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        return R @ vec