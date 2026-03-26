import numpy as np
from config import MU, DOCKING_TIME
from physics.collision import collision_probability


class OOS:
    def __init__(self, r, v):
        self.r = r.copy()
        self.v = v.copy()

        self.state = "IDLE"
        self.target = None

        self.fuel = 5.0

        self.dock_end_time = 0
        self.current_mission = None
        self.phase_direction = 0

    # ---------------- BASIC ----------------

    def propagate(self, rk4_step, dt):
        self.r, self.v = rk4_step(self.r, self.v, dt)

    def get_angle(self):
        return np.arctan2(self.r[1], self.r[0])

    # ---------------- ENTRY POINT ----------------

    def step(self, state, current_time):
        if self.state == "PHASING":
            self._handle_phasing(state)

        elif self.state == "DOCKING":
            self._handle_docking(current_time)

        elif self.state == "MANEUVER":
            self._handle_maneuver(state)

    # ---------------- START PHASING ----------------

    def start_phasing(self, state, current_time):
        target = self.target

        theta_oos = self.get_angle()
        theta_target = np.arctan2(state[target]["r"][1], state[target]["r"][0])

        phase_error = theta_target - theta_oos
        phase_error = (phase_error + np.pi) % (2 * np.pi) - np.pi

        direction = -1 if phase_error > 0 else 1

        tangent = self.v / np.linalg.norm(self.v)
        dv = 0.01 * direction * tangent

        self.v += dv
        self.fuel -= np.linalg.norm(dv)

        self.phase_direction = direction
        self.state = "PHASING"

        # 🔥 ADD THIS
        self.dock_end_time = current_time + DOCKING_TIME

        print(f"🚀 PHASING STARTED → {target}")
        print(f"Initial phase error: {phase_error:.4f}")

    # ---------------- PHASING ----------------

    def _handle_phasing(self, state):
        target = self.target

        theta_oos = self.get_angle()
        theta_target = np.arctan2(state[target]["r"][1], state[target]["r"][0])

        phase_error = theta_target - theta_oos
        phase_error = (phase_error + np.pi) % (2 * np.pi) - np.pi

        print(f"→ Phasing... error: {phase_error:.4f}")

        # dynamic direction
        direction = -1 if phase_error > 0 else 1

        # small correction
        tangent = self.v / np.linalg.norm(self.v)
        dv = 0.0005 * direction * tangent

        self.v += dv
        self.fuel -= np.linalg.norm(dv)

        # alignment
        if abs(phase_error) < 0.02:
            print("✅ Phase aligned → restoring orbit")

            r_norm = np.linalg.norm(self.r)
            v_circular = np.sqrt(MU / r_norm)

            current_speed = np.linalg.norm(self.v)
            dv_mag = v_circular - current_speed

            dv_restore = dv_mag * (self.v / np.linalg.norm(self.v))

            self.v += dv_restore
            self.fuel -= abs(dv_mag)

            self.state = "DOCKING"
           

    # ---------------- DOCKING ----------------

    def _handle_docking(self, current_time):
        print(f"→ Docking with {self.target}")

        if current_time >= self.dock_end_time:
            self.state = "MANEUVER"
            print("✅ DOCKED → Executing maneuver")

    # ---------------- MANEUVER ----------------

    def _handle_maneuver(self, state):
        target = self.target
        debris = target + "_DEBRIS"

        print("\n🚀 MANEUVER EXECUTION")

        pc_before, d_before = collision_probability(
            state[target], state[debris]
        )

        # GO / NO-GO
        if pc_before < 0.01 and d_before > 1.0:
            print("⚠️ Skipping maneuver (not a real threat)")
            self._finish_mission()
            return

        print("\n--- BEFORE ---")
        print(f"Min Distance: {d_before:.4f} km")
        print(f"Pc: {pc_before:.6f}")

        # perpendicular burn
        rel_v = state[target]["v"] - state[debris]["v"]
        rel_v_hat = rel_v / np.linalg.norm(rel_v)

        temp = np.array([1, 0, 0])
        if abs(np.dot(temp, rel_v_hat)) > 0.9:
            temp = np.array([0, 1, 0])

        perp = np.cross(rel_v_hat, temp)
        perp /= np.linalg.norm(perp)

        dv = perp * 0.01
        state[target]["v"] += dv

        print("\n--- ΔV APPLIED ---")
        print(f"ΔV Magnitude: {np.linalg.norm(dv):.6f} km/s")

        pc_after, d_after = collision_probability(
            state[target], state[debris]
        )

        print("\n--- AFTER ---")
        print(f"Min Distance: {d_after:.4f} km")
        print(f"Pc: {pc_after:.6f}")

        print("\n--- RESULT ---")
        print(f"Risk Reduction: {pc_before - pc_after:.6f}")

        self._finish_mission()

    # ---------------- CLEANUP ----------------

    def _finish_mission(self):
        self.state = "IDLE"
        self.target = None

        if self.current_mission:
            self.current_mission.completed = True
            self.current_mission = None