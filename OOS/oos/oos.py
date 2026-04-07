import numpy as np
from config import DELTA_T, MU, DOCKING_TIME

from physics.collision import collision_probability
from physics.propagation import rk4_step
from physics.cw import cw_transfer
from physics.planner import lambert_transfer_safe, is_far
from physics.lambert import predict_future


class OOS:
    def __init__(self, r, v):
        self.r = r.copy()
        self.v = v.copy()

        self.state = "IDLE"
        self.target = None
        self.current_mission = None

        self.fuel = 5.0

        self.transfer_end_time = None
        self.dock_end_time = None
        self.hold_end_time = None
        self.match_start_time = None
        
        self.failed_sats = set()

    # =============================
    # MAIN STEP
    # =============================
    def step(self, state, current_time):

        if self.state == "TRANSFER":
            self._handle_transfer(state, current_time)

        elif self.state == "VEL_MATCH":
            self._handle_velocity_match(state, current_time)

        elif self.state == "DOCKING_PREP":
            self._start_cw(state, current_time)

        elif self.state == "DOCKING":
            self._handle_docking(current_time)

        elif self.state == "MANEUVER":
            self._handle_maneuver(state)

        elif self.state == "HOLD":
            self._handle_hold(state, current_time)

        elif self.state == "PLANNING":
            self.start_mission(state, current_time)

    # =============================
    # START MISSION
    # =============================
    def start_mission(self, state, current_time):

        target = self.target
        target_obj = state[target]
        mission = self.current_mission

        print(f"\n🚀 STARTING MISSION → {target}")

        oos_state = {"r": self.r.copy(), "v": self.v.copy()}

        # CLOSE → CW
        if not is_far(oos_state, target_obj):
            print("🎯 Using CW (close target)")
            self._start_cw(state, current_time)
            return

        # FAR → LAMBERT
        print("🛰️ Using Lambert (far target)")

        transfer = lambert_transfer_safe(
            oos_state,
            target_obj,
            predict_future,
            mission.time_to_tca
        )

        if transfer is None:
            print("❌ Lambert failed")
            self.failed_sats.add(self.target)
            self._finish(success=False)
            return

        dv = transfer["v1"] - self.v
        self._apply_dv(dv)

        self.transfer_end_time = current_time + transfer["tof"]
        self.state = "TRANSFER"

    # =============================
    # TRANSFER
    # =============================
    def _handle_transfer(self, state, current_time):
        self._propagate()

        if current_time >= self.transfer_end_time:
            print("📍 Transfer complete → starting velocity match")
            self.state = "VEL_MATCH"
            self.match_start_time = current_time

    # =============================
    # VELOCITY MATCH (FIXED)
    # =============================
    def _handle_velocity_match(self, state, current_time):

        self._propagate()

        target_obj = state[self.target]

        rel_v = self.v - target_obj["v"]
        rel_v_mag = np.linalg.norm(rel_v)

        print(f"🔧 Matching velocity | rel_v = {rel_v_mag:.3f}")

        # SUCCESS
        if rel_v_mag < 0.2:
            print("✅ Velocity matched → CW")
            self.state = "DOCKING_PREP"
            return

        # TIMEOUT (prevents infinite loop)
        if current_time - self.match_start_time > 2000:
            print("❌ Velocity match timeout")
            self._finish(success=False)
            return

        # CONTROLLED DAMPING (CRITICAL FIX)
        dv = -rel_v * 0.2
        self._apply_dv(dv)

    # =============================
    # CW START
    # =============================
    def _start_cw(self, state, current_time):
        target_obj = state[self.target]

        rel_v = np.linalg.norm(self.v - target_obj["v"])
        dist = np.linalg.norm(self.r - target_obj["r"])

        print(f"📏 Pre-CW | d={dist:.2f}, rel_v={rel_v:.3f}")

        # SAFETY CHECK
        if rel_v > 0.3:
            print("❌ Too fast → back to velocity match")
            self.state = "VEL_MATCH"
            return

        print("🎯 Starting CW docking")

        oos_state = {"r": self.r.copy(), "v": self.v.copy()}

        dv = self._compute_best_cw(oos_state, target_obj)

        if dv is None:
            print("❌ CW failed")
            self._finish(success=False)
            return

        self._apply_dv(dv)

        self.dock_end_time = current_time + DOCKING_TIME
        self.state = "DOCKING"

    # =============================
    # DOCKING
    # =============================
    def _handle_docking(self, current_time):
        print(f"→ Docking with {self.target}")

        if current_time >= self.dock_end_time:
            print("✅ DOCKED → Executing maneuver")
            self.state = "MANEUVER"

    # =============================
    # MANEUVER
    # =============================
    def _handle_maneuver(self, state):

        target = self.target
        debris = target + "_DEBRIS"

        print("\n🚀 MANEUVER EXECUTION")

        pc, d = collision_probability(state[target], state[debris])
        print(f"Before → Pc: {pc:.6f}, d: {d:.4f}")

        if pc < 1e-4 and d > 1.0:
            print("⚠️ Not a real threat → skipping")
            self._finish()
            return

        dv = self._compute_avoidance_dv(state[target], state[debris])
        state[target]["v"] += dv

        pc2, d2 = collision_probability(state[target], state[debris])

        print(f"After → Pc: {pc2:.6f}, d: {d2:.4f}")
        print(f"ΔV: {np.linalg.norm(dv):.6f}")

        if hasattr(self, "metrics") and pc2 < 1e-5:
            self.metrics.log_collision_avoided()

        self.state = "HOLD"
        self.hold_end_time = None

    # =============================
    # HOLD
    # =============================
    def _handle_hold(self, state, current_time):

        target = self.target
        debris = target + "_DEBRIS"

        pc, d = collision_probability(state[target], state[debris])

        print(f"🟡 HOLD → Pc: {pc:.6f}, d: {d:.3f}")

        if pc < 1e-6:
            print("✅ Safe → leaving target")
            self._finish()
            return

        if self.hold_end_time is None:
            self.hold_end_time = current_time + 500

        if current_time >= self.hold_end_time:
            print("✅ Timeout safe → leaving")
            self._finish()
            return

        if pc > 1e-4 or d < 1.0:
            print("⚠️ Still risky → correcting")
            self.state = "MANEUVER"

    # =============================
    # HELPERS
    # =============================
    def _apply_dv(self, dv):
        self.v += dv
        self.fuel -= np.linalg.norm(dv)

        if hasattr(self, "metrics"):
            self.metrics.log_dv(dv)

    def _propagate(self):
        self.r, self.v = rk4_step(self.r, self.v, DELTA_T)

    def _compute_best_cw(self, oos_state, target_obj):
        best = None
        best_dv = float("inf")

        for tof in np.arange(50, 2000, 20):
            try:
                res = cw_transfer(oos_state, target_obj, tof)
                if res is not None and res["dv_mag"] < 5.0:
                    if res["dv_mag"] < best_dv:
                        best_dv = res["dv_mag"]
                        best = res
            except:
                continue

        if best is None:
            print("❌ No CW solution")
            return None

        return best["dv"]

    def _compute_avoidance_dv(self, sat, debris):
        rel_v = sat["v"] - debris["v"]
        rel_v_hat = rel_v / np.linalg.norm(rel_v)

        temp = np.array([1, 0, 0])
        if abs(np.dot(temp, rel_v_hat)) > 0.9:
            temp = np.array([0, 1, 0])

        perp = np.cross(rel_v_hat, temp)
        perp /= np.linalg.norm(perp)

        pc, _ = collision_probability(sat, debris)
        dv_mag = max(0.01, min(0.1, pc))

        print("Applying DV:", dv_mag)

        return perp * dv_mag

    # =============================
    # CLEANUP
    # =============================
    def _finish(self, success=True):

        self.state = "IDLE"
        self.target = None

        if hasattr(self, "metrics") and hasattr(self, "mission_start_time") and success:
            response_time = self.current_time - self.mission_start_time
            self.metrics.log_mission_complete(response_time)

        self.transfer_end_time = None
        self.dock_end_time = None
        self.hold_end_time = None
        self.match_start_time = None

        if self.current_mission:
            self.current_mission.completed = True
            self.current_mission = None