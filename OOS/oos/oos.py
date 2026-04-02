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

    # =============================
    # MAIN STEP
    # =============================
    def step(self, state, current_time):

        if self.state == "TRANSFER":
            self._handle_transfer(state, current_time)

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

        # -------- CLOSE → CW --------
        if not is_far(oos_state, target_obj):
            print("🎯 Using CW (close target)")
            self._start_cw(state, current_time)
            return

        # -------- FAR → LAMBERT --------
        print("🛰️ Using Lambert (far target)")

        transfer = lambert_transfer_safe(
            oos_state,
            target_obj,
            predict_future,
            mission.time_to_tca
        )

        if transfer is None:
            print("❌ Lambert failed")
            self._finish()
            return

        dv = transfer["v1"] - self.v
        self._apply_dv(dv)

        self.transfer_end_time = current_time + transfer["tof"]
        self.state = "TRANSFER"

    # =============================
    # TRANSFER (NO LOOP NOW)
    # =============================
    def _handle_transfer(self, state, current_time):
        self._propagate()

        target_obj = state[self.target]
        dist = np.linalg.norm(self.r - target_obj["r"])

        if dist < 200 or current_time >= self.transfer_end_time:
            print("📍 Transfer complete → switching to CW")

            self.transfer_end_time = None

            # 🔥 DIRECTLY GO TO CW (NO PLANNING LOOP)
            self._start_cw(state, current_time)

    # =============================
    # CW START (CRITICAL FIX)
    # =============================
    def _start_cw(self, state, current_time):
        target_obj = state[self.target]

        print("🎯 Starting CW docking")

        oos_state = {"r": self.r.copy(), "v": self.v.copy()}

        dv = self._compute_best_cw(oos_state, target_obj)
        if dv is None:
            print("❌ CW failed")
            self._finish()
            return

        self._apply_dv(dv)

        self.dock_end_time = current_time + DOCKING_TIME
        self.state = "DOCKING"

    # =============================
    # DOCKING
    # =============================
    def _handle_docking(self, current_time):
        print(f"→ Docking with {self.target}")

        if self.dock_end_time is None:
            raise RuntimeError("dock_end_time not set")

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

        self.state = "HOLD"
        self.hold_end_time = None

    # =============================
    # HOLD (SMART CONTROL)
    # =============================
    def _handle_hold(self, state, current_time):
        target = self.target
        debris = target + "_DEBRIS"

        pc, d = collision_probability(state[target], state[debris])

        print(f"🟡 HOLD → Pc: {pc:.6f}, d: {d:.3f}")

        if pc > 1e-4 or d < 1.0:
            print("⚠️ Still risky → correcting again")
            self.state = "MANEUVER"
            return

        if self.hold_end_time is None:
            self.hold_end_time = current_time + 500

        if current_time >= self.hold_end_time:
            print("✅ Safe → leaving target")
            self._finish()

    # =============================
    # HELPERS
    # =============================
    def _apply_dv(self, dv):
        self.v += dv
        self.fuel -= np.linalg.norm(dv)

    def _propagate(self):
        self.r, self.v = rk4_step(self.r, self.v, DELTA_T)

    def _compute_best_cw(self, oos_state, target_obj):
        best = None
        best_dv = float("inf")

        for tof in range(1000, 3000, 200):
            res = cw_transfer(oos_state, target_obj, tof)
            if res["dv_mag"] < best_dv:
                best_dv = res["dv_mag"]
                best = res

        return None if best is None else best["dv"]

    def _compute_avoidance_dv(self, sat, debris):
        rel_v = sat["v"] - debris["v"]
        rel_v_hat = rel_v / np.linalg.norm(rel_v)

        temp = np.array([1, 0, 0])
        if abs(np.dot(temp, rel_v_hat)) > 0.9:
            temp = np.array([0, 1, 0])

        perp = np.cross(rel_v_hat, temp)
        perp /= np.linalg.norm(perp)

        pc, _ = collision_probability(sat, debris)
        dv_mag = min(0.1, pc)

        return perp * dv_mag

    # =============================
    # CLEANUP
    # =============================
    def _finish(self):
        self.state = "IDLE"
        self.target = None

        self.transfer_end_time = None
        self.dock_end_time = None
        self.hold_end_time = None

        if self.current_mission:
            self.current_mission.completed = True
            self.current_mission = None