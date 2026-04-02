import numpy as np
from config import MAX_DISTANCE, MAX_LAMBERT_DV, CW_DISTANCE
from physics.cw import cw_docking_velocity
from physics.lambert import solve_lambert
from config import LAMBERT_TOF, MAX_LAMBERT_DV

class OOS:

    def __init__(self, r, v):
        self.r = r
        self.v = v
        self.state = "IDLE"
        self.target = None

    # -------------------
    # DECISION
    # -------------------
    def assign_mission(self, missions):

        if not missions:
            return

        # pick highest risk
        missions.sort(key=lambda m: -m.pc)
        self.target = missions[0].target
        self.state = "PLANNING"

    # -------------------
    # PLANNER
    # -------------------
    def plan(self):

        if self.target is None:
            return None

        rel = self.target.r - self.r
        dist = np.linalg.norm(rel)

        if dist > MAX_DISTANCE:
            return None

        if dist > CW_DISTANCE:
            return self._lambert_transfer()
        else:
            return self._cw_control()

    # -------------------
    # LAMBERT-LIKE (SIMPLIFIED)
    # -------------------



    def _lambert_transfer(self):

        v_new, dv = solve_lambert(
            self.r,
            self.v,
            self.target.r,
            self.target.v,
            LAMBERT_TOF
        )

        if v_new is None:
            print("[LAM] failed")
            return None

        print(f"[LAM] dv = {dv:.2f}")

        if dv > MAX_LAMBERT_DV:
            print("[LAM] rejected (high dv)")
            return None

        return v_new

    # -------------------
    # CW CONTROL
    # -------------------
    def _cw_control(self):

        rel_r = self.target.r - self.r
        rel_v = self.target.v - self.v

        dv = cw_docking_velocity(rel_r, rel_v)

        return self.v + dv

    # -------------------
    # STEP
    # -------------------
    def step(self, missions):

        if self.state == "IDLE":
            self.assign_mission(missions)

        elif self.state == "PLANNING":
            v_new = self.plan()

            if v_new is not None:
                self.v = v_new
                self.state = "TRANSFER"

        elif self.state == "TRANSFER":

            dist = np.linalg.norm(self.target.r - self.r)

            if dist < 50:
                print("✅ Docked with target")
                self.state = "IDLE"
                self.target = None