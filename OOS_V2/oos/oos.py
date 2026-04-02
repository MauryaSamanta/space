import numpy as np
from config import MAX_DISTANCE, MAX_LAMBERT_DV, CW_DISTANCE
from physics.cw import cw_docking_velocity


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
            return self._lambert_like()
        else:
            return self._cw_control()

    # -------------------
    # LAMBERT-LIKE (SIMPLIFIED)
    # -------------------
    def _lambert_like(self):

        direction = self.target.r - self.r
        direction = direction / np.linalg.norm(direction)

        desired_v = direction * np.linalg.norm(self.v)

        dv = np.linalg.norm(desired_v - self.v)

        if dv > MAX_LAMBERT_DV:
            return None

        return desired_v

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