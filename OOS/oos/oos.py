import numpy as np

class OOS:
    def __init__(self, r, v):
        self.r = r.copy()
        self.v = v.copy()

        self.state = "IDLE"
        self.target = None

        self.fuel = 5.0

        self.dock_end_time = 0

    def propagate(self, rk4_step, dt):
        self.r, self.v = rk4_step(self.r, self.v, dt)

    def get_angle(self):
        return np.arctan2(self.r[1], self.r[0])