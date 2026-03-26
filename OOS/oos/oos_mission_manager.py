import numpy as np
from physics.collision import collision_probability
from physics.propagation import rk4_step
from config import DELTA_T
# from simulation.engine import predict_tca  # adjust import if needed

def predict_tca(obj1, obj2, steps=500):
        r1, v1 = obj1["r"].copy(), obj1["v"].copy()
        r2, v2 = obj2["r"].copy(), obj2["v"].copy()

        min_d = float("inf")
        best_step = 0
        best_r2 = None

        for i in range(steps):
            r1, v1 = rk4_step(r1, v1, DELTA_T)
            r2, v2 = rk4_step(r2, v2, DELTA_T)

            d = np.linalg.norm(r1 - r2)

            if d < min_d:
                min_d = d
                best_step = i
                best_r2 = r2.copy()

        return min_d, best_step, best_r2
# -----------------------------
# MISSION OBJECT
# -----------------------------
class Mission:
    def __init__(self, sat, tca_steps, d_min, Pc_now, current_time, delta_t):
        self.sat = sat

        self.tca_steps = tca_steps
        self.tca_time = current_time + tca_steps * delta_t

        self.d_min = d_min
        self.Pc_now = Pc_now

        # priority score (you can tweak this)
        self.priority = Pc_now + (1.0 / (d_min + 1e-6))

        self.assigned = False
        self.completed = False

    def __repr__(self):
        return f"<Mission {self.sat} | priority={self.priority:.4f}>"


    
# -----------------------------
# GENERATE MISSIONS
# -----------------------------
def generate_missions(state, current_time, delta_t):
    missions = []

    for sat in state:
        if "_DEBRIS" in sat:
            continue

        debris = sat + "_DEBRIS"

        Pc_now, _ = collision_probability(state[sat], state[debris])
        d_min, t_steps, _ = predict_tca(state[sat], state[debris])
        # print(f"TCA steps: {t_steps}")
        # print(f"Distance at TCA: {d_min}")
        # EARLY detection condition
        if Pc_now > 0.01 or d_min < 0.5:
            mission = Mission(
                sat,
                t_steps,
                d_min,
                Pc_now,
                current_time,
                delta_t
            )
            missions.append(mission)

    return missions



# -----------------------------
# SORT MISSIONS BY PRIORITY
# -----------------------------
def prioritize_missions(missions):
    return sorted(missions, key=lambda m: m.priority, reverse=True)



# -----------------------------
# ASSIGN MISSION TO OOS
# -----------------------------
def assign_mission(oos, missions):
    for m in missions:
        if not m.assigned and not m.completed:
            oos.target = m.sat
            oos.state = "PLANNING"

            m.assigned = True
            oos.current_mission = m

            print(f"\n🛰️ Assigned mission → {m.sat}")
            return True

    return False