import numpy as np
from physics.collision import collision_probability
from physics.propagation import rk4_step
from config import DELTA_T, MU
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
        self.time_to_tca = tca_steps * delta_t

        self.d_min = d_min
        self.Pc_now = Pc_now

        time_factor = 1.0 / (self.time_to_tca + 1.0)  # 🔥 FIX

        self.priority = (
            10 * Pc_now +
            5 * time_factor +
            1 * (1.0 / (d_min + 1e-3))
        )

        self.assigned = False
        self.completed = False

    def __repr__(self):
        return f"<Mission {self.sat} | priority={self.priority:.4f} | TCA={self.time_to_tca:.2f}>"


    
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

        # ✅ EARLY detection
        if Pc_now > 5e-4 or (d_min < 1.0 and Pc_now > 1e-5):
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
def estimate_transfer_time(oos):
    r = np.linalg.norm(oos.r)
    orbital_period = 2 * np.pi * np.sqrt(r**3 / MU)
    return 0.25 * orbital_period  # rough estimate

def is_phasing_feasible(oos, target_obj, mission, max_delta_a=300):
    
    # current geometry
    theta_oos = np.arctan2(oos.r[1], oos.r[0])
    theta_target = np.arctan2(target_obj["r"][1], target_obj["r"][0])

    dtheta = theta_target - theta_oos
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi

    # orbit
    r = np.linalg.norm(oos.r)
    a0 = r  # near circular assumption
    T0 = 2 * np.pi * np.sqrt(a0**3 / MU)

    time_to_tca = mission.time_to_tca

    # try feasible Δa range
    for delta_a in [-300, -200, -150, -120, -100, -80, -60, -40, -20,
                     20, 40, 60, 80, 100, 120, 150, 200, 300]:

        # direction constraint
        if dtheta > 0 and delta_a > 0:
            continue
        if dtheta < 0 and delta_a < 0:
            continue

        a1 = a0 + delta_a
        T1 = 2 * np.pi * np.sqrt(a1**3 / MU)

        if abs(T0 - T1) < 1e-6:
            continue

        dt = (dtheta * T1 * T0) / (2 * np.pi * (T0 - T1))

        if dt < 0:
            dt += T1

        # ✅ allow multi-rev
        for k in range(3):
            dt_k = dt + k * T1

            if dt_k < time_to_tca - 300:
                return True

    return False

def assign_mission(oos, missions):
    best = None
    best_score = -1e9

    transfer_time = estimate_transfer_time(oos)

    for m in missions:
        if m.assigned or m.completed:
            continue

        
        
        # ❌ Skip unreachable missions
        if m.time_to_tca < transfer_time:
            continue

        

        if m.Pc_now < 1e-4:
            continue

        # simple fuel penalty
        fuel_penalty = 0.1 * transfer_time

        score = m.priority - fuel_penalty

        if score > best_score:
            best_score = score
            best = m

    if best:
        oos.target = best.sat
        oos.state = "PLANNING"

        best.assigned = True
        oos.current_mission = best

        if best.Pc_now < 5e-4:
            print("⚠️ Ignoring low-risk mission")
            return False

        print(f"\n🛰️ Assigned mission → {best.sat}")
        print(f"Priority: {best.priority:.4f} | TCA: {best.time_to_tca:.2f}")

        return True

    return False