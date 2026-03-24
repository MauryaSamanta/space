from fetch_tle import get_satellites
from simulate import get_trajectory
from collision import collision_probability
from poliastro.iod import lambert
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np
import math
import datetime

from scenario_generator import generate_scenario

# ---------- GLOBAL SETTINGS ----------
STEP_MINUTES = 2
DELTA_T = STEP_MINUTES * 60  # seconds
DOCKING_TIME = 300  # seconds
OOS_SPEED = 7500  # m/s

COOLDOWN_TIME = 2000  # seconds


# ---------- DISTANCE ----------
def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2 +
        (p1[2] - p2[2])**2
    )


def lambert_transfer(oos_r, oos_v, target_r, tof):
    try:
        r1 = np.array(oos_r) * u.km
        r2 = np.array(target_r) * u.km
        tof = tof * u.s

        (v1, v2), = lambert(Earth.k, r1, r2, tof)

        v1 = v1.to(u.km / u.s).value
        v2 = v2.to(u.km / u.s).value

        delta_v = np.linalg.norm(v1 - np.array(oos_v))

        return {
            "delta_v": delta_v,
            "v1": v1,
            "v2": v2   # 👈 ADD THIS
        }

    except Exception:
        return None
    
def generate_oos_trajectory(start_pos, velocity, steps, dt):
    traj = []
    pos = np.array(start_pos)
    vel = np.array(velocity)

    for i in range(steps):
        pos = pos + vel * dt
        traj.append(tuple(pos))

    return traj

def compute_pair_pc(sat1, sat2, trajectories):
    traj1 = trajectories[sat1]
    traj2 = trajectories[sat2]

    d, _, _ = closest_approach(traj1, traj2)
    pc = collision_probability(d)

    return d, pc

# ---------- CLOSEST APPROACH ----------
def closest_approach(traj1, traj2):
    min_dist = float("inf")
    min_step = 0

    for i in range(min(len(traj1), len(traj2))):
        d = distance(traj1[i], traj2[i])

        if d < min_dist:
            min_dist = d
            min_step = i

    t_collision = min_step * DELTA_T
    return min_dist, min_step, t_collision


# ---------- RENDEZVOUS TIME ----------
def estimate_rendezvous_time(oos_pos, sat_pos):
    d = distance(oos_pos, sat_pos)
    return d / OOS_SPEED


# ---------- APPLY ACTION ----------
def apply_action(traj, action, scale=300):
    ax, ay, az = action
    return [(x + ax*scale, y + ay*scale, z + az*scale) for (x, y, z) in traj]


# ---------- ACTION SPACE ----------
ACTIONS = [
    (1,0,0), (-1,0,0),
    (0,1,0), (0,-1,0),
    (0,0,1), (0,0,-1),
    (2,0,0), (0,2,0), (0,0,2),
    (-2,0,0), (0,-2,0), (0,0,-2)
]


# ---------- ACTION EVALUATION ----------
def compute_best_maneuver(target, trajectories):
    traj_target = trajectories[target]

    best_traj = None
    best_pc = float("inf")
    best_delta_v = None

    for other, traj_other in trajectories.items():
        if other == target:
            continue

        # --- FIND TCA ---
        min_dist, tca_idx, _ = closest_approach(traj_target, traj_other)

        p1 = np.array(traj_target[tca_idx])
        p2 = np.array(traj_other[tca_idx])

        # direction AWAY from collision
        direction = p1 - p2
        norm = np.linalg.norm(direction)

        if norm == 0:
            continue

        direction = direction / norm

        # --- TRY MULTIPLE ΔV SCALES ---
        for scale in [0.02, 0.05, 0.1]:
            # estimate velocity from trajectory
            v_est = (np.array(traj_target[1]) - np.array(traj_target[0])) / DELTA_T

            delta_v = scale * direction
            new_velocity = v_est + delta_v

            # generate new trajectory
            new_traj = []
            for i in range(len(traj_target)):
                    dt = i * DELTA_T
                    new_pos = np.array(traj_target[i]) + delta_v * dt
                    new_traj.append(tuple(new_pos))

            # --- EVALUATE RISK ---
            total_pc = 0
            for other2, traj_other2 in trajectories.items():
                if other2 == target:
                    continue

                d, _, _ = closest_approach(new_traj, traj_other2)
                total_pc += collision_probability(d)

            if total_pc < best_pc:
                best_pc = total_pc
                best_traj = new_traj
                best_delta_v = np.linalg.norm(delta_v)

    return {
        "trajectory": best_traj,
        "delta_v": best_delta_v
    }

# ---------- BUILD TASK LIST ----------
def build_tasks(trajectories):
    tasks = []

    for name in trajectories.keys():
        # only process real satellites
        if "DEBRIS" in name:
            continue

        sat_name = name
        debris_name = name + "_DEBRIS"

        if debris_name not in trajectories:
            continue

        traj_sat = trajectories[sat_name]
        traj_debris = trajectories[debris_name]

        d, _, t_collision = closest_approach(traj_sat, traj_debris)
        Pc = collision_probability(d)

        tasks.append({
            "sat1": sat_name,
            "sat2": debris_name,
            "Pc": Pc,
            "t_collision": t_collision
        })

    return tasks

def find_best_transfer_time(oos_pos, oos_vel, target_pos, t_collision):
    candidate_tofs = np.linspace(300, t_collision * 0.8, 8)  # try multiple TOFs

    best = None
    best_dv = float("inf")

    for tof in candidate_tofs:
        result = lambert_transfer(oos_pos, oos_vel, target_pos, tof)

        if result is None:
            continue

        delta_v = result["delta_v"]

        if delta_v < best_dv:
            best_dv = delta_v
            best = {
                "tof": tof,
                "delta_v": delta_v,
                "v1": result["v1"]
            }

    return best

# ---------- MAIN SIMULATION ----------
def run_simulation():
    # satellites = get_satellites(10)

    # FIXED START TIME (IMPORTANT)
    start_time = datetime.datetime.utcnow()

    # trajectories = {
    #     sat["name"]: get_trajectory(sat)
    #     for sat in satellites
    # }
    trajectories = generate_scenario(5)

    # OOS STATE
    sat_names = [k for k in trajectories.keys() if "DEBRIS" not in k]

    first_sat = sat_names[0]

    oos = {
        "position": trajectories[first_sat][0],
        "velocity": [0, 0, 0],
        "busy_until": 0,
        "fuel": 10000
    }

    cooldown = {}

    current_time = 0

    frames = []

    for step in range(500):
        print(f"\n========== STEP {step} ==========")
        frame = {
    "step": step,
    "time": current_time,   # ADD THIS
    "objects": {},
    "oos": list(oos["position"]),
    "events": [],
    "status": "Idle"
}

        for name, traj in trajectories.items():
            if step < len(traj):
                frame["objects"][name] = traj[step]

        frames.append(frame)
        # OOS busy check
        if current_time < oos["busy_until"]:
            print("OOS BUSY...")
            current_time += DELTA_T
            continue

        tasks = build_tasks(trajectories)

        # filter risky
        tasks = [t for t in tasks if t["Pc"] > 0.05]

        if not tasks:
            print("No high-risk collisions")
            current_time += DELTA_T
            continue

        feasible = []

        for t in tasks:
            frame["events"].append({
    "type": "conjunction",
    "sat1": t["sat1"],
    "sat2": t["sat2"],
    "Pc": t["Pc"],
    "tca": t["t_collision"],
    "time_to_tca": t["t_collision"] - current_time
})
            s1, s2 = t["sat1"], t["sat2"]

            pos1 = trajectories[s1][0]
            pos2 = trajectories[s2][0]

            # try both targets
            for target, pos in [(s1, pos1), (s2, pos2)]:

                # choose time of flight (important choice)
                transfer = find_best_transfer_time(
                oos["position"],
                oos["velocity"],
                pos,
                t["t_collision"]
            )

                if transfer is None:
                    continue

                tof = transfer["tof"]
                delta_v = transfer["delta_v"]

                # if result is None:
                #     continue

                # delta_v = result["delta_v"]

                # feasibility check
                if delta_v > oos["fuel"]:
                    continue

                total_time = tof + DOCKING_TIME

                if total_time > t["t_collision"]:
                    continue

                # cooldown check
                if target in cooldown and current_time < cooldown[target]:
                    continue

                priority = t["Pc"] / t["t_collision"]

                feasible.append({
                    "target": target,
                    "Pc": t["Pc"],
                    "time_left": t["t_collision"],
                    "delta_v": delta_v,
                    "tof": tof,
                    "pos": pos,
                    "priority": priority
                })

        if not feasible:
            print("No reachable collisions ❌")
            current_time += DELTA_T
            continue

        best = max(feasible, key=lambda x: x["priority"])

        transfer = find_best_transfer_time(
            oos["position"],
            oos["velocity"],
            best["pos"],
            best["time_left"]
        )

        if transfer is None:
            current_time += DELTA_T
            continue

        target = best["target"]
        frame["status"] = "Maneuver"
        frame["events"].append({
            "type": "maneuver",
            "target": target,
            "Pc": best["Pc"],
            "time_left": best["time_left"]
        })
        print(f"TARGET: {target}")
        print(f"Pc: {best['Pc']:.4f}")
        print(f"Time Left: {best['time_left']:.2f}s")

        result = compute_best_maneuver(target, trajectories)

        if result["trajectory"] is None:
            print("No valid maneuver found ❌")
            current_time += DELTA_T
            continue
        satA = None
        satB = None

        # find the task corresponding to this target
        for t in tasks:
            if t["sat1"] == target or t["sat2"] == target:
                satA = t["sat1"]
                satB = t["sat2"]
                break

        # compute BEFORE
        dist_before, pc_before = compute_pair_pc(satA, satB, trajectories)

        print("\n--- BEFORE MANEUVER ---")
        print(f"Pair: {satA} vs {satB}")
        print(f"Min Distance: {dist_before:.2f} km")
        print(f"Pc: {pc_before:.4f}")
        # apply best maneuver
        trajectories[target] = result["trajectory"]

        print(f"Applied ΔV: {result['delta_v']:.6f}")
        dist_after, pc_after = compute_pair_pc(satA, satB, trajectories)

        print("\n--- AFTER MANEUVER ---")
        print(f"Min Distance: {dist_after:.2f} km")
        print(f"Pc: {pc_after:.4f}")

        reduction = pc_before - pc_after

        print("\n--- RESULT ---")
        print(f"Risk Reduction: {reduction:.4f}")
        # update OOS
        steps = int(transfer["tof"] // DELTA_T)

        oos_traj = generate_oos_trajectory(
            oos["position"],
            transfer["v1"],
            steps,
            DELTA_T
        )

        oos["trajectory"] = oos_traj
        oos["traj_index"] = 0
        oos["busy_until"] = current_time + best["tof"] + DOCKING_TIME
        oos["fuel"] -= best["delta_v"]

        # cooldown
        cooldown[target] = current_time + COOLDOWN_TIME

        print(f"OOS BUSY UNTIL: {oos['busy_until']:.2f}")
        print(f"Fuel: {oos['fuel']}")
        if "trajectory" in oos and oos["traj_index"] < len(oos["trajectory"]):
            oos["position"] = oos["trajectory"][oos["traj_index"]]
            oos["traj_index"] += 1
        current_time += DELTA_T
       
    import json

    with open("../web/data.json", "w") as f:
        json.dump(frames, f)


# ---------- RUN ----------
if __name__ == "__main__":
    run_simulation()