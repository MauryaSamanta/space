from fetch_tle import get_satellites
from simulate import get_trajectory
from collision import collision_probability

import math
import datetime

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
def evaluate_actions(sat_name, trajectories):
    traj1 = trajectories[sat_name]

    original_risk = 0
    for other, traj2 in trajectories.items():
        if other == sat_name:
            continue
        d, _, _ = closest_approach(traj1, traj2)
        original_risk += collision_probability(d)

    best = None
    best_score = float("inf")

    for action in ACTIONS:
        new_traj = apply_action(traj1, action)

        total_risk = 0
        for other, traj2 in trajectories.items():
            if other == sat_name:
                continue
            d, _, _ = closest_approach(new_traj, traj2)
            total_risk += collision_probability(d)

        fuel = math.sqrt(sum(a*a for a in action)) * 300
        risk_reduction = original_risk - total_risk

        score = 800 * total_risk - 2000 * risk_reduction + 0.1 * fuel

        if score < best_score:
            best_score = score
            best = {
                "action": action,
                "risk_reduction": risk_reduction
            }

    return best


# ---------- BUILD TASK LIST ----------
def build_tasks(trajectories):
    tasks = []

    names = list(trajectories.keys())

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]

            d, _, t_collision = closest_approach(
                trajectories[n1],
                trajectories[n2]
            )

            Pc = collision_probability(d)

            tasks.append({
                "sat1": n1,
                "sat2": n2,
                "Pc": Pc,
                "t_collision": t_collision
            })

    return tasks


# ---------- MAIN SIMULATION ----------
def run_simulation():
    satellites = get_satellites(10)

    # FIXED START TIME (IMPORTANT)
    start_time = datetime.datetime.utcnow()

    trajectories = {
        sat["name"]: get_trajectory(sat)
        for sat in satellites
    }

    # OOS STATE
    oos = {
        "position": trajectories[satellites[0]["name"]][0],
        "busy_until": 0,
        "fuel": 10000
    }

    cooldown = {}

    current_time = 0

    for step in range(15):
        print(f"\n========== STEP {step} ==========")

        # OOS busy check
        if current_time < oos["busy_until"]:
            print("OOS BUSY...")
            current_time += DELTA_T
            continue

        tasks = build_tasks(trajectories)

        # filter risky
        tasks = [t for t in tasks if t["Pc"] > 0.2]

        if not tasks:
            print("No high-risk collisions")
            current_time += DELTA_T
            continue

        feasible = []

        for t in tasks:
            s1, s2 = t["sat1"], t["sat2"]

            pos1 = trajectories[s1][0]
            pos2 = trajectories[s2][0]

            t1 = estimate_rendezvous_time(oos["position"], pos1)
            t2 = estimate_rendezvous_time(oos["position"], pos2)

            if t1 < t2:
                target = s1
                t_r = t1
                pos = pos1
            else:
                target = s2
                t_r = t2
                pos = pos2

            total_time = t_r + DOCKING_TIME

            if total_time < t["t_collision"]:
                # cooldown check
                if target in cooldown and current_time < cooldown[target]:
                    continue

                priority = t["Pc"] / t["t_collision"]

                feasible.append({
                    "target": target,
                    "Pc": t["Pc"],
                    "time_left": t["t_collision"],
                    "rendezvous": t_r,
                    "pos": pos,
                    "priority": priority
                })

        if not feasible:
            print("No reachable collisions ❌")
            current_time += DELTA_T
            continue

        best = max(feasible, key=lambda x: x["priority"])

        target = best["target"]

        print(f"TARGET: {target}")
        print(f"Pc: {best['Pc']:.4f}")
        print(f"Time Left: {best['time_left']:.2f}s")

        action = evaluate_actions(target, trajectories)

        print(f"Action: {action['action']}")
        print(f"Risk Reduction: {action['risk_reduction']:.4f}")

        # apply
        trajectories[target] = apply_action(
            trajectories[target],
            action["action"]
        )

        # update OOS
        oos["position"] = best["pos"]
        oos["busy_until"] = current_time + best["rendezvous"] + DOCKING_TIME
        oos["fuel"] -= 100

        # cooldown
        cooldown[target] = current_time + COOLDOWN_TIME

        print(f"OOS BUSY UNTIL: {oos['busy_until']:.2f}")
        print(f"Fuel: {oos['fuel']}")

        current_time += DELTA_T


# ---------- RUN ----------
if __name__ == "__main__":
    run_simulation()