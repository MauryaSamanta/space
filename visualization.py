import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scenario_generator import generate_scenario
from edge import (
    build_tasks,
    compute_best_maneuver,
    compute_pair_pc,
    find_best_transfer_time,
    DELTA_T,
    DOCKING_TIME,
    COOLDOWN_TIME
)

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------
STEPS = 300

# -----------------------------
# SIMULATION + VISUALIZATION
# -----------------------------
def run_visual_simulation():

    trajectories = generate_scenario(3)

    sat_names = [k for k in trajectories if "DEBRIS" not in k]

    # OOS state
    oos = {
        "position": np.array(trajectories[sat_names[0]][0]),
        "velocity": np.array([0, 0, 0]),
        "busy_until": 0,
        "fuel": 10000
    }

    cooldown = {}
    current_time = 0

    # -----------------------------
    # MATPLOTLIB SETUP
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-9000, 9000)
    ax.set_ylim(-9000, 9000)
    ax.set_title("OOS Collision Avoidance Visualization")

    # orbit paths
    for name, traj in trajectories.items():
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]

        if "DEBRIS" in name:
            ax.plot(xs, ys, 'r--', alpha=0.3)
        else:
            ax.plot(xs, ys, 'b-', alpha=0.3)

    # dynamic dots
    dots = {}
    for name in trajectories:
        color = 'red' if "DEBRIS" in name else 'blue'
        dot, = ax.plot([], [], 'o', color=color)
        dots[name] = dot

    # OOS
    oos_dot, = ax.plot([], [], 'go', markersize=10)

    # event text
    event_text = ax.text(-8500, 8500, "", fontsize=10)

    events = []

    # -----------------------------
    # UPDATE FUNCTION
    # -----------------------------
    def update(step):
        nonlocal current_time, oos

        # update positions
        for name, traj in trajectories.items():
            if step < len(traj):
                x, y, _ = traj[step]
                dots[name].set_data(x, y)

        # update OOS
        oos_dot.set_data(oos["position"][0], oos["position"][1])

        # ---- LOGIC ----
        if current_time < oos["busy_until"]:
            event_text.set_text(f"⏳ OOS BUSY...")
            current_time += DELTA_T
            return

        tasks = build_tasks(trajectories)
        tasks = [t for t in tasks if t["Pc"] > 0.05]

        if not tasks:
            event_text.set_text("✅ No high-risk collisions")
            current_time += DELTA_T
            return

        feasible = []

        for t in tasks:
            s1, s2 = t["sat1"], t["sat2"]

            for target in [s1, s2]:
                pos = np.array(trajectories[target][step])

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

                if delta_v > oos["fuel"]:
                    continue

                if tof + DOCKING_TIME > t["t_collision"]:
                    continue

                if target in cooldown and current_time < cooldown[target]:
                    continue

                priority = t["Pc"] / (t["t_collision"] + 1e-6)

                feasible.append({
                    "target": target,
                    "Pc": t["Pc"],
                    "tof": tof,
                    "delta_v": delta_v,
                    "priority": priority
                })

        if not feasible:
            event_text.set_text("❌ No reachable collisions")
            current_time += DELTA_T
            return

        best = max(feasible, key=lambda x: x["priority"])
        target = best["target"]

        # ---- MANEUVER ----
        result = compute_best_maneuver(target, trajectories)

        if result["trajectory"] is None:
            event_text.set_text("❌ No valid maneuver")
            current_time += DELTA_T
            return

        # BEFORE
        satA = target
        satB = target + "_DEBRIS"

        dist_before, pc_before = compute_pair_pc(satA, satB, trajectories)

        # apply maneuver
        trajectories[target] = result["trajectory"]

        dist_after, pc_after = compute_pair_pc(satA, satB, trajectories)

        reduction = pc_before - pc_after

        # update OOS
        oos["position"] = np.array(trajectories[target][step])
        oos["busy_until"] = current_time + best["tof"] + DOCKING_TIME
        oos["fuel"] -= best["delta_v"]

        cooldown[target] = current_time + COOLDOWN_TIME

        # event display
        event_text.set_text(
            f"🚀 Target: {target}\n"
            f"Pc: {pc_before:.2f} → {pc_after:.2f}\n"
            f"ΔV: {result['delta_v']:.3f}"
        )

        current_time += DELTA_T

    # -----------------------------
    # ANIMATION
    # -----------------------------
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=STEPS,
        interval=80,
        repeat=True
    )

    plt.show()


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_visual_simulation()