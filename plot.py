import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

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
# SETTINGS
# -----------------------------
STEPS = 200

# -----------------------------
# INIT SIMULATION
# -----------------------------
trajectories = generate_scenario(3)

sat_names = [k for k in trajectories if "DEBRIS" not in k]
first_sat = sat_names[0]

oos = {
    "position": np.array(trajectories[first_sat][0]),
    "velocity": np.array([0, 0, 0]),
    "busy_until": 0,
    "fuel": 10000
}

cooldown = {}
current_time = 0

# -----------------------------
# PLOT SETUP
# -----------------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-9000, 9000)
ax.set_ylim(-9000, 9000)
ax.set_zlim(-9000, 9000)

ax.set_title("OOS Collision Avoidance (REAL-TIME)")

# orbit lines
for name, traj in trajectories.items():
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    zs = [p[2] for p in traj]

    if "DEBRIS" in name:
        ax.plot(xs, ys, zs, 'r--', alpha=0.2)
    else:
        ax.plot(xs, ys, zs, 'b-', alpha=0.2)

# dynamic points
dots = {}
for name in trajectories:
    color = 'red' if "DEBRIS" in name else 'blue'
    dot, = ax.plot([], [], [], 'o', color=color)
    dots[name] = dot

# OOS
oos_dot, = ax.plot([], [], [], 'go', markersize=8)

# text
text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

# draw Earth
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 30)

x = 6371 * np.outer(np.cos(u), np.sin(v))
y = 6371 * np.outer(np.sin(u), np.sin(v))
z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, alpha=0.3)

# -----------------------------
# UPDATE FUNCTION
# -----------------------------
def update(step):
    global current_time, oos

    # update satellite positions
    for name, traj in trajectories.items():
        if step < len(traj):
            x, y, z = traj[step]
            dots[name].set_data([x], [y])
            dots[name].set_3d_properties([z])

    # update OOS position
    oos_dot.set_data([oos["position"][0]], [oos["position"][1]])
    oos_dot.set_3d_properties([oos["position"][2]])

    # ---------------- LOGIC ----------------
    if current_time < oos["busy_until"]:
        text.set_text("⏳ OOS BUSY")
        current_time += DELTA_T
        return

    tasks = build_tasks(trajectories)
    tasks = [t for t in tasks if t["Pc"] > 0.05]

    if not tasks:
        text.set_text("✅ No collision")
        current_time += DELTA_T
        return

    feasible = []

    for t in tasks:
        for target in [t["sat1"], t["sat2"]]:
            if "DEBRIS" in t["sat1"]:
                target = t["sat2"]
            else:
                target = t["sat1"]
            pos = trajectories[target][step]

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
        text.set_text("❌ No reachable")
        current_time += DELTA_T
        return

    best = max(feasible, key=lambda x: x["priority"])
    target = best["target"]

    # BEFORE
    satA = target
    satB = target + "_DEBRIS"

    dist_before, pc_before = compute_pair_pc(satA, satB, trajectories)

    # APPLY MANEUVER
    result = compute_best_maneuver(target, trajectories)

    if result["trajectory"] is None:
        text.set_text("❌ Maneuver failed")
        current_time += DELTA_T
        return

    trajectories[target] = result["trajectory"]

    dist_after, pc_after = compute_pair_pc(satA, satB, trajectories)

    # update OOS
    oos["position"] = np.array(trajectories[target][step])
    oos["busy_until"] = current_time + best["tof"] + DOCKING_TIME
    oos["fuel"] -= best["delta_v"]

    cooldown[target] = current_time + COOLDOWN_TIME

    text.set_text(
        f"🚀 {target}\n"
        f"Pc: {pc_before:.2f} → {pc_after:.2f}"
    )

    current_time += DELTA_T


# -----------------------------
# ANIMATION
# -----------------------------
ani = FuncAnimation(
    fig,
    update,
    frames=STEPS,
    interval=80,
    repeat=True
)

plt.show()