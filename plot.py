import plotly.graph_objects as go
import numpy as np

from fetch_tle import get_satellites
from simulate import get_trajectory
from collision import collision_probability

from poliastro.iod import lambert
from poliastro.bodies import Earth
from poliastro.twobody.orbit import Orbit
from astropy import units as u

# ---------- LOAD ----------
satellites = get_satellites(3)

trajectories = {
    sat["name"]: get_trajectory(sat)
    for sat in satellites
}

# ---------- HELPERS ----------
def closest_approach(traj1, traj2):
    min_dist = float("inf")
    min_idx = 0

    for i in range(min(len(traj1), len(traj2))):
        d = np.linalg.norm(np.array(traj1[i]) - np.array(traj2[i]))
        if d < min_dist:
            min_dist = d
            min_idx = i

    return min_dist, min_idx

# ---------- FIND MOST RISKY PAIR ----------
names = list(trajectories.keys())

worst_pair = None
worst_pc = 0

for i in range(len(names)):
    for j in range(i+1, len(names)):
        n1, n2 = names[i], names[j]

        d, _ = closest_approach(
            trajectories[n1],
            trajectories[n2]
        )

        pc = collision_probability(d)

        if pc > worst_pc:
            worst_pc = pc
            worst_pair = (n1, n2)

satA, satB = worst_pair

trajA = trajectories[satA]
trajB = trajectories[satB]

# ---------- TCA ----------
_, tca_idx = closest_approach(trajA, trajB)

pA = np.array(trajA[tca_idx])
pB = np.array(trajB[tca_idx])

direction = pA - pB
direction = direction / np.linalg.norm(direction)

# ---------- APPLY SAME LOGIC AS SIMULATION ----------
best_traj = None
best_pc = float("inf")

for scale in [0.005, 0.01, 0.02]:
    new_traj = []

    for i in range(len(trajA)):
        new_traj.append(tuple(np.array(trajA[i]) + direction * scale * i * 10))

    d, _ = closest_approach(new_traj, trajB)
    pc = collision_probability(d)

    if pc < best_pc:
        best_pc = pc
        best_traj = new_traj

# ---------- LAMBERT (OPTIONAL OOS VISUAL) ----------
r1 = trajA[0]
r2 = trajB[0]

def lambert_path(r1, r2, tof=5000):
    r1 = np.array(r1) * u.km
    r2 = np.array(r2) * u.km
    tof = tof * u.s

    (v1, v2), = lambert(Earth.k, r1, r2, tof)

    orbit = Orbit.from_vectors(Earth, r1, v1)

    ts = np.linspace(0, tof.value, 100) * u.s

    pts = []
    for t in ts:
        pts.append(orbit.propagate(t).r.to(u.km).value)

    return pts

oos_path = lambert_path(r1, r2)

# ---------- PLOT ----------
fig = go.Figure()

# all satellites
for name, traj in trajectories.items():
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in traj],
        y=[p[1] for p in traj],
        z=[p[2] for p in traj],
        mode='lines',
        name=name,
        opacity=0.3
    ))

# highlight risky pair BEFORE
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in trajA],
    y=[p[1] for p in trajA],
    z=[p[2] for p in trajA],
    mode='lines',
    name=f"{satA} (BEFORE)",
    line=dict(width=6, dash='dash')
))

# AFTER maneuver
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in best_traj],
    y=[p[1] for p in best_traj],
    z=[p[2] for p in best_traj],
    mode='lines',
    name=f"{satA} (AFTER)",
    line=dict(width=6)
))

# other satellite
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in trajB],
    y=[p[1] for p in trajB],
    z=[p[2] for p in trajB],
    mode='lines',
    name=f"{satB}"
))

# OOS path
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in oos_path],
    y=[p[1] for p in oos_path],
    z=[p[2] for p in oos_path],
    mode='lines',
    name="OOS Path"
))

# EARTH
radius = 6371
phi = np.linspace(0, np.pi, 50)
theta = np.linspace(0, 2*np.pi, 50)

x = radius * np.outer(np.sin(phi), np.cos(theta))
y = radius * np.outer(np.sin(phi), np.sin(theta))
z = radius * np.outer(np.cos(phi), np.ones_like(theta))

fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.4, showscale=False))

fig.write_html("real_simulation_visualization.html")