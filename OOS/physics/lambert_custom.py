import numpy as np
import plotly.graph_objects as go

from poliastro.iod import lambert
from poliastro.bodies import Earth
from astropy import units as u

# -----------------------------
# CONSTANTS
# -----------------------------
MU = 398600.4418  # km^3/s^2
DT = 20
STEPS = 400
TOF = 4000

# -----------------------------
# RK4 PROPAGATION
# -----------------------------
def acceleration(r):
    return -MU * r / np.linalg.norm(r)**3

def rk4(r, v, dt):
    k1_r = v
    k1_v = acceleration(r)

    k2_r = v + 0.5 * dt * k1_v
    k2_v = acceleration(r + 0.5 * dt * k1_r)

    k3_r = v + 0.5 * dt * k2_v
    k3_v = acceleration(r + 0.5 * dt * k2_r)

    k4_r = v + dt * k3_v
    k4_v = acceleration(r + dt * k3_r)

    r_new = r + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    return r_new, v_new

# -----------------------------
# ORBIT INIT
# -----------------------------
def circular_orbit(radius, theta):
    r = np.array([radius*np.cos(theta), radius*np.sin(theta), 0])
    v_mag = np.sqrt(MU / radius)
    v = np.array([-v_mag*np.sin(theta), v_mag*np.cos(theta), 0])
    return r, v

# -----------------------------
# LAMBERT
# -----------------------------
def propagate_future(r, v):
    r_f, v_f = r.copy(), v.copy()
    for _ in range(int(TOF / DT)):
        r_f, v_f = rk4(r_f, v_f, DT)
    return r_f

def solve_lambert(r1, v1, r2, v2):
    r2_future = propagate_future(r2, v2)

    try:
        (v1_new, _), = lambert(
            Earth.k,
            r1 * u.km,
            r2_future * u.km,
            TOF * u.s
        )
        v1_new = v1_new.to(u.km/u.s).value
        dv = np.linalg.norm(v1_new - v1)
        return v1_new, dv
    except:
        return None, None

# -----------------------------
# INIT SYSTEM
# -----------------------------
R = 7000

angles = [0, 0.4, 0.8]

sats = []
for a in angles:
    r, v = circular_orbit(R, a)
    sats.append({"r": r, "v": v})

debris = []
for sat in sats:
    r = sat["r"] + np.random.normal(0, 0.05, 3)
    v = sat["v"] + np.random.normal(0, 0.0005, 3)
    debris.append({"r": r, "v": v})

r_oos, v_oos = circular_orbit(R, 1.5)
oos = {"r": r_oos, "v": v_oos, "target": 0, "state": "IDLE"}

# -----------------------------
# STORAGE
# -----------------------------
traj_sats = [[] for _ in sats]
traj_debris = [[] for _ in debris]
traj_oos = []

# -----------------------------
# SIMULATION
# -----------------------------
for step in range(STEPS):

    for i in range(len(sats)):
        sats[i]["r"], sats[i]["v"] = rk4(sats[i]["r"], sats[i]["v"], DT)
        debris[i]["r"], debris[i]["v"] = rk4(debris[i]["r"], debris[i]["v"], DT)

    oos["r"], oos["v"] = rk4(oos["r"], oos["v"], DT)

    target = sats[oos["target"]]
    dist = np.linalg.norm(target["r"] - oos["r"])

    if oos["state"] == "IDLE":
        v_new, dv = solve_lambert(oos["r"], oos["v"], target["r"], target["v"])
        if v_new is not None and dv < 1.0:
            oos["v"] = v_new
            oos["state"] = "TRANSFER"

    elif oos["state"] == "TRANSFER":
        if dist < 50:
            print(f"Docked with SAT {oos['target']}")
            oos["state"] = "DOCKED"

    elif oos["state"] == "DOCKED":
        sats[oos["target"]]["v"] += np.array([0, 0.01, 0])
        oos["target"] = (oos["target"] + 1) % len(sats)
        oos["state"] = "IDLE"

    for i in range(len(sats)):
        traj_sats[i].append(sats[i]["r"].copy())
        traj_debris[i].append(debris[i]["r"].copy())

    traj_oos.append(oos["r"].copy())

traj_sats = [np.array(t) for t in traj_sats]
traj_debris = [np.array(t) for t in traj_debris]
traj_oos = np.array(traj_oos)

# -----------------------------
# COLORS
# -----------------------------
sat_colors = ["green", "orange", "purple"]

# -----------------------------
# PLOTLY
# -----------------------------
frames = []

for i in range(STEPS):

    data = []

    # SATS
    for idx, t in enumerate(traj_sats):
        data.append(go.Scatter3d(
            x=t[:i,0], y=t[:i,1], z=t[:i,2],
            mode='lines',
            line=dict(color=sat_colors[idx]),
            name=f"SAT {idx}",
            showlegend=(i == 1)
        ))
        data.append(go.Scatter3d(
            x=[t[i,0]], y=[t[i,1]], z=[t[i,2]],
            mode='markers',
            marker=dict(color=sat_colors[idx], size=5),
            showlegend=False
        ))

    # DEBRIS
    for idx, t in enumerate(traj_debris):
        data.append(go.Scatter3d(
            x=t[:i,0], y=t[:i,1], z=t[:i,2],
            mode='lines',
            line=dict(color="red", dash="dot"),
            name=f"Debris {idx}",
            showlegend=(i == 1)
        ))
        data.append(go.Scatter3d(
            x=[t[i,0]], y=[t[i,1]], z=[t[i,2]],
            mode='markers',
            marker=dict(color="red", size=4),
            showlegend=False
        ))

    # OOS
    data.append(go.Scatter3d(
        x=traj_oos[:i,0], y=traj_oos[:i,1], z=traj_oos[:i,2],
        mode='lines',
        line=dict(color="blue", width=6),
        name="OOS",
        showlegend=(i == 1)
    ))
    data.append(go.Scatter3d(
        x=[traj_oos[i,0]], y=[traj_oos[i,1]], z=[traj_oos[i,2]],
        mode='markers',
        marker=dict(color="blue", size=6),
        showlegend=False
    ))

    frames.append(go.Frame(data=data))

fig = go.Figure(data=frames[1].data, frames=frames)

fig.update_layout(
    title="OOS Collision Avoidance Demo",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play", method="animate", args=[None])]
    )]
)

fig.write_html("oos_demo.html", auto_open=True)