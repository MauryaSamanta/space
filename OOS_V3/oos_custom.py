import numpy as np
import plotly.graph_objects as go

from poliastro.iod import lambert
from poliastro.bodies import Earth
from astropy import units as u

# -----------------------------
# CONSTANTS
# -----------------------------
MU = 398600.4418
DT = 20
STEPS = 400
TOF = 3000

# -----------------------------
# RK4
# -----------------------------
def acceleration(r):
    return -MU * r / np.linalg.norm(r)**3

def rk4(r, v, dt):
    k1_r = v
    k1_v = acceleration(r)

    k2_r = v + 0.5*dt*k1_v
    k2_v = acceleration(r + 0.5*dt*k1_r)

    k3_r = v + 0.5*dt*k2_v
    k3_v = acceleration(r + 0.5*dt*k2_r)

    k4_r = v + dt*k3_v
    k4_v = acceleration(r + dt*k3_r)

    r_new = r + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    return r_new, v_new

# -----------------------------
# ORBIT
# -----------------------------
def circular_orbit(r, theta):
    pos = np.array([r*np.cos(theta), r*np.sin(theta), 0])
    v_mag = np.sqrt(MU/r)
    vel = np.array([-v_mag*np.sin(theta), v_mag*np.cos(theta), 0])
    return pos, vel

# -----------------------------
# LAMBERT
# -----------------------------
def propagate_future(r, v):
    r_f, v_f = r.copy(), v.copy()
    for _ in range(int(TOF/DT)):
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
        return v1_new
    except:
        return None

# -----------------------------
# INIT (CONTROLLED SCENARIO)
# -----------------------------
R = 7000

# satellite
r_sat, v_sat = circular_orbit(R, 0)

# debris (slightly behind but same orbit → WILL collide)
r_deb, v_deb = circular_orbit(R, 0.02)

# OOS far away
r_oos, v_oos = circular_orbit(R, 1.5)

state = "IDLE"

# -----------------------------
# STORAGE
# -----------------------------
traj_sat = []
traj_deb = []
traj_oos = []

# -----------------------------
# SIMULATION
# -----------------------------
for step in range(STEPS):

    # propagate all
    r_sat, v_sat = rk4(r_sat, v_sat, DT)
    r_deb, v_deb = rk4(r_deb, v_deb, DT)
    r_oos, v_oos = rk4(r_oos, v_oos, DT)

    dist = np.linalg.norm(r_sat - r_oos)

    # OOS LOGIC
    if state == "IDLE":
        v_new = solve_lambert(r_oos, v_oos, r_sat, v_sat)
        if v_new is not None:
            v_oos = v_new
            state = "TRANSFER"

    elif state == "TRANSFER":
        if dist < 50:
            print("✅ Docked")
            state = "DOCKED"

    elif state == "DOCKED":
        # push satellite to avoid collision
        v_sat += np.array([0, 0.02, 0])
        print("🚀 Collision avoided")
        state = "DONE"

    # store
    traj_sat.append(r_sat.copy())
    traj_deb.append(r_deb.copy())
    traj_oos.append(r_oos.copy())

traj_sat = np.array(traj_sat)
traj_deb = np.array(traj_deb)
traj_oos = np.array(traj_oos)

# -----------------------------
# PLOT
# -----------------------------
frames = []

for i in range(STEPS):

    frames.append(go.Frame(data=[

        # SAT
        go.Scatter3d(
            x=traj_sat[:i,0], y=traj_sat[:i,1], z=traj_sat[:i,2],
            mode='lines',
            line=dict(color='green'),
            name="Satellite",
            showlegend=(i==1)
        ),
        go.Scatter3d(
            x=[traj_sat[i,0]], y=[traj_sat[i,1]], z=[traj_sat[i,2]],
            mode='markers',
            marker=dict(color='green', size=5),
            showlegend=False
        ),

        # DEBRIS
        go.Scatter3d(
            x=traj_deb[:i,0], y=traj_deb[:i,1], z=traj_deb[:i,2],
            mode='lines',
            line=dict(color='red', dash='dot'),
            name="Debris (collision path)",
            showlegend=(i==1)
        ),
        go.Scatter3d(
            x=[traj_deb[i,0]], y=[traj_deb[i,1]], z=[traj_deb[i,2]],
            mode='markers',
            marker=dict(color='red', size=4),
            showlegend=False
        ),

        # OOS
        go.Scatter3d(
            x=traj_oos[:i,0], y=traj_oos[:i,1], z=traj_oos[:i,2],
            mode='lines',
            line=dict(color='blue', width=6),
            name="OOS (Servicer)",
            showlegend=(i==1)
        ),
        go.Scatter3d(
            x=[traj_oos[i,0]], y=[traj_oos[i,1]], z=[traj_oos[i,2]],
            mode='markers',
            marker=dict(color='blue', size=6),
            showlegend=False
        ),

    ]))

fig = go.Figure(data=frames[1].data, frames=frames)

fig.update_layout(
    title="Single Satellite Collision Avoidance",
    scene=dict(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)"
    ),
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play", method="animate", args=[None])]
    )]
)

fig.write_html("single_oos_demo.html", auto_open=True)