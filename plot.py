import plotly.graph_objects as go
from simulate import get_satellites, get_trajectory
import numpy as np


satellites = get_satellites(3)

fig = go.Figure()

for sat in satellites:
    traj = get_trajectory(sat)

    x = [p[0] for p in traj]
    y = [p[1] for p in traj]
    z = [p[2] for p in traj]

    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',
        name=sat["name"]
    ))

# Create Earth sphere
radius = 6371  # km

phi = np.linspace(0, np.pi, 50)
theta = np.linspace(0, 2*np.pi, 50)

x = radius * np.outer(np.sin(phi), np.cos(theta))
y = radius * np.outer(np.sin(phi), np.sin(theta))
z = radius * np.outer(np.cos(phi), np.ones_like(theta))

fig.add_trace(go.Surface(
    x=x, y=y, z=z,
    colorscale='Blues',
    opacity=0.5,
    showscale=False
))

fig.show()