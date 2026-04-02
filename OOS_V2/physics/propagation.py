import numpy as np
from config import MU

def acceleration(r):
    norm = np.linalg.norm(r)
    return -MU * r / norm**3

def rk4_step(r, v, dt):

    def f(r, v):
        return v, acceleration(r)

    k1_v, k1_a = f(r, v)
    k2_v, k2_a = f(r + 0.5*dt*k1_v, v + 0.5*dt*k1_a)
    k3_v, k3_a = f(r + 0.5*dt*k2_v, v + 0.5*dt*k2_a)
    k4_v, k4_a = f(r + dt*k3_v, v + dt*k3_a)

    r_new = r + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    v_new = v + (dt/6)*(k1_a + 2*k2_a + 2*k3_a + k4_a)

    return r_new, v_new