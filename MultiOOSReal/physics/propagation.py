import numpy as np

MU = 398600  # km^3/s^2 (Earth)

def acceleration(r):
    r = np.array(r)
    norm = np.linalg.norm(r)
    return -MU * r / (norm**3)


def rk4_step(r, v, dt):
    r = np.array(r)
    v = np.array(v)

    def f_r(v): return v
    def f_v(r): return acceleration(r)

    k1_r = f_r(v)
    k1_v = f_v(r)

    k2_r = f_r(v + 0.5 * dt * k1_v)
    k2_v = f_v(r + 0.5 * dt * k1_r)

    k3_r = f_r(v + 0.5 * dt * k2_v)
    k3_v = f_v(r + 0.5 * dt * k2_r)

    k4_r = f_r(v + dt * k3_v)
    k4_v = f_v(r + dt * k3_r)

    r_new = r + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return r_new, v_new