import numpy as np

def cw_docking_velocity(rel_r, rel_v):
    # simple stabilizing control (not full CW solution)
    k = 0.001
    return -k * rel_r - 0.5 * rel_v