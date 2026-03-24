import numpy as np

def init_oos(initial_pos):
    return {
        "r": np.array(initial_pos),
        "v": np.array([0, 7.5, 0]),

        "state": "IDLE",
        "target": None,

        "arrival_time": 0,
        "dock_end_time": 0,

        "planned_dv": None
    }