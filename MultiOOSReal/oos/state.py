import numpy as np

def init_oos(initial_pos):
    return {
        "r": initial_pos.copy(),
        "v": None,  # will be set later

        "state": "IDLE",
        "target": None,

        # 🔥 NEW
        "phase_target_angle": None,

        "dock_end_time": 0,
        "planned_dv": None,

        "fuel": 5.0
    }