import numpy as np

class MetricsManager:
    def __init__(self):
        self.data = {
            "missions_total": 0,
            "missions_completed": 0,
            "collisions_detected": 0,
            "collisions_avoided": 0,
            "missed_collisions": 0,

            "total_dv": 0,
            "fuel_used_total": 0,

            "response_times": [],
            "dv_per_mission": [],

            "missions_per_oos": {},
        }

    def log_dv(self, dv):
        dv_mag = np.linalg.norm(dv)
        self.data["total_dv"] += dv_mag
        self.data["fuel_used_total"] += dv_mag
        self.data["dv_per_mission"].append(dv_mag)

    def log_mission_start(self):
        self.data["missions_total"] += 1

    def log_failure(self):
        if "failures" not in self.data:
            self.data["failures"] = 0
        self.data["failures"] += 1

    def log_mission_complete(self, response_time):
        self.data["missions_completed"] += 1
        self.data["response_times"].append(response_time)

    def log_collision_avoided(self):
        self.data["collisions_avoided"] += 1

    def log_wait_time(self, t):
        if "wait_times" not in self.data:
            self.data["wait_times"] = []
        self.data["wait_times"].append(t)

    def log_collision_detected(self):
        self.data["collisions_detected"] += 1

    def finalize(self):
        d = self.data

        d["avg_response_time"] = np.mean(d["response_times"]) if d["response_times"] else 0
        d["avg_dv_per_mission"] = np.mean(d["dv_per_mission"]) if d["dv_per_mission"] else 0
        d["avg_wait_time"] = np.mean(d.get("wait_times", [])) if d.get("wait_times") else 0
        d["success_rate"] = (
            d["missions_completed"] / max(1, d["missions_total"])
        )

        return d