import csv
import os

from ExperimentRunner import ExperimentRunner

class ExperimentManager:

    def __init__(self, save_path="results.csv"):
        self.results = []
        self.save_path = save_path

        # create file with header if not exists
        if not os.path.exists(self.save_path):
            with open(self.save_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "experiment_name",
                    "missions_total",
                    "missions_completed",
                    "collisions_avoided",
                    "missed_collisions",
                    "total_dv",
                    "fuel_used_total",
                    "avg_response_time",
                    "avg_dv_per_mission",
                    "success_rate"
                ])

    def run_experiment(self, config):
        print(f"\n🚀 Running: {config.name}")

        runner = ExperimentRunner(config)
        result = runner.run()

        self.results.append((config.name, result))

        # 🔥 SAVE TO CSV
        self._save_to_csv(config.name, result)

        return result

    def _save_to_csv(self, name, result):
        with open(self.save_path, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                name,
                result.get("missions_total", 0),
                result.get("missions_completed", 0),
                result.get("collisions_avoided", 0),
                result.get("missed_collisions", 0),
                result.get("total_dv", 0),
                result.get("fuel_used_total", 0),
                result.get("avg_response_time", 0),
                result.get("avg_dv_per_mission", 0),
                result.get("success_rate", 0),
            ])