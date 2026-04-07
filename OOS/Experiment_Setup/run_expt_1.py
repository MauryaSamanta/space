# COMPLETE EXPERIMENT SCRIPT - Copy & Paste Ready
from ExperimentConfig import ExperimentConfig
from ExperimentManager import ExperimentManager
import numpy as np
np.random.seed(42)

# 🧪 SCALING EXPERIMENT: Single OOS performance vs # satellites
experiments = [
    ExperimentConfig(name="1 OOS - 1 Sat", n_oos=1, n_satellites=1, n_orbits=1, fuel_per_oos=5.0, steps=150),
    ExperimentConfig(name="1 OOS - 2 Sats", n_oos=1, n_satellites=2, n_orbits=1, fuel_per_oos=5.0, steps=150),
    ExperimentConfig(name="1 OOS - 3 Sats", n_oos=1, n_satellites=3, n_orbits=1, fuel_per_oos=5.0, steps=150),
    ExperimentConfig(name="1 OOS - 4 Sats", n_oos=1, n_satellites=4, n_orbits=1, fuel_per_oos=5.0, steps=200),
]

print("🔬 SINGLE OOS SCALING EXPERIMENT")
print("="*60)
manager = ExperimentManager("single_oos_scaling_V4.csv")
results = []

for exp in experiments:
    result = manager.run_experiment(exp)
    results.append(result)
    print(f"✅ {exp.name}: {result['success_rate']:.1%} ({result['missions_completed']}/{result['missions_total']})")