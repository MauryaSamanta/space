# COMPLETE EXPERIMENT SCRIPT - Copy & Paste Ready
from ExperimentConfig import ExperimentConfig
from ExperimentManager import ExperimentManager
import numpy as np
np.random.seed(42)

experiments = [
    ExperimentConfig(name="f6 OOS - 24 Sats - R=100km", n_oos=6, n_satellites=24, fuel_per_oos=5.0, steps=10000,sensing_radius=100e3),
    # ExperimentConfig(name="2 OOS - 4 Sats", n_oos=2, n_satellites=4, fuel_per_oos=5.0, steps=400),
    # ExperimentConfig(name="2 OOS - 8 Sats", n_oos=2, n_satellites=8, fuel_per_oos=5.0, steps=400),
    # ExperimentConfig(name="2 OOS - 12 Sats", n_oos=2, n_satellites=12, fuel_per_oos=5.0, steps=400),
    # ExperimentConfig(name="3 OOS - 12 Sats", n_oos=3, n_satellites=12, fuel_per_oos=5.0, steps=400),
]

print("🔬 MULTI OOS SCALING EXPERIMENT")
print("="*60)
manager = ExperimentManager("multi_oos_scaling_V3.csv")
results = []

for exp in experiments:
    result = manager.run_experiment(exp)
    results.append(result)
    print(f"✅ {exp.name}: {result['success_rate']:.1%} ({result['missions_completed']}/{result['missions_total']})")