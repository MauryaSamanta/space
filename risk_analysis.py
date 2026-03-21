from fetch_tle import get_satellites
from simulate import get_trajectory
from collision import closest_approach, collision_probability

satellites = get_satellites(5)

trajectories = {}

# Generate trajectories
for sat in satellites:
    trajectories[sat["name"]] = get_trajectory(sat)

# Compare all pairs
n = len(satellites)

for i in range(n):
    for j in range(i+1, n):

        sat1 = satellites[i]["name"]
        sat2 = satellites[j]["name"]

        traj1 = trajectories[sat1]
        traj2 = trajectories[sat2]

        dist, step = closest_approach(traj1, traj2)
        Pc = collision_probability(dist)

        print(f"\n{sat1} vs {sat2}")
        print(f"Min Distance: {dist:.2f} km")
        print(f"Collision Probability: {Pc:.6e}")

        if Pc > 1e-4:
            print("⚠️ HIGH RISK")
        elif Pc > 1e-6:
            print("⚠️ MEDIUM RISK")
        else:
            print("SAFE")