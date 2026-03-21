from fetch_tle import get_satellites
from simulate import get_trajectory
from collision import closest_approach, collision_probability

import math


# ---------- ACTIONS ----------
ACTIONS = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1)
]


# ---------- APPLY MANEUVER ----------
def apply_action(traj, action, scale=150):
    ax, ay, az = action

    return [
        (x + ax*scale, y + ay*scale, z + az*scale)
        for (x, y, z) in traj
    ]


# ---------- BUILD GLOBAL GRAPH ----------
def build_global_graph(trajectories):
    graph = {}
    names = list(trajectories.keys())

    for name in names:
        graph[name] = {}

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1 = names[i]
            n2 = names[j]

            d, _ = closest_approach(
                trajectories[n1],
                trajectories[n2]
            )

            Pc = collision_probability(d)

            graph[n1][n2] = Pc
            graph[n2][n1] = Pc

    return graph


# ---------- NODE RISK ----------
def compute_node_risk(graph):
    node_risk = {}

    for node in graph:
        node_risk[node] = sum(graph[node].values())

    return node_risk


# ---------- TOTAL SYSTEM RISK ----------
def compute_total_risk(graph):
    total = 0

    visited = set()

    for n1 in graph:
        for n2 in graph[n1]:
            if (n2, n1) not in visited:
                total += graph[n1][n2]
                visited.add((n1, n2))

    return total


# ---------- SELECT TOP RISKY ----------
def select_top_risky(node_risk, k=3):
    return sorted(node_risk, key=node_risk.get, reverse=True)[:k]


# ---------- EVALUATE ACTION ----------
def evaluate_action(sat_name, trajectories):
    base_graph = build_global_graph(trajectories)
    base_risk = compute_total_risk(base_graph)

    best = None
    best_score = float("inf")

    for action in ACTIONS:
        new_traj = apply_action(trajectories[sat_name], action)

        temp_traj = trajectories.copy()
        temp_traj[sat_name] = new_traj

        new_graph = build_global_graph(temp_traj)
        new_risk = compute_total_risk(new_graph)

        fuel = math.sqrt(
            action[0]**2 +
            action[1]**2 +
            action[2]**2
        ) * 150

        risk_reduction = base_risk - new_risk

        score = (
            800 * new_risk
            - 2000 * risk_reduction
            + 0.1 * fuel
        )

        if score < best_score:
            best_score = score
            best = {
                "action": action,
                "score": score,
                "risk_reduction": risk_reduction,
                "new_risk": new_risk
            }

    return best


# ---------- MAIN LOOP ----------
def run_simulation():
    satellites = get_satellites(10)

    trajectories = {
        sat["name"]: get_trajectory(sat)
        for sat in satellites
    }

    for step in range(5):
        print(f"\n========== STEP {step} ==========")

        # Build graph
        graph = build_global_graph(trajectories)

        # Compute risks
        node_risk = compute_node_risk(graph)
        total_risk = compute_total_risk(graph)

        print(f"\nTOTAL SYSTEM RISK: {total_risk:.4f}")

        # Select top risky satellites
        candidates = select_top_risky(node_risk, k=3)

        print("\nTop Risky Satellites:")
        for c in candidates:
            print(f"{c}: {node_risk[c]:.4f}")

        best_global = None
        best_sat = None

        # Evaluate only candidates
        for sat_name in candidates:
            result = evaluate_action(sat_name, trajectories)

            print(f"\nEvaluating {sat_name}")
            print(f"Action: {result['action']}")
            print(f"Risk Reduction: {result['risk_reduction']:.4f}")

            if best_global is None or result["score"] < best_global["score"]:
                best_global = result
                best_sat = sat_name

        print(f"\n>>> SELECTED SATELLITE: {best_sat}")
        print(f"ACTION: {best_global['action']}")
        print(f"EXPECTED RISK REDUCTION: {best_global['risk_reduction']:.4f}")

        # Apply best action
        trajectories[best_sat] = apply_action(
            trajectories[best_sat],
            best_global["action"]
        )


# ---------- RUN ----------
if __name__ == "__main__":
    run_simulation()