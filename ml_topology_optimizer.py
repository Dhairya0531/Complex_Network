import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import (
    build_demand_schedule,
    collapse_multidigraph,
    parse_numeric,
    prepare_topology,
    run_simulation_with_waiting_time,
)

# --- CONFIGURATION ---
CITIES = [
    ("Bengaluru, India", "Bengaluru"),
    ("Berlin, Germany", "Berlin"),
    ("London, UK", "London"),
    ("Sydney, Australia", "Sydney")
]

EPOCHS = 10
SIM_STEPS = 30
POPULATION_SIZE = 2
LEARNING_RATE = 0.1


def get_graph_local(place):
    import networkx as nx
    import osmnx as ox

    try:
        raw_graph = ox.graph_from_place(place, network_type="drive", simplify=True)
    except:
        raw_graph = ox.graph_from_place(
            place, network_type="drive", simplify=True, retain_all=True
        )

    largest_component = max(nx.strongly_connected_components(raw_graph), key=len)
    raw_graph = raw_graph.subgraph(largest_component).copy()
    G = collapse_multidigraph(raw_graph)

    for u, v, data in G.edges(data=True):
        data["length"] = float(data.get("length", 1.0))
        data["speed_kph"] = parse_numeric(data.get("maxspeed"), 35.0)
        data["lanes"] = max(1, int(round(parse_numeric(data.get("lanes"), 1.0))))
        data["travel_time"] = data["length"] / max(data["speed_kph"] * 1000 / 3600, 1.0)
        data["capacity_per_cycle"] = max(1, int(data["lanes"] * 8))
    return G


def build_demand_local(steps, arrival_rate, seed, num_routes):
    local_rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(steps):
        arrivals = int(local_rng.poisson(arrival_rate))
        if arrivals == 0:
            schedule.append([])
            continue
        chosen_routes = local_rng.integers(0, num_routes, size=arrivals)
        schedule.append(chosen_routes.tolist())
    return schedule


def run_evaluation(graph, routes, demand, theta):
    nodes = list(graph.nodes())
    in_counts = np.array([graph.in_degree(n) for n in nodes], dtype=float)
    max_in = np.max(in_counts) if np.max(in_counts) > 0 else 1.0
    norm_degree = in_counts / max_in
    importance = np.array([graph.nodes[n].get("betweenness_norm", 0.0) for n in nodes])

    alpha_dynamic = theta[0] + theta[1] * norm_degree
    beta_dynamic = theta[2] * (1.0 - importance)
    gamma_dynamic = theta[3] + theta[4] * importance

    topology = prepare_topology(graph)
    topology["alpha_dynamic"] = alpha_dynamic
    topology["beta_dynamic"] = beta_dynamic
    topology["gamma_dynamic"] = gamma_dynamic

    res = run_simulation_with_waiting_time(
        graph, routes, demand, "dynamic_wtm", topology
    )
    return res["avg_travel_time"]


def optimize_city(city_name, city_label):
    print(f"\n>>> Starting Multi-City Optimization: {city_label} <<<")
    import networkx as nx

    G = get_graph_local(city_name)

    bc = nx.betweenness_centrality(G, k=min(40, len(G)), weight="travel_time")
    max_bc = max(bc.values()) if bc else 1.0
    for n in G.nodes():
        G.nodes[n]["betweenness_norm"] = bc.get(n, 0.0) / max_bc

    candidate_nodes = [
        n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0
    ]
    routes = []
    rng = np.random.default_rng(42)
    while len(routes) < 30:
        o, d = rng.choice(candidate_nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, o, d, weight="travel_time")
            if len(path) > 3:
                routes.append({"edges": list(zip(path[:-1], path[1:]))})
        except:
            continue

    demand = build_demand_local(SIM_STEPS, 18, 42, len(routes))

    theta = np.array([0.5, 0.5, 0.5, 0.2, 0.8])
    history = []

    for epoch in range(EPOCHS):
        best_loss = run_evaluation(G, routes, demand, theta)

        variations = []
        for _ in range(POPULATION_SIZE):
            noise = np.random.normal(0, LEARNING_RATE, size=5)
            test_theta = np.clip(theta + noise, 0.0, 1.0)
            loss = run_evaluation(G, routes, demand, test_theta)
            variations.append((test_theta, loss))

        variations.sort(key=lambda x: x[1])
        if variations[0][1] < best_loss:
            theta = variations[0][0]
            best_loss = variations[0][1]
        history.append(best_loss)

    print(
        f"Learned for {city_label}: Alpha={theta[0]:.2f}+{theta[1]:.2f}*deg | Beta={theta[2]:.2f}*(1-BC) | Gamma={theta[3]:.2f}+{theta[4]:.2f}*BC"
    )
    return theta, history


def run_multi_city_validation():
    all_results = {}

    for city_name, label in CITIES:
        theta, history = optimize_city(city_name, label)
        all_results[label] = {"theta": theta, "history": history}

    # Generate Comparative Summary Table
    df_data = []
    for label, res in all_results.items():
        th = res["theta"]
        df_data.append(
            {
                "City": label,
                "Alpha Formula": f"{th[0]:.2f} + {th[1]:.2f}*deg",
                "Beta Formula": f"{th[2]:.2f}*(1-BC)",
                "Gamma Formula": f"{th[3]:.2f} + {th[4]:.2f}*BC",
                "Final Travel Time": f"{res['history'][-1]:.1f}s",
            }
        )

    summary_df = pd.DataFrame(df_data)
    summary_df.to_csv("multi_city_optimization_results.csv", index=False)
    print("\n" + "=" * 80)
    print("MULTI-CITY CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Plot Comparison of Convergence
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    plt.figure(figsize=(10, 6))
    for label, res in all_results.items():
        # Normalize history to show % improvement
        start = res["history"][0]
        norm_hist = [100 * (h / start) for h in res["history"]]
        plt.plot(norm_hist, label=f"{label} (Optimized)", linewidth=3)

    plt.title("Formula Optimization Efficiency Across Multiple City Topologies")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Travel Time (%)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig("multi_city_convergence_comparison.png", dpi=300)
    print("\nComparison plot saved as 'multi_city_convergence_comparison.png'")


if __name__ == "__main__":
    run_multi_city_validation()
