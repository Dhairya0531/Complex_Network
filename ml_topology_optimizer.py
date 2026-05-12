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
    ("Sydney, Australia", "Sydney"),
]

EPOCHS = 15
SIM_STEPS = 60
POPULATION_SIZE = 3
LEARNING_RATE = 0.2


def get_graph_local(place):
    import networkx as nx
    import osmnx as ox

    print(f"\nFetching network for: {place}")
    try:
        raw_graph = ox.graph_from_place(place, network_type="drive", simplify=True)
    except:
        raw_graph = ox.graph_from_place(
            place, network_type="drive", simplify=True, retain_all=True
        )

    largest_component = max(nx.strongly_connected_components(raw_graph), key=len)
    raw_graph = raw_graph.subgraph(largest_component).copy()
    G = collapse_multidigraph(raw_graph)
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

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
    return res["avg_travel_time"] if not np.isnan(res["avg_travel_time"]) else 1e6


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
    while len(routes) < 50:
        o, d = rng.choice(candidate_nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, o, d, weight="travel_time")
            if len(path) > 3:
                routes.append({"edges": list(zip(path[:-1], path[1:]))})
        except:
            continue

    demand = build_demand_local(SIM_STEPS, 80, 42, len(routes))

    theta = np.array([0.5, 0.5, 0.5, 0.2, 0.8])
    history = []
    theta_history = [theta.copy()]

    for epoch in range(EPOCHS):
        current_loss = run_evaluation(G, routes, demand, theta)
        
        best_var_theta = theta.copy()
        best_var_loss = current_loss

        for _ in range(POPULATION_SIZE):
            noise = np.random.normal(0, LEARNING_RATE, size=5)
            test_theta = np.clip(theta + noise, 0.0, 1.0)
            loss = run_evaluation(G, routes, demand, test_theta)
            if loss < best_var_loss:
                best_var_loss = loss
                best_var_theta = test_theta

        theta = best_var_theta
        history.append(best_var_loss)
        theta_history.append(theta.copy())
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss = {best_var_loss:.2f}s")

    return theta, history, theta_history


def run_multi_city_validation():
    all_results = {}

    for city_name, label in CITIES:
        theta, history, theta_history = optimize_city(city_name, label)
        all_results[label] = {"theta": theta, "history": history, "theta_history": theta_history}

    # Plot Comparison of Convergence (Vertical Stack)
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'grid.alpha': 0.3
    })

    fig, axes = plt.subplots(len(all_results) * 2, 1, figsize=(10, 5 * len(all_results) * 2), constrained_layout=True)

    for i, (label, res) in enumerate(all_results.items()):
        # Top: Efficiency Plot
        ax_eff = axes[i * 2]
        ax_eff.plot(res["history"], color='black', linewidth=2.5, marker='o', markersize=4)
        ax_eff.set_title(f"Convergence of Traffic Efficiency: {label}")
        ax_eff.set_ylabel("Avg Travel Time (seconds)")
        ax_eff.set_xlabel("Epoch")
        ax_eff.grid(True, linestyle='--')
        
        # Bottom: Coefficients Plot
        ax_coeff = axes[i * 2 + 1]
        th_hist = np.array(res["theta_history"])
        th_labels = [r'$\alpha$ Bias', r'$\alpha$ Slope', r'$\beta$ Mult', r'$\gamma$ Bias', r'$\gamma$ Slope']
        for j in range(5):
            ax_coeff.plot(th_hist[:, j], label=th_labels[j], linewidth=2.5)
        
        ax_coeff.set_title(f"Convergence of Formula Coefficients: {label}")
        ax_coeff.set_ylabel("Coefficient Value")
        ax_coeff.set_xlabel("Epoch")
        ax_coeff.legend(loc='upper right', fontsize=10)
        ax_coeff.grid(True, linestyle='--')

    plt.savefig("multi_city_convergence_comparison.png", dpi=300)
    print("\nVertical comparison plot saved as 'multi_city_convergence_comparison.png'")


if __name__ == "__main__":
    run_multi_city_validation()
