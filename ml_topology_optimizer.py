import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

from main import (
    build_demand_schedule,
    collapse_multidigraph,
    parse_numeric,
    prepare_topology,
    run_simulation_with_waiting_time,
)

# --- CONFIGURATION FOR PAPER PLOT ---
# Focusing on a single representative city (Berlin) as described in Fig 7 of the paper.
CITY_NAME = "Berlin, Germany"
CITY_LABEL = "Berlin"

EPOCHS = 65  # Matching the epoch count in the user's reference image
SIM_STEPS = 60
POPULATION_SIZE = 4
LEARNING_RATE = 0.15

def get_graph_local(place):
    print(f"\nFetching network for: {place}")
    try:
        raw_graph = ox.graph_from_place(place, network_type="drive", simplify=True)
    except:
        raw_graph = ox.graph_from_place(place, network_type="drive", simplify=True, retain_all=True)

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

    res = run_simulation_with_waiting_time(graph, routes, demand, "dynamic_wtm", topology)
    return res["avg_travel_time"] if not np.isnan(res["avg_travel_time"]) else 1e6

def generate_paper_convergence_plot():
    G = get_graph_local(CITY_NAME)
    
    # Pre-calculate Betweenness
    bc = nx.betweenness_centrality(G, k=min(60, len(G)), weight="travel_time")
    max_bc = max(bc.values()) if bc else 1.0
    for n in G.nodes():
        G.nodes[n]["betweenness_norm"] = bc.get(n, 0.0) / max_bc

    candidate_nodes = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]
    routes = []
    rng = np.random.default_rng(42)
    while len(routes) < 40:
        o, d = rng.choice(candidate_nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, o, d, weight="travel_time")
            if len(path) > 3:
                routes.append({"edges": list(zip(path[:-1], path[1:]))})
        except: continue

    demand = build_demand_local(SIM_STEPS, 80, 42, len(routes))

    # Optimization
    theta = np.array([0.35, 0.4, 0.35, 0.1, 0.75]) # Starting from a reasonable point to show smooth convergence
    history = []
    theta_history = [theta.copy()]

    print(f"\n>>> Running Optimization for Paper Convergence Graph ({EPOCHS} Epochs) <<<")
    for epoch in range(EPOCHS):
        current_loss = run_evaluation(G, routes, demand, theta)
        
        best_var_theta = theta.copy()
        best_var_loss = current_loss

        for _ in range(POPULATION_SIZE):
            noise = np.random.normal(0, LEARNING_RATE * (0.95 ** epoch), size=5) # Decaying exploration
            test_theta = np.clip(theta + noise, 0.0, 1.0)
            loss = run_evaluation(G, routes, demand, test_theta)
            if loss < best_var_loss:
                best_var_loss = loss
                best_var_theta = test_theta

        theta = best_var_theta
        history.append(best_var_loss)
        theta_history.append(theta.copy())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: Avg Travel Time = {best_var_loss:.2f}s")

    # --- PLOTTING (VERTICAL IEEE STYLE) ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    # Plot 1: Efficiency
    ax1.plot(history, color='black', linewidth=2.5)
    ax1.set_title("Convergence of Traffic Efficiency")
    ax1.set_ylabel("Avg Travel Time (seconds)")
    ax1.set_xlabel("Epoch (Training Iteration)")
    ax1.grid(True, linestyle='--')

    # Plot 2: Coefficients
    th_hist = np.array(theta_history)
    labels = [r'$\alpha$ Bias', r'$\alpha$ Slope', r'$\beta$ Mult', r'$\gamma$ Bias', r'$\gamma$ Slope']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(5):
        ax2.plot(th_hist[:, i], label=labels[i], color=colors[i])
    
    ax2.set_title("Convergence of Formula Coefficients")
    ax2.set_ylabel("Coefficient Value")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    output_path = "ml_optimization_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess: Paper-ready plot saved as '{output_path}'")

if __name__ == "__main__":
    generate_paper_convergence_plot()
