import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from main import (
    build_demand_schedule,
    collapse_multidigraph,
    parse_numeric,
    prepare_topology,
    run_simulation_with_waiting_time,
)

# --- ROBUST ML CONFIG ---
CITIES = [
    ("Cubbon Park, Bengaluru, India", "Bengaluru"),
    ("Brandenburg Gate, Berlin, Germany", "Berlin"),
    ("Trafalgar Square, London, UK", "London"),
    ("Sydney Opera House, Sydney, Australia", "Sydney")
]
EPOCHS = 8
SIM_STEPS = 500 # Sufficient time for short routes
POPULATION_SIZE = 2
LEARNING_RATE = 0.1

def get_graph_local(place):
    import osmnx as ox
    raw_graph = ox.graph_from_address(place, dist=1200, network_type="drive", simplify=True)
    largest_component = max(nx.strongly_connected_components(raw_graph), key=len)
    G = collapse_multidigraph(raw_graph.subgraph(largest_component).copy())
    for u, v, data in G.edges(data=True):
        data["length"] = float(data.get("length", 1.0))
        data["speed_kph"] = parse_numeric(data.get("maxspeed"), 40.0)
        data["lanes"] = max(1, int(round(parse_numeric(data.get("lanes"), 1.0))))
        data["travel_time"] = data["length"] / max(data["speed_kph"] * 1000 / 3600, 1.0)
        data["capacity_per_cycle"] = max(1, int(data["lanes"] * 8))
    return G

def build_demand_local(steps, arrival_rate, seed, num_routes):
    local_rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(steps):
        arrivals = int(local_rng.poisson(arrival_rate))
        if arrivals == 0: schedule.append([])
        else: schedule.append(local_rng.integers(0, num_routes, size=arrivals).tolist())
    return schedule

def run_evaluation(graph, routes, demand, theta):
    nodes = list(graph.nodes())
    norm_deg = np.array([graph.in_degree(n) for n in nodes], dtype=float)
    norm_deg /= np.max(norm_deg) if np.max(norm_deg) > 0 else 1.0
    importance = np.array([graph.nodes[n].get("betweenness_norm", 0.0) for n in nodes])
    topology = prepare_topology(graph)
    topology["alpha_dynamic"] = theta[0] + theta[1] * norm_deg
    topology["beta_dynamic"] = theta[2] * (1.0 - importance)
    topology["gamma_dynamic"] = theta[3] + theta[4] * importance
    res = run_simulation_with_waiting_time(graph, routes, demand, "dynamic_wtm", topology)
    return res["avg_travel_time"] if not np.isnan(res["avg_travel_time"]) else 1e5

def optimize_city(city_name, city_label):
    print(f"\n>>> Optimizing {city_label} <<<")
    G = get_graph_local(city_name)
    bc = nx.betweenness_centrality(G, k=min(20, len(G)), weight="travel_time")
    max_bc = max(bc.values()) if bc else 1.0
    for n in G.nodes(): G.nodes[n]["betweenness_norm"] = bc.get(n, 0.0) / max_bc
    candidate_nodes = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]
    routes = []
    rng = np.random.default_rng(42)
    while len(routes) < 15:
        o, d = rng.choice(candidate_nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, o, d, weight="travel_time")
            if 3 <= len(path) <= 6: routes.append({"edges": list(zip(path[:-1], path[1:]))})
        except: continue
    demand = build_demand_local(SIM_STEPS, 6, 42, len(routes))
    theta = np.array([0.5, 0.5, 0.5, 0.2, 0.8]); history = []
    for epoch in range(EPOCHS):
        best_loss = run_evaluation(G, routes, demand, theta)
        for _ in range(POPULATION_SIZE):
            test_theta = np.clip(theta + np.random.normal(0, LEARNING_RATE, size=5), 0.0, 1.0)
            loss = run_evaluation(G, routes, demand, test_theta)
            if loss < best_loss: theta = test_theta; best_loss = loss
        history.append(best_loss)
    return theta, history

def run_multi_city_validation():
    all_results = {}
    for city_name, label in CITIES:
        theta, history = optimize_city(city_name, label)
        all_results[label] = {"theta": theta, "history": history}
    plt.rcParams.update({'font.size': 24, 'axes.linewidth': 2.5, 'axes.titlesize': 32, 'axes.labelsize': 28})
    plt.figure(figsize=(16, 10))
    for label, res in all_results.items():
        plt.plot([100*(h/res['history'][0]) for h in res['history']], label=label, linewidth=5.0)
    plt.title("Formula Optimization Convergence", fontweight='bold')
    plt.xlabel("Epoch", fontweight='bold'); plt.ylabel("Travel Time (%)", fontweight='bold')
    plt.legend(title="Cities", title_fontsize=24); plt.grid(True, linestyle=":"); plt.tight_layout()
    plt.savefig("ml_optimization_convergence.png", dpi=300)
    print("Success: ml_optimization_convergence.png")

if __name__ == "__main__":
    run_multi_city_validation()
