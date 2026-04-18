import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
from main import (
    prepare_topology, 
    build_demand_schedule, 
    run_simulation_with_waiting_time,
    collapse_multidigraph,
    parse_numeric
)

# --- ROBUST CONFIGURATION ---
CITIES = [
    ("Cubbon Park, Bengaluru, India", "Bengaluru"),
    ("Brandenburg Gate, Berlin, Germany", "Berlin"),
    ("Trafalgar Square, London, UK", "London"),
    ("Sydney Opera House, Sydney, Australia", "Sydney")
]
SIM_STEPS = 600
ARRIVAL_RATE = 8
NUM_TRIALS = 1

def get_graph_local(place, label):
    import osmnx as ox
    print(f"Fetching {label} patch...")
    raw_graph = ox.graph_from_address(place, dist=1200, network_type="drive", simplify=True)
    largest_component = max(nx.strongly_connected_components(raw_graph), key=len)
    raw_graph = raw_graph.subgraph(largest_component).copy()
    G = collapse_multidigraph(raw_graph)
    for u, v, data in G.edges(data=True):
        data["length"] = float(data.get("length", 1.0))
        data["speed_kph"] = parse_numeric(data.get("maxspeed"), 40.0)
        data["lanes"] = max(1, int(round(parse_numeric(data.get("lanes"), 1.0))))
        data["travel_time"] = data["length"] / max(data["speed_kph"] * 1000 / 3600, 1.0)
        data["capacity_per_cycle"] = max(1, int(data["lanes"] * 8))
    return G

def normalize_dict(d):
    if not d: return d
    max_val = max(d.values()) if max(d.values()) > 0 else 1.0
    return {k: v / max_val for k, v in d.items()}

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

def run_multi_city_comparison():
    city_results = []
    for city_name, label in CITIES:
        print(f"\n>>> Analyzing {label} <<<")
        G = get_graph_local(city_name, label)
        measures = {
            "Betweenness": normalize_dict(nx.betweenness_centrality(G, k=min(25, len(G)), weight="travel_time")),
            "Closeness": normalize_dict(nx.closeness_centrality(G)),
            "Degree": normalize_dict(dict(G.degree())),
            "PageRank": normalize_dict(nx.pagerank(G, weight="length"))
        }
        candidate_nodes = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]
        routes = []
        rng = np.random.default_rng(42)
        while len(routes) < 15:
            o, d = rng.choice(candidate_nodes, 2, replace=False)
            try:
                path = nx.shortest_path(G, o, d, weight="travel_time")
                if 3 <= len(path) <= 7:
                    routes.append({"edges": list(zip(path[:-1], path[1:]))})
            except: continue
        demand = build_demand_local(SIM_STEPS, ARRIVAL_RATE, 42, len(routes))
        for name, centrality_map in measures.items():
            print(f"  Testing {name}...")
            nx.set_node_attributes(G, centrality_map, "betweenness_norm") 
            topology = prepare_topology(G)
            res = run_simulation_with_waiting_time(G, routes, demand, "dynamic_wtm", topology)
            city_results.append({"City": label, "Measure": name, "Avg Travel Time": res['avg_travel_time'] if not np.isnan(res['avg_travel_time']) else 0})

    df = pd.DataFrame(city_results)
    pivot_df = df.pivot(index='City', columns='Measure', values='Avg Travel Time')
    plt.rcParams.update({'font.size': 24, 'axes.linewidth': 2.5, 'axes.titlesize': 32, 'axes.labelsize': 28})
    ax = pivot_df.plot(kind='bar', figsize=(18, 10), edgecolor='black', linewidth=2.0, color=['#2e86de', '#95a5a6', '#f1c40f', '#e74c3c'])
    bars, hatches = ax.patches, ['///', '\\\\', 'xx', '..']
    for i, bar in enumerate(bars): bar.set_hatch(hatches[i // len(pivot_df)])
    plt.ylabel("Avg Travel Time (s)", fontweight='bold')
    plt.xlabel("City", fontweight='bold')
    plt.title("Impact of Centrality Measure", fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend(title="Centrality Type", title_fontsize=24, frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("multi_city_centrality_comparison.png", dpi=300)
    print("Success: multi_city_centrality_comparison.png")

if __name__ == "__main__":
    run_multi_city_comparison()
