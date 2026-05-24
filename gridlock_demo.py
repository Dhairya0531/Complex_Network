#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd
from main import (
    parse_numeric,
    collapse_multidigraph,
    build_demand_schedule,
    prepare_topology,
    run_simulation_with_waiting_time
)

# Configuration for gridlock scenario
PLACE = "Berlin, Germany"
SIMULATION_STEPS = 180
ARRIVAL_RATE = 900  # Extremely high rate to saturate the network and trigger gridlock
NUM_OD_PAIRS = 40
RANDOM_SEED = 42

print("=" * 80)
print("GRIDLOCK GENERATION DEMO: Saturating the Network")
print("=" * 80)

# Fetch the network (reusing main's logic or cached data)
print(f"Loading driving network for: {PLACE}")
try:
    raw_graph = ox.graph_from_place(PLACE, network_type="drive", simplify=True)
except Exception as e:
    print(f"OSM fetch failed: {e}. Retrying with retain_all=True")
    raw_graph = ox.graph_from_place(PLACE, network_type="drive", simplify=True, retain_all=True)

largest_component = max(nx.strongly_connected_components(raw_graph), key=len)
raw_graph = raw_graph.subgraph(largest_component).copy()
G = collapse_multidigraph(raw_graph)

for u, v, data in G.edges(data=True):
    length = float(data.get("length", 1.0))
    speed_kph = parse_numeric(data.get("maxspeed"), 35.0)
    lanes = max(1, int(round(parse_numeric(data.get("lanes"), 1.0))))
    travel_time = length / max(speed_kph * 1000 / 3600, 1.0)

    data["length"] = length
    data["speed_kph"] = speed_kph
    data["lanes"] = lanes
    data["travel_time"] = travel_time
    data["capacity_per_cycle"] = max(1, int(lanes * 8))

# Centrality analysis
betweenness = nx.betweenness_centrality(G, k=60, weight="travel_time", seed=RANDOM_SEED)
max_bc = max(betweenness.values()) if betweenness else 1.0
node_importance = {n: v / max_bc for n, v in betweenness.items()}
nx.set_node_attributes(G, node_importance, "betweenness_norm")

candidate_nodes = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]
rng = np.random.default_rng(RANDOM_SEED)
route_bank = []
while len(route_bank) < NUM_OD_PAIRS:
    o, d = rng.choice(candidate_nodes, size=2, replace=False)
    try:
        path = nx.shortest_path(G, o, d, weight="travel_time")
        if len(path) > 3:
            route_bank.append({
                "origin": o,
                "destination": d,
                "path": path,
                "edges": list(zip(path[:-1], path[1:]))
            })
    except nx.NetworkXNoPath:
        continue

print(f"Precomputed routes: {len(route_bank)}")
topology = prepare_topology(G)
demand_schedule = build_demand_schedule(SIMULATION_STEPS, ARRIVAL_RATE, RANDOM_SEED)

results = {}
for controller in ["fixed", "backpressure", "dynamic_wtm"]:
    print(f"\nSimulating {controller.upper()} controller ...", flush=True)
    res = run_simulation_with_waiting_time(G, route_bank, demand_schedule, controller, topology)
    results[controller] = res

# Build comparison DataFrame
comp_data = []
for ctrl in ["fixed", "backpressure", "dynamic_wtm"]:
    r = results[ctrl]
    comp_data.append({
        "Controller": "WTM (Optimized)" if ctrl == "dynamic_wtm" else ctrl.upper(),
        "Avg Queue (Vehicles)": round(r["avg_queue_length"], 1),
        "Avg Travel Time (s)": round(r["avg_travel_time"], 1) if not np.isnan(r["avg_travel_time"]) else "N/A",
        "Throughput (Completed)": r["throughput"],
        "Completion Ratio": f"{r['completion_ratio'] * 100:.1f}%"
    })

df = pd.DataFrame(comp_data)
print("\n" + "=" * 80)
print("SIMULATION RESULTS UNDER HIGH CONGESTION (GRIDLOCK)")
print("=" * 80)
print(df.to_string(index=False))
print("=" * 80)

# --- PLOTTING GRIDLOCK COMPARISON ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.8,
    'axes.labelpad': 10,
    'axes.titlepad': 14,
    'grid.alpha': 0.3
})

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
colors = {"fixed": "#e74c3c", "backpressure": "#3498db", "dynamic_wtm": "#27ae60"}
linestyles = {"fixed": ":", "backpressure": "--", "dynamic_wtm": "-"}
markers = {"fixed": "s", "backpressure": "^", "dynamic_wtm": "o"}
labels = {"fixed": "Fixed-Time", "backpressure": "Backpressure", "dynamic_wtm": "Proposed WTM"}

for ctrl in ["fixed", "backpressure", "dynamic_wtm"]:
    history = results[ctrl]["queue_history"]
    ax.plot(
        history, 
        label=labels[ctrl], 
        color=colors[ctrl], 
        linestyle=linestyles[ctrl],
        marker=markers[ctrl],
        markevery=15,
        linewidth=2.5,
        markersize=7,
        markeredgecolor='black'
    )

ax.set_title("Queue Growth (Gridlock) Over Simulation Time", fontweight='bold', fontsize=15)
ax.set_ylabel("Total Network Queue Length (vehicles) [Lower is Better]", fontweight='bold', fontsize=13)
ax.set_xlabel("Simulation Step", fontweight='bold', fontsize=13)
ax.grid(True, linestyle='--')
ax.legend(loc="upper left", frameon=True, framealpha=0.95)

output_path = "gridlock_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSuccess: Gridlock comparison plot saved as '{output_path}'")
