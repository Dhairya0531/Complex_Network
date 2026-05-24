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

# --- CONFIGURATION ---
CITIES = [
    ("Cubbon Park, Bengaluru, India", "Bengaluru"),
    ("Brandenburg Gate, Berlin, Germany", "Berlin"),
    ("Trafalgar Square, London, UK", "London"),
    ("Sydney Opera House, Sydney, Australia", "Sydney")
]
SIM_STEPS = 100
ARRIVAL_RATE = 120

def get_graph_local(place, label):
    import osmnx as ox
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
        if arrivals == 0: schedule.append([]); continue
        schedule.append(local_rng.integers(0, num_routes, size=arrivals).tolist())
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
                if 3 <= len(path) <= 7: routes.append({"edges": list(zip(path[:-1], path[1:]))})
            except: continue
        num_trials = 10
        for name, centrality_map in measures.items():
            nx.set_node_attributes(G, centrality_map, "betweenness_norm") 
            topology = prepare_topology(G)
            
            travel_times = []
            for trial in range(num_trials):
                trial_demand = build_demand_local(SIM_STEPS, ARRIVAL_RATE, 42 + trial, len(routes))
                res = run_simulation_with_waiting_time(G, routes, trial_demand, "dynamic_wtm", topology)
                t_time = res['avg_travel_time']
                if not np.isnan(t_time):
                    travel_times.append(t_time)
            
            avg_t_time = np.mean(travel_times) if travel_times else 0
            city_results.append({
                "City": label, 
                "Measure": name, 
                "Avg Travel Time": avg_t_time
            })

    df = pd.DataFrame(city_results)
    pivot_df = df.pivot(index='City', columns='Measure', values='Avg Travel Time')
    
    # --- CLEAN PAPER PLOT ---
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.8,
        'axes.labelpad': 10,
        'axes.titlepad': 14,
        'xtick.major.pad': 6,
        'ytick.major.pad': 6,
        'legend.frameon': True,
    })
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    palette = ['#27ae60', '#3498db', '#e67e22', '#9b59b6']
    hatches = ['///', '\\\\\\\\', 'xx', '..']
    plt.rcParams['hatch.linewidth'] = 2.2
    pivot_df.plot(
        kind='bar',
        ax=ax,
        edgecolor='black',
        linewidth=1.8,
        color=palette,
        width=0.78,
    )

    # Apply grayscale colors, hatch patterns, and black borders for printing.
    for measure_idx, container in enumerate(ax.containers):
        color = palette[measure_idx % len(palette)]
        hatch = hatches[measure_idx % len(hatches)]
        for bar in container:
            bar.set_facecolor(color)
            bar.set_edgecolor('black')
            bar.set_hatch(hatch)
    
    ax.set_ylabel("Avg Travel Time (seconds) [Lower is Better]", fontweight='bold', fontsize=15)
    ax.set_xlabel("City", fontweight='bold', fontsize=15)
    ax.set_title("Impact of Centrality Metric on Performance", fontweight='bold', fontsize=20)
    ax.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    
    # Sync legend handles to have correct facecolors, edgecolors, and hatch patterns
    import matplotlib.patches as mpatches
    handles, labels = ax.get_legend_handles_labels()
    legend_patches = [
        mpatches.Patch(
            facecolor=palette[idx % len(palette)],
            edgecolor='black',
            hatch=hatches[idx % len(hatches)],
            label=labels[idx]
        )
        for idx in range(len(labels))
    ]
        
    ax.legend(
        handles=legend_patches,
        title="Centrality Type",
        title_fontsize=14,
        fontsize=12,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
    )
    fig.savefig("multi_city_centrality_comparison.png", dpi=300, bbox_inches='tight', pad_inches=0.25)
    print("Success: multi_city_centrality_comparison.png")

if __name__ == "__main__":
    run_multi_city_comparison()
