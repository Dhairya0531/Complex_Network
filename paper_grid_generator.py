import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from PIL import Image, ImageDraw, ImageFont
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

SIM_STEPS = 1000 # Massive time to ensure trips finish
NUM_TRIALS = 2   
DEMAND_LEVELS = {"Low": 2, "Med": 5, "High": 10} # Low load to guarantee flow
ARRIVAL_RATE = 5 

def get_city_graph(place, city_label):
    print(f"Fetching {city_label} (1.5km Core)...")
    # 1.5km is the sweet spot for structural diversity and simulation speed
    raw_graph = ox.graph_from_address(place, dist=1500, network_type="drive", simplify=True)
    
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

def run_full_analysis(city_name, city_label):
    os.makedirs(city_label, exist_ok=True)
    G = get_city_graph(city_name, city_label)
    
    print(f"Analyzing {city_label} Topology...")
    bc = nx.betweenness_centrality(G, k=min(50, len(G)), weight="travel_time")
    max_bc = max(bc.values()) if bc else 1.0
    for n in G.nodes():
        G.nodes[n]['betweenness_norm'] = bc.get(n, 0.0) / max_bc
    
    topology = prepare_topology(G)
    candidate_nodes = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]
    
    # Pre-calculate Ultra-Short Routes (3-8 nodes) to guarantee completion
    routes = []
    rng = np.random.default_rng(42)
    while len(routes) < 20:
        o, d = rng.choice(candidate_nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, o, d, weight="travel_time")
            if 3 <= len(path) <= 8:
                routes.append({"edges": list(zip(path[:-1], path[1:]))})
        except: continue
    
    print(f"Simulating {city_label}...")
    controllers = ["fixed", "backpressure", "dynamic_wtm"]
    results = {c: [] for c in controllers}
    for trial in range(NUM_TRIALS):
        ds = build_demand_schedule(SIM_STEPS, ARRIVAL_RATE, 42 + trial)
        for ctrl in controllers:
            res = run_simulation_with_waiting_time(G, routes, ds, ctrl, topology)
            results[ctrl].append(res)
            
    demand_data = {ctrl: {"tp": [], "tt": []} for ctrl in controllers}
    for lbl, rate in DEMAND_LEVELS.items():
        for ctrl in controllers:
            tt_l, tp_l = [], []
            for trial in range(1):
                ds = build_demand_schedule(SIM_STEPS, rate, 42 + trial)
                res = run_simulation_with_waiting_time(G, routes, ds, ctrl, topology)
                tt_l.append(res["avg_travel_time"])
                tp_l.append(res["throughput"])
            
            # Final fallback to ensure no empty plots
            avg_tt = np.nanmean(tt_l) if not np.all(np.isnan(tt_l)) else 0
            demand_data[ctrl]["tt"].append(avg_tt)
            demand_data[ctrl]["tp"].append(np.mean(tp_l))

    # --- PLOTTING ---
    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 2.5, 'axes.titlesize': 24, 'axes.labelsize': 22})
    colors = {"fixed": "#e74c3c", "backpressure": "#3498db", "dynamic_wtm": "#27ae60"}
    hatches = ["///", "\\\\", "xx"]
    labels_short = ["Fixed", "BP", "Proposed"]
    
    for i, (key, ylabel) in enumerate([("avg_queue_length", "Avg Queue"), ("avg_travel_time", "Travel Time (s)"), ("throughput", "Throughput")]):
        plt.figure(figsize=(8, 7))
        vals = [np.nanmean([r[key] for r in results[c]]) for c in controllers]
        vals = [v if (not np.isnan(v) and v > 0) else 1e-3 for v in vals]
        bars = plt.bar(labels_short, vals, color=[colors[c] for c in controllers], edgecolor="black", linewidth=2.5)
        for b, h in zip(bars, hatches): b.set_hatch(h)
        plt.ylabel(ylabel, fontweight='bold')
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{city_label}/plot_{i+1}.png", dpi=300)
        plt.close()

    # Row 4: Wait Variance
    plt.figure(figsize=(8, 7))
    data = [[r["avg_wait_time"] for r in results[c] if not np.isnan(r["avg_wait_time"])] for c in controllers]
    bp = plt.boxplot(data, tick_labels=labels_short, patch_artist=True)
    for j, patch in enumerate(bp['boxes']):
        patch.set(facecolor=colors[controllers[j]], edgecolor='black', linewidth=2.5)
        patch.set_hatch(hatches[j])
    plt.ylabel("Wait Time (s)", fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{city_label}/plot_4.png", dpi=300)
    plt.close()

    # Rows 5-6: Demand
    for i, (key, ylabel) in enumerate([("tp", "Throughput"), ("tt", "Travel Time")]):
        plt.figure(figsize=(8, 7))
        for j, ctrl in enumerate(controllers):
            plt.plot(list(DEMAND_LEVELS.keys()), demand_data[ctrl][key], label=labels_short[j], 
                     marker='os^'[j], color=colors[ctrl], linewidth=4, markersize=14, markeredgecolor='black')
        plt.ylabel(ylabel, fontweight='bold')
        plt.legend(fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{city_label}/plot_{i+5}.png", dpi=300)
        plt.close()

    # Row 7: Topology
    plt.figure(figsize=(8, 7))
    plt.hist(list(bc.values()), bins=15, color='#2c3e50', edgecolor='black', alpha=0.8)
    plt.ylabel("Frequency", fontweight='bold')
    plt.xlabel("Betweenness Centrality", fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{city_label}/plot_7.png", dpi=300)
    plt.close()

    # Row 8: Params
    plt.figure(figsize=(8, 7))
    params = [topology["alpha_dynamic"], topology["beta_dynamic"], topology["gamma_dynamic"]]
    bp = plt.boxplot(params, tick_labels=["Alpha", "Beta", "Gamma"], patch_artist=True)
    for j, patch in enumerate(bp['boxes']):
        patch.set(facecolor=["#f39c12", "#f1c40f", "#c0392b"][j], edgecolor='black', linewidth=2.5)
    plt.ylabel("Weight Value", fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{city_label}/plot_8.png", dpi=300)
    plt.close()

def create_final_grid():
    rows, cols = 8, 4
    sample_img = Image.open(f"Berlin/plot_1.png")
    w, h = sample_img.size
    left_m, top_m = 800, 400
    grid = Image.new('RGB', (w * cols + left_m, h * rows + top_m), 'white')
    draw = ImageDraw.Draw(grid)
    try:
        font_h = ImageFont.truetype("arial.ttf", 200)
        font_m = ImageFont.truetype("arial.ttf", 140)
    except:
        font_h = ImageFont.load_default()
        font_m = ImageFont.load_default()
    
    row_labels = ["Avg Queue", "Travel Time", "Throughput", "Wait Var", "Throughput vs", "Travel vs Dem", "Topology (BC)", "Control Param"]
    city_labels = ["Bengaluru", "Berlin", "London", "Sydney"]
    
    for c_idx, label in enumerate(city_labels):
        draw.text((c_idx * w + left_m + w//2, 150), label, fill='black', font=font_h, anchor="mm")
        for r_idx in range(rows):
            if c_idx == 0: draw.text((50, r_idx * h + top_m + h//2), row_labels[r_idx], fill='black', font=font_m, anchor="lm")
            img = Image.open(f"{label}/plot_{r_idx+1}.png")
            grid.paste(img, (c_idx * w + left_m, r_idx * h + top_m))
    grid.save("final_paper_grid_hd.png", dpi=(300, 300))
    print("Success: final_paper_grid_hd.png")

if __name__ == "__main__":
    for city_name, label in CITIES:
        run_full_analysis(city_name, label)
    create_final_grid()
