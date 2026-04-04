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

# --- CONFIGURATION ---
CITIES = [
    ("Cubbon Park, Bengaluru, India", "Bengaluru"),
    ("Brandenburg Gate, Berlin, Germany", "Berlin"),
    ("Trafalgar Square, London, UK", "London"),
    ("Sydney Opera House, Sydney, Australia", "Sydney")
]

# Grid Configuration: 8 Results (Rows) x 4 Cities (Cols)
METRICS = [
    "avg_queue", "avg_travel", "throughput", "wait_box",
    "tp_demand", "tt_demand", "bc_dist", "param_dist"
]

# Sim Settings
SIM_STEPS = 120 # Increased to allow Fixed to finish some trips
NUM_TRIALS = 5 
DEMAND_LEVELS = {"Low": 8, "Med": 18, "High": 30}
ARRIVAL_RATE = 15 # Slightly lower rate to prevent early saturation

def get_city_graph(place):
    print(f"Fetching {place}...")
    try:
        # Use a larger patch for better visualization in the final paper
        raw_graph = ox.graph_from_address(place, dist=1200, network_type="drive", simplify=True)
    except:
        raw_graph = ox.graph_from_place(place, network_type="drive", simplify=True, retain_all=True)
    
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

def run_full_analysis(city_name, city_label):
    os.makedirs(city_label, exist_ok=True)
    G = get_city_graph(city_name)
    
    # 1. Topology & Centrality
    bc = nx.betweenness_centrality(G, k=min(50, len(G)), weight="travel_time")
    max_bc = max(bc.values()) if bc else 1.0
    for n in G.nodes():
        G.nodes[n]['betweenness_norm'] = bc.get(n, 0.0) / max_bc
    
    topology = prepare_topology(G)
    
    # Pre-calculate Routes
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

    # 2. Main Simulation (Fixed vs Backpressure vs Proposed)
    controllers = ["fixed", "backpressure", "dynamic_wtm"]
    results = {c: [] for c in controllers}
    for trial in range(NUM_TRIALS):
        ds = build_demand_schedule(SIM_STEPS, ARRIVAL_RATE, 42 + trial)
        for ctrl in controllers:
            res = run_simulation_with_waiting_time(G, routes, ds, ctrl, topology)
            results[ctrl].append(res)
            
    # 3. Demand Analysis
    demand_data = {ctrl: {"tp": [], "tt": []} for ctrl in controllers}
    for lbl, rate in DEMAND_LEVELS.items():
        for ctrl in controllers:
            tt_l, tp_l = [], []
            for trial in range(3):
                ds = build_demand_schedule(SIM_STEPS, rate, 42 + trial)
                res = run_simulation_with_waiting_time(G, routes, ds, ctrl, topology)
                tt_l.append(res["avg_travel_time"])
                tp_l.append(res["throughput"])
            demand_data[ctrl]["tt"].append(np.nanmean(tt_l))
            demand_data[ctrl]["tp"].append(np.mean(tp_l))

    # --- PLOTTING ---
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.linewidth': 1.5
    })
    # RGB Palette: Fixed (Red), Backpressure (Blue), Proposed (Green)
    # Optimized for Greyscale: Light/Dotted, Medium/Dashed, Dark/Solid
    colors = {"fixed": "#e74c3c", "backpressure": "#3498db", "dynamic_wtm": "#27ae60"}
    hatches = ["///", "\\\\", "xx"]
    styles = [":", "--", "-"]
    markers = ["s", "^", "o"]
    labels_short = ["Fixed", "BP", "Proposed"]
    
    # Row 1-3: Bars (Queue, Travel, Throughput)
    for i, (key, ylabel) in enumerate([
        ("avg_queue_length", "Avg Queue"), 
        ("avg_travel_time", "Travel Time (s)"), 
        ("throughput", "Throughput")
    ]):
        plt.figure(figsize=(5, 4))
        vals = [np.mean([r[key] for r in results[c]]) for c in controllers]
        bars = plt.bar(labels_short, vals, color=[colors[c] for c in controllers], 
                       edgecolor="black", linewidth=1.5)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{city_label}/plot_{i+1}.png", dpi=300)
        plt.close()

    # Row 4: Wait Box
    plt.figure(figsize=(5, 4))
    data = [[r["avg_wait_time"] for r in results[c] if not np.isnan(r["avg_wait_time"])] for c in controllers]
    bp = plt.boxplot(data, tick_labels=labels_short, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        ctrl = controllers[i]
        patch.set(facecolor=colors[ctrl], edgecolor='black', linewidth=1.5)
        patch.set_hatch(hatches[i])
    plt.ylabel("Wait Time (s)")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{city_label}/plot_4.png", dpi=300)
    plt.close()

    # Row 5-6: Demand Lines
    for i, (key, ylabel) in enumerate([("tp", "Throughput"), ("tt", "Travel Time")]):
        plt.figure(figsize=(5, 4))
        for j, ctrl in enumerate(controllers):
            plt.plot(list(DEMAND_LEVELS.keys()), demand_data[ctrl][key], 
                     label=labels_short[j], marker=markers[j], linestyle=styles[j], 
                     color=colors[ctrl], markersize=8, linewidth=2.0, markeredgecolor='black')
        plt.ylabel(ylabel)
        plt.legend(fontsize=9)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{city_label}/plot_{i+5}.png", dpi=300)
        plt.close()

    # Row 7: BC Distribution
    plt.figure(figsize=(5, 4))
    plt.hist(list(bc.values()), bins=15, color='#2c3e50', edgecolor='black', linewidth=1.2, alpha=0.7)
    plt.ylabel("Frequency")
    plt.xlabel("Betweenness Centrality")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{city_label}/plot_7.png", dpi=300)
    plt.close()

    # Row 8: Param Box
    plt.figure(figsize=(5, 4))
    params = [topology["alpha_dynamic"], topology["beta_dynamic"], topology["gamma_dynamic"]]
    param_colors = ["#f39c12", "#f1c40f", "#c0392b"] # Orange, Yellow, Dark Red
    bp = plt.boxplot(params, tick_labels=["Alpha", "Beta", "Gamma"], patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set(facecolor=param_colors[i], edgecolor='black', linewidth=1.2, alpha=0.8)
    plt.ylabel("Value")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{city_label}/plot_8.png", dpi=300)
    plt.close()

def create_final_grid():
    # Load images and stitch
    rows = 8
    cols = 4
    sample_img = Image.open(f"{CITIES[0][1]}/plot_1.png")
    w, h = sample_img.size
    
    # Margin for Row Labels (Left) and Column Labels (Top)
    left_margin = 400
    top_margin = 150
    
    grid = Image.new('RGB', (w * cols + left_margin, h * rows + top_margin), 'white')
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font, fallback to default
    try:
        font_large = ImageFont.truetype("arial.ttf", 100)
        font_med = ImageFont.truetype("arial.ttf", 80)
    except:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
    
    row_labels = [
        "Avg Queue", "Travel Time", "Throughput", "Wait Variance",
        "Throughput vs Demand", "Travel vs Demand", "Topology (BC)", "Control Params"
    ]
    
    for c_idx, (name, label) in enumerate(CITIES):
        # Draw Column Title
        draw.text((c_idx * w + left_margin + w//4, 30), label, fill='black', font=font_large)
        for r_idx in range(rows):
            # Draw Row Label (only for the first column)
            if c_idx == 0:
                draw.text((20, r_idx * h + top_margin + h//2), row_labels[r_idx], fill='black', font=font_med)
                
            img = Image.open(f"{label}/plot_{r_idx+1}.png")
            grid.paste(img, (c_idx * w + left_margin, r_idx * h + top_margin))
            
    grid.save("final_paper_grid_hd.png", dpi=(300, 300))
    print("High-Res Greyscale Grid saved as final_paper_grid_hd.png")

if __name__ == "__main__":
    for city_name, label in CITIES:
        print(f"\nProcessing {label}...")
        run_full_analysis(city_name, label)
    create_final_grid()
