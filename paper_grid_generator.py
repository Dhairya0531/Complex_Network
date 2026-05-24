import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
NUM_TRIALS = 100   
DEMAND_LEVELS = {"Low": 50, "Med": 120, "High": 220}
ARRIVAL_RATE = 120 

def get_city_graph(place, city_label):
    print(f"Fetching {city_label}...")
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
    bc = nx.betweenness_centrality(G, k=min(40, len(G)), weight="travel_time")
    max_bc = max(bc.values()) if bc else 1.0
    for n in G.nodes(): G.nodes[n]['betweenness_norm'] = bc.get(n, 0.0) / max_bc
    topology = prepare_topology(G)
    candidate_nodes = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]
    routes = []
    rng = np.random.default_rng(42)
    while len(routes) < 20:
        o, d = rng.choice(candidate_nodes, 2, replace=False)
        try:
            path = nx.shortest_path(G, o, d, weight="travel_time")
            if 3 <= len(path) <= 8: routes.append({"edges": list(zip(path[:-1], path[1:]))})
        except: continue

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
            demand_data[ctrl]["tt"].append(np.nanmean(tt_l) if not np.all(np.isnan(tt_l)) else 0)
            demand_data[ctrl]["tp"].append(np.mean(tp_l))

    plt.rcParams.update({
        'font.size': 14,
        'axes.linewidth': 2.0,
        'axes.labelpad': 10,
        'axes.titlepad': 14,
        'xtick.major.pad': 6,
        'ytick.major.pad': 6,
        'legend.frameon': True,
    })
    colors = {"fixed": "#e74c3c", "backpressure": "#3498db", "dynamic_wtm": "#27ae60"}
    hatches = ["///", "\\\\\\\\", "xx"]
    labels_short = ["Fixed", "BP", "Proposed"]
    
    # Rows 1-3: Bars
    metrics = [
        ("avg_queue_length", "Avg Queue Length (vehicles) [Lower is Better]", "Avg Queue Length"),
        ("avg_travel_time", "Avg Travel Time (seconds) [Lower is Better]", "Avg Travel Time"),
        ("throughput", "Throughput (vehicles) [Higher is Better]", "Throughput")
    ]
    for i, (key, ylabel, title_suffix) in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
        vals = [np.nanmean([r[key] for r in results[c]]) for c in controllers]
        vals = [v if (not np.isnan(v) and v > 0) else 1e-3 for v in vals]
        bars = ax.bar(labels_short, vals, color=[colors[c] for c in controllers], edgecolor="black", linewidth=1.8)
        for b, h in zip(bars, hatches): b.set_hatch(h)
        ax.set_title(f"{city_label} - {title_suffix}", fontweight='bold', fontsize=18)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=15)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        fig.savefig(f"{city_label}/plot_{i+1}.png", dpi=300, bbox_inches='tight', pad_inches=0.18)
        plt.close(fig)

    # Row 4: Wait Variance
    fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    data = [[r["avg_wait_time"] for r in results[c] if not np.isnan(r["avg_wait_time"])] for c in controllers]
    bp = ax.boxplot(data, tick_labels=labels_short, patch_artist=True)
    for j, patch in enumerate(bp['boxes']):
        patch.set(facecolor=colors[controllers[j]], edgecolor='black', linewidth=3)
        patch.set_hatch(hatches[j])
    ax.set_title(f"{city_label} - Wait Time Spread", fontweight='bold', fontsize=18)
    ax.set_ylabel("Wait Time (seconds) [Lower is Better]", fontweight='bold', fontsize=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.savefig(f"{city_label}/plot_4.png", dpi=300, bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)

    # Rows 5-6: Demand
    demand_metrics = [
        ("tp", "Throughput (vehicles) [Higher is Better]", "Throughput vs Demand"),
        ("tt", "Avg Travel Time (seconds) [Lower is Better]", "Travel Time vs Demand")
    ]
    linestyles = [":", "--", "-"]
    markers = ["s", "^", "o"]
    for i, (key, ylabel, title_suffix) in enumerate(demand_metrics):
        fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
        for j, ctrl in enumerate(controllers):
            ax.plot(list(DEMAND_LEVELS.keys()), demand_data[ctrl][key], label=labels_short[j],
                    marker=markers[j], linestyle=linestyles[j], color=colors[ctrl], linewidth=2.8,
                    markersize=8, markeredgecolor='black')
        ax.set_title(f"{city_label} - {title_suffix}", fontweight='bold', fontsize=18)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=15)
        ax.set_xlabel("Traffic Demand", fontweight='bold', fontsize=15)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.legend(
            fontsize=12,
            frameon=True,
            framealpha=0.95,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
        )
        ax.set_xlim(-0.05, 2.05)
        ax.grid(True, linestyle='--', alpha=0.35)
        fig.savefig(f"{city_label}/plot_{i+5}.png", dpi=300, bbox_inches='tight', pad_inches=0.18)
        plt.close(fig)

    # Row 7: Topology
    fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    ax.hist(list(bc.values()), bins=15, color='#444444', edgecolor='black', alpha=0.8)
    ax.set_title(f"{city_label} - Centrality Distribution", fontweight='bold', fontsize=18)
    ax.set_ylabel("Frequency (Node Count) [n/a]", fontweight='bold', fontsize=15)
    ax.set_xlabel("Betweenness Centrality", fontweight='bold', fontsize=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.savefig(f"{city_label}/plot_7.png", dpi=300, bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)

    # Row 8: Params
    fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    params = [topology["alpha_dynamic"], topology["beta_dynamic"], topology["gamma_dynamic"]]
    bp = ax.boxplot(params, tick_labels=["Alpha", "Beta", "Gamma"], patch_artist=True)
    param_colors = ["#f39c12", "#f1c40f", "#c0392b"]
    for j, patch in enumerate(bp['boxes']):
        patch.set(facecolor=param_colors[j], edgecolor='black', linewidth=3)
    ax.set_title(f"{city_label} - Control Parameters", fontweight='bold', fontsize=18)
    ax.set_ylabel("Weight Value [n/a]", fontweight='bold', fontsize=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.savefig(f"{city_label}/plot_8.png", dpi=300, bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)

def create_final_grid():
    rows, cols = 8, 4
    # Larger dimensions to ensure legibility on A4 papers
    cell_w, cell_h = 1200, 900
    left_m, top_m = 20, 170
    grid = Image.new('RGB', (cell_w * cols + left_m, cell_h * rows + top_m), 'white')
    draw = ImageDraw.Draw(grid)
    try:
        font_h = ImageFont.truetype("arial.ttf", 64)
    except:
        font_h = ImageFont.load_default()
    
    city_labels = ["Bengaluru", "Berlin", "London", "Sydney"]
    
    for c_idx, label in enumerate(city_labels):
        draw.text((c_idx * cell_w + left_m + cell_w // 2, 70), label, fill='black', font=font_h, anchor="mm")
        for r_idx in range(rows):
            x0 = c_idx * cell_w + left_m
            y0 = r_idx * cell_h + top_m
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline="#d0d0d0", width=2)
            img = Image.open(f"{label}/plot_{r_idx+1}.png")
            fitted = ImageOps.contain(img, (cell_w - 24, cell_h - 24), Image.LANCZOS)
            paste_x = x0 + (cell_w - fitted.width) // 2
            paste_y = y0 + (cell_h - fitted.height) // 2
            grid.paste(fitted, (paste_x, paste_y))
    grid.save("final_paper_grid_hd.png", dpi=(300, 300))
    print("Success: final_paper_grid_hd.png")

if __name__ == "__main__":
    # Regenerate all plots for the four cities
    for place, label in CITIES:
        run_full_analysis(place, label)
    create_final_grid()
