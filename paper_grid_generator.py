import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw, ImageFont, ImageOps
from main import (
    prepare_topology, 
    run_simulation_with_waiting_time, 
)
from centrality_comparison import (
    ARRIVAL_RATE as BENCHMARK_ARRIVAL_RATE,
    CENTRALITY_SCALE,
    SIM_STEPS as BENCHMARK_SIM_STEPS,
    WTM_SCORE_WEIGHTS,
    build_bottleneck_routes,
    build_demand_local,
    get_graph_local,
)

# --- CONFIGURATION ---
CITIES = [
    ("Cubbon Park, Bengaluru, India", "Bengaluru"),
    ("Brandenburg Gate, Berlin, Germany", "Berlin"),
    ("Trafalgar Square, London, UK", "London"),
    ("Sydney Opera House, Sydney, Australia", "Sydney"),
]

SIM_STEPS = BENCHMARK_SIM_STEPS
NUM_TRIALS = 20
DEMAND_LEVELS = {"Low": 10, "Med": 16, "High": 24}
ARRIVAL_RATE = BENCHMARK_ARRIVAL_RATE


def get_city_graph(place, city_label):
    print(f"Building bottleneck benchmark for {city_label}...")
    return get_graph_local(place, city_label)

def run_full_analysis(city_name, city_label):
    os.makedirs(city_label, exist_ok=True)
    G = get_city_graph(city_name, city_label)
    bc = nx.betweenness_centrality(G, weight="travel_time", normalized=True)
    max_bc = max(bc.values()) if bc else 1.0
    for n in G.nodes():
        G.nodes[n]["betweenness_norm"] = bc.get(n, 0.0) / max_bc
    topology = prepare_topology(G)
    routes = build_bottleneck_routes(G, city_label, num_routes=14, seed=42)

    controllers = ["fixed", "backpressure", "dynamic_wtm"]
    results = {c: [] for c in controllers}
    for trial in range(NUM_TRIALS):
        ds = build_demand_local(SIM_STEPS, ARRIVAL_RATE, 42 + trial, routes)
        for ctrl in controllers:
            res = run_simulation_with_waiting_time(
                G,
                routes,
                ds,
                ctrl,
                topology,
                wtm_score_weights=WTM_SCORE_WEIGHTS,
                centrality_scale=CENTRALITY_SCALE,
            )
            results[ctrl].append(res)

    demand_data = {ctrl: {"tp": [], "tt": [], "tp_err": [], "tt_err": []} for ctrl in controllers}
    for lbl, rate in DEMAND_LEVELS.items():
        for ctrl in controllers:
            tt_l, tp_l = [], []
            for trial in range(3):
                ds = build_demand_local(SIM_STEPS, rate, 77 + trial, routes)
                res = run_simulation_with_waiting_time(
                    G,
                    routes,
                    ds,
                    ctrl,
                    topology,
                    wtm_score_weights=WTM_SCORE_WEIGHTS,
                    centrality_scale=CENTRALITY_SCALE,
                )
                tt_l.append(res["avg_travel_time"])
                tp_l.append(res["throughput"])
            demand_data[ctrl]["tt"].append(np.nanmean(tt_l) if not np.all(np.isnan(tt_l)) else 0)
            demand_data[ctrl]["tt_err"].append(np.nanstd(tt_l) if not np.all(np.isnan(tt_l)) else 0)
            demand_data[ctrl]["tp"].append(np.mean(tp_l))
            demand_data[ctrl]["tp_err"].append(np.std(tp_l))

    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'legend.fontsize': 19,
        'axes.linewidth': 2.2,
        'axes.labelpad': 10,
        'xtick.major.pad': 8,
        'ytick.major.pad': 8,
        'legend.frameon': True,
        'hatch.linewidth': 2.0,
        'hatch.color': 'black',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })
    
    # --- VIBRANT & PRINT-OPTIMIZED COLOR PALETTE ---
    # Rich, vibrant modern hues on display + distinct patterns & luminance for greyscale A4 printing
    bar_colors = {"fixed": "#ffb3ba", "backpressure": "#bae1ff", "dynamic_wtm": "#baffc9"}
    line_colors = {"fixed": "#e63946", "backpressure": "#1d3557", "dynamic_wtm": "#2a9d8f"}
    marker_face_colors = {"fixed": "#e63946", "backpressure": "#457b9d", "dynamic_wtm": "#2a9d8f"}
    hatches = ["///", "\\\\\\", "xxx"]
    labels_short = ["Fixed", "BP", "UrbSigOpt"]
    
    def save_multiformat(fig, prefix):
        fig.savefig(f"{prefix}.png", dpi=300, bbox_inches='tight', pad_inches=0.18)
        fig.savefig(f"{prefix}.pdf", bbox_inches='tight', pad_inches=0.18)
        fig.savefig(f"{prefix}.svg", bbox_inches='tight', pad_inches=0.18)

    # Rows 1-3: Bars with Error Bars (Quantifying trial randomization)
    metrics = [
        ("avg_queue_length", "Avg Queue Length (vehicles)\n[Lower is Better]", "Avg Queue Length"),
        ("avg_travel_time", "Avg Travel Time (seconds)\n[Lower is Better]", "Avg Travel Time"),
        ("throughput", "Throughput (vehicles)\n[Higher is Better]", "Throughput")
    ]
    for i, (key, ylabel, title_suffix) in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
        vals = [np.nanmean([r[key] for r in results[c]]) for c in controllers]
        vals = [v if (not np.isnan(v) and v > 0) else 1e-3 for v in vals]
        errs = [np.nanstd([r[key] for r in results[c]]) for c in controllers]
        errs = [e if not np.isnan(e) else 0.0 for e in errs]
        
        bars = ax.bar(
            labels_short, 
            vals, 
            yerr=errs, 
            capsize=7, 
            color=[bar_colors[c] for c in controllers], 
            edgecolor="black", 
            linewidth=2.2,
            error_kw=dict(elinewidth=2.2, ecolor='black', capthick=2.2)
        )
        for b, h in zip(bars, hatches): 
            b.set_hatch(h)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=22)
        ax.tick_params(axis='x', labelsize=21)
        ax.tick_params(axis='y', labelsize=21)
        ax.grid(axis='y', linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.2)
        save_multiformat(fig, f"{city_label}/plot_{i+1}")
        plt.close(fig)

    # Row 4: Wait Time Boxplot (Distribution across randomized trials)
    fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    data = [[r["avg_wait_time"] for r in results[c] if not np.isnan(r["avg_wait_time"])] for c in controllers]
    bp = ax.boxplot(
        data, 
        tick_labels=labels_short, 
        patch_artist=True, 
        widths=0.55,
        boxprops=dict(linewidth=2.2, color='black'),
        whiskerprops=dict(linewidth=2.2, color='black'),
        capprops=dict(linewidth=2.2, color='black'),
        medianprops=dict(linewidth=3.2, color='black'),
        flierprops=dict(marker='o', markersize=6, markerfacecolor='#555555', markeredgecolor='black', markeredgewidth=1.2, alpha=0.85)
    )
    for j, patch in enumerate(bp['boxes']):
        patch.set(facecolor=bar_colors[controllers[j]], edgecolor='black', linewidth=2.2)
        patch.set_hatch(hatches[j])
    ax.set_ylabel("Wait Time (seconds)\n[Lower is Better]", fontweight='bold', fontsize=22)
    ax.tick_params(axis='x', labelsize=21)
    ax.tick_params(axis='y', labelsize=21)
    ax.grid(axis='y', linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.2)
    save_multiformat(fig, f"{city_label}/plot_4")
    plt.close(fig)

    # Rows 5-6: Demand Performance with Error Lines (Stochastic Demand trials)
    demand_metrics = [
        ("tp", "Throughput (vehicles)\n[Higher is Better]", "Throughput vs Demand"),
        ("tt", "Avg Travel Time (seconds)\n[Lower is Better]", "Travel Time vs Demand")
    ]
    # Distinct line styles: Dotted (Fixed), Dash-dot (BP), Solid (UrbSigOpt)
    linestyles = [(0, (2.0, 2.0)), (0, (5, 2.5, 1.5, 2.5)), "-"]
    markers = ["s", "^", "o"]
    for i, (key, ylabel, title_suffix) in enumerate(demand_metrics):
        fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
        for j, ctrl in enumerate(controllers):
            ax.errorbar(
                list(DEMAND_LEVELS.keys()), 
                demand_data[ctrl][key], 
                yerr=demand_data[ctrl][f"{key}_err"],
                label=labels_short[j],
                marker=markers[j], 
                linestyle=linestyles[j], 
                color=line_colors[ctrl], 
                linewidth=3.2 if j < 2 else 3.6,
                markersize=13, 
                markeredgecolor='black',
                markeredgewidth=1.8,
                markerfacecolor=marker_face_colors[ctrl],
                capsize=7,
                capthick=2.2,
                elinewidth=2.2,
                ecolor=line_colors[ctrl]
            )
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=22)
        ax.set_xlabel("Traffic Demand", fontweight='bold', fontsize=22)
        ax.tick_params(axis='x', labelsize=21)
        ax.tick_params(axis='y', labelsize=21)
        ax.legend(
            fontsize=18,
            frameon=True,
            framealpha=0.96,
            facecolor='white',
            edgecolor='black',
            loc='best',
        )
        ax.set_xlim(-0.08, 2.08)
        ax.grid(True, linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.2)
        save_multiformat(fig, f"{city_label}/plot_{i+5}")
        plt.close(fig)

    # Row 7: Topology - Betweenness Centrality
    fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    ax.hist(list(bc.values()), bins=15, color='#457b9d', edgecolor='black', linewidth=1.8, alpha=0.9)
    ax.set_ylabel("Frequency (Node Count)", fontweight='bold', fontsize=24)
    ax.set_xlabel("Betweenness Centrality", fontweight='bold', fontsize=24)
    ax.tick_params(axis='x', labelsize=21)
    ax.tick_params(axis='y', labelsize=21)
    ax.grid(axis='y', linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.2)
    save_multiformat(fig, f"{city_label}/plot_7")
    plt.close(fig)

    # Row 8: Weight Parameters Distribution
    fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    params = [topology["alpha_dynamic"], topology["beta_dynamic"], topology["gamma_dynamic"]]
    bp = ax.boxplot(
        params, 
        tick_labels=["Alpha", "Beta", "Gamma"], 
        patch_artist=True, 
        widths=0.55,
        boxprops=dict(linewidth=2.2, color='black'),
        whiskerprops=dict(linewidth=2.2, color='black'),
        capprops=dict(linewidth=2.2, color='black'),
        medianprops=dict(linewidth=3.2, color='black'),
        flierprops=dict(marker='o', markersize=6, markerfacecolor='#555555', markeredgecolor='black', markeredgewidth=1.2, alpha=0.85)
    )
    param_colors = ["#f4a261", "#e9c46a", "#e76f51"]
    param_hatches = ["///", "\\\\\\", "xxx"]
    for j, patch in enumerate(bp['boxes']):
        patch.set(facecolor=param_colors[j], edgecolor='black', linewidth=2.2)
        patch.set_hatch(param_hatches[j])
    ax.set_ylabel("Weight Value", fontweight='bold', fontsize=24)
    ax.tick_params(axis='x', labelsize=21)
    ax.tick_params(axis='y', labelsize=21)
    ax.grid(axis='y', linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.2)
    save_multiformat(fig, f"{city_label}/plot_8")
    plt.close(fig)

    return {
        "results": results,
        "demand_data": demand_data,
        "bc": bc,
        "topology": topology,
    }


def create_unified_vector_and_hd_grid(city_datasets):
    """
    Generates a single unified figure containing the 8x4 grid,
    exporting vector PDF, vector SVG, and high-res 300 DPI PNG.
    """
    city_labels = ["Bengaluru", "Berlin", "London", "Sydney"]
    controllers = ["fixed", "backpressure", "dynamic_wtm"]
    labels_short = ["Fixed", "BP", "UrbSigOpt"]
    hatches = ["///", "\\\\\\", "xxx"]
    bar_colors = {"fixed": "#ffb3ba", "backpressure": "#bae1ff", "dynamic_wtm": "#baffc9"}
    line_colors = {"fixed": "#e63946", "backpressure": "#1d3557", "dynamic_wtm": "#2a9d8f"}
    marker_face_colors = {"fixed": "#e63946", "backpressure": "#457b9d", "dynamic_wtm": "#2a9d8f"}
    linestyles = [(0, (2.0, 2.0)), (0, (5, 2.5, 1.5, 2.5)), "-"]
    markers = ["s", "^", "o"]
    
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 16,
        'axes.linewidth': 1.8,
        'axes.labelpad': 8,
        'xtick.major.pad': 6,
        'ytick.major.pad': 6,
        'legend.frameon': True,
        'hatch.linewidth': 1.8,
        'hatch.color': 'black',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })

    fig, axes = plt.subplots(8, 4, figsize=(26, 36), constrained_layout=True)
    
    metrics = [
        ("avg_queue_length", "Avg Queue Length (vehicles)\n[Lower is Better]"),
        ("avg_travel_time", "Avg Travel Time (seconds)\n[Lower is Better]"),
        ("throughput", "Throughput (vehicles)\n[Higher is Better]")
    ]

    for col_idx, (place, city_label) in enumerate(CITIES):
        data_city = city_datasets[city_label]
        results = data_city["results"]
        demand_data = data_city["demand_data"]
        bc = data_city["bc"]
        topology = data_city["topology"]
        
        # Header title on top row
        axes[0, col_idx].set_title(city_label, fontsize=28, fontweight='bold', pad=16)

        # Rows 1-3: Bar plots
        for row_idx, (key, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            vals = [np.nanmean([r[key] for r in results[c]]) for c in controllers]
            vals = [v if (not np.isnan(v) and v > 0) else 1e-3 for v in vals]
            errs = [np.nanstd([r[key] for r in results[c]]) for c in controllers]
            errs = [e if not np.isnan(e) else 0.0 for e in errs]
            
            bars = ax.bar(
                labels_short, 
                vals, 
                yerr=errs, 
                capsize=6, 
                color=[bar_colors[c] for c in controllers], 
                edgecolor="black", 
                linewidth=1.8,
                error_kw=dict(elinewidth=1.8, ecolor='black', capthick=1.8)
            )
            for b, h in zip(bars, hatches): 
                b.set_hatch(h)
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=18)
            ax.tick_params(axis='x', labelsize=17)
            ax.tick_params(axis='y', labelsize=17)
            ax.grid(axis='y', linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.0)

        # Row 4: Wait Time Boxplot
        ax_w = axes[3, col_idx]
        data_wait = [[r["avg_wait_time"] for r in results[c] if not np.isnan(r["avg_wait_time"])] for c in controllers]
        bp_w = ax_w.boxplot(
            data_wait, 
            tick_labels=labels_short, 
            patch_artist=True, 
            widths=0.55,
            boxprops=dict(linewidth=1.8, color='black'),
            whiskerprops=dict(linewidth=1.8, color='black'),
            capprops=dict(linewidth=1.8, color='black'),
            medianprops=dict(linewidth=2.6, color='black'),
            flierprops=dict(marker='o', markersize=5, markerfacecolor='#555555', markeredgecolor='black', markeredgewidth=1.0, alpha=0.85)
        )
        for j, patch in enumerate(bp_w['boxes']):
            patch.set(facecolor=bar_colors[controllers[j]], edgecolor='black', linewidth=1.8)
            patch.set_hatch(hatches[j])
        ax_w.set_ylabel("Wait Time (seconds)\n[Lower is Better]", fontweight='bold', fontsize=18)
        ax_w.tick_params(axis='x', labelsize=17)
        ax_w.tick_params(axis='y', labelsize=17)
        ax_w.grid(axis='y', linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.0)

        # Row 5: Throughput vs Demand
        ax_tp = axes[4, col_idx]
        for j, ctrl in enumerate(controllers):
            ax_tp.errorbar(
                list(DEMAND_LEVELS.keys()), 
                demand_data[ctrl]["tp"], 
                yerr=demand_data[ctrl]["tp_err"],
                label=labels_short[j],
                marker=markers[j], 
                linestyle=linestyles[j], 
                color=line_colors[ctrl], 
                linewidth=2.8 if j < 2 else 3.2,
                markersize=11, 
                markeredgecolor='black',
                markeredgewidth=1.5,
                markerfacecolor=marker_face_colors[ctrl],
                capsize=6,
                capthick=1.8,
                elinewidth=1.8,
                ecolor=line_colors[ctrl]
            )
        ax_tp.set_ylabel("Throughput (vehicles)\n[Higher is Better]", fontweight='bold', fontsize=18)
        ax_tp.set_xlabel("Traffic Demand", fontweight='bold', fontsize=18)
        ax_tp.tick_params(axis='x', labelsize=17)
        ax_tp.tick_params(axis='y', labelsize=17)
        ax_tp.legend(fontsize=15, frameon=True, framealpha=0.96, facecolor='white', edgecolor='black', loc='best')
        ax_tp.set_xlim(-0.08, 2.08)
        ax_tp.grid(True, linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.0)

        # Row 6: Avg Travel Time vs Demand
        ax_tt = axes[5, col_idx]
        for j, ctrl in enumerate(controllers):
            ax_tt.errorbar(
                list(DEMAND_LEVELS.keys()), 
                demand_data[ctrl]["tt"], 
                yerr=demand_data[ctrl]["tt_err"],
                label=labels_short[j],
                marker=markers[j], 
                linestyle=linestyles[j], 
                color=line_colors[ctrl], 
                linewidth=2.8 if j < 2 else 3.2,
                markersize=11, 
                markeredgecolor='black',
                markeredgewidth=1.5,
                markerfacecolor=marker_face_colors[ctrl],
                capsize=6,
                capthick=1.8,
                elinewidth=1.8,
                ecolor=line_colors[ctrl]
            )
        ax_tt.set_ylabel("Avg Travel Time (seconds)\n[Lower is Better]", fontweight='bold', fontsize=18)
        ax_tt.set_xlabel("Traffic Demand", fontweight='bold', fontsize=18)
        ax_tt.tick_params(axis='x', labelsize=17)
        ax_tt.tick_params(axis='y', labelsize=17)
        ax_tt.legend(fontsize=15, frameon=True, framealpha=0.96, facecolor='white', edgecolor='black', loc='best')
        ax_tt.set_xlim(-0.08, 2.08)
        ax_tt.grid(True, linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.0)

        # Row 7: BC Distribution
        ax_bc = axes[6, col_idx]
        ax_bc.hist(list(bc.values()), bins=15, color='#457b9d', edgecolor='black', linewidth=1.6, alpha=0.9)
        ax_bc.set_ylabel("Frequency (Node Count)", fontweight='bold', fontsize=20)
        ax_bc.set_xlabel("Betweenness Centrality", fontweight='bold', fontsize=20)
        ax_bc.tick_params(axis='x', labelsize=17)
        ax_bc.tick_params(axis='y', labelsize=17)
        ax_bc.grid(axis='y', linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.0)

        # Row 8: Param Distribution
        ax_p = axes[7, col_idx]
        params = [topology["alpha_dynamic"], topology["beta_dynamic"], topology["gamma_dynamic"]]
        bp_p = ax_p.boxplot(
            params, 
            tick_labels=["Alpha", "Beta", "Gamma"], 
            patch_artist=True, 
            widths=0.55,
            boxprops=dict(linewidth=1.8, color='black'),
            whiskerprops=dict(linewidth=1.8, color='black'),
            capprops=dict(linewidth=1.8, color='black'),
            medianprops=dict(linewidth=2.6, color='black'),
            flierprops=dict(marker='o', markersize=5, markerfacecolor='#555555', markeredgecolor='black', markeredgewidth=1.0, alpha=0.85)
        )
        param_colors = ["#f4a261", "#e9c46a", "#e76f51"]
        param_hatches = ["///", "\\\\\\", "xxx"]
        for j, patch in enumerate(bp_p['boxes']):
            patch.set(facecolor=param_colors[j], edgecolor='black', linewidth=1.8)
            patch.set_hatch(param_hatches[j])
        ax_p.set_ylabel("Weight Value", fontweight='bold', fontsize=20)
        ax_p.tick_params(axis='x', labelsize=17)
        ax_p.tick_params(axis='y', labelsize=17)
        ax_p.grid(axis='y', linestyle='--', alpha=0.45, color='#c0c0c0', linewidth=1.0)

    print("Saving final_paper_grid.pdf (LaTeX vector)...")
    fig.savefig("final_paper_grid.pdf", bbox_inches='tight', pad_inches=0.1)
    print("Saving final_paper_grid.svg (Vector)...")
    fig.savefig("final_paper_grid.svg", bbox_inches='tight', pad_inches=0.1)
    print("Saving final_paper_grid_hd.png (High-Res Raster)...")
    fig.savefig("final_paper_grid_hd.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print("Success: Generated PDF, SVG, and HD PNG for full grid.")


if __name__ == "__main__":
    city_datasets = {}
    for place, label in CITIES:
        data = run_full_analysis(place, label)
        city_datasets[label] = data
    
    create_unified_vector_and_hd_grid(city_datasets)
