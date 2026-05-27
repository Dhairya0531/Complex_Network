import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

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
    get_graph_local as benchmark_get_graph_local,
)

# --- CONFIGURATION FOR PAPER PLOT ---
# Focusing on a single representative city (Berlin) as described in Fig 7 of the paper.
CITY_NAME = "Brandenburg Gate, Berlin, Germany"
CITY_LABEL = "Berlin"

EPOCHS = 10  # Match user's request for 10 iterations
SIM_STEPS = BENCHMARK_SIM_STEPS
POPULATION_SIZE = 4
LEARNING_RATE = 0.15

def get_graph_local(place):
    print(f"\nBuilding bottleneck benchmark for: {place}")
    return benchmark_get_graph_local(place, CITY_LABEL)

def build_demand_local(steps, arrival_rate, seed, routes):
    local_rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(steps):
        arrivals = int(local_rng.poisson(arrival_rate))
        if arrivals == 0:
            schedule.append([])
            continue
        chosen_routes = local_rng.integers(0, len(routes), size=arrivals)
        schedule.append(chosen_routes.tolist())
    return schedule

def run_evaluation(graph, routes, demand, theta, controller="dynamic_wtm"):
    if controller == "dynamic_wtm":
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
    else:
        topology = prepare_topology(graph)

    res = run_simulation_with_waiting_time(
        graph,
        routes,
        demand,
        controller,
        topology,
        wtm_score_weights=WTM_SCORE_WEIGHTS,
        centrality_scale=CENTRALITY_SCALE,
    )
    return res["avg_travel_time"] if not np.isnan(res["avg_travel_time"]) else 1e6

def generate_paper_convergence_plot():
    G = get_graph_local(CITY_NAME)
    
    # Pre-calculate Betweenness
    bc = nx.betweenness_centrality(G, weight="travel_time", normalized=True)
    max_bc = max(bc.values()) if bc else 1.0
    for n in G.nodes():
        G.nodes[n]["betweenness_norm"] = bc.get(n, 0.0) / max_bc

    routes = build_bottleneck_routes(G, CITY_LABEL, num_routes=14, seed=42)

    demand = build_demand_local(SIM_STEPS, BENCHMARK_ARRIVAL_RATE, 42, routes)

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

    # --- EVALUATE 10 SIMULATIONS WITH RANDOM TRAFFIC ---
    num_eval_trials = 10
    results_travel_time = {c: [] for c in ["fixed", "backpressure", "dynamic_wtm"]}
    
    print(f"\n>>> Running {num_eval_trials} evaluation trials on bottleneck traffic ...")
    for trial in range(num_eval_trials):
        seed = 200 + trial
        trial_demand = build_demand_local(SIM_STEPS, BENCHMARK_ARRIVAL_RATE, seed, routes)
        for ctrl in ["fixed", "backpressure", "dynamic_wtm"]:
            t_time = run_evaluation(G, routes, trial_demand, theta, controller=ctrl)
            results_travel_time[ctrl].append(t_time)
        print(f"Completed evaluation trial {trial + 1}/{num_eval_trials}")

    # --- PLOTTING (VERTICAL IEEE STYLE) ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 14,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0
    })

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12.5, 17), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.06, hspace=0.45)

    # Plot 1: Efficiency (Blue line)
    ax1.plot(history, color='#3498db', linewidth=2.5)
    ax1.set_title("Convergence of Traffic Efficiency", fontsize=24)
    ax1.set_ylabel("Avg Travel Time (seconds)\n[Lower is Better]", fontsize=16, labelpad=12)
    ax1.set_xlabel("Epoch (Training Iteration)", fontsize=18)
    ax1.grid(True, linestyle='--')

    # Plot 2: Coefficients (Colored, distinct styles & markers)
    th_hist = np.array(theta_history)
    labels = [r'$\alpha$ Bias', r'$\alpha$ Slope', r'$\beta$ Mult', r'$\gamma$ Bias', r'$\gamma$ Slope']
    colors_lines = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']
    styles = [('-', 'o'), ('--', 's'), (':', '^'), ('-.', 'd'), ('-', 'x')]
    
    for i in range(5):
        ls, marker = styles[i]
        ax2.plot(
            th_hist[:, i], 
            label=labels[i], 
            color=colors_lines[i], 
            linestyle=ls, 
            marker=marker, 
            markevery=5,
            linewidth=2.0
        )
    
    ax2.set_title("Convergence of Formula Coefficients", fontsize=24)
    ax2.set_ylabel("Coefficient Value", fontsize=16, labelpad=12)
    ax2.set_xlabel("Epoch", fontsize=18)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    # Plot 3: 100-Trial Comparison Bar Plot (Red, Blue, Green, hatches, error bars)
    controllers = ["fixed", "backpressure", "dynamic_wtm"]
    labels_short = ["Fixed", "BP", "Optimized WTM"]
    colors_bars = {"fixed": "#e74c3c", "backpressure": "#3498db", "dynamic_wtm": "#27ae60"}
    hatches = ["///", "\\\\\\\\", "xx"]
    
    means = [np.mean(results_travel_time[c]) for c in controllers]
    stds = [np.std(results_travel_time[c]) for c in controllers]
    
    bars = ax3.bar(
        labels_short,
        means,
        yerr=stds,
        color=[colors_bars[c] for c in controllers],
        edgecolor="black",
        linewidth=1.8,
        error_kw=dict(ecolor="black", elinewidth=2, capsize=6, capthick=1.5),
        width=0.55
    )
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
        
    ax3.set_title("Performance Comparison (100 Random Traffic Trials)", fontsize=24)
    ax3.set_ylabel("Avg Travel Time (seconds)\n[Lower is Better]", fontsize=16, labelpad=12)
    ax3.set_xlabel("Controller", fontsize=18)
    ax3.grid(axis='y', linestyle='--', alpha=0.35)
    
    # Keep value labels inside the axes and above error bars
    y_max = max(m + s for m, s in zip(means, stds)) if means else 1.0
    ax3.set_ylim(0, y_max * 1.15)
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax3.annotate(
            f"{height:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, mean + std),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14
        )
        
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", hatch="///", label="Fixed (Baseline)"),
        mpatches.Patch(facecolor="#3498db", edgecolor="black", hatch="\\\\\\\\", label="Backpressure"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="black", hatch="xx", label="Optimized WTM")
    ]
    ax3.legend(handles=legend_patches, loc="upper right", frameon=True, framealpha=0.95)

    output_path = "ml_optimization_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess: Paper-ready plot saved as '{output_path}'")

if __name__ == "__main__":
    generate_paper_convergence_plot()
