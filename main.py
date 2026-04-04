#!/usr/bin/env python
# coding: utf-8

import math
import os
import shutil
from collections import defaultdict, deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# --- CONFIGURATION ---
PLACE = "Nancy, France"
NETWORK_TYPE = "drive"
RANDOM_SEED = 42
SIMULATION_STEPS = 90
CYCLE_TIME = 60
MIN_GREEN = 15
MAX_GREEN = 45
ARRIVAL_RATE = 18
NUM_OD_PAIRS = 40
NUM_TRIALS = 20
BETWEENNESS_K = 80

rng = np.random.default_rng(RANDOM_SEED)
ox.settings.use_cache = True
ox.settings.log_console = False

# --- HELPER FUNCTIONS ---


def parse_numeric(value, default):
    if value is None:
        return default
    if isinstance(value, list):
        value = value[0]
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit() or ch == ".")
        return float(digits) if digits else default
    return float(value)


def collapse_multidigraph(multigraph):
    graph = nx.DiGraph()
    for node, data in multigraph.nodes(data=True):
        graph.add_node(node, **data)

    for u, v, data in multigraph.edges(data=True):
        edge_data = dict(data)
        edge_data["length"] = float(edge_data.get("length", 1.0))
        if graph.has_edge(u, v):
            if edge_data["length"] < graph[u][v]["length"]:
                graph[u][v].update(edge_data)
        else:
            graph.add_edge(u, v, **edge_data)
    return graph


# --- DATA INGESTION ---

cache_dir = os.path.expanduser("~/.cache/osmnx")
if os.path.exists(cache_dir):
    print(f"Clearing OSM cache: {cache_dir}")
    shutil.rmtree(cache_dir, ignore_errors=True)

try:
    print(f"Fetching network for: {PLACE}")
    raw_graph = ox.graph_from_place(PLACE, network_type=NETWORK_TYPE, simplify=True)
    print("Graph downloaded successfully")
except Exception as e:
    print(f"Error fetching graph: {e}")
    print("Retrying with alternative settings...")
    try:
        raw_graph = ox.graph_from_place(
            PLACE, network_type=NETWORK_TYPE, simplify=True, retain_all=True
        )
        print("Graph downloaded with retain_all=True")
    except Exception as e2:
        print(f"Retry failed: {e2}")
        raise

largest_component = max(nx.strongly_connected_components(raw_graph), key=len)
raw_graph = raw_graph.subgraph(largest_component).copy()

G = collapse_multidigraph(raw_graph)
G.graph["crs"] = raw_graph.graph.get("crs")

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

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# --- CENTRALITY ANALYSIS ---

k_value = min(BETWEENNESS_K, max(10, G.number_of_nodes() - 1))
betweenness = nx.betweenness_centrality(
    G,
    k=k_value,
    normalized=True,
    weight="travel_time",
    seed=RANDOM_SEED,
    endpoints=False,
)

max_bc = max(betweenness.values()) if betweenness else 1.0
node_importance = {
    node: (value / max_bc if max_bc else 0.0) for node, value in betweenness.items()
}
nx.set_node_attributes(G, node_importance, "betweenness_norm")

candidate_nodes = [
    node for node in G.nodes() if G.in_degree(node) > 0 and G.out_degree(node) > 0
]
print(f"Candidate intersections: {len(candidate_nodes)}")

# --- TRAFFIC DEMAND ---

route_bank = []
route_keys = set()
attempts = 0
max_attempts = NUM_OD_PAIRS * 40

while len(route_bank) < NUM_OD_PAIRS and attempts < max_attempts:
    origin, destination = rng.choice(candidate_nodes, size=2, replace=False)
    attempts += 1
    if (origin, destination) in route_keys:
        continue
    try:
        path = nx.shortest_path(G, origin, destination, weight="travel_time")
    except nx.NetworkXNoPath:
        continue
    if len(path) < 4:
        continue
    route_edges = list(zip(path[:-1], path[1:]))
    route_bank.append(
        {
            "origin": origin,
            "destination": destination,
            "path": path,
            "edges": route_edges,
        }
    )
    route_keys.add((origin, destination))

print(f"Routes prepared: {len(route_bank)}")


def build_demand_schedule(steps, arrival_rate, seed):
    local_rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(steps):
        arrivals = int(local_rng.poisson(arrival_rate))
        if arrivals == 0:
            schedule.append([])
            continue
        chosen_routes = local_rng.integers(0, len(route_bank), size=arrivals)
        schedule.append(chosen_routes.tolist())
    return schedule


demand_schedule = build_demand_schedule(SIMULATION_STEPS, ARRIVAL_RATE, RANDOM_SEED)
total_arrivals = sum(len(step) for step in demand_schedule)
print(f"Vehicles scheduled: {total_arrivals}")

# --- TOPOLOGY PREPARATION ---


def prepare_topology(graph):
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = list(graph.edges())
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}

    num_nodes, num_edges = len(nodes), len(edges)
    edge_target_idx = np.array([node_to_idx[v] for u, v, *k in edges], dtype=int)
    edge_source_idx = np.array([node_to_idx[u] for u, v, *k in edges], dtype=int)
    edge_caps = np.array(
        [float(graph.edges[e].get("capacity_per_cycle", 0.0)) for e in edges],
        dtype=float,
    )

    incoming_idx = [[] for _ in range(num_nodes)]
    outgoing_idx = [[] for _ in range(num_nodes)]
    for i, (src, tgt) in enumerate(zip(edge_source_idx, edge_target_idx)):
        incoming_idx[tgt].append(i)
        outgoing_idx[src].append(i)

    node_in_counts = np.array([len(e) for e in incoming_idx], dtype=float)
    node_in_counts_fixed = np.where(node_in_counts == 0, 1.0, node_in_counts)
    incoming_idx = [np.array(e, dtype=int) for e in incoming_idx]
    outgoing_idx = [np.array(e, dtype=int) for e in outgoing_idx]

    node_importance = np.array(
        [float(graph.nodes[n].get("betweenness_norm", 0.0)) for n in nodes], dtype=float
    )

    # DYNAMIC TOPOLOGY SCALING (Eqs. 7-9 in paper)
    alpha_dynamic = 0.5 + 0.5 * (
        node_in_counts / max(1.0, np.max(node_in_counts_fixed))
    )
    beta_dynamic = 0.5 * (1.0 - node_importance)
    gamma_dynamic = 0.2 + 0.8 * node_importance

    move_cap_fixed = np.maximum(
        1, np.floor(edge_caps / node_in_counts_fixed[edge_target_idx])
    ).astype(int)

    return {
        "nodes": nodes,
        "node_to_idx": node_to_idx,
        "edge_to_idx": edge_to_idx,
        "edges": edges,
        "incoming_idx": incoming_idx,
        "outgoing_idx": outgoing_idx,
        "edge_target_idx": edge_target_idx,
        "edge_source_idx": edge_source_idx,
        "move_cap_fixed": move_cap_fixed,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_importance": node_importance,
        "alpha_dynamic": alpha_dynamic,
        "beta_dynamic": beta_dynamic,
        "gamma_dynamic": gamma_dynamic,
        "edge_caps": edge_caps,
    }


# --- SIMULATION LOGIC ---


def choose_edge_for_node_wtm(
    incoming,
    queue_counts,
    edge_queues,
    step,
    edge_source_idx,
    node_importance,
    target_node_idx,
    topology_data,
):
    a = topology_data["alpha_dynamic"][target_node_idx]
    b = topology_data["beta_dynamic"][target_node_idx]
    g = topology_data["gamma_dynamic"][target_node_idx]
    edge_caps = topology_data["edge_caps"]

    cumulative_wait = np.zeros(len(incoming))
    for i, edge_idx in enumerate(incoming):
        queue = edge_queues[edge_idx]
        if len(queue) > 0:
            # Wait times use arrival_at_edge_step (3rd element in 4-tuple)
            wait_times = [step - arrival_at_edge_step for _, _, arrival_at_edge_step, _ in queue]
            cumulative_wait[i] = np.sum(wait_times)

    # Normalize metrics for selection consistent with timing logic.
    # We remove CYCLE_TIME from wait_pressure because wait_times is in cycles (steps).
    queue_pressure = queue_counts[incoming] / np.maximum(1.0, edge_caps[incoming])
    wait_pressure = cumulative_wait / np.maximum(1.0, edge_caps[incoming])
    source_importance = node_importance[edge_source_idx[incoming]]

    # MULTIPLICATIVE HUB AWARENESS: Hub status acts as a multiplier to traffic signals.
    # This ensures "Empty Hubs" don't steal green time from busy side roads.
    scores = (a * queue_pressure + b * wait_pressure) * (1.0 + g * source_importance)
    best_local_idx = np.argmax(scores)
    selected_idx = incoming[best_local_idx]
    return selected_idx, cumulative_wait[best_local_idx]


def run_simulation_with_waiting_time(
    graph,
    route_bank,
    demand_schedule,
    controller,
    topology=None,
):
    if topology is None:
        topology = prepare_topology(graph)

    node_to_idx = topology["node_to_idx"]
    edge_to_idx = topology["edge_to_idx"]
    incoming_idx = topology["incoming_idx"]
    edge_target_idx = topology["edge_target_idx"]
    edge_source_idx = topology["edge_source_idx"]
    num_nodes = topology["num_nodes"]
    num_edges = topology["num_edges"]
    nodes = topology["nodes"]
    node_importance = topology["node_importance"]
    edge_caps = topology["edge_caps"]
    alpha_dynamic = topology["alpha_dynamic"]
    beta_dynamic = topology["beta_dynamic"]
    gamma_dynamic = topology["gamma_dynamic"]

    outgoing_idx = topology["outgoing_idx"]

    indexed_routes = [[edge_to_idx[e] for e in r["edges"]] for r in route_bank]
    edge_queues = [deque() for _ in range(num_edges)]
    queue_counts = np.zeros(num_edges, dtype=int)
    move_caps = topology["move_cap_fixed"]  # used by fixed & backpressure

    completed_travel_times = []
    completed_wait_times = []
    total_injected = 0
    node_queue_totals = np.zeros(num_nodes, dtype=float)
    node_queue_peaks = np.zeros(num_nodes, dtype=float)
    queue_history = []
    # Per-vehicle cumulative wait tracker: vehicle_id -> total wait steps
    vehicle_wait_accumulator = defaultdict(float)

    for step, arrivals in enumerate(demand_schedule):
        transfers = []
        if np.any(queue_counts > 0):
            active_edges = np.flatnonzero(queue_counts)
            active_nodes = np.unique(edge_target_idx[active_edges])
            for v_idx in active_nodes:
                incoming = incoming_idx[v_idx]
                if controller == "fixed":
                    selected_idx = incoming[step % len(incoming)]
                elif controller == "backpressure":
                    # Simplified max-pressure: routing-agnostic highest incoming queue
                    selected_idx = incoming[np.argmax(queue_counts[incoming])]
                elif controller == "dynamic_wtm":
                    selected_idx, wait_time_sum = choose_edge_for_node_wtm(
                        incoming,
                        queue_counts,
                        edge_queues,
                        step,
                        edge_source_idx,
                        node_importance,
                        v_idx,
                        topology,
                    )

                # --- LIVE-DYNAMIC TIMING ALLOCATION ---
                if controller == "dynamic_wtm":
                    # Extract dynamic parameters for this node
                    a = alpha_dynamic[v_idx]
                    b = beta_dynamic[v_idx]
                    g = gamma_dynamic[v_idx]
                    edge_cap = edge_caps[selected_idx]

                    # Calculate Queue Pressure (Alpha term)
                    queue_pressure = queue_counts[selected_idx] / max(1.0, edge_cap)

                    # Calculate Wait Pressure (Beta term). Corrected scaling (removed CYCLE_TIME)
                    wait_pressure = wait_time_sum / max(1.0, edge_cap)

                    # Calculate Structural Importance (Gamma term)
                    source_node_idx = edge_source_idx[selected_idx]
                    importance_bonus = node_importance[source_node_idx]

                    # MULTIPLICATIVE SITUATIONAL PRIORITY:
                    situational_priority = np.clip(
                        (a * queue_pressure + b * wait_pressure)
                        * (1.0 + g * importance_bonus),
                        0.0,
                        1.0,
                    )

                    # Scale green time between MIN_GREEN and MAX_GREEN
                    dynamic_green = (
                        MIN_GREEN + (MAX_GREEN - MIN_GREEN) * situational_priority
                    )

                    # Convert green duration to vehicle capacity for this step
                    current_move_cap = max(
                        1, int(np.floor(edge_cap * (dynamic_green / CYCLE_TIME)))
                    )
                    move_count = min(queue_counts[selected_idx], current_move_cap)
                else:
                    move_count = min(
                        queue_counts[selected_idx], move_caps[selected_idx]
                    )

                if move_count > 0:
                    q = edge_queues[selected_idx]
                    for _ in range(move_count):
                        # vehicle_id is based on original injection_step
                        r_idx, injection_step, arrival_at_edge_step, pos = q.popleft()
                        vehicle_id = (r_idx, injection_step)
                        # Accumulate wait time at this specific edge
                        vehicle_wait_accumulator[vehicle_id] += max(0, step - arrival_at_edge_step)
                        transfers.append((r_idx, injection_step, pos + 1, step))
                    queue_counts[selected_idx] -= move_count

        for r_idx, injection_step, next_pos, serve_step in transfers:
            route = indexed_routes[r_idx]
            if next_pos >= len(route):
                # Total travel time: current step minus initial injection step
                # serve_step is current step here
                travel_time = serve_step - injection_step
                completed_travel_times.append(travel_time)
                vehicle_id = (r_idx, injection_step)
                completed_wait_times.append(vehicle_wait_accumulator.pop(vehicle_id, 0.0))
            else:
                next_e = route[next_pos]
                # Pass injection_step through and set serve_step as new arrival_at_edge_step
                edge_queues[next_e].append((r_idx, injection_step, serve_step, next_pos))
                queue_counts[next_e] += 1

        for route_id in arrivals:
            start_e = indexed_routes[route_id][0]
            # (r_idx, injection_step, arrival_at_edge_step, pos)
            edge_queues[start_e].append((route_id, step, step, 0))
            queue_counts[start_e] += 1
            total_injected += 1
            vehicle_wait_accumulator[(route_id, step)] = 0.0

        node_queues = np.bincount(
            edge_target_idx, weights=queue_counts.astype(float), minlength=num_nodes
        )
        node_queue_totals += node_queues
        np.maximum(node_queue_peaks, node_queues, out=node_queue_peaks)
        queue_history.append(np.sum(queue_counts))

    return {
        "controller": controller,
        "avg_queue_length": float(np.mean(queue_history)) if queue_history else 0.0,
        "avg_travel_time": float(np.mean(completed_travel_times)) * CYCLE_TIME
        if completed_travel_times
        else np.nan,
        "total_wait_time": float(np.sum(completed_wait_times)) * CYCLE_TIME
        if completed_wait_times
        else 0.0,
        "avg_wait_time": float(np.mean(completed_wait_times)) * CYCLE_TIME
        if completed_wait_times
        else np.nan,
        "throughput": len(completed_travel_times),
        "completion_ratio": len(completed_travel_times) / total_injected
        if total_injected
        else 0.0,
        "queue_history": queue_history,
        "max_wait_time": float(np.max(completed_wait_times)) * CYCLE_TIME
        if completed_wait_times
        else 0.0,
        "avg_node_queue": {
            n: node_queue_totals[i] / len(demand_schedule) for i, n in enumerate(nodes)
        },
        "peak_node_queue": {n: int(node_queue_peaks[i]) for i, n in enumerate(nodes)},
    }


if __name__ == "__main__":
    # --- EVALUATION ---

    print("=" * 80)
    print("SIGNAL SCHEDULING COMPARISON: Live-Dynamic Network Awareness")
    print("=" * 80)

    topology = prepare_topology(G)
    demand_schedule = build_demand_schedule(SIMULATION_STEPS, ARRIVAL_RATE, RANDOM_SEED)

    controllers_to_test = ["fixed", "backpressure", "dynamic_wtm"]
    wtm_results = []

    for controller in controllers_to_test:
        print(f"Testing {controller.upper():15s} ...", end=" ", flush=True)
        result = run_simulation_with_waiting_time(
            G, route_bank, demand_schedule, controller, topology=topology
        )
        wtm_results.append(result)
        print("Done")

    wtm_comparison_df = (
        pd.DataFrame(
            [
                {
                    "Controller": "PROPOSED (Live-Dynamic WTM)"
                    if r["controller"] == "dynamic_wtm"
                    else r["controller"].upper(),
                    "Avg Queue": round(r["avg_queue_length"], 2),
                    "Avg Travel Time (s)": round(r["avg_travel_time"], 2),
                    "Avg Wait Time (s)": round(r["avg_wait_time"], 2),
                    "Max Wait Time (s)": round(r["max_wait_time"], 2),
                    "Total Wait (s)": round(r["total_wait_time"], 0),
                    "Throughput": round(r["throughput"], 0),
                }
                for r in wtm_results
            ]
        )
        .sort_values("Avg Wait Time (s)")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 100)
    print("WAITING TIME COMPARISON: Fixed vs Backpressure vs Proposed (Live-Dynamic WTM)")
    print("=" * 100)
    print(wtm_comparison_df.to_string(index=False))
    print("=" * 100)

    # --- RESEARCH PLOTS ---

    import folium
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    controllers_list = ["fixed", "backpressure", "dynamic_wtm"]
    controller_labels = {
        "fixed": "Fixed (Baseline)",
        "backpressure": "Backpressure",
        "dynamic_wtm": "Proposed (Live-Dynamic WTM)",
    }
    controller_colors = {
        "fixed": "#95a5a6",  # Light Grey
        "backpressure": "#3498db",  # Blue
        "dynamic_wtm": "#27ae60",  # Green
    }
    controller_hatches = {
        "fixed": "//",
        "backpressure": "\\\\\\\\",
        "dynamic_wtm": "xx",
    }
    controller_ls = {"fixed": ":", "backpressure": "--", "dynamic_wtm": "-"}
    mk_map = {"fixed": "s", "backpressure": "^", "dynamic_wtm": "o"}

    per_trial_data = {c: [] for c in controllers_list}
    for trial in range(NUM_TRIALS):
        seed = RANDOM_SEED + trial
        ds = build_demand_schedule(SIMULATION_STEPS, ARRIVAL_RATE, seed)
        for ctrl in controllers_list:
            result = run_simulation_with_waiting_time(
                G, route_bank, ds, ctrl, topology=topology
            )
            per_trial_data[ctrl].append(result)
        if (trial + 1) % 5 == 0:
            print(f"Completed trial {trial + 1}/{NUM_TRIALS}")

    mean_metrics = {}
    for ctrl in controllers_list:
        runs = per_trial_data[ctrl]
        mean_metrics[ctrl] = {
            "avg_queue": np.mean([r["avg_queue_length"] for r in runs]),
            "avg_travel_time": np.nanmean([r["avg_travel_time"] for r in runs]),
            "throughput": np.mean([r["throughput"] for r in runs]),
            "avg_wait_time": np.nanmean([r["avg_wait_time"] for r in runs]),
            "max_wait_time": np.mean([r["max_wait_time"] for r in runs]),
        }

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    plt.rcParams["hatch.linewidth"] = 2.0
    bar_width = 0.55
    labels = [controller_labels[c] for c in controllers_list]
    labels_short = ["Fixed", "BP", "Proposed"]
    colors = [controller_colors[c] for c in controllers_list]
    hatches = [controller_hatches[c] for c in controllers_list]

    def _annotate_bars(ax, bars, fmt="{:.1f}"):
        ymax = max(b.get_height() for b in bars)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                fmt.format(bar.get_height()),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )

    # Plot 1: Avg Queue Length
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    values = [mean_metrics[c]["avg_queue"] for c in controllers_list]
    bars = ax1.bar(
        labels_short,
        values,
        color="white",
        edgecolor=colors,
        linewidth=2.0,
        width=bar_width,
        hatch=hatches,
    )
    _annotate_bars(ax1, bars)
    ax1.set_ylabel("Avg Queue (vehicles)")
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    if clean_vals:
        ax1.set_ylim(0, max(clean_vals) * 1.3)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("plot_1.png", dpi=200)

    # Plot 2: Avg Travel Time
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    values = [mean_metrics[c]["avg_travel_time"] for c in controllers_list]
    bars = ax2.bar(
        labels_short,
        values,
        color="white",
        edgecolor=colors,
        linewidth=2.0,
        width=bar_width,
        hatch=hatches,
    )
    _annotate_bars(ax2, bars)
    ax2.set_ylabel("Avg Travel Time (s)")
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    if clean_vals:
        ax2.set_ylim(0, max(clean_vals) * 1.3)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("plot_2.png", dpi=200)

    # Plot 3: Throughput
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    values = [mean_metrics[c]["throughput"] for c in controllers_list]
    bars = ax3.bar(
        labels_short,
        values,
        color="white",
        edgecolor=colors,
        linewidth=2.0,
        width=bar_width,
        hatch=hatches,
    )
    _annotate_bars(ax3, bars, fmt="{:.0f}")
    ax3.set_ylabel("Throughput (vehicles)")
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    if clean_vals:
        ax3.set_ylim(0, max(clean_vals) * 1.3)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("plot_3.png", dpi=200)

    # Plot 4: Waiting Time Boxplot
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    wait_distributions = [
        [r["avg_wait_time"] for r in per_trial_data[c] if not np.isnan(r["avg_wait_time"])]
        for c in controllers_list
    ]
    bp = ax4.boxplot(
        wait_distributions,
        tick_labels=labels_short,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="gold"),
    )
    for patch, col, h in zip(bp["boxes"], colors, hatches):
        patch.set_facecolor("white")
        patch.set_edgecolor(col)
        patch.set_linewidth(2.0)
        patch.set_hatch(h)
    ax4.set_ylabel("Avg Wait Time (s)")
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("plot_4.png", dpi=200)

    # Performance vs Demand analysis
    demand_levels = {"Low": 8, "Medium": 18, "High": 30}
    demand_results = {c: {"travel_time": [], "throughput": []} for c in controllers_list}
    for lbl, rate in demand_levels.items():
        for ctrl in controllers_list:
            tt_l, tp_l = [], []
            for trial in range(5):
                ds = build_demand_schedule(SIMULATION_STEPS, rate, RANDOM_SEED + trial)
                res = run_simulation_with_waiting_time(G, route_bank, ds, ctrl, topology)
                tt_l.append(res["avg_travel_time"])
                tp_l.append(res["throughput"])
            demand_results[ctrl]["travel_time"].append(np.nanmean(tt_l))
            demand_results[ctrl]["throughput"].append(np.mean(tp_l))

    # Plot 5: Throughput vs Demand
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    x_pos = np.arange(len(demand_levels))
    for ctrl in controllers_list:
        ax5.plot(
            x_pos,
            demand_results[ctrl]["throughput"],
            marker=mk_map[ctrl],
            color=controller_colors[ctrl],
            linestyle=controller_ls[ctrl],
            linewidth=2.0,
            label=labels_short[controllers_list.index(ctrl)],
        )
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(list(demand_levels.keys()))
    ax5.set_ylabel("Throughput")
    ax5.grid(alpha=0.3, linestyle="--")
    ax5.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plot_5.png", dpi=200)

    # Plot 6: Travel Time vs Demand
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    for ctrl in controllers_list:
        ax6.plot(
            x_pos,
            demand_results[ctrl]["travel_time"],
            marker=mk_map[ctrl],
            color=controller_colors[ctrl],
            linestyle=controller_ls[ctrl],
            linewidth=2.0,
            label=labels_short[controllers_list.index(ctrl)],
        )
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(list(demand_levels.keys()))
    ax6.set_ylabel("Avg Travel Time (s)")
    ax6.grid(alpha=0.3, linestyle="--")
    ax6.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plot_6.png", dpi=200)

    # Plot 7: BC Distribution
    fig7, ax7 = plt.subplots(figsize=(6, 5))
    ax7.hist(
        list(betweenness.values()),
        bins=20,
        color="#2c3e50",
        edgecolor="black",
        alpha=0.7,
    )
    ax7.set_ylabel("Frequency")
    ax7.set_xlabel("Betweenness Centrality")
    ax7.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("plot_7.png", dpi=200)

    # Plot 8: Param Dist
    fig8, ax8 = plt.subplots(figsize=(6, 5))
    params = [
        topology["alpha_dynamic"],
        topology["beta_dynamic"],
        topology["gamma_dynamic"],
    ]
    param_colors = ["#f39c12", "#f1c40f", "#c0392b"]
    bp8 = ax8.boxplot(
        params, tick_labels=["Alpha", "Beta", "Gamma"], patch_artist=True
    )
    for i, patch in enumerate(bp8["boxes"]):
        patch.set_facecolor(param_colors[i])
        patch.set_edgecolor("black")
        patch.set_alpha(0.8)
    ax8.set_ylabel("Value")
    ax8.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("plot_8.png", dpi=200)

    # Original plot5 and plot6 for backward compatibility (Heatmaps and Combined)
    # [Rest of original plotting logic for Heatmaps ...]

    print("\n" + "=" * 60)
    print("RESEARCH PLOTS GENERATED")
    print("=" * 60)

    report_filename = f"results_summary_{PLACE.split(',')[0].replace(' ', '_')}.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"TRAFFIC SIMULATION REPORT: {PLACE}\n")
        f.write("=" * 60 + "\n\n")
        f.write("LIVE-DYNAMIC TIMING ALLOCATION IMPLEMENTATION\n")
        f.write("=" * 60 + "\n\n")
        f.write("WAITING TIME COMPARISON:\n")
        f.write(wtm_comparison_df.to_string(index=False))
        f.write("\n" + "=" * 60 + "\n")
        f.write("\nNOTE: The 'Proposed (Dynamic WTM)' controller now uses Live-Dynamic\n")
        f.write("green light timing allocation, where each intersection adjusts its\n")
        f.write("green duration in real-time based on:\n")
        f.write("  - Alpha (α): Current queue pressure relative to capacity\n")
        f.write("  - Beta (β): Cumulative waiting time of vehicles\n")
        f.write("  - Gamma (γ): Structural importance (Betweenness Centrality)\n")

    print(f"\nResults saved to: {report_filename}\n")
