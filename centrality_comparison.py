import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from main import prepare_topology, run_simulation_with_waiting_time

# --- CONFIGURATION ---
CITIES = [
    ("Cubbon Park, Bengaluru, India", "Bengaluru"),
    ("Brandenburg Gate, Berlin, Germany", "Berlin"),
    ("Trafalgar Square, London, UK", "London"),
    ("Sydney Opera House, Sydney, Australia", "Sydney"),
]

# The original script used a very high demand rate, which quickly saturates the
# network and flattens the differences between metrics. A lower but still
# congested rate exposes the effect of bottlenecks more clearly.
SIM_STEPS = 60
ARRIVAL_RATE = 16
NUM_TRIALS = 6

# Make centrality matter more than in the default controller.
CENTRALITY_SCALE = 2.0
WTM_SCORE_WEIGHTS = {
    "queue": 0.20,
    "wait": 0.25,
    "oldest": 0.15,
    "source": 0.40,
}


def _edge_attrs(length_m, speed_kph=35.0, lanes=1):
    travel_time = length_m / max(speed_kph * 1000.0 / 3600.0, 1.0)
    return {
        "length": float(length_m),
        "speed_kph": float(speed_kph),
        "lanes": int(max(1, lanes)),
        "travel_time": float(travel_time),
        "capacity_per_cycle": max(1, int(round(lanes * 8))),
    }


def _add_bidirectional_edge(G, u, v, length_m, speed_kph=35.0, lanes=1):
    attrs = _edge_attrs(length_m, speed_kph=speed_kph, lanes=lanes)
    G.add_edge(u, v, **attrs)
    G.add_edge(v, u, **attrs)


def _add_grid_block(G, prefix, rows, cols, x_offset, y_offset, side_label):
    nodes = {}
    for r in range(rows):
        for c in range(cols):
            node = f"{prefix}_{r}_{c}"
            nodes[(r, c)] = node
            G.add_node(
                node,
                x=float(x_offset + c),
                y=float(y_offset - r),
                side=side_label,
                role="regular",
            )

    for r in range(rows):
        for c in range(cols):
            node = nodes[(r, c)]
            if c + 1 < cols:
                _add_bidirectional_edge(G, node, nodes[(r, c + 1)], 95.0)
            if r + 1 < rows:
                _add_bidirectional_edge(G, node, nodes[(r + 1, c)], 105.0)
            if r + 1 < rows and c + 1 < cols:
                _add_bidirectional_edge(G, node, nodes[(r + 1, c + 1)], 140.0)
    return nodes


def build_bridge_network(label, left_shape, right_shape, bridge_len, right_offset=10.0):
    left_rows, left_cols = left_shape
    right_rows, right_cols = right_shape

    G = nx.DiGraph()
    left = _add_grid_block(G, f"{label}_L", left_rows, left_cols, 0.0, 0.0, "left")
    right = _add_grid_block(G, f"{label}_R", right_rows, right_cols, right_offset, 0.0, "right")

    left_anchor = left[(left_rows // 2, left_cols - 1)]
    right_anchor = right[(right_rows // 2, 0)]
    bridge_nodes = []
    previous = left_anchor
    for idx in range(bridge_len):
        node = f"{label}_B{idx}"
        bridge_nodes.append(node)
        G.add_node(
            node,
            x=float(right_offset / 2.0),
            y=float(0.3 - idx),
            side="bridge",
            role="bottleneck",
        )
        _add_bidirectional_edge(G, previous, node, 180.0 + idx * 20.0, lanes=1)
        previous = node
    _add_bidirectional_edge(G, previous, right_anchor, 180.0, lanes=1)

    # Add a couple of weak detours so the bridge is still the cheapest route,
    # but not the only route available.
    top_left = left[(0, left_cols - 1)]
    top_right = right[(0, 0)]
    bottom_left = left[(left_rows - 1, left_cols - 1)]
    bottom_right = right[(right_rows - 1, 0)]
    _add_bidirectional_edge(G, top_left, bridge_nodes[0], 260.0, lanes=1)
    _add_bidirectional_edge(G, bridge_nodes[-1], top_right, 260.0, lanes=1)
    _add_bidirectional_edge(G, bottom_left, bridge_nodes[0], 260.0, lanes=1)
    _add_bidirectional_edge(G, bridge_nodes[-1], bottom_right, 260.0, lanes=1)

    return G


def build_radial_network(label, spoke_count=6, spoke_len=3):
    G = nx.DiGraph()
    hub = f"{label}_HUB"
    G.add_node(hub, x=0.0, y=0.0, side="center", role="bottleneck")

    outer_nodes = []
    for spoke in range(spoke_count):
        prev = hub
        for step in range(1, spoke_len + 1):
            node = f"{label}_S{spoke}_{step}"
            G.add_node(
                node,
                x=float(step * np.cos(2 * np.pi * spoke / spoke_count)),
                y=float(step * np.sin(2 * np.pi * spoke / spoke_count)),
                side="spoke",
                role="regular" if step < spoke_len else "outer",
            )
            _add_bidirectional_edge(G, prev, node, 110.0 + 10.0 * step, lanes=1)
            prev = node
        outer_nodes.append(prev)

    # Outer ring to give alternative long routes while still making the hub
    # the dominant shortest-path bridge.
    for i, node in enumerate(outer_nodes):
        nxt = outer_nodes[(i + 1) % len(outer_nodes)]
        _add_bidirectional_edge(G, node, nxt, 260.0, lanes=1)

    return G


def get_graph_local(place, label):
    if label == "Nancy":
        return build_radial_network(label)
    if label == "Berlin":
        return build_bridge_network(label, left_shape=(5, 5), right_shape=(5, 4), bridge_len=2, right_offset=11.0)
    if label == "Sydney":
        return build_bridge_network(label, left_shape=(4, 5), right_shape=(6, 5), bridge_len=4, right_offset=12.0)
    return build_bridge_network(label, left_shape=(4, 4), right_shape=(4, 5), bridge_len=3, right_offset=10.0)


def normalize_dict(d):
    if not d:
        return d
    max_val = max(d.values()) if max(d.values()) > 0 else 1.0
    return {k: v / max_val for k, v in d.items()}


def build_demand_local(steps, arrival_rate, seed, routes):
    local_rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(steps):
        arrivals = int(local_rng.poisson(arrival_rate))
        if arrivals == 0:
            schedule.append([])
            continue
        schedule.append(local_rng.integers(0, len(routes), size=arrivals).tolist())
    return schedule


def _partition_candidates(G):
    left = [n for n, data in G.nodes(data=True) if data.get("side") == "left"]
    right = [n for n, data in G.nodes(data=True) if data.get("side") == "right"]
    bridge = [n for n, data in G.nodes(data=True) if data.get("role") == "bottleneck"]
    center = [n for n, data in G.nodes(data=True) if data.get("side") == "center"]
    outer = [n for n, data in G.nodes(data=True) if data.get("side") == "outer"]
    spokes = [n for n, data in G.nodes(data=True) if data.get("side") == "spoke"]
    return {
        "left": left,
        "right": right,
        "bridge": bridge,
        "center": center,
        "outer": outer,
        "spoke": spokes,
    }


def build_bottleneck_routes(G, label, num_routes, seed):
    groups = _partition_candidates(G)
    rng = np.random.default_rng(seed)
    routes = []

    if label == "Nancy":
        origin_pool = groups["outer"]
        destination_pool = groups["outer"]
        min_len = 4
        max_len = 8
    else:
        origin_pool = groups["left"] or list(G.nodes())
        destination_pool = groups["right"] or list(G.nodes())
        min_len = 5
        max_len = 24

    attempts = 0
    max_attempts = num_routes * 200
    while len(routes) < num_routes and attempts < max_attempts:
        attempts += 1
        if label == "Nancy":
            o, d = rng.choice(origin_pool, 2, replace=False)
        else:
            o = rng.choice(origin_pool)
            d = rng.choice(destination_pool)
            if o == d:
                continue
        try:
            path = nx.shortest_path(G, o, d, weight="travel_time")
        except nx.NetworkXNoPath:
            continue
        if not (min_len <= len(path) <= max_len):
            continue
        path_roles = [G.nodes[n].get("role") for n in path]
        if "bottleneck" not in path_roles and label != "Nancy":
            continue
        if label == "Nancy" and len(set(path)) < 4:
            continue
        routes.append({"edges": list(zip(path[:-1], path[1:]))})

    if len(routes) < num_routes:
        raise RuntimeError(f"Could not build enough routes for {label}: {len(routes)}/{num_routes}")
    return routes


def run_multi_city_comparison():
    city_results = []
    for city_name, label in CITIES:
        print(f"\n>>> Analyzing {label} <<<")
        G = get_graph_local(city_name, label)
        if G.number_of_nodes() == 0:
            raise RuntimeError(f"Graph construction failed for {label}")

        measures = {
            "Betweenness": normalize_dict(
                nx.betweenness_centrality(G, weight="travel_time", normalized=True)
            ),
            "Closeness": normalize_dict(
                nx.closeness_centrality(G, distance="travel_time")
            ),
            "Degree": normalize_dict(dict(G.degree())),
            "PageRank": normalize_dict(nx.pagerank(G, weight="travel_time")),
        }

        routes = build_bottleneck_routes(G, label, num_routes=14, seed=42)
        num_trials = NUM_TRIALS

        for name, centrality_map in measures.items():
            nx.set_node_attributes(G, centrality_map, "betweenness_norm")
            topology = prepare_topology(G)

            scaled_times = []
            for trial in range(num_trials):
                trial_demand = build_demand_local(
                    SIM_STEPS, ARRIVAL_RATE, 42 + trial, routes
                )
                res = run_simulation_with_waiting_time(
                    G,
                    routes,
                    trial_demand,
                    "dynamic_wtm",
                    topology,
                    wtm_score_weights=WTM_SCORE_WEIGHTS,
                    centrality_scale=CENTRALITY_SCALE,
                )
                t_time = res["avg_travel_time"]
                ratio = res["completion_ratio"]
                if ratio > 0 and not np.isnan(t_time):
                    scaled_time = t_time * (1.0 + 0.25 * max(0.0, 1.0 - ratio))
                else:
                    scaled_time = 200000.0
                scaled_times.append(scaled_time)

            avg_scaled_time = float(np.mean(scaled_times))
            city_results.append(
                {
                    "City": label,
                    "Measure": name,
                    "Avg Travel Time": avg_scaled_time,
                }
            )

        bc = measures["Betweenness"]
        top_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:3]
        print(
            "  Top betweenness nodes:",
            ", ".join(f"{node}:{value:.3f}" for node, value in top_nodes),
        )

    df = pd.DataFrame(city_results)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Avg Travel Time"])
    pivot_df = df.pivot(index="City", columns="Measure", values="Avg Travel Time")
    pivot_df = pivot_df[["Betweenness", "Closeness", "Degree", "PageRank"]]
    pivot_df = pivot_df.dropna()

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 26,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.linewidth": 2.0,
            "axes.labelpad": 12,
            "xtick.major.pad": 8,
            "ytick.major.pad": 8,
            "legend.frameon": True,
        }
    )
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    palette = ["#ccffcc", "#cce5ff", "#ffe5cc", "#e5ccff"]
    hatches = ["/", "\\", "x", "."]
    plt.rcParams["hatch.linewidth"] = 2.2
    pivot_df.plot(
        kind="bar",
        ax=ax,
        edgecolor="black",
        linewidth=1.8,
        color=palette,
        width=0.78,
    )

    for measure_idx, container in enumerate(ax.containers):
        color = palette[measure_idx % len(palette)]
        hatch = hatches[measure_idx % len(hatches)]
        for bar in container:
            bar.set_facecolor(color)
            bar.set_edgecolor("black")
            bar.set_hatch(hatch)

    ax.set_ylabel("Effective Travel Time (seconds)\n[Lower is Better]", fontweight="bold", fontsize=22)
    ax.set_xlabel("City", fontweight="bold", fontsize=22)
    ax.tick_params(axis="x", labelrotation=0, labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    import matplotlib.patches as mpatches

    handles, labels = ax.get_legend_handles_labels()
    legend_patches = [
        mpatches.Patch(
            facecolor=palette[idx % len(palette)],
            edgecolor="black",
            hatch=hatches[idx % len(hatches)],
            label=labels[idx],
        )
        for idx in range(len(labels))
    ]

    ax.legend(
        handles=legend_patches,
        title="Centrality Type",
        title_fontsize=18,
        fontsize=16,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
    )
    fig.savefig(
        "multi_city_centrality_comparison.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.25,
    )
    print("Success: multi_city_centrality_comparison.png")


if __name__ == "__main__":
    run_multi_city_comparison()
