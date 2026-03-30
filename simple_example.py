import numpy as np
import networkx as nx
import pandas as pd
from collections import deque

# --- 1. CONFIGURATION ---
GRID_SIZE = 3  # 3x3 grid of intersections
SIM_STEPS = 50
ARRIVAL_RATE = 5
CYCLE_TIME = 60
MIN_GREEN = 15
MAX_GREEN = 45

# Weights for WTM Controller
ALPHA = 0.75  # Queue Length weight
BETA = 0.25   # Cumulative Wait Time weight
GAMMA = 0.2   # Network Centrality weight

# --- 2. NETWORK CREATION ---
def create_grid_network(size):
    G = nx.grid_2d_graph(size, size, periodic=False).to_directed()
    # Convert (x,y) nodes to simple integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Add road attributes
    for u, v, data in G.edges(data=True):
        data['capacity_per_cycle'] = 10
        data['travel_time'] = 1.0
        
    # Calculate Betweenness Centrality (Network Importance)
    centrality = nx.betweenness_centrality(G)
    max_c = max(centrality.values())
    for node in G.nodes():
        G.nodes[node]['importance'] = centrality[node] / max_c if max_c > 0 else 0
        
    return G

# --- 3. SIMULATION ENGINE ---
def run_simple_sim(G, controller_type):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    edges = list(G.edges())
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    node_to_incoming = [[] for _ in range(num_nodes)]
    for i, (u, v) in enumerate(edges):
        node_to_incoming[v].append(i)

    # State
    edge_queues = [deque() for _ in range(num_edges)]
    queue_counts = np.zeros(num_edges, dtype=int)
    completed_times = []
    total_injected = 0

    for step in range(SIM_STEPS):
        # A. Arrivals (Poisson)
        arrivals = np.random.poisson(ARRIVAL_RATE)
        for _ in range(arrivals):
            # Random origin-destination
            u, v = np.random.choice(num_nodes, 2, replace=False)
            try:
                path = nx.shortest_path(G, u, v)
                if len(path) > 1:
                    start_edge = edge_to_idx[(path[0], path[1])]
                    # Store (arrival_step, remaining_path_edges)
                    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    edge_queues[start_edge].append((step, path_edges[1:]))
                    queue_counts[start_edge] += 1
                    total_injected += 1
            except nx.NetworkXNoPath:
                continue

        # B. Signal Control & Movement
        transfers = []
        for v in range(num_nodes):
            incoming = node_to_incoming[v]
            if not incoming or not any(queue_counts[i] > 0 for i in incoming):
                continue
            
            # --- CONTROLLER LOGIC ---
            if controller_type == "FIXED":
                selected_idx = incoming[step % len(incoming)]
                move_cap = 5 # Fixed capacity
            
            elif controller_type == "BACKPRESSURE":
                # Purely local: pick longest queue
                selected_idx = incoming[np.argmax(queue_counts[incoming])]
                move_cap = 5
                
            elif controller_type == "WTM":
                # Proposed: Queue + Wait Time + Centrality
                scores = []
                for idx in incoming:
                    q_len = queue_counts[idx]
                    wait_time = sum(step - arr for arr, _ in edge_queues[idx])
                    centrality = G.nodes[edges[idx][0]]['importance']
                    
                    score = (ALPHA * q_len) + (BETA * wait_time) + (GAMMA * centrality)
                    scores.append(score)
                
                selected_idx = incoming[np.argmax(scores)]
                # Dynamic green time logic (simplified)
                move_cap = int(MIN_GREEN + (MAX_GREEN - MIN_GREEN) * (max(scores)/max(1, sum(scores))))

            # Move vehicles
            num_move = min(queue_counts[selected_idx], move_cap)
            for _ in range(num_move):
                arr_step, rem_path = edge_queues[selected_idx].popleft()
                queue_counts[selected_idx] -= 1
                if not rem_path:
                    completed_times.append(step - arr_step + 1)
                else:
                    next_e = edge_to_idx[rem_path[0]]
                    transfers.append((next_e, (arr_step, rem_path[1:])))

        # C. Apply Transfers
        for e_idx, data in transfers:
            edge_queues[e_idx].append(data)
            queue_counts[e_idx] += 1

    return {
        "throughput": len(completed_times),
        "avg_wait": np.mean(completed_times) if completed_times else 0,
        "remaining": sum(queue_counts)
    }

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print(f"Running Traffic Simulation on {GRID_SIZE}x{GRID_SIZE} Grid...")
    G = create_grid_network(GRID_SIZE)
    
    results = []
    for mode in ["FIXED", "BACKPRESSURE", "WTM"]:
        res = run_simple_sim(G, mode)
        results.append({"Mode": mode, **res})
    
    df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(df.to_string(index=False))
    
    print("\nExplanation:")
    print("- FIXED: Signals rotate regardless of traffic.")
    print("- BACKPRESSURE: Signals only care about the longest local queue.")
    print("- WTM (Proposed): Signals consider queue length, how long people have waited,")
    print("  AND how important the intersection is to the whole city network.")
