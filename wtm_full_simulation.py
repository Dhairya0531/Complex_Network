import numpy as np
import networkx as nx
import pandas as pd
from collections import deque

# --- 1. CONFIGURATION ---
GRID_SIZE = 3
SIM_STEPS = 10     # Running for 10 steps to keep the log readable
ARRIVAL_RATE = 4   # Vehicles per step
CYCLE_TIME = 60
MIN_GREEN = 15
MAX_GREEN = 45
EDGE_CAPACITY = 12

# --- 2. TOPOLOGY & PARAMETER CALCULATION ---
def setup_network():
    G = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE).to_directed()
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Calculate Node-Specific Parameters
    centrality = nx.betweenness_centrality(G)
    max_c = max(centrality.values()) if max(centrality.values()) > 0 else 1.0
    in_degrees = dict(G.in_degree())
    max_in = max(in_degrees.values()) if max(in_degrees.values()) > 0 else 1.0
    
    node_params = {}
    for node in G.nodes():
        norm_c = centrality[node] / max_c
        norm_d = in_degrees[node] / max_in
        
        # Formulas from the research
        alpha = 0.5 + 0.5 * norm_d
        beta  = 0.5 * (1.0 - norm_c)
        gamma = 0.2 + 0.8 * norm_c
        
        node_params[node] = {
            "alpha": alpha, "beta": beta, "gamma": gamma, "centrality": norm_c
        }
    return G, node_params

# --- 3. SIMULATION ENGINE ---
def run_simulation():
    G, node_params = setup_network()
    edges = list(G.edges())
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    num_nodes = G.number_of_nodes()
    
    # State: queues[edge_idx] = deque([arrival_step, ...])
    queues = [deque() for _ in range(len(edges))]
    completed_vehicles = 0
    total_wait_time = 0

    print("--- STEP 1: TOPOLOGY ANALYSIS (ALPHA, BETA, GAMMA) ---")
    param_df = pd.DataFrame.from_dict(node_params, orient='index')
    param_df.index.name = 'Node'
    print(param_df.round(3))
    print("\n--- STEP 2: STARTING SIMULATION ---")

    for step in range(1, SIM_STEPS + 1):
        print(f"\n>>> TIME STEP {step} <<<")
        
        # A. Arrivals
        new_arrivals = np.random.poisson(ARRIVAL_RATE)
        for _ in range(new_arrivals):
            u, v = np.random.choice(num_nodes, 2, replace=False)
            try:
                path = nx.shortest_path(G, u, v)
                if len(path) > 1:
                    e_idx = edge_to_idx[(path[0], path[1])]
                    queues[e_idx].append(step)
            except nx.NetworkXNoPath: pass
        
        # B. Intersection Logic (Dynamic WTM)
        step_transfers = []
        
        # We process nodes with incoming traffic
        for node in range(num_nodes):
            incoming_edges = [i for i, e in enumerate(edges) if e[1] == node]
            # Only process if there is traffic waiting
            if not any(len(queues[i]) > 0 for i in incoming_edges):
                continue
            
            p = node_params[node]
            best_e_idx = -1
            max_score = -1
            best_wait_sum = 0
            
            # Find which incoming road gets the green light
            for e_idx in incoming_edges:
                q_len = len(queues[e_idx])
                wait_sum = sum(step - arr for arr in queues[e_idx])
                src_node = edges[e_idx][0]
                src_imp = node_params[src_node]['centrality']
                
                # Multiplicative Situational Priority Formula
                score = (p['alpha'] * (q_len / EDGE_CAPACITY) + 
                         p['beta']  * (wait_sum / (EDGE_CAPACITY * CYCLE_TIME))) * \
                        (1.0 + p['gamma'] * src_imp)
                
                if score > max_score:
                    max_score = score
                    best_e_idx = e_idx
                    best_wait_sum = wait_sum
            
            # Calculate Dynamic Green Time for selected edge
            priority = np.clip(max_score, 0, 1.0)
            green_duration = MIN_GREEN + (MAX_GREEN - MIN_GREEN) * priority
            
            # Vehicles that can move in this cycle
            move_cap = max(1, int(EDGE_CAPACITY * (green_duration / CYCLE_TIME)))
            num_to_move = min(len(queues[best_e_idx]), move_cap)
            
            print(f"Intersection {node}: Selected Road {edges[best_e_idx]} | "
                  f"Priority: {priority:.2f} | Green: {green_duration:.1f}s | Moving: {num_to_move}")
            
            # Perform movement
            for _ in range(num_to_move):
                arr_step = queues[best_e_idx].popleft()
                # Simplified: Vehicles reach destination immediately for log clarity
                completed_vehicles += 1
                total_wait_time += (step - arr_step + 1)

    print("\n--- STEP 3: FINAL RESULTS ---")
    print(f"Total Throughput: {completed_vehicles} vehicles")
    print(f"Average Wait Time: {total_wait_time/completed_vehicles:.2f} steps" if completed_vehicles > 0 else "N/A")

if __name__ == "__main__":
    run_simulation()
