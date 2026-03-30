import numpy as np
import networkx as nx
import pandas as pd
from collections import deque

# --- 1. NETWORK TOPOLOGY SETUP ---
GRID_SIZE = 3
G = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE).to_directed()
mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)

# --- 2. PARAMETER CALCULATION LOGIC ---
# This is where the 'magic' of the research happens.
# We derive weights from the graph structure itself.

def calculate_dynamic_parameters(graph):
    # A. Betweenness Centrality (How much of a 'bridge' is this node?)
    centrality = nx.betweenness_centrality(graph)
    max_c = max(centrality.values()) if max(centrality.values()) > 0 else 1.0
    
    # B. In-Degree (How many roads feed into this node?)
    in_degrees = dict(graph.in_degree())
    max_in = max(in_degrees.values()) if max(in_degrees.values()) > 0 else 1.0
    
    node_data = []
    for node in graph.nodes():
        # Normalized metrics
        norm_importance = centrality[node] / max_c
        norm_degree = in_degrees[node] / max_in
        
        # Formulas from the project's 'main.py'
        alpha = 0.5 + 0.5 * norm_degree
        beta  = 0.5 * (1.0 - norm_importance)
        gamma = 0.2 + 0.8 * norm_importance
        
        node_data.append({
            "Node": node,
            "In-Degree": in_degrees[node],
            "Importance (BC)": round(norm_importance, 3),
            "Alpha (Queue)": round(alpha, 3),
            "Beta (Wait)": round(beta, 3),
            "Gamma (Network)": round(gamma, 3)
        })
    return pd.DataFrame(node_data)

# --- 3. SIMULATION RUN ---
def run_dynamic_sim(graph, param_df):
    steps = 30
    edge_queues = {edge: deque() for edge in graph.edges()}
    queue_counts = {edge: 0 for edge in graph.edges()}
    throughput = 0

    for step in range(steps):
        # Inject random traffic
        if np.random.rand() > 0.3:
            u, v = np.random.choice(list(graph.nodes()), 2, replace=False)
            try:
                path = nx.shortest_path(graph, u, v)
                if len(path) > 1:
                    e = (path[0], path[1])
                    edge_queues[e].append(step)
                    queue_counts[e] += 1
            except: pass

        # Process intersections
        for node in graph.nodes():
            incoming = [e for e in graph.edges() if e[1] == node]
            if not incoming or not any(queue_counts[e] > 0 for e in incoming):
                continue
            
            # Get this node's calculated parameters
            p = param_df.iloc[node]
            
            # Find best edge using Dynamic WTM formula:
            # Score = (Alpha * Queue) + (Beta * Wait) + (Gamma * Source Importance)
            best_edge = None
            max_score = -1
            
            for e in incoming:
                q = queue_counts[e]
                w = sum(step - start for start in edge_queues[e])
                src_imp = param_df.iloc[e[0]]["Importance (BC)"]
                
                score = (p["Alpha (Queue)"] * q) + \
                        (p["Beta (Wait)"] * w) + \
                        (p["Gamma (Network)"] * src_imp)
                
                if score > max_score:
                    max_score = score
                    best_edge = e
            
            # Move 1 vehicle
            if best_edge:
                edge_queues[best_edge].popleft()
                queue_counts[best_edge] -= 1
                throughput += 1

    return throughput

# --- 4. OUTPUT RESULTS ---
if __name__ == "__main__":
    print("Step 1: Calculating Dynamic Parameters for 3x3 Grid Intersections")
    print("-" * 70)
    df = calculate_dynamic_parameters(G)
    print(df.to_string(index=False))
    
    print("\nStep 2: Running Simulation using these Parameters...")
    tp = run_dynamic_sim(G, df)
    print(f"Total Throughput: {tp} vehicles served in 30 steps.")
    
    print("\nMathematical Reasoning:")
    print("1. Center Node (Node 4) has highest Importance (BC) -> Highest Gamma.")
    print("   This prioritizes the network's 'heart' to prevent gridlock.")
    print("2. Corner Nodes (0, 2, 6, 8) have lower In-Degree -> Lower Alpha.")
    print("   They are less sensitive to queue pressure and more to wait times.")
    print("3. Side nodes have higher Beta than center nodes.")
    print("   This ensures that cars on the outskirts aren't forgotten by the busy center.")
