import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation

# --- 1. CONFIGURATION ---
GRID_SIZE = 3
SIM_STEPS = 50
ARRIVAL_RATE = 3
MIN_GREEN = 15
MAX_GREEN = 45
EDGE_CAPACITY = 10

# --- 2. NETWORK SETUP ---
G = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE).to_directed()
mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)
pos = {i: (i % GRID_SIZE, GRID_SIZE - 1 - (i // GRID_SIZE)) for i in G.nodes()}

# Pre-calculate Weights (Alpha, Beta, Gamma)
centrality = nx.betweenness_centrality(G)
max_c = max(centrality.values()) if max(centrality.values()) > 0 else 1.0
node_params = {n: {"a": 0.8, "b": 0.2, "g": 0.5 + 0.5 * (centrality[n]/max_c)} for n in G.nodes()}

# State
queues = [deque() for _ in range(G.number_of_edges())]
edges = list(G.edges())
edge_to_idx = {e: i for i, e in enumerate(edges)}
history_wait = []
current_green_times = [MIN_GREEN] * G.number_of_nodes()

# --- 3. SIMULATION STEP ---
def update_sim(step):
    global current_green_times
    # A. Arrivals
    new_arrivals = np.random.poisson(ARRIVAL_RATE)
    for _ in range(new_arrivals):
        u, v = np.random.choice(G.number_of_nodes(), 2, replace=False)
        try:
            path = nx.shortest_path(G, u, v)
            if len(path) > 1:
                e_idx = edge_to_idx[(path[0], path[1])]
                queues[e_idx].append(step)
        except: pass

    # B. Processing Nodes
    node_congestion = []
    total_wait = 0
    vehicle_count = 0
    
    for node in range(G.number_of_nodes()):
        incoming = [i for i, e in enumerate(edges) if e[1] == node]
        q_sum = sum(len(queues[i]) for i in incoming)
        node_congestion.append(q_sum)
        
        if q_sum > 0:
            # Simple WTM Logic for Animation
            best_e = incoming[np.argmax([len(queues[i]) for i in incoming])]
            wait_sum = sum(step - arr for arr in queues[best_e])
            
            # Dynamic Green Calculation
            priority = np.clip((len(queues[best_e])/EDGE_CAPACITY + wait_sum/300), 0, 1)
            green = MIN_GREEN + (MAX_GREEN - MIN_GREEN) * priority
            current_green_times[node] = green
            
            # Move vehicles
            move_count = min(len(queues[best_e]), int(5 * (green/30)))
            for _ in range(move_count):
                queues[best_e].popleft()
        else:
            current_green_times[node] = MIN_GREEN

    # Global Metrics
    for q in queues:
        total_wait += sum(step - arr for arr in q)
        vehicle_count += len(q)
    history_wait.append(total_wait / max(1, vehicle_count))
    
    return node_congestion

# --- 4. VISUALIZATION ---
fig = plt.figure(figsize=(15, 6))
ax1 = plt.subplot(131) # Grid Map
ax2 = plt.subplot(132) # Wait Time Graph
ax3 = plt.subplot(133) # Green Time Bars

def animate(i):
    congestion = update_sim(i)
    
    # Panel 1: Grid
    ax1.clear()
    ax1.set_title(f"Traffic Grid (Step {i})\n(Red = Congested)", fontsize=10)
    nx.draw(G, pos, ax=ax1, node_color=congestion, 
            cmap=plt.cm.RdYlGn_r, node_size=800, 
            with_labels=True, font_color="white", font_weight="bold",
            edge_color="gray", arrows=True)
    
    # Panel 2: Wait Time History
    ax2.clear()
    ax2.set_title("Network Wait Time Trend", fontsize=10)
    ax2.plot(history_wait, color="blue", linewidth=2)
    ax2.set_ylabel("Avg Wait Steps")
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Green Light Durations
    ax3.clear()
    ax3.set_title("Active Green Light Timings (sec)", fontsize=10)
    ax3.bar(range(G.number_of_nodes()), current_green_times, color="green", alpha=0.7)
    ax3.set_ylim(0, 50)
    ax3.set_xlabel("Node ID")
    ax3.set_ylabel("Seconds")
    ax3.axhline(MIN_GREEN, color='red', linestyle='--', label="Min")
    ax3.axhline(MAX_GREEN, color='orange', linestyle='--', label="Max")

# Run
print("Starting Visualization... Close the window to stop.")
ani = animation.FuncAnimation(fig, animate, frames=SIM_STEPS, interval=500, repeat=False)
plt.tight_layout()
plt.show()
