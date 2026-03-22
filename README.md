# Urban Traffic Signal Optimization using Complex Networks

This project models and simulates urban traffic flow to evaluate network-aware traffic signal control algorithms against standard traditional approaches. Using real-world road networks from OpenStreetMap, the simulation compares controllers based on their ability to minimize vehicle wait times, reduce overall congestion, and improve network throughput.

## Project Architecture & Workflow

The architecture is built entirely in Python using a discrete-time simulation approach, heavily relying on Graph Theory. 

### 1. Data Ingestion & Graph Construction
The script downloads real-world driving networks (e.g., Nancy, France; Madrid, Spain; Manhattan, USA) using **OSMnx** and constructs a directed graph via **NetworkX**. It parses road attributes—such as maximum speed, length, and lane count—to determine realistic travel times and intersection capabilities (capacity per signal cycle).

### 2. Complex Network Analysis
A crucial part of the architecture is calculating the **Betweenness Centrality** for each intersection (node). Betweenness Centrality measures how often a node acts as a bridge along the shortest path between two other nodes. In an urban context, nodes with high betweenness centrality are structural bottlenecks.

### 3. Traffic Demand Generation
The simulator synthesizes realistic traffic demand by generating Origin-Destination (OD) pairs across the network. A schedule of vehicle arrivals is generated using a Poisson distribution (default arrival rate = 18 vehicles/min), replicating the stochastic nature of real-time urban traffic.

### 4. Discrete-Time Simulation Engine
Vehicles traverse the graph, waiting in edge queues at valid intersections. During each simulation step, an intersection "controller" allocates green light time, attempting to clear the queue by moving vehicles to their next downstream edge.

---

## Comparing the Controllers

We compare three distinct methodologies to solve the inefficiencies caused by static and purely localized traffic light scheduling:

### 1. Fixed Controller (Baseline)
Rigidly splits a fixed cycle time equally among all incoming roads. It has zero adaptability, causing massive inefficiencies when traffic loads are unbalanced.

### 2. Backpressure Controller (Local Adaptive)
Focuses entirely on *immediate* local conditions—calculating a "score" based on the length of incoming queues. It is effective for local congestion but "blind" to the overall city topology.

### 3. WTM (Waiting Time Minimization) Controller [Proposed]
The WTM controller introduces a hybrid, network-aware algorithm that accounts for local traffic volume, vehicle wait times, and global structural importance. 

**The Scoring Function:**
The controller determines the priority of an incoming road using a weighted equation:
`Priority = (ALPHA * Queue Length) + (BETA * Cumulative Wait Time) + (GAMMA * Betweenness Centrality)`

*   **Queue Length (ALPHA):** Weight assigned to the number of vehicles currently waiting.
*   **Wait Time (BETA):** Weight assigned to the total time vehicles have spent waiting at the intersection.
*   **Network Importance (GAMMA):** Weight assigned to the structural criticality of the intersection (Betweenness Centrality).

**Dynamic Green Light Duration:**
WTM dynamically scales the green time based on the priority score:
`Green Time = MIN_GREEN + [(MAX_GREEN - MIN_GREEN) * Priority]`

---

## City Performance Analysis

The model has been tested across various urban topologies. WTM consistently outperforms traditional models in **radial-concentric** cities where structural bottlenecks are prominent.

| City | Topology Type | WTM Performance vs Baselines | Best Parameters Found |
| :--- | :--- | :--- | :--- |
| **Nancy, France** | Radial / European | **Best Results**: +23.8% wait reduction vs Fixed. | $\alpha=0.75, \beta=0.0, \gamma=1.0$ |
| **Madrid, Spain** | Radial-Concentric | **Strong**: Outperforms Backpressure consistently. | $\alpha=0.75, \beta=0.25, \gamma=0.1$ |
| **Milan, Italy** | Concentric Rings | **Strong**: Efficient throughput in ring-road hubs. | $\alpha=0.5, \beta=0.0, \gamma=0.9$ |
| **Zaragoza, Spain** | Radial | **Strong**: Best queue minimization. | $\alpha=0.5, \beta=0.25, \gamma=0.0$ |
| **Manhattan, USA** | Rigid Grid | **Moderate**: Performance close to Backpressure. | $\alpha=1.0, \beta=0.0, \gamma=0.0$ |
| **Chandigarh, India** | Planned Grid | **Moderate**: Structural importance (Gamma) is less critical. | $\alpha=1.0, \beta=0.0, \gamma=0.2$ |
| **Moscow, Russia** | Massive Radial-Ring | **Complex**: High scale requires high Alpha. | $\alpha=1.0, \beta=0.25, \gamma=0.2$ |

### Key Research Findings:
1.  **Topology Matters**: WTM provides the highest gains in older, European-style radial cities where a few intersections (high Betweenness) "control" the flow of the entire city.
2.  **Grid Resilience**: In grid cities like Manhattan, traffic is more distributed, meaning local queue length (`alpha`) is usually sufficient, and the topological bonus (`gamma`) is less impactful.
3.  **Wait Time Weighting**: In highly congested scenarios, adding a small amount of `beta` (Wait Time) helps prevent "starvation" of minor roads.

---

## Technologies Used
* **Python 3**
* **OSMnx:** For downloading and mapping OpenStreetMap street geometries.
* **NetworkX:** For computing shortest paths and betweenness centrality.
* **NumPy / Pandas:** For simulation mathematics and metrics aggregation.
* **Matplotlib:** For visualizing simulation results and generating research plots.
* **Folium:** For generating interactive HTML heatmaps of city congestion.

## Usage
1.  **Install dependencies**: `pip install -r requirements.txt`
2.  **Configure**: Set `PLACE` in `main.py` (e.g., `"Madrid, Spain"`).
3.  **Run**: `python main.py`
4.  **Outputs**: 
    *   `plot1-6.png`: Research visualizations.
    *   `plot5_heatmap_wtm.html`: Interactive network congestion map.
    *   `results_summary_[City].txt`: Detailed controller comparison.
    *   `top_10_params_[City].csv`: Optimal parameter configurations.

---

## Future Direction: Dynamic Topology Scaling
The next phase of this project involves replacing global constants ($\alpha, \beta, \gamma$) with **Node-Specific Dynamic Weights**. By deriving weights directly from the local In-Degree (Hub-ness) and Betweenness (Bridge-ness) of each specific intersection, the network will become "Self-Adapting," removing the need for manual parameter tuning.
