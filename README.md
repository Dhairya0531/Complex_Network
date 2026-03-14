# Urban Traffic Signal Optimization using Complex Networks

This project models and simulates urban traffic flow to evaluate a novel, network-aware traffic signal control algorithm against standard traditional approaches. Using real-world road networks from OpenStreetMap, the simulation compares controllers based on their ability to minimize vehicle wait times, reduce overall congestion, and improve network throughput.

## Project Architecture & Workflow

The architecture is built entirely in Python using a discrete-time simulation approach, heavily relying on Graph Theory. 

### 1. Data Ingestion & Graph Construction
The script downloads real-world driving networks (e.g., Nancy, France) using **OSMnx** and constructs a directed graph via **NetworkX**. It parses road attributes—such as maximum speed, length, and lane count—to determine realistic travel times and intersection capabilities (capacity per signal cycle).

### 2. Complex Network Analysis
A crucial part of the architecture is calculating the **Betweenness Centrality** for each intersection (node). Betweenness Centrality measures how often a node acts as a bridge along the shortest path between two other nodes. In an urban context, nodes with high betweenness centrality are structural bottlenecks.

### 3. Traffic Demand Generation
The simulator synthesizes realistic traffic demand by generating Origin-Destination (OD) pairs across the network. A schedule of vehicle arrivals is generated using a Poisson distribution, replicating the stochastic nature of real-time urban traffic.

### 4. Discrete-Time Simulation Engine
Vehicles traverse the graph, waiting in edge queues at valid intersections. During each simulation step, an intersection "controller" allocates green light time, attempting to clear the queue by moving vehicles to their next downstream edge.

## Comparing the Controllers

The core objective is to solve the inefficiencies caused by static and purely localized traffic light scheduling. We compare three distinct methodologies:

### 1. Fixed Controller (Baseline)
Typical of most older intersections. It rigidly splits a fixed cycle time equally among all incoming roads. It has zero adaptability to real-time traffic variance, causing massive inefficiencies when off-balance traffic loads occur.

### 2. Backpressure Controller (Local Adaptive)
An adaptive approach that evaluates traffic dynamically. It focuses entirely on *immediate* conditions—calculating a "score" based on the length of incoming queues minus the length of the downstream queues. While this effectively mitigates immediate local congestion, it assigns fixed durations for green lights and is utterly "blind" to the overall topology of the city grid.

### 3. The Proposed Solution (Global + Local Network-Aware)
The proposed controller introduces a hybrid, dynamically scaled algorithm that accounts for **both** local traffic volume and global structural importance. 

**How it works:**
The controller determines the priority of an incoming road using a weighted equation:
`Priority = (ALPHA * Local Vehicle Load) + (BETA * Intersection Betweenness Centrality)`

* **Local Demand (ALPHA - 75%):** How backed up the current intersection is compared to its maximum localized queue.
* **Global Importance (BETA - 25%):** How critical this intersection is to the overall city grid. High-centrality nodes are essentially "arteries" that, if congested, will cascade traffic standstills throughout the city.

**Dynamic Green Light Duration:**
Unlike the Fixed and Backpressure models, the Proposed algorithm dynamically scales to the urgency of the moment.
`Green Time = MIN_GREEN(15s) + [(MAX_GREEN(45s) - MIN_GREEN(15s)) * Priority]`

**Why it performs better:** 
By proactively clearing high-centrality intersections *longer* when localized load increases, the algorithm prevents localized gridlock from mutating into a network-wide traffic jam, radically reducing average queues and improving end-to-end travel throughput.

## Technologies Used
* **Python 3**
* **OSMnx:** For downloading and mapping OpenStreetMap street geometries.
* **NetworkX:** For computing shortest paths, strongly connected components, and betweenness centrality.
* **NumPy / Pandas:** For simulation mathematics, dataset manipulation, and metrics aggregation.
* **Matplotlib:** For visualizing simulation results, queue dynamics, and graph plotting.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the simulation directly via the Jupyter Notebook (`main.ipynb`).
3. Results will map node bottlenecks and output a detailed comparison graph measuring throughput and wait time.
