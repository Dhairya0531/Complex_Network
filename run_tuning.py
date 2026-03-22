import itertools
import numpy as np
from tuning_simulation import prepare_topology, run_simulation_with_waiting_time, G, route_bank, build_demand_schedule, SIMULATION_STEPS, ARRIVAL_RATE, RANDOM_SEED

# Reduced parameter grid to speed up execution
param_grid = {
    "alpha_off": [0.3, 0.5],
    "alpha_mul": [0.3, 0.5],
    "beta_mul": [0.5, 0.7],
    "gamma_off": [0.2],
    "gamma_mul": [0.6, 0.8],
}

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []
for i, params in enumerate(param_combinations):
    print(f"Testing {i+1}/{len(param_combinations)}: {params}")
    topology = prepare_topology(G, params=params)
    ds = build_demand_schedule(SIMULATION_STEPS, ARRIVAL_RATE, RANDOM_SEED)
    res = run_simulation_with_waiting_time(G, route_bank, ds, "dynamic_wtm", topology=topology, params=params)
    results.append({"params": params, "result": res})
    print(f"Throughput: {res['throughput']}, Avg Wait: {res['avg_wait_time']}")

# Find best parameters
# Objective: Throughput max, Wait Time < Backpressure Wait Time (48.37)
backpressure_wait = 48.37
best_results = [r for r in results if r['result']['avg_wait_time'] < backpressure_wait]
best_results.sort(key=lambda x: x['result']['throughput'], reverse=True)

if best_results:
    print(f"Best: {best_results[0]}")
    with open("best_parameters.txt", "w") as f:
        f.write(str(best_results[0]))
else:
    print("No parameters found meeting criteria.")
