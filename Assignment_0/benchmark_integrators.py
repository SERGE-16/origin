"""Compare explicit_euler vs. rk4 on speed, two ways:

1. At the same (shared) timestep -- isolates the pure per-step cost
   difference (RK4 does 4 dynamics evaluations per step vs. Euler's 1).
2. At each integrator's own largest numerically-accurate timestep (from
   find_max_timestep.py) -- isolates the cost of actually getting an
   accurate answer, which is what you care about in practice.
"""

import timeit

import numpy as np

from models import pendulum as model
from integrators import explicit_euler, rk4
from simulate import run_simulation
from find_max_timestep import find_largest_accurate_timestep

INTEGRATORS = {"explicit_euler": explicit_euler, "rk4": rk4}

params = model.generate_params()
params["damping_coeff"] = 0.0  # undamped: energy should be exactly conserved
initial_state = np.array([np.pi / 4, 0.0])
sim_time = 5.0
tolerance = 1e-2  # 1% relative energy drift allowed
n_repeats = 3  # timeit repeats per case

# --- Step 1: find each integrator's largest numerically-accurate timestep ---
print(f"Finding largest accurate timestep per integrator (tolerance={tolerance:.0%}, sim_time={sim_time}s)...")
max_dt = {}
for name, integrator in INTEGRATORS.items():
    max_dt[name] = find_largest_accurate_timestep(
        model, integrator, params, initial_state, sim_time, tolerance=tolerance
    )
    print(f"  {name:>14s}: {max_dt[name]:.3e} s")

# --- Step 2: timeit comparison at a SHARED timestep ---
# Use the stricter (smaller) of the two, so both integrators are actually
# numerically valid at the timestep being timed.
shared_dt = min(max_dt.values())
print(f"\nSame timestep for both (dt = {shared_dt:.3e} s, the stricter of the two):")
for name, integrator in INTEGRATORS.items():
    elapsed = timeit.timeit(
        lambda integrator=integrator: run_simulation(
            model, integrator, params, initial_state, shared_dt, sim_time
        ),
        number=n_repeats,
    )
    print(f"  {name:>14s}: {elapsed / n_repeats:.4f} s/run")

# --- Step 3: timeit comparison at each integrator's OWN largest accurate timestep ---
print("\nEach integrator at its own largest accurate timestep:")
for name, integrator in INTEGRATORS.items():
    dt = max_dt[name]
    elapsed = timeit.timeit(
        lambda integrator=integrator, dt=dt: run_simulation(
            model, integrator, params, initial_state, dt, sim_time
        ),
        number=n_repeats,
    )
    print(f"  {name:>14s} (dt={dt:.3e}): {elapsed / n_repeats:.4f} s/run")