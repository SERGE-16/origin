# prior to looking at the results, I considered .001 output mean and .001% fraction difference from the highest fidelity 
# grid as being the cutoff point, with output mean defined as the average poincare angular velocity
from pathlib import Path
import numpy as np
import pandas as pd
from models import inverted_pendulum_walker as model

# PARAMETERS
params = {
    "gravity": 9.81,                  # m/s^2
    "length": 1.0,                    # m
    "mass": 1.0,                      # kg
    "incline": 0.06,                  # rad
    "angle_of_attack": np.pi / 8,     # rad
    "ankle_torque": 0.0,              # N m
    "damping_coeff": 0.0              # kg*m^2/s
}

# Grid resolutions to test
theta_dot_resolutions = [101, 121, 141, 161, 181, 201, 221] #changed bounds because 101 wasn't good enought
alpha_resolutions = [11] #found no change across alpha search

timestep = 1e-4

# POINCARE SECTION
def poincare_section_event(prev_state, next_state, params):
    theta_prev = prev_state[0]
    theta_next = next_state[0]
    return theta_prev < 0 and theta_next >= 0

# INTERPOLATION
def interpolate_state(previous_state, next_state, target_theta):
    theta_prev = previous_state[0]
    theta_next = next_state[0]

    if theta_next == theta_prev:
        return next_state.copy()

    fraction = (target_theta - theta_prev) / (theta_next - theta_prev)
    return previous_state + fraction * (next_state - previous_state)

# POINCARE MAP
def poincare_map(theta_dot, alpha, params, timestep=1e-4):
    step_params = params.copy()
    step_params["angle_of_attack"] = alpha
    step_params["ankle_torque"] = 0.0

    gamma = step_params["incline"]
    theta_td = gamma + alpha

    # Start on the Poincare section: [theta, theta_dot] = [0, theta_dot]
    state = np.array([0.0, theta_dot])
    max_steps = 20000

    # FIND TOUCHDOWN
    touchdown_found = False

    for _ in range(max_steps):
        next_state = state + timestep * model.dynamics(0, state, step_params)

        if state[0] < theta_td and next_state[0] >= theta_td:
            touchdown_state = interpolate_state(state, next_state, theta_td)
            state = model.event_dynamics(touchdown_state, step_params)
            touchdown_found = True
            break

        state = next_state

    if not touchdown_found:
        return np.nan

    # FIND NEXT POINCARE SECTION
    for _ in range(max_steps):
        next_state = state + timestep * model.dynamics(0, state, step_params)

        if poincare_section_event(state, next_state, step_params):
            section_state = interpolate_state(state, next_state, 0.0)

            # Only accept the forward-moving branch
            if section_state[1] > 0:
                return section_state[1]

            return np.nan

        state = next_state

    return np.nan

# BUILD ONE POINCARE TABLE
def build_poincare_table(theta_dot_grid, alpha_grid):
    table = np.full((len(theta_dot_grid), len(alpha_grid)), np.nan)

    for i, theta_dot in enumerate(theta_dot_grid):
        for j, alpha in enumerate(alpha_grid):
            table[i, j] = poincare_map(theta_dot, alpha, params, timestep=timestep)

    return table

# METRICS FOR EACH GRID
def calculate_metrics(theta_dot_grid, alpha_grid, poincare_table):
    total_entries = poincare_table.size
    valid_entries = np.count_nonzero(np.isfinite(poincare_table))
    invalid_entries = total_entries - valid_entries

    valid_fraction = valid_entries / total_entries

    if valid_entries > 0:
        valid_values = poincare_table[np.isfinite(poincare_table)]

        output_min = np.min(valid_values)
        output_max = np.max(valid_values)
        output_mean = np.mean(valid_values)
    else:
        output_min = np.nan
        output_max = np.nan
        output_mean = np.nan

    return {
        "theta_dot_resolution": len(theta_dot_grid),
        "alpha_resolution": len(alpha_grid),
        "theta_dot_spacing": (theta_dot_grid[1] - theta_dot_grid[0] if len(theta_dot_grid) > 1 else np.nan),
        "alpha_spacing": (alpha_grid[1] - alpha_grid[0] if len(alpha_grid) > 1 else np.nan),
        "total_entries": total_entries,
        "valid_entries": valid_entries,
        "invalid_entries": invalid_entries,
        "valid_fraction": valid_fraction,
        "output_min": output_min,
        "output_max": output_max,
        "output_mean": output_mean,}

# RUN GRID RESOLUTION TEST
results = {}
metric_rows = []

theta_dot_max = np.sqrt(2 * params["gravity"] / params["length"])

for theta_dot_res in theta_dot_resolutions:
    for alpha_res in alpha_resolutions:

        print(f"Testing theta_dot = {theta_dot_res}, alpha = {alpha_res}")

        # Create the grids
        theta_dot_grid = np.linspace(0, theta_dot_max, theta_dot_res)

        alpha_grid = np.linspace(np.pi / 8, np.pi / 7, alpha_res)

        # Build Poincare table
        poincare_table = build_poincare_table(theta_dot_grid, alpha_grid)

        # Store full result in memory
        results[(theta_dot_res, alpha_res)] = {"theta_dot_grid": theta_dot_grid, "alpha_grid": alpha_grid, "poincare_table": poincare_table}

        # Calculate and store key metrics
        metric_rows.append(calculate_metrics(theta_dot_grid, alpha_grid, poincare_table))

print("Grid resolution test complete")

# SAVE FULL RESULTS
npz_path = Path("grid_resolution_results.npz")
np.savez(npz_path, results=np.array(results, dtype=object))

print(f"Full results saved to: {npz_path.resolve()}")

# SAVE KEY METRICS AS CSV
metrics_df = pd.DataFrame(metric_rows)

csv_path = Path("grid_resolution_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
print(f"Key metrics saved to: {csv_path.resolve()}")

# PRINT A SHORT SUMMARY
print("\nKEY METRICS:")
print(metrics_df[["theta_dot_resolution", "alpha_resolution", "theta_dot_spacing", "alpha_spacing", "valid_fraction", "output_min", "output_max", "output_mean", ]].to_string(index=False))
