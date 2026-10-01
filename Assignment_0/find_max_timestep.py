"""Bisection search for the largest timestep that keeps a simulation
numerically accurate, judged by energy conservation.

For an undamped, unforced conservative system (damping_coeff = 0), total
mechanical energy is analytically constant. A fixed-step integrator will
let it drift; the drift grows with timestep, and blows up outright past
the integrator's stability limit. We define "numerically accurate" as
"total energy stays within `tolerance` (relative) of its initial value
for the whole run", and bisect on timestep to find the largest one that
still satisfies that.
"""

import numpy as np

from simulate import run_simulation


def relative_energy_drift(model, params, state_traj):
    """Max relative deviation of total energy from its initial value."""
    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy
    if not np.all(np.isfinite(total_energy)):
        return np.inf
    return float(np.max(np.abs(total_energy - total_energy[0])) / np.abs(total_energy[0]))


def is_accurate(model, integrator, params, initial_state, sim_time, timestep, tolerance):
    _, state_traj = run_simulation(model, integrator, params, initial_state, timestep, sim_time)
    return relative_energy_drift(model, params, state_traj) <= tolerance


def find_largest_accurate_timestep(
    model,
    integrator,
    params,
    initial_state,
    sim_time,
    tolerance=1e-2,
    dt_lower=1e-5,
    dt_upper=1.0,
    n_bisections=20,
):
    """Bisect for the largest ``timestep`` in ``[dt_lower, dt_upper]`` for
    which the energy-drift check in ``is_accurate`` passes.

    Assumes accuracy is (roughly) monotonically non-increasing in
    timestep, i.e. once it fails at some dt it keeps failing for larger
    dt. That holds here because larger steps only accumulate more local
    truncation error / push further into instability.
    """
    if not is_accurate(model, integrator, params, initial_state, sim_time, dt_lower, tolerance):
        raise ValueError(
            f"Even the smallest candidate timestep ({dt_lower:.1e} s) is not accurate "
            "enough -- lower dt_lower or loosen tolerance."
        )
    if is_accurate(model, integrator, params, initial_state, sim_time, dt_upper, tolerance):
        # The whole bracket is accurate; dt_upper itself is the best we tested.
        return dt_upper

    lower, upper = dt_lower, dt_upper
    for _ in range(n_bisections):
        mid = 0.5 * (lower + upper)
        if is_accurate(model, integrator, params, initial_state, sim_time, mid, tolerance):
            lower = mid
        else:
            upper = mid
    return lower


if __name__ == "__main__":
    from models import pendulum as model
    from integrators import explicit_euler, rk4

    params = model.generate_params()
    params["damping_coeff"] = 0.0  # undamped: energy should be exactly conserved
    initial_state = np.array([np.pi / 4, 0.0])
    sim_time = 3.0
    tolerance = 1e-2  # 1% relative energy drift allowed

    for name, integrator in [("explicit_euler", explicit_euler), ("rk4", rk4)]:
        max_dt = find_largest_accurate_timestep(
            model, integrator, params, initial_state, sim_time, tolerance=tolerance
        )
        print(f"{name:>14s}: largest accurate timestep = {max_dt:.3e} s")