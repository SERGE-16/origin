"""Shared fixed-timestep simulation loop.

Every model in ``models/`` exposes ``dynamics(t, state, params)`` and
every integrator in ``integrators/`` exposes ``step(dynamics, t, state,
timestep, params)``. ``run_simulation`` just wires those two conventions
together in one place so every script (the energy sanity checks, the
timestep/timing benchmark, the bouncing-ball checks, ...) shares the
exact same loop instead of re-implementing it.
"""

import numpy as np


def run_simulation(model, integrator, params, initial_state, timestep, sim_time):
    """Simulate ``model`` forward from ``initial_state`` for ``sim_time`` seconds.

    Parameters
    ----------
    model : module
        Exposes ``dynamics(t, state, params) -> state_derivative``.
    integrator : module
        Exposes ``step(dynamics, t, state, timestep, params) -> next_state``.
    params : dict
        Forwarded to ``model.dynamics`` at every step.
    initial_state : np.ndarray, shape (n,)
    timestep : float
        Fixed step size (s).
    sim_time : float
        Total simulated duration (s).

    Returns
    -------
    time_traj : np.ndarray, shape (n_timesteps,)
    state_traj : np.ndarray, shape (n, n_timesteps)
    """
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((initial_state.shape[0], n_timesteps))
    state_traj[:, 0] = initial_state

    for step_idx, t in enumerate(time_traj[:-1]):
        state_traj[:, step_idx + 1] = integrator.step(
            model.dynamics, t, state_traj[:, step_idx], timestep, params
        )

    return time_traj, state_traj
