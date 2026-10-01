"""Classic fourth-order Runge-Kutta (RK4) integrator.

Same signature as ``integrators.explicit_euler`` so it drops into any
script that already does::

    from integrators import explicit_euler as integrator

by changing only that import line to::

    from integrators import rk4 as integrator

RK4 evaluates the dynamics four times per step (at the start, twice at
the midpoint, and once at the end) and combines them into a weighted
average slope. It is fourth-order accurate: local error O(timestep**5),
global error O(timestep**4). That buys a much larger stable/accurate
timestep than Euler at the cost of 4x the dynamics evaluations per step
-- the timestep-vs-cost tradeoff this assignment's benchmark is built to
measure.

Reference: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
"""

import numpy as np


def step(dynamics, t, state, timestep, params):
    """Advance ``state`` by one classic RK4 step.

    Parameters
    ----------
    dynamics : callable
        Function with signature ``dynamics(t, state, params) -> state_derivative``,
        e.g. ``models.pendulum.dynamics``. Must return an array the same
        shape as ``state``.
    t : float
        Current simulation time (s).
    state : np.ndarray, shape (n,)
        Current state vector.
    timestep : float
        Step size (s).
    params : dict
        Model parameters, forwarded unchanged to ``dynamics``.

    Returns
    -------
    np.ndarray, shape (n,)
        State at ``t + timestep``.
    """
    half_step = timestep / 2.0

    k1 = dynamics(t, state, params)
    k2 = dynamics(t + half_step, state + half_step * k1, params)
    k3 = dynamics(t + half_step, state + half_step * k2, params)
    k4 = dynamics(t + timestep, state + timestep * k3, params)

    next_state = state + (timestep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state