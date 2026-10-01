"""Explicit (forward) Euler integrator.

Mirrors the ``models`` package convention: a model exposes
``dynamics(t, state, params) -> state_derivative``. An integrator here
exposes a single ``step(...)`` function that advances a state forward by
one timestep, given that model's dynamics function. Because every
integrator in this package shares that signature, swapping schemes is a
one-line change in the importing script, e.g.::

    from integrators import explicit_euler as integrator
    # ... later ...
    next_state = integrator.step(model.dynamics, t, state, timestep, params)

Explicit Euler is first-order accurate: local error is O(timestep**2) per
step, global error O(timestep). It is cheap (one dynamics evaluation per
step) but requires a small timestep to stay accurate, especially for
oscillatory systems like the pendulum.
"""

import numpy as np


def step(dynamics, t, state, timestep, params):
    """Advance ``state`` by one explicit-Euler step.

    Parameters
    ----------
    dynamics : callable
        Function with signature ``dynamics(t, state, params) -> state_derivative``,
        e.g. ``models.pendulum.dynamics``. Must return an array the same
        shape as ``state``.
    t : float
        Current simulation time (s). Unused by Euler itself (the slope is
        only ever evaluated at the start of the step) but kept in the
        signature so every integrator in this package is interchangeable.
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
    state_derivative = dynamics(t, state, params)
    next_state = state + timestep * state_derivative
    return next_state
