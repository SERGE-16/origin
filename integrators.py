import numpy as np


# from the wikipedia article: t is t_n and timestep is h; returns the state at t + h
def rk4(dynamics, t, state, timestep, params):
    state = np.asarray(state, dtype=float)
    k1 = dynamics(t, state, params)
    k2 = dynamics(t + timestep / 2, state + k1 * timestep / 2, params)
    k3 = dynamics(t + timestep / 2, state + k2 * timestep / 2, params)
    k4 = dynamics(t + timestep, state + timestep * k3, params)
    return state + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler(dynamics, t, state, timestep, params):
    state = np.asarray(state, dtype=float)
    return state + timestep * dynamics(t, state, params)
