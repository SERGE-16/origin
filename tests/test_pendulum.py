import inspect

import numpy as np

from Assignment_0.models import pendulum as model
from Assignment_0.integrators import explicit_euler_step


def test_pendulum_energy():
    params = model.generate_params()
    params["damping_coeff"] = 0
    state = np.array([0.1, 0.0])

    initial_energy = model.calculate_energy(state, params)

    dt = 1e-3
    for i in range(100):
        state = explicit_euler_step(model.dynamics, i * dt, state, dt, params)

    final_energy = model.calculate_energy(state, params)

    assert np.isclose(np.sum(initial_energy), np.sum(final_energy))


def test_pendulum_torque():
    """
    Tests that torque works
    by checking that simulating gravity with the torque parameter matches regularly having gravity.
    """

    # With Gravity, No Other Torque
    params = model.generate_params()
    params["gravity"] = 9.8
    params["torque"] = 0

    state_G = np.array([0.1, 0.0])

    dt = 1e-3
    for i in range(100):
        state_G = explicit_euler_step(model.dynamics, i * dt, state_G, dt, params)

    # With Other Torque, No Gravity
    params = model.generate_params()
    params["gravity"] = 0

    state_T = np.array([0.1, 0.0])

    dt = 1e-3
    for i in range(100):
        params["torque"] = 9.8 * np.sin(state_T[0])
        state_T = explicit_euler_step(model.dynamics, i * dt, state_T, dt, params)

    assert np.isclose(state_G[0], state_T[0])
    assert np.isclose(state_G[1], state_T[1])


def test_pendulum_damping():
    """
    Test that damping works
    by testing at a bunch of initial conditions and damping coefficients
    that energy strictly reduces after a run.
    """

    params = model.generate_params()
    params["damping_coeff"] = 1.0
    params["mass"] = 1.0

    def check_energy_loss(state, params):
        last_energy = model.calculate_energy(state, params)

        dt = 1e-3
        for i in range(1000):
            state = explicit_euler_step(model.dynamics, i * dt, state, dt, params)

        final_energy = model.calculate_energy(state, params)

        assert np.sum(final_energy) <= np.sum(last_energy)

    for i in range(10):
        for j in range(10):
            for k in range(10):
                params["damping_coeff"] = k * 0.1 + 0.1
                check_energy_loss(np.array([i, j]), params)
