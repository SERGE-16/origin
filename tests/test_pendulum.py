import numpy as np

from integrators import rk4
from models import pendulum

TIMESTEP = 0.01
STEPS = 100
INITIAL_STATE = np.array([0.5, 0.0])  # angle 0 is upright; avoid equilibria


def total_energy(state, params):
    kinetic_energy, potential_energy = pendulum.calculate_energy(state, params)
    return kinetic_energy + potential_energy


def simulate(params, state=INITIAL_STATE, steps=STEPS, timestep=TIMESTEP):
    """Return the trajectory as a ``(2, steps + 1)`` array."""
    trajectory = [state]
    for k in range(steps):
        state = rk4(pendulum.dynamics, k * timestep, state, timestep, params)
        trajectory.append(state)
    return np.array(trajectory).T


def test_energy_conserved_without_damping_or_torque():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0

    energy = total_energy(simulate(params), params)

    # RK4 drifts ~1e-8 per step, so allow a small absolute tolerance per step.
    assert np.all(np.isclose(np.diff(energy), 0.0, atol=1e-6))
    assert np.all(np.isclose(energy, energy[0]))


def test_torque_work_equals_energy_change():
    # With no damping, dE/dt = torque * angular_velocity, so dE = torque * d(angle).
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 2.0

    trajectory = simulate(params)
    energy = total_energy(trajectory, params)
    work = params["torque"] * (trajectory[0, -1] - trajectory[0, 0])

    assert not np.isclose(energy[-1], energy[0])  # torque actually did something
    assert np.isclose(energy[-1] - energy[0], work)


def test_torque_balances_gravity():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    angle = INITIAL_STATE[0]
    params["torque"] = (
        -params["mass"] * params["gravity"] * params["length"] * np.sin(angle)
    )

    derivative = pendulum.dynamics(0.0, np.array([angle, 0.0]), params)

    assert np.allclose(derivative, 0.0)


def test_damping_dissipates_energy():
    # With no torque, dE/dt = -damping_coeff * angular_velocity**2 <= 0.
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.5
    params["torque"] = 0.0

    trajectory = simulate(params)
    energy = total_energy(trajectory, params)
    angular_velocity = trajectory[1]
    dissipated = params["damping_coeff"] * np.trapezoid(
        angular_velocity**2, dx=TIMESTEP
    )

    assert np.all(np.diff(energy) <= 1e-12)
    assert energy[-1] < energy[0]
    assert np.isclose(energy[0] - energy[-1], dissipated, rtol=1e-3)
