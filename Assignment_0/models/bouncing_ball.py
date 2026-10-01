"""Bouncing ball dynamics.

Expected dynamics (written before implementing, per the assignment)
---------------------------------------------------------------------
State: ``[height, velocity]`` (m, m/s), height measured from the ground
to the ball's center.

While airborne (height above the ground), the only force is gravity, so
the trajectory between bounces should be a plain downward-opening
parabola in height-vs-time -- free fall, then a symmetric flight back
up, exactly like tossing a ball.

At contact the ball can't be treated as an ideal instantaneous
collision if we want to reuse the same fixed-step ``integrator.step``
loop as the pendulum (an instantaneous velocity flip would need the
integrator or the outer loop to special-case an event, breaking the
"drop-in integrator" design). Instead this model uses a *compliant
contact*: once the ball penetrates the ground, a stiff spring pushes it
back out, and (optionally) a damper bleeds energy out of that push. That
keeps ``dynamics(t, state, params)`` a plain, if stiff, ODE that any of
the integrators can step through unmodified -- the bounce is just a
region where the acceleration briefly spikes upward instead of being
constant.

With the contact damping set to zero, contact is perfectly elastic: the
ball should bounce back to *exactly* its starting height forever, and
total mechanical energy (kinetic + gravitational + the spring's stored
energy) should stay constant, mirroring the pendulum's undamped energy
check. With damping turned on (the default), each bounce should return
to a lower apex than the last, and the ball should settle to rest on
the ground, again mirroring the pendulum's damped case ("energy bleeds
out until it comes to a standstill").

Sanity checks this suggests (implemented in bouncing_ball_check.py)
---------------------------------------------------------------------
1. Undamped: total energy stays constant (relative drift near zero,
   bounded only by integrator error) -- same idea as the pendulum's
   frictionless energy check.
2. Damped: total energy is non-increasing and trends to the resting
   equilibrium's energy level (ball at rest, sitting on the ground).
3. Damped: successive bounce apex heights are non-increasing (each
   bounce should not go higher than the last).
4. The ball should never fall meaningfully below the ground -- if the
   timestep is too large relative to the contact stiffness, an
   integrator can "tunnel" through the spring (step clean over the
   force spike) and the ball falls through the floor. That's a useful,
   visible failure mode for the timestep-accuracy discussion.
"""

import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    radius = params["radius"]
    ground_stiffness = params["ground_stiffness"]
    ground_damping = params["ground_damping"]

    height = state[0]
    velocity = state[1]

    penetration = radius - height  # > 0 once the ball's surface is below ground level
    if penetration > 0.0:
        # Spring pushes up proportional to penetration; damper resists the
        # closing velocity. Clip at zero so contact only ever pushes the
        # ball away from the ground, never pulls it in (no adhesion).
        contact_force = max(ground_stiffness * penetration - ground_damping * velocity, 0.0)
    else:
        contact_force = 0.0

    acceleration = -gravity + contact_force / mass
    state_derivative = np.array([velocity, acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity (m/s^2)
        "mass": 0.5,  # ball mass (kg)
        "radius": 0.05,  # ball radius (m) -- contact happens at height = radius
        "ground_stiffness": 5e4,  # contact spring stiffness (N/m)
        "ground_damping": 20.0,  # contact damping (N*s/m); 0 => perfectly elastic bounces
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``.

    Total mechanical energy = kinetic + gravitational potential + the
    contact spring's stored (elastic) potential energy. Including the
    spring term matters: while the ball is in contact, energy is
    genuinely stored in the compressed "ground", not lost -- only
    ``ground_damping`` actually removes energy from the system.
    """
    gravity = params["gravity"]
    mass = params["mass"]
    radius = params["radius"]
    ground_stiffness = params["ground_stiffness"]

    height = state[0]  # indexes entire row "vectorized" if state is (2, N)
    velocity = state[1]

    penetration = np.maximum(radius - height, 0.0)

    kinetic_energy = 0.5 * mass * velocity**2
    potential_energy = mass * gravity * height + 0.5 * ground_stiffness * penetration**2
    return kinetic_energy, potential_energy