"""InvertedPendulumWalker starter model, with visualization provided.

Implement the model functions for Assignment 2. The visualizer works independently
of those functions; it draws a supplied state without advancing the simulation.

--------------------------------------------------------------------------------
Derivation notes (kept here rather than in the report, since they explain the
exact formulas used below)
--------------------------------------------------------------------------------
Coordinates match ``visualize``: theta is measured clockwise from upward
vertical, the stance foot is fixed, and the hub (point mass / hip) sits at

    hub = foot + length * [sin(theta), cos(theta)]

so theta > 0 means the hub has swung out ahead of the stance foot in +x
(downhill/forward), and theta < 0 means the hub is still trailing behind it.
The ground through the stance foot has slope ``incline`` (positive = downhill
to the right), and the swing leg hangs at angle ``theta - 2*angle_of_attack``
from vertical (i.e. angle_of_attack, alpha, is HALF the interior angle between
the two legs, exactly like the rimless wheel).

1) Continuous dynamics (single support, point-mass pendulum pivoting about the
   fixed stance foot), with an ankle torque tau summed in:

       theta_ddot = (m g l sin(theta) + tau) / (m l^2)

   This is an *inverted* pendulum: theta = 0 (upright) is an unstable
   equilibrium, since theta_ddot has the same sign as theta for small theta.

2) Touchdown guard. Working out when the swing foot's height matches the
   sloped ground (see the writeup / whiteboard derivation) gives the same
   clean result as the fixed-alpha rimless wheel, generalized to a per-step
   alpha:

       theta_td = alpha + incline

   i.e. touchdown happens when the (still-fixed) old stance leg reaches
   alpha + incline from vertical -- independent of the leg's angular velocity.

3) Impact map. Modeling contact as an instantaneous, non-slipping, inelastic
   collision that removes the radial (leg-parallel) momentum component and
   conserves the tangential component resolved onto the new leg direction
   gives the standard rimless-wheel result, again generalized to per-step
   alpha:

       theta+           = incline - alpha        (new stance angle, from vertical)
       theta_dot+        = theta_dot- * cos(2*alpha)

   (theta_old - theta_new = 2*alpha at the instant of impact, which is where
   the cos(2*alpha) factor comes from.)

These three formulas are exactly what event_guard/event_dynamics/dynamics
implement below.
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_params():
    """Default parameters for the inverted pendulum walker.

    ``incline`` is the assignment's slope gamma = 0.06 rad. ``angle_of_attack``
    and ``ankle_torque`` are control inputs -- they are given sane defaults here
    (mid-range alpha, zero ankle torque) but are expected to be overwritten
    every step/timestep by whatever policy is driving the simulation.
    """
    alpha_bounds = (np.pi / 8, np.pi / 7)
    torque_bound_fracs = (-0.1, 0.05)  # multiples of m*g*l

    params = {
        "gravity": 9.81,  # gravity, m/s^2
        "length": 1.0,  # leg length, m
        "mass": 1.0,  # point mass at the hub, kg
        "incline": 0.06,  # ground slope gamma, rad (positive = downhill to +x)
        "angle_of_attack": float(np.mean(alpha_bounds)),  # alpha, rad (control input, per step)
        "ankle_torque": 0.0,  # tau, N*m (control input, per timestep)
        "alpha_bounds": alpha_bounds,
        "torque_bound_fracs": torque_bound_fracs,
    }
    return params


def torque_bounds(params):
    """Return (tau_min, tau_max) in N*m from the stored fractions of m*g*l."""
    mgl = params["mass"] * params["gravity"] * params["length"]
    frac_min, frac_max = params["torque_bound_fracs"]
    return frac_min * mgl, frac_max * mgl


def dynamics(t, state, params):
    """Single-support (stance-phase) dynamics: inverted pendulum + ankle torque."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    ankle_torque = params.get("ankle_torque", 0.0)

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle) + ankle_torque
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def event_guard(previous_state, next_state, params):
    """Touchdown guard: True iff theta crossed alpha + incline this step.

    theta increases monotonically to alpha + incline during a normal forward
    stance phase (see module docstring), so a simple sign-crossing test on
    theta - theta_td is sufficient and robust to the fixed integration step.
    """
    theta_td = params["angle_of_attack"] + params["incline"]
    return previous_state[0] < theta_td <= next_state[0]


def event_dynamics(state, params):
    """Touchdown impact map: pivot from the old stance foot to the new one.

    ``state`` is the pre-impact state [theta-, theta_dot-] (should satisfy
    theta- == angle_of_attack + incline, up to integration tolerance).
    Returns the post-impact state [theta+, theta_dot+] measured about the new
    stance foot (formerly the swing foot).
    """
    alpha = params["angle_of_attack"]
    incline = params["incline"]
    _, angular_velocity_minus = state

    theta_plus = incline - alpha
    angular_velocity_plus = angular_velocity_minus * np.cos(2 * alpha)
    return np.array([theta_plus, angular_velocity_plus])


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``.

    Total mechanical energy E = KE + PE is conserved whenever ankle_torque=0
    (verify: dE/dt = m l^2 theta_dot theta_ddot + m g l (-sin theta) theta_dot
    = m l^2 theta_dot [theta_ddot - (g/l) sin theta] = 0 under the dynamics
    above). It jumps down at each touchdown impact (theta_dot shrinks by
    cos(2*alpha) < 1), which is the walker's only source of dissipation.
    """
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy


def visualize(
    state,
    params,
    ax=None,
    *,
    show_swing=True,
    stance_position=(0.0, 0.0),
    view_limits=None,
):
    """Draw one walker pose and return a Matplotlib Axes.

    Parameters
    ----------
    state : array-like, shape (2,)
        [theta, angular_velocity], in radians and radians/second. Theta is
        measured clockwise from upward vertical; positive x points right.
    params : dict
        ``length`` is the leg length in meters. ``incline`` is the ground's
        downhill slope angle in radians (positive slopes descend to the right).
        ``angle_of_attack`` is HALF the angle between the stance and forward swing
        legs, in radians; it is needed only when show_swing=True.
        ``ankle_torque`` (optional, default 0) is displayed in N m, with positive
        torque acting in the positive theta direction. Other keys are ignored.
    ax : matplotlib.axes.Axes, optional
        Axes to clear and reuse. If omitted, create a figure. This function
        neither shows nor saves it: use plt.show() or ax.figure.savefig(...).
    show_swing : bool
        Draw a straight forward swing leg at the supplied angle_of_attack. Set False
        while the swing leg is held clear or while balancing. Swing motion is
        not part of the two-state model and is not inferred from theta.
    stance_position : pair of floats
        Current stance foot's (x, y) in meters, default (0, 0). The two-state
        model does not track translation; supply foot positions if desired.
        Ground passes through this point at the supplied incline.
    view_limits : (xmin, xmax, ymin, ymax), optional
        Fixed camera bounds in meters. By default the view follows the stance
        foot with bounds that fit both legs at any angle. Supply the same bounds
        each frame for a stationary world view.

    Notes
    -----
    Draws the supplied pose; contact events belong in the simulation.
    Reuse ax for frame sequences; use evenly spaced simulation times for playback
    at a fixed frame rate, and pass the parameters actually used at each frame.
    """
    state = np.asarray(state, dtype=float)
    foot = np.asarray(stance_position, dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("state must contain two finite values: [theta, velocity].")
    if foot.shape != (2,) or not np.all(np.isfinite(foot)):
        raise ValueError("stance_position must contain two finite values: [x, y].")
    length = float(params["length"])
    incline = float(params["incline"])
    torque = float(params.get("ankle_torque", 0.0))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive.")
    if not np.isfinite(incline) or abs(incline) >= np.pi / 2:
        raise ValueError("incline must be finite and between -pi/2 and pi/2.")
    if not np.isfinite(torque):
        raise ValueError("ankle_torque must be finite.")
    if show_swing:
        angle_of_attack = float(params["angle_of_attack"])
        if not np.isfinite(angle_of_attack):
            raise ValueError("angle_of_attack must be finite.")

    if view_limits is None:
        radius = 2.15 * length
        view_limits = (
            foot[0] - radius,
            foot[0] + radius,
            foot[1] - radius,
            foot[1] + radius,
        )
    limits = np.asarray(view_limits, dtype=float)
    if (
        limits.shape != (4,)
        or not np.all(np.isfinite(limits))
        or limits[0] >= limits[1]
        or limits[2] >= limits[3]
    ):
        raise ValueError(
            "view_limits must be (xmin, xmax, ymin, ymax) with increasing bounds."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.clear()
    theta, angular_velocity = state
    hub = foot + length * np.array([np.sin(theta), np.cos(theta)])

    ground_x = np.array(limits[:2])
    ground_y = foot[1] - np.tan(incline) * (ground_x - foot[0])
    ax.fill_between(ground_x, ground_y, limits[2], color="#eee7dc", zorder=0)
    ax.plot(ground_x, ground_y, color="#7b6651", linewidth=2, label="Ground")
    ax.plot(
        [foot[0], foot[0]],
        [foot[1], foot[1] + 1.25 * length],
        ":",
        color="0.7",
        linewidth=1,
        label="Vertical",
    )

    if show_swing:
        swing_angle = theta - 2 * angle_of_attack
        swing_foot = hub - length * np.array([np.sin(swing_angle), np.cos(swing_angle)])
        swing_color = "#df8a25"
        ax.plot(
            [hub[0], swing_foot[0]],
            [hub[1], swing_foot[1]],
            "--",
            color=swing_color,
            linewidth=2.5,
            label="Swing leg",
            zorder=3,
        )
        ax.plot(
            *swing_foot,
            "o",
            color=swing_color,
            markersize=7,
            zorder=4,
            label="Swing foot",
        )

    stance_color = "#23699b"
    ax.plot(
        [foot[0], hub[0]],
        [foot[1], hub[1]],
        color=stance_color,
        linewidth=4,
        label="Stance leg",
        zorder=4,
    )
    ax.plot(*foot, "s", color="#333333", markersize=8, zorder=5, label="Stance foot")
    ax.plot(
        *hub,
        "o",
        color=stance_color,
        markeredgecolor="white",
        markersize=17,
        zorder=6,
        label="Hub",
    )
    ax.text(
        0.03,
        0.97,
        f"$\\theta$ = {theta:.3f} rad\n"
        f"$\\dot\\theta$ = {angular_velocity:.3f} rad/s\n"
        f"$\\tau$ = {torque:.3f} N m",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set(
        xlim=limits[:2],
        ylim=limits[2:],
        xlabel="x (m)",
        ylabel="y (m)",
        title="Inverted pendulum walker",
    )
    ax.set_aspect("equal", adjustable="box")
    return ax