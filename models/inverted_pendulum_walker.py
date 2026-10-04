import matplotlib.pyplot as plt
import numpy as np


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s) - shouldn't need it
        "ankle_torque": 0.0,  # optional ankle torque (N m) - control input
        "incline": .06, #slope of ground
        "angle_of_attack": np.pi/8 #alpha
    }
    return params


def generate_initial_condition():
    """Start upright with enough forward velocity to step over the top."""
    return np.array([0.0, 4.0])


def dynamics(t, state, params):
    #verbatim from rimless_wheel
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]
    ankle_torque=params["ankle_torque"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
        - damping_coeff * angular_velocity
        + ankle_torque # Ankle torque, deviation from rimless wheel
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def get_control_bounds(params):
    angle_of_attack_min=np.pi/8
    angle_of_attack_max=np.pi/7

    length=params["length"]
    mass=params["mass"]
    gravity=params["gravity"]

    tau_min=-.1*length*mass*gravity
    tau_max=.05*length*mass*gravity

    return angle_of_attack_min, angle_of_attack_max, tau_min, tau_max


def event_guard(previous_state, next_state, params):
    angle_of_attack=params["angle_of_attack"]
    incline=params["incline"]

    theta_prev=previous_state[0]
    theta_next=next_state[0]

    #swing leg reaches the ground
    #theta_TD = gamma + alpha
    theta_td=incline+angle_of_attack

    prev_event=theta_prev-theta_td
    next_event=theta_next-theta_td

    return (prev_event <=0 and next_event>=0) #returns boolean, if threshold is passed

def event_dynamics(state, params):
    alpha=params["angle_of_attack"]

    theta=state[0]
    theta_dot=state[1]

    #new stance leg is the old swing leg
    #swing leg is 2*alpha away from the old stance leg
    new_theta=theta-2*alpha

    #angular velocity just after impact - angular velocity is lost
    new_theta_dot=theta_dot*np.cos(2*alpha)

    return np.array([new_theta, new_theta_dot])


def calculate_energy(state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle=state[0]
    angular_velocity=state[1]

    #verbatim from previous assignments
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
    length : float
        Leg length in meters.
    incline : float
        Ground's downhill slope angle in radians.
    angle_of_attack : float
        Half the angle between the stance and forward swing legs.
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

    ax.fill_between(
        ground_x,
        ground_y,
        limits[2],
        color="#eee7dc",
        zorder=0
    )

    ax.plot(
        ground_x,
        ground_y,
        color="#7b6651",
        linewidth=2,
        label="Ground"
    )

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

        swing_foot = hub - length * np.array(
            [np.sin(swing_angle), np.cos(swing_angle)]
        )

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

    ax.plot(
        *foot,
        "s",
        color="#333333",
        markersize=8,
        zorder=5,
        label="Stance foot"
    )

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
