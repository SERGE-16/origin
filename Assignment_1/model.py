"""
Core model of the rimless wheel -- SCIPY-FREE VERSION.

Same physics and API as the scipy version (RimlessWheelParams,
continuous_dynamics, energy, reset_map, simulate), but the ODE
integration and impact-event detection are hand-rolled instead of using
scipy.integrate.solve_ivp:

  - fixed-step RK4 for the smooth stance-phase integration (instead of
    adaptive RK45)
  - manual bisection, re-integrating with fine substeps inside the
    bracketing step, to locate the impact time (instead of scipy's
    built-in dense-output root finding on events)

See the docstring of `simulate()` for how the two pieces fit together.
"""

from dataclasses import dataclass
import numpy as np

G_EARTH = 9.81  # m/s^2


@dataclass
class RimlessWheelParams:
    l: float = 1.0          # spoke length [m]
    N: int = 8              # number of spokes
    gamma: float = 0.08     # downhill ground slope [rad]
    m: float = 1.0          # hub point mass [kg] (cancels out of the dynamics)
    g: float = G_EARTH

    @property
    def alpha(self) -> float:
        """Half-angle between adjacent spokes: alpha = pi/N."""
        return np.pi / self.N

    def forward_guard_angle(self) -> float:
        return self.gamma + self.alpha

    def backward_guard_angle(self) -> float:
        return self.gamma - self.alpha


# --------------------------------------------------------------------------
# Continuous dynamics
# --------------------------------------------------------------------------
def continuous_dynamics(x, p: RimlessWheelParams):
    """xdot = f(x) for the pinned inverted-pendulum stance phase. No time
    dependence, so we don't bother threading a `t` argument through."""
    theta, theta_dot = x
    theta_ddot = (p.g / p.l) * np.sin(theta)
    return np.array([theta_dot, theta_ddot])


def energy(x, p: RimlessWheelParams):
    """
    E = 1/2 m l^2 thetadot^2 + m g l cos(theta), conserved during stance
    (height of the point mass above the current pivot is +l*cos(theta)).
    """
    theta, theta_dot = x
    return 0.5 * p.m * p.l**2 * theta_dot**2 + p.m * p.g * p.l * np.cos(theta)


def guard_forward(x, p: RimlessWheelParams):
    return x[0] - p.forward_guard_angle()


def guard_backward(x, p: RimlessWheelParams):
    return x[0] - p.backward_guard_angle()


# --------------------------------------------------------------------------
# Hand-rolled RK4 integrator
# --------------------------------------------------------------------------
def rk4_step(x, dt, p: RimlessWheelParams):
    """One fixed-step classical RK4 step for the (time-invariant) dynamics."""
    k1 = continuous_dynamics(x, p)
    k2 = continuous_dynamics(x + 0.5 * dt * k1, p)
    k3 = continuous_dynamics(x + 0.5 * dt * k2, p)
    k4 = continuous_dynamics(x + dt * k3, p)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_fixed(x0, duration, p: RimlessWheelParams, n_substeps):
    """Integrate `duration` seconds forward from x0 using n_substeps equal
    RK4 substeps. Used both for the main stepping loop and for the fine
    re-integration used inside event bisection."""
    if n_substeps <= 0:
        return np.array(x0, dtype=float)
    dt = duration / n_substeps
    x = np.array(x0, dtype=float)
    for _ in range(n_substeps):
        x = rk4_step(x, dt, p)
    return x


# --------------------------------------------------------------------------
# Event location by bisection
# --------------------------------------------------------------------------
def _locate_event_bisect(x_prev, dt_bracket, guard_fn, p: RimlessWheelParams,
                          tol=1e-10, max_iter=60, n_substeps_refine=8):
    """
    Given a state x_prev at the start of a bracket of duration dt_bracket
    during which guard_fn changes sign exactly once, bisect on the
    fraction of dt_bracket to locate the zero crossing to time-tolerance
    `tol`, re-integrating (with n_substeps_refine RK4 substeps) from
    x_prev each time to evaluate guard_fn at the candidate time.
    """
    g_lo = guard_fn(x_prev, p)
    t_lo, t_hi = 0.0, dt_bracket
    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        x_mid = integrate_fixed(x_prev, t_mid, p, n_substeps_refine)
        g_mid = guard_fn(x_mid, p)
        if g_lo * g_mid <= 0.0:
            t_hi = t_mid
        else:
            t_lo, g_lo = t_mid, g_mid
        if (t_hi - t_lo) < tol:
            break
    t_event = t_hi
    x_event = integrate_fixed(x_prev, t_event, p, n_substeps_refine)
    return t_event, x_event


# --------------------------------------------------------------------------
# Reset map (identical physics to the scipy version)
# --------------------------------------------------------------------------
def reset_map(x_minus, p: RimlessWheelParams, direction: str):
    theta_minus, theta_dot_minus = x_minus
    two_alpha = 2.0 * p.alpha
    theta_dot_plus = theta_dot_minus * np.cos(two_alpha)
    if direction == "forward":
        theta_plus = theta_minus - two_alpha
    elif direction == "backward":
        theta_plus = theta_minus + two_alpha
    else:
        raise ValueError(direction)
    return np.array([theta_plus, theta_dot_plus])


# --------------------------------------------------------------------------
# Hybrid simulation
# --------------------------------------------------------------------------
class SimResult:
    def __init__(self):
        self.t = []       # list of time arrays, one per stance phase
        self.x = []       # list of state arrays (2, n_pts), one per stance phase
        self.events = []  # list of 'forward' / 'backward' per transition
        self.step_times = [0.0]

    def flatten(self):
        t_all, x_all = [], []
        for t_seg, x_seg in zip(self.t, self.x):
            t_all.append(t_seg)
            x_all.append(x_seg)
        return np.concatenate(t_all), np.concatenate(x_all, axis=1)


def simulate(x0, p: RimlessWheelParams, t_max=40.0, max_steps=400,
             dt_outer=5e-3, n_substeps_refine=6, event_tol=1e-9):
    """
    Simulate the rimless wheel hybrid system from initial state x0, using
    a hand-rolled fixed-step RK4 inner loop (step size dt_outer) plus
    bisection-based event location -- no scipy.

    `max_steps` counts IMPACT EVENTS, not fixed-size integration substeps.
    """
    x = np.array(x0, dtype=float)
    result = SimResult()
    t_total = 0.0
    status = "max_steps"

    for step in range(max_steps):
        if t_total >= t_max:
            status = "timeout"
            break

        t_local = 0.0
        x_local = x.copy()
        ts_phase = [0.0]
        xs_phase = [x_local.copy()]
        event_found = False
        direction = None
        x_event = None

        while True:
            remaining = t_max - (t_total + t_local)
            if remaining <= 0:
                break
            dt = min(dt_outer, remaining)

            g_fwd_prev = guard_forward(x_local, p)
            g_bwd_prev = guard_backward(x_local, p)
            x_new = rk4_step(x_local, dt, p)
            g_fwd_new = guard_forward(x_new, p)
            g_bwd_new = guard_backward(x_new, p)

            fwd_cross = (g_fwd_prev < 0.0) and (g_fwd_new >= 0.0)
            bwd_cross = (g_bwd_prev > 0.0) and (g_bwd_new <= 0.0)

            if fwd_cross or bwd_cross:
                guard_fn = guard_forward if fwd_cross else guard_backward
                direction = "forward" if fwd_cross else "backward"
                t_event_rel, x_event = _locate_event_bisect(
                    x_local, dt, guard_fn, p, tol=event_tol,
                    n_substeps_refine=n_substeps_refine)
                t_local += t_event_rel
                ts_phase.append(t_local)
                xs_phase.append(x_event.copy())
                event_found = True
                break
            else:
                t_local += dt
                x_local = x_new
                ts_phase.append(t_local)
                xs_phase.append(x_local.copy())

        result.t.append(t_total + np.array(ts_phase))
        result.x.append(np.array(xs_phase).T)  # shape (2, n_pts)
        t_total += t_local

        if not event_found:
            status = "timeout"
            break

        x = reset_map(x_event, p, direction)
        result.events.append(direction)
        result.step_times.append(t_total)
    else:
        status = "max_steps"

    return result, status