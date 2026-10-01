"""
Poincare section / step-to-step return map, fixed point, and Floquet
multiplier for the rimless wheel -- SCIPY-FREE VERSION.

Same as the scipy version, except the fixed-point root-find uses the
plain bisection in rootfind.py instead of scipy.optimize.brentq. The
closed-form algebra (return map, fixed point, Floquet multiplier) is
pure numpy either way and needed no changes; see that derivation below.

We take the Poincare section at the FORWARD guard (theta = gamma+alpha,
theta_dot > 0), i.e. the instant just before each forward impact. The
return map

    r : theta_dot_n  |->  theta_dot_{n+1}

sends the pre-impact speed at one forward crossing to the pre-impact
speed at the next one (passing through the reset + one stance swing in
between -- and, if necessary, through any intervening backward
rock/bounce, handled generally by re-using the hybrid simulator).

Closed-form shortcut
---------------------
Whenever the post-reset swing clears theta=gamma+alpha WITHOUT turning
back (guaranteed if gamma >= alpha, and true near the fixed point studied
below even when gamma < alpha, as we verify numerically), energy
conservation during the stance phase gives an exact algebraic map. With
    E = 1/2 theta_dot^2 + (g/l) cos(theta)   conserved during stance,
    theta^+ = gamma - alpha,  theta_dot^+ = theta_dot^- * cos(2 alpha),
solving E(theta^+, theta_dot^+) = E(gamma+alpha, theta_dot_next) gives

    r(v) = sqrt( v^2 * cos(2 alpha)^2  +  (4 g / l) * sin(gamma) * sin(alpha) )

Setting r(v*) = v* and solving gives the fixed point in closed form:

    v*^2 = (g/l) * sin(gamma) / ( sin(alpha) * cos(alpha)^2 )

and differentiating r(v) at v=v* gives the Floquet multiplier in closed
form too (since r(v*) = v*):

    dr/dv |_{v*} = v* * cos(2 alpha)^2 / r(v*) = cos(2 alpha)^2

i.e. for THIS system, in the "always-clears" regime, the analytic
prediction is that the Floquet multiplier of the rolling limit cycle
equals cos^2(2 alpha) exactly -- independent of gamma! We use this as a
closed-form cross-check against the numerical (finite-difference,
general-simulator-based) estimate.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from rootfind import bisect
from model import RimlessWheelParams, simulate, reset_map


def closed_form_return_map(v, p: RimlessWheelParams):
    two_alpha = 2 * p.alpha
    C = (4 * p.g / p.l) * np.sin(p.gamma) * np.sin(p.alpha)
    return np.sqrt(v**2 * np.cos(two_alpha) ** 2 + C)


def closed_form_fixed_point(p: RimlessWheelParams):
    v2 = (p.g / p.l) * np.sin(p.gamma) / (np.sin(p.alpha) * np.cos(p.alpha) ** 2)
    return np.sqrt(v2)


def closed_form_floquet(p: RimlessWheelParams):
    return np.cos(2 * p.alpha) ** 2


def escape_speed_threshold(p: RimlessWheelParams):
    """
    Minimum pre-impact forward-guard speed v such that the post-reset swing
    (starting at theta=gamma-alpha) has enough kinetic energy to climb over
    the potential "hump" at theta=0 (the unstable balance point) and reach
    theta=gamma+alpha WITHOUT turning back -- i.e. the smallest v for which
    the single-hop return map r(v) is meaningful at all.

    If gamma >= alpha, the hump (theta=0) is never inside the stance wedge
    [gamma-alpha, gamma+alpha] to begin with, so every swing is monotonic
    and the threshold is 0 (even a dead-stop restart rolls forward).
    """
    if p.gamma >= p.alpha:
        return 0.0
    delta_E = (p.g / p.l) * (1 - np.cos(p.alpha - p.gamma))
    theta_dot_plus_min = np.sqrt(2 * delta_E)
    return theta_dot_plus_min / np.cos(2 * p.alpha)


def general_return_map(v, p: RimlessWheelParams, max_substeps=30):
    """
    Robust one-Poincare-step map using the full hybrid simulator: reset as
    if `v` were the pre-impact forward-guard speed, then integrate through
    the stance phase(s) -- handling any intervening backward bounce -- up
    until the NEXT forward-guard crossing, and return that speed.

    This matches closed_form_return_map exactly whenever the swing clears
    monotonically, but stays correct even if it doesn't.
    """
    x_plus = reset_map([p.forward_guard_angle(), v], p, "forward")
    result, status = simulate(x_plus, p, t_max=30.0, max_steps=max_substeps)
    for k, ev in enumerate(result.events):
        if ev == "forward":
            # first forward crossing after (and including) the initial reset
            return abs(result.x[k][1, -1])
    return np.nan  # never made it back to the forward guard (fell over)


def build_return_map_data(p: RimlessWheelParams, v_min=None, v_max=4.0, n=60):
    v_thresh = escape_speed_threshold(p)
    if v_min is None:
        v_min = max(1e-3, v_thresh + 1e-3 * max(v_thresh, 1.0))
    vs = np.linspace(v_min, v_max, n)
    r_general = np.array([general_return_map(v, p) for v in vs])
    r_closed = closed_form_return_map(vs, p)
    return vs, r_general, r_closed


def find_fixed_point_numeric(p: RimlessWheelParams, v_lo=None, v_hi=6.0):
    """
    Bracket-and-bisect for the fixed point of the general (robust,
    hybrid-sim) return map, using the plain bisection in rootfind.py.
    The naive bracket [v_thresh, v_hi] can fail when the true fixed point
    sits very close to the escape-speed threshold (near the critical
    slope, see sweep.py) -- a small multiplicative margin above threshold
    can accidentally overshoot the root itself. To stay robust we scan a
    fine grid of candidate v_lo values just above threshold and use the
    first one that brackets a sign change with v_hi.
    """
    v_thresh = escape_speed_threshold(p)
    f = lambda v: general_return_map(v, p) - v

    if v_lo is not None:
        return bisect(f, v_lo, v_hi)

    candidates = v_thresh + np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]) * max(v_thresh, 1.0)
    f_hi = f(v_hi)
    for v_lo_try in candidates:
        f_lo = f(v_lo_try)
        if np.isfinite(f_lo) and f_lo * f_hi < 0:
            return bisect(f, v_lo_try, v_hi)
    raise ValueError(f"could not bracket a root above threshold {v_thresh} for gamma={p.gamma}, N={p.N}")


def estimate_floquet_multiplier(p: RimlessWheelParams, v_star, eps=1e-3, use_general=True):
    rmap = general_return_map if use_general else closed_form_return_map
    r_plus = rmap(v_star + eps, p)
    r_minus = rmap(v_star - eps, p)
    return (r_plus - r_minus) / (2 * eps)


def plot_return_map(vs, r_general, r_closed, v_star, savepath):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot(vs, vs, "k--", lw=1, label="identity ($\\dot\\theta_{n+1}=\\dot\\theta_n$)")
    ax.plot(vs, r_closed, color="0.7", lw=4, alpha=0.6, label="closed-form (energy conservation)")
    ax.plot(vs, r_general, color="#2a6f97", lw=1.8, label="general hybrid-sim return map")
    ax.plot(v_star, v_star, "o", color="crimson", ms=9, zorder=5,
            label=f"fixed point $\\dot\\theta^*$={v_star:.4f} rad/s")
    ax.set_xlabel(r"$\dot\theta_n$  (speed at forward-guard crossing $n$) [rad/s]")
    ax.set_ylabel(r"$\dot\theta_{n+1}$  (speed at next crossing) [rad/s]")
    ax.set_title("Step-to-step return map (Poincare section at forward guard)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"saved {savepath}")


if __name__ == "__main__":
    p = RimlessWheelParams(N=8, gamma=0.2, l=1.0)

    v_star_closed = closed_form_fixed_point(p)
    v_star_numeric = find_fixed_point_numeric(p)
    print(f"closed-form fixed point   v* = {v_star_closed:.6f} rad/s")
    print(f"numeric (general sim) fp  v* = {v_star_numeric:.6f} rad/s")

    lam_closed = closed_form_floquet(p)
    lam_numeric = estimate_floquet_multiplier(p, v_star_numeric, eps=1e-3, use_general=True)
    print(f"closed-form Floquet multiplier  cos^2(2 alpha) = {lam_closed:.6f}")
    print(f"numeric (finite-diff, general sim) multiplier  = {lam_numeric:.6f}")
    print("=> |multiplier| < 1  -> limit cycle is LOCALLY EXPONENTIALLY STABLE.")

    v_thresh = escape_speed_threshold(p)
    print(f"escape-speed threshold (below which the 1-hop map isn't meaningful): {v_thresh:.4f} rad/s")
    vs, r_general, r_closed = build_return_map_data(p, v_max=4.0, n=60)
    plot_return_map(vs, r_general, r_closed, v_star_numeric, "figures/return_map.png")