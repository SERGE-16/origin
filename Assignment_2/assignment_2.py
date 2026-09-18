"""Assignment 2: Inverted Pendulum Walker.

Reconstructed experiment/report script (the real ``assignment_2.py`` starter
wasn't available to build this from, so this script is written directly from
the assignment prompt; see the accompanying notes for the modeling choices
made along the way). It:

  1. Designs a saturated feedback-linearization ankle-torque controller and
     grid-searches its region of attraction (RoA) around standing.
  2. Picks the mid-stance crossing (theta = 0) as the Poincare section and
     builds a 2D lookup table (state theta_dot_k, control alpha) for the
     discrete step-to-step map, using a grid-resolution convergence check to
     pick the coarsest adequate grid.
  3. Backward-induces, from the states already inside the RoA, how many
     further steps (and which alpha policy) bring any swept theta_dot_k to a
     standstill.
  4. Simulates the full hybrid controller (stepping until inside the RoA,
     then ankle balancing) from an example initial condition.

Everything is shown live in matplotlib windows -- nothing is written to
disk. All 4 static plots and the live walking animation are opened together
by one plt.show() call at the end of main(), so closing (or Ctrl+C-ing) any
one of them does not lose the others; they're all already built and open.

Run with: uv run python assignment_2.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from models.inverted_pendulum_walker import (
    generate_params,
    dynamics,
    event_guard,
    event_dynamics,
    calculate_energy,
    torque_bounds,
    visualize,
)
from integrators import rk4 as default_integrator


# ==============================================================================
# Integrator
# ==============================================================================
# Uses integrators/rk4.py (integrators/explicit_euler.py is also available --
# every function below that integrates takes an `integrator` module as an
# argument, defaulting to RK4, so swapping is one call-site change, e.g.:
#
#   from integrators import explicit_euler
#   compute_roa(params, ..., integrator=explicit_euler)
#
# Both expose the same step(dynamics, t, state, timestep, params) -> state
# signature (see their docstrings), which is exactly what params['ankle_torque']
# / params['angle_of_attack'] being held fixed across a step assumes: the
# controller only updates once per timestep, zero-order-held across it.


# ==============================================================================
# Part 1: Ankle-torque balancing controller (feedback linearization + PD)
# ==============================================================================
# Strategy: cancel the destabilizing gravity torque (feedback linearization),
# then add a critically-damped PD term to pull (theta, theta_dot) -> (0, 0).
# tau is saturated to the assignment's bounds, so this only truly linearizes
# the dynamics near theta=0; further out the torque budget isn't enough to
# fully cancel gravity, which is exactly why the RoA below is finite and
# needs a numerical search rather than a clean analytical shape.
KP = 4.0  # rad/s^2 per rad
KD = 4.0  # rad/s^2 per (rad/s)  (critically damped: kd = 2*sqrt(kp))


def ankle_control(state, params, kp=KP, kd=KD):
    """Saturated feedback-linearizing PD controller. Vectorizes over state
    shaped (2,) or (2, N)."""
    gravity, length, mass = params["gravity"], params["length"], params["mass"]
    theta, theta_dot = state[0], state[1]
    tau_ff = -mass * gravity * length * np.sin(theta)  # cancel gravity
    tau_fb = mass * length**2 * (-kp * theta - kd * theta_dot)  # PD to origin
    tau = tau_ff + tau_fb
    tau_min, tau_max = torque_bounds(params)
    return np.clip(tau, tau_min, tau_max)


def compute_roa(
    params,
    theta_range,
    theta_dot_range,
    n_theta=101,
    n_theta_dot=101,
    t_final=6.0,
    dt=0.002,
    theta_tol=0.01,
    theta_dot_tol=0.02,
    integrator=default_integrator,
):
    """Grid-search the RoA of ``ankle_control`` by simulating the closed loop
    (dynamics + saturated tau, no stepping/impacts) forward from every grid
    point and checking convergence to (0, 0). Vectorized: every grid point is
    integrated simultaneously as one (2, n_theta*n_theta_dot) state array.
    """
    theta_grid = np.linspace(*theta_range, n_theta)
    theta_dot_grid = np.linspace(*theta_dot_range, n_theta_dot)
    TH, THD = np.meshgrid(theta_grid, theta_dot_grid, indexing="ij")
    state = np.vstack([TH.ravel(), THD.ravel()])

    p = dict(params)
    n_steps = int(round(t_final / dt))
    diverged = np.zeros(state.shape[1], dtype=bool)
    for _ in range(n_steps):
        p["ankle_torque"] = ankle_control(state, p)
        state = integrator.step(dynamics, 0.0, state, dt, p)
        diverged |= np.abs(state[0]) > np.pi / 2

    converged = (
        (~diverged) & (np.abs(state[0]) < theta_tol) & (np.abs(state[1]) < theta_dot_tol)
    )
    return theta_grid, theta_dot_grid, converged.reshape(n_theta, n_theta_dot)


def in_roa(theta, theta_dot, theta_grid, theta_dot_grid, roa_mask):
    """Nearest-grid-point RoA membership test for an arbitrary state."""
    if not (theta_grid[0] <= theta <= theta_grid[-1]):
        return False
    if not (theta_dot_grid[0] <= theta_dot <= theta_dot_grid[-1]):
        return False
    i = int(np.argmin(np.abs(theta_grid - theta)))
    j = int(np.argmin(np.abs(theta_dot_grid - theta_dot)))
    return bool(roa_mask[i, j])


def plot_roa(theta_grid, theta_dot_grid, roa_mask):
    """Build the RoA figure and leave it open (no save, no close) -- shown
    later by the single plt.show() call at the end of main()."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    TH, THD = np.meshgrid(theta_grid, theta_dot_grid, indexing="ij")
    ax.pcolormesh(TH, THD, roa_mask, cmap="Greens", vmin=0, vmax=1.6, shading="auto")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.axvline(0, color="0.6", linewidth=0.8)
    ax.plot(0, 0, "k*", markersize=14, label="Standing equilibrium")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_title("Region of attraction of the saturated ankle controller")
    ax.legend(loc="upper left")
    return fig


# ==============================================================================
# Part 2: Poincare map / lookup table (mid-stance section, theta = 0)
# ==============================================================================
# Picking theta = 0 (mid-stance) as the Poincare section makes theta constant
# by construction (satisfying the assignment's requirement) and, unlike the
# touchdown section, is independent of the control alpha, which is what makes
# theta_dot alone a valid state for the return map. Passive dynamics conserve
# specific energy e = 0.5*theta_dot^2 + (g/l)*cos(theta) whenever
# ankle_torque = 0 (see calculate_energy's docstring), so the whole mid-stance
# -> mid-stance map can be written in closed form instead of integrated:
#
#   theta_td            = alpha + incline
#   theta_dot_td         = sqrt(theta_dot_k^2 + (2g/l)(1 - cos(theta_td)))      [energy: 0 -> theta_td]
#   theta_dot_plus       = theta_dot_td * cos(2*alpha)                          [impact]
#   theta_plus           = incline - alpha
#   e_plus               = 0.5*theta_dot_plus^2 + (g/l)*cos(theta_plus)
#   theta_dot_{k+1}       = sqrt(2*(e_plus - g/l))     if e_plus > g/l, else the
#                           step FAILS (the walker never climbs back to
#                           theta=0 -- it stumbles and falls back on the new
#                           leg without completing the step).
def poincare_step(theta_dot_k, alpha, params):
    """Closed-form mid-stance -> mid-stance map. Returns theta_dot_{k+1}, or
    NaN where the step fails to clear theta=0 (a genuine failure mode, not a
    numerical one -- see the "dead zone" discussion in build_lookup_table)."""
    g, l, incline = params["gravity"], params["length"], params["incline"]
    theta_td = alpha + incline
    theta_dot_td = np.sqrt(np.maximum(theta_dot_k**2 + (2 * g / l) * (1 - np.cos(theta_td)), 0.0))
    theta_dot_plus = theta_dot_td * np.cos(2 * alpha)
    theta_plus = incline - alpha
    e_plus = 0.5 * theta_dot_plus**2 + (g / l) * np.cos(theta_plus)
    e_needed = g / l  # specific energy of (theta, theta_dot) = (0, 0)
    ok = e_plus > e_needed
    return np.where(ok, np.sqrt(np.maximum(2 * (e_plus - e_needed), 0.0)), np.nan)


def roa_capture_speed(theta_grid, theta_dot_grid, roa_mask):
    """Largest theta_dot (>=0) such that (theta=0, theta_dot) is inside the
    ankle controller's RoA -- i.e. states the lookup table can hand straight
    to the balance controller with zero further steps."""
    i0 = int(np.argmin(np.abs(theta_grid)))
    row = roa_mask[i0]
    positive = theta_dot_grid >= 0
    captured = row & positive
    if not captured.any():
        return 0.0
    return float(theta_dot_grid[captured].max())


def build_lookup_table(params, roa_capture, N, M, max_iters=200):
    """Backward-induction lookup table over the mid-stance Poincare section.

    theta_dot_grid sweeps [0, sqrt(2g/l)] (Froude number 2, per the
    assignment). alpha_grid sweeps the allowed [pi/8, pi/7]. steps_to_stand[i]
    is the fewest steps needed to bring theta_dot_grid[i] into the RoA
    (0 = already inside); policy_alpha[i] is the alpha achieving that.
    Grid points that stay at -1 are a genuine failure/unrecoverable set: no
    alpha in the allowed range avoids the walker stumbling on the very next
    touchdown (see the report notes).
    """
    theta_dot_max = np.sqrt(2 * params["gravity"] / params["length"])
    alpha_lo, alpha_hi = params["alpha_bounds"]
    theta_dot_grid = np.linspace(0.0, theta_dot_max, N)
    alpha_grid = np.linspace(alpha_lo, alpha_hi, M)

    steps_to_stand = np.full(N, -1, dtype=int)
    policy_alpha = np.full(N, np.nan)
    steps_to_stand[theta_dot_grid <= roa_capture] = 0

    TD, AL = np.meshgrid(theta_dot_grid, alpha_grid, indexing="ij")  # (N, M)
    NEXT = poincare_step(TD, AL, params)
    finite = np.isfinite(NEXT)
    with np.errstate(invalid="ignore"):
        idx_next = np.clip(np.round(NEXT / theta_dot_max * (N - 1)).astype(int), 0, N - 1)

    for n in range(1, max_iters + 1):
        unclassified = steps_to_stand < 0
        if not unclassified.any():
            break
        candidate = np.where(finite, steps_to_stand[idx_next], -999)
        reaches_prev = candidate == (n - 1)
        newly = unclassified & reaches_prev.any(axis=1)
        if not newly.any():
            break
        for i in np.where(newly)[0]:
            # Among alphas that reach a (n-1)-step state, prefer the one
            # landing on the fastest (highest-margin) such state.
            cols = np.where(reaches_prev[i])[0]
            best_col = cols[np.argmax(NEXT[i, cols])]
            policy_alpha[i] = alpha_grid[best_col]
        steps_to_stand[newly] = n

    return theta_dot_grid, alpha_grid, steps_to_stand, policy_alpha


def choose_grid_resolution(params, roa_capture, N_candidates, M, tol=0.01):
    """Resolution-convergence criterion: resample each candidate grid's
    classification onto a common fine reference grid (nearest-neighbor) and
    compare to the next-finer candidate. Pick the coarsest N whose mean
    disagreement against the next step up is below `tol`.
    """
    theta_dot_max = np.sqrt(2 * params["gravity"] / params["length"])
    ref_N = 4001
    ref_grid = np.linspace(0, theta_dot_max, ref_N)

    results = {}
    resampled = {}
    for N in N_candidates:
        grid, _, steps, _ = build_lookup_table(params, roa_capture, N=N, M=M)
        idx = np.clip(np.round(ref_grid / theta_dot_max * (N - 1)).astype(int), 0, N - 1)
        resampled[N] = steps[idx]
        results[N] = steps

    chosen = N_candidates[-1]
    report = []
    for a, b in zip(N_candidates[:-1], N_candidates[1:]):
        disagreement = (resampled[a] != resampled[b]).mean()
        report.append((a, b, disagreement))
        if disagreement < tol and chosen == N_candidates[-1]:
            chosen = a
    return chosen, report


def plot_lookup_table(theta_dot_grid, alpha_grid, steps_to_stand, policy_alpha):
    """Build the lookup-table figure and leave it open (no save, no close)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    ax = axes[0]
    max_n = max(int(steps_to_stand.max()), 1)
    cmap = plt.get_cmap("viridis", max_n + 1)
    unclassified = steps_to_stand < 0
    ax.scatter(
        theta_dot_grid[~unclassified],
        steps_to_stand[~unclassified],
        c=steps_to_stand[~unclassified],
        cmap=cmap,
        vmin=0,
        vmax=max_n,
        s=10,
    )
    if unclassified.any():
        ax.scatter(
            theta_dot_grid[unclassified],
            np.full(unclassified.sum(), -0.5),
            color="crimson",
            s=10,
            label="unrecoverable (no valid alpha)",
        )
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel(r"$\dot\theta_k$ at mid-stance (rad/s)")
    ax.set_ylabel("steps to standstill")
    ax.set_title("Backward-induction policy: steps to standstill")

    ax2 = axes[1]
    classified = steps_to_stand >= 1
    sc = ax2.scatter(
        theta_dot_grid[classified],
        policy_alpha[classified],
        c=steps_to_stand[classified],
        cmap=cmap,
        vmin=0,
        vmax=max_n,
        s=10,
    )
    ax2.axhline(alpha_grid.min(), color="0.7", linestyle="--", linewidth=0.8)
    ax2.axhline(alpha_grid.max(), color="0.7", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(r"$\dot\theta_k$ at mid-stance (rad/s)")
    ax2.set_ylabel(r"policy $\alpha$ (rad)")
    ax2.set_title("Chosen step angle of attack")
    fig.colorbar(sc, ax=ax2, label="steps to standstill")

    return fig


def plot_alpha_vs_state(theta_dot_grid, alpha_grid, steps_to_stand, policy_alpha):
    """Standalone control-input plot: the lookup table's policy alpha_k
    (the control input, chosen once per step) as a function of the
    mid-stance state theta_dot_k. Same data as the right-hand panel of
    plot_lookup_table, split out on its own since alpha-vs-state is a
    distinct input/state relationship worth showing by itself. Built and
    left open (no save, no close).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    max_n = max(int(steps_to_stand.max()), 1)
    cmap = plt.get_cmap("viridis", max_n + 1)
    classified = steps_to_stand >= 1
    sc = ax.scatter(
        theta_dot_grid[classified], policy_alpha[classified],
        c=steps_to_stand[classified], cmap=cmap, vmin=0, vmax=max_n, s=16,
    )
    ax.axhline(alpha_grid.min(), color="0.6", linestyle="--", linewidth=1.0, label=r"$\alpha$ bounds")
    ax.axhline(alpha_grid.max(), color="0.6", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"state $\dot\theta_k$ at mid-stance (rad/s)")
    ax.set_ylabel(r"control input $\alpha$ (rad)")
    ax.set_title(r"Control policy: $\alpha$ vs. mid-stance state $\dot\theta_k$")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, label="steps to standstill")
    return fig


def find_boundary_theta_dots(params, roa_capture, M, coarse_N, fine_N, max_points=4):
    """Locate mid-stance velocities that build_lookup_table classifies
    differently at ``coarse_N`` vs. a much finer ``fine_N`` -- i.e. the
    specific initial conditions a coarser grid gets wrong, found
    principled-ly rather than by guessing round numbers. Mismatches cluster
    right at classification-boundary points (where steps_to_stand flips
    from one integer to the next), so contiguous runs of disagreement in
    the fine grid are grouped and one representative theta_dot is taken
    from each run, spreading the returned points across the swept range.
    """
    grid_fine, _, steps_fine, _ = build_lookup_table(params, roa_capture, N=fine_N, M=M)
    grid_coarse, _, steps_coarse, _ = build_lookup_table(params, roa_capture, N=coarse_N, M=M)
    idx_coarse = np.clip(
        np.round(grid_fine / grid_fine.max() * (len(grid_coarse) - 1)).astype(int),
        0, len(grid_coarse) - 1,
    )
    mismatch_idx = np.where(steps_fine != steps_coarse[idx_coarse])[0]
    if not len(mismatch_idx):
        return []
    groups = np.split(mismatch_idx, np.where(np.diff(mismatch_idx) > 1)[0] + 1)
    return [float(grid_fine[g[0]]) for g in groups][:max_points]


def grid_resolution_study(params, roa_capture, N_candidates, M, sample_theta_dots):
    """For each candidate grid size N, build the full lookup table and read
    off the steps-to-standstill classification for each of
    ``sample_theta_dots`` (nearest-grid-point). Returns
    {theta_dot_0: [steps_at_N0, steps_at_N1, ...]} in the same order as
    N_candidates -- the raw numbers behind the grid-resolution convergence
    plot, and the evidence for "a lower resolution would not have been
    good enough" (whichever sample still disagrees with the finest grid at
    the next-coarser N).
    """
    results = {td0: [] for td0 in sample_theta_dots}
    for N in N_candidates:
        grid, _, steps, _ = build_lookup_table(params, roa_capture, N=N, M=M)
        for td0 in sample_theta_dots:
            idx = int(np.argmin(np.abs(grid - td0)))
            results[td0].append(int(steps[idx]))
    return results


def plot_grid_resolution_study(N_candidates, sample_theta_dots, results):
    """Grid-resolution convergence study: for several representative
    mid-stance velocities theta_dot_0, show how their steps-to-standstill
    classification changes as the lookup-table grid size N (the
    discretization of the theta_dot "state" axis) is refined. The curves
    should visibly flatten out by the chosen resolution; anywhere a
    lower-N point still disagrees with the finest grid is exactly where
    that resolution would not yet have been good enough. Built and left
    open (no save, no close).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    cmap = plt.get_cmap("viridis", max(len(sample_theta_dots), 2))
    for i, td0 in enumerate(sample_theta_dots):
        ax.plot(N_candidates, results[td0], "o-", color=cmap(i),
                 label=fr"$\dot\theta_0 \approx$ {td0:.3f} rad/s")
    ax.set_xscale("log")
    ax.set_xticks(N_candidates)
    ax.set_xticklabels([str(n) for n in N_candidates])
    ax.set_xlabel("grid size $N$ (state-axis resolution)")
    ax.set_ylabel("steps to standstill (classification)")
    ax.set_title("Grid-resolution convergence for representative initial conditions")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def lookup_policy(theta_dot_k, theta_dot_grid, policy_alpha):
    """Nearest-grid-point alpha for a given mid-stance velocity."""
    idx = int(np.argmin(np.abs(theta_dot_grid - theta_dot_k)))
    return policy_alpha[idx]


def choose_example_ic(theta_dot_grid, steps_to_stand, min_steps=3):
    """Pick a genuine (not guessed) initial condition that needs at least
    ``min_steps`` steps to reach the RoA, by reading the lookup table's own
    ``steps_to_stand`` rather than hand-picking theta_dot_0 and hoping.

    Uses the worst case actually present in the swept table: since no swept
    state needs more steps than this, its own steps_to_stand value *is* "the
    maximum number of steps the walker can continue walking before reaching
    the RoA" for that initial condition (there's nothing in the sweep that
    takes longer). Returns (theta_dot_0, steps_needed).
    """
    max_steps = int(steps_to_stand.max())
    if max_steps < min_steps:
        raise ValueError(
            f"No swept initial condition needs >= {min_steps} steps; the "
            f"worst case found is {max_steps} steps. Widen the swept range "
            "or refine the grid."
        )
    candidates = theta_dot_grid[steps_to_stand == max_steps]
    idx = int(np.argmin(np.abs(theta_dot_grid - np.median(candidates))))
    return float(theta_dot_grid[idx]), max_steps


def plot_steps_to_standstill(theta_dot_grid, steps_to_stand, roa_capture, theta_dot_0, steps_needed):
    """Standalone visualization of how many steps it takes to reach
    standstill, as a function of the mid-stance initial condition theta_dot_0
    -- i.e. steps_to_stand read directly off the lookup table, with the
    direct-capture boundary and the chosen example initial condition marked.
    Built and left open (no save, no close).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    max_n = max(int(steps_to_stand.max()), 1)
    cmap = plt.get_cmap("viridis", max_n + 1)
    unclassified = steps_to_stand < 0
    sc = ax.scatter(
        theta_dot_grid[~unclassified],
        steps_to_stand[~unclassified],
        c=steps_to_stand[~unclassified],
        cmap=cmap,
        vmin=0,
        vmax=max_n,
        s=14,
    )
    if unclassified.any():
        ax.scatter(
            theta_dot_grid[unclassified],
            np.full(unclassified.sum(), -0.5),
            color="crimson",
            s=14,
            label="unrecoverable (no valid alpha)",
        )
    ax.axvline(roa_capture, color="0.5", linestyle="--", linewidth=1.0, label="direct-capture speed")
    ax.plot(
        [theta_dot_0], [steps_needed], "*", color="black", markersize=18,
        markeredgecolor="white", zorder=5,
        label=f"example IC: {theta_dot_0:.3f} rad/s -> {steps_needed} steps",
    )
    ax.set_xlabel(r"$\dot\theta_0$ at mid-stance (rad/s)")
    ax.set_ylabel("steps to standstill")
    ax.set_title("Steps to reach standstill vs. mid-stance initial condition")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax, label="steps to standstill")
    return fig


def count_steps_taken(trajectory):
    """Number of touchdown events (foot-strikes) that occurred while the
    walker was still stepping (before the controller switched to
    'balance'), found from jumps in stance_x, plus their simulation times.
    Compares against the lookup table's predicted steps_needed as a sanity
    check that the closed-loop simulation matches the Poincare-map policy.
    """
    modes = np.asarray(trajectory["mode"])
    stance_x = trajectory["stance_x"]
    t = trajectory["t"]
    jumps = np.where(np.diff(stance_x) > 1e-9)[0] + 1
    touchdown_modes = modes[jumps - 1]  # mode of the stance phase that just ended
    stepping = touchdown_modes == "step"
    n_steps = int(np.sum(stepping))
    touchdown_times = t[jumps][stepping]
    return n_steps, touchdown_times


# ==============================================================================
# Part 3: Full hybrid simulation
# ==============================================================================
def simulate(
    theta_dot_0,
    params,
    theta_grid,
    theta_dot_grid_roa,
    roa_mask,
    theta_dot_grid_table,
    policy_alpha,
    dt=0.001,
    t_max=15.0,
    integrator=default_integrator,
):
    """Simulate from mid-stance (theta=0, theta_dot=theta_dot_0), stepping
    with the lookup-table policy until the state enters the ankle
    controller's RoA, then balancing. Records theta, theta_dot, ankle_torque,
    angle_of_attack, mode ('step'/'balance'), and touchdown times/positions
    (translated forward in x by the step length) for animation."""
    p = dict(params)
    state = np.array([0.0, theta_dot_0])
    t = 0.0

    mode = "step"
    alpha = lookup_policy(theta_dot_0, theta_dot_grid_table, policy_alpha)
    if np.isnan(alpha):
        raise ValueError(
            f"theta_dot_0={theta_dot_0} is in the unrecoverable set of the lookup "
            "table -- pick another initial condition."
        )
    p["angle_of_attack"] = alpha
    p["ankle_torque"] = 0.0

    stance_x = 0.0  # x-position of the current stance foot, for the animation
    seen_midstance = True  # we start exactly at mid-stance

    times, thetas, theta_dots, torques, alphas, modes, stance_xs = [], [], [], [], [], [], []

    n_steps = int(round(t_max / dt))
    for _ in range(n_steps):
        times.append(t)
        thetas.append(state[0])
        theta_dots.append(state[1])
        torques.append(p["ankle_torque"])
        alphas.append(p["angle_of_attack"])
        modes.append(mode)
        stance_xs.append(stance_x)

        if mode == "balance":
            p["ankle_torque"] = float(ankle_control(state, p))
            state = integrator.step(dynamics, t, state, dt, p)
        else:
            p["ankle_torque"] = 0.0
            prev_state = state
            state = integrator.step(dynamics, t, state, dt, p)

            if not seen_midstance and prev_state[0] < 0.0 <= state[0]:
                seen_midstance = True
                alpha = lookup_policy(state[1], theta_dot_grid_table, policy_alpha)
                if np.isnan(alpha):
                    raise ValueError(
                        f"Reached an unrecoverable mid-stance velocity {state[1]:.4f} "
                        "rad/s -- the walker cannot avoid stumbling with the allowed "
                        "alpha range."
                    )
                p["angle_of_attack"] = alpha

            if event_guard(prev_state, state, p):
                # Touchdown: apply the impact map and start a new stance phase.
                foot_step = 2 * p["length"] * np.sin(p["angle_of_attack"]) * np.cos(state[0] - p["angle_of_attack"])
                # (chord length between the two feet, projected onto x; see note below)
                state = event_dynamics(state, p)
                stance_x += foot_step
                seen_midstance = False

            if in_roa(state[0], state[1], theta_grid, theta_dot_grid_roa, roa_mask):
                mode = "balance"

        t += dt

    return {
        "t": np.array(times),
        "theta": np.array(thetas),
        "theta_dot": np.array(theta_dots),
        "torque": np.array(torques),
        "alpha": np.array(alphas),
        "mode": modes,
        "stance_x": np.array(stance_xs),
    }


def animate(trajectory, params, frame_dt=1.0 / 30, tail_seconds=2.0):
    """Build the live walking animation (matplotlib.animation.FuncAnimation)
    and return it. Does NOT call plt.show() itself -- main() does that once,
    after building this and all 4 static plots, so every window opens
    together. You must keep the returned object alive (assign it to a
    variable) until after plt.show() returns, or it can be garbage-collected
    mid-playback and the animation will freeze.
    """
    t = trajectory["t"]
    t_end = t[-1]
    # Stop the animation shortly after the walker settles, rather than
    # padding out the full t_max of balancing.
    settle_idx = np.where(np.array(trajectory["mode"]) == "balance")[0]
    if settle_idx.size:
        t_end = min(t_end, t[settle_idx[0]] + tail_seconds)

    frame_times = np.arange(0.0, t_end, frame_dt)
    frame_idx = np.searchsorted(t, frame_times)
    frame_idx = np.clip(frame_idx, 0, len(t) - 1)

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    x_span = trajectory["stance_x"][frame_idx[-1]] + 2.5 * params["length"]
    view_limits = (-1.5 * params["length"], x_span, -1.2 * params["length"], 1.6 * params["length"])

    def draw(i):
        idx = frame_idx[i]
        state = np.array([trajectory["theta"][idx], trajectory["theta_dot"][idx]])
        p = dict(params)
        p["angle_of_attack"] = trajectory["alpha"][idx]
        p["ankle_torque"] = trajectory["torque"][idx]
        show_swing = trajectory["mode"][idx] == "step"
        visualize(
            state,
            p,
            ax=ax,
            show_swing=show_swing,
            stance_position=(trajectory["stance_x"][idx], 0.0),
            view_limits=view_limits,
        )
        ax.set_title(f"t = {t[idx]:.2f} s   [{trajectory['mode'][idx]}]")
        return (ax,)

    # Caller (main()) must keep this reference alive until plt.show() returns;
    # otherwise it can get garbage-collected mid-playback and freeze.
    anim = FuncAnimation(
        fig, draw, frames=len(frame_idx), interval=1000 * frame_dt, blit=False, repeat=False,
    )
    return anim


def plot_trajectory_diagnostics(trajectory, touchdown_times, steps_predicted):
    """As before (theta, theta_dot, torque vs. time), plus a marker + label
    at each touchdown that occurred while still stepping, and a text box
    comparing the simulated step count against the lookup table's
    prediction -- this is "the maximum number of steps the walker can
    continue walking before reaching the RoA" for this initial condition,
    made visible on the trajectory itself. Built and left open (no save, no
    close).
    """
    n_steps = len(touchdown_times)
    fig, axes = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True, layout="constrained")
    axes[0].plot(trajectory["t"], trajectory["theta"], color="#23699b")
    axes[0].set_ylabel(r"$\theta$ (rad)")
    axes[1].plot(trajectory["t"], trajectory["theta_dot"], color="#df8a25")
    axes[1].set_ylabel(r"$\dot\theta$ (rad/s)")
    axes[2].plot(trajectory["t"], trajectory["torque"], color="#3f7f3f")
    axes[2].set_ylabel(r"$\tau$ (N m)")
    axes[2].set_xlabel("time (s)")

    for ax in axes:
        ax.grid(alpha=0.3)
        for td_t in touchdown_times:
            ax.axvline(td_t, color="0.55", linestyle="--", linewidth=0.9)

    axes[0].text(
        0.02,
        0.03,
        f"Reached RoA after {n_steps} steps\n(lookup table predicted {steps_predicted})",
        transform=axes[0].transAxes,
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    # The touchdowns are bunched together relative to the full 0..t_max view
    # (the walker is only "stepping" for the first second or so before it
    # settles), so per-line "step N" labels on the main panel would just
    # overlap. Zoom into that window in an inset instead, where there's
    # room to label each step individually.
    if n_steps:
        pad = 0.15 * max(touchdown_times[-1], 1e-3)
        window = (trajectory["t"] >= 0) & (trajectory["t"] <= touchdown_times[-1] + pad)
        inset = axes[0].inset_axes([0.42, 0.42, 0.56, 0.55])
        inset.plot(trajectory["t"][window], trajectory["theta"][window], color="#23699b")
        y0, y1 = inset.get_ylim()
        for i, td_t in enumerate(touchdown_times, start=1):
            inset.axvline(td_t, color="0.55", linestyle="--", linewidth=0.9)
            inset.annotate(
                str(i),
                xy=(td_t, y1),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=7,
                color="0.3",
                ha="center",
            )
        inset.set_xlim(0, touchdown_times[-1] + pad)
        inset.set_title("steps (zoom)", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.3)

    fig.suptitle("Full hybrid controller: state and ankle torque vs. time")
    return fig


def poincare_crossings(trajectory):
    """Find where the trajectory crosses the Poincare section (theta = 0,
    mid-stance) while stepping -- the same event simulate() uses internally
    to trigger a lookup-table re-query, recovered here from the recorded
    theta/mode arrays instead. Returns (indices, times, theta_dots) for each
    crossing, in chronological order.
    """
    theta = trajectory["theta"]
    modes = np.asarray(trajectory["mode"])
    prev_theta, curr_theta = theta[:-1], theta[1:]
    crossing = (prev_theta < 0.0) & (curr_theta >= 0.0) & (modes[:-1] == "step")
    idx = np.where(crossing)[0] + 1  # sample just after the crossing, theta ~ 0
    return idx, trajectory["t"][idx], trajectory["theta_dot"][idx]


def plot_poincare_slice(trajectory):
    """Phase-space (theta, theta_dot) view of the trajectory, with the
    Poincare section (theta = 0, mid-stance) drawn as a vertical line and
    each crossing of it marked and numbered in sequence -- these numbered
    points are exactly the theta_dot_k samples the lookup table's return map
    (poincare_step / build_lookup_table) operates on. Built and left open
    (no save, no close).
    """
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")

    theta = trajectory["theta"]
    theta_dot = trajectory["theta_dot"]
    modes = np.asarray(trajectory["mode"])

    # Full phase trajectory. Plotted as separate stepping/balancing segments
    # so the color communicates which controller was active; the impact map
    # shows up naturally as the near-vertical jumps in theta at each
    # touchdown, since theta and theta_dot both reset discontinuously there.
    stepping = modes == "step"
    ax.plot(theta[stepping], theta_dot[stepping], color="#df8a25", linewidth=1.2, label="Stepping")
    ax.plot(theta[~stepping], theta_dot[~stepping], color="#23699b", linewidth=1.2, label="Balancing")

    # The Poincare section itself.
    ax.axvline(0, color="0.4", linestyle="--", linewidth=1.2, label=r"Poincare section ($\theta=0$, mid-stance)")

    # Section crossings -- the actual discrete return-map samples.
    idx, t_cross, theta_dot_cross = poincare_crossings(trajectory)
    if len(idx):
        sc = ax.scatter(
            np.zeros_like(theta_dot_cross), theta_dot_cross,
            c=np.arange(1, len(idx) + 1), cmap="viridis",
            s=70, zorder=5, edgecolor="white", linewidth=1.0,
            label="Section crossings ($\\dot\\theta_k$)",
        )
        for k, td in enumerate(theta_dot_cross, start=1):
            ax.annotate(str(k), xy=(0, td), xytext=(8, 0), textcoords="offset points", fontsize=9, va="center")
        fig.colorbar(sc, ax=ax, label="crossing index $k$")

    ax.plot(0, 0, "k*", markersize=14, zorder=6, label="Standing equilibrium")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_title("Phase trajectory and Poincare section ($\\theta=0$)")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_poincare_sequence(trajectory):
    """Poincare-section return-map sequence: crossing index k (x-axis) vs.
    the mid-stance velocity theta_dot_k (y-axis) recorded at that crossing
    -- i.e. the actual discrete samples the lookup table's return map
    (poincare_step / build_lookup_table) is built to predict, laid out in
    the order they occurred. k=0 is the initial condition itself (the
    simulation starts exactly at mid-stance), and each subsequent k is a
    later mid-stance crossing. Built and left open (no save, no close).
    """
    idx, t_cross, theta_dot_cross = poincare_crossings(trajectory)
    k = np.arange(1, len(idx) + 1)
    k_all = np.concatenate([[0], k])
    theta_dot_all = np.concatenate([[trajectory["theta_dot"][0]], theta_dot_cross])

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.plot(k_all, theta_dot_all, "o-", color="#23699b", markersize=7, zorder=3)
    for kk, td in zip(k_all, theta_dot_all):
        ax.annotate(
            f"{td:.3f}", xy=(kk, td), xytext=(0, 9), textcoords="offset points",
            fontsize=8, ha="center", color="0.3",
        )
    ax.axhline(0, color="0.7", linewidth=0.8, zorder=1)
    ax.set_xlabel("crossing index $k$")
    ax.set_ylabel(r"$\dot\theta_k$ at mid-stance (rad/s)")
    ax.set_xticks(k_all)
    ax.set_title("Poincare section sequence: mid-stance velocity per step")
    ax.grid(alpha=0.3)
    return fig


# ==============================================================================
# Main
# ==============================================================================
def run_pipeline():
    """Run the full assignment pipeline -- RoA search, lookup-table
    construction with a grid-resolution convergence study (both the
    whole-grid disagreement check and a per-initial-condition study),
    example initial-condition selection, and the closed-loop simulation --
    and return every intermediate result needed to build all of the
    report's plots. Builds no figures and shows/saves nothing itself, so
    main() (which shows everything interactively) and
    generate_report_figures.py (which saves everything to disk) both call
    this once and are guaranteed to get identical numbers.
    """
    params = generate_params()
    print(f"incline (gamma) = {params['incline']} rad")
    print(f"alpha bounds     = {params['alpha_bounds']}")
    print(f"torque bounds    = {torque_bounds(params)} N*m")

    # --- Part 1: RoA of the ankle controller ---
    alpha_lo = params["alpha_bounds"][0]
    theta_box = alpha_lo + params["incline"]  # smallest possible touchdown angle
    print("\nGrid-searching ankle-controller RoA ...")
    theta_grid, theta_dot_grid_roa, roa_mask = compute_roa(
        params, (-theta_box, theta_box), (-2.0, 2.0), n_theta=101, n_theta_dot=101
    )
    roa_capture = roa_capture_speed(theta_grid, theta_dot_grid_roa, roa_mask)
    print(f"RoA fraction of grid: {roa_mask.mean():.3f}")
    print(f"Direct-capture speed at mid-stance (theta=0): {roa_capture:.4f} rad/s")

    # --- Part 2: lookup table, with a grid-resolution convergence check ---
    print("\nChecking lookup-table grid-resolution convergence ...")
    N_candidates = [51, 101, 201, 401, 801]
    M = 41
    chosen_N, grid_report = choose_grid_resolution(params, roa_capture, N_candidates, M=M, tol=0.01)
    for a, b, disagreement in grid_report:
        print(f"  N={a:4d} vs N={b:4d}: {disagreement*100:.3f}% of grid reclassified")
    print(f"Chosen coarsest-adequate grid: N={chosen_N} (theta_dot), M={M} (alpha)")

    theta_dot_grid_table, alpha_grid, steps_to_stand, policy_alpha = build_lookup_table(
        params, roa_capture, N=chosen_N, M=M
    )
    print(f"Max steps to standstill over swept range: {int(steps_to_stand.max())}")
    print(f"Unrecoverable fraction of swept range: {(steps_to_stand < 0).mean()*100:.2f}%")

    # --- Part 3: simulate an example initial condition ---
    # Read the initial condition off the lookup table instead of guessing:
    # this picks the worst case actually present in the swept grid, so its
    # own steps_to_stand *is* the maximum number of steps the walker can
    # continue walking before reaching the RoA for this initial condition.
    theta_dot_0, steps_predicted = choose_example_ic(theta_dot_grid_table, steps_to_stand, min_steps=3)
    print(
        f"\nChosen example initial condition: theta_dot_0 = {theta_dot_0:.4f} rad/s "
        f"at mid-stance -> lookup table predicts {steps_predicted} steps to reach the RoA "
        "(the worst case anywhere in the swept range)."
    )
    print(f"Simulating from theta_dot_0 = {theta_dot_0:.4f} rad/s at mid-stance ...")
    trajectory = simulate(
        theta_dot_0, params, theta_grid, theta_dot_grid_roa, roa_mask,
        theta_dot_grid_table, policy_alpha,
    )
    n_steps_sim, touchdown_times = count_steps_taken(trajectory)
    print(f"Simulated trajectory took {n_steps_sim} steps to reach the RoA (predicted {steps_predicted}).")
    print(f"Reached balancing mode at t = {trajectory['t'][np.array(trajectory['mode'])=='balance'][0]:.3f} s")

    # --- Grid-resolution study, per initial condition ---
    # The check above (chosen via choose_grid_resolution) reports what
    # fraction of the WHOLE grid gets reclassified between resolutions.
    # This instead tracks specific, representative initial conditions
    # individually across every candidate N, so we can show exactly which
    # states a too-coarse grid gets wrong -- not just an aggregate
    # percentage. The most informative points to track are the ones
    # sitting right at a classification boundary (found automatically by
    # comparing N=201 against a much finer N=801 reference), since that's
    # exactly where a coarser grid is at risk of giving the wrong answer;
    # the chosen example IC is included too, for continuity with the
    # trajectory plot.
    boundary_points = find_boundary_theta_dots(params, roa_capture, M, coarse_N=201, fine_N=801, max_points=4)
    sample_theta_dots = sorted({round(p, 4) for p in boundary_points} | {round(theta_dot_0, 4)})
    print(f"\nGrid-resolution study sample initial conditions (rad/s): {sample_theta_dots}")
    grid_study_results = grid_resolution_study(params, roa_capture, N_candidates, M, sample_theta_dots)
    for td0 in sample_theta_dots:
        row = grid_study_results[td0]
        print(f"  theta_dot_0={td0:.3f}: " + " ".join(f"N={n}->{s}" for n, s in zip(N_candidates, row)))

    return {
        "params": params,
        "theta_grid": theta_grid,
        "theta_dot_grid_roa": theta_dot_grid_roa,
        "roa_mask": roa_mask,
        "roa_capture": roa_capture,
        "N_candidates": N_candidates,
        "M": M,
        "chosen_N": chosen_N,
        "grid_report": grid_report,
        "theta_dot_grid_table": theta_dot_grid_table,
        "alpha_grid": alpha_grid,
        "steps_to_stand": steps_to_stand,
        "policy_alpha": policy_alpha,
        "theta_dot_0": theta_dot_0,
        "steps_predicted": steps_predicted,
        "trajectory": trajectory,
        "touchdown_times": touchdown_times,
        "n_steps_sim": n_steps_sim,
        "sample_theta_dots": sample_theta_dots,
        "grid_study_results": grid_study_results,
    }


def main():
    data = run_pipeline()

    plot_roa(data["theta_grid"], data["theta_dot_grid_roa"], data["roa_mask"])
    plot_lookup_table(data["theta_dot_grid_table"], data["alpha_grid"], data["steps_to_stand"], data["policy_alpha"])
    plot_alpha_vs_state(data["theta_dot_grid_table"], data["alpha_grid"], data["steps_to_stand"], data["policy_alpha"])
    plot_steps_to_standstill(
        data["theta_dot_grid_table"], data["steps_to_stand"], data["roa_capture"],
        data["theta_dot_0"], data["steps_predicted"],
    )
    plot_grid_resolution_study(data["N_candidates"], data["sample_theta_dots"], data["grid_study_results"])
    plot_trajectory_diagnostics(data["trajectory"], data["touchdown_times"], data["steps_predicted"])
    plot_poincare_slice(data["trajectory"])
    plot_poincare_sequence(data["trajectory"])

    # All 8 static figures above are already built (just not shown yet --
    # plt.subplots() alone doesn't pop up a window). Build the animation
    # too, then show everything -- 8 static plots + the live animation --
    # together with a single plt.show() call, which blocks until you close
    # all of the windows (or Ctrl+C).
    anim = animate(data["trajectory"], data["params"])  # noqa: F841 (must stay alive through plt.show())
    print("\nShowing all plots and the animation now (close the windows, or Ctrl+C, to exit) ...")
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()