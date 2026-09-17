from pathlib import Path

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

OUTPUT_DIR = os.path.join("output", "assignment_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def plot_roa(theta_grid, theta_dot_grid, roa_mask, path):
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
    fig.savefig(path, dpi=160)
    plt.close(fig)


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


def plot_lookup_table(theta_dot_grid, alpha_grid, steps_to_stand, policy_alpha, path):
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

    fig.savefig(path, dpi=160)
    plt.close(fig)


def lookup_policy(theta_dot_k, theta_dot_grid, policy_alpha):
    """Nearest-grid-point alpha for a given mid-stance velocity."""
    idx = int(np.argmin(np.abs(theta_dot_grid - theta_dot_k)))
    return policy_alpha[idx]


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
    """Play the simulated trajectory back live in a matplotlib window.

    Opens an interactive animation (matplotlib.animation.FuncAnimation) and
    blocks until you close the window. Nothing is written to disk -- if you
    want a saved copy later, this is the function to point at a writer
    (e.g. anim.save('walker.gif', writer=PillowWriter(fps=...))).
    """
    t = trajectory["t"]
    t_end = t[-1]
    # Stop the animation shortly after the walker settles, rather than
    # padding out the full t_max of balancing.git 
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

    # Keep a reference to the animation alive for the duration of plt.show();
    # otherwise it can get garbage-collected mid-playback.
    anim = FuncAnimation(
        fig, draw, frames=len(frame_idx), interval=1000 * frame_dt, blit=False, repeat=False,
    )
    plt.show()
    return anim


def plot_trajectory_diagnostics(trajectory, path):
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, layout="constrained")
    axes[0].plot(trajectory["t"], trajectory["theta"], color="#23699b")
    axes[0].set_ylabel(r"$\theta$ (rad)")
    axes[1].plot(trajectory["t"], trajectory["theta_dot"], color="#df8a25")
    axes[1].set_ylabel(r"$\dot\theta$ (rad/s)")
    axes[2].plot(trajectory["t"], trajectory["torque"], color="#3f7f3f")
    axes[2].set_ylabel(r"$\tau$ (N m)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Full hybrid controller: state and ankle torque vs. time")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
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
    plot_roa(theta_grid, theta_dot_grid_roa, roa_mask, os.path.join(OUTPUT_DIR, "roa.png"))
    roa_capture = roa_capture_speed(theta_grid, theta_dot_grid_roa, roa_mask)
    print(f"RoA fraction of grid: {roa_mask.mean():.3f}")
    print(f"Direct-capture speed at mid-stance (theta=0): {roa_capture:.4f} rad/s")

    # --- Part 2: lookup table, with a grid-resolution convergence check ---
    print("\nChecking lookup-table grid-resolution convergence ...")
    N_candidates = [51, 101, 201, 401, 801]
    chosen_N, report = choose_grid_resolution(params, roa_capture, N_candidates, M=41, tol=0.01)
    for a, b, disagreement in report:
        print(f"  N={a:4d} vs N={b:4d}: {disagreement*100:.3f}% of grid reclassified")
    print(f"Chosen coarsest-adequate grid: N={chosen_N} (theta_dot), M=41 (alpha)")

    theta_dot_grid_table, alpha_grid, steps_to_stand, policy_alpha = build_lookup_table(
        params, roa_capture, N=chosen_N, M=41
    )
    plot_lookup_table(
        theta_dot_grid_table, alpha_grid, steps_to_stand, policy_alpha,
        os.path.join(OUTPUT_DIR, "lookup_table.png"),
    )
    print(f"Max steps to standstill over swept range: {int(steps_to_stand.max())}")
    print(f"Unrecoverable fraction of swept range: {(steps_to_stand < 0).mean()*100:.2f}%")

    # --- Part 3: simulate + animate an example initial condition ---
    theta_dot_0 = 2.0  # rad/s at mid-stance -- needs a couple of steps, per the table
    print(f"\nSimulating from theta_dot_0 = {theta_dot_0} rad/s at mid-stance ...")
    trajectory = simulate(
        theta_dot_0, params, theta_grid, theta_dot_grid_roa, roa_mask,
        theta_dot_grid_table, policy_alpha,
    )
    n_impacts = int(np.sum(np.diff((np.array(trajectory["mode"]) == "step").astype(int)) != 0))
    print(f"Reached balancing mode at t = {trajectory['t'][np.array(trajectory['mode'])=='balance'][0]:.3f} s")

    plot_trajectory_diagnostics(trajectory, os.path.join(OUTPUT_DIR, "trajectory.png"))

    print("Opening animation window (close it to exit) ...")
    animate(trajectory, params)


if __name__ == "__main__":
    main()