"""
Brute-force estimate of the regions of attraction (RoA) for the rimless
wheel, by gridding the state space and simulating each grid point out to
(numerical) steady state.

State-space domain
-------------------
The stance leg is only physically in contact for
    theta in [gamma - alpha, gamma + alpha]
(that IS the wedge between the two guards), so that is the natural domain
to grid theta over. theta_dot is gridded over a symmetric range wide
enough to comfortably bracket the rolling limit cycle's fixed point.

Expected attractors (before looking at data)
----------------------------------------------
For gamma > 0 we expect exactly TWO attractors:
  (1) the stable forward-rolling limit cycle (downhill walking gait) --
      every impact loses a fixed KE fraction but every stance phase gains
      a fixed (speed-independent) amount of KE from rolling further
      downhill, and these balance at one particular speed (see README).
  (2) "falls over" / comes to rest -- insufficient energy to escape the
      wedge forward, so the wheel rocks back and forth, losing energy at
      every impact (both forward AND backward impacts dissipate), and
      decays toward the unstable balance point theta=thetadot=0.
A third, backward-rolling attractor is not physically expected to be
persistent (rolling backward means climbing the slope AND dissipating at
every impact -- purely double loss -- so it must decay), but we still
classify it as its own bucket in case a corner of parameter space (e.g.
very large N, tiny dissipation) reveals something unexpected.

Speed trick (per the assignment's hint)
------------------------------------------
A fixed tiny time-step (e.g. explicit Euler) needs a very small dt to
localize the impact accurately, but that dt is needlessly small during
the slow, smooth part of the swing. Instead we use adaptive-step RK45
(scipy.integrate.solve_ivp) whose *event* handling locates the guard
crossing via root-finding on the local polynomial interpolant of the last
accepted step -- so the impact time comes out to solver tolerance
regardless of step size, while the solver is free to take large steps
during the smooth swing. This is the single biggest speedup here (easily
10-100x fewer function evaluations than a fixed small-dt integrator at
matched accuracy). The grid evaluation is also embarrassingly parallel
across grid points, so we additionally parallelize with multiprocessing.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
from model import RimlessWheelParams, simulate

REST, FORWARD, BACKWARD, UNRESOLVED = 0, 1, 2, 3
LABELS = {REST: "at rest / fell over", FORWARD: "forward rolling limit cycle",
          BACKWARD: "backward rolling", UNRESOLVED: "unresolved"}
COLORS = {REST: "#cfcfcf", FORWARD: "#2a6f97", BACKWARD: "#c1440e", UNRESOLVED: "#f2c14e"}


def classify_trajectory(x0, p: RimlessWheelParams, t_max=15.0, max_steps=40,
                         v_rest_tol=5e-3, v_roll_tol=0.05):
    result, status = simulate(x0, p, t_max=t_max, max_steps=max_steps)
    if len(result.events) == 0:
        # never even reached the first guard -> stalled immediately
        final_speed = result.x[-1][1, -1]
    else:
        final_speed = result.x[-1][1, -1]

    if abs(final_speed) < v_rest_tol:
        return REST
    # Require a consistent-sign, non-tiny tail of post-impact speeds to
    # call it a genuine sustained rolling limit cycle (not a transient).
    tail = result.events[-5:] if len(result.events) >= 5 else result.events
    if len(tail) >= 3 and abs(final_speed) > v_roll_tol:
        if final_speed > 0 and all(e == "forward" for e in tail):
            return FORWARD
        if final_speed < 0 and all(e == "backward" for e in tail):
            return BACKWARD
    if abs(final_speed) < v_roll_tol:
        return REST
    return UNRESOLVED


def _worker(args):
    theta0, thetadot0, p = args
    return classify_trajectory([theta0, thetadot0], p)


def compute_roa_grid(p: RimlessWheelParams, n_theta=61, n_thetadot=61,
                      thetadot_max=8.0, n_workers=8, margin=0.0):
    theta_lo = p.backward_guard_angle() + margin
    theta_hi = p.forward_guard_angle() - margin
    thetas = np.linspace(theta_lo, theta_hi, n_theta)
    thetadots = np.linspace(-thetadot_max, thetadot_max, n_thetadot)
    TH, THD = np.meshgrid(thetas, thetadots, indexing="ij")

    jobs = [(TH[i, j], THD[i, j], p) for i in range(n_theta) for j in range(n_thetadot)]
    with Pool(n_workers) as pool:
        labels_flat = pool.map(_worker, jobs)
    labels = np.array(labels_flat).reshape(n_theta, n_thetadot)
    return thetas, thetadots, labels


def plot_roa(thetas, thetadots, labels, p: RimlessWheelParams, savepath):
    fig, ax = plt.subplots(figsize=(7, 6))
    present = sorted(np.unique(labels))
    cmap = ListedColormap([COLORS[k] for k in present])
    remap = {k: i for i, k in enumerate(present)}
    remapped = np.vectorize(remap.get)(labels)
    im = ax.pcolormesh(np.degrees(thetas), thetadots, remapped.T, cmap=cmap,
                        shading="nearest", vmin=-0.5, vmax=len(present) - 0.5)
    ax.axvline(np.degrees(p.gamma), color="k", lw=0.8, ls=":", label=r"$\theta=\gamma$ (vertical)")
    ax.set_xlabel(r"$\theta$ [deg]  (stance-leg angle from vertical)")
    ax.set_ylabel(r"$\dot\theta$ [rad/s]")
    ax.set_title(f"Region-of-attraction map  (N={p.N}, $\\gamma$={np.degrees(p.gamma):.1f}$^\\circ$)\n"
                 f"grid over the physical wedge $\\theta\\in[\\gamma-\\alpha,\\gamma+\\alpha]$")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in present]
    ax.legend(handles, [LABELS[k] for k in present], loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"saved {savepath}")


if __name__ == "__main__":
    p = RimlessWheelParams(N=8, gamma=0.15, l=1.0)
    thetas, thetadots, labels = compute_roa_grid(p, n_theta=71, n_thetadot=71, thetadot_max=8.0)
    plot_roa(thetas, thetadots, labels, p, "figures/roa_map.png")

    from collections import Counter
    counts = Counter(labels.flatten().tolist())
    total = labels.size
    for k, v in counts.items():
        print(f"  {LABELS[k]:30s}: {v:5d} / {total} grid points ({100*v/total:.1f}%)")