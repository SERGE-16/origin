
"""
Sweeps over the slope gamma and the number of spokes N.

For each parameter setting we report:
  - whether a rolling limit cycle exists at all (see "critical slope" note
    below -- this is the headline finding of the gamma sweep)
  - the limit cycle's fixed-point speed v*
  - its Floquet multiplier (closed-form AND finite-difference numeric)
  - a coarse-grid estimate of the rolling basin's share of state space

Critical slope
--------------
The closed-form fixed point v*(gamma) -> 0 as gamma -> 0 (no slope, no
energy input), while the escape-speed threshold v_thresh(gamma) (minimum
speed needed to clear the theta=0 "hump" in one swing, see poincare.py)
stays roughly constant until gamma approaches alpha. So for small gamma,
v*(gamma) < v_thresh(gamma): the "solution" predicted by naive energy
conservation is not self-consistent (a wheel moving at that speed would
NOT actually clear the hump), which means no simple one-hop rolling limit
cycle exists there at all. There is therefore a CRITICAL SLOPE gamma_c(N)
below which the wheel cannot sustain any rolling gait -- every initial
condition eventually falls to rest -- and only above which a stable
rolling limit cycle (and a nonzero rolling basin of attraction) appears.
We locate gamma_c by bisecting on v*(gamma) - v_thresh(gamma) = 0.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from rootfind import bisect
from model import RimlessWheelParams
from poincare import (closed_form_fixed_point, closed_form_floquet, closed_form_return_map,
                       escape_speed_threshold, find_fixed_point_numeric, estimate_floquet_multiplier)
from roa import compute_roa_grid, FORWARD


def critical_gamma(N, l=1.0, g=9.81, lo=1e-4, hi=None):
    alpha = np.pi / N
    if hi is None:
        hi = alpha * 0.999  # gamma_c is always < alpha (at gamma=alpha, threshold=0 < v*, self-consistent)

    def f(gamma):
        p = RimlessWheelParams(N=N, gamma=gamma, l=l, g=g)
        return closed_form_fixed_point(p) - escape_speed_threshold(p)

    if f(lo) > 0:
        return 0.0  # already self-consistent at essentially zero slope (shouldn't happen for finite N)
    if f(hi) < 0:
        return hi
    return bisect(f, lo, hi)


def analyze_one(p: RimlessWheelParams, roa_grid_n=23, roa_vmax_factor=2.5, n_workers=2):
    v_thresh = escape_speed_threshold(p)
    v_star_closed = closed_form_fixed_point(p)
    self_consistent = v_star_closed > v_thresh

    out = dict(gamma=p.gamma, N=p.N, alpha=p.alpha, v_thresh=v_thresh,
               v_star_closed=v_star_closed, self_consistent=self_consistent)

    if self_consistent:
        try:
            v_star_num = find_fixed_point_numeric(p, v_hi=max(6.0, 3 * v_star_closed))
        except ValueError:
            v_star_num = np.nan
        lam_closed = closed_form_floquet(p)
        if not np.isnan(v_star_num):
            eps = min(0.3 * (v_star_num - v_thresh), 0.05 * v_star_num)
            eps = max(eps, 1e-5)
            lam_num = estimate_floquet_multiplier(p, v_star_num, eps=eps, use_general=True)
        else:
            lam_num = np.nan
        out.update(v_star_numeric=v_star_num, lam_closed=lam_closed, lam_numeric=lam_num)
    else:
        out.update(v_star_numeric=np.nan, lam_closed=np.nan, lam_numeric=np.nan)

    thetadot_max = roa_vmax_factor * max(v_star_closed, 1.0)
    thetas, thetadots, labels = compute_roa_grid(
        p, n_theta=roa_grid_n, n_thetadot=roa_grid_n, thetadot_max=thetadot_max, n_workers=n_workers)
    rolling_fraction = np.mean(labels == FORWARD)
    out["rolling_fraction"] = rolling_fraction
    return out


def sweep_gamma(N=8, gammas=None, l=1.0):
    alpha = np.pi / N
    gc = critical_gamma(N, l=l)
    if gammas is None:
        gammas = np.concatenate([
            np.linspace(0.3 * gc, 0.9 * gc, 3),
            np.linspace(1.05 * gc, alpha * 1.3, 8),
        ])
    rows = []
    for gamma in gammas:
        p = RimlessWheelParams(N=N, gamma=gamma, l=l)
        row = analyze_one(p)
        rows.append(row)
        print(f"  gamma={np.degrees(gamma):5.2f}deg  self_consistent={row['self_consistent']!s:5}  "
              f"v*={row['v_star_numeric']:.3f}  lam_closed={row['lam_closed']:.3f}  "
              f"lam_num={row['lam_numeric']:.3f}  rolling_frac={row['rolling_fraction']:.2f}")
    return gc, rows


def sweep_N(gamma=0.18, Ns=range(6, 13), l=1.0):
    rows = []
    for N in Ns:
        p = RimlessWheelParams(N=N, gamma=gamma, l=l)
        gc = critical_gamma(N, l=l)
        row = analyze_one(p)
        row["gamma_critical"] = gc
        rows.append(row)
        print(f"  N={N:2d}  alpha={np.degrees(row['alpha']):5.2f}deg  gamma_c={np.degrees(gc):5.2f}deg  "
              f"v*={row['v_star_numeric']:.3f}  lam_closed={row['lam_closed']:.3f}  "
              f"lam_num={row['lam_numeric']:.3f}  rolling_frac={row['rolling_fraction']:.2f}")
    return rows


def plot_gamma_sweep(N, gc, rows, savepath):
    gammas_deg = np.degrees([r["gamma"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(gammas_deg, [r["rolling_fraction"] for r in rows], "o-", color="#2a6f97")
    ax.axvline(np.degrees(gc), color="crimson", ls="--", lw=1.2, label=r"critical slope $\gamma_c$")
    ax.set_xlabel(r"$\gamma$ [deg]"); ax.set_ylabel("rolling-gait basin fraction of grid")
    ax.set_title("RoA size vs slope"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(gammas_deg, [r["v_star_closed"] for r in rows], "-", color="0.7", lw=4, alpha=0.6, label="closed-form $v^*$")
    ax.plot(gammas_deg, [r["v_star_numeric"] for r in rows], "o-", color="#2a6f97", ms=4, label="numeric $v^*$")
    ax.axvline(np.degrees(gc), color="crimson", ls="--", lw=1.2)
    ax.set_xlabel(r"$\gamma$ [deg]"); ax.set_ylabel(r"limit-cycle speed $\dot\theta^*$ [rad/s]")
    ax.set_title("Limit-cycle speed vs slope"); ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(gammas_deg, [r["lam_closed"] for r in rows], "-", color="0.7", lw=4, alpha=0.6, label=r"closed-form $\cos^2(2\alpha)$")
    ax.plot(gammas_deg, [r["lam_numeric"] for r in rows], "o-", color="#2a6f97", ms=4, label="numeric (finite diff)")
    ax.axvline(np.degrees(gc), color="crimson", ls="--", lw=1.2)
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"$\gamma$ [deg]"); ax.set_ylabel("Floquet multiplier")
    ax.set_title("Floquet multiplier vs slope"); ax.legend(fontsize=8)

    fig.suptitle(f"Slope ($\\gamma$) sweep, N={N} spokes fixed")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"saved {savepath}")


def plot_N_sweep(rows, savepath):
    Ns = [r["N"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(Ns, [r["rolling_fraction"] for r in rows], "o-", color="#2a6f97")
    ax.set_xlabel("N (number of spokes)"); ax.set_ylabel("rolling-gait basin fraction of grid")
    ax.set_title("RoA size vs N")

    ax = axes[1]
    ax.plot(Ns, [np.degrees(r["gamma_critical"]) for r in rows], "o-", color="#c1440e")
    ax.set_xlabel("N"); ax.set_ylabel(r"critical slope $\gamma_c$ [deg]")
    ax.set_title("Critical slope vs N")

    ax = axes[2]
    ax.plot(Ns, [r["lam_closed"] for r in rows], "-", color="0.7", lw=4, alpha=0.6, label=r"closed-form $\cos^2(2\alpha)$")
    ax.plot(Ns, [r["lam_numeric"] for r in rows], "o-", color="#2a6f97", ms=4, label="numeric")
    ax.set_xlabel("N"); ax.set_ylabel("Floquet multiplier")
    ax.set_title(r"Floquet multiplier vs N  ($\gamma$ fixed)"); ax.legend(fontsize=8)

    fig.suptitle(f"Spoke-count (N) sweep, $\\gamma$ fixed = {np.degrees(rows[0]['gamma']):.1f} deg")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"saved {savepath}")


if __name__ == "__main__":
    print("=== gamma sweep (N=8) ===")
    gc, gamma_rows = sweep_gamma(N=8)
    plot_gamma_sweep(8, gc, gamma_rows, "figures/sweep_gamma.png")

    print("\n=== N sweep (gamma=0.18 rad ~ 10.3 deg) ===")
    N_rows = sweep_N(gamma=0.18, Ns=range(6, 13))
    plot_N_sweep(N_rows, "figures/sweep_N.png")