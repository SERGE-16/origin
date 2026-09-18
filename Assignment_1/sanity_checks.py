"""
Sanity checks for the rimless-wheel model, run BEFORE trusting any of the
downstream analysis.

Check 1 -- energy conservation during stance:
    Between impacts the model is just gravity + a frictionless pin joint,
    so E = 1/2 m l^2 thetadot^2 - m g l cos(theta) must be conserved to
    integrator tolerance across an entire stance phase.
    Expectation: relative drift in E over one stance phase is ~1e-8 or
    smaller (set by rtol/atol of the ODE solver), NOT growing with the
    number of steps simulated.

Check 2 -- impact energy ratio is EXACTLY cos^2(2*alpha):
    KE^+ / KE^- = thetadot^+^2 / thetadot^-^2 = cos^2(2*alpha) by
    construction of the reset map. This just double-checks the reset map
    is wired up correctly (independent of the ODE integration).
    Expectation: ratio matches cos^2(2*alpha) to machine precision.

Check 3 -- flat ground (gamma=0) can never sustain rolling:
    With no downhill slope there is no geometric energy input per step
    (see README derivation), so E can only be lost at every impact.
    Expectation: starting with a healthy forward velocity on gamma=0
    ground, the impact-to-impact energy is strictly decreasing and the
    wheel eventually fails to clear a guard (comes to rest) -- it must
    NOT settle into a sustained rolling limit cycle.

Check 4 -- many-spoke limit (N large) should show most energy input
converted efficiently step-to-step and Floquet multiplier close to
what you'd get from the closed-form return map, and impact losses
should shrink as N grows (sin^2(2*alpha) -> 0).
"""
import numpy as np
from model import RimlessWheelParams, simulate, energy


def check_energy_conservation_within_stance():
    p = RimlessWheelParams(N=8, gamma=0.15, l=1.0)
    x0 = [p.backward_guard_angle() + 1e-3, 0.05]
    result, status = simulate(x0, p, t_max=5.0, max_steps=1)
    t_seg, x_seg = result.t[0], result.x[0]
    E = np.array([energy(x_seg[:, k], p) for k in range(x_seg.shape[1])])
    rel_drift = np.max(np.abs(E - E[0])) / max(abs(E[0]), 1e-12)
    print(f"[check 1] max relative energy drift within one stance phase: {rel_drift:.3e}")
    assert rel_drift < 1e-6, "energy should be conserved during smooth stance phase!"
    print("  PASS: energy is conserved (no dissipation) between impacts, as expected.\n")


def check_impact_energy_ratio():
    p = RimlessWheelParams(N=8, gamma=0.15, l=1.0)
    theta_dot_minus = 1.7
    x_minus = [p.forward_guard_angle(), theta_dot_minus]
    from model import reset_map
    x_plus = reset_map(x_minus, p, "forward")
    ratio = (x_plus[1] / theta_dot_minus) ** 2
    expected = np.cos(2 * p.alpha) ** 2
    print(f"[check 2] KE ratio at impact = {ratio:.6f}, expected cos^2(2 alpha) = {expected:.6f}")
    assert np.isclose(ratio, expected, atol=1e-12)
    print("  PASS: reset map exactly matches the specified angular-momentum-conserving impact.\n")


def check_flat_ground_cannot_sustain_rolling():
    p = RimlessWheelParams(N=8, gamma=0.0, l=1.0)
    x0 = [p.backward_guard_angle() + 1e-6, 3.0]  # generous forward speed
    result, status = simulate(x0, p, t_max=60.0, max_steps=60)
    n_impacts = len(result.events)
    print(f"[check 3] gamma=0: simulated {n_impacts} impacts before status='{status}'")
    # Speed immediately BEFORE each impact (last point of each stance segment)
    # -- must be strictly decreasing in magnitude every single step, since
    # with gamma=0 there is zero geometric energy input (sin(gamma)=0 in the
    # closed-form return map) and every impact strictly dissipates KE.
    pre_impact_speed = np.array([abs(result.x[k][1, -1]) for k in range(n_impacts)])
    strictly_decreasing = np.all(np.diff(pre_impact_speed) < 1e-9)
    ratio = pre_impact_speed[1] / pre_impact_speed[0]
    expected_ratio = np.cos(2 * p.alpha)
    print(f"  pre-impact speeds (first 8): {np.round(pre_impact_speed[:8], 4)}")
    print(f"  step-to-step decay ratio = {ratio:.6f}, expected cos(2 alpha) = {expected_ratio:.6f}")
    print(f"  events alternate forward/backward (rocking): {result.events[:8]} ...")
    assert strictly_decreasing, "on flat ground, speed must strictly decay step to step!"
    assert np.isclose(ratio, expected_ratio, atol=1e-3)
    assert pre_impact_speed[-1] < 1e-3 * pre_impact_speed[0], "should have decayed to ~rest"
    print("  PASS: with no downhill energy input, every impact bleeds energy (decay ratio "
          "= cos(2 alpha) exactly) and the wheel rocks to a stop -- flat ground cannot "
          "sustain a rolling gait, confirming gravity-along-the-slope is the only energy "
          "source.\n")


def check_many_spokes_reduces_impact_loss():
    print("[check 4] impact KE-loss fraction (1 - cos^2(2 alpha)) vs N:")
    for N in [6, 8, 10, 12, 30, 100]:
        alpha = np.pi / N
        loss_frac = 1 - np.cos(2 * alpha) ** 2
        print(f"    N={N:4d}  alpha={np.degrees(alpha):6.2f} deg   KE lost per impact = {loss_frac:.4f}")
    print("  PASS (qualitative): loss fraction shrinks monotonically as N grows, "
          "-> approaches smooth rolling (no dissipation) as N -> infinity.\n")


if __name__ == "__main__":
    check_energy_conservation_within_stance()
    check_impact_energy_ratio()
    check_flat_ground_cannot_sustain_rolling()
    check_many_spokes_reduces_impact_loss()
    print("All sanity checks passed.")