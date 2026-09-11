"""
Draws the model schematic requested by the assignment ("sketch it out, with
parameters and states annotated") -- generated with matplotlib instead of by
hand, but showing exactly what a hand sketch should: the wheel, the ground
slope, all spokes, and every parameter/state annotated.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from model import RimlessWheelParams


def draw_schematic(p: RimlessWheelParams, theta=0.12, savepath="figures/schematic.png"):
    alpha = p.alpha
    gamma = p.gamma
    l = p.l

    fig, ax = plt.subplots(figsize=(7, 6))

    # Ground line (slope), drawn long, centered near hub's ground contact
    hub_ground_x, hub_ground_y = 0.0, 0.0
    L = 2.2 * l
    ax.plot([hub_ground_x - L * np.cos(gamma), hub_ground_x + L * np.cos(gamma)],
            [hub_ground_y + L * np.sin(gamma), hub_ground_y - L * np.sin(gamma)],
            color="0.3", lw=2, zorder=1)
    ax.annotate("ground (slope $\\gamma$ downhill $\\rightarrow$)",
                (hub_ground_x + 0.55 * L * np.cos(gamma), hub_ground_y - 0.55 * L * np.sin(gamma) - 0.08),
                fontsize=9, color="0.3")

    # Hub position: stance foot is pinned at origin; hub is at angle theta
    # from TRUE vertical (not from the ground normal).
    hub = np.array([l * np.sin(theta), l * np.cos(theta)])

    # True vertical reference (dashed)
    ax.plot([0, 0], [0, 1.3 * l], "k--", lw=1, zorder=1)
    ax.annotate("upward vertical", (0.03, 1.28 * l), fontsize=8, color="0.2")

    # All N spokes, evenly spaced by 2*alpha, centered on the ground-normal
    # direction (gamma from vertical); stance spoke is highlighted.
    N = p.N
    # Spoke k sits at angle gamma + (k - (N-1)/2)*2*alpha from vertical, for
    # a wheel "resting" symmetrically about the slope normal; we then offset
    # the whole wheel by (theta - gamma) so the CURRENT stance spoke is at
    # angle theta (matches the hub location drawn above).
    base_angles = gamma + (np.arange(N) - (N - 1) / 2.0) * 2 * alpha
    offset = theta - gamma
    spoke_angles = base_angles + offset
    for a in spoke_angles:
        tip = np.array([l * np.sin(a), l * np.cos(a)])
        is_stance = np.isclose(a, theta)
        ax.plot([hub[0], tip[0]], [hub[1], tip[1]],
                 color=("crimson" if is_stance else "0.6"),
                 lw=(2.5 if is_stance else 1.2), zorder=3 if is_stance else 2)

    # Hub point mass
    ax.plot(*hub, "o", color="crimson", ms=12, zorder=5)
    ax.annotate("point mass $m$ @ hub", (hub[0] + 0.06, hub[1] + 0.03), fontsize=9)

    # theta angle arc (from vertical to stance spoke)
    arc_r = 0.35 * l
    arc_thetas = np.linspace(0, theta, 40)
    ax.plot(arc_r * np.sin(arc_thetas), arc_r * np.cos(arc_thetas), color="crimson", lw=1.5)
    ax.annotate(r"$\theta$", (arc_r * np.sin(theta / 2) + 0.05, arc_r * np.cos(theta / 2)),
                fontsize=12, color="crimson")

    # gamma angle arc (from vertical to ground normal)
    arc_r2 = 0.55 * l
    arc_gammas = np.linspace(0, gamma, 40)
    ax.plot(arc_r2 * np.sin(arc_gammas), arc_r2 * np.cos(arc_gammas), color="0.3", lw=1.5)
    ax.annotate(r"$\gamma$", (arc_r2 * np.sin(gamma / 2) + 0.05, arc_r2 * np.cos(gamma / 2) + 0.05),
                fontsize=11, color="0.3")

    # alpha half-angle annotation between two adjacent spokes at the guard config
    ax.annotate(r"$2\alpha = 2\pi/N$" + "\n(inter-spoke angle)",
                (0.62 * l, 0.15 * l), fontsize=9, color="0.4")
    ax.annotate("$l$ (spoke length)", (hub[0] * 0.5 - 0.35, hub[1] * 0.5), fontsize=9, color="crimson")

    ax.plot(0, 0, "k^", ms=10, zorder=5)
    ax.annotate("stance contact\n(pinned, no slip)", (0.05, -0.18), fontsize=8)

    ax.set_xlim(-1.3 * l, 1.6 * l)
    ax.set_ylim(-0.6 * l, 1.5 * l)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Rimless wheel model  (N={N} spokes, l={l} m, $\\gamma$={np.degrees(gamma):.1f}$^\\circ$)\n"
        f"state $x=[\\theta,\\dot\\theta]$   params: $l, N, \\gamma, g$",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"saved {savepath}")


if __name__ == "__main__":
    draw_schematic(RimlessWheelParams(N=8, gamma=0.12), theta=0.15)