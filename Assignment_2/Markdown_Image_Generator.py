"""Generate the static figures referenced by REPORT.md, saved to
report_figures/.

This is a separate, save-to-disk entry point -- it does NOT touch
assignment_2.py's own main(), which stays a pure show-everything-live
script with nothing written to disk (per your earlier request). It imports
run_pipeline() and the individual plot_* functions straight from
assignment_2.py, so the numbers in the report are guaranteed to match a
live run of assignment_2.py exactly (same code, same computation, just
saved instead of shown).

Run with: uv run python generate_report_figures.py
"""

import os

import matplotlib.pyplot as plt

from assignment_2 import (
    run_pipeline,
    plot_roa,
    plot_lookup_table,
    plot_alpha_vs_state,
    plot_steps_to_standstill,
    plot_grid_resolution_study,
    plot_trajectory_diagnostics,
    plot_poincare_slice,
    plot_poincare_sequence,
)

OUTPUT_DIR = "report_figures"


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {os.path.abspath(path)}")


def main():
    data = run_pipeline()

    save(
        plot_roa(data["theta_grid"], data["theta_dot_grid_roa"], data["roa_mask"]),
        "roa.png",
    )
    save(
        plot_lookup_table(
            data["theta_dot_grid_table"], data["alpha_grid"],
            data["steps_to_stand"], data["policy_alpha"],
        ),
        "lookup_table.png",
    )
    save(
        plot_alpha_vs_state(
            data["theta_dot_grid_table"], data["alpha_grid"],
            data["steps_to_stand"], data["policy_alpha"],
        ),
        "alpha_vs_state.png",
    )
    save(
        plot_steps_to_standstill(
            data["theta_dot_grid_table"], data["steps_to_stand"], data["roa_capture"],
            data["theta_dot_0"], data["steps_predicted"],
        ),
        "steps_to_standstill.png",
    )
    save(
        plot_grid_resolution_study(
            data["N_candidates"], data["sample_theta_dots"], data["grid_study_results"],
        ),
        "grid_resolution_study.png",
    )
    save(
        plot_trajectory_diagnostics(
            data["trajectory"], data["touchdown_times"], data["steps_predicted"],
        ),
        "trajectory.png",
    )
    save(plot_poincare_slice(data["trajectory"]), "poincare_phase_slice.png")
    save(plot_poincare_sequence(data["trajectory"]), "poincare_sequence.png")

    print("\nAll report figures saved.")


if __name__ == "__main__":
    main()