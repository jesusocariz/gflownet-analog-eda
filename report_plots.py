"""Regenerate the reference-set figure used by the current technical report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import sobol_order
from pareto_experiment import area_proxy, nondominated_mask, objectives
from spice_simulator import evaluate_spice


def main() -> None:
    """Plot trade-offs in the locked 1,024-design ngspice reference set."""
    output_dir = Path("results/advanced_pareto_ngspice")
    output_dir.mkdir(parents=True, exist_ok=True)

    designs = sobol_order(16_384, 2026)[:1_024]
    responses = evaluate_spice(designs)
    objective_values = objectives(designs, responses)
    valid = responses[:, 3] > 0.5
    nondominated = nondominated_mask(objective_values, valid)
    area = area_proxy(designs)

    panels = [
        (responses[:, 2], responses[:, 0], "Power (mW)", "Gain (dB)"),
        (responses[:, 2], responses[:, 1], "Power (mW)", "Bandwidth (MHz)"),
        (area, responses[:, 0], "Area proxy", "Gain (dB)"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for axis, (horizontal, vertical, x_label, y_label) in zip(axes, panels):
        axis.scatter(
            horizontal[valid & ~nondominated],
            vertical[valid & ~nondominated],
            s=10,
            c="#b7bdc6",
            alpha=0.45,
            label="Valid reference designs",
        )
        axis.scatter(
            horizontal[nondominated],
            vertical[nondominated],
            s=24,
            c="#dc3522",
            edgecolor="white",
            linewidth=0.3,
            label="Non-dominated reference set",
        )
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=0.18)

    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Trade-offs in the locked 1,024-design ngspice reference set",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "reference_tradeoffs.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
