"""Multi-objective search for a diverse analog-design Pareto front.

This is intentionally separate from experiment.py, which remains the global
surrogate-accuracy study for the final report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment import (
    batch_diversity, diverse_topk, fit_ensemble, gfn_select, predictions,
    sobol_order,
)
from simulator import all_designs, decode, evaluate
from spice_simulator import evaluate_spice


def area_proxy(x: np.ndarray) -> np.ndarray:
    """Technology-independent proxy; lower is better, not physical layout area."""
    p = decode(x)
    transistor = p["w_um"] * p["l_um"]
    resistor = 0.20 * p["rd_kohm"]
    capacitor = 1.50 * p["cl_pf"]
    return transistor + resistor + capacitor


def objectives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """All objectives in minimisation form: [-gain, -bandwidth, power, area]."""
    return np.column_stack([-y[:, 0], -y[:, 1], y[:, 2], area_proxy(x)])


def nondominated_mask(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    valid = np.ones(len(values), dtype=bool) if valid is None else valid.astype(bool)
    result = np.zeros(len(values), dtype=bool)
    indices = np.flatnonzero(valid)
    for i in indices:
        dominated = np.any(np.all(values[indices] <= values[i], axis=1) &
                           np.any(values[indices] < values[i], axis=1))
        result[i] = not dominated
    return result


def normalise(values: np.ndarray, ideal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.clip((values - ideal) / (reference - ideal + 1e-12), 0.0, 1.2)


def monte_carlo_hypervolume(front: np.ndarray, samples: np.ndarray) -> float:
    """Deterministic common-sample estimate in the unit reference box."""
    if len(front) == 0:
        return 0.0
    front = np.clip(front, 0.0, 1.0)
    dominated = np.zeros(len(samples), dtype=bool)
    for point in front:
        dominated |= np.all(point <= samples, axis=1)
    return float(dominated.mean())


def igd(front: np.ndarray, reference_front: np.ndarray) -> float:
    if len(front) == 0:
        return float("inf")
    return float(np.sqrt(((reference_front[:, None, :] - front[None, :, :]) ** 2).sum(2)).min(1).mean())


def pareto_metrics(x: np.ndarray, y: np.ndarray, ideal: np.ndarray, reference: np.ndarray,
                   reference_front: np.ndarray, hv_samples: np.ndarray) -> dict[str, float]:
    f = normalise(objectives(x, y), ideal, reference)
    mask = nondominated_mask(f, y[:, 3] > 0.5)
    front = f[mask]
    return {
        "hypervolume": monte_carlo_hypervolume(front, hv_samples),
        "igd": igd(front, reference_front),
        "n_nondominated": int(mask.sum()),
        "archive_diversity": batch_diversity(x[mask]),
    }


def preference_score(mean: np.ndarray, pool: np.ndarray, weights: np.ndarray,
                     ideal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    predicted = np.column_stack([-mean[:, 0], -mean[:, 1], mean[:, 2], area_proxy(pool)])
    z = normalise(predicted, ideal, reference)
    # Achievement scalarisation: each preference exposes a different Pareto region.
    chebyshev = np.max(weights[:, None, :] * z[None, :, :], axis=2)
    return -chebyshev.min(axis=0)


def run(method, seed, rounds, batch_size, gfn_steps, universe, oracle,
        x_reference, y_reference, ideal, reference, reference_front, hv_samples,
        diversity_weight):
    rng = np.random.default_rng(seed)
    observed = rng.choice(len(universe), 24, replace=False).tolist()
    sobol = sobol_order(4096, seed)
    lookup_all = {tuple(x): i for i, x in enumerate(universe)}
    rows = []
    for round_id in range(rounds + 1):
        x_obs = universe[observed]; y_obs = oracle(x_obs)
        metrics = pareto_metrics(x_obs, y_obs, ideal, reference, reference_front, hv_samples)
        rows.append({"method": method, "seed": seed, "round": round_id,
                     "budget": len(observed), **metrics,
                     "last_batch_diversity": batch_diversity(x_obs[-batch_size:]) if round_id else np.nan})
        if round_id == rounds:
            break
        available = np.ones(len(universe), dtype=bool); available[observed] = False
        pool_idx = np.flatnonzero(available); pool = universe[pool_idx]
        if method == "random":
            selected = rng.choice(pool_idx, batch_size, replace=False)
        elif method == "sobol":
            seen = {tuple(x) for x in x_obs}
            chosen = [x for x in sobol if tuple(x) not in seen][:batch_size]
            selected = np.asarray([lookup_all[tuple(x)] for x in chosen])
        else:
            models, scaler = fit_ensemble(x_obs, y_obs, seed + round_id)
            mean, uncertainty = predictions(models, scaler, pool)
            weights = rng.dirichlet(np.ones(4), size=32)
            score = preference_score(mean, pool, weights, ideal, reference)
            score += 0.15 * uncertainty
            predicted_valid = 1.0 / (1.0 + np.exp(-(mean[:, 0] - 3.0) / 2.0))
            score += 0.35 * predicted_valid
            if method == "scalarisation":
                chosen = diverse_topk(pool, score, batch_size, diversity_weight)
            elif method == "gflownet_pareto":
                scaled = (score - score.min()) / (np.ptp(score) + 1e-8)
                reward = np.exp(4.0 * scaled)
                chosen = gfn_select(pool, reward, batch_size, seed * 100 + round_id,
                                    gfn_steps, diversity_weight)
            else:
                raise ValueError(method)
            lookup = {tuple(x): i for i, x in zip(pool_idx, pool)}
            selected = np.asarray([lookup[tuple(x)] for x in chosen])
        observed.extend(selected.tolist())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["compact", "ngspice"], default="compact")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gfn-steps", type=int, default=250)
    parser.add_argument("--reference-size", type=int, default=None)
    parser.add_argument("--diversity-weight", type=float, default=0.4)
    args = parser.parse_args()

    universe = all_designs()
    oracle = evaluate if args.backend == "compact" else evaluate_spice
    reference_size = args.reference_size or (len(universe) if args.backend == "compact" else 512)
    if reference_size == len(universe):
        x_reference = universe
    else:
        order = sobol_order(8192, 2026)
        x_reference = order[:reference_size]
    print(f"evaluating fixed {args.backend} reference set ({len(x_reference)} points)", flush=True)
    y_reference = oracle(x_reference)
    f_reference = objectives(x_reference, y_reference)
    valid_reference = y_reference[:, 3] > 0.5
    valid_f = f_reference[valid_reference]
    ideal = np.quantile(valid_f, 0.01, axis=0)
    reference = np.quantile(valid_f, 0.99, axis=0)
    reference_front_mask = nondominated_mask(f_reference, valid_reference)
    reference_front = normalise(f_reference[reference_front_mask], ideal, reference)
    hv_samples = np.random.default_rng(4242).random((40_000, 4))

    methods = ["random", "sobol", "scalarisation", "gflownet_pareto"]
    rows = []
    for method in methods:
        for seed in args.seeds:
            print(f"running {method}, seed={seed}", flush=True)
            rows.extend(run(method, seed, args.rounds, args.batch_size, args.gfn_steps,
                            universe, oracle, x_reference, y_reference, ideal, reference,
                            reference_front, hv_samples, args.diversity_weight))
    out = Path("results") / f"pareto_{args.backend}"; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(x_reference, columns=decode(x_reference).keys()).to_csv(out / "reference_design_indices.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method, group in df.groupby("method"):
        agg = group.groupby("budget").agg(hv=("hypervolume", "mean"), igd=("igd", "mean"),
                                             count=("n_nondominated", "mean"))
        axes[0].plot(agg.index, agg.hv, marker="o", label=method)
        axes[1].plot(agg.index, agg.igd, marker="o", label=method)
        axes[2].plot(agg.index, agg["count"], marker="o", label=method)
    axes[0].set(xlabel="SPICE evaluations", ylabel="Hypervolume", title="Pareto quality (higher is better)")
    axes[1].set(xlabel="SPICE evaluations", ylabel="IGD", title="Reference-front distance (lower is better)")
    axes[2].set(xlabel="SPICE evaluations", ylabel="Non-dominated designs", title="Archive size")
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "pareto_curves.png", dpi=180)
    final = df[df["round"] == args.rounds].groupby("method")[["hypervolume", "igd", "n_nondominated", "archive_diversity"]].mean()
    print(final.sort_values("hypervolume", ascending=False))


if __name__ == "__main__":
    main()

