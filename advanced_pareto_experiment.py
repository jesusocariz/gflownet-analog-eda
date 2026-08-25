"""Advanced Pareto experiment: stronger baselines, metrics and five seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiment import (GFlowNet, batch_diversity, diverse_topk, fit_ensemble,
                        gfn_select, predictions, sample_trajectories, sobol_order)
from pareto_experiment import (
    area_proxy, igd, monte_carlo_hypervolume, nondominated_mask, normalise,
    objectives, preference_score,
)
from simulator import N_DIMS, N_VALUES, all_designs, evaluate
from spice_simulator import evaluate_spice


def objective_spacing(front: np.ndarray) -> float:
    """Uniformity of nearest-neighbour distances; lower is better."""
    if len(front) < 3:
        return float("nan")
    distances = np.sqrt(((front[:, None, :] - front[None, :, :]) ** 2).sum(2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(1)
    return float(nearest.std(ddof=1))


def categorical_select(pool, reward, k, seed, diversity_weight):
    """Ablation: sample directly from the same terminal reward as the GFlowNet."""
    rng = np.random.default_rng(seed)
    probability = reward / reward.sum()
    candidate_count = min(len(pool), max(256, 20 * k))
    candidates_idx = rng.choice(len(pool), candidate_count, replace=False, p=probability)
    candidates = pool[candidates_idx]
    return diverse_topk(candidates, reward[candidates_idx], k, diversity_weight)


def nsga2_surrogate_select(pool, predicted, k, seed, generations=35, population_size=160,
                           gfn_donors=None):
    """Compact NSGA-II-style search over surrogate predictions.

    It uses non-dominated sorting plus crowding distance, tournament selection,
    uniform crossover and discrete mutation. Final evaluations remain ngspice.
    """
    rng = np.random.default_rng(seed)
    score_lookup = {tuple(x): f for x, f in zip(pool, predicted)}
    available = {tuple(x) for x in pool}

    def rank_and_crowding(pop):
        f = np.asarray([score_lookup[tuple(x)] for x in pop])
        remaining = np.arange(len(pop)); rank = np.full(len(pop), 10_000); level = 0
        while len(remaining):
            nd = nondominated_mask(f[remaining]); chosen = remaining[nd]
            rank[chosen] = level; remaining = remaining[~nd]; level += 1
        crowd = np.zeros(len(pop))
        for r in np.unique(rank):
            ids = np.flatnonzero(rank == r)
            if len(ids) <= 2:
                crowd[ids] = np.inf; continue
            for j in range(f.shape[1]):
                order = ids[np.argsort(f[ids, j])]
                crowd[order[[0, -1]]] = np.inf
                span = np.ptp(f[order, j]) + 1e-12
                crowd[order[1:-1]] += (f[order[2:], j] - f[order[:-2], j]) / span
        return rank, crowd

    pop = pool[rng.choice(len(pool), min(population_size, len(pool)), replace=False)]
    for _ in range(generations):
        rank, crowd = rank_and_crowding(pop)
        children = []
        for _ in range(len(pop)):
            pair = rng.choice(len(pop), 2, replace=False)
            parent = pop[pair[np.lexsort((-crowd[pair], rank[pair]))[0]]]
            mate = pop[rng.integers(len(pop))]
            # In the hybrid, a high-reward GFlowNet design becomes a third
            # parent and therefore an informed mutation direction.
            if gfn_donors is not None and rng.random() < .65:
                donor = gfn_donors[rng.integers(len(gfn_donors))]
                child = np.where(rng.random(N_DIMS) < .35, donor,
                                 np.where(rng.random(N_DIMS) < .5, parent, mate)).copy()
            else:
                child = np.where(rng.random(N_DIMS) < .5, parent, mate).copy()
            mutation = rng.random(N_DIMS) < (1.0 / N_DIMS)
            child[mutation] = rng.integers(N_VALUES, size=mutation.sum())
            if tuple(child) in available:
                children.append(child)
        combined = np.unique(np.vstack([pop, children]) if children else pop, axis=0)
        rank, crowd = rank_and_crowding(combined)
        order = np.lexsort((-crowd, rank)); pop = combined[order[:population_size]]
    f = np.asarray([score_lookup[tuple(x)] for x in pop])
    nd = pop[nondominated_mask(f)]
    if len(nd) < k:
        nd = pop
    return diverse_topk(nd, np.ones(len(nd)), k, diversity_weight=1.0)


class FeedbackGFlowNet:
    """Persistent GFlowNet used as an intelligent NSGA-II mutation operator."""

    def __init__(self, seed):
        torch.manual_seed(seed)
        self.model = GFlowNet()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        self.generator = torch.Generator().manual_seed(seed)

    def update(self, pool, reward, successful, steps):
        reward_map = {tuple(x): float(r) for x, r in zip(pool, reward)}
        maximum = float(np.max(reward))
        # Successful ngspice-evaluated Pareto designs receive an explicit boost.
        for x in successful:
            reward_map[tuple(x)] = max(reward_map.get(tuple(x), 0.0), 2.0 * maximum)
        floor = max(float(np.quantile(reward, .05)), 1e-4)
        for _ in range(steps):
            states, log_pf = sample_trajectories(self.model, 128, self.generator)
            terminal_reward = torch.tensor(
                [reward_map.get(tuple(x.tolist()), floor) for x in states], dtype=torch.float32)
            loss = ((self.model.log_z + log_pf - torch.log(terminal_reward + 1e-8)) ** 2).mean()
            self.optimiser.zero_grad(); loss.backward(); self.optimiser.step()

    def donors(self, available, n):
        accepted = []; allowed = {tuple(x) for x in available}
        with torch.no_grad():
            for _ in range(50):
                states, _ = sample_trajectories(self.model, max(128, n * 4), self.generator)
                for x in states.numpy():
                    key = tuple(x)
                    if key in allowed and key not in accepted: accepted.append(key)
                if len(accepted) >= n: break
        if not accepted:
            return available[:n]
        return np.asarray(accepted[:n], dtype=np.int64)


def metrics(x, y, ideal, reference, reference_front, hv_samples):
    f = normalise(objectives(x, y), ideal, reference)
    mask = nondominated_mask(f, y[:, 3] > .5); front = f[mask]
    return {
        "hypervolume": monte_carlo_hypervolume(front, hv_samples),
        "igd": igd(front, reference_front),
        "n_nondominated": int(mask.sum()),
        "objective_spacing": objective_spacing(front),
        "archive_input_diversity": batch_diversity(x[mask]),
    }


def run(method, seed, rounds, batch_size, gfn_steps, universe, oracle,
        ideal, reference, reference_front, hv_samples, diversity_weight):
    rng = np.random.default_rng(seed)
    observed = rng.choice(len(universe), 24, replace=False).tolist()
    feedback_gfn = FeedbackGFlowNet(seed + 50_000) if method == "gfn_nsga2_feedback" else None
    sobol = sobol_order(8192, seed); all_lookup = {tuple(x): i for i, x in enumerate(universe)}
    rows = []
    for round_id in range(rounds + 1):
        x_obs = universe[observed]; y_obs = oracle(x_obs)
        rows.append({"method": method, "seed": seed, "round": round_id,
                     "budget": len(observed), **metrics(x_obs, y_obs, ideal, reference,
                                                        reference_front, hv_samples),
                     "last_batch_diversity": batch_diversity(x_obs[-batch_size:]) if round_id else np.nan})
        if round_id == rounds: break
        available_mask = np.ones(len(universe), bool); available_mask[observed] = False
        pool_idx = np.flatnonzero(available_mask); pool = universe[pool_idx]
        if method == "random":
            selected = rng.choice(pool_idx, batch_size, replace=False)
        elif method == "sobol":
            seen = {tuple(x) for x in x_obs}
            chosen = np.asarray([x for x in sobol if tuple(x) not in seen][:batch_size])
            selected = np.asarray([all_lookup[tuple(x)] for x in chosen])
        else:
            models, scaler = fit_ensemble(x_obs, y_obs, seed + round_id, n_models=7)
            mean, uncertainty = predictions(models, scaler, pool)
            weights = rng.dirichlet(np.ones(4), size=64)
            score = preference_score(mean, pool, weights, ideal, reference) + .15 * uncertainty
            score += .35 / (1.0 + np.exp(-(mean[:, 0] - 3.0) / 2.0))
            scaled = (score - score.min()) / (np.ptp(score) + 1e-8)
            reward = np.exp(4.0 * scaled)
            if method == "scalarisation":
                chosen = diverse_topk(pool, score, batch_size, diversity_weight)
            elif method == "categorical_reward":
                chosen = categorical_select(pool, reward, batch_size,
                                            seed * 1000 + round_id, diversity_weight)
            elif method == "gflownet_pareto":
                chosen = gfn_select(pool, reward, batch_size, seed * 1000 + round_id,
                                    gfn_steps, diversity_weight)
            elif method in {"nsga2_surrogate", "gfn_nsga2_feedback"}:
                predicted_f = normalise(np.column_stack([-mean[:, 0], -mean[:, 1],
                                                          mean[:, 2], area_proxy(pool)]),
                                         ideal, reference)
                donors = None
                if method == "gfn_nsga2_feedback":
                    observed_f = normalise(objectives(x_obs, y_obs), ideal, reference)
                    success_mask = nondominated_mask(observed_f, y_obs[:, 3] > .5)
                    # The same surrogate reward guides unexplored space, while
                    # evaluated non-dominated successes close the feedback loop.
                    feedback_gfn.update(pool, reward, x_obs[success_mask],
                                        max(60, gfn_steps // 2))
                    donors = feedback_gfn.donors(pool, 256)
                chosen = nsga2_surrogate_select(pool, predicted_f, batch_size,
                                                seed * 1000 + round_id,
                                                gfn_donors=donors)
            else: raise ValueError(method)
            lookup = {tuple(x): i for x, i in zip(pool, pool_idx)}
            selected = np.asarray([lookup[tuple(x)] for x in chosen])
        observed.extend(selected.tolist())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GFlowNet and evolutionary Pareto-search methods."
    )
    parser.add_argument("--backend", choices=["compact", "ngspice"], default="ngspice")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gfn-steps", type=int, default=250)
    parser.add_argument("--reference-size", type=int, default=1024)
    parser.add_argument("--diversity-weight", type=float, default=.3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/advanced_pareto_<backend>).",
    )
    args = parser.parse_args()
    universe = all_designs(); oracle = evaluate if args.backend == "compact" else evaluate_spice
    x_ref = sobol_order(16384, 2026)[:args.reference_size]
    print(f"evaluating locked reference ({len(x_ref)} points)", flush=True)
    y_ref = oracle(x_ref); f_ref = objectives(x_ref, y_ref); valid = y_ref[:, 3] > .5
    ideal = np.quantile(f_ref[valid], .01, axis=0); reference = np.quantile(f_ref[valid], .99, axis=0)
    ref_front = normalise(f_ref[nondominated_mask(f_ref, valid)], ideal, reference)
    hv_samples = np.random.default_rng(4242).random((100_000, 4))
    methods = ["random", "sobol", "scalarisation", "categorical_reward",
               "nsga2_surrogate", "gflownet_pareto", "gfn_nsga2_feedback"]
    rows = []
    for method in methods:
        for seed in args.seeds:
            print(f"running {method}, seed={seed}", flush=True)
            rows.extend(run(method, seed, args.rounds, args.batch_size, args.gfn_steps,
                            universe, oracle, ideal, reference, ref_front, hv_samples,
                            args.diversity_weight))
    out = args.output_dir or Path("results") / f"advanced_pareto_{args.backend}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = [("hypervolume", "Hypervolume", True), ("igd", "IGD", False),
              ("n_nondominated", "Non-dominated archive", True),
              ("objective_spacing", "Objective spacing", False)]
    for method, group in df.groupby("method"):
        for ax, (field, label, _) in zip(axes.flat, fields):
            agg = group.groupby("budget")[field].agg(["mean", "std"])
            ax.plot(agg.index, agg["mean"], marker="o", label=method)
            ax.fill_between(agg.index, agg["mean"]-agg["std"], agg["mean"]+agg["std"], alpha=.10)
            ax.set(xlabel="ngspice evaluations", ylabel=label); ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(out / "advanced_curves.png", dpi=200)
    final = df[df["round"] == args.rounds]
    stats = final.groupby("method")[[x[0] for x in fields] + ["archive_input_diversity"]].agg(["mean", "std"])
    stats.to_csv(out / "final_statistics.csv")
    print(stats.round(4).to_string())


if __name__ == "__main__":
    main()
