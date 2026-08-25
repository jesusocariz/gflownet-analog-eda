"""Compare active-learning acquisition policies on the same oracle budget."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from simulator import N_DIMS, N_VALUES, all_designs, evaluate, target_feasible
from spice_simulator import evaluate_spice


def features(x: np.ndarray) -> np.ndarray:
    return x.astype(float) / (N_VALUES - 1)


def fit_ensemble(x: np.ndarray, y: np.ndarray, seed: int, n_models: int = 5):
    scaler = StandardScaler().fit(y[:, :3])
    ys = scaler.transform(y[:, :3])
    models = []
    for i in range(n_models):
        model = ExtraTreesRegressor(
            n_estimators=80, min_samples_leaf=2, max_features=0.8,
            bootstrap=True, random_state=seed * 101 + i, n_jobs=-1,
        )
        model.fit(features(x), ys)
        models.append(model)
    return models, scaler


def predictions(models, scaler, pool):
    pred_s = np.stack([m.predict(features(pool)) for m in models])
    mean = scaler.inverse_transform(pred_s.mean(0))
    std = pred_s.std(0).mean(1)
    return mean, std


def acquisition(mean: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    gain_ok = 1.0 / (1.0 + np.exp(-(mean[:, 0] - 18.0) / 2.0))
    bw_ok = 1.0 / (1.0 + np.exp(-(mean[:, 1] - 8.0) / 3.0))
    power_ok = 1.0 / (1.0 + np.exp((mean[:, 2] - 0.35) / 0.08))
    return uncertainty + 0.5 * gain_ok * bw_ok * power_ok


def nearest_distance(candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Normalised Euclidean novelty relative to a reference set."""
    if len(reference) == 0:
        return np.ones(len(candidates))
    a = features(candidates); b = features(reference)
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(2)).min(1) / np.sqrt(N_DIMS)


def diverse_topk(candidates: np.ndarray, scores: np.ndarray, k: int,
                 diversity_weight: float) -> np.ndarray:
    """Greedy quality-diversity selection with scores scaled to [0, 1]."""
    quality = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
    remaining = np.ones(len(candidates), dtype=bool)
    chosen = []
    for _ in range(min(k, len(candidates))):
        if chosen:
            diversity = nearest_distance(candidates, candidates[np.asarray(chosen)])
        else:
            diversity = np.ones(len(candidates))
        objective = quality + diversity_weight * diversity
        objective[~remaining] = -np.inf
        idx = int(np.argmax(objective)); chosen.append(idx); remaining[idx] = False
    return candidates[np.asarray(chosen)]


def batch_diversity(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    z = features(x)
    distances = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(2)) / np.sqrt(N_DIMS)
    return float(distances[np.triu_indices(len(z), 1)].mean())


class GFlowNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_DIMS * (N_VALUES + 1), 96), torch.nn.ReLU(),
            torch.nn.Linear(96, 96), torch.nn.ReLU(),
            torch.nn.Linear(96, N_VALUES),
        )
        self.log_z = torch.nn.Parameter(torch.tensor(0.0))

    def logits(self, state, step):
        one_hot = torch.nn.functional.one_hot(state + 1, N_VALUES + 1).float().flatten(1)
        return self.net(one_hot)


def sample_trajectories(model, n, generator):
    state = torch.full((n, N_DIMS), -1, dtype=torch.long)
    log_pf = torch.zeros(n)
    for step in range(N_DIMS):
        logits = model.logits(state, step)
        probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        log_pf += torch.log(probs.gather(1, action[:, None]).squeeze(1) + 1e-12)
        state[:, step] = action
    return state, log_pf


def gfn_select(pool, reward, batch_size, seed, steps, diversity_weight=0.0):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = GFlowNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    reward_map = {tuple(x): float(r) for x, r in zip(pool, reward)}
    floor = max(float(np.quantile(reward, 0.1)), 1e-4)
    for _ in range(steps):
        states, log_pf = sample_trajectories(model, 128, generator)
        r = torch.tensor([reward_map.get(tuple(x.tolist()), floor) for x in states])
        # A fixed construction order gives one backward trajectory, log P_B=0.
        loss = ((model.log_z + log_pf - torch.log(r + 1e-8)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    sampled = []
    available = {tuple(x) for x in pool}
    with torch.no_grad():
        for _ in range(40):
            states, _ = sample_trajectories(model, max(128, batch_size * 8), generator)
            for x in states.numpy():
                key = tuple(x)
                if key in available and key not in sampled:
                    sampled.append(key)
            if len(sampled) >= max(batch_size * 20, 256):
                break
    if len(sampled) < batch_size:
        sampled.extend(tuple(x) for x in pool[np.argsort(reward)[::-1]] if tuple(x) not in sampled)
    candidates = np.asarray(sampled, dtype=np.int64)
    reward_map = {tuple(x): r for x, r in zip(pool, reward)}
    scores = np.asarray([reward_map[tuple(x)] for x in candidates])
    return diverse_topk(candidates, scores, batch_size, diversity_weight)


def sobol_order(n: int, seed: int) -> np.ndarray:
    u = qmc.Sobol(N_DIMS, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n))))
    x = np.minimum((u * N_VALUES).astype(int), N_VALUES - 1)
    _, idx = np.unique(x, axis=0, return_index=True)
    return x[np.sort(idx)]


def run(method, seed, rounds, batch_size, gfn_steps, universe, oracle, x_test, y_test,
        diversity_weight):
    rng = np.random.default_rng(seed)
    initial = rng.choice(len(universe), 24, replace=False).tolist()
    observed = list(initial)
    sobol = sobol_order(4096, seed)
    rows = []
    for round_id in range(rounds + 1):
        x_obs = universe[observed]; y_obs = oracle(x_obs)
        models, scaler = fit_ensemble(x_obs, y_obs, seed + round_id)
        test_mean, _ = predictions(models, scaler, x_test)
        # Normalise each metric so bandwidth units do not dominate the score.
        scale = np.std(y_test[:, :3], axis=0) + 1e-8
        rmse = root_mean_squared_error(y_test[:, :3] / scale, test_mean / scale)
        rows.append({"method": method, "seed": seed, "round": round_id,
                     "budget": len(observed), "rmse": rmse,
                     "feasible_found": int(target_feasible(y_obs).sum()),
                     "unique_feasible": int(np.unique(x_obs[target_feasible(y_obs)], axis=0).shape[0]),
                     "last_batch_diversity": batch_diversity(x_obs[-batch_size:]) if round_id else np.nan})
        if round_id == rounds:
            break
        mask = np.ones(len(universe), dtype=bool); mask[observed] = False
        pool_idx = np.flatnonzero(mask); pool = universe[pool_idx]
        if method == "random":
            selected = rng.choice(pool_idx, batch_size, replace=False)
        elif method == "sobol":
            observed_set = {tuple(x) for x in x_obs}
            candidates = [x for x in sobol if tuple(x) not in observed_set][:batch_size]
            lookup = {tuple(x): i for i, x in enumerate(universe)}
            selected = np.asarray([lookup[tuple(x)] for x in candidates])
        else:
            mean, std = predictions(models, scaler, pool)
            score = acquisition(mean, std)
            novelty = nearest_distance(pool, x_obs)
            if method in {"uncertainty", "uncertainty_diverse"}:
                weight = diversity_weight if method.endswith("diverse") else 0.0
                chosen = diverse_topk(pool, score, batch_size, weight)
                lookup = {tuple(x): i for i, x in zip(pool_idx, pool)}
                selected = np.asarray([lookup[tuple(x)] for x in chosen])
            elif method in {"gflownet", "gflownet_diverse"}:
                # Novelty to all observations is part of the terminal reward;
                # pairwise batch diversity is added during final set selection.
                reward_diversity = diversity_weight if method.endswith("diverse") else 0.0
                score_with_novelty = score + reward_diversity * novelty
                reward = np.exp(3.0 * (score_with_novelty - score_with_novelty.min()) /
                                (np.ptp(score_with_novelty) + 1e-8))
                weight = diversity_weight if method.endswith("diverse") else 0.0
                chosen = gfn_select(pool, reward, batch_size, seed * 100 + round_id,
                                    gfn_steps, weight)
                lookup = {tuple(x): i for i, x in zip(pool_idx, pool)}
                selected = np.asarray([lookup[tuple(x)] for x in chosen])
            else:
                raise ValueError(method)
        observed.extend(selected.tolist())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gfn-steps", type=int, default=300)
    parser.add_argument("--backend", choices=["compact", "ngspice"], default="compact")
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--diversity-weight", type=float, default=0.6)
    args = parser.parse_args()
    universe = all_designs()
    oracle = evaluate if args.backend == "compact" else evaluate_spice
    test_size = args.test_size or (2000 if args.backend == "compact" else 128)
    test_idx = np.random.default_rng(2026).choice(len(universe), test_size, replace=False)
    x_test = universe[test_idx]; y_test = oracle(x_test)
    methods = ["random", "sobol", "uncertainty", "uncertainty_diverse",
               "gflownet", "gflownet_diverse"]
    rows = []
    for method in methods:
        for seed in args.seeds:
            print(f"running {method}, seed={seed}", flush=True)
            rows.extend(run(method, seed, args.rounds, args.batch_size, args.gfn_steps,
                            universe, oracle, x_test, y_test, args.diversity_weight))
    out = Path("results") / args.backend; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method, group in df.groupby("method"):
        agg = group.groupby("budget").agg(rmse=("rmse", "mean"), feasible=("feasible_found", "mean"))
        axes[0].plot(agg.index, agg.rmse, marker="o", label=method)
        axes[1].plot(agg.index, agg.feasible, marker="o", label=method)
        diversity = group.groupby("budget").last_batch_diversity.mean()
        axes[2].plot(diversity.index, diversity, marker="o", label=method)
    axes[0].set(xlabel="Oracle evaluations", ylabel="Surrogate RMSE", title="Data efficiency")
    axes[1].set(xlabel="Oracle evaluations", ylabel="Feasible designs found", title="Design discovery")
    axes[2].set(xlabel="Oracle evaluations", ylabel="Mean pairwise distance", title="Batch diversity")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=180)
    print(df.groupby(["method", "budget"])[["rmse", "feasible_found"]].mean().groupby(level=0).tail(1))


if __name__ == "__main__":
    main()
