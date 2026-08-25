import numpy as np

from simulator import N_DIMS, all_designs, evaluate
from spice_simulator import evaluate_spice
from experiment import batch_diversity, diverse_topk


def test_space_and_outputs():
    x = all_designs()
    assert x.shape == (5 ** N_DIMS, N_DIMS)
    y = evaluate(x[:20])
    assert y.shape == (20, 4)
    assert np.isfinite(y).all()
    assert set(np.unique(y[:, 3])).issubset({0.0, 1.0})


def test_ngspice_smoke(tmp_path):
    y = evaluate_spice(np.array([[2, 2, 2, 2, 2, 2]]), tmp_path / "cache.json")
    assert y.shape == (1, 4)
    assert np.isfinite(y).all()
    assert y[0, 0] > 0.0
    assert y[0, 1] > 0.0


def test_explicit_diversity_changes_selection():
    candidates = np.array([[0] * 6, [1] * 6, [2] * 6, [4] * 6])
    scores = np.array([1.0, 0.99, 0.98, 0.80])
    plain = diverse_topk(candidates, scores, 2, diversity_weight=0.0)
    diverse = diverse_topk(candidates, scores, 2, diversity_weight=1.0)
    assert batch_diversity(diverse) > batch_diversity(plain)
