import numpy as np

from advanced_pareto_experiment import (FeedbackGFlowNet, categorical_select,
                                        nsga2_surrogate_select, objective_spacing)


def test_objective_spacing_uniform_front():
    uniform = np.column_stack([np.linspace(0, 1, 6), np.linspace(1, 0, 6)])
    clustered = uniform.copy(); clustered[1:5] = clustered[0] + .01
    assert objective_spacing(uniform) < objective_spacing(clustered)


def test_categorical_selection_is_unique():
    pool = np.asarray([[i // 5, i % 5, 0, 0, 0, 0] for i in range(25)])
    chosen = categorical_select(pool, np.ones(25), 8, seed=0, diversity_weight=.3)
    assert len(np.unique(chosen, axis=0)) == 8


def test_gflownet_donors_integrate_with_nsga2():
    pool = np.asarray([[a, b, 0, 0, 0, 0] for a in range(5) for b in range(5)])
    predicted = np.column_stack([pool[:, 0], pool[:, 1], -pool[:, 0], -pool[:, 1]])
    feedback = FeedbackGFlowNet(seed=0)
    feedback.update(pool, np.linspace(1, 3, len(pool)), pool[:3], steps=2)
    donors = feedback.donors(pool, 8)
    chosen = nsga2_surrogate_select(pool, predicted, 5, seed=0, generations=2,
                                    population_size=12, gfn_donors=donors)
    assert chosen.shape == (5, 6)
    assert len(np.unique(chosen, axis=0)) == 5
