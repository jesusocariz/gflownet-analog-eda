import numpy as np

from pareto_experiment import area_proxy, monte_carlo_hypervolume, nondominated_mask


def test_nondominated_mask():
    values = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    assert nondominated_mask(values).tolist() == [True, True, False, True]


def test_hypervolume_improves_with_better_front():
    samples = np.random.default_rng(0).random((10_000, 2))
    assert monte_carlo_hypervolume(np.array([[0.2, 0.2]]), samples) > \
           monte_carlo_hypervolume(np.array([[0.5, 0.5]]), samples)


def test_area_proxy_is_positive():
    assert np.all(area_proxy(np.array([[0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4]])) > 0)
