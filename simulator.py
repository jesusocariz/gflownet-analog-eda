"""Small deterministic analog-design oracle used before connecting SPICE."""

from __future__ import annotations

import itertools
import numpy as np


GRID = {
    "w_um": np.array([2.0, 4.0, 8.0, 16.0, 32.0]),
    "l_um": np.array([0.18, 0.27, 0.36, 0.54, 0.72]),
    "id_ua": np.array([20.0, 40.0, 80.0, 160.0, 320.0]),
    "rd_kohm": np.array([2.0, 4.0, 8.0, 16.0, 32.0]),
    "cl_pf": np.array([0.2, 0.5, 1.0, 2.0, 5.0]),
    "vdd_v": np.array([1.2, 1.5, 1.8, 2.1, 2.5]),
}
N_VALUES = 5
N_DIMS = len(GRID)


def all_designs() -> np.ndarray:
    return np.asarray(list(itertools.product(range(N_VALUES), repeat=N_DIMS)), dtype=np.int64)


def decode(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.atleast_2d(x)
    return {name: values[x[:, i]] for i, (name, values) in enumerate(GRID.items())}


def evaluate(x: np.ndarray) -> np.ndarray:
    """Return [gain_db, bandwidth_mhz, power_mw, feasible].

    The equations are intentionally transparent. They preserve the central
    trade-offs needed by the acquisition benchmark without pretending to be a
    foundry model.
    """
    p = decode(x)
    w, l = p["w_um"], p["l_um"]
    current = p["id_ua"] * 1e-6
    rd = p["rd_kohm"] * 1e3
    cl = p["cl_pf"] * 1e-12
    vdd = p["vdd_v"]

    mu_cox = 220e-6
    vov = np.sqrt(2.0 * current / (mu_cox * w / l))
    gm = 2.0 * current / np.maximum(vov, 1e-6)
    lam = 0.08 * (0.18 / l) ** 0.65
    ro = 1.0 / np.maximum(lam * current, 1e-12)
    rout = 1.0 / (1.0 / rd + 1.0 / ro)

    cgs = 0.70e-15 * w * l + 0.12e-15 * w
    cgd = 0.10e-15 * w + 0.04e-15 * w / np.sqrt(l / 0.18)
    gain = gm * rout / (1.0 + 0.025 * (w / l) ** 0.35)
    gain_db = 20.0 * np.log10(np.maximum(gain, 1e-9))
    bandwidth_mhz = 1.0 / (2.0 * np.pi * rout * (cl + cgs + cgd)) / 1e6
    power_mw = vdd * current * 1e3

    vds = vdd - current * rd
    feasible = (vds > vov + 0.12) & (vds > 0.15) & (gain_db > 0.0)
    return np.column_stack([gain_db, bandwidth_mhz, power_mw, feasible.astype(float)])


def target_feasible(y: np.ndarray) -> np.ndarray:
    """A useful but non-trivial specification region."""
    return (y[:, 3] > 0.5) & (y[:, 0] >= 18.0) & (y[:, 1] >= 8.0) & (y[:, 2] <= 0.35)

