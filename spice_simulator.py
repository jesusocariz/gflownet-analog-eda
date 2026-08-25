"""ngspice backend with a persistent cache for the common-source benchmark."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from simulator import GRID, decode


CACHE_PATH = Path(__file__).with_name("results") / "ngspice_cache.json"


def _netlist(indices: np.ndarray) -> str:
    p = {k: float(v[0]) for k, v in decode(indices).items()}
    # The Level-1 model is intentionally portable. It is a real circuit
    # simulation, but not a substitute for a PDK model.
    kp = 220e-6
    vto = 0.45
    vov_est = np.sqrt(2.0 * p["id_ua"] * 1e-6 / (kp * p["w_um"] / p["l_um"]))
    vg = vto + vov_est
    return f"""Common-source amplifier active-learning benchmark
.model nch nmos level=1 kp={kp} vto={vto} lambda=0.08 gamma=0.35 phi=0.7 cgso=0.15n cgdo=0.08n
.param VDD={p['vdd_v']} VG={vg} RD={p['rd_kohm']}k CL={p['cl_pf']}p
VDD vdd 0 {{VDD}}
VIN in 0 dc {{VG}} ac 1
RD vdd out {{RD}}
M1 out in 0 0 nch w={p['w_um']}u l={p['l_um']}u
CL out 0 {{CL}}
.control
op
wrdata op.dat v(out) i(VDD) v(in)
ac dec 80 10 10G
wrdata ac.dat frequency vdb(out)
quit
.endc
.end
"""


def _simulate_one(x: np.ndarray, timeout: float = 10.0) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="gfn-spice-") as tmp:
        netlist = Path(tmp) / "circuit.cir"
        log = Path(tmp) / "ngspice.log"
        netlist.write_text(_netlist(x), encoding="utf-8")
        try:
            subprocess.run(
                ["ngspice", "-b", "-o", str(log), str(netlist)],
                check=True, timeout=timeout, capture_output=True, text=True, cwd=tmp,
            )
            op = np.loadtxt(Path(tmp) / "op.dat").reshape(-1)
            # Columns are scale/value pairs for v(out), i(VDD), and v(in).
            vout, supply_current, vin = float(op[1]), float(op[3]), float(op[5])
            values = {"vout": vout, "vov": vin - 0.45,
                      "power": -float(decode(x)["vdd_v"][0]) * supply_current * 1000.0}
            ac = np.loadtxt(Path(tmp) / "ac.dat")
            # wrdata repeats the scale column for each requested vector.
            frequency, gain_curve = ac[:, 0], ac[:, -1]
            gain_db = float(gain_curve[0])
            below = np.flatnonzero(gain_curve <= gain_db - 3.0)
            bandwidth = float(frequency[below[0]]) if len(below) else float(frequency[-1])
            feasible = values["vout"] > values["vov"] + 0.05 and gain_db > 0
            return np.array([
                gain_db, bandwidth / 1e6,
                values["power"], float(feasible),
            ])
        except (subprocess.SubprocessError, OSError, ValueError):
            return np.array([-80.0, 0.0, 10.0, 0.0])


def evaluate_spice(x: np.ndarray, cache_path: Path = CACHE_PATH) -> np.ndarray:
    """Evaluate designs in ngspice, reusing every previously simulated point."""
    x = np.atleast_2d(x).astype(int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}
    rows = []
    changed = False
    for design in x:
        key = ",".join(map(str, design.tolist()))
        if key not in cache:
            cache[key] = _simulate_one(design).tolist()
            changed = True
        rows.append(cache[key])
    if changed:
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return np.asarray(rows, dtype=float)
