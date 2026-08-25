# GFlowNet-guided multi-objective analog circuit design

A reproducible research prototype exploring whether Generative Flow Networks
(GFlowNets) can improve data-efficient Pareto search in analog electronic design
automation (EDA).

The benchmark optimises a common-source NMOS amplifier over a discrete space of
15,625 designs. It compares random and Sobol sampling, scalarisation, direct
categorical reward sampling, surrogate-assisted NSGA-II, a standalone Pareto
GFlowNet, and a feedback hybrid in which a persistent GFlowNet acts as a learned
mutation donor for NSGA-II.

The four minimisation objectives are negative gain, negative bandwidth, power,
and a technology-independent area proxy. Circuit validity, operating point,
gain, bandwidth, and power are evaluated with ngspice.

## Main result

Under a matched budget of 96 circuit evaluations and five paired seeds, the
GFlowNet--NSGA-II hybrid obtains the best mean hypervolume and IGD. The
differences from surrogate NSGA-II are small and not statistically significant
at this sample size. The standalone GFlowNet produces the largest and most
uniformly spaced non-dominated archive. These results motivate a larger study;
they do not establish superiority over NSGA-II.

The full methodology, plots, results, and limitations are in
[`GFlowNet_Analog_EDA_Report_v2.pdf`](GFlowNet_Analog_EDA_Report_v2.pdf).

## Requirements

- Python 3.10 or newer
- [ngspice](https://ngspice.sourceforge.io/) on `PATH` for real simulations
- XeLaTeX only if rebuilding the PDF report

Install ngspice on macOS with `brew install ngspice`, or on Ubuntu/Debian with
`sudo apt-get install ngspice`. Then create a virtual environment and install
the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

## Reproduce the reported experiment

From the repository root:

```bash
python3 -m pytest -q
python3 advanced_pareto_experiment.py \
  --backend ngspice \
  --reference-size 1024 \
  --seeds 0 1 2 3 4 \
  --rounds 6 \
  --batch-size 12 \
  --gfn-steps 250 \
  --diversity-weight 0.3
python3 report_plots.py
xelatex GFlowNet_Analog_EDA_Report_v2.tex
```

The complete run can take several minutes. `results/ngspice_cache.json` stores
previous simulations and makes reruns deterministic and faster. It may be
deleted to force every ngspice evaluation to run again.

For a quick end-to-end check using the analytical development oracle:

```bash
python3 advanced_pareto_experiment.py \
  --backend compact --reference-size 128 --seeds 0 \
  --rounds 1 --batch-size 4 --gfn-steps 5 \
  --output-dir results/smoke_compact
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `advanced_pareto_experiment.py` | Main seven-method comparison and feedback hybrid |
| `experiment.py` | Shared surrogate, GFlowNet, Sobol, and diversity utilities |
| `pareto_experiment.py` | Objectives, dominance, hypervolume, and IGD utilities |
| `simulator.py` | Discrete design space and fast analytical oracle |
| `spice_simulator.py` | ngspice netlist, parser, and persistent cache |
| `report_plots.py` | Figures used by the report |
| `test_*.py` | Unit and ngspice integration tests |
| `results/advanced_pareto_ngspice/` | Data and figures reported in v2 |
| `GFlowNet_Analog_EDA_Report_v2.*` | Current report source and PDF |

The earlier scripts remain because the main experiment imports their tested
surrogate and Pareto utilities. Only the locked `advanced_pareto_ngspice`
results used by the current report are versioned.

## Scope and limitations

This is a learning-oriented benchmark, not a production analog-design tool. The
portable Level-1 MOS model is useful for reproducibility but is not a foundry
PDK. The area objective is a proxy rather than extracted layout area, the design
space is quantised, and the statistical comparison uses only five seeds. A
stronger follow-up should use an open PDK, process/voltage/temperature corners,
Monte Carlo mismatch, tighter wall-clock accounting, and more paired runs.

## License

Code is released under the MIT License. The report and generated figures are
provided for research and portfolio use; please cite the repository metadata in
[`CITATION.cff`](CITATION.cff) when reusing this work.
