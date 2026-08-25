# Primary project objective

## Diverse Pareto-front discovery for analog circuit design

The primary objective is to determine whether a multi-objective GFlowNet can
recover a high-quality and diverse approximation of the analog-design Pareto
front using fewer high-fidelity SPICE evaluations than established sampling and
scalarisation baselines.

The present benchmark optimises four conflicting quantities:

1. maximise low-frequency voltage gain;
2. maximise −3 dB bandwidth;
3. minimise DC power consumption;
4. minimise a transparent area proxy based on transistor geometry, resistance
   and capacitance.

All candidate designs must also satisfy the simulator and transistor
operating-region validity checks. Performance is judged primarily by
**hypervolume per SPICE evaluation**, with inverted generational distance (IGD),
number of non-dominated designs and archive/input-space diversity as supporting
metrics.

The research question is:

> Under an identical SPICE-call budget, does reward-proportional GFlowNet
> sampling discover a broader and higher-quality set of Pareto-optimal analog
> designs than random sampling, Sobol sampling and preference scalarisation?

The earlier global-surrogate-accuracy study remains a secondary experiment. Its
code and results are intentionally retained in `experiment.py`,
`PRELIMINARY_FINDINGS.md`, `NGSPICE_FINDINGS.md` and `results/{compact,ngspice}`
for inclusion as complementary evidence in the final report.

