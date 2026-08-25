# Advanced Pareto experiment findings

## Locked protocol

- 1,024-point Sobol ngspice reference set.
- Five seeds and 96 evaluations per method (24 initial + six batches of 12).
- Four objectives: maximise gain and bandwidth; minimise power and area proxy.
- 100,000 common Monte Carlo points for deterministic hypervolume comparison.
- Seven-model Extra Trees surrogate ensemble.
- Baselines: random, Sobol, preference scalarisation, direct categorical sampling
  from the GFlowNet reward, and surrogate-assisted NSGA-II.
- GFlowNet: 250 Trajectory-Balance updates per acquisition round.

## Final results

| Method | Hypervolume ↑ | IGD ↓ | Non-dominated ↑ | Spacing ↓ |
|---|---:|---:|---:|---:|
| GFlowNet–NSGA-II feedback | **0.526 ± 0.046** | **0.161 ± 0.008** | 32.6 ± 4.3 | 0.131 ± 0.013 |
| Surrogate NSGA-II | 0.520 ± 0.053 | 0.164 ± 0.016 | 33.0 ± 1.9 | 0.138 ± 0.024 |
| Scalarisation | 0.461 ± 0.032 | 0.175 ± 0.007 | 37.2 ± 5.5 | 0.108 ± 0.043 |
| Categorical reward | 0.410 ± 0.056 | 0.179 ± 0.017 | 39.4 ± 4.7 | 0.110 ± 0.035 |
| Pareto GFlowNet | 0.401 ± 0.045 | 0.178 ± 0.013 | **41.6 ± 4.3** | **0.097 ± 0.026** |
| Sobol | 0.370 ± 0.048 | 0.184 ± 0.011 | 28.0 ± 1.0 | 0.124 ± 0.025 |
| Random | 0.290 ± 0.067 | 0.197 ± 0.016 | 27.4 ± 3.0 | 0.139 ± 0.022 |

## Conclusions

The feedback hybrid obtains the best mean hypervolume and IGD, narrowly ahead
of surrogate-assisted NSGA-II. The mean changes are +0.006 hypervolume and
-0.003 IGD. With only five paired seeds, Wilcoxon tests are not significant
(`p=0.8125` and `p=1.0` respectively), so this is evidence of competitiveness,
not superiority.

The standalone GFlowNet produces the largest non-dominated archive and the
lowest spacing statistic, indicating a numerous and relatively uniform set of
objective-space trade-offs. The hybrid inherits NSGA-II's stronger front
extension while modestly improving its mean spacing.

The categorical ablation obtains slightly higher hypervolume than the GFlowNet
using the same terminal reward. Therefore, the current results do not establish
that amortised GFlowNet training improves terminal solution quality. They do
suggest that flow-based sampling changes coverage: it finds more non-dominated
solutions with more uniform nearest-neighbour spacing.

This is a useful research result. More paired seeds are needed to establish
whether feedback reliably improves NSGA-II. The next GFlowNet should be trained directly
against a hypervolume-improvement or archive-conditioned reward, rather than an
approximate preference score. Evaluation should then include training time and
repeat the locked comparison on an open-PDK circuit.
