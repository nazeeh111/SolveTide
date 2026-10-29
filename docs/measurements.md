# Executed measurements

Local CPU research results, three chronological repeats per held-out trajectory. Timings include assembly, feature/model/basis work, classical setup/solves and residual verification. Independent direct-reference checks are recorded outside those timings. Scheduler load varied; these finite examples establish no universal acceleration or latency guarantee.

The trained paired quota/refresh model is **disabled** by development validation. The learned rows below explicitly force that frozen model for measurement; deployed CLI/editor use ordinary warm starts when disabled. Confirmation does not enable or tune it.

| Family | Method | Median sequence(s) | Failed steps /120 | Retained MiB max | Forward error max | Gradient error max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| heat | warm | 0.6587 | 3 | 0.13 | 7.1e-09 | 6.7e-10 |
| heat | direct | 0.8638 | 0 | 0.98 | 4.4e-16 | 3.9e-15 |
| heat | cold | 0.9401 | 0 | 0.10 | 4.4e-07 | 2.2e-08 |
| heat | fixed | 0.9428 | 0 | 0.27 | 6.7e-09 | 2e-09 |
| heat | balanced | 0.9743 | 0 | 0.27 | 7.2e-09 | 1e-09 |
| heat | learned | 0.9794 | 1 | 0.27 | 5.1e-09 | 6.3e-10 |
| heat | amg_reuse | 1.1199 | 0 | 0.41 | 1.8e-07 | 9.2e-08 |
| heat | amg | 1.1847 | 0 | 0.41 | 1.5e-09 | 4.7e-10 |
| heat | joint | 1.2401 | 0 | 0.27 | 6.7e-09 | 2e-09 |
| heat | drift | 1.6009 | 0 | 0.27 | 7.1e-09 | 2.3e-09 |
| heat | separate | 1.6979 | 0 | 0.37 | 5.3e-09 | 4e-10 |
| heat | gcrot | 2.4363 | 0 | 0.29 | 5e-06 | 7.1e-06 |
| elasticity | direct | 1.8238 | 0 | 4.60 | 1e-13 | 3.1e-13 |
| elasticity | joint | 2.1842 | 0 | 0.80 | 2.4e-09 | 7.7e-10 |
| elasticity | warm | 2.4140 | 12 | 0.53 | 3.5e-09 | 2.2e-10 |
| elasticity | separate | 2.5474 | 0 | 1.25 | 2.2e-09 | 1.2e-10 |
| elasticity | drift | 2.7261 | 0 | 0.80 | 3.5e-09 | 1.2e-09 |
| elasticity | cold | 3.0787 | 0 | 0.47 | 1.9e-09 | 2.8e-10 |
| elasticity | balanced | 3.0928 | 0 | 0.80 | 2.8e-09 | 2.5e-10 |
| elasticity | amg | 3.1290 | 6 | 2.03 | 5.6e-10 | 2.1e-10 |
| elasticity | amg_reuse | 3.4126 | 6 | 2.03 | 6.6e-08 | 4.2e-08 |
| elasticity | learned | 3.6615 | 0 | 0.75 | 1.9e-09 | 2.2e-10 |
| elasticity | fixed | 4.0224 | 0 | 0.80 | 2.9e-09 | 7.1e-10 |
| elasticity | gcrot | 4.8605 | 0 | 0.84 | 5.3e-06 | 7.8e-06 |

All retained basis controls have total rank 12, including separate 6+6 banks. Factor/AMG storage differs and is reported honestly. A failed step is retained in totals; a cheaper failed run is not a successful speedup. `gcrot` iterations are outer iterations; matrix-vector counts expose inner work. AMG includes symmetric smoothing and elasticity rigid-body candidates. The implemented memory correction is an initial projection, not full augmented CG.

## Complete inverse design

Identical target, start, bounds and stopping criteria per method; target generation and final direct verification included. Reported successful objectives below come from final direct solves. Failed attempts retain their consumed time and last accepted state in raw traces; they are not successful optimization speedups.

| Family | Method | Successful runs /3 | Median observed time(s) | Max verified objective among successes | Median evaluations |
| --- | --- | ---: | ---: | ---: | ---: |
| heat | direct | 3 | 0.2311 | 2.81e-15 | 20 |
| heat | warm | 3 | 0.4390 | 2.81e-15 | 20 |
| heat | fixed | 3 | 0.4031 | 2.81e-15 | 20 |
| heat | joint | 3 | 0.3938 | 2.81e-15 | 20 |
| heat | learned | 3 | 0.4829 | 2.81e-15 | 20 |
| heat | amg_reuse | 3 | 0.5444 | 2.81e-15 | 20 |
| elasticity | direct | 3 | 0.6538 | 4e-18 | 20 |
| elasticity | warm | 0 | 0.1647 | unavailable | 2 |
| elasticity | fixed | 3 | 1.2923 | 4e-18 | 20 |
| elasticity | joint | 3 | 1.2098 | 4.01e-18 | 20 |
| elasticity | learned | 0 | 0.1824 | unavailable | 2 |
| elasticity | amg_reuse | 3 | 1.4309 | 4e-18 | 20 |

## Isolated process memory

One separate process per family/method, n40. Process peak RSS includes Python, imports, native libraries, assembly and temporary workspaces; retained storage is a separate metric. These are measurements, not hard OS allocation limits.

| Family | Method | Process peak RSS(MiB) | Retained storage(MiB) |
| --- | --- | ---: | ---: |
| heat | direct | 70.8 | 0.94 |
| heat | warm | 64.2 | 0.12 |
| heat | gcrot | 66.2 | 0.27 |
| heat | fixed | 64.2 | 0.27 |
| heat | joint | 64.0 | 0.27 |
| heat | learned | 64.1 | 0.24 |
| heat | amg_reuse | 70.7 | 0.39 |
| heat | separate | 64.2 | 0.37 |
| elasticity | direct | 92.8 | 4.29 |
| elasticity | warm | 75.6 | 0.50 |
| elasticity | gcrot | 81.1 | 0.80 |
| elasticity | fixed | 75.4 | 0.80 |
| elasticity | joint | 80.0 | 0.80 |
| elasticity | learned | 75.9 | 0.75 |
| elasticity | amg_reuse | 85.1 | 1.93 |
| elasticity | separate | 77.1 | 1.25 |

## Measured editor response tiers

Actual changed-input loopback requests include computation, serialization, transport and parse. Browser paint is excluded. Three local observations per row; maximum is observed, not a latency bound. No universal real-time claim.

| Family | Edge divisions | Median complete response(s) | Max observed(s) | Refused /3 |
| --- | ---: | ---: | ---: | ---: |
| heat | 20 | 0.0322 | 0.0781 | 0 |
| heat | 32 | 0.0604 | 0.0626 | 0 |
| heat | 48 | 0.1667 | 0.1688 | 0 |
| elasticity | 20 | 0.0882 | 0.0940 | 0 |
| elasticity | 32 | 0.2200 | 0.2570 | 0 |
| elasticity | 48 | 0.4785 | 0.5924 | 0 |

Raw [confirmation](../experiments/confirmation.json), [optimization](../experiments/optimization.json), [memory](../experiments/memory.json), [training](../experiments/training.json), and [independent numerical validation](../experiments/numerical-validation.json) retain statuses and traces. Source hashes, protocol and model hashes bind each study.

PCGBandit, G-RANS and SOLVAX were studied as references and were not benchmarked here. No superiority over them is claimed. Fixed chronological order does not use SKR offline sorting. Small PDE examples, one machine, original synthetic data and a simple nodal objective limit generalization. Solver convergence does not certify discretization or model truth.
