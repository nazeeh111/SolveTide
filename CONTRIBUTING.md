# Contributing

Reproduce the numerical tests before proposing changes. Include the actual constrained equations, parameters, tolerances, chronological input sequence and observed status for solver defects. Keep physical-model changes separate from solver-performance claims. New acceleration claims need full setup/verification cost, matched tolerances, held-out trajectories and complete optimization comparisons.

Do not silently relax acceptance or reuse incompatible cache mappings. New families must use the public API and supply original discrete derivatives plus independent forward/gradient checks.
