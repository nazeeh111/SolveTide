SolveTide 0.1.0 is a checked sparse-simulation runtime and research preview.

- One typed Python API for constrained heat and linear elasticity, original-equation forward/adjoint checks, bounded shared correction memory, and a trained paired quota/refresh policy.
- Local field editor, versioned equation exports, standalone standard-library checker, separately installed inverse-design client, editable architecture and actual browser replay.
- 104 numerical, failure, lifecycle and integration tests. Independent small-case direct/finite-difference checks, mesh/conditioning studies, 288 held-out method/sequence runs, 36 complete optimization attempts and isolated process-memory measurements.

The learned policy failed the development gate and is disabled. No general speedup or real-time breakthrough is claimed. The held-out study retains 28 adjoint refusals; the optimization study retains six failed iterative elasticity attempts. Sparse direct completed all 6 family/repeat optimization cases and was the fastest successful method locally. Numerical convergence concerns stored equations, not physical truth or a solution-error enclosure.

Install the runtime wheel and, optionally, the separate design-example wheel. The source archive includes reproduction scripts, original synthetic data, trained weights, all controls and failures, pinned dependencies and exact version hashes. `checker.py` and the example ZIP can be used with Python standard library only. `SHA256SUMS` identifies the released bytes, not a cryptographic author signature.

See the README, docs/measurements.md, docs/reproducibility.md and docs/walkthrough.md for scope, results and owner instructions. The 15-second demo is an edited replay of actual browser states, not real-time footage. No public compute service is deployed.
