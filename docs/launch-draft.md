# Launch draft (not posted)

I built SolveTide to test a practical question: can a simulation runtime reuse work across design edits and objective-gradient solves without skipping the actual equations?

The initial release supports constrained heat diffusion and linear elasticity through one Python API, a local editor and a separate inverse-design client. Every accepted forward and adjoint result is checked against the assembled sparse system, and changed mesh or constraint mappings invalidate saved state.

The joint-memory policy is trained, but development validation did not justify enabling it. Warm starts remain the default. The release includes the negative result, mature solver controls, chronological held-out trajectories, gradient and mesh checks, original-equation exports and an independent checker.

Source, reproducible evidence and installable release: https://github.com/nazeeh111/SolveTide

Feedback on the numerical contracts, fair comparisons and useful external integrations would be welcome. No external method superiority, adoption or real-time performance is claimed.
