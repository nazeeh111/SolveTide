# Owner walkthrough

1. Install the pinned wheel and run `solvetide studio`. Open the printed loopback URL. Start with heat; compute and inspect the convergence state, residual, backward error and all four sensitivities.
2. Use the local material edit. The old image becomes stale immediately; compute the changed system and inspect changed operator identity plus refreshed memory. This is a new numerical solution, not cached image interpolation.
3. Change mesh resolution or prescribed edge. Compute and inspect `mapping_invalidated`. Unsupported old directions are discarded.
4. Use the one-iteration limit. The new result refuses when the declared tolerance is unmet; no current field/gradient/export is offered. Run sparse-direct fallback and inspect the freshly accepted solution.
5. Switch to elasticity and repeat. The same API solves displacement with consistent constraints and adjoints.
6. Select learned policy. The development gate controls whether it runs. In this release validation disables it, so the effective strategy is ordinary warm start with no learned-model overhead. Forced research evaluation is separate.
7. Export the fresh result. Editor exports contain current fields, IDs and original equations. CLI bundles additionally include discrete parameter derivatives and the standalone checker contract. Run `solvetide verify` on a CLI bundle.
8. Install the separate design client and run `solvetide-design`. Read its stopping state, recovered coefficients and independently recomputed final objective. A faster individual solve does not by itself mean faster optimization.
9. Read the confirmation, optimization and memory reports. Treat failures and negative speed results as results. See conditioning and mesh studies before interpreting an accurate discrete solve as an accurate physical model.

Exports retain exact input versions and equations; restarting the application begins with empty solve memory. Research protocol, trained weights, editable diagram, scripts and permitted evidence are included in source. No public compute endpoint is deployed.
