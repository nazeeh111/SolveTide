# Public API and statuses

`assemble(family='heat', n=16, parameters=None, boundary='left') -> System` builds immutable float64 reduced equations. Read `A`, `b`, analytic derivative `(dA,db)` pairs, free/fixed mappings, prescribed values, node coordinates, triangle cells and version identities. `System.objective(x,target)` returns the nodal-mean objective and its derivative over free DOFs; `System.expand(x)` restores boundary DOFs.

`SolverConfig` declares numerical tolerances, classical work, time, input-byte and factor-storage limits, strategy and retained direction budget. Invalid configurations refuse. `Runtime(config,policy)` holds one chronological stream. `solve(system,rhs=None,role='primal',cancel=None) -> SolveResult` computes a candidate and checks original equations. `gradient(system,forward,target=None,cancel=None) -> GradientResult` rechecks that exact primal result and solves the original transpose equations. The supported matrices are symmetric, so the same reduced operator is used. Do not share one Runtime across threads; editor requests serialize computation.

The result's `accepted` property requires both a converged status and data. Failed solves always have `x=None`. Failed gradients have no fresh objective/gradient. Never display an intermediate candidate as an accepted new field. `summary()` excludes arrays and produces strict-JSON-safe diagnostics; `state()` reports schema1 cache metadata. `reset()` discards all process-local state. A new process also starts empty; there is no disk-cache loading API.

| Status | Meaning |
| --- | --- |
| converged | Original residual and backward-error rules met |
| max_work / work_limit | Classical iteration or matrix-vector budget exhausted |
| residual_rejected / backward_error_rejected | Independent acceptance failed |
| cancelled / time_limit | Cooperative stop; no current solution |
| resource_limit | Dimension, byte, magnitude or conservative stored-factor budget refused |
| invalid_input / numerical_failure | Unsupported data or failed numerical arithmetic/setup |
| forward_failed / forward_rejected / stale_forward | Gradient requires a current accepted primal |
| adjoint_failed / invalid_objective / invalid_derivative | No current gradient |

`SolveResult` includes true residual, relative residual, backward error, conditioning enclosure, iterations, counted matrix-vector work, total/stage times, retained memory, cache event, actual reused rank, planning features/action and exact identities. Gershgorin lower bounds often fail to provide a useful conditioning bound; null means unavailable. The independent conditioning study uses small-system eigenvalues and states its limits.

`RankPolicy.load()` reads only versioned JSON coefficients. `enabled=False` is a deployment gate: CLI/editor select a plain warm Runtime rather than invoking the model or maintaining its bank. Researchers can explicitly force the policy for the frozen comparison. The runtime still checks all fields/gradients. See [runtime contracts](runtime.md) for update formulas and policy interfaces.

The two-family API is intended as a reusable solve/sensitivity component. External applications can provide their own objective derivative as `rhs` with `role='adjoint'` and compute explicit parameter terms. External custom matrices must establish SPD and supply consistent derivatives/constraints. The public built-in objective has no explicit parameter term and excludes prescribed DOFs; it is not a universal objective/autodifferentiation system.
