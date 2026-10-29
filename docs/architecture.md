# Architecture

```mermaid
flowchart TD
  E[Changed coefficients, load, or prescribed value] --> A[scikit-fem assembly and constraint elimination]
  A --> I[Actual operator, RHS, mapping and precision identities]
  I --> P[Shared primal-adjoint memory and paired quota/refresh policy]
  P --> C[Projected initial guess]
  C --> K[Fixed-preconditioner SciPy classical solve]
  K --> V[Original residual and backward-error acceptance]
  V -->|accepted| F[Field and objective]
  F --> D[Original discrete derivative and transpose solve]
  D --> V2[Independent adjoint residual acceptance]
  V2 --> G[Implicit objective gradient]
  V -->|failed or cancelled| R[Explicit status, no current field]
  V2 -->|failed| R
  G --> S[Local field editor or separate inverse-design SDK client]
  S --> B[Versioned equations and result export]
  B --> X[Standalone standard-library checker]
```

The editable diagram source is [architecture.mmd](architecture.mmd). The solver is independent of the field editor. The external application in `integration/design_app` installs as a second distribution and imports only public runtime interfaces.

`assembly.py` owns constrained discrete equations and analytic parameter derivatives. `runtime.py` owns the process-local cache, correction bank, classical methods and acceptance. `policy.py` owns compact data-only joint-action weights. `artifacts.py` owns versioned CLI exports and `checker.py` independently recomputes their stored-equation checks. `server.py` serializes editing work and revokes obsolete results.

A correction bank stores at most the configured number of orthonormal vectors shared between primal and adjoint solves. The trained plan assigns role quotas summing to that budget and a refresh/skip choice; rank is actual valid available directions, not reserved empty slots. Changed operator products are recomputed. Changed mesh, DOF ordering, constraints or precision revoke mappings. No factorization is reused under a different operator identity. Failed candidates do not become accepted memory.

This release does not persist reusable memory to disk. A restart is an empty runtime; exported results are evidence, never silently loaded as a solve cache. Replaying exported inputs reassembles and recomputes.
