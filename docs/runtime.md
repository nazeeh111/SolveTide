# Solver contract and joint memory

SolveTide uses classical sparse solvers. A learned policy can choose a bounded
initial correction; it cannot accept a numerical answer. Supported assemblers
establish symmetry and positive definiteness by positive material coefficients
and explicit Dirichlet constraints. Runtime checks on externally supplied
matrices check shape, finite data, symmetry and positive diagonal; these do not
constitute a proof of positive definiteness. External callers must supply SPD
matrices. Arbitrary singular and indefinite models are outside the API contract.

## Accepted results and limits

An accepted solve has `status == "converged"` and a non-None, read-only `x`.
Every other status has `x=None`, even when a solver produced an intermediate
candidate. Gradient computation requires an accepted primal solution of the
current equations, recomputes its residual, and requires a separately accepted
adjoint. A failed adjoint exposes neither a new objective nor a gradient.

The independent acceptance computation uses the copied, original CSR matrix:

```
r = b - A @ x
||r||2 <= max(atol, rtol * ||b||2)
eta = ||r||inf / (||A||inf * ||x||inf + ||b||inf)
eta <= max(rtol, atol / denominator)
```

A zero denominator and zero residual give eta=0. This is normwise backward
error, not a solution-error bound. `conditioning` reports a Gershgorin spectral
enclosure. Only a strictly positive lower bound supplies the displayed upper
bound on the spectral condition number; otherwise the condition bound is None.
No discrete error estimate establishes physical truth or discretization accuracy.

Default limits are rtol=1e-9, atol=1e-11, 2,000 iterations, ten seconds,
40,000 unknowns, 2,000,000 nonzeros, 256 MiB input and 256 MiB stored-factor space.
Direct factorization is additionally limited to 4,000 unknowns. Individual
input matrix/RHS entries and accepted solution components must have absolute
magnitude at most 1e150. Larger finite inputs refuse instead of risking
overflow; this conservative numerical envelope is not a condition-number
guarantee. Vector norms use scaled BLAS arithmetic, and nonfinite residual,
tolerance, backward-error denominator or gradient arithmetic refuses. A result
summary replaces any nonfinite diagnostic number with JSON null.

Before SuperLU factorization, a conservative stored-factor bound must fit:

```
L + U entries <= n*(n+1)
bound = 16*n*(n+1) + 32*n + 16 bytes
```

This allows float64 values with int64 indices, two int64 row-pointer arrays,
and two int64 permutation arrays. The bound is checked before calling `splu`,
and a one-byte budget therefore refuses without invoking factorization. Actual
materialized L/U/permutation storage is checked again after factorization.
Larger sparse systems can refuse even when their actual fill would fit; fair
direct comparisons must use systems within this stated cap. SuperLU temporary
workspace, native representation duplication, allocator overhead and process
RSS are outside this stored-array bound. This is not a hard process memory cap.

Cancellation and deadlines are checked before/after opaque SciPy setup
calls and inside iterative matrix-vector products/callbacks. They are cooperative,
not process-level preemption. Iterative matrix-vector work is capped at
`3*maxiter+10`, including GCROT's inner work. GCROT's reported iterations are outer
iterations, while `matvecs` reports the separately capped inner work.

`cancel` is a zero-argument callable, for example `threading.Event().is_set`.
A Runtime instance is a single chronological stream and is not thread-safe.

## Identities, invalidation and cache state

The runtime canonicalizes a private CSR matrix copy and hashes actual matrix
storage, shape, RHS bytes, mesh coordinates, cells, free/fixed indices, family,
precision and assembler mapping identity. It does not trust a supplied operator
hash. Mutation of a caller matrix changes the next identity. A different mapping
or precision clears memory, warm states, factors and AMG state. A changed
operator refreshes any used matrix-applied basis products from the new matrix;
rank zero skips those products entirely. Local
coefficient edits therefore recompute the entire accepted solution.

- `cold`: zero initial state, fixed Jacobi CG.
- `warm`: previous accepted vector of the same role, fixed Jacobi CG.
- `fixed`: fixed-rank chronological orthonormal solution recycling, then CG.
- `joint`: bounded, shared cost-weighted primal/adjoint solution memory, then CG.
- `drift`: joint control rejecting the initial correction when relative
  Frobenius operator drift exceeds 0.15 (configurable).
- `learned`: joint memory with a policy-selected primal/adjoint allocation
  plan and correction ranks. The plan is chosen before the primal and held
  through its paired adjoint. Legacy choose-only rank policies are also
  supported. Missing/invalid policies fall back to rank zero; exceptions or
  invalid plans disable that policy for the stream.
- `gcrot`: SciPy flexible GCROT(m,k), m=20, k=memory_budget, with every C product
  discarded and recomputed for the current operator. This is the established
  recycled Krylov comparison, distinct from the initial-correction method.
- `direct`: SuperLU factorization reused only for exactly matching operator
  bytes. A changed RHS can reuse it, a changed operator cannot.
- `amg`: PyAMG smoothed aggregation rebuilt on changed operators; fixed during
  each CG call. Elasticity supplies restricted rigid-body candidates for x/y
  translation and rotation (-y,x); constrained components are removed, so
  these are near-null modes, not singular modes of the supported operator.
  Both smoothing sweeps and the declared hierarchy symmetry are symmetric. `amg_reuse` keeps the old hierarchy across compatible edits,
  also fixed during each call. Neither is a varying CG preconditioner.

All strategies recheck the original equations after solving. Factorization and
AMG failures are explicit numerical failures, not a claim that an answer solved.

## Specific joint update

Let Q be the retained Euclidean-orthonormal directions. Accepted primal and
adjoint solutions are orthogonalized against Q with two passes; linearly
dependent candidates (relative residual norm <=1e-10) do not increase rank.
The total number of Q directions across both roles is at most memory_budget.
There are additionally at most two accepted warm vectors, one per role. Those
vectors, matrix snapshot, factors and hierarchy storage are counted in reported
retained `memory_bytes`; they are not part of Q's rank budget. Transient work
arrays, Python object overhead and BLAS allocation are not measured peak RSS.

For each solve, starting from its role's warm state x0:

1. Extract current residual projection features; choose rank first. When rank
   is positive, recompute AQ with the actual current operator. Compute energy relevance
   `abs(q_i.T @ (b-A@x0))/sqrt(q_i.T @ A @ q_i)`.
2. `fixed` uses chronological retained directions. The joint variants select
   the requested rank by descending current relevance.
3. Form H=Q.T@A@Q, diagonalize this small symmetric matrix, and discard
   nonpositive/numerically deficient modes (relative cutoff 1e-13).
4. Apply `x0 += Q @ solve(H, Q.T @ (b-A@x0))` in the retained eigenspace.
   Reject nonfinite corrections or residual amplification above twice the
   initial residual. This is an initial Galerkin correction, not augmented CG.
5. Run ordinary fixed-preconditioner CG and independently test the result.
6. On acceptance, update the role's measured iterations and residual reduction.
   A new direction receives utility `max(1e-6,reduction)*(1+iterations)`.
   Selected directions blend 80% prior utility and 20% current measured work
   reduction. Utilities decay by 0.95 after a new retained direction.
7. Learned paired plans first enforce their role quotas by evicting the
   lowest-utility directions of any over-quota role. A zero role quota prunes
   its existing directions and prevents new directions of that role. This
   deterministic cache pruning happens at plan adoption, even if the upcoming
   solve fails; no failed solution is ever retained. For other joint variants,
   if over budget, evict the smallest utility divided by the
   number of directions belonging to that role, preserving a role's last
   direction when another role has more than one. This is the explicit bounded
   role-allocation rule. `fixed` evicts the oldest direction instead.

This is an experimental cost-aware solution-memory rule. It does not establish
novelty for recycling, POD, residual verification, or primal/adjoint reuse.
The overhead can exceed iteration savings and must be included in studies.

## Learned joint policy interface

Pass `Runtime(SolverConfig(strategy="learned"), policy=policy)`, where the
preferred method is `policy.plan(features, max_rank_budget)`. It returns:

```python
{"primal_rank": 6, "adjoint_rank": 6,
 "primal_quota": 6, "adjoint_quota": 6, "refresh": True}
```

All ranks and quotas are nonnegative integers, ranks cannot exceed their role
quotas, and both quotas sum to at most memory_budget. The role quota controls
how many retained directions originate from that role. Both roles can draw
correction directions from the shared space according to current energy
relevance, up to their chosen rank and the currently available dimension.
`refresh=False` skips both initial corrections while maintaining accepted
history under the quotas. Every requested matrix-applied product is freshly
formed; no old AQ can cross an operator change. Refresh here means activating
the refreshed correction work, not reusing a stale product when it is false.

The policy is evaluated only before the primal. Its allocation and raw feature
vector are frozen for the matching operator/mapping adjoint. An orphan adjoint
uses a classical warm fallback. A next primal can choose a different allocation.
The nine numerical planning features are:

| Feature | Meaning |
| --- | --- |
| operator_drift | Relative Frobenius norm of current minus previous operator |
| residual_projection | Norm of Q.T times initial primal residual divided by its norm |
| previous_primal_reduction | Last primal initial-correction residual reduction |
| previous_adjoint_reduction | Last adjoint initial-correction residual reduction |
| previous_primal_iterations | Last accepted primal iteration count |
| previous_adjoint_iterations | Last accepted adjoint iteration count |
| available_primal_rank | Number of retained primal-origin directions |
| available_adjoint_rank | Number of retained adjoint-origin directions |
| dimension | Reduced system dimension |

`SolveResult.policy_features` and `policy_action` preserve the exact raw primal
planning information and adopted action; the paired adjoint records the same
information. `Runtime.state()` also exposes them. Invalid plans, quotas beyond
the budget, exceptions and planning overhead above the historical total
classical primal/adjoint solve cost disable the policy and skip corrections.
Historical costs are exponentially averaged from accepted solves only.

For comparison and compatibility, `policy.choose(features,max_rank)` can
instead return one current solve rank. Its seven features are operator_drift,
residual_projection, previous_reduction, previous_iterations, available_rank,
role_adjoint and dimension. This legacy path does not train joint quotas.
Returned ranks are clamped to [0,max_rank]; nonfinite values/exceptions disable
it. Its planning overhead is compared with the current role's classical cost.

Feature extraction, operator comparison, any current AQ, rank selection and
policy inference are timed. Rank zero skips AQ and the projected solve; Q.T@r
feature extraction and identity/operator-drift checks still incur cost. This
limited safeguard does not prove total memory overhead is profitable; the
complete measured comparisons determine that. A trained policy must be fixed
before confirmation trajectories run. Choosing a plan from a future held-out
solution would not be an online experiment. Training labels must measure the
complete forward/adjoint pair and memory effects, not independent rank-only
solve times.

## Gradient equation

For reduced equations A(p)x=b(p) and an objective J(x), solve
`A.T @ lambda = dJ/dx` using the same constraint mapping. Then
`dJ/dp = lambda.T @ (db/dp - dA/dp @ x)`. The assemblers include the derivative
of nonzero Dirichlet elimination in db/dp. Complex objective values, objective derivatives, dA and db refuse before
conversion to real binary64. There is no differentiation through
saved iterates or detached approximate fields. The current API's objective
has no explicit dependence on p; external explicit objective terms must be
added by the application. Current primal/adjoint rtol are identical.

Timing components can overlap: `policy` planning time lies inside `basis_update_projection`. Do not sum these nested components. Use the authoritative `total` or measured sequence wall time for full cost; the reports also retain assembly and independent reference-check time separately. Planning guard time includes quota management, not just regression arithmetic.
