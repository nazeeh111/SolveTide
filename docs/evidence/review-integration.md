# Independent optimization integration review

**Verdict: no actionable blocker found for the requested failure-retention, outcome-labeling and learned-gate requirements.**

Reviewed immutable snapshot005:

| Source | SHA-256 |
| --- | --- |
| integration/design_app/design_app.py | 88cfdaa99a222cfa747e2add072733c08eab3896d9f581803246999f64d0571b |
| studies/optimization.py | 06fd4a3c893425f7e8d4783cde731a1d707d7f2f8a98c4705ccb5649139f9641 |

The canonical source and experiment files were read only. Copies and hashes are in canonical-005 under this uniquely named review folder.

## Executed focused evidence

`repro-integration-005.py` imports the frozen application with injected solver results and executes the frozen full study runner using a capture-only writer. It does not perform physical optimization or write canonical experiment outputs. Evidence is `review-integration-005-evidence.json`.

- An accepted evaluation followed by an adjoint refusal returns `success=False`, `status='solver_failed'`, and null top-level `parameters`, `objective` and `direct_checked_objective`.
- The refused proposal appears under `failed_parameters`; the previous fully accepted evaluation remains under `last_accepted_parameters` and `last_accepted_objective`. History records the refusal status, its solver summaries and a null current objective.
- A refusal on the very first evaluation leaves both last-accepted fields null.
- With the development policy disabled, a normal `strategy='learned'` request loads without force, uses the ordinary warm runtime and records requested learned/effective warm/force false. Its runtime receives no policy object.
- An explicit `force_policy=True` research call uses learned and records force true. The production CLI exposes no implicit forcing option. The research study runner explicitly passes the flag, and nonlearned strategies still record force false.
- With every run's second evaluation deliberately refused, the complete frozen runner finishes all **36 unique combinations** of six methods, two families and three repeats, retains all 36 failure records, and leaves all refused achieved-outcome fields null. One capture-only final write is reached; the first failure does not abort the study.

## Source contract check

The runtime, tolerances, two-coefficient target, initial point, bounds and solver budget remain the same across methods: n=24 in the study, material/contrast truth1.8/4.2, start0.65/0.6, rtol1e-9, atol1e-11, basis cap12, per-solve15s, L-BFGS-B maxiter40, gtol1e-8 and ftol1e-12. The new code changes how refusal is recorded and how production versus forced research policy selection is represented. Successful or ordinary optimizer-stopped endpoints still receive a separate direct objective computation; a solver refusal returns before such an endpoint is labeled achieved.

This verdict validates control flow and metadata with injected failures. It does not certify the numerical outcomes or timings in the final optimization artifact. Those require the independent artifact audit and actual numerical runs being handled by the parent reviewer.
