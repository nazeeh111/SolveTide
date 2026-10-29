# Reproduce SolveTide

Python3.12 is the validated interpreter. Runtime versions are pinned in `pyproject.toml`; study/dev extras pin the free optional packages. `requirements-study.lock` records all installed transitive study/build versions for an exact environment. Install from the release wheel or from the versioned Git tag. The dependency-free standalone `checker.py` can inspect a supplied result independently.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,study]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m studies.validate
```

Use a new output path for CLI and application exports. Files are never silently overwritten.

## Train and evaluate

`studies/protocol.json` is frozen before the delivered training and confirmation. Development uses seeds101–104, meshes8/12/16 and boundaries left/all. Whole seed104 trajectories validate; they do not fit weights. Confirmation uses seeds901–904, meshes20/28/40, boundaries right/both, unseen whole trajectories and larger coefficient contrasts. Deliberate mesh/constraint edits exercise invalidation. Chronology is preserved inside each stream; method order is shuffled across repeats. Each method has the same residual rules and maximum classical work.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.run train
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.run confirm
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.optimization
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.memory
```

Training fits six ridge regressions predicting complete forward-adjoint pair cost for fixed quota/refresh actions. Each pair includes assembly, validation/identity, features, inference, matrix-applied basis, orthogonalization/retention, classical setup/iteration and residual checks. Quotas are feasible capacities; actual ranks are saved. Failures remain in training records with a conservative30s penalty. This is a one-step cost heuristic whose retention consequences are checked by complete chronological validation, not sequence-optimal control. Deployment requires at least5% lower median sequence time than warm and unlearned joint, no failures and improvement for both families. Confirmation never changes weights or enables the model. Forced learned evaluation is clearly distinguished from the actual disabled deployment fallback.

The superseded rank-only development attempt is retained in `experiments/superseded-v0`. It is rejected evidence, not the model or confirmation. No failed preliminary result is silently counted as a success.

Direct-reference field/gradient checks are outside timed production solves and recorded separately. Mesh, finite-difference and conditioning studies are independent of the timing comparison. Sequence time counts complete computations, without plotting and oracle checks. Setup/import and release validation scope are explicit in generated evidence. CPU scheduling load and limited repetitions prevent robust general speedup claims.

All correction methods use a total retained rank12; the separate control uses two role-specific banks6+6. LU and AMG retain their appropriate structures, whose actual storage is counted rather than equated to twelve vectors. The memory study runs each method in an independent process and records high-water RSS, including interpreter/imports, assembly and native temporary workspaces. Retained storage accounting and process RSS are separate metrics; neither is a hard OS memory cap. Direct factors have a conservative stored-array bound before allocation and an actual storage check afterward; opaque native workspace is outside the bound. Input dimensions and basis sizes are bounded before work.

## Independent recipient installation

Build both distributions, install into a fresh environment, and run from an empty directory with Python isolated mode. No checkout imports, private files or previous caches are needed.

```sh
python -m build --no-isolation
python -m build --no-isolation integration/design_app --outdir dist
python3.12 -m venv recipient
recipient/bin/python -m pip install dist/solvetide-0.1.0-py3-none-any.whl
recipient/bin/python -m pip install --no-deps dist/solvetide_design_example-0.1.0-py3-none-any.whl
python scripts/installed_check.py --python recipient/bin/python --output installed-check.json
recipient/bin/solvetide-design --output inverse-design.json
```

The design client synthesizes a target from material1.8/contrast4.2 with a separate direct solve. L-BFGS-B optimizes positive log-material and log-contrast from0.65/0.6, using public fields and implicit gradients. All methods have the same target, initialization, bounds, tolerances and stopping rules. Final objectives are checked with direct solves and included in full run time. Synthetic target recovery does not establish real experimental identifiability.

The final development run retains actual adopted plans as well as requested guarded-action labels. Seven forward-quota proposals triggered the runtime's inference-cost safeguard and skipped refresh while retaining9+3 quotas; three skip proposals recorded zero applied ranks, with6+6 quotas and no refresh. These guarded outcomes are included in complete-pair targets rather than relabeled as if a full correction ran. The model estimates the cost of requesting a guarded policy action, not a guaranteed realized correction rank. This limited local regression is not an optimal control solution.

An initial optimization driver stopped at a refused adjoint and left later methods unexecuted. Its aborted log and earlier client are retained under `experiments/superseded-optimization`. The client now returns a structured failed-run report with no current design/objective/gradient, and the full comparison continues through every method. Methods, starts, targets, bounds and tolerances were unchanged. Production client requests honor the development gate; the comparison explicitly passes `force_policy=True` to evaluate the disabled learned proposal. No confirmation data were used to tune the model or solver.
