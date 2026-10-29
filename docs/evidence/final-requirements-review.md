# Focused runtime and paired-policy requirements review

**Verdict: no outstanding blocker found in the inspected runtime, paired-policy training code, or declared memory-invalidation contract.** The previous complex-target and non-JSON diagnostic defects are corrected and independently rechecked. This is a scoped code/numerical review, not confirmation that all release gates have been executed.

## Immutable artifacts

- Snapshot003 runtime: `9aabcf3375ddbc2bcde1e709d3a67cd0e099a5808a85117b592b4c4b2fe52bdc`.
- Snapshot004 runtime, including stored-factor preflight: `98108d59c3532d04b15ee37c2211f73765c04d4fa90c610d14f11a0b4f549e28`.
- Assembly for both snapshots: `31d7c447c7b444b8a23fb06fd1a253d7284a710f8ee77eefc3ffc0994ff6f8dd`.
- Snapshot003 policy: `e4ab1d254843c8d52e79292da68dc2558a53c05a4ad63e471810b03668acd59a`.
- Snapshot003 study runner: `3b9cec7ca67f5f8d7e1fc9e69a3666ff8f8383c708420ef71393075ed6e80e68`.

Sources and hashes are retained in canonical-003 and canonical-004 under this review folder. The canonical project was read only.

## Executed focused rechecks

`recheck-003.py`, evidence `review-runtime-003-evidence.json`:

- A complex objective target returns `invalid_input`, no gradient and no casting warning.
- Finite external inputs above the declared 1e150 magnitude limit return `invalid_input`, no field and strict-JSON-safe summary.
- A synthetic changing SPD stream confirms one policy call before each primal, the identical plan and raw feature vector on its following adjoint, combined rank at most four and each role's quota at most two. Current-operator results agree with independent direct solutions. Final basis orthogonality error is 3.10e-16.
- Mapping changes return `mapping_invalidated` with zero reused rank.
- A fitted synthetic failed action with recorded time .001 seconds is penalized to a predicted 30 seconds. Failure examples are retained rather than becoming attractive cheap actions.
- All six public action plans satisfy their shared budget and role quotas, including the two zero-quota one-role controls and the skip/refresh distinction.

`recheck-factor-004.py`, evidence `review-factor-004-evidence.json`:

- A one-byte factor budget returns `resource_limit` before any call to SuperLU.
- An ordinary direct solve still converges with one factorization.
- The conservative bound correctly covers dense triangular L/U data and indices, their pointer arrays and both permutations. It is a stored-factor bound. Opaque SuperLU workspace and process peak RSS remain outside it, as the detailed runtime documentation states.

## Paired-policy methodology

The six trained actions allocate primal/adjoint quotas and ranks under the same retained basis cap; `refresh=False` skips both corrections. A zero quota both prunes that role's old directions and prevents retaining new ones. Used correction directions may originate from either role, by current-operator relevance, so a quota describes retained origin rather than exclusive solve ownership. This matches the documented joint-space method.

Training measures complete chronological forward/adjoint pairs with assembly, features, inference, correction work, classical solves, residual checks and retention included in pair wall time. It runs complete streams under constant development actions, fits per-action ridge models and validates the adaptive policy on separate whole development trajectories. This is a local cost heuristic; the data do not prove globally optimal sequence allocation or eliminate action-history distribution shift.

At inspection, 720 training records were retained and all had converged. Whole seed104 trajectories are excluded from fitting for each family. The deployment gate is applied independently to heat and elasticity, requires at least 5% improvement over both warm and joint, and requires zero learned validation failures. The inspected trained document is disabled because that gate failed. An owner-triggered final training rerun was in progress after source freeze; its final outputs must be verified separately.

The confirmation function forces evaluation of learned proposals even if deployment is disabled, without rewriting weights or the gate, and asserts the policy file hash is unchanged. It draws separate seeds, finer held-out meshes and right/both primary boundary regimes. Deliberate final mapping changes can return to a known boundary regime; these are explicit invalidation tests. Confirmation output should remain an evaluation record, not be used to revise the gate or claim default deployment was enabled. No confirmation result was used by this review to recommend tuning.

## Numerical and memory contract

Only classical accepted solutions can supply new memory directions or measured utility updates. Mapping changes reset caches; original A/RHS identities are recomputed from bytes. Every used A@Q is freshly formed from the current operator. Learned corrections alter the initial guess only, and ordinary CG's preconditioner is fixed during each call. The original-equation residual and normwise backward error decide acceptance; stale forward results cannot be differentiated. Earlier broad checks of both PDE families and finite-difference gradients are recorded in review-runtime.md.

The fixed chronological recycling control qualifies as the requested fixed-rank recycling baseline; it is not labeled POD. GCROT is an executed established Krylov recycling method with old C products discarded and U vectors copied. Shared strategies retain at most12 basis vectors; the separate control uses6+6. Reported storage includes retained warm states, copied matrices, factors or hierarchies and, for the separate control, both runtimes. This is **matched basis-rank capacity**, not identical retained bytes or matched peak-RSS use. GCROT's Arnoldi workspace, Q/AQ temporaries and native allocations are not measured process memory. Existing protocol/runner language makes that distinction; release comparisons must preserve it.

## Limits of the verdict

Three repeated timings and one development-validation trajectory per family provide limited statistical evidence. Performance conclusions should stay local and exploratory, especially where repeated wall times vary. No general acceleration or real-time claim follows from passing correctness. Full held-out comparisons, inverse-design outcomes, independent installation and public release/CI verification remain separate executed-evidence gates. This review does not certify those gates merely from their source code.

Minor documentation synchronization: at inspection README.md still said factor storage was checked after allocation, while detailed runtime.md correctly described the newly added preflight. The README should mention both preflight and final measured check. This outdated description understates the new safeguard and is not an unresolved numerical blocker.
