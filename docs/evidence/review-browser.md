# Independent browser and exported-result checks

Canonical files were read only. All captures and review evidence were saved in this distinct review folder. No publishing was performed by this chat.

## Frozen-source sequence

The owner restarted the editor on IPv4 loopback port8769 in PID89603. `editor-final-source.json` preserves the reported startup hashes. The final runtime hash is `98108d59c3532d04b15ee37c2211f73765c04d4fa90c610d14f11a0b4f549e28`; assembly is `31d7c447c7b444b8a23fb06fd1a253d7284a710f8ee77eefc3ffc0994ff6f8dd`. I reloaded the page after restart, then executed these actual UI actions in order. Screenshots are observations of computed results, not mockups. Pauses and automation reconnections are not evidence of solver latency.

| Capture prefix | Actual action and observed result |
|---|---|
| final-01-initial | Compute heat, n20, material/source/contrast1, zero boundary, warm strategy. Accepted; field range0..0.5006; cache empty; export enabled. |
| final-02-local-edit | Click Small local edit +5%. Contrast1.05, changed field range0..0.4981; accepted; operator_refreshed; export enabled. |
| final-03-remesh | Click Change mesh. n20 to32; accepted range0..0.4982; mapping_invalidated; zero reused rank. |
| final-04-failure | Click Limit to1 iteration. Cold maxiter1; Rejected max_work, no accepted gradient, export disabled. Faded old image explicitly labeled unaccepted for current inputs in accessibility text. |
| final-05-direct-fallback | Click Recompute with direct solver. Accepted fresh direct solve, factor_built, relative residual2.02e-13, backward error1.96e-16; export enabled. |

All five `.jpg` screenshots and `-ax.txt` full accessibility trees exist. The first two also have DOM-derived `.json` records with input values and full inspection details. Later DOM evaluation became intermittent, so the remaining evidence uses native screenshots and accessibility observations. `final-capture-manifest.json` records byte sizes and SHA256 values. The screenshots use the actual viewport and scrolling position; no synthetic scene or accelerated timing was represented as real-time interaction. Browser zoom key input had no effect. A full-page screenshot request failed; regular screenshots succeeded. The browser connection reset once during the attempted failure action; after reconnecting, state inspection showed that action had not occurred, so it was executed and verified successfully.

## Earlier same-editor checks and limits

Before the final restart, I also verified:

- Learned strategy explicitly reports the disabled development gate, requested strategy learned, effective strategy warm and policy_disabled true. `current-desktop-learned-gate.jpg` and its full accessibility tree preserve that observation. Final trained weights are separately owned by the build chat and must not be inferred from this earlier observation.
- Editing boundary value to0.1 before recomputation marks the image stale and disables export; `current-stale-ax.txt` preserves the result.
- Changing family to elasticity at n32 produced accepted fields/gradients and mapping invalidation.
- Mobile390x844 rendered a readable single-column arrangement; DOM/body widths both390, no horizontal overflow. `current-mobile-top.jpg`, `current-mobile-field.jpg` and desktop captures preserve the observations. Viewport override was reset afterward.

These earlier captures were made before the final runtime restart. Do not relabel them as screenshots of the final frozen process. Their numerical exports were independently verified below.

## Real browser download verification

Downloaded JSON through the editor Export button, read the actual saved files, copied only those exports into this review folder, independently reassembled the original equations, solved with SciPy SuperLU, and formed independent direct adjoints. `browser-export-verification.json` records hashes and results.

| Export | Relative field error vs independent direct | Maximum relative gradient error | Boundaries/operator identity |
|---|---:|---:|---|
| snapshot002-browser-export.json, heat revision5 | 2.30e-16 | 1.80e-14 | exact/matching |
| current-browser-elasticity-export.json, elasticity revision6 | 1.37e-15 | 4.36e-13 | exact/matching |

No remaining editor correctness blocker was observed. Release CI, final trained-policy gate, held-out comparison, clean installation and public artifacts remain separate gates handled by the canonical owner.
