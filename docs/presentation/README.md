# A local edit, checked against the whole discrete system

This actual source run uses `e639abebfbfef204d21a3af851fed5d6369f2ae3`, tree
`3a8cdafc58a9ff9111e9bf5935bff5b4cadb75fe`. It is a bounded synthetic heat-editor
journey, not a new performance study or physical heat experiment.

## Make the edit

Start the [documented local field editor](../../README.md#run-locally). Keep Heat
diffusion, 20×20 mesh divisions, left-edge support, material 1, source 1, boundary
value 0 and the default warm-start strategy. Set Local coefficient contrast to
**2.00**, compute and export. Then press **Small local edit +5%**, which changes
contrast to **2.10** and recomputes. Export the new result into a separate file.

The actual downloads are [before](editor-before.json) and [after](editor-after.json).
Each has all **441 nodal field values**, **800 triangles**, input/operator identities,
forward CSR equations, acceptance metrics and the four reported objective gradients.
The unit-square model is dimensionless: these are model temperatures and model
coordinates, not kelvins, metres or measured material properties.

The objective is `0.5 * mean(T_free²)` over unconstrained nodal degrees of freedom,
not a spatial integral. It changed from **0.06394799174272758** to
**0.06363670781793167**. The reported contrast derivative changed from
**−0.0032481663021744524** to **−0.002983305735726991**. The largest absolute nodal
change was **0.0024670508786768797 model units**. These values describe this
specific accepted discrete problem; they are not uncertainty estimates or a
claim of physical accuracy.

## Read the figure honestly

The [wide SVG](field-edit.svg), [portrait SVG](field-edit-mobile.svg), PNG counterparts
and [standalone HTML](field-edit.html) use the actual exports, not a proposed UI.
The dashed top-right quadrant marks the changed conductivity region. Both
absolute fields use the same temperature limits. The signed difference has its
own explicitly labeled symmetric scale, centered at zero; it is not drawn on the
absolute-temperature scale.

Every source mesh cell is retained. As in the real editor, each triangle's color
is the arithmetic mean of its three nodal values. Difference is computed from all
441 after-minus-before values, then averaged per triangle. There is no smoothing,
interpolated replacement mesh, fitting, new solve, normalization or excluded sample
in the renderer. [Inputs, limits, transforms and hashes](figure-manifest.json) ·
[editable existing-NumPy/Matplotlib renderer](render.py).

The before and after data come from sequential browser exports in the same local
runtime. Their timings and iteration counts remain raw observations; neither the
figure nor this comparison establishes general acceleration or a latency guarantee.
The learned policy is not selected and remains development-disabled in the
product. Established warm starts are the default.

## Reopen and check the equations

The editor and CLI have **different export schemas**. The editor carries its
forward equations but not the complete adjoint vector/derivative record. The
included [small standard-library reader](check-editor.py) recomputes only its
stored forward residual. It does not independently verify the editor's reported
gradient. [Actual editor residual checks](editor-checks.json).

For a full independent algebra check, the actual CLI command also ran:

```sh
solvetide solve --family heat --n 20 --contrast 2 --output checked-equations.json
python solvetide/checker.py checked-equations.json
```

The [full native artifact](checked-equations.json) includes original CSR equations,
forward and adjoint vectors, all four discrete derivative inputs and its digest.
The [unchanged exported checker](checker.py) uses Python's standard library and
returned [checked](independent-check.json): forward residual norm
`4.8002874651423286e-11`, adjoint residual norm `1.2766318745047081e-11`, and matching
adjoint gradient contractions. This is a separate solve of the same stated initial
problem, not a claim that the editor JSON has the CLI schema.

After extracting the [complete evidence ZIP](field-evidence.zip) into a fresh directory:

```sh
python3 -I check-editor.py editor-before.json editor-after.json
python3 -I checker.py checked-equations.json
```

Both commands were actually run from a fresh extraction using isolated Python.
Every path in `MANIFEST.json` is included and hash-checked. The bundle requires no
SolveTide, NumPy, SciPy or scikit-fem installation for these stored-equation checks.
New solves require the full source/dependencies; copying an old accepted field
is not a new solve. Figure hashes bind the delivered files, not a promise that
all renderer versions or later executions produce identical timings/image bytes.

## Failure and recovery

The real **Limit to 1 iteration** control produced `Rejected · max_work`; no new
field was returned, export was disabled and the server export route returned
HTTP 409. **Recompute with direct solver** then accepted a fresh fallback. Before
any solve, export also returned 409. Desktop 1440px and phone 375px paths were
exercised. The new mobile CSS makes inputs 16px, units 12px and actions 14px while
preserving the current interface identity and wide workspace.

There is a separate **known current-main display limitation**: changing the physical
model can relabel the faded earlier field before another solve is accepted. The
export is correctly disabled and the text marks it stale, but the displayed quantity
label can belong to the newly selected family. A separately held engineering branch
owns that identity repair. It is **not merged or claimed by this candidate**; this
recorded comparison stays within the heat family. Recompute before interpreting
or exporting changed inputs.

## Boundaries and provenance

Residuals and backward error check stored floating-point discrete equations.
They do not bound solution error in an ill-conditioned system, discretization
error or physical-model error. No new material calibration, safety-critical
certification, universal PDE support or trained-policy result is established.
The numerical core, checker, study records, defaults and held branch remain unchanged.

See the complete [equations](../equations.md), [runtime contracts](../runtime.md),
[historical studies and measurements](../measurements.md), [source/API](../../README.md),
[MIT license](../../LICENSE) and [dependency notices](../licenses.md). NumPy,
SciPy and scikit-fem provide the declared numerical foundations; no external
solver source or pretrained model was copied into this presentation.
