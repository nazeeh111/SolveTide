# Supported equations and derivatives

Both assemblers use scikit-fem 12, continuous piecewise-linear (P1) triangles on the unit square `[0,1]²`. `n` is the number of edge intervals; there are `(n+1)²` vertices and `2n²` triangles. Supported `n` is 4 through 128. The parameter `contrast` multiplies the top-right quadrant's material. The coefficient is evaluated at scikit-fem quadrature points. Even `n` aligns the quadrant interfaces with the mesh. Odd `n` is supported as the corresponding quadrature-defined discrete model, not an exact geometric interface representation.

## Heat

The steady scalar model is

`-div(k grad(T)) = source`,

with `k(x,y) = material * [1 + (contrast-1) I(x>0.5 and y>0.5)]`. The weak form is `∫ k grad(T)·grad(v) = ∫ source v`. Conductivity and contrast must be positive. Natural zero normal heat flux applies on unprescribed edges. There are no time stepping, radiation, convection, or nonlinear material effects.

## Elasticity

The two-dimensional plane-strain model is

`-div(sigma(u)) = (0, source)`,

`sigma = 2 mu epsilon(u) + lambda tr(epsilon(u)) I`,

`epsilon(u) = [grad(u) + grad(u)^T]/2`,

`mu = E/[2(1+nu)]`, `lambda = E nu/[(1+nu)(1-2nu)]`,

with fixed `nu=0.3` and `E(x,y)` equal to the conductivity coefficient above. The weak form is `∫ [2 mu epsilon(u):epsilon(v) + lambda div(u)div(v)] = ∫ source v_y`. This is a small-strain, linear, isotropic displacement model, not a geometrically nonlinear structural simulation. Natural zero traction applies on unprescribed edges. Displacement arrays interleave horizontal and vertical components at each node: `ux0, uy0, ux1, uy1, ...`.

## Constraints and rigid modes

`boundary="left"`, `"right"`, `"both"`, or `"all"` prescribes a full edge, the two vertical edges, or all edges. Heat prescribes `T=boundary_value`. Elasticity prescribes `u_x=0` and `u_y=boundary_value`. All elasticity constraints fix both components over multiple separated nodes, removing two translations and one rotation. Unsupported unconstrained or custom partial-node mappings are rejected rather than silently regularized. Neumann-only diffusion and unconstrained bodies are outside this API.

Let `F` and `D` denote the free and fixed degrees of freedom, with prescribed full vector `g`. The actual reduced system is

`A = K_FF`, `b = f_F - K_FD g_D`.

The nonempty Dirichlet support and positive material establish the mathematical SPD contract for these assembled models. Floating-point conditioning can still make an extreme case fail a solver check. The runtime must report that failure. Matrix symmetry is not by itself evidence of positive definiteness for arbitrary externally supplied matrices.

## Exact discrete sensitivity inputs

`System.derivatives[name]` stores `(dA/dp, db/dp)` for each supported parameter. If `K0` is the uniform coefficient stiffness and `Kpatch` is the quadrant stiffness, then

`K = material * [K0 + (contrast-1) Kpatch]`.

The derivatives of `K` with respect to material and contrast are assembled analytically from those same forms. Source changes only the uniform load. For every parameter,

`db/dp = df_F/dp - (dK_FD/dp) g_D - K_FD (dg_D/dp)`.

In particular, nonzero prescribed values change the material and contrast right-hand-side derivatives. Boundary-value derivatives use the original stiffness and prescribed-value direction. The adjoint must use `dJ/dp = explicit_dJ/dp + lambdaᵀ(db/dp - dA/dp x)` with `Aᵀ lambda = dJ/dx`, including all these terms. No derivative is taken through a cached field.

`System.objective(x,target)` is `0.5 * mean((x-target)²)` over **free DOFs**, and its derivative is `(x-target)/Nfree`. It is a nodal mean, not a spatial integral or full-DOF objective. The target is held fixed; it may be zero, a finite real scalar, or a real-valued array of exactly `Nfree` values. Complex fields and targets are rejected before any float conversion, including complex scalars and NumPy arrays. Constraints' objective contributions are excluded. An application with a different objective can supply its own adjoint right-hand side and explicit parameter terms.

## Identity and storage

Every system records SHA-256 identities for the CSR operator bytes, reduced RHS, mesh/constraint mapping, float64 precision, and prescribed boundary data. Operator and RHS hashes distinguish factorization reuse from field reuse. A boundary-value change retains the mapping but changes the reduced RHS. A constrained-edge, family, or mesh change changes the mapping. All stored NumPy arrays and sparse storage arrays are marked read-only, and metadata/derivative maps are read-only. Python's frozen dataclass and read-only buffers protect normal API use, not malicious low-level mutation. Runtime acceptance remains based on the current equations.

`System.expand(x)` restores prescribed values for display/export. A computed displacement or temperature can change globally after a local contrast edit. No untouched region receives a validity mark from locality alone.

## Evidence and limits

`tests/test_assembly.py` checks all supported boundary/family combinations for symmetry and positive eigenvalues on small systems, nonzero constant temperature/rigid prescribed translation, all four equation derivatives against central differences, identity behavior, mutation refusal, invalid inputs and refinement against the independent analytic heat solution `T=x-x²/2` for unit source and left support. The analytic comparison decreases at the expected second-order rate for nodal RMS field error.

The solve residual concerns the discrete equations. Mesh refinement concerns discretization. Neither establishes material truth, realistic loading, safety factors, manufacturing feasibility, or safety-critical certification. The unit square and parameter values use internally consistent dimensionless units. An application is responsible for dimensional scaling and its physical model.
