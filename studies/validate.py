"""Independent release numerical checks. Run from the repository root.

The reference uses sparse LU and independent residual iterative refinement,
not Runtime.solve or its retained memory. Central differences assemble and
solve both perturbed equations independently. Nothing in this report certifies
the physical model or bounds the continuous PDE solution error.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from decimal import Decimal, localcontext
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from solvetide import Runtime, SolverConfig, assemble

SCHEMA = "solvetide-numerical-validation-v1"
PARAMETERS = {"material": 1.7, "contrast": 3.5, "source": .8, "boundary_value": .13}
STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _precise_residual(A: csr_matrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    # Small correctness cases use 60-digit Decimal arithmetic on the exact
    # stored binary64 values. This is independent of BLAS sparse matvec and
    # works on arm64 where np.longdouble has no additional mantissa bits.
    if A.shape[0] <= 512:
        residual = []
        with localcontext() as context:
            context.prec = 60
            dx = [Decimal.from_float(float(value)) for value in x]
            for i in range(A.shape[0]):
                total = Decimal(0)
                for index in range(A.indptr[i], A.indptr[i+1]):
                    total += Decimal.from_float(float(A.data[index])) * dx[A.indices[index]]
                residual.append(float(Decimal.from_float(float(b[i])) - total))
        return np.asarray(residual)
    return (np.asarray(b, dtype=np.longdouble)
            - A.astype(np.longdouble) @ np.asarray(x, dtype=np.longdouble))


def direct_reference(A: csr_matrix, b: np.ndarray) -> np.ndarray:
    """Independent float64 sparse LU with accurately recomputed refinement.

    Small systems use 60-digit Decimal residuals of the stored values. Larger
    refinement cases use longdouble, which equals double on some platforms.
    This is not an exact or interval-certified answer.
    """
    factor = splu(A.tocsc())
    value = factor.solve(np.asarray(b, dtype=np.float64))
    for _ in range(3):
        residual = _precise_residual(A, b, value)
        correction = factor.solve(np.asarray(residual, dtype=np.float64))
        candidate = value + correction
        if np.linalg.norm(_precise_residual(A, b, candidate), np.inf) > np.linalg.norm(residual, np.inf):
            break
        value = candidate
    return value


def _relative(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(b), np.finfo(float).tiny))


def _residual(A, b, x) -> dict:
    # Recomputed independently from public runtime diagnostics.
    r = _precise_residual(A, b, x)
    row_norm = float(np.max(np.asarray(abs(A).sum(axis=1))))
    denominator = row_norm * float(np.linalg.norm(x, np.inf)) + float(np.linalg.norm(b, np.inf))
    return {"residual_l2": float(np.linalg.norm(r)),
            "relative_residual_l2": float(np.linalg.norm(r) / max(np.linalg.norm(b), np.finfo(float).tiny)),
            "backward_error_inf": float(np.linalg.norm(r, np.inf)) / denominator if denominator else 0.0}


def check_case(family: str, n: int, steps=STEPS) -> dict:
    system = assemble(family, n, PARAMETERS, boundary="left")
    target = .07
    config = SolverConfig(strategy="joint", rtol=1e-11, atol=1e-13,
                          maxiter=6000, max_seconds=30, memory_budget=12)
    runtime = Runtime(config)
    forward = runtime.solve(system)
    if not forward.accepted:
        return {"family": family, "n": n, "passed": False, "failure": forward.summary()}
    sensitivity = runtime.gradient(system, forward, target=target)
    if not sensitivity.accepted:
        return {"family": family, "n": n, "passed": False,
                "failure": {"status": sensitivity.status, "message": sensitivity.message}}
    reference = direct_reference(system.A, system.b)
    objective, objective_rhs = system.objective(reference, target)
    adjoint = direct_reference(system.A.T.tocsr(), objective_rhs)
    direct_gradients = {name: float(adjoint @ (db - dA @ reference))
                        for name, (dA, db) in system.derivatives.items()}
    differences = {}
    for name in PARAMETERS:
        rows = []
        for step in steps:
            plus, minus = dict(PARAMETERS), dict(PARAMETERS)
            plus[name] += step
            minus[name] -= step
            sp = assemble(family, n, plus)
            sm = assemble(family, n, minus)
            jp, _ = sp.objective(direct_reference(sp.A, sp.b), target)
            jm, _ = sm.objective(direct_reference(sm.A, sm.b), target)
            fd = (jp - jm) / (2 * step)
            analytic = sensitivity.gradient[name]
            rows.append({"step": step, "central_difference": fd,
                         "absolute_error": abs(fd - analytic),
                         "relative_error": abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12)})
        differences[name] = {"runtime_gradient": sensitivity.gradient[name],
                             "independent_direct_adjoint_gradient": direct_gradients[name],
                             "direct_absolute_error": abs(sensitivity.gradient[name] - direct_gradients[name]),
                             "steps": rows, "best_relative_fd_error": min(row["relative_error"] for row in rows)}
    relative_forward = _relative(forward.x, reference)
    relative_adjoint = _relative(sensitivity.adjoint.x, adjoint)
    gradient_error = max(item["direct_absolute_error"] / max(abs(item["independent_direct_adjoint_gradient"]), 1e-12)
                         for item in differences.values())
    best_fd = max(item["best_relative_fd_error"] for item in differences.values())
    return {
        "family": family, "n": n, "ndofs": system.ndofs, "nonzeros": system.A.nnz,
        "parameters": dict(system.parameters), "target": target, "boundary": system.boundary,
        "identity": dict(system.identity), "status": forward.status, "adjoint_status": sensitivity.adjoint.status,
        "objective": sensitivity.objective, "reference_objective": objective,
        "relative_forward_error_vs_refined_lu": relative_forward,
        "relative_adjoint_error_vs_refined_lu": relative_adjoint,
        "independent_forward_residual": _residual(system.A, system.b, forward.x),
        "independent_adjoint_residual": _residual(system.A.T.tocsr(), objective_rhs, sensitivity.adjoint.x),
        "gradients": differences,
        "gates": {"relative_forward_error_max": 1e-7, "relative_adjoint_error_max": 1e-7,
                  "relative_direct_gradient_error_max": 1e-7, "best_relative_fd_error_max": 1e-5},
        "passed": bool(relative_forward < 1e-7 and relative_adjoint < 1e-7 and gradient_error < 1e-7 and best_fd < 1e-5),
    }


def heat_refinement(meshes=(8, 16, 32, 64)) -> dict:
    rows = []
    parameters = {"material": 1.7, "contrast": 1.0, "source": .8, "boundary_value": .13}
    previous_error = None
    for n in meshes:
        s = assemble("heat", n, parameters)
        field = s.expand(direct_reference(s.A, s.b))
        x = s.coordinates[0]
        exact = .13 + .8 / 1.7 * (x - .5 * x * x)
        rms = float(np.sqrt(np.mean((field - exact) ** 2)))
        rows.append({"n": n, "ndofs": s.ndofs, "nodal_rms_error": rms,
                     "max_absolute_error": float(np.max(abs(field-exact))),
                     "observed_order": None if previous_error is None else float(np.log2(previous_error / rms))})
        previous_error = rms
    orders = [r["observed_order"] for r in rows if r["observed_order"] is not None]
    return {"family": "heat", "analytic_solution": "T=0.13+(0.8/1.7)*(x-x^2/2)",
            "equations": "-1.7*Laplacian(T)=0.8; T(0,y)=0.13; zero normal flux on other edges",
            "rows": rows, "minimum_expected_order": 1.7,
            "passed": bool(all(order > 1.7 for order in orders))}


def elasticity_refinement(meshes=(8, 16, 32), reference_n=64) -> dict:
    p = {"material": 1.7, "contrast": 1.0, "source": .8, "boundary_value": .13}
    fine = assemble("elasticity", reference_n, p)
    fine_field = fine.expand(direct_reference(fine.A, fine.b)).reshape(-1, 2)
    fine_indices = {tuple(np.rint(point * reference_n).astype(int)): i
                    for i, point in enumerate(fine.coordinates.T)}
    rows = []
    for n in meshes:
        if reference_n % n:
            raise ValueError("elasticity refinement requires nested meshes")
        system = assemble("elasticity", n, p)
        solution = direct_reference(system.A, system.b)
        field = system.expand(solution).reshape(-1, 2)
        indices = [fine_indices[tuple(np.rint(point * reference_n).astype(int))] for point in system.coordinates.T]
        sampled = fine_field[indices]
        rows.append({"n": n, "ndofs": system.ndofs,
                     "vector_nodal_rms_difference_to_reference": float(np.sqrt(np.mean((field-sampled)**2))),
                     "relative_vector_difference_to_reference": _relative(field, sampled),
                     "free_dof_mean_square_objective": system.objective(solution)[0]})
    # An exact rigid translation consistency check, independent of gradient code.
    translation = assemble("elasticity", 8, {"source": 0., "boundary_value": .37, "contrast": 7.})
    translated = translation.expand(direct_reference(translation.A, translation.b)).reshape(-1, 2)
    translation_error = float(np.max(abs(translated - np.array([0., .37]))))
    errors = [row["vector_nodal_rms_difference_to_reference"] for row in rows]
    return {"family": "elasticity", "reference_n": reference_n, "reference_ndofs": fine.ndofs,
            "parameters": p, "rows": rows, "rigid_translation_max_error": translation_error,
            "limit": "Nested-node comparison against n=64 is a consistency/refinement observation, not an error bound. Clamped/free corners reduce regularity. The nodal-mean objective also changes quadrature with mesh resolution.",
            "passed": bool(translation_error < 1e-10 and all(b < a for a, b in zip(errors, errors[1:])))}


def conditioning_case(family: str, contrast: float, n=6) -> dict:
    s = assemble(family, n, {"contrast": contrast, "source": .8, "boundary_value": .13})
    eigenvalues, eigenvectors = np.linalg.eigh(s.A.toarray())
    lower, upper = float(eigenvalues[0]), float(eigenvalues[-1])
    # Labeled manufactured RHS: an algebraic demonstration on the actual FEM
    # operator. It is not the assembler's uniform body/heat source and not a
    # field produced by Runtime. Its exact chosen field is the largest mode.
    exact = eigenvectors[:, -1]
    manufactured_rhs = s.A @ exact
    bad = exact + .1 * eigenvectors[:, 0]
    diagnostics = _residual(s.A, manufactured_rhs, bad)
    physical_reference = direct_reference(s.A, s.b)
    result = Runtime(SolverConfig(strategy="cold", rtol=1e-9, atol=1e-11,
                                  maxiter=6000, max_seconds=30)).solve(s)
    return {"family": family, "n": n, "contrast": contrast,
            "spectral_estimate": {"method": "dense symmetric eigensolver on the stored small matrix",
                                  "lambda_min_estimate": lower, "lambda_max_estimate": upper,
                                  "condition_2_estimate": upper/lower,
                                  "limit": "Floating-point estimates, not rigorous enclosures; accuracy degrades at high condition numbers."},
            "manufactured_bad_candidate": {"relative_solution_error": _relative(bad, exact),
                                            **diagnostics,
                                            "satisfies_relative_residual_1e_9": diagnostics["relative_residual_l2"] <= 1e-9,
                                            "description": "Chosen field is the largest eigenmode; add 10% smallest-mode error. Manufactured RHS, not the physical uniform source. This is not a runtime-returned field."},
            "physical_runtime": {"status": result.status, "accepted": result.accepted,
                                 "relative_forward_error_vs_refined_lu": _relative(result.x, physical_reference) if result.accepted else None,
                                 "returned_field_on_failure": result.x is not None if not result.accepted else None,
                                 "diagnostics": _residual(s.A, s.b, result.x) if result.accepted else None}}


def failure_checks() -> dict:
    s = assemble("heat", 8, PARAMETERS)
    runtime = Runtime(SolverConfig(strategy="joint"))
    accepted = runtime.solve(s)
    cancelled = runtime.solve(assemble("heat", 8, dict(PARAMETERS, source=.9)), cancel=lambda: True)
    cancellation_gradient = runtime.gradient(s, cancelled)
    limited_runtime = Runtime(SolverConfig(strategy="cold", maxiter=1, rtol=1e-14, atol=1e-15))
    limited = limited_runtime.solve(s)
    limited_gradient = limited_runtime.gradient(s, limited)
    changed = assemble("heat", 8, dict(PARAMETERS, contrast=4.))
    stale = runtime.gradient(changed, accepted)
    invalidated = runtime.solve(assemble("heat", 10, PARAMETERS))
    reference = direct_reference(assemble("heat", 10, PARAMETERS).A, assemble("heat", 10, PARAMETERS).b)
    rows = {
        "initial_forward_accepted": accepted.accepted,
        "cancel_status": cancelled.status, "cancel_returns_no_field": cancelled.x is None,
        "cancel_gradient_status": cancellation_gradient.status,
        "cancel_returns_no_gradient": cancellation_gradient.gradient is None,
        "max_work_status": limited.status, "max_work_returns_no_field": limited.x is None,
        "max_work_gradient_status": limited_gradient.status,
        "max_work_returns_no_gradient": limited_gradient.gradient is None,
        "changed_operator_gradient_status": stale.status, "changed_operator_returns_no_gradient": stale.gradient is None,
        "changed_mesh_cache_event": invalidated.cache_event,
        "changed_mesh_accepted": invalidated.accepted,
        "changed_mesh_relative_error_vs_lu": _relative(invalidated.x, reference) if invalidated.accepted else None,
    }
    rows["passed"] = bool(accepted.accepted and cancelled.status == "cancelled" and cancelled.x is None
                          and cancellation_gradient.gradient is None and not limited.accepted and limited.x is None
                          and limited_gradient.gradient is None and stale.status == "stale_forward" and stale.gradient is None
                          and invalidated.accepted and invalidated.cache_event.startswith("mapping_invalidated")
                          and rows["changed_mesh_relative_error_vs_lu"] < 1e-6)
    return rows


def run_validation() -> dict:
    start = perf_counter()
    cases = [check_case(family, n) for family in ("heat", "elasticity") for n in (6, 10)]
    heat = heat_refinement()
    elasticity = elasticity_refinement()
    conditions = [conditioning_case(family, contrast) for family in ("heat", "elasticity") for contrast in (1., 1e4, 1e8)]
    failures = failure_checks()
    demonstrated = any(c["manufactured_bad_candidate"]["satisfies_relative_residual_1e_9"]
                       and c["manufactured_bad_candidate"]["relative_solution_error"] > .09 for c in conditions)
    root = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
              for path in (root/"solvetide/assembly.py", root/"solvetide/runtime.py", root/"studies/validate.py")}
    return {"schema": SCHEMA, "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-fem")},
            "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant),
            "small_case_residual_arithmetic": "Decimal 60 significant digits on stored binary64 operands (nDOF<=512)"},
            "source_sha256": hashes, "forward_and_gradient_cases": cases,
            "heat_analytic_refinement": heat, "elasticity_refinement": elasticity,
            "conditioning_stress": conditions, "failure_and_restart_checks": failures,
            "small_residual_large_error_demonstrated": demonstrated,
            "reference_method": "Independent SciPy SuperLU with up to three residual refinement steps. Small nDOF<=512 cases use 60-digit Decimal residuals on stored binary64 values; larger mesh-refinement cases use longdouble. Central differences independently reassemble both perturbed inputs.",
            "limits": ["Numerical reference agreement is not a rigorous solution enclosure.",
                       "The residual and backward error concern stored discrete equations, not discretization or model error.",
                       "Gradient finite differences hold target and mesh/constraints fixed.",
                       "This is a correctness study, not evidence of acceleration."],
            "elapsed_seconds": perf_counter()-start,
            "passed": bool(all(c["passed"] for c in cases) and heat["passed"] and elasticity["passed"] and failures["passed"] and demonstrated)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/numerical-validation.json"))
    args = parser.parse_args()
    report = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".new")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
