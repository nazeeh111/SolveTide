"""Exercise the independent numerical oracle, not stored status labels."""
import numpy as np
import pytest
from scipy.sparse import diags

from studies.validate import (check_case, conditioning_case, direct_reference,
                              failure_checks, heat_refinement)


def test_direct_reference_handles_known_ill_conditioned_system():
    diagonal = np.geomspace(1e-8, 1., 32)
    matrix = diags(diagonal).tocsr()
    expected = np.linspace(-1., 2., 32)
    actual = direct_reference(matrix, matrix @ expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("family", ["heat", "elasticity"])
def test_independent_gradients_of_nonzero_boundary_problem(family):
    report = check_case(family, 4, steps=(1e-3, 1e-5))
    assert report["passed"], report
    assert set(report["gradients"]) == {"material", "contrast", "source", "boundary_value"}


def test_small_residual_is_not_small_field_error():
    report = conditioning_case("heat", 1e8, n=4)
    candidate = report["manufactured_bad_candidate"]
    assert candidate["relative_solution_error"] > .09
    assert candidate["relative_residual_l2"] < 1e-9
    assert candidate["backward_error_inf"] < 1e-9


def test_failed_or_cancelled_solve_never_supplies_fresh_field_or_gradient():
    assert failure_checks()["passed"]


def test_independent_analytic_heat_convergence():
    assert heat_refinement(meshes=(8, 16, 32))["passed"]


def test_independent_residual_preserves_small_term_between_large_cancelling_terms():
    from scipy.sparse import csr_matrix
    from studies.validate import _precise_residual
    # Float64 products are exactly represented, but ordinary summation can
    # erase the unit term. The independent Decimal residual retains it.
    matrix = csr_matrix(np.array([[1e16, 1., -1e16]]))
    assert _precise_residual(matrix, np.array([1.]), np.ones(3))[0] == 0.
