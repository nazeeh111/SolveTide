import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from solvetide.assembly import assemble


@pytest.mark.parametrize("family", ["heat", "elasticity"])
@pytest.mark.parametrize("boundary", ["left", "right", "both", "all"])
def test_supported_system_is_symmetric_positive_definite(family, boundary):
    s = assemble(family, 4, {"contrast": 4.0}, boundary)
    np.testing.assert_allclose(s.A.toarray(), s.A.toarray().T, atol=1e-14)
    assert np.linalg.eigvalsh(s.A.toarray())[0] > 0
    assert s.components == (2 if family == "elasticity" else 1)
    assert s.ndofs == len(s.b)


@pytest.mark.parametrize("family", ["heat", "elasticity"])
@pytest.mark.parametrize("boundary", ["left", "right", "both", "all"])
def test_nonzero_prescribed_constant_or_translation(family, boundary):
    s = assemble(family, 6, {"source": 0, "boundary_value": 0.37, "contrast": 2}, boundary)
    full = s.expand(spsolve(s.A, s.b))
    if family == "heat":
        np.testing.assert_allclose(full, 0.37, atol=2e-13)
    else:
        np.testing.assert_allclose(full[::2], 0.0, atol=2e-13)
        np.testing.assert_allclose(full[1::2], 0.37, atol=2e-13)


@pytest.mark.parametrize("family", ["heat", "elasticity"])
def test_discrete_derivatives_include_boundary_elimination(family):
    p = {"material": 2.1, "source": 0.7, "boundary_value": 0.3, "contrast": 3.2}
    s = assemble(family, 6, p)
    h = 1e-5
    for name in p:
        plus, minus = dict(p), dict(p)
        plus[name] += h
        minus[name] -= h
        a = assemble(family, 6, plus)
        b = assemble(family, 6, minus)
        dA, db = s.derivatives[name]
        np.testing.assert_allclose((a.A.toarray()-b.A.toarray())/(2*h), dA.toarray(), rtol=2e-9, atol=3e-9)
        np.testing.assert_allclose((a.b-b.b)/(2*h), db, rtol=2e-9, atol=3e-9)


def test_identities_have_precise_invalidation():
    s = assemble()
    rhs = assemble(parameters={"source": 2})
    value = assemble(parameters={"boundary_value": 2})
    op = assemble(parameters={"material": 2})
    mapping = assemble(boundary="right")
    assert s.identity["operator"] == rhs.identity["operator"] == value.identity["operator"]
    assert s.identity["rhs"] != rhs.identity["rhs"]
    assert s.identity["mapping"] == rhs.identity["mapping"] == op.identity["mapping"]
    assert s.identity["operator"] != op.identity["operator"]
    assert s.identity["mapping"] != mapping.identity["mapping"]
    with pytest.raises(ValueError):
        s.A.data[0] = 9
    with pytest.raises(ValueError):
        s.b[0] = 9
    with pytest.raises(TypeError):
        s.parameters["source"] = 9


@pytest.mark.parametrize("kwargs", [
    {"family":"fluid"}, {"boundary":"none"}, {"n":3}, {"n":129}, {"n":True}, {"n":4.0},
    {"parameters":{"material":0}}, {"parameters":{"contrast":-1}},
    {"parameters":{"source":float("nan")}}, {"parameters":{"boundary_value":float("inf")}},
    {"parameters":{"other":1}}, {"parameters":{"material":True}},
])
def test_bad_inputs_refuse(kwargs):
    with pytest.raises(ValueError):
        assemble(**kwargs)


def test_heat_mesh_refinement_against_analytic_solution():
    # -u''=1; u(0)=0, u'(1)=0, natural boundaries in y.
    errors = []
    for n in (8, 16, 32):
        s = assemble(n=n)
        full = s.expand(spsolve(s.A, s.b))
        x = s.coordinates[0]
        exact = x - 0.5*x*x
        errors.append(np.sqrt(np.mean((full-exact)**2)))
    assert errors[1] < errors[0] / 3
    assert errors[2] < errors[1] / 3


def test_objective_target_shape_and_finiteness():
    s = assemble(n=4)
    x = spsolve(s.A, s.b)
    j, dj = s.objective(x, target=0.1)
    assert j > 0
    np.testing.assert_allclose(dj, (x-.1)/len(x))
    for bad in ([1, 2], float("nan")):
        with pytest.raises(ValueError):
            s.objective(x, bad)


@pytest.mark.parametrize("complex_value", [1+2j, np.array(1+2j), np.array([1+2j])])
def test_objective_rejects_complex_targets_before_float_conversion(complex_value):
    system = assemble(n=4)
    field = spsolve(system.A, system.b)
    with pytest.raises(ValueError, match="real-valued"):
        system.objective(field, target=complex_value)


@pytest.mark.parametrize("representation", ["list", "array"])
def test_field_operations_reject_complex_before_float_conversion(representation):
    system = assemble(n=4)
    field = spsolve(system.A, system.b).astype(complex)
    field[0] += 2j
    value = field.tolist() if representation == "list" else field
    with pytest.raises(ValueError, match="real-valued"):
        system.expand(value)
    with pytest.raises(ValueError, match="real-valued"):
        system.objective(value)


@pytest.mark.parametrize("value", [1+2j, np.complex128(1+2j), np.complex128(1+0j)])
def test_assembly_rejects_complex_parameters_before_float_conversion(value):
    with pytest.raises(ValueError, match="finite real scalar"):
        assemble(parameters={"source": value})
