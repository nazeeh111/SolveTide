from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from solvetide.runtime import Runtime, SolverConfig


def system(n=40, material=1., source=1., mapping="line"):
    base = sparse.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format="csr")
    b = source * np.sin(np.linspace(.1, 2, n))
    return SimpleNamespace(A=material*base, b=b,
                           identity={"mapping": mapping},
                           derivatives={"material": (base, np.zeros(n)),
                                        "source": (sparse.csr_matrix((n,n)), b/source)},
                           objective=lambda x, target=None: (.5*np.mean((x-(0 if target is None else target))**2),
                                                             (x-(0 if target is None else target))/n))


@pytest.mark.parametrize("strategy", ["cold", "warm", "fixed", "joint", "drift", "direct", "gcrot", "amg", "amg_reuse"])
def test_strategies_true_residual_and_derivative(strategy):
    runtime = Runtime(SolverConfig(strategy=strategy, rtol=1e-10, memory_budget=4))
    for m in [1., 1.01, 1.04]:
        s = system(material=m)
        forward = runtime.solve(s)
        assert forward.accepted, forward.summary()
        np.testing.assert_allclose(forward.x, spsolve(s.A, s.b), rtol=2e-9, atol=1e-10)
        gradient = runtime.gradient(s, forward)
        assert gradient.accepted, gradient
        assert forward.residual_norm <= max(runtime.config.atol, runtime.config.rtol*np.linalg.norm(s.b))
        assert gradient.gradient["material"] == pytest.approx(-2*gradient.objective/m, rel=1e-8)
        assert gradient.gradient["source"] == pytest.approx(2*gradient.objective, rel=1e-8)
        assert forward.backward_error < 1e-9
    assert runtime.state()["rank"] <= 4


def test_changed_operator_and_mapping_invalidation():
    r = Runtime(SolverConfig(memory_budget=3))
    a = r.solve(system())
    g = r.gradient(system(), a)
    assert g.accepted
    assert r.state()["rank"] > 0
    s = system(material=1.03)
    updated = r.solve(s)
    assert updated.cache_event == "operator_refreshed"
    np.testing.assert_allclose(updated.x, spsolve(s.A,s.b), rtol=1e-8)
    assert r.gradient(s,a).status == "stale_forward"
    different = r.solve(system(mapping="new constraints"))
    assert different.cache_event == "mapping_invalidated"
    assert different.reused_rank == 0


def test_identities_recomputed_when_caller_mutates():
    s = system()
    r = Runtime()
    first = r.solve(s)
    s.A.data *= 2
    assert r.gradient(s,first).status == "stale_forward"
    new = r.solve(s)
    assert new.cache_event == "operator_refreshed"
    np.testing.assert_allclose(new.x, first.x/2, rtol=1e-8)


def test_failure_never_exposes_candidate_and_cannot_differentiate():
    r = Runtime(SolverConfig(maxiter=1))
    failed = r.solve(system())
    assert not failed.accepted
    assert failed.x is None
    assert r.state()["rank"] == 0
    assert r.gradient(system(),failed).gradient is None


def test_cancellation_and_deadline():
    r = Runtime()
    cancelled = r.solve(system(), cancel=lambda: True)
    assert cancelled.status == "cancelled"
    assert cancelled.x is None
    timed = Runtime(SolverConfig(max_seconds=1e-12)).solve(system())
    assert timed.status == "time_limit"
    assert timed.x is None


def test_cancellation_during_iteration():
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls > 7
    r = Runtime()
    result = r.solve(system(n=100),cancel=cancel)
    assert result.status == "cancelled"
    assert result.x is None
    assert r.state()["rank"] == 0


def test_nonfinite_asymmetric_rejected():
    r = Runtime()
    s = system()
    s.b[0] = np.nan
    assert r.solve(s).status == "invalid_input"
    s = system()
    s.A[0,1] = 3
    assert r.solve(s).status == "invalid_input"


def test_direct_exact_operator_reuse_only():
    r = Runtime(SolverConfig(strategy="direct"))
    first = r.solve(system())
    assert first.accepted and "factor_built" in first.cache_event
    second = r.solve(system(source=2))
    assert second.accepted and "factor_reused" in second.cache_event
    changed = r.solve(system(material=2))
    assert changed.accepted and "factor_built" in changed.cache_event


def test_learned_bad_policy_falls_back_and_is_disabled():
    class Bad:
        def choose(self, features, max_rank):
            return np.nan
    r = Runtime(SolverConfig(strategy="learned"), policy=Bad())
    result = r.solve(system())
    assert result.accepted
    assert r.state()["policy_disabled"]
    assert result.reused_rank == 0
    assert set(result.policy_features) >= {"operator_drift", "residual_projection", "previous_reduction"}


def test_learned_rank_bound_and_json_state():
    import json
    class Excess:
        def choose(self, features, max_rank):
            return 100000
    r = Runtime(SolverConfig(strategy="learned", memory_budget=2), policy=Excess())
    for k in range(5):
        s = system(source=1+k*.1)
        f = r.solve(s)
        r.gradient(s,f)
    assert r.state()["rank"] <= 2
    json.dumps(r.state(), allow_nan=False)
    json.dumps(f.summary(), allow_nan=False)
    r.reset()
    assert r.state()["rank"] == 0
    assert r.state()["memory_bytes"] == 0


@pytest.mark.parametrize("kwargs", [{"rtol":float("nan")},{"maxiter":0},{"memory_budget":33},{"strategy":"unknown"},{"max_seconds":0}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        SolverConfig(**kwargs)


def test_gcrot_failed_solve_does_not_change_solution_memory(monkeypatch):
    from solvetide import runtime as module
    r = Runtime(SolverConfig(strategy="gcrot", memory_budget=4))
    first = r.solve(system(n=20))
    assert first.accepted
    saved = [(None, u.copy()) for _, u in r._gcrot]
    warm = r._warm["primal"].copy()
    def fail(A,b,**kwargs):
        for _, u in kwargs["CU"]:
            u[:] = 111
        kwargs["CU"].append((None,np.ones_like(b)))
        return np.zeros_like(b), 1
    monkeypatch.setattr(module.sla,"gcrotmk", fail)
    failed = r.solve(system(n=20,source=2))
    assert failed.x is None
    assert len(r._gcrot) == len(saved)
    for (_, actual), (_, before) in zip(r._gcrot,saved):
        np.testing.assert_array_equal(actual,before)
    np.testing.assert_array_equal(r._warm["primal"],warm)


def test_slow_policy_disables_inference_cost(monkeypatch):
    import time
    class Slow:
        def choose(self,features,max_rank):
            time.sleep(.005)
            return max_rank
    r = Runtime(SolverConfig(strategy="learned"),policy=Slow())
    first = r.solve(system())
    assert first.accepted
    # A recorded tiny classical cost models a near-instant warm solve.
    r._stats["primal"]["classical_seconds"] = 1e-6
    second = r.solve(system(source=2))
    assert second.accepted
    assert r.state()["policy_disabled"]
    assert second.reused_rank == 0
    assert "inference" in second.message


@pytest.mark.parametrize("family", ["heat", "elasticity"])
def test_real_fem_original_equations_and_gradient_steps(family):
    from solvetide.assembly import assemble
    parameters = {"material":1.3,"source":.7,"contrast":2.,"boundary_value":.1}
    s = assemble(family,n=6,parameters=parameters)
    r = Runtime(SolverConfig(rtol=1e-11,atol=1e-13))
    f = r.solve(s)
    assert f.accepted, f.summary()
    reference = spsolve(s.A,s.b)
    np.testing.assert_allclose(f.x,reference,rtol=2e-8,atol=1e-10)
    g = r.gradient(s,f)
    assert g.accepted
    for parameter in parameters:
        for h in [1e-3,1e-4,1e-5]:
            plus = dict(parameters); plus[parameter] += h
            minus = dict(parameters); minus[parameter] -= h
            sp = assemble(family,n=6,parameters=plus)
            sm = assemble(family,n=6,parameters=minus)
            jp = sp.objective(spsolve(sp.A,sp.b))[0]
            jm = sm.objective(spsolve(sm.A,sm.b))[0]
            fd = (jp-jm)/(2*h)
            assert g.gradient[parameter] == pytest.approx(fd,rel=5e-5,abs=1e-8)


def test_complex_system_refuses_silent_real_cast():
    s = system()
    s.A = s.A.astype(complex)
    s.A.data[0] += 1j
    assert Runtime().solve(s).status == "invalid_input"


def test_joint_plan_is_selected_once_and_held_for_adjoint():
    class Planner:
        calls = 0
        seen = None
        def plan(self, features, max_rank_budget):
            self.calls += 1
            self.seen = dict(features)
            return {"primal_rank":3,"adjoint_rank":1,"primal_quota":3,"adjoint_quota":1,"refresh":True}
    policy = Planner()
    r = Runtime(SolverConfig(strategy="learned",memory_budget=4), policy=policy)
    s = system()
    f = r.solve(s)
    g = r.gradient(s,f)
    assert g.accepted
    assert policy.calls == 1
    assert f.policy_action == g.adjoint.policy_action
    assert f.policy_features == policy.seen == g.adjoint.policy_features
    assert set(f.policy_features) == {"operator_drift","residual_projection","previous_primal_reduction",
        "previous_adjoint_reduction","previous_primal_iterations","previous_adjoint_iterations",
        "available_primal_rank","available_adjoint_rank","dimension"}
    assert r.state()["policy_action"] == f.policy_action


def test_joint_quota_switch_prunes_excluded_role_and_skip_disables_both():
    class Planner:
        action = {"primal_rank":2,"adjoint_rank":2,"primal_quota":2,"adjoint_quota":2,"refresh":True}
        def plan(self, features, budget):
            return self.action.copy()
    policy = Planner()
    r = Runtime(SolverConfig(strategy="learned",memory_budget=4),policy=policy)
    # This test isolates allocation, not a scheduler-dependent timing guard.
    for stats in r._stats.values():
        stats["classical_seconds"] = 1e6
    for k in range(3):
        s = system(n=20)
        s.b += .01*k*np.cos(np.arange(20))
        f = r.solve(s); assert f.accepted
        assert r.gradient(s,f).accepted
    assert "adjoint" in r.state()["roles"]
    # Suppress timing fallback here to isolate allocation semantics.
    for stats in r._stats.values():
        stats["classical_seconds"] = 1.
    policy.action = {"primal_rank":4,"adjoint_rank":0,"primal_quota":4,"adjoint_quota":0,"refresh":True}
    s = system(n=20,source=2)
    f = r.solve(s); g = r.gradient(s,f)
    assert g.accepted
    assert "adjoint" not in r.state()["roles"]
    assert g.adjoint.reused_rank == 0
    assert f.reused_rank <= 4
    policy.action = {"primal_rank":2,"adjoint_rank":2,"primal_quota":2,"adjoint_quota":2,"refresh":False}
    f = r.solve(system(n=20,source=3)); g = r.gradient(system(n=20,source=3),f)
    assert g.accepted
    assert f.reused_rank == g.adjoint.reused_rank == 0
    assert r.state()["roles"].count("primal") <= 2
    assert r.state()["roles"].count("adjoint") <= 2


def test_invalid_joint_plan_disables_and_falls_back():
    class Invalid:
        def plan(self,features,budget):
            return {"primal_rank":5,"adjoint_rank":0,"primal_quota":5,"adjoint_quota":0,"refresh":True}
    r = Runtime(SolverConfig(strategy="learned",memory_budget=4),policy=Invalid())
    f = r.solve(system())
    assert f.accepted
    assert r.state()["policy_disabled"]
    assert f.reused_rank == 0


def test_joint_plan_no_failed_solution_retention():
    class Planner:
        def plan(self,features,budget):
            return {"primal_rank":2,"adjoint_rank":2,"primal_quota":2,"adjoint_quota":2,"refresh":True}
    r = Runtime(SolverConfig(strategy="learned",memory_budget=4,maxiter=1),policy=Planner())
    f = r.solve(system())
    assert not f.accepted
    assert r.state()["rank"] == 0
    assert not r._warm


def test_amg_elasticity_supplies_constrained_rigid_modes(monkeypatch):
    import pyamg
    from solvetide.assembly import assemble
    real = pyamg.smoothed_aggregation_solver
    seen = {}
    def capture(A,**kwargs):
        seen.update(kwargs)
        return real(A,**kwargs)
    monkeypatch.setattr(pyamg,"smoothed_aggregation_solver",capture)
    s = assemble("elasticity",n=8)
    r = Runtime(SolverConfig(strategy="amg"))
    f = r.solve(s)
    assert f.accepted, f.summary()
    assert seen["B"].shape == (s.ndofs,3)
    assert np.linalg.matrix_rank(seen["B"]) == 3
    assert seen["symmetry"] == "symmetric"
    assert seen["presmoother"][1]["sweep"] == "symmetric"
    assert seen["postsmoother"][1]["sweep"] == "symmetric"
    np.testing.assert_allclose(f.x,spsolve(s.A,s.b),rtol=1e-7,atol=1e-10)


def test_failed_joint_solve_does_not_update_measured_utilities():
    r = Runtime(SolverConfig(strategy="joint"))
    s = system(n=40)
    assert r.solve(s).accepted
    saved = list(r._utility)
    saved_stats = {role: dict(values) for role,values in r._stats.items()}
    r.config = SolverConfig(strategy="joint",maxiter=1)
    edited = system(n=40)
    edited.b += .1*np.cos(np.arange(40))
    failed = r.solve(edited)
    assert not failed.accepted
    assert r._utility == saved
    assert r._stats == saved_stats


@pytest.mark.parametrize("which", ["dA", "db", "objective", "dJdx"])
def test_gradient_rejects_complex_before_cast(which):
    s = system(n=8)
    r = Runtime()
    f = r.solve(s)
    assert f.accepted
    if which in {"dA", "db"}:
        dA,db = s.derivatives["material"]
        if which == "dA":
            dA = dA.astype(complex); dA.data[0] += 1j
        else:
            db = db.astype(complex); db[0] += 1j
        s.derivatives["material"] = (dA,db)
    elif which == "objective":
        s.objective = lambda x,target=None: (1+1j,np.ones_like(x))
    else:
        s.objective = lambda x,target=None: (1.,np.ones_like(x,dtype=complex)*(1+1j))
    g = r.gradient(s,f)
    assert not g.accepted
    assert g.gradient is None
    assert g.status == ("invalid_derivative" if which in {"dA","db"} else "invalid_objective")


def test_extreme_finite_input_refuses_and_summary_is_json_safe():
    import json
    s = system(n=2)
    s.A = sparse.eye(2,format="csr")
    s.b[:] = 1e200
    f = Runtime().solve(s)
    assert not f.accepted
    assert f.x is None
    assert f.status == "invalid_input"
    json.dumps(f.summary(),allow_nan=False)
    # Diagnostic sanitization is also defensive for constructed/imported results.
    f.residual_norm = float("inf")
    f.conditioning["condition_upper_bound"] = float("nan")
    clean = f.summary()
    assert clean["residual_norm"] is None
    assert clean["conditioning"]["condition_upper_bound"] is None
    json.dumps(clean,allow_nan=False)


def test_gradient_overflow_refuses_instead_of_exposing_infinity():
    s = system(n=2)
    s.A = sparse.eye(2,format="csr")
    s.b[:] = 1e100
    # Use a linear objective so its value and accepted adjoint stay finite.
    s.objective = lambda x,target=None: (float(np.sum(x)),np.full_like(x,1e100))
    s.derivatives = {"overflow": (sparse.eye(2,format="csr")*1e200,np.zeros(2))}
    r = Runtime(SolverConfig(strategy="direct"))
    f = r.solve(s)
    assert f.accepted
    g = r.gradient(s,f)
    assert not g.accepted
    assert g.gradient is None
    assert g.status == "numerical_failure"


def test_direct_preallocation_bound_refuses_before_factorization(monkeypatch):
    from solvetide import runtime as module
    called = False
    def forbidden(*args,**kwargs):
        nonlocal called
        called = True
        raise AssertionError("splu must not run beyond conservative stored-factor budget")
    monkeypatch.setattr(module.sla,"splu",forbidden)
    f = Runtime(SolverConfig(strategy="direct",max_factor_bytes=1)).solve(system(n=2))
    assert f.status == "resource_limit"
    assert not f.accepted and f.x is None
    assert "conservative factor storage bound" in f.message
    assert not called


def test_direct_storage_bound_accepts_budget_boundary_and_counts_permutations():
    n = 8
    bound = 16*n*(n+1)+32*n+16
    f = Runtime(SolverConfig(strategy="direct",max_factor_bytes=bound-1)).solve(system(n=n))
    assert f.status == "resource_limit"
    r = Runtime(SolverConfig(strategy="direct",max_factor_bytes=bound))
    f = r.solve(system(n=n))
    assert f.accepted
    factor = r._factor
    expected = sum(a.nbytes for mat in (factor.L,factor.U) for a in (mat.data,mat.indices,mat.indptr)) + factor.perm_r.nbytes + factor.perm_c.nbytes
    assert r._factor_bytes == expected
    assert expected <= bound
