"""Residual-checked sparse SPD solves and bounded joint primal/adjoint memory.

Memory supplies an initial Galerkin correction only. SciPy's classical solver,
with a fixed preconditioner during each call, still decides the candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import norm as _stable_norm
from scipy.sparse import linalg as sla

Array = NDArray[np.float64]
MAX_MAGNITUDE = 1e150


def _json_safe(value: Any) -> Any:
    """Preserve diagnostic structure while excluding invalid JSON numbers."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _norm(value: Any, ord: Any = None) -> float:
    # BLAS scaled vector norms avoid squaring into overflow/underflow. Inputs
    # are validated separately; nonfinite intermediate norms explicitly refuse.
    return float(_stable_norm(value, ord=ord, check_finite=False))
STRATEGIES = {"cold", "warm", "fixed", "joint", "drift", "learned", "direct", "gcrot", "amg", "amg_reuse"}


@dataclass(frozen=True)
class SolverConfig:
    rtol: float = 1e-9
    atol: float = 1e-11
    maxiter: int = 2000
    memory_budget: int = 12
    strategy: str = "joint"
    max_seconds: float = 10.0
    drift_threshold: float = 0.15
    max_input_bytes: int = 256 * 1024 * 1024
    max_factor_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if not all(math.isfinite(v) and v >= 0 for v in (self.rtol, self.atol)) or max(self.rtol, self.atol) == 0:
            raise ValueError("finite nonnegative tolerances, at least one positive, required")
        if not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if not isinstance(self.memory_budget, int) or not 0 <= self.memory_budget <= 32:
            raise ValueError("memory_budget must be an integer between 0 and 32")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        if not math.isfinite(self.drift_threshold) or self.drift_threshold < 0:
            raise ValueError("drift_threshold must be finite and nonnegative")
        if min(self.max_input_bytes, self.max_factor_bytes) < 1:
            raise ValueError("memory limits must be positive")


@dataclass
class SolveResult:
    status: str
    x: Array | None = None
    iterations: int = 0
    residual_norm: float | None = None
    relative_residual: float | None = None
    backward_error: float | None = None
    elapsed_seconds: float = 0.0
    reused_rank: int = 0
    cache_event: str = "empty"
    timings: dict[str, float] = field(default_factory=dict)
    memory_bytes: int = 0
    matvecs: int = 0
    message: str = ""
    identity: dict[str, str] = field(default_factory=dict)
    role: str = "primal"
    conditioning: dict[str, Any] = field(default_factory=dict)
    policy_features: dict[str, float] = field(default_factory=dict)
    policy_action: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "converged" and self.x is not None

    def summary(self) -> dict[str, Any]:
        """JSON-safe solver state without potentially large field arrays."""
        return _json_safe({k: v for k, v in vars(self).items() if k != "x"})


@dataclass
class GradientResult:
    status: str
    objective: float | None = None
    gradient: dict[str, float] | None = None
    adjoint: SolveResult | None = None
    message: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == "converged" and self.gradient is not None


class _Stop(Exception):
    def __init__(self, status: str):
        self.status = status


def _bytes(a: Any) -> bytes:
    a = np.ascontiguousarray(a)
    return str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()


def _hash_matrix(A: sparse.csr_matrix) -> str:
    h = sha256(str(A.shape).encode())
    for a in (A.indptr, A.indices, A.data):
        h.update(_bytes(a))
    return h.hexdigest()


def _identities(system: Any, A: sparse.csr_matrix, b: Array) -> dict[str, str]:
    # Recompute from bytes, not a caller-supplied operator/rhs identity.
    mapping = sha256(str(A.shape).encode())
    for attr in ("coordinates", "cells", "free", "fixed"):
        if hasattr(system, attr):
            mapping.update(_bytes(getattr(system, attr)))
    mapping.update(str(getattr(system, "family", "external")).encode())
    mapping.update(str(getattr(system, "identity", {}).get("mapping", "")).encode())
    return {"operator": _hash_matrix(A), "rhs": sha256(_bytes(b)).hexdigest(),
            "mapping": mapping.hexdigest(), "precision": str(A.dtype)}


class Runtime:
    """One chronological edit stream; do not share an instance across threads.

    A mapping change resets all saved state. A changed operator always refreshes
    products from actual matrix bytes. A previous result is never returned for a
    new solve, even when a warm start happens to satisfy its new equations.
    """

    def __init__(self, config: SolverConfig | None = None, policy: Any = None):
        self.config = config or SolverConfig()
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self._vectors: list[Array] = []
        self._roles: list[str] = []
        self._utility: list[float] = []
        self._warm: dict[str, Array] = {}
        self._last: dict[str, str] | None = None
        self._previous_A: sparse.csr_matrix | None = None
        self._stats = {r: {"iterations": 0.0, "reduction": 0.0, "classical_seconds": 0.0} for r in ("primal", "adjoint")}
        self._last_policy_features: dict[str, float] = {}
        self._pair_plan: dict[str, Any] | None = None
        self._pair_features: dict[str, float] = {}
        self._pair_identity: tuple[str, str] | None = None
        self._factor: Any = None
        self._factor_id: str | None = None
        self._factor_bytes = 0
        self._amg: Any = None
        self._amg_id: str | None = None
        self._gcrot: list[Any] = []
        self._policy_disabled = False
        self._solves = 0

    def _memory_bytes(self) -> int:
        basis = sum(v.nbytes for v in self._vectors)
        warm = sum(v.nbytes for v in self._warm.values())
        previous = 0 if self._previous_A is None else sum(a.nbytes for a in (self._previous_A.data, self._previous_A.indices, self._previous_A.indptr))
        gcrot = sum((0 if c is None else c.nbytes) + u.nbytes for c, u in self._gcrot)
        amg = 0
        if self._amg is not None:
            for level in self._amg.levels:
                candidates = getattr(level, "B", None)
                if candidates is not None:
                    amg += candidates.nbytes
                for name in ("A", "P", "R"):
                    mat = getattr(level, name, None)
                    if mat is not None:
                        amg += sum(a.nbytes for a in (mat.data, mat.indices, mat.indptr))
        return basis + warm + previous + gcrot + self._factor_bytes + amg

    def state(self) -> dict[str, Any]:
        return {"schema_version": 1, "strategy": self.config.strategy, "rank": len(self._vectors),
                "memory_budget": self.config.memory_budget, "roles": list(self._roles),
                "role_statistics": {r: dict(v) for r, v in self._stats.items()}, "memory_bytes": self._memory_bytes(),
                "identity": None if self._last is None else dict(self._last), "solves": self._solves,
                "policy_features": dict(self._last_policy_features),
                "policy_action": {} if self._pair_plan is None else dict(self._pair_plan),
                "policy_disabled": self._policy_disabled}

    @staticmethod
    def _validate_plan(plan: Any, budget: int) -> dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("joint policy plan must be a dict")
        keys = ("primal_rank", "adjoint_rank", "primal_quota", "adjoint_quota")
        checked: dict[str, Any] = {}
        for key in keys:
            value = plan.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"plan {key} must be a nonnegative integer")
            checked[key] = int(value)
        if not isinstance(plan.get("refresh"), (bool, np.bool_)):
            raise ValueError("plan refresh must be boolean")
        checked["refresh"] = bool(plan["refresh"])
        if checked["primal_quota"] + checked["adjoint_quota"] > budget:
            raise ValueError("joint plan quotas exceed shared memory budget")
        if any(checked[f"{role}_rank"] > checked[f"{role}_quota"] for role in ("primal", "adjoint")):
            raise ValueError("joint plan ranks exceed role quotas")
        return checked

    def _prune_plan_quotas(self) -> None:
        if self.config.strategy != "learned" or self._pair_plan is None:
            return
        for role in ("primal", "adjoint"):
            quota = self._pair_plan[f"{role}_quota"]
            while self._roles.count(role) > quota:
                candidates = [i for i, r in enumerate(self._roles) if r == role]
                remove = min(candidates, key=lambda i: self._utility[i])
                for data in (self._vectors, self._roles, self._utility):
                    data.pop(remove)

    @staticmethod
    def _amg_candidates(system: Any, dimension: int) -> Array:
        """Constrained rigid-body near-nullspace for interleaved 2D elasticity."""
        if getattr(system, "family", "") == "elasticity" and hasattr(system, "coordinates") and hasattr(system, "free"):
            coordinates = np.asarray(system.coordinates)
            free = np.asarray(system.free, dtype=np.int64)
            if coordinates.ndim == 2 and coordinates.shape[0] == 2 and free.shape == (dimension,):
                nodes = coordinates.shape[1]
                modes = np.zeros((2 * nodes, 3), dtype=np.float64)
                modes[0::2, 0] = 1.0
                modes[1::2, 1] = 1.0
                modes[0::2, 2] = -coordinates[1]
                modes[1::2, 2] = coordinates[0]
                candidates = modes[free]
                # Removing constrained components makes these near-null modes,
                # not true nullspaces of the supported SPD operator.
                _, singular, vh = np.linalg.svd(candidates, full_matrices=False)
                keep = singular > max(float(singular[0]) * 1e-12, 1e-30)
                return candidates @ vh[keep].T
        return np.ones((dimension, 1), dtype=np.float64)

    def _retain(self, x: Array, role: str, reduction: float, iterations: int, classical_seconds: float) -> None:
        cap = self.config.memory_budget
        prior = self._stats[role]["classical_seconds"]
        self._stats[role] = {"iterations": float(iterations), "reduction": float(reduction),
                             "classical_seconds": classical_seconds if prior == 0 else 0.8 * prior + 0.2 * classical_seconds}
        # Warm vectors are explicitly bounded to one per role, additional to the
        # correction-space rank budget and included in memory accounting.
        if self.config.strategy != "cold":
            self._warm[role] = x.copy()
        if cap == 0 or self.config.strategy not in {"fixed", "joint", "drift", "learned"}:
            return
        if self.config.strategy == "learned" and self._pair_plan is not None and self._pair_plan[f"{role}_quota"] == 0:
            return
        q = x.copy()
        original = _norm(q)
        if original == 0:
            return
        for _ in range(2):
            for v in self._vectors:
                q -= v * np.dot(v, q)
        qnorm = _norm(q)
        if qnorm <= 1e-10 * original:
            return
        q /= qnorm
        self._vectors.append(q)
        self._roles.append(role)
        self._utility.append(max(1e-6, reduction) * (1.0 + iterations))
        self._prune_plan_quotas()
        if len(self._vectors) > cap:
            if self.config.strategy == "fixed":
                remove = 0
            else:
                # Cost-aware allocation: preserve the last direction of a role
                # while both roles have evidence; otherwise evict least utility
                # per retained direction, with gradual aging after each solve.
                counts = {r: self._roles.count(r) for r in ("primal", "adjoint")}
                scores = [u / max(1, counts[r]) for u, r in zip(self._utility, self._roles)]
                eligible = [i for i, r in enumerate(self._roles) if counts[r] > 1]
                remove = min(eligible or list(range(len(scores))), key=lambda i: scores[i])
            for data in (self._vectors, self._roles, self._utility):
                data.pop(remove)
            # Removing an orthonormal vector preserves remaining orthogonality.
        self._prune_plan_quotas()
        self._utility = [0.95 * u for u in self._utility]

    def solve(self, system: Any, rhs: Array | None = None, role: str = "primal",
              cancel: Callable[[], bool] | None = None) -> SolveResult:
        start = perf_counter()
        result = SolveResult(status="invalid_input", role=role)
        timings: dict[str, float] = {}
        config = self.config
        deadline = start + config.max_seconds

        def check() -> None:
            if cancel is not None and cancel():
                raise _Stop("cancelled")
            if perf_counter() >= deadline:
                raise _Stop("time_limit")

        def finish(status: str, message: str = "") -> SolveResult:
            result.status = status
            result.message = message
            if status != "converged":
                result.x = None
            result.elapsed_seconds = perf_counter() - start
            timings["total"] = result.elapsed_seconds
            result.timings = timings
            result.memory_bytes = self._memory_bytes()
            return result

        try:
            check()
            if role not in ("primal", "adjoint"):
                return finish("invalid_input", "role must be primal or adjoint")
            t = perf_counter()
            supplied_rhs = system.b if rhs is None else rhs
            if np.iscomplexobj(system.A) or np.iscomplexobj(supplied_rhs):
                return finish("invalid_input", "only real-valued SPD systems are supported")
            A = sparse.csr_matrix(system.A, dtype=np.float64, copy=True)
            A.sum_duplicates()
            A.sort_indices()
            A.eliminate_zeros()
            b = np.array(system.b if rhs is None else rhs, dtype=np.float64, copy=True)
            if A.shape[0] == 0 or A.shape[0] != A.shape[1] or b.shape != (A.shape[0],):
                return finish("invalid_input", "square nonempty A and matching vector b required")
            if A.shape[0] > 40000 or A.nnz > 2000000:
                return finish("resource_limit", "matrix dimension or nonzero limit exceeded")
            if sum(a.nbytes for a in (A.data, A.indices, A.indptr, b)) > config.max_input_bytes:
                return finish("resource_limit", "input byte limit exceeded")
            if not np.isfinite(A.data).all() or not np.isfinite(b).all():
                return finish("invalid_input", "matrix and rhs must be finite")
            if (A.nnz and np.max(np.abs(A.data)) > MAX_MAGNITUDE) or np.max(np.abs(b)) > MAX_MAGNITUDE:
                return finish("invalid_input", "matrix and rhs magnitudes must not exceed 1e150")
            if np.any(A.diagonal() <= 0):
                return finish("invalid_input", "positive diagonal required for supported SPD systems")
            asym = A - A.T
            anorm = float(sla.norm(A, ord=np.inf))
            if asym.nnz and float(np.max(np.abs(asym.data))) > 1e-12 * max(1.0, anorm):
                return finish("invalid_input", "only symmetric positive-definite systems are supported")
            identity = _identities(system, A, b)
            result.identity = identity
            previous = self._last
            if previous is None:
                result.cache_event = "empty"
            elif previous["mapping"] != identity["mapping"] or previous["precision"] != identity["precision"]:
                self.reset()
                result.cache_event = "mapping_invalidated"
            elif previous["operator"] != identity["operator"]:
                result.cache_event = "operator_refreshed"
            elif previous["rhs"] != identity["rhs"]:
                result.cache_event = "rhs_changed"
            else:
                result.cache_event = "unchanged_equations"
            drift = 0.0
            if self._previous_A is not None and self._previous_A.shape == A.shape:
                drift = float(sla.norm(A - self._previous_A) / max(sla.norm(self._previous_A), np.finfo(float).tiny))
            if not math.isfinite(drift):
                # Saturation indicates an enormous operator change, never a
                # similarity claim. It keeps policy features valid JSON.
                drift = float(np.finfo(float).max)
            self._last = identity
            self._previous_A = A.copy()
            timings["validation_identity"] = perf_counter() - t
            check()
            diag = A.diagonal()
            rowsum = np.asarray(abs(A).sum(axis=1)).ravel()
            lower = float(np.min(2 * diag - rowsum))
            upper = float(np.max(rowsum))
            result.conditioning = {"method": "Gershgorin spectral enclosure", "lower_bound": lower,
                                   "upper_bound": upper,
                                   "condition_upper_bound": upper / lower if lower > 0 else None,
                                   "limit": "A nonpositive lower bound gives no condition bound; no physical or discretization error certificate."}
            threshold = max(config.atol, config.rtol * float(_norm(b)))
            warm_strategies = {"warm", "fixed", "joint", "drift", "learned", "gcrot", "amg", "amg_reuse"}
            x0 = self._warm.get(role, np.zeros_like(b)).copy() if config.strategy in warm_strategies else np.zeros_like(b)
            initial_r = b - A @ x0
            initial_norm = float(_norm(initial_r))
            if not math.isfinite(threshold) or not math.isfinite(initial_norm) or not np.isfinite(initial_r).all():
                return finish("numerical_failure", "nonfinite initial residual or tolerance arithmetic")
            t = perf_counter()
            Q = np.column_stack(self._vectors) if self._vectors else np.empty((b.size, 0))
            projection = float(_norm(Q.T @ initial_r) / max(initial_norm, np.finfo(float).tiny))
            features = {"operator_drift": drift, "residual_projection": projection,
                        "previous_reduction": self._stats[role]["reduction"],
                        "previous_iterations": self._stats[role]["iterations"],
                        "available_rank": float(Q.shape[1]), "role_adjoint": float(role == "adjoint"),
                        "dimension": float(b.size)}
            result.policy_features = features
            self._last_policy_features = dict(features)
            rank = Q.shape[1]
            if config.strategy in {"cold", "warm", "direct", "gcrot", "amg", "amg_reuse"}:
                rank = 0
            if config.strategy == "drift" and drift > config.drift_threshold:
                rank = 0
            tpolicy = perf_counter()
            uses_plan = config.strategy == "learned" and self.policy is not None and hasattr(self.policy, "plan")
            if uses_plan:
                pair_features = {
                    "operator_drift": drift, "residual_projection": projection,
                    "previous_primal_reduction": self._stats["primal"]["reduction"],
                    "previous_adjoint_reduction": self._stats["adjoint"]["reduction"],
                    "previous_primal_iterations": self._stats["primal"]["iterations"],
                    "previous_adjoint_iterations": self._stats["adjoint"]["iterations"],
                    "available_primal_rank": float(self._roles.count("primal")),
                    "available_adjoint_rank": float(self._roles.count("adjoint")),
                    "dimension": float(b.size),
                }
                pair_identity = (identity["operator"], identity["mapping"])
                if role == "primal":
                    self._pair_features = pair_features
                    self._pair_identity = pair_identity
                    if not self._policy_disabled:
                        try:
                            self._pair_plan = self._validate_plan(self.policy.plan(pair_features, config.memory_budget), config.memory_budget)
                        except Exception:
                            self._policy_disabled = True
                            self._pair_plan = None
                            result.message = "policy disabled after invalid joint plan; warm fallback"
                elif self._pair_identity != pair_identity:
                    # An orphan adjoint does not infer a new plan from future
                    # sensitivity information. It uses the classical fallback.
                    self._pair_plan = None
                if self._pair_plan is not None:
                    if self._policy_disabled:
                        self._pair_plan = dict(self._pair_plan, primal_rank=0, adjoint_rank=0, refresh=False)
                    self._prune_plan_quotas()
                    result.policy_action = dict(self._pair_plan)
                    rank = min(self._pair_plan[f"{role}_rank"], len(self._vectors)) if self._pair_plan["refresh"] else 0
                else:
                    rank = 0
                result.policy_features = dict(self._pair_features if self._pair_identity == pair_identity else pair_features)
                self._last_policy_features = dict(result.policy_features)
                Q = np.column_stack(self._vectors) if self._vectors else np.empty((b.size, 0))
            elif config.strategy == "learned":
                if self.policy is None or self._policy_disabled:
                    rank = 0
                else:
                    try:
                        proposed = self.policy.choose(features, rank) if hasattr(self.policy, "choose") else self.policy(features, rank)
                        if isinstance(proposed, bool) or not np.isfinite(proposed):
                            raise ValueError("nonfinite rank")
                        rank = min(rank, max(0, int(proposed)))
                    except Exception:
                        self._policy_disabled = True
                        rank = 0
                        result.message = "policy disabled after invalid prediction; classical fallback"
            timings["policy"] = perf_counter() - tpolicy
            historical = sum(stat["classical_seconds"] for stat in self._stats.values()) if uses_plan else self._stats[role]["classical_seconds"]
            if config.strategy == "learned" and self.policy is not None and historical > 0 and timings["policy"] > historical:
                self._policy_disabled = True
                rank = 0
                if self._pair_plan is not None:
                    self._pair_plan = dict(self._pair_plan, primal_rank=0, adjoint_rank=0, refresh=False)
                    result.policy_action = dict(self._pair_plan)
                result.message = "policy disabled: inference exceeds historical classical solve time"
            # Do not pay A@Q when no correction is requested. Every used
            # product is freshly formed from the current matrix; none is cached.
            AQ = A @ Q if rank else np.empty((b.size, 0))
            energies = np.sum(Q * AQ, axis=0) if rank else np.empty(0)
            selected = np.arange(rank)
            pending_utility_updates: list[tuple[int, float]] = []
            if rank and config.strategy in {"joint", "drift", "learned"}:
                relevance = np.abs(Q.T @ initial_r) / np.sqrt(np.maximum(energies, np.finfo(float).tiny))
                # Joint retention already allocates roles by measured work. The
                # current operator's energy relevance selects this solve's rank.
                selected = np.argsort(-relevance, kind="stable")[:rank]
            if rank:
                q, aq = Q[:, selected], AQ[:, selected]
                small = q.T @ aq
                small = (small + small.T) * 0.5
                eigenvalues, eigenvectors = np.linalg.eigh(small)
                keep = eigenvalues > max(1e-13 * max(float(eigenvalues[-1]), 1e-30), 1e-30)
                if np.any(keep):
                    z = eigenvectors[:, keep]
                    correction = q @ (z @ ((z.T @ (q.T @ initial_r)) / eigenvalues[keep]))
                    candidate = x0 + correction
                    corrected_norm = float(_norm(b - A @ candidate))
                    # A Galerkin correction minimizes energy error, not 2-norm
                    # residual. Reject gross residual amplification explicitly.
                    if np.isfinite(candidate).all() and corrected_norm <= 2 * max(initial_norm, threshold):
                        x0 = candidate
                        result.reused_rank = int(np.count_nonzero(keep))
                        reduction = max(0.0, 1.0 - corrected_norm / max(initial_norm, np.finfo(float).tiny))
                        for index in selected:
                            pending_utility_updates.append((int(index), reduction * (1 + self._stats[role]["iterations"])))
            timings["basis_update_projection"] = perf_counter() - t
            after_projection = float(_norm(b - A @ x0))
            reduction = max(0.0, 1.0 - after_projection / max(initial_norm, np.finfo(float).tiny))
            check()
            t = perf_counter()
            max_matvecs = 3 * config.maxiter + 10

            def matvec(v: Array) -> Array:
                check()
                result.matvecs += 1
                if result.matvecs > max_matvecs:
                    raise _Stop("work_limit")
                return A @ v

            op = sla.LinearOperator(A.shape, matvec=matvec, dtype=np.float64)
            M = sla.LinearOperator(A.shape, matvec=lambda v: v / diag, dtype=np.float64)
            info = 0
            pending_gcrot = None
            if _norm(b - A @ x0) <= threshold:
                x = x0
            elif config.strategy == "direct":
                if b.size > 4000:
                    return finish("resource_limit", "direct factorization limited to 4000 unknowns")
                # A dense triangular L and U together have n*(n+1)
                # entries. Bound each value plus its column index using
                # float64+int64 (16 bytes), two int64 row-pointer arrays,
                # and two int64 permutation arrays before asking SuperLU
                # to allocate anything. This is stored-factor space only,
                # not a process-RSS or opaque workspace guarantee.
                factor_storage_bound = 16 * b.size * (b.size + 1) + 32 * b.size + 16
                if factor_storage_bound > config.max_factor_bytes:
                    return finish("resource_limit", f"conservative factor storage bound {factor_storage_bound} exceeds configured byte budget")
                if self._factor_id != identity["operator"]:
                    self._factor = sla.splu(A.tocsc())
                    self._factor_bytes = sum(a.nbytes for mat in (self._factor.L, self._factor.U) for a in (mat.data, mat.indices, mat.indptr)) + self._factor.perm_r.nbytes + self._factor.perm_c.nbytes
                    if self._factor_bytes > config.max_factor_bytes:
                        self._factor = None
                        self._factor_id = None
                        self._factor_bytes = 0
                        return finish("resource_limit", "factor fill exceeded configured byte budget")
                    self._factor_id = identity["operator"]
                    result.cache_event += ":factor_built"
                else:
                    result.cache_event += ":factor_reused"
                check()
                x = self._factor.solve(b)
                result.iterations = 1
            else:
                if config.strategy in {"amg", "amg_reuse"}:
                    import pyamg
                    if self._amg is None or (config.strategy == "amg" and self._amg_id != identity["operator"]):
                        B = self._amg_candidates(system, b.size)
                        self._amg = pyamg.smoothed_aggregation_solver(
                            A, B=B, symmetry="symmetric",
                            presmoother=("gauss_seidel", {"sweep": "symmetric"}),
                            postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
                        )
                        self._amg_id = identity["operator"]
                        result.cache_event += ":amg_built"
                    else:
                        result.cache_event += ":amg_reused"
                    M = self._amg.aspreconditioner(cycle="V")
                def callback(_: Array) -> None:
                    result.iterations += 1
                    check()
                if config.strategy == "gcrot":
                    # Drop all C products; SciPy recomputes them under current A.
                    CU = [(None, u.copy()) for _, u in self._gcrot]
                    x, info = sla.gcrotmk(op, b, x0=x0, rtol=config.rtol, atol=config.atol,
                                         maxiter=config.maxiter, M=M, callback=callback, m=20,
                                         k=config.memory_budget, CU=CU, discard_C=True)
                    pending_gcrot = [(None, u.copy()) for _, u in CU[-config.memory_budget:]] if config.memory_budget else []
                else:
                    # M is held fixed within CG; learned memory never acts as a
                    # varying CG preconditioner.
                    x, info = sla.cg(op, b, x0=x0, rtol=config.rtol, atol=config.atol,
                                     maxiter=config.maxiter, M=M, callback=callback)
            timings["classical_solve_setup"] = perf_counter() - t
            check()
            t = perf_counter()
            if not np.isfinite(x).all() or np.max(np.abs(x)) > MAX_MAGNITUDE:
                return finish("numerical_failure", "candidate is nonfinite or exceeds the 1e150 magnitude limit")
            residual = b - A @ x
            residual_norm = float(_norm(residual))
            denominator = anorm * float(_norm(x, np.inf)) + float(_norm(b, np.inf))
            backward = float(_norm(residual, np.inf)) / denominator if denominator else 0.0
            result.residual_norm = residual_norm
            result.relative_residual = residual_norm / max(float(_norm(b)), np.finfo(float).tiny)
            result.backward_error = backward
            timings["independent_residual_verification"] = perf_counter() - t
            if (not np.isfinite(residual).all() or not all(math.isfinite(v) for v in
                    (residual_norm, denominator, backward, result.relative_residual, threshold))):
                return finish("numerical_failure", "nonfinite residual norm, tolerance or backward-error arithmetic")
            if info < 0:
                return finish("numerical_failure", f"classical solver breakdown ({info})")
            if residual_norm > threshold:
                return finish("max_work" if info > 0 else "residual_rejected", "true residual exceeds declared tolerance")
            if backward > max(config.rtol, config.atol / max(denominator, np.finfo(float).tiny)):
                return finish("backward_error_rejected", "normwise backward error exceeds tolerance")
            t = perf_counter()
            if pending_gcrot is not None:
                self._gcrot = pending_gcrot
            for index, measured_utility in pending_utility_updates:
                self._utility[index] = 0.8 * self._utility[index] + 0.2 * measured_utility
            self._retain(x, role, reduction, result.iterations, timings["classical_solve_setup"])
            self._solves += 1
            result.x = x.copy()
            result.x.flags.writeable = False
            timings["memory_retention"] = perf_counter() - t
            return finish("converged", result.message)
        except _Stop as exc:
            return finish(exc.status, "solve interrupted; no field or gradient accepted")
        except (ValueError, TypeError, AttributeError) as exc:
            return finish("invalid_input", str(exc))
        except (RuntimeError, np.linalg.LinAlgError, ImportError) as exc:
            return finish("numerical_failure", str(exc))

    def gradient(self, system: Any, forward: SolveResult, target: Array | None = None,
                 cancel: Callable[[], bool] | None = None) -> GradientResult:
        if not forward.accepted or forward.role != "primal":
            return GradientResult("forward_failed", message="an accepted primal solve is required")
        try:
            if np.iscomplexobj(system.A) or np.iscomplexobj(system.b):
                return GradientResult("invalid_input", message="only real-valued equations are supported")
            A = sparse.csr_matrix(system.A, dtype=np.float64, copy=True)
            A.sum_duplicates(); A.sort_indices(); A.eliminate_zeros()
            b = np.asarray(system.b, dtype=np.float64)
            if _identities(system, A, b) != forward.identity:
                return GradientResult("stale_forward", message="equations or mapping changed since forward solve")
            # Do not trust the public result's flag or a caller-mutated field.
            residual = b - A @ forward.x
            residual_norm = _norm(residual)
            if not math.isfinite(residual_norm) or not np.isfinite(residual).all() or residual_norm > max(self.config.atol, self.config.rtol * _norm(b)):
                return GradientResult("forward_rejected", message="forward field no longer satisfies its equations")
            value, dJdx = system.objective(forward.x, target)
            if np.iscomplexobj(value) or np.iscomplexobj(dJdx) or np.asarray(value).ndim != 0:
                return GradientResult("invalid_objective", message="objective scalar and derivative must be real-valued")
            value = float(value)
            dJdx = np.asarray(dJdx, dtype=np.float64)
            if not np.isfinite(value) or dJdx.shape != b.shape or not np.isfinite(dJdx).all():
                return GradientResult("invalid_objective", message="objective and derivative must be finite and shape-matched")
            # The supported matrix is symmetric; this is the transpose solve.
            adjoint = self.solve(system, rhs=dJdx, role="adjoint", cancel=cancel)
            if not adjoint.accepted:
                return GradientResult("adjoint_failed", adjoint=adjoint, message=adjoint.message)
            gradients = {}
            for name, (dA, db) in system.derivatives.items():
                if np.iscomplexobj(dA) or np.iscomplexobj(db):
                    return GradientResult("invalid_derivative", adjoint=adjoint, message=f"complex derivative {name} is unsupported")
                dA = sparse.csr_matrix(dA, dtype=np.float64)
                db = np.asarray(db, dtype=np.float64)
                if dA.shape != A.shape or db.shape != b.shape or not np.isfinite(dA.data).all() or not np.isfinite(db).all():
                    return GradientResult("invalid_derivative", adjoint=adjoint, message=f"bad derivative {name}")
                with np.errstate(over="ignore", invalid="ignore"):
                    derivative_rhs = db - dA @ forward.x
                    derivative_value = float(np.dot(adjoint.x, derivative_rhs))
                if not np.isfinite(derivative_rhs).all() or not math.isfinite(derivative_value):
                    return GradientResult("numerical_failure", adjoint=adjoint, message=f"nonfinite gradient arithmetic for {name}")
                gradients[name] = derivative_value
            if not all(math.isfinite(v) for v in gradients.values()):
                return GradientResult("numerical_failure", adjoint=adjoint, message="nonfinite gradient")
            return GradientResult("converged", float(value), gradients, adjoint)
        except (ValueError, TypeError, AttributeError, OverflowError, FloatingPointError) as exc:
            return GradientResult("invalid_input", message=str(exc))
