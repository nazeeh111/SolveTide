"""Constrained P1 finite-element systems and exact discrete parameter derivatives.

Only the supported assemblers establish the SPD contract.  A small residual
is a numerical solve check, not a certificate about the continuous model.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, LinearForm, MeshTri, asm
from skfem.helpers import ddot, dot, grad, sym_grad, trace

Array = NDArray[np.float64]
DEFAULT_PARAMETERS = {"material": 1.0, "source": 1.0, "boundary_value": 0.0, "contrast": 1.0}
BOUNDARIES = ("left", "right", "both", "all")
FAMILIES = ("heat", "elasticity")


def _readonly(value: NDArray) -> NDArray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _csr(matrix) -> csr_matrix:
    result = csr_matrix(matrix, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    # Freeze the sparse storage too: stale identities must not follow edits.
    for array in (result.data, result.indices, result.indptr):
        array.flags.writeable = False
    return result


def _hash(*items) -> str:
    h = sha256()
    for item in items:
        if hasattr(item, "dtype"):
            a = np.ascontiguousarray(item)
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(str(item).encode())
        h.update(b"\0")
    return h.hexdigest()


@dataclass(frozen=True)
class System:
    """An assembled, constrained binary64 system. ``x`` always means free DOFs.

    Displacement DOFs are interleaved x,y per node. ``expand`` restores the
    fixed values. The derivatives are of the reduced equations, including
    elimination of the nonzero prescribed boundary values.
    """

    A: csr_matrix
    b: Array
    derivatives: Mapping[str, tuple[csr_matrix, Array]]
    coordinates: Array
    cells: NDArray[np.int64]
    free: NDArray[np.int64]
    fixed: NDArray[np.int64]
    prescribed: Array
    family: str
    n: int
    parameters: Mapping[str, float]
    boundary: str
    identity: Mapping[str, str]
    assembly_seconds: float

    def expand(self, x: Array) -> Array:
        if np.iscomplexobj(x):
            raise ValueError("solution must be real-valued")
        value = np.asarray(x, dtype=np.float64)
        if value.shape != self.b.shape or not np.all(np.isfinite(value)):
            raise ValueError("solution must be a finite vector of free DOFs")
        full = self.prescribed.copy()
        full[self.free] = value
        return full

    def objective(self, x: Array, target=None) -> tuple[float, Array]:
        """Mean-square field mismatch over free DOFs, not a spatial integral.

        ``target`` is zero, a finite scalar, or a vector of free DOFs. It is
        held fixed in parameter differentiation. Prescribed DOFs are excluded.
        """
        if np.iscomplexobj(x) or (target is not None and np.iscomplexobj(target)):
            raise ValueError("objective field and target must be real-valued")
        value = np.asarray(x, dtype=np.float64)
        if value.shape != self.b.shape or not np.all(np.isfinite(value)):
            raise ValueError("objective requires a finite free-DOF field")
        goal = np.asarray(0.0 if target is None else target, dtype=np.float64)
        if goal.ndim > 0 and goal.shape != value.shape:
            raise ValueError("target must be scalar or match the free-DOF field")
        if not np.all(np.isfinite(goal)):
            raise ValueError("target must be finite")
        error = value - goal
        return float(0.5 * np.mean(error * error)), error / len(error)

    @property
    def ndofs(self) -> int:
        return self.A.shape[0]

    @property
    def components(self) -> int:
        return 2 if self.family == "elasticity" else 1


def assemble(family: str = "heat", n: int = 16, parameters: Mapping[str, float] | None = None,
             boundary: str = "left") -> System:
    """Assemble heat or supported plane-strain elasticity on a unit square.

    ``n`` is the number of intervals along each edge (4..128). Natural zero
    flux/traction applies on unprescribed edges. A positive ``material`` is
    conductivity or Young's modulus. ``contrast`` multiplies it in the
    top-right quadrant; ``source`` is uniform heat or vertical body force.
    ``boundary_value`` prescribes temperature or vertical displacement;
    horizontal prescribed displacement is zero.
    """
    start = perf_counter()
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 4 <= n <= 128:
        raise ValueError("n must be an integer from 4 through 128")
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}; unconstrained systems are unsupported")
    supplied = {} if parameters is None else dict(parameters)
    unknown = supplied.keys() - DEFAULT_PARAMETERS.keys()
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}")
    p = dict(DEFAULT_PARAMETERS)
    for key, value in supplied.items():
        if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value) or not np.isscalar(value):
            raise ValueError(f"{key} must be a finite real scalar")
        try:
            p[key] = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a finite real scalar") from exc
    if not all(np.isfinite(v) for v in p.values()):
        raise ValueError("parameters must be finite")
    if not all(1e-10 <= p[k] <= 1e10 for k in ("material", "contrast")):
        raise ValueError("material and contrast must lie in [1e-10, 1e10]")
    if max(abs(p["source"]), abs(p["boundary_value"])) > 1e12:
        raise ValueError("source and boundary_value magnitude must be <= 1e12")

    points = np.linspace(0.0, 1.0, int(n) + 1)
    mesh = MeshTri.init_tensor(points, points)
    scalar = ElementTriP1()
    basis = Basis(mesh, scalar if family == "heat" else ElementVector(scalar))

    def edge(x):
        left = np.isclose(x[0], 0.0)
        right = np.isclose(x[0], 1.0)
        if boundary == "left":
            return left
        if boundary == "right":
            return right
        if boundary == "both":
            return left | right
        return left | right | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    fixed = basis.get_dofs(edge).all()
    free = np.setdiff1d(np.arange(basis.N), fixed)
    if len(fixed) == 0 or len(free) == 0:
        raise ValueError("a nonempty supported free/fixed partition is required")
    # The supported edge always has multiple nodes, eliminating all 2D rigid
    # translations and rotation when both displacement components are fixed.
    prescribed_unit = np.zeros(basis.N)
    if family == "heat":
        prescribed_unit[fixed] = 1.0
    else:
        prescribed_unit[np.intersect1d(basis.nodal_dofs[1], fixed)] = 1.0
    prescribed = p["boundary_value"] * prescribed_unit

    @BilinearForm
    def uniform_form(u, v, w):
        if family == "heat":
            return dot(grad(u), grad(v))
        eu, ev = sym_grad(u), sym_grad(v)
        # Plane strain, E=1, nu=0.3. E is varied by material/contrast.
        nu = 0.3
        mu = 1.0 / (2.0 * (1.0 + nu))
        lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return 2.0 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev)

    @BilinearForm
    def patch_form(u, v, w):
        patch = (w.x[0] > 0.5) & (w.x[1] > 0.5)
        return patch * uniform_form.form(u, v, w)

    @LinearForm
    def load_form(v, w):
        return v if family == "heat" else v[1]

    uniform = asm(uniform_form, basis).tocsr()
    patch = asm(patch_form, basis).tocsr()
    material_derivative = uniform + (p["contrast"] - 1.0) * patch
    full_A = p["material"] * material_derivative
    contrast_derivative = p["material"] * patch
    load = asm(load_form, basis)
    full_b = p["source"] * load
    A = _csr(full_A[free][:, free])
    b = _readonly(full_b[free] - full_A[free] @ prescribed)
    zero = _csr(csr_matrix(A.shape))
    derivatives = {
        "material": (_csr(material_derivative[free][:, free]),
                     _readonly(-(material_derivative[free] @ prescribed))),
        "contrast": (_csr(contrast_derivative[free][:, free]),
                     _readonly(-(contrast_derivative[free] @ prescribed))),
        "source": (zero, _readonly(load[free])),
        "boundary_value": (zero, _readonly(-(full_A[free] @ prescribed_unit))),
    }
    coordinates = _readonly(mesh.p)
    cells = _readonly(mesh.t.astype(np.int64))
    free = _readonly(free.astype(np.int64))
    fixed = _readonly(fixed.astype(np.int64))
    identity = {
        "schema": "solvetide-system-v1",
        "operator": _hash(A.shape, A.indptr, A.indices, A.data),
        "rhs": _hash(b),
        "mapping": _hash("p1-v1", family, coordinates, cells, free, fixed),
        "precision": "float64",
        "boundary": _hash(boundary, fixed, prescribed),
    }
    return System(A, b, MappingProxyType(derivatives), coordinates, cells, free, fixed,
                  _readonly(prescribed), family, int(n), MappingProxyType(p), boundary,
                  MappingProxyType(identity), perf_counter() - start)
