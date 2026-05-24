"""Edge-vector generalized Poisson's ratio for the bipartite mechanism
(task 4).

The simulator's :meth:`auxetic.simulation.Simulator.poissons_ratio`
(SPEC §7.4) measures the auxetic response as a *bounding-box* lateral /
axial strain. This module defines a local, per-triangle alternative
built from how a triangle's **edge connection points** move as the
rotating-units mechanism actuates.

Geometry (matching :mod:`auxetic.bipartite`): a triangle with corners
``P_0, P_1, P_2`` has centroid ``M`` and hinges
``T_c = P_c + t·(M - P_c)`` with ``t = 1/(1+C)``. Under actuation
``theta`` each corner kite rotates rigidly about its hinge.

Why not the corners?  The actuated corners satisfy
``Q_c = M + B·(P_c - M)`` with a **shape-independent** linear part
``B = t·R(θ) + (1-t)·I``, so their strain ``ε = t(cosθ-1)·I`` is
isotropic for *every* triangle — a single triangle's corner motion is
always perfectly auxetic (``ν = -1``). The shape and ``C`` dependence
lives in the **connection points** along the edges: each triangle edge
``(a, b)`` carries two perpendicular feet — ``E_ab`` (dropped from
hinge ``T_a``) and ``E_ba`` (from ``T_b``) — that rotate about
*different* hinges, so the bond midpoint ``(E_ab + E_ba)/2`` moves
non-affinely. The triangle of the three edge bond-midpoints therefore
deforms anisotropically in a shape/``C``-dependent way.

The metric: build the rest (``theta=0``) and actuated edge-midpoint
triangles, recover the affine deformation gradient ``A`` between them,
the small-strain tensor ``ε = (A + Aᵀ)/2 - I``, and a generalized
Poisson's ratio:

- **principal** (default): ``ν = -ε₂/ε₁`` with eigenstrains ordered
  ``|ε₁| ≥ |ε₂|``. Equilateral → ``ν = -1`` (isotropic); anisotropic
  shapes drift toward 0 / positive.
- **directional**: pass ``axis`` for ``ν = -ε_perp/ε_axial`` along a
  chosen loading direction.

numpy only — no GUI, no other deps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _as_triangle(triangle: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangle, dtype=float)
    if tri.shape != (3, 2):
        raise ValueError(f"triangle must be a (3, 2) array, got {tri.shape}")
    return tri


def hinge_fraction(C: float) -> float:
    """Hinge position fraction ``t = 1/(1+C)`` (Acuna et al. step 3).
    ``C`` must be positive; ``C → 0`` puts hinges at the centroid, large
    ``C`` puts them near the corners."""
    if not (C > 0.0):
        raise ValueError(f"C must be > 0, got {C}")
    return 1.0 / (1.0 + float(C))


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(float(theta)), math.sin(float(theta))
    return np.array([[c, -s], [s, c]])


def _perp_foot(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Foot of the perpendicular from ``point`` onto the line ``a``–``b``
    (matches ``auxetic.bipartite``'s edge-point construction)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return a.copy()
    s = float(np.dot(point - a, ab)) / denom
    return a + s * ab


def actuated_corners(triangle: np.ndarray, C: float, theta: float) -> np.ndarray:
    """The three kite corners after actuating by ``theta`` (each rotated
    rigidly about its own hinge). Provided for completeness — note this
    motion is shape-independent (isotropic), see the module docstring."""
    P = _as_triangle(triangle)
    t = hinge_fraction(C)
    M = P.mean(axis=0)
    T = P + t * (M - P)
    R = _rotation(theta)
    return (P - T) @ R.T + T


def edge_midpoint_triangle(triangle: np.ndarray, C: float,
                           theta: float) -> np.ndarray:
    """The triangle of the three per-edge bond midpoints after actuation.

    For each triangle edge ``(a, b)``, the two corner kites drop
    perpendicular feet ``E_ab``/``E_ba`` onto it; each foot is carried by
    its kite's rigid rotation about that kite's hinge. The edge's
    connection point is the midpoint of the two actuated feet. Returns the
    ``(3, 2)`` triangle of those midpoints for edges (0,1), (1,2), (2,0).
    At ``theta = 0`` this is the triangle of edge bond-midpoints at rest."""
    P = _as_triangle(triangle)
    t = hinge_fraction(C)
    M = P.mean(axis=0)
    T = [P[c] + t * (M - P[c]) for c in range(3)]
    R = _rotation(theta)

    def about(p: np.ndarray, c: int) -> np.ndarray:
        return R @ (np.asarray(p, dtype=float) - T[c]) + T[c]

    mids = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        foot_ab = _perp_foot(T[a], P[a], P[b])
        foot_ba = _perp_foot(T[b], P[b], P[a])
        mids.append(0.5 * (about(foot_ab, a) + about(foot_ba, b)))
    return np.array(mids)


def edge_vector_deformation_gradient(triangle: np.ndarray, C: float,
                                     theta: float) -> np.ndarray:
    """2×2 affine deformation gradient ``A`` between the rest
    (``theta=0``) and actuated edge-midpoint triangles. Raises
    ``ValueError`` for a degenerate (collinear) triangle."""
    rest = edge_midpoint_triangle(triangle, C, 0.0)
    defo = edge_midpoint_triangle(triangle, C, theta)
    E_rest = np.column_stack([rest[1] - rest[0], rest[2] - rest[0]])
    E_def = np.column_stack([defo[1] - defo[0], defo[2] - defo[0]])
    if abs(float(np.linalg.det(E_rest))) < 1e-15:
        raise ValueError("degenerate triangle: rest edge-midpoints collinear")
    return E_def @ np.linalg.inv(E_rest)


def triangle_strain_tensor(triangle: np.ndarray, C: float,
                           theta: float) -> np.ndarray:
    """Symmetric 2×2 small-strain tensor ``ε = (A + Aᵀ)/2 - I`` of the
    edge-midpoint-triangle deformation. Rotation-free; ``ε = 0`` at
    ``theta = 0``."""
    A = edge_vector_deformation_gradient(triangle, C, theta)
    return 0.5 * (A + A.T) - np.eye(2)


def generalized_poisson_ratio(triangle: np.ndarray, C: float, theta: float,
                              *, axis: np.ndarray | None = None,
                              eps: float = 1e-9) -> float:
    """Generalized Poisson's ratio of the edge-vector deformation.

    With ``axis=None`` (default) returns the principal ratio
    ``ν = -ε₂/ε₁`` (eigenstrains ordered ``|ε₁| ≥ |ε₂|``). With ``axis``
    given (a 2-vector loading direction) returns the directional ratio
    ``ν = -ε_perp/ε_axial``.

    Returns ``nan`` when the relevant axial strain is below ``eps`` (e.g.
    at ``theta = 0``, where the deformation — and hence the ratio — is
    undefined)."""
    strain = triangle_strain_tensor(triangle, C, theta)

    if axis is not None:
        d = np.asarray(axis, dtype=float)
        nd = float(np.linalg.norm(d))
        if nd < 1e-15:
            raise ValueError("axis must be a non-zero 2-vector")
        d = d / nd
        n = np.array([-d[1], d[0]])
        eps_axial = float(d @ strain @ d)
        eps_perp = float(n @ strain @ n)
        if abs(eps_axial) < eps:
            return float("nan")
        return -eps_perp / eps_axial

    evals = np.linalg.eigvalsh(strain)
    order = np.argsort(np.abs(evals))[::-1]
    e1, e2 = float(evals[order[0]]), float(evals[order[1]])
    if abs(e1) < eps:
        return float("nan")
    return -e2 / e1


# ---------------------------------------------------------------------------
# Shape helpers (build the equilateral → isosceles → scalene axis)
# ---------------------------------------------------------------------------

def equilateral_triangle(side: float = 1.0) -> np.ndarray:
    """Equilateral triangle with the given side length, base on the x-axis."""
    return np.array([[0.0, 0.0], [side, 0.0],
                     [side / 2.0, side * math.sqrt(3.0) / 2.0]])


def apex_triangle(apex_x: float, apex_y: float) -> np.ndarray:
    """Triangle with unit base ``(0,0)-(1,0)`` and apex at
    ``(apex_x, apex_y)``. Apex ``(0.5, √3/2)`` is equilateral; ``(0.5, h)``
    is isosceles; any other apex is scalene."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [float(apex_x), float(apex_y)]])


def morph_triangle(s: float) -> np.ndarray:
    """A one-parameter family sweeping equilateral → isosceles → scalene.

    ``s = 0`` is equilateral. For ``s ∈ (0, 0.5]`` the apex stays centred
    over the base but drops in height (isosceles, non-equilateral). For
    ``s ∈ (0.5, 1]`` the apex also slides sideways (scalene). Base is the
    unit segment ``(0,0)-(1,0)``."""
    s = float(np.clip(s, 0.0, 1.0))
    h_eq = math.sqrt(3.0) / 2.0
    if s <= 0.5:
        u = s / 0.5
        height = h_eq * (1.0 - u) + (0.45 * h_eq) * u
        apex_x = 0.5
    else:
        u = (s - 0.5) / 0.5
        height = 0.45 * h_eq
        apex_x = 0.5 + 0.45 * u
    return apex_triangle(apex_x, height)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

@dataclass
class PoissonSweep:
    """Result of sweeping the generalized Poisson's ratio over a shape
    axis and a ``C`` axis at a fixed actuation angle.

    ``shape_values`` and ``C_values`` are the 1-D axes; ``ratios`` is the
    ``(len(shape_values), len(C_values))`` grid of ν, with ``nan`` where
    the metric is undefined. ``triangles`` keeps the (3, 2) triangle used
    for each shape row."""

    shape_values: np.ndarray
    C_values: np.ndarray
    ratios: np.ndarray
    theta: float
    triangles: list


def sweep_poisson(triangles: list, C_values: np.ndarray,
                  theta: float) -> np.ndarray:
    """Generalized Poisson's ratio for every (triangle, C) pair at a fixed
    ``theta``. Returns a ``(len(triangles), len(C_values))`` array."""
    C_values = np.asarray(C_values, dtype=float)
    out = np.empty((len(triangles), len(C_values)), dtype=float)
    for i, tri in enumerate(triangles):
        for j, C in enumerate(C_values):
            out[i, j] = generalized_poisson_ratio(tri, float(C), theta)
    return out


def sweep_shape_and_C(shape_values: np.ndarray, C_values: np.ndarray,
                      theta: float = 0.1) -> PoissonSweep:
    """Sweep the generalized Poisson's ratio across a shape axis
    (``morph_triangle`` from equilateral → scalene) and a ``C`` axis at a
    small fixed actuation ``theta``. Returns a :class:`PoissonSweep`."""
    shape_values = np.asarray(shape_values, dtype=float)
    C_values = np.asarray(C_values, dtype=float)
    triangles = [morph_triangle(float(s)) for s in shape_values]
    ratios = sweep_poisson(triangles, C_values, theta)
    return PoissonSweep(shape_values=shape_values, C_values=C_values,
                        ratios=ratios, theta=float(theta), triangles=triangles)
