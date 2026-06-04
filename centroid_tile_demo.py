"""Standalone centroid-tile construction demo.

Given N 2D points (N >= 3):
1. Build a Delaunay triangulation.
2. For every triangle (P0, P1, P2), with shrink c:
    C    = incenter(P0, P1, P2)                -- weighted vertex average
                                                    weights = lengths of the
                                                    opposite edges
                                                    (a = |P1-P2|, b = |P0-P2|,
                                                    c_w = |P0-P1|;
                                                    C = (a*P0 + b*P1 + c_w*P2)
                                                        / (a + b + c_w))
    v_i  = P_i - C
    T_i  = C + c * v_i                         -- inner-triangle vertex
                                                    (c=0 -> all T's at C,
                                                    c=1 -> T_i = P_i)
    foot of T on edge(P_a, P_b):
        d = P_b - P_a
        t = ((T - P_a) . d) / (d . d)
        foot = P_a + t * d
    - Fill inner triangle T0-T1-T2.
    - Six foot points (canonical order): P001, P101, P112, P212, P002, P202.

3. Each T_i owns a quadrilateral "wing" panel of four corners:
    wing(T0) = (T0, P001, P0,  P002)
    wing(T1) = (T1, P101, P1,  P112)
    wing(T2) = (T2, P212, P2,  P202)
    With the kirigami actuation angle theta, every panel pivots about the
    vertex that connects it to an inner triangle:
        - Each wing rotates rigidly by +theta about its own T_i.
        - Each link polygon (the purple panel that joins two inner
          triangles across a shared Delaunay edge) rotates rigidly by
          +theta about its T-pivot on the UPSTREAM inner triangle (the
          BFS-parent side of the dual graph from the chosen root).
        - The root inner triangle stays fixed at its rest position. Every
          other inner triangle is TRANSLATED (orientation preserved) so
          its pivot T-vertex stays attached to its upstream link. Because
          (T_n_A - T_n_B) = c * (P_A - P_B) = (T_m_A - T_m_B) for the two
          endpoints A, B of any shared edge with uniform shrink c, both
          links across the same shared edge agree on the same translation
          (the auxetic kinematic constraint).
    The shared Delaunay vertex P between adjacent wings stays a single
    point at every theta -- it's the kirigami hinge. As a result, wings,
    inner triangles, and link polygons all keep their REST SHAPES under
    the whole theta sweep; only their positions change.
    theta = 0 reproduces the original (un-opened) tile.

The math is fully vectorized over Delaunay triangles. Static geometry
(vertices, centroids, edge anchors, projection denominators) is built
once into `TilingGeometry`; each (c, theta) evaluation is a few
einsums. Matplotlib artists are persistent and updated in place.

Run:
    python centroid_tile_demo.py
"""
from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, RadioButtons
from scipy.spatial import Delaunay, QhullError


# Triangle-local edge table: each edge as (start, end) vertex index in {0, 1, 2}.
# Edge 0 = P0-P1,  edge 1 = P1-P2,  edge 2 = P0-P2.
_EDGES = np.array([[0, 1], [1, 2], [0, 2]])

# Six foot points per triangle in canonical order:
#   P001, P101, P112, P212, P002, P202.
# Each is identified by its owning T_index and the edge it lies on.
_T_FOR_FOOT    = np.array([0, 1, 1, 2, 0, 2])
_EDGE_FOR_FOOT = np.array([0, 0, 1, 1, 2, 2])
_FOOT_NAMES = ("P001", "P101", "P112", "P212", "P002", "P202")

# Wing membership per T (slot into the canonical foot array).
# Wing(T_i) = (foot_a, P_i, foot_b), traversed CCW-ish around the quad.
#   T0: (P001 [slot 0], P0, P002 [slot 4])
#   T1: (P101 [slot 1], P1, P112 [slot 2])
#   T2: (P212 [slot 3], P2, P202 [slot 5])
_WING_FOOT_A = np.array([0, 1, 3])
_WING_FOOT_B = np.array([4, 2, 5])


# ---------------------------------------------------------------------------
# Single-triangle helpers (small, allocation-free; used for tests / one-offs).
# ---------------------------------------------------------------------------
def foot_point(T: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Foot of the perpendicular from T onto the infinite line through A, B."""
    d = B - A
    return A + (np.dot(T - A, d) / np.dot(d, d)) * d


def _rotation_matrix(theta: float) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]])


def _rotate_about(point: np.ndarray, pivot: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate `point` about `pivot` by R. Broadcasts over leading axes."""
    return pivot + (point - pivot) @ R.T


def _aabb(*arrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (min, max) over a set of point arrays.

    Each input is flattened to (-1, 2) before reduction.
    """
    pts = np.concatenate([a.reshape(-1, 2) for a in arrs], axis=0)
    return pts.min(axis=0), pts.max(axis=0)


def _order_ccw(polys: np.ndarray) -> np.ndarray:
    """Reorder each polygon's vertices CCW about its centroid.

    Input/Output shape (..., K, 2). For points in convex position this
    yields the simple (non-self-intersecting) convex polygon, which is
    what we want for the link quads.
    """
    centroid = polys.mean(axis=-2, keepdims=True)
    rel = polys - centroid
    ang = np.arctan2(rel[..., 1], rel[..., 0])
    order = np.argsort(ang, axis=-1)
    return np.take_along_axis(polys, order[..., None], axis=-2)


def _quadratic_bezier(p0, p1, p2, n_samples: int = 12) -> np.ndarray:
    """Sample `n_samples` points along the quadratic Bezier p0 -> p1 -> p2.

    p1 is the control point (the curve passes through p0 and p2 and is
    tangent to p0->p1 at p0 and to p1->p2 at p2). Adapted from the STL
    fillet tool's corner-rounding routine.
    """
    t = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    p0, p1, p2 = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _find_junctions(polys, tol: float) -> list:
    """Cluster polygon corners by position; keep clusters spanning >=2 polygons.

    Bucketed by rounded coordinate (O(V)): corners that coincide exactly --
    which is the case at every kirigami joint, since the inner triangle, its
    wings, and the links all reference the *same* T / foot / shared-vertex
    arrays -- land in the same bucket. Returns a list of clusters, each a list
    of (poly_index, vertex_index, position).
    """
    inv = 1.0 / tol
    buckets: dict[tuple[int, int], list] = {}
    for pi, poly in enumerate(polys):
        for vi in range(len(poly)):
            v = np.asarray(poly[vi], dtype=float)
            key = (int(round(v[0] * inv)), int(round(v[1] * inv)))
            buckets.setdefault(key, []).append((pi, vi, v))
    return [cl for cl in buckets.values() if len({c[0] for c in cl}) >= 2]


def _build_joint_bridge(junction, polys, d: float, samples: int = 10,
                        n_inner: int = 0):
    """Bezier 'flower' that welds the pieces meeting at one junction.

    Uniform-radius fillet. Every arm leaving the joint centre backs off by the
    SAME distance

        radius = d * min(0.5 * shortest_leg, 0.25 * shortest_inner_edge)

    so the back-off points sit on a circle around the corner and the resulting
    Bezier flower is symmetric. The two bounds, whichever is smaller:

    * 0.5 * shortest_leg -- half the shortest arm of any kind at the joint, i.e.
      the MIDPOINT of the shortest leg. Keeps the flower from overrunning the
      nearest panel edge (two fillets sharing a leg meet no further than its
      midpoint).
    * 0.25 * shortest_inner_edge -- a quarter of the shortest inner-triangle
      edge incident at this corner. Inner-triangle edges are identified by their
      source polygon: the first `n_inner` entries of `polys` are the inner
      triangles, so an arm coming from one of those is a genuine inner-triangle
      edge (a T-to-T edge of the SAME inner triangle) -- as opposed to a link
      cross-arm to a NEIGHBOUR's T, which is also T-to-T but is NOT an inner
      edge. Under the uniform radius each wing-foot arm pairs with one of the
      corner's two incident inner edges, so the binding value is their shorter.

    `d` in [0, 1] scales the radius: d = 0 collapses every arm onto the centre
    (a sharp join); larger d grows the flower until it reaches the shortest
    leg's midpoint or a quarter of the shortest inner edge, whichever comes
    first.

    The back-off points are sorted by angle around the junction and consecutive
    ones are joined by a quadratic Bezier whose control point is the junction
    centre -- the arc D_i -> (centre) -> D_{i+1}. Returns (P, 2) or None.
    """
    center = np.mean([c[2] for c in junction], axis=0)
    arms = []                       # (unit_dir, leg_len, angle, is_inner_edge)
    for pi, vi, _ in junction:
        poly = np.asarray(polys[pi], dtype=float)
        n = len(poly)
        is_inner = pi < n_inner
        for ni in ((vi - 1) % n, (vi + 1) % n):
            vec = poly[ni] - center
            L = float(np.linalg.norm(vec))
            if L > 1e-9:
                arms.append(
                    (vec / L, L, float(np.arctan2(vec[1], vec[0])), is_inner)
                )
    if len(arms) < 2:
        return None
    arms.sort(key=lambda e: e[2])

    # Uniform back-off radius: the smaller of half the shortest leg and a
    # quarter of the shortest incident inner-triangle edge, scaled by d.
    half_leg = 0.5 * min(e[1] for e in arms)
    inner_lens = [e[1] for e in arms if e[3]]
    quarter_inner = 0.25 * min(inner_lens) if inner_lens else float("inf")
    radius = d * min(half_leg, quarter_inner)
    if radius < 1e-9:
        return None

    backoffs = [center + e[0] * radius for e in arms]
    pts = []
    for i in range(len(backoffs)):
        bez = _quadratic_bezier(
            backoffs[i], center, backoffs[(i + 1) % len(backoffs)], samples
        )
        pts.extend(bez[:-1])           # drop duplicated arc endpoints
    return np.asarray(pts) if len(pts) >= 3 else None


def build_joint_bridges(polys, d: float, n_inner: int = 0,
                        samples: int = 10) -> list[np.ndarray]:
    """Bezier bridge polygons welding every joint where >=2 polygons meet.

    `polys` is the full set of rendered panels. The FIRST `n_inner` of them must
    be the inner-triangle polygons (the caller concatenates inner + wings +
    links in that order); an arm sourced from one of those counts as an
    inner-triangle edge for the "quarter of the inner triangle" bound, which
    excludes link cross-arms that are also T-to-T. `d` in [0, 1] is the fillet
    fraction.

    At each joint every incident arm backs off from the corner by the same
    radius = d * min(0.5 * shortest leg, 0.25 * shortest inner-triangle edge),
    and a quadratic-Bezier arc (control point = the corner) joins consecutive
    back-off points. d = 0 -> sharp single-point join. See `_build_joint_bridge`.
    """
    if d <= 1e-6 or len(polys) < 2:
        return []
    allv = np.concatenate([np.asarray(p, dtype=float) for p in polys], axis=0)
    scale = float(np.linalg.norm(allv.max(axis=0) - allv.min(axis=0)))
    if scale < 1e-12:
        return []
    tol = scale * 1e-6
    juncs = _find_junctions(polys, tol)
    bridges = []
    for jn in juncs:
        b = _build_joint_bridge(jn, polys, d, samples, n_inner)
        if b is not None:
            bridges.append(b)
    return bridges


def _segments_cross(p1, p2, p3, p4) -> np.ndarray:
    """Vectorized proper-crossing test for segments p1p2 vs p3p4. (K,2) inputs."""
    def cz(a, b, c):  # z of (b-a) x (c-a)
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
               (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    d1, d2 = cz(p3, p4, p1), cz(p3, p4, p2)
    d3, d4 = cz(p1, p2, p3), cz(p1, p2, p4)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _quad_fixed_lead_edge(A, B, C, D) -> np.ndarray:
    """Order quad corners as [A, B, ?, ?] keeping A-B adjacent and simple.

    A, B, C, D are (K, 2). The lead edge A->B is preserved; the remaining
    pair (C, D) is placed as (C, D) unless that self-intersects, in which
    case it is swapped to (D, C).

    When the lead edge A-B itself crosses the segment C-D (C and D lie on
    opposite sides of A-B), no simple quad can keep A-B as an edge -- this
    happens only at extreme wing rotations. There we fall back to a plain
    CCW ordering so the polygon is at least simple. Returns (K, 4, 2).
    """
    swap = _segments_cross(B, C, D, A)           # connecting edges cross -> swap C/D
    order = np.where(swap[:, None, None],
                     np.stack([A, B, D, C], axis=1),
                     np.stack([A, B, C, D], axis=1))
    degenerate = _segments_cross(A, B, C, D)     # A-B crosses C-D: no A-B-edge quad
    if np.any(degenerate):
        ccw = _order_ccw(np.stack([A, B, C, D], axis=1))
        order = np.where(degenerate[:, None, None], ccw, order)
    return order


def _triangle_at(geom: "TilingGeometry", xy) -> int:
    """Index of the Delaunay triangle containing point xy, or -1 if none.

    Uses barycentric coordinates against every triangle (vectorized).
    """
    p = np.asarray(xy, dtype=float)
    A, B, C = geom.P[:, 0], geom.P[:, 1], geom.P[:, 2]
    v0, v1, v2 = B - A, C - A, p - A
    d00 = np.einsum("mi,mi->m", v0, v0)
    d01 = np.einsum("mi,mi->m", v0, v1)
    d11 = np.einsum("mi,mi->m", v1, v1)
    d20 = np.einsum("mi,mi->m", v2, v0)
    d21 = np.einsum("mi,mi->m", v2, v1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-15, 1e-15, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
    idx = np.where(inside)[0]
    return int(idx[0]) if idx.size else -1


def _incenter(P: np.ndarray) -> np.ndarray:
    """Incenter(s) of triangle(s). P: (..., 3, 2). Returns (..., 2).

    The incenter is the weighted average of vertices with weights equal to
    the lengths of the OPPOSITE edges. Equivalently, it is the intersection
    of the three angle bisectors and is equidistant from the three edges
    (that distance is the inradius). Lies strictly inside the triangle.

    Differs from the centroid (P0 + P1 + P2) / 3 for non-equilateral
    triangles; coincides with the centroid only for equilaterals.
    """
    # opp[..., i] = |P_j - P_k| for (j, k) = the two non-i vertices.
    opp = np.linalg.norm(
        P[..., [1, 0, 0], :] - P[..., [2, 2, 1], :], axis=-1
    )                                                  # (..., 3)
    return (opp[..., None] * P).sum(axis=-2) / opp.sum(axis=-1, keepdims=True)


def _anchor_points(P: np.ndarray, anchor: str = "incenter") -> np.ndarray:
    """Per-triangle anchor point the T's shrink toward. P: (..., 3, 2).

    anchor="centroid" -> arithmetic mean of vertices;
    anchor="incenter" -> incenter (equidistant from the three edges).
    """
    if anchor == "centroid":
        return P.mean(axis=-2)
    if anchor == "incenter":
        return _incenter(P)
    raise ValueError(f"anchor must be 'centroid' or 'incenter'; got {anchor!r}")


def _bfs_propagate(
    triangles: np.ndarray, neighbors: np.ndarray, T_rest: np.ndarray,
    theta: float, root: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """BFS from `root` inner triangle; return per-triangle (translation, parent).

    Each Delaunay triangle's inner triangle is TRANSLATED (no rotation) to its
    world position. The translation is determined by the rigid rotation of the
    link polygon joining it to its BFS parent: the link rotates by `theta`
    about the parent-side T-pivot, so the downstream pivot lands at

        T_n_world = T_m_world + R(theta) @ (T_n_rest - T_m_rest)
        t[n]      = T_n_world - T_n_rest
                  = t[m] + (R(theta) - I) @ (T_n_rest - T_m_rest)

    With uniform shrink factor `c`, both endpoints A, B of any shared edge
    yield the same translation -- (T_n_A - T_n_B) = c * (P_A - P_B) =
    (T_m_A - T_m_B) -- so the kinematics close up and inner triangles keep
    their orientation (the auxetic constraint). With per-triangle `c`
    overrides the equality breaks; we pick the first shared-edge endpoint
    we encounter and the structure is then only approximately rigid.

    For cross-edges in the BFS (cycles in the dual graph) we skip already-
    visited triangles; under the auxetic constraint the cycle closes
    automatically, so the BFS tree is enough.

    Returns:
        t:      (M, 2) translation of each inner triangle (rest -> world)
        parent: (M,) BFS parent of each triangle (-1 for component roots)
    """
    M = len(triangles)
    t = np.zeros((M, 2))
    parent = np.full(M, -1, dtype=int)
    visited = np.zeros(M, dtype=bool)
    R_th = _rotation_matrix(theta)
    starts = [root] + [i for i in range(M) if i != root]
    for start in starts:
        if visited[start]:
            continue
        visited[start] = True
        queue = [start]
        while queue:
            m = queue.pop(0)
            for kk in range(3):
                n = int(neighbors[m, kk])
                if n < 0 or visited[n]:
                    continue
                # The scipy edge kk in m corresponds to _SCIPY_TO_MY_EDGE[kk] in our table.
                # Pick the first endpoint as the propagation pivot (the second endpoint
                # gives the same translation under the auxetic constraint).
                my_edge = int(_SCIPY_TO_MY_EDGE[kk])
                ep_local_m = int(_EDGES[my_edge, 0])
                global_ep = int(triangles[m, ep_local_m])
                ep_local_n = int(np.where(triangles[n] == global_ep)[0][0])
                delta = T_rest[n, ep_local_n] - T_rest[m, ep_local_m]
                t[n] = t[m] + (R_th @ delta) - delta
                parent[n] = m
                visited[n] = True
                queue.append(n)
    return t, parent


def poisson_ratios(
    init_size: tuple[float, float],
    rot_size: tuple[float, float],
    strain_tol: float = 5e-4,
) -> tuple[float | None, float | None]:
    """Engineering Poisson's ratios from bounding-box dimensions.

    Given the (W, H) of the structure before and after deformation,
    returns (nu_xy, nu_yx) where
        eps_x = (W - W0) / W0,   eps_y = (H - H0) / H0
        nu_xy = -eps_y / eps_x   (axial load in x, transverse response in y)
        nu_yx = -eps_x / eps_y.

    For the auxetic case both axes shrink (or grow) together -> nu < 0.

    Each component is None when its denominator strain falls below
    `strain_tol` (default 0.05%), because the ratio is then dominated
    by numerical noise rather than real deformation.
    """
    W0, H0 = init_size
    WR, HR = rot_size
    eps_x = (WR - W0) / W0 if W0 > 1e-12 else 0.0
    eps_y = (HR - H0) / H0 if H0 > 1e-12 else 0.0
    nu_xy = -eps_y / eps_x if abs(eps_x) > strain_tol else None
    nu_yx = -eps_x / eps_y if abs(eps_y) > strain_tol else None
    return nu_xy, nu_yx


def construct_triangle_tile(
    P0: np.ndarray, P1: np.ndarray, P2: np.ndarray,
    c: float, theta: float = 0.0, anchor: str = "incenter",
) -> dict[str, np.ndarray]:
    """T's and the six foot points for one triangle (named-dict wrapper).

    `anchor` selects the point the T's shrink toward ("incenter" or
    "centroid"). For theta != 0 the inner triangle T0-T1-T2 stays fixed
    and each wing rotates rigidly about its T_i pivot by theta:
        (P001, P002)  rotates about T0
        (P101, P112)  rotates about T1
        (P212, P202)  rotates about T2
    A single isolated triangle has no link polygons, so BFS propagation is
    trivial -- this matches the multi-triangle compute_wings on a tile
    that happens to be the BFS root.
    """
    P = np.stack([P0, P1, P2])
    C = _anchor_points(P, anchor)
    T = C + c * (P - C)
    out: dict[str, np.ndarray] = {"T0": T[0], "T1": T[1], "T2": T[2]}
    for name, ti, ei in zip(_FOOT_NAMES, _T_FOR_FOOT, _EDGE_FOR_FOOT):
        a, b = _EDGES[ei]
        out[name] = foot_point(T[ti], P[a], P[b])
    if theta != 0.0:
        R = _rotation_matrix(theta)
        for name, ti in zip(_FOOT_NAMES, _T_FOR_FOOT):
            out[name] = _rotate_about(out[name], T[ti], R)
    return out


# ---------------------------------------------------------------------------
# Vectorized geometry: build once per point-set, query per (c, theta).
# ---------------------------------------------------------------------------
@dataclass
class TilingGeometry:
    """Static per-triangle geometry. Independent of (c, theta)."""
    points: np.ndarray      # (N, 2)
    triangles: np.ndarray   # (M, 3)    vertex indices into points
    P: np.ndarray           # (M, 3, 2) triangle vertices
    C: np.ndarray           # (M, 2)    incenters (per-triangle anchor point)
    v: np.ndarray           # (M, 3, 2) v_i = P_i - C
    edge_A: np.ndarray      # (M, 3, 2) edge start points
    edge_d: np.ndarray      # (M, 3, 2) edge direction vectors (B - A)
    edge_dd: np.ndarray     # (M, 3)    precomputed d . d denominators
    neighbors: np.ndarray   # (M, 3)    scipy Delaunay neighbors (-1 = boundary)

    # Precomputed indexing for shared-edge link polygons. For each shared
    # Delaunay edge there are two link polygons (one at each endpoint).
    # All arrays are length K = 2 * (number of shared edges).
    link_tri_m: np.ndarray   # (K,) triangle index, "m" side
    link_tri_n: np.ndarray   # (K,) triangle index, "n" side
    link_ti_m: np.ndarray    # (K,) local T-index in m for this endpoint
    link_ti_n: np.ndarray    # (K,) local T-index in n for this endpoint
    link_slot_m: np.ndarray  # (K,) wing slot (0 or 2) for foot in m on shared edge
    link_slot_n: np.ndarray  # (K,) wing slot (0 or 2) for foot in n on shared edge

    @classmethod
    def build(cls, points: np.ndarray, anchor: str = "incenter") -> "TilingGeometry":
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must be (N, 2); got shape {pts.shape}")
        if len(pts) < 3:
            raise ValueError(f"need at least 3 points; got {len(pts)}")
        d = Delaunay(pts)
        tri = d.simplices
        neighbors = d.neighbors
        P = pts[tri]
        C = _anchor_points(P, anchor)
        v = P - C[:, None, :]
        edge_A = P[:, _EDGES[:, 0], :]
        edge_B = P[:, _EDGES[:, 1], :]
        edge_d = edge_B - edge_A
        edge_dd = np.einsum("mei,mei->me", edge_d, edge_d)
        link_idx = _build_link_indexing(tri, neighbors)
        return cls(
            pts, tri, P, C, v, edge_A, edge_d, edge_dd, neighbors,
            *link_idx,
        )


# scipy.Delaunay's edge k is opposite vertex k. Map scipy edge index ->
# index into our local _EDGES table.
#   scipy 0 (verts 1, 2) -> _EDGES[1] (P1-P2)
#   scipy 1 (verts 0, 2) -> _EDGES[2] (P0-P2)
#   scipy 2 (verts 0, 1) -> _EDGES[0] (P0-P1)
_SCIPY_TO_MY_EDGE = np.array([1, 2, 0])

# (T_index, my_edge_index) -> wing slot index (0 = foot_a, 2 = foot_b).
# Derived from _WING_FOOT_A / _WING_FOOT_B (see _FOOT_NAMES table).
_WING_FOOT_SLOT = {
    (0, 0): 0, (0, 2): 2,
    (1, 0): 0, (1, 1): 2,
    (2, 1): 0, (2, 2): 2,
}


def _build_link_indexing(
    tri: np.ndarray, neighbors: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Walk shared Delaunay edges once, return six index arrays for compute_links.

    Each shared edge contributes two link polygons (one per endpoint).
    Returns (link_tri_m, link_tri_n, link_ti_m, link_ti_n,
            link_slot_m, link_slot_n).
    """
    tri_m, tri_n, ti_m, ti_n, slot_m, slot_n = [], [], [], [], [], []
    M = len(tri)
    for m in range(M):
        for k_scipy in range(3):
            n = int(neighbors[m, k_scipy])
            if n < 0 or n < m:
                # Boundary edge, or already processed from the other side.
                continue
            my_edge_m = int(_SCIPY_TO_MY_EDGE[k_scipy])
            ep_a_m, ep_b_m = (int(x) for x in _EDGES[my_edge_m])
            global_a = int(tri[m, ep_a_m])
            global_b = int(tri[m, ep_b_m])
            n_simp = tri[n]
            ep_a_n = int(np.where(n_simp == global_a)[0][0])
            ep_b_n = int(np.where(n_simp == global_b)[0][0])
            # Find the edge in n containing exactly these two local indices.
            my_edge_n = next(
                kk for kk in range(3)
                if set(_EDGES[kk].tolist()) == {ep_a_n, ep_b_n}
            )
            for (ep_m, ep_n) in [(ep_a_m, ep_a_n), (ep_b_m, ep_b_n)]:
                tri_m.append(m); tri_n.append(n)
                ti_m.append(ep_m); ti_n.append(ep_n)
                slot_m.append(_WING_FOOT_SLOT[(ep_m, my_edge_m)])
                slot_n.append(_WING_FOOT_SLOT[(ep_n, my_edge_n)])
    return tuple(np.array(a, dtype=int) for a in (tri_m, tri_n, ti_m, ti_n, slot_m, slot_n))


def compute_T(geom: TilingGeometry, c) -> np.ndarray:
    """Inner-triangle vertices for every Delaunay triangle. Shape (M, 3, 2).

    `c` may be a scalar (same shrink for all triangles) or a length-M array
    (per-triangle shrink). T's depend only on c; theta does NOT move them.
    """
    c_arr = np.asarray(c, dtype=float)
    if c_arr.ndim == 0:
        return geom.C[:, None, :] + c_arr * geom.v
    return geom.C[:, None, :] + c_arr[:, None, None] * geom.v


def compute_feet(geom: TilingGeometry, T: np.ndarray) -> np.ndarray:
    """Un-rotated foot points (on the outer edges). Shape (M, 6, 2).

    These are the "rest" foot points — perpendicular projections of each
    T onto its two outer edges. The wing rotation is applied separately.
    """
    T_sel  = T[:, _T_FOR_FOOT, :]
    A_sel  = geom.edge_A[:, _EDGE_FOR_FOOT, :]
    d_sel  = geom.edge_d[:, _EDGE_FOR_FOOT, :]
    dd_sel = geom.edge_dd[:, _EDGE_FOR_FOOT]
    t = np.einsum("mki,mki->mk", T_sel - A_sel, d_sel) / dd_sel
    return A_sel + t[..., None] * d_sel


def compute_wings(
    geom: TilingGeometry, c: float, theta: float = 0.0, root: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner triangles (translated) and wings (rotated about T_i) for rigid kirigami.

    Every wing rotates by +theta about its OWN T_i (the vertex of its inner
    triangle that anchors it). Inner triangles translate but do not rotate;
    `_bfs_propagate` walks the dual graph from `root` and computes each
    triangle's translation from the rigid rotation of its parent link. The
    root inner triangle stays fixed.

    The shared Delaunay vertex P between two wings (one in tile m, one in
    tile n) stays a single point at every theta -- the wing rotations and
    the link rotation agree on its world position by construction.

    Returns:
        T:     (M, 3, 2)        -- inner-triangle vertices in world space
                                   (root tile at rest; others translated).
        wings: (M, 3, 3, 2)     -- (triangle, T_index, point_in_wing, xy)
                                   point_in_wing order: [foot_a, P_i, foot_b].
                                   Together with T_i these form the quad
                                   (T_i, foot_a, P_i, foot_b) traversed in order.
    """
    T_rest = compute_T(geom, c)
    feet = compute_feet(geom, T_rest)
    foot_a = feet[:, _WING_FOOT_A, :]                       # (M, 3, 2)
    foot_b = feet[:, _WING_FOOT_B, :]                       # (M, 3, 2)
    wings_rest = np.stack([foot_a, geom.P, foot_b], axis=2) # (M, 3, 3, 2)
    if theta == 0.0:
        return T_rest, wings_rest
    t, _ = _bfs_propagate(geom.triangles, geom.neighbors, T_rest, theta, root)
    T_world = T_rest + t[:, None, :]                        # (M, 3, 2)
    R = _rotation_matrix(theta)
    pivot_rest  = T_rest[:, :, None, :]                     # (M, 3, 1, 2)
    pivot_world = T_world[:, :, None, :]                    # (M, 3, 1, 2)
    return T_world, pivot_world + (wings_rest - pivot_rest) @ R.T


def tile_state(
    geom: TilingGeometry, c: float, theta: float = 0.0, root: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot (T, wings) evaluation. See compute_wings for shapes."""
    return compute_wings(geom, c, theta, root)


def _feet_rest_index(T_local: int, edge_local: int) -> int | None:
    """Index into compute_feet's (M, 6, 2) of the foot at (T_local, edge_local)."""
    for fi in range(6):
        if int(_T_FOR_FOOT[fi]) == T_local and int(_EDGE_FOR_FOOT[fi]) == edge_local:
            return fi
    return None


def _other_adjacent_edge(T_local: int, this_edge: int) -> int:
    """The edge (other than `this_edge`) of the local triangle that contains T_local."""
    for e in range(3):
        if e == this_edge:
            continue
        if T_local in _EDGES[e].tolist():
            return e
    raise ValueError(f"no other edge contains T={T_local} besides {this_edge}")


def compute_links(
    geom: TilingGeometry, c: float, theta: float = 0.0, root: int = 0,
) -> list[np.ndarray]:
    """Rigid joint polygons at every Delaunay vertex shared by K >= 2 inner triangles.

    For each shared Delaunay vertex v, the K incident inner triangles are
    ordered angularly around v and ONE polygon is built that walks the joint
    perimeter:

    - K == 2 (two triangles share a single Delaunay edge meeting at v):
      6 corners [T_m, F_m_non_shared, P_m, P_n, F_n_non_shared, T_n].
      F_m / F_n are the non-shared-edge feet (the wing's "other" foot at v);
      P_m and P_n are the shared vertex v, coincident in both wings at all
      theta (the kirigami hinge -- one zero-length edge).

    - K >= 3 INTERIOR (closed cycle around v, no boundary edges):
      K corners [T_0, T_1, ..., T_{K-1}] -- just the inner-triangle T's
      connected directly in angular order around v. Triangle (K=3),
      quadrilateral (K=4), pentagon (K=5), hexagon (K=6), etc. No legs,
      no central P -- this is the convex K-gon spanning the joint T's.

    - K >= 3 BOUNDARY (v lies on the convex hull, fan with open ends):
      K + 3 corners [T_0, T_1, ..., T_{K-1}, outer_last, P, outer_first].
      K T's + 2 outer-edge feet (one for each fan-end wing) + 1 central P.
      Consecutive T's are connected directly; the polygon then descends to
      the convex-hull vertex P along the CCW-side boundary edge and rises
      back along the CW-side boundary edge. Hexagon for K = 3 (3 reds +
      2 oranges + 1 yellow in the annotated UI), heptagon for K = 4, etc.

    Each polygon is RIGID under the auxetic constraint: every corner is
    obtained by rotating its rest position about the first incident T
    (the chosen pivot) by `theta` and translating by that pivot's
    BFS-propagated displacement. Wing feet rigid with wing 0 (the pivot)
    follow directly; wing feet rigid with wing k (k > 0) match by the
    auxetic identity t_k = (R(theta) - I)(T_k_rest - T_0_rest), which
    folds the wing-k rotation into the same single rigid transform.

    Returns a LIST of (P, 2) arrays -- one polygon per joint vertex. The
    polygon size P varies with K (6 for K = 2, 2K for K >= 3).
    """
    from collections import defaultdict

    T_rest = compute_T(geom, c)
    feet_rest = compute_feet(geom, T_rest)

    # Vertex -> [(triangle_index, local_T_index_in_that_triangle), ...]
    incidence: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for m, tri in enumerate(geom.triangles):
        for li in range(3):
            incidence[int(tri[li])].append((m, li))

    if theta == 0.0:
        t = np.zeros((len(geom.triangles), 2))
    else:
        t, _ = _bfs_propagate(
            geom.triangles, geom.neighbors, T_rest, theta, root,
        )
    T_world = T_rest + t[:, None, :]
    R = _rotation_matrix(theta) if theta != 0.0 else np.eye(2)

    polygons: list[np.ndarray] = []
    for v, incs in incidence.items():
        K = len(incs)
        if K < 2:
            continue
        v_pos = np.asarray(geom.points[v], dtype=float)

        # Order incidences angularly around v (CCW at rest).
        angles = np.array([
            np.arctan2(
                T_rest[m, li, 1] - v_pos[1],
                T_rest[m, li, 0] - v_pos[0],
            )
            for (m, li) in incs
        ])
        incs_ord = [incs[i] for i in np.argsort(angles)]

        # Is `v` a boundary (convex-hull) vertex? -- True iff any incident
        # triangle has a Delaunay edge at `v` with no neighbour across it.
        is_boundary_v = any(
            int(geom.neighbors[m, k]) == -1
            for (m, li) in incs
            for k in range(3)
            if k != li
        )

        if K == 2:
            # Existing 6-corner formula: 2 T's + 2 outer-edge feet + P twice
            # (kirigami hinge). At a K=2 vertex the two adjacent edges of each
            # wing are exactly one shared edge + one boundary edge, so the
            # "non-shared" foot IS the outer-edge foot.
            (m_, li_m), (n_, li_n) = incs_ord
            kk_m = next(
                (kk for kk in range(3) if int(geom.neighbors[m_, kk]) == n_),
                None,
            )
            kk_n = next(
                (kk for kk in range(3) if int(geom.neighbors[n_, kk]) == m_),
                None,
            )
            if kk_m is None or kk_n is None:
                # Triangles share v but not an edge (rare, e.g. fan boundary).
                continue
            my_edge_m = int(_SCIPY_TO_MY_EDGE[kk_m])
            my_edge_n = int(_SCIPY_TO_MY_EDGE[kk_n])
            ns_edge_m = _other_adjacent_edge(li_m, my_edge_m)
            ns_edge_n = _other_adjacent_edge(li_n, my_edge_n)
            fi_m = _feet_rest_index(li_m, ns_edge_m)
            fi_n = _feet_rest_index(li_n, ns_edge_n)
            corners_rest = np.stack([
                T_rest[m_, li_m],
                feet_rest[m_, fi_m],
                v_pos,
                v_pos,
                feet_rest[n_, fi_n],
                T_rest[n_, li_n],
            ])
        elif is_boundary_v:
            # K >= 3 BOUNDARY fan: K T's + 2 outer-edge feet (one at each fan
            # end) + 1 central P  ==  K + 3 corners.
            #
            # [T_0, T_1, ..., T_{K-1}, outer_foot_last, P, outer_foot_first]
            #
            # Consecutive T's are connected DIRECTLY along the inner triangle
            # edges of their adjoining wings (no shared-edge leg corner). The
            # polygon then "drops down" along the CCW-side boundary edge to
            # the convex-hull vertex P, and back up along the CW-side boundary
            # edge to T_0 -- matching the maroon-outline hexagon (3 red T's,
            # 2 orange outer-edge feet, 1 yellow P).
            m_first, li_first = incs_ord[0]
            m_last, li_last = incs_ord[K - 1]
            bk_first = next(
                (k for k in range(3)
                 if k != li_first and int(geom.neighbors[m_first, k]) == -1),
                None,
            )
            bk_last = next(
                (k for k in range(3)
                 if k != li_last and int(geom.neighbors[m_last, k]) == -1),
                None,
            )
            if bk_first is None or bk_last is None:
                # Could not identify both fan-end boundary edges; skip.
                continue
            my_edge_first = int(_SCIPY_TO_MY_EDGE[bk_first])
            my_edge_last = int(_SCIPY_TO_MY_EDGE[bk_last])
            fi_first = _feet_rest_index(li_first, my_edge_first)
            fi_last = _feet_rest_index(li_last, my_edge_last)
            if fi_first is None or fi_last is None:
                continue
            outer_foot_first = feet_rest[m_first, fi_first]
            outer_foot_last = feet_rest[m_last, fi_last]
            corners_list = [T_rest[m, li] for (m, li) in incs_ord]
            corners_list.append(outer_foot_last)
            corners_list.append(v_pos)
            corners_list.append(outer_foot_first)
            corners_rest = np.stack(corners_list)
        else:
            # K >= 3 INTERIOR cyclic: just connect the K inner-triangle T's
            # directly in angular order around v. K corners total -- a
            # triangle (K=3), quadrilateral (K=4), pentagon (K=5), hexagon
            # (K=6), etc. No central P, no shared-edge legs -- the polygon
            # is the convex K-gon spanning the joint T's, matching the
            # yellow-outlined K-gon in the annotated UI.
            corners_list = [T_rest[m, li] for (m, li) in incs_ord]
            corners_rest = np.stack(corners_list)

        # Rigid rotation about the first incident T (auxetic constraint
        # guarantees all corners ride along consistently, regardless of
        # which wing they're individually rigid with).
        m_root, li_root = incs_ord[0]
        pivot_rest = T_rest[m_root, li_root]
        pivot_world = T_world[m_root, li_root]
        rel = corners_rest - pivot_rest
        polygons.append(pivot_world + rel @ R.T)

    return polygons


def junction_wing_mask(geom: TilingGeometry) -> np.ndarray:
    """Boolean (M, 3): True where a wing sits at a shared-edge endpoint.

    These "junction wings" are absorbed into the link hexagons, so the viewer
    skips drawing them. Wings at boundary corners (no shared edge) stay False.
    """
    mask = np.zeros((len(geom.triangles), 3), dtype=bool)
    if len(geom.link_tri_m):
        mask[geom.link_tri_m, geom.link_ti_m] = True
        mask[geom.link_tri_n, geom.link_ti_n] = True
    return mask


# ---------------------------------------------------------------------------
# STL export: extrude the flat panels into a thin slab and write binary STL.
# ---------------------------------------------------------------------------
def _signed_area(poly: np.ndarray) -> float:
    """Signed area of a 2D polygon (CCW positive). poly: (N, 2)."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clean_polygon(poly: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop consecutive duplicate vertices (incl. wrap-around). (N, 2) -> (M, 2).

    The K=2 link polygons repeat the shared Delaunay vertex (the kirigami hinge)
    as a zero-length edge; this collapses such repeats so the cap triangulator
    sees a clean simple polygon.
    """
    pts = np.asarray(poly, dtype=float)
    if len(pts) == 0:
        return pts
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tol:
        keep.pop()
    return np.asarray(keep)


def _point_in_triangle(p, a, b, c) -> bool:
    """True if p lies inside triangle abc (2D, boundary excluded)."""
    def side(u, v, w):
        return (u[0] - w[0]) * (v[1] - w[1]) - (v[0] - w[0]) * (u[1] - w[1])
    b1, b2, b3 = side(p, a, b) < 0.0, side(p, b, c) < 0.0, side(p, c, a) < 0.0
    return (b1 == b2) and (b2 == b3)


def _triangulate_polygon(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon -> vertex-index triples.

    Robust to non-convex (but not self-intersecting) polygons such as the link
    panels and the Bezier joint 'flowers'. `poly` should already be CCW and
    de-duplicated; the returned triples index into `poly`.
    """
    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n))

    def is_convex(i0, i1, i2):
        a, b, c = poly[i0], poly[i1], poly[i2]
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0

    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        m = len(idx)
        for ii in range(m):
            i0, i1, i2 = idx[(ii - 1) % m], idx[ii], idx[(ii + 1) % m]
            if not is_convex(i0, i1, i2):
                continue
            a, b, c = poly[i0], poly[i1], poly[i2]
            if any(jj not in (i0, i1, i2)
                   and _point_in_triangle(poly[jj], a, b, c) for jj in idx):
                continue
            tris.append((i0, i1, i2))
            del idx[ii]
            break
        else:
            break                               # no ear found (degenerate); stop
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def _polygon_prism(poly2d: np.ndarray, z0: float, z1: float):
    """Extrude a 2D polygon to a prism. Returns (T, 3, 3) triangles or None.

    Builds the top cap (+z), bottom cap (-z) and the side walls, wound so the
    outward normals point away from the solid.
    """
    poly = _clean_polygon(poly2d)
    if len(poly) < 3:
        return None
    if _signed_area(poly) < 0.0:
        poly = poly[::-1]                        # canonical CCW
    n = len(poly)
    caps = _triangulate_polygon(poly)
    if not caps:
        return None
    z_lo, z_hi = (z0, z1) if z1 >= z0 else (z1, z0)
    bottom = np.column_stack([poly, np.full(n, z_lo)])
    top = np.column_stack([poly, np.full(n, z_hi)])
    tris = []
    for (i, j, k) in caps:
        tris.append([top[i], top[j], top[k]])           # +z cap (CCW from above)
        tris.append([bottom[i], bottom[k], bottom[j]])  # -z cap (reversed)
    for ii in range(n):
        a, b = ii, (ii + 1) % n
        tris.append([bottom[a], bottom[b], top[b]])      # outward side wall
        tris.append([bottom[a], top[b], top[a]])
    return np.asarray(tris, dtype=float)


def _write_stl_binary(path: str, tris: np.ndarray) -> None:
    """Write triangles (T, 3, 3) to a binary STL file with computed normals."""
    tris = np.asarray(tris, dtype=float)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    nrm = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 1e-12)
    rec = np.zeros(len(tris), dtype=np.dtype([
        ("normal", "<f4", (3,)),
        ("v", "<f4", (3, 3)),
        ("attr", "<u2"),
    ]))
    rec["normal"] = nrm
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)                    # 80-byte header
        f.write(struct.pack("<I", len(tris)))    # triangle count
        f.write(rec.tobytes())


def export_stl(polygons, path: str, thickness: float) -> int:
    """Extrude each 2D polygon by `thickness` (centred on z=0) and write binary STL.

    `polygons` is an iterable of (N, 2) arrays (the rendered panels + joint
    bridges). Overlapping prisms are written as a triangle 'soup' -- valid for
    slicers that union solids on import. Returns the number of triangles written.
    """
    half = 0.5 * float(thickness)
    chunks = []
    for poly in polygons:
        prism = _polygon_prism(np.asarray(poly, dtype=float).reshape(-1, 2), -half, half)
        if prism is not None:
            chunks.append(prism)
    if not chunks:
        return 0
    tris = np.concatenate(chunks, axis=0)
    _write_stl_binary(path, tris)
    return len(tris)


# ---------------------------------------------------------------------------
# Interactive viewer.
# ---------------------------------------------------------------------------
def show(
    points: np.ndarray,
    c_init: float = 0.5,
    theta_init: float = 0.0,
) -> None:
    """Open an interactive matplotlib window.

    Sliders:
    * c      -- global shrink factor.
    * theta  -- kirigami actuation angle, in degrees. Every wing rotates
                by +theta about its T_i pivot; every link polygon rotates
                rigidly by +theta about its pivot on the upstream inner
                triangle (BFS parent side). The root inner triangle stays
                fixed; other inner triangles translate (orientation
                preserved) so the rigid links stay attached.
    * spin   -- rigid rotation of the WHOLE structure in world space
                (about the centroid of the points), in degrees.
    * fillet -- joint smoothing fraction d in [0, 1]. At each joint (a T
                corner where >=2 pieces meet) the incident edges are rounded by
                a quadratic-Bezier bridge whose control point is the corner.
                Every arm backs off from the corner by the SAME radius
                d * min(1/2 * shortest leg, 1/4 * shortest inner-triangle edge),
                whichever bound is smaller -- so the flower is symmetric and
                never overruns the shortest leg's midpoint or a quarter of the
                inner triangle. d = 0 -> sharp (panels meet at the single joint
                point).

    Radio buttons:
    * anchor -- point the T's shrink toward: "incenter" or "centroid".

    Mouse (hold a modifier and click on the main plot):
    * Shift-click a triangle -> select it; the c slider then edits only
        that triangle's c, AND that triangle becomes the BFS root that
        stays fixed during the theta sweep. Shift-click empty space to
        deselect (root falls back to triangle 0).
    * Ctrl-click-drag a point -> move it; re-triangulates live.
    * Alt-click -> add a new point at the cursor.

    Keys:
    * c -> print the current spun coordinates (points, T's, wings, links)
            to the console.
    * e -> export the current (spun, actuated, filleted) structure to a binary
            STL file in the working directory. Every rendered panel -- inner
            triangles, wings, links and the Bezier joint bridges -- is extruded
            into a thin slab (thickness 5% of the bounding-box diagonal); the
            file name is timestamped so repeated exports don't clobber.

    Re-triangulating (moving or adding a point) resets any per-triangle c
    overrides, because triangle indices are not stable across triangulations.
    Clicks are interpreted in the spun view and mapped back so the stored
    point data stays in the un-spun frame; spin is a view + export transform.
    """
    # 'c' is a default matplotlib nav shortcut (back); free it for our handler.
    if "c" in plt.rcParams["keymap.back"]:
        plt.rcParams["keymap.back"].remove("c")

    state = {
        "geom": TilingGeometry.build(points, anchor="incenter"),
        "pts": np.asarray(points, dtype=float).copy(),
        "anchor": "incenter",
        "c_global": float(c_init),
        "c_over": {},        # triangle index -> c override
        "sel_tri": None,     # selected triangle (c editing target)
        "drag": None,        # index of point currently being dragged
        "syncing": False,    # guard against slider set_val feedback
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    plt.subplots_adjust(bottom=0.26, left=0.16)

    outer_lc = LineCollection(
        [], colors="lightgray", linewidths=0.7, linestyles="--", zorder=1,
    )
    sel_lc = LineCollection([], colors="#ff7f0e", linewidths=2.4, zorder=2)
    ax.add_collection(outer_lc)
    ax.add_collection(sel_lc)

    wing_pc = PolyCollection(
        [], facecolors="#d4eddb", edgecolors="#2ca02c",
        linewidths=0.9, alpha=0.85, zorder=3,
    )
    link_pc = PolyCollection(
        [], facecolors="#c5b0e0", edgecolors="#5e3580",
        linewidths=1.0, alpha=0.85, zorder=3,
    )
    inner_pc = PolyCollection(
        [], facecolors="#bcd7f0", edgecolors="#1f77b4",
        linewidths=1.2, zorder=4,
    )
    # Bezier "bridge" panels that weld the joints (where >=2 pieces meet at a
    # point). Drawn on top so the smoothed joints are visible.
    joint_pc = PolyCollection(
        [], facecolors="#e9c46a", edgecolors="#b8860b",
        linewidths=0.8, alpha=0.85, zorder=5,
    )
    init_bbox = Rectangle(
        (0, 0), 0, 0, fill=False,
        edgecolor="#888888", linewidth=1.0, linestyle="--", zorder=7,
    )
    rot_bbox = Rectangle(
        (0, 0), 0, 0, fill=False,
        edgecolor="#d62728", linewidth=1.3, linestyle="-", zorder=8,
    )
    ax.add_patch(init_bbox)
    ax.add_patch(rot_bbox)
    ax.add_collection(wing_pc)
    ax.add_collection(link_pc)
    ax.add_collection(inner_pc)
    ax.add_collection(joint_pc)
    pts_scatter = ax.scatter([], [], s=25, c="black", zorder=6)
    title = ax.set_title("")

    # Generous fixed axis limits (extra room so spin doesn't clip the corners).
    span = np.ptp(state["pts"], axis=0)
    pad = 0.55 * max(float(span.max()), 1.0)
    lo = state["pts"].min(axis=0)
    hi = state["pts"].max(axis=0)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)

    def effective_c() -> np.ndarray:
        geom = state["geom"]
        arr = np.full(len(geom.triangles), state["c_global"], dtype=float)
        for i, cv in state["c_over"].items():
            if 0 <= i < arr.size:
                arr[i] = cv
        return arr

    def current_root() -> int:
        """Selected triangle becomes the BFS root; default 0."""
        sel = state["sel_tri"]
        M = len(state["geom"].triangles)
        if sel is None or not (0 <= sel < M):
            return 0
        return int(sel)

    def spin_apply(arr):
        """Rigidly rotate coords about the structure centre by the spin angle."""
        a = np.asarray(arr, dtype=float)
        phi = np.radians(s_spin.val)
        if phi == 0.0 or a.size == 0:
            return a
        center = state["pts"].mean(axis=0)
        return center + (a - center) @ _rotation_matrix(phi).T

    def unspin(x, y):
        """Map a spun-view click back to un-spun data coordinates."""
        p = np.array([x, y], dtype=float)
        phi = np.radians(s_spin.val)
        if phi == 0.0:
            return p
        center = state["pts"].mean(axis=0)
        return center + (p - center) @ _rotation_matrix(-phi).T

    def refresh_static() -> None:
        geom = state["geom"]
        A = geom.P[:, _EDGES[:, 0], :]
        B = geom.P[:, _EDGES[:, 1], :]
        outer_lc.set_segments(spin_apply(np.stack([A, B], axis=2).reshape(-1, 2, 2)))
        pts_scatter.set_offsets(spin_apply(state["pts"]))
        sel = state["sel_tri"]
        if sel is None or sel >= len(geom.triangles):
            sel_lc.set_segments([])
        else:
            P = spin_apply(geom.P[sel])
            sel_lc.set_segments([P[[0, 1]], P[[1, 2]], P[[2, 0]]])

    def redraw() -> None:
        geom = state["geom"]
        theta = np.radians(s_th.val)
        c = effective_c()
        root = current_root()
        T, wings = tile_state(geom, c, theta, root)
        T0, w0 = (T, wings) if theta == 0.0 else tile_state(geom, c, 0.0, root)

        Ts, ws = spin_apply(T), spin_apply(wings)
        T0s, w0s = spin_apply(T0), spin_apply(w0)
        # compute_links now returns a list of per-vertex joint polygons with
        # variable corner counts (6 for K=2 vertices, 2K for K>=3 vertices),
        # so apply spin per-polygon.
        link_list = [spin_apply(p) for p in compute_links(geom, c, theta, root)]

        # Sharp panels, then Bezier bridges welding the joints between them.
        # Junction wings are absorbed into the joint polygons, so skip drawing them.
        d_fillet = s_fillet.val
        quads = np.concatenate([Ts[:, :, None, :], ws], axis=2).reshape(-1, 4, 2)
        keep_wing = ~junction_wing_mask(geom).reshape(-1)
        inner_list = list(Ts)
        wing_list = [q for q, k in zip(quads, keep_wing) if k]
        inner_pc.set_verts(inner_list)
        wing_pc.set_verts(wing_list)
        link_pc.set_verts(link_list)
        joint_pc.set_verts(build_joint_bridges(
            inner_list + wing_list + link_list, d_fillet, len(inner_list)))

        mn0, mx0 = _aabb(T0s, w0s)
        mnR, mxR = _aabb(Ts, ws)
        w0w, w0h = mx0 - mn0
        wRw, wRh = mxR - mnR
        init_bbox.set_bounds(mn0[0], mn0[1], w0w, w0h)
        rot_bbox.set_bounds(mnR[0], mnR[1], wRw, wRh)
        dw = 100.0 * (wRw / w0w - 1.0) if w0w > 1e-12 else 0.0
        dh = 100.0 * (wRh / w0h - 1.0) if w0h > 1e-12 else 0.0
        ratio = f" = {dw / dh:+.1f}" if abs(dh) > 1e-9 else ""

        sel = state["sel_tri"]
        if sel is None:
            ctxt = f"c = {state['c_global']:.3f}"
            if state["c_over"]:
                ctxt += f" (+{len(state['c_over'])} overrides)"
        else:
            ctxt = f"c[tri {sel}] = {effective_c()[sel]:.3f}"
        title.set_text(
            f"{ctxt},  theta = {np.degrees(theta):5.1f},  spin = {s_spin.val:5.1f} deg,"
        )
        refresh_static()
        fig.canvas.draw_idle()

    def rebuild(new_pts: np.ndarray) -> bool:
        """Re-triangulate from new_pts. Returns False if degenerate (no change)."""
        try:
            geom = TilingGeometry.build(new_pts, anchor=state["anchor"])
        except (QhullError, ValueError):
            return False
        state["pts"] = np.asarray(new_pts, dtype=float)
        state["geom"] = geom
        state["c_over"] = {}
        state["sel_tri"] = None
        redraw()
        return True

    # ---- sliders -----------------------------------------------------------
    ax_c      = plt.axes([0.25, 0.175, 0.6, 0.025])
    ax_th     = plt.axes([0.25, 0.130, 0.6, 0.025])
    ax_spin   = plt.axes([0.25, 0.085, 0.6, 0.025])
    ax_fillet = plt.axes([0.25, 0.040, 0.6, 0.025])
    s_c      = Slider(ax_c,      "c",           0.0,    1.0,   valinit=c_init)
    s_th     = Slider(ax_th,     "theta (deg)", -180.0, 180.0, valinit=np.degrees(theta_init))
    s_spin   = Slider(ax_spin,   "spin (deg)",  -180.0, 180.0, valinit=0.0)
    s_fillet = Slider(ax_fillet, "fillet (d)",  0.0,    1.0,   valinit=0.0)

    # ---- anchor radio ------------------------------------------------------
    ax_radio = plt.axes([0.005, 0.80, 0.135, 0.13])
    ax_radio.set_title("anchor", fontsize=9)
    radio = RadioButtons(ax_radio, ("incenter", "centroid"), active=0)

    def on_anchor(label: str) -> None:
        state["anchor"] = label
        # Same points -> same triangulation -> per-triangle c stays valid.
        state["geom"] = TilingGeometry.build(state["pts"], anchor=label)
        redraw()
    radio.on_clicked(on_anchor)

    def on_c(val: float) -> None:
        if state["syncing"]:
            return
        if state["sel_tri"] is not None:
            state["c_over"][state["sel_tri"]] = val
        else:
            state["c_global"] = val
        redraw()
    s_c.on_changed(on_c)
    s_th.on_changed(lambda _: redraw())
    s_spin.on_changed(lambda _: redraw())
    s_fillet.on_changed(lambda _: redraw())

    # ---- mouse interaction -------------------------------------------------
    def _mods(event):
        k = event.key or ""
        return ("shift" in k, ("control" in k or "ctrl" in k), "alt" in k)

    def on_press(event) -> None:
        if event.inaxes is not ax or event.xdata is None:
            return
        shift, ctrl, alt = _mods(event)
        if ctrl:
            px = ax.transData.transform(spin_apply(state["pts"]))
            d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
            i = int(np.argmin(d))
            if d[i] <= 15.0:
                state["drag"] = i
        elif alt:
            rebuild(np.vstack([state["pts"], unspin(event.xdata, event.ydata)]))
        elif shift:
            ti = _triangle_at(state["geom"], unspin(event.xdata, event.ydata))
            state["sel_tri"] = ti if ti >= 0 else None
            if state["sel_tri"] is not None:
                state["syncing"] = True
                s_c.set_val(effective_c()[state["sel_tri"]])
                state["syncing"] = False
            redraw()

    def on_motion(event) -> None:
        if state["drag"] is None or event.inaxes is not ax or event.xdata is None:
            return
        pts = state["pts"].copy()
        pts[state["drag"]] = unspin(event.xdata, event.ydata)
        drag = state["drag"]
        if rebuild(pts):            # rebuild resets drag/sel; restore drag
            state["drag"] = drag

    def on_release(_event) -> None:
        state["drag"] = None

    def _export_stl_current() -> None:
        """Extrude the current (spun, actuated, filleted) panels to a binary STL."""
        geom = state["geom"]
        theta = np.radians(s_th.val)
        c = effective_c()
        root = current_root()
        T, wings = tile_state(geom, c, theta, root)
        Ts, ws = spin_apply(T), spin_apply(wings)
        quads = np.concatenate([Ts[:, :, None, :], ws], axis=2).reshape(-1, 4, 2)
        keep_wing = ~junction_wing_mask(geom).reshape(-1)
        inner_list = list(Ts)
        wing_list = [q for q, k in zip(quads, keep_wing) if k]
        link_list = [spin_apply(p) for p in compute_links(geom, c, theta, root)]
        bridges = build_joint_bridges(
            inner_list + wing_list + link_list, s_fillet.val, len(inner_list))
        polys = inner_list + wing_list + link_list + bridges
        # Slab thickness: 5% of the in-plane bounding-box diagonal.
        allv = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in polys], axis=0)
        diag = float(np.linalg.norm(allv.max(0) - allv.min(0)))
        thickness = 0.05 * diag if diag > 1e-9 else 1.0
        path = os.path.abspath(time.strftime("centroid_tile_%Y%m%d_%H%M%S.stl"))
        ntri = export_stl(polys, path, thickness)
        print(f"[STL] {ntri} triangles, thickness {thickness:.4f}  ->  {path}")

    def on_key(event) -> None:
        if event.key == "e":
            _export_stl_current()
            return
        if event.key != "c":
            return
        geom = state["geom"]
        theta = np.radians(s_th.val)
        c = effective_c()
        root = current_root()
        T, wings = tile_state(geom, c, theta, root)
        links_s = [spin_apply(p) for p in compute_links(geom, c, theta, root)]
        pts_s, T_s = spin_apply(state["pts"]), spin_apply(T)
        wings_s = spin_apply(wings)
        print("=" * 64)
        print(f"spin={s_spin.val:.1f} deg  theta={np.degrees(theta):.1f} deg  "
            f"anchor={state['anchor']}  c_global={state['c_global']:.3f}  root={root}")
        if state["c_over"]:
            print("  c overrides:", {k: round(v, 3) for k, v in state["c_over"].items()})
        print(f"\npoints ({len(pts_s)}):")
        for i, p in enumerate(pts_s):
            print(f"  P{i}: [{p[0]:.4f}, {p[1]:.4f}]")
        print(f"\ninner triangles T ({len(T_s)}):")
        for i, t in enumerate(T_s):
            print(f"  tri{i}: {np.round(t, 4).tolist()}")
        print(f"\nwings ({len(wings_s) * 3} quads) [T_i, foot_a, P_i, foot_b]:")
        for i in range(len(wings_s)):
            for j in range(3):
                quad = np.vstack([T_s[i, j], wings_s[i, j]])
                print(f"  tri{i} wing{j}: {np.round(quad, 4).tolist()}")
        print(f"\nlinks ({len(links_s)} joint polygons):")
        for i, h in enumerate(links_s):
            print(f"  link{i} ({len(h)} corners): {np.round(h, 4).tolist()}")
        print("=" * 64)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Controls: shift-click triangle = per-tri c | ctrl-drag = move point"
        " | alt-click = add point | fillet slider = smooth joints"
        " | press 'c' = dump spun coords | press 'e' = export STL")
    refresh_static()
    redraw()
    plt.show()


if __name__ == "__main__":
    example_points = np.array([
        [0.0, 0.0],
        [1.0, 1.73],
        [2.0, 0.0]
    ])
    show(example_points, c_init=0.5)
