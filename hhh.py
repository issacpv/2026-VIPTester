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
    With the kirigami joint angle theta, the THREE non-T corners of each
    wing rotate rigidly about T_i by theta:
        (P001, P0, P002)  rotates about T0
        (P101, P1, P112)  rotates about T1
        (P212, P2, P202)  rotates about T2
    T_i and the inner triangle T0-T1-T2 are fixed by theta; only the
    wing panels move. theta=0 reproduces the original (un-opened) tile.

The math is fully vectorized over Delaunay triangles. Static geometry
(vertices, centroids, edge anchors, projection denominators) is built
once into `TilingGeometry`; each (c, theta) evaluation is a few
einsums. Matplotlib artists are persistent and updated in place.

Run:
    python centroid_tile_demo.py
"""
from __future__ import annotations

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


def _build_joint_bridge(junction, polys, frac: float, samples: int = 10):
    """Bezier 'flower' that welds the pieces meeting at one junction.

    Reproduces the STL fillet tool's bridge construction: collect every edge
    of every participating polygon that ends at the junction, back off along
    it by a radius, sort the back-off points by angle around the junction, and
    join consecutive ones with a quadratic Bezier whose control point is the
    junction centre. The control point pulls each arc toward the joint, so the
    polygon sits like a fillet over the meeting corners. Returns (P, 2) or None.
    """
    center = np.mean([c[2] for c in junction], axis=0)
    edges = []
    for pi, vi, _ in junction:
        poly = np.asarray(polys[pi], dtype=float)
        n = len(poly)
        for ni in ((vi - 1) % n, (vi + 1) % n):
            d = poly[ni] - center
            L = float(np.linalg.norm(d))
            if L > 1e-9:
                edges.append((d / L, L, float(np.arctan2(d[1], d[0]))))
    if len(edges) < 2:
        return None
    edges.sort(key=lambda e: e[2])
    radius = frac * 0.45 * float(np.median([e[1] for e in edges]))
    if radius < 1e-9:
        return None
    backoffs = [center + e[0] * min(radius, e[1] * 0.45) for e in edges]
    pts = []
    for i in range(len(backoffs)):
        bez = _quadratic_bezier(
            backoffs[i], center, backoffs[(i + 1) % len(backoffs)], samples
        )
        pts.extend(bez[:-1])           # drop duplicated arc endpoints
    return np.asarray(pts) if len(pts) >= 3 else None


def build_joint_bridges(polys, frac: float, samples: int = 10) -> list[np.ndarray]:
    """Bezier bridge polygons welding every joint where >=2 polygons meet.

    `polys` is the full set of rendered panels (inner triangles + wings +
    links). `frac` in [0, 1] scales the bridge radius (0 = no smoothing).
    """
    if frac <= 1e-6 or len(polys) < 2:
        return []
    allv = np.concatenate([np.asarray(p, dtype=float) for p in polys], axis=0)
    scale = float(np.linalg.norm(allv.max(axis=0) - allv.min(axis=0)))
    if scale < 1e-12:
        return []
    bridges = []
    for jn in _find_junctions(polys, tol=scale * 1e-6):
        b = _build_joint_bridge(jn, polys, frac, samples)
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
    "centroid"). For theta != 0 each foot is rotated about its OWNING T
    (P001/P002 about T0, P101/P112 about T1, P212/P202 about T2) — the
    kirigami wing rotation. T_i itself does not move with theta. At
    theta = 0 the dict is identical to the un-rotated construction.
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
    geom: TilingGeometry, c: float, theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Wings rotated about each T_i by theta.

    Returns:
        T:     (M, 3, 2)        -- inner-triangle vertices (unaffected by theta)
        wings: (M, 3, 3, 2)     -- (triangle, T_index, point_in_wing, xy)
                                point_in_wing order: [foot_a, P_i, foot_b].
                                Together with T_i these form the quad
                                (T_i, foot_a, P_i, foot_b) traversed in order.
    """
    T = compute_T(geom, c)
    feet = compute_feet(geom, T)
    foot_a = feet[:, _WING_FOOT_A, :]                       # (M, 3, 2)
    foot_b = feet[:, _WING_FOOT_B, :]                       # (M, 3, 2)
    wings = np.stack([foot_a, geom.P, foot_b], axis=2)      # (M, 3, 3, 2)
    if theta == 0.0:
        return T, wings
    R = _rotation_matrix(theta)
    pivot = T[:, :, None, :]                                # (M, 3, 1, 2)
    return T, pivot + (wings - pivot) @ R.T


def tile_state(
    geom: TilingGeometry, c: float, theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot (T, wings) evaluation. See compute_wings for shapes."""
    return compute_wings(geom, c, theta)


def compute_links(
    geom: TilingGeometry, c: float, theta: float = 0.0,
) -> np.ndarray:
    """Hexagons that join inner triangles across each shared Delaunay edge.

    For each shared edge between adjacent triangles m and n, two link hexagons
    are produced (one at each endpoint of the shared edge). Corner order:

        [T_m, F_m, P_m, P_n, F_n, T_n]

    where T_m / T_n are the inner-triangle vertices at the shared endpoint,
    P_m / P_n are the two (rotated) copies of the shared vertex, and F_m / F_n
    are each T's wing foot on the NON-shared edge (the wing's other foot). This
    hexagon is exactly the union of the two junction wings and the inner-vertex
    link, so it lets us drop those green junction wings and draw a single
    purple panel. The closing edge T_n -> T_m joins the two inner-triangle
    vertices DIRECTLY; the P_m -> P_n side is the shared vertex, which splits
    open as theta rotates the wings (at theta = 0 the P copies coincide).

    Returns: (K, 6, 2) where K = 2 * (number of shared edges).
            Empty array of shape (0, 6, 2) when there are no shared edges.
    """
    T, wings = compute_wings(geom, c, theta)
    K = len(geom.link_tri_m)
    if K == 0:
        return np.zeros((0, 6, 2))
    T_m = T[geom.link_tri_m, geom.link_ti_m]                       # (K, 2)
    T_n = T[geom.link_tri_n, geom.link_ti_n]
    P_m = wings[geom.link_tri_m, geom.link_ti_m, 1]               # shared vertex in m's wing
    P_n = wings[geom.link_tri_n, geom.link_ti_n, 1]               # shared vertex in n's wing
    # Non-shared-edge foot = the wing's OTHER foot (slots are 0 and 2).
    ns_m = 2 - geom.link_slot_m
    ns_n = 2 - geom.link_slot_n
    F_m = wings[geom.link_tri_m, geom.link_ti_m, ns_m]
    F_n = wings[geom.link_tri_n, geom.link_ti_n, ns_n]
    return np.stack([T_m, F_m, P_m, P_n, F_n, T_n], axis=1)       # (K, 6, 2)


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
    * theta  -- wing rotation (kirigami joint angle), in degrees.
    * spin   -- rigid rotation of the WHOLE structure in world space
                (about the centroid of the points), in degrees.
    * fillet -- joint smoothing (0 = none, 1 = max): adds a quadratic-Bezier
                bridge panel at each joint (where >=2 pieces meet at a point --
                the T hinges, feet, and shared vertices) to weld them smoothly.

    Radio buttons:
    * anchor -- point the T's shrink toward: "incenter" or "centroid".

    Mouse (hold a modifier and click on the main plot):
    * Shift-click a triangle -> select it; the c slider then edits only
        that triangle's c. Shift-click empty space to deselect.
    * Ctrl-click-drag a point -> move it; re-triangulates live.
    * Alt-click -> add a new point at the cursor.

    Keys:
    * c -> print the current spun coordinates (points, T's, wings, links)
            to the console.

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
        T, wings = tile_state(geom, c, theta)
        T0, w0 = (T, wings) if theta == 0.0 else tile_state(geom, c, 0.0)

        Ts, ws = spin_apply(T), spin_apply(wings)
        T0s, w0s = spin_apply(T0), spin_apply(w0)
        links = spin_apply(compute_links(geom, c, theta))

        # Sharp panels, then Bezier bridges welding the joints between them.
        # Junction wings are absorbed into the link hexagons, so skip drawing them.
        ffrac = s_fillet.val
        quads = np.concatenate([Ts[:, :, None, :], ws], axis=2).reshape(-1, 4, 2)
        keep_wing = ~junction_wing_mask(geom).reshape(-1)
        inner_list = list(Ts)
        wing_list = [q for q, k in zip(quads, keep_wing) if k]
        link_list = list(links)
        inner_pc.set_verts(inner_list)
        wing_pc.set_verts(wing_list)
        link_pc.set_verts(link_list)
        joint_pc.set_verts(build_joint_bridges(inner_list + wing_list + link_list, ffrac))

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
            f"  fillet = {ffrac:.2f}   Δ {dw:+.1f}% / {dh:+.1f}%{ratio}"
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
    s_fillet = Slider(ax_fillet, "fillet",      0.0,    1.0,   valinit=0.0)

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

    def on_key(event) -> None:
        if event.key != "c":
            return
        geom = state["geom"]
        theta = np.radians(s_th.val)
        c = effective_c()
        T, wings = tile_state(geom, c, theta)
        links = compute_links(geom, c, theta)
        pts_s, T_s = spin_apply(state["pts"]), spin_apply(T)
        wings_s, links_s = spin_apply(wings), spin_apply(links)
        print("=" * 64)
        print(f"spin={s_spin.val:.1f} deg  theta={np.degrees(theta):.1f} deg  "
            f"anchor={state['anchor']}  c_global={state['c_global']:.3f}")
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
        print(f"\nlinks ({len(links_s)} hexagons):")
        for i, h in enumerate(links_s):
            print(f"  link{i}: {np.round(h, 4).tolist()}")
        print("=" * 64)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Controls: shift-click triangle = per-tri c | ctrl-drag = move point"
        " | alt-click = add point | fillet slider = smooth joints"
        " | press 'c' = dump spun coords")
    refresh_static()
    redraw()
    plt.show()


if __name__ == "__main__":
    example_points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [1.73, 1.0]
    ])
    show(example_points, c_init=0.5)
