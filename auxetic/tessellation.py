"""Equilateral-fill tessellation of an arbitrary planar region (task 5).

Given a polygonal region and a target tile density, fill the interior
with a near-equilateral triangular lattice and close the boundary with
the (necessarily irregular) isosceles / scalene triangles that bridge
the regular interior grid to the region's edges.

Pipeline:

1. Lay an ideal equilateral point grid (rows offset by half a cell,
   spaced ``edge·√3/2``) over the region's bounding box.
2. Keep grid points that are inside the polygon by at least
   ``interior_margin`` — this leaves a clean band near the boundary for
   the closer triangles instead of slivers.
3. Resample the polygon boundary so no boundary edge is longer than
   ``edge`` (keeps the closer triangles well-shaped).
4. Delaunay-triangulate the union of interior + boundary points and drop
   any triangle whose centroid falls outside the polygon (so concave
   regions are respected).

The interior triangles inherit the grid's equilateral shape; the ring of
triangles touching the boundary are the isosceles/scalene closers.

numpy / scipy only — no GUI, no other deps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay


@dataclass
class TessellationResult:
    """Output of :func:`generate_tessellation`.

    ``points`` is the full (N, 2) vertex array: the first ``n_boundary``
    rows are the (resampled) boundary points, the rest are interior grid
    points. ``simplices`` is the (M, 3) array of triangle vertex indices
    into ``points`` covering the region. ``boundary`` is the (possibly
    resampled) ordered polygon the tessellation was clipped to."""

    points: np.ndarray
    simplices: np.ndarray
    boundary: np.ndarray
    n_boundary: int

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    @property
    def n_triangles(self) -> int:
        return int(len(self.simplices))

    def interior_triangle_mask(self) -> np.ndarray:
        """Boolean mask over ``simplices``: triangles all of whose
        vertices are interior grid points (index >= ``n_boundary``).
        These are the regular near-equilateral tiles; the complement is
        the ring of boundary closers."""
        return np.all(self.simplices >= self.n_boundary, axis=1)


# ---------------------------------------------------------------------------
# Polygon helpers (pure numpy)
# ---------------------------------------------------------------------------

def polygon_area(polygon: np.ndarray) -> float:
    """Absolute area of a simple polygon via the shoelace formula."""
    p = np.asarray(polygon, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def points_in_polygon(pts: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Boolean mask: which of ``pts`` lie inside ``polygon`` (even-odd
    ray-casting, vectorised over points). Points exactly on an edge are
    reported inconsistently (as with any ray-cast) — callers that care
    use a margin."""
    pts = np.asarray(pts, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    x = pts[:, 0]
    y = pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= crosses
        j = i
    return inside


def _dist_points_to_segment(pts: np.ndarray, a: np.ndarray,
                            b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-300
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def distance_to_polygon(pts: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Minimum distance from each point to the polygon boundary (the
    closed edge loop), regardless of inside/outside."""
    pts = np.asarray(pts, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    d = np.full(len(pts), np.inf)
    n = len(poly)
    for i in range(n):
        d = np.minimum(d, _dist_points_to_segment(pts, poly[i], poly[(i + 1) % n]))
    return d


def resample_polygon(polygon: np.ndarray, max_segment: float) -> np.ndarray:
    """Return the polygon's boundary as an ordered point loop in which no
    consecutive segment is longer than ``max_segment``. Each original
    vertex is preserved (it starts an edge); long edges get evenly-spaced
    interior points inserted."""
    poly = np.asarray(polygon, dtype=float)
    n = len(poly)
    out: list[np.ndarray] = []
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        length = float(np.linalg.norm(b - a))
        k = max(1, int(math.ceil(length / max_segment))) if max_segment > 0 else 1
        for s in range(k):  # include a, exclude b (next edge contributes it)
            out.append(a + (b - a) * (s / k))
    return np.asarray(out, dtype=float)


def equilateral_grid(xmin: float, ymin: float, xmax: float, ymax: float,
                     edge: float) -> np.ndarray:
    """Triangular (equilateral) point lattice covering the bbox. Rows are
    spaced ``edge·√3/2`` apart; odd rows are offset by ``edge/2`` so each
    interior point has six equidistant neighbours."""
    if edge <= 0:
        raise ValueError(f"edge must be positive, got {edge}")
    dy = edge * math.sqrt(3.0) / 2.0
    rows = []
    n_rows = int(math.floor((ymax - ymin) / dy)) + 2
    for j in range(n_rows):
        y = ymin + j * dy
        offset = (edge / 2.0) if (j % 2) else 0.0
        n_cols = int(math.floor((xmax - xmin) / edge)) + 2
        xs = xmin + offset + np.arange(n_cols) * edge
        col = np.column_stack([xs, np.full(len(xs), y)])
        rows.append(col)
    return np.vstack(rows) if rows else np.zeros((0, 2))


def _dedup(pts: np.ndarray, tol: float) -> np.ndarray:
    """Drop near-duplicate points (within ``tol``), preserving order of
    first appearance."""
    if len(pts) == 0:
        return pts
    keys = np.round(pts / tol).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def triangle_angles(tri_pts: np.ndarray) -> np.ndarray:
    """Interior angles (degrees) of a triangle, sorted ascending."""
    p = np.asarray(tri_pts, dtype=float)
    a = float(np.linalg.norm(p[1] - p[2]))
    b = float(np.linalg.norm(p[0] - p[2]))
    c = float(np.linalg.norm(p[0] - p[1]))
    A = math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c + 1e-300)))))
    B = math.degrees(math.acos(max(-1.0, min(1.0, (a * a + c * c - b * b) / (2 * a * c + 1e-300)))))
    C = 180.0 - A - B
    return np.array(sorted([A, B, C]))


def equilateral_deviation(tri_pts: np.ndarray) -> float:
    """Max absolute deviation (degrees) of a triangle's angles from 60°.
    0 == perfectly equilateral."""
    return float(np.max(np.abs(triangle_angles(tri_pts) - 60.0)))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def edge_from_triangle_count(area: float, n_triangles: int) -> float:
    """Equilateral side length that tiles ``area`` into roughly
    ``n_triangles`` triangles (each of area ``√3/4·edge²``)."""
    if n_triangles <= 0:
        raise ValueError("n_triangles must be positive")
    return math.sqrt(4.0 * area / (math.sqrt(3.0) * n_triangles))


def generate_tessellation(boundary: np.ndarray,
                          target_edge: float | None = None,
                          *,
                          n_triangles: int | None = None,
                          resample_boundary: bool = True,
                          interior_margin: float | None = None
                          ) -> TessellationResult:
    """Tessellate the region bounded by ``boundary`` (an ordered (N, 2)
    polygon, N ≥ 3) into near-equilateral triangles.

    Exactly one of ``target_edge`` (the equilateral side length) or
    ``n_triangles`` (a target count, converted to an edge length from the
    region area) sets the density. ``interior_margin`` (default
    ``0.5·edge``) is how far inside the boundary an interior grid point
    must sit; it controls the width of the boundary-closer band.

    Returns a :class:`TessellationResult` whose triangles cover the
    region: interior triangles are near-equilateral, boundary triangles
    are the isosceles/scalene closers.
    """
    poly = np.asarray(boundary, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(
            f"boundary must be an (N>=3, 2) polygon, got shape {poly.shape}")

    area = polygon_area(poly)
    if area <= 0.0:
        raise ValueError("boundary polygon has zero area (degenerate/collinear)")

    if target_edge is None:
        if n_triangles is None:
            raise ValueError("provide either target_edge or n_triangles")
        target_edge = edge_from_triangle_count(area, int(n_triangles))
    edge = float(target_edge)
    if edge <= 0.0:
        raise ValueError(f"target_edge must be positive, got {edge}")
    if interior_margin is None:
        interior_margin = 0.5 * edge

    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)

    # Interior equilateral grid points, kept if comfortably inside.
    grid = equilateral_grid(float(xmin), float(ymin), float(xmax), float(ymax), edge)
    if len(grid):
        inside = points_in_polygon(grid, poly)
        grid = grid[inside]
    if len(grid):
        d = distance_to_polygon(grid, poly)
        grid = grid[d >= interior_margin]

    # Boundary points (resampled so closer triangles stay well-shaped).
    # Dedup boundary and interior separately so the boundary points keep
    # a contiguous block at the front (index < n_boundary) — interior
    # grid points are >= interior_margin inside, so they never collide
    # with boundary points.
    tol = max(1e-12, 1e-6 * edge)
    bnd = resample_polygon(poly, edge) if resample_boundary else poly.copy()
    bnd = _dedup(bnd, tol)
    if len(grid):
        grid = _dedup(grid, tol)
        all_pts = np.vstack([bnd, grid])
    else:
        all_pts = bnd
    n_boundary = int(len(bnd))

    if len(all_pts) < 3:
        raise ValueError("too few points to triangulate; reduce target_edge")

    tri = Delaunay(all_pts)
    simplices = np.asarray(tri.simplices)

    # Clip triangles whose centroid lies outside the polygon (Delaunay
    # fills the convex hull; this carves concavities back out).
    centroids = all_pts[simplices].mean(axis=1)
    keep = points_in_polygon(centroids, poly)
    simplices = simplices[keep]

    return TessellationResult(points=all_pts, simplices=simplices,
                              boundary=bnd, n_boundary=n_boundary)
