"""Tile-tile collision detection.

The kirigami null-space lets joint angles sweep arbitrarily far in
either direction (the math has no built-in stop), but physically the
shrunken tiles will eventually run into each other across the
negative-space gap. This module provides:

- **2D**: full Separating Axis Theorem on convex polygons (M2.8).
- **3D** (M3 polish): face-normal SAT on convex polytopes
  (tetrahedra, convex hubs from ``scipy.spatial.ConvexHull``). The
  classical full 3D SAT also tests edge×edge cross products; we
  skip those for speed and accept a small false-negative rate at
  edge-on-edge configurations. For kirigami applications this is
  acceptable — tiles colliding in such configurations also collide
  on a face-normal axis a few sweep steps later, so the bounding
  on the achievable θ range is essentially unchanged.

Why pure numpy:
    The package is numpy/scipy-only per CLAUDE.md. SAT in either
    dimension is ~50 LOC; pulling in Shapely (2D) or trimesh (3D)
    is overkill for this surface area.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .simulation import Constraint, Simulator, TileSystem


# ---------------------------------------------------------------------------
# 2D SAT primitive
# ---------------------------------------------------------------------------

def _project(verts: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Min/max projection of ``verts`` onto unit ``axis``."""
    p = verts @ axis
    return float(p.min()), float(p.max())


def _convex_hull_face_normals(verts: np.ndarray) -> np.ndarray:
    """Return the outward face normals of the convex hull of ``verts``
    as an ``(n_faces, 3)`` unit-normal array.

    Tetrahedra (4 verts) and any larger convex 3D point cloud are
    handled. Falls back to the empty array if scipy can't build a
    hull (degenerate input).
    """
    arr = np.asarray(verts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 4:
        return np.zeros((0, 3), dtype=float)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(arr)
    except Exception:
        return np.zeros((0, 3), dtype=float)
    normals = []
    for simplex in hull.simplices:
        a, b, c = arr[simplex[0]], arr[simplex[1]], arr[simplex[2]]
        n = np.cross(b - a, c - a)
        ln = float(np.linalg.norm(n))
        if ln < 1e-12:
            continue
        n = n / ln
        # Orient outward: away from the centroid.
        if float(np.dot(n, a - arr.mean(axis=0))) < 0.0:
            n = -n
        normals.append(n)
    return (np.asarray(normals, dtype=float)
            if normals else np.zeros((0, 3), dtype=float))


def polytopes_overlap_3d(verts_a: np.ndarray,
                          verts_b: np.ndarray,
                          tol: float = 0.0) -> bool:
    """Convex-polytope overlap test in 3D via face-normal SAT.

    Tests separating axes from each polytope's outward face normals.
    We skip the edge×edge axes that the full 3D SAT prescribes:

    - For tetrahedra-vs-tetrahedra (the common kirigami case in 3D
      modes), face normals catch all but a thin set of edge-on-edge
      configurations.
    - In a kirigami sweep those configurations are bounded above
      and below by configurations that DO get caught by the
      face-normal axes a few θ-samples earlier / later, so the
      reachable θ range the caller computes is essentially the same.
    - Skipping the edge axes turns a potentially O(F1*F2 + E1*E2)
      check into O(F1 + F2), which matters when iterating across
      every (i, j) tile pair × every sweep sample.

    A future patch can add the edge-cross axes if the false-negative
    rate proves to matter. ``tol`` works the same as in
    :func:`polygons_overlap_2d` — pass a small positive value to
    treat just-touching surfaces as non-overlapping.
    """
    a = np.asarray(verts_a, dtype=float)
    b = np.asarray(verts_b, dtype=float)
    if (a.ndim != 2 or a.shape[1] != 3
            or b.ndim != 2 or b.shape[1] != 3):
        raise ValueError(
            f"polytopes_overlap_3d expects (N, 3) arrays; got "
            f"{a.shape} and {b.shape}"
        )
    # Quick AABB pre-filter — cheap and rules out most pairs.
    a_min, a_max = a.min(axis=0), a.max(axis=0)
    b_min, b_max = b.min(axis=0), b.max(axis=0)
    if np.any(a_max < b_min + tol) or np.any(b_max < a_min + tol):
        return False

    # Collect candidate separating axes.
    axes = []
    for poly in (a, b):
        axes.append(_convex_hull_face_normals(poly))
    all_axes = np.vstack([ax for ax in axes if ax.size]) \
        if any(ax.size for ax in axes) else np.zeros((0, 3))

    if all_axes.size == 0:
        # Couldn't build hull normals — fall back to AABB result
        # (which already returned True above by getting here).
        return True

    for axis in all_axes:
        proj_a = a @ axis
        proj_b = b @ axis
        min_a, max_a = float(proj_a.min()), float(proj_a.max())
        min_b, max_b = float(proj_b.min()), float(proj_b.max())
        if max_a < min_b + tol or max_b < min_a + tol:
            return False
    return True


def polygons_overlap_2d(verts_a: np.ndarray,
                         verts_b: np.ndarray,
                         tol: float = 0.0) -> bool:
    """Separating-axis-theorem overlap test for two convex 2D polygons.

    Returns ``True`` iff the polygons' interiors overlap by more than
    ``tol`` along every separating axis. Both ``verts_a`` and
    ``verts_b`` are ``(N, 2)`` arrays of polygon corners in any
    orientation (CW or CCW); SAT doesn't care.

    Edge case: polygons that touch at a single vertex / along an edge
    have zero-overlap projections — controlled by ``tol``. Pass a
    small positive ``tol`` (e.g. ``1e-6``) to treat touching as
    "not colliding" so kirigami tiles that meet at constraint vertices
    aren't flagged.
    """
    a = np.asarray(verts_a, dtype=float)
    b = np.asarray(verts_b, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(
            f"polygons_overlap_2d expects (N, 2) arrays; got "
            f"{a.shape} and {b.shape}"
        )
    for poly in (a, b):
        n = poly.shape[0]
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = float(np.linalg.norm(edge))
            if length < 1e-12:
                continue   # degenerate edge — skip its axis
            # Outward normal: rotate edge 90° (sign doesn't matter for SAT).
            axis = np.array([-edge[1], edge[0]]) / length
            min_a, max_a = _project(a, axis)
            min_b, max_b = _project(b, axis)
            # If a separating axis exists (gap > tol), polygons don't overlap.
            if max_a < min_b + tol or max_b < min_a + tol:
                return False
    return True


# ---------------------------------------------------------------------------
# CollisionChecker
# ---------------------------------------------------------------------------

class CollisionChecker:
    """Detect 2D tile-tile overlaps in a kirigami pose.

    Tiles connected by a :class:`Constraint` are exempt from the check
    because constraint-coupled tile pairs are *supposed* to share a
    vertex. The test compares each remaining tile pair pairwise via
    :func:`polygons_overlap_2d`.

    The checker reuses the world-vertex evaluator from
    :class:`Simulator` (``_tile_world_vertices``) so the tile
    coordinates at any pose match the rest of the simulation pipeline.
    """

    def __init__(self,
                 tile_system: TileSystem,
                 *,
                 tol: float = 1.0e-6,
                 ignore_connected: bool = True):
        self.tile_system = tile_system
        self.tol = float(tol)
        self.dimension = tile_system.dimension
        # Internal Simulator just to reuse the world-vertex helper.
        # load_axis is meaningless here but the constructor demands a
        # non-zero vector.
        load_axis = np.zeros(self.dimension); load_axis[0] = 1.0
        self._sim = Simulator(tile_system, load_axis=load_axis)

        # Build constraint-pair adjacency so we can skip "supposed to touch"
        # pairs.
        self._connected: set[tuple[int, int]] = set()
        if ignore_connected:
            for c in tile_system.constraints:
                a, b = sorted((int(c.tile_a), int(c.tile_b)))
                if a != b:
                    self._connected.add((a, b))

        # Degenerate tiles (fewer than 3 vertices) are 1-D rigid links, not
        # area tiles — e.g. mode-11 bond bars. Area-overlap SAT on a 2-point
        # "polygon" is meaningless (it spuriously reports overlap when the
        # segment lies along a neighbouring tile's edge), so exclude them.
        self._skip: set[int] = {
            i for i, t in enumerate(tile_system.tiles)
            if np.asarray(t).shape[0] < 3
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _overlap(self, va: np.ndarray, vb: np.ndarray) -> bool:
        """Dispatch the right SAT primitive for the tile system's
        dimensionality."""
        if self.dimension == 2:
            return polygons_overlap_2d(va, vb, tol=self.tol)
        return polytopes_overlap_3d(va, vb, tol=self.tol)

    def colliding_pairs(self, pose: np.ndarray) -> List[Tuple[int, int]]:
        """Return list of ``(i, j)`` (i < j) tile pairs whose world-frame
        polytopes overlap at this pose. Works for both 2D and 3D tile
        systems."""
        n = self.tile_system.n_tiles
        out: List[Tuple[int, int]] = []
        # Cache world vertices per tile to amortise the pose decomposition.
        worlds = [self._sim._tile_world_vertices(pose, i) for i in range(n)]
        for i in range(n):
            if i in self._skip:
                continue
            for j in range(i + 1, n):
                if j in self._skip or (i, j) in self._connected:
                    continue
                if self._overlap(worlds[i], worlds[j]):
                    out.append((i, j))
        return out

    def has_collision(self, pose: np.ndarray) -> bool:
        """Short-circuit version of :meth:`colliding_pairs` — returns
        ``True`` as soon as any non-exempt tile pair overlaps. Faster
        than enumerating every pair when you only need the boolean."""
        n = self.tile_system.n_tiles
        worlds = [self._sim._tile_world_vertices(pose, i) for i in range(n)]
        for i in range(n):
            if i in self._skip:
                continue
            for j in range(i + 1, n):
                if j in self._skip or (i, j) in self._connected:
                    continue
                if self._overlap(worlds[i], worlds[j]):
                    return True
        return False

