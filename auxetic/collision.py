"""Tile-tile collision detection (M2.8).

The kirigami null-space lets joint angles sweep arbitrarily far in
either direction (the math has no built-in stop), but physically the
shrunken tiles will eventually run into each other across the
negative-space gap. This module provides a 2D **separating axis
theorem** (SAT) overlap test plus a :class:`CollisionChecker` wrapper
that filters out the tile pairs that are *supposed* to touch (those
joined by a kirigami constraint at a shared vertex).

Why SAT and not Shapely:
    The package is numpy/scipy-only per CLAUDE.md. SAT for two convex
    polygons in 2D is ~30 lines and exact; pulling in Shapely just for
    intersection testing is overkill and adds a native dependency.

3D collision (mode-3 / 6 / 9 tetrahedra) is intentionally out of scope
in M2 — that needs GJK or convex-hull SAT in 3D, which is a
substantially bigger primitive. For now,
:func:`CollisionChecker.has_collision` returns ``False`` on 3D tile
systems with a one-time warning, so callers don't have to special-case
the dimension.
"""

from __future__ import annotations

import warnings
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def colliding_pairs(self, pose: np.ndarray) -> List[Tuple[int, int]]:
        """Return list of ``(i, j)`` (i < j) tile pairs whose 2D world
        polygons overlap at this pose. Empty list for 3D tile systems
        (with a one-time warning) and for poses that produce no
        overlaps."""
        if self.dimension != 2:
            self._warn_3d_unsupported()
            return []
        n = self.tile_system.n_tiles
        out: List[Tuple[int, int]] = []
        # Cache world vertices per tile to amortise the pose decomposition.
        worlds = [self._sim._tile_world_vertices(pose, i) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in self._connected:
                    continue
                if polygons_overlap_2d(worlds[i], worlds[j], tol=self.tol):
                    out.append((i, j))
        return out

    def has_collision(self, pose: np.ndarray) -> bool:
        """Short-circuit version of :meth:`colliding_pairs` — returns
        ``True`` as soon as any non-exempt tile pair overlaps. Faster
        than enumerating every pair when you only need the boolean.
        """
        if self.dimension != 2:
            self._warn_3d_unsupported()
            return False
        n = self.tile_system.n_tiles
        worlds = [self._sim._tile_world_vertices(pose, i) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in self._connected:
                    continue
                if polygons_overlap_2d(worlds[i], worlds[j], tol=self.tol):
                    return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _warned_3d = False

    @classmethod
    def _warn_3d_unsupported(cls) -> None:
        if cls._warned_3d:
            return
        cls._warned_3d = True
        warnings.warn(
            "CollisionChecker is 2D-only in M2 — tile-tile overlap "
            "detection for 3D tetrahedra is not implemented yet. "
            "All 3D collision queries return False."
        )
