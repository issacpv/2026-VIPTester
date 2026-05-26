"""Library of 2D tile templates for the compose-from-tiles workflow.

Each template is a small point cloud + its triangulation, expressed in
lattice ([0, 1]) space and **centred on its centroid**, so dropping a
tile places its centre at the cursor. Composing tiles
(:mod:`auxetic.composition`) welds near-coincident vertices, so tiles
that line up share vertices/edges; the existing mode-11 bipartite
generator then renders the fused triangulation as one auxetic shape.

Tiles are triangulated into the same kind of triangle network the
random/grid modes produce, so nothing downstream needs to special-case
them — only the *source* of the points + triangulation differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default tile edge length in lattice space — a quarter of the unit cell,
# so a handful of tiles compose a visible patch.
TILE_EDGE = 0.25


@dataclass(frozen=True)
class TileTemplate:
    """A library tile: a centred point cloud + its triangulation.

    ``points`` is an ``(N, 2)`` array centred on its centroid; ``simplices``
    is an ``(M, 3)`` int array of triangle vertex indices. ``name`` is the
    library label / drag identifier.
    """

    name: str
    points: np.ndarray
    simplices: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.simplices.shape[0])


def _centred(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return pts - pts.mean(axis=0)


def _equilateral_triangle(edge: float = TILE_EDGE) -> TileTemplate:
    h = edge * np.sqrt(3.0) / 2.0
    pts = _centred([[0.0, 0.0], [edge, 0.0], [edge / 2.0, h]])
    return TileTemplate("Triangle", pts, np.array([[0, 1, 2]], dtype=np.int64))


def _equilateral_triangle_down(edge: float = TILE_EDGE) -> TileTemplate:
    # The up-triangle rotated 180° (negation of the centred points). With
    # translation-only placement, two *identical* up-triangles can share
    # only a single vertex; an up + a down triangle share a full edge (two
    # vertices weld), composing the classic rhombus / triangular grid.
    up = _equilateral_triangle(edge)
    return TileTemplate("Triangle-down", -up.points, up.simplices.copy())


def _square(edge: float = TILE_EDGE) -> TileTemplate:
    pts = _centred([[0.0, 0.0], [edge, 0.0], [edge, edge], [0.0, edge]])
    # Two triangles sharing the 0–2 diagonal.
    simp = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return TileTemplate("Square", pts, simp)


def _hexagon(edge: float = TILE_EDGE) -> TileTemplate:
    # Regular hexagon: six outer vertices (circumradius = edge) + centre,
    # fanned into six triangles. The centre is index 6.
    ang = np.deg2rad(np.arange(6) * 60.0)
    outer = np.column_stack([edge * np.cos(ang), edge * np.sin(ang)])
    pts = _centred(np.vstack([outer, [[0.0, 0.0]]]))
    simp = np.array([[6, i, (i + 1) % 6] for i in range(6)], dtype=np.int64)
    return TileTemplate("Hexagon", pts, simp)


# Public library, keyed by tile name (also the drag identifier).
TILE_LIBRARY: dict[str, TileTemplate] = {
    t.name: t for t in (_equilateral_triangle(), _equilateral_triangle_down(),
                        _square(), _hexagon())
}


def get_tile(name: str) -> TileTemplate:
    """Look up a tile template by name; raises ``KeyError`` if unknown."""
    return TILE_LIBRARY[name]
