"""Rotating-cuboids 3D kirigami (mode 10).

This is the 3D analog of the 2D rotating-squares mechanism. The
existing 3D modes (3, 6, 9) use **tetrahedral kirigami** — each
Delaunay simplex is shrunk into a small tetrahedron — which doesn't
reproduce the canonical "block-collapse" auxetic visual users expect
for 3D structures. This module provides a parallel cuboid pipeline:
an ``n×n×n`` grid of points carves space into ``(n-1)³`` unit cubes,
each cube is shrunk toward its centroid by ``ratio``, and adjacent
cubes are pinned at one shared edge of the face between them. The
edge orientation alternates by cell-position parity per Grima et al.
to produce a coherent auxetic kinematic mode.

Vertex layout (8 corners per cube):

    idx 0: (0, 0, 0)   idx 4: (0, 0, 1)
    idx 1: (1, 0, 0)   idx 5: (1, 0, 1)
    idx 2: (0, 1, 0)   idx 6: (0, 1, 1)
    idx 3: (1, 1, 0)   idx 7: (1, 1, 1)

Encoded as ``vert_idx(di, dj, dk) = di + 2*dj + 4*dk``. The
constraint generator picks one edge of each shared face to pin, with
the edge offset alternating by the parity of the perpendicular
indices — see :func:`generate_cuboids` for the exact rule.

This module returns the geometric data needed for the simulator
(:class:`auxetic.simulation.TileSystem`) and the export pipeline
(STL / kirigami text formats). Modes 1–9 are untouched.
"""

from __future__ import annotations

from itertools import product
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Vertex bookkeeping
# ---------------------------------------------------------------------------

# Direction triples for the 8 cube corners, in vertex-index order.
# i.e. CUBE_CORNERS[v] gives the (di, dj, dk) offset of corner v.
CUBE_CORNERS: tuple[tuple[int, int, int], ...] = tuple(
    (di, dj, dk)
    for dk in (0, 1)
    for dj in (0, 1)
    for di in (0, 1)
)
assert len(CUBE_CORNERS) == 8


def vert_idx(di: int, dj: int, dk: int) -> int:
    """Vertex index inside a tile from a (di, dj, dk) corner offset.

    Inverse of indexing into :data:`CUBE_CORNERS`. Used by the
    constraint generator to pick specific corners of a cube tile.
    """
    return int(di) + 2 * int(dj) + 4 * int(dk)


# 12 triangles for an axis-aligned cube, indexed into the 8-corner
# layout above. Used by :func:`triangles_for_cube` (and from there by
# the STL export path for mode 10). Each triangle wound CCW seen from
# outside the cube so VTK / printers compute the right normals.
_CUBE_TRIANGLES: tuple[tuple[int, int, int], ...] = (
    # -z face (k=0): corners 0, 1, 2, 3
    (0, 2, 1), (1, 2, 3),
    # +z face (k=1): corners 4, 5, 6, 7
    (4, 5, 6), (5, 7, 6),
    # -y face (j=0): corners 0, 1, 4, 5
    (0, 1, 4), (1, 5, 4),
    # +y face (j=1): corners 2, 3, 6, 7
    (2, 6, 3), (3, 6, 7),
    # -x face (i=0): corners 0, 2, 4, 6
    (0, 4, 2), (2, 4, 6),
    # +x face (i=1): corners 1, 3, 5, 7
    (1, 3, 5), (3, 7, 5),
)


def triangles_for_cube(verts: np.ndarray) -> List[np.ndarray]:
    """Return 12 (3, 3) triangle arrays describing the convex hull of
    an 8-vertex cuboid. ``verts`` may be axis-aligned, sheared, or
    rotated — :func:`triangles_for_cube` only assumes the corner order
    matches :data:`CUBE_CORNERS`."""
    arr = np.asarray(verts, dtype=float)
    if arr.shape != (8, 3):
        raise ValueError(
            f"triangles_for_cube expects (8, 3); got {arr.shape}")
    return [
        np.stack([arr[a], arr[b], arr[c]], axis=0)
        for a, b, c in _CUBE_TRIANGLES
    ]


# ---------------------------------------------------------------------------
# Cuboid kirigami generator
# ---------------------------------------------------------------------------

def generate_cuboids(
        n: int = 3,
        ratio: float = 0.35,
        ) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int, int]]]:
    """Build a rotating-cuboids 3D kirigami of resolution ``n``.

    - ``n`` : grid resolution along each axis. The lattice has ``n³``
      grid points and ``(n-1)³`` cuboid cells.
    - ``ratio`` : shrink factor (0 = no shrink, 1 = collapse to centroid).
      Identical semantics to the existing modes 1–9.

    Returns ``(points, tiles, constraints)``:
    - ``points`` : ``(n³, 3)`` array of grid point coordinates in
      ``[0, 1]³`` (lattice space, same convention as the rest of the
      package).
    - ``tiles``  : list of ``(8, 3)`` arrays, one per cuboid cell.
      Vertex order matches :data:`CUBE_CORNERS`. Coordinates are
      already shrunk toward the cell centroid by ``ratio``.
    - ``constraints`` : list of ``(tile_a, vert_a, tile_b, vert_b,
      ctype)`` tuples. Two constraints per face-adjacent cube pair —
      together they pin one full edge of the shared face. The edge
      offset alternates by perpendicular-index parity to give the
      rotating-cubes auxetic mechanism a coherent kinematic mode.

    Constraint topology (rotating cubes):
        Cubes A=(i,j,k) and B=(i+1,j,k) share the face at x=i+1.
        We pin the edge along **y** at z=z_off where
        ``z_off = (j + k) mod 2``. Cyclic rules apply for y- and
        z-faces. With this scheme the constraint graph is bipartite
        (color cubes by ``(i+j+k) mod 2``), so the bipartite-rotation
        mode selector in :class:`auxetic.simulation.Simulator` picks
        a coherent alternating-rotation auxetic mode for free.
    """
    if int(n) < 2:
        raise ValueError(f"n must be >= 2 (need at least one cube); got {n}")
    n = int(n)

    # ---- Grid points (n^3) -----------------------------------------
    coords = np.linspace(0.0, 1.0, n)
    points = np.array(
        [(x, y, z) for z in coords for y in coords for x in coords],
        dtype=float,
    )

    def pidx(i: int, j: int, k: int) -> int:
        """Flat index into ``points`` for grid coord (i, j, k)."""
        return i + n * j + n * n * k

    # ---- Cuboid cells (one tile per cell) --------------------------
    tiles: List[np.ndarray] = []
    cube_index_of: dict[tuple[int, int, int], int] = {}
    cell_indices: List[tuple[int, int, int]] = []
    for ci, (i, j, k) in enumerate(
        (i, j, k) for k in range(n - 1) for j in range(n - 1) for i in range(n - 1)
    ):
        # Indices of the 8 corners of this cell, in CUBE_CORNERS order.
        corner_idx = [
            pidx(i + di, j + dj, k + dk) for di, dj, dk in CUBE_CORNERS
        ]
        corners = points[corner_idx]
        centroid = corners.mean(axis=0)
        shrunk = (1.0 - ratio) * corners + ratio * centroid
        tiles.append(np.asarray(shrunk, dtype=float))
        cube_index_of[(i, j, k)] = ci
        cell_indices.append((i, j, k))

    # ---- Constraints: one edge per face-adjacent pair --------------
    constraints: List[Tuple[int, int, int, int, int]] = []
    for ci, (i, j, k) in enumerate(cell_indices):
        # +x face: shared face at x=i+1; cube A's right face / B's left face.
        # Edge along y, picking z=z_off based on (j+k) parity.
        nidx = cube_index_of.get((i + 1, j, k))
        if nidx is not None:
            z_off = (j + k) % 2
            constraints.append(
                (ci, vert_idx(1, 0, z_off), nidx, vert_idx(0, 0, z_off), 1))
            constraints.append(
                (ci, vert_idx(1, 1, z_off), nidx, vert_idx(0, 1, z_off), 1))
        # +y face: edge along z, picking x=x_off based on (k+i) parity.
        nidx = cube_index_of.get((i, j + 1, k))
        if nidx is not None:
            x_off = (k + i) % 2
            constraints.append(
                (ci, vert_idx(x_off, 1, 0), nidx, vert_idx(x_off, 0, 0), 1))
            constraints.append(
                (ci, vert_idx(x_off, 1, 1), nidx, vert_idx(x_off, 0, 1), 1))
        # +z face: edge along x, picking y=y_off based on (i+j) parity.
        nidx = cube_index_of.get((i, j, k + 1))
        if nidx is not None:
            y_off = (i + j) % 2
            constraints.append(
                (ci, vert_idx(0, y_off, 1), nidx, vert_idx(0, y_off, 0), 1))
            constraints.append(
                (ci, vert_idx(1, y_off, 1), nidx, vert_idx(1, y_off, 0), 1))

    return points, tiles, constraints
