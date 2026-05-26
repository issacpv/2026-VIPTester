"""Compose a triangulated mesh from library tiles, welding
near-coincident vertices.

The compose-from-tiles workflow (GUI: the Tile Library panel) drops small
tile templates (see :mod:`auxetic.tile_library`) onto the canvas and
fuses them where their vertices line up. These pure functions do the
geometry: ``add_tile`` appends a (shifted) tile's points + triangles to
an existing mesh and ``weld_points`` merges vertices that fall within a
tolerance, remapping the triangulation and dropping any triangle that
collapses (two of its corners welded together).

A welded mesh is an **explicit** triangulation — adjacent tiles that
share an edge become two triangles sharing that edge, which is exactly
what the mode-11 bipartite generator fuses into a single auxetic shape.
The triangulation must therefore be preserved verbatim (never
re-Delaunayed), so the caller installs it via
``Lattice._set_points_and_tri`` with ``preserve_triangulation`` set.

numpy only — no GUI, no scipy.
"""

from __future__ import annotations

import numpy as np

# Default weld radius in lattice ([0, 1]) space. Roughly a quarter of the
# default tile edge (0.25), so dropping a tile within a comfortable mouse
# distance of an existing vertex fuses them.
DEFAULT_WELD_TOL = 0.06

# Snap radius (lattice space): a dropped tile whose nearest vertex lands
# within this distance of an existing vertex is translated so the two
# coincide exactly (see ``snap_tile_offset``). About half a tile edge, so
# a roughly-aimed drop locks cleanly into place rather than landing
# skewed. Larger than the weld tolerance so the snap engages first, then
# the now-exact coincidence welds.
SNAP_RADIUS = 0.12


def snap_tile_offset(existing_points: np.ndarray, tile_points: np.ndarray,
                     offset, snap_radius: float = SNAP_RADIUS) -> np.ndarray:
    """Adjust a drop ``offset`` so the placed tile snaps onto existing
    geometry.

    Finds the nearest (tile vertex, existing vertex) pair after applying
    ``offset``; if that pair is within ``snap_radius``, returns an offset
    translated so the two coincide **exactly**. Tiles have a fixed
    orientation, so this pure translation is enough to lock a congruent
    tile into alignment — a shared edge / the other shared vertices line
    up exactly too — which is what makes a dropped tile land *level*
    instead of pivoting off a single approximate weld. Returns the
    original ``offset`` (as a 2-vector) when there's nothing close enough
    to snap to, or no existing points.
    """
    off = np.asarray(offset, dtype=float).reshape(2)
    ex = np.asarray(existing_points, dtype=float).reshape(-1, 2)
    if ex.shape[0] == 0:
        return off
    placed = np.asarray(tile_points, dtype=float).reshape(-1, 2) + off
    # Pairwise distances (n_tile, n_existing); pick the closest pair.
    d = np.linalg.norm(placed[:, None, :] - ex[None, :, :], axis=2)
    ti, ei = np.unravel_index(int(np.argmin(d)), d.shape)
    if float(d[ti, ei]) <= float(snap_radius):
        return off + (ex[ei] - placed[ti])
    return off


def weld_points(points: np.ndarray, simplices: np.ndarray,
                tol: float = DEFAULT_WELD_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices within Euclidean distance ``tol`` and remap the
    triangulation.

    Returns ``(new_points, new_simplices)``. Points are clustered greedily
    in input order: each point either joins the first already-kept point
    within ``tol`` or becomes a new kept point. Triangles are reindexed to
    the kept points; any triangle that ends up with a repeated vertex
    (its corners welded together) is dropped as degenerate.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    simp = np.asarray(simplices, dtype=np.int64).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return pts.copy(), simp.copy()

    kept: list[np.ndarray] = []
    old_to_new = np.empty(n, dtype=np.int64)
    tol2 = float(tol) * float(tol)
    for i in range(n):
        p = pts[i]
        match = -1
        for j, q in enumerate(kept):
            d = p - q
            if float(d[0] * d[0] + d[1] * d[1]) <= tol2:
                match = j
                break
        if match < 0:
            old_to_new[i] = len(kept)
            kept.append(p)
        else:
            old_to_new[i] = match

    new_pts = np.asarray(kept, dtype=float).reshape(-1, 2)

    out: list[list[int]] = []
    for tri in simp:
        a, b, c = (int(old_to_new[int(v)]) for v in tri)
        if a != b and b != c and a != c:
            out.append([a, b, c])
    new_simp = (np.asarray(out, dtype=np.int64).reshape(-1, 3)
                if out else np.zeros((0, 3), dtype=np.int64))
    return new_pts, new_simp


def add_tile(points: np.ndarray, simplices: np.ndarray,
             tile_points: np.ndarray, tile_simplices: np.ndarray,
             offset=(0.0, 0.0),
             weld_tol: float = DEFAULT_WELD_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Append a tile to an existing mesh (shifted by ``offset``) and weld.

    ``points`` / ``simplices`` are the current mesh (may be empty).
    ``tile_points`` / ``tile_simplices`` are a template (see
    :mod:`auxetic.tile_library`); they are translated by ``offset`` (the
    drop location, applied to the template's centred coordinates) and
    their triangles are reindexed onto the appended block before welding
    the union. Welding fuses any tile vertex that lands within
    ``weld_tol`` of an existing vertex — the merge-on-proximity behaviour.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    simp = np.asarray(simplices, dtype=np.int64).reshape(-1, 3)
    tp = np.asarray(tile_points, dtype=float).reshape(-1, 2) + np.asarray(
        offset, dtype=float)
    ts = np.asarray(tile_simplices, dtype=np.int64).reshape(-1, 3) + len(pts)

    merged_pts = np.vstack([pts, tp]) if len(pts) else tp
    merged_simp = np.vstack([simp, ts]) if len(simp) else ts
    return weld_points(merged_pts, merged_simp, weld_tol)
