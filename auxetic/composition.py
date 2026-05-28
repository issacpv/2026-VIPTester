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
                tol: float = DEFAULT_WELD_TOL, *,
                return_kept: bool = False):
    """Merge vertices within Euclidean distance ``tol`` and remap the
    triangulation.

    Returns ``(new_points, new_simplices)``. Points are clustered greedily
    in input order: each point either joins the first already-kept point
    within ``tol`` or becomes a new kept point. Triangles are reindexed to
    the kept points; any triangle that ends up with a repeated vertex
    (its corners welded together) is dropped as degenerate.

    With ``return_kept=True`` the result is
    ``(new_points, new_simplices, kept_index)`` where ``kept_index`` is an
    int array of the indices (into the input ``simplices``) of the
    triangles that survived — letting a caller carry a per-triangle array
    (e.g. tile/ownership ids) through the weld in lockstep.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    simp = np.asarray(simplices, dtype=np.int64).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        if return_kept:
            return (pts.copy(), simp.copy(),
                    np.arange(len(simp), dtype=np.int64))
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
    kept_tris: list[int] = []
    for ti, tri in enumerate(simp):
        a, b, c = (int(old_to_new[int(v)]) for v in tri)
        if a != b and b != c and a != c:
            out.append([a, b, c])
            kept_tris.append(ti)
    new_simp = (np.asarray(out, dtype=np.int64).reshape(-1, 3)
                if out else np.zeros((0, 3), dtype=np.int64))
    if return_kept:
        return new_pts, new_simp, np.asarray(kept_tris, dtype=np.int64)
    return new_pts, new_simp


def add_tile(points: np.ndarray, simplices: np.ndarray,
             tile_points: np.ndarray, tile_simplices: np.ndarray,
             offset=(0.0, 0.0),
             weld_tol: float = DEFAULT_WELD_TOL, *,
             return_kept: bool = False):
    """Append a tile to an existing mesh (shifted by ``offset``) and weld.

    ``points`` / ``simplices`` are the current mesh (may be empty).
    ``tile_points`` / ``tile_simplices`` are a template (see
    :mod:`auxetic.tile_library`); they are translated by ``offset`` (the
    drop location, applied to the template's centred coordinates) and
    their triangles are reindexed onto the appended block before welding
    the union. Welding fuses any tile vertex that lands within
    ``weld_tol`` of an existing vertex — the merge-on-proximity behaviour.

    With ``return_kept=True`` the result is
    ``(new_points, new_simplices, kept_index)``; ``kept_index`` indexes
    into the *merged* triangle list (the existing ``simplices`` followed
    by the tile's), naming the triangles that survived the weld — so the
    caller can keep a per-triangle array aligned across the add.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    simp = np.asarray(simplices, dtype=np.int64).reshape(-1, 3)
    tp = np.asarray(tile_points, dtype=float).reshape(-1, 2) + np.asarray(
        offset, dtype=float)
    ts = np.asarray(tile_simplices, dtype=np.int64).reshape(-1, 3) + len(pts)

    merged_pts = np.vstack([pts, tp]) if len(pts) else tp
    merged_simp = np.vstack([simp, ts]) if len(simp) else ts
    return weld_points(merged_pts, merged_simp, weld_tol,
                       return_kept=return_kept)


# Perpendicular distance below which a vertex counts as lying *on* a
# triangle edge, and the parametric margin that excludes the edge's own
# endpoints. Welding/snapping makes shared vertices coincide exactly and
# congruent axis-aligned tiles stay collinear, so a hanging node sits on
# the host edge to floating-point precision — a tight tolerance is enough
# and avoids false positives on merely-nearby vertices.
_ON_EDGE_PERP_TOL = 1e-7
_ON_EDGE_END_TOL = 1e-9


def _interior_param(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """If ``p`` lies strictly between ``a`` and ``b`` (off the endpoints,
    within ``_ON_EDGE_PERP_TOL`` of the line), return its parameter ``s`` in
    ``(0, 1)`` along ``a -> b``; otherwise ``None``."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-18:
        return None
    s = float((p - a) @ ab) / L2
    if s <= _ON_EDGE_END_TOL or s >= 1.0 - _ON_EDGE_END_TOL:
        return None
    perp = p - (a + s * ab)
    if float(perp @ perp) > _ON_EDGE_PERP_TOL * _ON_EDGE_PERP_TOL:
        return None
    return s


def split_t_junctions(points: np.ndarray, simplices: np.ndarray, *,
                      max_passes: int = 64, return_parents: bool = False):
    """Make a composed triangulation **conforming** by splitting any edge
    that has another tile's vertex sitting in its interior — a *T-junction*
    (hanging node).

    Tiles of different sizes can't share a full edge: a double-size square's
    edge is twice as long as a unit tile's, so the unit tile's corner lands
    at the *midpoint* of the big edge, where the big tile has no vertex. That
    hanging node leaves the big and small tiles touching at a single corner;
    under the mode-11 bipartite mechanism the under-coupled unit rotates free
    and a point "breaks off". Splitting the host triangle at the hanging node
    turns the single touch into a fully shared edge, so the existing
    bipartite foot/corner fusion couples the tiles into one mechanism — no
    simulation change required.

    Points are never moved or added (the hanging node is already a vertex);
    only the host triangle's connectivity changes. When there is no
    T-junction the simplices are returned unchanged, in the same order, so
    uniform-size compositions stay byte-for-byte identical.

    With ``return_parents=True`` also returns an int array ``parents`` where
    ``parents[i]`` is the index (into the input ``simplices``) of the
    triangle new simplex ``i`` came from, so a caller can carry a
    per-triangle array (e.g. tile/ownership ids) through the split.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    simp = np.asarray(simplices, dtype=np.int64).reshape(-1, 3)
    n_pts = len(pts)
    cur: list[list[int]] = [[int(a), int(b), int(c)] for a, b, c in simp]
    parents: list[int] = list(range(len(cur)))

    for _ in range(int(max_passes)):
        out: list[list[int]] = []
        out_parents: list[int] = []
        changed = False
        for tri, par in zip(cur, parents):
            a, b, c = tri
            tri_set = {a, b, c}
            split = False
            # Try each edge; split the first one carrying interior vertices.
            # Remaining edges are handled on the next pass (cheap, and keeps
            # the fan construction simple).
            for u, w, opp in ((a, b, c), (b, c, a), (c, a, b)):
                interior = []
                for v in range(n_pts):
                    if v in tri_set:
                        continue
                    s = _interior_param(pts[v], pts[u], pts[w])
                    if s is not None:
                        interior.append((s, v))
                if not interior:
                    continue
                interior.sort()
                chain = [u] + [v for _, v in interior] + [w]
                for k in range(len(chain) - 1):
                    out.append([chain[k], chain[k + 1], opp])
                    out_parents.append(par)
                changed = True
                split = True
                break
            if not split:
                out.append([a, b, c])
                out_parents.append(par)
        cur, parents = out, out_parents
        if not changed:
            break

    new_simp = (np.asarray(cur, dtype=np.int64).reshape(-1, 3)
                if cur else np.zeros((0, 3), dtype=np.int64))
    if return_parents:
        return pts, new_simp, np.asarray(parents, dtype=np.int64)
    return pts, new_simp
