"""Tests for the per-edge Delaunay flip helpers in ``auxetic.geometry``.

The flip primitive is 2D-only in M1 (3D tetrahedral flips are
intentionally deferred). These tests verify:

- ``flippable_edges`` returns only interior edges of strictly convex
  quads, sorted, with ``i < j``.
- ``apply_edge_flips`` is a no-op for empty / unrecognised flip sets.
- A single flip changes ``simplices`` and gives proper CCW orientation.
- Flipping an edge then flipping the new edge returns to the original
  triangulation (involution via the *new* diagonal, not by re-applying
  the same canonical edge twice).
- 3D inputs and unsupported axes return empty / unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay

from auxetic.geometry import (
    _FlippedTri,
    _orient_ccw,
    _quad_is_strictly_convex,
    apply_edge_flips,
    edge_flip_apexes,
    flippable_edges,
    generate_points,
)


# ---------------------------------------------------------------------------
# A canonical 4-point square: two ways to triangulate, one flippable edge.
#
#   3 -- 2
#   |  / |       canonical Delaunay picks diagonal 1-3 or 0-2 depending on
#   | /  |       jitter; either way the OTHER diagonal is the only flip.
#   0 -- 1
# ---------------------------------------------------------------------------

SQUARE = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
])


def _square_tri():
    return Delaunay(SQUARE)


# ---------------------------------------------------------------------------
# flippable_edges
# ---------------------------------------------------------------------------

def test_flippable_square_returns_one_edge():
    tri = _square_tri()
    edges = flippable_edges(tri, SQUARE)
    # There is exactly one shared diagonal between the two triangles.
    assert len(edges) == 1
    a, b = edges[0]
    assert a < b
    # The diagonal is one of the two square diagonals: (0,2) or (1,3).
    assert (a, b) in [(0, 2), (1, 3)]


def test_flippable_edges_are_sorted_with_i_less_than_j():
    pts = np.random.RandomState(42).rand(10, 2)
    tri = Delaunay(pts)
    edges = flippable_edges(tri, pts)
    assert edges == sorted(edges)
    for a, b in edges:
        assert a < b


def test_flippable_edges_excludes_boundary():
    pts = np.random.RandomState(42).rand(10, 2)
    tri = Delaunay(pts)
    edges = flippable_edges(tri, pts)
    # Build the boundary edge set as ConvexHull edges; those must NOT be flippable.
    from scipy.spatial import ConvexHull
    hull_edges = set()
    hull = ConvexHull(pts)
    for s in hull.simplices:
        a, b = sorted((int(s[0]), int(s[1])))
        hull_edges.add((a, b))
    for e in edges:
        assert e not in hull_edges


def test_flippable_returns_empty_for_3d_points():
    pts = np.random.RandomState(0).rand(8, 3)
    tri = Delaunay(pts)
    assert flippable_edges(tri, pts) == []


def test_flippable_returns_empty_for_no_simplices():
    class EmptyTri:
        simplices = np.zeros((0, 3), dtype=int)
    assert flippable_edges(EmptyTri(), SQUARE) == []


def test_quad_strictly_convex_helper_directions():
    pa = np.array([0.0, 0.0]); pb = np.array([1.0, 1.0])
    pc = np.array([1.0, 0.0]); pd = np.array([0.0, 1.0])
    # c and d on opposite sides of (a,b) — convex quad
    assert _quad_is_strictly_convex(pa, pb, pc, pd)
    # When c and d are on the same side the quad is non-convex.
    pd_same_side = np.array([0.5, -0.2])
    assert not _quad_is_strictly_convex(pa, pb, pc, pd_same_side)


# ---------------------------------------------------------------------------
# apply_edge_flips
# ---------------------------------------------------------------------------

def test_apply_empty_flips_returns_input_unchanged():
    tri = _square_tri()
    out = apply_edge_flips(tri, SQUARE, set())
    # Same object — no allocation done
    assert out is tri


def test_apply_unknown_edges_returns_unmodified_simplices():
    tri = _square_tri()
    out = apply_edge_flips(tri, SQUARE, {(0, 99)})
    np.testing.assert_array_equal(np.sort(out.simplices, axis=1),
                                   np.sort(tri.simplices, axis=1))


def test_single_flip_changes_simplices_on_square():
    tri = _square_tri()
    edges = flippable_edges(tri, SQUARE)
    assert len(edges) == 1
    flipped = apply_edge_flips(tri, SQUARE, {edges[0]})
    # The two triangles should now share the *other* diagonal.
    orig_set = {tuple(sorted(int(v) for v in s)) for s in tri.simplices}
    new_set  = {tuple(sorted(int(v) for v in s)) for s in flipped.simplices}
    assert orig_set != new_set
    assert len(new_set) == 2  # still two triangles


def test_flipping_new_diagonal_returns_to_original():
    tri = _square_tri()
    a, b = flippable_edges(tri, SQUARE)[0]
    flipped = apply_edge_flips(tri, SQUARE, {(a, b)})
    new_edges = flippable_edges(flipped, SQUARE)
    assert len(new_edges) == 1  # same shape, one flippable diagonal
    re_flipped = apply_edge_flips(flipped, SQUARE, set(new_edges))
    orig_set = {tuple(sorted(int(v) for v in s)) for s in tri.simplices}
    re_set   = {tuple(sorted(int(v) for v in s)) for s in re_flipped.simplices}
    assert orig_set == re_set


def test_flipped_simplices_are_ccw_oriented():
    tri = _square_tri()
    edges = flippable_edges(tri, SQUARE)
    flipped = apply_edge_flips(tri, SQUARE, set(edges))
    for s in flipped.simplices:
        a, b, c = (int(v) for v in s)
        u = SQUARE[b] - SQUARE[a]
        v = SQUARE[c] - SQUARE[a]
        cross = float(u[0]) * float(v[1]) - float(u[1]) * float(v[0])
        assert cross >= 0.0, f"triangle {s} is CW (cross={cross})"


def test_apply_flips_returns_simplices_proxy_with_correct_attribute():
    tri = _square_tri()
    out = apply_edge_flips(tri, SQUARE, set(flippable_edges(tri, SQUARE)))
    assert isinstance(out, _FlippedTri)
    assert hasattr(out, "simplices")
    assert out.simplices.shape == tri.simplices.shape


def test_flips_do_not_create_new_or_remove_old_vertices():
    pts = np.random.RandomState(7).rand(12, 2)
    tri = Delaunay(pts)
    edges = flippable_edges(tri, pts)
    flipped = apply_edge_flips(tri, pts, set(edges))
    used_before = set(int(v) for s in tri.simplices for v in s)
    used_after  = set(int(v) for s in flipped.simplices for v in s)
    assert used_before == used_after


# ---------------------------------------------------------------------------
# Integration with generate_points: edge flips work on the random Delaunay
# output, which is the realistic use case in the GUI.
# ---------------------------------------------------------------------------

def test_generate_points_then_flip():
    np.random.seed(123)
    pts, tri = generate_points(20, 1)
    edges = flippable_edges(tri, pts)
    assert len(edges) > 0
    flipped = apply_edge_flips(tri, pts, {edges[0]})
    assert flipped.simplices.shape == tri.simplices.shape
    # At least the first flipped triangle differs.
    assert not np.array_equal(flipped.simplices, tri.simplices)


def test_orient_ccw_swaps_when_needed():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # CCW: a, b, c with b at +x and c at +y → cross (b-a) × (c-a) > 0
    assert _orient_ccw([0, 1, 2], pts) == [0, 1, 2]
    # CW: swap b and c → should flip
    assert _orient_ccw([0, 2, 1], pts) == [0, 1, 2]


# ---------------------------------------------------------------------------
# edge_flip_apexes — the corner pair a flip would connect.
# ---------------------------------------------------------------------------

def test_edge_flip_apexes_square_returns_other_diagonal():
    """For the canonical square, the apexes of the current diagonal are
    exactly the two endpoints of the *other* diagonal — and together
    the four indices span all of {0, 1, 2, 3}."""
    tri = _square_tri()
    edge = flippable_edges(tri, SQUARE)[0]
    apx = edge_flip_apexes(tri, edge)
    assert apx is not None
    c, d = apx
    assert c < d                                  # returned sorted
    assert {edge[0], edge[1], c, d} == {0, 1, 2, 3}
    # The apex pair is the OTHER square diagonal.
    assert (c, d) in [(0, 2), (1, 3)]
    assert (c, d) != tuple(sorted(edge))


def test_edge_flip_apexes_accepts_unsorted_edge():
    tri = _square_tri()
    a, b = flippable_edges(tri, SQUARE)[0]
    # Passing the edge reversed must give the same apex pair.
    assert edge_flip_apexes(tri, (b, a)) == edge_flip_apexes(tri, (a, b))


def test_edge_flip_apexes_boundary_edge_returns_none():
    """A hull (boundary) edge borders only one triangle — no flip,
    so no apex pair."""
    from scipy.spatial import ConvexHull
    tri = _square_tri()
    hull = ConvexHull(SQUARE)
    a, b = sorted((int(hull.simplices[0][0]), int(hull.simplices[0][1])))
    assert edge_flip_apexes(tri, (a, b)) is None


def test_edge_flip_apexes_unknown_edge_returns_none():
    tri = _square_tri()
    assert edge_flip_apexes(tri, (0, 99)) is None


def test_edge_flip_apexes_3d_returns_none():
    pts = np.random.RandomState(0).rand(8, 3)
    tri = Delaunay(pts)
    assert edge_flip_apexes(tri, (0, 1)) is None


def test_edge_flip_apexes_matches_apply_edge_flips_result():
    """The apexes reported for an edge must be exactly the new edge
    that ``apply_edge_flips`` introduces when that edge is flipped."""
    np.random.seed(321)
    pts, tri = generate_points(16, 1)
    edge = flippable_edges(tri, pts)[0]
    apx = edge_flip_apexes(tri, edge)
    assert apx is not None
    flipped = apply_edge_flips(tri, pts, {edge})
    # The flipped triangulation should contain the apex pair as an edge.
    new_edges = set()
    for s in flipped.simplices:
        v = [int(x) for x in s]
        for k in range(3):
            e = tuple(sorted((v[k], v[(k + 1) % 3])))
            new_edges.add(e)
    assert apx in new_edges
