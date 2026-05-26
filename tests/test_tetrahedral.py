"""Tests for ``auxetic.tetrahedral`` — the 3D tetrahedral auxetic (mode
12), the volumetric analogue of the 2D bipartite tile.

Each tetrahedron emits five rigid solids: one internal tetra (set B, the
tetra of contracted hinge vertices ``t_i = C·S + (1-C)·P_i``) and four
corner polyhedra (set A, one per corner — the convex solid spanning the
corner, its hinge, three perpendicular edge-feet and three perpendicular
face-feet).
"""

import numpy as np
import pytest

from auxetic.tetrahedral import (
    TetrahedralNetwork,
    TetraPolyhedron,
    build_tetrahedral_network,
)
from auxetic.lattice import Lattice, _3D_MODES, _DELAUNAY_MODES, _TETRAHEDRAL_MODES


def _unit_tetra():
    """The canonical corner tetra on the origin + unit axes."""
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    simps = np.array([[0, 1, 2, 3]])
    return pts, simps


def _regular_tetra():
    """A (near-)regular tetrahedron — symmetric corners for clean tests."""
    pts = np.array([
        [1.0,  1.0,  1.0],
        [1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ])
    simps = np.array([[0, 1, 2, 3]])
    return pts, simps


# ---------------------------------------------------------------------------
# Per-tetrahedron solid emission
# ---------------------------------------------------------------------------

def test_single_tetra_emits_five_solids():
    pts, simps = _unit_tetra()
    net = build_tetrahedral_network(pts, simps, C=0.5)
    assert len(net.polyhedra) == 5
    assert len(net.set_b) == 1      # one internal tetra
    assert len(net.set_a) == 4      # four corner polyhedra


def test_internal_tetra_vertices_are_the_contracted_corners():
    pts, simps = _regular_tetra()
    C = 0.4
    net = build_tetrahedral_network(pts, simps, C=C)
    S = pts.mean(axis=0)
    expected = np.array([C * S + (1.0 - C) * p for p in pts])
    internal = net.set_b[0].vertices
    assert internal.shape == (4, 3)
    assert np.allclose(internal, expected)


def test_internal_tetra_centroid_is_the_tetra_centroid():
    pts, simps = _regular_tetra()
    net = build_tetrahedral_network(pts, simps, C=0.6)
    # Contraction toward S leaves the centroid of the four hinges at S.
    assert np.allclose(net.set_b[0].vertices.mean(axis=0), pts.mean(axis=0))


def test_corner_polyhedron_has_eight_vertices_and_anchors_on_its_corner():
    pts, simps = _unit_tetra()
    net = build_tetrahedral_network(pts, simps, C=0.5)
    corners = net.set_a
    assert len(corners) == 4
    anchored = {int(c.corner_point_index) for c in corners}
    assert anchored == {0, 1, 2, 3}
    for c in corners:
        # corner P, hinge t, 3 edge-feet, 3 face-feet
        assert c.vertices.shape == (8, 3)
        assert np.allclose(c.vertices[0], pts[c.corner_point_index])


def test_corner_edge_points_lie_on_their_incident_edges():
    """The three canonical edge points sit a fixed fraction (``_HINGE_
    FRACTION·C``) of the way from the corner toward each incident edge's
    far corner — defined from the shared edge alone, not the hinge."""
    from auxetic.tetrahedral import _HINGE_FRACTION
    pts, simps = _regular_tetra()
    C = 0.5
    net = build_tetrahedral_network(pts, simps, C=C)
    f = _HINGE_FRACTION * C
    for c in net.set_a:
        ci = int(c.corner_point_index)
        others = [o for o in range(4) if o != ci]
        edge_pts = c.vertices[2:5]      # [P, t, edge(3), face(3)]
        expected = [pts[ci] + f * (pts[o] - pts[ci]) for o in others]
        assert np.allclose(edge_pts, expected)


def test_shared_face_corner_polyhedra_fuse_into_a_hinge():
    """Two tetrahedra sharing a face compute IDENTICAL canonical points on
    that face and its edges, so their corner polyhedra at a shared corner
    meet at ≥2 coincident vertices — a revolute hinge, not a ball joint.
    This cross-tetra fusion is what makes the 3D mechanism coherent (the
    point-joint MVP shared only the single corner, leaving it floppy)."""
    pts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.3, 0.3, -1.0],
    ])
    simps = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])   # share face {0, 1, 2}
    net = build_tetrahedral_network(pts, simps, C=0.5)
    c0 = [p for p in net.set_a if p.corner_point_index == 0]
    assert len(c0) == 2
    va, vb = c0[0].vertices, c0[1].vertices
    shared = sum(1 for a in va
                 if any(np.allclose(a, b, atol=1e-9) for b in vb))
    assert shared >= 2      # shared corner P_0 + ≥1 shared face/edge point


def test_corner_polyhedra_are_non_degenerate_at_mid_C():
    """At a mid contraction the eight corner points span a real volume."""
    from scipy.spatial import ConvexHull
    pts, simps = _regular_tetra()
    net = build_tetrahedral_network(pts, simps, C=0.5)
    for c in net.set_a:
        hull = ConvexHull(c.vertices)
        assert hull.volume > 1e-6


def test_C_zero_collapses_corners_and_internal_equals_full_tetra():
    pts, simps = _unit_tetra()
    net = build_tetrahedral_network(pts, simps, C=0.0)
    # Internal tetra == the original corners.
    assert np.allclose(net.set_b[0].vertices, pts)
    # Every corner solid collapses onto its corner (hinge == corner, so
    # all perpendicular feet land on the corner too).
    for c in net.set_a:
        assert np.allclose(c.vertices, pts[c.corner_point_index])


def test_C_one_collapses_internal_tetra_to_the_centroid():
    pts, simps = _regular_tetra()
    net = build_tetrahedral_network(pts, simps, C=1.0)
    S = pts.mean(axis=0)
    assert np.allclose(net.set_b[0].vertices, np.tile(S, (4, 1)))


def test_invalid_C_raises():
    pts, simps = _unit_tetra()
    with pytest.raises(ValueError):
        build_tetrahedral_network(pts, simps, C=-0.1)
    with pytest.raises(ValueError):
        build_tetrahedral_network(pts, simps, C=1.5)


def test_input_shape_validation():
    with pytest.raises(ValueError):
        build_tetrahedral_network(np.zeros((4, 2)), np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError):
        build_tetrahedral_network(np.zeros((4, 3)), np.array([[0, 1, 2]]))


def test_multiple_tetrahedra_emit_five_each():
    # Two tetrahedra sharing a face (the (0,1,2) triangle).
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.3, 0.3, -1.0],
    ])
    simps = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    net = build_tetrahedral_network(pts, simps, C=0.5)
    assert len(net.polyhedra) == 10
    assert {p.tetra_index for p in net.polyhedra} == {0, 1}


# ---------------------------------------------------------------------------
# Lattice integration (mode 12)
# ---------------------------------------------------------------------------

def test_mode_12_is_registered_as_3d_delaunay_tetrahedral():
    assert 12 in _3D_MODES
    assert 12 in _DELAUNAY_MODES
    assert 12 in _TETRAHEDRAL_MODES


def test_mode_12_lattice_generates_3d_points_and_tetrahedra():
    lat = Lattice(mode=12, n_points=12, ratio=0.35, seed=3, C=0.5)
    assert lat.points.ndim == 2 and lat.points.shape[1] == 3
    simplices = np.asarray(lat.tri.simplices)
    assert simplices.shape[1] == 4      # tetrahedra, not triangles


def test_mode_12_lattice_builds_export_triangles():
    lat = Lattice(mode=12, n_points=12, ratio=0.35, seed=3, C=0.5)
    tris = lat.build_export_triangles(verbose=False)
    assert len(tris) > 0
    assert np.asarray(tris[0]).shape == (3, 3)


def test_mode_12_C_change_changes_geometry():
    # C=0 collapses the corner polyhedra to points (only the internal
    # tetra renders); a mid C opens them into real solids. So flipping C
    # and clearing the cache must change the triangle count — proving the
    # live C drives the (rebuilt-each-call) mode-12 geometry.
    lat = Lattice(mode=12, n_points=12, ratio=0.35, seed=3, C=0.0)
    n_low = len(lat.build_export_triangles(verbose=False))
    lat.C = 0.5
    lat._clear_caches()
    n_high = len(lat.build_export_triangles(verbose=False))
    assert n_high > n_low


def test_mode_12_to_stl_writes_a_file(tmp_path):
    lat = Lattice(mode=12, n_points=12, ratio=0.35, seed=5, C=0.5)
    out = tmp_path / "tetra.stl"
    lat.to_stl(str(out), verbose=False)
    assert out.exists() and out.stat().st_size > 0


def test_mode_12_kirigami_emits_tiles_and_constraints():
    """Mode 12 now has a kirigami model: collect_kirigami yields one
    internal tetra + four corner polyhedra per tetrahedron, plus the
    hinge constraints from their fused vertices. (Earlier it raised — the
    rotating-polyhedra mechanism was deferred.)"""
    lat = Lattice(mode=12, n_points=12, ratio=0.35, seed=5, C=0.5)
    n_tetra = len(np.asarray(lat.tri.simplices))
    tiles, source, constraints = lat.collect_kirigami()
    assert len(tiles) == 5 * n_tetra
    assert len(constraints) > 0


def test_mode_12_out_of_range_C_renders_without_crashing():
    """A stale mode-11-style C (e.g. 2.0) on a mode-12 lattice must clamp
    in the render path rather than raise (the build core stays strict)."""
    lat = Lattice(mode=12, n_points=10, ratio=0.35, seed=1, C=2.0)
    tris = lat.build_export_triangles(verbose=False)
    assert len(tris) > 0
