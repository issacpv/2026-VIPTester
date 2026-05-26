"""Tests for the compose-from-tiles geometry (pure, no GUI).

``auxetic.composition`` welds near-coincident vertices and appends tiles;
``auxetic.tile_library`` provides the 2D templates. Together they back
the Tile Library drag-and-drop workflow: dropping a tile next to existing
ones fuses shared vertices/edges so the mode-11 bipartite generator
renders one big auxetic shape.
"""

import numpy as np
import pytest

from auxetic.composition import (
    DEFAULT_WELD_TOL,
    SNAP_RADIUS,
    add_tile,
    snap_tile_offset,
    weld_points,
)
from auxetic.lattice import Lattice
from auxetic.tile_library import TILE_LIBRARY, get_tile


# ---------------------------------------------------------------------------
# weld_points
# ---------------------------------------------------------------------------

def test_weld_merges_coincident_points_and_remaps():
    # Two triangles, the second a copy of the first shifted so one vertex
    # lands exactly on a vertex of the first.
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],   # tri 0
                    [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])   # tri 1 (shares (1,0))
    simp = np.array([[0, 1, 2], [3, 4, 5]])
    new_pts, new_simp = weld_points(pts, simp, tol=1e-6)
    assert len(new_pts) == 5           # (1,0) merged: 6 → 5
    assert len(new_simp) == 2          # both triangles survive
    # Every simplex index is in range and no triangle is degenerate.
    assert new_simp.max() < len(new_pts)
    for tri in new_simp:
        assert len(set(int(v) for v in tri)) == 3


def test_weld_drops_degenerate_triangles():
    # A triangle whose two corners weld together collapses and is dropped.
    pts = np.array([[0.0, 0.0], [1e-9, 0.0], [0.5, 1.0]])
    simp = np.array([[0, 1, 2]])
    new_pts, new_simp = weld_points(pts, simp, tol=1e-3)
    assert len(new_pts) == 2
    assert len(new_simp) == 0


def test_weld_no_merges_leaves_mesh_unchanged():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    simp = np.array([[0, 1, 2]])
    new_pts, new_simp = weld_points(pts, simp, tol=1e-6)
    assert len(new_pts) == 3
    assert len(new_simp) == 1


def test_weld_empty_is_safe():
    new_pts, new_simp = weld_points(np.zeros((0, 2)), np.zeros((0, 3), int))
    assert new_pts.shape == (0, 2)
    assert new_simp.shape == (0, 3)


# ---------------------------------------------------------------------------
# add_tile
# ---------------------------------------------------------------------------

def test_add_tile_to_empty_is_just_the_tile():
    tri = get_tile("Triangle")
    pts, simp = add_tile(np.zeros((0, 2)), np.zeros((0, 3), int),
                         tri.points, tri.simplices, offset=(0.5, 0.5))
    assert len(pts) == 3
    assert len(simp) == 1
    # Centred template + offset places the centroid at the drop point.
    assert np.allclose(pts.mean(axis=0), [0.5, 0.5], atol=1e-9)


def test_two_squares_share_an_edge_when_translated_one_edge():
    sq = get_tile("Square")
    edge = float(sq.points[:, 0].max() - sq.points[:, 0].min())
    pts, simp = add_tile(np.zeros((0, 2)), np.zeros((0, 3), int),
                         sq.points, sq.simplices, offset=(0.0, 0.0))
    pts, simp = add_tile(pts, simp, sq.points, sq.simplices,
                         offset=(edge, 0.0))
    # 4 + 4 − 2 welded shared-edge vertices = 6 points; 2 + 2 = 4 triangles.
    assert len(pts) == 6
    assert len(simp) == 4


def test_up_and_down_triangles_fuse_into_a_rhombus():
    up = get_tile("Triangle")
    down = get_tile("Triangle-down")
    pts, simp = add_tile(np.zeros((0, 2)), np.zeros((0, 3), int),
                         up.points, up.simplices, offset=(0.0, 0.0))
    # Translate the down-triangle so its top edge lands on the up-triangle's
    # bottom edge → two vertices weld → a 4-point rhombus.
    off_y = float(up.points[:, 1].min() - down.points[:, 1].max())
    pts, simp = add_tile(pts, simp, down.points, down.simplices,
                         offset=(0.0, off_y), weld_tol=DEFAULT_WELD_TOL)
    assert len(pts) == 4
    assert len(simp) == 2


# ---------------------------------------------------------------------------
# snap_tile_offset — drops lock onto existing geometry (so they land level)
# ---------------------------------------------------------------------------

def test_snap_pulls_a_nearby_drop_onto_an_existing_vertex():
    sq = get_tile("Square")
    base = sq.points + np.array([0.4, 0.5])      # one square already placed
    # Drop the next square ~0.09 off the ideal edge-adjacent position.
    snapped = snap_tile_offset(base, sq.points, (0.73, 0.55))
    placed = sq.points + snapped
    # After snapping, some placed vertex coincides *exactly* with a base one.
    dmin = np.min(np.linalg.norm(
        placed[:, None, :] - base[None, :, :], axis=2))
    assert dmin < 1e-9


def test_snap_is_noop_when_nothing_is_close():
    sq = get_tile("Square")
    base = sq.points + np.array([0.4, 0.5])
    far = (0.4 + 5 * SNAP_RADIUS, 0.5 + 5 * SNAP_RADIUS)
    assert np.allclose(snap_tile_offset(base, sq.points, far), far)


def test_snap_is_noop_with_no_existing_points():
    sq = get_tile("Square")
    assert np.allclose(
        snap_tile_offset(np.zeros((0, 2)), sq.points, (0.3, 0.3)), (0.3, 0.3))


def test_sloppy_second_drop_still_welds_level():
    """A roughly-aimed second drop must snap + weld into the same 6-point
    mesh a precise drop would give — i.e. it lands level, not skewed."""
    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4, 0.5))
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.73, 0.55))  # sloppy
    assert lat.n_points == 6
    assert len(lat.tri.simplices) == 4


# ---------------------------------------------------------------------------
# tile_library templates
# ---------------------------------------------------------------------------

def test_library_has_the_expected_tiles():
    assert set(TILE_LIBRARY) == {"Triangle", "Triangle-down", "Square", "Hexagon"}


@pytest.mark.parametrize("name", ["Triangle", "Triangle-down", "Square", "Hexagon"])
def test_template_is_valid_and_centred(name):
    t = get_tile(name)
    assert t.points.shape[1] == 2
    assert t.simplices.shape[1] == 3
    # Indices in range.
    assert int(t.simplices.max()) < t.n_points
    # Centred on centroid.
    assert np.allclose(t.points.mean(axis=0), [0.0, 0.0], atol=1e-9)
    # Every triangle has positive area (non-degenerate).
    for tri in t.simplices:
        a, b, c = t.points[tri]
        # 2D scalar cross (np.cross on 2-vectors is deprecated in NumPy 2).
        area = 0.5 * abs(float((b[0] - a[0]) * (c[1] - a[1])
                               - (b[1] - a[1]) * (c[0] - a[0])))
        assert area > 1e-6


# ---------------------------------------------------------------------------
# Lattice.compose_add_tile integration
# ---------------------------------------------------------------------------

def _square_edge() -> float:
    sq = get_tile("Square")
    return float(sq.points[:, 0].max() - sq.points[:, 0].min())


def test_compose_add_tile_seeds_then_welds():
    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    edge = _square_edge()
    # First drop seeds a fresh composition and enters mode 11.
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4, 0.5))
    assert lat.mode == 11
    assert lat.preserve_triangulation is True
    assert lat.n_points == 4
    assert len(lat.tri.simplices) == 2
    # Second drop one edge over → shares an edge → 2 vertices weld.
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4 + edge, 0.5))
    assert lat.n_points == 6
    assert len(lat.tri.simplices) == 4


def test_compose_point_move_preserves_triangulation():
    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.5, 0.5))
    before = np.asarray(lat.tri.simplices).copy()
    moved = lat.points.copy()
    moved[0] += [0.01, 0.0]
    lat.regenerate_from_points(moved)   # a 2D Delaunay mode would re-triangulate
    assert np.array_equal(before, np.asarray(lat.tri.simplices))


def test_regenerate_exits_composition():
    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.5, 0.5))
    assert lat.preserve_triangulation is True
    lat.regenerate()   # explicit re-roll
    assert lat.preserve_triangulation is False


def test_composed_lattice_builds_bipartite_and_exports():
    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.5, 0.5))
    net = lat.build_bipartite()
    # 2 triangles × (1 central + 3 corner) = 8 polygons.
    assert len(net.polygons) == 8
    tris = lat.build_export_triangles(verbose=False)
    assert len(tris) > 0


def test_scale_points_scales_about_centroid_preserving_triangulation():
    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.5, 0.5))
    p0 = lat.points.copy()
    simp0 = np.asarray(lat.tri.simplices).copy()
    c0 = p0.mean(axis=0)

    lat.scale_points(3.0)
    assert np.allclose(lat.points, c0 + (p0 - c0) * 3.0)
    assert np.allclose(lat.points.mean(axis=0), c0)             # centroid fixed
    assert np.array_equal(simp0, np.asarray(lat.tri.simplices))  # tri preserved
    assert lat.preserve_triangulation is True

    lat.scale_points(1.0 / 3.0)
    assert np.allclose(lat.points, p0)                          # exactly reversible


def _edge_set(lat):
    s = set()
    for t in lat.tri.simplices:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            s.add(tuple(sorted((int(a), int(b)))))
    return s


def test_flip_composed_edge_swaps_the_diagonal():
    """Edge flip on a composed (preserve_triangulation) mesh must actually
    swap the diagonal — the bug was that _triangulate short-circuits and
    never applied edge_flips, so the flip recorded (red) but never took."""
    from auxetic import geometry as _geom

    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    e = float(sq.points[:, 0].max() - sq.points[:, 0].min())
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4, 0.5))
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4 + e, 0.5))

    flips = _geom.flippable_edges(lat.tri, lat.points)
    assert flips, "expected a flippable diagonal in the composed mesh"
    edge = tuple(sorted(flips[0]))
    assert edge in _edge_set(lat)

    assert lat.flip_composed_edge(edge) is True
    assert edge not in _edge_set(lat)              # diagonal actually swapped
    assert lat.preserve_triangulation is True      # still a composition


def test_scale_points_on_delaunay_lattice_is_uniform():
    lat = Lattice(mode=1, n_points=8, seed=2)
    p0 = lat.points.copy()
    c0 = p0.mean(axis=0)
    lat.scale_points(2.5)
    assert np.allclose(lat.points, c0 + (p0 - c0) * 2.5)


def test_mode11_sweep_stops_at_real_collision_not_analytic_jamming():
    """A composed mode-11 lattice's kinematic sweep must bound the rotation
    at the first real polygon collision (the physical jamming limit), not
    over-rotate to the larger analytic jamming angle and collapse into
    self-overlap."""
    from auxetic.simulation import Simulator, TileSystem
    from auxetic.collision import CollisionChecker

    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    e = float(sq.points[:, 0].max() - sq.points[:, 0].min())
    for off in [(0.4, 0.5), (0.4 + e, 0.5), (0.4, 0.5 - e), (0.4 + e, 0.5 - e)]:
        lat.compose_add_tile(sq.points, sq.simplices, offset=off)

    jam = float(lat.bipartite_jamming_angle())
    ts = TileSystem.from_lattice(lat)
    sim = Simulator(ts, np.array([0.0, -1.0]))
    res = sim.sweep_mechanism(max_actuation=jam, collision_stop=True)

    max_act = float(np.abs(np.asarray(res.theta_samples)).max())
    # Stops strictly short of the analytic jamming angle ...
    assert max_act < jam - 1e-3
    # ... and every reachable pose is collision-free (no collapse).
    cc = CollisionChecker(ts, tol=1e-6)
    assert not any(cc.has_collision(p) for p in res.poses)
