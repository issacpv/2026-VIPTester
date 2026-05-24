"""Tests for ``auxetic.bipartite`` — the per-triangle auxetic tile.

Each triangle emits four rigid polygons: one central polygon (set B,
the triangle of centroid hinges) and three corner kites (set A, one per
corner, the quad ``[P_i, E_ij, T_i, E_ik]``). Adjacent triangles fuse on
shared edges because they place the shared corner node and the shared
edge points at the same locations.
"""

import numpy as np
import pytest

from auxetic.bipartite import (
    BipartiteNetwork,
    BipartitePolygon,
    build_bipartite_network,
    jamming_angle,
)


def _equilateral():
    """A single upward triangle: apex on top, two base corners."""
    pts = np.array([
        [0.5, 0.9],   # P0 apex
        [0.1, 0.1],   # P1 base-left
        [0.9, 0.1],   # P2 base-right
    ])
    simps = np.array([[0, 1, 2]])
    return pts, simps


def _symmetric_rhombus():
    """Two triangles sharing the P0-P2 diagonal, symmetric about it so
    the shared-edge points from both triangles coincide (clean fusion)."""
    pts = np.array([
        [0.0,  0.0],   # P0 shared
        [1.0,  1.0],   # P1 top tip
        [2.0,  0.0],   # P2 shared
        [1.0, -1.0],   # P3 bottom tip
    ])
    simps = np.array([
        [0, 1, 2],     # triangle 0
        [0, 2, 3],     # triangle 1
    ])
    return pts, simps


# ---------------------------------------------------------------------------
# Per-triangle polygon emission
# ---------------------------------------------------------------------------

def test_single_triangle_emits_four_polygons():
    pts, simps = _equilateral()
    net = build_bipartite_network(pts, simps, C=1.0)
    assert len(net.polygons) == 4
    assert len(net.set_b) == 1     # one central polygon
    assert len(net.set_a) == 3     # three corner kites


def test_central_polygon_is_a_triangle():
    pts, simps = _equilateral()
    net = build_bipartite_network(pts, simps, C=1.0)
    (central,) = net.set_b
    assert central.degree == 3
    assert central.kind == "central"


def test_corner_kites_are_quads():
    pts, simps = _equilateral()
    net = build_bipartite_network(pts, simps, C=1.0)
    assert len(net.set_a) == 3
    for kite in net.set_a:
        assert kite.degree == 4
        assert kite.kind == "corner"
        # The kite is anchored at a real lattice corner.
        assert 0 <= kite.corner_point_index < len(pts)
        assert np.allclose(kite.node_xy, pts[kite.corner_point_index])


def test_kite_contains_its_node_and_hinge():
    """Each kite must include the corner node P_i and the centroid hinge
    T_i = P_i + t(M - P_i) among its vertices."""
    pts, simps = _equilateral()
    M = pts.mean(axis=0)
    t = 0.5  # C = 1
    net = build_bipartite_network(pts, simps, C=1.0)
    for kite in net.set_a:
        Pi = pts[kite.corner_point_index]
        Ti = Pi + t * (M - Pi)
        verts = kite.vertices
        assert np.any(np.all(np.isclose(verts, Pi), axis=1)), "node missing"
        assert np.any(np.all(np.isclose(verts, Ti), axis=1)), "hinge missing"


def test_kite_hinge_index_points_to_the_hinge():
    pts, simps = _equilateral()
    M = pts.mean(axis=0)
    t = 0.5
    net = build_bipartite_network(pts, simps, C=1.0)
    for kite in net.set_a:
        Pi = pts[kite.corner_point_index]
        Ti = Pi + t * (M - Pi)
        assert kite.hinge_index >= 0
        assert np.allclose(kite.vertices[kite.hinge_index], Ti)
        # inner_edges() returns the two edges incident to that hinge.
        edges = kite.inner_edges()
        assert len(edges) == 2
        for e in edges:
            # The hinge is an endpoint of each inner edge.
            assert np.allclose(e[0], Ti) or np.allclose(e[1], Ti)


def test_inner_edges_are_perpendicular_to_faces():
    """Each kite's inner edge (foot -> hinge) must be perpendicular to
    the triangle edge it sits on — the whole point of the rebuild.
    Tested on a scalene triangle so it isn't an equilateral fluke."""
    pts = np.array([[0.5, 0.95], [0.05, 0.1], [0.8, 0.05]])
    simps = np.array([[0, 1, 2]])
    net = build_bipartite_network(pts, simps, C=1.3)
    for kite in net.set_a:
        ci = kite.corner_point_index
        Pi = pts[ci]
        others = [pts[i] for i in range(3) if i != ci]
        for edge in kite.inner_edges():
            d = edge[1] - edge[0]            # foot -> hinge
            # Perpendicular to (at least) one incident triangle edge.
            perps = [abs(float(np.dot(d, (o - Pi)))) for o in others]
            assert min(perps) < 1e-9


def test_hinge_is_shared_between_kite_and_central():
    """The defining connection: every centroid hinge T_i is a vertex of
    BOTH the central polygon and the corresponding corner kite — that's
    where the kite pivots against the central polygon."""
    pts, simps = _equilateral()
    net = build_bipartite_network(pts, simps, C=1.0)
    (central,) = net.set_b

    def has_vertex(poly, v):
        return bool(np.any(np.all(np.isclose(poly.vertices, v), axis=1)))

    M = pts.mean(axis=0)
    t = 0.5
    for kite in net.set_a:
        Pi = pts[kite.corner_point_index]
        Ti = Pi + t * (M - Pi)
        assert has_vertex(central, Ti)
        assert has_vertex(kite, Ti)


# ---------------------------------------------------------------------------
# C controls hinge position / kite size
# ---------------------------------------------------------------------------

def test_C_one_places_hinges_at_corner_centroid_midpoint():
    pts, simps = _equilateral()
    M = pts.mean(axis=0)
    net = build_bipartite_network(pts, simps, C=1.0)
    (central,) = net.set_b
    expected = np.array([0.5 * (pts[i] + M) for i in range(3)])
    # Central vertices are the three midpoints (order-independent).
    for e in expected:
        assert np.any(np.all(np.isclose(central.vertices, e), axis=1))


def test_smaller_C_grows_the_kites():
    """Kite arm length is t·|M - P_i| with t = 1/(1+C); smaller C → larger
    t → larger kites (the 'solid' limit). Compare kite areas."""
    pts, simps = _equilateral()

    def total_kite_area(C):
        net = build_bipartite_network(pts, simps, C=C)
        area = 0.0
        for k in net.set_a:
            v = k.vertices
            area += abs(0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1)
                                     - np.roll(v[:, 0], -1) * v[:, 1]))
        return area

    assert total_kite_area(0.3) > total_kite_area(1.0) > total_kite_area(4.0)


# ---------------------------------------------------------------------------
# Fusion across a shared edge
# ---------------------------------------------------------------------------

def test_bonds_one_per_edge_and_lie_on_triangle_edges():
    pts, simps = _equilateral()
    net = build_bipartite_network(pts, simps, C=1.0)
    assert len(net.bonds) == 3        # one per triangle edge

    def cross(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    tri_edges = [(0, 1), (1, 2), (2, 0)]
    for bond in net.bonds:
        assert bond.shape == (2, 2)
        on_some_edge = False
        for (i, j) in tri_edges:
            A, B = pts[i], pts[j]
            if (abs(cross(B - A, bond[0] - A)) < 1e-9
                    and abs(cross(B - A, bond[1] - A)) < 1e-9):
                on_some_edge = True
                break
        assert on_some_edge, "bond endpoints are not collinear with a triangle edge"


def test_rhombus_has_six_bonds():
    pts, simps = _symmetric_rhombus()
    net = build_bipartite_network(pts, simps, C=1.0)
    assert len(net.bonds) == 6        # 2 triangles × 3 edges


def test_rhombus_emits_eight_polygons():
    pts, simps = _symmetric_rhombus()
    net = build_bipartite_network(pts, simps, C=1.0)
    assert len(net.polygons) == 8       # 2 triangles × 4
    assert len(net.set_b) == 2
    assert len(net.set_a) == 6


def test_shared_corner_kites_fuse_on_the_diagonal():
    """The two triangles both own corner P0 (and P2). Their kites at a
    shared corner must share the corner node AND the edge point that
    lies on the shared diagonal — that coincidence is the fusion."""
    pts, simps = _symmetric_rhombus()
    net = build_bipartite_network(pts, simps, C=1.0)

    # The two kites anchored at corner index 0, one from each triangle.
    p0_kites = [k for k in net.set_a if k.corner_point_index == 0]
    assert len(p0_kites) == 2
    ka, kb = p0_kites
    assert ka.triangle_index != kb.triangle_index

    va = {tuple(np.round(v, 9)) for v in ka.vertices}
    vb = {tuple(np.round(v, 9)) for v in kb.vertices}
    shared = va & vb
    # Shared: the corner node P0 itself, plus the edge point on the
    # shared P0-P2 diagonal.
    assert tuple(np.round(pts[0], 9)) in shared
    assert len(shared) == 2


# ---------------------------------------------------------------------------
# Kinematic rotation (central fixed, kites rotate about hinges)
# ---------------------------------------------------------------------------

def test_theta_zero_matches_rest():
    pts, simps = _equilateral()
    rest = build_bipartite_network(pts, simps, C=1.0)
    same = build_bipartite_network(pts, simps, C=1.0, theta=0.0)
    for a, b in zip(rest.polygons, same.polygons):
        assert np.allclose(a.vertices, b.vertices)


def test_rotation_holds_central_polygon_fixed():
    pts, simps = _equilateral()
    rest = build_bipartite_network(pts, simps, C=1.0)
    rot  = build_bipartite_network(pts, simps, C=1.0, theta=0.3)
    # The central polygon (set B) is identical — it does not move.
    (cr,) = rest.set_b
    (cc,) = rot.set_b
    assert np.allclose(cr.vertices, cc.vertices)


def test_rotation_spins_each_kite_about_its_hinge():
    pts, simps = _equilateral()
    theta = 0.25
    rest = build_bipartite_network(pts, simps, C=1.0)
    rot  = build_bipartite_network(pts, simps, C=1.0, theta=theta)
    c, s = np.cos(theta), np.sin(theta)
    Rm = np.array([[c, -s], [s, c]])
    rest_by_corner = {k.corner_point_index: k for k in rest.set_a}
    for kite in rot.set_a:
        rk = rest_by_corner[kite.corner_point_index]
        hinge = kite.vertices[kite.hinge_index]
        # The hinge itself is the pivot — unchanged.
        assert np.allclose(hinge, rk.vertices[rk.hinge_index])
        # Every vertex equals the rest vertex rotated about the hinge.
        for v_rot, v_rest in zip(kite.vertices, rk.vertices):
            expected = Rm @ (v_rest - hinge) + hinge
            assert np.allclose(v_rot, expected)


def test_rotation_is_rigid_edge_lengths_preserved():
    pts, simps = _equilateral()
    rest = build_bipartite_network(pts, simps, C=1.0)
    rot  = build_bipartite_network(pts, simps, C=1.0, theta=0.4)
    rest_by_corner = {k.corner_point_index: k for k in rest.set_a}
    for kite in rot.set_a:
        rk = rest_by_corner[kite.corner_point_index]
        for i in range(kite.degree):
            j = (i + 1) % kite.degree
            d_rot = np.linalg.norm(kite.vertices[i] - kite.vertices[j])
            d_rest = np.linalg.norm(rk.vertices[i] - rk.vertices[j])
            assert d_rot == pytest.approx(d_rest)


def test_jamming_angle_is_positive_and_below_pi_over_2():
    pts, simps = _equilateral()
    jam = jamming_angle(pts, simps, C=1.0)
    assert 0.0 < jam < np.pi / 2.0 + 1e-9


def test_jamming_angle_brings_inner_edge_onto_central_edge():
    """Rotating by exactly the jamming angle should bring at least one
    kite inner edge collinear with the central-polygon edge it collides
    with (the contact configuration)."""
    pts, simps = _equilateral()
    jam = jamming_angle(pts, simps, C=1.0)
    net = build_bipartite_network(pts, simps, C=1.0, theta=jam)
    (central,) = net.set_b

    def cross(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    # Some inner edge, extended from its hinge, is collinear with a
    # central-polygon edge sharing that hinge.
    cverts = central.vertices
    found = False
    for kite in net.set_a:
        hinge = kite.vertices[kite.hinge_index]
        # central edges incident to this hinge
        hidx = [i for i in range(len(cverts))
                if np.allclose(cverts[i], hinge)]
        if not hidx:
            continue
        ci = hidx[0]
        central_dirs = [cverts[(ci + 1) % len(cverts)] - hinge,
                        cverts[(ci - 1) % len(cverts)] - hinge]
        for edge in kite.inner_edges():
            d = (edge[0] - hinge) if np.allclose(edge[1], hinge) else (edge[1] - hinge)
            for cd in central_dirs:
                if (np.linalg.norm(d) > 1e-9 and np.linalg.norm(cd) > 1e-9
                        and abs(cross(d, cd)) < 1e-6
                        and np.dot(d, cd) > 0):
                    found = True
    assert found


# ---------------------------------------------------------------------------
# Winding + validation
# ---------------------------------------------------------------------------

def test_every_polygon_is_ccw():
    pts, simps = _symmetric_rhombus()
    net = build_bipartite_network(pts, simps, C=1.0)
    for poly in net.polygons:
        v = poly.vertices
        area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1)
                            - np.roll(v[:, 0], -1) * v[:, 1])
        assert area > 0.0


def test_hinges_property_dedups_central_vertices():
    pts, simps = _symmetric_rhombus()
    net = build_bipartite_network(pts, simps, C=1.0)
    # Two centroids, three hinges each, all distinct → 6 hinges.
    assert net.hinges.shape == (6, 2)


def test_input_validation():
    with pytest.raises(ValueError, match="points must be"):
        build_bipartite_network(np.zeros((3, 3)), np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="simplices must be"):
        build_bipartite_network(np.zeros((3, 2)), np.array([[0, 1]]))
    with pytest.raises(ValueError, match="C must be > 0"):
        build_bipartite_network(np.zeros((3, 2)), np.array([[0, 1, 2]]), C=0.0)
    with pytest.raises(ValueError, match="C must be > 0"):
        build_bipartite_network(np.zeros((3, 2)), np.array([[0, 1, 2]]), C=-1.0)
