"""Tests for ``auxetic.collision`` — 2D SAT polygon overlap and the
``CollisionChecker`` that wraps it for kirigami pose evaluation.

The detector is 2D-only in M2; 3D tile systems return no collisions
with a one-time warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from auxetic.collision import (
    CollisionChecker,
    polygons_overlap_2d,
)
from auxetic.simulation import Constraint, TileSystem


# ---------------------------------------------------------------------------
# polygons_overlap_2d — primitive
# ---------------------------------------------------------------------------

UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_separated_squares_do_not_overlap():
    a = UNIT_SQUARE.copy()
    b = UNIT_SQUARE.copy() + np.array([2.0, 0.0])  # 1 unit gap on the +x side
    assert polygons_overlap_2d(a, b) is False


def test_overlapping_squares_do_overlap():
    a = UNIT_SQUARE.copy()
    b = UNIT_SQUARE.copy() + np.array([0.5, 0.0])  # half-overlap
    assert polygons_overlap_2d(a, b) is True


def test_touching_edge_with_positive_tol_is_not_overlap():
    """Two squares sharing an edge have zero gap. With a positive
    tolerance they should NOT count as overlapping — this is the
    "constraint-coupled tiles touching at a vertex" scenario."""
    a = UNIT_SQUARE.copy()
    b = UNIT_SQUARE.copy() + np.array([1.0, 0.0])  # share the right edge
    assert polygons_overlap_2d(a, b, tol=1e-6) is False


def test_touching_vertex_with_positive_tol_is_not_overlap():
    a = UNIT_SQUARE.copy()
    b = UNIT_SQUARE.copy() + np.array([1.0, 1.0])  # touch only at (1, 1)
    assert polygons_overlap_2d(a, b, tol=1e-6) is False


def test_concentric_overlap():
    """One square fully containing another."""
    inner = (UNIT_SQUARE - 0.5) * 0.4 + 0.5
    assert polygons_overlap_2d(UNIT_SQUARE, inner) is True


def test_triangle_overlap_via_sat():
    a = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
    b = np.array([[1.0, 0.5], [3.0, 0.5], [2.0, 2.5]])
    assert polygons_overlap_2d(a, b) is True


def test_rotated_squares_overlap_when_centers_close():
    R = np.array([[np.cos(0.7), -np.sin(0.7)],
                   [np.sin(0.7),  np.cos(0.7)]])
    a = UNIT_SQUARE - 0.5    # centre at origin
    b = (UNIT_SQUARE - 0.5) @ R.T + np.array([0.6, 0.0])
    assert polygons_overlap_2d(a, b) is True


def test_invalid_dim_raises():
    with pytest.raises(ValueError, match="expects \\(N, 2\\)"):
        polygons_overlap_2d(np.zeros((3, 3)), UNIT_SQUARE)


# ---------------------------------------------------------------------------
# CollisionChecker — pose-aware wrapper
# ---------------------------------------------------------------------------

def _two_far_squares() -> TileSystem:
    """Two unit squares far apart, no constraint between them."""
    t0 = UNIT_SQUARE.copy()
    t1 = UNIT_SQUARE.copy() + np.array([3.0, 0.0])
    return TileSystem(2, [t0, t1], constraints=[])


def _two_overlapping_squares() -> TileSystem:
    t0 = UNIT_SQUARE.copy()
    t1 = UNIT_SQUARE.copy() + np.array([0.3, 0.0])  # overlap by 0.7 units in x
    return TileSystem(2, [t0, t1], constraints=[])


def _two_touching_at_vertex_with_constraint() -> TileSystem:
    """Two squares sharing the (1, 0) vertex, joined by a constraint
    so the pair should NOT count as colliding."""
    t0 = UNIT_SQUARE.copy()
    t1 = UNIT_SQUARE.copy() + np.array([1.0, 0.0])
    # tile 0's vertex 1 == tile 1's vertex 0 (both at (1, 0))
    constraints = [Constraint(0, 1, 1, 0, 1)]
    return TileSystem(2, [t0, t1], constraints=constraints)


def test_no_collision_at_rest_when_separated():
    cc = CollisionChecker(_two_far_squares())
    assert cc.has_collision(cc._sim.rest_pose()) is False
    assert cc.colliding_pairs(cc._sim.rest_pose()) == []


def test_collision_detected_at_rest_when_overlapping():
    cc = CollisionChecker(_two_overlapping_squares())
    pairs = cc.colliding_pairs(cc._sim.rest_pose())
    assert pairs == [(0, 1)]
    assert cc.has_collision(cc._sim.rest_pose()) is True


def test_constraint_pair_not_flagged_at_rest():
    """Tiles that share a constraint vertex are coincident — without
    the exemption they'd register as colliding even at rest."""
    cc = CollisionChecker(_two_touching_at_vertex_with_constraint())
    assert cc.has_collision(cc._sim.rest_pose()) is False
    assert cc.colliding_pairs(cc._sim.rest_pose()) == []


def test_constraint_exemption_can_be_disabled():
    """``ignore_connected=False`` should make even constraint-coupled
    pairs participate in the check (useful for diagnostics)."""
    cc = CollisionChecker(
        _two_touching_at_vertex_with_constraint(),
        ignore_connected=False,
        tol=-1e-3,   # negative tol so touching counts as overlap
    )
    # The two squares share an edge — with negative tol the projection
    # gap doesn't separate them.
    assert cc.has_collision(cc._sim.rest_pose()) is True


def test_pose_translation_makes_separated_tiles_collide():
    """Translate tile 1 in along -x in pose-space until it overlaps tile 0."""
    cc = CollisionChecker(_two_far_squares())
    pose = cc._sim.rest_pose().copy()
    # Pose layout: [tx0, ty0, theta0, tx1, ty1, theta1].
    pose[3] = -2.5  # shift tile 1 enough to overlap tile 0
    assert cc.has_collision(pose) is True


def test_pose_rotation_brings_tiles_together():
    """Rotate tile 0 — at rest tiles are far apart, but a 90° rotation
    around tile 0's origin can bring its vertices into tile 1's space."""
    cc = CollisionChecker(_two_overlapping_squares())
    # Already overlapping at rest — sanity check.
    assert cc.has_collision(cc._sim.rest_pose()) is True
    # Apply a tiny CCW rotation; still overlapping.
    pose = cc._sim.rest_pose().copy()
    pose[2] = 0.05
    assert cc.has_collision(pose) is True


# ---------------------------------------------------------------------------
# 3D smoke
# ---------------------------------------------------------------------------

def test_3d_collision_returns_false_with_warning():
    ts = TileSystem(
        3,
        tiles=[
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5],
                      [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]]),
        ],
        constraints=[],
    )
    # Reset the once-warned flag so this test can independently see it.
    CollisionChecker._warned_3d = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cc = CollisionChecker(ts)
        result = cc.has_collision(cc._sim.rest_pose())
    assert result is False
    assert any("2D-only" in str(w.message) for w in caught)
