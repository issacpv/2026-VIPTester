"""Task 6a: expose the points + bounding box the Poisson calc tracks.

``Simulator.all_world_vertices`` / ``bbox_bounds`` / ``bbox_corners`` give
the GUI the geometry to visualise — the tracked point cloud and the
axis-aligned bounding box at any pose, so it can show rest vs compressed.
Pure tests; no GUI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxetic import Lattice, TileSystem, Simulator


def _sim(mode=1, **kw):
    lat = Lattice(mode=mode, ratio=0.35, **kw)
    ts = TileSystem.from_lattice(lat)
    load_axis = (np.array([0.0, -1.0]) if ts.dimension == 2
                 else np.array([0.0, -1.0, 0.0]))
    return Simulator(ts, load_axis=load_axis)


def test_all_world_vertices_shape_and_concat():
    sim = _sim(mode=1, n_points=6, seed=3)
    rest = sim.rest_pose()
    v = sim.all_world_vertices(rest)
    assert v.ndim == 2 and v.shape[1] == sim.dimension
    manual = np.concatenate(
        [sim._tile_world_vertices(rest, i) for i in range(sim.n_tiles)],
        axis=0)
    np.testing.assert_allclose(v, manual)


def test_bbox_bounds_and_corners_consistent():
    sim = _sim(mode=1, n_points=6, seed=3)
    rest = sim.rest_pose()
    lo, hi = sim.bbox_bounds(rest)
    assert np.all(hi >= lo)
    # extent == hi - lo == _bbox_extents (the refactor preserved it).
    np.testing.assert_allclose(hi - lo, sim._bbox_extents(rest))
    corners = sim.bbox_corners(rest)
    assert corners.shape == (2 ** sim.dimension, sim.dimension)
    # Corners exactly span the bounds.
    np.testing.assert_allclose(corners.min(axis=0), lo)
    np.testing.assert_allclose(corners.max(axis=0), hi)


def test_bbox_changes_under_actuation():
    """A compressed (actuated) pose has a different bbox than rest — the
    contrast the Poisson tracking visualises."""
    sim = _sim(mode=1, n_points=6, seed=3)
    mode = sim.identify_kirigami_mode()
    if mode is None:
        pytest.skip("no kirigami mode for this lattice")
    rest = sim.rest_pose()
    actuated = sim.project_to_manifold(rest + 0.05 * mode)
    assert not np.allclose(sim._bbox_extents(rest), sim._bbox_extents(actuated))


def test_bbox_3d_has_eight_corners():
    sim = _sim(mode=6, n_points=8)
    corners = sim.bbox_corners(sim.rest_pose())
    assert corners.shape == (8, 3)


def test_bbox_extreme_vertices_sit_on_the_bounds():
    sim = _sim(mode=1, n_points=6, seed=3)
    rest = sim.rest_pose()
    ext = sim.bbox_extreme_vertices(rest)
    dim = sim.dimension
    assert ext.shape == (dim, 2, dim)
    lo, hi = sim.bbox_bounds(rest)
    verts = sim.all_world_vertices(rest)
    for d in range(dim):
        # min-side vertex sits on the lo[d] plane, max-side on hi[d].
        np.testing.assert_allclose(ext[d, 0, d], lo[d])
        np.testing.assert_allclose(ext[d, 1, d], hi[d])
        # Both are actual tracked vertices.
        assert np.any(np.all(np.isclose(verts, ext[d, 0]), axis=1))
        assert np.any(np.all(np.isclose(verts, ext[d, 1]), axis=1))


def test_expansion_pose_is_the_axial_maximum():
    """Batch 3 task 1: the most axially-EXPANDED sweep pose (argmax of the
    axial bbox extent) is distinct from the most-compressed pose (argmin)
    and really is the axial maximum — the geometry behind the third
    'expansion' bounds box. Its corners/extremes are well-formed."""
    sim = _sim(mode=1, n_points=6, seed=3)
    mode = sim.identify_kirigami_mode()
    if mode is None:
        pytest.skip("no kirigami mode for this lattice")
    result = sim.sweep_theta()
    extents = np.asarray(result.bbox_extents, dtype=float)
    axial = sim._axial_index()
    if float(np.ptp(extents[:, axial])) < 1e-9:
        pytest.skip("axial extent is flat for this lattice; no expansion extreme")
    comp_idx = int(np.argmin(extents[:, axial]))
    exp_idx = int(np.argmax(extents[:, axial]))
    assert exp_idx != comp_idx
    assert extents[exp_idx, axial] == extents[:, axial].max()
    exp_pose = result.poses[exp_idx]
    assert sim.bbox_corners(exp_pose).shape == (2 ** sim.dimension, sim.dimension)
    assert sim.bbox_extreme_vertices(exp_pose).shape == (
        sim.dimension, 2, sim.dimension)


def test_empty_tile_system_is_safe():
    sim = _sim(mode=1, n_points=6, seed=3)
    sim.n_tiles = 0                      # force the degenerate guard path
    assert sim.all_world_vertices(sim.rest_pose()).shape == (0, sim.dimension)
    lo, hi = sim.bbox_bounds(sim.rest_pose())
    np.testing.assert_allclose(lo, np.zeros(sim.dimension))
    np.testing.assert_allclose(hi, np.zeros(sim.dimension))
