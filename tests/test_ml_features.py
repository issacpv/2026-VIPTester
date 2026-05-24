"""Tests for ``auxetic_ml.features`` — handcrafted lattice feature
extractor.

The feature schema is fixed-size (``FEATURE_DIM = 25``) and
deterministic. Same lattice → same vector. Each schema slice has a
documented meaning so a regression-tree baseline could read the
vector for sanity checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from auxetic import Lattice
from auxetic_ml.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_VERSION,
    N_MODES,
    lattice_features,
)


def test_feature_schema_size_and_names():
    assert FEATURE_VERSION == 1
    assert FEATURE_DIM == 25
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert all(isinstance(n, str) for n in FEATURE_NAMES)


def test_features_have_correct_shape_and_dtype():
    lat = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    feats = lattice_features(lat)
    assert feats.shape == (FEATURE_DIM,)
    assert feats.dtype == np.float64


def test_features_are_deterministic():
    lat = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    a = lattice_features(lat)
    b = lattice_features(lat)
    np.testing.assert_array_equal(a, b)


def test_features_one_hot_encodes_mode():
    for mode in (1, 2, 4, 6):
        n_pts = 8 if mode == 6 else (6 if mode in (4, 5) else 5)
        lat = Lattice(mode=mode, n_points=n_pts, ratio=0.35, seed=42)
        feats = lattice_features(lat)
        # First N_MODES entries are the one-hot.
        one_hot = feats[:N_MODES]
        assert one_hot.sum() == pytest.approx(1.0)
        assert one_hot[mode - 1] == 1.0


def test_features_reflect_n_points_and_ratio():
    lat_a = Lattice(mode=1, n_points=5,  ratio=0.35, seed=42)
    lat_b = Lattice(mode=1, n_points=10, ratio=0.50, seed=42)
    fa = lattice_features(lat_a)
    fb = lattice_features(lat_b)
    # n_points slot is index N_MODES; ratio is N_MODES + 1.
    assert fa[N_MODES]      == 5.0
    assert fb[N_MODES]      == 10.0
    assert fa[N_MODES + 1]  == pytest.approx(0.35)
    assert fb[N_MODES + 1]  == pytest.approx(0.50)


def test_features_handle_3d_lattice():
    lat = Lattice(mode=6, n_points=8, ratio=0.35, seed=42)
    feats = lattice_features(lat)
    assert feats.shape == (FEATURE_DIM,)
    # No NaNs / Infs anywhere — the 3D path uses real point coords.
    assert np.all(np.isfinite(feats))


def test_features_change_after_edge_flip():
    """The ``n_edge_flips`` slot should reflect mutations."""
    from auxetic.geometry import flippable_edges
    lat = Lattice(mode=1, n_points=12, seed=7)
    edges = flippable_edges(lat.tri, lat.points)
    assert len(edges) > 0
    base = lattice_features(lat)
    lat.edge_flips = {edges[0]}
    after = lattice_features(lat)
    # n_edge_flips slot is at N_MODES + 3.
    assert base[N_MODES + 3]  == 0.0
    assert after[N_MODES + 3] == 1.0


def test_features_change_with_joint_angle():
    lat = Lattice(mode=1, n_points=5, seed=42)
    base = lattice_features(lat)
    lat.joint_angle = 0.7
    after = lattice_features(lat)
    # joint_angle is the LAST slot.
    assert base[-1]  == pytest.approx(0.0)
    assert after[-1] == pytest.approx(0.7)


def test_features_handle_lattice_with_no_points_gracefully():
    """Defensive path — features() shouldn't crash on a degenerate
    lattice with missing point cloud."""
    lat = Lattice(mode=1, n_points=5, seed=42)
    saved_points = lat.points
    lat.points = None
    try:
        feats = lattice_features(lat)
        assert feats.shape == (FEATURE_DIM,)
        assert np.all(np.isfinite(feats))
    finally:
        lat.points = saved_points
