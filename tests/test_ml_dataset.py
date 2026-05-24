"""Tests for ``auxetic_ml.dataset`` — sample generation and storage.

The dataset pipeline orchestrates the M2 dynamic sim across many
``(lattice, action)`` pairs and saves the results. These tests pin
the contracts: deterministic sampling, save/load roundtrip, action
space coverage.

We use small lattices and short sim durations to keep total
test-suite runtime manageable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from auxetic import Lattice
from auxetic_ml.dataset import (
    Action,
    GROUND_FACES,
    Sample,
    SampleLabel,
    generate_samples,
    load_samples,
    sample_action,
    save_samples,
)


def _factory(mode=1, n_points=5, seed=42, ratio=0.35):
    def make():
        return Lattice(mode=mode, n_points=n_points, ratio=ratio, seed=seed)
    return make


# ---------------------------------------------------------------------------
# sample_action
# ---------------------------------------------------------------------------

def test_sample_action_returns_valid_record():
    rng = np.random.default_rng(0)
    a = sample_action(rng, [(0, 1), (2, 3)])
    assert a.ground_face in GROUND_FACES
    for e in a.edge_flips:
        assert e in [(0, 1), (2, 3)]
    assert -np.pi <= a.pre_rotation_rad <= np.pi


def test_sample_action_is_deterministic_with_seed():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    edges = [(0, 1), (2, 3), (4, 5)]
    actions_a = [sample_action(rng_a, edges) for _ in range(10)]
    actions_b = [sample_action(rng_b, edges) for _ in range(10)]
    for a, b in zip(actions_a, actions_b):
        assert a.ground_face == b.ground_face
        assert a.edge_flips  == b.edge_flips
        assert a.pre_rotation_rad == pytest.approx(b.pre_rotation_rad)


def test_sample_action_flip_probability_zero_yields_no_flips():
    rng = np.random.default_rng(0)
    edges = [(i, i + 1) for i in range(5)]
    for _ in range(20):
        a = sample_action(rng, edges, flip_probability=0.0)
        assert a.edge_flips == ()


def test_sample_action_flip_probability_one_includes_all_edges():
    rng = np.random.default_rng(0)
    edges = [(0, 1), (2, 3), (4, 5)]
    a = sample_action(rng, edges, flip_probability=1.0)
    assert set(a.edge_flips) == set(edges)


# ---------------------------------------------------------------------------
# generate_samples
# ---------------------------------------------------------------------------

def test_generate_samples_count_matches():
    samples = generate_samples(_factory(), n_samples=3, seed=0,
                                duration=0.02)
    assert len(samples) == 3


def test_generate_samples_records_have_correct_types():
    samples = generate_samples(_factory(), n_samples=2, seed=0,
                                duration=0.02)
    for s in samples:
        assert isinstance(s, Sample)
        assert isinstance(s.action, Action)
        assert isinstance(s.label, SampleLabel)
        assert s.features.ndim == 1
        # Label fields are finite numbers (or known-finite bools).
        assert np.isfinite(s.label.compression_efficiency)
        assert np.isfinite(s.label.stability_score)
        assert isinstance(s.label.converged, bool)


def test_generate_samples_is_deterministic_with_seed():
    s_a = generate_samples(_factory(), n_samples=3, seed=99, duration=0.02)
    s_b = generate_samples(_factory(), n_samples=3, seed=99, duration=0.02)
    for a, b in zip(s_a, s_b):
        assert a.action.ground_face == b.action.ground_face
        assert a.action.edge_flips  == b.action.edge_flips
        assert a.action.pre_rotation_rad == pytest.approx(
            b.action.pre_rotation_rad)
        np.testing.assert_array_equal(a.features, b.features)


def test_generate_samples_lattice_meta_populated():
    samples = generate_samples(_factory(mode=4, n_points=9), n_samples=1,
                                seed=0, duration=0.02)
    s = samples[0]
    assert s.lattice_meta["mode"]     == 4
    assert s.lattice_meta["n_points"] == 9


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_samples(tmp_path):
    samples = generate_samples(_factory(), n_samples=4, seed=0,
                                duration=0.02)
    out_dir = save_samples(samples, tmp_path / "ds")
    assert (out_dir / "manifest.json").is_file()
    loaded = load_samples(out_dir)
    assert len(loaded) == len(samples)
    for orig, back in zip(samples, loaded):
        np.testing.assert_array_equal(orig.features, back.features)
        assert orig.action.ground_face      == back.action.ground_face
        assert orig.action.edge_flips       == back.action.edge_flips
        assert orig.action.pre_rotation_rad == pytest.approx(
            back.action.pre_rotation_rad)
        assert orig.label.compression_efficiency == pytest.approx(
            back.label.compression_efficiency)
        assert orig.label.stability_score == pytest.approx(
            back.label.stability_score)
        assert orig.label.converged == back.label.converged
        assert orig.lattice_meta == back.lattice_meta


def test_save_writes_per_sample_files_and_manifest(tmp_path):
    samples = generate_samples(_factory(), n_samples=3, seed=0,
                                duration=0.02)
    out = save_samples(samples, tmp_path / "ds")
    files = sorted(p.name for p in out.iterdir() if p.suffix == ".npz")
    assert files == ["sample_0000.npz", "sample_0001.npz", "sample_0002.npz"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_samples"] == 3
    assert manifest["files"]     == files


def test_load_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_samples(tmp_path / "no_such_dir")


def test_save_handles_empty_force_set(tmp_path):
    samples = generate_samples(_factory(), n_samples=2, seed=0,
                                duration=0.02)
    # Force at least one sample to have no edge flips.
    samples[0] = Sample(
        features=samples[0].features,
        action=Action(ground_face=None, edge_flips=(),
                      pre_rotation_rad=0.0),
        label=samples[0].label,
        lattice_meta=samples[0].lattice_meta,
    )
    out = save_samples(samples, tmp_path / "ds")
    loaded = load_samples(out)
    assert loaded[0].action.edge_flips == ()
    assert loaded[0].action.ground_face is None
