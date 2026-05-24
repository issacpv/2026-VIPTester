"""Tests for the v4 → v5 preset migration and the mode-11 constant
size ratio ``C`` added to the generation block.

v5 adds a single field — ``generation.C`` — consulted only by mode 11
(the bipartite auxetic). Every pre-v5 preset must load with C defaulted
to 1.0 and otherwise byte-for-byte identical behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from auxetic import Lattice
from auxetic_studio.preset import (
    PRESET_VERSION,
    _migrate_v4_to_v5,
    _stub_generation,
    load_preset,
    save_preset,
)


def _v4_payload(mode=1, n_points=5):
    """A hand-rolled v4 payload — the on-disk shape from M2, before C
    existed in the generation block."""
    return {
        "version":   4,
        "mode":      mode,
        "n_points":  n_points,
        "ratio":     0.35,
        "nz_layers": 2,
        "points":    [[0.1, 0.2], [0.3, 0.4], [0.6, 0.7],
                       [0.8, 0.9], [0.5, 0.5]][:n_points],
        "shape_params": {
            "ngon_thickness":      0.03,
            "hub_size_factor":     0.75,
            "joint_sphere_radius": 0.015,
            "strut_radius":        0.02,
        },
        "generation": {
            "density_axis": "none", "density_law": "uniform",
            "density_strength": 1.0,
            "edge_flips": [], "mesh_path": None, "mesh_vertices": None,
            "unit_scale_cm": 1.0,
        },
        "dynamics": {},
        "view_state": {
            "rigid_rotation_quat": [1.0, 0.0, 0.0, 0.0],
            "flipped":             False,
            "joint_angle_deg":     0.0,
        },
        "metadata": {"name": "", "created": "", "modified": "", "notes": ""},
    }


# ---------------------------------------------------------------------------
# Version + stub
# ---------------------------------------------------------------------------

def test_preset_version_at_least_5():
    # v5 introduced generation.C; the exact current version is pinned in
    # the newest version's test (test_preset_v6). This only asserts v5's
    # features have not been rolled back.
    assert PRESET_VERSION >= 5


def test_stub_generation_includes_C():
    assert _stub_generation()["C"] == 1.0


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migrate_v4_to_v5_adds_default_C():
    out = _migrate_v4_to_v5(_v4_payload())
    assert out["version"] == 5
    assert out["generation"]["C"] == 1.0


def test_migrate_v4_to_v5_preserves_existing_C():
    payload = _v4_payload()
    payload["generation"]["C"] = 2.5   # hand-edited file
    out = _migrate_v4_to_v5(payload)
    assert out["generation"]["C"] == 2.5


def test_load_v4_file_defaults_C(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps(_v4_payload()))
    lat = load_preset(str(p))
    assert lat.C == 1.0


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_mode_11_roundtrip(tmp_path):
    lat = Lattice(mode=11, n_points=4, seed=1)
    lat.C = 3.5
    pts = lat.points.copy()

    p = tmp_path / "bip.json"
    save_preset(str(p), lat)

    data = json.loads(p.read_text())
    assert data["version"] == PRESET_VERSION
    assert data["generation"]["C"] == 3.5

    lat2 = load_preset(str(p))
    assert lat2.mode == 11
    assert lat2.C == 3.5
    assert np.allclose(lat2.points, pts)


def test_non_bipartite_roundtrip_keeps_default_C(tmp_path):
    """A mode-1 lattice saves and reloads with C=1.0 — the field is
    inert for every mode except 11, but still persisted."""
    lat = Lattice(mode=1, n_points=5, seed=3)
    p = tmp_path / "m1.json"
    save_preset(str(p), lat)
    lat2 = load_preset(str(p))
    assert lat2.mode == 1
    assert lat2.C == 1.0
