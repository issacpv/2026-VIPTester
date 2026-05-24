"""Tests for the v3 → v4 preset migration and the v4 ``dynamics``
block introduced in M2.

The v4 schema adds the dynamic-simulator state:
- ``forces`` — list of user-defined force vectors (location kind,
  direction, magnitude, optional tile/vertex indices).
- ``ground_face`` — which bbox face the lattice "stands on", or ``None``.
- ``pre_rotation_quat`` / ``pre_joint_angle_deg`` — optional sim-init
  overrides of the orientation already in ``view_state``.
- ``fixed_tiles`` — explicit list of pinned tile indices.
- ``config`` — dt, duration, joint stiffness/damping, gravity, etc.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from auxetic import Lattice
from auxetic_studio.preset import (
    PRESET_VERSION,
    _migrate_v3_to_v4,
    _stub_dynamics,
    load_preset,
    save_preset,
)


# ---------------------------------------------------------------------------
# Fixtures: a hand-rolled v3 payload (the on-disk shape of the M1 codebase).
# ---------------------------------------------------------------------------

def _v3_payload(mode=1, n_points=5):
    return {
        "version":   3,
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
        "view_state": {
            "rigid_rotation_quat": [1.0, 0.0, 0.0, 0.0],
            "flipped":             False,
            "joint_angle_deg":     0.0,
        },
        "metadata": {
            "name": "", "created": "", "modified": "", "notes": "",
        },
    }


# ---------------------------------------------------------------------------
# Migration: v3 → v4
# ---------------------------------------------------------------------------

def test_migrate_v3_to_v4_adds_dynamics_block():
    out = _migrate_v3_to_v4(_v3_payload())
    assert out["version"] == 4
    assert "dynamics" in out
    assert set(out["dynamics"].keys()) == set(_stub_dynamics().keys())


def test_migrate_v3_to_v4_defaults_are_empty_or_neutral():
    out = _migrate_v3_to_v4(_v3_payload())
    d = out["dynamics"]
    assert d["forces"]              == []
    assert d["ground_face"]         is None
    assert d["pre_rotation_quat"]   is None
    assert d["pre_joint_angle_deg"] is None
    assert d["fixed_tiles"]         == []
    assert d["config"]["dt"] > 0.0
    assert d["config"]["duration"] > 0.0


def test_migrate_v3_to_v4_does_not_clobber_existing_dynamics():
    src = _v3_payload()
    src["dynamics"] = {
        "forces": [{"kind": "tile_centroid", "tile_index": 0,
                     "direction": [0.0, -1.0, 0.0], "magnitude": 5.0}],
        "ground_face": "-y",
    }
    out = _migrate_v3_to_v4(src)
    # User's existing partial block is preserved...
    assert out["dynamics"]["ground_face"] == "-y"
    assert len(out["dynamics"]["forces"]) == 1
    # ...and missing fields are filled with defaults.
    assert out["dynamics"]["fixed_tiles"]       == []
    assert "dt" in out["dynamics"]["config"]


def test_migrate_v3_to_v4_fills_missing_inner_config_keys():
    """Inner-config fields the user didn't set should pick up the
    package-level defaults from ``_stub_dynamics``."""
    from auxetic_studio.preset import _stub_dynamics
    src = _v3_payload()
    src["dynamics"] = {"config": {"dt": 5e-4}}    # only dt set
    out = _migrate_v3_to_v4(src)
    defaults = _stub_dynamics()["config"]
    assert out["dynamics"]["config"]["dt"]               == 5e-4
    assert out["dynamics"]["config"]["duration"]         == defaults["duration"]
    assert out["dynamics"]["config"]["joint_stiffness"]  == defaults["joint_stiffness"]
    assert "gravity_cm_per_s2" in out["dynamics"]["config"]


# ---------------------------------------------------------------------------
# Loading v3 file in current code → v4 dynamics defaults appear
# ---------------------------------------------------------------------------

def test_load_v3_file_succeeds_with_default_dynamics(tmp_path):
    p = tmp_path / "v3.auxlat"
    p.write_text(json.dumps(_v3_payload()), encoding="utf-8")
    lat = load_preset(str(p))
    assert lat.dynamics_state["forces"] == []
    assert lat.dynamics_state["ground_face"] is None
    assert "dt" in lat.dynamics_state["config"]


# ---------------------------------------------------------------------------
# v4 round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_dynamics_default(tmp_path):
    """Default dynamics_state survives save/load unchanged."""
    lat = Lattice(mode=1, n_points=5, seed=42)
    p = str(tmp_path / "default.auxlat")
    save_preset(p, lat)
    loaded = load_preset(p)
    assert loaded.dynamics_state == lat.dynamics_state


def test_save_load_roundtrip_dynamics_with_forces(tmp_path):
    lat = Lattice(mode=1, n_points=5, seed=42)
    lat.dynamics_state["forces"] = [{
        "location_kind": "tile_centroid",
        "tile_index": 0,
        "direction": [0.0, -1.0, 0.0],
        "magnitude": 12.5,
    }]
    lat.dynamics_state["ground_face"] = "-y"
    lat.dynamics_state["fixed_tiles"] = [3, 4]
    p = str(tmp_path / "forces.auxlat")
    save_preset(p, lat)
    loaded = load_preset(p)
    assert loaded.dynamics_state["forces"][0]["magnitude"] == 12.5
    assert loaded.dynamics_state["ground_face"]            == "-y"
    assert loaded.dynamics_state["fixed_tiles"]            == [3, 4]


def test_save_load_roundtrip_dynamics_config_overrides(tmp_path):
    lat = Lattice(mode=1, n_points=5, seed=42)
    lat.dynamics_state["config"]["dt"] = 5e-4
    lat.dynamics_state["config"]["duration"] = 0.5
    lat.dynamics_state["config"]["joint_stiffness"] = 5e3
    p = str(tmp_path / "cfg.auxlat")
    save_preset(p, lat)
    loaded = load_preset(p)
    assert loaded.dynamics_state["config"]["dt"]              == 5e-4
    assert loaded.dynamics_state["config"]["duration"]        == 0.5
    assert loaded.dynamics_state["config"]["joint_stiffness"] == 5e3


def test_save_writes_version_4():
    assert PRESET_VERSION == 4
