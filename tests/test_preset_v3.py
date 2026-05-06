"""Tests for the v2 → v3 preset migration and the v3 ``generation``
block introduced in M1.

The v3 schema adds:
- ``density_axis``, ``density_law``, ``density_strength`` — biased
  random sampling for modes 1, 2, 3.
- ``edge_flips`` — per-edge Delaunay diagonal flips (2D only).
- ``mesh_path`` + ``mesh_vertices`` — provenance for modes 7, 8, 9.
- ``unit_scale_cm`` — physical scale, used by the M2 dynamic sim.

A v2 file must load cleanly with all defaults (no behaviour change).
A v3 file must round-trip every new field.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from auxetic import Lattice
from auxetic_studio.preset import (
    PRESET_VERSION,
    _migrate_v2_to_v3,
    _stub_generation,
    load_preset,
    save_preset,
)


# ---------------------------------------------------------------------------
# Migration: v2 → v3
# ---------------------------------------------------------------------------

def _v2_payload(mode=1, n_points=5):
    """Hand-rolled v2 preset dict (the on-disk shape from the v2 codebase)."""
    return {
        "version":   2,
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
        "view_state": {
            "rigid_rotation_quat": [1.0, 0.0, 0.0, 0.0],
            "flipped":             False,
            "joint_angle_deg":     0.0,
        },
        "metadata": {
            "name": "", "created": "", "modified": "", "notes": "",
        },
    }


def test_migrate_v2_to_v3_adds_generation_block():
    out = _migrate_v2_to_v3(_v2_payload())
    assert out["version"] == 3
    assert "generation" in out
    assert set(out["generation"].keys()) == set(_stub_generation().keys())


def test_migrate_v2_to_v3_defaults_are_behavior_preserving():
    out = _migrate_v2_to_v3(_v2_payload())
    g = out["generation"]
    assert g["density_axis"]     == "none"
    assert g["density_law"]      == "uniform"
    assert g["density_strength"] == 1.0
    assert g["edge_flips"]       == []
    assert g["mesh_path"]        is None
    assert g["mesh_vertices"]    is None
    assert g["unit_scale_cm"]    == 1.0


def test_migrate_v2_to_v3_does_not_clobber_existing_generation():
    """If a partial 'generation' block is somehow already present (e.g.
    a hand-rolled preset), migration should fill missing keys but not
    overwrite the user's existing values."""
    src = _v2_payload()
    src["generation"] = {"density_axis": "x", "density_law": "linear"}
    out = _migrate_v2_to_v3(src)
    assert out["generation"]["density_axis"] == "x"
    assert out["generation"]["density_law"]  == "linear"
    assert out["generation"]["density_strength"] == 1.0  # filled by migration
    assert out["generation"]["unit_scale_cm"]    == 1.0


# ---------------------------------------------------------------------------
# Loading a v2 file goes through the migration path
# ---------------------------------------------------------------------------

def test_load_v2_file_succeeds_with_default_generation(tmp_path):
    p = tmp_path / "v2.auxlat"
    p.write_text(json.dumps(_v2_payload()), encoding="utf-8")
    lat = load_preset(str(p))
    # Lattice gets the v3 defaults from the stub generation block.
    assert lat.density_axis     == "none"
    assert lat.density_law      == "uniform"
    assert lat.density_strength == 1.0
    assert lat.edge_flips       == set()
    assert lat.mesh_path        is None
    assert lat.mesh_vertices    is None
    assert lat.unit_scale_cm    == 1.0


# ---------------------------------------------------------------------------
# v3 round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_density_gradient(tmp_path):
    lat = Lattice(mode=1, n_points=12, seed=42,
                  density_axis="x", density_law="linear",
                  density_strength=0.7)
    p = str(tmp_path / "dens.auxlat")
    save_preset(p, lat)
    loaded = load_preset(p)
    assert loaded.density_axis     == "x"
    assert loaded.density_law      == "linear"
    assert loaded.density_strength == 0.7


def test_save_load_roundtrip_edge_flips(tmp_path):
    lat = Lattice(mode=1, n_points=12, seed=42)
    lat.edge_flips = {(2, 5), (3, 7)}
    lat.regenerate_from_points(lat.points)  # apply
    p = str(tmp_path / "flips.auxlat")
    save_preset(p, lat)
    loaded = load_preset(p)
    assert loaded.edge_flips == {(2, 5), (3, 7)}


def test_save_load_roundtrip_mesh_import(tmp_path):
    obj = tmp_path / "tiny.obj"
    obj.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nv 0.5 0.5 0\n",
        encoding="utf-8",
    )
    lat = Lattice.from_mesh(str(obj), dim="2D")
    p = str(tmp_path / "mesh.auxlat")
    save_preset(p, lat)
    loaded = load_preset(p)
    assert loaded.mode == 7
    assert loaded.mesh_path == str(obj)
    assert loaded.mesh_vertices is not None
    np.testing.assert_allclose(loaded.mesh_vertices, lat.mesh_vertices)


def test_save_load_roundtrip_unit_scale_cm(tmp_path):
    lat = Lattice(mode=1, n_points=5, seed=42, unit_scale_cm=2.5)
    p = str(tmp_path / "scale.auxlat")
    save_preset(p, lat)
    loaded = load_preset(p)
    assert loaded.unit_scale_cm == 2.5


def test_save_writes_current_preset_version():
    """Sanity: PRESET_VERSION is at least 3 (M1 introduced the
    generation block). The exact current value advances with each
    milestone — see ``auxetic_studio/preset.PRESET_VERSION``."""
    assert PRESET_VERSION >= 3


def test_v3_generation_block_round_trips_unchanged(tmp_path):
    """A freshly-saved preset stores the generation block. When the
    schema advances past v3 we still want to confirm the M1 fields are
    written, so the assertion intentionally targets the generation
    sub-dict, not the version integer."""
    lat = Lattice(mode=1, n_points=5, seed=42,
                  density_axis="y", density_law="exp", density_strength=1.5)
    p = tmp_path / "fresh.auxlat"
    save_preset(str(p), lat)
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["generation"]["density_axis"]     == "y"
    assert on_disk["generation"]["density_law"]      == "exp"
    assert on_disk["generation"]["density_strength"] == 1.5
