"""Tests for the v5 → v6 preset migration and the ``bezier`` block.

v6 adds an opt-in ``bezier`` block (curved strut edges). Every pre-v6
preset must load with curving OFF, i.e. byte-for-byte identical export
geometry, and a v6 preset must round-trip the bezier settings.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from auxetic import Lattice
from auxetic_studio.preset import (
    PRESET_VERSION,
    _migrate_v5_to_v6,
    _stub_bezier,
    load_preset,
    save_preset,
)


def _v5_payload(mode=1, n_points=5):
    """A hand-rolled v5 payload — the on-disk shape before the bezier
    block existed."""
    return {
        "version":   5,
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
            "unit_scale_cm": 1.0, "C": 1.0,
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

def test_preset_version_is_6():
    assert PRESET_VERSION == 6


def test_stub_bezier_defaults_off():
    stub = _stub_bezier()
    assert stub["enabled"] is False
    assert stub["strength"] == 0.25
    assert stub["segments"] == 12


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migrate_v5_to_v6_adds_bezier_off():
    out = _migrate_v5_to_v6(_v5_payload())
    assert out["version"] == 6
    assert out["bezier"]["enabled"] is False
    assert out["bezier"]["strength"] == 0.25
    assert out["bezier"]["segments"] == 12


def test_migrate_v5_to_v6_preserves_existing_bezier():
    payload = _v5_payload()
    payload["bezier"] = {"enabled": True, "strength": 0.4, "segments": 20}
    out = _migrate_v5_to_v6(payload)
    assert out["bezier"] == {"enabled": True, "strength": 0.4, "segments": 20}


def test_load_v5_file_defaults_bezier_off(tmp_path):
    p = tmp_path / "old_v5.json"
    p.write_text(json.dumps(_v5_payload()))
    lat = load_preset(str(p))
    assert lat.bezier_enabled is False
    assert lat.bezier_strength == 0.25
    assert lat.bezier_segments == 12


def test_load_v5_file_exports_identically_to_default(tmp_path):
    """A migrated v5 preset (no bezier block) must export the same STL as
    a default lattice built from the same parameters — curving is off."""
    p = tmp_path / "old_v5.json"
    p.write_text(json.dumps(_v5_payload(mode=1, n_points=5)))
    loaded = load_preset(str(p))

    direct = Lattice(mode=1, n_points=5, ratio=0.35, nz_layers=2)
    direct.regenerate_from_points(np.asarray(loaded.points))

    a, b = tmp_path / "loaded.stl", tmp_path / "direct.stl"
    loaded.to_stl(str(a), verbose=False)
    direct.to_stl(str(b), verbose=False)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read()[80:] == fb.read()[80:]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_bezier_roundtrip(tmp_path):
    lat = Lattice(mode=1, n_points=6, seed=3)
    lat.set_bezier(enabled=True, strength=0.4, segments=20)

    p = tmp_path / "bez.json"
    save_preset(str(p), lat)

    data = json.loads(p.read_text())
    assert data["version"] == 6
    assert data["bezier"] == {"enabled": True, "strength": 0.4, "segments": 20}

    lat2 = load_preset(str(p))
    assert lat2.bezier_enabled is True
    assert lat2.bezier_strength == 0.4
    assert lat2.bezier_segments == 20


def test_default_lattice_saves_bezier_off(tmp_path):
    lat = Lattice(mode=1, n_points=5, seed=3)
    p = tmp_path / "m1.json"
    save_preset(str(p), lat)
    data = json.loads(p.read_text())
    assert data["bezier"]["enabled"] is False
    lat2 = load_preset(str(p))
    assert lat2.bezier_enabled is False


def test_bezier_settings_survive_roundtrip_and_curve(tmp_path):
    """After a save/load cycle, the reloaded lattice actually exports
    curved geometry (more SCAD cylinders than its straight twin)."""
    lat = Lattice(mode=6, n_points=8)
    lat.set_bezier(enabled=True, strength=0.3, segments=5)
    p = tmp_path / "curved.json"
    save_preset(str(p), lat)
    lat2 = load_preset(str(p))

    straight = Lattice(mode=6, n_points=8)
    s_scad, c_scad = tmp_path / "s.scad", tmp_path / "c.scad"
    straight.to_scad(str(s_scad), verbose=False)
    lat2.to_scad(str(c_scad), verbose=False)
    n_straight = s_scad.read_text().count("cylinder(")
    n_curved = c_scad.read_text().count("cylinder(")
    assert n_curved == n_straight * 5
