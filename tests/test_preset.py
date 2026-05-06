"""Stage 4 preset tests.

Six tests, mapped to the prompt:

1. ``test_save_load_roundtrip_unedited`` — save/load preserves the
   scalar fields, points, and shape_params of an as-regenerated lattice.
2. ``test_save_load_roundtrip_after_edits`` — *load-bearing*: edited
   points survive a save/load cycle exactly. (Stage 3's whole point
   depends on this.)
3. ``test_load_legacy_preset`` — a hand-written Stage-2-shaped JSON
   migrates cleanly to a Lattice with sensible defaults.
4. ``test_reject_future_version`` — ``version: 99`` raises with a
   readable message.
5. ``test_metadata_created_preserved_across_saves`` — repeated saves
   advance ``modified`` but leave ``created`` alone.
6. ``test_view_state_round_trips_unmodified`` — view_state survives
   save/load even though nothing acts on it yet.
"""

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from auxetic import Lattice
from auxetic_studio.preset import save_preset, load_preset, PRESET_VERSION


# ---------------------------------------------------------------------------
# 1. Round-trip on an unedited (freshly regenerated) lattice
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_unedited(tmp_path):
    src = Lattice(mode=2, n_points=6, ratio=0.42, nz_layers=3, seed=42)

    path = str(tmp_path / "preset.auxlat")
    save_preset(path, src)
    dst = load_preset(path)

    assert dst.mode      == src.mode
    assert dst.n_points  == src.n_points
    assert dst.ratio     == pytest.approx(src.ratio)
    assert dst.nz_layers == src.nz_layers

    np.testing.assert_array_equal(dst.points, src.points)

    # SPEC §5.1 shape_params block.
    assert dst.ngon_thickness      == pytest.approx(src.ngon_thickness)
    assert dst.hub_size_factor     == pytest.approx(src.hub_size_factor)
    assert dst.joint_sphere_radius == pytest.approx(src.joint_sphere_radius)
    assert dst.strut_radius        == pytest.approx(src.strut_radius)


# ---------------------------------------------------------------------------
# 2. Round-trip preserves manual edits — the load-bearing test
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_after_edits(tmp_path):
    """Stage 3's edit work depends on this passing: if a saved preset
    silently re-rolls points on load instead of restoring them, every
    user edit is lost on file open."""
    src = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)

    # Modify points: shift each one.
    edited = src.points.copy()
    edited[0] = [0.99, 0.01]
    edited[2] = [0.10, 0.90]
    src.regenerate_from_points(edited)
    # Sanity: regenerate_from_points must NOT touch points_original.
    assert not np.allclose(src.points, src.points_original)

    path = str(tmp_path / "edited.auxlat")
    save_preset(path, src)
    dst = load_preset(path)

    np.testing.assert_array_equal(dst.points, edited)
    # After load, points_original must equal the loaded points so
    # "Reset to Original" returns to the as-loaded state.
    np.testing.assert_array_equal(dst.points_original, dst.points)


# ---------------------------------------------------------------------------
# 3. Legacy migration: hand-rolled Stage-2-shaped file
# ---------------------------------------------------------------------------

def test_load_legacy_preset(tmp_path):
    """Mirrors the Stage 2 stub format: nested ``lattice`` dict, no
    ``points`` field, version-less. ``_migrate_legacy`` should fill
    defaults; the loader's missing-points branch should fall back to a
    fresh regenerate()."""
    legacy = {
        # No "version" — triggers _is_legacy.
        "lattice": {
            "mode":      4,
            "n_points":  6,
            "ratio":     0.40,
            "nz_layers": 2,
            "seed":      42,
        },
        "view_state": {  # Stage 2's old view_state shape — gets discarded.
            "view_mode": 0,
            "camera_position": [0.0, 0.0, 0.0],
        },
        "metadata": {
            "name":   "old preset",
            "author": "someone",
            "notes":  "from stage 2",
            "created": "2026-04-01T00:00:00Z",
        },
    }
    path = str(tmp_path / "legacy.auxlat")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(legacy, f)

    lattice = load_preset(path)

    assert lattice.mode      == 4
    assert lattice.n_points  == 6
    assert lattice.ratio     == pytest.approx(0.40)
    assert lattice.nz_layers == 2

    # Missing-points branch ⇒ a fresh regenerate populated points.
    assert lattice.points is not None
    assert lattice.points.shape[0] >= 3

    # Shape params got geometry-module defaults.
    from auxetic import geometry as g
    assert lattice.ngon_thickness  == pytest.approx(g.NGON_THICKNESS)
    assert lattice.hub_size_factor == pytest.approx(g.HUB_SIZE_FACTOR)

    # Metadata: "created" preserved, "modified" populated.
    assert lattice.metadata["name"]    == "old preset"
    assert lattice.metadata["created"] == "2026-04-01T00:00:00Z"
    assert lattice.metadata["modified"]  # non-empty


# ---------------------------------------------------------------------------
# 4. Future-version rejection
# ---------------------------------------------------------------------------

def test_reject_future_version(tmp_path):
    future = {
        "version":   99,
        "mode":      1,
        "n_points":  5,
        "ratio":     0.35,
        "nz_layers": 2,
        "points":    [],
        "shape_params": {},
        "view_state":   {},
        "metadata":     {},
    }
    path = str(tmp_path / "future.auxlat")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(future, f)

    with pytest.raises(ValueError, match="newer than this application supports"):
        load_preset(path)


# ---------------------------------------------------------------------------
# 5. metadata.created vs metadata.modified across saves
# ---------------------------------------------------------------------------

def test_metadata_created_preserved_across_saves(tmp_path):
    lat = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    path = str(tmp_path / "meta.auxlat")

    # Inject deterministic timestamps via the test seam so the test
    # doesn't depend on wall-clock granularity.
    save_preset(path, lat, _now_fn=lambda: "2026-04-29T00:00:00Z")
    with open(path) as f:
        data1 = json.load(f)

    save_preset(path, lat, _now_fn=lambda: "2026-04-29T01:00:00Z")
    with open(path) as f:
        data2 = json.load(f)

    # `created` must be preserved across saves...
    assert data2["metadata"]["created"] == data1["metadata"]["created"]
    assert data2["metadata"]["created"] == "2026-04-29T00:00:00Z"
    # ...while `modified` advances each save.
    assert data2["metadata"]["modified"] != data1["metadata"]["modified"]
    assert data2["metadata"]["modified"] == "2026-04-29T01:00:00Z"


# ---------------------------------------------------------------------------
# 6. view_state survives save/load even though nothing acts on it yet
# ---------------------------------------------------------------------------

def test_view_state_round_trips_unmodified(tmp_path):
    lat = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    # Fill with non-default values so we can detect any silent drop.
    lat.view_state = {
        "rigid_rotation_quat": [0.7071, 0.7071, 0.0, 0.0],
        "flipped":             True,
        "joint_angle_deg":     45.0,
    }

    path = str(tmp_path / "vs.auxlat")
    save_preset(path, lat)
    dst = load_preset(path)

    assert dst.view_state == lat.view_state


# ---------------------------------------------------------------------------
# Bonus: confirm save_preset writes a v1-shaped file (cheap structural check)
# ---------------------------------------------------------------------------

def test_save_writes_v1_schema(tmp_path):
    lat = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    path = str(tmp_path / "schema.auxlat")
    save_preset(path, lat)
    with open(path) as f:
        data = json.load(f)

    expected_keys = {
        "version", "mode", "n_points", "ratio", "nz_layers",
        "points", "shape_params", "generation", "dynamics",
        "view_state", "metadata",
    }
    assert set(data.keys()) == expected_keys
    assert data["version"] == PRESET_VERSION

    # shape_params has SPEC §5.1's four fields.
    assert set(data["shape_params"].keys()) == {
        "ngon_thickness", "hub_size_factor",
        "joint_sphere_radius", "strut_radius",
    }
    # view_state has SPEC §5.1's three fields.
    assert set(data["view_state"].keys()) == {
        "rigid_rotation_quat", "flipped", "joint_angle_deg",
    }
    # generation block (v3, M1) carries the density / edge-flip / mesh
    # fields. Defaults are documented in ``preset._stub_generation``.
    assert set(data["generation"].keys()) == {
        "density_axis", "density_law", "density_strength",
        "edge_flips", "mesh_path", "mesh_vertices", "unit_scale_cm",
    }
    # dynamics block (v4, M2) — Newtonian sim parameters.
    assert set(data["dynamics"].keys()) == {
        "forces", "ground_face", "pre_rotation_quat",
        "pre_joint_angle_deg", "fixed_tiles", "config",
    }
