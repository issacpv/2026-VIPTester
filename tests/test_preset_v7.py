"""Preset v7 — persistence of a Tile-Library composition.

v7 adds a ``compose`` block carrying the user-authored triangulation, so
a composed lattice reloads with its exact mesh (never re-Delaunayed). The
v6→v7 migration marks older presets as not-composed, so they load exactly
as before.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from auxetic.lattice import Lattice
from auxetic.tile_library import get_tile
from auxetic_studio.preset import (
    PRESET_VERSION,
    _migrate_v6_to_v7,
    load_preset,
    save_preset,
)


def test_preset_version_is_current():
    # Bumped to 8 when per-triangle C overrides (tri_ids + piece_C) were
    # added to the compose block; v7 composition behaviour is unchanged.
    assert PRESET_VERSION == 8


def _composed_lattice() -> tuple[Lattice, np.ndarray]:
    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    edge = float(sq.points[:, 0].max() - sq.points[:, 0].min())
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4, 0.5))
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4 + edge, 0.5))
    return lat, np.asarray(lat.tri.simplices).copy()


def test_composed_lattice_round_trips(tmp_path):
    lat, simp_before = _composed_lattice()
    path = str(tmp_path / "compose.json")
    save_preset(path, lat)
    loaded = load_preset(path)

    assert loaded.mode == 11
    assert loaded.preserve_triangulation is True
    assert loaded.n_points == lat.n_points
    # The authored triangulation is restored verbatim — NOT re-Delaunayed.
    assert np.array_equal(simp_before, np.asarray(loaded.tri.simplices))
    assert np.allclose(loaded.points, lat.points)


def test_loaded_composition_survives_a_point_move(tmp_path):
    lat, simp_before = _composed_lattice()
    path = str(tmp_path / "compose.json")
    save_preset(path, lat)
    loaded = load_preset(path)

    moved = loaded.points.copy()
    moved[0] += [0.005, 0.0]
    loaded.regenerate_from_points(moved)
    assert np.array_equal(simp_before, np.asarray(loaded.tri.simplices))


def test_non_composed_lattice_saves_compose_off(tmp_path):
    lat = Lattice(mode=1, n_points=6, seed=1)
    path = str(tmp_path / "plain.json")
    save_preset(path, lat)
    loaded = load_preset(path)
    assert loaded.preserve_triangulation is False


def test_migration_v6_adds_non_composed_block():
    v6 = {"version": 6, "mode": 1, "points": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]}
    out = _migrate_v6_to_v7(v6)
    assert out["version"] == 7
    assert out["compose"]["preserve_triangulation"] is False
    assert out["compose"]["simplices"] is None


def test_old_v6_preset_loads_non_composed(tmp_path):
    """A v6-style preset (no compose block) must load as a normal
    Delaunay lattice, not a composition."""
    import json
    v6 = {
        "version": 6,
        "mode": 1,
        "n_points": 3,
        "ratio": 0.35,
        "nz_layers": 2,
        "points": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
    }
    path = tmp_path / "v6.json"
    path.write_text(json.dumps(v6), encoding="utf-8")
    loaded = load_preset(str(path))
    assert loaded.preserve_triangulation is False
    assert loaded.mode == 1
