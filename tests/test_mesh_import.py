"""Tests for ``auxetic.mesh_io`` — STL/OBJ vertex import, deduplication,
unit-cube normalisation, and uniform decimation.

The repo ships with a real STL at ``auxetic_lattice.stl`` (54 KB); the
tests use it as smoke coverage and synthesise tiny OBJ/STL files in
``tmp_path`` for the unit cases.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from auxetic.mesh_io import (
    decimate_uniform,
    normalize_to_unit_cube,
    read_mesh_vertices,
)


REPO_ROOT     = Path(__file__).resolve().parent.parent
SHIPPED_STL   = REPO_ROOT / "auxetic_lattice.stl"


# ---------------------------------------------------------------------------
# OBJ import (no external dep, simplest case).
# ---------------------------------------------------------------------------

def _write_obj(tmp_path, lines):
    p = tmp_path / "test.obj"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_read_obj_basic_vertices(tmp_path):
    lines = [
        "# unit square plus an apex",
        "v 0 0 0",
        "v 1 0 0",
        "v 0 1 0",
        "v 1 1 0",
        "v 0.5 0.5 1",
    ]
    path = _write_obj(tmp_path, lines)
    verts = read_mesh_vertices(str(path))
    assert verts.shape == (5, 3)
    assert np.allclose(verts[0], [0, 0, 0])
    assert np.allclose(verts[-1], [0.5, 0.5, 1])


def test_read_obj_dedupes_repeated_vertices(tmp_path):
    lines = [
        "v 0 0 0",
        "v 1 0 0",
        "v 0 0 0",  # duplicate
        "v 0 1 0",
        "v 1 0 0",  # duplicate
    ]
    path = _write_obj(tmp_path, lines)
    verts = read_mesh_vertices(str(path))
    assert verts.shape == (3, 3)


def test_read_obj_skips_non_v_lines(tmp_path):
    lines = [
        "# comment",
        "vt 0.0 1.0",      # texture coords
        "vn 0.0 0.0 1.0",  # normals
        "v 0 0 0",
        "v 1 1 1",
        "f 1 2 3",         # face — irrelevant
    ]
    path = _write_obj(tmp_path, lines)
    verts = read_mesh_vertices(str(path))
    assert verts.shape == (2, 3)


def test_read_obj_short_or_garbage_v_lines_are_skipped(tmp_path):
    lines = [
        "v 0 0",         # only two coords — skip
        "v foo bar baz", # non-numeric — skip
        "v 1 2 3",       # valid
    ]
    path = _write_obj(tmp_path, lines)
    verts = read_mesh_vertices(str(path))
    assert verts.shape == (1, 3)
    assert np.allclose(verts[0], [1, 2, 3])


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "weird.ply"
    p.write_text("ply\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        read_mesh_vertices(str(p))


# ---------------------------------------------------------------------------
# STL import (depends on numpy-stl, which is already a project dep).
# ---------------------------------------------------------------------------

def test_read_shipped_stl_has_unique_vertices():
    if not SHIPPED_STL.exists():
        pytest.skip("auxetic_lattice.stl not present")
    verts = read_mesh_vertices(str(SHIPPED_STL))
    assert verts.ndim == 2
    assert verts.shape[1] == 3
    assert verts.shape[0] > 0
    # Dedup: every row should be unique to ~1e-9.
    rounded = np.round(verts, 9)
    unique = np.unique(rounded, axis=0)
    assert unique.shape[0] == verts.shape[0]


def test_read_synthetic_stl(tmp_path):
    """Build a tiny STL with numpy-stl, read it back."""
    pytest.importorskip("stl")
    from stl import mesh as stl_mesh

    triangles = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ], dtype=float)
    m = stl_mesh.Mesh(np.zeros(triangles.shape[0], dtype=stl_mesh.Mesh.dtype))
    m.vectors = triangles
    out = tmp_path / "tiny.stl"
    m.save(str(out))

    verts = read_mesh_vertices(str(out))
    assert verts.shape == (4, 3)  # 2 tris share 2 verts → 4 unique


# ---------------------------------------------------------------------------
# normalize_to_unit_cube
# ---------------------------------------------------------------------------

def test_normalize_basic_box():
    verts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 2.0],
        [-2.0, -3.0, 0.5],
    ])
    out = normalize_to_unit_cube(verts)
    assert np.all(out >= 0.0 - 1e-12)
    assert np.all(out <= 1.0 + 1e-12)
    # Min and max should hit 0 and 1 on every axis with extent.
    np.testing.assert_allclose(out.min(axis=0), [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(out.max(axis=0), [1, 1, 1], atol=1e-12)


def test_normalize_collapsed_axis_set_to_half():
    """A 2D-in-3D mesh (z constant) should map z to 0.5 on every vertex."""
    verts = np.array([
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
    ])
    out = normalize_to_unit_cube(verts)
    np.testing.assert_allclose(out[:, 2], 0.5, atol=1e-12)


def test_normalize_empty_returns_empty():
    out = normalize_to_unit_cube(np.zeros((0, 3)))
    assert out.shape == (0, 3)


# ---------------------------------------------------------------------------
# decimate_uniform
# ---------------------------------------------------------------------------

def test_decimate_below_target_returns_input():
    verts = np.random.RandomState(0).rand(10, 3)
    out = decimate_uniform(verts, n=20)
    np.testing.assert_array_equal(out, verts)


def test_decimate_to_smaller_count():
    verts = np.random.RandomState(0).rand(100, 3)
    out = decimate_uniform(verts, n=25, seed=42)
    assert out.shape == (25, 3)
    # Every output row must come from the input set.
    rounded_in = {tuple(np.round(r, 9)) for r in verts}
    for r in out:
        assert tuple(np.round(r, 9)) in rounded_in


def test_decimate_is_deterministic_with_seed():
    verts = np.random.RandomState(0).rand(100, 3)
    a = decimate_uniform(verts, n=25, seed=42)
    b = decimate_uniform(verts, n=25, seed=42)
    np.testing.assert_array_equal(a, b)


def test_decimate_returns_sorted_indices():
    """Sorted indices preserve the relative ordering of the original mesh,
    which makes downstream debugging easier."""
    verts = np.arange(60).reshape(20, 3).astype(float)
    out = decimate_uniform(verts, n=5, seed=0)
    # Each row of `out` should appear in `verts` in the same monotone order.
    in_first_col = verts[:, 0]
    out_first_col = out[:, 0]
    assert list(out_first_col) == sorted(out_first_col)
    assert all(v in in_first_col for v in out_first_col)


# ---------------------------------------------------------------------------
# Lattice.from_mesh end-to-end integration
# ---------------------------------------------------------------------------

def _write_grid_obj(tmp_path, n=5):
    """Make an OBJ with an n×n grid of vertices in z=0 (so 2D import works)
    plus a single vertex at z=1 to exercise 3D import."""
    rows = []
    for j in range(n):
        for i in range(n):
            x = i / (n - 1)
            y = j / (n - 1)
            rows.append(f"v {x} {y} 0.0")
    rows.append("v 0.5 0.5 1.0")  # apex for 3D
    p = tmp_path / "grid.obj"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def test_lattice_from_mesh_2d_pipeline_ok(tmp_path):
    from auxetic import Lattice
    obj = _write_grid_obj(tmp_path, n=4)
    lat = Lattice.from_mesh(str(obj), dim="2D", ratio=0.35)
    assert lat.mode == 7
    assert lat.points.shape[1] == 2
    assert lat.mesh_path == str(obj)
    # STL export must succeed for the new mode.
    stl = tmp_path / "out.stl"
    lat.to_stl(str(stl), verbose=False)
    assert stl.exists() and stl.stat().st_size > 0


def test_lattice_from_mesh_2_5d_extruded(tmp_path):
    from auxetic import Lattice
    obj = _write_grid_obj(tmp_path, n=4)
    lat = Lattice.from_mesh(str(obj), dim="2.5D", nz_layers=3)
    assert lat.mode == 8
    assert lat.points.shape[1] == 2
    stl = tmp_path / "out.stl"
    lat.to_stl(str(stl), verbose=False)
    assert stl.exists() and stl.stat().st_size > 0


def test_lattice_from_mesh_3d_pipeline_ok(tmp_path):
    from auxetic import Lattice
    obj = _write_grid_obj(tmp_path, n=3)
    lat = Lattice.from_mesh(str(obj), dim=3)
    assert lat.mode == 9
    assert lat.points.shape[1] == 3
    stl = tmp_path / "out.stl"
    lat.to_stl(str(stl), verbose=False)
    assert stl.exists() and stl.stat().st_size > 0


def test_lattice_from_mesh_decimation_caps_count(tmp_path):
    from auxetic import Lattice
    obj = _write_grid_obj(tmp_path, n=5)  # 26 vertices total
    # Use 2D so 3D Delaunay degeneracy (coplanar points after random
    # decimation) doesn't muddle this test — decimation is dim-agnostic.
    lat = Lattice.from_mesh(str(obj), dim="2D", decimate_to=10, seed=42)
    assert lat.mesh_vertices.shape[0] == 10


def test_lattice_from_mesh_invalid_dim_raises(tmp_path):
    from auxetic import Lattice
    obj = _write_grid_obj(tmp_path, n=3)
    with pytest.raises(ValueError, match="dim must be"):
        Lattice.from_mesh(str(obj), dim="banana")


def test_lattice_from_mesh_empty_mesh_raises(tmp_path):
    from auxetic import Lattice
    obj = tmp_path / "empty.obj"
    obj.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No vertices"):
        Lattice.from_mesh(str(obj), dim="2D")


def test_lattice_mesh_mode_without_vertices_raises():
    from auxetic import Lattice
    # Build a normal mode-1 lattice, then try to flip to mode 9 without
    # mesh_vertices — regenerate must complain.
    lat = Lattice(mode=1, n_points=5, seed=0)
    lat.mode = 9
    lat.mesh_vertices = None
    with pytest.raises(RuntimeError, match="mesh_vertices"):
        lat.regenerate()


def test_lattice_with_edge_flips_passes_through_export(tmp_path):
    """Lattice with non-empty edge_flips still exports — the apply_edge_flips
    pass produces a tri whose simplices the STL pipeline accepts."""
    from auxetic import Lattice
    from auxetic.geometry import flippable_edges
    lat = Lattice(mode=1, n_points=12, seed=7)
    edges = flippable_edges(lat.tri, lat.points)
    assert edges, "fixture should have flippable edges"
    lat.edge_flips = {edges[0]}
    lat.regenerate_from_points(lat.points)  # re-triangulate w/ flip
    stl = tmp_path / "out.stl"
    lat.to_stl(str(stl), verbose=False)
    assert stl.exists() and stl.stat().st_size > 0


def test_lattice_with_density_gradient_does_not_match_default(tmp_path):
    from auxetic import Lattice
    lat_default = Lattice(mode=1, n_points=20, seed=42)
    lat_biased  = Lattice(mode=1, n_points=20, seed=42,
                          density_axis="x", density_law="linear",
                          density_strength=1.0)
    # Same seed, biased path → x-mean shifted higher than uniform.
    assert lat_biased.points[:, 0].mean() > lat_default.points[:, 0].mean() - 0.01
    # And export still works.
    stl = tmp_path / "biased.stl"
    lat_biased.to_stl(str(stl), verbose=False)
    assert stl.exists() and stl.stat().st_size > 0
