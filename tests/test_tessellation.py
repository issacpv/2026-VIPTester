"""Tests for the equilateral-fill tessellation generator (task 5).

Covers the polygon helpers, the generator's coverage / interior
near-equilaterality / boundary-closure behaviour, and the
Lattice.from_tessellation integration. Pure (no GUI) — drives the
auxetic package directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxetic.lattice import Lattice
from auxetic.tessellation import (
    TessellationResult,
    distance_to_polygon,
    edge_from_triangle_count,
    equilateral_deviation,
    equilateral_grid,
    generate_tessellation,
    points_in_polygon,
    polygon_area,
    resample_polygon,
    triangle_angles,
)


UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
# Concave L-shape, true area = 3.
L_SHAPE = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)


def _tri_area(p):
    """Triangle area from a (3, 2) vertex array (explicit 2D determinant,
    avoids the deprecated 2D np.cross)."""
    (x0, y0), (x1, y1), (x2, y2) = p
    return abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2.0


def _covered_area(res):
    return sum(_tri_area(res.points[s]) for s in res.simplices)


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------

def test_polygon_area_square_and_triangle():
    assert polygon_area(UNIT_SQUARE) == pytest.approx(1.0)
    tri = np.array([[0, 0], [1, 0], [0, 1]], float)
    assert polygon_area(tri) == pytest.approx(0.5)
    assert polygon_area(L_SHAPE) == pytest.approx(3.0)


def test_points_in_polygon_basic():
    pts = np.array([[0.5, 0.5], [-0.1, 0.5], [1.5, 0.5], [0.99, 0.99]], float)
    inside = points_in_polygon(pts, UNIT_SQUARE)
    assert list(inside) == [True, False, False, True]


def test_points_in_polygon_concavity():
    # The notch of the L (point (1.5, 1.5)) is OUTSIDE the region.
    pts = np.array([[0.5, 0.5], [1.5, 1.5], [0.5, 1.5]], float)
    inside = points_in_polygon(pts, L_SHAPE)
    assert list(inside) == [True, False, True]


def test_resample_polygon_respects_max_segment():
    out = resample_polygon(UNIT_SQUARE, 0.25)
    # No consecutive segment longer than the cap.
    loop = np.vstack([out, out[0]])
    seglen = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    assert seglen.max() <= 0.25 + 1e-9
    # Original corners are still present.
    for corner in UNIT_SQUARE:
        assert np.any(np.all(np.isclose(out, corner), axis=1))


def test_equilateral_grid_row_spacing():
    grid = equilateral_grid(0.0, 0.0, 1.0, 1.0, 0.2)
    ys = np.unique(np.round(grid[:, 1], 9))
    drow = np.diff(ys)
    assert np.allclose(drow, 0.2 * np.sqrt(3) / 2.0)


def test_triangle_angles_equilateral_and_right():
    eq = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], float)
    assert np.allclose(triangle_angles(eq), [60, 60, 60], atol=1e-6)
    right = np.array([[0, 0], [1, 0], [0, 1]], float)
    assert np.allclose(triangle_angles(right), [45, 45, 90], atol=1e-6)


def test_equilateral_deviation_zero_for_equilateral():
    eq = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], float)
    assert equilateral_deviation(eq) == pytest.approx(0.0, abs=1e-6)


def test_edge_from_triangle_count_roundtrips_area():
    area = 5.0
    n = 80
    edge = edge_from_triangle_count(area, n)
    per_tri = np.sqrt(3) / 4 * edge ** 2
    assert per_tri * n == pytest.approx(area)


# ---------------------------------------------------------------------------
# Generator: coverage
# ---------------------------------------------------------------------------

def test_square_full_coverage():
    res = generate_tessellation(UNIT_SQUARE, target_edge=0.18)
    assert _covered_area(res) == pytest.approx(polygon_area(UNIT_SQUARE), rel=1e-6)


def test_concave_region_is_carved_out():
    """The covered area must equal the L's true area (3), not the convex
    hull's area (3.5) — concavity triangles are clipped."""
    res = generate_tessellation(L_SHAPE, target_edge=0.2)
    assert _covered_area(res) == pytest.approx(3.0, rel=1e-6)
    # And no triangle centroid lies in the notch.
    cent = res.points[res.simplices].mean(axis=1)
    assert np.all(points_in_polygon(cent, L_SHAPE))


def test_all_points_inside_or_on_boundary():
    res = generate_tessellation(UNIT_SQUARE, target_edge=0.2)
    # Every vertex is inside or within a hair of the boundary.
    inside = points_in_polygon(res.points, UNIT_SQUARE)
    onbnd = distance_to_polygon(res.points, UNIT_SQUARE) < 1e-9
    assert np.all(inside | onbnd)


# ---------------------------------------------------------------------------
# Generator: interior near-equilaterality + boundary closers
# ---------------------------------------------------------------------------

def test_interior_triangles_are_equilateral():
    res = generate_tessellation(UNIT_SQUARE, target_edge=0.18)
    mask = res.interior_triangle_mask()
    assert mask.sum() >= 10  # a meaningful interior region exists
    devs = np.array([equilateral_deviation(res.points[s])
                     for s in res.simplices[mask]])
    # Interior tiles inherit the grid's equilateral shape exactly.
    assert devs.max() < 0.01


def test_boundary_triangles_exist_and_close_the_region():
    res = generate_tessellation(UNIT_SQUARE, target_edge=0.18)
    mask = res.interior_triangle_mask()
    n_boundary_tris = int((~mask).sum())
    # There is a ring of closer triangles touching the boundary...
    assert n_boundary_tris > 0
    # ...and together with the interior they still cover the whole region.
    assert _covered_area(res) == pytest.approx(1.0, rel=1e-6)


def test_most_boundary_closers_are_not_equilateral():
    """The closers bridging grid to boundary are isosceles/scalene — they
    should deviate from 60° more than the (perfect) interior tiles."""
    res = generate_tessellation(UNIT_SQUARE, target_edge=0.18)
    mask = res.interior_triangle_mask()
    closer_devs = np.array([equilateral_deviation(res.points[s])
                            for s in res.simplices[~mask]])
    # At least some closers are clearly irregular.
    assert closer_devs.max() > 5.0


# ---------------------------------------------------------------------------
# Generator: density control
# ---------------------------------------------------------------------------

def test_smaller_edge_gives_more_triangles():
    coarse = generate_tessellation(UNIT_SQUARE, target_edge=0.3)
    fine = generate_tessellation(UNIT_SQUARE, target_edge=0.12)
    assert fine.n_triangles > coarse.n_triangles


def test_n_triangles_density_is_in_the_ballpark():
    res = generate_tessellation(UNIT_SQUARE, n_triangles=80)
    # The grid + clipping won't hit the count exactly, but should be the
    # right order of magnitude.
    assert 40 <= res.n_triangles <= 160


# ---------------------------------------------------------------------------
# Generator: validation
# ---------------------------------------------------------------------------

def test_requires_a_density():
    with pytest.raises(ValueError):
        generate_tessellation(UNIT_SQUARE)


def test_rejects_degenerate_boundary():
    with pytest.raises(ValueError):
        generate_tessellation(np.array([[0, 0], [1, 0]], float), target_edge=0.2)


def test_rejects_zero_area_boundary():
    collinear = np.array([[0, 0], [1, 0], [2, 0]], float)
    with pytest.raises(ValueError):
        generate_tessellation(collinear, target_edge=0.2)


# ---------------------------------------------------------------------------
# Lattice integration
# ---------------------------------------------------------------------------

def test_lattice_from_tessellation_builds_2d_lattice():
    lat = Lattice.from_tessellation(UNIT_SQUARE, target_edge=0.2, mode=1)
    assert lat.mode == 1
    assert lat.points.shape[1] == 2
    # Points live in the unit square (lattice convention).
    assert lat.points.min() >= -1e-9
    assert lat.points.max() <= 1.0 + 1e-9
    assert len(lat.tri.simplices) > 0


def test_lattice_from_tessellation_preserves_triangulation():
    res = generate_tessellation(UNIT_SQUARE, target_edge=0.2)
    lat = Lattice.from_tessellation(UNIT_SQUARE, target_edge=0.2, mode=1,
                                    preserve_triangulation=True)
    assert len(lat.tri.simplices) == res.n_triangles


def test_lattice_from_tessellation_normalization_preserves_equilaterality():
    # Uniform scaling keeps interior tiles equilateral in lattice space.
    lat = Lattice.from_tessellation(UNIT_SQUARE, target_edge=0.18, mode=1)
    devs = np.array([equilateral_deviation(lat.points[s])
                     for s in lat.tri.simplices])
    # A solid fraction of tiles are (near-)equilateral after normalization.
    assert (devs < 0.5).sum() >= 10


def test_lattice_from_tessellation_drives_exporters(tmp_path):
    lat = Lattice.from_tessellation(UNIT_SQUARE, target_edge=0.25, mode=1)
    stl = tmp_path / "t.stl"
    scad = tmp_path / "t.scad"
    lat.to_stl(str(stl), verbose=False)
    lat.to_scad(str(scad), verbose=False)
    assert stl.exists() and stl.stat().st_size > 0
    assert "polyhedron(" in scad.read_text()


def test_lattice_from_tessellation_rejects_3d_mode():
    with pytest.raises(ValueError):
        Lattice.from_tessellation(UNIT_SQUARE, target_edge=0.2, mode=6)


def test_lattice_from_tessellation_count_density():
    lat = Lattice.from_tessellation(UNIT_SQUARE, n_triangles=60, mode=1)
    assert len(lat.tri.simplices) > 0
