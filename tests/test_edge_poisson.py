"""Tests for the edge-vector generalized Poisson's ratio (task 4).

Covers the per-triangle metric (equilateral is the isotropic ν = -1
known case; the metric varies with shape and C), the documented
corner-motion isotropy that motivates using edge connection points, the
shape/C sweep, and input validation. Pure (no GUI).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxetic.lattice import Lattice
from auxetic.edge_poisson import (
    PoissonSweep,
    actuated_corners,
    apex_triangle,
    edge_midpoint_triangle,
    equilateral_triangle,
    generalized_poisson_ratio,
    hinge_fraction,
    morph_triangle,
    sweep_poisson,
    sweep_shape_and_C,
    triangle_strain_tensor,
)


EQ = equilateral_triangle()


# ---------------------------------------------------------------------------
# Known case: equilateral is isotropic (ν = -1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.05, 0.2, 0.4])
@pytest.mark.parametrize("C", [0.3, 1.0, 3.0])
def test_equilateral_is_minus_one(theta, C):
    assert generalized_poisson_ratio(EQ, C, theta) == pytest.approx(-1.0, abs=1e-6)


def test_equilateral_directional_is_minus_one_any_axis():
    for axis in ([1, 0], [0, 1], [1, 1], [2, -3]):
        assert generalized_poisson_ratio(EQ, 1.0, 0.2, axis=axis) == \
            pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Zero / degenerate actuation
# ---------------------------------------------------------------------------

def test_zero_actuation_is_nan():
    assert np.isnan(generalized_poisson_ratio(EQ, 1.0, 0.0))


def test_zero_actuation_strain_is_zero():
    strain = triangle_strain_tensor(EQ, 1.0, 0.0)
    assert np.allclose(strain, 0.0)


def test_strain_tensor_is_symmetric():
    strain = triangle_strain_tensor(apex_triangle(0.2, 0.6), 1.0, 0.2)
    assert np.allclose(strain, strain.T)


# ---------------------------------------------------------------------------
# The metric VARIES with shape and C
# ---------------------------------------------------------------------------

def test_isosceles_differs_from_equilateral():
    iso = apex_triangle(0.5, 1.3)        # symmetric but taller -> isosceles
    nu = generalized_poisson_ratio(iso, 1.0, 0.15)
    assert abs(nu - (-1.0)) > 0.1        # no longer the isotropic -1


def test_scalene_differs_from_equilateral():
    scal = apex_triangle(0.15, 0.55)
    nu = generalized_poisson_ratio(scal, 1.0, 0.15)
    assert abs(nu - (-1.0)) > 0.1


def test_metric_varies_with_C_on_scalene():
    scal = apex_triangle(0.2, 0.6)
    vals = [generalized_poisson_ratio(scal, C, 0.15) for C in (0.3, 1.0, 3.0)]
    # The three C values give three distinct ratios.
    assert max(vals) - min(vals) > 0.1


def test_shape_sweep_starts_at_minus_one_and_moves_away():
    s_vals = np.linspace(0.0, 1.0, 6)
    nus = [generalized_poisson_ratio(morph_triangle(s), 1.0, 0.15) for s in s_vals]
    assert nus[0] == pytest.approx(-1.0, abs=1e-6)          # equilateral
    # The scalene end is well away from -1.
    assert abs(nus[-1] - (-1.0)) > 0.5


# ---------------------------------------------------------------------------
# Documented corner-motion isotropy (why we use edge connection points)
# ---------------------------------------------------------------------------

def test_corner_motion_is_isotropic_even_for_scalene():
    """The actuated corners deform by a shape-independent gradient
    B = t·R(θ)+(1-t)·I, whose symmetric part is a scalar multiple of I —
    so a corner-based Poisson ratio would be identically -1. This test
    pins that property (the reason the metric uses edge midpoints)."""
    P = apex_triangle(0.2, 0.6)          # clearly scalene
    Q0 = actuated_corners(P, 1.0, 0.0)   # == P (identity at theta 0)
    Q = actuated_corners(P, 1.0, 0.25)
    E0 = np.column_stack([Q0[1] - Q0[0], Q0[2] - Q0[0]])
    E = np.column_stack([Q[1] - Q[0], Q[2] - Q[0]])
    A = E @ np.linalg.inv(E0)
    sym = 0.5 * (A + A.T)
    assert abs(sym[0, 1]) < 1e-9                 # no shear
    assert abs(sym[0, 0] - sym[1, 1]) < 1e-9     # equal normal strains => isotropic


def test_edge_midpoint_triangle_rest_lies_on_edges():
    """At theta=0 each edge midpoint sits on its triangle edge."""
    P = apex_triangle(0.3, 0.7)
    mids = edge_midpoint_triangle(P, 1.0, 0.0)
    for (a, b), m in zip(((0, 1), (1, 2), (2, 0)), mids):
        # m is a convex combination of P[a], P[b] (on the segment).
        ab = P[b] - P[a]
        t = np.dot(m - P[a], ab) / np.dot(ab, ab)
        assert -1e-9 <= t <= 1 + 1e-9
        perp = np.linalg.norm((m - P[a]) - t * ab)
        assert perp < 1e-9


# ---------------------------------------------------------------------------
# Sweep API
# ---------------------------------------------------------------------------

def test_sweep_poisson_shape():
    tris = [morph_triangle(s) for s in (0.0, 0.5, 1.0)]
    grid = sweep_poisson(tris, [0.5, 1.0, 2.0], theta=0.15)
    assert grid.shape == (3, 3)
    assert np.allclose(grid[0], -1.0, atol=1e-6)   # equilateral row


def test_sweep_shape_and_C_dataclass():
    sw = sweep_shape_and_C(np.linspace(0, 1, 5), [0.5, 1.0, 2.0], theta=0.12)
    assert isinstance(sw, PoissonSweep)
    assert sw.ratios.shape == (5, 3)
    assert len(sw.triangles) == 5
    assert np.allclose(sw.ratios[0], -1.0, atol=1e-6)     # equilateral row
    # Some shape row differs from the equilateral row.
    assert np.any(np.abs(sw.ratios[-1] - (-1.0)) > 0.1)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def test_morph_triangle_endpoints():
    eq = morph_triangle(0.0)
    # s=0 is equilateral: all sides equal.
    sides = [np.linalg.norm(eq[(i + 1) % 3] - eq[i]) for i in range(3)]
    assert np.allclose(sides, sides[0], atol=1e-9)
    # s=1 is scalene: all sides distinct.
    sc = morph_triangle(1.0)
    s2 = sorted(np.linalg.norm(sc[(i + 1) % 3] - sc[i]) for i in range(3))
    assert s2[1] - s2[0] > 1e-3 and s2[2] - s2[1] > 1e-3


def test_hinge_fraction():
    assert hinge_fraction(1.0) == pytest.approx(0.5)
    assert hinge_fraction(3.0) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_rejects_nonpositive_C():
    with pytest.raises(ValueError):
        generalized_poisson_ratio(EQ, 0.0, 0.2)
    with pytest.raises(ValueError):
        hinge_fraction(-1.0)


def test_rejects_bad_triangle_shape():
    with pytest.raises(ValueError):
        triangle_strain_tensor(np.zeros((4, 2)), 1.0, 0.2)


def test_rejects_zero_axis():
    with pytest.raises(ValueError):
        generalized_poisson_ratio(EQ, 1.0, 0.2, axis=[0, 0])


# ---------------------------------------------------------------------------
# Lattice integration: Lattice.edge_vector_poisson_ratio
# ---------------------------------------------------------------------------

def test_lattice_single_equilateral_triangle_is_minus_one():
    lat = Lattice(mode=1, n_points=3, seed=1)
    lat.regenerate_from_points(equilateral_triangle())
    assert lat.edge_vector_poisson_ratio() == pytest.approx(-1.0, abs=1e-6)


def test_lattice_3d_mode_is_nan():
    lat = Lattice(mode=6, n_points=8)
    assert math.isnan(lat.edge_vector_poisson_ratio())


def test_lattice_2d_mode_is_finite():
    lat = Lattice(mode=1, n_points=8, seed=2)
    assert math.isfinite(lat.edge_vector_poisson_ratio())


def test_lattice_zero_actuation_is_nan():
    lat = Lattice(mode=1, n_points=3, seed=1)
    lat.regenerate_from_points(equilateral_triangle())
    assert math.isnan(lat.edge_vector_poisson_ratio(theta=0.0))


def test_lattice_equilateral_tessellation_is_auxetic():
    """An equilateral-fill tessellation is dominated by equilateral
    interior tiles, so its mean edge-vector ν is strongly negative."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    lat = Lattice.from_tessellation(square, target_edge=0.16, mode=1)
    nu = lat.edge_vector_poisson_ratio()
    assert math.isfinite(nu)
    # Clearly auxetic on average (interior equilateral tiles dominate the
    # mean; the scalene boundary closers pull it up from the ideal -1).
    assert nu < -0.2


def test_lattice_eqhex_full_structure_is_minus_one():
    """The EqHex repro (mode-11 hexagon of equilateral tiles): the
    full-structure edge-vector ν is ~-1, the meaningful auxetic value the
    task-6b readout surfaces — even though the bbox ν reads ~0 for this
    symmetric rotating-units mechanism."""
    eqhex = np.array([
        [0.0, 1.7320508075688772], [1.0, 0.0], [1.0, 3.4641016151377544],
        [2.0, 1.7320508075688772], [3.0, 0.0], [3.0, 3.4641016151377544],
        [4.0, 1.7320508075688772],
    ])
    lat = Lattice(mode=11, n_points=7, ratio=0.35)
    lat.regenerate_from_points(eqhex)
    assert lat.edge_vector_poisson_ratio() == pytest.approx(-1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Per-triangle pick: Lattice.poisson_ratio_at_point (task 6c)
# ---------------------------------------------------------------------------

def test_poisson_ratio_at_point_picks_containing_triangle():
    """A lattice-space point maps to its Delaunay triangle, and the returned
    ν matches that triangle's direct generalized_poisson_ratio."""
    lat = Lattice(mode=1, n_points=8, seed=2)
    simplices = np.asarray(lat.tri.simplices)
    pts = np.asarray(lat.points)
    tri0 = pts[simplices[0]]
    idx, nu = lat.poisson_ratio_at_point(tri0.mean(axis=0), world=False)
    assert idx == 0
    expected = generalized_poisson_ratio(tri0, float(lat.C), 0.1)
    np.testing.assert_allclose(nu, expected, equal_nan=True)


def test_poisson_ratio_at_point_equilateral_is_minus_one():
    lat = Lattice(mode=1, n_points=3, seed=1)
    lat.regenerate_from_points(equilateral_triangle())
    pts = np.asarray(lat.points)
    idx, nu = lat.poisson_ratio_at_point(pts.mean(axis=0), world=False)
    assert idx is not None
    assert nu == pytest.approx(-1.0, abs=1e-6)


def test_poisson_ratio_at_point_world_identity_matches_lattice():
    """With the default identity orientation, a world-frame point maps to
    the same triangle as the equivalent lattice-space point."""
    lat = Lattice(mode=1, n_points=8, seed=2)
    pts = np.asarray(lat.points)
    centroid = pts[np.asarray(lat.tri.simplices)[1]].mean(axis=0)
    idx_local, nu_local = lat.poisson_ratio_at_point(centroid, world=False)
    idx_world, nu_world = lat.poisson_ratio_at_point(
        np.array([centroid[0], centroid[1], 0.0]), world=True)
    assert idx_world == idx_local
    np.testing.assert_allclose(nu_world, nu_local, equal_nan=True)


def test_poisson_ratio_at_point_3d_is_none():
    lat = Lattice(mode=6, n_points=8)
    idx, nu = lat.poisson_ratio_at_point(np.array([0.5, 0.5, 0.5]), world=True)
    assert idx is None
    assert math.isnan(nu)


def test_poisson_ratio_at_point_outside_hull_falls_back():
    """A point well outside the convex hull still resolves to the nearest
    triangle rather than failing."""
    lat = Lattice(mode=1, n_points=8, seed=2)
    idx, nu = lat.poisson_ratio_at_point(np.array([99.0, 99.0]), world=False)
    assert idx is not None
    assert 0 <= idx < len(np.asarray(lat.tri.simplices))
