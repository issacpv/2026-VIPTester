"""Tests for the density-gradient extension to ``auxetic.geometry.generate_points``.

The default behaviour (``density_axis="none"``, ``density_law="uniform"``)
must reproduce the legacy V20 random sample byte-for-byte; that's covered
elsewhere in ``tests/test_regression.py``. This file exercises the new
biased path: that ``"linear"``, ``"log"``, ``"exp"`` warps move the
sample distribution in the expected direction along the chosen axis,
that grid modes ignore the gradient, and that the inverse-CDF outputs
stay inside [0, 1].
"""

from __future__ import annotations

import numpy as np
import pytest

from auxetic.geometry import _density_bias_inv_cdf, generate_points


N_SAMPLES = 2000  # large enough to make mean comparisons stable
SEED      = 42


# ---------------------------------------------------------------------------
# Default-preservation: same legacy points whether we pass the new kwargs
# explicitly at their defaults or not at all.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5, 6])
def test_defaults_match_positional_call(mode):
    n_points = 5 if mode in (1, 2, 3) else (6 if mode in (4, 5) else 8)

    np.random.seed(SEED)
    pts_a, _ = generate_points(n_points, mode)

    np.random.seed(SEED)
    pts_b, _ = generate_points(
        n_points, mode,
        density_axis="none", density_law="uniform", density_strength=1.0,
    )

    np.testing.assert_array_equal(pts_a, pts_b)


# ---------------------------------------------------------------------------
# Inverse-CDF helper: range and shape sanity.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", ["linear", "log", "exp"])
@pytest.mark.parametrize("strength", [-1.0, -0.5, 0.5, 1.0])
def test_inv_cdf_stays_in_unit_interval(law, strength):
    u = np.linspace(0.0, 1.0, 257)
    x = _density_bias_inv_cdf(u, law, strength)
    assert np.all(x >= 0.0 - 1e-12), f"{law} s={strength} produced x < 0"
    assert np.all(x <= 1.0 + 1e-12), f"{law} s={strength} produced x > 1"
    # Endpoints should be preserved exactly (within fp tolerance).
    assert abs(x[0] - 0.0) < 1e-9
    assert abs(x[-1] - 1.0) < 1e-9


def test_inv_cdf_uniform_law_is_identity():
    u = np.linspace(0.0, 1.0, 100)
    np.testing.assert_array_equal(_density_bias_inv_cdf(u, "uniform", 5.0), u)


def test_inv_cdf_zero_strength_is_identity():
    u = np.linspace(0.0, 1.0, 100)
    for law in ("linear", "log", "exp"):
        out = _density_bias_inv_cdf(u, law, 0.0)
        np.testing.assert_allclose(out, u, atol=1e-12)


# ---------------------------------------------------------------------------
# Biased sampling: distributions shift in the expected direction.
# ---------------------------------------------------------------------------

def _sample(mode, *, axis, law, strength, n=N_SAMPLES, seed=SEED):
    np.random.seed(seed)
    pts, _ = generate_points(
        n, mode,
        density_axis=axis, density_law=law, density_strength=strength,
    )
    return pts


@pytest.mark.parametrize("mode", [1, 2])  # 2D random Delaunay modes
def test_linear_positive_strength_biases_toward_one(mode):
    pts = _sample(mode, axis="x", law="linear", strength=1.0)
    assert pts[:, 0].mean() > 0.55, "expected mean(x) > 0.5 with positive strength"


@pytest.mark.parametrize("mode", [1, 2])
def test_linear_negative_strength_biases_toward_zero(mode):
    pts = _sample(mode, axis="x", law="linear", strength=-1.0)
    assert pts[:, 0].mean() < 0.45, "expected mean(x) < 0.5 with negative strength"


@pytest.mark.parametrize("mode", [1, 2])
def test_log_positive_strength_biases_toward_one(mode):
    pts = _sample(mode, axis="x", law="log", strength=2.0)
    assert pts[:, 0].mean() > 0.55, "log inverse-CDF (concave) should cluster near x=1 for s>0"


@pytest.mark.parametrize("mode", [1, 2])
def test_exp_positive_strength_biases_toward_zero(mode):
    pts = _sample(mode, axis="x", law="exp", strength=2.0)
    assert pts[:, 0].mean() < 0.45, "exp inverse-CDF (convex) should cluster near x=0 for s>0"


def test_log_and_exp_are_inverses():
    """Sampling with ``law="log"`` and ``law="exp"`` at the same |s| should
    produce mean(x) values that are mirrored around 0.5 (the laws are
    each other's inverse-CDFs)."""
    pts_log = _sample(mode=1, axis="x", law="log", strength=2.0)
    pts_exp = _sample(mode=1, axis="x", law="exp", strength=2.0)
    log_mean = pts_log[:, 0].mean()
    exp_mean = pts_exp[:, 0].mean()
    assert abs((log_mean - 0.5) + (exp_mean - 0.5)) < 0.05


def test_axis_y_biases_y_not_x():
    pts = _sample(mode=1, axis="y", law="log", strength=2.0)
    assert pts[:, 1].mean() > 0.55, "y-axis bias should shift y mean"
    # x mean should be near 0.5 (uniform); allow slack for small-N noise
    assert abs(pts[:, 0].mean() - 0.5) < 0.05


def test_axis_z_works_for_3d_mode():
    pts = _sample(mode=3, axis="z", law="log", strength=2.0)
    assert pts[:, 2].mean() > 0.55


def test_axis_z_silently_ignored_for_2d_mode():
    """Mode 1 has no z coord; asking to bias z should fall back to uniform."""
    pts = _sample(mode=1, axis="z", law="log", strength=2.0)
    # Should be a plain uniform 2D draw — mean ≈ 0.5 on both axes.
    assert abs(pts[:, 0].mean() - 0.5) < 0.05
    assert abs(pts[:, 1].mean() - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Grid modes ignore density gradients.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,n", [(4, 9), (5, 9), (6, 8)])
def test_grid_modes_ignore_density(mode, n):
    np.random.seed(SEED)
    pts_default, _ = generate_points(n, mode)
    np.random.seed(SEED)
    pts_biased, _ = generate_points(
        n, mode, density_axis="x", density_law="exp", density_strength=2.0,
    )
    np.testing.assert_array_equal(pts_default, pts_biased)


# ---------------------------------------------------------------------------
# Negative axis silently uniform: unknown axis name falls through.
# ---------------------------------------------------------------------------

def test_unknown_axis_falls_through_to_uniform():
    pts = _sample(mode=1, axis="bogus", law="log", strength=2.0)
    # Same shape, distribution should look uniform
    assert pts.shape == (N_SAMPLES, 2)
    assert abs(pts[:, 0].mean() - 0.5) < 0.05
    assert abs(pts[:, 1].mean() - 0.5) < 0.05
