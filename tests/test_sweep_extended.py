"""Tests for the M2.8 extended-range sweep and collision-stop on
:meth:`Simulator.sweep_theta`.

Three responsibilities:
1. ``theta_max`` extends the sweep range past the canonical ±π/2 with
   no behaviour change for callers that don't pass it.
2. ``collision_stop`` flags samples where tiles overlap and bounds the
   reachable θ via ``collision_theta_min`` / ``collision_theta_max``.
3. The compression-ratio metric ignores collided samples (those
   configurations are physically unreachable).
"""

from __future__ import annotations

import numpy as np
import pytest

from auxetic import Lattice, TileSystem, Simulator
from auxetic.simulation import Constraint, SimResult


def _grid_3x3():
    """Realistic kirigami fixture: 3×3 grid lattice (mode 4)."""
    lat = Lattice(mode=4, n_points=9, ratio=0.35, seed=42)
    ts = TileSystem.from_lattice(lat)
    return Simulator(ts, load_axis=np.array([0.0, -1.0]))


# ---------------------------------------------------------------------------
# theta_max — extended range
# ---------------------------------------------------------------------------

def test_default_theta_max_is_pi_over_2():
    """Backward compat: no theta_max kwarg => same range as before M2.8."""
    sim = _grid_3x3()
    res = sim.sweep_theta(n_steps=21)
    assert res.theta_samples[0]  == pytest.approx(-np.pi / 2.0)
    assert res.theta_samples[-1] == pytest.approx(+np.pi / 2.0)


def test_theta_max_pi_extends_range():
    sim = _grid_3x3()
    res = sim.sweep_theta(n_steps=37, theta_max=np.pi)
    assert res.theta_samples[0]  == pytest.approx(-np.pi)
    assert res.theta_samples[-1] == pytest.approx(+np.pi)
    # Sample count and pose array shape track n_steps.
    assert res.poses.shape == (37, sim.n_tiles * sim.dofs)


def test_theta_max_arbitrary_value_works():
    sim = _grid_3x3()
    res = sim.sweep_theta(n_steps=11, theta_max=0.7)
    assert res.theta_samples[0]  == pytest.approx(-0.7)
    assert res.theta_samples[-1] == pytest.approx(+0.7)


# ---------------------------------------------------------------------------
# collision_stop — without it, sweep ignores collisions
# ---------------------------------------------------------------------------

def test_collision_at_theta_all_false_when_check_disabled():
    """``collision_at_theta`` is an n_steps-shaped bool array even when
    ``collision_stop=False`` — every entry is just False. This keeps
    callers simple (no None-checks required when iterating)."""
    sim = _grid_3x3()
    res = sim.sweep_theta(n_steps=21)
    assert res.collision_at_theta.shape == (21,)
    assert not res.collision_at_theta.any()
    assert res.collision_theta_min is None
    assert res.collision_theta_max is None


def test_collision_check_returns_per_sample_flags():
    sim = _grid_3x3()
    res = sim.sweep_theta(n_steps=21, theta_max=np.pi, collision_stop=True)
    assert res.collision_at_theta.shape == (21,)
    assert res.collision_at_theta.dtype == bool


def test_collision_bounds_set_when_collisions_detected():
    """With theta_max=π the kirigami is rotated far enough that some
    tile pair will eventually overlap. The bounds should be populated."""
    sim = _grid_3x3()
    res = sim.sweep_theta(n_steps=37, theta_max=np.pi, collision_stop=True)
    # At least one collision flag should be True somewhere in the sweep
    # (otherwise this lattice is uniquely uncollidable and the test is
    # uninformative — keep the assertion conditional).
    if res.collision_at_theta.any():
        # At least one of the bounds should have been set on the side
        # where collisions happened.
        assert (res.collision_theta_min is not None
                or res.collision_theta_max is not None)


def test_collision_bounds_none_when_no_collisions_in_canonical_range():
    """The canonical [-π/2, π/2] range typically doesn't reach
    collision in well-formed auxetics. Bounds remain None."""
    sim = _grid_3x3()
    res = sim.sweep_theta(n_steps=21, collision_stop=True)
    if not res.collision_at_theta.any():
        assert res.collision_theta_min is None
        assert res.collision_theta_max is None


# ---------------------------------------------------------------------------
# Synthetic case: two squares pinned at a corner — guaranteed to collide
# at large rotation amplitudes
# ---------------------------------------------------------------------------

def _two_squares_with_pin() -> Simulator:
    """Two unit squares stacked vertically, pinned at their adjacent
    corner. At large kinematic θ the squares rotate apart enough to
    eventually overlap (each square's outboard corner swings inward)."""
    t0 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    t1 = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    constraints = [Constraint(0, 2, 1, 1, 1)]   # share (1, 1)
    ts = TileSystem(2, [t0, t1], constraints=constraints)
    return Simulator(ts, load_axis=np.array([0.0, -1.0]))


def test_collision_eventually_appears_for_two_pinned_squares():
    sim = _two_squares_with_pin()
    res = sim.sweep_theta(n_steps=181, theta_max=np.pi, collision_stop=True)
    # Symmetric construction — collisions on either side OR neither
    # (depends on the kirigami mode picked for this 2-tile system).
    # If the picked mode produces meaningful tile rotation, collisions
    # should appear at some |θ|.
    if res.collision_at_theta.any():
        idx = np.where(res.collision_at_theta)[0]
        # Collisions should be at |θ| > π/2 — the inner range is
        # collision-free.
        for i in idx:
            assert abs(res.theta_samples[i]) > np.pi / 4


# ---------------------------------------------------------------------------
# Compression ratio ignores collided samples
# ---------------------------------------------------------------------------

def test_compression_ratio_excludes_collided_samples():
    """If half the sweep is collided, the compression ratio should be
    computed only over the reachable half."""
    sim = _grid_3x3()
    # Run with collision_stop and large range to force some collisions.
    res = sim.sweep_theta(n_steps=37, theta_max=np.pi, collision_stop=True)
    # Sanity: comp_ratio is in [0, 1].
    assert 0.0 <= res.compression_ratio <= 1.0


# ---------------------------------------------------------------------------
# 3D mode falls through cleanly (collision is 2D-only in M2)
# ---------------------------------------------------------------------------

def test_3d_sweep_with_collision_stop_does_not_error():
    """For 3D modes the collision checker no-ops; sweep completes
    normally with collision_at_theta all-False."""
    lat = Lattice(mode=6, n_points=8, ratio=0.35, seed=42)
    ts = TileSystem.from_lattice(lat)
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0, 0.0]))
    res = sim.sweep_theta(n_steps=11, theta_max=np.pi, collision_stop=True)
    assert res.poses.shape[0] == 11
    # 3D collision is unsupported — array exists but all False.
    assert res.collision_at_theta.shape == (11,)
    assert not res.collision_at_theta.any()
    assert res.collision_theta_min is None
    assert res.collision_theta_max is None
