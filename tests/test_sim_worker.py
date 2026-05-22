"""Task 5: the kinematic solve runs on a worker thread so a long sweep
doesn't freeze the UI. ``_solve_kinematic`` is the pure compute the worker
runs (and the synchronous ``run_simulation`` path runs too). This pins the
acceptance — worker output is numerically identical to driving the
``Simulator`` directly — by comparing ``_solve_kinematic`` against the same
``Simulator`` calls made inline.

Pure: builds ``Lattice`` / ``TileSystem`` / ``Simulator`` directly, no Qt
widget and no thread, so it's immune to the GUI teardown race and safe to
run alone. (The QThread machinery itself mirrors the proven
``predictor_panel`` worker and isn't asserted headlessly.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxetic import Lattice, TileSystem, Simulator
from auxetic_studio.simulation_panel import _solve_kinematic


def _inputs(lat):
    ts = TileSystem.from_lattice(lat)
    load_axis = (np.array([0.0, -1.0]) if ts.dimension == 2
                 else np.array([0.0, -1.0, 0.0]))
    return ts, load_axis


def test_solve_kinematic_matches_direct_simulator_mode1():
    lat = Lattice(mode=1, n_points=6, ratio=0.35, seed=3)
    ts, load_axis = _inputs(lat)

    # Worker compute (what runs off the UI thread).
    simulator, sim_result, poissons, locked, info = _solve_kinematic(
        ts, load_axis, False, 0.0)

    # Same calls made directly on a fresh Simulator.
    sim2 = Simulator(ts, load_axis=load_axis)
    sr2 = sim2.sweep_theta(n_steps=181, theta_max=np.pi, collision_stop=True)
    p2 = sim2.poissons_ratio()
    l2, _ = sim2.is_locked()

    np.testing.assert_allclose(sim_result.theta_samples, sr2.theta_samples)
    np.testing.assert_allclose(sim_result.bbox_extents, sr2.bbox_extents)
    np.testing.assert_allclose(poissons, p2, equal_nan=True)
    assert locked == l2


def test_solve_kinematic_matches_direct_simulator_mode11():
    eqtri = np.array([[0.0, 0.0], [1.0, np.sqrt(3.0)], [2.0, 0.0]])
    lat = Lattice(mode=11, n_points=3, ratio=0.35)
    lat.regenerate_from_points(eqtri)
    ts, load_axis = _inputs(lat)
    jam = float(lat.bipartite_jamming_angle())

    simulator, sim_result, poissons, locked, info = _solve_kinematic(
        ts, load_axis, True, jam)

    sim2 = Simulator(ts, load_axis=load_axis)
    sr2 = sim2.sweep_mechanism(max_actuation=jam)

    np.testing.assert_allclose(sim_result.actuation_angles,
                               sr2.actuation_angles)
    np.testing.assert_allclose(sim_result.bbox_extents, sr2.bbox_extents)
    # Mode 11 reuses the sweep's own locking verdict.
    assert locked == sr2.locked


def test_solve_kinematic_returns_usable_simulator():
    """The returned Simulator is the one the sweep ran on, so the panel can
    keep using it (pose lookups, anchor frames) after the worker finishes."""
    lat = Lattice(mode=1, n_points=6, ratio=0.35, seed=3)
    ts, load_axis = _inputs(lat)
    simulator, sim_result, poissons, locked, info = _solve_kinematic(
        ts, load_axis, False, 0.0)
    assert isinstance(simulator, Simulator)
    assert simulator.tile_system is ts
