"""Kinematic-simulation tests for mode 12 (3D tetrahedral auxetic).

These exercise the path the Simulation panel's "Run Simulation" button
drives — ``TileSystem.from_lattice`` → ``Simulator`` — but at the core
level (no Qt), so they're fast and deterministic.

The mode-12 mechanism only works because the canonical edge/face points
fuse across adjacent tetrahedra into revolute hinges (see
``auxetic.tetrahedral``). Without that fusion the rigid pieces touch
only at single points (3D ball joints) and the constraint Jacobian has a
~96-dimensional floppy null space — no meaningful mode. With it, the
null space collapses to the 6 rigid-body modes plus a handful of genuine
mechanism DOF, and the solver identifies a kirigami mode.
"""

from __future__ import annotations

import math

import numpy as np

from auxetic.lattice import Lattice
from auxetic.simulation import Simulator, TileSystem


def _mode12_tilesystem(n: int = 8, C: float = 0.5, seed: int = 42):
    lat = Lattice(mode=12, n_points=n, ratio=0.35, seed=seed, C=C)
    return TileSystem.from_lattice(lat), lat


# ---------------------------------------------------------------------------
# TileSystem assembly
# ---------------------------------------------------------------------------

def test_mode12_collect_kirigami_no_longer_raises():
    """Regression on the earlier guard: mode 12 used to raise here."""
    lat = Lattice(mode=12, n_points=8, ratio=0.35, seed=1, C=0.5)
    tiles, source, constraints = lat.collect_kirigami()
    assert len(tiles) > 0
    assert len(constraints) > 0


def test_mode12_tilesystem_is_3d_with_five_tiles_per_tetra():
    ts, lat = _mode12_tilesystem()
    n_tetra = len(np.asarray(lat.tri.simplices))
    assert ts.dimension == 3
    # 1 internal tetra + 4 corner polyhedra per tetrahedron.
    assert ts.n_tiles == 5 * n_tetra
    assert ts.n_constraints > 0


def test_mode12_internal_and_corner_tile_vertex_counts():
    ts, _ = _mode12_tilesystem()
    internal = [t for t, s in zip(ts.tiles, ts.tile_source)
                if s.get("type") == "tetrahedron"]
    corner = [t for t, s in zip(ts.tiles, ts.tile_source)
              if s.get("type") == "hub_polyhedron"]
    assert internal and corner
    assert all(t.shape == (4, 3) for t in internal)   # internal tetra
    assert all(t.shape == (8, 3) for t in corner)      # corner polyhedron


# ---------------------------------------------------------------------------
# The mechanism is coherent (hinge fusion), not a floppy ball-joint mess
# ---------------------------------------------------------------------------

def _nullspace_dim(sim: Simulator) -> int:
    J = sim.assemble_jacobian(sim.rest_pose())
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-9 * max(1.0, sv.max())))
    return J.shape[1] - rank


def test_mode12_mechanism_is_low_dof_not_floppy():
    ts, _ = _mode12_tilesystem()
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0, 0.0]))
    nulldim = _nullspace_dim(sim)
    # 6 rigid-body modes + a few genuine mechanism DOF — emphatically NOT
    # the ~96 of the un-fused point-joint construction.
    assert 6 <= nulldim < 30


def test_mode12_jacobian_is_3d_shaped():
    ts, _ = _mode12_tilesystem()
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0, 0.0]))
    J = sim.assemble_jacobian(sim.rest_pose())
    # 3 scalar equations per constraint; 6 DOF per tile in 3D.
    assert J.shape == (ts.n_constraints * 3, ts.n_tiles * 6)


def test_mode12_identifies_a_kirigami_mode():
    ts, _ = _mode12_tilesystem()
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0, 0.0]))
    mode = sim.identify_kirigami_mode()
    assert mode is not None
    assert mode.shape == (ts.n_tiles * 6,)
    assert np.linalg.norm(mode) > 0.0


def test_mode12_poissons_ratio_is_finite():
    # Small lattice keeps the projection solves quick.
    ts, _ = _mode12_tilesystem(n=6)
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0, 0.0]))
    pr = sim.poissons_ratio()
    vals = pr if isinstance(pr, (tuple, list, np.ndarray)) else [pr]
    finite = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    # The mechanism must yield at least one finite transverse-response
    # ratio (auxetic or not — that's geometry-dependent).
    assert len(finite) >= 1
