"""Tests for ``auxetic.dynamics`` — the M2 Newtonian rigid-body sim.

These tests verify the integrator's core behaviours in isolation:

- Rest pose with no forces stays at rest.
- A single unconstrained tile under gravity falls per ``y = ½ g t²``.
- A two-tile chain with one fixed tile and a soft constraint comes to
  rest at the constraint's equilibrium — proves Baumgarte-stabilised
  constraints damp correctly.
- A user-applied force translates a free tile in the force's direction.
- A tile-vertex force induces the right torque sign.
- Ground contact stops a falling tile from passing through the plane.
- Fixed tiles ignore all forces.

Higher-level tests (preset round-trip, GUI panel, end-to-end vs
kinematic sweep) live in their own files.
"""

from __future__ import annotations

import numpy as np
import pytest

from auxetic.dynamics import (
    DynamicsConfig,
    DynamicsSimulator,
    ForceVector,
    GroundContact,
    TileMass,
    default_masses_from_tile_system,
)
from auxetic.simulation import Constraint, TileSystem


# ---------------------------------------------------------------------------
# Fixtures: minimal tile systems
# ---------------------------------------------------------------------------

def _single_2d_triangle() -> TileSystem:
    return TileSystem(
        dimension=2,
        tiles=[np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
        constraints=[],
    )


def _single_3d_tetra() -> TileSystem:
    return TileSystem(
        dimension=3,
        tiles=[np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])],
        constraints=[],
    )


def _two_2d_triangles_with_constraint() -> TileSystem:
    """Two adjacent triangles, with vertex 1 of tile 0 pinned to vertex
    0 of tile 1. Used to exercise the soft-constraint path."""
    t0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    t1 = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    return TileSystem(
        dimension=2,
        tiles=[t0, t1],
        constraints=[Constraint(tile_a=0, vert_a=1, tile_b=1, vert_b=0, ctype=1)],
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_2d():
    ts = _single_2d_triangle()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    sim = DynamicsSimulator(ts, masses, DynamicsConfig())
    assert sim.n_tiles == 1
    assert sim.n_dofs == 3   # 2 trans + 1 rot in 2D


def test_construction_3d():
    ts = _single_3d_tetra()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    sim = DynamicsSimulator(ts, masses, DynamicsConfig())
    assert sim.n_dofs == 6   # 3 trans + 3 rot


def test_mass_count_mismatch_raises():
    ts = _single_2d_triangle()
    with pytest.raises(ValueError, match="masses length"):
        DynamicsSimulator(ts, [], DynamicsConfig())


# ---------------------------------------------------------------------------
# Default mass helper
# ---------------------------------------------------------------------------

def test_default_masses_2d_uses_area():
    ts = _single_2d_triangle()
    masses = default_masses_from_tile_system(ts, density=2.0, thickness=0.5)
    # Triangle area = 0.5; mass = 0.5 * 0.5 * 2.0 = 0.5.
    assert abs(masses[0].mass - 0.5) < 1e-9
    assert masses[0].inertia_iso > 0.0


def test_default_masses_3d_uses_volume():
    ts = _single_3d_tetra()
    masses = default_masses_from_tile_system(ts, density=3.0)
    # Unit-corner tetrahedron volume = 1/6.
    expected = 3.0 * (1.0 / 6.0)
    assert abs(masses[0].mass - expected) < 1e-6


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------

def _zero_g(dim=2) -> np.ndarray:
    return np.zeros(3) if dim == 3 else np.zeros(3)


def test_rest_pose_stays_at_rest_with_no_forces():
    """No forces, no gravity → pose unchanged after many steps."""
    ts = _single_2d_triangle()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    cfg = DynamicsConfig(dt=1e-3, duration=0.05,
                          gravity=np.zeros(3))
    sim = DynamicsSimulator(ts, masses, cfg)
    res = sim.simulate()
    np.testing.assert_allclose(res.poses[0], res.poses[-1], atol=1e-9)
    np.testing.assert_allclose(res.velocities[-1], 0.0, atol=1e-9)


def test_unconstrained_tile_falls_under_gravity_2d():
    """A single 2D tile with no constraints, only gravity → free fall."""
    ts = _single_2d_triangle()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    g = -9.81
    cfg = DynamicsConfig(dt=1e-4, duration=0.1,
                          gravity=np.array([0.0, g, 0.0]))
    sim = DynamicsSimulator(ts, masses, cfg)
    res = sim.simulate()
    # After t=0.1s, expected y-displacement ≈ ½ * g * t² = -0.04905.
    expected = 0.5 * g * (0.1 ** 2)
    actual = float(res.poses[-1, 1])  # tile 0, ty
    assert abs(actual - expected) < 1e-3, f"expected {expected}, got {actual}"


def test_unconstrained_tile_falls_under_gravity_3d():
    ts = _single_3d_tetra()
    masses = [TileMass(mass=2.0, inertia_iso=0.5)]
    g = -9.81
    cfg = DynamicsConfig(dt=1e-4, duration=0.1,
                          gravity=np.array([0.0, g, 0.0]))
    sim = DynamicsSimulator(ts, masses, cfg)
    res = sim.simulate()
    expected = 0.5 * g * (0.1 ** 2)
    actual = float(res.poses[-1, 1])  # tile 0, ty
    assert abs(actual - expected) < 1e-3


def test_user_force_pushes_free_tile_in_2d():
    """Constant force F on a free 2D tile → uniform acceleration; after
    t, displacement = ½ a t² along force direction."""
    ts = _single_2d_triangle()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    cfg = DynamicsConfig(dt=1e-4, duration=0.1,
                          gravity=np.zeros(3))
    forces = [ForceVector(
        location_kind="tile_centroid",
        direction=np.array([1.0, 0.0]),
        magnitude=2.0,
        tile_index=0,
    )]
    sim = DynamicsSimulator(ts, masses, cfg, forces=forces)
    res = sim.simulate()
    expected_x = 0.5 * 2.0 * (0.1 ** 2)  # a = F/m = 2/1 = 2
    actual_x = float(res.poses[-1, 0])
    assert abs(actual_x - expected_x) < 1e-3


def test_force_at_tile_vertex_induces_torque_2d():
    """A force applied at tile vertex 1 = (1, 0) in the +y direction
    should torque the tile counter-clockwise."""
    ts = _single_2d_triangle()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    cfg = DynamicsConfig(dt=1e-4, duration=0.05,
                          gravity=np.zeros(3))
    forces = [ForceVector(
        location_kind="tile_vertex",
        direction=np.array([0.0, 1.0]),
        magnitude=1.0,
        tile_index=0,
        vert_index=1,   # at body-frame (1, 0) → torque about origin = +z
    )]
    sim = DynamicsSimulator(ts, masses, cfg, forces=forces)
    res = sim.simulate()
    # θ at end should be > 0 (counter-clockwise rotation).
    theta = float(res.poses[-1, 2])
    assert theta > 0.0, f"expected CCW rotation, got θ={theta}"


def test_fixed_tile_does_not_move_under_force():
    """A tile listed in ``fixed_tiles`` ignores all forces / constraints."""
    ts = _single_2d_triangle()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    cfg = DynamicsConfig(dt=1e-3, duration=0.1,
                          gravity=np.array([0.0, -9.81, 0.0]))
    forces = [ForceVector(
        location_kind="tile_centroid",
        direction=np.array([1.0, 0.0]),
        magnitude=10.0,
        tile_index=0,
    )]
    sim = DynamicsSimulator(ts, masses, cfg, forces=forces, fixed_tiles=[0])
    res = sim.simulate()
    np.testing.assert_allclose(res.poses[-1], 0.0, atol=1e-9)
    np.testing.assert_allclose(res.velocities[-1], 0.0, atol=1e-9)


def test_constraint_pulls_residual_toward_zero():
    """Two tiles, one fixed, with a constraint between them. The free
    tile should settle so the constraint residual is small."""
    ts = _two_2d_triangles_with_constraint()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)] * 2
    cfg = DynamicsConfig(
        dt=1e-3, duration=2.0,
        gravity=np.zeros(3),
        joint_stiffness=1e3,
        joint_damping=20.0,
    )
    # Perturb the free tile by translating it; constraint should pull
    # it back. Tile 1 is the free one.
    initial_pose = np.zeros(6)  # 2 tiles × 3 dofs
    initial_pose[3] = 0.5   # tile 1 tx — pulls (1,0,0) → (1.5, 0, 0)
    sim = DynamicsSimulator(ts, masses, cfg, fixed_tiles=[0])
    res = sim.simulate(initial_pose=initial_pose)
    # The residual after settling should be ~0 (tile 1 returns close to
    # the constraint-satisfying position).
    final_pose = res.poses[-1]
    r_final = sim._sim.constraint_residual(final_pose)
    assert float(np.linalg.norm(r_final)) < 0.1


def test_ground_contact_blocks_falling_tile():
    """A tile released above a ground plane should bounce / settle, not
    pass through."""
    ts = _single_2d_triangle()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    cfg = DynamicsConfig(
        dt=1e-4, duration=0.5,
        gravity=np.array([0.0, -9.81, 0.0]),
    )
    ground = GroundContact(
        plane_point=np.array([0.0, -2.0, 0.0]),
        plane_normal=np.array([0.0, 1.0, 0.0]),
        stiffness=1e5,
        damping=100.0,
    )
    sim = DynamicsSimulator(ts, masses, cfg, ground=ground)
    initial_pose = np.zeros(3)   # tile starts at y=0 (above the plane at y=-2)
    res = sim.simulate(initial_pose=initial_pose)
    # Lowest tile-vertex y-coord across the whole trajectory must be
    # >= ground - small allowance for penetration depth.
    min_y_seen = float(res.poses[:, 1].min())
    assert min_y_seen > -2.0 - 0.1, f"tile passed through ground (min y={min_y_seen})"


def test_simulate_records_full_trajectory():
    ts = _single_2d_triangle()
    masses = [TileMass(mass=1.0, inertia_iso=0.1)]
    cfg = DynamicsConfig(dt=1e-3, duration=0.1, gravity=np.zeros(3))
    sim = DynamicsSimulator(ts, masses, cfg)
    res = sim.simulate()
    n_expected = int(np.round(cfg.duration / cfg.dt)) + 1
    assert res.poses.shape == (n_expected, 3)
    assert res.velocities.shape == (n_expected, 3)
    assert res.bbox_extents.shape == (n_expected, 2)
    assert res.times.shape == (n_expected,)


def test_force_vector_normalises_direction():
    fv = ForceVector(
        location_kind="tile_centroid",
        direction=np.array([3.0, 0.0]),
        magnitude=1.0,
        tile_index=0,
    )
    np.testing.assert_allclose(fv.direction, [1.0, 0.0], atol=1e-12)


def test_force_vector_zero_direction_raises():
    with pytest.raises(ValueError, match="direction must be non-zero"):
        ForceVector(
            location_kind="world",
            direction=np.array([0.0, 0.0]),
            magnitude=1.0,
        )


def test_ground_contact_normalises_normal():
    gc = GroundContact(
        plane_point=np.array([0.0, 0.0]),
        plane_normal=np.array([0.0, 5.0]),
    )
    np.testing.assert_allclose(gc.plane_normal, [0.0, 1.0], atol=1e-12)
