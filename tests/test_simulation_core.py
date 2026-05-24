"""Stage 6a tests: TileSystem + Simulator core.

Twelve tests, mapped to the prompt:

1.  test_tilesystem_from_lattice_2d
2.  test_tilesystem_from_lattice_3d
3.  test_tilesystem_files_roundtrip
4.  test_jacobian_dimension_2d
5.  test_jacobian_dimension_3d
6.  test_jacobian_residual_consistency
7.  test_null_space_includes_rigid_modes_2d
8.  test_null_space_includes_rigid_modes_3d
9.  test_kirigami_mode_excludes_rigid
10. test_projection_recovers_rest_pose
11. test_projection_corrects_off_manifold
12. test_load_axis_transforms_with_rigid_rotation

These tests do **not** import any GUI module. The Stage 4 conftest
hook still runs against them but it's a no-op for non-Qt code.
"""

import os
import sys

import numpy as np
import pytest
import scipy.linalg
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from auxetic import Lattice, TileSystem, Simulator, Constraint


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _lattice_2d():
    """A small mode-1 lattice with enough points that the kirigami
    mode is non-trivial. n=8 gives a richer constraint graph than the
    n=5 used elsewhere — useful for the kinematic-mode tests."""
    return Lattice(mode=1, n_points=8, ratio=0.35, seed=42)


def _lattice_3d():
    return Lattice(mode=6, n_points=8, ratio=0.35, seed=42)


def _sim_2d():
    L = _lattice_2d()
    ts = TileSystem.from_lattice(L)
    return Simulator(ts, load_axis=np.array([0.0, 1.0]))


def _sim_3d():
    L = _lattice_3d()
    ts = TileSystem.from_lattice(L)
    return Simulator(ts, load_axis=np.array([0.0, 1.0, 0.0]))


# ---------------------------------------------------------------------------
# 1. TileSystem.from_lattice in 2D
# ---------------------------------------------------------------------------

def test_tilesystem_from_lattice_2d():
    ts = TileSystem.from_lattice(_lattice_2d())
    assert ts.dimension == 2
    assert ts.n_tiles > 0
    for t in ts.tiles:
        assert t.ndim == 2 and t.shape[1] == 2
    assert ts.n_constraints > 0
    for c in ts.constraints:
        assert isinstance(c, Constraint)
        assert 0 <= c.tile_a < ts.n_tiles
        assert 0 <= c.tile_b < ts.n_tiles


# ---------------------------------------------------------------------------
# 2. TileSystem.from_lattice in 3D
# ---------------------------------------------------------------------------

def test_tilesystem_from_lattice_3d():
    ts = TileSystem.from_lattice(_lattice_3d())
    assert ts.dimension == 3
    assert ts.n_tiles > 0
    for t in ts.tiles:
        assert t.ndim == 2 and t.shape[1] == 3
    assert ts.n_constraints > 0


# ---------------------------------------------------------------------------
# 3. files round-trip
# ---------------------------------------------------------------------------

def test_tilesystem_files_roundtrip(tmp_path):
    """Use mode 6: kirigami exporter writes 3D vertices, so loading
    back with dimension=3 reproduces the in-memory representation."""
    L = _lattice_3d()
    verts_path = str(tmp_path / "verts.txt")
    cons_path  = str(tmp_path / "cons.txt")
    L.to_kirigami(verts_path, cons_path, verbose=False)

    ts_files   = TileSystem.from_files(verts_path, cons_path, dimension=3)
    ts_lattice = TileSystem.from_lattice(L)

    assert ts_files.dimension == ts_lattice.dimension
    assert ts_files.n_tiles == ts_lattice.n_tiles
    assert ts_files.n_constraints == ts_lattice.n_constraints

    for tf, tl in zip(ts_files.tiles, ts_lattice.tiles):
        assert tf.shape == tl.shape
        # Tolerance bounded by the file format's %.9g precision.
        np.testing.assert_allclose(tf, tl, atol=1e-9)

    for cf, cl in zip(ts_files.constraints, ts_lattice.constraints):
        assert (cf.tile_a, cf.vert_a, cf.tile_b, cf.vert_b, cf.ctype) == \
               (cl.tile_a, cl.vert_a, cl.tile_b, cl.vert_b, cl.ctype)


# ---------------------------------------------------------------------------
# 4 + 5. Jacobian dimensions
# ---------------------------------------------------------------------------

def test_jacobian_dimension_2d():
    sim = _sim_2d()
    J = sim.assemble_jacobian(sim.rest_pose())
    assert J.shape == (sim.n_constraints * 2, sim.n_tiles * 3)


def test_jacobian_dimension_3d():
    sim = _sim_3d()
    J = sim.assemble_jacobian(sim.rest_pose())
    assert J.shape == (sim.n_constraints * 3, sim.n_tiles * 6)


# ---------------------------------------------------------------------------
# 6. Jacobian / residual consistency under small perturbation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["2d", "3d"])
def test_jacobian_residual_consistency(which):
    """Linearisation check: J @ δ should equal r(p+δ) - r(p) up to
    O(δ²). We perturb at the rest pose (where the residual is zero)
    and at a small non-zero pose, since both regimes matter for the
    projection step."""
    sim = _sim_2d() if which == "2d" else _sim_3d()
    rng = np.random.default_rng(0)
    delta_scale = 1e-6

    for base_scale in (0.0, 1e-3):
        base_pose = base_scale * rng.standard_normal(sim.n_tiles * sim.dofs)
        delta     = delta_scale * rng.standard_normal(base_pose.shape)

        r0 = sim.constraint_residual(base_pose)
        r1 = sim.constraint_residual(base_pose + delta)
        J  = sim.assemble_jacobian(base_pose)

        diff = (r1 - r0) - J @ delta
        # ‖diff‖ should scale like δ² (here ≤ 1e-9). We allow some
        # slack so a slightly larger constant doesn't make this flaky.
        assert np.linalg.norm(diff) < 1e-8, (
            f"linearisation error too large at base_scale={base_scale}: "
            f"||diff||={np.linalg.norm(diff)}"
        )


# ---------------------------------------------------------------------------
# 7 + 8. Null space contains all rigid-body modes
# ---------------------------------------------------------------------------

def _null_space(sim) -> np.ndarray:
    return scipy.linalg.null_space(
        sim.assemble_jacobian(sim.rest_pose()), rcond=1e-8,
    )


def test_null_space_includes_rigid_modes_2d():
    sim = _sim_2d()
    null = _null_space(sim)
    # 2D rigid-body modes: tx + ty + rotation ⇒ at least 3.
    assert null.shape[1] >= 3, (
        f"expected null-space dimension ≥ 3 (rigid + kirigami), "
        f"got {null.shape[1]}"
    )

    rigid = sim._build_rigid_basis()  # (N_pose, 3)
    J = sim.assemble_jacobian(sim.rest_pose())
    for k in range(rigid.shape[1]):
        residual = J @ rigid[:, k]
        assert np.linalg.norm(residual) < 1e-9, (
            f"rigid mode {k} not in null space: ||J @ rigid_k||="
            f"{np.linalg.norm(residual)}"
        )


def test_null_space_includes_rigid_modes_3d():
    sim = _sim_3d()
    null = _null_space(sim)
    # 3D rigid-body modes: 3 translations + 3 rotations ⇒ at least 6.
    assert null.shape[1] >= 6

    rigid = sim._build_rigid_basis()  # (N_pose, 6)
    J = sim.assemble_jacobian(sim.rest_pose())
    for k in range(rigid.shape[1]):
        residual = J @ rigid[:, k]
        assert np.linalg.norm(residual) < 1e-9


# ---------------------------------------------------------------------------
# 9. Kirigami mode excludes rigid-body modes
# ---------------------------------------------------------------------------

def test_kirigami_mode_excludes_rigid():
    sim = _sim_2d()
    mode = sim.identify_kirigami_mode()
    if mode is None:
        pytest.skip("This system has no kirigami mode")

    rigid = sim._build_rigid_basis()
    rigid_orth = scipy.linalg.orth(rigid)
    # Project mode onto rigid subspace; should be ~zero.
    overlap = rigid_orth.T @ mode
    assert np.linalg.norm(overlap) < 1e-9, (
        f"kirigami mode has non-trivial overlap with rigid subspace: "
        f"||overlap||={np.linalg.norm(overlap)}"
    )


# ---------------------------------------------------------------------------
# 10. Projection from a near-manifold point recovers it
# ---------------------------------------------------------------------------

def test_projection_recovers_rest_pose():
    """Perturb along the kirigami mode and verify the projection lands
    on the manifold with the kirigami-mode amplitude preserved.

    Note: for under-constrained systems the Jacobian has more null
    directions than just the kirigami mode. ``trf``'s minimum-norm
    Gauss-Newton step is free to drift along ANY null direction
    while still driving the residual to zero, so we don't expect the
    projected pose to literally equal the perturbed pose. What
    *should* be invariant is the projection of the displacement onto
    the kirigami mode itself — ``J·m = 0`` gives the projection no
    incentive to remove the kirigami component."""
    sim = _sim_2d()
    mode = sim.identify_kirigami_mode()
    if mode is None:
        pytest.skip("This system has no kirigami mode")

    eps = 1e-3
    rest = sim.rest_pose()
    perturbed = rest + eps * mode
    projected = sim.project_to_manifold(perturbed)

    r_after = np.linalg.norm(sim.constraint_residual(projected))
    assert r_after < 1e-9, f"residual after projection: {r_after}"

    # Mode amplitude before vs. after projection (mode is unit-norm
    # so this is a scalar projection). The kirigami mode is exactly
    # in the null space at REST, but the manifold curves away from
    # the linear approximation by O(ε²) per step, so we expect the
    # projection to correct the amplitude by O(ε²) as well.
    amp_before = float(np.dot(perturbed - rest, mode))
    amp_after  = float(np.dot(projected - rest, mode))
    assert abs(amp_after - amp_before) < 100 * eps**2, (
        f"kirigami-mode amplitude not preserved: "
        f"before={amp_before}, after={amp_after}, "
        f"diff={abs(amp_after - amp_before)}, eps²={eps**2}"
    )


# ---------------------------------------------------------------------------
# 11. Projection corrects an off-manifold perturbation
# ---------------------------------------------------------------------------

def test_projection_corrects_off_manifold():
    """Random perturbation almost certainly has a non-zero
    component perpendicular to the manifold. Projection should drive
    the residual to zero and reduce it from its starting value."""
    sim = _sim_2d()
    rng = np.random.default_rng(1)

    eps = 1e-3
    perturbed = sim.rest_pose() + eps * rng.standard_normal(
        sim.n_tiles * sim.dofs)

    r_before = np.linalg.norm(sim.constraint_residual(perturbed))
    projected = sim.project_to_manifold(perturbed)
    r_after  = np.linalg.norm(sim.constraint_residual(projected))

    assert r_after < 1e-9, f"residual after projection: {r_after}"
    assert r_after < r_before, (
        f"projection didn't reduce residual: before={r_before}, "
        f"after={r_after}"
    )


# ---------------------------------------------------------------------------
# 12. World transform applied to tiles when building from a rotated lattice
# ---------------------------------------------------------------------------

def test_load_axis_transforms_with_rigid_rotation():
    """SPEC §6.1 + §8: world_transform applies to tile vertices on
    the way into TileSystem so a rotated lattice's kirigami mode
    aligns differently with the load axis."""
    L = _lattice_3d()
    ts_canonical = TileSystem.from_lattice(L)

    L.rigid_rotation = Rotation.from_euler("z", 90, degrees=True)
    ts_rotated = TileSystem.from_lattice(L)

    assert ts_rotated.dimension == ts_canonical.dimension
    assert ts_rotated.n_tiles == ts_canonical.n_tiles

    # Verify each tile reflects the rotation. world_transform pivots
    # around centroid (0.5, 0.5, 0.5); we apply it manually here and
    # compare.
    M = L.world_transform()
    for tc, tr in zip(ts_canonical.tiles, ts_rotated.tiles):
        n = tc.shape[0]
        homo = np.hstack([tc, np.ones((n, 1))])
        expected = (M @ homo.T).T[:, :3]
        np.testing.assert_allclose(tr, expected, atol=1e-9)
