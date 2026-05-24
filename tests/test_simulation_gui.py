"""Stage 6c tests: SimulationPanel + View3D.show_pose wiring.

Nine tests, mapped to the prompt:

1. test_simulation_panel_run_button_populates_results
2. test_joint_slider_drives_pose_when_result_present
3. test_slider_at_zero_degrees_corresponds_to_minus_pi_over_2_radians
4. test_simulation_invalidated_on_lattice_change
5. test_play_button_disabled_without_result
6. test_load_axis_uses_world_frame_y
7. test_run_simulation_does_not_crash_on_degenerate_lattice
8. test_outdated_state_clears_pose_in_view_3d
9. test_play_button_animates_via_qtimer
"""

import math
import os
import sys

import numpy as np
import pytest

# Force headless Qt before importing anything that touches QtGui/QtWidgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from scipy.spatial.transform import Rotation

from auxetic import Lattice, Simulator, TileSystem
from auxetic_studio.simulation_panel import (
    slider_to_simulator_theta,
    simulator_theta_to_slider,
)
from auxetic_studio.views import View3D, _apply_tile_pose


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def main_window(qapp):
    from auxetic_studio import MainWindow
    win = MainWindow(headless_3d=True)
    yield win
    win.close()


# ---------------------------------------------------------------------------
# 1. Run Simulation populates the panel state and readout
# ---------------------------------------------------------------------------

def test_simulation_panel_run_button_populates_results(main_window):
    win = main_window
    panel = win.simulation_panel

    assert panel._sim_result is None  # nothing yet
    panel.run_simulation()

    assert panel._sim_result is not None
    assert panel._tile_system is not None
    assert panel._poissons_ratio is not None
    assert panel._locking_info is not None
    text = panel.readout.text()
    assert "Poisson's ratio" in text


# ---------------------------------------------------------------------------
# 2. Slider drives View3D.show_pose when a fresh result is present
# ---------------------------------------------------------------------------

def test_joint_slider_drives_pose_when_result_present(main_window):
    """Slider at 90° corresponds to simulator θ = 0 (rest pose). The
    nearest sample in the trajectory is the centre of the array. We
    verify ``view_3d.show_pose`` was called with a tile_system + a
    pose vector matching that sample.

    Note: the slider already sits at 90° at construction (initial
    ``joint_angle = 0`` rad → slider 90°), so we have to nudge it
    away first to force a ``valueChanged`` signal when we land at 90."""
    win = main_window
    panel = win.simulation_panel
    panel.run_simulation()

    # Nudge slider away from the rest position, then reset the spy.
    panel.slider.setValue(int(round(45.0 * panel.SLIDER_SCALE)))
    win.view_3d.last_show_pose_args = None

    # Slider at 90° → simulator θ ≈ 0 (rest, middle of the trajectory).
    panel.slider.setValue(int(round(90.0 * panel.SLIDER_SCALE)))

    args = win.view_3d.last_show_pose_args
    assert args is not None, "show_pose was not called"
    tile_system, pose = args
    assert tile_system is panel._tile_system

    # The trajectory has 181 samples on [-π/2, +π/2]; θ=0 is at index 90.
    expected_pose = panel._sim_result.poses[90]
    np.testing.assert_allclose(pose, expected_pose, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Convention helpers (the boundary)
# ---------------------------------------------------------------------------

def test_joint_angle_release_leaves_view_in_posed_state(main_window):
    """After committing a joint-angle change (e.g. slider release ->
    JointAngleChangeCommand), the View3D should remain in the POSED
    configuration rather than snapping back to the canonical lattice.

    Regression: ``_refresh_state`` used to call ``_refresh_views``
    AFTER ``simulation_panel.refresh_from_lattice``. The panel's
    ``show_pose`` call ran first and then was immediately overwritten
    by the canonical ``View3D.update_lattice`` render, so users saw
    the figure snap back to rest after letting go of the slider.
    Reordered so the panel's pose-driving is the last step.
    """
    win = main_window
    panel = win.simulation_panel

    # Run a kinematic sim so the panel has a sim_result that
    # ``_drive_pose_from_slider`` can index into.
    panel.run_simulation()
    assert panel._sim_result is not None

    # Commit a non-zero joint angle (simulating slider release).
    target_rad = 0.5
    win._on_joint_angle_change_requested(0.0, target_rad)

    # Lattice's joint_angle is the new value.
    assert win.lattice.joint_angle == pytest.approx(target_rad)

    # View3D's last show_pose call should reflect a non-zero theta —
    # i.e., the pose at the slider's current θ is what's displayed,
    # not the rest pose. ``last_show_pose_args`` is set by
    # ``View3D.show_pose`` (used by tests as an injection point).
    assert panel._view_3d.last_show_pose_args is not None, (
        "View3D should be displaying a posed mesh, not the canonical "
        "mesh — _refresh_state's call order regressed."
    )


def test_run_simulation_uses_extended_range_with_collision_check(main_window):
    """``run_simulation`` should request the M2.8 extended sweep
    (theta_max=π, collision_stop=True) so the plot can shade
    physically-unreachable θ regions."""
    win = main_window
    panel = win.simulation_panel
    panel.run_simulation()
    res = panel._sim_result
    assert res is not None
    # theta_samples should now span (-π, +π) rather than (-π/2, +π/2).
    assert res.theta_samples[0]  == pytest.approx(-math.pi,  abs=1e-9)
    assert res.theta_samples[-1] == pytest.approx(+math.pi, abs=1e-9)
    # collision_at_theta is allocated even if no collisions hit (M2.8).
    assert res.collision_at_theta.shape == (len(res.theta_samples),)


def test_collision_shading_added_to_plot_when_collisions_exist(main_window):
    """If the sweep result records collision bounds, the panel's
    kinematic plot should add ``axvspan`` shading for those regions."""
    win = main_window
    panel = win.simulation_panel
    panel.run_simulation()
    # We can't reliably force a collision on the default lattice — but
    # we can simulate the result having bounds and verify the shading
    # path runs without error.
    panel._sim_result.collision_theta_min = -math.pi * 0.9
    panel._sim_result.collision_theta_max = +math.pi * 0.9
    panel._update_plot()
    # Two spans should be drawn (one per side).
    assert len(panel._collision_spans) == 2


def test_slider_convention_extended_to_full_pi_range():
    """The two convention helpers — the only place slider↔simulator
    mapping is allowed to live — implement the M2.8 extended mapping:
    the slider widget keeps its 0–180° physical range but each
    physical degree corresponds to 2 mathematical degrees, so the
    full slider range maps to ``[-π, +π]`` math radians.

    Pre-M2.8 the mapping was slider ``[0, 180]`` ↔ math
    ``[-π/2, +π/2]`` (SPEC §6.2 bistable cycle). Doubling the math
    range lets the kinematic sweep explore the full ±180° rotation
    until collisions intervene; the GUI plot shades the unreachable
    spans so the user still sees the bistable region clearly.

    Rest is at slider 90° ↔ math 0 in BOTH conventions.
    """
    assert slider_to_simulator_theta(0.0)   == pytest.approx(-math.pi)
    assert slider_to_simulator_theta(90.0)  == pytest.approx(0.0,    abs=1e-12)
    assert slider_to_simulator_theta(180.0) == pytest.approx(+math.pi)

    # Inverse round-trip
    assert simulator_theta_to_slider(-math.pi) == pytest.approx(0.0,   abs=1e-9)
    assert simulator_theta_to_slider(0.0)      == pytest.approx(90.0,  abs=1e-9)
    assert simulator_theta_to_slider(+math.pi) == pytest.approx(180.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Lattice modification invalidates the simulation
# ---------------------------------------------------------------------------

def test_simulation_invalidated_on_lattice_change(main_window):
    win = main_window
    panel = win.simulation_panel
    panel.run_simulation()
    assert panel._is_outdated is False
    assert panel._sim_result is not None

    # Push a parameter change command (mode 1 → mode 6) — this is a
    # structural change and must invalidate.
    inspector = win.inspector
    inspector.select_mode(6)

    assert panel._is_outdated is True
    assert "outdated" in panel.readout.text().lower()


# ---------------------------------------------------------------------------
# 5. Play button is gated on (fresh result, not outdated)
# ---------------------------------------------------------------------------

def test_play_button_disabled_without_result(main_window):
    win = main_window
    panel = win.simulation_panel

    # Fresh panel — no result yet.
    assert panel.play_button.isEnabled() is False

    # After running — enabled.
    panel.run_simulation()
    assert panel.play_button.isEnabled() is True

    # After invalidation — disabled again.
    panel.mark_outdated()
    assert panel.play_button.isEnabled() is False


# ---------------------------------------------------------------------------
# 6. Load axis fixed in world frame (-Y)
# ---------------------------------------------------------------------------

def test_load_axis_uses_world_frame_y(main_window):
    """SPEC §6.1 / Stage 6c contract: the load axis is fixed at
    world-frame -Y regardless of how the lattice has been rotated.
    The tile vertices already reflect the rotation (Stage 6a's
    from_lattice applies world_transform), so a fixed load axis is
    what makes a rotated lattice line up differently against it."""
    win = main_window

    # 2D mode — load axis should be 2-vector [0, -1].
    win.lattice = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    win.lattice.rigid_rotation = Rotation.from_euler("z", 45, degrees=True)
    win.simulation_panel.set_lattice(win.lattice)
    win.simulation_panel.run_simulation()
    sim = win.simulation_panel._simulator
    assert sim.dimension == 2
    np.testing.assert_allclose(sim.load_axis, np.array([0.0, -1.0]))

    # 3D mode — load axis should be 3-vector [0, -1, 0].
    win.lattice = Lattice(mode=6, n_points=8, ratio=0.35, seed=42)
    win.lattice.rigid_rotation = Rotation.from_euler("z", 90, degrees=True)
    win.simulation_panel.set_lattice(win.lattice)
    win.simulation_panel.run_simulation()
    sim = win.simulation_panel._simulator
    assert sim.dimension == 3
    np.testing.assert_allclose(sim.load_axis, np.array([0.0, -1.0, 0.0]))


# ---------------------------------------------------------------------------
# 7. Simulator failures surface in the readout, never crash the GUI
# ---------------------------------------------------------------------------

def test_run_simulation_does_not_crash_on_degenerate_lattice(main_window):
    """Construct a lattice that's likely to raise during simulation
    setup (mode 1 with extremely few points), patch the simulator to
    force a failure, click Run Simulation, verify the readout shows
    the error and the panel/window are still responsive."""
    win = main_window
    panel = win.simulation_panel

    # Force a guaranteed failure — monkey-patch TileSystem.from_lattice
    # to raise. The panel's try/except must catch it.
    import auxetic_studio.simulation_panel as sp_module
    original = sp_module.TileSystem.from_lattice

    class _BoomError(RuntimeError):
        pass

    def _boom(_lattice):
        raise _BoomError("synthetic test failure")

    sp_module.TileSystem.from_lattice = staticmethod(_boom)
    try:
        panel.run_simulation()  # must NOT raise
    finally:
        sp_module.TileSystem.from_lattice = original

    # Result was cleared; error surfaced in readout; window still alive.
    assert panel._sim_result is None
    assert panel._last_error is not None
    assert "synthetic test failure" in panel._last_error
    assert "Simulation failed" in panel.readout.text()
    # Sanity: the panel's slider still responds.
    panel.slider.setValue(int(round(90.0 * panel.SLIDER_SCALE)))


# ---------------------------------------------------------------------------
# 8. Invalidating clears the View3D pose
# ---------------------------------------------------------------------------

def test_outdated_state_clears_pose_in_view_3d(main_window):
    win = main_window
    panel = win.simulation_panel
    panel.run_simulation()

    # Drive the slider to set a pose in View3D. Nudge to 45° (away
    # from the construction-time 90°) to force a valueChanged event.
    panel.slider.setValue(int(round(45.0 * panel.SLIDER_SCALE)))
    assert win.view_3d.last_show_pose_args is not None
    assert win.view_3d._pose_view_active is True

    # Invalidating must call clear_pose() which drops show_pose state.
    panel.mark_outdated()
    assert win.view_3d._pose_view_active is False
    assert win.view_3d.last_show_pose_args is None


# ---------------------------------------------------------------------------
# 9. Play button drives the slider via QTimer
# ---------------------------------------------------------------------------

def test_play_button_animates_via_qtimer(main_window, qapp):
    """The Play toggle starts a QTimer that advances the slider on
    each tick. We don't time-pump via ``QTest.qWait`` because that
    spins Qt's full event loop — which on Windows + Python 3.14 +
    PyQt6 + offscreen Qt + accumulated state from prior tests in
    the same process triggers an access violation in VTK's offscreen
    GL path. Instead, we drive the timer slot directly to verify
    the *structural claim* (timer is active, ticks advance the
    slider, toggling off stops the timer) without needing a real
    event loop run."""
    win = main_window
    panel = win.simulation_panel
    panel.run_simulation()

    start_value = panel.slider.value()
    panel.play_button.setChecked(True)
    assert panel._play_timer.isActive(), "play timer not started by toggle"

    # Drive the timer slot manually, simulating several ticks.
    for _ in range(10):
        panel._on_play_tick()

    advanced_value = panel.slider.value()
    assert advanced_value != start_value, (
        "slider did not advance during play "
        f"(start={start_value}, after={advanced_value})"
    )

    # Toggling off stops the timer.
    panel.play_button.setChecked(False)
    assert not panel._play_timer.isActive(), "play timer not stopped by toggle"


# ---------------------------------------------------------------------------
# Stage 6c.5 — pose render emits the full deformed lattice mesh
# (hubs, struts, joint spheres), not just the tile triangles.
# ---------------------------------------------------------------------------

def test_show_pose_includes_hub_and_strut_geometry():
    """The pose-rendered mesh must include hub solids, strut tubes, and
    joint spheres — the same primitives ``Lattice.to_stl`` produces.
    The Stage 6c implementation rendered only ~4 triangles for a small
    mode-1 lattice where the static path produces ~800; this test
    catches that regression by demanding order-of-magnitude equality
    (within 20%)."""
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    static_triangles = L.build_export_triangles(verbose=False)

    ts = TileSystem.from_lattice(L)
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0]))
    rest_pose = sim.rest_pose()
    pose_triangles = View3D._build_pose_mesh_triangles(ts, rest_pose)

    n_static = len(static_triangles)
    n_pose   = len(pose_triangles)

    assert n_static > 100, (
        f"static path produced suspiciously few triangles ({n_static}); "
        "test premise broken"
    )
    assert n_pose >= int(0.8 * n_static), (
        f"pose render produced too few triangles: {n_pose} vs static "
        f"{n_static}. Bug regression watermark was ~4 triangles."
    )
    assert n_pose <= int(1.2 * n_static), (
        f"pose render produced too many triangles: {n_pose} vs static "
        f"{n_static}"
    )


def test_show_pose_at_rest_matches_canonical_lattice_geometry():
    """For a mode-1 lattice at rest pose with identity rigid_rotation,
    the pose render and the static ``Lattice.to_stl`` path go through
    the same primitives (``extrude_polygon_solid``, ``tube_mesh``,
    ``sphere_mesh``) on the same canonical geometry, so the resulting
    triangle sets should match position-for-position. We compare via
    sorted centroids — invariant under triangle-emission order
    differences between the two pipelines."""
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    static_triangles = L.build_export_triangles(verbose=False)

    ts = TileSystem.from_lattice(L)
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0]))
    rest_pose = sim.rest_pose()
    pose_triangles = View3D._build_pose_mesh_triangles(ts, rest_pose)

    def centroids_sorted(triangles):
        cs = np.array([np.mean(np.asarray(t, dtype=float), axis=0)
                       for t in triangles])
        # Lex-sort by (x, y, z) for stable comparison across paths.
        order = np.lexsort((cs[:, 2], cs[:, 1], cs[:, 0]))
        return cs[order]

    static_c = centroids_sorted(static_triangles)
    pose_c   = centroids_sorted(pose_triangles)

    assert static_c.shape == pose_c.shape, (
        f"triangle counts differ: static={static_c.shape[0]}, "
        f"pose={pose_c.shape[0]}"
    )
    np.testing.assert_allclose(pose_c, static_c, atol=1e-6)


def test_show_pose_includes_central_hub_for_mode_6():
    """For a mode-6 lattice with at least one central hub, verify the
    pose render emits ``hub_polyhedron`` geometry. n_points=27 → 3×3×3
    grid; the centre point at (0.5, 0.5, 0.5) sits in all 8 octants so
    ``is_central_hub`` returns True for it."""
    from auxetic.geometry import collect_export_geometry_from_posed_tiles

    L = Lattice(mode=6, n_points=27, ratio=0.35, seed=42)
    ts = TileSystem.from_lattice(L)

    hub_polyhedron_indices = [
        i for i, src in enumerate(ts.tile_source)
        if src.get('type') == 'hub_polyhedron'
    ]
    assert len(hub_polyhedron_indices) >= 1, (
        "n_points=27 should produce at least one central hub; got types: "
        f"{set(s.get('type') for s in ts.tile_source)}"
    )

    sim = Simulator(ts, load_axis=np.array([0.0, -1.0, 0.0]))
    rest_pose = sim.rest_pose()

    posed_tiles = [_apply_tile_pose(ts.tiles[i], rest_pose, i, ts.dimension)
                   for i in range(len(ts.tiles))]
    strut_curves, solid_triangles, joint_positions = (
        collect_export_geometry_from_posed_tiles(
            posed_tiles, ts.tile_source, ts.dimension
        )
    )

    # Every tCOH vertex is on the hull, so each appears in ≥ 1 surface
    # triangle. The hub_center is interior — not in any triangle. So
    # we expect at least len(hub_verts) - 1 of the hub_polyhedron's
    # own vertex positions to register as joint positions.
    hub_idx = hub_polyhedron_indices[0]
    hub_verts = posed_tiles[hub_idx]
    matched = sum(
        1 for v in hub_verts if tuple(np.round(v, 8)) in joint_positions
    )
    assert matched >= len(hub_verts) - 1, (
        f"hub_polyhedron vertices missing from joint positions: "
        f"{matched}/{len(hub_verts)}"
    )

    # The tCOH triangulation contributes many surface triangles. Verify
    # a substantial number of solid triangles touch a hub vertex —
    # without the hub_polyhedron path they'd be absent entirely.
    hub_vert_keys = {tuple(np.round(v, 8)) for v in hub_verts}
    tris_touching_hub = sum(
        1 for tri in solid_triangles
        if any(tuple(np.round(v, 8)) in hub_vert_keys for v in tri)
    )
    assert tris_touching_hub >= 30, (
        f"too few solid triangles touch hub_polyhedron vertices: "
        f"{tris_touching_hub} (expected ≥ 30 from a tCOH triangulation)"
    )


def test_promoted_constraint_residual_check():
    """Promote the Stage 6c diagnostic to a permanent invariant: at
    rest pose AND at θ = ±π/4 along the kirigami mode, the per-tile
    pose application used by the renderer must keep every constraint's
    pinned vertices coincident in world space. If the renderer is
    using the wrong pose convention, this fires before any visual gap
    appears."""
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    ts = TileSystem.from_lattice(L)
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0]))

    mode = sim.identify_kirigami_mode()
    assert mode is not None, "this lattice has no kirigami mode"

    test_thetas = [0.0, +math.pi / 4.0, -math.pi / 4.0]
    for theta in test_thetas:
        if theta == 0.0:
            pose = sim.rest_pose()
        else:
            pose = sim.project_to_manifold(sim.rest_pose() + theta * mode)
        # Constraint residual via the simulator (analytic).
        r = sim.constraint_residual(pose)
        assert float(np.linalg.norm(r)) < 1e-6, (
            f"simulator residual at θ={theta:+.4f}: "
            f"{float(np.linalg.norm(r)):.3e}"
        )

        # Independent residual via the renderer's pose-application path
        # (proves the two conventions agree).
        posed_tiles = [_apply_tile_pose(ts.tiles[i], pose, i, ts.dimension)
                       for i in range(len(ts.tiles))]
        worst = 0.0
        for c in ts.constraints:
            v_a = posed_tiles[c.tile_a][c.vert_a]
            v_b = posed_tiles[c.tile_b][c.vert_b]
            worst = max(worst, float(np.linalg.norm(v_a - v_b)))
        assert worst < 1e-6, (
            f"renderer-path constraint residual at θ={theta:+.4f}: "
            f"{worst:.3e}"
        )
