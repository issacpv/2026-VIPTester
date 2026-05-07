"""GUI integration tests for the M2 "Run Dynamic" hook on the
SimulationPanel.

These exercise the wiring between the lattice's ``dynamics_state``,
the panel's ``run_dynamics`` slot, and the on-screen readout. They
intentionally stay coarse — the dynamics-engine math is covered by
``tests/test_dynamics.py``.
"""

from __future__ import annotations

import os

# Force offscreen Qt before any import that touches Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def main_window():
    """Spin up a MainWindow with a small mode-1 lattice and yield it."""
    from PyQt6.QtWidgets import QApplication
    from auxetic_studio.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(headless_3d=True)
    yield win
    try:
        win.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------

def test_panel_has_run_dynamic_button(main_window):
    panel = main_window.simulation_panel
    assert hasattr(panel, "run_dynamic_button")
    assert panel.run_dynamic_button.text() == "Run Dynamic"


def test_panel_starts_with_no_dynamics_result(main_window):
    panel = main_window.simulation_panel
    assert panel._dynamics_result is None
    assert panel._dynamics_error  is None


# ---------------------------------------------------------------------------
# Click → result
# ---------------------------------------------------------------------------

def test_clicking_run_dynamic_produces_result(main_window):
    """End-to-end: clicking Run Dynamic on a fresh mode-1 lattice
    populates ``_dynamics_result`` with a coherent DynamicsResult."""
    panel = main_window.simulation_panel
    # Use a small dt and short duration so the test runs fast.
    panel._lattice.dynamics_state["config"]["dt"]       = 1.0e-3
    panel._lattice.dynamics_state["config"]["duration"] = 0.05

    panel.run_dynamic_button.click()

    assert panel._dynamics_error is None, panel._dynamics_error
    assert panel._dynamics_result is not None
    res = panel._dynamics_result
    assert res.poses.ndim == 2
    assert res.times.shape[0] == res.poses.shape[0]


def test_run_dynamic_readout_includes_dynamic_section(main_window):
    panel = main_window.simulation_panel
    panel._lattice.dynamics_state["config"]["dt"]       = 1.0e-3
    panel._lattice.dynamics_state["config"]["duration"] = 0.05
    panel.run_dynamic_button.click()
    text = panel.readout.text()
    assert "Dynamic sim" in text
    assert "Final compression" in text
    assert "Converged" in text


# ---------------------------------------------------------------------------
# Lattice change drops the dynamics result
# ---------------------------------------------------------------------------

def test_lattice_change_invalidates_dynamics_result(main_window):
    panel = main_window.simulation_panel
    panel._lattice.dynamics_state["config"]["dt"]       = 1.0e-3
    panel._lattice.dynamics_state["config"]["duration"] = 0.05
    panel.run_dynamic_button.click()
    assert panel._dynamics_result is not None

    # A regenerate-style change calls mark_outdated via _on_lattice_structurally_changed.
    panel.mark_outdated()
    assert panel._dynamics_result is None
    assert "Dynamic sim" not in panel.readout.text()


# ---------------------------------------------------------------------------
# Build helper end-to-end (uses the real Lattice → DynamicsSimulator path)
# ---------------------------------------------------------------------------

def test_build_dynamics_simulator_from_lattice_smoke(main_window):
    """Independent of the GUI panel: the helper turns a live Lattice
    into a runnable DynamicsSimulator."""
    from auxetic.dynamics import build_dynamics_simulator_from_lattice
    lattice = main_window.lattice
    ds = build_dynamics_simulator_from_lattice(lattice)
    assert ds.n_tiles == ds.tile_system.n_tiles
    # Override duration to keep the smoke test fast.
    ds.config.duration = 0.02
    res = ds.simulate()
    assert res.poses.shape[0] >= 2


def test_build_with_ground_face_pins_those_tiles(main_window):
    """Setting ground_face = '-y' should auto-add tiles touching the
    bottom bbox face to fixed_tiles."""
    from auxetic.dynamics import build_dynamics_simulator_from_lattice
    lattice = main_window.lattice
    lattice.dynamics_state["ground_face"] = "-y"
    ds = build_dynamics_simulator_from_lattice(lattice)
    # At least one tile should be pinned for any non-degenerate lattice.
    assert len(ds.fixed_tiles) >= 1


# ---------------------------------------------------------------------------
# Mode toggle (Kinematic / Dynamic) — added in M2.7
# ---------------------------------------------------------------------------

def test_panel_has_mode_toggle(main_window):
    panel = main_window.simulation_panel
    assert hasattr(panel, "mode_kinematic_radio")
    assert hasattr(panel, "mode_dynamic_radio")
    assert panel.mode_kinematic_radio.isChecked()  # default
    assert not panel.mode_dynamic_radio.isChecked()


def test_dynamics_config_box_visible_only_in_dynamic_mode(main_window):
    """Use ``isHidden`` (explicit show/hide state) rather than
    ``isVisible`` because under ``QT_QPA_PLATFORM=offscreen`` the dock's
    parent isn't on-screen, so ``isVisible`` reports False even for
    widgets that are explicitly shown."""
    panel = main_window.simulation_panel
    assert panel._dynamics_config_box.isHidden()
    panel.mode_dynamic_radio.setChecked(True)
    assert not panel._dynamics_config_box.isHidden()
    panel.mode_kinematic_radio.setChecked(True)
    assert panel._dynamics_config_box.isHidden()


def test_running_dynamic_auto_switches_to_dynamic_mode(main_window):
    panel = main_window.simulation_panel
    panel._lattice.dynamics_state["config"]["duration"] = 0.05
    assert panel._scrub_mode == "kinematic"
    panel.run_dynamic_button.click()
    assert panel._scrub_mode == "dynamic"
    assert panel.mode_dynamic_radio.isChecked()


def test_slider_in_dynamic_mode_scrubs_through_time(main_window):
    """Moving the slider in dynamic mode should change the View3D pose
    by indexing into ``dynamics_result.poses`` rather than the
    kinematic ``sim_result``."""
    panel = main_window.simulation_panel
    panel._lattice.dynamics_state["config"]["duration"] = 0.05
    panel.run_dynamic_button.click()
    res = panel._dynamics_result
    assert res is not None and res.poses.shape[0] >= 2

    # Move the slider to 50%; the resolved index should be near n/2.
    midpoint = int((panel.slider.minimum() + panel.slider.maximum()) / 2)
    panel.slider.setValue(midpoint)
    # Verify panel's internal dispatcher routed via dynamic path.
    assert panel._scrub_mode == "dynamic"


def test_ground_face_combo_writes_through_to_lattice(main_window):
    panel = main_window.simulation_panel
    panel.mode_dynamic_radio.setChecked(True)   # so the combo is visible & wired
    # Pick "-y" via its data role.
    idx = panel.ground_face_combo.findData("-y")
    assert idx >= 0
    panel.ground_face_combo.setCurrentIndex(idx)
    assert panel._lattice.dynamics_state["ground_face"] == "-y"

    # Reset to none.
    idx = panel.ground_face_combo.findData("none")
    panel.ground_face_combo.setCurrentIndex(idx)
    assert panel._lattice.dynamics_state["ground_face"] is None


def test_ground_face_change_invalidates_dynamic_result(main_window):
    """Changing the ground face mid-session should drop a stale
    dynamics result so the readout doesn't lie about the new config."""
    panel = main_window.simulation_panel
    panel._lattice.dynamics_state["config"]["duration"] = 0.05
    panel.run_dynamic_button.click()
    assert panel._dynamics_result is not None

    panel.mode_dynamic_radio.setChecked(True)
    idx = panel.ground_face_combo.findData("-y")
    panel.ground_face_combo.setCurrentIndex(idx)
    assert panel._dynamics_result is None


# ---------------------------------------------------------------------------
# Force-table editor (M2.9)
# ---------------------------------------------------------------------------

def test_force_table_starts_empty(main_window):
    panel = main_window.simulation_panel
    assert panel.forces_table.rowCount() == 0


def test_add_force_appends_default_row(main_window):
    panel = main_window.simulation_panel
    panel._on_add_force()
    forces = main_window.lattice.dynamics_state["forces"]
    assert len(forces) == 1
    f = forces[0]
    assert f["tile_index"]    == 0
    assert f["vert_index"]    == -1
    assert f["location_kind"] == "tile_centroid"
    assert f["direction"]     == [1.0, 0.0, 0.0]
    assert f["magnitude"]     == 1.0
    # Table should now have one row populated.
    assert panel.forces_table.rowCount() == 1


def test_add_force_pushes_undoable_command(main_window):
    panel = main_window.simulation_panel
    initial_count = main_window.undo_stack.count()
    panel._on_add_force()
    assert main_window.undo_stack.count() == initial_count + 1
    # Undo restores the empty force list.
    main_window.undo_stack.undo()
    assert main_window.lattice.dynamics_state["forces"] == []
    # Redo brings the force back.
    main_window.undo_stack.redo()
    assert len(main_window.lattice.dynamics_state["forces"]) == 1


def test_remove_force_drops_selected_row(main_window):
    panel = main_window.simulation_panel
    # Add two forces so we have a non-trivial selection.
    panel._on_add_force()
    panel._on_add_force()
    # Select the first row and remove.
    panel.forces_table.selectRow(0)
    panel._on_remove_force()
    assert len(main_window.lattice.dynamics_state["forces"]) == 1


def test_remove_force_with_no_selection_is_noop(main_window):
    panel = main_window.simulation_panel
    panel._on_add_force()
    panel.forces_table.clearSelection()
    panel.forces_table.setCurrentCell(-1, -1)
    panel._on_remove_force()
    # No removal happened.
    assert len(main_window.lattice.dynamics_state["forces"]) == 1


def test_editing_magnitude_cell_propagates_to_lattice(main_window):
    panel = main_window.simulation_panel
    panel._on_add_force()
    # Programmatically edit the magnitude cell (column 5).
    panel.forces_table.item(0, 5).setText("4.5")
    assert main_window.lattice.dynamics_state["forces"][0]["magnitude"] == 4.5


def test_editing_vertex_to_positive_value_switches_to_tile_vertex_kind(main_window):
    panel = main_window.simulation_panel
    panel._on_add_force()
    # Default is centroid (vert_index = -1). Set vertex to 2.
    panel.forces_table.item(0, 1).setText("2")
    f = main_window.lattice.dynamics_state["forces"][0]
    assert f["vert_index"]    == 2
    assert f["location_kind"] == "tile_vertex"


def test_force_changes_invalidate_dynamics_result(main_window):
    """Like the ground-face change, editing forces should drop a
    stale dynamics result so the readout doesn't lie."""
    panel = main_window.simulation_panel
    panel._lattice.dynamics_state["config"]["duration"] = 0.05
    panel.run_dynamic_button.click()
    assert panel._dynamics_result is not None
    panel._on_add_force()
    assert panel._dynamics_result is None


# ---------------------------------------------------------------------------
# Force-arrow glyphs in View3D (M3 polish)
# ---------------------------------------------------------------------------

def test_force_glyphs_empty_when_no_forces(main_window):
    """No forces, no arrows — even after a sim run."""
    panel = main_window.simulation_panel
    panel.run_simulation()
    panel._refresh_force_glyphs()
    glyphs = main_window.view_3d.last_force_glyphs
    assert glyphs == [] or glyphs is None


def test_force_glyphs_populate_after_run_sim_with_forces(main_window):
    """One force, one arrow. The glyph data is recorded in
    ``View3D.last_force_glyphs`` regardless of whether the underlying
    VTK interactor renders (works headless too)."""
    panel = main_window.simulation_panel
    # Add a force first so it's present when run_simulation caches the
    # tile system.
    panel._on_add_force()
    panel.run_simulation()
    panel._refresh_force_glyphs()
    glyphs = main_window.view_3d.last_force_glyphs
    assert glyphs is not None
    assert len(glyphs) == 1
    origin, direction, mag_norm = glyphs[0]
    assert origin.shape == (3,)
    assert direction.shape == (3,)
    # Largest magnitude in the list normalises to 1.0.
    assert mag_norm == pytest.approx(1.0)


def test_force_glyph_normalisation_picks_largest():
    """Two forces with magnitudes 1 and 5 → mag_norms 0.2 and 1.0."""
    import numpy as np
    from auxetic_studio.views import View3D
    from auxetic.simulation import TileSystem

    tiles = [np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])]
    ts = TileSystem(2, tiles, constraints=[])
    forces = [
        {"location_kind": "tile_centroid", "tile_index": 0,
         "vert_index": -1, "direction": [1.0, 0.0, 0.0], "magnitude": 1.0},
        {"location_kind": "tile_centroid", "tile_index": 0,
         "vert_index": -1, "direction": [0.0, 1.0, 0.0], "magnitude": 5.0},
    ]
    glyphs = View3D._compute_glyph_data(ts, forces)
    assert len(glyphs) == 2
    mags = sorted(g[2] for g in glyphs)
    assert mags[0] == pytest.approx(0.2)
    assert mags[1] == pytest.approx(1.0)


def test_force_glyph_uses_tile_centroid_when_vert_index_negative():
    import numpy as np
    from auxetic_studio.views import View3D
    from auxetic.simulation import TileSystem

    # Triangle centroid at (0.5, 0.333…)
    tiles = [np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])]
    ts = TileSystem(2, tiles, constraints=[])
    forces = [{
        "location_kind": "tile_centroid", "tile_index": 0,
        "vert_index": -1, "direction": [1.0, 0.0, 0.0], "magnitude": 1.0,
    }]
    glyphs = View3D._compute_glyph_data(ts, forces)
    assert len(glyphs) == 1
    origin = glyphs[0][0]
    np.testing.assert_allclose(origin[:2], [0.5, 1.0 / 3.0], atol=1e-6)


def test_force_glyph_uses_specific_vertex_when_kind_is_tile_vertex():
    import numpy as np
    from auxetic_studio.views import View3D
    from auxetic.simulation import TileSystem

    tiles = [np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])]
    ts = TileSystem(2, tiles, constraints=[])
    forces = [{
        "location_kind": "tile_vertex", "tile_index": 0,
        "vert_index": 1, "direction": [0.0, 1.0, 0.0], "magnitude": 2.0,
    }]
    glyphs = View3D._compute_glyph_data(ts, forces)
    origin = glyphs[0][0]
    np.testing.assert_allclose(origin[:2], [1.0, 0.0], atol=1e-12)


def test_force_glyph_skips_invalid_tile_or_vertex_indices():
    """Out-of-range tile_index or vert_index → silently skip the
    record so the panel doesn't crash on partially-edited tables."""
    import numpy as np
    from auxetic_studio.views import View3D
    from auxetic.simulation import TileSystem

    tiles = [np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])]
    ts = TileSystem(2, tiles, constraints=[])
    forces = [
        {"location_kind": "tile_centroid", "tile_index": 99,
         "vert_index": -1, "direction": [1.0, 0.0, 0.0], "magnitude": 1.0},
        {"location_kind": "tile_vertex", "tile_index": 0,
         "vert_index": 7, "direction": [1.0, 0.0, 0.0], "magnitude": 1.0},
    ]
    assert View3D._compute_glyph_data(ts, forces) == []


def test_force_glyph_skips_zero_direction():
    """A zero-direction force can't be normalised → drop it from the
    glyph list (the dataset / table editor would never produce one in
    practice but the helper guards against it)."""
    import numpy as np
    from auxetic_studio.views import View3D
    from auxetic.simulation import TileSystem

    tiles = [np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])]
    ts = TileSystem(2, tiles, constraints=[])
    forces = [{
        "location_kind": "tile_centroid", "tile_index": 0,
        "vert_index": -1, "direction": [0.0, 0.0, 0.0], "magnitude": 1.0,
    }]
    assert View3D._compute_glyph_data(ts, forces) == []


def test_force_table_populates_from_lattice_on_set_lattice(main_window):
    """When the panel is rebound to a lattice that already has forces
    (e.g. after File → Open of a v4 preset), the table should reflect
    the loaded forces immediately."""
    panel = main_window.simulation_panel
    main_window.lattice.dynamics_state["forces"] = [
        {"location_kind": "tile_centroid", "tile_index": 1,
         "vert_index": -1, "direction": [0.0, -1.0, 0.0], "magnitude": 7.5},
    ]
    panel.set_lattice(main_window.lattice)
    assert panel.forces_table.rowCount() == 1
    assert panel.forces_table.item(0, 0).text() == "1"
    assert panel.forces_table.item(0, 5).text() == "7.5000"
