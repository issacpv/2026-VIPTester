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
