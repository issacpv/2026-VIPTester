"""GUI wiring for the Bezier-edges control (task 1).

Drives the InspectorPanel bezier widgets through a headless MainWindow
(the proven-stable construction path other GUI tests use; a directly
constructed top-level panel destabilises offscreen Qt teardown). Because
MainWindow wires ``inspector.parameterChanged`` to the undoable
``ParameterChangeCommand`` path, toggling a widget here exercises the
full inspector -> command -> lattice flow end to end.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def main_window():
    from PyQt6.QtWidgets import QApplication
    from auxetic_studio.main_window import MainWindow
    QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(headless_3d=True)
    yield win
    try:
        win.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Widgets exist with the right defaults
# ---------------------------------------------------------------------------

def test_inspector_has_bezier_widgets(main_window):
    insp = main_window.inspector
    assert hasattr(insp, "bezier_enabled_check")
    assert hasattr(insp, "bezier_strength_spin")
    assert hasattr(insp, "bezier_segments_spin")
    assert insp.bezier_enabled_check.isChecked() is False
    assert insp.bezier_strength_spin.value() == pytest.approx(0.25)
    assert insp.bezier_segments_spin.value() == 12


# ---------------------------------------------------------------------------
# End-to-end: widget action -> command -> lattice
# ---------------------------------------------------------------------------

def test_toggling_checkbox_updates_lattice(main_window):
    win = main_window
    assert win.lattice.bezier_enabled is False
    win.inspector.bezier_enabled_check.setChecked(True)
    assert win.lattice.bezier_enabled is True


def test_toggling_checkbox_is_undoable(main_window):
    win = main_window
    win.inspector.bezier_enabled_check.setChecked(True)
    assert win.lattice.bezier_enabled is True
    win.undo_stack.undo()
    assert win.lattice.bezier_enabled is False


def test_segments_spin_updates_lattice(main_window):
    win = main_window
    win.inspector.bezier_segments_spin.setValue(8)
    assert win.lattice.bezier_segments == 8


def test_strength_commit_updates_lattice(main_window):
    win = main_window
    win.inspector.bezier_strength_spin.setValue(0.4)
    win.inspector._on_bezier_strength_committed()  # editingFinished surrogate
    assert win.lattice.bezier_strength == pytest.approx(0.4)


def test_bezier_change_does_not_regenerate_points(main_window):
    """Toggling bezier must not re-roll the point cloud (regenerate=False)."""
    win = main_window
    before = win.lattice.points.copy()
    win.inspector.bezier_enabled_check.setChecked(True)
    import numpy as np
    assert np.array_equal(win.lattice.points, before)


def test_enabling_then_exporting_curves_geometry(main_window, tmp_path):
    """End-to-end: enabling via the checkbox makes to_scad emit curved
    struts (more cylinders than the straight export)."""
    win = main_window
    # Use a lattice known to have struts.
    from auxetic import Lattice
    win.lattice = Lattice(mode=6, n_points=8, ratio=0.35)
    win.inspector.set_lattice(win.lattice)

    straight = tmp_path / "straight.scad"
    win.lattice.to_scad(str(straight), verbose=False)
    n_straight = straight.read_text().count("cylinder(")

    win.inspector.bezier_enabled_check.setChecked(True)
    win.inspector.bezier_segments_spin.setValue(5)
    curved = tmp_path / "curved.scad"
    win.lattice.to_scad(str(curved), verbose=False)
    n_curved = curved.read_text().count("cylinder(")

    assert n_straight >= 1
    assert n_curved == n_straight * 5


def test_refresh_from_lattice_syncs_widgets(main_window):
    win = main_window
    win.lattice.set_bezier(enabled=True, strength=0.5, segments=20)
    win.inspector.refresh_from_lattice()
    insp = win.inspector
    assert insp.bezier_enabled_check.isChecked() is True
    assert insp.bezier_strength_spin.value() == pytest.approx(0.5)
    assert insp.bezier_segments_spin.value() == 20
