"""Stage 3 edit-mode tests.

Five tests, mapped to the prompt:

1. ``test_edit_toggle_disabled_in_3d_modes`` — the Edit toolbar/menu
   action is disabled when ``lattice.mode in {3, 6}`` and enabled
   when ``mode == 1``.
2. ``test_regenerate_from_points_updates_triangulation`` — calling
   ``Lattice.regenerate_from_points`` swaps points and re-triangulates
   so the new simplex set matches a fresh Delaunay over the moved points.
3. ``test_move_point_command_undoable`` — push a ``MovePointCommand``,
   verify the move, then undo and verify the original position
   is restored exactly.
4. ``test_reset_to_original_after_edits`` — apply several moves, then
   ``reset_to_original()``, and verify ``points`` matches the snapshot.
5. ``test_delete_blocked_at_three_points`` — at the 3-point floor, the
   Delete action no-ops and the status bar carries an explanation.
"""

import os
import sys

# Force headless Qt before importing anything that touches QtGui/QtWidgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt6.QtWidgets import QApplication

from scipy.spatial import Delaunay

from auxetic import Lattice
from auxetic_studio.commands import MovePointCommand


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
# 1. Edit-mode gating by lattice.mode
# ---------------------------------------------------------------------------

def test_edit_toggle_disabled_in_3d_modes(main_window):
    """Per the §4.1.1 deferral: edit must be disabled in 3D modes
    (with the prompt-mandated tooltip), and enabled in 2D modes.
    M1 added modes 7 / 8 / 9 — 7 and 8 are 2D mesh-import (editable)
    and 9 is 3D mesh-import (not editable)."""
    win = main_window
    expected_disabled_tip = (
        "3D editing is not supported in this version. "
        "Switch to a 2D mode to edit points."
    )

    # Mode 3 → disabled, with tooltip.
    win.lattice.mode = 3
    win._update_edit_action_enabled()
    assert not win.edit_action.isEnabled()
    assert win.edit_action.toolTip() == expected_disabled_tip

    # Mode 6 → still disabled.
    win.lattice.mode = 6
    win._update_edit_action_enabled()
    assert not win.edit_action.isEnabled()
    assert win.edit_action.toolTip() == expected_disabled_tip

    # Mode 1 → enabled.
    win.lattice.mode = 1
    win._update_edit_action_enabled()
    assert win.edit_action.isEnabled()
    # Enabled state should NOT carry the disabled-mode tooltip.
    assert win.edit_action.toolTip() != expected_disabled_tip


# ---------------------------------------------------------------------------
# 2. regenerate_from_points refreshes the triangulation
# ---------------------------------------------------------------------------

def test_regenerate_from_points_updates_triangulation():
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    new_pts = L.points.copy()
    new_pts[0] = [0.99, 0.01]  # move point 0 well outside its original spot

    L.regenerate_from_points(new_pts)

    # Points propagated.
    np.testing.assert_array_equal(L.points, new_pts)
    # n_points should track the actual count.
    assert L.n_points == len(new_pts)

    # Triangulation reflects the new positions: simplex set must match
    # an independent Delaunay over the same moved points (compared as
    # sets of sorted index tuples since SciPy can permute simplex order
    # and within-simplex vertex order).
    expected = {tuple(sorted(s)) for s in Delaunay(new_pts).simplices}
    actual   = {tuple(sorted(s)) for s in L.tri.simplices}
    assert actual == expected


def test_regenerate_from_points_validates_dimension():
    """Sanity guard on the input shape — 2D mode rejects 3D arrays."""
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    bad = np.zeros((5, 3))
    with pytest.raises(ValueError):
        L.regenerate_from_points(bad)


# ---------------------------------------------------------------------------
# 3. MovePointCommand round-trips
# ---------------------------------------------------------------------------

def test_move_point_command_undoable():
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    old = L.points[0].astype(float).copy()
    new = old + np.array([0.1, 0.05])

    cmd = MovePointCommand(L, 0, old, new)
    cmd.redo()
    np.testing.assert_allclose(L.points[0], new)

    cmd.undo()
    np.testing.assert_allclose(L.points[0], old)


# ---------------------------------------------------------------------------
# 4. reset_to_original restores the snapshot
# ---------------------------------------------------------------------------

def test_reset_to_original_after_edits():
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    original = L.points_original.copy()

    # Several distinct moves so we know `points` has actually drifted.
    for i in range(3):
        old = L.points[i].astype(float).copy()
        new = old + np.array([0.05, 0.05])
        MovePointCommand(L, i, old, new).redo()
    # Sanity: edits actually changed the points.
    assert not np.allclose(L.points, original)

    L.reset_to_original()
    np.testing.assert_array_equal(L.points, original)
    np.testing.assert_array_equal(L.points, L.points_original)


# ---------------------------------------------------------------------------
# 5. Delete is blocked at the 3-point floor
# ---------------------------------------------------------------------------

def test_delete_blocked_at_three_points(main_window):
    win = main_window
    # Build a lattice that's already at the 3-point floor.
    win.lattice = Lattice(mode=1, n_points=3, ratio=0.35, seed=42)
    win.inspector.set_lattice(win.lattice)
    win._update_edit_action_enabled()

    # Enter edit mode and select point 0.
    win.edit_action.setChecked(True)  # toggled → set_edit_mode(True)
    assert win.edit_mode is True
    win.view_2d.selected_index = 0

    points_before = win.lattice.points.copy()
    win.delete_selected_point()

    # Points must be untouched.
    np.testing.assert_array_equal(win.lattice.points, points_before)

    # Status bar must explain the block.
    msg = win.statusBar().currentMessage()
    assert msg, f"expected a status message, got empty: {msg!r}"
    assert "3" in msg or "minimum" in msg.lower() or "least" in msg.lower(), (
        f"status message {msg!r} does not explain the 3-point floor"
    )
