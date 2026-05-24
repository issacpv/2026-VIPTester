"""GUI tests for the Coordinates panel — the tabular point editor.

The panel lists every lattice point with editable X / Y cells; typing a
value routes through the same undoable ``MovePointCommand`` the
viewport drag path uses. These tests drive the panel's table directly
(``QTableWidgetItem.setText`` fires ``cellChanged`` synchronously, just
like a user committing a cell edit).

Scope is 2D / 2.5D only — in 3D modes the table is cleared and
disabled.
"""

from __future__ import annotations

import os

# Force offscreen Qt before any import that touches Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def main_window():
    """A MainWindow on the default mode-1 lattice."""
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
# Existence / wiring
# ---------------------------------------------------------------------------

def test_main_window_has_coordinates_panel(main_window):
    win = main_window
    assert hasattr(win, "coordinates_panel")
    from auxetic_studio.coordinates_panel import CoordinatesPanel
    assert isinstance(win.coordinates_panel, CoordinatesPanel)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_table_lists_every_point(main_window):
    win = main_window
    cp = win.coordinates_panel
    pts = np.asarray(win.lattice.points, dtype=float)
    assert cp.table.rowCount() == pts.shape[0]
    # Every X / Y cell matches the lattice point (to display precision).
    for row in range(pts.shape[0]):
        x = float(cp.table.item(row, 1).text())
        y = float(cp.table.item(row, 2).text())
        assert x == pytest.approx(pts[row, 0], abs=1e-6)
        assert y == pytest.approx(pts[row, 1], abs=1e-6)


def test_index_column_is_read_only(main_window):
    from PyQt6.QtCore import Qt
    cp = main_window.coordinates_panel
    for row in range(cp.table.rowCount()):
        flags = cp.table.item(row, 0).flags()
        assert not (flags & Qt.ItemFlag.ItemIsEditable)
        # X / Y cells, by contrast, must stay editable.
        assert cp.table.item(row, 1).flags() & Qt.ItemFlag.ItemIsEditable
        assert cp.table.item(row, 2).flags() & Qt.ItemFlag.ItemIsEditable


# ---------------------------------------------------------------------------
# Editing a cell moves the point
# ---------------------------------------------------------------------------

def test_editing_x_cell_moves_the_point(main_window):
    win = main_window
    cp = win.coordinates_panel
    cp.table.item(2, 1).setText("0.7770")
    assert win.lattice.points[2][0] == pytest.approx(0.777, abs=1e-9)


def test_editing_y_cell_keeps_x_bit_exact(main_window):
    """Editing only the Y cell must not perturb X — the un-edited axis
    is taken from the lattice, not the display-rounded cell."""
    win = main_window
    cp = win.coordinates_panel
    exact_x = float(win.lattice.points[3][0])
    cp.table.item(3, 2).setText("0.123456")
    assert win.lattice.points[3][1] == pytest.approx(0.123456, abs=1e-9)
    # X is bit-exact, not merely close.
    assert win.lattice.points[3][0] == exact_x


def test_edit_pushes_one_undo_command(main_window):
    win = main_window
    cp = win.coordinates_panel
    n_before = win.undo_stack.count()
    cp.table.item(1, 1).setText("0.4200")
    assert win.undo_stack.count() == n_before + 1


def test_edit_is_undoable_and_redoable(main_window):
    win = main_window
    cp = win.coordinates_panel
    original = float(win.lattice.points[0][0])
    cp.table.item(0, 1).setText("0.9000")
    assert win.lattice.points[0][0] == pytest.approx(0.9, abs=1e-9)

    win.undo_stack.undo()
    assert win.lattice.points[0][0] == pytest.approx(original, abs=1e-12)

    win.undo_stack.redo()
    assert win.lattice.points[0][0] == pytest.approx(0.9, abs=1e-9)


def test_committing_unchanged_value_pushes_no_command(main_window):
    """Re-entering the value already shown is a no-op — no undo entry."""
    win = main_window
    cp = win.coordinates_panel
    n_before = win.undo_stack.count()
    current = cp.table.item(0, 1).text()
    cp.table.item(0, 1).setText(current)
    assert win.undo_stack.count() == n_before


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_invalid_text_reverts_and_pushes_no_command(main_window):
    win = main_window
    cp = win.coordinates_panel
    n_before = win.undo_stack.count()
    before_x = float(win.lattice.points[1][0])

    cp.table.item(1, 1).setText("not-a-number")

    # No command pushed, lattice untouched.
    assert win.undo_stack.count() == n_before
    assert win.lattice.points[1][0] == pytest.approx(before_x, abs=1e-12)
    # The cell reverted to the lattice value.
    assert float(cp.table.item(1, 1).text()) == pytest.approx(
        before_x, abs=1e-6)


# ---------------------------------------------------------------------------
# Sync with other edit paths
# ---------------------------------------------------------------------------

def test_table_refreshes_after_external_point_change(main_window):
    """A drag-style move (routed through the same command) must show up
    in the table after the lattice refreshes."""
    win = main_window
    cp = win.coordinates_panel
    old = np.asarray(win.lattice.points[4][:2], dtype=float)
    new = old + np.array([0.05, -0.03])
    win._on_point_move_completed(4, old, new)
    # Table now reflects the moved point.
    assert float(cp.table.item(4, 1).text()) == pytest.approx(
        new[0], abs=1e-6)
    assert float(cp.table.item(4, 2).text()) == pytest.approx(
        new[1], abs=1e-6)


# ---------------------------------------------------------------------------
# 3D-mode gating
# ---------------------------------------------------------------------------

def test_3d_mode_clears_and_disables_table(main_window):
    win = main_window
    win.inspector.select_mode(3)          # 3D random
    cp = win.coordinates_panel
    assert cp.table.rowCount() == 0
    assert cp.table.isEnabled() is False


def test_switching_back_to_2d_repopulates_table(main_window):
    win = main_window
    win.inspector.select_mode(3)          # 3D — table cleared
    assert win.coordinates_panel.table.rowCount() == 0
    win.inspector.select_mode(1)          # back to 2D
    cp = win.coordinates_panel
    assert cp.table.isEnabled() is True
    assert cp.table.rowCount() == len(win.lattice.points)


# ---------------------------------------------------------------------------
# Mode 11 (bipartite auxetic) is a 2D editable mode
# ---------------------------------------------------------------------------

def test_mode_11_enables_coordinate_editing(main_window):
    """Regression: mode 11 was missing from the panel's editable-mode
    list, so the table disabled itself for bipartite-auxetic lattices."""
    win = main_window
    win.inspector.select_mode(11)
    cp = win.coordinates_panel
    assert cp.table.isEnabled() is True
    assert cp.table.rowCount() == len(win.lattice.points)


# ---------------------------------------------------------------------------
# Expression input (sqrt(3), pi, ...)
# ---------------------------------------------------------------------------

def test_expression_input_sets_point(main_window):
    win = main_window
    cp = win.coordinates_panel
    cp.table.item(0, 1).setText("sqrt(2)/2")
    assert win.lattice.points[0][0] == pytest.approx(2 ** 0.5 / 2, abs=1e-9)


def test_expression_allows_values_outside_unit_square(main_window):
    """The (1, sqrt(3)) equilateral-triangle use case: coordinates above
    1 are accepted (the 2D view auto-ranges)."""
    win = main_window
    cp = win.coordinates_panel
    cp.table.item(0, 2).setText("sqrt(3)")
    assert win.lattice.points[0][1] == pytest.approx(3 ** 0.5, abs=1e-9)


def test_bad_expression_reverts_cell(main_window):
    win = main_window
    cp = win.coordinates_panel
    n_before = win.undo_stack.count()
    before_y = float(win.lattice.points[1][1])
    cp.table.item(1, 2).setText("sqrt(")          # syntax error
    assert win.undo_stack.count() == n_before
    assert win.lattice.points[1][1] == pytest.approx(before_y, abs=1e-12)


def test_parse_coordinate_numbers_and_expressions():
    import math
    from auxetic_studio.coordinates_panel import parse_coordinate
    assert parse_coordinate("0") == 0.0
    assert parse_coordinate("2.5") == 2.5
    assert parse_coordinate("-1.5") == -1.5
    assert parse_coordinate("sqrt(3)") == pytest.approx(math.sqrt(3))
    assert parse_coordinate("1 + sqrt(3)") == pytest.approx(1 + math.sqrt(3))
    assert parse_coordinate("2*pi/3") == pytest.approx(2 * math.pi / 3)
    assert parse_coordinate("cos(0)") == pytest.approx(1.0)
    assert parse_coordinate("3**2") == pytest.approx(9.0)
    assert parse_coordinate("(1+2)*4") == pytest.approx(12.0)


def test_parse_coordinate_rejects_bad_and_unsafe():
    from auxetic_studio.coordinates_panel import parse_coordinate
    for bad in ["", "   ", "abc", "x + 1", "__import__('os')",
                "os.system('x')", "[1, 2]", "lambda: 1", "1/0",
                "sqrt(-1)", "1j"]:
        with pytest.raises(ValueError):
            parse_coordinate(bad)
