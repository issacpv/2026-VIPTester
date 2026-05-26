"""GUI tests for the Tile Library (compose-from-tiles).

Covers the wiring on top of the pure composition geometry (tested in
``tests/test_composition.py``):

- the Library panel lists draggable tile templates;
- View2D accepts the tile drag payload and maps a drop into lattice
  space, emitting ``tileDropRequested``;
- a drop is an undoable ``AddTileCommand`` that composes/welds the tile
  and switches the lattice into the mode-11 compose state.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtCore import QByteArray, QMimeData, Qt
from PyQt6.QtWidgets import QApplication

from auxetic_studio.library_panel import LibraryPanel, TILE_MIME


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def win(qapp):
    from auxetic_studio.main_window import MainWindow
    w = MainWindow(headless_3d=True)
    yield w
    try:
        w.close()
    except Exception:
        pass


def _tile_mime(name: str) -> QMimeData:
    md = QMimeData()
    md.setData(TILE_MIME, QByteArray(name.encode("utf-8")))
    return md


# ---------------------------------------------------------------------------
# Library panel
# ---------------------------------------------------------------------------

def test_library_panel_lists_all_tiles(qapp):
    panel = LibraryPanel()
    names = [panel.list.item(i).data(Qt.ItemDataRole.UserRole)
             for i in range(panel.list.count())]
    assert set(names) == {"Triangle", "Triangle-down", "Square", "Hexagon"}
    assert panel.list.dragEnabled() is True


def test_main_window_has_library_dock(win):
    assert hasattr(win, "library_panel")
    assert isinstance(win.library_panel, LibraryPanel)


# ---------------------------------------------------------------------------
# View2D drop target
# ---------------------------------------------------------------------------

def test_view2d_accepts_drops(win):
    assert win.view_2d.acceptDrops() is True


def test_has_tile_payload_discriminates(win):
    class _Ev:
        def __init__(self, md):
            self._md = md

        def mimeData(self):
            return self._md

    assert win.view_2d._has_tile_payload(_Ev(_tile_mime("Square"))) is True
    plain = QMimeData()
    plain.setText("not a tile")
    assert win.view_2d._has_tile_payload(_Ev(plain)) is False


# Note: the full drop→coordinate-mapping path (dropEvent) is not unit
# tested — it needs a rendered window, and forcing one (``win.show()``)
# trips the documented Windows + PyQt6 + offscreen VTK abort (see
# tests/conftest.py). The mapping reuses ``world_to_canonical_2d``
# (covered by the rotation/edit tests) plus stock pyqtgraph
# scene↔view mapping; the compose/undo logic below is what's load-bearing.


# ---------------------------------------------------------------------------
# Drop → undoable compose command
# ---------------------------------------------------------------------------

def test_dropping_a_tile_composes_and_is_undoable(win):
    n_before = win.undo_stack.count()
    win._on_tile_dropped("Square", 0.45, 0.5)
    assert win.lattice.mode == 11
    assert win.lattice.preserve_triangulation is True
    assert win.lattice.n_points == 4
    assert win.undo_stack.count() == n_before + 1

    win.undo_stack.undo()
    # Back to the pre-compose lattice.
    assert win.lattice.preserve_triangulation is False


def test_two_adjacent_drops_weld(win):
    win._on_tile_dropped("Square", 0.4, 0.5)
    win._on_tile_dropped("Square", 0.65, 0.5)   # one edge (0.25) over
    # 4 + 4 − 2 welded shared-edge vertices.
    assert win.lattice.n_points == 6
    assert len(win.lattice.tri.simplices) == 4


def test_drop_forces_2d_view(win):
    from auxetic_studio.main_window import VIEW_2D
    win._on_tile_dropped("Triangle", 0.5, 0.5)
    assert win.stack.currentIndex() == VIEW_2D


# ---------------------------------------------------------------------------
# All panels consolidated into one tab strip
# ---------------------------------------------------------------------------

def test_all_panels_share_one_tab_strip(win):
    group = win.tabifiedDockWidgets(win._inspector_dock)
    titles = {d.windowTitle() for d in group}
    assert titles == {"Coordinates", "Tile Library", "Simulation", "Predictor"}


# ---------------------------------------------------------------------------
# Tile size + model scale
# ---------------------------------------------------------------------------

def _span(win) -> float:
    p = np.asarray(win.lattice.points, dtype=float)
    return float((p.max(axis=0) - p.min(axis=0)).max())


def test_library_tile_size_controls_drop_scale(win):
    # Default tile edge is 0.25; a square dropped alone spans one edge.
    win._on_tile_dropped("Square", 0.5, 0.5)
    assert _span(win) == pytest.approx(0.25, abs=1e-6)
    # Doubling the tile size doubles the dropped structure.
    win.undo_stack.undo()
    win.library_panel.tile_size_spin.setValue(0.5)
    win._on_tile_dropped("Square", 0.5, 0.5)
    assert _span(win) == pytest.approx(0.5, abs=1e-6)


def test_scale_model_button_scales_and_is_undoable(win):
    win._on_tile_dropped("Square", 0.5, 0.5)
    span0 = _span(win)
    n_before = win.undo_stack.count()
    win.library_panel.scale_spin.setValue(2.0)
    win.library_panel._on_scale_clicked()          # emits scaleModelRequested(2.0)
    assert win.undo_stack.count() == n_before + 1
    assert _span(win) == pytest.approx(2.0 * span0, rel=1e-6)

    win.undo_stack.undo()
    assert _span(win) == pytest.approx(span0, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge flip on a composed lattice (regression: it used to record-but-not-apply)
# ---------------------------------------------------------------------------

def _edges(lat):
    s = set()
    for t in lat.tri.simplices:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            s.add(tuple(sorted((int(a), int(b)))))
    return s


def test_edge_flip_on_composed_lattice_actually_swaps_diagonal(win):
    from auxetic import geometry as _geom
    # Two edge-adjacent squares → a flippable interior diagonal.
    win._on_tile_dropped("Square", 0.4, 0.5)
    win._on_tile_dropped("Square", 0.65, 0.5)
    lat = win.lattice
    flips = _geom.flippable_edges(lat.tri, lat.points)
    assert flips, "expected a flippable diagonal in the composed mesh"
    edge = tuple(sorted(flips[0]))
    assert edge in _edges(lat)

    win._on_edge_flip_requested(edge, edge in lat.edge_flips)
    assert edge not in _edges(lat)        # the diagonal actually swapped

    win.undo_stack.undo()
    assert edge in _edges(lat)            # undo restores the original diagonal
