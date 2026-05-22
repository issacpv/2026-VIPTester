"""GUI integration tests for the two-step edge-flip gesture.

The edge-flip UI (M1, redesigned) is a deliberate two-step gesture:

1. Click a flippable (blue) or flipped (red) triangulation edge — it
   turns **green** to show it's selected, and the two apex corners a
   flip would connect light up **amber**.
2. Hold Ctrl and click both amber corners. Once both are confirmed the
   view fires ``edgeFlipRequested`` and the flip lands on the undo
   stack.

These tests drive ``View2D``'s handlers directly (``_on_edge_clicked``
/ ``_on_corner_clicked``) — the same slots the ``_EdgeItem`` /
scatter signals are wired to — so the gesture logic is exercised
without synthesising pyqtgraph mouse events.
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
    """A MainWindow on the default mode-1 lattice (which has flippable
    edges (2,4) and (3,4))."""
    from PyQt6.QtWidgets import QApplication
    from auxetic_studio.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(headless_3d=True)
    yield win
    try:
        win.close()
    except Exception:
        pass


def _first_flippable_edge(lattice):
    from auxetic import geometry as g
    edges = g.flippable_edges(lattice.tri, np.asarray(lattice.points, float))
    assert edges, "test lattice must have at least one flippable edge"
    return tuple(sorted(edges[0]))


# ---------------------------------------------------------------------------
# Entering edge mode
# ---------------------------------------------------------------------------

def test_edge_mode_enables_corner_picking(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    assert win.view_2d.edge_mode is True
    # The scatter must accept clicks so corners can be Ctrl+clicked.
    assert win.view_2d._scatter._corner_pick_enabled is True
    # No edge is selected until the user clicks one.
    assert win.view_2d._edge_selected is None


# ---------------------------------------------------------------------------
# Step 1 — click an edge to select it
# ---------------------------------------------------------------------------

def test_clicking_edge_selects_it_and_computes_apexes(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)

    v._on_edge_clicked(edge)

    assert v._edge_selected == edge
    # Apex corners are computed and are distinct from the edge endpoints.
    assert v._edge_apexes is not None
    c, d = v._edge_apexes
    assert c != d
    assert c not in edge and d not in edge
    # Nothing flipped yet — selecting is step 1 only.
    assert edge not in win.lattice.edge_flips


def test_selected_edge_renders_green(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)
    v._on_edge_clicked(edge)
    # The matching _EdgeItem must carry the selected flag.
    selected_items = [it for it in v._edge_items if it.selected]
    assert len(selected_items) == 1
    assert selected_items[0].edge == edge


def test_clicking_selected_edge_again_deselects(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)
    v._on_edge_clicked(edge)
    assert v._edge_selected == edge
    # Second click on the same edge cancels the gesture.
    v._on_edge_clicked(edge)
    assert v._edge_selected is None
    assert v._edge_apexes is None


# ---------------------------------------------------------------------------
# Step 2 — Ctrl+click the two corners to confirm
# ---------------------------------------------------------------------------

def test_ctrl_clicking_both_corners_flips_the_edge(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)

    v._on_edge_clicked(edge)
    c, d = v._edge_apexes

    # First corner — confirmed, but no flip yet.
    v._on_corner_clicked(c, True)
    assert c in v._picked_corners
    assert edge not in win.lattice.edge_flips

    # Second corner — both confirmed, flip lands.
    v._on_corner_clicked(d, True)
    assert edge in win.lattice.edge_flips
    # Gesture resets after the flip.
    assert v._edge_selected is None


def test_one_corner_alone_does_not_flip(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)
    v._on_edge_clicked(edge)
    c, _d = v._edge_apexes
    v._on_corner_clicked(c, True)
    assert edge not in win.lattice.edge_flips
    # Edge stays selected, waiting for the second corner.
    assert v._edge_selected == edge


def test_corner_click_without_ctrl_is_ignored(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)
    v._on_edge_clicked(edge)
    c, d = v._edge_apexes
    # Plain clicks (no Ctrl) must not confirm corners.
    v._on_corner_clicked(c, False)
    v._on_corner_clicked(d, False)
    assert v._picked_corners == set()
    assert edge not in win.lattice.edge_flips


def test_ctrl_clicking_wrong_corner_is_ignored(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)
    v._on_edge_clicked(edge)
    apexes = set(v._edge_apexes)
    n_pts = len(win.lattice.points)
    # A vertex that is neither an apex nor an endpoint of the edge.
    bad = next(i for i in range(n_pts)
               if i not in apexes and i not in edge)
    v._on_corner_clicked(bad, True)
    assert bad not in v._picked_corners
    assert edge not in win.lattice.edge_flips


def test_corner_click_with_no_edge_selected_is_noop(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    # No edge selected — Ctrl+clicking a corner does nothing harmful.
    v._on_corner_clicked(0, True)
    assert v._picked_corners == set()
    assert v._edge_selected is None


# ---------------------------------------------------------------------------
# Undo / round-trip
# ---------------------------------------------------------------------------

def test_flip_via_gesture_is_undoable(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)

    v._on_edge_clicked(edge)
    c, d = v._edge_apexes
    v._on_corner_clicked(c, True)
    v._on_corner_clicked(d, True)
    assert edge in win.lattice.edge_flips

    win.undo_stack.undo()
    assert edge not in win.lattice.edge_flips

    win.undo_stack.redo()
    assert edge in win.lattice.edge_flips


# ---------------------------------------------------------------------------
# Status-bar guidance
# ---------------------------------------------------------------------------

def test_status_messages_guide_the_gesture(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)

    seen: list[str] = []
    v.edgeFlipStatus.connect(seen.append)

    v._on_edge_clicked(edge)
    assert any("selected" in m.lower() for m in seen)

    seen.clear()
    c, d = v._edge_apexes
    v._on_corner_clicked(c, True)
    assert any("other" in m.lower() or "one corner" in m.lower()
               for m in seen)


def test_leaving_edge_mode_clears_selection(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)
    v._on_edge_clicked(edge)
    assert v._edge_selected is not None
    # Turning edge mode off must drop the in-progress gesture.
    win.edge_action.setChecked(False)
    assert v._edge_selected is None
    assert v._scatter._corner_pick_enabled is False


# ---------------------------------------------------------------------------
# Ctrl+hover node enlargement — a fat click target so the confirming
# Ctrl+click can't slip onto an adjacent edge.
# ---------------------------------------------------------------------------

def test_ctrl_hover_enlarges_the_node(main_window):
    from auxetic_studio.views import _CTRL_HOVER_SIZE, _HOVER_SIZE
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    edge = _first_flippable_edge(win.lattice)
    v._on_edge_clicked(edge)
    c, d = v._edge_apexes

    # Ctrl+hovering apex c inflates it; the other apex keeps its size.
    v._on_corner_hover_changed(c)
    assert v._scatter.data[c]["size"] == _CTRL_HOVER_SIZE
    assert v._scatter.data[d]["size"] == _HOVER_SIZE

    # Hover off — the node shrinks back to the candidate size.
    v._on_corner_hover_changed(-1)
    assert v._scatter.data[c]["size"] == _HOVER_SIZE


def test_ctrl_hover_state_clears_on_exit(main_window):
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    v._on_corner_hover_changed(2)
    assert v._ctrl_hover_corner == 2
    win.edge_action.setChecked(False)
    assert v._ctrl_hover_corner == -1


def test_scatter_renders_above_edge_items(main_window):
    """The node scatter must sit above the edge items in Z-order so a
    Ctrl+click on a node is hit-tested against the node first."""
    win = main_window
    win.edge_action.setChecked(True)
    v = win.view_2d
    assert v._edge_items, "edge mode should render edge items"
    for item in v._edge_items:
        assert v._scatter.zValue() > item.zValue()


# ---------------------------------------------------------------------------
# Click gating — only a Ctrl+click claims a node; a plain click is left
# unaccepted so it falls through to the edge underneath.
# ---------------------------------------------------------------------------

class _FakeClickEvent:
    """Minimal stand-in for a pyqtgraph MouseClickEvent."""

    def __init__(self, ctrl: bool):
        from PyQt6.QtCore import Qt
        self._mods = (Qt.KeyboardModifier.ControlModifier if ctrl
                      else Qt.KeyboardModifier.NoModifier)
        self.accepted: bool | None = None

    def button(self):
        from PyQt6.QtCore import Qt
        return Qt.MouseButton.LeftButton

    def modifiers(self):
        return self._mods

    def pos(self):
        return None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def test_ctrl_click_on_node_is_accepted(main_window):
    """A Ctrl+click over a node is accepted by the scatter — combined
    with its higher Z-order, the node wins the click over any edge."""
    win = main_window
    win.edge_action.setChecked(True)
    sc = win.view_2d._scatter
    sc._index_at = lambda pos: 3            # stub hit-testing to node 3

    got: list[tuple[int, bool]] = []
    sc.sigCornerClicked.connect(lambda i, ctrl: got.append((i, ctrl)))

    ev = _FakeClickEvent(ctrl=True)
    sc.mouseClickEvent(ev)
    assert ev.accepted is True
    assert got == [(3, True)]


def test_plain_click_on_node_falls_through_to_edge(main_window):
    """A non-Ctrl click over a node is left UNACCEPTED so pyqtgraph
    dispatches it to the edge underneath — the user is selecting an
    edge, not picking a corner."""
    win = main_window
    win.edge_action.setChecked(True)
    sc = win.view_2d._scatter
    sc._index_at = lambda pos: 3

    got: list[tuple[int, bool]] = []
    sc.sigCornerClicked.connect(lambda i, ctrl: got.append((i, ctrl)))

    ev = _FakeClickEvent(ctrl=False)
    sc.mouseClickEvent(ev)
    assert ev.accepted is False              # not accepted → falls through
    assert got == []                         # no corner-click emitted
