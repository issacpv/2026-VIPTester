"""View widgets for the central QStackedWidget.

View2D — pyqtgraph PlotWidget with a background grid, a scatter of
lattice points, and (when edit mode is on) draggable point handles
backed by a custom ``DraggablePointsItem`` ScatterPlotItem subclass.
The 2D view also has an Edge mode (M1) that renders triangulation
edges as clickable line segments — a click toggles the per-edge
Delaunay diagonal flip and emits ``edgeFlipRequested``.

View3D — pyvistaqt QtInteractor showing the STL mesh produced by
Lattice.to_stl() into a temp file.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsPolygonItem

import pyvista as pv

from auxetic import geometry as _geom
from auxetic_studio.camera_controls import (
    ZOOM_WHEEL_STEP as _ZOOM_WHEEL_STEP,
    dolly_toward_cursor,
)

try:
    from pyvistaqt import QtInteractor
    _PYVISTAQT_AVAILABLE = True
except Exception:  # pragma: no cover - import-time platform issues
    QtInteractor = None
    _PYVISTAQT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Poisson-bounds overlay: the seven extremal-pose boxes (SPEC §7.4 viz).
# Single source of truth shared with the Simulation panel, which colours the
# matching show/hide checkboxes to each box's wireframe. Each entry is
# ``(key, short label, wireframe + point colour)``; the tuple order is both
# the draw order and the checkbox order. Keys match
# ``Simulator.extremal_pose_indices``.
# ---------------------------------------------------------------------------
POISSON_BOXES: tuple[tuple[str, str, str], ...] = (
    ("initial",         "Initial",   "#8a8a8a"),
    ("compressed_pos",  "Comp +θ",   "#ff4d4d"),
    ("compressed_neg",  "Comp −θ",   "#ff9a3d"),
    # The overall expanded footprint: a single envelope enclosing the four
    # directional reach boxes below (biggest +x/−x/+y/−y of the sweep). Bold
    # black so the outer frame reads against the (light) viewport + panel —
    # white washed out on both.
    ("footprint",       "Footprint", "#000000"),
    ("expansion_pos_x", "+X",        "#33dd55"),
    ("expansion_neg_x", "−X",        "#00bcd4"),
    ("expansion_pos_y", "+Y",        "#4d79ff"),
    ("expansion_neg_y", "−Y",        "#c04dff"),
)
POISSON_BOX_COLORS: dict[str, str] = {k: c for k, _lbl, c in POISSON_BOXES}
POISSON_BOX_KEYS: tuple[str, ...] = tuple(k for k, _lbl, _c in POISSON_BOXES)


# Edit-mode visual styling.
_NORMAL_SIZE   = 10.0
_HOVER_SIZE    = 13.0
_SELECTED_SIZE = 14.0
# Edge-mode: a node the user is Ctrl+hovering inflates to this size —
# a fat click target so the flip-confirm Ctrl+click can't slip onto an
# adjacent edge.
_CTRL_HOVER_SIZE = 22.0

_NORMAL_BRUSH   = pg.mkBrush(30, 100, 200, 200)
_HOVER_BRUSH    = pg.mkBrush(255, 180, 60, 230)
_SELECTED_BRUSH = pg.mkBrush(220, 60, 60, 230)
_NORMAL_PEN     = pg.mkPen("k", width=0.8)

# Snap step for Shift-drag (lattice space).
_SNAP_STEP = 0.05

# Mode-11 auxetic-tile styling. Each triangle yields three corner kites
# (set A) and one central polygon (set B). Corner kites are filled blue
# and the central polygon green; the shared centroid hinges
# (central-polygon vertices) are dotted black. The kite fill is kept
# light so the darker-blue inner-edge highlight (below) stays legible
# on top of it.
_BIP_SET_A_BRUSH  = pg.mkBrush(120, 170, 235, 150)
_BIP_SET_A_PEN    = pg.mkPen(40, 90, 170, width=1.5)
_BIP_SET_B_BRUSH  = pg.mkBrush(90, 200, 110, 150)
_BIP_SET_B_PEN    = pg.mkPen(30, 130, 50, width=1.5)
# The two inner kite edges (edge-point -> hinge) are perpendicular to
# the triangle faces; drawn blue to make that hinge geometry explicit.
_BIP_INNER_EDGE_PEN = pg.mkPen(30, 90, 230, width=2.5)
# Bonds connecting adjacent kites along each triangle edge — black.
_BIP_BOND_PEN       = pg.mkPen(0, 0, 0, width=3.0)
# A degree-2 polygon (should not occur for the kite construction, kept
# as a defensive fallback) renders as a "hinge bar" segment.
_BIP_HINGE_BAR_PEN = pg.mkPen(60, 60, 60, width=2.0)
_BIP_HINGE_BRUSH   = pg.mkBrush(20, 20, 20, 230)
_BIP_HINGE_SIZE    = 6.0


def _snap(value: float) -> float:
    return round(value / _SNAP_STEP) * _SNAP_STEP


class DraggablePointsItem(pg.ScatterPlotItem):
    """A ScatterPlotItem with hover / click / drag handlers gated by an
    ``edit_enabled`` flag.

    Emits index-based signals only — coordinate transforms, snap-to-
    grid policy, and lattice mutation all happen in ``View2D`` /
    ``MainWindow`` so this class stays a thin Qt-event adapter.
    """

    sigPointClicked     = pyqtSignal(int)
    sigPointDragStart   = pyqtSignal(int)
    sigPointDragLive    = pyqtSignal(int, float, float, bool)  # idx, x, y, snap
    sigPointDragFinish  = pyqtSignal(int, float, float, bool)  # idx, x, y, snap
    sigHoverChanged     = pyqtSignal(int)                      # -1 if no hover
    sigCornerClicked    = pyqtSignal(int, bool)                # idx, ctrl_held
    sigCornerHoverChanged = pyqtSignal(int)   # node Ctrl+hovered, -1 if none

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._edit_enabled        = False
        # Edge-mode corner picking — left-clicks emit ``sigCornerClicked``
        # instead of the edit-mode click signal. Independent of
        # ``_edit_enabled``; the two are never on at once.
        self._corner_pick_enabled = False
        self._drag_index          = -1
        self._hover_index         = -1
        # Node currently under a Ctrl+hover in corner-pick mode (-1 if
        # none). Drives the enlarged click-target affordance.
        self._ctrl_hover_index    = -1

        # Default: don't intercept any mouse buttons (ViewBox pans, etc.).
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    # ------------------------------------------------------------------

    def _sync_accepted_buttons(self) -> None:
        """Accept left-clicks when either edit mode or corner-pick mode
        is on. Hover events are wanted in both: edit mode uses them for
        the drag affordance, corner-pick mode for the Ctrl+hover
        node-enlarge affordance."""
        want_clicks = self._edit_enabled or self._corner_pick_enabled
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton if want_clicks
            else Qt.MouseButton.NoButton
        )
        self.setAcceptHoverEvents(want_clicks)

    def setEditEnabled(self, enabled: bool) -> None:
        if self._edit_enabled == enabled:
            return
        self._edit_enabled = enabled
        self._sync_accepted_buttons()
        if not enabled:
            if self._hover_index != -1:
                self._hover_index = -1
                self.sigHoverChanged.emit(-1)
            self._drag_index = -1

    def setCornerPickEnabled(self, enabled: bool) -> None:
        """Enable click-to-pick-corner handling for 2D edge mode. When
        on, a Ctrl+click on a point emits ``sigCornerClicked`` and a
        Ctrl+hover emits ``sigCornerHoverChanged`` for the node-enlarge
        affordance."""
        enabled = bool(enabled)
        if self._corner_pick_enabled == enabled:
            return
        self._corner_pick_enabled = enabled
        if not enabled and self._ctrl_hover_index != -1:
            # Leaving corner-pick mode — drop any stale hover state.
            self._ctrl_hover_index = -1
            self.sigCornerHoverChanged.emit(-1)
        self._sync_accepted_buttons()

    # ------------------------------------------------------------------

    def _index_at(self, pos) -> int:
        pts = self.pointsAt(pos)
        return int(pts[0].index()) if pts else -1

    def mouseClickEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore(); return
        if not (self._edit_enabled or self._corner_pick_enabled):
            ev.ignore(); return
        idx = self._index_at(ev.pos())
        if idx < 0:
            ev.ignore(); return

        if self._edit_enabled:
            ev.accept()
            self.sigPointClicked.emit(idx)
            return

        # Corner-pick mode. Only a **Ctrl+click** claims the node — that
        # is the flip-confirm gesture. The scatter sits above the edge
        # items in Z-order, so accepting here makes the node win the
        # click over any edge passing under it. A plain (non-Ctrl) click
        # is deliberately left UNACCEPTED so it falls through to the
        # edge underneath — the user is selecting an edge, not a corner.
        ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if not ctrl:
            ev.ignore(); return
        ev.accept()
        self.sigCornerClicked.emit(idx, True)

    def mouseDragEvent(self, ev):
        if not self._edit_enabled or ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore(); return

        if ev.isStart():
            idx = self._index_at(ev.buttonDownPos())
            if idx < 0:
                self._drag_index = -1
                ev.ignore(); return
            self._drag_index = idx
            ev.accept()
            self.sigPointDragStart.emit(idx)
            return

        if self._drag_index < 0:
            ev.ignore(); return

        ev.accept()
        pos = ev.pos()
        x, y = float(pos.x()), float(pos.y())
        snap = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if ev.isFinish():
            idx = self._drag_index
            self._drag_index = -1
            self.sigPointDragFinish.emit(idx, x, y, snap)
        else:
            self.sigPointDragLive.emit(self._drag_index, x, y, snap)

    def hoverEvent(self, ev):
        if self._edit_enabled:
            # Edit mode — hover drives the drag affordance highlight.
            if ev.isExit():
                if self._hover_index != -1:
                    self._hover_index = -1
                    self.sigHoverChanged.emit(-1)
                return
            idx = self._index_at(ev.pos())
            if idx != self._hover_index:
                self._hover_index = idx
                self.sigHoverChanged.emit(idx)
            return

        if self._corner_pick_enabled:
            # Corner-pick mode — a node under a Ctrl+hover enlarges into
            # a fat click target. Plain (non-Ctrl) hovers don't, so the
            # affordance only appears when the user is actually about to
            # Ctrl+click a corner.
            new_idx = -1
            if not ev.isExit():
                ctrl = bool(
                    ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                if ctrl:
                    new_idx = self._index_at(ev.pos())
            if new_idx != self._ctrl_hover_index:
                self._ctrl_hover_index = new_idx
                self.sigCornerHoverChanged.emit(new_idx)


_EDGE_FLIPPED_PEN    = pg.mkPen(220, 60, 60,   width=2.0)
_EDGE_FLIPPABLE_PEN  = pg.mkPen(30, 100, 200,  width=1.4)
_EDGE_FROZEN_PEN     = pg.mkPen(170, 170, 170, width=1.0,
                                 style=Qt.PenStyle.DashLine)
# Selected edge — green, thick. Set when the user clicks a flippable /
# flipped edge to begin the two-step flip gesture.
_EDGE_SELECTED_PEN   = pg.mkPen(40, 180, 70, width=3.2)

# Edge-mode corner highlights. Amber = a corner a flip of the selected
# edge can connect (Ctrl-click target); green = a corner the user has
# already Ctrl-clicked to confirm.
_CORNER_CANDIDATE_BRUSH = pg.mkBrush(255, 170, 40, 235)
_CORNER_PICKED_BRUSH    = pg.mkBrush(40, 180, 70, 240)


class _EdgeItem(pg.PlotCurveItem):
    """One clickable triangulation edge in 2D edge mode.

    Four styles encode the edge's role:
    - Green, solid, thick: currently selected (first step of a flip).
    - Red, solid, thick: currently flipped from the canonical Delaunay diagonal.
    - Blue, solid: flippable (interior + convex quad), but not yet flipped.
    - Grey, dashed: not flippable (boundary edge or non-convex quad).
    """

    sigEdgeClicked = pyqtSignal(object)  # edge tuple (i, j) with i < j

    def __init__(self, edge, *, flipped: bool, flippable: bool,
                 selected: bool = False):
        super().__init__()
        self.edge = (int(edge[0]), int(edge[1]))
        self.flipped = bool(flipped)
        self.flippable = bool(flippable)
        self.selected = bool(selected)
        if self.selected:
            pen = _EDGE_SELECTED_PEN
        elif self.flipped:
            pen = _EDGE_FLIPPED_PEN
        elif self.flippable:
            pen = _EDGE_FLIPPABLE_PEN
        else:
            pen = _EDGE_FROZEN_PEN
        self.setPen(pen)
        # Only flippable / flipped edges respond to clicks; frozen
        # boundary edges are inert.
        if self.flippable or self.flipped:
            self.setClickable(True, width=8)
            self.sigClicked.connect(self._on_clicked)

    def _on_clicked(self, _curve):
        self.sigEdgeClicked.emit(self.edge)


class View2D(QWidget):
    """2-D scatter of lattice points over a grid.

    In edit mode, exposes ``selected_index`` and emits:
    - ``pointSelected(int)`` on click
    - ``pointMoveCompleted(int, ndarray, ndarray)`` on drag-release
      (old + new positions in lattice space)

    In edge mode (M1), renders triangulation edges as clickable line
    segments and emits ``edgeFlipRequested(edge, already_flipped)``
    when one is clicked. Edit mode and edge mode are mutually
    exclusive.
    """

    pointSelected       = pyqtSignal(int)
    pointMoveCompleted  = pyqtSignal(int, object, object)
    edgeFlipRequested   = pyqtSignal(object, bool)  # edge tuple, already_flipped
    # Human-readable guidance for the status bar as the user steps
    # through the select-edge → Ctrl+click-corners flip gesture.
    edgeFlipStatus      = pyqtSignal(str)
    # Tile Library: a tile was dropped on the canvas. Carries
    # (tile_name, canonical_x, canonical_y) — the drop point already
    # mapped into lattice space. MainWindow turns it into a compose
    # command.
    tileDropRequested   = pyqtSignal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setAspectLocked(True)
        self.plot.setBackground("w")
        self.plot.setLabel("bottom", "x")
        self.plot.setLabel("left", "y")

        self._scatter = DraggablePointsItem(
            size=_NORMAL_SIZE,
            brush=_NORMAL_BRUSH,
            pen=_NORMAL_PEN,
        )
        self._scatter.sigPointClicked.connect(self._on_point_clicked)
        self._scatter.sigPointDragStart.connect(self._on_drag_start)
        self._scatter.sigPointDragLive.connect(self._on_drag_live)
        self._scatter.sigPointDragFinish.connect(self._on_drag_finish)
        self._scatter.sigHoverChanged.connect(self._on_hover_changed)
        self._scatter.sigCornerClicked.connect(self._on_corner_clicked)
        self._scatter.sigCornerHoverChanged.connect(
            self._on_corner_hover_changed)

        # Accept tile drags from the Library panel (see dropEvent).
        self.setAcceptDrops(True)

        self.plot.addItem(self._scatter)
        # Keep the node scatter above the triangulation edge items in
        # Z-order. A Ctrl+click that lands on a node is then tried
        # against the scatter first — so the node wins the click over
        # any edge running under it (see DraggablePointsItem.mouseClickEvent).
        self._scatter.setZValue(10)
        layout.addWidget(self.plot)

        # Mode-11 bipartite polygons live below the nodes/edges. Filled
        # polygons + hinge-bar segments are transient QGraphicsItems
        # rebuilt each refresh; the hinge dots are one reusable scatter.
        self._bipartite_items: list = []
        self._hinge_scatter = pg.ScatterPlotItem(
            size=_BIP_HINGE_SIZE, brush=_BIP_HINGE_BRUSH,
            pen=pg.mkPen(None))
        self._hinge_scatter.setZValue(-3)
        self.plot.addItem(self._hinge_scatter)

        self._lattice         = None
        self._edit_mode       = False
        self._edge_mode       = False
        self._edge_items: list[_EdgeItem] = []
        self.selected_index   = -1
        self._hover_index     = -1
        self._drag_old_pos    = None  # 2-vector captured at drag start

        # ---- edge-flip gesture state -------------------------------------
        # The flip is a two-step gesture: (1) click a flippable / flipped
        # edge to select it (it turns green), (2) Ctrl+click both of the
        # quad's apex corners (highlighted amber) to confirm. State here
        # tracks where the user is in that gesture.
        self._edge_selected:  tuple[int, int] | None = None
        self._edge_apexes:    tuple[int, int] | None = None
        self._picked_corners: set[int] = set()
        # Node currently under a Ctrl+hover — rendered enlarged so the
        # confirming Ctrl+click has a generous target. -1 when none.
        self._ctrl_hover_corner: int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_lattice(self, lattice) -> None:
        self._lattice = lattice
        if self.selected_index >= len(lattice.points):
            self.selected_index = -1
        if self._hover_index >= len(lattice.points):
            self._hover_index = -1
        # 3D modes can't show edges meaningfully; force-exit edge mode
        # if the lattice switched into one. The toolbar enable-state
        # is also kept consistent in MainWindow.
        if self._edge_mode and lattice is not None and lattice.mode in (3, 6, 9):
            self._edge_mode = False
        # A lattice update means the triangulation may have changed
        # (e.g. right after an edge flip lands) — drop any in-progress
        # edge selection so stale vertex indices don't linger.
        self._clear_edge_selection()
        self._refresh_visuals()
        self._refresh_edges()
        self._refresh_bipartite()
        self._auto_range()

    def set_edit_mode(self, on: bool) -> None:
        self._edit_mode = bool(on)
        self._scatter.setEditEnabled(self._edit_mode)
        if self._edit_mode:
            # Edit and edge modes are mutually exclusive.
            self._edge_mode = False
            self._scatter.setCornerPickEnabled(False)
            self._clear_edge_selection()
        else:
            self.selected_index = -1
            self._hover_index = -1
        self._refresh_visuals()
        self._refresh_edges()

    def set_edge_mode(self, on: bool) -> None:
        """Toggle the per-edge Delaunay flip mode.

        When ``on`` is True, triangulation edges are rendered as
        clickable line segments. Flipping is a two-step gesture:

        1. Click a blue (flippable) or red (flipped) edge — it turns
           **green** to show it's selected, and the two corners a flip
           would connect light up **amber**.
        2. Hold Ctrl and click both amber corners. Once both are
           confirmed the view fires ``edgeFlipRequested`` with the edge
           tuple and its current flipped state.

        Clicking the green edge again deselects it; clicking a
        different edge switches the selection. Edit and edge modes are
        mutually exclusive; entering edge mode silently exits edit mode.
        """
        self._edge_mode = bool(on)
        if self._edge_mode:
            self._edit_mode = False
            self._scatter.setEditEnabled(False)
            self.selected_index = -1
            self._hover_index = -1
            self._scatter.setCornerPickEnabled(True)
        else:
            self._scatter.setCornerPickEnabled(False)
            self._clear_edge_selection()
        self._refresh_visuals()
        self._refresh_edges()

    def _clear_edge_selection(self) -> None:
        """Reset all transient edge-mode interaction state (selection,
        confirmed corners, and the Ctrl+hover target) back to idle."""
        self._edge_selected   = None
        self._edge_apexes     = None
        self._picked_corners  = set()
        self._ctrl_hover_corner = -1

    @property
    def edit_mode(self) -> bool:
        return self._edit_mode

    @property
    def edge_mode(self) -> bool:
        return self._edge_mode

    # ------------------------------------------------------------------
    # Scatter handlers
    # ------------------------------------------------------------------

    def _on_point_clicked(self, idx: int) -> None:
        self.selected_index = idx
        self._refresh_visuals()
        self.pointSelected.emit(idx)

    def _on_drag_start(self, idx: int) -> None:
        if self._lattice is None:
            return
        # Record the exact CANONICAL starting position from the lattice
        # (not the event position) so undo restores the precise prior
        # coord regardless of the current world transform.
        self._drag_old_pos = self._lattice.points[idx, :2].astype(float).copy()
        self.selected_index = idx
        self._refresh_visuals()

    def _on_drag_live(self, idx: int, x: float, y: float, snap: bool) -> None:
        if snap:
            x, y = _snap(x), _snap(y)
        # Drag positions are in WORLD space (the view renders transformed
        # points). The visual override stays in world space too — we
        # don't inverse-transform until the drag finishes.
        self._refresh_visuals(drag_override=(idx, x, y))

    def _on_drag_finish(self, idx: int, x: float, y: float, snap: bool) -> None:
        if snap:
            x, y = _snap(x), _snap(y)
        # The drag end position is in WORLD space (after world_transform).
        # The lattice stores CANONICAL points, so inverse-transform here
        # before pushing the move command.
        new_canonical = self.world_to_canonical_2d(x, y)
        old = (self._drag_old_pos.copy()
               if self._drag_old_pos is not None
               else self._lattice.points[idx, :2].astype(float).copy())
        new = np.array(new_canonical, dtype=float)
        self._drag_old_pos = None
        self.pointMoveCompleted.emit(idx, old, new)

    def world_to_canonical_2d(self, x_world: float, y_world: float) -> tuple[float, float]:
        """Inverse the lattice's world_transform to recover canonical
        coordinates from a position drawn in the (transformed) 2D view.
        Used by the edit drag handler so a user dragging a rotated
        point still ends up storing the right canonical coordinate.

        Returns (x, y) — z is dropped since 2D modes store 2-vectors."""
        if self._lattice is None:
            return float(x_world), float(y_world)
        M_inv = np.linalg.inv(self._lattice.world_transform())
        p = M_inv @ np.array([x_world, y_world, 0.0, 1.0])
        return float(p[0]), float(p[1])

    # ------------------------------------------------------------------
    # Tile Library drag-and-drop (drop target)
    # ------------------------------------------------------------------

    def _has_tile_payload(self, event) -> bool:
        from .library_panel import TILE_MIME
        md = event.mimeData()
        return md is not None and md.hasFormat(TILE_MIME)

    def dragEnterEvent(self, event) -> None:   # noqa: N802 (Qt override)
        if self._has_tile_payload(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:    # noqa: N802 (Qt override)
        if self._has_tile_payload(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:        # noqa: N802 (Qt override)
        """A Library tile was dropped on the canvas. Map the drop point
        from widget → scene → data → canonical lattice coords and emit
        ``tileDropRequested`` so MainWindow can compose the tile there."""
        from .library_panel import TILE_MIME
        if not self._has_tile_payload(event):
            event.ignore()
            return
        name = bytes(event.mimeData().data(TILE_MIME)).decode("utf-8")
        try:
            # QDropEvent.position() (Qt6) is in this widget's coords.
            pos = event.position().toPoint()
            plot_pt = self.plot.mapFrom(self, pos)
            scene_pt = self.plot.mapToScene(plot_pt)
            view_pt = self.plot.getPlotItem().vb.mapSceneToView(scene_pt)
            cx, cy = self.world_to_canonical_2d(view_pt.x(), view_pt.y())
        except Exception:
            event.ignore()
            return
        self.tileDropRequested.emit(name, float(cx), float(cy))
        event.acceptProposedAction()

    def _on_hover_changed(self, idx: int) -> None:
        if idx == self._hover_index:
            return
        self._hover_index = idx
        self._refresh_visuals()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh_visuals(self, drag_override=None) -> None:
        """Re-set the scatter data with current per-point styling.

        Points are rendered in WORLD space (after ``world_transform``)
        per SPEC §6 — so a rotated lattice actually appears rotated.
        Edit-mode drags record canonical coordinates by inverse-
        transforming the drag end position; see ``_on_drag_finish``.

        ``drag_override`` is ``(idx, x, y)`` when a drag is in progress —
        the dragged point's position (in world coords) is overridden
        visually only and the lattice is left untouched until
        ``_on_drag_finish``."""
        if self._lattice is None:
            return
        # Use transformed_points so rotation is visible.
        pts = np.asarray(self._lattice.transformed_points(), dtype=float)
        n = len(pts)
        if n == 0:
            self._scatter.setData([], [])
            return

        xs = pts[:, 0].astype(float).copy()
        ys = pts[:, 1].astype(float).copy() if pts.shape[1] > 1 else np.zeros(n)

        sizes   = np.full(n, _NORMAL_SIZE)
        brushes = [_NORMAL_BRUSH] * n

        if drag_override is not None:
            di, dx, dy = drag_override
            if 0 <= di < n:
                xs[di] = dx
                ys[di] = dy
                sizes[di]   = _SELECTED_SIZE
                brushes[di] = _SELECTED_BRUSH
        elif self._edit_mode:
            if 0 <= self._hover_index < n and self._hover_index != self.selected_index:
                sizes[self._hover_index]   = _HOVER_SIZE
                brushes[self._hover_index] = _HOVER_BRUSH
            if 0 <= self.selected_index < n:
                sizes[self.selected_index]   = _SELECTED_SIZE
                brushes[self.selected_index] = _SELECTED_BRUSH
        elif self._edge_mode:
            # When an edge is selected, highlight the two corners a flip
            # would connect. Amber = candidate (Ctrl+click target),
            # green = already confirmed by a Ctrl+click.
            if self._edge_apexes is not None:
                for ci in self._edge_apexes:
                    if 0 <= ci < n:
                        if ci in self._picked_corners:
                            sizes[ci]   = _SELECTED_SIZE
                            brushes[ci] = _CORNER_PICKED_BRUSH
                        else:
                            sizes[ci]   = _HOVER_SIZE
                            brushes[ci] = _CORNER_CANDIDATE_BRUSH
            # A node under a Ctrl+hover inflates into a fat click
            # target. Its colour is left as-is (amber apex / green
            # picked / normal) — the size jump alone signals "this is
            # what your Ctrl+click will hit".
            ch = self._ctrl_hover_corner
            if 0 <= ch < n:
                sizes[ch] = _CTRL_HOVER_SIZE

        self._scatter.setData(
            x=xs, y=ys, size=sizes, brush=brushes, pen=_NORMAL_PEN,
        )

    def _refresh_edges(self) -> None:
        """Tear down old ``_EdgeItem``s and rebuild them from the
        current triangulation. Cheap enough to call on every refresh
        because typical lattices have ≤300 edges."""
        for item in self._edge_items:
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self._edge_items.clear()

        if self._lattice is None or not self._edge_mode:
            return
        # Edge mode only renders for 2D (and 2.5D, which stores 2-vec
        # points internally). 3D modes have no per-edge concept.
        pts_canon = np.asarray(self._lattice.points, dtype=float)
        if pts_canon.ndim != 2 or pts_canon.shape[1] != 2:
            return

        pts_world = np.asarray(self._lattice.transformed_points(), dtype=float)
        try:
            flippable = set(_geom.flippable_edges(self._lattice.tri, pts_canon))
        except Exception:
            flippable = set()
        flipped_now = set(self._lattice.edge_flips)

        seen: set[tuple[int, int]] = set()
        simplices = np.asarray(self._lattice.tri.simplices)
        for simplex in simplices:
            verts = [int(v) for v in simplex]
            for k in range(3):
                a, b = verts[k], verts[(k + 1) % 3]
                if a > b: a, b = b, a
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                item = _EdgeItem(
                    (a, b),
                    flipped   = (a, b) in flipped_now,
                    flippable = (a, b) in flippable,
                    selected  = (self._edge_selected == (a, b)),
                )
                item.setData(
                    x=[pts_world[a, 0], pts_world[b, 0]],
                    y=[pts_world[a, 1], pts_world[b, 1]],
                )
                item.sigEdgeClicked.connect(self._on_edge_clicked)
                self.plot.addItem(item)
                self._edge_items.append(item)

    def _refresh_bipartite(self) -> None:
        """Render the mode-11 bipartite polygon network beneath the nodes.

        Set-A (corner) polygons are blue, set-B (centroid) polygons green
        — the antiferromagnet analogy from Acuna et al. 2022. By the
        strict paper recipe, polygons on the lattice boundary are
        degenerate: a degree-2 corner draws as a hinge-bar segment and a
        degree-1 corner contributes only its hinge dot. Shared polygon
        vertices (the hinges) are dotted on top.

        Polygon vertices are pushed through ``Lattice.transform_points``
        so the network tracks the node scatter under a rigid rotation.
        No-ops (and clears) for every non-bipartite mode."""
        vb = self.plot.getViewBox()
        for item in self._bipartite_items:
            try:
                vb.removeItem(item)
            except Exception:
                pass
        self._bipartite_items.clear()

        lat = self._lattice
        if lat is None or getattr(lat, "mode", None) not in (11,):
            self._hinge_scatter.setData([], [])
            return

        try:
            net = lat.build_bipartite()
        except Exception:
            # A transient bad triangulation (e.g. mid-edit collinear
            # points) shouldn't crash the view — just clear and bail.
            self._hinge_scatter.setData([], [])
            return

        for poly in net.polygons:
            if poly.degree == 0:
                continue
            vw = np.asarray(
                lat.transform_points(np.asarray(poly.vertices, float)),
                dtype=float)

            if poly.degree >= 3:
                qpoly = QPolygonF([QPointF(float(x), float(y))
                                   for x, y in vw])
                item = QGraphicsPolygonItem(qpoly)
                if poly.set_label == 'A':
                    item.setBrush(_BIP_SET_A_BRUSH)
                    item.setPen(_BIP_SET_A_PEN)
                else:
                    item.setBrush(_BIP_SET_B_BRUSH)
                    item.setPen(_BIP_SET_B_PEN)
                item.setZValue(-10)
                vb.addItem(item)
                self._bipartite_items.append(item)

                # Highlight the kite's two perpendicular inner edges blue.
                for edge in poly.inner_edges():
                    ew = np.asarray(lat.transform_points(edge), dtype=float)
                    seg = pg.PlotCurveItem(
                        x=ew[:, 0], y=ew[:, 1], pen=_BIP_INNER_EDGE_PEN)
                    seg.setZValue(-4)
                    vb.addItem(seg)
                    self._bipartite_items.append(seg)
            elif poly.degree == 2:
                seg = pg.PlotCurveItem(
                    x=[vw[0, 0], vw[1, 0]],
                    y=[vw[0, 1], vw[1, 1]],
                    pen=_BIP_HINGE_BAR_PEN)
                seg.setZValue(-5)
                vb.addItem(seg)
                self._bipartite_items.append(seg)

        # Purple bonds connecting adjacent kites along each triangle edge.
        for bond in getattr(net, "bonds", ()):
            bw = np.asarray(lat.transform_points(np.asarray(bond, float)),
                            dtype=float)
            seg = pg.PlotCurveItem(
                x=bw[:, 0], y=bw[:, 1], pen=_BIP_BOND_PEN)
            seg.setZValue(-6)
            vb.addItem(seg)
            self._bipartite_items.append(seg)

        # Hinge dots: the shared centroid hinges (central-polygon
        # vertices), where the corner kites pivot against the central
        # polygon — the black dots in the target tile.
        hinges = np.asarray(net.hinges, dtype=float)
        if hinges.size:
            hw = np.asarray(lat.transform_points(hinges), dtype=float)
            self._hinge_scatter.setData(x=hw[:, 0], y=hw[:, 1])
        else:
            self._hinge_scatter.setData([], [])

    def _on_edge_clicked(self, edge) -> None:
        """Step 1 of the flip gesture: select the clicked edge.

        Clicking the already-selected (green) edge deselects it;
        clicking any other flippable / flipped edge switches the
        selection to it. Selecting an edge computes the two apex
        corners a flip would connect and highlights them amber for the
        Ctrl+click confirmation step.
        """
        if self._lattice is None:
            return
        a, b = sorted((int(edge[0]), int(edge[1])))

        # Clicking the selected edge again cancels the gesture.
        if self._edge_selected == (a, b):
            self._clear_edge_selection()
            self._refresh_edges()
            self._refresh_visuals()
            self.edgeFlipStatus.emit("Edge deselected.")
            return

        apexes = _geom.edge_flip_apexes(self._lattice.tri, (a, b))
        if apexes is None:
            self.edgeFlipStatus.emit("That edge can't be flipped.")
            return

        self._edge_selected  = (a, b)
        self._edge_apexes    = apexes
        self._picked_corners = set()
        self._refresh_edges()
        self._refresh_visuals()
        self.edgeFlipStatus.emit(
            f"Edge {a}-{b} selected — hold Ctrl and click the two amber "
            f"corners to flip it."
        )

    def _on_corner_clicked(self, idx: int, ctrl_held: bool) -> None:
        """Step 2 of the flip gesture: Ctrl+click the apex corners.

        Only Ctrl+clicks on the two amber apex corners count. Once both
        have been confirmed the view fires ``edgeFlipRequested`` for the
        selected edge and resets the gesture.
        """
        if self._lattice is None or not self._edge_mode:
            return
        if self._edge_selected is None or self._edge_apexes is None:
            self.edgeFlipStatus.emit(
                "Select an edge first — click a blue or red line.")
            return
        if not ctrl_held:
            self.edgeFlipStatus.emit(
                "Hold Ctrl while clicking an amber corner to confirm "
                "the flip.")
            return
        idx = int(idx)
        if idx not in self._edge_apexes:
            self.edgeFlipStatus.emit(
                "That corner isn't part of the selected edge — Ctrl+click "
                "one of the two amber corners.")
            return

        self._picked_corners.add(idx)
        self._refresh_visuals()

        if set(self._picked_corners) >= set(self._edge_apexes):
            # Both apex corners confirmed — perform the flip.
            edge = self._edge_selected
            already_flipped = edge in self._lattice.edge_flips
            self._clear_edge_selection()
            self.edgeFlipStatus.emit(
                f"Flipping edge {edge[0]}-{edge[1]}…")
            self.edgeFlipRequested.emit(edge, already_flipped)
        else:
            self.edgeFlipStatus.emit(
                "One corner confirmed — Ctrl+click the other amber "
                "corner to flip the edge.")

    def _on_corner_hover_changed(self, idx: int) -> None:
        """The scatter reports the node under a Ctrl+hover (``-1`` when
        none). Enlarge it so the confirming Ctrl+click has a generous
        target and can't slip onto an adjacent edge."""
        idx = int(idx)
        if idx == self._ctrl_hover_corner:
            return
        self._ctrl_hover_corner = idx
        self._refresh_visuals()

    def _auto_range(self) -> None:
        if self._lattice is None:
            return
        # Use transformed_points so the view fits the lattice in its
        # current oriented frame.
        pts = self._lattice.transformed_points()
        if len(pts) == 0:
            return
        xs = pts[:, 0]
        ys = pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts))
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        pad = 0.1 * max(x_max - x_min, y_max - y_min, 1e-3)
        self.plot.setXRange(x_min - pad, x_max + pad, padding=0)
        self.plot.setYRange(y_min - pad, y_max + pad, padding=0)


class View3D(QWidget):
    """3-D viewer that loads the STL mesh emitted by ``Lattice.to_stl()``.

    Rotation handling: per SPEC §6, the rigid rotation is applied to the
    actor via ``vtkActor.SetUserTransform()`` — NOT baked into the mesh.
    Baking would force a full STL re-export on every rotation change
    and kill interactivity. The STL on disk is rendered in canonical
    frame; the actor's user transform applies world_transform on top.

    Note: ``Lattice.to_stl()`` itself applies world_transform to the
    written file too (per SPEC §9 export policy), so we save the STL
    with rotation TEMPORARILY suppressed (saving canonical points) and
    rotate the actor afterward. This keeps the viewer interactive while
    still letting File→Export emit oriented geometry.

    The VTK render-window initialisation that backs ``QtInteractor`` can
    fail under the headless Qt platforms used in CI/tests (e.g. an access
    violation on Windows when ``QT_QPA_PLATFORM=offscreen``). When that
    happens the viewer falls back to an empty placeholder — the rest of
    the shell still works, just without 3D rendering.
    """

    # Emitted when the user left-clicks a point on a rendered surface
    # (surface-point picking). Carries the picked world-space point as a
    # length-3 ``np.ndarray``, or ``None`` if the click hit empty space.
    # The SimulationPanel resolves the point to the nearest polygon tile
    # to drive the "anchor view to a polygon" feature.
    surfacePointPicked = pyqtSignal(object)

    # Emitted on a **Ctrl**+left-click on a rendered surface (task 6c).
    # Carries the picked world-space point (length-3 ndarray) or ``None`` on
    # a miss. MainWindow maps it to the lattice triangle and shows that
    # triangle's generalized Poisson ratio.
    trianglePoissonPicked = pyqtSignal(object)

    def __init__(self, parent=None, *, force_placeholder: bool = False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.interactor = None
        self._mesh_actor = None
        self._tmp_stl_path = os.path.join(
            tempfile.gettempdir(), "auxetic_studio_view3d.stl"
        )
        # Cached lattice ref so ``clear_pose`` can fall back to the
        # default ``Lattice.to_stl()`` rendering without the caller
        # having to re-supply the lattice.
        self._cached_lattice = None
        # Test-friendly tap: record the most recent show_pose call
        # arguments so a headless test can spy without monkeypatching.
        self.last_show_pose_args: tuple | None = None
        # State flag — True between show_pose and clear_pose.
        self._pose_view_active = False
        # M3-polish: list of VTK actors holding force-arrow glyphs.
        # Replaced wholesale on each ``set_force_glyphs`` call.
        self._force_actors: list = []
        # Test-friendly tap mirroring last_show_pose_args.
        self.last_force_glyphs: list | None = None
        # Piston-mode visualisation: ground + piston plate actors.
        self._piston_actors: list = []
        # Test-friendly tap — records the most recent piston-viz call.
        self.last_piston_visualization: dict | None = None
        # "Anchor view to a polygon": outline actor ringing the anchored
        # tile, and a test-friendly tap recording the last highlight call.
        self._anchor_actor = None
        self.last_anchor_highlight = None
        # Poisson-tracking overlay (task 6a): bbox wireframes + per-axis
        # extreme points for the rest ("initial") and current ("final",
        # compressed) poses. Test-friendly tap records the last call.
        self._poisson_actors: list = []
        self.last_poisson_tracking: dict | None = None

        # Track whether the orientation/widget extras are active so
        # tests / external callers can introspect.
        self.has_grid           = False
        self.has_axes           = False
        self.has_view_cube      = False
        self._view_cube_widget  = None

        # Zoom-to-cursor wheel observers. Tags + raw interactor handle are
        # kept so the wiring can be introspected / torn down; populated by
        # ``_install_zoom_to_cursor`` and left empty when headless.
        self.has_zoom_to_cursor   = False
        self._zoom_observer_tags: list[int] = []
        self._zoom_iren           = None

        if force_placeholder or not _PYVISTAQT_AVAILABLE:
            return

        try:
            self.interactor = QtInteractor(self)
            layout.addWidget(self.interactor.interactor)
        except Exception:
            # VTK / Qt platform mismatch — leave the widget empty.
            self.interactor = None

        if self.interactor is not None:
            self._install_3d_navigation_aids()
            self._enable_tile_picking()
            self._install_zoom_to_cursor()

    # ------------------------------------------------------------------
    # Surface-point picking (anchor-view-to-polygon)
    # ------------------------------------------------------------------

    def _enable_tile_picking(self) -> None:
        """Enable left-click surface-point picking so the user can click a
        rendered polygon to anchor the view to it. The pick fires on a
        left click (drag still orbits the camera); the resolved point is
        emitted via ``surfacePointPicked`` for the panel to map to a tile.

        Guarded — picking relies on VTK observers that vary by version and
        can be absent under headless platforms. A failure here just leaves
        picking off; the rest of the viewer is unaffected."""
        try:
            self.interactor.enable_surface_point_picking(
                callback=self._on_surface_pick,
                show_message=False,
                show_point=False,
                left_clicking=True,
                clear_on_no_selection=False,
            )
        except Exception:
            pass

    def _on_surface_pick(self, *args) -> None:
        """Picking callback. PyVista passes the picked point; we forward a
        clean length-3 point (or ``None`` on a miss). A plain left-click
        drives the anchor feature (``surfacePointPicked``); a Ctrl+left-click
        instead requests the per-triangle Poisson readout
        (``trianglePoissonPicked``, task 6c). The held modifier is read from
        ``QApplication`` at pick time since the VTK callback doesn't carry it."""
        try:
            from PyQt6.QtWidgets import QApplication
            point = args[0] if args else None
            pt = None
            if point is not None:
                arr = np.asarray(point, dtype=float).ravel()
                if arr.size >= 3 and bool(np.all(np.isfinite(arr[:3]))):
                    pt = arr[:3].copy()
            ctrl = bool(QApplication.keyboardModifiers()
                        & Qt.KeyboardModifier.ControlModifier)
            if ctrl:
                self.trianglePoissonPicked.emit(pt)
            else:
                self.surfacePointPicked.emit(pt)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Zoom-to-cursor (mouse-wheel dolly toward the point under the cursor)
    # ------------------------------------------------------------------

    def _install_zoom_to_cursor(self) -> None:
        """Replace the default centred wheel-zoom with one that dollies
        toward the point under the cursor.

        Wired as **high-priority** observers (10.0) on the raw VTK
        interactor for the wheel events, each of which aborts the
        invocation (``vtkCommand.SetAbortFlag``) so the trackball style's
        built-in centred dolly (priority 0.0) never runs. Fully guarded:
        any failure (or a headless interactor) just leaves the default
        zoom in place rather than taking down the viewer."""
        if self.interactor is None:
            return
        try:
            iren = self.interactor.iren.interactor  # raw vtkRenderWindowInteractor
        except Exception:
            return
        if iren is None:
            return
        try:
            t_fwd = iren.AddObserver(
                "MouseWheelForwardEvent", self._on_wheel_forward, 10.0)
            t_bwd = iren.AddObserver(
                "MouseWheelBackwardEvent", self._on_wheel_backward, 10.0)
        except Exception:
            return
        self._zoom_iren = iren
        self._zoom_observer_tags = [t_fwd, t_bwd]
        self.has_zoom_to_cursor = True

    def _on_wheel_forward(self, caller, event) -> None:
        """Wheel-forward = zoom in (camera moves toward the cursor point)."""
        self._zoom_to_cursor(1.0 + _ZOOM_WHEEL_STEP)
        self._abort_event(caller)

    def _on_wheel_backward(self, caller, event) -> None:
        """Wheel-backward = zoom out (camera moves away from the cursor point)."""
        self._zoom_to_cursor(1.0 / (1.0 + _ZOOM_WHEEL_STEP))
        self._abort_event(caller)

    def _abort_event(self, caller) -> None:
        """Set the abort flag on this object's wheel observers so the
        trackball style's lower-priority centred-dolly handler is skipped
        for the current wheel event."""
        try:
            for tag in self._zoom_observer_tags:
                cmd = caller.GetCommand(tag)
                if cmd is not None:
                    cmd.SetAbortFlag(1)
        except Exception:
            pass

    def _cursor_world_point(self, renderer, camera):
        """World-space point under the cursor, projected onto the camera's
        focal plane. Returns the focal point itself when the conversion is
        unavailable (so a centred dolly is the safe fallback)."""
        focal = camera.GetFocalPoint()
        try:
            x, y = self.interactor.iren.interactor.GetEventPosition()
            # Display-space depth of the focal plane, then read the world
            # point at the cursor (x, y) at that same depth.
            renderer.SetWorldPoint(focal[0], focal[1], focal[2], 1.0)
            renderer.WorldToDisplay()
            depth = renderer.GetDisplayPoint()[2]
            renderer.SetDisplayPoint(float(x), float(y), depth)
            renderer.DisplayToWorld()
            w = renderer.GetWorldPoint()
            if abs(w[3]) > 1e-12:
                return (w[0] / w[3], w[1] / w[3], w[2] / w[3])
        except Exception:
            pass
        return focal

    def _zoom_to_cursor(self, factor: float) -> None:
        """Dolly the camera toward (or away from) the point under the
        cursor by ``factor`` (>1 in, <1 out). Headless / failure safe."""
        if self.interactor is None:
            return
        try:
            renderer = self.interactor.renderer
            camera = renderer.GetActiveCamera()
            target = self._cursor_world_point(renderer, camera)
            new_pos, new_foc = dolly_toward_cursor(
                camera.GetPosition(), camera.GetFocalPoint(), target, factor)
            camera.SetPosition(float(new_pos[0]), float(new_pos[1]),
                               float(new_pos[2]))
            camera.SetFocalPoint(float(new_foc[0]), float(new_foc[1]),
                                 float(new_foc[2]))
            # Parallel projection ignores camera distance, so the dolly
            # alone wouldn't change apparent size — scale the parallel
            # extent too. Scaling by 1/factor keeps the cursor point fixed
            # in this mode as well (see dolly_toward_cursor docstring).
            if camera.GetParallelProjection():
                camera.SetParallelScale(camera.GetParallelScale() / float(factor))
            renderer.ResetCameraClippingRange()
            self.interactor.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3D navigation aids (Fusion-360-style grid + axes triad + view cube)
    # ------------------------------------------------------------------

    def _install_3d_navigation_aids(self) -> None:
        """Add the standard 3D-app navigation widgets to the interactor:

        - **Floor grid + bbox** (PyVista's ``show_grid``) — gives the
          user a frame of reference instead of a black void.
        - **XYZ axes triad** in the lower-left (``add_axes``) — small
          coloured triad showing world orientation. Always visible.
        - **Clickable view cube** in the upper-right
          (``add_camera_orientation_widget``) — clickable cube faces
          snap the camera to top / front / side / etc., the same UX as
          Fusion 360 / SolidWorks / OnShape.

        Each call is wrapped in try/except because the underlying VTK
        widget classes vary by version and we don't want a missing
        widget to take down the rest of the viewer.
        """
        try:
            self.interactor.show_grid(color="grey", show_zaxis=True)
            self.has_grid = True
        except Exception:
            self.has_grid = False
        try:
            self.interactor.add_axes(
                interactive=False,
                line_width=2,
            )
            self.has_axes = True
        except Exception:
            self.has_axes = False
        # PyVista exposes the VTK camera-orientation widget via
        # ``add_camera_orientation_widget``. Some older PyVista builds
        # don't have this method — fall back gracefully.
        try:
            widget = self.interactor.add_camera_orientation_widget()
            self._view_cube_widget = widget
            self.has_view_cube = True
            # Disable the default animated transition. With it on,
            # clicking the Top / Bottom faces produces a visible swing
            # because the "up vector" is degenerate at the poles —
            # the camera path goes through an unintended intermediate
            # orientation. Most CAD apps snap instantly and that's
            # the closer match to user expectation here.
            try:
                widget.SetAnimate(False)
            except Exception:
                # Older VTK uses AnimateOff(); newer uses SetAnimate.
                try:
                    widget.AnimateOff()
                except Exception:
                    pass
        except Exception:
            self._view_cube_widget = None
            self.has_view_cube = False

    # ------------------------------------------------------------------
    # Camera presets (Iso / Fit)
    # ------------------------------------------------------------------
    #
    # These drive the CAMERA only — they don't touch the lattice's
    # rigid_rotation field. Conceptually distinct from the
    # InspectorPanel's "Top/Front/Side/Reset" buttons, which orient the
    # LATTICE in world space (and go through the undo stack). The
    # InspectorPanel buttons rotate the auxetic; these buttons rotate
    # the user's viewpoint.

    def camera_isometric(self) -> None:
        """Standard ISO 3/4 view."""
        if self.interactor is None:
            return
        try:
            self.interactor.view_isometric()
        except Exception:
            pass

    def camera_fit(self) -> None:
        """Re-frame the camera so the current scene fills the
        viewport. Useful after a big lattice change."""
        if self.interactor is None:
            return
        try:
            self.interactor.reset_camera()
        except Exception:
            pass

    # ------------------------------------------------------------------

    def update_lattice(self, lattice):
        # Cache for ``clear_pose`` fall-back, regardless of whether
        # we're headless or not. Tests rely on the cache being set.
        self._cached_lattice = lattice
        if self.interactor is None:
            return
        try:
            # Save STL in canonical frame (suppress rotation), then
            # apply rotation via the actor — see class docstring.
            saved_rotation = lattice.rigid_rotation
            saved_flipped  = lattice.flipped
            try:
                from scipy.spatial.transform import Rotation as _R
                lattice.rigid_rotation = _R.identity()
                lattice.flipped = False
                lattice.to_stl(self._tmp_stl_path, verbose=False)
            finally:
                lattice.rigid_rotation = saved_rotation
                lattice.flipped = saved_flipped
            mesh = pv.read(self._tmp_stl_path)
        except Exception:
            return

        self._mesh_actor = _swap_mesh_actor(
            self.interactor, self._mesh_actor, mesh,
            color="lightsteelblue", show_edges=False, smooth_shading=True,
        )
        if self._mesh_actor is not None:
            self._apply_user_transform(lattice)
            try:
                self.interactor.reset_camera()
            except Exception:
                pass

    def _apply_user_transform(self, lattice) -> None:
        """Push ``lattice.world_transform()`` onto the mesh actor as a
        VTK user transform — applied at render time, no mesh data
        touched."""
        if self._mesh_actor is None:
            return
        try:
            import vtk
        except Exception:
            return
        M = lattice.world_transform()
        vtk_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_mat.SetElement(i, j, float(M[i, j]))
        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_mat)
        try:
            self._mesh_actor.SetUserTransform(transform)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pose-driven rendering (Stage 6c — simulation playback path)
    # ------------------------------------------------------------------

    def show_pose(self, tile_system, pose, highlight_tile=None) -> None:
        """Render the tile system's vertices transformed by ``pose``.

        Used during simulation playback to show the lattice deformed
        along the kirigami mode. ``tile_system.tiles`` is already in
        world frame (per ``TileSystem.from_lattice``'s contract), so
        we apply only the per-tile pose on top — no separate
        world-transform user-transform.

        The mesh built here goes through the same
        ``build_export_triangles`` pipeline ``Lattice.to_stl`` uses;
        hubs, struts, and joint spheres all appear deformed alongside
        the tile faces. Shape parameters (strut radius, sphere counts,
        etc.) are pulled from the cached lattice when available so the
        pose render matches the static render's appearance.

        ``highlight_tile`` (optional) is the index of the polygon the
        view is anchored to; when set, its posed outline is drawn as a
        ring above the structure so the user can see which polygon they
        clicked. The caller passes the *relativized* pose, so the
        anchored polygon sits at its rest placement (stationary) — the
        ring marks that fixed reference.

        ``last_show_pose_args`` is set to ``(tile_system, pose.copy())``
        so headless tests can verify the call without monkeypatching;
        ``last_anchor_highlight`` mirrors the anchor ring's vertices (or
        ``None``)."""
        import numpy as _np
        pose_arr = _np.asarray(pose, dtype=float)
        # If the dynamics integrator diverged before clamping (or the
        # caller passed a stale Inf/NaN pose), don't try to render —
        # VTK's grid axis label calc and Jacobi eigensolver crash on
        # non-finite bounds.
        if not _np.all(_np.isfinite(pose_arr)):
            return
        self._pose_view_active = True
        self.last_show_pose_args = (tile_system, pose_arr.copy())

        # Anchor outline verts — computed (and exposed for tests) even
        # when headless; the actual actor is only drawn below.
        anchor_verts = self._compute_anchor_verts(
            tile_system, pose_arr, highlight_tile)
        self.last_anchor_highlight = (
            None if anchor_verts is None else anchor_verts.copy())

        if self.interactor is None:
            return

        triangles = self._build_pose_mesh_triangles(
            tile_system, pose, lattice=self._cached_lattice,
        )
        if not triangles:
            self._update_anchor_outline(None)
            return

        try:
            mesh = _triangles_to_polydata(triangles)
        except Exception:
            return

        self._mesh_actor = _swap_mesh_actor(
            self.interactor, self._mesh_actor, mesh,
            color="lightsteelblue", show_edges=False, smooth_shading=True,
        )
        # Rebuild the anchor overlay without rendering, then issue a single
        # render so the mesh + ring land in the same frame (no flicker).
        self._update_anchor_outline(anchor_verts, mesh, render=False)
        try:
            self.interactor.render()
        except Exception:
            pass

    def _compute_anchor_verts(self, tile_system, pose, tile_idx):
        """Posed vertices (Nx3) of the anchored tile, or ``None``. 2D tile
        systems are lifted to z=0. Pure geometry — safe to call headless."""
        if tile_idx is None or tile_system is None:
            return None
        try:
            idx = int(tile_idx)
        except (TypeError, ValueError):
            return None
        if not (0 <= idx < tile_system.n_tiles):
            return None
        try:
            v = _apply_tile_pose(
                tile_system.tiles[idx], pose, idx, tile_system.dimension)
            v = np.asarray(v, dtype=float)
        except Exception:
            return None
        if v.shape[0] < 2:
            return None
        if tile_system.dimension == 2:
            return np.hstack([v, np.zeros((v.shape[0], 1))])
        return v

    def _update_anchor_outline(self, verts3d, mesh=None, render: bool = True) -> None:
        """Draw (or clear) the gold ring marking the anchored polygon.
        Lifted just above the rendered solids and forced to render on-top
        (depth-independent) so it's equally visible from top, bottom, and
        iso — the opaque mesh no longer occludes it when viewed from below.

        Internal actor swaps are issued with ``render=False``. Pass
        ``render=False`` from a caller (e.g. ``show_pose``) that batches a
        single render after the whole scene is rebuilt, so playback doesn't
        flicker; the default renders once on return for standalone callers."""
        if self.interactor is None:
            return
        if self._anchor_actor is not None:
            try:
                self.interactor.remove_actor(self._anchor_actor, render=False)
            except Exception:
                pass
            self._anchor_actor = None
        if verts3d is not None and len(verts3d) >= 2:
            try:
                v = np.asarray(verts3d, dtype=float).copy()
                if mesh is not None and getattr(mesh, "n_points", 0):
                    ztop = float(np.asarray(mesh.points)[:, 2].max())
                else:
                    ztop = float(v[:, 2].max())
                span = float(np.ptp(v[:, :2])) or 1.0
                v[:, 2] = ztop + 0.05 * span
                loop = np.vstack([v, v[0]])
                poly = pv.lines_from_points(loop)
                self._anchor_actor = self.interactor.add_mesh(
                    poly, color="#ffae00", line_width=6, pickable=False,
                    render_lines_as_tubes=True, render=False,
                )
                _force_actor_on_top(self._anchor_actor)
            except Exception:
                self._anchor_actor = None
        if render:
            try:
                self.interactor.render()
            except Exception:
                pass

    def clear_pose(self) -> None:
        """Drop the simulation-playback mesh and re-render the cached
        lattice via the default ``Lattice.to_stl`` path. Called by the
        SimulationPanel when the simulation is invalidated."""
        self._pose_view_active = False
        self.last_show_pose_args = None
        self._update_anchor_outline(None)
        self.last_anchor_highlight = None
        self.clear_poisson_tracking()
        if self._cached_lattice is not None:
            self.update_lattice(self._cached_lattice)

    # ------------------------------------------------------------------
    # Poisson-tracking overlay (task 6a)
    # ------------------------------------------------------------------

    def show_poisson_tracking(self, boxes) -> None:
        """Draw the Poisson-bounds overlay — up to seven extremal-pose
        axis-aligned bounding boxes: the rest ("initial"), the two most
        axially-compressed poses on the +θ and −θ halves of the sweep, and
        the four farthest-reach poses in +x / −x / +y / −y. Each box is a
        wireframe plus the per-axis extreme vertices that define it, drawn in
        the box's own colour (see :data:`POISSON_BOXES`).

        ``boxes`` maps a box key to a ``(corners, extremes, visible)`` triple:
          - ``corners``  — ``(2**dim, dim)`` bbox corner array, or ``None``
          - ``extremes`` — ``(dim, 2, dim)`` per-axis min/max vertices, or ``None``
          - ``visible``  — bool. A hidden box is neither drawn nor recorded
            (its ``last_poisson_tracking`` geometry is ``None`` — the
            headless-tap convention).

        ``last_poisson_tracking`` is recorded as ``{key: {"corners", "extremes",
        "visible"}}`` for every known key (geometry ``None`` when hidden or
        absent). All geometry is computed in ``auxetic`` (Simulator) and passed
        in — this method only renders. Headless-safe (records the tap then
        no-ops without an interactor). Unknown / missing keys are ignored."""
        rec: dict[str, dict] = {}
        for key, _label, _color in POISSON_BOXES:
            spec = boxes.get(key) if boxes else None
            if spec is None:
                corners, extremes, visible = None, None, False
            else:
                corners, extremes, visible = spec
                visible = bool(visible)
            show = visible and corners is not None
            rec[key] = {
                "corners":  np.asarray(corners, float) if show else None,
                "extremes": (np.asarray(extremes, float)
                             if (show and extremes is not None) else None),
                "visible":  visible,
            }
        self.last_poisson_tracking = rec
        if self.interactor is None:
            return
        self._clear_poisson_actors(render=False)
        try:
            for key, _label, color in POISSON_BOXES:
                entry = rec[key]
                if entry["corners"] is None:
                    continue
                # Initial is a thin grey reference frame; the actuated /
                # reach boxes draw thicker so they read on top of it; the
                # overall footprint envelope is boldest of all.
                if key == "initial":
                    width, psize = 2, 11
                elif key == "footprint":
                    width, psize = 4, 13
                else:
                    width, psize = 3, 13
                self._add_bbox_wireframe(entry["corners"], color=color, width=width)
                if entry["extremes"] is not None:
                    self._add_extreme_points(entry["extremes"], (color,), size=psize)
        except Exception:
            pass
        try:
            self.interactor.render()
        except Exception:
            pass

    def _add_bbox_wireframe(self, corners, *, color: str, width: int) -> None:
        c = np.asarray(corners, dtype=float)
        if c.ndim != 2 or c.shape[0] == 0:
            return
        dim = c.shape[1]
        lo = c.min(axis=0)
        hi = c.max(axis=0)
        if dim == 2:
            loop = np.array([
                [lo[0], lo[1], 0.0], [hi[0], lo[1], 0.0],
                [hi[0], hi[1], 0.0], [lo[0], hi[1], 0.0],
                [lo[0], lo[1], 0.0],
            ])
            mesh = pv.lines_from_points(loop)
            actor = self.interactor.add_mesh(
                mesh, color=color, line_width=width,
                render_lines_as_tubes=True, pickable=False, render=False)
        else:
            box = pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
            actor = self.interactor.add_mesh(
                box, color=color, style="wireframe", line_width=width,
                pickable=False, render=False)
        if actor is not None:
            self._poisson_actors.append(actor)

    def _add_extreme_points(self, extremes, colors, *, size: int) -> None:
        e = np.asarray(extremes, dtype=float)        # (dim, 2, dim)
        if e.ndim != 3:
            return
        for d in range(e.shape[0]):
            pts = e[d]
            if pts.shape[1] == 2:
                pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])
            actor = self.interactor.add_mesh(
                pv.PolyData(pts), color=colors[d % len(colors)],
                render_points_as_spheres=True, point_size=size,
                pickable=False, render=False)
            if actor is not None:
                self._poisson_actors.append(actor)

    def _clear_poisson_actors(self, *, render: bool = True) -> None:
        if self.interactor is None:
            self._poisson_actors = []
            return
        for actor in self._poisson_actors:
            try:
                self.interactor.remove_actor(actor, render=False)
            except Exception:
                pass
        self._poisson_actors = []
        if render:
            try:
                self.interactor.render()
            except Exception:
                pass

    def clear_poisson_tracking(self) -> None:
        """Remove the Poisson-tracking overlay (no-op if not shown)."""
        self.last_poisson_tracking = None
        self._clear_poisson_actors(render=True)

    # ------------------------------------------------------------------
    # Force-arrow glyphs (M2 polish — visual feedback for ForceVectors)
    # ------------------------------------------------------------------

    def set_force_glyphs(self, tile_system, forces) -> None:
        """Render one arrow per :class:`ForceVector` at its world
        application point. Called by the simulation panel whenever
        the force list, the lattice, or the cached tile system
        changes.

        ``forces`` is the raw list of force dicts as stored in
        ``lattice.dynamics_state['forces']``. ``tile_system`` is the
        same TileSystem used by the kinematic / dynamic solvers — its
        tile vertices are already in world frame.

        Arrow length is normalised across the active force list so
        the largest force renders at a fixed visual size and smaller
        forces are proportionally shorter; this avoids tiny invisible
        arrows when one big force dominates the dynamics.
        """
        # Always update the test-friendly tap, even when headless.
        self.last_force_glyphs = self._compute_glyph_data(tile_system, forces)
        self._clear_force_actors()
        if self.interactor is None:
            return
        if not self.last_force_glyphs:
            try:
                self.interactor.render()
            except Exception:
                pass
            return
        # Pick a render scale so the largest force is ~10% of the
        # bbox diagonal — visible but not dominating.
        scale = self._glyph_render_scale(tile_system, self.last_force_glyphs)
        try:
            for origin, direction, mag_norm in self.last_force_glyphs:
                arrow = pv.Arrow(
                    start=tuple(origin),
                    direction=tuple(direction),
                    scale=float(scale * max(0.1, mag_norm)),
                )
                actor = self.interactor.add_mesh(
                    arrow, color="crimson", show_edges=False,
                )
                self._force_actors.append(actor)
            self.interactor.render()
        except Exception:
            # If glyph rendering fails (e.g. VTK quirks), drop the
            # actors and continue without crashing the panel.
            self._clear_force_actors()

    def clear_force_glyphs(self) -> None:
        """Remove all force arrows. Used when forces are emptied or
        the panel is rebound to a lattice without forces."""
        self.last_force_glyphs = []
        self._clear_force_actors()
        if self.interactor is not None:
            try:
                self.interactor.render()
            except Exception:
                pass

    @staticmethod
    def _compute_glyph_data(tile_system, forces) -> list:
        """Pure helper, no Qt / VTK touching — returns
        ``[(origin_xyz, dir_xyz, mag_normalised), ...]``. Used as the
        test-friendly tap and consumed by the actor-creation path."""
        import numpy as _np
        if not forces or tile_system is None:
            return []
        # Extract magnitudes for normalisation (keep raw too for the
        # "no scaling" branch when only one non-zero force exists).
        raw_mags = [
            float(f.get("magnitude", 0.0)) for f in forces
        ]
        max_mag = max((abs(m) for m in raw_mags), default=1.0)
        if max_mag < 1e-12:
            max_mag = 1.0
        out = []
        n_tiles = tile_system.n_tiles
        dim = tile_system.dimension
        for f, mag in zip(forces, raw_mags):
            tile_idx = int(f.get("tile_index", -1))
            if not (0 <= tile_idx < n_tiles):
                continue
            tile_verts = _np.asarray(tile_system.tiles[tile_idx], dtype=float)
            kind = str(f.get("location_kind", "tile_centroid"))
            if kind == "tile_vertex":
                v_idx = int(f.get("vert_index", -1))
                if not (0 <= v_idx < tile_verts.shape[0]):
                    continue
                origin = tile_verts[v_idx]
            else:   # tile_centroid (default) or anything else falls back
                origin = tile_verts.mean(axis=0)
            # Direction (3D padded for 2D tiles)
            d = list(f.get("direction") or [1.0, 0.0, 0.0])
            while len(d) < 3:
                d.append(0.0)
            origin3 = _np.zeros(3, dtype=float)
            origin3[: min(3, dim)] = origin[: min(3, dim)]
            direction3 = _np.asarray(d[:3], dtype=float)
            n = float(_np.linalg.norm(direction3))
            if n < 1e-12:
                continue
            direction3 = direction3 / n
            mag_norm = float(abs(mag) / max_mag)
            out.append((origin3, direction3, mag_norm))
        return out

    @staticmethod
    def _glyph_render_scale(tile_system, glyph_data) -> float:
        """Pick a per-arrow length scale (for ``pv.Arrow(scale=...)``)
        that renders the largest force at ~10% of the lattice's bbox
        diagonal."""
        import numpy as _np
        try:
            all_v = _np.vstack(tile_system.tiles)
            if all_v.shape[1] == 2:
                all_v = _np.hstack([all_v, _np.zeros((all_v.shape[0], 1))])
            bbox = all_v.max(axis=0) - all_v.min(axis=0)
            diag = float(_np.linalg.norm(bbox))
            return max(diag * 0.10, 0.05)
        except Exception:
            return 0.1

    def _clear_force_actors(self) -> None:
        if self.interactor is None:
            self._force_actors.clear()
            return
        for actor in self._force_actors:
            try:
                self.interactor.remove_actor(actor)
            except Exception:
                pass
        self._force_actors.clear()

    # ------------------------------------------------------------------
    # Piston-mode visualisation (ground plate + moving piston plate)
    # ------------------------------------------------------------------

    def set_piston_visualization(self,
                                  tile_system,
                                  current_pose,
                                  *,
                                  initial_pose=None,
                                  axis_idx: int = 1) -> None:
        """Render two thin slabs flanking the lattice along the piston
        axis:

        - **Ground plate** at the *initial-pose* lattice bottom — a
          fixed reference for the user's eye showing where the bottom
          slab is anchored.
        - **Piston plate** at the *current-pose* lattice top — moves
          downward as the user scrubs through the dynamic trajectory,
          visualising how far the piston has pressed in.

        Both plates align with the **world axes** (so a pre-rotated
        lattice doesn't tilt them), extend 1.2× the lattice's lateral
        extent so they fully cover the deforming structure, and are
        thin in the piston axis (5% of the bbox diagonal).

        ``initial_pose=None`` falls back to ``current_pose`` (no
        compression depicted — both plates touch the lattice).
        ``axis_idx`` matches the piston axis the dynamics builder used
        (default 1 = world-Y, matching ``_piston_setup``).
        """
        info = self._compute_piston_data(
            tile_system, current_pose,
            initial_pose=initial_pose, axis_idx=axis_idx,
        )
        self.last_piston_visualization = info
        self._clear_piston_actors()
        if self.interactor is None or info is None:
            return
        try:
            ground_mesh = pv.Cube(
                center=tuple(info["ground_center"]),
                x_length=float(info["plate_size"][0]),
                y_length=float(info["plate_size"][1]),
                z_length=float(info["plate_size"][2]),
            )
            piston_mesh = pv.Cube(
                center=tuple(info["piston_center"]),
                x_length=float(info["plate_size"][0]),
                y_length=float(info["plate_size"][1]),
                z_length=float(info["plate_size"][2]),
            )
            ground_actor = self.interactor.add_mesh(
                ground_mesh, color="#5a5a5a", opacity=0.6,
                show_edges=True, edge_color="#1a1a1a",
            )
            piston_actor = self.interactor.add_mesh(
                piston_mesh, color="#a0a0a8", opacity=0.7,
                show_edges=True, edge_color="#303030",
                metallic=True, specular=0.7,
            )
            self._piston_actors.extend([ground_actor, piston_actor])
            self.interactor.render()
        except Exception:
            self._clear_piston_actors()

    def clear_piston_visualization(self) -> None:
        """Drop the ground + piston actors."""
        self.last_piston_visualization = None
        self._clear_piston_actors()
        if self.interactor is not None:
            try:
                self.interactor.render()
            except Exception:
                pass

    @staticmethod
    def _compute_piston_data(tile_system, current_pose,
                              *, initial_pose=None, axis_idx: int = 1
                              ) -> dict | None:
        """Compute ground and piston plate centres + size from poses.
        Pure helper so headless tests can verify the geometry without
        any VTK dependency."""
        import numpy as _np
        if tile_system is None or tile_system.n_tiles == 0:
            return None
        cur_verts = _world_verts_3d(tile_system, current_pose)
        if cur_verts.shape[0] == 0:
            return None
        # Bail out on non-finite verts. Caller may pass a divergent
        # dynamics pose; rendering an Inf-sized cube blows up VTK's
        # axis labelling (vtkAxisActor: Number of labels = INT_MIN).
        if not _np.all(_np.isfinite(cur_verts)):
            return None
        ref_verts = (cur_verts if initial_pose is None
                     else _world_verts_3d(tile_system, initial_pose))
        if not _np.all(_np.isfinite(ref_verts)):
            return None

        # Lateral axes are everything except the piston axis. We work
        # in 3D throughout (2D lattices are padded to z=0 in
        # ``_world_verts_3d``).
        ax = int(axis_idx)
        if ax not in (0, 1, 2):
            ax = 1
        lat = [i for i in range(3) if i != ax]

        bbox_min = cur_verts.min(axis=0)
        bbox_max = cur_verts.max(axis=0)
        bbox_size = bbox_max - bbox_min
        diag = float(_np.linalg.norm(bbox_size))
        thickness = max(diag * 0.05, 0.02)

        # Ground anchored at the INITIAL bottom — doesn't move.
        ground_axis_y = float(ref_verts[:, ax].min())
        # Piston tracks the CURRENT top — moves with compression.
        piston_axis_y = float(cur_verts[:, ax].max())

        center_lat = (bbox_min[lat] + bbox_max[lat]) / 2.0
        ground_center = _np.zeros(3)
        ground_center[lat[0]] = center_lat[0]
        ground_center[lat[1]] = center_lat[1]
        ground_center[ax]     = ground_axis_y - thickness / 2.0
        piston_center = _np.zeros(3)
        piston_center[lat[0]] = center_lat[0]
        piston_center[lat[1]] = center_lat[1]
        piston_center[ax]     = piston_axis_y + thickness / 2.0

        # Plate size: 1.2x the lateral extents in the perpendicular
        # axes; thickness along the piston axis.
        size = _np.zeros(3)
        size[lat[0]] = max(float(bbox_size[lat[0]]) * 1.2, thickness)
        size[lat[1]] = max(float(bbox_size[lat[1]]) * 1.2, thickness)
        size[ax]     = thickness

        return {
            "ground_center":  ground_center,
            "piston_center":  piston_center,
            "plate_size":     size,
            "axis_idx":       ax,
            "ground_axis_y":  ground_axis_y,
            "piston_axis_y":  piston_axis_y,
        }

    def _clear_piston_actors(self) -> None:
        if self.interactor is None:
            self._piston_actors.clear()
            return
        for actor in self._piston_actors:
            try:
                self.interactor.remove_actor(actor)
            except Exception:
                pass
        self._piston_actors.clear()

    @staticmethod
    def _build_pose_mesh_triangles(tile_system, pose, *, lattice=None):
        """Apply ``pose`` to ``tile_system`` and return the full deformed
        lattice mesh — tile solids, hub solids, strut tubes, and joint
        spheres — as a flat list of 3-vertex triangles.

        The pipeline mirrors ``Lattice.to_stl``: pose-transformed tile
        vertices feed into ``collect_export_geometry_from_posed_tiles``
        (which dispatches per ``tile_source[i]['type']`` and detects
        struts via shared canonical lattice keys), then the resulting
        ``(strut_curves, all_triangles, joint_positions)`` triple feeds
        ``build_export_triangles`` unchanged. ``lattice`` (when
        supplied) provides the shape parameters so the pose render
        matches the static render's tube radii and sphere counts."""
        from auxetic.geometry import (
            collect_export_geometry_from_posed_tiles,
            build_export_triangles,
        )

        posed_tiles = [
            _apply_tile_pose(tile_system.tiles[i], pose, i, tile_system.dimension)
            for i in range(len(tile_system.tiles))
        ]

        strut_curves, solid_triangles, joint_positions = (
            collect_export_geometry_from_posed_tiles(
                posed_tiles, tile_system.tile_source, tile_system.dimension,
            )
        )

        kwargs = {"verbose": False}
        if lattice is not None:
            # Mirror ``Lattice.shape_params`` defaults used by
            # ``Lattice.to_stl`` so hub/strut/sphere proportions match
            # the static render. Sphere ring/segment counts aren't
            # exposed on Lattice, so they fall through to the module
            # constants ``build_export_triangles`` reads from.
            kwargs["strut_radius"]        = float(lattice.strut_radius)
            kwargs["joint_sphere_radius"] = float(lattice.joint_sphere_radius)
        return build_export_triangles(
            strut_curves, solid_triangles, joint_positions, **kwargs
        )

    def close(self):
        if self.interactor is not None:
            try:
                self.interactor.close()
            except Exception:
                pass
        super().close()


def _apply_tile_pose(tile, pose, tile_idx, dimension):
    """Apply per-tile pose ``[tx, ty, θ]`` (2D) or
    ``[tx, ty, tz, rx, ry, rz]`` (3D, axis-angle) to ``tile``'s vertex
    array. Same convention as ``Simulator.assemble_jacobian`` — rotate
    around the origin, then translate — so a constraint-satisfying
    ``pose`` produces constraint-satisfying world-frame vertices.

    Used by ``View3D._build_pose_mesh_triangles`` and the matching
    test that promotes the Stage 6c diagnostic to a permanent
    invariant."""
    from scipy.spatial.transform import Rotation as _R

    dofs = 3 if dimension == 2 else 6
    s = tile_idx * dofs
    if dimension == 2:
        tx, ty, theta = float(pose[s]), float(pose[s + 1]), float(pose[s + 2])
        c, sn = np.cos(theta), np.sin(theta)
        R = np.array([[c, -sn], [sn, c]])
        return np.asarray(tile, dtype=float) @ R.T + np.array([tx, ty])
    else:
        t = np.asarray(pose[s:s + 3], dtype=float)
        omega = np.asarray(pose[s + 3:s + 6], dtype=float)
        if float(np.linalg.norm(omega)) < 1e-12:
            R = np.eye(3)
        else:
            R = _R.from_rotvec(omega).as_matrix()
        return np.asarray(tile, dtype=float) @ R.T + t


def _world_verts_3d(tile_system, pose) -> np.ndarray:
    """Stack every tile's world-frame vertices for ``pose`` into a
    single ``(N, 3)`` array. 2D lattices are padded with z=0 so the
    piston-visualisation helper can treat both dim cases uniformly."""
    if tile_system is None or tile_system.n_tiles == 0:
        return np.zeros((0, 3), dtype=float)
    dim = tile_system.dimension
    rows = []
    for ti in range(tile_system.n_tiles):
        v = _apply_tile_pose(tile_system.tiles[ti], pose, ti, dim)
        if dim == 2:
            v3 = np.hstack([v, np.zeros((v.shape[0], 1))])
        else:
            v3 = v
        rows.append(np.asarray(v3, dtype=float))
    return np.vstack(rows) if rows else np.zeros((0, 3), dtype=float)


def _triangles_to_polydata(triangles):
    """Stack a list of (3, 3) triangles into a pyvista PolyData."""
    import numpy as _np
    arr = _np.asarray(triangles, dtype=float)        # (n_tris, 3, 3)
    n_tris = arr.shape[0]
    points = arr.reshape(-1, 3)                       # (3 * n_tris, 3)
    # PyVista face encoding: [n_verts, i0, i1, i2, n_verts, ...]
    faces = _np.empty(n_tris * 4, dtype=_np.int64)
    faces[0::4] = 3
    indices = _np.arange(n_tris * 3, dtype=_np.int64).reshape(n_tris, 3)
    faces[1::4] = indices[:, 0]
    faces[2::4] = indices[:, 1]
    faces[3::4] = indices[:, 2]
    return pv.PolyData(points, faces)


def _force_actor_on_top(actor) -> bool:
    """Make an overlay actor render over the solid mesh from *any* viewing
    angle by pushing its rasterized depth toward the near plane via large
    negative coincident-topology offsets.

    The anchor-polygon ring is lifted just above the structure, so from the
    top it sits over the mesh — but from below the opaque mesh would occlude
    it. Forcing the ring on-top keeps it equally visible top / bottom / iso
    without changing the top-down look. Returns ``True`` if the offset was
    applied (the actor exposed a VTK mapper), ``False`` otherwise.
    """
    get_mapper = getattr(actor, "GetMapper", None)
    mapper = get_mapper() if callable(get_mapper) else getattr(actor, "mapper", None)
    if mapper is None:
        return False
    try:
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0.0, -66000.0)
        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, -66000.0)
        mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-66000.0)
    except Exception:
        return False
    return True


def _swap_mesh_actor(interactor, old_actor, mesh, **mesh_kwargs):
    """Replace ``old_actor`` with a fresh actor for ``mesh`` *without*
    rendering an intermediate empty frame.

    Both the removal and the add are issued with ``render=False`` so the
    caller can trigger a single render once the whole scene (mesh +
    overlays) is rebuilt. This is what stops the simulation-playback
    flicker: with the default ``render=True``, ``remove_actor`` renders a
    frame that has no mesh actor before ``add_mesh`` puts the new one in,
    so a 30 fps sweep strobes the lattice in and out. Returns the new
    actor, or ``None`` if the add failed.
    """
    if old_actor is not None:
        try:
            interactor.remove_actor(old_actor, render=False)
        except Exception:
            pass
    try:
        return interactor.add_mesh(mesh, render=False, **mesh_kwargs)
    except Exception:
        return None
