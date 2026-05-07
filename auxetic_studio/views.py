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
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout

import pyvista as pv

from auxetic import geometry as _geom

try:
    from pyvistaqt import QtInteractor
    _PYVISTAQT_AVAILABLE = True
except Exception:  # pragma: no cover - import-time platform issues
    QtInteractor = None
    _PYVISTAQT_AVAILABLE = False


# Edit-mode visual styling.
_NORMAL_SIZE   = 10.0
_HOVER_SIZE    = 13.0
_SELECTED_SIZE = 14.0

_NORMAL_BRUSH   = pg.mkBrush(30, 100, 200, 200)
_HOVER_BRUSH    = pg.mkBrush(255, 180, 60, 230)
_SELECTED_BRUSH = pg.mkBrush(220, 60, 60, 230)
_NORMAL_PEN     = pg.mkPen("k", width=0.8)

# Snap step for Shift-drag (lattice space).
_SNAP_STEP = 0.05


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._edit_enabled  = False
        self._drag_index    = -1
        self._hover_index   = -1

        # Default: don't intercept any mouse buttons (ViewBox pans, etc.).
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    # ------------------------------------------------------------------

    def setEditEnabled(self, enabled: bool) -> None:
        if self._edit_enabled == enabled:
            return
        self._edit_enabled = enabled
        self.setAcceptHoverEvents(enabled)
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton if enabled else Qt.MouseButton.NoButton
        )
        if not enabled:
            if self._hover_index != -1:
                self._hover_index = -1
                self.sigHoverChanged.emit(-1)
            self._drag_index = -1

    # ------------------------------------------------------------------

    def _index_at(self, pos) -> int:
        pts = self.pointsAt(pos)
        return int(pts[0].index()) if pts else -1

    def mouseClickEvent(self, ev):
        if not self._edit_enabled or ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore(); return
        idx = self._index_at(ev.pos())
        if idx < 0:
            ev.ignore(); return
        ev.accept()
        self.sigPointClicked.emit(idx)

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
        if not self._edit_enabled:
            return
        if ev.isExit():
            if self._hover_index != -1:
                self._hover_index = -1
                self.sigHoverChanged.emit(-1)
            return
        idx = self._index_at(ev.pos())
        if idx != self._hover_index:
            self._hover_index = idx
            self.sigHoverChanged.emit(idx)


_EDGE_FLIPPED_PEN    = pg.mkPen(220, 60, 60,   width=2.0)
_EDGE_FLIPPABLE_PEN  = pg.mkPen(30, 100, 200,  width=1.4)
_EDGE_FROZEN_PEN     = pg.mkPen(170, 170, 170, width=1.0,
                                 style=Qt.PenStyle.DashLine)


class _EdgeItem(pg.PlotCurveItem):
    """One clickable triangulation edge in 2D edge mode.

    Three styles encode the edge's role:
    - Red, solid, thick: currently flipped from the canonical Delaunay diagonal.
    - Blue, solid: flippable (interior + convex quad), but not yet flipped.
    - Grey, dashed: not flippable (boundary edge or non-convex quad).
    """

    sigEdgeClicked = pyqtSignal(object)  # edge tuple (i, j) with i < j

    def __init__(self, edge, *, flipped: bool, flippable: bool):
        super().__init__()
        self.edge = (int(edge[0]), int(edge[1]))
        self.flipped = bool(flipped)
        self.flippable = bool(flippable)
        if self.flipped:
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

        self.plot.addItem(self._scatter)
        layout.addWidget(self.plot)

        self._lattice         = None
        self._edit_mode       = False
        self._edge_mode       = False
        self._edge_items: list[_EdgeItem] = []
        self.selected_index   = -1
        self._hover_index     = -1
        self._drag_old_pos    = None  # 2-vector captured at drag start

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
        self._refresh_visuals()
        self._refresh_edges()
        self._auto_range()

    def set_edit_mode(self, on: bool) -> None:
        self._edit_mode = bool(on)
        self._scatter.setEditEnabled(self._edit_mode)
        if self._edit_mode:
            # Edit and edge modes are mutually exclusive.
            self._edge_mode = False
        else:
            self.selected_index = -1
            self._hover_index = -1
        self._refresh_visuals()
        self._refresh_edges()

    def set_edge_mode(self, on: bool) -> None:
        """Toggle the per-edge Delaunay flip mode.

        When ``on`` is True, triangulation edges are rendered as
        clickable line segments — a click fires ``edgeFlipRequested``
        with the edge tuple and current flipped state. Edit and edge
        modes are mutually exclusive; entering edge mode silently exits
        edit mode.
        """
        self._edge_mode = bool(on)
        if self._edge_mode:
            self._edit_mode = False
            self._scatter.setEditEnabled(False)
            self.selected_index = -1
            self._hover_index = -1
        self._refresh_visuals()
        self._refresh_edges()

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
                )
                item.setData(
                    x=[pts_world[a, 0], pts_world[b, 0]],
                    y=[pts_world[a, 1], pts_world[b, 1]],
                )
                item.sigEdgeClicked.connect(self._on_edge_clicked)
                self.plot.addItem(item)
                self._edge_items.append(item)

    def _on_edge_clicked(self, edge) -> None:
        if self._lattice is None:
            return
        a, b = int(edge[0]), int(edge[1])
        already_flipped = (a, b) in self._lattice.edge_flips
        self.edgeFlipRequested.emit((a, b), already_flipped)

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

        # Track whether the orientation/widget extras are active so
        # tests / external callers can introspect.
        self.has_grid           = False
        self.has_axes           = False
        self.has_view_cube      = False
        self._view_cube_widget  = None

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
    # Camera presets (Top / Front / Side / Iso / Fit)
    # ------------------------------------------------------------------
    #
    # These drive the CAMERA only — they don't touch the lattice's
    # rigid_rotation field. Conceptually distinct from the
    # InspectorPanel's "Top/Front/Side/Reset" buttons, which orient the
    # LATTICE in world space (and go through the undo stack). The
    # InspectorPanel buttons rotate the auxetic; these buttons rotate
    # the user's viewpoint.

    def camera_top(self) -> None:
        """Look straight down the +Z axis (top-down)."""
        if self.interactor is None:
            return
        try:
            self.interactor.view_xy()
        except Exception:
            pass

    def camera_front(self) -> None:
        """Look along the +Y axis (front view)."""
        if self.interactor is None:
            return
        try:
            self.interactor.view_xz()
        except Exception:
            pass

    def camera_side(self) -> None:
        """Look along the +X axis (right side)."""
        if self.interactor is None:
            return
        try:
            self.interactor.view_yz()
        except Exception:
            pass

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

        if self._mesh_actor is not None:
            try:
                self.interactor.remove_actor(self._mesh_actor)
            except Exception:
                pass
            self._mesh_actor = None

        try:
            self._mesh_actor = self.interactor.add_mesh(
                mesh, color="lightsteelblue", show_edges=False, smooth_shading=True
            )
            self._apply_user_transform(lattice)
            self.interactor.reset_camera()
        except Exception:
            self._mesh_actor = None

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

    def show_pose(self, tile_system, pose) -> None:
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

        ``last_show_pose_args`` is set to ``(tile_system, pose.copy())``
        so headless tests can verify the call without monkeypatching."""
        import numpy as _np
        self._pose_view_active = True
        self.last_show_pose_args = (tile_system, _np.asarray(pose, dtype=float).copy())

        if self.interactor is None:
            return

        triangles = self._build_pose_mesh_triangles(
            tile_system, pose, lattice=self._cached_lattice,
        )
        if not triangles:
            return

        try:
            mesh = _triangles_to_polydata(triangles)
        except Exception:
            return

        if self._mesh_actor is not None:
            try:
                self.interactor.remove_actor(self._mesh_actor)
            except Exception:
                pass
            self._mesh_actor = None
        try:
            self._mesh_actor = self.interactor.add_mesh(
                mesh, color="lightsteelblue", show_edges=False, smooth_shading=True,
            )
        except Exception:
            self._mesh_actor = None

    def clear_pose(self) -> None:
        """Drop the simulation-playback mesh and re-render the cached
        lattice via the default ``Lattice.to_stl`` path. Called by the
        SimulationPanel when the simulation is invalidated."""
        self._pose_view_active = False
        self.last_show_pose_args = None
        if self._cached_lattice is not None:
            self.update_lattice(self._cached_lattice)

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
