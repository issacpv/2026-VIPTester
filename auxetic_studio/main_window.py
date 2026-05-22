"""Main application window."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QUndoStack
from PyQt6.QtWidgets import (
    QMainWindow,
    QStackedWidget,
    QDockWidget,
    QToolBar,
    QFileDialog,
    QMessageBox,
    QStatusBar,
)

from auxetic import Lattice

from .views import View2D, View3D
from .inspector import InspectorPanel
from .preset import save_preset, load_preset
from .simulation_panel import SimulationPanel
from .predictor_panel import PredictorPanel
from .coordinates_panel import CoordinatesPanel
from .commands import (
    MovePointCommand,
    ParameterChangeCommand,
    DeletePointCommand,
    ResetToOriginalCommand,
    RotationChangeCommand,
    FlipCommand,
    FlipEdgeCommand,
    ForceListChangeCommand,
    JointAngleChangeCommand,
    RecommendationApplyCommand,
)


VIEW_2D = 0
VIEW_3D = 1

# Modes for which 2D point editing is allowed (per the §4.1.1 deferral
# captured in the prompt: 3D editing is out of scope for this stage).
# Modes 7 and 8 are 2D / 2.5D mesh-import — they share the editor surface
# with the random/grid 2D modes since their points live in 2-vector
# canonical form too.
_EDITABLE_MODES = (1, 2, 4, 5, 7, 8, 11)
# Modes for which the per-edge Delaunay flip GUI is meaningful (M1).
# Grid modes (4, 5) have flippable diagonals too, but their canonical
# triangulation is constructed deterministically and the user usually
# wants those preserved — flipping is still allowed but starts from
# the canonical state. Mode 11 (bipartite auxetic) is 2D Delaunay, so
# an edge flip simply picks the other diagonal — exactly the tool for
# choosing which way a placed rhombus splits into two tiles.
_EDGE_FLIP_MODES = (1, 2, 4, 5, 7, 8, 11)
_EDIT_DISABLED_TOOLTIP = (
    "3D editing is not supported in this version. "
    "Switch to a 2D mode to edit points."
)
_EDIT_ENABLED_TOOLTIP = "Toggle edit mode (drag points to reshape the lattice)"
_EDGE_DISABLED_TOOLTIP = (
    "Per-edge Delaunay flips are 2D-only in M1. Switch to a 2D / 2.5D mode."
)
_EDGE_ENABLED_TOOLTIP = (
    "Toggle edge mode — click an edge to select it, then Ctrl+click "
    "its two corners to flip the diagonal"
)


class MainWindow(QMainWindow):
    def __init__(self, parent=None, *, headless_3d: bool = False):
        """``headless_3d=True`` skips the VTK QtInteractor and uses a
        placeholder for the 3D view — useful for tests under
        ``QT_QPA_PLATFORM=offscreen`` where VTK's render-window
        initialisation can crash."""
        super().__init__(parent)
        self.setWindowTitle("Auxetic Studio")
        self.resize(1200, 800)

        # ---- model -------------------------------------------------------
        self.lattice = Lattice(mode=1, n_points=5, ratio=0.35, nz_layers=2, seed=42)
        self._current_path: str | None = None

        # ---- undo stack --------------------------------------------------
        self.undo_stack = QUndoStack(self)

        # ---- central stacked views ---------------------------------------
        self.view_2d = View2D(self)
        self.view_3d = View3D(self, force_placeholder=headless_3d)

        self.view_2d.pointMoveCompleted.connect(self._on_point_move_completed)
        self.view_2d.edgeFlipRequested.connect(self._on_edge_flip_requested)
        # Edge-flip gesture guidance → status bar. Lambda defers the
        # ``statusBar()`` lookup to emit-time, since the status bar is
        # constructed later in __init__ than the views.
        self.view_2d.edgeFlipStatus.connect(
            lambda msg: self.statusBar().showMessage(msg))
        # Ctrl-click a triangle in the 3D view → show that triangle's
        # generalized Poisson ratio in the status bar (task 6c).
        self.view_3d.trianglePoissonPicked.connect(
            self._on_triangle_poisson_picked)

        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.view_2d)   # index VIEW_2D
        self.stack.addWidget(self.view_3d)   # index VIEW_3D
        self.setCentralWidget(self.stack)

        # ---- inspector dock (right) --------------------------------------
        self.inspector = InspectorPanel(self.lattice, self)
        self.inspector.parameterChanged.connect(self._on_inspector_parameter_changed)
        self.inspector.rotationChangeRequested.connect(self._on_rotation_change_requested)
        self.inspector.flipChangeRequested.connect(self._on_flip_change_requested)
        self.inspector.meshImportRequested.connect(self._on_mesh_import_requested)

        dock = QDockWidget("Inspector", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(self.inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._inspector_dock = dock

        # ---- simulation panel dock (right, below inspector) --------------
        # SPEC §6.2 / §7 — Stage 6c ships the simulator runner UI.
        # The panel holds a reference to View3D so it can drive
        # ``show_pose`` directly during slider scrub/play, without
        # MainWindow having to mediate every frame.
        self.simulation_panel = SimulationPanel(
            self.lattice, view_3d=self.view_3d, parent=self,
        )
        self.simulation_panel.jointAngleChangeRequested.connect(
            self._on_joint_angle_change_requested
        )
        self.simulation_panel.forcesChangeRequested.connect(
            self._on_forces_change_requested
        )
        # Same handler the Inspector's orientation widgets use — the
        # dynamics pane's orientation sliders are a second entry point
        # to the same lattice field, not a parallel state.
        self.simulation_panel.rotationChangeRequested.connect(
            self._on_rotation_change_requested
        )
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.simulation_panel
        )
        self._simulation_dock = self.simulation_panel

        # ---- predictor panel dock (M3) ----------------------------------
        # Tabified with the simulation dock so the right column doesn't
        # get crowded; users switch between Simulation and Predictor
        # via the dock-tab strip.
        self.predictor_panel = PredictorPanel(self.lattice, parent=self)
        self.predictor_panel.applyRecommendationRequested.connect(
            self._on_apply_recommendation_requested
        )
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.predictor_panel
        )
        self.tabifyDockWidget(self.simulation_panel, self.predictor_panel)
        # Keep Simulation as the visible tab on first launch.
        self.simulation_panel.raise_()
        self._predictor_dock = self.predictor_panel

        # ---- coordinates panel dock (tabular point editor) ---------------
        # A keyboard alternative to the viewport's drag-to-move editing:
        # lists every point with editable X / Y cells. Tabified with the
        # Inspector since both are lattice-geometry panels.
        self.coordinates_panel = CoordinatesPanel(self.lattice, parent=self)
        self.coordinates_panel.pointMoveRequested.connect(
            self._on_coordinate_edit_requested
        )
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.coordinates_panel
        )
        self.tabifyDockWidget(self._inspector_dock, self.coordinates_panel)
        # Keep the Inspector as the visible tab on first launch.
        self._inspector_dock.raise_()
        self._coordinates_dock = self.coordinates_panel

        # ---- left toolbar ------------------------------------------------
        self._build_toolbar()

        # ---- menu bar ----------------------------------------------------
        self._build_menus()

        # ---- status bar --------------------------------------------------
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

        # initial state
        self._update_edit_action_enabled()
        self._update_edge_action_enabled()
        self._refresh_views()

    # =====================================================================
    # Construction helpers
    # =====================================================================

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(20, 20))
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)

        self.act_view_2d = QAction("2D", self)
        self.act_view_2d.setCheckable(True)
        self.act_view_2d.setChecked(True)
        self.act_view_2d.triggered.connect(lambda: self._set_view_mode(VIEW_2D))

        self.act_view_3d = QAction("3D", self)
        self.act_view_3d.setCheckable(True)
        self.act_view_3d.triggered.connect(lambda: self._set_view_mode(VIEW_3D))

        view_group = QActionGroup(self)
        view_group.addAction(self.act_view_2d)
        view_group.addAction(self.act_view_3d)
        view_group.setExclusive(True)

        tb.addAction(self.act_view_2d)
        tb.addAction(self.act_view_3d)
        tb.addSeparator()

        # Edit-mode toggle (mode-gated; see _update_edit_action_enabled).
        self.edit_action = QAction("Edit", self)
        self.edit_action.setCheckable(True)
        self.edit_action.setToolTip(_EDIT_ENABLED_TOOLTIP)
        self.edit_action.toggled.connect(self.set_edit_mode)
        tb.addAction(self.edit_action)

        # Edge-mode toggle (M1) — click triangulation edges to flip
        # their Delaunay diagonal. Mutually exclusive with Edit mode.
        self.edge_action = QAction("Edge", self)
        self.edge_action.setCheckable(True)
        self.edge_action.setToolTip(_EDGE_ENABLED_TOOLTIP)
        self.edge_action.toggled.connect(self.set_edge_mode)
        tb.addAction(self.edge_action)

        # SPEC §6 orientation-mode toggle. Per the scope-fallback clause
        # in the Stage 5 prompt, the 3D gizmo (and the matching 2D
        # rotation handle) ship in Stage 5.5; this toggle exists in the
        # toolbar so the layout matches §10 but is disabled until then.
        self.orientation_action = QAction("Orientation", self)
        self.orientation_action.setCheckable(True)
        self.orientation_action.setEnabled(False)
        self.orientation_action.setToolTip(
            "On-canvas rotation gizmo lands in Stage 5.5. "
            "Use the inspector's Orientation section for now."
        )
        tb.addAction(self.orientation_action)

        regen_act = QAction("Regenerate", self)
        regen_act.triggered.connect(self._regenerate)
        tb.addAction(regen_act)

        tb.addSeparator()

        # ---- Camera-preset buttons (3D viewport orientation) ------------
        # Distinct from the Inspector's Top/Front/Side/Reset preset
        # buttons: those rotate the LATTICE; these rotate the CAMERA.
        # The clickable view cube in the View3D corner does the same
        # job — these buttons are a redundant-but-handy shortcut and
        # also work when the cube widget is unavailable (older PyVista).
        self.cam_iso_act = QAction("Iso", self)
        self.cam_iso_act.setToolTip(
            "Camera: isometric 3/4 view. Doesn't change the lattice.")
        self.cam_iso_act.triggered.connect(lambda: self.view_3d.camera_isometric())
        tb.addAction(self.cam_iso_act)

        self.cam_fit_act = QAction("Fit", self)
        self.cam_fit_act.setToolTip(
            "Reframe the camera so the whole lattice is visible.")
        self.cam_fit_act.triggered.connect(lambda: self.view_3d.camera_fit())
        tb.addAction(self.cam_fit_act)

        self._toolbar = tb

    def _build_menus(self):
        mb = self.menuBar()

        # ---- File --------------------------------------------------------
        file_menu = mb.addMenu("&File")

        new_act = QAction("&New", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._on_new)
        file_menu.addAction(new_act)

        open_act = QAction("&Open…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        save_act = QAction("&Save", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._on_save)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save &As…", self)
        save_as_act.setShortcut("Ctrl+Shift+S")
        save_as_act.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        # Export submenu
        export_menu = file_menu.addMenu("&Export")
        for label, slot in [
            ("STL…",       self._on_export_stl),
            ("OBJ…",       self._on_export_obj),
            ("SCAD…",      self._on_export_scad),
            ("Kirigami…",  self._on_export_kirigami),
        ]:
            act = QAction(label, self)
            act.triggered.connect(slot)
            export_menu.addAction(act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # ---- Edit (real Undo/Redo + Edit Mode + Reset to Original) -------
        edit_menu = mb.addMenu("&Edit")

        # Real undo/redo backed by self.undo_stack — replaces Stage 2 stubs.
        self.undo_action = self.undo_stack.createUndoAction(self, "&Undo")
        self.undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        edit_menu.addAction(self.undo_action)

        self.redo_action = self.undo_stack.createRedoAction(self, "&Redo")
        self.redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        # Edit Mode menu item shares the same QAction as the toolbar
        # button — toggling either propagates to both.
        self.edit_action.setText("&Edit Mode")
        edit_menu.addAction(self.edit_action)

        self.reset_action = QAction("&Reset to Original", self)
        self.reset_action.triggered.connect(self._on_reset_to_original)
        edit_menu.addAction(self.reset_action)

        edit_menu.addSeparator()

        self.delete_action = QAction("&Delete Selected Point", self)
        self.delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self.delete_action.triggered.connect(self.delete_selected_point)
        edit_menu.addAction(self.delete_action)

        edit_menu.addSeparator()
        self._add_stub(edit_menu, "Preferences…")

        # ---- View --------------------------------------------------------
        view_menu = mb.addMenu("&View")
        view_menu.addAction(self.act_view_2d)
        view_menu.addAction(self.act_view_3d)
        view_menu.addSeparator()
        toggle_inspector = self._inspector_dock.toggleViewAction()
        toggle_inspector.setText("Show Inspector")
        view_menu.addAction(toggle_inspector)

        toggle_sim = self._simulation_dock.toggleViewAction()
        toggle_sim.setText("Show Simulation Panel")
        view_menu.addAction(toggle_sim)

        toggle_coords = self._coordinates_dock.toggleViewAction()
        toggle_coords.setText("Show Coordinates")
        view_menu.addAction(toggle_coords)

        # ---- Simulate (stubs) -------------------------------------------
        sim_menu = mb.addMenu("&Simulate")
        for label in ("Run Simulation", "Pause", "Reset"):
            self._add_stub(sim_menu, label)

        # ---- Help -------------------------------------------------------
        help_menu = mb.addMenu("&Help")
        about_act = QAction("&About Auxetic Studio", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)
        self._add_stub(help_menu, "Documentation")

    def _add_stub(self, menu, label):
        act = QAction(label, self)
        act.triggered.connect(lambda _, name=label: self._not_implemented(name))
        menu.addAction(act)
        return act

    # =====================================================================
    # View management
    # =====================================================================

    def _set_view_mode(self, mode):
        self.stack.setCurrentIndex(mode)
        if mode == VIEW_2D:
            self.act_view_2d.setChecked(True)
            self.statusBar().showMessage("2D view")
        else:
            self.act_view_3d.setChecked(True)
            self.statusBar().showMessage("3D view")

    def _refresh_views(self):
        self.view_2d.update_lattice(self.lattice)
        self.view_3d.update_lattice(self.lattice)
        self.statusBar().showMessage(
            f"mode={self.lattice.mode}  n={self.lattice.n_points}  "
            f"ratio={self.lattice.ratio:.3f}  nz={self.lattice.nz_layers}"
        )

    def _regenerate(self):
        # Re-roll: this discards manual edits per SPEC §4.3. Push as a
        # parameter command-style operation so it lands on the undo stack.
        # (Implementing the §4.3 confirmation dialog is out of scope here.)
        self.lattice.regenerate()
        self.undo_stack.clear()  # re-roll throws away edit history
        self._on_lattice_structurally_changed()

    def _refresh_state(self):
        """Refresh inspector + simulation panel + views + mode-gated
        edit action. Called after every command (redo/undo) and from
        any other path that mutates the lattice.

        Order matters: the canonical 3D render via ``_refresh_views``
        runs FIRST, then ``simulation_panel.refresh_from_lattice``
        drives the posed mesh on top of it. Reversing this order
        means the panel's ``show_pose`` call gets immediately
        overwritten by the canonical render, and the user-released
        joint-angle slider snaps the visualisation back to rest
        (regression reported as "the figure returns to original
        structure after letting go of the slider").
        """
        self.inspector.refresh_from_lattice()
        self.coordinates_panel.refresh_from_lattice()
        self.predictor_panel.refresh_metrics()
        self._update_edit_action_enabled()
        self._update_edge_action_enabled()
        self._refresh_views()
        # Must be last — drives the posed mesh on top of the canonical
        # render that ``_refresh_views`` just installed.
        self.simulation_panel.refresh_from_lattice()

    def _on_lattice_structurally_changed(self):
        """Like ``_refresh_state``, but also invalidates the simulation
        panel. Used as the ``on_change`` callback for every command
        EXCEPT ``JointAngleChangeCommand`` — the joint angle is the
        simulator's parameter, not its input, so scrubbing the
        slider/spinbox doesn't change the trajectory or metrics
        and shouldn't force a rerun.

        Per SPEC §7.7 / Stage 6c prompt: ``regenerate``, point edits,
        rotation/flip, mode/ratio/nz_layers, and preset load all
        invalidate; joint-angle changes don't."""
        self._refresh_state()
        self.simulation_panel.mark_outdated()

    # =====================================================================
    # Edit mode
    # =====================================================================

    def _update_edit_action_enabled(self) -> None:
        """Mode-gate the Edit toolbar button + menu item.

        When the current lattice mode is 3D (3 or 6) the edit action is
        disabled with the prompt-mandated tooltip. If edit mode was on
        when the user switched to a 3D mode, force it off — there's no
        coherent meaning to "editing a 3D lattice in 2D" without the
        full §4.1.1 work."""
        editable = self.lattice.mode in _EDITABLE_MODES
        self.edit_action.setEnabled(editable)
        self.edit_action.setToolTip(
            _EDIT_ENABLED_TOOLTIP if editable else _EDIT_DISABLED_TOOLTIP
        )
        if not editable and self.edit_action.isChecked():
            self.edit_action.setChecked(False)  # triggers set_edit_mode(False)

    def set_edit_mode(self, on: bool) -> None:
        """Slot for ``edit_action.toggled``. Tells View2D and forces the
        viewport to 2D since 3D editing isn't supported in this stage."""
        on = bool(on)
        if on and self.lattice.mode not in _EDITABLE_MODES:
            self.edit_action.setChecked(False)
            return
        if on and self.edge_action.isChecked():
            # Mutually exclusive with edge mode.
            self.edge_action.setChecked(False)
        self.view_2d.set_edit_mode(on)
        if on:
            self._set_view_mode(VIEW_2D)
            self.statusBar().showMessage("Edit mode ON — drag points to move them")
        else:
            self.statusBar().showMessage("Edit mode OFF")

    def _update_edge_action_enabled(self) -> None:
        """Mode-gate the Edge toolbar button. 3D modes have no per-edge
        flip semantics in M1, so the action is disabled there."""
        editable = self.lattice.mode in _EDGE_FLIP_MODES
        self.edge_action.setEnabled(editable)
        self.edge_action.setToolTip(
            _EDGE_ENABLED_TOOLTIP if editable else _EDGE_DISABLED_TOOLTIP
        )
        if not editable and self.edge_action.isChecked():
            self.edge_action.setChecked(False)  # triggers set_edge_mode(False)

    def set_edge_mode(self, on: bool) -> None:
        """Slot for ``edge_action.toggled``. Mirrors ``set_edit_mode``
        for the per-edge Delaunay flip workflow."""
        on = bool(on)
        if on and self.lattice.mode not in _EDGE_FLIP_MODES:
            self.edge_action.setChecked(False)
            return
        if on and self.edit_action.isChecked():
            self.edit_action.setChecked(False)
        self.view_2d.set_edge_mode(on)
        if on:
            self._set_view_mode(VIEW_2D)
            self.statusBar().showMessage(
                "Edge mode ON — click a blue/red edge to select it (turns "
                "green), then Ctrl+click the two amber corners to flip it")
        else:
            self.statusBar().showMessage("Edge mode OFF")

    @property
    def edit_mode(self) -> bool:
        return bool(self.edit_action.isChecked())

    @property
    def edge_mode(self) -> bool:
        return bool(self.edge_action.isChecked())

    def delete_selected_point(self) -> None:
        """Public so tests (and the Delete shortcut) can both call it.
        Enforces SPEC §4.1's "must leave ≥ 3 points" invariant."""
        if not self.edit_mode:
            return
        idx = self.view_2d.selected_index
        if idx < 0 or idx >= len(self.lattice.points):
            self.statusBar().showMessage("No point selected to delete")
            return
        if len(self.lattice.points) <= 3:
            self.statusBar().showMessage(
                "Cannot delete: lattice must have at least 3 points"
            )
            return
        cmd = DeletePointCommand(self.lattice, idx,
                                  on_change=self._on_lattice_structurally_changed)
        self.undo_stack.push(cmd)
        # The deleted index disappears from the array; clear selection.
        self.view_2d.selected_index = -1

    # =====================================================================
    # Edit-flow signal handlers
    # =====================================================================

    def _on_inspector_parameter_changed(self, field, old_value, new_value):
        """Push a parameter change onto the undo stack rather than
        mutating the lattice directly. Per SPEC §4.2, parameter changes
        are undoable.

        The mode-11 ``C`` (constant size ratio) only repositions hinges
        on the existing triangulation — re-rolling the point cloud would
        destroy the user's placed lattice — so it is applied with
        ``regenerate=False``. Every other parameter re-generates."""
        # C and the bezier-strut options only affect derived geometry on
        # the existing triangulation — re-rolling the point cloud would
        # discard the user's placed lattice, so they apply without a
        # regenerate (the command invalidates the export cache instead).
        _no_regen = ("C", "bezier_enabled", "bezier_strength", "bezier_segments")
        cmd = ParameterChangeCommand(
            self.lattice, field, old_value, new_value,
            on_change=self._on_lattice_structurally_changed,
            regenerate=(field not in _no_regen),
        )
        self.undo_stack.push(cmd)

    def _on_point_move_completed(self, idx, old_pos, new_pos):
        """Drag-release in View2D: one undoable step per move (SPEC §4.2)."""
        cmd = MovePointCommand(
            self.lattice, idx, old_pos, new_pos,
            on_change=self._on_lattice_structurally_changed,
        )
        self.undo_stack.push(cmd)

    def _on_coordinate_edit_requested(self, idx, old_pos, new_pos):
        """Cell commit in the Coordinates panel: same undoable
        ``MovePointCommand`` path as a viewport drag-release, so typing
        a coordinate and dragging a point are interchangeable and both
        land on the undo stack."""
        cmd = MovePointCommand(
            self.lattice, idx, old_pos, new_pos,
            on_change=self._on_lattice_structurally_changed,
        )
        self.undo_stack.push(cmd)

    def _on_edge_flip_requested(self, edge, already_flipped):
        """Edge click in View2D → ``FlipEdgeCommand`` (M1)."""
        cmd = FlipEdgeCommand(
            self.lattice, edge, bool(already_flipped),
            on_change=self._on_lattice_structurally_changed,
        )
        self.undo_stack.push(cmd)

    def _on_triangle_poisson_picked(self, point):
        """Ctrl-click in the 3D view → show the picked triangle's generalized
        Poisson ratio in the status bar (task 6c). Geometry + ν come from
        ``Lattice.poisson_ratio_at_point``; this only formats and displays."""
        if point is None:
            return
        try:
            idx, nu = self.lattice.poisson_ratio_at_point(point, world=True)
        except Exception:
            return
        if idx is None:
            self.statusBar().showMessage(
                "Per-triangle ν: unavailable (3D mode / no 2D triangulation)",
                6000)
        elif isinstance(nu, float) and nu != nu:        # NaN
            self.statusBar().showMessage(
                f"Triangle {idx}: ν unavailable (degenerate)", 6000)
        else:
            self.statusBar().showMessage(
                f"Triangle {idx}: generalized Poisson's ratio ν = {nu:+.4f}",
                8000)

    def _on_reset_to_original(self):
        cmd = ResetToOriginalCommand(
            self.lattice, on_change=self._on_lattice_structurally_changed,
        )
        self.undo_stack.push(cmd)
        self.statusBar().showMessage("Reset to original points")

    # ---- SPEC §6 orientation flow ----------------------------------------

    def _on_rotation_change_requested(self, old_rotation, new_rotation):
        """Inspector → ``RotationChangeCommand``. Per SPEC §6.1 rigid
        rotation is undoable just like any other parameter change."""
        cmd = RotationChangeCommand(
            self.lattice, old_rotation, new_rotation,
            on_change=self._on_lattice_structurally_changed,
        )
        self.undo_stack.push(cmd)

    def _on_flip_change_requested(self, old_value, new_value):
        cmd = FlipCommand(
            self.lattice, old_value, new_value,
            on_change=self._on_lattice_structurally_changed,
        )
        self.undo_stack.push(cmd)

    def _on_joint_angle_change_requested(self, old_rad, new_rad):
        """SimulationPanel debounces — emits once per slider release.
        SPEC §6.3 keeps joint angle DISTINCT from rigid rotation; this
        handler is intentionally separate from
        ``_on_rotation_change_requested``."""
        cmd = JointAngleChangeCommand(
            self.lattice, old_rad, new_rad,
            on_change=self._refresh_state,
        )
        self.undo_stack.push(cmd)

    def _on_forces_change_requested(self, old_forces, new_forces):
        """SimulationPanel emits one (old, new) pair per add / remove /
        edit on the dynamics force table. Wrap as an undoable command
        so the user can revert experimental load cases.

        Force-list edits invalidate the most-recent dynamics result,
        so the on_change callback also drops it (handled by
        ``_on_dynamics_state_changed``)."""
        cmd = ForceListChangeCommand(
            self.lattice, old_forces, new_forces,
            on_change=self._on_dynamics_state_changed,
        )
        self.undo_stack.push(cmd)

    def _on_apply_recommendation_requested(self,
                                            ground_face,
                                            edge_flips,
                                            joint_angle_rad: float):
        """Predictor panel emits this when the user clicks Apply on a
        recommendation. Wrap as a single ``RecommendationApplyCommand``
        so the three lattice fields move atomically and undo restores
        the prior config in one click."""
        cmd = RecommendationApplyCommand(
            self.lattice,
            ground_face, edge_flips, joint_angle_rad,
            on_change=self._on_dynamics_state_changed,
        )
        self.undo_stack.push(cmd)

    def _on_dynamics_state_changed(self):
        """Invalidate the simulation panel's stale dynamics result and
        repaint everything that depends on the lattice. Used as the
        ``on_change`` callback for force-edit commands."""
        # Drop the dynamics result; user must rerun.
        self.simulation_panel._dynamics_result = None
        self.simulation_panel._dynamics_error  = None
        self._refresh_state()

    # =====================================================================
    # File menu handlers
    # =====================================================================

    def _on_new(self):
        self.lattice = Lattice(mode=1, n_points=5, ratio=0.35, nz_layers=2, seed=42)
        self.inspector.set_lattice(self.lattice)
        self.simulation_panel.set_lattice(self.lattice)
        self.predictor_panel.set_lattice(self.lattice)
        self.coordinates_panel.set_lattice(self.lattice)
        self.undo_stack.clear()
        self._current_path = None
        self.setWindowTitle("Auxetic Studio")
        self._refresh_state()

    def _on_open(self, _checked=False, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Preset", "", "Auxetic Preset (*.json);;All Files (*)"
            )
        if not path:
            return
        try:
            self.lattice = load_preset(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.inspector.set_lattice(self.lattice)
        self.simulation_panel.set_lattice(self.lattice)
        self.predictor_panel.set_lattice(self.lattice)
        self.coordinates_panel.set_lattice(self.lattice)
        self.undo_stack.clear()
        self._current_path = path
        self.setWindowTitle(f"Auxetic Studio — {os.path.basename(path)}")
        self._refresh_state()

    def _on_mesh_import_requested(self, path: str, decimate_to):
        """Replace ``self.lattice`` with the result of
        ``Lattice.from_mesh(path)``. The dimensionality follows the
        inspector's current ``dim_combo`` selection at the moment of
        the import. Like File → Open this is a major action that
        clears the undo stack."""
        from auxetic import Lattice
        # Pull dim from the inspector's current state, not from the
        # lattice mode (which may still reflect the pre-import mode).
        dim_label = str(self.inspector.dim_combo.currentData())
        try:
            n = int(decimate_to) if decimate_to is not None else None
            new_lattice = Lattice.from_mesh(path, dim=dim_label, decimate_to=n)
        except Exception as e:
            QMessageBox.critical(self, "Mesh import failed", str(e))
            # Revert the inspector's combos to the still-current lattice.
            self.inspector.refresh_from_lattice()
            return
        self.lattice = new_lattice
        self.inspector.set_lattice(self.lattice)
        self.simulation_panel.set_lattice(self.lattice)
        self.predictor_panel.set_lattice(self.lattice)
        self.coordinates_panel.set_lattice(self.lattice)
        self.undo_stack.clear()
        self._current_path = None
        self.setWindowTitle(
            f"Auxetic Studio — {os.path.basename(path)} (mesh)")
        self._refresh_state()

    def _on_save(self):
        if self._current_path:
            self._save_to(self._current_path)
        else:
            self._on_save_as()

    def _on_save_as(self, _checked=False, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Preset", "preset.json",
                "Auxetic Preset (*.json);;All Files (*)",
            )
        if not path:
            return
        self._save_to(path)
        self._current_path = path
        self.setWindowTitle(f"Auxetic Studio — {os.path.basename(path)}")

    def _save_to(self, path: str):
        try:
            save_preset(path, self.lattice)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # ---- Export ----------------------------------------------------------

    def _ask_path(self, caption: str, default_name: str, filters: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, caption, default_name, filters)
        return path or None

    def export_stl(self, path: str) -> None:
        self.lattice.to_stl(path, verbose=False)

    def export_obj(self, path: str) -> None:
        self.lattice.to_obj(path, verbose=False)

    def export_scad(self, path: str) -> None:
        self.lattice.to_scad(path, verbose=False)

    def export_kirigami(self, vertices_path: str, constraints_path: str) -> None:
        self.lattice.to_kirigami(vertices_path, constraints_path, verbose=False)

    def _on_export_stl(self):
        path = self._ask_path("Export STL", "auxetic_lattice.stl",
                              "STL (*.stl);;All Files (*)")
        if path: self.export_stl(path); self.statusBar().showMessage(f"Exported STL: {path}")

    def _on_export_obj(self):
        path = self._ask_path("Export OBJ", "auxetic_lattice.obj",
                              "OBJ (*.obj);;All Files (*)")
        if path: self.export_obj(path); self.statusBar().showMessage(f"Exported OBJ: {path}")

    def _on_export_scad(self):
        path = self._ask_path("Export SCAD", "auxetic_lattice.scad",
                              "SCAD (*.scad);;All Files (*)")
        if path: self.export_scad(path); self.statusBar().showMessage(f"Exported SCAD: {path}")

    def _on_export_kirigami(self):
        verts_path = self._ask_path(
            "Export Kirigami — Vertices", "vertices.txt",
            "Text (*.txt);;All Files (*)")
        if not verts_path: return
        consts_path = self._ask_path(
            "Export Kirigami — Constraints", "constraints.txt",
            "Text (*.txt);;All Files (*)")
        if not consts_path: return
        self.export_kirigami(verts_path, consts_path)
        self.statusBar().showMessage(f"Exported Kirigami: {verts_path}, {consts_path}")

    # =====================================================================
    # Stubs / about
    # =====================================================================

    def _not_implemented(self, name: str):
        QMessageBox.information(
            self, name, f"{name} is not yet implemented."
        )

    def _on_about(self):
        QMessageBox.about(
            self, "About Auxetic Studio",
            "Auxetic Studio — application shell.\n"
            "Backed by the auxetic/ kirigami lattice library.",
        )

    # =====================================================================
    # Cleanup
    # =====================================================================

    def closeEvent(self, event):
        # pyvistaqt requires explicit close to release the VTK render window
        try:
            self.view_3d.close()
        except Exception:
            pass
        # Stop the simulation panel's QTimer + close its matplotlib
        # figure explicitly — Qt's deferred-deletion path is not
        # synchronous enough to guarantee these are gone before the
        # next test's QTest.qWait pumps the event loop. Without this,
        # the play-animation test crashes when run after a few prior
        # MainWindow lifecycles in the same process.
        try:
            self.simulation_panel.shutdown()
        except Exception:
            pass
        try:
            self.predictor_panel.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
