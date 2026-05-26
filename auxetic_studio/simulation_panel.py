"""Simulation panel dock — SPEC §6.2 / §7.

Wires the simulator (Stages 6a/6b) into the GUI:

- "Run Simulation" button computes a SimResult, Poisson's ratio, and
  locking status, all from the live ``Lattice`` via
  ``TileSystem.from_lattice``.
- The joint-angle slider, when a fresh result is present, drives
  ``View3D.show_pose`` along the recorded trajectory.
- The Play button animates the slider through the full bistable cycle.
- A matplotlib plot shows ``bbox_extent[axial] vs slider θ``, with a
  vertical marker tracking the slider position.

Convention boundary (per SPEC §6.2 and ``CLAUDE.md`` "Conventions worth
knowing"): the slider is in **physical degrees** [0°, 180°], the
simulator and ``SimResult`` are in **mathematical radians**
[-π/2, +π/2]. The two helpers at the top of this file are the only
place the conversion happens. No third convention is allowed.
"""

from __future__ import annotations

import math
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QCheckBox,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

from auxetic import TileSystem, Simulator
from auxetic.dynamics import build_dynamics_simulator_from_lattice
from auxetic_studio.views import POISSON_BOXES


# ---------------------------------------------------------------------------
# Convention helpers — the ONLY place slider↔simulator conversion lives.
# Per SPEC §6.2, slider is physical degrees, simulator is mathematical
# radians. ``CLAUDE.md`` "Conventions worth knowing" forbids any third
# convention; if you find yourself writing ``theta * 2`` or
# ``theta + pi/2`` outside these two functions, you're either
# reinventing the wheel or adding a bug.
# ---------------------------------------------------------------------------

def slider_to_simulator_theta(slider_deg: float) -> float:
    """Slider ``[0, 180]`` physical degrees -> simulator ``[-π, +π]``
    radians. M2.8 doubled the math range so the slider can scrub a
    full 180° rotation from rest in either direction (the kirigami
    can keep rotating until tile-tile collisions stop it).

    Original SPEC §6.2 convention was slider 0-180 → math -π/2 to
    +π/2 (the bistable cycle). With collision detection now bounding
    the reachable range, we cover the full ±π and let collisions —
    shaded on the plot — show what's physically achievable.

    Rest is at slider=90° (math 0) for both conventions.
    """
    return float(np.radians((slider_deg - 90.0) * 2.0))


def simulator_theta_to_slider(theta_rad: float) -> float:
    """Inverse of :func:`slider_to_simulator_theta` — simulator radians
    in ``[-π, +π]`` map back to ``[0, 180]`` physical slider degrees."""
    return float(np.degrees(theta_rad) / 2.0 + 90.0)


def _rotations_close(a: Rotation, b: Rotation, tol: float = 1e-9) -> bool:
    """Quaternion-equivalent comparison — duplicated from the Inspector
    so the orientation sliders here can suppress no-op emits without
    pulling that helper into the public API."""
    qa = np.asarray(a.as_quat())
    qb = np.asarray(b.as_quat())
    return bool(np.linalg.norm(qa - qb) < tol or np.linalg.norm(qa + qb) < tol)


# Slider range and resolution (sub-degree granularity via tenths-of-deg
# integer values, since QSlider doesn't do floats natively).
_SLIDER_MIN_DEG  = 0.0
_SLIDER_MAX_DEG  = 180.0
_SLIDER_REST_DEG = 90.0
_SLIDER_SCALE    = 10  # int slider value = degrees × 10
# The math-θ extremes the slider corresponds to, used by the collision-
# shading spans on the kinematic plot.
_SLIDER_MIN_DEG_THETA_MIN = -np.pi   # slider 0°   → math -π
_SLIDER_MAX_DEG_THETA_MAX = +np.pi   # slider 180° → math +π


def _mode11_overlap_spans(x_deg, margin_frac: float = 0.15):
    """Red 'overlap' shading spans + x-limits for a mode-11 kinematic plot.

    The bipartite mechanism follows its rigid manifold and stops at the
    jamming angle — the exact point where the polygons first touch — so
    (unlike the over-rotating ``sweep_theta`` modes) there is no
    overlapping region *inside* the swept range to flag. Everything past
    the reachable ends is overlap. This returns two red spans flanking the
    reachable x-range (in slider degrees) plus widened x-limits so the
    margins are visible past the curve.

    Returns ``([(lo, hi), ...], (xlo, xhi))``; empty spans + ``None``
    x-limits for a degenerate (zero-width) sweep, so nothing is shaded.
    """
    x = np.asarray(x_deg, dtype=float).ravel()
    if x.size < 2:
        return [], None
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return [], None
    margin = float(margin_frac) * (hi - lo)
    spans = [(hi, hi + margin), (lo - margin, lo)]
    xlim = (lo - margin * 1.15, hi + margin * 1.15)
    return spans, xlim

# Animation: 30 fps, full bistable cycle in ~3 s.
_PLAY_FPS_INTERVAL_MS = 33
_PLAY_CYCLE_FRAMES    = 90

# Orientation slider (Dynamic mode): one slider per Euler axis. Range
# matches the inspector's spinboxes — [-180°, +180°]. Same tenths-of-deg
# integer encoding as the joint slider so QSlider can hold sub-degree
# values.
_ORIENT_SLIDER_MIN_DEG = -180.0
_ORIENT_SLIDER_MAX_DEG =  180.0
_ORIENT_SLIDER_SCALE   = 10


def _format_sim_error(exc: Exception) -> str:
    """Format an exception for the readout. Must be called inside the
    ``except`` block so ``format_exc`` reflects the live exception."""
    return (f"{type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc().splitlines()[-1]}")


def _solve_kinematic(tile_system, load_axis, is_mode_11: bool, jam: float):
    """Run the quasi-static kinematic sweep + Poisson's ratio + locking
    for a tile system. Pure compute — no GUI, no ``Lattice`` access — so
    it produces identical results on the UI thread (``run_simulation``) or
    a worker thread (:class:`_SimWorker`). Inputs come from
    ``_build_sim_inputs`` on the main thread (the only place the live
    lattice is read). Returns
    ``(simulator, sim_result, poissons, locked, info)``."""
    simulator = Simulator(tile_system, load_axis=load_axis)
    if is_mode_11:
        # Mode 11 is a large-amplitude rotating-units mechanism: follow the
        # curved 1-DOF manifold via sweep_mechanism (the fixed-rest-mode
        # sweep_theta saturates part-way). ``collision_stop`` bounds the
        # march at the first real polygon collision — the physical jamming
        # limit — so the units stop where they actually touch instead of
        # rotating on into a self-overlapping collapse (the analytic
        # ``jam`` angle overestimates the reachable range).
        sim_result = simulator.sweep_mechanism(max_actuation=jam,
                                               collision_stop=True)
    else:
        # M2.8: sweep the full ±π range with tile-tile collision detection
        # so the plot shows what's physically reachable.
        sim_result = simulator.sweep_theta(
            n_steps=181, theta_max=np.pi, collision_stop=True,
        )
    poissons = simulator.poissons_ratio()
    if is_mode_11:
        # Reuse the real sweep's locking/compression; is_locked() would
        # re-sweep with the tiny default ±π/2 amplitude and wrongly report
        # a working auxetic as locked.
        locked, info = sim_result.locked, sim_result.locking_info
    else:
        locked, info = simulator.is_locked()
    return simulator, sim_result, poissons, locked, info


class _SimWorker(QObject):
    """Runs :func:`_solve_kinematic` off the UI thread so a long sweep
    doesn't freeze the app (mirrors ``predictor_panel._TrainerWorker``).

    Emits:
    - ``finished(payload)`` — ``(tile_system, simulator, sim_result,
      poissons, locked, info)`` on success
    - ``failed(message)``   — formatted error text on any exception
    """
    finished = pyqtSignal(object)
    failed   = pyqtSignal(str)

    def __init__(self, tile_system, load_axis, is_mode_11: bool, jam: float,
                 parent: QObject | None = None):
        super().__init__(parent)
        self._tile_system = tile_system
        self._load_axis   = load_axis
        self._is_mode_11  = is_mode_11
        self._jam         = jam

    def run(self) -> None:
        """Slot connected to ``QThread.started``."""
        try:
            results = _solve_kinematic(
                self._tile_system, self._load_axis,
                self._is_mode_11, self._jam)
            self.finished.emit((self._tile_system, *results))
        except Exception as exc:
            self.failed.emit(_format_sim_error(exc))


class SimulationPanel(QDockWidget):
    # Class-level alias of the module constant, so tests / external
    # callers don't have to import the module to compute slider values.
    SLIDER_SCALE = _SLIDER_SCALE
    SLIDER_REST_DEG = _SLIDER_REST_DEG

    # Pushed on slider release / spinbox commit (debounced, one per
    # interaction) so the QUndoStack doesn't get spammed.
    jointAngleChangeRequested = pyqtSignal(float, float)  # old_rad, new_rad

    # Emitted when ``run_simulation`` succeeds. The MainWindow connects
    # this so it can update its status bar / window title etc.
    simulationCompleted = pyqtSignal()

    # Stage 5-era hook (unused here — Play is wired internally now).
    playRequested = pyqtSignal()

    # M2.9 — emitted when the user adds / removes / edits a force in
    # the dynamics force table. Carries (old_forces, new_forces) so
    # the MainWindow can wrap the change in a ForceListChangeCommand.
    forcesChangeRequested = pyqtSignal(object, object)

    # Emitted when the orientation sliders (Dynamic mode) commit a
    # change to ``lattice.rigid_rotation``. MainWindow wraps the
    # (old, new) ``Rotation`` pair in a ``RotationChangeCommand`` so
    # undo/redo works the same as the Inspector's orientation widgets.
    rotationChangeRequested = pyqtSignal(object, object)

    # ------------------------------------------------------------------

    def __init__(self, lattice, view_3d=None, parent=None):
        super().__init__("Simulation", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._lattice  = lattice
        self._view_3d  = view_3d        # may be None in early-construction
        self._suspend  = False
        # Captured at slider press; the angle to compare against on
        # release so the QUndoStack receives one (old, new) pair.
        self._press_angle_rad: float | None = None
        # Same role for the orientation sliders — captured at slider
        # press so release can emit (old_rotation, new_rotation).
        self._press_rotation: Rotation | None = None

        # ---- simulation result state -------------------------------------
        self._sim_result      = None
        self._tile_system     = None
        self._simulator       = None
        self._poissons_ratio  = None      # bbox ν (SPEC §7.4), whole structure
        self._edge_poisson_ratio = None   # full-structure edge-vector ν (mean)
        self._locking_info    = None
        self._is_outdated     = False     # True after lattice change until rerun
        self._last_error      = None      # exception text if last run failed

        # ---- async kinematic-solve worker (keeps the GUI responsive) -----
        # The toolbar Run button drives the solve on a QThread so a long
        # sweep can't freeze the event loop; run_simulation() itself stays
        # synchronous for programmatic/test callers.
        self._sim_thread    = None
        self._sim_worker    = None
        self._sim_cancelled = False

        # ---- anchor-view-to-polygon state (kinematic scrub only) ---------
        # The polygon tile the 3D view is rendered relative to (None =
        # world frame). Set by clicking a tile in the 3D view. The most
        # recently displayed kinematic pose is kept so a click can be
        # resolved to the nearest polygon in the geometry actually on screen.
        self._anchor_tile: int | None = None
        self._displayed_pose = None

        # ---- M2 dynamic-sim state ----------------------------------------
        self._dynamics_result = None      # most recent DynamicsResult, if any
        self._dynamics_error  = None      # exception text from last Run Dynamic
        # Which trajectory the slider scrubs / the plot shows. Set by
        # the Mode radio at the top of the panel; defaults to the
        # existing kinematic behaviour so nothing changes for users
        # who never click Dynamic.
        self._scrub_mode      = "kinematic"  # "kinematic" or "dynamic"

        # ---- play state --------------------------------------------------
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(_PLAY_FPS_INTERVAL_MS)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._play_phase = 0              # frame counter for the cycle

        # ---- widgets -----------------------------------------------------
        body = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(8, 8, 8, 8)

        self._build_mode_toggle(outer)
        self._build_run_row(outer)
        self._build_dynamics_config_box(outer)
        self._build_joint_slider_box(outer)
        self._build_poisson_bounds_box(outer)
        self._build_plot(outer)
        self._build_readout(outer)

        outer.addStretch(1)
        self.setWidget(body)

        self.refresh_from_lattice()
        self._update_state_dependent_ui()
        self._wire_view3d(self._view_3d)

    # ==================================================================
    # Construction helpers
    # ==================================================================

    def _build_mode_toggle(self, outer):
        """Mode radio: Kinematic / Dynamic. Drives which trajectory
        the slider scrubs and what the plot displays."""
        row = QWidget(self)
        layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("<b>Mode:</b>"))
        self.mode_kinematic_radio = QRadioButton("Kinematic", row)
        self.mode_dynamic_radio   = QRadioButton("Dynamic",   row)
        self.mode_kinematic_radio.setChecked(True)
        self.mode_kinematic_radio.setToolTip(
            "Quasi-static θ-sweep — Poisson's ratio, locking criterion")
        self.mode_dynamic_radio.setToolTip(
            "Newtonian dynamic sim — scrub slider through time")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.mode_kinematic_radio, 0)
        self._mode_group.addButton(self.mode_dynamic_radio,   1)
        self._mode_group.idToggled.connect(self._on_mode_toggled)
        layout.addWidget(self.mode_kinematic_radio)
        layout.addWidget(self.mode_dynamic_radio)
        layout.addStretch(1)
        outer.addWidget(row)

    def _build_dynamics_config_box(self, outer):
        """Dynamics config: ground-face dropdown + force-table editor.

        The table edits ``lattice.dynamics_state['forces']`` directly.
        Each row is one force vector with columns Tile / Vertex / dx /
        dy / dz / Magnitude. Vertex = -1 means "applied at tile
        centroid" (no torque); Vertex >= 0 means "applied at that body-
        frame vertex" (induces torque).
        """
        self._dynamics_config_box = QGroupBox("Dynamics setup", self)
        v = QVBoxLayout(self._dynamics_config_box)

        # ---- Piston compression row (primary workflow) ----------------
        # Auto-pins the bottom of the lattice (in world frame, after
        # the rigid rotation set via the Inspector) and pushes a
        # downward force totalling N Newtons on the top vertices.
        # Sets all the manual dynamics knobs below to "ignored" while
        # piston_force_n > 0.
        piston_row = QWidget(self._dynamics_config_box)
        pi_layout = QHBoxLayout(piston_row)
        pi_layout.setContentsMargins(0, 0, 0, 0)
        pi_layout.addWidget(QLabel("<b>Piston force (N):</b>"))
        self.piston_force_spin = QDoubleSpinBox(self._dynamics_config_box)
        self.piston_force_spin.setRange(0.0, 1.0e4)
        self.piston_force_spin.setDecimals(2)
        self.piston_force_spin.setSingleStep(0.5)
        self.piston_force_spin.setValue(5.0)
        self.piston_force_spin.setToolTip(
            "Total downward force a virtual piston applies to the top of "
            "the lattice (after pre-rotation via the Inspector's "
            "Orientation controls). Set to 0 to use the manual "
            "Ground face + Forces below."
        )
        self.piston_force_spin.valueChanged.connect(
            self._on_piston_force_changed)
        pi_layout.addWidget(self.piston_force_spin, 1)
        v.addWidget(piston_row)

        piston_help = QLabel(
            "<i>Pre-rotate the lattice via the orientation sliders below, "
            "then click <b>Run Dynamic</b>. The piston pushes from the "
            "top of the lattice in its current world orientation.</i>",
            self._dynamics_config_box,
        )
        piston_help.setTextFormat(Qt.TextFormat.RichText)
        piston_help.setWordWrap(True)
        v.addWidget(piston_help)

        # ---- Orientation sliders (drives lattice.rigid_rotation) -------
        # Three Euler XYZ sliders, applied extrinsic X→Y→Z (matches the
        # Inspector's orientation block). Lets the user pre-rotate the
        # lattice without leaving the dynamics workflow.
        orient_box = QGroupBox("Lattice orientation", self._dynamics_config_box)
        ov = QFormLayout(orient_box)
        self._orient_sliders: dict[str, QSlider] = {}
        self._orient_spins:   dict[str, QDoubleSpinBox] = {}
        for axis in ("X", "Y", "Z"):
            row = QWidget(orient_box)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            sl = QSlider(Qt.Orientation.Horizontal, row)
            sl.setRange(int(_ORIENT_SLIDER_MIN_DEG * _ORIENT_SLIDER_SCALE),
                        int(_ORIENT_SLIDER_MAX_DEG * _ORIENT_SLIDER_SCALE))
            sl.setSingleStep(_ORIENT_SLIDER_SCALE)
            sl.setPageStep(_ORIENT_SLIDER_SCALE * 10)
            sl.valueChanged.connect(
                lambda v, a=axis: self._on_orient_slider_changed(a, v))
            sl.sliderPressed.connect(self._on_orient_slider_pressed)
            sl.sliderReleased.connect(self._on_orient_slider_released)
            sp = QDoubleSpinBox(row)
            sp.setRange(_ORIENT_SLIDER_MIN_DEG, _ORIENT_SLIDER_MAX_DEG)
            sp.setDecimals(1)
            sp.setSingleStep(1.0)
            sp.setSuffix("°")
            sp.editingFinished.connect(self._on_orient_spin_committed)
            sp.valueChanged.connect(
                lambda v, a=axis: self._on_orient_spin_value_changed(a, v))
            sp.setToolTip(
                f"Rotation about world {axis}, applied in XYZ extrinsic order.")
            row_layout.addWidget(sl, 1)
            row_layout.addWidget(sp)
            ov.addRow(QLabel(axis), row)
            self._orient_sliders[axis] = sl
            self._orient_spins[axis] = sp
        v.addWidget(orient_box)
        self._orient_box = orient_box

        # ---- Ground face row -------------------------------------------
        gf_row = QWidget(self._dynamics_config_box)
        gf_layout = QHBoxLayout(gf_row); gf_layout.setContentsMargins(0, 0, 0, 0)
        gf_layout.addWidget(QLabel("Ground face:"))
        self.ground_face_combo = QComboBox(self._dynamics_config_box)
        for label in ("none", "+x", "-x", "+y", "-y", "+z", "-z"):
            self.ground_face_combo.addItem(label, label)
        self.ground_face_combo.setToolTip(
            "Which face of the lattice's bbox sits on the ground plane. "
            "Tiles touching that face are auto-pinned during the sim.")
        self.ground_face_combo.currentIndexChanged.connect(
            self._on_ground_face_changed)
        gf_layout.addWidget(self.ground_face_combo, 1)
        v.addWidget(gf_row)

        # ---- Force table -----------------------------------------------
        v.addWidget(QLabel("External forces:"))
        self.forces_table = QTableWidget(0, 6, self._dynamics_config_box)
        self.forces_table.setHorizontalHeaderLabels(
            ["Tile", "Vertex", "dx", "dy", "dz", "Magnitude (N)"])
        self.forces_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.forces_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.forces_table.setEditTriggers(
            QAbstractItemView.EditTrigger.AllEditTriggers)
        # Cell-changed: a single edit pushed via the
        # ``forcesChangeRequested`` signal — one undo entry per commit.
        self.forces_table.cellChanged.connect(self._on_force_cell_changed)
        v.addWidget(self.forces_table)

        # ---- Add / Remove buttons --------------------------------------
        btn_row = QWidget(self._dynamics_config_box)
        btn_layout = QHBoxLayout(btn_row); btn_layout.setContentsMargins(0, 0, 0, 0)
        self.add_force_button = QPushButton("+ Add force", btn_row)
        self.add_force_button.setToolTip(
            "Append a default force (centroid of tile 0, +x direction, 1 N)")
        self.add_force_button.clicked.connect(self._on_add_force)
        self.remove_force_button = QPushButton("− Remove selected", btn_row)
        self.remove_force_button.setToolTip(
            "Remove the highlighted row from the force list")
        self.remove_force_button.clicked.connect(self._on_remove_force)
        btn_layout.addWidget(self.add_force_button)
        btn_layout.addWidget(self.remove_force_button)
        btn_layout.addStretch(1)
        v.addWidget(btn_row)

        outer.addWidget(self._dynamics_config_box)
        # Hidden by default; only visible in Dynamic mode.
        self._dynamics_config_box.setVisible(False)

    def _build_run_row(self, outer):
        row = QWidget(self)
        layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0)
        self.run_button = QPushButton("Run Simulation", row)
        self.run_button.setToolTip(
            "Quasi-static kirigami sweep (Poisson's ratio, locking)")
        self.run_button.clicked.connect(self._start_sim_async)
        layout.addWidget(self.run_button)

        # M2 — Newtonian dynamic sim. Reads forces / ground face / config
        # from lattice.dynamics_state.
        self.run_dynamic_button = QPushButton("Run Dynamic", row)
        self.run_dynamic_button.setToolTip(
            "Newtonian rigid-body sim with gravity, contact, and any "
            "user forces from the preset's dynamics block")
        self.run_dynamic_button.clicked.connect(self.run_dynamics)
        layout.addWidget(self.run_dynamic_button)

        self.play_button = QPushButton("▶ Play", row)
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.toggled.connect(self._on_play_toggled)
        layout.addWidget(self.play_button)
        layout.addStretch(1)
        outer.addWidget(row)

    def _build_joint_slider_box(self, outer):
        box = QGroupBox("Joint angle (kirigami DOF)", self)
        form = QFormLayout(box)

        slider_row = QWidget(box)
        slider_layout = QHBoxLayout(slider_row); slider_layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider(Qt.Orientation.Horizontal, slider_row)
        self.slider.setRange(int(_SLIDER_MIN_DEG * _SLIDER_SCALE),
                              int(_SLIDER_MAX_DEG * _SLIDER_SCALE))
        self.slider.setSingleStep(_SLIDER_SCALE)
        self.slider.setPageStep(_SLIDER_SCALE * 10)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)

        self.spin = QDoubleSpinBox(slider_row)
        self.spin.setRange(_SLIDER_MIN_DEG, _SLIDER_MAX_DEG)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(1.0)
        self.spin.setSuffix("°")
        self.spin.editingFinished.connect(self._on_spin_committed)
        self.spin.valueChanged.connect(self._on_spin_changed)

        slider_layout.addWidget(self.slider, 1)
        slider_layout.addWidget(self.spin)

        form.addRow(QLabel("θ"), slider_row)
        outer.addWidget(box)

    def _build_poisson_bounds_box(self, outer):
        """Per-bound visibility toggles for the Poisson-bounds overlay.
        Eight checkboxes (default ON), one per extremal-pose box defined in
        ``views.POISSON_BOXES``: Initial, the two most-compressed poses
        (Comp ±θ), the overall expanded Footprint, and the four farthest-reach
        poses (±X / ±Y). Each checkbox's text is coloured to match its box's
        wireframe. Pure GUI visibility state — toggling a box re-renders the
        overlay; the geometry is unchanged. Laid out 4-over-4 (summary boxes /
        directional boxes)."""
        box = QGroupBox("Poisson bounds", self)
        grid = QGridLayout(box); grid.setContentsMargins(8, 4, 8, 4)
        self._poisson_bound_cbs: dict[str, QCheckBox] = {}
        for n, (key, label, color) in enumerate(POISSON_BOXES):
            cb = QCheckBox(label, box)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {color};")
            # Re-render the overlay live when a box is toggled.
            cb.toggled.connect(self._update_poisson_tracking)
            self._poisson_bound_cbs[key] = cb
            # Row 0: the four "summary" boxes (rest, the two compressions, and
            # the overall footprint). Row 1: the four directional reach boxes.
            grid.addWidget(cb, 0 if n < 4 else 1, n % 4)
        box.setToolTip(
            "Show/hide the extremal kinematic bounding boxes: the rest pose, "
            "the most-compressed pose on each rotation direction (±θ), and "
            "the farthest-reach pose in each of ±X / ±Y.")
        outer.addWidget(box)

    def _build_plot(self, outer):
        self._figure = Figure(figsize=(4.5, 2.5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_xlabel("Joint angle θ (degrees)")
        self._ax.set_ylabel("Bbox extent along load axis")
        self._ax.set_xlim(_SLIDER_MIN_DEG, _SLIDER_MAX_DEG)
        self._plot_line = None     # set once a sweep result is available
        self._plot_marker = self._ax.axvline(
            _SLIDER_REST_DEG, color="red", linestyle="--", linewidth=1, alpha=0.7,
        )
        # Shaded "unreachable due to collision" spans (M2.8). Replaced
        # on each plot update.
        self._collision_spans: list = []
        outer.addWidget(self._canvas)

    def _build_readout(self, outer):
        self.readout = QLabel(self)
        self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setWordWrap(True)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self.readout)

    # ==================================================================
    # Public API
    # ==================================================================

    def set_view_3d(self, view_3d) -> None:
        """The MainWindow may construct the panel before the View3D
        (depending on construction order); use this to inject the
        reference once it's available."""
        self._view_3d = view_3d
        self._wire_view3d(view_3d)

    def _wire_view3d(self, view_3d) -> None:
        """Connect the View3D's surface-pick signal so clicking a polygon
        anchors the kinematic view to it. Idempotent — safe to call from
        both ``__init__`` and ``set_view_3d``."""
        if view_3d is None or not hasattr(view_3d, "surfacePointPicked"):
            return
        try:
            view_3d.surfacePointPicked.disconnect(self._on_surface_point_picked)
        except (TypeError, RuntimeError):
            pass
        try:
            view_3d.surfacePointPicked.connect(self._on_surface_point_picked)
        except (TypeError, RuntimeError):
            pass

    # ==================================================================
    # Anchor view to a polygon (kinematic scrub only)
    # ==================================================================

    def _on_surface_point_picked(self, point) -> None:
        """Handle a 3D surface-pick. Resolves the clicked point to the
        nearest polygon in the displayed pose and toggles it as the anchor
        (click the anchored polygon again, or empty space, to release).
        Anchoring only affects the kinematic scrub."""
        if self._scrub_mode != "kinematic":
            return
        if point is None:
            if self._anchor_tile is not None:
                self._set_anchor_tile(None)
            return
        tile = self._resolve_clicked_tile(point)
        if tile is None:
            return
        new_anchor = None if tile == self._anchor_tile else tile
        self._set_anchor_tile(new_anchor)

    def _resolve_clicked_tile(self, point) -> int | None:
        """Nearest polygon (>=3 vertices) to ``point`` in the currently
        displayed kinematic pose. Bonds / degenerate bars are skipped so
        the anchor is always a real polygon."""
        if self._simulator is None or self._tile_system is None:
            return None
        pose = self._displayed_pose
        if pose is None:
            return None
        dim = self._tile_system.dimension
        pt = np.asarray(point, dtype=float).ravel()[:dim]
        best, best_d = None, float("inf")
        for i in range(self._tile_system.n_tiles):
            if self._tile_system.tiles[i].shape[0] < 3:
                continue
            centroid = self._simulator._tile_world_vertices(pose, i).mean(axis=0)
            d = float(np.linalg.norm(centroid - pt))
            if d < best_d:
                best_d, best = d, i
        return best

    def _set_anchor_tile(self, tile: int | None) -> None:
        """Set (or clear) the anchored polygon and re-render the current
        kinematic pose in that polygon's frame."""
        if tile == self._anchor_tile:
            return
        self._anchor_tile = tile
        self._drive_pose_from_slider(self._slider_value_deg())
        self._update_state_dependent_ui()

    def shutdown(self) -> None:
        """Stop the play timer, join any running kinematic-solve worker
        thread, and close the matplotlib figure explicitly. Called by
        ``MainWindow.closeEvent`` to release these resources synchronously
        rather than letting them ride Qt's deferred-deletion path — that's
        too slow for the test suite, where back-to-back MainWindow
        lifetimes in one process otherwise let stale ``QTimer``, worker
        threads, and matplotlib canvas state survive into the next test's
        event loop."""
        thread = self._sim_thread
        if thread is not None:
            try:
                thread.quit()
                thread.wait(2000)
            except Exception:
                pass
        if hasattr(self, "_play_timer") and self._play_timer is not None:
            try:
                self._play_timer.stop()
            except Exception:
                pass
        if hasattr(self, "_figure") and self._figure is not None:
            try:
                # Close the underlying matplotlib figure (drops the
                # FigureCanvas's strong ref to the figure backing it).
                import matplotlib.pyplot as _plt
                _plt.close(self._figure)
            except Exception:
                pass

    def set_lattice(self, lattice) -> None:
        """Re-bind to a new Lattice (e.g. after File → Open). The
        pending simulation result is invalidated since it was computed
        for the old lattice."""
        self._lattice = lattice
        self.mark_outdated()
        self.refresh_from_lattice()

    def refresh_from_lattice(self) -> None:
        """Sync the slider position from ``lattice.joint_angle`` —
        physical-degree slider mirrors mathematical-radian field via
        the convention helpers. Also drives the plot marker and (if
        a fresh result is present) the View3D pose so undo/redo of
        joint angle ends up at a coherent visual state."""
        slider_deg = simulator_theta_to_slider(float(self._lattice.joint_angle))
        slider_deg = max(_SLIDER_MIN_DEG, min(_SLIDER_MAX_DEG, slider_deg))
        self._suspend = True
        try:
            self.slider.setValue(int(round(slider_deg * _SLIDER_SCALE)))
            self.spin.setValue(slider_deg)
            # Sync the piston-force spinbox from the lattice.
            self.piston_force_spin.setValue(
                float(self._lattice.dynamics_state.get("piston_force_n", 0.0))
            )
            # Sync the ground-face combo from the lattice's dynamics_state.
            gf = self._lattice.dynamics_state.get("ground_face")
            target = "none" if gf is None else str(gf)
            gi = self.ground_face_combo.findData(target)
            if gi >= 0:
                self.ground_face_combo.setCurrentIndex(gi)
            # Sync the orientation sliders / spinboxes from
            # lattice.rigid_rotation so undo/redo keeps them in step.
            self._sync_orient_widgets_from_lattice()
        finally:
            self._suspend = False
        # Populate the force table from the current lattice. This call
        # also goes through ``_suspend`` to avoid emitting cellChanged
        # for every populated cell.
        self._populate_forces_table_from_lattice()
        self._refresh_force_glyphs()
        self._drive_pose_from_slider(slider_deg)

    def _refresh_force_glyphs(self) -> None:
        """Render (or clear) the View3D force-arrow glyphs to match
        ``lattice.dynamics_state['forces']``. Requires a cached
        TileSystem to know where each tile sits in world space — we
        keep that cached after Run Simulation / Run Dynamic, so
        glyphs only appear once a sim has been run.

        UX justification: arrow placement requires tile centroids
        which are NOT lattice point coords (each tile is a small
        triangle inside a Delaunay simplex). Building the TileSystem
        ad-hoc on every refresh is expensive; reusing the cached one
        is free and costs only a "run sim once first" requirement
        before glyphs appear.
        """
        if self._view_3d is None:
            return
        forces = self._lattice.dynamics_state.get("forces") or []
        if self._tile_system is None or not forces:
            try:
                self._view_3d.clear_force_glyphs()
            except Exception:
                pass
            return
        try:
            self._view_3d.set_force_glyphs(self._tile_system, forces)
        except Exception:
            pass

    def mark_outdated(self) -> None:
        """Called by MainWindow whenever the lattice changes (regenerate,
        edit, rotation, mode, preset load). Stops playback, drops the
        pose-driven view, and disables the slider's pose-driving path
        until the user re-runs.

        ``self._sim_result`` is kept around so the readout can still
        display the stale values dimmed below the "outdated" banner.
        The M2 dynamic-sim result is dropped outright since there's no
        equivalent "stale" view for it yet."""
        if self._is_outdated:
            return
        self._is_outdated = True
        # The lattice changed, so the anchored polygon's index may no
        # longer refer to the same tile — release the anchor (and the
        # cached displayed pose) to a clean world-frame state.
        self._anchor_tile = None
        self._displayed_pose = None
        # Dynamic sim result is invalidated — its trajectory was for the
        # pre-change lattice. Clear it; the user clicks Run Dynamic again.
        self._dynamics_result = None
        self._dynamics_error  = None
        if self._play_timer.isActive():
            self._play_timer.stop()
        self.play_button.blockSignals(True)
        self.play_button.setChecked(False)
        self.play_button.blockSignals(False)
        if self._view_3d is not None:
            try:
                self._view_3d.clear_pose()
            except Exception:
                pass
            try:
                self._view_3d.clear_piston_visualization()
            except Exception:
                pass
        self._update_state_dependent_ui()

    # ==================================================================
    # Run Simulation
    # ==================================================================

    def run_simulation(self) -> None:
        """Run the quasi-static kirigami sweep synchronously and apply the
        result. Kept synchronous for programmatic / test callers; the
        toolbar Run button instead uses the non-blocking
        :meth:`_start_sim_async`. Any simulator failure is surfaced via the
        readout, never crashes the GUI."""
        self._last_error = None
        try:
            tile_system, load_axis, is_mode_11, jam = self._build_sim_inputs()
            results = _solve_kinematic(tile_system, load_axis, is_mode_11, jam)
        except Exception as exc:
            self._apply_sim_failure(_format_sim_error(exc))
            return
        self._apply_sim_result(tile_system, *results)

    def _build_sim_inputs(self):
        """Read the live lattice into the immutable inputs the solver
        needs. Main-thread only (the lattice QObject isn't thread-safe);
        the resulting ``tile_system`` is plain numpy data safe to hand to a
        worker thread.

        The load axis is fixed in world frame at -Y. ``from_lattice``
        already applied ``lattice.world_transform()`` to the tile vertices,
        so the load axis stays put regardless of how the lattice is
        rotated — rotating the lattice rotates the tile vertices in world
        frame, leaving the load axis fixed."""
        tile_system = TileSystem.from_lattice(self._lattice)
        if tile_system.dimension == 2:
            load_axis = np.array([0.0, -1.0])
        else:
            load_axis = np.array([0.0, -1.0, 0.0])
        is_mode_11 = int(getattr(self._lattice, "mode", 0)) == 11
        jam = (float(self._lattice.bipartite_jamming_angle())
               if is_mode_11 else 0.0)
        return tile_system, load_axis, is_mode_11, jam

    def _apply_sim_result(self, tile_system, simulator, sim_result,
                          poissons, locked, info) -> None:
        """Store a successful solve and refresh the UI (main thread)."""
        self._sim_result      = sim_result
        self._tile_system     = tile_system
        self._simulator       = simulator
        self._poissons_ratio  = poissons
        # Whole-lattice edge-vector ν (geometry-only; cheap, computed once
        # here rather than per readout refresh). Captures the true auxetic
        # value for symmetric mechanisms where the bbox ν reads ~0.
        try:
            self._edge_poisson_ratio = float(
                self._lattice.edge_vector_poisson_ratio())
        except Exception:
            self._edge_poisson_ratio = None
        self._locking_info    = info
        self._is_outdated     = False
        # Drop a stale anchor whose index no longer exists (tile count
        # changed); otherwise keep it across re-runs of the same lattice.
        if (self._anchor_tile is not None
                and self._anchor_tile >= tile_system.n_tiles):
            self._anchor_tile = None
        self._update_plot()
        self._update_state_dependent_ui()
        self._update_poisson_tracking()
        self.simulationCompleted.emit()

    def _apply_sim_failure(self, error_text: str) -> None:
        """Clear stored state after a failed solve and surface the error."""
        self._last_error     = error_text
        self._sim_result     = None
        self._tile_system    = None
        self._simulator      = None
        self._poissons_ratio = None
        self._edge_poisson_ratio = None
        self._locking_info   = None
        self._is_outdated    = False
        self._update_state_dependent_ui()
        self._update_poisson_tracking()

    def _update_poisson_tracking(self) -> None:
        """Refresh the 3D Poisson-bounds overlay: eight bounding boxes — the
        rest pose, the two most axially-compressed poses on the +θ and −θ
        halves of the sweep, the four farthest-reach poses in +X / −X / +Y /
        −Y (each with the per-axis extreme points that define it), and the
        overall expanded **footprint** that encloses those four directional
        boxes. Pose selection lives in the Simulator
        (:meth:`Simulator.extremal_pose_indices`). When a polygon is anchored
        the selection is done in that polygon's frame (``anchor=`` passed
        through) AND the chosen poses are relativized to it for drawing — so
        the reach/extent picks describe the structure as it's actually shown
        and the boxes (and footprint) genuinely enclose the on-screen lattice
        at every θ. The footprint is built from the four directional boxes'
        (already-relativized) corners, so it stays correct in either frame.
        Each box's checkbox gates its visibility (pure visibility; geometry
        unchanged). Clears the overlay when there's no fresh kinematic result.
        No-op without a 3D view. All geometry comes from the Simulator
        (auxetic/), not here."""
        view = self._view_3d
        if view is None:
            return
        sim = self._simulator
        result = self._sim_result
        if sim is None or result is None or self._is_outdated:
            view.clear_poisson_tracking()
            return
        try:
            # When a polygon is anchored, _drive_pose_from_slider draws every
            # frame relativized to that polygon (relativize_pose). Select the
            # extremal poses in that SAME frame (anchor=) and draw the chosen
            # poses relativized too, so the boxes enclose what's on screen at
            # every θ. No anchor → absolute selection + absolute draw.
            anchor = self._anchor_tile
            indices = sim.extremal_pose_indices(result, anchor=anchor)

            def _visible(key: str) -> bool:
                cb = self._poisson_bound_cbs.get(key)
                return cb.isChecked() if cb is not None else True

            boxes: dict[str, tuple] = {}
            for key, idx in indices.items():
                pose = result.poses[idx]
                if anchor is not None:
                    pose = sim.relativize_pose(pose, anchor)
                boxes[key] = (
                    sim.bbox_corners(pose),
                    sim.bbox_extreme_vertices(pose),
                    _visible(key),
                )
            # Overall expanded footprint: the envelope of the four directional
            # reach boxes (their biggest +x/−x/+y/−y). Built from the corners
            # already computed above, so it inherits whatever frame they're in
            # (absolute or anchor-relativized) for free. No per-axis extreme
            # points — it's a synthetic envelope, not a single pose.
            directional = [boxes[k][0] for k in (
                "expansion_pos_x", "expansion_neg_x",
                "expansion_pos_y", "expansion_neg_y")]
            boxes["footprint"] = (
                sim.aabb_corners_enclosing(directional),
                None,
                _visible("footprint"),
            )
            view.show_poisson_tracking(boxes)
        except Exception:
            view.clear_poisson_tracking()

    # ---- Non-blocking solve (toolbar Run button) ---------------------
    def _start_sim_async(self) -> None:
        """Run the kinematic solve on a worker thread so the UI stays
        responsive. A second click while running cancels: the result is
        discarded and the UI frees up when the thread finishes — the thread
        is allowed to run to completion, never force-killed (unsafe in Qt)."""
        if self._sim_thread is not None:
            # Already running → this click requests cancel.
            self._sim_cancelled = True
            self.run_button.setEnabled(False)
            self.run_button.setText("Cancelling…")
            return
        self._last_error = None
        try:
            tile_system, load_axis, is_mode_11, jam = self._build_sim_inputs()
        except Exception as exc:
            self._apply_sim_failure(_format_sim_error(exc))
            return

        self._sim_cancelled = False
        self.run_button.setText("Cancel")
        self.run_button.setToolTip("Cancel the running simulation.")

        thread = QThread(self)
        worker = _SimWorker(tile_system, load_axis, is_mode_11, jam)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_sim_finished)
        worker.failed.connect(self._on_sim_failed)
        # Cleanup chain — quit the thread when the worker is done, then
        # delete both objects (mirrors predictor_panel).
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_sim_thread_finished)
        self._sim_thread = thread
        self._sim_worker = worker
        thread.start()

    def _on_sim_finished(self, payload) -> None:
        if self._sim_cancelled:
            return   # result abandoned; UI resets in _on_sim_thread_finished
        self._apply_sim_result(*payload)

    def _on_sim_failed(self, message: str) -> None:
        if self._sim_cancelled:
            return
        self._apply_sim_failure(message)

    def _on_sim_thread_finished(self) -> None:
        # Thread + worker are scheduled for deletion via deleteLater;
        # clear refs and restore the Run button.
        self._sim_thread    = None
        self._sim_worker    = None
        self._sim_cancelled = False
        self.run_button.setText("Run Simulation")
        self.run_button.setToolTip(
            "Quasi-static kirigami sweep (Poisson's ratio, locking)")
        self.run_button.setEnabled(True)

    def run_dynamics(self) -> None:
        """Click handler for "Run Dynamic" (M2). Wraps
        :func:`auxetic.dynamics.build_dynamics_simulator_from_lattice`
        and stores the resulting :class:`DynamicsResult`. Like
        ``run_simulation`` this is exception-safe."""
        self._dynamics_error = None
        try:
            ds = build_dynamics_simulator_from_lattice(self._lattice)
            self._dynamics_result = ds.simulate()
            # Stash the tile system so the slider can drive the View3D
            # without rebuilding it. The dynamics solver and kinematic
            # solver use the same ``TileSystem.from_lattice`` output.
            self._tile_system = ds.tile_system
        except Exception as exc:
            self._dynamics_error = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc().splitlines()[-1]}"
            )
            self._dynamics_result = None
        # If the user ran Dynamic explicitly, switch the scrub to
        # Dynamic mode so they can see the trajectory.
        if self._dynamics_result is not None and not self.mode_dynamic_radio.isChecked():
            self.mode_dynamic_radio.setChecked(True)
        # Snap the scrub slider to 0 (start of trajectory) so the user
        # sees the un-compressed initial pose with the piston plate at
        # its starting height. From there they can scrub forward, click
        # Play, or watch the floor + piston move as compression
        # progresses. Without this reset, a previous slider position
        # (e.g. 90° from a kinematic sweep) would land us mid-trajectory
        # and the piston would appear already partially descended.
        if self._dynamics_result is not None:
            self._suspend = True
            try:
                self.slider.setValue(int(round(_SLIDER_MIN_DEG * _SLIDER_SCALE)))
                self.spin.setValue(_SLIDER_MIN_DEG)
            finally:
                self._suspend = False
        self._update_plot()
        self._drive_pose_from_slider(self._slider_value_deg())
        self._update_state_dependent_ui()

    # ==================================================================
    # Plot rendering
    # ==================================================================

    def _update_plot(self) -> None:
        # Pick what to plot based on the current scrub mode.
        if self._scrub_mode == "dynamic":
            self._update_plot_dynamic()
            return
        self._update_plot_kinematic()

    def _update_plot_kinematic(self) -> None:
        self._ax.set_xlabel("Joint angle θ (degrees)")
        # Drop any prior collision-shading spans before redrawing.
        for span in self._collision_spans:
            try:
                span.remove()
            except Exception:
                pass
        self._collision_spans.clear()

        if self._sim_result is None:
            if self._plot_line is not None:
                self._plot_line.remove()
                self._plot_line = None
            self._canvas.draw_idle()
            return

        axial = self._simulator._axial_index()
        is_mode11 = int(getattr(self._lattice, "mode", 0)) == 11
        if is_mode11:
            # Mode 11's sweep is parameterised by physical actuation
            # (closure) angle, mapped to the slider the same way
            # _mode11_pose_index_for_slider does: 90°→rest, 180°→+jamming.
            jam_deg = math.degrees(float(self._lattice.bipartite_jamming_angle()))
            if jam_deg < 1e-6:
                jam_deg = 90.0
            x_deg = (_SLIDER_REST_DEG
                     + np.degrees(self._sim_result.actuation_angles)
                     / jam_deg * 90.0)
        else:
            x_deg = np.array([
                simulator_theta_to_slider(t) for t in self._sim_result.theta_samples
            ])
        y = self._sim_result.bbox_extents[:, axial]

        if self._plot_line is None:
            (self._plot_line,) = self._ax.plot(x_deg, y, color="steelblue")
        else:
            self._plot_line.set_data(x_deg, y)

        # M2.8 — shade the θ ranges where tile-tile collisions block
        # further rotation, so the user sees what's reachable.
        result = self._sim_result
        if result.collision_theta_min is not None:
            x_lo = simulator_theta_to_slider(_SLIDER_MIN_DEG_THETA_MIN)
            x_hi = simulator_theta_to_slider(result.collision_theta_min)
            self._collision_spans.append(
                self._ax.axvspan(x_lo, x_hi, color="#dd5050", alpha=0.15)
            )
        if result.collision_theta_max is not None:
            x_lo = simulator_theta_to_slider(result.collision_theta_max)
            x_hi = simulator_theta_to_slider(_SLIDER_MAX_DEG_THETA_MAX)
            self._collision_spans.append(
                self._ax.axvspan(x_lo, x_hi, color="#dd5050", alpha=0.15)
            )

        # Mode 11 (bipartite): its mechanism stops at the jamming angle =
        # the overlap onset, so there is no overlapping region inside the
        # swept range to flag the way the sweep_theta modes do. Shade the
        # area just past the reachable ends red ("polygons overlap past
        # here") and widen the x-limits so that margin is visible.
        mode11_xlim = None
        if is_mode11:
            spans, mode11_xlim = _mode11_overlap_spans(x_deg)
            for a, b in spans:
                self._collision_spans.append(
                    self._ax.axvspan(a, b, color="#dd5050", alpha=0.15))

        self._ax.relim(); self._ax.autoscale_view(scalex=False, scaley=True)
        if mode11_xlim is not None:
            self._ax.set_xlim(*mode11_xlim)
        else:
            self._ax.set_xlim(_SLIDER_MIN_DEG, _SLIDER_MAX_DEG)
        self._canvas.draw_idle()

    def _update_plot_dynamic(self) -> None:
        """Plot bbox extent (along the dynamic sim's load axis) vs.
        slider scrub position. The slider here is a 0–100% time
        cursor, mapped onto ``[_SLIDER_MIN_DEG, _SLIDER_MAX_DEG]`` so
        the same widget can drive either trajectory."""
        self._ax.set_xlabel("Time (slider 0–100%)")
        if self._dynamics_result is None:
            if self._plot_line is not None:
                self._plot_line.remove()
                self._plot_line = None
            self._canvas.draw_idle()
            return
        r = self._dynamics_result
        # Pick the same axial direction the dynamics solver used (gravity
        # / load-axis convention: axis-1 in 2D, axis-1 in 3D).
        axial = 1
        n = r.bbox_extents.shape[0]
        if n < 2:
            return
        scrub_pos = np.linspace(_SLIDER_MIN_DEG, _SLIDER_MAX_DEG, n)
        y = r.bbox_extents[:, axial]
        if self._plot_line is None:
            (self._plot_line,) = self._ax.plot(scrub_pos, y, color="darkorange")
        else:
            self._plot_line.set_data(scrub_pos, y)
            self._plot_line.set_color("darkorange")
        self._ax.relim(); self._ax.autoscale_view(scalex=False, scaley=True)
        self._ax.set_xlim(_SLIDER_MIN_DEG, _SLIDER_MAX_DEG)
        self._canvas.draw_idle()

    def _update_marker(self, slider_deg: float) -> None:
        self._plot_marker.set_xdata([slider_deg, slider_deg])
        self._canvas.draw_idle()

    # ==================================================================
    # Readout / state-dependent UI
    # ==================================================================

    def _update_state_dependent_ui(self) -> None:
        """Update the readout text, plot dimming, and Play-button
        enabled state based on (sim_result presence, outdated flag,
        last_error). Single source of truth for "what does the panel
        look like right now"."""
        has_result = self._sim_result is not None
        fresh = has_result and not self._is_outdated
        # Play is also available in Dynamic mode when a fresh dynamics
        # result is loaded — animates the piston pressing down through
        # the trajectory.
        has_dyn = self._dynamics_result is not None
        play_enabled = fresh or has_dyn

        self.play_button.setEnabled(play_enabled)
        # Plot greys out when outdated — visually consistent with the
        # readout's dimmed-stale pattern.
        if self._plot_line is not None:
            self._plot_line.set_alpha(1.0 if (fresh or has_dyn) else 0.3)
            self._canvas.draw_idle()

        self.readout.setText(self._compose_readout_html())

    def _compose_readout_html(self) -> str:
        anchor_html    = self._compose_anchor_readout_html()
        kinematic_html = self._compose_kinematic_readout_html()
        dynamic_html   = self._compose_dynamic_readout_html()
        return anchor_html + kinematic_html + dynamic_html

    def _compose_anchor_readout_html(self) -> str:
        """A small banner naming the anchored polygon, shown while the
        kinematic view is locked to a tile's frame."""
        if self._anchor_tile is None:
            return ""
        return (
            f"<p style='color:#b8860b'><b>View anchored to polygon "
            f"#{self._anchor_tile}</b> — click it again (or empty space) "
            f"to release.</p>"
        )

    def _compose_kinematic_readout_html(self) -> str:
        if self._last_error is not None:
            return (
                f"<p style='color:#b00020'><b>Simulation failed</b></p>"
                f"<pre style='font-size:9pt'>{self._last_error}</pre>"
            )
        if self._sim_result is None:
            return "<p><i>Click Run Simulation to begin.</i></p>"

        # Success rendering (fresh) or stale (outdated).
        nu_html       = self._format_poissons_ratio_html()
        edge_nu_html  = self._format_edge_poisson_html()
        lock_html     = self._format_locked_html()
        comp_pct      = (self._locking_info or {}).get("compression_ratio", 0.0) * 100.0
        proj          = (self._locking_info or {}).get("mode_projection",   0.0)
        body = (
            "<table style='border-spacing:0;'>"
            f"<tr><td><b>Full-structure ν (edge-vector):</b></td>"
            f"<td>{edge_nu_html}</td></tr>"
            f"<tr><td><b>Poisson's ratio (bbox):</b></td><td>{nu_html}</td></tr>"
            f"<tr><td><b>Locked status:</b></td><td>{lock_html}</td></tr>"
            f"<tr><td><b>Compression ratio:</b></td><td>{comp_pct:.1f}%</td></tr>"
            f"<tr><td><b>Mode projection:</b></td><td>{proj:.3f}</td></tr>"
            "</table>"
        )
        if self._is_outdated:
            return (
                "<p style='color:#b00020'><b>Simulation outdated — "
                "click Run Simulation again.</b></p>"
                f"<div style='opacity:0.5'>{body}</div>"
            )
        return body

    def _compose_dynamic_readout_html(self) -> str:
        """Format the dynamic-sim result, if any. Appended below the
        kinematic readout so users see both."""
        if self._dynamics_error is not None:
            return (
                f"<hr><p style='color:#b00020'>"
                f"<b>Dynamic sim failed</b></p>"
                f"<pre style='font-size:9pt'>{self._dynamics_error}</pre>"
            )
        if self._dynamics_result is None:
            return ""
        r = self._dynamics_result
        comp_pct = float(r.final_compression) * 100.0
        ke_final = float(r.energy_trace["kinetic"][-1])
        converged_html = (
            "<span style='color:#2c7a2c'>yes</span>" if r.converged
            else "<span style='color:#b00020'>no</span>"
        )
        return (
            "<hr>"
            "<p><b>Dynamic sim</b></p>"
            "<table style='border-spacing:0;'>"
            f"<tr><td><b>Final compression:</b></td><td>{comp_pct:.1f}%</td></tr>"
            f"<tr><td><b>Final KE:</b></td><td>{ke_final:.3e}</td></tr>"
            f"<tr><td><b>Converged:</b></td><td>{converged_html}</td></tr>"
            f"<tr><td><b>Steps:</b></td><td>{len(r.times)}</td></tr>"
            "</table>"
        )

    def _format_poissons_ratio_html(self) -> str:
        nu = self._poissons_ratio
        if nu is None:
            return "—"
        if isinstance(nu, tuple):
            # 3D: three values, NaN at axial index becomes "—".
            labels = ("ν_x", "ν_y", "ν_z")
            cells  = []
            for label, v in zip(labels, nu):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append(f"{label} = — <i>(load axis)</i>")
                else:
                    cells.append(f"{label} = {float(v):.4f}")
            return "<br>".join(cells)
        # 2D: scalar.
        if isinstance(nu, float) and math.isnan(nu):
            return "— <i>(no axial extension)</i>"
        return f"{float(nu):.4f}"

    def _format_edge_poisson_html(self) -> str:
        """Whole-lattice edge-vector generalized Poisson's ratio (mean over
        all triangles) — geometry-only, distinct from the bbox ν above. For
        symmetric rotating-units mechanisms (e.g. EqHex) the bbox ν can read
        ~0 while this captures the true auxetic value (-1 for equilateral
        tiles). Ctrl-click a triangle in the 3D view for one triangle's ν."""
        nu = self._edge_poisson_ratio
        if nu is None:
            return "—"
        if isinstance(nu, float) and math.isnan(nu):
            return "— <i>(3D / no 2D triangles)</i>"
        return f"{float(nu):+.4f}"

    def _format_locked_html(self) -> str:
        info = self._locking_info or {}
        locked = info.get("locked")
        # ``is_locked`` returns ``(bool, dict)`` separately; we stored
        # only the dict, so derive locked from the reason string.
        reason = info.get("reason", "")
        if reason and reason != "not locked":
            return f"<span style='color:#b00020'>Locked: {reason}</span>"
        return "<span style='color:#107010'>Not locked</span>"

    # ==================================================================
    # Slider behavior — delegates to pose-driving when fresh
    # ==================================================================

    @property
    def is_outdated(self) -> bool:
        return self._is_outdated

    def _slider_value_deg(self) -> float:
        return float(self.slider.value()) / _SLIDER_SCALE

    def _on_slider_pressed(self) -> None:
        # Stash the angle at press so release can emit (old, new).
        self._press_angle_rad = float(self._lattice.joint_angle)

    def _on_slider_released(self) -> None:
        if self._suspend or self._press_angle_rad is None:
            return
        new_rad = slider_to_simulator_theta(self._slider_value_deg())
        old_rad = self._press_angle_rad
        self._press_angle_rad = None
        if abs(new_rad - old_rad) < 1e-12:
            return
        self.jointAngleChangeRequested.emit(old_rad, new_rad)

    def _on_slider_changed(self, value: int) -> None:
        if self._suspend:
            return
        self._suspend = True
        try:
            slider_deg = self._slider_value_deg()
            self.spin.setValue(slider_deg)
        finally:
            self._suspend = False
        self._drive_pose_from_slider(slider_deg)

    def _on_spin_changed(self, value: float) -> None:
        if self._suspend:
            return
        self._suspend = True
        try:
            self.slider.setValue(int(round(value * _SLIDER_SCALE)))
        finally:
            self._suspend = False
        self._drive_pose_from_slider(value)

    def _on_spin_committed(self) -> None:
        if self._suspend:
            return
        new_rad = slider_to_simulator_theta(self.spin.value())
        old_rad = float(self._lattice.joint_angle)
        if abs(new_rad - old_rad) < 1e-12:
            return
        self.jointAngleChangeRequested.emit(old_rad, new_rad)

    def _drive_pose_from_slider(self, slider_deg: float) -> None:
        """If a fresh simulation result is present, render the pose at
        the trajectory sample nearest the slider's position. Always
        update the plot marker (so the marker still tracks the slider
        even when no result is available).

        Dispatches on ``self._scrub_mode``:

        - ``"kinematic"`` — slider degrees map to θ via the SPEC §6.2
          convention helpers; pose comes from
          ``self._sim_result.poses``.
        - ``"dynamic"``   — slider degrees are treated as a 0–100% scrub
          across the trajectory time grid; pose comes from
          ``self._dynamics_result.poses``.
        """
        self._update_marker(slider_deg)
        if self._view_3d is None:
            return

        if self._scrub_mode == "dynamic":
            if self._dynamics_result is None:
                return
            poses = self._dynamics_result.poses
            n = poses.shape[0]
            if n == 0:
                return
            frac = max(0.0, min(1.0,
                                 (slider_deg - _SLIDER_MIN_DEG)
                                 / max(_SLIDER_MAX_DEG - _SLIDER_MIN_DEG, 1e-9)))
            idx = int(round(frac * (n - 1)))
            # Defensive: even with the simulator's divergence guard,
            # a stale / partially-clamped trajectory might still hold a
            # non-finite pose. Walk backward to the nearest finite one
            # so the renderer never receives Inf/NaN.
            if not np.all(np.isfinite(poses[idx])):
                while idx > 0 and not np.all(np.isfinite(poses[idx])):
                    idx -= 1
            try:
                # Reuse the kinematic tile_system if present (the
                # dynamics solver is built on the same TileSystem).
                ts = self._tile_system
                if ts is None:
                    from auxetic import TileSystem
                    ts = TileSystem.from_lattice(self._lattice)
                self._view_3d.show_pose(ts, poses[idx])
                # M3-polish: ground + piston plate visualisation.
                # Static ground at the initial-pose bottom; piston at
                # the current pose's top so it tracks the compression.
                # Only meaningful when piston mode is active (the
                # default workflow); skip for manual mode.
                if float(
                    self._lattice.dynamics_state.get("piston_force_n", 0.0)
                ) > 0.0:
                    self._view_3d.set_piston_visualization(
                        ts, poses[idx], initial_pose=poses[0],
                    )
                else:
                    self._view_3d.clear_piston_visualization()
            except Exception:
                pass
            return
        # Kinematic (default) — clear any leftover piston plates from
        # a previous dynamic-mode scrub.
        if self._view_3d is not None:
            try:
                self._view_3d.clear_piston_visualization()
            except Exception:
                pass
        if self._sim_result is None or self._is_outdated:
            return
        sr = self._sim_result
        if getattr(self._lattice, "mode", None) == 11:
            # Mode 11's sweep is parameterised by physical closure angle;
            # map the slider to it (90°→rest, 180°→jamming) and pick the
            # nearest swept pose by actuation (see the helper).
            idx = self._mode11_pose_index_for_slider(slider_deg)
        else:
            theta_rad = slider_to_simulator_theta(slider_deg)
            idx = int(np.argmin(np.abs(sr.theta_samples - theta_rad)))
        # When a polygon is anchored, render every frame in that polygon's
        # frame so it stays fixed and the rest moves relative to it.
        raw_pose = sr.poses[idx]
        if self._anchor_tile is not None and self._simulator is not None:
            pose = self._simulator.relativize_pose(raw_pose, self._anchor_tile)
        else:
            pose = raw_pose
        self._displayed_pose = np.asarray(pose, dtype=float)
        try:
            self._view_3d.show_pose(self._tile_system, pose,
                                    highlight_tile=self._anchor_tile)
        except Exception:
            pass

    def _mode11_pose_index_for_slider(self, slider_deg: float) -> int:
        """Pose index whose *physical actuation* (closure) angle matches
        the slider, for the mode-11 manifold-following sweep.

        The slider maps linearly onto the mechanism's reachable range:
        90°→rest (0 closure), 180°→+jamming (full closure, holes shut),
        0°→−jamming. Poses are picked by their recorded
        ``actuation_angles`` — the kite-vs-central relative rotation, i.e.
        what the eye reads as "how far the units turned". Selecting on
        that (rather than a single tile's rotation, which is ~half the
        closure because the two families counter-rotate) is what removes
        the old factor-of-2: slider 135° now shows ~45° of closure, not
        ~90°."""
        sr = self._sim_result
        act_deg = np.degrees(np.asarray(sr.actuation_angles, dtype=float))
        if act_deg.size == 0:
            return 0
        jam_deg = math.degrees(float(self._lattice.bipartite_jamming_angle()))
        if jam_deg < 1e-6:
            return int(np.argmin(np.abs(act_deg)))
        target = (float(slider_deg) - _SLIDER_REST_DEG) / 90.0 * jam_deg
        return int(np.argmin(np.abs(act_deg - target)))

    # ==================================================================
    # Mode toggle handler
    # ==================================================================

    def _on_mode_toggled(self, button_id: int, checked: bool) -> None:
        if not checked:
            return   # only react to the "becoming-checked" half of the toggle
        self._scrub_mode = "dynamic" if button_id == 1 else "kinematic"
        self._dynamics_config_box.setVisible(self._scrub_mode == "dynamic")
        # Stop any running playback — the tick handler reads scrub_mode
        # at the top, so a stale cycle from the previous mode would
        # otherwise keep firing.
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.blockSignals(True)
            self.play_button.setChecked(False)
            self.play_button.setText("▶ Play")
            self.play_button.blockSignals(False)
        # Refresh the plot for the new mode and re-drive the slider so
        # the View3D pose updates to the right trajectory.
        self._update_plot()
        self._drive_pose_from_slider(self._slider_value_deg())
        self._update_state_dependent_ui()

    # ==================================================================
    # Ground-face handler
    # ==================================================================

    def _on_piston_force_changed(self, new_value: float) -> None:
        """Write through to ``lattice.dynamics_state['piston_force_n']``
        and invalidate any stale dynamics result so the readout
        doesn't lie about the new load case. Mirrors the ground-face
        handler — direct mutation, not undo-stack-tracked (the spinbox
        is a config knob, not a geometry edit)."""
        if self._suspend:
            return
        new_value = float(new_value)
        if self._lattice.dynamics_state.get("piston_force_n", 0.0) == new_value:
            return
        self._lattice.dynamics_state["piston_force_n"] = new_value
        self._dynamics_result = None
        self._dynamics_error  = None
        self._update_state_dependent_ui()

    def _on_ground_face_changed(self, _idx: int) -> None:
        if self._suspend:
            return
        label = str(self.ground_face_combo.currentData())
        new_value = None if label == "none" else label
        # Direct mutation of dynamics_state — this isn't a
        # geometry-changing edit so it doesn't go through the undo
        # stack.
        if self._lattice.dynamics_state.get("ground_face") != new_value:
            self._lattice.dynamics_state["ground_face"] = new_value
            # Clear stale dynamic result; user must rerun.
            self._dynamics_result = None
            self._dynamics_error  = None
            self._update_state_dependent_ui()

    # ==================================================================
    # Orientation slider handlers
    # ==================================================================

    def _build_rotation_from_orient_widgets(self) -> Rotation:
        """Read the three orientation spinboxes back into a scipy
        ``Rotation``. Slider and spin are kept in lock-step via
        ``_on_orient_slider_changed`` / ``_on_orient_spin_value_changed``,
        so the spin values are the canonical source."""
        x = float(self._orient_spins["X"].value())
        y = float(self._orient_spins["Y"].value())
        z = float(self._orient_spins["Z"].value())
        return Rotation.from_euler("xyz", [x, y, z], degrees=True)

    def _on_orient_slider_pressed(self) -> None:
        # Stash the current rotation so release can emit (old, new).
        self._press_rotation = self._lattice.rigid_rotation

    def _on_orient_slider_changed(self, axis: str, value: int) -> None:
        if self._suspend:
            return
        self._suspend = True
        try:
            deg = float(value) / _ORIENT_SLIDER_SCALE
            self._orient_spins[axis].setValue(deg)
        finally:
            self._suspend = False

    def _on_orient_slider_released(self) -> None:
        if self._suspend:
            return
        old = self._press_rotation
        self._press_rotation = None
        if old is None:
            old = self._lattice.rigid_rotation
        new = self._build_rotation_from_orient_widgets()
        if _rotations_close(old, new):
            return
        self.rotationChangeRequested.emit(old, new)

    def _on_orient_spin_value_changed(self, axis: str, value: float) -> None:
        if self._suspend:
            return
        self._suspend = True
        try:
            sl = self._orient_sliders[axis]
            sl.setValue(int(round(value * _ORIENT_SLIDER_SCALE)))
        finally:
            self._suspend = False

    def _on_orient_spin_committed(self) -> None:
        if self._suspend:
            return
        new = self._build_rotation_from_orient_widgets()
        old = self._lattice.rigid_rotation
        if _rotations_close(old, new):
            return
        self.rotationChangeRequested.emit(old, new)

    def _sync_orient_widgets_from_lattice(self) -> None:
        """Pull ``lattice.rigid_rotation`` back into the orientation
        sliders + spinboxes. Decompose as XYZ extrinsic to match the
        Inspector's spinboxes; suppress scipy's gimbal-lock warning so
        Top/Front/Side preset rotations don't spam test output."""
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                x, y, z = self._lattice.rigid_rotation.as_euler(
                    "xyz", degrees=True)
        except Exception:
            x = y = z = 0.0
        for axis, deg in (("X", x), ("Y", y), ("Z", z)):
            self._orient_spins[axis].setValue(float(deg))
            self._orient_sliders[axis].setValue(
                int(round(float(deg) * _ORIENT_SLIDER_SCALE)))

    # ==================================================================
    # Force-table handlers (M2.9)
    # ==================================================================

    def _populate_forces_table_from_lattice(self) -> None:
        """Sync the table widget from ``lattice.dynamics_state['forces']``.
        Called by ``refresh_from_lattice`` and after every force-list
        commit so undo/redo reflects in the table."""
        forces = list(self._lattice.dynamics_state.get("forces") or [])
        # ``cellChanged`` fires for every setItem call; suspend during
        # the populate so we don't ping-pong commits.
        self._suspend = True
        try:
            self.forces_table.setRowCount(len(forces))
            for row, f in enumerate(forces):
                self._populate_forces_row(row, f)
        finally:
            self._suspend = False

    def _populate_forces_row(self, row: int, f: dict) -> None:
        """Populate one row of the table from a force dict."""
        # tile_index → "Tile" cell
        self.forces_table.setItem(
            row, 0, QTableWidgetItem(str(int(f.get("tile_index", 0)))))
        # vert_index → "Vertex" cell (-1 means centroid)
        v_idx = int(f.get("vert_index", -1))
        # If the dict's location_kind says centroid, force vert to -1.
        if str(f.get("location_kind", "tile_centroid")) == "tile_centroid":
            v_idx = -1
        self.forces_table.setItem(
            row, 1, QTableWidgetItem(str(v_idx)))
        # direction (3 components)
        d = list(f.get("direction") or [1.0, 0.0, 0.0])
        while len(d) < 3:
            d.append(0.0)
        for k in range(3):
            self.forces_table.setItem(
                row, 2 + k, QTableWidgetItem(f"{float(d[k]):.4f}"))
        # magnitude
        self.forces_table.setItem(
            row, 5, QTableWidgetItem(f"{float(f.get('magnitude', 1.0)):.4f}"))

    def _build_forces_from_table(self) -> list:
        """Read the table widget back into a list of force dicts.
        Invalid cells fall back to sensible defaults so users can edit
        partial rows without crashing."""
        out = []
        n_rows = self.forces_table.rowCount()
        for row in range(n_rows):
            try:
                tile_idx = int(self.forces_table.item(row, 0).text())
            except (AttributeError, ValueError):
                tile_idx = 0
            try:
                vert_idx = int(self.forces_table.item(row, 1).text())
            except (AttributeError, ValueError):
                vert_idx = -1
            d = []
            for k in range(3):
                try:
                    d.append(float(self.forces_table.item(row, 2 + k).text()))
                except (AttributeError, ValueError):
                    d.append(0.0)
            try:
                mag = float(self.forces_table.item(row, 5).text())
            except (AttributeError, ValueError):
                mag = 1.0
            # Auto-classify location_kind based on vert_idx.
            kind = "tile_vertex" if vert_idx >= 0 else "tile_centroid"
            # ForceVector requires non-zero direction — default to +x
            # if the user's edit zeroed everything out.
            if all(abs(v) < 1e-12 for v in d):
                d = [1.0, 0.0, 0.0]
            out.append({
                "location_kind": kind,
                "tile_index":    tile_idx,
                "vert_index":    vert_idx,
                "direction":     d,
                "magnitude":     mag,
            })
        return out

    def _emit_forces_change_if_modified(self) -> None:
        """Compare the table's reconstructed force list against what's
        currently on the lattice. If they differ, emit
        ``forcesChangeRequested`` so the MainWindow can wrap the diff
        in an undoable command."""
        if self._suspend:
            return
        old = list(self._lattice.dynamics_state.get("forces") or [])
        new = self._build_forces_from_table()
        if old != new:
            self.forcesChangeRequested.emit(old, new)

    def _on_force_cell_changed(self, _row: int, _col: int) -> None:
        self._emit_forces_change_if_modified()

    def _on_add_force(self) -> None:
        """Append a sensible default force (tile 0 centroid, +x, 1 N)."""
        old = list(self._lattice.dynamics_state.get("forces") or [])
        new = old + [{
            "location_kind": "tile_centroid",
            "tile_index":    0,
            "vert_index":    -1,
            "direction":     [1.0, 0.0, 0.0],
            "magnitude":     1.0,
        }]
        self.forcesChangeRequested.emit(old, new)

    def _on_remove_force(self) -> None:
        """Remove the currently selected row from the force list."""
        row = self.forces_table.currentRow()
        if row < 0:
            return
        old = list(self._lattice.dynamics_state.get("forces") or [])
        if not (0 <= row < len(old)):
            return
        new = list(old[:row]) + list(old[row + 1:])
        self.forcesChangeRequested.emit(old, new)

    # ==================================================================
    # Play / animate
    # ==================================================================

    def _on_play_toggled(self, checked: bool) -> None:
        # In Dynamic mode play sweeps the slider 0%→100% so the user
        # sees the piston pressing in from rest; in Kinematic mode it
        # oscillates through the full bistable cycle.
        kine_ready = (self._sim_result is not None
                      and not self._is_outdated
                      and self._scrub_mode == "kinematic")
        dyn_ready  = (self._dynamics_result is not None
                      and self._scrub_mode == "dynamic")
        if checked and (kine_ready or dyn_ready):
            self._play_phase = 0
            self._play_timer.start()
            self.play_button.setText("⏸ Pause")
        else:
            self._play_timer.stop()
            self.play_button.setText("▶ Play")
            # If we got toggled OFF, leave slider where it landed.

    def _on_play_tick(self) -> None:
        if self._scrub_mode == "dynamic":
            # Linear forward sweep, then loop. 0% → 100% over the cycle.
            self._play_phase = (self._play_phase + 1) % _PLAY_CYCLE_FRAMES
            frac = self._play_phase / max(_PLAY_CYCLE_FRAMES - 1, 1)
            slider_deg = _SLIDER_MIN_DEG + frac * (
                _SLIDER_MAX_DEG - _SLIDER_MIN_DEG)
        else:
            # 90 frames/cycle, sine oscillation:
            # phase 0   : slider 90° (rest)
            # phase π/2 : slider 180° (compressed-B)
            # phase π   : slider 90° (back through rest)
            # phase 3π/2: slider 0° (compressed-A)
            # phase 2π  : slider 90° (cycle complete)
            self._play_phase = (self._play_phase + 1) % _PLAY_CYCLE_FRAMES
            phase = 2.0 * math.pi * (self._play_phase / _PLAY_CYCLE_FRAMES)
            slider_deg = _SLIDER_REST_DEG + 90.0 * math.sin(phase)
        # Programmatic slider move — _on_slider_changed will drive
        # the pose via _drive_pose_from_slider.
        self._suspend = True
        try:
            self.slider.setValue(int(round(slider_deg * _SLIDER_SCALE)))
            self.spin.setValue(slider_deg)
        finally:
            self._suspend = False
        # Drive pose explicitly (we suppressed it via _suspend above).
        self._drive_pose_from_slider(slider_deg)
