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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QPushButton,
)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure

from auxetic import TileSystem, Simulator
from auxetic.dynamics import build_dynamics_simulator_from_lattice


# ---------------------------------------------------------------------------
# Convention helpers — the ONLY place slider↔simulator conversion lives.
# Per SPEC §6.2, slider is physical degrees, simulator is mathematical
# radians. ``CLAUDE.md`` "Conventions worth knowing" forbids any third
# convention; if you find yourself writing ``theta * 2`` or
# ``theta + pi/2`` outside these two functions, you're either
# reinventing the wheel or adding a bug.
# ---------------------------------------------------------------------------

def slider_to_simulator_theta(slider_deg: float) -> float:
    """Slider [0, 180] degrees -> simulator [-pi/2, +pi/2] radians.

    See SPEC §6.2 for the two-parameterization convention.
    """
    return float(np.radians(slider_deg - 90.0))


def simulator_theta_to_slider(theta_rad: float) -> float:
    """Simulator [-pi/2, +pi/2] radians -> slider [0, 180] degrees.

    See SPEC §6.2 for the two-parameterization convention.
    """
    return float(np.degrees(theta_rad) + 90.0)


# Slider range and resolution (sub-degree granularity via tenths-of-deg
# integer values, since QSlider doesn't do floats natively).
_SLIDER_MIN_DEG  = 0.0
_SLIDER_MAX_DEG  = 180.0
_SLIDER_REST_DEG = 90.0
_SLIDER_SCALE    = 10  # int slider value = degrees × 10

# Animation: 30 fps, full bistable cycle in ~3 s.
_PLAY_FPS_INTERVAL_MS = 33
_PLAY_CYCLE_FRAMES    = 90


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

        # ---- simulation result state -------------------------------------
        self._sim_result      = None
        self._tile_system     = None
        self._simulator       = None
        self._poissons_ratio  = None
        self._locking_info    = None
        self._is_outdated     = False     # True after lattice change until rerun
        self._last_error      = None      # exception text if last run failed

        # ---- M2 dynamic-sim state ----------------------------------------
        self._dynamics_result = None      # most recent DynamicsResult, if any
        self._dynamics_error  = None      # exception text from last Run Dynamic

        # ---- play state --------------------------------------------------
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(_PLAY_FPS_INTERVAL_MS)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._play_phase = 0              # frame counter for the cycle

        # ---- widgets -----------------------------------------------------
        body = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(8, 8, 8, 8)

        self._build_run_row(outer)
        self._build_joint_slider_box(outer)
        self._build_plot(outer)
        self._build_readout(outer)

        outer.addStretch(1)
        self.setWidget(body)

        self.refresh_from_lattice()
        self._update_state_dependent_ui()

    # ==================================================================
    # Construction helpers
    # ==================================================================

    def _build_run_row(self, outer):
        row = QWidget(self)
        layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0)
        self.run_button = QPushButton("Run Simulation", row)
        self.run_button.setToolTip(
            "Quasi-static kirigami sweep (Poisson's ratio, locking)")
        self.run_button.clicked.connect(self.run_simulation)
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

    def shutdown(self) -> None:
        """Stop the play timer and close the matplotlib figure
        explicitly. Called by ``MainWindow.closeEvent`` to release
        these resources synchronously rather than letting them ride
        Qt's deferred-deletion path — that's too slow for the test
        suite, where back-to-back MainWindow lifetimes in one process
        otherwise let stale ``QTimer`` and matplotlib canvas state
        survive into the next test's event loop."""
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
        finally:
            self._suspend = False
        self._drive_pose_from_slider(slider_deg)

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
        self._update_state_dependent_ui()

    # ==================================================================
    # Run Simulation
    # ==================================================================

    def run_simulation(self) -> None:
        """Click handler for "Run Simulation". Wraps the entire pipeline
        in try/except — any simulator failure is surfaced via the
        readout, never crashes the GUI."""
        self._last_error = None
        try:
            tile_system = TileSystem.from_lattice(self._lattice)
            # Load axis fixed in world frame at -Y. Stage 6a's
            # from_lattice already applied lattice.world_transform()
            # to tile vertices, so the simulator's load axis stays
            # fixed regardless of how the lattice is rotated — which
            # is exactly the intent: rotating the lattice rotates the
            # tile vertices in world frame, leaving load axis put.
            if tile_system.dimension == 2:
                load_axis = np.array([0.0, -1.0])
            else:
                load_axis = np.array([0.0, -1.0, 0.0])
            simulator   = Simulator(tile_system, load_axis=load_axis)
            sim_result  = simulator.sweep_theta()
            poissons    = simulator.poissons_ratio()
            locked, info = simulator.is_locked()
        except Exception as exc:
            self._last_error = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc().splitlines()[-1]}"
            )
            self._sim_result     = None
            self._tile_system    = None
            self._simulator      = None
            self._poissons_ratio = None
            self._locking_info   = None
            self._is_outdated    = False
            self._update_state_dependent_ui()
            return

        # Success: replace stored state.
        self._sim_result      = sim_result
        self._tile_system     = tile_system
        self._simulator       = simulator
        self._poissons_ratio  = poissons
        self._locking_info    = info
        self._is_outdated     = False

        self._update_plot()
        self._update_state_dependent_ui()
        self.simulationCompleted.emit()

    def run_dynamics(self) -> None:
        """Click handler for "Run Dynamic" (M2). Wraps
        :func:`auxetic.dynamics.build_dynamics_simulator_from_lattice`
        and stores the resulting :class:`DynamicsResult`. Like
        ``run_simulation`` this is exception-safe."""
        self._dynamics_error = None
        try:
            ds = build_dynamics_simulator_from_lattice(self._lattice)
            self._dynamics_result = ds.simulate()
        except Exception as exc:
            self._dynamics_error = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc().splitlines()[-1]}"
            )
            self._dynamics_result = None
        self._update_state_dependent_ui()

    # ==================================================================
    # Plot rendering
    # ==================================================================

    def _update_plot(self) -> None:
        if self._sim_result is None:
            if self._plot_line is not None:
                self._plot_line.remove()
                self._plot_line = None
            self._canvas.draw_idle()
            return

        axial = self._simulator._axial_index()
        x_deg = np.array([
            simulator_theta_to_slider(t) for t in self._sim_result.theta_samples
        ])
        y = self._sim_result.bbox_extents[:, axial]

        if self._plot_line is None:
            (self._plot_line,) = self._ax.plot(x_deg, y, color="steelblue")
        else:
            self._plot_line.set_data(x_deg, y)
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

        self.play_button.setEnabled(fresh)
        # Plot greys out when outdated — visually consistent with the
        # readout's dimmed-stale pattern.
        if self._plot_line is not None:
            self._plot_line.set_alpha(1.0 if fresh else 0.3)
            self._canvas.draw_idle()

        self.readout.setText(self._compose_readout_html())

    def _compose_readout_html(self) -> str:
        kinematic_html = self._compose_kinematic_readout_html()
        dynamic_html   = self._compose_dynamic_readout_html()
        return kinematic_html + dynamic_html

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
        lock_html     = self._format_locked_html()
        comp_pct      = (self._locking_info or {}).get("compression_ratio", 0.0) * 100.0
        proj          = (self._locking_info or {}).get("mode_projection",   0.0)
        body = (
            "<table style='border-spacing:0;'>"
            f"<tr><td><b>Poisson's ratio:</b></td><td>{nu_html}</td></tr>"
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
        the trajectory sample nearest the slider's θ. Always update
        the plot marker (so the marker still tracks the slider even
        when no result is available)."""
        self._update_marker(slider_deg)
        if self._sim_result is None or self._is_outdated:
            return
        if self._view_3d is None:
            return
        theta_rad = slider_to_simulator_theta(slider_deg)
        idx = int(np.argmin(np.abs(self._sim_result.theta_samples - theta_rad)))
        try:
            self._view_3d.show_pose(self._tile_system,
                                     self._sim_result.poses[idx])
        except Exception:
            pass

    # ==================================================================
    # Play / animate
    # ==================================================================

    def _on_play_toggled(self, checked: bool) -> None:
        if checked and self._sim_result is not None and not self._is_outdated:
            self._play_phase = 0
            self._play_timer.start()
            self.play_button.setText("⏸ Pause")
        else:
            self._play_timer.stop()
            self.play_button.setText("▶ Play")
            # If we got toggled OFF, leave slider where it landed.

    def _on_play_tick(self) -> None:
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
