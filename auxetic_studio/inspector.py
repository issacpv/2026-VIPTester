"""Right-hand inspector panel.

Hosts Qt widgets bound to:

- ``lattice.mode`` (derived from a dimensionality + strategy pair in
  M1) / ``n_points`` / ``ratio`` / ``nz_layers`` — emitted via
  ``parameterChanged(field, old, new)``.
- ``lattice.density_axis`` / ``density_law`` / ``density_strength``
  (M1 — random-mode density gradient) — same signal.
- ``lattice.rigid_rotation`` (SPEC §6.1) — emitted via
  ``rotationChangeRequested(old_rotation, new_rotation)``. The widget
  set is mode-dependent: a single Z-angle spinbox in 2D modes; three
  Euler XYZ spinboxes (extrinsic, applied X→Y→Z) in 3D modes.
- ``lattice.flipped`` (SPEC §6.1 special case) — emitted via
  ``flipChangeRequested(old, new)``.
- Mesh import — emitted via ``meshImportRequested(path, decimate_to)``;
  MainWindow handles the actual ``Lattice.from_mesh`` rebind.

The MainWindow catches these signals and wraps them in undoable
commands; the inspector itself never mutates the lattice — undo/redo
must always end up at a coherent UI + lattice state.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QLabel,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGridLayout,
    QFileDialog,
    QMessageBox,
)
from scipy.spatial.transform import Rotation


# Mode integer ↔ (dimensionality, strategy) mapping. Modes 1–6 are the
# pre-existing random/grid combinations; modes 7/8/9 are the M1
# mesh-import variants (dim 2D / 2.5D / 3D respectively).
DIM_LABELS      = ["2D", "2.5D", "3D"]
STRATEGY_LABELS = ["Random", "Grid", "Mesh import", "Cuboid grid"]

# (dim, strategy) tuple → mode integer.
# "Cuboid grid" is 3D-only — picking it from 2D / 2.5D auto-switches
# the dim combo to 3D in ``_on_strategy_changed``.
_DIM_STRAT_TO_MODE: dict[tuple[str, str], int] = {
    ("2D",   "Random"):       1,
    ("2.5D", "Random"):       2,
    ("3D",   "Random"):       3,
    ("2D",   "Grid"):         4,
    ("2.5D", "Grid"):         5,
    ("3D",   "Grid"):         6,
    ("2D",   "Mesh import"):  7,
    ("2.5D", "Mesh import"):  8,
    ("3D",   "Mesh import"):  9,
    ("3D",   "Cuboid grid"):  10,
}
# Inverse for refresh_from_lattice — only canonical (dim, strategy)
# pairs map back. Mode 10 → ("3D", "Cuboid grid").
_MODE_TO_DIM_STRAT: dict[int, tuple[str, str]] = {
    v: k for k, v in _DIM_STRAT_TO_MODE.items()
}

_RANDOM_MODES = (1, 2, 3)
_GRID_MODES   = (4, 5, 6)
_MESH_MODES   = (7, 8, 9)
_CUBOID_MODES = (10,)
_2D_MODES     = (1, 2, 4, 5, 7, 8)   # 2D and 2.5D
_3D_MODES     = (3, 6, 9, 10)

# Vertex-count threshold above which mesh import prompts for decimation.
_MESH_DECIMATE_PROMPT_THRESHOLD = 500
_MESH_DECIMATE_DEFAULT          = 500


# SPEC §6 preset orientations (3D mode). One source of truth per the
# prompt. Top / Reset both = identity; the 3D-mode UI exposes both
# anyway because Reset is the explicit "snap back" button.
PRESET_ROTATIONS: dict[str, Rotation] = {
    "Top":   Rotation.identity(),
    "Front": Rotation.from_euler("x", 90, degrees=True),
    "Side":  Rotation.from_euler("y", 90, degrees=True),
    "Reset": Rotation.identity(),
}


class InspectorPanel(QWidget):
    parameterChanged        = pyqtSignal(str, object, object)
    rotationChangeRequested = pyqtSignal(object, object)
    flipChangeRequested     = pyqtSignal(bool, bool)
    # path: str, decimate_to: int | None
    meshImportRequested     = pyqtSignal(str, object)

    def __init__(self, lattice, parent=None):
        super().__init__(parent)
        self._lattice = lattice
        self._suspend = False
        # Last-known good (dim, strategy) for revert-on-cancel during
        # mesh-import flow.
        self._last_valid_dim_idx      = 0
        self._last_valid_strategy_idx = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # =================================================================
        # Lattice section
        # =================================================================
        lat_box = QGroupBox("Lattice", self)
        form = QFormLayout(lat_box)

        self.dim_combo = QComboBox(lat_box)
        for label in DIM_LABELS:
            self.dim_combo.addItem(label, label)
        self.dim_combo.currentIndexChanged.connect(self._on_dim_changed)

        self.strategy_combo = QComboBox(lat_box)
        for label in STRATEGY_LABELS:
            self.strategy_combo.addItem(label, label)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)

        self.n_points_spin = QSpinBox(lat_box)
        self.n_points_spin.setRange(2, 4096)
        self.n_points_spin.valueChanged.connect(self._on_n_points_changed)

        self.ratio_spin = QDoubleSpinBox(lat_box)
        self.ratio_spin.setRange(0.0, 0.99)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setDecimals(3)
        self.ratio_spin.valueChanged.connect(self._on_ratio_changed)

        self.nz_layers_spin = QSpinBox(lat_box)
        self.nz_layers_spin.setRange(2, 64)
        self.nz_layers_spin.valueChanged.connect(self._on_nz_layers_changed)

        form.addRow(QLabel("Dimensionality"), self.dim_combo)
        form.addRow(QLabel("Strategy"),       self.strategy_combo)
        form.addRow(QLabel("N points"),       self.n_points_spin)
        form.addRow(QLabel("Ratio"),          self.ratio_spin)
        form.addRow(QLabel("Nz layers"),      self.nz_layers_spin)

        # ---- Mesh import row (visible only when strategy = Mesh) -------
        self._mesh_row = QWidget(lat_box)
        mesh_layout = QHBoxLayout(self._mesh_row)
        mesh_layout.setContentsMargins(0, 0, 0, 0)
        self.mesh_path_label = QLabel("(no mesh loaded)", self._mesh_row)
        self.mesh_path_label.setWordWrap(True)
        self.choose_mesh_button = QPushButton("Choose file…", self._mesh_row)
        self.choose_mesh_button.clicked.connect(self._on_choose_mesh_clicked)
        mesh_layout.addWidget(self.mesh_path_label, 1)
        mesh_layout.addWidget(self.choose_mesh_button, 0)
        form.addRow(QLabel("Mesh"), self._mesh_row)

        outer.addWidget(lat_box)

        # =================================================================
        # Density gradient section (visible only when strategy = Random)
        # =================================================================
        self._density_box = QGroupBox("Density gradient (random modes)", self)
        dens_form = QFormLayout(self._density_box)

        self.density_axis_combo = QComboBox(self._density_box)
        for label in ("none", "x", "y", "z"):
            self.density_axis_combo.addItem(label, label)
        self.density_axis_combo.currentIndexChanged.connect(self._on_density_axis_changed)

        self.density_law_combo = QComboBox(self._density_box)
        for label in ("uniform", "linear", "log", "exp"):
            self.density_law_combo.addItem(label, label)
        self.density_law_combo.currentIndexChanged.connect(self._on_density_law_changed)

        self.density_strength_spin = QDoubleSpinBox(self._density_box)
        self.density_strength_spin.setRange(-2.0, 2.0)
        self.density_strength_spin.setSingleStep(0.1)
        self.density_strength_spin.setDecimals(2)
        self.density_strength_spin.editingFinished.connect(
            self._on_density_strength_committed)

        dens_form.addRow(QLabel("Axis"),     self.density_axis_combo)
        dens_form.addRow(QLabel("Law"),      self.density_law_combo)
        dens_form.addRow(QLabel("Strength"), self.density_strength_spin)

        outer.addWidget(self._density_box)

        # =================================================================
        # Orientation section (SPEC §6)
        # =================================================================
        orient_box = QGroupBox("Orientation", self)
        ov = QVBoxLayout(orient_box)

        # 2D row: single Z angle.
        self._row_2d = QWidget(orient_box)
        row_2d_layout = QFormLayout(self._row_2d)
        self.z_angle_spin = QDoubleSpinBox(self._row_2d)
        self.z_angle_spin.setRange(-180.0, 180.0)
        self.z_angle_spin.setSingleStep(1.0)
        self.z_angle_spin.setDecimals(2)
        self.z_angle_spin.setSuffix("°")
        self.z_angle_spin.editingFinished.connect(self._on_z_angle_committed)
        row_2d_layout.addRow(QLabel("Z (degrees)"), self.z_angle_spin)
        ov.addWidget(self._row_2d)

        # 3D row: Euler XYZ spinboxes.
        self._row_3d = QWidget(orient_box)
        row_3d_form = QFormLayout(self._row_3d)
        self.euler_spins: dict[str, QDoubleSpinBox] = {}
        for axis in ("X", "Y", "Z"):
            sb = QDoubleSpinBox(self._row_3d)
            sb.setRange(-180.0, 180.0)
            sb.setSingleStep(1.0)
            sb.setDecimals(2)
            sb.setSuffix("°")
            sb.editingFinished.connect(self._on_euler_committed)
            sb.setToolTip("Applied in XYZ extrinsic order (X first, then Y, then Z)")
            row_3d_form.addRow(QLabel(axis), sb)
            self.euler_spins[axis] = sb
        ov.addWidget(self._row_3d)

        # 3D-only preset buttons row.
        self._preset_row = QWidget(orient_box)
        preset_layout = QGridLayout(self._preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        self.preset_buttons: dict[str, QPushButton] = {}
        for col, name in enumerate(("Top", "Front", "Side", "Reset")):
            btn = QPushButton(name, self._preset_row)
            btn.clicked.connect(lambda _checked=False, n=name: self._on_preset_clicked(n))
            preset_layout.addWidget(btn, 0, col)
            self.preset_buttons[name] = btn
        ov.addWidget(self._preset_row)

        # Mirror button (formerly "Flip" — renamed in M1 to free the
        # word "flip" for per-edge Delaunay diagonal flips). Toggles
        # ``lattice.flipped``, which is a 180° X-axis rigid rotation.
        flip_row = QWidget(orient_box)
        flip_layout = QHBoxLayout(flip_row)
        flip_layout.setContentsMargins(0, 0, 0, 0)
        self.flip_button = QPushButton("Mirror", flip_row)
        self.flip_button.setCheckable(True)
        self.flip_button.setToolTip(
            "Mirror the lattice (180° rotation about the X axis)")
        self.flip_button.toggled.connect(self._on_flip_toggled)
        flip_layout.addWidget(self.flip_button)
        flip_layout.addStretch(1)
        ov.addWidget(flip_row)

        outer.addWidget(orient_box)
        outer.addStretch(1)

        self.refresh_from_lattice()

    # =====================================================================
    # public API
    # =====================================================================

    @property
    def lattice(self):
        return self._lattice

    def set_lattice(self, lattice):
        """Re-bind to a new Lattice instance (e.g. after File → Open)."""
        self._lattice = lattice
        self.refresh_from_lattice()

    def refresh_from_lattice(self):
        """Pull current values from the lattice into the widgets without
        re-emitting any change signal."""
        self._suspend = True
        try:
            mode = int(self._lattice.mode)
            dim, strat = _MODE_TO_DIM_STRAT.get(mode, ("2D", "Random"))
            di = self.dim_combo.findData(dim)
            si = self.strategy_combo.findData(strat)
            if di >= 0:
                self.dim_combo.setCurrentIndex(di)
                self._last_valid_dim_idx = di
            if si >= 0:
                self.strategy_combo.setCurrentIndex(si)
                self._last_valid_strategy_idx = si

            self.n_points_spin.setValue(int(self._lattice.n_points))
            self.ratio_spin.setValue(float(self._lattice.ratio))
            self.nz_layers_spin.setValue(int(self._lattice.nz_layers))

            # Density widgets: visible only for Random strategy.
            ai = self.density_axis_combo.findData(
                str(getattr(self._lattice, "density_axis", "none")))
            if ai >= 0: self.density_axis_combo.setCurrentIndex(ai)
            li = self.density_law_combo.findData(
                str(getattr(self._lattice, "density_law", "uniform")))
            if li >= 0: self.density_law_combo.setCurrentIndex(li)
            self.density_strength_spin.setValue(
                float(getattr(self._lattice, "density_strength", 1.0)))

            # Mesh row: show current mesh path if any.
            mp = getattr(self._lattice, "mesh_path", None)
            self.mesh_path_label.setText(
                str(mp) if mp else "(no mesh loaded)")

            self._update_visibility()
            self._sync_orientation_widgets()

            self.flip_button.setChecked(bool(self._lattice.flipped))
        finally:
            self._suspend = False

    def select_mode(self, mode: int) -> None:
        """Programmatically pick the mode integer ``mode``, mirroring the
        effect of the user choosing the corresponding (dim, strategy)
        pair. Emits ``parameterChanged`` like a normal user interaction.

        Used by GUI tests written before the M1 dim+strategy split.
        """
        if mode not in _MODE_TO_DIM_STRAT:
            raise ValueError(f"select_mode: unknown mode {mode!r}")
        dim, strat = _MODE_TO_DIM_STRAT[mode]
        old_mode = int(self._lattice.mode)
        self._suspend = True
        try:
            di = self.dim_combo.findData(dim)
            si = self.strategy_combo.findData(strat)
            if di >= 0: self.dim_combo.setCurrentIndex(di)
            if si >= 0: self.strategy_combo.setCurrentIndex(si)
            self._last_valid_dim_idx      = self.dim_combo.currentIndex()
            self._last_valid_strategy_idx = self.strategy_combo.currentIndex()
            self._update_visibility()
        finally:
            self._suspend = False
        if mode != old_mode:
            self._emit_param("mode", old_mode, mode)

    # =====================================================================
    # parameter change handlers (lattice section)
    # =====================================================================

    def _emit_param(self, field: str, old, new) -> None:
        if self._suspend:
            return
        if old == new:
            return
        self.parameterChanged.emit(field, old, new)

    def _derived_mode(self) -> int:
        dim   = str(self.dim_combo.currentData())
        strat = str(self.strategy_combo.currentData())
        return _DIM_STRAT_TO_MODE.get((dim, strat), int(self._lattice.mode))

    def _on_dim_changed(self, _idx):
        if self._suspend:
            return
        # Cuboid grid is 3D-only — if the user switches dim away from
        # 3D while strategy is Cuboid, auto-flip strategy back to Grid
        # (the closest equivalent). Mirrors the auto-flip-to-3D when
        # the user picks Cuboid from a non-3D dim.
        new_dim   = str(self.dim_combo.currentData())
        cur_strat = str(self.strategy_combo.currentData())
        if new_dim != "3D" and cur_strat == "Cuboid grid":
            si_grid = self.strategy_combo.findData("Grid")
            if si_grid >= 0:
                self._suspend = True
                try:
                    self.strategy_combo.setCurrentIndex(si_grid)
                    self._last_valid_strategy_idx = si_grid
                finally:
                    self._suspend = False
        # Mesh-strategy + new dim with no mesh loaded → can't change yet.
        # Fall through and emit anyway; if the lattice has mesh_vertices
        # the new mode (7/8/9) will work, otherwise regenerate raises and
        # the user is expected to choose a mesh.
        self._last_valid_dim_idx = self.dim_combo.currentIndex()
        self._emit_mode_from_combos()
        self._update_visibility()

    def _on_strategy_changed(self, _idx):
        if self._suspend:
            return
        new_strat = str(self.strategy_combo.currentData())
        # Cuboid grid is 3D-only — auto-flip the dim combo to 3D so
        # ``_DIM_STRAT_TO_MODE`` resolves to mode 10. We do this in
        # ``_suspend`` so the dim change doesn't fire its own
        # parameterChanged emission; we'll emit one mode change below.
        if new_strat == "Cuboid grid":
            di_3d = self.dim_combo.findData("3D")
            if di_3d >= 0 and self.dim_combo.currentIndex() != di_3d:
                self._suspend = True
                try:
                    self.dim_combo.setCurrentIndex(di_3d)
                    self._last_valid_dim_idx = di_3d
                finally:
                    self._suspend = False
        if new_strat == "Mesh import":
            # Mesh strategy needs a file. Open the chooser; revert combo
            # if the user cancels.
            chosen = self._prompt_for_mesh_file()
            if not chosen:
                # Revert without emitting.
                self._suspend = True
                try:
                    self.strategy_combo.setCurrentIndex(self._last_valid_strategy_idx)
                finally:
                    self._suspend = False
                return
            path, decimate_to = chosen
            # MainWindow handles the lattice swap via Lattice.from_mesh.
            self._last_valid_strategy_idx = self.strategy_combo.currentIndex()
            self._update_visibility()
            self.meshImportRequested.emit(path, decimate_to)
            return
        self._last_valid_strategy_idx = self.strategy_combo.currentIndex()
        self._emit_mode_from_combos()
        self._update_visibility()

    def _emit_mode_from_combos(self) -> None:
        new_mode = self._derived_mode()
        old_mode = int(self._lattice.mode)
        if new_mode != old_mode:
            self._emit_param("mode", old_mode, new_mode)

    def _on_n_points_changed(self, value):
        self._emit_param("n_points", int(self._lattice.n_points), int(value))

    def _on_ratio_changed(self, value):
        self._emit_param("ratio", float(self._lattice.ratio), float(value))

    def _on_nz_layers_changed(self, value):
        self._emit_param("nz_layers", int(self._lattice.nz_layers), int(value))

    # ---- density gradient handlers ---------------------------------------

    def _on_density_axis_changed(self, _idx):
        new = str(self.density_axis_combo.currentData())
        old = str(getattr(self._lattice, "density_axis", "none"))
        self._emit_param("density_axis", old, new)

    def _on_density_law_changed(self, _idx):
        new = str(self.density_law_combo.currentData())
        old = str(getattr(self._lattice, "density_law", "uniform"))
        self._emit_param("density_law", old, new)

    def _on_density_strength_committed(self):
        if self._suspend:
            return
        new = float(self.density_strength_spin.value())
        old = float(getattr(self._lattice, "density_strength", 1.0))
        self._emit_param("density_strength", old, new)

    # ---- mesh import handlers --------------------------------------------

    def _on_choose_mesh_clicked(self):
        if self._suspend:
            return
        chosen = self._prompt_for_mesh_file()
        if not chosen:
            return
        path, decimate_to = chosen
        self.meshImportRequested.emit(path, decimate_to)

    def _prompt_for_mesh_file(self) -> tuple[str, object] | None:
        """Open a file chooser, then (for large meshes) prompt for
        decimation. Returns ``(path, decimate_to_or_None)`` or ``None``
        if the user cancelled."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import mesh", "", "Mesh files (*.stl *.obj)")
        if not path:
            return None
        n_verts = self._peek_mesh_vertex_count(path)
        decimate_to: object = None
        if n_verts > _MESH_DECIMATE_PROMPT_THRESHOLD:
            mb = QMessageBox(self)
            mb.setIcon(QMessageBox.Icon.Question)
            mb.setWindowTitle("Large mesh")
            mb.setText(
                f"This mesh has {n_verts} unique vertices. Importing all "
                f"of them may be slow.")
            mb.setInformativeText("How would you like to proceed?")
            keep_btn     = mb.addButton(
                "Use all (slow)",        QMessageBox.ButtonRole.AcceptRole)
            decimate_btn = mb.addButton(
                f"Decimate to {_MESH_DECIMATE_DEFAULT}",
                QMessageBox.ButtonRole.AcceptRole)
            mb.addButton(QMessageBox.StandardButton.Cancel)
            mb.exec()
            clicked = mb.clickedButton()
            if clicked is decimate_btn:
                decimate_to = _MESH_DECIMATE_DEFAULT
            elif clicked is keep_btn:
                decimate_to = None
            else:
                return None
        return path, decimate_to

    @staticmethod
    def _peek_mesh_vertex_count(path: str) -> int:
        try:
            from auxetic.mesh_io import read_mesh_vertices
            return int(read_mesh_vertices(path).shape[0])
        except Exception:
            return 0

    # =====================================================================
    # orientation handlers (SPEC §6)
    # =====================================================================

    def _update_visibility(self) -> None:
        """Show/hide the conditional rows based on the current dim and
        strategy. Called from ``refresh_from_lattice`` and after every
        combo change."""
        mode  = self._derived_mode()
        is_2d = mode in _2D_MODES
        is_random = mode in _RANDOM_MODES
        is_grid   = mode in _GRID_MODES
        is_mesh   = mode in _MESH_MODES

        # Orientation rows depend only on dim.
        self._row_2d.setVisible(is_2d)
        self._row_3d.setVisible(not is_2d)
        self._preset_row.setVisible(not is_2d)

        # Density gradient is a random-mode-only feature.
        self._density_box.setVisible(is_random)

        # Mesh row only meaningful for mesh modes.
        self._mesh_row.setVisible(is_mesh)

        # nz_layers is only used by extruded (2.5D) modes — 2 / 5 / 8.
        # Existing pre-M1 UI showed the spinbox unconditionally; preserve
        # that for backwards compatibility (no test expects it hidden).

    # Back-compat alias for the original method name. test_rotation.py
    # and other internal call-sites used the orientation-only name.
    def _update_orientation_visibility(self) -> None:
        self._update_visibility()

    def _sync_orientation_widgets(self) -> None:
        """Read lattice.rigid_rotation back into the spinboxes. Always
        decompose as XYZ extrinsic for consistency between 2D (only Z
        meaningful) and 3D (all three).

        The XYZ Euler decomposition hits a singularity at the canonical
        90°-about-X / 90°-about-Y orientations (e.g. the "Front" / "Side"
        preset buttons). scipy emits a UserWarning in that case and
        sets the third angle to zero — fine for display, so we silence
        it explicitly rather than letting it leak into test output."""
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                x, y, z = self._lattice.rigid_rotation.as_euler("xyz", degrees=True)
        except Exception:
            x = y = z = 0.0
        if self._lattice.mode in _2D_MODES:
            # In 2D modes only Z is meaningful — but if the rotation was
            # set to something else (e.g. before the user switched modes)
            # we still display Z so they can see/correct it.
            self.z_angle_spin.setValue(float(z))
        else:
            self.euler_spins["X"].setValue(float(x))
            self.euler_spins["Y"].setValue(float(y))
            self.euler_spins["Z"].setValue(float(z))

    def _emit_rotation(self, new_rotation: Rotation) -> None:
        if self._suspend:
            return
        old = self._lattice.rigid_rotation
        # Compare as quaternion to avoid Euler ambiguity.
        if _rotations_close(old, new_rotation):
            return
        self.rotationChangeRequested.emit(old, new_rotation)

    def _on_z_angle_committed(self) -> None:
        if self._suspend:
            return
        new_z = float(self.z_angle_spin.value())
        new_rot = Rotation.from_euler("xyz", [0.0, 0.0, new_z], degrees=True)
        self._emit_rotation(new_rot)

    def _on_euler_committed(self) -> None:
        if self._suspend:
            return
        x = float(self.euler_spins["X"].value())
        y = float(self.euler_spins["Y"].value())
        z = float(self.euler_spins["Z"].value())
        new_rot = Rotation.from_euler("xyz", [x, y, z], degrees=True)
        self._emit_rotation(new_rot)

    def _on_preset_clicked(self, name: str) -> None:
        if self._suspend:
            return
        new_rot = PRESET_ROTATIONS[name]
        self._emit_rotation(new_rot)

    def _on_flip_toggled(self, checked: bool) -> None:
        if self._suspend:
            return
        old = bool(self._lattice.flipped)
        new = bool(checked)
        if old == new:
            return
        self.flipChangeRequested.emit(old, new)


def _rotations_close(a: Rotation, b: Rotation, tol: float = 1e-9) -> bool:
    import numpy as np
    qa = np.asarray(a.as_quat())
    qb = np.asarray(b.as_quat())
    # Quaternion equivalence: q ~ -q (both encode the same rotation).
    return bool(np.linalg.norm(qa - qb) < tol or np.linalg.norm(qa + qb) < tol)
