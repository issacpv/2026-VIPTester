"""Tile Library dock — a palette of 2D tile templates the user drags
onto the 2D canvas to compose a lattice.

Each list entry is a drag source carrying the tile's name in a custom
MIME type (:data:`TILE_MIME`). ``View2D`` accepts the drop, maps the drop
point into lattice space, and asks the ``MainWindow`` to compose the tile
(welding any vertices that land near existing ones — see
:mod:`auxetic.composition`). This panel owns no geometry; it only emits
drags.
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QMimeData, QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDrag, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from auxetic.tile_library import TILE_EDGE, TILE_LIBRARY, TileTemplate

# Custom drag payload: the tile name, UTF-8 encoded. A dedicated MIME type
# (rather than plain text) means the canvas only accepts genuine tile
# drags, not arbitrary text drops.
TILE_MIME = "application/x-auxetic-tile"


def _tile_pixmap(template: TileTemplate, size: int = 48,
                 pad: int = 6) -> QPixmap:
    """Render a small thumbnail of the tile's triangulation."""
    px = QPixmap(size, size)
    px.fill(QColor(0, 0, 0, 0))
    pts = template.points
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = float((mx - mn).max()) or 1.0
    scale = (size - 2 * pad) / span

    def to_px(p):
        x = pad + (float(p[0]) - mn[0]) * scale
        # Flip Y: lattice +y is up, screen +y is down.
        y = size - pad - (float(p[1]) - mn[1]) * scale
        return QPointF(x, y)

    painter = QPainter(px)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(60, 90, 160), 1.5))
        painter.setBrush(QColor(120, 160, 230, 90))
        for tri in template.simplices:
            poly = QPolygonF([to_px(pts[int(v)]) for v in tri])
            painter.drawPolygon(poly)
    finally:
        painter.end()
    return px


class _TileListWidget(QListWidget):
    """A list whose items start a tile drag carrying :data:`TILE_MIME`."""

    def startDrag(self, supportedActions) -> None:  # noqa: N802 (Qt override)
        item = self.currentItem()
        if item is None:
            return
        name = str(item.data(Qt.ItemDataRole.UserRole))
        mime = QMimeData()
        mime.setData(TILE_MIME, QByteArray(name.encode("utf-8")))
        drag = QDrag(self)
        drag.setMimeData(mime)
        pm = item.icon().pixmap(QSize(48, 48))
        if not pm.isNull():
            drag.setPixmap(pm)
            drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction)


class LibraryPanel(QDockWidget):
    """Right-side dock listing draggable tile templates."""

    # Emitted when the user clicks "Scale model" — carries the multiplier
    # to apply to the current composition (MainWindow wraps it in an
    # undoable ScalePointsCommand).
    scaleModelRequested = pyqtSignal(float)
    # Per-triangle C (mode 11): the user changed the selected triangle's
    # C ratio, or asked to clear its override. MainWindow turns these into
    # undoable commands against the currently-selected triangle.
    triangleCRatioChanged = pyqtSignal(float)
    triangleCResetRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Tile Library", parent)
        # Simplex index of the canvas-selected triangle (-1 = none). Set
        # by MainWindow via ``set_selected_triangle``.
        self._selected_idx = -1
        # Guards the C spinbox signal while we update it programmatically.
        self._suspend = False
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        body = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            "Drag a tile onto the 2D canvas. Drop it so its vertices line "
            "up with existing ones — nearby points merge into one shape.",
            body,
        )
        info.setWordWrap(True)
        outer.addWidget(info)

        self.list = _TileListWidget(body)
        self.list.setIconSize(QSize(48, 48))
        self.list.setDragEnabled(True)
        self.list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection)
        for name, template in TILE_LIBRARY.items():
            item = QListWidgetItem(QIcon(_tile_pixmap(template)), name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(item)
        outer.addWidget(self.list)

        # ---- size controls ------------------------------------------------
        form = QFormLayout()

        # Tile size: the edge length (lattice units) of newly-dropped
        # tiles. Bigger tiles → bigger structures per drop.
        self.tile_size_spin = QDoubleSpinBox(body)
        self.tile_size_spin.setRange(0.02, 1.0)
        self.tile_size_spin.setSingleStep(0.05)
        self.tile_size_spin.setDecimals(3)
        self.tile_size_spin.setValue(float(TILE_EDGE))
        self.tile_size_spin.setToolTip(
            "Edge length (lattice units) of tiles dropped from the library. "
            "Larger tiles build bigger structures per drop.")
        form.addRow(QLabel("Tile size"), self.tile_size_spin)
        outer.addLayout(form)

        # Scale model: multiply the whole current composition about its
        # centre — grows its footprint and (importantly) its exported
        # STL/OBJ size. The 2D view auto-fits, so this won't change the
        # on-screen size, only the absolute / export scale.
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale model ×"))
        self.scale_spin = QDoubleSpinBox(body)
        self.scale_spin.setRange(0.1, 20.0)
        self.scale_spin.setSingleStep(0.25)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setValue(2.0)
        scale_row.addWidget(self.scale_spin)
        self.scale_button = QPushButton("Apply", body)
        self.scale_button.setToolTip(
            "Resize the whole model by this factor about its centre "
            "(footprint + export size). On-screen size is unchanged "
            "because the 2D view auto-fits.")
        self.scale_button.clicked.connect(self._on_scale_clicked)
        scale_row.addWidget(self.scale_button)
        outer.addLayout(scale_row)

        # ---- per-triangle C ratio (mode 11) ------------------------------
        # Shift+click a triangle on the 2D canvas to select it, then set
        # *that triangle's* C ratio here — every other triangle keeps the
        # global C (Inspector). Lets one composed shape mix meshy and
        # solid tiles.
        self.triangle_label = QLabel(
            "Shift+click a triangle on the canvas to set its C ratio.", body)
        self.triangle_label.setWordWrap(True)
        outer.addWidget(self.triangle_label)

        tri_row = QHBoxLayout()
        tri_row.addWidget(QLabel("Triangle C"))
        self.triangle_c_spin = QDoubleSpinBox(body)
        self.triangle_c_spin.setRange(0.05, 20.0)
        self.triangle_c_spin.setSingleStep(0.05)
        self.triangle_c_spin.setDecimals(3)
        self.triangle_c_spin.setValue(1.0)
        self.triangle_c_spin.setEnabled(False)
        self.triangle_c_spin.setToolTip(
            "Size ratio C for the selected triangle only. Smaller C → "
            "solider tile; larger C → meshier. Overrides the global C for "
            "this one triangle.")
        self.triangle_c_spin.valueChanged.connect(self._on_triangle_c_changed)
        tri_row.addWidget(self.triangle_c_spin)
        self.triangle_reset_button = QPushButton("Reset", body)
        self.triangle_reset_button.setEnabled(False)
        self.triangle_reset_button.setToolTip(
            "Clear this triangle's override and return it to the global C.")
        self.triangle_reset_button.clicked.connect(self._on_triangle_c_reset)
        tri_row.addWidget(self.triangle_reset_button)
        outer.addLayout(tri_row)

        self.setWidget(body)

    def tile_edge(self) -> float:
        """Current Library tile edge length (lattice units)."""
        return float(self.tile_size_spin.value())

    def _on_scale_clicked(self) -> None:
        factor = float(self.scale_spin.value())
        if factor > 0.0:
            self.scaleModelRequested.emit(factor)

    def set_selected_triangle(self, idx: int, c_value: float,
                              has_override: bool) -> None:
        """Reflect the canvas selection in the per-triangle C control.

        ``idx >= 0`` enables the spinbox and shows the triangle's current
        C (its override or the global value); ``idx < 0`` disables it and
        restores the prompt. The spinbox signal is blocked during the
        update so this doesn't fire a (recursive) change command."""
        self._selected_idx = int(idx)
        enabled = self._selected_idx >= 0
        self._suspend = True
        try:
            self.triangle_c_spin.setEnabled(enabled)
            self.triangle_reset_button.setEnabled(enabled and bool(has_override))
            if enabled:
                self.triangle_c_spin.setValue(float(c_value))
                tag = "overridden" if has_override else "global C"
                self.triangle_label.setText(
                    f"Triangle #{self._selected_idx} selected ({tag}) — "
                    f"set its C ratio.")
            else:
                self.triangle_label.setText(
                    "Shift+click a triangle on the canvas to set its C ratio.")
        finally:
            self._suspend = False

    def _on_triangle_c_changed(self, value: float) -> None:
        if self._suspend or self._selected_idx < 0:
            return
        self.triangleCRatioChanged.emit(float(value))

    def _on_triangle_c_reset(self) -> None:
        if self._selected_idx < 0:
            return
        self.triangleCResetRequested.emit()
