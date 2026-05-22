"""Coordinates panel dock — tabular view + keyboard editing of points.

Gives a keyboard alternative to the 2D viewport's drag-to-move Edit
mode: every lattice point is listed with editable X / Y cells. Typing a
value and committing it (Enter / focus-out) routes through the same
undoable :class:`~auxetic_studio.commands.MovePointCommand` the drag
path uses, so the table, the 2D view, and the undo stack stay
consistent.

Scope: 2D / 2.5D modes only (1, 2, 4, 5, 7, 8, 11) — matching the
viewport Edit mode. 2.5D modes store 2-vector points (extruded into
z-layers at render time), so the table shows X / Y for them too. In 3D
modes the table is cleared and an explanatory note shown instead.

A coordinate cell accepts a plain number *or* a math expression, so a
perfect equilateral triangle can be entered as ``(0, 0)``,
``(1, sqrt(3))``, ``(2, 0)`` directly. Expressions are evaluated by a
restricted ``ast`` walker (``parse_coordinate``) — only numeric
literals, ``+ - * / ** %``, parentheses, a whitelist of math functions
(``sqrt``, ``sin``, ``cos``, ...) and constants (``pi``, ``e``, ``tau``)
are allowed; anything else reverts the cell.

Architecture: like the other panels this is a thin Qt layer over the
``Lattice`` model. It never mutates the lattice directly — a committed
edit emits ``pointMoveRequested`` and the MainWindow wraps it in an
undoable command, so undo/redo always lands at a coherent state.
"""

from __future__ import annotations

import ast
import math
import operator

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)


# Modes whose points are stored as editable 2-vectors (2D + 2.5D).
# Mirrors ``auxetic_studio.main_window._EDITABLE_MODES``; 3D modes
# (3, 6, 9, 10) store 3-vectors and point editing is deferred for them.
# Mode 11 (bipartite auxetic) is 2D and editable like the others.
_2D_EDITABLE_MODES = (1, 2, 4, 5, 7, 8, 11)

# Decimal places shown for / accepted from a coordinate cell. Lattice
# space is the unit square per CLAUDE.md, so 6 places is sub-micro
# resolution — ample for a coordinate editor.
_COORD_DECIMALS = 6


# ---------------------------------------------------------------------------
# Safe coordinate-expression evaluation.
#
# Lets a cell accept ``sqrt(3)``, ``1 + sqrt(3)``, ``2*pi/3`` etc. — not
# just plain decimals — without ever running arbitrary code. We parse to
# an AST and walk it, permitting only numeric literals, the basic
# arithmetic operators, unary +/-, parenthesised sub-expressions, calls
# to whitelisted math functions, and a few named constants.
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "cbrt": (lambda x: math.copysign(abs(x) ** (1 / 3), x)),
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "hypot": math.hypot,
    "exp": math.exp, "log": math.log, "log10": math.log10, "log2": math.log2,
    "abs": abs, "pow": pow,
    "radians": math.radians, "degrees": math.degrees,
    "floor": math.floor, "ceil": math.ceil,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return float(_BINOPS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return float(_UNARYOPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTS:
        return float(_ALLOWED_CONSTS[node.id])
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _ALLOWED_FUNCS and not node.keywords):
        args = [_eval_node(a) for a in node.args]
        return float(_ALLOWED_FUNCS[node.func.id](*args))
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def parse_coordinate(text: str) -> float:
    """Evaluate a coordinate cell's text to a float.

    Accepts a plain number or a safe math expression (``sqrt(3)``,
    ``1 + sqrt(3)``, ``2*pi/3``). Raises ``ValueError`` on empty input,
    a syntax error, or any disallowed construct (names, attribute
    access, comprehensions, etc.), so callers can revert the cell."""
    text = text.strip()
    if not text:
        raise ValueError("empty coordinate")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid coordinate expression: {text!r}") from exc
    try:
        value = _eval_node(tree)
    except ArithmeticError as exc:
        # e.g. 1/0, log(0) — surface as ValueError so the cell reverts
        # rather than letting the exception escape the Qt signal handler.
        raise ValueError(f"arithmetic error in {text!r}: {exc}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite coordinate: {text!r}")
    return value


class CoordinatesPanel(QDockWidget):
    """Right-side dock listing every lattice point with editable X / Y
    cells. A keyboard alternative to the viewport's drag editing."""

    # Emitted when the user commits a coordinate edit. Carries
    # ``(index, old_xy, new_xy)`` as numpy 2-vectors; the MainWindow
    # wraps it in a :class:`MovePointCommand` so the edit is undoable.
    pointMoveRequested = pyqtSignal(int, object, object)

    def __init__(self, lattice, parent=None):
        super().__init__("Coordinates", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._lattice = lattice
        # Guards ``cellChanged`` while the table is being repopulated,
        # so a programmatic populate doesn't masquerade as a user edit.
        self._suspend = False

        body  = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(8, 8, 8, 8)

        self._info = QLabel(self)
        self._info.setWordWrap(True)
        self._info.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(self._info)

        self.table = QTableWidget(0, 3, body)
        self.table.setHorizontalHeaderLabels(["#", "X", "Y"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.table.cellChanged.connect(self._on_cell_changed)
        outer.addWidget(self.table)

        self.setWidget(body)
        self.refresh_from_lattice()

    # ==================================================================
    # Public API
    # ==================================================================

    def set_lattice(self, lattice) -> None:
        """Re-bind to a new Lattice (after File → New / Open / mesh
        import). The table repopulates from the new lattice's points."""
        self._lattice = lattice
        self.refresh_from_lattice()

    def refresh_from_lattice(self) -> None:
        """Repopulate the table from ``lattice.points``.

        In 2D / 2.5D modes every point gets an editable row; in 3D modes
        the table is cleared, disabled, and an explanatory note shown
        (3D point editing is out of scope — see CLAUDE.md)."""
        mode = int(getattr(self._lattice, "mode", 1))
        editable = mode in _2D_EDITABLE_MODES

        # cellChanged fires for every setItem below — suspend so the
        # repopulate isn't mistaken for a user edit.
        self._suspend = True
        try:
            if not editable:
                self.table.setRowCount(0)
                self.table.setEnabled(False)
                self._info.setText(
                    "<i>Coordinate editing is available in 2D / 2.5D modes "
                    "only. Switch to a 2D mode to view and edit point "
                    "coordinates.</i>"
                )
                return

            self.table.setEnabled(True)
            pts = np.asarray(self._lattice.points, dtype=float)
            n = pts.shape[0]
            self._info.setText(
                f"<b>{n}</b> point{'s' if n != 1 else ''} — double-click an "
                f"X or Y cell to type a value or expression "
                f"(e.g. <code>sqrt(3)</code>, <code>2*pi/3</code>). "
                f"Each edit is undoable (Ctrl+Z)."
            )
            self.table.setRowCount(n)
            for row in range(n):
                self._populate_row(row, pts[row])
        finally:
            self._suspend = False

    # ==================================================================
    # Internals
    # ==================================================================

    def _populate_row(self, row: int, point: np.ndarray) -> None:
        """Fill one table row from a lattice point 2-vector."""
        # Index column — read-only.
        idx_item = QTableWidgetItem(str(row))
        idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 0, idx_item)
        # X / Y — editable.
        x = float(point[0])
        y = float(point[1]) if point.shape[0] > 1 else 0.0
        self.table.setItem(row, 1, QTableWidgetItem(f"{x:.{_COORD_DECIMALS}f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{y:.{_COORD_DECIMALS}f}"))

    def _on_cell_changed(self, row: int, col: int) -> None:
        """A coordinate cell was committed. Emit an undoable move.

        Only the *edited* column's value is read from the table — the
        other coordinate is taken from ``lattice.points`` so it keeps
        its exact stored value rather than being rounded to the cell's
        display precision."""
        if self._suspend or col == 0:
            return
        pts = np.asarray(self._lattice.points, dtype=float)
        if not (0 <= row < pts.shape[0]):
            return
        old_xy = pts[row, :2].astype(float).copy()

        axis = col - 1   # col 1 → x (axis 0), col 2 → y (axis 1)
        item = self.table.item(row, col)
        text = item.text() if item is not None else ""
        try:
            new_value = parse_coordinate(text)
        except (TypeError, ValueError):
            # Bad input / unparseable expression — revert the cell to the
            # lattice value, no move.
            self._revert_cell(row, col, old_xy[axis])
            return

        new_xy = old_xy.copy()
        new_xy[axis] = new_value
        if np.allclose(new_xy, old_xy, atol=1e-12):
            return
        self.pointMoveRequested.emit(int(row), old_xy, new_xy)

    def _revert_cell(self, row: int, col: int, value: float) -> None:
        """Rewrite a single cell back to ``value`` without re-emitting
        a change (used when a typed value fails to parse)."""
        self._suspend = True
        try:
            item = self.table.item(row, col)
            if item is not None:
                item.setText(f"{float(value):.{_COORD_DECIMALS}f}")
        finally:
            self._suspend = False
