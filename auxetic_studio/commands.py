"""QUndoCommand subclasses backing edit / orientation operations.

Per SPEC §4.2 every edit funnels through a ``QUndoStack``:
- ``MovePointCommand`` — one step per drag-release (not per pixel)
- ``ParameterChangeCommand`` — covers mode / n_points / ratio / nz_layers
- ``DeletePointCommand`` — Delete-key removal of a selected point
- ``ResetToOriginalCommand`` — Edit → Reset to Original

SPEC §6 orientation operations (Stage 5) — also undoable, also
distinct fields per §6.3 ("two distinct rotation concepts"):
- ``RotationChangeCommand`` — rigid lattice rotation
- ``FlipCommand`` — flip toggle (a special case of rigid rotation,
  stored separately so the UI can show a 'flipped' indicator
  without inspecting the quaternion)
- ``JointAngleChangeCommand`` — kirigami DOF, in radians

Each command takes an optional ``on_change`` callback so the
``MainWindow`` can refresh both views (and the inspector spinboxes)
after redo or undo without the command needing to know about Qt views.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtGui import QUndoCommand
from scipy.spatial.transform import Rotation

from auxetic import Lattice


Callback = Optional[Callable[[], None]]


class MovePointCommand(QUndoCommand):
    """Move a single point. ``old_pos`` and ``new_pos`` are 2-vectors
    for 2D modes / 3-vectors for 3D modes — but per the §4.1.1 deferral
    edit mode is currently 2D-only, so in practice these are 2D."""

    def __init__(self, lattice: Lattice, index: int,
                 old_pos: np.ndarray, new_pos: np.ndarray,
                 on_change: Callback = None):
        super().__init__(f"Move point {index}")
        self.lattice = lattice
        self.index   = int(index)
        self.old_pos = np.array(old_pos, dtype=float).copy()
        self.new_pos = np.array(new_pos, dtype=float).copy()
        self.on_change = on_change

    def _apply(self, pos: np.ndarray) -> None:
        pts = self.lattice.points.copy()
        pts[self.index] = pos
        self.lattice.regenerate_from_points(pts)
        if self.on_change is not None:
            self.on_change()

    def redo(self) -> None: self._apply(self.new_pos)
    def undo(self) -> None: self._apply(self.old_pos)


class ParameterChangeCommand(QUndoCommand):
    """Change one of the lattice's scalar attributes (``mode``,
    ``n_points``, ``ratio``, ``nz_layers``).

    A parameter change goes through ``regenerate()``, which re-rolls
    points and refreshes ``points_original``. If the lattice has a
    ``seed`` (the default for the studio's ``MainWindow``), the re-roll
    is deterministic and undo restores the prior parameter value
    cleanly."""

    _ALLOWED = ("mode", "n_points", "ratio", "nz_layers",
                # M1 additions — density gradient and physical scale.
                "density_axis", "density_law", "density_strength",
                "unit_scale_cm")

    def __init__(self, lattice: Lattice, field: str,
                 old_value: Any, new_value: Any,
                 on_change: Callback = None):
        if field not in self._ALLOWED:
            raise ValueError(f"ParameterChangeCommand: unsupported field {field!r}")
        super().__init__(f"Change {field}")
        self.lattice    = lattice
        self.field      = field
        self.old_value  = old_value
        self.new_value  = new_value
        self.on_change  = on_change

    def _apply(self, value: Any) -> None:
        setattr(self.lattice, self.field, value)
        self.lattice.regenerate()
        if self.on_change is not None:
            self.on_change()

    def redo(self) -> None: self._apply(self.new_value)
    def undo(self) -> None: self._apply(self.old_value)


class DeletePointCommand(QUndoCommand):
    """Remove a single point. The removed coordinate is stashed so undo
    can re-insert it at the same index. Note: the §4.1 minimum-3-points
    invariant is enforced by the *caller*; this command will happily
    drop the count below 3 if pushed."""

    def __init__(self, lattice: Lattice, index: int,
                 on_change: Callback = None):
        super().__init__(f"Delete point {index}")
        self.lattice    = lattice
        self.index      = int(index)
        self.removed    = lattice.points[index].copy()
        self.on_change  = on_change

    def redo(self) -> None:
        pts = np.delete(self.lattice.points, self.index, axis=0)
        self.lattice.regenerate_from_points(pts)
        if self.on_change is not None:
            self.on_change()

    def undo(self) -> None:
        pts = np.insert(self.lattice.points, self.index, self.removed, axis=0)
        self.lattice.regenerate_from_points(pts)
        if self.on_change is not None:
            self.on_change()


class ResetToOriginalCommand(QUndoCommand):
    """Restore ``lattice.points`` to ``points_original`` (the snapshot
    captured at the last ``regenerate()``). On undo, the pre-reset
    points are restored — so a user can experiment with reset and back
    out of it."""

    def __init__(self, lattice: Lattice, on_change: Callback = None):
        super().__init__("Reset to Original")
        self.lattice          = lattice
        self.before_points    = lattice.points.copy()
        self.on_change        = on_change

    def redo(self) -> None:
        self.lattice.reset_to_original()
        if self.on_change is not None:
            self.on_change()

    def undo(self) -> None:
        self.lattice.regenerate_from_points(self.before_points)
        if self.on_change is not None:
            self.on_change()


# ---------------------------------------------------------------------------
# SPEC §6 orientation commands
#
# IMPORTANT (§6.3): rigid_rotation, flipped, and joint_angle are three
# distinct fields and live in three distinct commands. Combining them
# into one "orientation change" command would conflate concepts that
# the spec explicitly tells us to keep separate.
# ---------------------------------------------------------------------------


class RotationChangeCommand(QUndoCommand):
    """Set ``lattice.rigid_rotation``. ``old_rotation`` and
    ``new_rotation`` are ``scipy.spatial.transform.Rotation`` objects."""

    def __init__(self, lattice: Lattice,
                 old_rotation: Rotation, new_rotation: Rotation,
                 on_change: Callback = None):
        super().__init__("Change rotation")
        self.lattice       = lattice
        self.old_rotation  = old_rotation
        self.new_rotation  = new_rotation
        self.on_change     = on_change

    def _apply(self, rotation: Rotation) -> None:
        self.lattice.rigid_rotation = rotation
        if self.on_change is not None:
            self.on_change()

    def redo(self) -> None: self._apply(self.new_rotation)
    def undo(self) -> None: self._apply(self.old_rotation)


class FlipCommand(QUndoCommand):
    """Toggle ``lattice.flipped``. SPEC §6.1 calls flip a special case
    of rigid rotation, but it lives in its own field so the UI can
    show a 'flipped' indicator without inspecting the quaternion."""

    def __init__(self, lattice: Lattice,
                 old_value: bool, new_value: bool,
                 on_change: Callback = None):
        super().__init__("Flip")
        self.lattice    = lattice
        self.old_value  = bool(old_value)
        self.new_value  = bool(new_value)
        self.on_change  = on_change

    def _apply(self, value: bool) -> None:
        self.lattice.flipped = bool(value)
        if self.on_change is not None:
            self.on_change()

    def redo(self) -> None: self._apply(self.new_value)
    def undo(self) -> None: self._apply(self.old_value)


class FlipEdgeCommand(QUndoCommand):
    """Toggle a single Delaunay edge in ``lattice.edge_flips``.

    Each invocation flips one (i, j) edge: redo adds it to the set,
    undo removes it (or vice-versa). After mutating the set the
    command re-triangulates via ``regenerate_from_points`` so the
    downstream geometry sees the new diagonal layout.

    M1: 2D only — 3D tetrahedral flips are deferred (see SPEC and
    plan file). The caller is responsible for filtering the edge
    against ``flippable_edges``.
    """

    def __init__(self, lattice: Lattice,
                 edge: tuple[int, int],
                 already_flipped: bool,
                 on_change: Callback = None):
        super().__init__(f"Flip edge {edge[0]}-{edge[1]}")
        a, b = sorted((int(edge[0]), int(edge[1])))
        self.lattice         = lattice
        self.edge            = (a, b)
        self.already_flipped = bool(already_flipped)
        self.on_change       = on_change

    def _set_membership(self, present: bool) -> None:
        flips = set(self.lattice.edge_flips)
        if present:
            flips.add(self.edge)
        else:
            flips.discard(self.edge)
        self.lattice.edge_flips = flips
        # Re-triangulate from the existing points so the flip is applied.
        self.lattice.regenerate_from_points(self.lattice.points)
        if self.on_change is not None:
            self.on_change()

    def redo(self) -> None:
        # If the edge was NOT already flipped, redo flips it (adds);
        # if it WAS flipped, redo unflips (removes).
        self._set_membership(not self.already_flipped)

    def undo(self) -> None:
        self._set_membership(self.already_flipped)


class JointAngleChangeCommand(QUndoCommand):
    """Set ``lattice.joint_angle`` (radians). Pushed on slider release,
    not on every tick — the SimulationPanel debounces."""

    def __init__(self, lattice: Lattice,
                 old_angle_rad: float, new_angle_rad: float,
                 on_change: Callback = None):
        super().__init__("Change joint angle")
        self.lattice         = lattice
        self.old_angle_rad   = float(old_angle_rad)
        self.new_angle_rad   = float(new_angle_rad)
        self.on_change       = on_change

    def _apply(self, angle_rad: float) -> None:
        self.lattice.joint_angle = float(angle_rad)
        if self.on_change is not None:
            self.on_change()

    def redo(self) -> None: self._apply(self.new_angle_rad)
    def undo(self) -> None: self._apply(self.old_angle_rad)
