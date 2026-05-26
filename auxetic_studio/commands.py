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
                "unit_scale_cm",
                # Mode-11 constant size ratio. Routed with
                # regenerate=False (see below): C only repositions hinges
                # on the existing triangulation, so re-rolling points
                # would needlessly destroy the user's placed lattice.
                "C",
                # Task 1 bezier-strut options. Also regenerate=False —
                # they only affect derived export/render geometry, never
                # the point cloud — but they DO change the cached strut
                # curves, so the non-regenerate branch invalidates the
                # export-geometry cache below.
                "bezier_enabled", "bezier_strength", "bezier_segments")

    def __init__(self, lattice: Lattice, field: str,
                 old_value: Any, new_value: Any,
                 on_change: Callback = None,
                 *, regenerate: bool = True):
        if field not in self._ALLOWED:
            raise ValueError(f"ParameterChangeCommand: unsupported field {field!r}")
        super().__init__(f"Change {field}")
        self.lattice    = lattice
        self.field      = field
        self.old_value  = old_value
        self.new_value  = new_value
        self.on_change  = on_change
        # When False, the attribute is set without re-rolling the point
        # cloud. Used for parameters that only affect derived geometry
        # (e.g. mode-11 ``C``), never the point generation itself.
        self.regenerate = bool(regenerate)

    def _apply(self, value: Any) -> None:
        setattr(self.lattice, self.field, value)
        if self.regenerate:
            self.lattice.regenerate()
        else:
            # Non-regenerating params (C, bezier_*) leave the point cloud
            # untouched but change derived export/render geometry — drop
            # the cached strut/triangle/joint data so the next render or
            # export rebuilds it with the new value.
            self.lattice._clear_caches()
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


class AddTileCommand(QUndoCommand):
    """Drop a Tile-Library tile onto the composed mesh (undoable).

    ``redo`` calls :meth:`Lattice.compose_add_tile`, which welds the tile
    into the current composition (or seeds a fresh one on the first drop)
    and switches the lattice to mode 11 with ``preserve_triangulation``.
    ``undo`` restores the full prior state — points, the exact prior
    triangulation, mode, the preserve flag and ``points_original`` — so a
    drop is a single clean step even across the first (compose-entering)
    drop.
    """

    def __init__(self, lattice: Lattice, tile_name: str,
                 offset_xy, weld_tol: float | None = None,
                 on_change: Callback = None, tile_scale: float = 1.0):
        super().__init__(f"Add tile {tile_name}")
        from auxetic.tile_library import get_tile
        self.lattice    = lattice
        self.tile_name  = str(tile_name)
        self.offset     = (float(offset_xy[0]), float(offset_xy[1]))
        self.weld_tol   = weld_tol
        self.on_change  = on_change
        # Library tile-size factor: the template is scaled by this before
        # composing, so the user can drop larger tiles. Weld/snap radii
        # scale with it so alignment behaves the same at any tile size.
        self.tile_scale = float(tile_scale)
        self.template   = get_tile(self.tile_name)
        self._prev: dict | None = None

    def _snapshot(self) -> dict:
        lat = self.lattice
        return {
            "points":    None if lat.points is None else lat.points.copy(),
            "simplices": (None if lat.tri is None
                          else np.asarray(lat.tri.simplices).copy()),
            "mode":      int(lat.mode),
            "preserve":  bool(lat.preserve_triangulation),
            "points_original": (None if lat.points_original is None
                                else lat.points_original.copy()),
        }

    def redo(self) -> None:
        from auxetic.composition import DEFAULT_WELD_TOL, SNAP_RADIUS
        if self._prev is None:
            self._prev = self._snapshot()
        s = self.tile_scale
        pts = self.template.points * s                 # bigger tile
        base_tol = DEFAULT_WELD_TOL if self.weld_tol is None else self.weld_tol
        self.lattice.compose_add_tile(
            pts, self.template.simplices, offset=self.offset,
            weld_tol=base_tol * s, snap_radius=SNAP_RADIUS * s)
        if self.on_change is not None:
            self.on_change()

    def undo(self) -> None:
        from auxetic import geometry as _geom
        p = self._prev or {}
        lat = self.lattice
        lat.mode = p.get("mode", lat.mode)
        lat.preserve_triangulation = p.get("preserve", False)
        lat.points_original = p.get("points_original")
        prev_points = p.get("points")
        prev_simplices = p.get("simplices")
        if prev_points is not None and prev_simplices is not None:
            # Restore the exact prior triangulation (wrap in _FlippedTri —
            # downstream reads only ``.simplices``, so this is valid for a
            # prior Delaunay or composed state alike).
            lat._set_points_and_tri(prev_points,
                                    _geom._FlippedTri(prev_simplices))
        elif prev_points is not None:
            lat._set_points_and_tri(prev_points, None)
        if self.on_change is not None:
            self.on_change()


class ScalePointsCommand(QUndoCommand):
    """Uniformly scale the lattice's points about their centroid by
    ``factor`` (enlarge / shrink the whole model — footprint + export
    size). Uniform scaling about the centroid is exactly reversible, so
    undo just scales by ``1/factor`` (no snapshot needed); the centroid is
    a fixed point of the scaling, so it returns to the original
    coordinates."""

    def __init__(self, lattice: Lattice, factor: float,
                 on_change: Callback = None):
        super().__init__(f"Scale ×{float(factor):g}")
        self.lattice   = lattice
        self.factor    = float(factor)
        self.on_change = on_change

    def redo(self) -> None:
        self.lattice.scale_points(self.factor)
        if self.on_change is not None:
            self.on_change()

    def undo(self) -> None:
        if self.factor != 0.0:
            self.lattice.scale_points(1.0 / self.factor)
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
        # A composed (Tile-Library) mesh has no canonical Delaunay to
        # toggle against — ``_triangulate`` never re-Delaunays it, so the
        # ``edge_flips`` set is never applied there. Flip it in place
        # instead (snapshotting the simplices for undo).
        self._composed = bool(getattr(lattice, "preserve_triangulation", False))
        self._prev_simplices = None

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
        if self._composed:
            # Direct in-place diagonal swap of the authored mesh.
            self._prev_simplices = np.asarray(
                self.lattice.tri.simplices).copy()
            self.lattice.flip_composed_edge(self.edge)
            if self.on_change is not None:
                self.on_change()
            return
        # Delaunay modes: if the edge was NOT already flipped, redo flips
        # it (adds to the set); if it WAS flipped, redo unflips (removes).
        self._set_membership(not self.already_flipped)

    def undo(self) -> None:
        if self._composed:
            from auxetic import geometry as _geom
            if self._prev_simplices is not None:
                self.lattice._set_points_and_tri(
                    self.lattice.points,
                    _geom._FlippedTri(self._prev_simplices))
            if self.on_change is not None:
                self.on_change()
            return
        self._set_membership(self.already_flipped)


class ForceListChangeCommand(QUndoCommand):
    """Replace ``lattice.dynamics_state['forces']`` with a new list.

    Single command type covers add / remove / edit — each user
    interaction snapshots the full force list before and after, so
    undo/redo restores the exact prior state regardless of which
    operation triggered the change. Force lists are tiny (typically
    < 10 entries), so deep-copying them per command is negligible.

    Auto-invalidates the dynamics result via the ``on_change``
    callback so a stale trajectory doesn't survive past a force
    edit.
    """

    def __init__(self, lattice: Lattice,
                 old_forces: list, new_forces: list,
                 on_change: Callback = None):
        super().__init__("Edit forces")
        import copy
        self.lattice    = lattice
        self.old_forces = copy.deepcopy(list(old_forces))
        self.new_forces = copy.deepcopy(list(new_forces))
        self.on_change  = on_change

    def _apply(self, forces: list) -> None:
        import copy
        self.lattice.dynamics_state["forces"] = copy.deepcopy(forces)
        if self.on_change is not None:
            self.on_change()

    def redo(self) -> None: self._apply(self.new_forces)
    def undo(self) -> None: self._apply(self.old_forces)


class RecommendationApplyCommand(QUndoCommand):
    """Atomically install a (ground_face, edge_flips, joint_angle)
    triple — typically the output of :func:`auxetic_ml.model.predict_best_action`.

    Three lattice fields move together so the user sees ONE undo
    entry per "Apply Recommendation" click rather than three. Stores
    the prior values for undo so reverting restores the exact
    pre-apply configuration.
    """

    def __init__(self,
                 lattice: Lattice,
                 new_ground_face,
                 new_edge_flips,
                 new_joint_angle: float,
                 on_change: Callback = None):
        super().__init__("Apply prediction")
        import copy
        self.lattice            = lattice
        self.old_ground_face    = lattice.dynamics_state.get("ground_face")
        self.old_edge_flips     = set(lattice.edge_flips)
        self.old_joint_angle    = float(lattice.joint_angle)
        self.new_ground_face    = (None if new_ground_face is None
                                   else str(new_ground_face))
        self.new_edge_flips     = set(
            tuple(sorted((int(a), int(b)))) for a, b in (new_edge_flips or ())
        )
        self.new_joint_angle    = float(new_joint_angle)
        self.on_change          = on_change

    def _apply(self, ground_face, edge_flips, joint_angle: float) -> None:
        self.lattice.dynamics_state["ground_face"] = ground_face
        self.lattice.edge_flips                    = set(edge_flips)
        self.lattice.joint_angle                   = float(joint_angle)
        # Re-triangulate so the new edge_flips set takes effect.
        if self.lattice.points is not None:
            self.lattice.regenerate_from_points(self.lattice.points)
        if self.on_change is not None:
            self.on_change()

    def redo(self) -> None:
        self._apply(self.new_ground_face, self.new_edge_flips,
                    self.new_joint_angle)

    def undo(self) -> None:
        self._apply(self.old_ground_face, self.old_edge_flips,
                    self.old_joint_angle)


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
