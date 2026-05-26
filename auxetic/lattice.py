"""High-level Lattice class.

A Lattice owns the point cloud, its triangulation, the shrink ratio, and
the mode. It also owns the **rigid lattice rotation** and the **joint
angle** as two distinct fields per SPEC §6.3 — these mean different
things and must never be conflated:

- ``rigid_rotation`` (SPEC §6.1) — orients the whole lattice in world
  space. Applied at render time and at export time. Does **not** modify
  ``points``.
- ``joint_angle``   (SPEC §6.2) — internal kirigami DOF in radians.
  Affects tile positions via the simulation (not yet implemented).
- ``flipped``       — special-cased mirror; redundant-with-rotation
  but stored separately so the UI can show a "flipped" indicator
  without inspecting the quaternion.

The class exposes:

- ``regenerate()`` — re-roll points from scratch (using ``mode``,
  ``n_points``, and the optional ``seed``); also (re)captures
  ``points_original``.
- ``regenerate_from_points(new_points)`` — keep the user's edits and
  re-triangulate around them. Does NOT touch ``points_original``.
- ``reset_to_original()`` — restore ``points`` to the snapshot taken
  during the last ``regenerate()`` (or load).
- ``world_transform()`` — 4×4 homogeneous matrix combining ``flipped``
  and ``rigid_rotation``, applied around the lattice centroid (0.5,
  0.5, 0.5). Joint angle is **not** part of this matrix (per SPEC §6.3).
- ``transformed_points()`` — points after ``world_transform`` (used by
  views and exports). Stored points are never modified.
- ``to_stl()`` / ``to_obj()`` / ``to_scad()`` / ``to_kirigami()`` —
  exporters that consume the current points + triangulation, with
  ``world_transform`` applied to vertex positions per SPEC §9.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

from . import geometry as _geom
from . import tiles as _tiles
from . import export as _export


# Modes whose points live in 3D (n×3); everything else is 2D (n×2).
# Modes 7, 8, 9 are mesh-import variants of 1, 2, 3 respectively.
# Mode 10 is the rotating-cuboids 3D kirigami (sibling of mode 6 but
# with cube tiles instead of tetrahedra).
# Mode 11 is the bipartite-polygon auxetic (Acuna et al. 2022): a 2D
# point cloud is Delaunay-triangulated and the corner/centroid
# bipartition drives the polygon network (see ``auxetic.bipartite``).
# Mode 12 is the 3D tetrahedral auxetic — the volumetric analogue of
# mode 11: a 3D point cloud is Delaunay-tetrahedralised and each tetra
# emits an internal tetra + four corner polyhedra (see
# ``auxetic.tetrahedral``).
_3D_MODES = (3, 6, 9, 10, 12)
_DELAUNAY_MODES = (1, 2, 3, 7, 8, 9, 11, 12)  # modes that re-Delaunay on each retriangulation
_CUBOID_MODES = (10,)                  # modes that bypass Delaunay entirely
_BIPARTITE_MODES = (11,)               # modes built via auxetic.bipartite
_TETRAHEDRAL_MODES = (12,)             # modes built via auxetic.tetrahedral

# Lattice-space centroid that all rigid rotations / flips pivot around.
_CENTROID = np.array([0.5, 0.5, 0.5])


def _normalize_to_unit_square(pts: np.ndarray) -> np.ndarray:
    """Uniformly scale + centre an (N, 2) point set into the unit square
    [0, 1]². Uniform (single-factor) scaling preserves triangle shape so
    a tessellation's equilateral interior stays equilateral in lattice
    space. A degenerate (zero-extent) input is returned unchanged."""
    pts = np.asarray(pts, dtype=float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = float((mx - mn).max())
    if span <= 0.0:
        return pts.copy()
    scaled = (pts - mn) / span
    # Centre the shorter axis within [0, 1].
    scaled += (1.0 - scaled.max(axis=0)) / 2.0
    return scaled


class Lattice:
    def __init__(self, mode=1, n_points=5, ratio=0.35, nz_layers=2, seed=None,
                 ngon_thickness: float | None = None,
                 hub_size_factor: float | None = None,
                 joint_sphere_radius: float | None = None,
                 strut_radius: float | None = None,
                 # M1 generation extensions — every default below is the
                 # "no change vs. V20" value, so the regression suite
                 # produces byte-identical output without these kwargs.
                 density_axis: str = "none",
                 density_law:  str = "uniform",
                 density_strength: float = 1.0,
                 edge_flips=None,
                 mesh_path: str | None = None,
                 mesh_vertices: np.ndarray | None = None,
                 unit_scale_cm: float = 1.0,
                 # Mode-11 bipartite auxetic: constant size ratio C =
                 # b_ji / a_ij (Acuna et al. 2022, step 3). Ignored by
                 # every other mode. C=1 is the symmetric midpoint case.
                 C: float = 1.0,
                 # Bezier-curved strut edges (opt-in). Default OFF so every
                 # mode's export is byte-for-byte identical to before.
                 # ``bezier_strength`` is the perpendicular control-point
                 # offset as a fraction of strut length; ``bezier_segments``
                 # is the polyline tessellation density (>=2 to curve).
                 bezier_enabled: bool = False,
                 bezier_strength: float = 0.25,
                 bezier_segments: int = 12):
        self.mode      = mode
        self.n_points  = n_points
        self.ratio     = ratio
        self.nz_layers = nz_layers
        self.seed      = seed
        self.C         = float(C)

        # ---- Bezier-curved strut edges (opt-in, SPEC task 1) ------------
        # When enabled, struts export as tessellated quadratic-Bezier
        # polylines instead of straight 2-point segments. Default OFF =>
        # byte-identical export. ``_clear_caches`` is triggered by
        # :meth:`set_bezier` so a settings change re-runs the geometry.
        self.bezier_enabled  = bool(bezier_enabled)
        self.bezier_strength = float(bezier_strength)
        self.bezier_segments = int(bezier_segments)

        # Shape parameters per SPEC §5.1's ``shape_params`` block.
        self.ngon_thickness      = (_geom.NGON_THICKNESS      if ngon_thickness      is None else float(ngon_thickness))
        self.hub_size_factor     = (_geom.HUB_SIZE_FACTOR     if hub_size_factor     is None else float(hub_size_factor))
        self.joint_sphere_radius = (_geom.JOINT_SPHERE_RADIUS if joint_sphere_radius is None else float(joint_sphere_radius))
        self.strut_radius        = (_geom.STRUT_RADIUS        if strut_radius        is None else float(strut_radius))

        # ---- M1 generation extensions ------------------------------------
        # Density gradient — biases random sample distribution along an
        # axis. Active only for random Delaunay modes (1, 2, 3); grid and
        # mesh-import modes ignore these knobs.
        self.density_axis     = str(density_axis)
        self.density_law      = str(density_law)
        self.density_strength = float(density_strength)

        # Set of (i, j) edge tuples (i < j) currently flipped from the
        # canonical Delaunay diagonal. Applied by ``apply_edge_flips``
        # after each (re)triangulation. 2D-only; ignored for 3D modes.
        self.edge_flips: set[tuple[int, int]] = set()
        if edge_flips:
            for e in edge_flips:
                a, b = sorted((int(e[0]), int(e[1])))
                self.edge_flips.add((a, b))

        # Mesh-import state. ``mesh_vertices`` is normalised to [0, 1]^3
        # by ``Lattice.from_mesh``; ``mesh_path`` is informational.
        self.mesh_path = mesh_path
        self.mesh_vertices = (np.asarray(mesh_vertices, dtype=float)
                              if mesh_vertices is not None else None)

        # Mode-10 cuboid-kirigami state: list of (8, 3) tile arrays plus
        # the explicit (tile_a, vert_a, tile_b, vert_b, ctype) constraint
        # tuples. None for non-cuboid modes. Set during ``regenerate``
        # via :func:`auxetic.cuboid_kirigami.generate_cuboids`.
        self.cuboid_tiles: list | None = None
        self.cuboid_constraints: list | None = None

        # Lattice unit -> physical scale. Used by the M2 dynamic
        # simulator to convert default forces/masses into SI; the
        # geometry pipeline itself is unit-agnostic.
        self.unit_scale_cm = float(unit_scale_cm)

        # ---- M2 dynamic-simulator state ---------------------------------
        # Stored as a plain dict so save/load pickup is straightforward.
        # Defaults configure a "piston compression" load case so a
        # fresh "Run Dynamic" click produces visible motion: the
        # bottom of the lattice (in world frame, after rigid_rotation)
        # is auto-pinned, the top gets a downward force totalling
        # ``piston_force_n``. Users who want manual control can set
        # ``piston_force_n = 0`` and configure ``ground_face`` +
        # ``forces`` directly.
        self.dynamics_state: dict = {
            "piston_force_n":      5.0,    # total compressive force (N).
                                            # 0 = disable piston mode.
            "forces":              [],     # list of dicts (see preset v4)
            "ground_face":         None,   # "+x"/"-x"/"+y"/"-y"/"+z"/"-z"
            "pre_rotation_quat":   None,   # override view_state.rigid_rotation_quat
            "pre_joint_angle_deg": None,   # override view_state.joint_angle_deg
            "fixed_tiles":         [],
            "config": {
                "dt":                       1.0e-3,
                "duration":                 1.0,
                # Stiffness picked so a 5 N piston force on a unit-cell
                # kirigami produces visible compression without blowing
                # up under explicit Euler. Users tune for their geometry.
                "joint_stiffness":          5.0e2,
                "joint_damping":            5.0e0,
                "gravity_cm_per_s2":        [0.0, 0.0, 0.0],
                "convergence_kinetic_thresh": 1.0e-5,
            },
        }

        # ---- SPEC §6: rotation + joint state ----------------------------
        # Two distinct concepts per §6.3 — kept in separate fields and
        # never combined into a single value at any layer (package, GUI,
        # preset).
        self.rigid_rotation: Rotation = Rotation.identity()  # §6.1
        self.joint_angle:    float    = 0.0                  # §6.2 (radians)
        self.flipped:        bool     = False                # §6.1 special case

        self.metadata: dict = {
            "name":     "",
            "created":  "",
            "modified": "",
            "notes":    "",
        }

        # Tile-Library compose state: when True, ``points``/``tri`` hold a
        # user-authored explicit triangulation that must survive point
        # edits (never re-Delaunayed). Set by ``compose_add_tile`` and by
        # the preset v7 loader; cleared by ``regenerate`` (a re-roll exits
        # compose). ``_triangulate`` short-circuits on it.
        self.preserve_triangulation: bool = False

        self.points: np.ndarray | None = None
        self.tri = None
        self.points_original: np.ndarray | None = None

        self._strut_curves    = None
        self._solid_triangles = None
        self._joint_positions = None

        self.regenerate()

    # ==================================================================
    # Construction from imported meshes (modes 7, 8, 9)
    # ==================================================================

    @classmethod
    def from_mesh(cls, path, *,
                   dim,
                   decimate_to: int | None = None,
                   ratio: float = 0.35,
                   nz_layers: int = 2,
                   seed: int | None = None,
                   ngon_thickness: float | None = None,
                   hub_size_factor: float | None = None,
                   joint_sphere_radius: float | None = None,
                   strut_radius: float | None = None,
                   unit_scale_cm: float = 1.0) -> "Lattice":
        """Build a Lattice from an STL or OBJ file's vertices.

        ``dim`` selects the lattice dimensionality (and therefore which
        of mode 7 / 8 / 9 is used). Accepts ``2`` / ``"2D"``,
        ``2.5`` / ``"2.5D"``, or ``3`` / ``"3D"``.

        ``decimate_to`` caps the number of vertices used after dedup;
        ``None`` means keep them all (use with care for large meshes —
        Delaunay scales poorly).
        """
        from . import mesh_io as _mesh_io

        raw = _mesh_io.read_mesh_vertices(str(path))
        if raw.shape[0] == 0:
            raise ValueError(f"No vertices found in mesh: {path!r}")
        if decimate_to is not None and raw.shape[0] > decimate_to:
            raw = _mesh_io.decimate_uniform(raw, n=int(decimate_to), seed=seed)
        norm = _mesh_io.normalize_to_unit_cube(raw)

        mode_map = {2: 7, "2D": 7, "2d": 7,
                    2.5: 8, "2.5D": 8, "2.5d": 8,
                    3: 9, "3D": 9, "3d": 9}
        mode = mode_map.get(dim)
        if mode is None:
            raise ValueError(
                f"dim must be 2, 2.5, 3, or '2D'/'2.5D'/'3D'; got {dim!r}")

        return cls(
            mode=mode,
            n_points=int(norm.shape[0]),
            ratio=ratio,
            nz_layers=nz_layers,
            seed=seed,
            ngon_thickness=ngon_thickness,
            hub_size_factor=hub_size_factor,
            joint_sphere_radius=joint_sphere_radius,
            strut_radius=strut_radius,
            mesh_path=str(path),
            mesh_vertices=norm,
            unit_scale_cm=unit_scale_cm,
        )

    # ==================================================================
    # Construction from a region tessellation (task 5)
    # ==================================================================

    @classmethod
    def from_tessellation(cls, boundary, *,
                          target_edge: float | None = None,
                          n_triangles: int | None = None,
                          mode: int = 1,
                          ratio: float = 0.35,
                          normalize: bool = True,
                          preserve_triangulation: bool = True,
                          **kwargs) -> "Lattice":
        """Build a 2D Lattice from an equilateral-fill tessellation of the
        region bounded by ``boundary`` (see
        :func:`auxetic.tessellation.generate_tessellation`).

        Density is set by ``target_edge`` or ``n_triangles`` (exactly
        one). ``mode`` must be a 2D mode (1/2/4/5/11). With
        ``normalize`` (default), points are uniformly scaled+centred into
        the unit square so the lattice keeps its [0, 1] convention while
        preserving triangle shape.

        With ``preserve_triangulation`` (default) the tessellation's
        clipped triangulation is installed verbatim — important for
        concave regions, where a plain Delaunay would fill the
        concavities. Note: a subsequent point edit or reset re-Delaunays
        (the standard 2D edit path), which only matches the original for
        convex regions.
        """
        from . import tessellation as _tess

        if mode in _3D_MODES:
            raise ValueError(
                f"from_tessellation is 2D-only; mode {mode} is a 3D mode")

        result = _tess.generate_tessellation(
            boundary, target_edge, n_triangles=n_triangles)
        pts = np.asarray(result.points, dtype=float)
        if normalize:
            pts = _normalize_to_unit_square(pts)

        lat = cls(mode=mode, n_points=len(pts), ratio=ratio, **kwargs)
        if preserve_triangulation:
            lat._set_points_and_tri(pts, _geom._FlippedTri(result.simplices))
        else:
            lat.regenerate_from_points(pts)
        lat.points_original = pts.copy()
        return lat

    # ==================================================================
    # view_state — SPEC §5.1's serializable view block, derived from
    # the §6 fields. Implemented as a property so a Stage 4 caller
    # that does ``lat.view_state = {dict}`` still works (the setter
    # parses the dict back into the real fields).
    # ==================================================================

    @property
    def view_state(self) -> dict:
        """SPEC §5.1 view_state, derived from the §6 fields each call."""
        quat_xyzw = self.rigid_rotation.as_quat()  # scipy default: [x, y, z, w]
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]),
                     float(quat_xyzw[1]), float(quat_xyzw[2])]
        return {
            "rigid_rotation_quat": quat_wxyz,
            "flipped":             bool(self.flipped),
            "joint_angle_deg":     math.degrees(float(self.joint_angle)),
        }

    @view_state.setter
    def view_state(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("view_state must be a dict")
        if "rigid_rotation_quat" in value:
            wxyz = value["rigid_rotation_quat"]
            xyzw = [float(wxyz[1]), float(wxyz[2]),
                    float(wxyz[3]), float(wxyz[0])]
            self.rigid_rotation = Rotation.from_quat(xyzw)
        if "flipped" in value:
            self.flipped = bool(value["flipped"])
        if "joint_angle_deg" in value:
            self.joint_angle = math.radians(float(value["joint_angle_deg"]))

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _clear_caches(self):
        self._strut_curves    = None
        self._solid_triangles = None
        self._joint_positions = None

    def _set_points_and_tri(self, points: np.ndarray, tri) -> None:
        self.points   = points
        self.tri      = tri
        self.n_points = len(points)
        self._clear_caches()

    def _triangulate(self, points: np.ndarray):
        # A Tile-Library composition carries an explicit, user-authored
        # triangulation. A point move (same count) must keep that exact
        # connectivity — re-Delaunaying would discard the composed mesh
        # and fill any concavities. So short-circuit before the Delaunay
        # path. (Tile adds / welds change the count and go through
        # ``compose_add_tile`` → ``_set_points_and_tri`` directly, not here.)
        if (self.preserve_triangulation and self.tri is not None
                and self.points is not None
                and len(points) == len(self.points)):
            return self.tri
        if self.mode in _DELAUNAY_MODES:
            tri = Delaunay(points)
        elif self.tri is not None and self.points is not None and len(points) == len(self.points):
            # Grid modes (4, 5, 6) keep their canonical symmetric simplices
            # when the user just edits a point — re-Delaunay would discard
            # the grid's deliberate diagonal layout.
            tri = self.tri
        else:
            tri = Delaunay(points)
        if self.edge_flips:
            tri = _geom.apply_edge_flips(tri, points, self.edge_flips)
        return tri

    # ==================================================================
    # Re-roll / edit / reset
    # ==================================================================

    def regenerate(self):
        # A re-roll generates a fresh point cloud, so it exits any
        # Tile-Library composition (the authored triangulation is gone).
        self.preserve_triangulation = False
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.mode in _CUBOID_MODES:
            from . import cuboid_kirigami as _cuboid
            n = max(2, int(round(self.n_points ** (1.0 / 3.0))))
            points, tiles, constraint_records = _cuboid.generate_cuboids(
                n=n, ratio=self.ratio,
            )
            # Mode-10 has no Delaunay simplices — store None for `tri`
            # and populate the cuboid-specific fields. Downstream code
            # gates on ``self.mode in _CUBOID_MODES`` rather than
            # ``self.tri is None``.
            self.cuboid_tiles = tiles
            self.cuboid_constraints = constraint_records
            self._set_points_and_tri(points, None)
            self.points_original = points.copy()
            return self
        # Non-cuboid modes use the usual generate-points + Delaunay path.
        self.cuboid_tiles = None
        self.cuboid_constraints = None
        if self.mode in (7, 8, 9):
            if self.mesh_vertices is None:
                raise RuntimeError(
                    f"mode {self.mode} requires mesh_vertices — "
                    f"construct via Lattice.from_mesh()."
                )
            points, tri = _geom.points_from_mesh_vertices(
                self.mesh_vertices, self.mode)
        else:
            points, tri = _geom.generate_points(
                self.n_points, self.mode,
                density_axis=self.density_axis,
                density_law=self.density_law,
                density_strength=self.density_strength,
            )
        if self.edge_flips:
            tri = _geom.apply_edge_flips(tri, points, self.edge_flips)
        self._set_points_and_tri(points, tri)
        self.points_original = points.copy()
        return self

    def regenerate_from_points(self, new_points: np.ndarray) -> None:
        new_points = np.asarray(new_points, dtype=float)
        expected_dim = 3 if self.mode in _3D_MODES else 2
        if new_points.ndim != 2 or new_points.shape[1] != expected_dim:
            raise ValueError(
                f"regenerate_from_points: expected (N, {expected_dim}) array "
                f"for mode {self.mode}, got shape {new_points.shape}"
            )
        new_tri = self._triangulate(new_points)
        self._set_points_and_tri(new_points, new_tri)

    def reset_to_original(self) -> None:
        """Restore ``self.points`` to the cached ``points_original``.

        ``points_original`` is rewritten by ``regenerate()`` and by the
        preset loader (which sets it to the as-loaded points). After
        loading a saved preset, "Reset to Original" returns to the
        **as-loaded** state, not the as-randomly-generated state from a
        previous session. ``points_original`` is intentionally not
        stored in presets — it tracks the most recent regenerate/load
        checkpoint.

        Note: ``reset_to_original`` does not touch ``rigid_rotation``,
        ``flipped``, or ``joint_angle`` — those are orientation /
        simulation state, not point edits."""
        if self.points_original is None:
            return
        original = self.points_original.copy()
        new_tri = self._triangulate(original)
        self._set_points_and_tri(original, new_tri)

    # ==================================================================
    # Tile-Library composition (compose-from-tiles workflow)
    # ==================================================================

    def compose_add_tile(self, tile_points, tile_simplices, *,
                         offset=(0.0, 0.0),
                         weld_tol: float | None = None,
                         snap_radius: float | None = None):
        """Drop a library tile onto the composed mesh, welding any tile
        vertex that lands within ``weld_tol`` (lattice space) of an
        existing vertex.

        The first call — when the lattice is not yet composing — seeds a
        fresh composition with just this tile; later calls add to it. The
        method switches the lattice to mode 11 (the 2D bipartite auxetic)
        and sets ``preserve_triangulation`` so the authored mesh survives
        subsequent point edits and renders fused (shared edges → one
        auxetic shape). Returns the new ``(points, simplices)``.

        The placement geometry is the pure :mod:`auxetic.composition`
        ``add_tile`` (templates are centred, so ``offset`` is where the
        tile centre lands)."""
        from . import composition as _composition

        tol = (_composition.DEFAULT_WELD_TOL if weld_tol is None
               else float(weld_tol))
        snap = (_composition.SNAP_RADIUS if snap_radius is None
                else float(snap_radius))
        if (self.preserve_triangulation and self.points is not None
                and self.tri is not None):
            base_pts = np.asarray(self.points, dtype=float).reshape(-1, 2)
            base_simp = np.asarray(self.tri.simplices,
                                   dtype=np.int64).reshape(-1, 3)
        else:
            base_pts = np.zeros((0, 2), dtype=float)
            base_simp = np.zeros((0, 3), dtype=np.int64)

        # Snap the drop so the tile's nearest vertex lands exactly on the
        # nearest existing vertex — a roughly-aimed drop then locks into
        # alignment (shared edge coincides) instead of skewing off one
        # approximate weld. No-op for the first (seed) tile.
        offset = _composition.snap_tile_offset(
            base_pts, tile_points, offset, snap)

        new_pts, new_simp = _composition.add_tile(
            base_pts, base_simp, tile_points, tile_simplices,
            offset=offset, weld_tol=tol)

        self.mode = 11
        self.preserve_triangulation = True
        self._set_points_and_tri(new_pts, _geom._FlippedTri(new_simp))
        self.points_original = new_pts.copy()
        return self.points, np.asarray(self.tri.simplices)

    def scale_points(self, factor: float) -> None:
        """Uniformly scale the lattice's points about their centroid by
        ``factor`` (so the structure keeps its position but grows /
        shrinks). Enlarges the model's footprint in the unit cell and its
        exported STL/OBJ size — handy for pushing a small composed
        structure up to a usable / printable scale.

        Uniform scaling about the centroid leaves the triangulation
        topology unchanged, so a composed (``preserve_triangulation``)
        mesh keeps its authored simplices and a Delaunay lattice keeps its
        (scale-invariant) triangulation — only the coordinates move. The
        on-screen 2D view auto-fits, so this changes the structure's
        relative footprint and its export size rather than its apparent
        size in the viewport."""
        if self.points is None or self.points.shape[0] == 0:
            return
        f = float(factor)
        pts = np.asarray(self.points, dtype=float)
        centroid = pts.mean(axis=0)
        scaled = centroid + (pts - centroid) * f
        # ``_triangulate`` is preserve-aware: it keeps the explicit
        # composed simplices (same count) and re-Delaunays otherwise (a
        # uniform scale is Delaunay-invariant, so the topology is stable).
        self._set_points_and_tri(scaled, self._triangulate(scaled))

    def flip_composed_edge(self, edge) -> bool:
        """Swap the diagonal of the quad whose current diagonal is
        ``edge``, in a composed (``preserve_triangulation``) mesh.

        The Delaunay-mode edge flip records the flip in ``edge_flips`` and
        re-applies it after every re-triangulation — but a composed mesh
        has no canonical Delaunay to toggle against (its triangulation IS
        the authored design, and ``_triangulate`` deliberately never
        re-Delaunays it). So here the flip is a **direct, in-place edit**
        of the stored simplices; undo restores the prior simplices via the
        command. Returns True iff the flip applied (the edge was a
        flippable, strictly-convex interior diagonal); a boundary or
        non-convex edge leaves the mesh unchanged and returns False.

        To flip back, flip the *new* diagonal — it swaps to the original."""
        if self.points is None or self.tri is None:
            return False
        a, b = sorted((int(edge[0]), int(edge[1])))
        new_tri = _geom.apply_edge_flips(self.tri, self.points, {(a, b)})
        after: set[tuple[int, int]] = set()
        for s in np.asarray(new_tri.simplices):
            verts = [int(v) for v in s]
            for i in range(3):
                x, y = verts[i], verts[(i + 1) % 3]
                after.add((x, y) if x < y else (y, x))
        if (a, b) in after:
            return False   # boundary / non-convex — nothing changed
        self._set_points_and_tri(
            np.asarray(self.points, dtype=float), new_tri)
        return True

    # ==================================================================
    # Bezier-curved strut edges (SPEC task 1)
    # ==================================================================

    def set_bezier(self, *,
                   enabled: bool | None = None,
                   strength: float | None = None,
                   segments: int | None = None) -> None:
        """Update the Bezier-strut export options and invalidate the
        cached export geometry so the next export/render re-runs with
        the new settings. Only the provided fields change.

        Curving is active only when ``enabled`` is true, ``strength`` is
        non-zero, and ``segments >= 2``; otherwise struts export straight
        (byte-for-byte identical to curves-off)."""
        if enabled is not None:
            self.bezier_enabled = bool(enabled)
        if strength is not None:
            self.bezier_strength = float(strength)
        if segments is not None:
            self.bezier_segments = int(segments)
        self._clear_caches()

    # ==================================================================
    # SPEC §6: world transform (rigid rotation + flip)
    # ==================================================================

    def world_transform(self, *,
                         rigid_rotation: Rotation | None = None,
                         flipped: bool | None = None) -> np.ndarray:
        """4×4 homogeneous matrix combining ``flipped`` and
        ``rigid_rotation`` around the lattice centroid (0.5, 0.5, 0.5).

        Order per the SPEC §6 mandate: translate centroid to origin →
        apply flip → apply rigid_rotation → translate back.

        Joint angle is **not** included — per SPEC §6.3 it is a
        separate concept that affects tile positions via the simulation,
        not the rigid orientation.

        Optional ``rigid_rotation`` / ``flipped`` overrides allow a
        view to compute a live-preview transform during a drag without
        mutating the lattice."""
        rigid = self.rigid_rotation if rigid_rotation is None else rigid_rotation
        flip  = self.flipped         if flipped        is None else flipped

        # T(-c)
        T_minus = np.eye(4)
        T_minus[:3, 3] = -_CENTROID

        # Flip = 180° about X when set; identity otherwise. SPEC §6.1
        # specifies this exact representation.
        if flip:
            R_flip_3 = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        else:
            R_flip_3 = np.eye(3)
        R_flip = np.eye(4)
        R_flip[:3, :3] = R_flip_3

        # Rigid rotation
        R_rigid = np.eye(4)
        R_rigid[:3, :3] = rigid.as_matrix()

        # T(+c)
        T_plus = np.eye(4)
        T_plus[:3, 3] = _CENTROID

        return T_plus @ R_rigid @ R_flip @ T_minus

    def has_nontrivial_transform(self) -> bool:
        """True if ``world_transform()`` is *not* the identity matrix.
        Used by the export path to short-circuit the transform when
        nothing has actually been rotated — otherwise float drift would
        break byte-identical export round-trips on identity rotation."""
        if self.flipped:
            return True
        return float(self.rigid_rotation.magnitude()) > 1e-12

    @staticmethod
    def _apply_matrix(M: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply 4×4 ``M`` to a (N, D) point array (D ∈ {2, 3}). When
        D=2 the points are lifted to z=0 for the transform and the
        z-component is dropped on return."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError(f"_apply_matrix expects (N, D) array, got {pts.shape}")
        n, d = pts.shape
        if d == 2:
            pts3 = np.hstack([pts, np.zeros((n, 1))])
        elif d == 3:
            pts3 = pts
        else:
            raise ValueError(f"_apply_matrix: unsupported point dim {d}")
        homo = np.hstack([pts3, np.ones((n, 1))])
        out  = (M @ homo.T).T[:, :3]
        return out[:, :2] if d == 2 else out

    def transformed_points(self) -> np.ndarray:
        """``points`` after applying ``world_transform``. Same dim as
        canonical points (Nx2 or Nx3). Stored points unchanged."""
        return self._apply_matrix(self.world_transform(), self.points)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply the current ``world_transform`` to an arbitrary (N, D)
        point array (D ∈ {2, 3}).

        Lets a view render *derived* geometry — e.g. mode-11 bipartite
        polygon vertices — in the same oriented frame as
        ``transformed_points()`` without reaching into the private
        matrix helper. Stored points are untouched."""
        return self._apply_matrix(self.world_transform(), points)

    def _transform_collection(self, collection: Iterable, M: np.ndarray) -> list:
        """Map ``_apply_matrix`` over an iterable of point arrays
        (e.g. triangle list, strut curves, tile vertex arrays)."""
        return [self._apply_matrix(M, np.asarray(item, dtype=float))
                for item in collection]

    # ==================================================================
    # Export pipeline — applies world_transform per SPEC §9
    # ==================================================================

    def _ensure_export_geometry(self):
        # Mode-11 geometry depends on the live joint angle (the kite
        # rotation), which changes without a re-triangulation — so the
        # cache would go stale. Rebuild it every call instead.
        if self.mode in _BIPARTITE_MODES:
            (self._strut_curves,
             self._solid_triangles,
             self._joint_positions) = _geom.collect_export_geometry(
                self.points, self.tri, self.ratio, self.mode, self.nz_layers,
                bipartite_C=self.C, bipartite_theta=self._bipartite_theta(),
                bezier_enabled=self.bezier_enabled,
                bezier_strength=self.bezier_strength,
                bezier_segments=self.bezier_segments)
            return
        # Mode-12 geometry depends on the live ``C`` (the internal-tetra
        # contraction), which changes without a re-triangulation — so,
        # like mode 11, rebuild it every call instead of caching.
        if self.mode in _TETRAHEDRAL_MODES:
            (self._strut_curves,
             self._solid_triangles,
             self._joint_positions) = _geom.collect_export_geometry(
                self.points, self.tri, self.ratio, self.mode, self.nz_layers,
                tetra_C=self.C,
                bezier_enabled=self.bezier_enabled,
                bezier_strength=self.bezier_strength,
                bezier_segments=self.bezier_segments)
            return
        if self._strut_curves is None:
            (self._strut_curves,
             self._solid_triangles,
             self._joint_positions) = _geom.collect_export_geometry(
                self.points, self.tri, self.ratio, self.mode, self.nz_layers,
                cuboid_tiles=self.cuboid_tiles,
                cuboid_constraints=self.cuboid_constraints,
                bezier_enabled=self.bezier_enabled,
                bezier_strength=self.bezier_strength,
                bezier_segments=self.bezier_segments)

    def build_export_triangles(self, **kwargs):
        """Final triangle list including strut tubes + joint spheres."""
        self._ensure_export_geometry()
        return _geom.build_export_triangles(
            self._strut_curves,
            self._solid_triangles,
            self._joint_positions,
            **kwargs)

    def to_stl(self, path, verbose=True, **build_kwargs):
        triangles = self.build_export_triangles(verbose=verbose, **build_kwargs)
        if self.has_nontrivial_transform():
            triangles = self._transform_collection(triangles, self.world_transform())
        _export.export_stl_direct(path, triangles, verbose=verbose)
        return triangles

    def to_obj(self, path, verbose=True, **build_kwargs):
        triangles = self.build_export_triangles(verbose=verbose, **build_kwargs)
        if self.has_nontrivial_transform():
            triangles = self._transform_collection(triangles, self.world_transform())
        _export.export_obj_direct(path, triangles, verbose=verbose)
        return triangles

    def to_scad(self, path, verbose=True, **build_kwargs):
        self._ensure_export_geometry()
        strut_curves = self._strut_curves
        triangles = self.build_export_triangles(verbose=verbose, **build_kwargs)
        if self.has_nontrivial_transform():
            M = self.world_transform()
            strut_curves = self._transform_collection(strut_curves, M)
            triangles    = self._transform_collection(triangles, M)
        _export.export_to_scad(path, strut_curves, triangles,
                               mode=self.mode, n_points=self.n_points,
                               ratio=self.ratio, verbose=verbose)
        return triangles

    def bipartite_jamming_angle(self) -> float:
        """The mechanism's jamming angle (radians) for the current
        mode-11 lattice — the largest ``|theta|`` the kites can rotate
        about their hinges before colliding with the central polygons.
        Returns ``pi/2`` for non-bipartite modes."""
        if self.mode not in _BIPARTITE_MODES or self.tri is None:
            return float(math.pi / 2.0)
        from . import bipartite as _bip
        return _bip.jamming_angle(self.points,
                                  np.asarray(self.tri.simplices), self.C)

    def edge_vector_poisson_ratio(self, theta: float = 0.1) -> float:
        """Lattice-level generalized Poisson's ratio from the edge-vector
        metric (see :mod:`auxetic.edge_poisson`), averaged over the
        lattice's triangles at a small probe actuation ``theta`` (radians).

        Each triangle's bipartite rotating-units mechanism is actuated by
        ``theta`` and its edge-connection-point deformation distilled into
        a principal Poisson's ratio; the lattice value is the mean over
        all (non-degenerate, finite) triangles. Uses the lattice's
        ``C``. Returns ``nan`` for 3D modes, for a lattice with no 2D
        triangulation, or when no triangle yields a finite ratio (e.g.
        ``theta = 0``)."""
        from . import edge_poisson as _ep

        if self.mode in _3D_MODES or self.tri is None:
            return float("nan")
        pts = np.asarray(self.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return float("nan")
        simplices = np.asarray(self.tri.simplices)
        if simplices.ndim != 2 or simplices.shape[1] != 3:
            return float("nan")

        vals: list[float] = []
        for s in simplices:
            tri = pts[s]
            try:
                nu = _ep.generalized_poisson_ratio(tri, float(self.C), float(theta))
            except ValueError:
                continue
            if np.isfinite(nu):
                vals.append(float(nu))
        return float(np.mean(vals)) if vals else float("nan")

    def poisson_ratio_at_point(self, point, theta: float = 0.1,
                               *, world: bool = True) -> tuple[int | None, float]:
        """Generalized (edge-vector) Poisson's ratio of the single lattice
        triangle at ``point`` — backs the GUI's Ctrl-click-a-triangle
        readout (task 6c), the per-triangle counterpart of the
        full-structure :meth:`edge_vector_poisson_ratio`.

        ``point`` is a world-frame ``(x, y[, z])`` when ``world=True`` (the
        default; mapped back to lattice space via the inverse
        ``world_transform``), else it is already in lattice space. The
        containing Delaunay triangle is found with ``tri.find_simplex``; a
        point outside the convex hull falls back to the nearest triangle by
        centroid.

        Returns ``(triangle_index, nu)`` — ``(None, nan)`` for 3D modes or a
        lattice with no 2D triangulation; ``nu`` is ``nan`` for a degenerate
        triangle or ``theta == 0``. Selection + ν live here, not the GUI."""
        from . import edge_poisson as _ep

        if self.mode in _3D_MODES or self.tri is None:
            return None, float("nan")
        pts = np.asarray(self.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None, float("nan")
        simplices = np.asarray(self.tri.simplices)
        if simplices.ndim != 2 or simplices.shape[1] != 3:
            return None, float("nan")

        p = np.asarray(point, dtype=float).ravel()
        if world:
            try:
                M_inv = np.linalg.inv(self.world_transform())
            except np.linalg.LinAlgError:
                return None, float("nan")
            h = np.array([p[0], p[1], p[2] if p.size > 2 else 0.0, 1.0])
            xy = (M_inv @ h)[:2]
        else:
            xy = p[:2]

        idx = int(self.tri.find_simplex(xy))
        if idx < 0:
            # Outside the hull — nearest triangle by centroid.
            centroids = pts[simplices].mean(axis=1)
            idx = int(np.argmin(np.linalg.norm(centroids - xy, axis=1)))
        tri = pts[simplices[idx]]
        try:
            nu = float(_ep.generalized_poisson_ratio(
                tri, float(self.C), float(theta)))
        except ValueError:
            nu = float("nan")
        return idx, nu

    def _bipartite_theta(self) -> float:
        """Actuation angle for *static* mode-11 rendering — always 0
        (rest).

        The earlier deterministic per-kite spin (rotate each kite about
        its hinge) pinwheeled because the kites weren't tied to each
        other. It's been replaced by the coherent floppy-mode mechanism
        the kirigami Simulator finds over the kite + central + bond tile
        system. So the 2D/3D *design* views always show the rest tile;
        the *deformation* is produced by the simulation (Run Simulation
        → scrub), which keeps every hinge connected."""
        return 0.0

    def build_bipartite(self, theta: float | None = None):
        """Build the bipartite polygon network for the current mode-11
        lattice, actuated by the rotating-units mechanism.

        Each corner kite is rotated about its hinge by ``theta`` (the
        central polygons stay fixed). When ``theta`` is ``None`` the
        lattice's ``joint_angle`` is used (clamped to the jamming angle),
        so the simulation/kinematic angle slider drives the rotation.

        Returns an :class:`auxetic.bipartite.BipartiteNetwork`. Polygons
        are in **canonical** lattice space; pass each polygon's
        ``vertices`` through :meth:`transform_points` to draw them in the
        oriented world frame.

        Raises ``RuntimeError`` for any non-bipartite mode."""
        if self.mode not in _BIPARTITE_MODES:
            raise RuntimeError(
                f"build_bipartite() is only valid for bipartite modes "
                f"{_BIPARTITE_MODES}; current mode is {self.mode}")
        from . import bipartite as _bip
        simplices = np.asarray(self.tri.simplices)
        th = self._bipartite_theta() if theta is None else float(theta)
        return _bip.build_bipartite_network(self.points, simplices,
                                            self.C, theta=th)

    def collect_kirigami(self):
        tiles, source = _tiles.collect_kirigami_tiles(
            self.points, self.tri, self.ratio, self.mode, self.nz_layers,
            bipartite_C=self.C, bipartite_theta=self._bipartite_theta())
        constraints = _tiles.build_kirigami_constraints(tiles, source)
        return tiles, source, constraints

    def to_kirigami(self, vertices_path, constraints_path, verbose=True):
        """Kirigami export. Per SPEC §9, vertices are emitted in the
        oriented frame; constraints are connectivity (tile/vertex
        indices) so they're unaffected by rotation."""
        tiles, source, constraints = self.collect_kirigami()
        if self.has_nontrivial_transform():
            tiles = self._transform_collection(tiles, self.world_transform())
        _export.export_kirigami_vertices(vertices_path, tiles, verbose=verbose)
        _export.export_kirigami_constraints(constraints_path, constraints, verbose=verbose)
        return tiles, constraints
