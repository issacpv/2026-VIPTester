"""Quasi-static kinematic simulator (SPEC §7.3 / §7.4 / §7.5).

Stage 6a delivered ``TileSystem`` (input adapter) + ``Simulator`` core
(Jacobian, null-space, projection). Stage 6b adds the θ-sweep
trajectory, Poisson's-ratio computation, and locking detection. The
GUI panel hookup lands in 6c.

Per ``docs/solver_evaluation.md``, the algorithmic primitives are
``scipy.linalg.null_space`` (kinematic-mode identification) and
``scipy.optimize.least_squares`` (constraint projection). Everything in
this module is the modeling layer that turns a ``constraints.txt``-
shaped record into the residual + analytic Jacobian those primitives
consume.

Pose layout (the contract):

- ``dofs_per_tile == 3`` in 2D: ``[tx, ty, θ]``
- ``dofs_per_tile == 6`` in 3D: ``[tx, ty, tz, rx, ry, rz]`` where
  ``(rx, ry, rz)`` is an axis-angle rotation vector
- The "rest pose" is ``zeros(n_tiles * dofs_per_tile)`` — every tile's
  translation is zero and every tile's rotation is identity, so vertex
  world positions are exactly ``tile_system.tiles[i][v]``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Constraint:
    """One row of ``constraints.txt``: pin (tile_a, vert_a) to
    (tile_b, vert_b). ``ctype`` is preserved verbatim from the file
    (1 = default tile-to-tile, 2 = explicit "top" connection in
    layered 2D modes); the v1 solver ignores it but it round-trips
    through file I/O for downstream consumers."""
    tile_a: int
    vert_a: int
    tile_b: int
    vert_b: int
    ctype:  int


@dataclass
class SimResult:
    """Result of ``Simulator.sweep_theta`` (SPEC §7.6).

    - ``theta_samples`` (radians, shape ``(n_steps,)``) — the sweep
      parameter values. Default range is the canonical bistable
      ``np.linspace(-π/2, π/2, n_steps)`` (rest at θ=0, compressed
      states at θ=±π/2 per SPEC §6.2 mathematical parameterization).
      The M2.8 ``theta_max`` extension allows ranges up to ±π.
    - ``poses`` (shape ``(n_steps, n_tiles * dofs_per_tile)``) — the
      projected pose at each step.
    - ``bbox_extents`` (shape ``(n_steps, dimension)``) — bounding-box
      ``(max - min)`` per spatial dimension over all tile vertices in
      the projected configuration.
    - ``compression_ratio`` — ``(max(axial) - min(axial)) / max(axial)``
      where the axial dimension is the index whose unit vector best
      aligns with ``Simulator.load_axis``.
    - ``locked`` / ``locking_info`` — composite criterion per SPEC §7.5
      (also exposed by ``Simulator.is_locked``).
    - ``collision_at_theta`` (M2.8) — bool array of length ``n_steps``,
      ``True`` for samples where tiles overlap. Set to all-``False``
      when collision checking is disabled or unsupported (3D modes).
    - ``collision_theta_min``/``collision_theta_max`` (M2.8) — the
      θ values at which the lattice first collides on the negative
      and positive halves of the sweep. ``None`` if no collision was
      seen on that side. Used by the GUI to render shaded
      "unreachable" regions on the sweep plot.
    """
    theta_samples:     np.ndarray
    poses:             np.ndarray
    bbox_extents:      np.ndarray
    compression_ratio: float
    locked:            bool
    locking_info:      dict
    collision_at_theta:    np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool))
    collision_theta_min:   float | None = None
    collision_theta_max:   float | None = None


# ---------------------------------------------------------------------------
# TileSystem
# ---------------------------------------------------------------------------

class TileSystem:
    """Decouples the solver from its input source. Two constructors
    (``from_lattice`` for the live GUI path, ``from_files`` for
    ``vertices.txt`` + ``constraints.txt`` round-trips), one consumer
    (``Simulator``).
    """

    def __init__(self, dimension: int,
                 tiles: list[np.ndarray],
                 constraints: list[Constraint],
                 tile_source: list[dict] | None = None):
        if dimension not in (2, 3):
            raise ValueError(f"TileSystem dimension must be 2 or 3, got {dimension}")
        for i, t in enumerate(tiles):
            if t.ndim != 2 or t.shape[1] != dimension:
                raise ValueError(
                    f"tile {i} has shape {t.shape}; expected (N, {dimension})")
        if tile_source is None:
            # Stub: one ``{'type': 'unknown'}`` per tile. Pose-rendering
            # in ``collect_export_geometry_from_posed_tiles`` raises
            # ``ValueError`` when it sees ``'unknown'`` types — only the
            # constraint-based simulation path tolerates a stub source.
            tile_source = [{'type': 'unknown'} for _ in tiles]
        if len(tile_source) != len(tiles):
            raise ValueError(
                f"tile_source length {len(tile_source)} != tiles length "
                f"{len(tiles)}"
            )
        self.dimension   = dimension
        self.tiles       = [np.asarray(t, dtype=float) for t in tiles]
        self.constraints = list(constraints)
        self.tile_source = list(tile_source)

    # ------------------------------------------------------------------

    @classmethod
    def from_lattice(cls, lattice) -> "TileSystem":
        """Build a TileSystem from a live ``Lattice``.

        The simulator works in **world frame**, so we apply the
        lattice's ``world_transform`` to tile vertices before storing —
        this is what lets a rotated lattice's kirigami mode line up
        differently against the load axis (SPEC §6.3 / §8)."""
        from . import tiles as _tiles  # local import: avoid circular at module load

        # Mode-10 cuboid kirigami stores its tiles + constraints
        # directly on the lattice (no Delaunay simplices), so we skip
        # the simplex-based collector and use the precomputed data.
        if (int(lattice.mode) == 10
                and getattr(lattice, "cuboid_tiles", None) is not None):
            tile_arrays_3d = [np.asarray(t, dtype=float).copy()
                               for t in lattice.cuboid_tiles]
            constraint_tuples = list(lattice.cuboid_constraints or [])
            source = [{"type": "cuboid", "cube_idx": i}
                      for i in range(len(tile_arrays_3d))]
        else:
            tile_arrays_3d, source = _tiles.collect_kirigami_tiles(
                lattice.points, lattice.tri, lattice.ratio,
                lattice.mode, lattice.nz_layers,
            )
            constraint_tuples = _tiles.build_kirigami_constraints(
                tile_arrays_3d, source,
            )

        # Apply the lattice's world transform (rigid rotation + flip,
        # around centroid). collect_kirigami_tiles always returns 3D
        # vertex arrays, so transform in 3D.
        if lattice.has_nontrivial_transform():
            M = lattice.world_transform()
            tile_arrays_3d = [
                _apply_4x4_to_points(M, t) for t in tile_arrays_3d
            ]

        # Dimension inference: kirigami tiles for the flat 2D modes
        # (1, 4) all live in a single z-plane (which stays planar
        # under any pure-Z rotation). Drop the z-coord and report 2D.
        # For modes 2/5 (extruded) and 3/6 (native 3D) the z-coord
        # carries information — keep the full 3D representation.
        if _is_planar(tile_arrays_3d):
            dimension = 2
            tiles = [t[:, :2].copy() for t in tile_arrays_3d]
        else:
            dimension = 3
            tiles = [t.copy() for t in tile_arrays_3d]

        constraints = [Constraint(*tup) for tup in constraint_tuples]
        # ``source`` is keyed in canonical lattice frame and survives the
        # 2D-planar slicing path unchanged — only the vertex arrays get
        # sliced down to (N, 2). The ``vertex_keys`` it carries are used
        # by the pose-rendering path to detect strut pairs.
        return cls(dimension, tiles, constraints, tile_source=source)

    @classmethod
    def from_files(cls, vertices_path: str, constraints_path: str,
                   dimension: int) -> "TileSystem":
        """Parse the same ``vertices.txt`` + ``constraints.txt`` format
        that ``Lattice.to_kirigami`` writes.

        ``dimension`` is required because the file format doesn't
        encode it: each vertex is just space-separated floats, and you
        only know whether they group as 2-vectors or 3-vectors by
        looking at the lattice that produced the file. Trailing
        newlines and blank lines are tolerated.

        Note: file-loaded TileSystems don't support pose rendering —
        the kirigami file format doesn't carry tile-type metadata, so
        every entry of ``tile_source`` is stubbed as
        ``{'type': 'unknown'}``. The constraint-based simulation path
        works fine; only ``collect_export_geometry_from_posed_tiles``
        rejects unknown-type sources."""
        if dimension not in (2, 3):
            raise ValueError(f"dimension must be 2 or 3, got {dimension}")

        tiles: list[np.ndarray] = []
        with open(vertices_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                vals = [float(tok) for tok in stripped.split()]
                if len(vals) % dimension != 0:
                    raise ValueError(
                        f"{vertices_path}:{line_no}: {len(vals)} floats "
                        f"is not a multiple of dimension {dimension}"
                    )
                arr = np.asarray(vals, dtype=float).reshape(-1, dimension)
                tiles.append(arr)

        constraints: list[Constraint] = []
        with open(constraints_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                toks = stripped.split()
                if len(toks) != 5:
                    raise ValueError(
                        f"{constraints_path}:{line_no}: expected 5 ints, "
                        f"got {len(toks)}"
                    )
                ta, va, tb, vb, ct = (int(t) for t in toks)
                constraints.append(Constraint(ta, va, tb, vb, ct))

        return cls(dimension, tiles, constraints)

    # ------------------------------------------------------------------

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Quasi-static kinematic solver around a fixed ``TileSystem``.

    Methods are organised around SPEC §7.3's algorithm:

    - ``assemble_jacobian(pose)`` — constraint Jacobian J at any pose
    - ``identify_kirigami_mode()`` — null space of J(rest), with
      rigid-body modes stripped, then load-axis selection if multi-D
    - ``project_to_manifold(perturbed_pose)`` — Gauss-Newton (via
      ``scipy.optimize.least_squares``) drives constraint residual to
      zero
    """

    def __init__(self, tile_system: TileSystem, load_axis: np.ndarray):
        self.tile_system = tile_system
        load_axis = np.asarray(load_axis, dtype=float).flatten()
        if load_axis.shape != (tile_system.dimension,):
            raise ValueError(
                f"load_axis must have shape ({tile_system.dimension},), "
                f"got {load_axis.shape}"
            )
        n = float(np.linalg.norm(load_axis))
        if n < 1e-12:
            raise ValueError("load_axis must have non-zero norm")
        self.load_axis = load_axis / n

        # Cache for shape parameters used everywhere.
        self.dimension = tile_system.dimension
        self.dofs      = 3 if self.dimension == 2 else 6
        self.n_tiles   = tile_system.n_tiles
        self.n_constraints = tile_system.n_constraints

    # ==================================================================
    # Pose application
    # ==================================================================

    def rest_pose(self) -> np.ndarray:
        return np.zeros(self.n_tiles * self.dofs, dtype=float)

    def _decompose_pose(self, pose: np.ndarray, i: int):
        """Return ``(t, R)`` for tile ``i`` under ``pose``.

        ``t`` is the translation vector, ``R`` is a (dim, dim) rotation
        matrix.
        """
        s = i * self.dofs
        if self.dimension == 2:
            tx, ty, theta = pose[s], pose[s + 1], pose[s + 2]
            c, sn = np.cos(theta), np.sin(theta)
            R = np.array([[c, -sn], [sn, c]])
            return np.array([tx, ty]), R
        else:
            t = pose[s:s + 3].copy()
            omega = pose[s + 3:s + 6]
            if np.linalg.norm(omega) < 1e-12:
                R = np.eye(3)
            else:
                R = Rotation.from_rotvec(omega).as_matrix()
            return t, R

    def _world_vertex(self, pose: np.ndarray, tile_i: int, vert_i: int) -> np.ndarray:
        t, R = self._decompose_pose(pose, tile_i)
        v = self.tile_system.tiles[tile_i][vert_i]
        return R @ v + t

    def _tile_world_vertices(self, pose: np.ndarray, tile_i: int) -> np.ndarray:
        """All vertices of tile ``tile_i`` in world frame under ``pose``.
        Shape ``(n_verts, dimension)``."""
        t, R = self._decompose_pose(pose, tile_i)
        tile = self.tile_system.tiles[tile_i]
        return tile @ R.T + t

    # ==================================================================
    # Constraint residual
    # ==================================================================

    def constraint_residual(self, pose: np.ndarray) -> np.ndarray:
        """Stack ``(R_a v_a + t_a) - (R_b v_b + t_b)`` for every
        constraint into a flat residual vector. Length =
        ``n_constraints * dimension``."""
        r = np.empty(self.n_constraints * self.dimension, dtype=float)
        # Cache per-tile (t, R) so we don't re-decompose.
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def get(i: int):
            v = cache.get(i)
            if v is None:
                v = self._decompose_pose(pose, i)
                cache[i] = v
            return v

        for k, c in enumerate(self.tile_system.constraints):
            t_a, R_a = get(c.tile_a)
            t_b, R_b = get(c.tile_b)
            v_a = self.tile_system.tiles[c.tile_a][c.vert_a]
            v_b = self.tile_system.tiles[c.tile_b][c.vert_b]
            r[k * self.dimension:(k + 1) * self.dimension] = (
                (R_a @ v_a + t_a) - (R_b @ v_b + t_b)
            )
        return r

    # ==================================================================
    # Jacobian
    # ==================================================================

    def assemble_jacobian(self, pose: np.ndarray) -> np.ndarray:
        """Constraint Jacobian J of shape
        ``(n_constraints * dimension, n_tiles * dofs_per_tile)``.

        Rows = one block of ``dimension`` rows per constraint
        (one per coordinate of the coincidence requirement).
        Columns = one block of ``dofs_per_tile`` columns per tile.

        Each constraint touches exactly two tiles; the row block has
        zeros everywhere except in those two column blocks. Translation
        derivatives are ±I; rotation derivatives use the closed-form
        ∂(R(ω) v)/∂ω at the current ω (see ``_rot_deriv_*``).

        Implementation note: dense storage. The systems we run on
        (≤ a few hundred tiles, ≤ a few hundred constraints) fit
        comfortably in memory and clarity beats premature sparse-
        matrix optimisation. Drop in ``scipy.sparse`` when profiling
        says it matters.
        """
        M = self.n_constraints * self.dimension
        N = self.n_tiles * self.dofs
        J = np.zeros((M, N), dtype=float)
        if M == 0 or N == 0:
            return J

        # Cache rotation-related quantities per tile so we don't recompute
        # for tiles that appear in many constraints.
        rot_cache: dict[int, dict] = {}

        def get_rot(i: int):
            v = rot_cache.get(i)
            if v is None:
                s = i * self.dofs
                if self.dimension == 2:
                    theta = pose[s + 2]
                    c, sn = np.cos(theta), np.sin(theta)
                    v = {
                        "theta": theta,
                        "dRdtheta": np.array([[-sn, -c], [c, -sn]]),
                    }
                else:
                    omega = pose[s + 3:s + 6]
                    v = {"omega": omega, "Jr": _so3_right_jacobian(omega),
                         "R":     _so3_rotation_matrix(omega)}
                rot_cache[i] = v
            return v

        I_dim = np.eye(self.dimension)

        for k, c in enumerate(self.tile_system.constraints):
            row = k * self.dimension
            v_a = self.tile_system.tiles[c.tile_a][c.vert_a]
            v_b = self.tile_system.tiles[c.tile_b][c.vert_b]
            col_a = c.tile_a * self.dofs
            col_b = c.tile_b * self.dofs

            # Translation derivatives
            J[row:row + self.dimension, col_a:col_a + self.dimension] += I_dim
            J[row:row + self.dimension, col_b:col_b + self.dimension] -= I_dim

            # Rotation derivatives
            if self.dimension == 2:
                ra = get_rot(c.tile_a)
                rb = get_rot(c.tile_b)
                J[row:row + 2, col_a + 2] += ra["dRdtheta"] @ v_a
                J[row:row + 2, col_b + 2] -= rb["dRdtheta"] @ v_b
            else:
                ra = get_rot(c.tile_a)
                rb = get_rot(c.tile_b)
                # ∂(R(ω) v)/∂ω = -R(ω) [v]_× J_r(ω) (right-Jacobian
                # convention; reduces to -[v]_× at ω = 0).
                dR_a = -ra["R"] @ _skew(v_a) @ ra["Jr"]
                dR_b = -rb["R"] @ _skew(v_b) @ rb["Jr"]
                J[row:row + 3, col_a + 3:col_a + 6] += dR_a
                J[row:row + 3, col_b + 3:col_b + 6] -= dR_b

        return J

    # ==================================================================
    # Null-space identification (SPEC §7.3 step 2)
    # ==================================================================

    def _build_rigid_basis(self) -> np.ndarray:
        """Construct the rigid-body subspace explicitly (SPEC §7.3).

        2D: 3 columns — translation in x, y, and rotation about
        the global centroid in the plane.
        3D: 6 columns — translation in x, y, z, and rotation about
        the global centroid around each of the three axes.

        For a rotation by ``δθ`` around axis ``e`` through centroid
        ``C``, every tile shares the same rotation parameter (so its
        local rotation parameter equals ``e``) and translates by
        ``-(e × C)`` so that the centroid maps to itself."""
        all_verts = (np.concatenate(self.tile_system.tiles, axis=0)
                     if self.tile_system.tiles else np.zeros((0, self.dimension)))
        if len(all_verts) > 0:
            C = all_verts.mean(axis=0)
        else:
            C = np.zeros(self.dimension)

        N = self.n_tiles * self.dofs
        if self.dimension == 2:
            n_rigid = 3
        else:
            n_rigid = 6
        basis = np.zeros((N, n_rigid))

        if self.dimension == 2:
            # Translation x, y
            for i in range(self.n_tiles):
                basis[i * 3 + 0, 0] = 1.0
                basis[i * 3 + 1, 1] = 1.0
                # Rotation about centroid: ω_i = 1, t_i = -R(π/2) C =
                # [C_y, -C_x] for a counter-clockwise δθ.
                basis[i * 3 + 0, 2] = C[1]
                basis[i * 3 + 1, 2] = -C[0]
                basis[i * 3 + 2, 2] = 1.0
        else:
            # Translation x, y, z
            for i in range(self.n_tiles):
                basis[i * 6 + 0, 0] = 1.0
                basis[i * 6 + 1, 1] = 1.0
                basis[i * 6 + 2, 2] = 1.0
                # Rotation about each axis through centroid C.
                # ω_i = e_k, t_i = -(e_k × C).
                for k in range(3):
                    e_k = np.zeros(3); e_k[k] = 1.0
                    t_part = -np.cross(e_k, C)
                    basis[i * 6 + 0:i * 6 + 3, 3 + k] = t_part
                    basis[i * 6 + 3 + k,        3 + k] = 1.0
        return basis

    def identify_kirigami_mode(self) -> np.ndarray | None:
        """SPEC §7.3 steps 2-3: compute null space of J(rest), strip
        rigid-body modes, select the kirigami mode by load-axis
        projection.

        Returns a unit-norm pose-vector representing the chosen mode,
        or ``None`` if the kirigami subspace is empty (system is fully
        constrained)."""
        J_rest = self.assemble_jacobian(self.rest_pose())
        raw_null = scipy.linalg.null_space(J_rest, rcond=1e-8)

        rigid_basis = self._build_rigid_basis()
        rigid_orth = scipy.linalg.orth(rigid_basis)

        # Project raw null-space onto the orthogonal complement of the
        # rigid-body subspace.
        if rigid_orth.shape[1] > 0:
            kirigami_part = raw_null - rigid_orth @ (rigid_orth.T @ raw_null)
        else:
            kirigami_part = raw_null

        # Extract a clean orthonormal basis for the residual subspace.
        if kirigami_part.size == 0:
            return None
        U, S, _Vt = np.linalg.svd(kirigami_part, full_matrices=False)
        # Anything significantly above zero is a real mode; the rest
        # is what was already rigid.
        tol = max(1e-8, 1e-8 * (S[0] if S.size else 0))
        n_kirigami = int(np.sum(S > tol))
        if n_kirigami == 0:
            return None
        kirigami_basis = U[:, :n_kirigami]

        if n_kirigami == 1:
            return kirigami_basis[:, 0]

        # PRIMARY: bipartite-alternating-rotation projection.
        #
        # If the tile-constraint graph is bipartite (typical for grid
        # and grid-like kirigami topologies), the canonical
        # rotating-squares mode has tiles in one color rotating one
        # way and tiles in the other color rotating the opposite way.
        # We build that target vector (±1 in each tile's rotation DOF
        # by colour), project it onto the kirigami null space, and use
        # the result if a non-trivial projection exists. This gives
        # the visually-coherent pattern users expect — all tiles of
        # one colour rotate the same direction.
        alternating = self._project_alternating_rotation_onto_null(
            kirigami_basis,
        )
        if alternating is not None:
            return alternating

        # FALLBACK: pick the mode with the strongest second-order
        # bbox-extent curvature along the load axis. Used when the
        # constraint graph isn't bipartite (rare in practice) or the
        # alternating target projects to ~0.
        #
        # Earlier drafts of this fallback used a per-vertex-mean
        # displacement heuristic. That formulation hits the SPEC §7.5
        # dead-end: vertex displacements in symmetric auxetic modes
        # cancel by construction, so the "best" score went to a
        # non-auxetic translation mode and the negative spaces
        # rotated incoherently. Curvature-based scoring (``+δ`` and
        # ``-δ`` both compress for symmetric modes, so the second
        # difference is non-zero and signed) is the right criterion.
        scores = np.array([
            self._mode_bbox_change_alignment(kirigami_basis[:, j])
            for j in range(n_kirigami)
        ])
        max_score = scores.max()
        if max_score < 1e-12:
            warnings.warn(
                "All kirigami modes orthogonal to load axis; picking first."
            )
            return kirigami_basis[:, 0]
        within_5pct = scores >= 0.95 * max_score
        n_close = int(within_5pct.sum())
        if n_close > 1:
            warnings.warn(
                f"Degenerate kirigami mode selection: {n_close} modes "
                f"within tolerance"
            )
        return kirigami_basis[:, int(np.argmax(scores))]

    def _project_alternating_rotation_onto_null(
            self, kirigami_basis: np.ndarray) -> np.ndarray | None:
        """If the tile-constraint graph is bipartite, build the
        target vector ``[+1, +1, ..., -1, -1, ...]`` over per-tile
        rotation DOFs (signs by 2-colouring) and project it onto the
        provided kirigami null-space basis.

        Returns the unit-norm projected mode if non-trivial, else
        ``None`` (signalling the caller to fall back to bbox-curvature
        scoring).

        Visual intent: the rotating-squares pattern. Tiles of one
        colour rotate one way; tiles of the other colour rotate the
        other way. The projection finds the kirigami mode closest to
        that target — the SAME-SIGN block within each colour is what
        makes the negative-space tiles rotate coherently.
        """
        if self.n_tiles < 2:
            return None

        # Build the tile-tile adjacency from the constraint records.
        adj: list[list[int]] = [[] for _ in range(self.n_tiles)]
        for c in self.tile_system.constraints:
            if c.tile_a != c.tile_b:
                adj[c.tile_a].append(c.tile_b)
                adj[c.tile_b].append(c.tile_a)

        # 2-colour each connected component via BFS. Bail if any edge
        # would force two tiles into the same colour.
        color = [-1] * self.n_tiles
        for root in range(self.n_tiles):
            if color[root] != -1:
                continue
            color[root] = 0
            queue = [root]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return None   # not bipartite — fall back

        # Build the target: ±1 in each tile's rotation DOF, zero in
        # translation DOFs and in disconnected tiles.
        target = np.zeros(self.n_tiles * self.dofs, dtype=float)
        rot_offset = 2 if self.dimension == 2 else 5  # last DOF in 2D/3D
        # In 3D the rotation DOFs are 3 axes; we put the target on the
        # axis most aligned with the load axis so the alternating
        # rotation produces compression along that axis.
        for ti, c in enumerate(color):
            if c < 0:
                continue
            s = ti * self.dofs
            sign = +1.0 if c == 0 else -1.0
            if self.dimension == 2:
                target[s + 2] = sign
            else:
                # In 3D, rotate about the load axis so the auxetic
                # motion's rotation axis matches the load direction.
                # ω_i = sign * load_axis (3-vector); the rotation DOFs
                # at indices [s+3, s+4, s+5] hold (ω_x, ω_y, ω_z).
                target[s + 3:s + 6] = sign * self.load_axis

        norm = float(np.linalg.norm(target))
        if norm < 1e-12:
            return None
        target = target / norm

        # Project onto the kirigami null space.
        coeffs = kirigami_basis.T @ target
        proj   = kirigami_basis @ coeffs
        proj_norm = float(np.linalg.norm(proj))
        if proj_norm < 1e-3:
            # Target is essentially orthogonal to the null space —
            # the alternating-rotation pattern isn't kinematically
            # achievable for this topology; let the bbox-curvature
            # fallback try.
            return None

        picked = proj / proj_norm
        # Sanity check: the picked mode must compress along the load
        # axis. If it doesn't, the bipartite target is misaligned with
        # the load axis (e.g., load axis chosen at 45°) and the
        # bbox-curvature fallback is more useful.
        if self._mode_bbox_change_alignment(picked) < 1e-9:
            return None
        return picked

    def _mode_bbox_change_alignment(self, mode_vec: np.ndarray) -> float:
        """Score a candidate kirigami mode by how strongly the
        bounding-box extent CURVES along the load axis at the rest pose.

        The criterion combines two ideas:

        1. **Direction**: bbox-change direction (not mean per-vertex
           displacement) — sibling of
           :meth:`_displacement_direction_of_mode`. The §7.5 rationale
           applies: averaging per-vertex displacements zeros out
           symmetric auxetic modes by construction.
        2. **Order**: second-order in ε. A genuine auxetic mode is
           SYMMETRIC about rest — both ``+ε`` and ``-ε`` compress the
           bbox the same way, so the first-order term is ~0 and only
           the curvature ``d²(bbox)/dε²`` is informative. First-order
           scoring (which the very first attempt at this fix used)
           collapses inside the degenerate subspace of multiple axial
           modes — it can't tell pure auxetic from pure axial-shift,
           and noise sets the picked combination.

        Returns ``|d²(bbox · load_axis)/dε²|`` evaluated by central
        differencing. Larger = stronger axial compression curvature =
        more auxetic.
        """
        delta = 1.0e-3
        rest    = float(self._bbox_extents(self.rest_pose())               @ self.load_axis)
        forward = float(self._bbox_extents(self.rest_pose() + delta * mode_vec) @ self.load_axis)
        backward = float(self._bbox_extents(self.rest_pose() - delta * mode_vec) @ self.load_axis)
        # Central second difference: f(+δ) + f(-δ) - 2 f(0) ≈ δ² · f''(0)
        curvature = (forward + backward - 2.0 * rest) / (delta * delta)
        return float(abs(curvature))

    # ==================================================================
    # Projection (SPEC §7.3 step 4)
    # ==================================================================

    def project_to_manifold(self, perturbed_pose: np.ndarray) -> np.ndarray:
        """Gauss-Newton (via ``scipy.optimize.least_squares`` with
        ``method='trf'``) drives the constraint residual to zero
        starting from ``perturbed_pose``.

        Returns the projected pose. The analytic Jacobian is supplied
        for stability and speed — without it ``trf`` would
        finite-difference, which both costs more evaluations and
        compounds noise from rotation parameterisation."""
        x0 = np.asarray(perturbed_pose, dtype=float).flatten()
        if x0.shape != (self.n_tiles * self.dofs,):
            raise ValueError(
                f"perturbed_pose shape {x0.shape}; expected "
                f"({self.n_tiles * self.dofs},)"
            )
        # Fast path: a few explicit Gauss-Newton iterations with the
        # min-norm step. From a warm-started initial guess the
        # residual is small (~Δθ² off-manifold from manifold curvature)
        # and a single step usually drives it below the threshold —
        # without scipy's trust-region overhead, which on near-
        # converged inputs spends many iterations chasing tight
        # gradient tolerances. Falls through to ``trf`` if GN diverges
        # or stalls.
        x = x0.copy()
        r = self.constraint_residual(x)
        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-9:
            return x
        for _ in range(5):
            J = self.assemble_jacobian(x)
            try:
                delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
            except np.linalg.LinAlgError:
                break
            x_new = x + delta
            r_new = self.constraint_residual(x_new)
            r_new_norm = float(np.linalg.norm(r_new))
            if r_new_norm < 1e-9:
                return x_new
            if r_new_norm > r_norm:
                # Diverging: fall through to scipy's robust trf.
                break
            x, r, r_norm = x_new, r_new, r_new_norm

        result = least_squares(
            self.constraint_residual,
            x,
            jac=self.assemble_jacobian,
            method="trf",
            xtol=1e-9,
            ftol=1e-9,
            # Tightening gtol below the default 1e-8 — at default
            # trf halts on the gradient-optimality criterion well
            # before the absolute residual gets below 1e-9 on a
            # curved manifold.
            gtol=1e-12,
        )
        return result.x

    # ==================================================================
    # Bounding-box helpers (used by sweep + Poisson + locking)
    # ==================================================================

    def _bbox_extents(self, pose: np.ndarray) -> np.ndarray:
        """``(max - min)`` per spatial dimension over all tile vertices
        in the given pose. Shape ``(dimension,)``."""
        if self.n_tiles == 0:
            return np.zeros(self.dimension)
        all_verts = np.concatenate(
            [self._tile_world_vertices(pose, i) for i in range(self.n_tiles)],
            axis=0,
        )
        return all_verts.max(axis=0) - all_verts.min(axis=0)

    def _axial_index(self) -> int:
        """Spatial-dimension index whose unit vector best aligns with
        ``load_axis``."""
        return int(np.argmax(np.abs(self.load_axis)))

    # ==================================================================
    # SPEC §7.3 step 5-6: θ-sweep
    # ==================================================================

    def sweep_theta(self, n_steps: int = 181, *,
                     warm_start: bool = True,
                     theta_max: float | None = None,
                     collision_stop: bool = False,
                     collision_tol: float = 1.0e-6,
                     ) -> SimResult:
        """Sweep θ along the kirigami mode and project to the manifold
        at each step.

        Per SPEC §6.2, the simulator works in the *mathematical*
        joint-angle parameterization: rest at θ=0. The default range
        is the canonical bistable ``[-π/2, +π/2]``; the M2.8
        ``theta_max`` parameter extends it (set to ``np.pi`` for
        ``[-π, +π]`` — full 180° rotation in either direction).

        SPEC §7.3 step 5–6: at each θ_i, compute ``pose = θ_i * mode``
        (linear extrapolation around the rest pose), project to the
        constraint manifold via ``project_to_manifold``, and record the
        bounding-box extents. The previous step's projected pose is the
        warm-start initial guess for the next step's projection — this
        is what makes the inner loop fast (without it, every step
        starts from the linear extrapolation, which is increasingly
        wrong as |θ| grows and the manifold curves away).

        ``warm_start=False`` disables warm-starting (every step
        projects from the cold linear extrapolation). Used by the
        ``test_warm_start_speeds_up_sweep`` test to compare timings.

        ``collision_stop=True`` (M2.8) runs a 2D tile-tile collision
        check at every projected pose. The first θ on the negative
        half of the sweep where a collision appears bounds
        ``collision_theta_min``; the first on the positive half bounds
        ``collision_theta_max``. Samples beyond those bounds still get
        recorded (the array shape stays predictable) but their
        ``collision_at_theta`` flag is True. Always False for 3D
        tile systems (collision detection is 2D-only in M2).
        """
        if theta_max is None:
            theta_max = float(np.pi / 2.0)
        n_pose = self.n_tiles * self.dofs
        theta_samples = np.linspace(-theta_max, theta_max, n_steps, dtype=float)
        poses        = np.zeros((n_steps, n_pose), dtype=float)
        bbox_extents = np.zeros((n_steps, self.dimension), dtype=float)
        collision_flags = np.zeros(n_steps, dtype=bool)
        collision_theta_min: float | None = None
        collision_theta_max: float | None = None

        # Lazy collision checker — only constructed when requested AND
        # supported (2D dimension; the helper module also no-ops 3D).
        collider = None
        if collision_stop and self.dimension == 2:
            from .collision import CollisionChecker
            collider = CollisionChecker(self.tile_system, tol=collision_tol)

        mode = self.identify_kirigami_mode()
        if mode is None:
            # No kirigami mode — system is fully constrained or rigid.
            # Trajectory is just the rest pose at every θ; bbox is constant.
            rest = self.rest_pose()
            for i in range(n_steps):
                poses[i] = rest
                bbox_extents[i] = self._bbox_extents(rest)
            comp_ratio = 0.0
            locked, locking_info = self._compute_locking(None, comp_ratio)
            return SimResult(
                theta_samples=theta_samples, poses=poses,
                bbox_extents=bbox_extents,
                compression_ratio=comp_ratio,
                locked=locked, locking_info=locking_info,
                collision_at_theta=collision_flags,
            )

        prev_pose = None
        prev_theta = 0.0
        for i, theta in enumerate(theta_samples):
            if not warm_start or prev_pose is None:
                initial = float(theta) * mode
            else:
                initial = prev_pose + float(theta - prev_theta) * mode
            projected = self.project_to_manifold(initial)
            poses[i] = projected
            bbox_extents[i] = self._bbox_extents(projected)
            if collider is not None and collider.has_collision(projected):
                collision_flags[i] = True
                if theta < 0.0 and collision_theta_min is None:
                    # First collision on the negative half (sweeping toward
                    # rest the linspace passes here first, but the FIRST
                    # collision on this half going outward from rest is the
                    # one closest to 0, which is the LAST one we'll see in
                    # the linspace order before crossing rest. Track all
                    # negative-half collisions and pick the maximum θ later.
                    pass
                elif theta > 0.0 and collision_theta_max is None:
                    collision_theta_max = float(theta)
            prev_pose = projected
            prev_theta = theta

        # Resolve collision_theta_min: for the negative half, the
        # *innermost* (closest to 0) collision is the bound — anything
        # farther out has already collided. Pick the LARGEST negative θ
        # that has a collision (i.e., the smallest |θ|).
        neg_collisions = theta_samples[(theta_samples < 0.0) & collision_flags]
        if neg_collisions.size > 0:
            collision_theta_min = float(neg_collisions.max())  # closest to 0
        # collision_theta_max already set on first hit going forward.

        # SPEC §7.4 Option A compression ratio. Only consider non-collided
        # samples — values past the collision boundary are physically
        # unreachable and would distort the metric.
        valid_mask = ~collision_flags
        axial = self._axial_index()
        axial_extent = bbox_extents[:, axial]
        if valid_mask.any():
            valid_axial = axial_extent[valid_mask]
            max_axial = float(valid_axial.max())
            if max_axial > 0:
                comp_ratio = (max_axial - float(valid_axial.min())) / max_axial
            else:
                comp_ratio = 0.0
        else:
            comp_ratio = 0.0

        locked, locking_info = self._compute_locking(mode, comp_ratio)
        return SimResult(
            theta_samples=theta_samples, poses=poses,
            bbox_extents=bbox_extents,
            compression_ratio=comp_ratio,
            locked=locked, locking_info=locking_info,
            collision_at_theta=collision_flags,
            collision_theta_min=collision_theta_min,
            collision_theta_max=collision_theta_max,
        )

    # ==================================================================
    # SPEC §7.4 Poisson's ratio
    # ==================================================================

    def poissons_ratio(self):
        """ν = -ε_lateral / ε_axial computed from a small perturbation
        along the kirigami mode at the rest pose.

        Returns ``np.nan`` (2D) or a 3-tuple with ``np.nan`` at the
        axial index (3D) when the axial strain is below 1e-12 (the
        mode doesn't compress the structure axially). Locking is
        tracked separately by ``is_locked``.

        For 2D, returns a single scalar.
        For 3D, returns ``(ν_x, ν_y, ν_z)`` with the axial-direction
        entry replaced by ``np.nan``."""
        mode = self.identify_kirigami_mode()
        if mode is None:
            return float("nan") if self.dimension == 2 else (
                float("nan"),) * 3

        delta = 1e-4
        rest = self.rest_pose()
        rest_extent      = self._bbox_extents(rest)
        perturbed_pose   = rest + delta * mode
        projected        = self.project_to_manifold(perturbed_pose)
        perturbed_extent = self._bbox_extents(projected)

        # Strain per dimension. Guard against zero rest extents (a
        # degenerate axis would mean the lattice has no thickness in
        # that direction — would only happen for hand-pathological
        # systems, but doesn't blow up here).
        with np.errstate(divide="ignore", invalid="ignore"):
            strain = np.where(
                rest_extent > 1e-12,
                (perturbed_extent - rest_extent) / rest_extent,
                0.0,
            )

        axial = self._axial_index()
        eps_axial = float(strain[axial])
        if abs(eps_axial) < 1e-12:
            # Pure-shear / rotational mode — no axial extension to
            # divide by. Locking criterion will flag this.
            return float("nan") if self.dimension == 2 else tuple(
                float("nan") for _ in range(3)
            )

        if self.dimension == 2:
            lateral = 1 - axial
            return float(-strain[lateral] / eps_axial)
        else:
            out = []
            for d in range(3):
                if d == axial:
                    out.append(float("nan"))
                else:
                    out.append(float(-strain[d] / eps_axial))
            return tuple(out)

    # ==================================================================
    # SPEC §7.5 locking detection
    # ==================================================================

    def _displacement_direction_of_mode(self, mode: np.ndarray) -> np.ndarray:
        """Direction along which the kirigami mode produces axial
        motion of the structure as a whole, unit-normalised. SPEC §7.5
        takes the dot product of this direction with the load axis as
        its locking metric.

        Computed as the bounding-box change direction under a small
        perturbation along the mode at rest. Captures the physically
        meaningful question — does the mode move the structure along
        this axis? — directly.

        Note: an earlier draft of this helper averaged the per-vertex
        displacement vectors and used that as the direction. That
        formulation cancels by construction for any symmetric auxetic
        mode (vertex displacements come in equal-and-opposite pairs;
        rotating squares is the canonical example), making every
        well-formed auxetic appear locked. SPEC §7.5 documents this
        dead end and prescribes the bbox-change definition used here.
        """
        delta = 1e-4
        rest_extent  = self._bbox_extents(self.rest_pose())
        perturbed    = self.rest_pose() + delta * mode
        new_extent   = self._bbox_extents(perturbed)
        delta_extent = new_extent - rest_extent
        norm = float(np.linalg.norm(delta_extent))
        if norm < 1e-12:
            return np.zeros(self.dimension)
        return delta_extent / norm

    def _compute_locking(self, mode: np.ndarray | None,
                          comp_ratio: float) -> tuple[bool, dict]:
        """Apply the SPEC §7.5 composite criterion. Shared between
        ``sweep_theta`` (which already has its own ``comp_ratio``) and
        ``is_locked`` (which runs a quick coarse sweep to obtain one)."""
        if mode is None:
            return (
                True,
                {
                    "mode_projection":   0.0,
                    "compression_ratio": float(comp_ratio),
                    "reason":            "no kirigami mode (system fully constrained)",
                },
            )
        mode_dir = self._displacement_direction_of_mode(mode)
        proj = abs(float(np.dot(mode_dir, self.load_axis)))

        reasons = []
        if proj < 0.05:
            reasons.append(
                f"mode misaligned with load axis (projection={proj:.3f})"
            )
        if comp_ratio < 0.05:
            reasons.append(f"compression ratio below 5% ({comp_ratio:.1%})")

        return (
            len(reasons) > 0,
            {
                "mode_projection":   float(proj),
                "compression_ratio": float(comp_ratio),
                "reason":            "; ".join(reasons) if reasons else "not locked",
            },
        )

    def is_locked(self) -> tuple[bool, dict]:
        """Composite SPEC §7.5 criterion: the lattice is locked if the
        kirigami mode is misaligned with the load axis (``|mode·load|
        < 0.05``) OR the compression ratio is below 5%.

        Runs a coarse θ-sweep (``n_steps=37``, every 5°) to obtain the
        compression ratio. Sufficient resolution to detect locking
        without the cost of the full 181-step sweep."""
        mode = self.identify_kirigami_mode()
        # Quick coarse sweep for the compression check; we deliberately
        # don't reuse this result outside is_locked because callers who
        # want the full 181-sample trajectory should call sweep_theta
        # themselves.
        result = self.sweep_theta(n_steps=37)
        return self._compute_locking(mode, result.compression_ratio)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_4x4_to_points(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Multiply every row of ``points`` (Nx3) by the 4×4 matrix ``M``."""
    n = len(points)
    homo = np.hstack([points, np.ones((n, 1))])
    return (M @ homo.T).T[:, :3]


def _is_planar(tile_arrays_3d: list[np.ndarray], tol: float = 1e-8) -> bool:
    """True when every vertex across every tile shares the same z
    (within ``tol``). Used by ``TileSystem.from_lattice`` to decide
    whether to drop the z-coord and run a 2D simulation."""
    if not tile_arrays_3d:
        return True
    all_z = np.concatenate([t[:, 2] for t in tile_arrays_3d])
    return float(all_z.max() - all_z.min()) < tol


def _skew(v: np.ndarray) -> np.ndarray:
    """3D cross-product matrix [v]_× such that [v]_× w = v × w."""
    return np.array([
        [0.0, -v[2],  v[1]],
        [v[2],  0.0, -v[0]],
        [-v[1], v[0],  0.0],
    ])


def _so3_rotation_matrix(omega: np.ndarray) -> np.ndarray:
    """``Rotation.from_rotvec(omega).as_matrix()`` short-circuited
    at ``omega ≈ 0`` to avoid scipy's small-angle warning path."""
    if np.linalg.norm(omega) < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(omega).as_matrix()


def _so3_right_jacobian(omega: np.ndarray) -> np.ndarray:
    """Right-Jacobian of SO(3): J_r(ω) such that
    ``∂(R(ω) v)/∂ω = -R(ω) [v]_× J_r(ω)``.

    Closed form:
        J_r(ω) = I - (1 - cos θ)/θ²  [ω]_×
                   + (θ - sin θ)/θ³ [ω]_×²
    where θ = ‖ω‖. At θ = 0, J_r = I (use the Taylor limit).
    """
    theta = float(np.linalg.norm(omega))
    if theta < 1e-9:
        return np.eye(3)
    K = _skew(omega)
    K2 = K @ K
    return (
        np.eye(3)
        - (1.0 - np.cos(theta)) / (theta * theta) * K
        + (theta - np.sin(theta)) / (theta ** 3) * K2
    )
