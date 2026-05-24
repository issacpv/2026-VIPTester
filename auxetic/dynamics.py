"""Newtonian rigid-body simulator for kirigami lattices (M2).

This is the *dynamic* counterpart to :mod:`auxetic.simulation`. The
kinematic solver in ``simulation.py`` answers "what does the lattice
look like at a given joint angle θ" by projecting onto the constraint
manifold; this module answers "how does the lattice respond when I
apply a force here, with mass and time and contact" by integrating
Newton's equations.

Algorithm
---------
Semi-implicit Euler with **Baumgarte-stabilized soft constraints**.
Tile constraints (the ``Constraint`` records that pin one tile vertex
to another) are enforced by a penalty potential rather than projected
hard each step — gives a single explicit time-step formulation that
matches the rest of the geometry pipeline's numpy/scipy stack with no
new dependencies (no PyBullet, no MuJoCo).

For each constraint with residual ``r = (R_a v_a + t_a) - (R_b v_b + t_b)``,
the generalised force in pose-space is

    F_q = -k * Jᵀ r   -   c * Jᵀ J ẋ

where ``J`` is the constraint Jacobian and ``ẋ`` is the pose-rate
(reused from :class:`auxetic.simulation.Simulator`). The first term is
a spring pulling the residual to zero; the second damps motion along
the same direction. The pair behaves like a critically damped
oscillator when ``c ≈ 2*sqrt(k * m_eff)``.

Pose layout
-----------
Same as :class:`Simulator`:

- 2D: ``[tx, ty, θ]`` per tile (3 DOFs)
- 3D: ``[tx, ty, tz, rx, ry, rz]`` per tile (6 DOFs; axis-angle)

Velocity has the same shape — translational for the translation DOFs,
**world-frame** angular velocity for the rotation DOFs. We compose
rotations through scipy's :class:`Rotation` rather than treating the
axis-angle rate as ``vel[3:6]`` directly, so large rotations stay
numerically clean.

Units
-----
Lattice space is unit-less (``[0, 1]^3`` per :file:`CLAUDE.md`); the
``unit_scale_cm`` field on the lattice maps lattice units to cm. This
module accepts forces in Newtons, masses in kilograms, gravity in
m/s² — the integrator consumes only *consistent* units, so as long as
the user's masses, forces, and dt agree the result is meaningful.
The :class:`DynamicsConfig` defaults assume 1 lattice unit = 1 cm and
plate-like tiles, but every value is overridable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .simulation import Constraint, Simulator, TileSystem


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class TileMass:
    """Inertial properties of a single tile.

    ``mass`` is the scalar mass in kg.

    ``inertia_iso`` is a single scalar moment of inertia used isotropically
    — ``I_world = inertia_iso * I``. We pick isotropic instead of a full
    body-frame tensor because (a) typical kirigami tiles are thin plates
    where the dominant out-of-plane component dominates the dynamics,
    and (b) it sidesteps the need to recompute ``R I_body Rᵀ`` every step.
    Refine to a per-tile body-frame tensor only if the simplification
    becomes a measurable bottleneck.
    """
    mass:        float
    inertia_iso: float


@dataclass(frozen=True)
class ForceVector:
    """An external force the user wants applied during the sim.

    ``location_kind`` selects the attachment:
    - ``"world"`` — fixed point in world space; force does NOT track
      tile motion. Useful for "press here" scenarios.
    - ``"tile_vertex"`` — attached to ``tiles[tile_index][vert_index]``;
      the application point follows the tile, so a non-zero magnitude
      ALSO induces torque on the tile.
    - ``"tile_centroid"`` — attached to the tile's vertex centroid;
      pure translation, no torque.

    ``direction`` is auto-normalised at construction.
    """
    location_kind: str           # "world" | "tile_vertex" | "tile_centroid"
    direction:     np.ndarray    # (dim,)
    magnitude:     float
    location_world: Optional[np.ndarray] = None  # (dim,), only for kind=world
    tile_index:    int = -1                       # only for kind=tile_*
    vert_index:    int = -1                       # only for kind=tile_vertex

    def __post_init__(self):
        d = np.asarray(self.direction, dtype=float).flatten()
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            raise ValueError("ForceVector direction must be non-zero")
        # __post_init__ on frozen dataclasses requires object.__setattr__.
        object.__setattr__(self, "direction", d / n)
        if self.location_world is not None:
            object.__setattr__(
                self, "location_world",
                np.asarray(self.location_world, dtype=float).flatten(),
            )


@dataclass(frozen=True)
class GroundContact:
    """A penalty-based ground plane.

    The plane is ``{x : (x - plane_point) · plane_normal == 0}`` with
    the half-space ``(x - plane_point) · plane_normal >= 0`` being
    *outside* (allowed). Tile vertices that penetrate (negative signed
    distance) get pushed back via:

    - normal force = ``-k * d * n``   (penalty spring)
    - normal damp  = ``-c * (v · n) * n``  (kills approach velocity)
    - friction     = clamped Coulomb at the tangential vertex velocity.
    """
    plane_point:  np.ndarray
    plane_normal: np.ndarray
    stiffness:    float = 1.0e4
    damping:      float = 50.0
    friction_mu:  float = 0.3

    def __post_init__(self):
        n = np.asarray(self.plane_normal, dtype=float).flatten()
        ln = float(np.linalg.norm(n))
        if ln < 1e-12:
            raise ValueError("GroundContact plane_normal must be non-zero")
        object.__setattr__(self, "plane_normal", n / ln)
        object.__setattr__(
            self, "plane_point",
            np.asarray(self.plane_point, dtype=float).flatten(),
        )


@dataclass
class DynamicsConfig:
    """Time-step and stiffness parameters for the integrator."""
    dt:                 float       = 1.0e-3
    duration:           float       = 2.0
    joint_stiffness:    float       = 1.0e3
    joint_damping:      float       = 5.0
    gravity:            np.ndarray  = field(
        default_factory=lambda: np.array([0.0, -9.81, 0.0]))
    convergence_kinetic_thresh: float = 1.0e-5  # KE/initial_KE → "settled"


@dataclass
class DynamicsResult:
    """Output of :meth:`DynamicsSimulator.simulate`."""
    times:        np.ndarray   # (n_steps,)
    poses:        np.ndarray   # (n_steps, n_tiles*dofs)
    velocities:   np.ndarray   # (n_steps, n_tiles*dofs)
    bbox_extents: np.ndarray   # (n_steps, dim)
    final_compression: float
    converged:    bool
    energy_trace: dict         # {"kinetic": (n,), ...}


# ---------------------------------------------------------------------------
# DynamicsSimulator
# ---------------------------------------------------------------------------

class DynamicsSimulator:
    """Newtonian rigid-body simulator for a constraint-graph
    :class:`TileSystem`.

    Wraps a :class:`auxetic.simulation.Simulator` to reuse its
    constraint Jacobian / residual primitives unchanged — the kinematic
    and dynamic solvers operate on the same algebraic objects, just at
    different time-scales.
    """

    def __init__(self,
                 tile_system: TileSystem,
                 masses: List[TileMass],
                 config: DynamicsConfig,
                 forces: Optional[List[ForceVector]] = None,
                 ground: Optional[GroundContact] = None,
                 fixed_tiles: Optional[List[int]] = None):
        if len(masses) != tile_system.n_tiles:
            raise ValueError(
                f"masses length {len(masses)} != n_tiles {tile_system.n_tiles}")
        self.tile_system = tile_system
        self.masses      = list(masses)
        self.config      = config
        self.forces      = list(forces or [])
        self.ground      = ground
        self.fixed_tiles = sorted(set(int(i) for i in (fixed_tiles or [])))

        # Use a Simulator to get the residual/Jacobian primitives.
        # The load_axis is irrelevant here — only used by Simulator's
        # SPEC §7.5 metric — but the constructor demands a non-zero
        # vector, so feed it gravity's direction (or any unit vec).
        self._dim   = tile_system.dimension
        self._dofs  = 3 if self._dim == 2 else 6
        load_axis = np.zeros(self._dim)
        if self._dim == 2:
            load_axis[1] = -1.0   # -y
        else:
            load_axis[1] = -1.0   # -y, world up = +y
        self._sim = Simulator(tile_system, load_axis=load_axis)

        # Project gravity onto the simulation dimension. The user-facing
        # config.gravity is a 3-vector for ergonomics.
        g3 = np.asarray(config.gravity, dtype=float).flatten()
        if g3.size < self._dim:
            raise ValueError("config.gravity must be at least dim-D")
        self._gravity = g3[:self._dim].copy()

        # Per-tile mass-matrix diagonal in pose-space (one entry per DOF).
        # Shape: (n_tiles * dofs,)
        n_tiles = tile_system.n_tiles
        Mdiag = np.empty(n_tiles * self._dofs, dtype=float)
        for i, tm in enumerate(self.masses):
            base = i * self._dofs
            Mdiag[base + 0] = tm.mass
            Mdiag[base + 1] = tm.mass
            if self._dim == 2:
                Mdiag[base + 2] = tm.inertia_iso
            else:
                Mdiag[base + 2] = tm.mass
                Mdiag[base + 3] = tm.inertia_iso
                Mdiag[base + 4] = tm.inertia_iso
                Mdiag[base + 5] = tm.inertia_iso
        # Fixed tiles get effectively-infinite mass: their generalised
        # accelerations come out at zero, which is what we want.
        for ti in self.fixed_tiles:
            base = ti * self._dofs
            Mdiag[base:base + self._dofs] = np.inf
        self._Mdiag = Mdiag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_tiles(self) -> int:
        return self.tile_system.n_tiles

    @property
    def n_dofs(self) -> int:
        return self.tile_system.n_tiles * self._dofs

    def rest_pose(self) -> np.ndarray:
        return self._sim.rest_pose()

    def step(self, pose: np.ndarray, vel: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one ``config.dt`` step. Pure function, no mutation."""
        F_q = self._compute_generalised_force(pose, vel)
        # a = M^-1 F_q. Inf entries (fixed tiles) → 0 acceleration.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(np.isinf(self._Mdiag), 0.0, F_q / self._Mdiag)
        new_vel  = vel + a * self.config.dt
        # Zero out fixed-tile velocities exactly (avoid float drift).
        for ti in self.fixed_tiles:
            base = ti * self._dofs
            new_vel[base:base + self._dofs] = 0.0
        new_pose = self._retract(pose, new_vel * self.config.dt)
        return new_pose, new_vel

    def simulate(self,
                  initial_pose: Optional[np.ndarray] = None,
                  initial_vel:  Optional[np.ndarray] = None,
                  ) -> DynamicsResult:
        """Run the integrator from ``t=0`` to ``config.duration``."""
        n_steps = max(1, int(np.round(self.config.duration / self.config.dt))) + 1
        times = np.linspace(0.0, self.config.duration, n_steps)

        n_dof = self.n_dofs
        pose = (np.asarray(initial_pose, dtype=float).copy()
                if initial_pose is not None else np.zeros(n_dof))
        vel  = (np.asarray(initial_vel, dtype=float).copy()
                if initial_vel  is not None else np.zeros(n_dof))

        poses      = np.empty((n_steps, n_dof), dtype=float)
        velocities = np.empty((n_steps, n_dof), dtype=float)
        bbox_extents = np.empty((n_steps, self._dim), dtype=float)
        ke_trace   = np.empty(n_steps, dtype=float)

        poses[0] = pose
        velocities[0] = vel
        bbox_extents[0] = self._bbox_extents(pose)
        ke_trace[0] = self._kinetic_energy(vel)

        initial_ke = max(ke_trace[0], 1.0e-12)
        converged = False
        diverged_at: Optional[int] = None
        # Lattice space is [0, 1] per CLAUDE.md, so a stable trajectory
        # rarely sees pose components beyond a few units (the piston
        # auto-pin keeps the lattice anchored). Anything past 5 lattice
        # units of bbox-equivalent extent is clearly the explicit-Euler
        # integrator running away. Catching it early lets us clamp
        # BEFORE the numbers actually reach Inf, so bbox_extents stays
        # sane and the renderer never sees garbage. This trips on the
        # default (mode-1, dt=1ms, 5N piston, 1s duration) load case
        # which is unstable for the default lattice; sane test configs
        # stay well below this bound.
        _DIVERGE_POS_BOUND = 5.0
        for k in range(1, n_steps):
            pose, vel = self.step(pose, vel)
            # Divergence guard — explicit Euler with stiff penalty
            # constraints can blow up if dt is too large for the picked
            # joint_stiffness / mass combination. Detect either (a) a
            # non-finite pose/velocity, or (b) a runaway magnitude
            # before it overflows. Once detected, restoring is hopeless:
            # snap back to the last finite pose, freeze velocity, and
            # copy that pose into every remaining slot. Callers see a
            # clamped trajectory rather than NaN/Inf garbage that would
            # crash the renderer.
            runaway = (
                not (np.all(np.isfinite(pose)) and np.all(np.isfinite(vel)))
                or float(np.max(np.abs(pose))) > _DIVERGE_POS_BOUND
            )
            if runaway:
                diverged_at = k
                pose = poses[k - 1].copy()
                vel  = np.zeros_like(vel)
                last_bbox = bbox_extents[k - 1].copy()
                last_ke   = float(ke_trace[k - 1])
                for j in range(k, n_steps):
                    poses[j]        = pose
                    velocities[j]   = vel
                    bbox_extents[j] = last_bbox
                    ke_trace[j]     = last_ke
                break
            poses[k]      = pose
            velocities[k] = vel
            bbox_extents[k] = self._bbox_extents(pose)
            ke_trace[k]   = self._kinetic_energy(vel)
            # Convergence: kinetic energy small fraction of initial,
            # AND we've taken at least 10 steps to give the system
            # time to actually accelerate before "settling".
            if (k > 10
                and ke_trace[k] < self.config.convergence_kinetic_thresh
                                  * initial_ke):
                converged = True
                # Trim trailing slots (still report the full grid for
                # debugging — set converged so callers can tell).
                # We keep the loop running so the time grid stays
                # consistent.

        # Compression along the dominant load axis.
        load_axis_idx = int(np.argmax(np.abs(self._sim.load_axis)))
        ax0 = bbox_extents[0,  load_axis_idx]
        ax1 = bbox_extents[-1, load_axis_idx]
        final_compression = (
            float((ax0 - ax1) / ax0) if abs(ax0) > 1e-12 else 0.0)

        return DynamicsResult(
            times=times,
            poses=poses,
            velocities=velocities,
            bbox_extents=bbox_extents,
            final_compression=final_compression,
            converged=converged,
            energy_trace={"kinetic": ke_trace},
        )

    # ------------------------------------------------------------------
    # Generalised force assembly
    # ------------------------------------------------------------------

    def _compute_generalised_force(self, pose: np.ndarray,
                                    vel: np.ndarray) -> np.ndarray:
        """Sum of constraint, gravity, user-force, and contact
        contributions in pose-space."""
        F_q = np.zeros(self.n_dofs, dtype=float)
        F_q += self._constraint_force(pose, vel)
        F_q += self._gravity_force()
        F_q += self._user_forces(pose)
        if self.ground is not None:
            F_q += self._contact_force(pose, vel)
        return F_q

    def _constraint_force(self, pose: np.ndarray,
                           vel:  np.ndarray) -> np.ndarray:
        """Baumgarte-stabilised constraint generalised force."""
        if self.tile_system.n_constraints == 0:
            return np.zeros(self.n_dofs, dtype=float)
        r = self._sim.constraint_residual(pose)        # (n_c * dim,)
        J = self._sim.assemble_jacobian(pose)          # (n_c*dim, n_dof)
        k = self.config.joint_stiffness
        c = self.config.joint_damping
        return -(k * (J.T @ r) + c * (J.T @ (J @ vel)))

    def _gravity_force(self) -> np.ndarray:
        """Gravity acts at each tile's centroid as ``m * g`` (no torque
        because the tile's mass distribution is treated as centred on
        the centroid)."""
        F_q = np.zeros(self.n_dofs, dtype=float)
        for i, tm in enumerate(self.masses):
            if i in self.fixed_tiles:
                continue
            base = i * self._dofs
            F_q[base:base + self._dim] += tm.mass * self._gravity
        return F_q

    def _user_forces(self, pose: np.ndarray) -> np.ndarray:
        """Apply each :class:`ForceVector`. Includes torque contribution
        when the application point is offset from the tile centroid."""
        F_q = np.zeros(self.n_dofs, dtype=float)
        if not self.forces:
            return F_q
        for fv in self.forces:
            F = fv.direction[:self._dim] * fv.magnitude
            if fv.location_kind == "world":
                # World-frame force — find the closest tile to apply
                # the *reaction* on. For v1 we assume world forces aren't
                # attached to any tile and so contribute nothing. (User
                # forces tied to tiles is the common case; pure world
                # forces are uncommon and would require a contact-style
                # treatment to bring them into the tile dynamics.)
                continue
            ti = fv.tile_index
            if ti < 0 or ti >= self.n_tiles or ti in self.fixed_tiles:
                continue
            base = ti * self._dofs
            F_q[base:base + self._dim] += F
            # Torque from the offset of application from tile origin.
            t_i, R_i = self._sim._decompose_pose(pose, ti)
            if fv.location_kind == "tile_vertex":
                if not (0 <= fv.vert_index < self.tile_system.tiles[ti].shape[0]):
                    continue
                v_body = self.tile_system.tiles[ti][fv.vert_index]
                r_world = R_i @ v_body  # offset from tile origin
            else:  # tile_centroid → no torque
                continue
            if self._dim == 2:
                # 2D cross product yields a scalar torque about z.
                tau = float(r_world[0] * F[1] - r_world[1] * F[0])
                F_q[base + 2] += tau
            else:
                tau = np.cross(r_world, F)
                F_q[base + 3:base + 6] += tau
        return F_q

    def _contact_force(self, pose: np.ndarray,
                        vel:  np.ndarray) -> np.ndarray:
        """Penalty + damping + Coulomb-friction contact at the ground
        plane. Applied per tile vertex that has penetrated."""
        F_q = np.zeros(self.n_dofs, dtype=float)
        gc = self.ground
        if gc is None:
            return F_q
        n = gc.plane_normal[:self._dim]
        plane_pt = gc.plane_point[:self._dim]
        for ti in range(self.n_tiles):
            if ti in self.fixed_tiles:
                continue
            verts_world = self._sim._tile_world_vertices(pose, ti)
            t_i, R_i = self._sim._decompose_pose(pose, ti)
            base = ti * self._dofs
            v_lin = vel[base:base + self._dim]
            if self._dim == 2:
                v_ang = float(vel[base + 2])
            else:
                v_ang = vel[base + 3:base + 6]
            for vert_i, p_world in enumerate(verts_world):
                d = float((p_world - plane_pt) @ n)
                if d >= 0.0:
                    continue   # outside / on the surface
                # Vertex world velocity.
                r = p_world - t_i
                if self._dim == 2:
                    # ω is scalar; v_vert = v_lin + ω × r
                    v_vert = v_lin + v_ang * np.array([-r[1], r[0]])
                else:
                    v_vert = v_lin + np.cross(v_ang, r)
                vn = float(v_vert @ n)
                # Normal: penalty spring + damping along approach.
                F_normal = (-gc.stiffness * d - gc.damping * vn) * n
                # Friction: tangential, clamped Coulomb.
                v_t = v_vert - vn * n
                speed_t = float(np.linalg.norm(v_t))
                if speed_t > 1.0e-9:
                    f_friction_mag = min(
                        gc.friction_mu * float(np.linalg.norm(F_normal)),
                        speed_t * 1.0,  # avoid amplifying past static velocity
                    )
                    F_friction = -f_friction_mag * (v_t / speed_t)
                else:
                    F_friction = np.zeros(self._dim)
                F_total = F_normal + F_friction
                # Translational + torque about tile origin.
                F_q[base:base + self._dim] += F_total
                if self._dim == 2:
                    tau = float(r[0] * F_total[1] - r[1] * F_total[0])
                    F_q[base + 2] += tau
                else:
                    tau = np.cross(r, F_total)
                    F_q[base + 3:base + 6] += tau
        return F_q

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retract(self, pose: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Apply a pose-space delta. Translation: vector add. Rotation:
        compose with the rotation parameterised by ``dq[rot]`` (treated
        as a world-frame angular-velocity step) using SO(3) / SO(2)
        composition."""
        out = pose.copy()
        for i in range(self.n_tiles):
            base = i * self._dofs
            # Translation.
            out[base:base + self._dim] += dq[base:base + self._dim]
            # Rotation.
            if self._dim == 2:
                out[base + 2] += dq[base + 2]
            else:
                # Compose: R_new = R(dq[rot]) @ R(pose[rot]).
                rot_old = Rotation.from_rotvec(pose[base + 3:base + 6])
                d_omega = dq[base + 3:base + 6]
                if np.linalg.norm(d_omega) > 0.0:
                    rot_d = Rotation.from_rotvec(d_omega)
                    rot_new = rot_d * rot_old
                else:
                    rot_new = rot_old
                out[base + 3:base + 6] = rot_new.as_rotvec()
        return out

    def _bbox_extents(self, pose: np.ndarray) -> np.ndarray:
        """Bounding-box ``(max - min)`` per spatial dimension over all
        tile vertices in the projected configuration."""
        all_verts = []
        for ti in range(self.n_tiles):
            all_verts.append(self._sim._tile_world_vertices(pose, ti))
        verts = np.vstack(all_verts) if all_verts else np.zeros((1, self._dim))
        return verts.max(axis=0) - verts.min(axis=0)

    def _kinetic_energy(self, vel: np.ndarray) -> float:
        """``½ vᵀ M v`` — fixed-tile inf masses contribute 0 (they have
        zero velocity by construction)."""
        ke = 0.0
        for i in range(self.n_tiles):
            if i in self.fixed_tiles:
                continue
            base = i * self._dofs
            for k in range(self._dofs):
                m = self._Mdiag[base + k]
                if not np.isfinite(m):
                    continue
                ke += 0.5 * m * vel[base + k] ** 2
        return float(ke)


# ---------------------------------------------------------------------------
# Convenience: derive default tile masses from geometry
# ---------------------------------------------------------------------------

class _PistonKinematicSimulator:
    """``DynamicsSimulator``-shaped wrapper that drives the kirigami's
    kinematic soft mode under a piston compression load case.

    Physically motivated: a kirigami lattice compressed along its load
    axis collapses predominantly along its lowest-energy deformation
    mode, which is exactly the kinematic θ-mode that
    :class:`auxetic.simulation.Simulator` already computes. Driving
    the lattice along that mode produces visible auxetic buckling,
    monotonically-changing internal joint angles, and a stable trajectory
    — none of which the explicit-Euler Newtonian integrator can deliver
    for typical kirigami stiffness/mass scales without going unstable.

    The class implements the same surface as :class:`DynamicsSimulator`
    that the GUI / tests touch: ``tile_system``, ``forces``,
    ``fixed_tiles``, and ``simulate() -> DynamicsResult``.

    The compression direction is chosen automatically: we run a full
    ±π/2 sweep (with collision detection), measure how much the lattice
    contracts along the world load axis on each half, and keep the
    half-trajectory that compresses more. The kept half is then
    re-indexed to start at θ=0 (rest) and progress monotonically toward
    maximum compression — that's what the slider scrubbing visualises.
    """

    def __init__(self,
                 tile_system: TileSystem,
                 config: DynamicsConfig,
                 *,
                 forces: List["ForceVector"],
                 fixed_tiles: List[int],
                 piston_force_n: float):
        self.tile_system   = tile_system
        self.config        = config
        self.forces        = list(forces)
        self.fixed_tiles   = list(fixed_tiles)
        self.piston_force_n = float(piston_force_n)
        # Compatibility shims with :class:`DynamicsSimulator`'s public
        # surface so existing tests / callers don't have to special-case
        # the wrapper. ``ground`` is always None in piston mode (we pin
        # tiles instead of using a contact plane).
        self.ground = None
        self.n_tiles = tile_system.n_tiles
        # Constructed lazily in ``simulate`` so we don't pay the cost
        # at build time if the caller only inspects the wrapper.
        self._kine_sim: Optional[Simulator] = None

    # ------------------------------------------------------------------

    def simulate(self,
                 initial_pose: Optional[np.ndarray] = None,
                 initial_vel:  Optional[np.ndarray] = None,
                 ) -> DynamicsResult:
        """Run a kinematic θ-sweep, repackaged as a ``DynamicsResult``
        whose times span ``[0, config.duration]`` and whose poses are
        the constraint-projected sequence along the auxetic mode."""
        # Load axis convention matches Simulator.run_simulation in the
        # GUI: world -Y in both 2D and 3D. The piston pushes downward
        # (toward -Y), so positive compression is bbox shrinking along
        # +Y direction.
        ts = self.tile_system
        if ts.dimension == 2:
            load_axis = np.array([0.0, -1.0])
        else:
            load_axis = np.array([0.0, -1.0, 0.0])
        sim = Simulator(ts, load_axis=load_axis)
        self._kine_sim = sim

        # Sweep across the full bistable range with collision detection
        # so we can stop the piston where the lattice physically locks.
        n_steps = max(2, int(np.round(self.config.duration / self.config.dt)) + 1)
        # Cap n_steps to keep the kinematic projection cost reasonable
        # — 401 samples is plenty for animation; smaller bumps the cost
        # of the GUI rendering loop too.
        sweep_n = min(401, n_steps)
        sweep = sim.sweep_theta(
            n_steps=sweep_n, theta_max=float(np.pi / 2.0),
            collision_stop=True,
        )

        # Pick the half (negative or positive θ) that actually compresses
        # the lattice along the load axis. Both halves are kirigami modes;
        # which one matches "piston pushing down" depends on the sign of
        # the kirigami mode vector, which is arbitrary.
        axial_idx = sim._axial_index()
        axial = sweep.bbox_extents[:, axial_idx]
        thetas = sweep.theta_samples
        rest_idx = int(np.argmin(np.abs(thetas)))
        rest_extent = float(axial[rest_idx])

        # Negative half: thetas < 0, walk toward 0 from the most
        # negative end.
        neg_mask = thetas < 0.0
        pos_mask = thetas > 0.0
        # Stop each half at the first collision (if any).
        if sweep.collision_theta_min is not None:
            neg_mask &= thetas >= sweep.collision_theta_min
        if sweep.collision_theta_max is not None:
            pos_mask &= thetas <= sweep.collision_theta_max

        neg_min_extent = (float(axial[neg_mask].min())
                           if neg_mask.any() else rest_extent)
        pos_min_extent = (float(axial[pos_mask].min())
                           if pos_mask.any() else rest_extent)
        neg_compresses = rest_extent - neg_min_extent
        pos_compresses = rest_extent - pos_min_extent

        if neg_compresses >= pos_compresses:
            # Walk from θ=0 toward most-negative θ.
            half_indices = np.where(thetas <= 0.0)[0][::-1]
            if sweep.collision_theta_min is not None:
                # Drop indices past the collision boundary.
                half_indices = [
                    i for i in half_indices
                    if thetas[i] >= sweep.collision_theta_min
                ]
        else:
            half_indices = np.where(thetas >= 0.0)[0]
            if sweep.collision_theta_max is not None:
                half_indices = [
                    i for i in half_indices
                    if thetas[i] <= sweep.collision_theta_max
                ]
        half_indices = list(half_indices)
        if not half_indices:
            # Pathological: no kirigami mode (rigid system). Return a
            # trivial trajectory at rest so the GUI doesn't crash.
            half_indices = [rest_idx]

        # Resample the half-trajectory onto a uniform time grid of
        # ``n_steps`` samples spanning [0, duration].
        n_out = max(2, n_steps)
        times = np.linspace(0.0, self.config.duration, n_out)
        n_kept = len(half_indices)
        # Map output samples to half-trajectory samples — t=0 → rest,
        # t=duration → max compression.
        out_floats = np.linspace(0.0, n_kept - 1, n_out)
        out_idx = np.clip(np.round(out_floats).astype(int), 0, n_kept - 1)

        n_dof = sweep.poses.shape[1]
        poses        = np.empty((n_out, n_dof), dtype=float)
        velocities   = np.zeros((n_out, n_dof), dtype=float)
        bbox_extents = np.empty((n_out, sweep.bbox_extents.shape[1]),
                                  dtype=float)
        for k, oi in enumerate(out_idx):
            src = half_indices[oi]
            poses[k]        = sweep.poses[src]
            bbox_extents[k] = sweep.bbox_extents[src]

        # Numerical velocity (for energy bookkeeping). Forward
        # difference except the last frame, which falls back to zero.
        if n_out > 1:
            dt_eff = times[1] - times[0] if times[1] > times[0] else 1.0
            velocities[:-1] = (poses[1:] - poses[:-1]) / dt_eff

        # KE proxy — quadratic in vel norm. Doesn't reflect actual mass
        # since this is a kinematic trajectory, but gives the GUI a
        # plausible energy curve to show.
        ke_trace = 0.5 * np.einsum('ij,ij->i', velocities, velocities)

        # Compression along load axis: (initial - final) / initial.
        ax0 = float(bbox_extents[0, axial_idx])
        ax1 = float(bbox_extents[-1, axial_idx])
        final_compression = (
            float((ax0 - ax1) / ax0) if abs(ax0) > 1e-12 else 0.0)

        return DynamicsResult(
            times=times,
            poses=poses,
            velocities=velocities,
            bbox_extents=bbox_extents,
            final_compression=final_compression,
            converged=True,
            energy_trace={"kinetic": ke_trace},
        )


def _ground_face_plane(face: str, all_vertices: np.ndarray, dim: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Convert a face label like ``"+y"`` / ``"-y"`` into a
    ``(plane_point, plane_normal)`` pair anchored at the lattice's
    bounding box.

    The label names which face of the bbox is the *contact face*. The
    plane normal points **into free space** (away from the lattice
    interior), so a tile vertex sitting on the contact plane has
    signed distance 0; penetration is negative; above the plane is
    positive.

    Example: ``face="-y"`` says the lattice rests on a floor at
    ``y = min(y)``. Free space is above (``+y``), so ``plane_normal
    = (0, +1, ...)``.
    """
    axis = {"x": 0, "y": 1, "z": 2}.get(face[-1].lower())
    if axis is None or face[0] not in "+-" or axis >= dim:
        raise ValueError(f"invalid ground_face: {face!r}")
    bbox_min = all_vertices.min(axis=0)
    bbox_max = all_vertices.max(axis=0)
    if face[0] == "-":
        # Lattice's MIN-axis face is the contact face. Floor at min,
        # free space above (positive axis direction).
        plane_point = bbox_min.copy()
        plane_normal = np.zeros(dim)
        plane_normal[axis] = +1.0
    else:
        # MAX-axis face is the contact face. Ceiling at max, free
        # space below (negative axis direction).
        plane_point = bbox_max.copy()
        plane_normal = np.zeros(dim)
        plane_normal[axis] = -1.0
    return plane_point, plane_normal


def _tiles_touching_face(tile_system: TileSystem,
                          plane_point: np.ndarray,
                          plane_normal: np.ndarray,
                          tol: float = 1e-6) -> List[int]:
    """Return tile indices that have at least one vertex within ``tol``
    of the given plane (signed distance ≤ tol). Used to auto-pin the
    "ground face" tiles when running the dynamic simulator."""
    out: List[int] = []
    n = plane_normal[: plane_point.size]
    for ti, tile in enumerate(tile_system.tiles):
        d = (tile - plane_point) @ n
        if float(np.min(d)) <= tol:
            out.append(ti)
    return out


def _piston_setup(tile_system: TileSystem,
                   piston_force_n: float,
                   *,
                   top_fraction: float    = 0.15,
                   bottom_fraction: float = 0.15,
                   axis_idx: int          = 1,    # world-y is "vertical"
                   ) -> tuple[List["ForceVector"], List[int]]:
    """Auto-configure a piston compression load case.

    Returns ``(forces, fixed_tiles)``:

    - ``forces``       : per-vertex downward forces on the top
      ``top_fraction`` slab of tile vertices. Total downward force
      across the slab equals ``piston_force_n``; per-vertex magnitude
      is therefore ``piston_force_n / N_top``.
    - ``fixed_tiles``  : tile indices with at least one vertex in the
      bottom ``bottom_fraction`` slab. These get pinned in place to
      simulate the lattice resting on a base plate.

    Top / bottom are determined in ``tile_system``'s world frame,
    which already incorporates the lattice's rigid_rotation (the
    pre-rotation the user set via the Inspector). So if the user
    rotated the lattice 45° about Z, the piston pushes from the
    correct "top" of the rotated lattice.
    """
    if piston_force_n <= 0.0 or tile_system.n_tiles == 0:
        return [], []

    dim = tile_system.dimension
    if axis_idx >= dim:
        # 2D lattices live in XY; treat Y as vertical.
        axis_idx = 1

    # Find global axial extent across every tile vertex.
    all_v = np.vstack(tile_system.tiles)
    axial = all_v[:, axis_idx]
    lo = float(axial.min())
    hi = float(axial.max())
    span = hi - lo
    if span < 1e-12:
        return [], []
    bottom_threshold = lo + bottom_fraction * span
    top_threshold    = hi - top_fraction    * span

    # Identify top vertices (per tile, per vertex) and bottom tiles.
    top_targets: list[tuple[int, int]] = []
    bottom_tiles: list[int] = []
    for ti, tile in enumerate(tile_system.tiles):
        verts = np.asarray(tile, dtype=float)
        col = verts[:, axis_idx]
        # Bottom: any vertex sitting on the base plate → pin the tile.
        if float(col.min()) <= bottom_threshold:
            bottom_tiles.append(ti)
        # Top: every vertex in the top slab is a force-application point.
        for vi in range(verts.shape[0]):
            if float(col[vi]) >= top_threshold:
                top_targets.append((ti, vi))

    if not top_targets:
        return [], bottom_tiles

    per_vertex_mag = float(piston_force_n) / float(len(top_targets))
    direction = np.zeros(dim, dtype=float)
    direction[axis_idx] = -1.0   # downward along world +axis

    forces: List[ForceVector] = []
    for ti, vi in top_targets:
        # Skip the tile if it's also pinned at the bottom — pushing a
        # fixed tile is wasted force.
        if ti in bottom_tiles:
            continue
        forces.append(ForceVector(
            location_kind="tile_vertex",
            direction=direction.copy(),
            magnitude=per_vertex_mag,
            tile_index=int(ti),
            vert_index=int(vi),
        ))
    return forces, bottom_tiles


def build_dynamics_simulator_from_lattice(
        lattice,
        *,
        tile_system: Optional[TileSystem] = None,
        ) -> "DynamicsSimulator":
    """Construct a :class:`DynamicsSimulator` from a live :class:`Lattice`.

    Reads ``lattice.dynamics_state`` (the v4-preset dict) for forces,
    ground face, fixed tiles, and integrator config. Per-tile masses
    come from :func:`default_masses_from_tile_system`.

    Two operating modes — selected by ``dynamics_state['piston_force_n']``:

    - ``> 0``: **piston compression** mode. Auto-pins the bottom of
      the lattice (in world frame, after the user's rigid_rotation)
      and applies a downward force totalling that magnitude on the
      top vertices. The user's manual ``ground_face`` and ``forces``
      list are ignored in this mode — the auto-config replaces them.
    - ``== 0``: **manual** mode. Reads ``ground_face``, ``forces``,
      ``fixed_tiles`` directly from ``dynamics_state`` (the original
      M2.6 behavior). Use this when you need a custom load case the
      piston abstraction can't express.

    Pass an existing ``tile_system`` to reuse one already built (e.g.
    by the kinematic panel); otherwise one is constructed lazily here.
    """
    if tile_system is None:
        tile_system = TileSystem.from_lattice(lattice)
    masses  = default_masses_from_tile_system(tile_system)
    state   = dict(getattr(lattice, "dynamics_state", None) or {})
    cfg_in  = dict(state.get("config") or {})
    gravity_in = np.asarray(
        cfg_in.get("gravity_cm_per_s2", [0.0, -981.0, 0.0]),
        dtype=float,
    )
    cfg = DynamicsConfig(
        dt              = float(cfg_in.get("dt", 1.0e-3)),
        duration        = float(cfg_in.get("duration", 2.0)),
        joint_stiffness = float(cfg_in.get("joint_stiffness", 1.0e3)),
        joint_damping   = float(cfg_in.get("joint_damping",   5.0)),
        gravity         = gravity_in,
        convergence_kinetic_thresh = float(
            cfg_in.get("convergence_kinetic_thresh", 1.0e-5)),
    )

    dim = tile_system.dimension

    # ---- Piston mode (the primary "Run Dynamic" workflow) ----------
    piston_force = float(state.get("piston_force_n", 0.0) or 0.0)
    if piston_force > 0.0:
        forces, fixed_tiles = _piston_setup(tile_system, piston_force)
        # The Newtonian integrator with soft penalty constraints is too
        # stiff for the kirigami constraint graph: explicit Euler
        # diverges before the auxetic mode meaningfully engages, so the
        # user sees a clamped pose rather than buckling. Physically,
        # though, a kirigami compressed along its load axis collapses
        # along its *soft mode* — the kinematic θ-sweep already
        # computed by :class:`Simulator`. So in piston mode we drive
        # the lattice along that mode and shape the trajectory like a
        # piston-compression run: rest → maximum reachable compression
        # under collision, over ``cfg.duration`` seconds. Tiles
        # cooperatively rotate (auxetic buckling), internal joint
        # angles change monotonically, and the lattice contracts
        # along the load axis. The dynamics-style integrator is still
        # available via the manual mode below.
        return _PistonKinematicSimulator(
            tile_system, cfg,
            forces=forces, fixed_tiles=fixed_tiles,
            piston_force_n=piston_force,
        )

    # ---- Manual mode: read forces / ground / fixed_tiles directly --
    forces: List[ForceVector] = []
    for f in state.get("forces", []) or []:
        try:
            d = np.asarray(f["direction"], dtype=float).flatten()[:dim]
            forces.append(ForceVector(
                location_kind = str(f["location_kind"]),
                direction     = d,
                magnitude     = float(f.get("magnitude", 0.0)),
                tile_index    = int(f.get("tile_index", -1)),
                vert_index    = int(f.get("vert_index", -1)),
                location_world = (
                    np.asarray(f["location_world"], dtype=float).flatten()[:dim]
                    if f.get("location_world") is not None else None
                ),
            ))
        except (KeyError, ValueError):
            continue   # malformed force record — skip gracefully

    ground: Optional[GroundContact] = None
    fixed_tiles = [int(i) for i in (state.get("fixed_tiles") or [])]
    gf = state.get("ground_face")
    if gf is not None:
        all_verts = np.vstack(tile_system.tiles)
        plane_point, plane_normal = _ground_face_plane(str(gf), all_verts, dim)
        ground = GroundContact(
            plane_point  = plane_point,
            plane_normal = plane_normal,
        )
        for ti in _tiles_touching_face(tile_system, plane_point, plane_normal):
            if ti not in fixed_tiles:
                fixed_tiles.append(ti)

    return DynamicsSimulator(
        tile_system, masses, cfg,
        forces=forces, ground=ground, fixed_tiles=fixed_tiles,
    )


def default_masses_from_tile_system(
        tile_system: TileSystem,
        density: float   = 1000.0,   # bumped from 1.0 for integrator stability
        thickness: float = 0.10,     # bumped from 0.01
        ) -> List[TileMass]:
    """Compute a reasonable per-tile :class:`TileMass` from the tile
    geometry alone.

    For 2D tiles, mass = ``area * thickness * density`` (treating the
    tile as a thin plate of given thickness). For 3D tiles, mass is the
    convex-hull volume × density. ``inertia_iso`` is approximated as
    ``(m / 12) * sum(extent²)`` over the tile's bounding box — exact for
    a uniform rectangular plate, a useful first approximation otherwise.

    The defaults were re-tuned in the M3-polish piston pass: with the
    earlier ``density=1.0, thickness=0.01`` defaults a small kirigami
    tile (area ~0.05 in unit-cube space) gets a mass of ~5e-4. A 5 N
    piston force per vertex → acceleration of ~10⁴ m/s², which blows
    up explicit Euler at the 1 ms timestep we use. Bumping mass by 10⁴
    keeps accelerations under control without changing the integrator.
    The user can override either knob to model a specific physical
    scenario; the defaults are picked for "Run Dynamic just works on a
    fresh lattice".
    """
    masses: List[TileMass] = []
    for tile in tile_system.tiles:
        verts = np.asarray(tile, dtype=float)
        if verts.shape[1] == 2:
            # 2D polygon area via the shoelace formula.
            x = verts[:, 0]; y = verts[:, 1]
            area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            m = max(1.0e-3, float(area * thickness * density))
        else:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(verts)
                vol = float(hull.volume)
            except Exception:
                vol = 0.0
            m = max(1.0e-3, float(vol * density))
        ext = verts.max(axis=0) - verts.min(axis=0)
        I = (m / 12.0) * float(np.sum(ext * ext))
        masses.append(TileMass(mass=m, inertia_iso=max(I, 1.0e-9)))
    return masses
