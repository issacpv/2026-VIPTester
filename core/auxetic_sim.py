"""
Auxetic simulation — 3-D hub-and-tet mechanism.

Architecture
────────────
Central hub   → Truncated-cuboctahedron (TCO) rigid body, force-driven to
                spin in place each step.

Surrounding   → Each non-corner lattice point owns one n-gon PRISM body
hub prisms      (hexagonal, octagonal, …) whose centre is pinned to the
                lattice position by a POINT2POINT-to-world constraint.
                That pin allows free rotation but prevents drift, so the
                prism can only spin about its own centre.

Corner prisms → Same prism geometry but mass = 0 (static).  They act as
                the outermost fixed anchors of the mechanism.

Tetrahedra    → Convex rigid bodies with POINT2POINT ball joints connecting
                each of their four corners to whichever hub-prism (or TCO)
                body owns that lattice vertex.  The joint pivot is defined
                in each hub body's LOCAL frame at the shrunk-corner offset.

Mechanism
─────────
1. TCO spins  →  its local-frame pivot vectors sweep arcs in world space
2.            →  ball joints drag the tet corners along those arcs
3.            →  each tet rotates
4.            →  tet corners attached to surrounding prisms exert off-centre
                 forces on those prism bodies
5.            →  each prism experiences a net torque  →  it spins about its
                 anchored centre
6.            →  the next ring of tets is dragged  →  wave propagates out
"""

import sys
import os
import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'grid'))
import displayAuxeticV20 as _dav                   # noqa: E402
from displayAuxeticV20 import (                    # noqa: E402
    generate_points,
    collect_kirigami_tiles,
    build_kirigami_constraints,
    export_kirigami_vertices,
    export_kirigami_constraints,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.geometry import create_solid_geometry, create_extruded_geometry  # noqa: E402


# ── Lattice / geometry ────────────────────────────────────────────────────────
N_POINTS = _dav.n_points
MODE     = _dav.mode
RATIO    = _dav.ratio

# ── Physics ───────────────────────────────────────────────────────────────────
HUB_OMEGA      = -2.0       # rad/s; negative → CW viewed from +Z
TIMESTEP       = 1 / 240.0
SUBSTEPS       = 60         # sub-steps per simulation step
SOLVER_ITERS   = 300        # Gauss-Seidel iterations per sub-step
JOINT_FORCE    = 10_000     # N; ball-joint force cap (~172× needed headroom)

TET_MASS       = 0.02       # kg; tetrahedron mass
HUB_PRISM_MASS = 0.008      # kg; rotating prism mass
PRISM_INERTIA  = 3e-4       # kg·m²; uniform diagonal of prism inertia tensor

TET_LIN_DAMP   = 0.30       # tet linear damping
TET_ANG_DAMP   = 0.30       # tet angular damping
PRISM_LIN_DAMP = 0.95       # resist translational drift from the anchor
PRISM_ANG_DAMP = 0.05       # low — prisms should spin freely

# ── Visuals ───────────────────────────────────────────────────────────────────
HUB_SPOKE_LEN  = 0.08       # length of the TCO rotation-indicator line
HUB_R_FALLBACK = 0.008      # radius of stub sphere for corner hubs < 3 corners

C_HUB_CENTRAL  = [0.15, 0.45, 0.95, 1.00]   # blue   – TCO
C_HUB_ROTATING = [0.35, 0.60, 0.85, 0.90]   # sky    – rotating prisms
C_HUB_FIXED    = [0.55, 0.55, 0.60, 0.80]   # grey   – fixed corner prisms
C_TET          = [0.95, 0.55, 0.12, 0.93]   # orange – tetrahedra


class AuxeticSim:
    """Build and run the 3-D auxetic mechanism simulation."""

    def __init__(self):
        self._hub_angle      = 0.0
        self._hub_driver     = []   # [(body_id, centre_array), …]
        self._hub_center_pos = None
        self._spoke_id       = -1
        self._paused         = False

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self):
        """Connect PyBullet, build all bodies, anchors, and joints."""
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=TIMESTEP,
            numSubSteps=SUBSTEPS,
            numSolverIterations=SOLVER_ITERS,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        pts, tri = generate_points(N_POINTS, MODE)
        pts_3d   = np.asarray(pts, float)

        # ── Kirigami export (unchanged from before) ───────────────────────────
        _export_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'grid')
        tiles, tile_source = collect_kirigami_tiles(pts, tri, RATIO, MODE, 1)
        raw_constraints    = build_kirigami_constraints(tiles, tile_source)
        export_kirigami_vertices(
            os.path.join(_export_dir, 'vertices.txt'), tiles)
        export_kirigami_constraints(
            os.path.join(_export_dir, 'constraints.txt'), raw_constraints)

        # ── Shrunk tetrahedra ─────────────────────────────────────────────────
        tet_verts_world = []
        simplex_list    = []
        for simplex in tri.simplices:
            tet      = pts_3d[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = (1.0 - RATIO) * tet + RATIO * centroid
            tet_verts_world.append(shrunk)
            simplex_list.append(simplex)

        # ── Classify lattice hubs ─────────────────────────────────────────────
        grid_center = pts_3d.mean(axis=0)
        central_idx = int(np.argmin(np.linalg.norm(pts_3d - grid_center, axis=1)))

        mn, mx = pts_3d.min(axis=0), pts_3d.max(axis=0)
        tol = 1e-6

        def _is_corner(pos):
            """ALL three coords at a face of the bounding box."""
            return all(
                abs(pos[ax] - mn[ax]) < tol or abs(pos[ax] - mx[ax]) < tol
                for ax in range(3)
            )

        # ── Per-hub shrunk-corner groups ──────────────────────────────────────
        groups = _dav.build_3d_groups(pts_3d, tri, RATIO)
        ngon_t = _dav.ngon_thickness

        # ── Create hub bodies ─────────────────────────────────────────────────
        # hub_body_id[h_idx]  → PyBullet body ID
        # hub_body_ctr[h_idx] → body reference centre (numpy array)
        #                        used to compute local-frame pivot offsets
        hub_body_id  = {}
        hub_body_ctr = {}
        all_hub_ids  = []

        for h_idx, hub_pos in enumerate(pts_3d):
            pts_list = groups.get(tuple(hub_pos), [])

            if h_idx == central_idx:
                # ── Central TCO hub ───────────────────────────────────────────
                hid, ctr = self._make_tco_hub(hub_pos, pts_list)

            elif len(pts_list) >= 3:
                # ── Surrounding n-gon prism hub ───────────────────────────────
                try:
                    fixed = _is_corner(hub_pos)
                    hid, ctr = self._make_prism_hub(pts_list, ngon_t,
                                                    fixed=fixed)
                except Exception:
                    # Fallback: tiny static sphere
                    hid, ctr = self._make_fallback_sphere(hub_pos,
                                                          fixed=True)
            else:
                # ── Corner hub with too few shrunk corners ────────────────────
                hid, ctr = self._make_fallback_sphere(hub_pos, fixed=True)

            hub_body_id[h_idx]  = hid
            hub_body_ctr[h_idx] = np.asarray(ctr, float)
            all_hub_ids.append(hid)

        self._hub_driver     = [(hub_body_id[central_idx],
                                 pts_3d[central_idx].copy())]
        self._hub_center_pos = pts_3d[central_idx].copy()

        # ── Disable hub-hub collisions ────────────────────────────────────────
        for i in range(len(all_hub_ids)):
            for j in range(i + 1, len(all_hub_ids)):
                p.setCollisionFilterPair(all_hub_ids[i], all_hub_ids[j],
                                         -1, -1, enableCollision=0)

        # ── Create tetrahedra ─────────────────────────────────────────────────
        tet_ids     = []
        tet_centers = []
        for shrunk in tet_verts_world:
            center = shrunk.mean(axis=0)
            tid    = self._make_tet_body(shrunk, center)
            tet_ids.append(tid)
            tet_centers.append(center)
            p.changeDynamics(tid, -1,
                             linearDamping=TET_LIN_DAMP,
                             angularDamping=TET_ANG_DAMP)

        # ── Disable hub-tet and tet-tet collisions ────────────────────────────
        for t_id in tet_ids:
            for h_id in all_hub_ids:
                p.setCollisionFilterPair(h_id, t_id, -1, -1, enableCollision=0)
        for i in range(len(tet_ids)):
            for j in range(i + 1, len(tet_ids)):
                p.setCollisionFilterPair(tet_ids[i], tet_ids[j],
                                         -1, -1, enableCollision=0)

        # ── Ball joints: hub-prism body ↔ tet at each shrunk corner ───────────
        # The pivot is specified in each body's LOCAL frame.  Because the
        # prism bodies are anchored (only rotate), a force at an off-centre
        # pivot produces a pure torque that spins the prism.
        n_joints = 0
        for t_idx, (simplex, shrunk) in enumerate(
                zip(simplex_list, tet_verts_world)):
            t_center = tet_centers[t_idx]
            for v in range(4):
                h_idx = simplex[v]
                if h_idx not in hub_body_id:
                    continue
                joint_world = shrunk[v]
                # Offset from the hub body's reference centre → local pivot
                pivot_hub = (joint_world - hub_body_ctr[h_idx]).tolist()
                pivot_tet = (joint_world - t_center).tolist()

                c_id = p.createConstraint(
                    hub_body_id[h_idx], -1, tet_ids[t_idx], -1,
                    p.JOINT_POINT2POINT, [0, 0, 0],
                    pivot_hub, pivot_tet)
                p.changeConstraint(c_id, maxForce=JOINT_FORCE)
                n_joints += 1

        # ── Camera ────────────────────────────────────────────────────────────
        p.resetDebugVisualizerCamera(
            cameraDistance=2.2,
            cameraYaw=40,
            cameraPitch=-28,
            cameraTargetPosition=grid_center.tolist(),
        )

        n_corner = sum(1 for hp in pts_3d if _is_corner(hp))
        n_free   = max(0, len(pts_3d) - n_corner - 1)
        print(f"[AuxeticSim] {len(pts_3d)} hubs "
              f"({n_corner} fixed corners, {n_free} rotating, 1 TCO), "
              f"{len(tet_ids)} tets, {n_joints} ball joints")
        print(f"[AuxeticSim] Central TCO idx={central_idx} "
              f"at {pts_3d[central_idx].round(3)}, ω={HUB_OMEGA:.2f} rad/s")
        print("[AuxeticSim] Space = pause/resume   Q = quit")

    def run(self):
        """Main simulation loop."""
        while p.isConnected():
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("[AuxeticSim] Quitting.")
                break
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                self._paused = not self._paused
                print("[AuxeticSim]",
                      "Paused." if self._paused else "Resumed.")

            if not self._paused:
                self._drive_hub()
                p.stepSimulation()

        if p.isConnected():
            p.disconnect()

    # ── Private body factories ────────────────────────────────────────────────

    @staticmethod
    def _make_tco_hub(hub_pos, pts_list):
        """Truncated-cuboctahedron body for the central hub.
        Returns (body_id, centre_array)."""
        scale = (_dav._hub_scale_for_tcoh(hub_pos, pts_list)
                 if pts_list else 0.05)
        tco_v, _, _, _ = _dav.make_truncated_cuboctahedron(hub_pos, scale)
        col_verts, vis_idx, _, vis_normals, vis_verts = \
            create_solid_geometry(tco_v)
        col = p.createCollisionShape(p.GEOM_MESH, vertices=col_verts)
        vis = p.createVisualShape(
            p.GEOM_MESH, vertices=vis_verts, indices=vis_idx,
            normals=vis_normals, rgbaColor=C_HUB_CENTRAL,
            specularColor=[0.1, 0.1, 0.1])
        hid = p.createMultiBody(
            baseMass=1.0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=hub_pos.tolist())
        return hid, hub_pos.copy()

    @staticmethod
    def _make_prism_hub(pts_list, ngon_t, fixed=False):
        """N-gon prism body for a surrounding hub.

        fixed=True  → mass=0 static anchor (corner hubs).
        fixed=False → small-mass body whose centre is pinned to the lattice
                      point by a POINT2POINT-to-world constraint; the body
                      is free to rotate about that pinned centre.

        Returns (body_id, prism_centre_array).
        """
        ordered = _dav.convex_order_3d(np.array(pts_list))
        if ordered is None:
            raise ValueError("convex_order_3d returned None")

        _, vis_idx, prism_ctr_list, vis_normals, vis_verts = \
            create_extruded_geometry(ordered.tolist(), ngon_t)
        prism_ctr = np.array(prism_ctr_list)

        color = C_HUB_FIXED if fixed else C_HUB_ROTATING
        vis = p.createVisualShape(
            p.GEOM_MESH, vertices=vis_verts, indices=vis_idx,
            normals=vis_normals, rgbaColor=color,
            specularColor=[0.05, 0.05, 0.05])

        if fixed:
            hid = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=prism_ctr.tolist())
        else:
            # Tiny sphere collision shape gives PyBullet a non-degenerate
            # inertia tensor which we then override.
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)
            hid = p.createMultiBody(
                baseMass=HUB_PRISM_MASS, baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=prism_ctr.tolist())
            p.changeDynamics(hid, -1,
                             localInertiaDiagonal=[PRISM_INERTIA] * 3,
                             linearDamping=PRISM_LIN_DAMP,
                             angularDamping=PRISM_ANG_DAMP)
            # Pin the prism's centre to its world lattice position.
            # Only rotational DOF remains free.
            p.createConstraint(
                hid, -1, -1, -1,
                p.JOINT_POINT2POINT, [0, 0, 0],
                [0, 0, 0],           # at the prism body's own centre
                prism_ctr.tolist())  # world anchor point

        return hid, prism_ctr

    @staticmethod
    def _make_fallback_sphere(hub_pos, fixed=True):
        """Tiny sphere for corner hubs that have too few shrunk corners."""
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=HUB_R_FALLBACK, rgbaColor=C_HUB_FIXED)
        hid = p.createMultiBody(
            baseMass=0 if fixed else 0.001,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=hub_pos.tolist())
        return hid, hub_pos.copy()

    # ── Drive loop ────────────────────────────────────────────────────────────

    def _drive_hub(self):
        """Advance the TCO angle and teleport it to the new orientation."""
        self._hub_angle += HUB_OMEGA * TIMESTEP
        orn = p.getQuaternionFromEuler([0.0, 0.0, self._hub_angle])
        for body_id, centre in self._hub_driver:
            p.resetBasePositionAndOrientation(body_id, centre.tolist(), orn)
            p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0],
                                [0.0, 0.0, HUB_OMEGA])

        # Draw a spoke to visualise the current rotation angle
        c   = self._hub_center_pos
        ca  = self._hub_angle
        tip = c + np.array([HUB_SPOKE_LEN * np.cos(ca),
                             HUB_SPOKE_LEN * np.sin(ca),
                             0.02])
        if self._spoke_id < 0:
            self._spoke_id = p.addUserDebugLine(
                c.tolist(), tip.tolist(), [1.0, 0.2, 0.0],
                lineWidth=3, lifeTime=0)
        else:
            self._spoke_id = p.addUserDebugLine(
                c.tolist(), tip.tolist(), [1.0, 0.2, 0.0],
                lineWidth=3, lifeTime=0,
                replaceItemUniqueId=self._spoke_id)

    # ── Tet factory ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_tet_body(shrunk_verts, center):
        """Convex rigid body for one shrunk tetrahedron."""
        try:
            col_verts, vis_idx, _, vis_normals, vis_verts = \
                create_solid_geometry(shrunk_verts)
        except Exception:
            half = np.ptp(shrunk_verts, axis=0) / 2 + 1e-3
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half.tolist())
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half.tolist(),
                                      rgbaColor=C_TET)
            return p.createMultiBody(
                baseMass=TET_MASS, baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis, basePosition=center.tolist())

        vis_shape = p.createVisualShape(
            p.GEOM_MESH, vertices=vis_verts, indices=vis_idx,
            normals=vis_normals, rgbaColor=C_TET,
            specularColor=[0.08, 0.08, 0.08])
        col_shape = p.createCollisionShape(p.GEOM_MESH, vertices=col_verts)
        return p.createMultiBody(
            baseMass=TET_MASS,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=center.tolist())
