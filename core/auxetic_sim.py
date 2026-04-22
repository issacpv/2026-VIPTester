"""
Auxetic simulation — 3-D hub-and-tet mechanism.

Architecture
────────────
Central hub   → Truncated-cuboctahedron (TCO) rigid body, force-driven to
                spin in place each step.

Surrounding   → Each non-corner lattice point owns one n-gon PRISM body
hub prisms      whose centre is pinned to the lattice position by a
                POINT2POINT-to-world constraint. The pin prevents drift so
                the prism can only spin about its own centre.

Corner prisms → Same prism geometry but DYNAMIC (no world pin): they are
                free to translate and rotate, driven entirely by the ball
                joints connecting them to their six neighbouring tetrahedra.

Tetrahedra    → Convex rigid bodies with POINT2POINT ball joints connecting
                each of their four corners to whichever hub-prism (or TCO)
                body owns that lattice vertex. The joint pivot is defined
                in each hub body's LOCAL frame at the shrunk-corner offset.

Mechanism
─────────
1. TCO spins  →  its local-frame pivot vectors sweep arcs in world space
2.            →  ball joints drag the tet corners along those arcs
3.            →  each tet rotates
4.            →  tet corners attached to surrounding prisms exert off-centre
                 forces on those prism bodies
5.            →  pinned prisms spin; corner prisms translate + spin
6.            →  the next ring of tets is dragged  →  wave propagates out
"""

import sys
import os
import time
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


# ── Lattice / geometry — mirror displayAuxeticV20 settings exactly ────────────
N_POINTS = _dav.n_points
MODE     = _dav.mode
RATIO    = _dav.ratio

# ── Physics ───────────────────────────────────────────────────────────────────
HUB_OMEGA      = -0.5       # rad/s; negative → CW viewed from +Z
TIMESTEP       = 1 / 240.0
SUBSTEPS       = 60
SOLVER_ITERS   = 300
JOINT_FORCE    = 10_000

TET_MASS       = 0.02
HUB_PRISM_MASS = 0.008
PRISM_INERTIA  = 3e-4

TET_LIN_DAMP   = 0.30
TET_ANG_DAMP   = 0.30
PRISM_LIN_DAMP = 0.95   # high — resists drift for world-pinned prisms
PRISM_ANG_DAMP = 0.05   # low  — prisms should spin freely
CORNER_LIN_DAMP = 0.30  # moderate — corners are free to move
CORNER_ANG_DAMP = 0.30

# ── Visuals ───────────────────────────────────────────────────────────────────
HUB_SPOKE_LEN  = 0.08
HUB_R_FALLBACK = 0.008

C_HUB_CENTRAL  = [0.15, 0.45, 0.95, 1.00]   # blue   – TCO hub
C_HUB_ROTATING = [0.35, 0.60, 0.85, 0.90]   # sky    – world-pinned prisms
C_HUB_CORNER   = [0.35, 0.60, 0.85, 0.90]   # same   – corner prisms (dynamic)
C_HUB_FALLBACK = [0.55, 0.55, 0.60, 0.80]   # grey   – fallback spheres
C_TET          = [0.95, 0.55, 0.12, 0.93]   # orange – tetrahedra


class AuxeticSim:
    """Build and run the 3-D auxetic mechanism simulation."""

    def __init__(self):
        self._hub_angle      = 0.0
        self._hub_driver     = []
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

        # ── Kirigami export ───────────────────────────────────────────────────
        _export_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'grid')
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
            """True only if ALL three coordinates are at the grid boundary."""
            return all(
                abs(pos[ax] - mn[ax]) < tol or abs(pos[ax] - mx[ax]) < tol
                for ax in range(3)
            )

        groups = _dav.build_3d_groups(pts_3d, tri, RATIO)
        ngon_t = _dav.ngon_thickness

        # ── Create hub bodies ─────────────────────────────────────────────────
        hub_body_id  = {}   # h_idx → PyBullet body ID
        hub_body_ctr = {}   # h_idx → body reference centre (numpy array)
        all_hub_ids  = []

        for h_idx, hub_pos in enumerate(pts_3d):
            pts_list = groups.get(tuple(hub_pos), [])

            if h_idx == central_idx:
                hid, ctr = self._make_tco_hub(hub_pos, pts_list)

            elif len(pts_list) >= 3:
                try:
                    is_corner = _is_corner(hub_pos)
                    hid, ctr  = self._make_prism_hub(
                        pts_list, ngon_t, corner=is_corner)
                except Exception:
                    hid, ctr = self._make_fallback_sphere(hub_pos)
            else:
                hid, ctr = self._make_fallback_sphere(hub_pos)

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

        # ── Ball joints: hub body ↔ tet at each shrunk corner ─────────────────
        n_joints = 0
        for t_idx, (simplex, shrunk) in enumerate(
                zip(simplex_list, tet_verts_world)):
            t_center = tet_centers[t_idx]
            for v in range(4):
                h_idx = simplex[v]
                if h_idx not in hub_body_id:
                    continue
                joint_world = shrunk[v]
                pivot_hub   = (joint_world - hub_body_ctr[h_idx]).tolist()
                pivot_tet   = (joint_world - t_center).tolist()

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
              f"({n_corner} dynamic corner prisms, {n_free} pinned prisms, 1 TCO), "
              f"{len(tet_ids)} tets, {n_joints} ball joints")
        print(f"[AuxeticSim] Central TCO idx={central_idx} "
              f"at {pts_3d[central_idx].round(3)}, ω={HUB_OMEGA:.2f} rad/s")
        print("[AuxeticSim] Controls: Space = pause/resume   Q = quit")

    def run(self):
        """Main simulation loop."""
        while p.isConnected():
            keys = p.getKeyboardEvents()

            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("[AuxeticSim] Quitting.")
                break
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                self._paused = not self._paused
                print("[AuxeticSim]", "Paused." if self._paused else "Resumed.")

            if not self._paused:
                self._drive_hub()
                p.stepSimulation()

            time.sleep(TIMESTEP)

        if p.isConnected():
            p.disconnect()

    # ── Private body factories ────────────────────────────────────────────────

    @staticmethod
    def _make_tco_hub(hub_pos, pts_list):
        """Truncated-cuboctahedron body for the central hub.
        Returns (body_id, centre_array).
        """
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
    def _make_prism_hub(pts_list, ngon_t, corner=False):
        """N-gon prism body for a surrounding hub.

        corner=False → world-pinned at prism centre via POINT2POINT-to-world
                       so the body can only spin, not drift.
        corner=True  → free dynamic body (translates + rotates), pulled
                       entirely by the ball joints to neighbouring tetrahedra.

        Returns (body_id, prism_centre_array).
        """
        ordered = _dav.convex_order_3d(np.array(pts_list))
        if ordered is None:
            raise ValueError("convex_order_3d returned None")

        _, vis_idx, prism_ctr_list, vis_normals, vis_verts = \
            create_extruded_geometry(ordered.tolist(), ngon_t)
        prism_ctr = np.array(prism_ctr_list)

        color = C_HUB_CORNER if corner else C_HUB_ROTATING
        vis = p.createVisualShape(
            p.GEOM_MESH, vertices=vis_verts, indices=vis_idx,
            normals=vis_normals, rgbaColor=color,
            specularColor=[0.05, 0.05, 0.05])

        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)
        hid = p.createMultiBody(
            baseMass=HUB_PRISM_MASS, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=prism_ctr.tolist())
        p.changeDynamics(hid, -1,
                         localInertiaDiagonal=[PRISM_INERTIA] * 3,
                         linearDamping=CORNER_LIN_DAMP if corner else PRISM_LIN_DAMP,
                         angularDamping=CORNER_ANG_DAMP if corner else PRISM_ANG_DAMP)

        if not corner:
            # Pin the prism's centre to its world lattice position.
            p.createConstraint(
                hid, -1, -1, -1,
                p.JOINT_POINT2POINT, [0, 0, 0],
                [0, 0, 0],
                prism_ctr.tolist())

        return hid, prism_ctr

    @staticmethod
    def _make_fallback_sphere(hub_pos):
        """Tiny static sphere for hubs that have too few shrunk corners."""
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=HUB_R_FALLBACK, rgbaColor=C_HUB_FALLBACK)
        hid = p.createMultiBody(
            baseMass=0,
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
