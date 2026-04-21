"""
Auxetic simulation: central hub rotates CW in XY plane; tetrahedra rotate
~90° relative to the hub as the off-centre ball joints pull them around.

Geometry comes from displayAuxeticV20.generate_points() (mode-6 symmetric 3-D
grid).  All four corners of every tetrahedron are shrunk toward its own
centroid (ratio controls how far), so each corner sits at a point that is
slightly off-centre from the nearest lattice hub.  When the central hub rotates,
those off-centre pivots sweep arcs that force the tetrahedra to rotate.
Boundary lattice-point hubs are fixed; interior non-central hubs are light
free-floating bodies so the mechanism can propagate outward.
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
N_POINTS  = _dav.n_points
MODE      = _dav.mode
RATIO     = _dav.ratio

# ── Physics ───────────────────────────────────────────────────────────────────
HUB_OMEGA        = -0.5      # rad/s; negative → CW when viewed from +Z (above)
TIMESTEP         = 1 / 240.0
SUBSTEPS         = 30        # more substeps → more stable constraints
SOLVER_ITERS     = 100       # constraint solver iterations per substep
JOINT_FORCE      = 500       # realistic for TET_MASS=0.02 at this scale
TET_MASS         = 0.02
LIN_DAMP         = 0.60
ANG_DAMP         = 0.60

# ── Visuals ───────────────────────────────────────────────────────────────────
HUB_DISK_RADIUS = 0.06
HUB_DISK_HEIGHT = 0.012
HUB_SPOKE_LEN   = 0.055

HUB_R_INTERIOR = 0.018
HUB_R_BOUNDARY = 0.012

C_HUB_CENTRAL  = [0.15, 0.45, 0.95, 1.00]
C_HUB_INTERIOR = [0.50, 0.75, 0.95, 0.75]
C_HUB_BOUNDARY = [0.35, 0.60, 0.85, 0.90]
C_TET          = [0.95, 0.55, 0.12, 0.93]
C_HUB_POLYGON  = [0.35, 0.60, 0.85, 0.90]   # extruded hub face colour


class AuxeticSim:
    """Build and run the auxetic mechanism simulation."""

    def __init__(self):
        self._hub_angle  = 0.0
        self._hub_driver = []
        self._paused     = False

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self):
        """Connect PyBullet, generate geometry, create bodies and joints."""
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

        # ── Export vertices / constraints ─────────────────────────────────────
        _export_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'grid')
        tiles, tile_source = collect_kirigami_tiles(pts, tri, RATIO, MODE, 1)
        raw_constraints    = build_kirigami_constraints(tiles, tile_source)
        export_kirigami_vertices(
            os.path.join(_export_dir, 'vertices.txt'), tiles)
        export_kirigami_constraints(
            os.path.join(_export_dir, 'constraints.txt'), raw_constraints)

        # ── Build shrunk tetrahedra ───────────────────────────────────────────
        tet_verts_world = []
        simplex_list    = []
        for simplex in tri.simplices:
            tet      = pts_3d[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = (1.0 - RATIO) * tet + RATIO * centroid
            tet_verts_world.append(shrunk)
            simplex_list.append(simplex)

        # ── Classify lattice hubs ─────────────────────────────────────────────
        grid_center     = pts_3d.mean(axis=0)
        dists_to_center = np.linalg.norm(pts_3d - grid_center, axis=1)
        central_idx     = int(np.argmin(dists_to_center))

        mn, mx = pts_3d.min(axis=0), pts_3d.max(axis=0)
        tol = 1e-6

        def _is_corner(pos):
            """True only if ALL three coordinates are at the grid boundary (a corner vertex)."""
            return all(
                abs(pos[ax] - mn[ax]) < tol or abs(pos[ax] - mx[ax]) < tol
                for ax in range(3)
            )

        def _is_boundary(pos):
            """True if ANY coordinate is at the grid boundary (face/edge/corner)."""
            return any(
                abs(pos[ax] - mn[ax]) < tol or abs(pos[ax] - mx[ax]) < tol
                for ax in range(3)
            )

        # ── Create hub sphere bodies ──────────────────────────────────────────
        # Only corner hubs (all 3 coords at boundary) are fixed.
        # Edge and face hubs are free-floating so the mechanism propagates.
        hub_ids = []
        for h_idx, hub_pos in enumerate(pts_3d):
            if h_idx == central_idx:
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=HUB_DISK_RADIUS, height=HUB_DISK_HEIGHT)
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=HUB_DISK_RADIUS, length=HUB_DISK_HEIGHT,
                    rgbaColor=C_HUB_CENTRAL)
                mass = 1.0
            elif _is_corner(hub_pos):
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=HUB_R_BOUNDARY)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=HUB_R_BOUNDARY,
                                          rgbaColor=C_HUB_BOUNDARY)
                mass = 0.0          # corner hubs: fully fixed anchors
            else:
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=HUB_R_INTERIOR)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=HUB_R_INTERIOR,
                                          rgbaColor=C_HUB_INTERIOR)
                mass = 0.005        # edge/face/interior hubs: free to move

            hid = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=hub_pos.tolist(),
            )
            hub_ids.append(hid)

        self._hub_driver     = [(hub_ids[central_idx], pts_3d[central_idx].copy())]
        self._hub_center_pos = pts_3d[central_idx].copy()
        self._spoke_id       = -1

        # Disable hub-hub collisions
        for i in range(len(hub_ids)):
            for j in range(i + 1, len(hub_ids)):
                p.setCollisionFilterPair(hub_ids[i], hub_ids[j], -1, -1,
                                         enableCollision=0)

        # ── Create tetrahedron bodies ─────────────────────────────────────────
        tet_ids     = []
        tet_centers = []
        for shrunk in tet_verts_world:
            center = shrunk.mean(axis=0)
            tet_id = self._make_tet_body(shrunk, center)
            tet_ids.append(tet_id)
            tet_centers.append(center)
            p.changeDynamics(tet_id, -1,
                             linearDamping=LIN_DAMP, angularDamping=ANG_DAMP)

        # Disable tet-tet and hub-tet collisions
        for t_id in tet_ids:
            for h_id in hub_ids:
                p.setCollisionFilterPair(h_id, t_id, -1, -1, enableCollision=0)
        for i in range(len(tet_ids)):
            for j in range(i + 1, len(tet_ids)):
                p.setCollisionFilterPair(tet_ids[i], tet_ids[j], -1, -1,
                                         enableCollision=0)

        # ── Create ball joints: hub <-> tet at each shrunk corner ─────────────
        n_joints = 0
        for t_idx, (simplex, shrunk) in enumerate(zip(simplex_list, tet_verts_world)):
            t_center = tet_centers[t_idx]
            for v in range(4):
                h_idx       = simplex[v]
                hub_id      = hub_ids[h_idx]
                hub_pos     = pts_3d[h_idx]
                joint_world = shrunk[v]

                pivot_hub = (joint_world - hub_pos).tolist()
                pivot_tet = (joint_world - t_center).tolist()

                c_id = p.createConstraint(
                    hub_id, -1, tet_ids[t_idx], -1,
                    p.JOINT_POINT2POINT, [0, 0, 0],
                    pivot_hub, pivot_tet,
                )
                p.changeConstraint(c_id, maxForce=JOINT_FORCE)
                n_joints += 1

        # ── Add extruded hub polygon visuals ──────────────────────────────────
        # For each hub lattice point, gather the shrunk tet corners that meet
        # there and extrude them into a polygon prism (visual only, mass=0).
        groups = _dav.build_3d_groups(pts_3d, tri, RATIO)
        ngon_t = _dav.ngon_thickness

        for h_idx, hub_pos in enumerate(pts_3d):
            if h_idx == central_idx:
                continue  # central hub already rendered as rotating disk
            if not _is_corner(hub_pos):
                continue  # skip moving hubs — their polygon would stay static
            key      = tuple(hub_pos)
            pts_list = groups.get(key, [])
            if len(pts_list) < 3:
                continue
            try:
                ordered = _dav.convex_order_3d(np.array(pts_list))
                if ordered is None:
                    continue
                _, vis_idx, center, vis_normals, vis_verts = \
                    create_extruded_geometry(ordered.tolist(), ngon_t)
                vis = p.createVisualShape(
                    p.GEOM_MESH,
                    vertices=vis_verts,
                    indices=vis_idx,
                    normals=vis_normals,
                    rgbaColor=C_HUB_POLYGON,
                    specularColor=[0.05, 0.05, 0.05],
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=vis,
                    basePosition=center,
                )
            except Exception:
                pass

        # ── Camera ────────────────────────────────────────────────────────────
        p.resetDebugVisualizerCamera(
            cameraDistance=2.2,
            cameraYaw=40,
            cameraPitch=-28,
            cameraTargetPosition=grid_center.tolist(),
        )

        n_corner   = sum(1 for hp in pts_3d if _is_corner(hp))
        n_free     = max(0, len(pts_3d) - n_corner - 1)
        print(f"[AuxeticSim] Lattice: {len(pts_3d)} hubs "
              f"({n_corner} fixed corners, {n_free} free edge/face/interior), "
              f"{len(tet_ids)} tets, {n_joints} ball joints")
        print(f"[AuxeticSim] Central hub idx={central_idx} "
              f"at {pts_3d[central_idx].round(3)}, omega={HUB_OMEGA:.2f} rad/s CW")
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _drive_hub(self):
        """Advance the central hub angle and force its transform each step."""
        self._hub_angle += HUB_OMEGA * TIMESTEP
        orn = p.getQuaternionFromEuler([0.0, 0.0, self._hub_angle])
        for body_id, center in self._hub_driver:
            p.resetBasePositionAndOrientation(body_id, center.tolist(), orn)
            p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0],
                                [0.0, 0.0, HUB_OMEGA])

        c  = self._hub_center_pos
        ca = self._hub_angle
        tip = c + np.array([HUB_SPOKE_LEN * np.cos(ca),
                             HUB_SPOKE_LEN * np.sin(ca),
                             HUB_DISK_HEIGHT])
        if self._spoke_id < 0:
            self._spoke_id = p.addUserDebugLine(
                c.tolist(), tip.tolist(), [1.0, 0.2, 0.0],
                lineWidth=3, lifeTime=0)
        else:
            self._spoke_id = p.addUserDebugLine(
                c.tolist(), tip.tolist(), [1.0, 0.2, 0.0],
                lineWidth=3, lifeTime=0,
                replaceItemUniqueId=self._spoke_id)

    @staticmethod
    def _make_tet_body(shrunk_verts, center):
        """Create a convex-mesh rigid body for one shrunk tetrahedron."""
        try:
            col_verts, vis_idx, _, vis_normals, vis_verts = \
                create_solid_geometry(shrunk_verts)
        except Exception:
            half = np.ptp(shrunk_verts, axis=0) / 2 + 1e-3
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half.tolist())
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half.tolist(),
                                      rgbaColor=C_TET)
            return p.createMultiBody(
                baseMass=TET_MASS,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=center.tolist(),
            )

        vis_shape = p.createVisualShape(
            p.GEOM_MESH,
            vertices=vis_verts,
            indices=vis_idx,
            normals=vis_normals,
            rgbaColor=C_TET,
            specularColor=[0.08, 0.08, 0.08],
        )
        col_shape = p.createCollisionShape(p.GEOM_MESH, vertices=col_verts)
        return p.createMultiBody(
            baseMass=TET_MASS,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=center.tolist(),
        )
