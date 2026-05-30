"""
Geometry utilities for the kirigami simulation project.
"""
import numpy as np
import pybullet as p
from scipy.spatial import ConvexHull

def create_ground_plane(z, thickness=1, friction=0.5,color=(0.35, 0.35, 0.35, 1.0)):
    """Create a large axis-aligned ground box.

    Args:
        z (float): World Z coordinate of the box center (NOT the top surface).
        thickness (float): Half-thickness of the box (full thickness = 2 * thickness).
        color (tuple[float,float,float,float]): RGBA color.

    Returns:
        int: PyBullet body ID of the created static ground.
    """
    
    ground_collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[50000, 50000, thickness]
    )
    ground_visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[50000, 50000, thickness],
        rgbaColor=list(color),
        visualFrameOrientation=[0, 0, 0, 1], # one can create a tilted ground for better view in some cases
        specularColor=[0, 0, 0] # the ground would not reflect light
    )

    ground_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=ground_collision_shape,
        baseVisualShapeIndex=ground_visual_shape,
        basePosition=[0, 0, z-thickness]
    )
    p.changeDynamics(ground_id, -1, lateralFriction=friction)
    return ground_id

    

def build_extruded_visual_mesh(planar_vertices, normal, brick_thickness, center):
    n = len(planar_vertices)
    top = planar_vertices + brick_thickness * normal
    bottom = planar_vertices

    vis_vertices = []
    vis_normals = []
    vis_indices = []

    def add(v, nrm):
        vis_vertices.append((v - center).tolist())
        vis_normals.append(nrm.tolist())
        return len(vis_vertices) - 1

    # Bottom cap: CCW when viewed from outside (outside is -normal)
    bot_idx = [add(v, -normal) for v in bottom]
    for i in range(1, n - 1):
        vis_indices += [bot_idx[0], bot_idx[i + 1], bot_idx[i]]  # fan, reversed order

    # Top cap: CCW when viewed from outside (outside is +normal)
    top_idx = [add(v, normal) for v in top]
    for i in range(1, n - 1):
        vis_indices += [top_idx[0], top_idx[i], top_idx[i + 1]]  # fan

    # Sides: duplicate vertices per side, use outward normal per side
    poly_center = np.mean(planar_vertices, axis=0)
    for i in range(n):
        j = (i + 1) % n
        edge = planar_vertices[j] - planar_vertices[i]
        side_n = np.cross(normal, edge)
        side_n_norm = np.linalg.norm(side_n)
        if side_n_norm < 1e-12:
            continue
        side_n = side_n / side_n_norm

        # Ensure outward (point away from polygon center)
        mid = 0.5 * (planar_vertices[i] + planar_vertices[j])
        if np.dot(side_n, mid - poly_center) < 0:
            side_n = -side_n

        v0 = add(bottom[i], side_n)  # bottom i
        v1 = add(bottom[j], side_n)  # bottom j
        v2 = add(top[j],    side_n)  # top j
        v3 = add(top[i],    side_n)  # top i

        # CCW winding when viewed from outside
        vis_indices += [v0, v1, v2, v0, v2, v3]

    return vis_vertices, vis_indices, vis_normals


def create_extruded_geometry(vertices_per_tile, brick_thickness):
    
    planar_vertices = np.array(vertices_per_tile)
    
    center = np.mean(planar_vertices, axis=0)

    vec1 = planar_vertices[1] - planar_vertices[0]
    vec2 = planar_vertices[2] - planar_vertices[0]
    normal = np.cross(vec1, vec2)
    norm_val = np.linalg.norm(normal)
    if norm_val == 0:
        print("Warning: Degenerate polygon. Using default normal [0,0,1].")
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / norm_val

    top_vertices = planar_vertices + brick_thickness * normal
    bottom_vertices = planar_vertices.copy()

    center = center + 0.5 * brick_thickness * normal  # center of the solid

    # Physics/constraints vertices (keep your original structure: bottom then top)
    local_verts = [(v - center).tolist() for v in bottom_vertices] + \
                  [(v - center).tolist() for v in top_vertices]

    # Visual mesh with proper normals
    vis_vertices, visual_indices, vis_normals = build_extruded_visual_mesh(
        planar_vertices, normal, brick_thickness, center
    )

    return (local_verts, visual_indices, center.tolist(), vis_normals, vis_vertices)

def is_planar(vertices, tol=1e-6):
    """True if all input 3D points lie on a common plane (or are colinear)."""
    verts = np.asarray(vertices, dtype=float)
    if len(verts) < 4:
        return True
    v0 = verts[0]
    d1 = verts[1] - v0
    normal = None
    for k in range(2, len(verts)):
        n = np.cross(d1, verts[k] - v0)
        if np.linalg.norm(n) > tol:
            normal = n / np.linalg.norm(n)
            break
    if normal is None:
        return True
    return np.all(np.abs((verts - v0) @ normal) <= tol)


def create_solid_geometry(vertices_3d):
    """Build a solid convex rigid body from an arbitrary 3D point set.

    Returns the same tuple shape as create_extruded_geometry so the caller
    can hand the result straight to create_brick_body.
    """
    pts = np.asarray(vertices_3d, dtype=float)
    if len(pts) < 4:
        raise ValueError("create_solid_geometry requires at least 4 non-coplanar points")

    hull = ConvexHull(pts)
    center = pts.mean(axis=0)

    local_verts = [(pt - center).tolist() for pt in pts]

    vis_vertices = []
    vis_normals = []
    vis_indices = []
    for simplex, eq in zip(hull.simplices, hull.equations):
        n = np.array(eq[:3], dtype=float)
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Align triangle winding with outward normal (CCW seen from outside).
        if np.dot(np.cross(b - a, c - a), n) < 0:
            b, c = c, b
        base = len(vis_vertices)
        for v in (a, b, c):
            vis_vertices.append((v - center).tolist())
            vis_normals.append(n.tolist())
        vis_indices += [base, base + 1, base + 2]

    return (local_verts, vis_indices, center.tolist(), vis_normals, vis_vertices)


def create_brick_body(col_verts, visual_indices, center, vis_normals, vis_vertices, mass=1.0):
    r = 0.8 + np.random.uniform(-0.1, 0.1)
    g = 0.4 + np.random.uniform(-0.1, 0.1)
    b = 0.2 + np.random.uniform(-0.1, 0.1)
    brick_color = [min(1, r), min(1, g), min(1, b), 1]

    vis_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        vertices=vis_vertices,       # use duplicated vertices
        indices=visual_indices,
        normals=vis_normals,         # correct per-face normals
        rgbaColor=brick_color,
        specularColor=[0.1, 0.1, 0.1]  # reduce specular if you want less glare
    )

    col_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        vertices=col_verts  # convex hull from physics verts (bottom+top)
    )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=center
    )
    return body_id


def create_constraints_between_bricks(bricks, constraints_with_types, local_verts_list, is_solid_list=None):
    """Create constraints between bricks based on vertex connections and type.

    Args:
        bricks: List of brick body IDs
        constraints_with_types: List of constraints (f_i, v_j, f_p, v_q, type)
                                For extruded tiles: type=1 bottom vertex, type=2 top vertex.
                                For solid tiles: v_j indexes directly into the local vertex
                                list (the type flag is ignored).
        local_verts_list: List of local vertex coordinates for each brick
        is_solid_list: Optional list of bools. Entry i=True means tile i is a solid body
                       with a single flat vertex list (no bottom/top halves).

    Returns:
       list: created_constraint_ids: List of created constraint IDs
    """
    created_constraints = []

    def resolve_pivot(face, v_idx, ctype):
        verts = local_verts_list[face]
        is_solid = bool(is_solid_list[face]) if is_solid_list is not None else False
        if is_solid:
            return verts[v_idx]
        num = len(verts) // 2
        if ctype == 1:
            return verts[v_idx]
        if ctype == 2:
            return verts[num + v_idx]
        return None

    for f_i, v_j, f_p, v_q, constraint_type in constraints_with_types:
        pivot_in_1 = resolve_pivot(f_i, v_j, constraint_type)
        pivot_in_2 = resolve_pivot(f_p, v_q, constraint_type)
        if pivot_in_1 is None or pivot_in_2 is None:
            print(f"Warning: Unknown constraint type {constraint_type} for constraint {(f_i, v_j, f_p, v_q)}. Skipping.")
            continue

        c_id = p.createConstraint(
            bricks[f_i], -1, bricks[f_p], -1,
            p.JOINT_POINT2POINT, [0, 0, 0],
            pivot_in_1, pivot_in_2
        )
        created_constraints.append(c_id)

    return created_constraints


def transform_local_to_world_coordinates(body_ids, local_vertices):
    """
    Transform local vertex coordinates to world coordinates based on current body poses.
    
    This function takes local vertex coordinates (relative to body centers) and transforms
    them to world coordinates using the current position and orientation of each body
    from the physics simulation.
    
    Args:
        body_ids: List of PyBullet body IDs
        local_vertices: List of local vertex coordinate arrays for each body
                       [[[x1,y1,z1], [x2,y2,z2], ...], ...]

    Returns:
        list: List of transformed vertices in world coordinates for each body
              Each element contains the world coordinates of all vertices for that body
    """
    transformed_vertices = []
    
    for i, body_id in enumerate(body_ids):
        if i >= len(local_vertices):
            continue
            
        # Get current pose of this brick
        current_pos, current_orn = p.getBasePositionAndOrientation(body_id)
        
        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(current_orn)).reshape(3, 3)
        
        # Vectorized transformation of all vertices for this brick
        local_verts = np.array(local_vertices[i])  # Shape: (n_vertices, 3)
        world_verts = (rotation_matrix @ local_verts.T).T + np.array(current_pos)  # Shape: (n_vertices, 3)
        
        # Add this brick's transformed vertices to the result
        transformed_vertices.append(world_verts.tolist())
    
    return transformed_vertices

def compute_min_z(vertices):
        """
        Return the minimal z from a List[List[List[float]]]
        """
        if not vertices:
            return None
        z = []
        for tile in vertices:
            for v in tile:
                z.append(v[2])
        return min(z)