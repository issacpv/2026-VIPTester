"""Kirigami tile collection and constraint building.

Each kirigami "tile" is a polygonal/polyhedral patch with N >= 3 vertices.
Constraints link coincident vertices on different tiles so that PyKirigami
treats them as a shared joint.
"""

import numpy as np
from scipy.spatial import ConvexHull

from .geometry import (
    build_3d_groups,
    is_central_hub,
    hub_scale_for_tcoh,
    make_truncated_cuboctahedron,
    convex_order_3d,
    order_hub_ring_xy,
)


def collect_kirigami_tiles(points_nd, tri, ratio, mode, nz_layers):
    """Polygonal tile faces for kirigami export.

    For 3D modes (3, 6) each shrunken tetrahedron is one tile (4 vertices)
    so PyKirigami treats it as a single rigid polyhedron rather than 4
    independent triangular faces.
    """
    tiles       = []
    tile_source = []

    def add_tile(verts_3d, source_meta):
        verts_3d = np.asarray(verts_3d, float)
        if len(verts_3d) >= 3:
            tiles.append(verts_3d)
            tile_source.append(source_meta)

    if mode in [3, 6, 9]:
        pts_norm = points_nd

        groups       = build_3d_groups(pts_norm, tri, ratio)
        TOL_KEY      = 9
        central_keys = set()
        for key, pts_list in groups.items():
            if is_central_hub(pts_list):
                central_keys.add(tuple(np.round(np.asarray(key, float), TOL_KEY)))

        def _key(pt):
            return tuple(np.round(np.asarray(pt, float), TOL_KEY))

        for s_idx, simplex in enumerate(tri.simplices):
            tet      = pts_norm[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = np.empty_like(tet)
            for i in range(4):
                if _key(tet[i]) in central_keys:
                    shrunk[i] = tet[i]
                else:
                    shrunk[i] = (1 - ratio) * tet[i] + ratio * centroid

            add_tile(shrunk, {
                'type':         'tetrahedron',
                'simplex_idx':  s_idx,
                'vertex_keys':  [
                    (float(pts_norm[simplex[i], 0]),
                     float(pts_norm[simplex[i], 1]),
                     float(pts_norm[simplex[i], 2]))
                    for i in range(4)
                ],
            })

        for key, pts_list in groups.items():
            hub_center = np.array(key, float)
            if len(pts_list) < 3:
                continue
            pts_arr = np.array(pts_list)

            if is_central_hub(pts_list):
                scale = hub_scale_for_tcoh(hub_center, pts_list)
                verts, oct_faces, hex_faces, sq_faces = make_truncated_cuboctahedron(
                    hub_center, scale)
                hub_verts = np.vstack([verts, hub_center.reshape(1, 3)])
                add_tile(hub_verts, {
                    'type':       'hub_polyhedron',
                    'hub_center': hub_center,
                    'hub_key':    key,
                })
            else:
                ordered = convex_order_3d(pts_arr)
                if ordered is not None:
                    add_tile(ordered, {
                        'type':       'hub_face',
                        'hub_center': hub_center,
                        'hub_key':    key,
                    })

    else:
        points_2d = points_nd
        layers    = [0] if mode in [1, 4, 7] else list(range(nz_layers))

        for z_idx in layers:
            z_val = 0.0 if mode in [1, 4, 7] else z_idx / max(nz_layers - 1, 1)

            for s_idx, simplex in enumerate(tri.simplices):
                tri_pts = points_2d[simplex]
                c2d     = tri_pts.mean(axis=0)
                t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                     for i in range(3)])
                t3d     = np.hstack([t2d, np.full((3, 1), z_val)])
                add_tile(t3d, {
                    'type':        'tri_face',
                    'simplex_idx': s_idx,
                    'z_idx':       z_idx,
                    'z_val':       z_val,
                    'vertex_keys': [
                        (float(points_2d[simplex[i], 0]),
                         float(points_2d[simplex[i], 1]),
                         float(z_val))
                        for i in range(3)
                    ],
                })

            groups = {tuple(p): [] for p in points_2d}
            for s_idx, simplex in enumerate(tri.simplices):
                tri_pts = points_2d[simplex]
                c2d     = tri_pts.mean(axis=0)
                t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                     for i in range(3)])
                t3d     = np.hstack([t2d, np.full((3, 1), z_val)])
                for i, v in enumerate(tri_pts):
                    groups[tuple(v)].append(t3d[i])

            for key, pts_list in groups.items():
                if len(pts_list) >= 3:
                    try:
                        pts_arr = np.array(pts_list)
                        hull    = ConvexHull(pts_arr[:, :2])
                        verts   = pts_arr[hull.vertices, :2]
                        ring    = order_hub_ring_xy(verts, z_val)
                        v3d     = np.hstack([ring, np.full((len(ring), 1), z_val)])
                        add_tile(v3d, {
                            'type':    'hub_face',
                            'hub_key': key,
                            'z_val':   z_val,
                        })
                    except Exception:
                        pass

    return tiles, tile_source


def build_kirigami_constraints(tiles, tile_source):
    """Vertex-to-vertex constraints between tiles whose vertices coincide.

    Type 1 = default; type 2 = explicit "top" connection in layered 2D modes
    (when z is in the upper half of the global z-range).
    """
    TOLERANCE_DECIMALS = 5
    constraints = []

    pos_map = {}
    for t_idx, tile in enumerate(tiles):
        for v_idx, vert in enumerate(tile):
            key = tuple(np.round(vert, TOLERANCE_DECIMALS))
            pos_map.setdefault(key, []).append((t_idx, v_idx))

    all_z = [v[2] for tile in tiles for v in tile]
    z_min, z_max = (min(all_z), max(all_z)) if all_z else (0.0, 0.0)
    z_mid = (z_min + z_max) / 2.0

    seen = set()
    for key, occupants in pos_map.items():
        if len(occupants) < 2:
            continue
        for i in range(len(occupants)):
            for j in range(i + 1, len(occupants)):
                t1, v1 = occupants[i]
                t2, v2 = occupants[j]
                if t1 == t2:
                    continue

                if t1 < t2:
                    a, va, b, vb = t1, v1, t2, v2
                else:
                    a, va, b, vb = t2, v2, t1, v1
                pair_key = (a, va, b, vb)
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                z = tiles[a][va][2]
                ctype = 2 if (z_max > z_min and z > z_mid + 1e-9) else 1
                constraints.append((a, va, b, vb, ctype))

    return constraints
