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


def collect_kirigami_tiles(points_nd, tri, ratio, mode, nz_layers,
                           bipartite_C=1.0, bipartite_theta=0.0):
    """Polygonal tile faces for kirigami export.

    For 3D modes (3, 6) each shrunken tetrahedron is one tile (4 vertices)
    so PyKirigami treats it as a single rigid polyhedron rather than 4
    independent triangular faces.

    Mode 11 (bipartite auxetic) produces one flat tile per polygon — the
    central triangle and the three corner kites of each triangle — at the
    current actuation angle ``bipartite_theta``. There is exactly one
    z-layer (the structure is planar), so the simulator never sees the
    spurious second layer the generic 2.5D path would have built.
    """
    tiles       = []
    tile_source = []

    def add_tile(verts_3d, source_meta):
        verts_3d = np.asarray(verts_3d, float)
        if len(verts_3d) >= 3:
            tiles.append(verts_3d)
            tile_source.append(source_meta)

    if mode == 11:
        from .bipartite import build_bipartite_network
        net = build_bipartite_network(
            points_nd, np.asarray(tri.simplices),
            C=bipartite_C, theta=bipartite_theta)
        for p_idx, poly in enumerate(net.polygons):
            v3d = np.hstack([np.asarray(poly.vertices, float),
                             np.zeros((poly.degree, 1))])
            add_tile(v3d, {
                'type':          'tri_face',   # flat polygon → extruded on export
                'kind':          poly.kind,    # 'central' or 'corner'
                'poly_idx':      p_idx,
                'triangle_index': poly.triangle_index,
            })
        # Bonds are rigid 2-vertex bars linking adjacent kites along each
        # shared edge. They are STRUCTURAL, not decorative: each corner
        # kite is otherwise pinned only to its central polygon (one
        # hinge) and would be free to spin; the bonds tie it to its
        # neighbours so the whole network coheres into a single auxetic
        # floppy mode (validated on rhombus + hex patches). Appended
        # directly because ``add_tile`` rejects <3-vertex tiles.
        for b_idx, bond in enumerate(net.bonds):
            seg = np.hstack([np.asarray(bond, float), np.zeros((2, 1))])
            tiles.append(seg)
            tile_source.append({'type': 'bond', 'bond_idx': b_idx})
        return tiles, tile_source

    if mode == 12:
        # Mode 12 (3D tetrahedral auxetic): one rigid tile per internal
        # tetra (4 verts) + one per corner polyhedron (8 verts). The
        # canonical edge/face points fuse across adjacent tetrahedra
        # (see auxetic.tetrahedral), so ``build_kirigami_constraints``
        # turns the coincident vertices into the hinge constraints the
        # kinematic mechanism rides on. ``bipartite_C`` carries the
        # lattice's C in both call sites; clamp to the valid [0, 1]
        # contraction range (mode 11's C can exceed 1).
        from .tetrahedral import build_tetrahedral_network
        c_clamped = min(max(float(bipartite_C), 0.0), 1.0)
        net = build_tetrahedral_network(
            points_nd, np.asarray(tri.simplices), C=c_clamped)
        for poly in net.polyhedra:
            verts = np.asarray(poly.vertices, dtype=float)
            if poly.set_label == 'B':
                add_tile(verts, {
                    'type':      'tetrahedron',   # 4-vertex solid
                    'tetra_idx': poly.tetra_index,
                })
            else:
                add_tile(verts, {
                    'type':      'hub_polyhedron',  # convex corner solid
                    'tetra_idx': poly.tetra_index,
                    'corner':    poly.corner_point_index,
                })
        return tiles, tile_source

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
