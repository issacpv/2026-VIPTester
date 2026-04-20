import numpy as np
from itertools import permutations
from scipy.spatial import Delaunay, ConvexHull

# ==========================
# USER SETTINGS
# ==========================
mode        = 6
n_points    = 16
ratio       = 0.5
nz_layers   = 2

# --- Shape settings ---
ngon_thickness  = 0.03
hub_size_factor = 0.75

# --- Joint sphere settings ---
joint_sphere_radius  = 0.015   # set to 0 to disable
joint_sphere_rings   = 6       # latitude subdivisions
joint_sphere_segments = 8      # longitude subdivisions

# --- Export settings ---
export_scad      = False
export_stl       = True
export_obj       = False
export_vertices  = True   # NEW: export kirigami vertices file
export_constraints = True # NEW: export kirigami constraints file
export_scad_path = "auxetic_lattice.scad"
export_stl_path  = "auxetic_lattice.stl"
export_obj_path  = "auxetic_lattice.obj"
export_vertices_path    = "vertices.txt"    # NEW
export_constraints_path = "constraints.txt" # NEW

strut_radius   = 0.02
face_thickness = 0.015
scad_segments  = 8


# ==========================
# SPHERE MESH
# ==========================

def sphere_mesh(center, radius, rings, segments):
    """
    UV-sphere triangles centred at `center`.
    `rings`    = number of latitude bands (poles add +2 implicit caps).
    `segments` = number of longitude slices.
    Returns a list of [a, b, c] triangles (outward normals).
    """
    if radius < 1e-12:
        return []
    center = np.asarray(center, float)
    # Build vertex grid: (rings+1) latitude rows × segments columns
    verts = []
    for i in range(rings + 1):
        phi = np.pi * i / rings          # 0 … π
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            verts.append(center + np.array([x, y, z]))

    def vidx(i, j):
        return i * segments + (j % segments)

    tris = []
    for i in range(rings):
        for j in range(segments):
            a = vidx(i,     j)
            b = vidx(i,     j + 1)
            c = vidx(i + 1, j)
            d = vidx(i + 1, j + 1)
            # two triangles per quad; skip degenerate ones at poles
            if i > 0:
                tris.append([verts[a], verts[b], verts[c]])
            if i < rings - 1:
                tris.append([verts[b], verts[d], verts[c]])
    return tris


# ==========================
# TRIANGULATION
# ==========================

def triangulate_3d_grid_symmetric(nx, ny, nz):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    def idx(i, j, k):
        return i * (ny * nz) + j * nz + k

    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    tetrahedra = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c000 = idx(i,   j,   k  )
                c001 = idx(i,   j,   k+1)
                c010 = idx(i,   j+1, k  )
                c011 = idx(i,   j+1, k+1)
                c100 = idx(i+1, j,   k  )
                c101 = idx(i+1, j,   k+1)
                c110 = idx(i+1, j+1, k  )
                c111 = idx(i+1, j+1, k+1)

                sx = (i + 0.5) >= cx
                sy = (j + 0.5) >= cy
                sz = (k + 0.5) >= cz

                if sx and sy and sz:
                    a, b = c111, c000
                    ring = [c110, c010, c011, c001, c101, c100]
                elif not sx and sy and sz:
                    a, b = c011, c100
                    ring = [c001, c000, c010, c110, c111, c101]
                elif sx and not sy and sz:
                    a, b = c101, c010
                    ring = [c100, c000, c001, c011, c111, c110]
                elif sx and sy and not sz:
                    a, b = c110, c001
                    ring = [c111, c011, c010, c000, c100, c101]
                elif not sx and not sy and sz:
                    a, b = c001, c110
                    ring = [c000, c100, c101, c111, c011, c010]
                elif not sx and sy and not sz:
                    a, b = c010, c101
                    ring = [c011, c111, c110, c100, c000, c001]
                elif sx and not sy and not sz:
                    a, b = c100, c011
                    ring = [c110, c111, c101, c001, c000, c010]
                else:
                    a, b = c000, c111
                    ring = [c100, c110, c010, c011, c001, c101]

                for r in range(6):
                    tetrahedra.append([a, b, ring[r], ring[(r + 1) % 6]])

    class MockTri:
        def __init__(self, s): self.simplices = np.array(s)

    return points, MockTri(tetrahedra)


def triangulate_grid_symmetric(nx, ny):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    def idx(i, j): return j * nx + i

    center = np.array([0.5, 0.5])
    triangles = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            bl, br = idx(i, j), idx(i + 1, j)
            tl, tr = idx(i, j + 1), idx(i + 1, j + 1)
            qc = points[[bl, br, tl, tr]].mean(axis=0)
            to_c = center - qc
            if to_c[0] * to_c[1] > 0:
                triangles += [[bl, br, tr], [bl, tr, tl]]
            else:
                triangles += [[br, tr, tl], [br, tl, bl]]

    return points, np.array(triangles)


# ==========================
# POINT GENERATION
# ==========================

def generate_points(n_points, mode):
    if mode == 3:
        pts = np.random.rand(n_points, 3)
        return pts, Delaunay(pts)
    elif mode == 6:
        cbrt = max(2, round(n_points ** (1 / 3)))
        return triangulate_3d_grid_symmetric(cbrt, cbrt, cbrt)
    elif mode in [4, 5]:
        def factor_pair(n):
            for i in range(int(np.sqrt(n)), 0, -1):
                if n % i == 0: return i, n // i
            return 1, n
        nx2d, ny2d = factor_pair(n_points)
        pts, tris = triangulate_grid_symmetric(nx2d, ny2d)

        class MockTri:
            def __init__(self, s): self.simplices = np.array(s)

        return pts, MockTri(tris)
    else:
        pts = np.random.rand(n_points, 2)
        return pts, Delaunay(pts)


# ==========================
# TRUNCATED CUBOCTAHEDRON
# ==========================

def make_truncated_cuboctahedron(center, scale):
    a, b, c = 1.0, 1.0 + np.sqrt(2), 1.0 + 2 * np.sqrt(2)
    raw = set()
    for perm in permutations([a, b, c]):
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    raw.add((perm[0]*sx, perm[1]*sy, perm[2]*sz))

    verts   = np.array(sorted(raw), dtype=float)
    mean_r  = np.mean(np.linalg.norm(verts, axis=1))
    verts   = verts * (scale / mean_r)
    center  = np.asarray(center, float)
    verts  += center

    def collect_face(normal_axis, sign, expected_count):
        coords    = (verts - center)[:, normal_axis]
        threshold = sign * (coords.max() if sign > 0 else -coords.min()) * 0.999
        face      = [i for i, v in enumerate(verts - center)
                     if sign * v[normal_axis] >= threshold]
        return face if len(face) == expected_count else []

    oct_faces = []
    for axis in range(3):
        for sign in [1, -1]:
            face = collect_face(axis, sign, 8)
            if face:
                pts   = verts[face]
                fc    = pts.mean(axis=0)
                other = [a for a in range(3) if a != axis]
                angles = np.arctan2(
                    (pts - fc)[:, other[1]],
                    (pts - fc)[:, other[0]])
                oct_faces.append([face[i] for i in np.argsort(angles)])

    hex_faces = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                diag      = np.array([sx, sy, sz], float)
                dots      = (verts - center) @ diag
                threshold = dots.max() * 0.999
                face      = [i for i, d in enumerate(dots) if d >= threshold]
                if len(face) == 6:
                    pts  = verts[face]
                    fc   = pts.mean(axis=0)
                    dn   = diag / np.linalg.norm(diag)
                    u    = np.cross(dn, [1, 0, 0])
                    if np.linalg.norm(u) < 1e-6:
                        u = np.cross(dn, [0, 1, 0])
                    u   /= np.linalg.norm(u)
                    v_ax = np.cross(dn, u)
                    off  = pts - fc
                    angles = np.arctan2(off @ v_ax, off @ u)
                    hex_faces.append([face[i] for i in np.argsort(angles)])

    sq_faces    = []
    edge_normals = []
    for i in range(3):
        for j in range(3):
            if i != j:
                for si in [1, -1]:
                    for sj in [1, -1]:
                        n = np.zeros(3)
                        n[i] = si; n[j] = sj
                        edge_normals.append(n / np.linalg.norm(n))
    seen = set()
    for n in edge_normals:
        nkey = tuple(np.round(n, 6))
        if nkey in seen: continue
        seen.add(nkey)
        dots      = (verts - center) @ n
        threshold = dots.max() * 0.999
        face      = [i for i, d in enumerate(dots) if d >= threshold]
        if len(face) == 4:
            pts  = verts[face]
            fc   = pts.mean(axis=0)
            u    = np.cross(n, [1, 0, 0])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(n, [0, 1, 0])
            u   /= np.linalg.norm(u)
            v_ax = np.cross(n, u)
            off  = pts - fc
            angles = np.arctan2(off @ v_ax, off @ u)
            sq_faces.append([face[i] for i in np.argsort(angles)])

    return verts, oct_faces, hex_faces, sq_faces


def is_central_hub(pts_list):
    if len(pts_list) < 8:
        return False
    pts      = np.array(pts_list)
    centroid = pts.mean(axis=0)
    offsets  = pts - centroid
    octants  = set()
    for off in offsets:
        sx = 1 if off[0] > 1e-9 else (-1 if off[0] < -1e-9 else 0)
        sy = 1 if off[1] > 1e-9 else (-1 if off[1] < -1e-9 else 0)
        sz = 1 if off[2] > 1e-9 else (-1 if off[2] < -1e-9 else 0)
        if sx != 0 and sy != 0 and sz != 0:
            octants.add((sx, sy, sz))
    return len(octants) == 8


# ==========================
# GEOMETRY HELPERS
# ==========================

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < 1e-12 else v / n


def convex_order_3d(pts):
    pts      = np.asarray(pts, float)
    if len(pts) < 3:
        return None
    centroid = pts.mean(axis=0)
    offsets  = pts - centroid
    _, _, Vt = np.linalg.svd(offsets)
    normal   = Vt[-1]
    mags     = np.linalg.norm(offsets, axis=1)
    u        = offsets[np.argmax(mags)]
    u        = u - np.dot(u, normal) * normal
    if np.linalg.norm(u) < 1e-12:
        return None
    u  = u / np.linalg.norm(u)
    v  = np.cross(normal, u)
    if np.linalg.norm(v) < 1e-12:
        return None
    v      = v / np.linalg.norm(v)
    coords = offsets @ np.stack([u, v], axis=1)
    return pts[np.argsort(np.arctan2(coords[:, 1], coords[:, 0]))]


def newell_normal(poly):
    n      = len(poly)
    normal = np.zeros(3)
    for i in range(n):
        j         = (i + 1) % n
        normal[0] += (poly[i][1] - poly[j][1]) * (poly[i][2] + poly[j][2])
        normal[1] += (poly[i][2] - poly[j][2]) * (poly[i][0] + poly[j][0])
        normal[2] += (poly[i][0] - poly[j][0]) * (poly[i][1] + poly[j][1])
    nn = np.linalg.norm(normal)
    return normal / nn if nn > 1e-10 else None


def order_hub_ring_xy(vertices_xy, z_plane):
    xy = np.asarray(vertices_xy, float)
    if len(xy) < 3:
        return xy
    v3 = np.hstack([xy, np.full((len(xy), 1), z_plane)])
    o  = convex_order_3d(v3)
    return o[:, :2] if o is not None else xy


# ==========================
# SOLID MESH BUILDERS
# ==========================

def triangles_for_solid_tetrahedron(tet_verts):
    v          = np.asarray(tet_verts, float)
    triangles  = []
    face_indices = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    for fi in face_indices:
        a, b, c  = v[fi[0]], v[fi[1]], v[fi[2]]
        opposite = v[list(set(range(4)) - set(fi))[0]]
        normal   = np.cross(b - a, c - a)
        if np.dot(normal, a - opposite) < 0:
            b, c = c, b
        triangles.append([a, b, c])
    return triangles


def triangles_for_convex_solid(pts):
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return []
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    centroid  = pts.mean(axis=0)
    triangles = []
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        normal  = np.cross(b - a, c - a)
        if np.dot(normal, a - centroid) < 0:
            b, c = c, b
        triangles.append([a, b, c])
    return triangles


def _extrude_polygon_solid(verts_3d):
    verts_3d   = np.asarray(verts_3d, float)
    normal     = newell_normal(verts_3d)
    if normal is None:
        return []
    top        = verts_3d
    bottom     = verts_3d - normal * ngon_thickness
    n          = len(verts_3d)
    tris       = []
    centroid_t = top.mean(axis=0)
    for i in range(1, n - 1):
        t0, ti, ti1 = top[0], top[i], top[i + 1]
        if np.dot(np.cross(ti - t0, ti1 - t0), normal) < 0:
            tris.append([t0, ti1, ti])
        else:
            tris.append([t0, ti, ti1])
        b0, bi, bi1 = bottom[0], bottom[i], bottom[i + 1]
        if np.dot(np.cross(bi - b0, bi1 - b0), -normal) < 0:
            tris.append([b0, bi1, bi])
        else:
            tris.append([b0, bi, bi1])
    for i in range(n):
        j        = (i + 1) % n
        a, b, c, d = top[i], top[j], bottom[j], bottom[i]
        mid      = (a + b + c + d) / 4.0
        n1       = np.cross(b - a, d - a)
        if np.dot(n1, mid - centroid_t) < 0:
            tris.append([a, d, b])
            tris.append([b, d, c])
        else:
            tris.append([a, b, d])
            tris.append([b, c, d])
    return tris


# ==========================
# HUB EXPORT
# ==========================

def _hub_scale_for_tcoh(hub_center, pts_list):
    mean_dist = np.mean([np.linalg.norm(p - hub_center) for p in pts_list])
    a, b, c   = 1.0, 1.0 + np.sqrt(2), 1.0 + 2 * np.sqrt(2)
    all_verts = []
    for perm in permutations([a, b, c]):
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    all_verts.append([perm[0]*sx, perm[1]*sy, perm[2]*sz])
    all_verts = np.array(all_verts, dtype=float)
    mean_r    = np.mean(np.linalg.norm(all_verts, axis=1))
    oct_fc_dist = c / mean_r
    return (mean_dist / oct_fc_dist) * hub_size_factor


def dispatch_hub_export(key, pts_list, all_triangles):
    hub_center = np.array(key, float)
    if is_central_hub(pts_list):
        scale                         = _hub_scale_for_tcoh(hub_center, pts_list)
        verts, oct_faces, hex_faces, sq_faces = make_truncated_cuboctahedron(hub_center, scale)
        all_triangles.extend(triangles_for_convex_solid(verts))
    else:
        pts_arr = np.array(pts_list)
        ordered = convex_order_3d(pts_arr)
        if ordered is not None:
            all_triangles.extend(_extrude_polygon_solid(ordered))


# ==========================
# 3D GROUP BUILDER
# ==========================

def build_3d_groups(pts_norm, tri, ratio):
    groups = {tuple(p): [] for p in pts_norm}
    for simplex in tri.simplices:
        tet      = pts_norm[simplex]
        centroid = tet.mean(axis=0)
        shrunk   = np.array([(1 - ratio) * tet[i] + ratio * centroid
                              for i in range(4)])
        for i, vertex in enumerate(pts_norm[simplex]):
            groups[tuple(vertex)].append(shrunk[i])
    return groups


# ==========================
# GEOMETRY COLLECTION
# ==========================

def collect_export_geometry(points_nd, tri, ratio, mode, nz_layers):
    strut_curves  = []
    all_triangles = []
    # Deduplicated set of joint positions for sphere placement
    joint_positions = set()

    def add_strut(p0, p1):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        if np.linalg.norm(p1 - p0) > 1e-9:
            strut_curves.append(np.array([p0, p1]))

    def register_joint(pt):
        """Round to a grid to deduplicate near-coincident joints."""
        key = tuple(np.round(pt, 8))
        joint_positions.add(key)

    if mode in [3, 6]:
        pts_norm = points_nd
        groups   = build_3d_groups(pts_norm, tri, ratio)

        for simplex in tri.simplices:
            tet      = pts_norm[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = np.array([(1 - ratio) * tet[i] + ratio * centroid
                                  for i in range(4)])
            all_triangles.extend(triangles_for_solid_tetrahedron(shrunk))
            # Register each shrunken corner as a joint
            for pt in shrunk:
                register_joint(pt)

        for key, pts_list in groups.items():
            if len(pts_list) == 2:
                add_strut(pts_list[0], pts_list[1])
            elif len(pts_list) >= 3:
                dispatch_hub_export(key, pts_list, all_triangles)

    else:
        points_2d = points_nd
        layers    = [0] if mode in [1, 4] else list(range(nz_layers))

        for simplex in tri.simplices:
            tri_pts = points_2d[simplex]
            c2d     = tri_pts.mean(axis=0)
            t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                 for i in range(3)])
            if mode in [1, 4]:
                face3 = np.hstack([t2d, np.zeros((3, 1))])
                all_triangles.extend(_extrude_polygon_solid(face3))
                for pt in face3:
                    register_joint(pt)
            elif mode in [2, 5]:
                for z_idx in range(nz_layers - 1):
                    zb = z_idx / (nz_layers - 1)
                    zt = (z_idx + 1) / (nz_layers - 1)
                    bp = np.hstack([t2d, np.full((3, 1), zb)])
                    tp = np.hstack([t2d, np.full((3, 1), zt)])
                    all_triangles.extend(_extrude_polygon_solid(tp))
                    for pt in bp: register_joint(pt)
                    for pt in tp: register_joint(pt)
                    for i in range(3):
                        j    = (i + 1) % 3
                        quad = np.array([bp[i], bp[j], tp[j], tp[i]])
                        all_triangles.extend(_extrude_polygon_solid(quad))

        for z_idx in layers:
            z_val  = 0.0 if mode in [1, 4] else z_idx / (nz_layers - 1)
            groups = {tuple(p): [] for p in points_2d}

            for simplex in tri.simplices:
                tri_pts = points_2d[simplex]
                c2d     = tri_pts.mean(axis=0)
                t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                     for i in range(3)])
                t3d     = np.hstack([t2d, np.full((3, 1), z_val)])
                for i, v in enumerate(tri_pts):
                    groups[tuple(v)].append(t3d[i])
                for pt in t3d:
                    register_joint(pt)

            for key, pts_list in groups.items():
                if len(pts_list) == 2:
                    p0, p1 = pts_list
                    if mode in [1, 4]:
                        add_strut(p0, p1)
                    else:
                        zb = z_idx / (nz_layers - 1)
                        zt = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                        add_strut([p0[0], p0[1], zb], [p1[0], p1[1], zb])
                        add_strut([p0[0], p0[1], zt], [p1[0], p1[1], zt])
                        add_strut([p0[0], p0[1], zb], [p0[0], p0[1], zt])
                        add_strut([p1[0], p1[1], zb], [p1[0], p1[1], zt])
                elif len(pts_list) >= 3:
                    try:
                        pts_arr = np.array(pts_list)
                        hull    = ConvexHull(pts_arr[:, :2])
                        verts   = pts_arr[hull.vertices, :2]
                        if mode in [1, 4]:
                            ring = order_hub_ring_xy(verts, z_val)
                            v3d  = np.hstack([ring, np.full((len(ring), 1), z_val)])
                            all_triangles.extend(_extrude_polygon_solid(v3d))
                        else:
                            zb   = z_idx / (nz_layers - 1)
                            zt   = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                            ring = order_hub_ring_xy(verts, zb)
                            nr   = len(ring)
                            bv   = np.hstack([ring, np.full((nr, 1), zb)])
                            tv   = np.hstack([ring, np.full((nr, 1), zt)])
                            if abs(zt - zb) >= 1e-9:
                                all_triangles.extend(triangles_for_convex_solid(np.vstack([bv, tv])))
                            else:
                                all_triangles.extend(_extrude_polygon_solid(bv))
                    except Exception:
                        pass

    return strut_curves, all_triangles, joint_positions


# ==========================
# MESH BUILD
# ==========================

def build_export_triangles(strut_curves, all_triangles, joint_positions):
    result = list(all_triangles)

    def tube_mesh(path, radius, segments):
        path  = np.asarray(path, float)
        rings = []
        for k, pt in enumerate(path):
            tang = path[1] - path[0] if k == 0 else \
                   path[-1] - path[-2] if k == len(path) - 1 else \
                   path[k + 1] - path[k - 1]
            tn = np.linalg.norm(tang)
            if tn < 1e-12: continue
            tang  = tang / tn
            perp  = np.cross(tang, [0, 0, 1])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(tang, [0, 1, 0])
            perp  = perp / np.linalg.norm(perp)
            perp2 = np.cross(tang, perp)
            rings.append([pt + radius * (np.cos(2 * np.pi * s / segments) * perp +
                                          np.sin(2 * np.pi * s / segments) * perp2)
                           for s in range(segments)])
        tris = []
        for i in range(len(rings) - 1):
            r0, r1 = rings[i], rings[i + 1]
            for s in range(segments):
                s1 = (s + 1) % segments
                tris += [[r0[s], r1[s], r1[s1]], [r0[s], r1[s1], r0[s1]]]
        for s in range(segments):
            s1 = (s + 1) % segments
            tris.append([path[0],  rings[0][-1 - s1],  rings[0][-1 - s]])
            tris.append([path[-1], rings[-1][s],        rings[-1][s1]])
        return tris

    for path in strut_curves:
        result += tube_mesh(path, strut_radius, scad_segments)

    # --- Joint spheres ---
    if joint_sphere_radius > 1e-12:
        n_spheres = 0
        for pt_key in joint_positions:
            center = np.array(pt_key, float)
            result += sphere_mesh(center, joint_sphere_radius,
                                  joint_sphere_rings, joint_sphere_segments)
            n_spheres += 1
        print(f"  Joint spheres: {n_spheres} placed")

    return result


# ==========================
# WRITERS
# ==========================

def _fmt(v):
    return f"[{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}]"


def export_to_scad(scad_path, strut_curves, triangles):
    def scad_cylinder(p0, p1, radius, fn):
        d   = p1 - p0
        L   = np.linalg.norm(d)
        if L < 1e-10: return ""
        dh  = d / L
        dot = np.clip(dh[2], -1, 1)
        ang = np.degrees(np.arccos(dot))
        ax  = np.cross([0, 0, 1], dh)
        an  = np.linalg.norm(ax)
        ax  = ax / an if an > 1e-10 else np.array([1, 0, 0])
        return (f"translate({_fmt(p0)})\n"
                f"  rotate(a={ang:.6f},v={_fmt(ax)})\n"
                f"    cylinder(h={L:.6f},r={radius:.6f},$fn={fn});\n")

    lines = [f"// auxetic lattice  mode={mode}  n_points={n_points}  ratio={ratio}\n",
             f"$fn={scad_segments};\nrender(convexity=10)\nunion(){{\n",
             "  // struts\n"]
    for pts in strut_curves:
        lines.append(scad_cylinder(pts[0], pts[1], strut_radius, scad_segments))
    if triangles:
        all_v, all_f, vi = [], [], 0
        for tri in triangles:
            a, b, c = (np.asarray(tri[k], float) for k in range(3))
            all_v  += [a, b, c]
            all_f.append([vi, vi + 1, vi + 2])
            vi     += 3
        vs = ", ".join(_fmt(v) for v in all_v)
        fs = ", ".join("[" + ",".join(str(x) for x in f) + "]" for f in all_f)
        lines.append(f"polyhedron(points=[{vs}],faces=[{fs}],convexity=10);\n")
    lines.append("}\n")
    with open(scad_path, "w") as f:
        f.writelines(lines)
    print(f"  SCAD: {scad_path}  ({len(strut_curves)} struts, {len(triangles)} tris)")


def export_stl_direct(stl_path, triangles):
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        print("  numpy-stl not installed — skipping STL"); return
    import os
    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        for j in range(3):
            m.vectors[i][j] = np.asarray(tri[j], float)
    m.save(stl_path)
    print(f"  STL: {stl_path}  ({os.path.getsize(stl_path) // 1024} KB)")


def export_obj_direct(obj_path, triangles):
    import os
    lines   = ["# auxetic lattice\n"]
    v_count = 0
    n_count = 0
    for tri in triangles:
        a, b, c = (np.asarray(tri[k], float) for k in range(3))
        nv      = np.cross(b - a, c - a)
        nn      = np.linalg.norm(nv)
        if nn < 1e-14: continue
        nv      = nv / nn
        vb      = v_count + 1
        for p in (a, b, c):
            lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        v_count += 3
        n_count += 1
        lines.append(f"vn {nv[0]:.9g} {nv[1]:.9g} {nv[2]:.9g}\n")
        lines.append(f"f {vb}//{n_count} {vb+1}//{n_count} {vb+2}//{n_count}\n")
    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    print(f"  OBJ: {obj_path}  ({os.path.getsize(obj_path) // 1024} KB, {n_count} tris)")


# ==========================
# NEW: KIRIGAMI TILE EXPORT
# ==========================

# ==========================
# REPLACEMENT: KIRIGAMI TILE EXPORT
# ==========================
# Drop-in replacements for collect_kirigami_tiles() and
# build_kirigami_constraints() in your script.

def collect_kirigami_tiles(points_nd, tri, ratio, mode, nz_layers):
    """
    Collect polygonal tile faces for kirigami export.

    For 3D modes (3, 6): each shrunken tetrahedron is emitted as ONE tile
    containing all 4 vertices, so PyKirigami treats it as a single rigid
    polyhedron rather than 4 independent triangular faces.
    Hub polygons are emitted as before.
    """
    tiles       = []
    tile_source = []

    def add_tile(verts_3d, source_meta):
        verts_3d = np.asarray(verts_3d, float)
        if len(verts_3d) >= 3:
            tiles.append(verts_3d)
            tile_source.append(source_meta)

    if mode in [3, 6]:
        pts_norm = points_nd

        # Identify central-hub lattice points first so tet-corner shrinking
        # can skip them: a tet's hub-facing corner stays at the lattice point
        # so all tets meeting at that hub share exactly one world position
        # (single ball joint → hub retains rotational DOF).
        groups       = build_3d_groups(pts_norm, tri, ratio)
        TOL_KEY      = 9
        central_keys = set()
        for key, pts_list in groups.items():
            if is_central_hub(pts_list):
                central_keys.add(tuple(np.round(np.asarray(key, float), TOL_KEY)))

        def _key(pt):
            return tuple(np.round(np.asarray(pt, float), TOL_KEY))

        # --- Each tetrahedron is ONE tile (4 vertices, full 3D solid) ---
        for s_idx, simplex in enumerate(tri.simplices):
            tet      = pts_norm[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = np.empty_like(tet)
            for i in range(4):
                if _key(tet[i]) in central_keys:
                    shrunk[i] = tet[i]  # stay pinned at the hub center
                else:
                    shrunk[i] = (1 - ratio) * tet[i] + ratio * centroid

            add_tile(shrunk, {
                'type':         'tetrahedron',
                'simplex_idx':  s_idx,
                'vertex_keys':  [tuple(pts_norm[simplex[i]]) for i in range(4)],
            })

        # --- Hubs ---
        for key, pts_list in groups.items():
            hub_center = np.array(key, float)
            if len(pts_list) < 3:
                continue
            pts_arr = np.array(pts_list)

            if is_central_hub(pts_list):
                scale = _hub_scale_for_tcoh(hub_center, pts_list)
                verts, oct_faces, hex_faces, sq_faces = make_truncated_cuboctahedron(
                    hub_center, scale)
                # Attach to the tet ring via a single shared point at hub_center.
                # The coincidence-based constraint pass will then link this one
                # vertex to every tet corner sitting at hub_center, giving one
                # multi-way ball joint — the hub is free to rotate.
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
        # 2-D / extruded modes unchanged from before
        points_2d = points_nd
        layers    = [0] if mode in [1, 4] else list(range(nz_layers))

        for z_idx in layers:
            z_val = 0.0 if mode in [1, 4] else z_idx / max(nz_layers - 1, 1)

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
    """
    Build vertex-to-vertex constraints between tiles.

    Connection-type assignment:
      - type 1 (default): tile-to-tile connection at a shared lattice point
      - type 2: reserved for explicit "top" connections in layered 2D modes

    Because each tetrahedron is now a single tile (4 vertices), there are no
    intra-tet constraints needed — the tile itself enforces rigidity.
    Constraints only link DIFFERENT tiles whose vertices coincide (e.g.
    tet corners meeting at a hub).
    """
    TOLERANCE_DECIMALS = 5
    constraints = []

    # Map rounded position → list of (tile_idx, vert_idx)
    pos_map = {}
    for t_idx, tile in enumerate(tiles):
        for v_idx, vert in enumerate(tile):
            key = tuple(np.round(vert, TOLERANCE_DECIMALS))
            pos_map.setdefault(key, []).append((t_idx, v_idx))

    # Determine global z-range for type-1/type-2 split (only meaningful in 2D modes)
    all_z = [v[2] for tile in tiles for v in tile]
    z_min, z_max = (min(all_z), max(all_z)) if all_z else (0.0, 0.0)
    z_mid = (z_min + z_max) / 2.0

    seen = set()
    for key, occupants in pos_map.items():
        if len(occupants) < 2:
            continue
        # Pairwise constraints between DIFFERENT tiles only
        for i in range(len(occupants)):
            for j in range(i + 1, len(occupants)):
                t1, v1 = occupants[i]
                t2, v2 = occupants[j]
                if t1 == t2:
                    continue  # skip self-pairs (no intra-tile constraint needed)

                # Canonical ordering to dedupe
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

def export_kirigami_vertices(path, tiles):
    """
    Write vertices file: one tile per line, 3n space-separated floats.
    x1 y1 z1 x2 y2 z2 ... xn yn zn
    """
    import os
    lines = []
    for tile in tiles:
        coords = []
        for vert in tile:
            coords.extend([f"{vert[0]:.9g}", f"{vert[1]:.9g}", f"{vert[2]:.9g}"])
        lines.append(" ".join(coords) + "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    print(f"  Vertices: {path}  ({len(tiles)} tiles, "
          f"{os.path.getsize(path) // 1024} KB)")


def export_kirigami_constraints(path, constraints):
    """
    Write constraints file:
    tile1_index vertex1_index tile2_index vertex2_index connection_type
    All indices are 0-based.
    """
    import os
    lines = []
    for (t1, v1, t2, v2, ctype) in constraints:
        lines.append(f"{t1} {v1} {t2} {v2} {ctype}\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    print(f"  Constraints: {path}  ({len(constraints)} connections, "
          f"{os.path.getsize(path) // 1024} KB)")


# ==========================
# ENTRY POINT
# ==========================

def run_export(points_nd, tri, ratio, mode, nz_layers):
    import os
    print("\n--- Export ---")
    base    = os.path.dirname(os.path.abspath(__file__))
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(base, p)

    strut_curves, all_triangles, joint_positions = collect_export_geometry(
        points_nd, tri, ratio, mode, nz_layers)
    print(f"  Geometry: {len(strut_curves)} struts, {len(all_triangles)} solid tris, "
          f"{len(joint_positions)} joint positions")

    triangles = build_export_triangles(strut_curves, all_triangles, joint_positions)
    print(f"  Final mesh: {len(triangles)} triangles (incl. strut tubes + joint spheres)")

    if export_scad:
        export_to_scad(resolve(export_scad_path), strut_curves, triangles)
    if export_stl:
        export_stl_direct(resolve(export_stl_path), triangles)
    if export_obj:
        export_obj_direct(resolve(export_obj_path), triangles)

    # --- Kirigami tile export ---
    if export_vertices or export_constraints:
        print("\n--- Kirigami Tile Export ---")
        tiles, tile_source = collect_kirigami_tiles(
            points_nd, tri, ratio, mode, nz_layers)
        print(f"  Collected {len(tiles)} tiles")

        if export_vertices:
            export_kirigami_vertices(resolve(export_vertices_path), tiles)

        if export_constraints:
            constraints = build_kirigami_constraints(tiles, tile_source)
            print(f"  Built {len(constraints)} constraints")
            export_kirigami_constraints(resolve(export_constraints_path), constraints)

    print("--------------\n")


points_nd, tri_nd = generate_points(n_points, mode)
run_export(points_nd, tri_nd, ratio, mode, nz_layers)