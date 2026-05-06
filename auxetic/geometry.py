"""Geometric primitives: point generation, triangulation, hub detection,
mesh helpers, and STL-mesh assembly.

Module-level constants (`NGON_THICKNESS`, `HUB_SIZE_FACTOR`, `STRUT_RADIUS`,
`SCAD_SEGMENTS`, `JOINT_SPHERE_*`) match the originals from the V20 script
so default behavior is byte-identical.
"""

import numpy as np
from itertools import permutations
from scipy.spatial import Delaunay, ConvexHull


NGON_THICKNESS = 0.03
HUB_SIZE_FACTOR = 0.75

JOINT_SPHERE_RADIUS = 0.015
JOINT_SPHERE_RINGS = 6
JOINT_SPHERE_SEGMENTS = 8

STRUT_RADIUS = 0.02
SCAD_SEGMENTS = 8


def sphere_mesh(center, radius, rings, segments):
    if radius < 1e-12:
        return []
    center = np.asarray(center, float)
    verts = []
    for i in range(rings + 1):
        phi = np.pi * i / rings
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
            if i > 0:
                tris.append([verts[a], verts[b], verts[c]])
            if i < rings - 1:
                tris.append([verts[b], verts[d], verts[c]])
    return tris


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


def _density_bias_inv_cdf(u: np.ndarray, law: str, strength: float) -> np.ndarray:
    """Map uniform samples ``u`` ∈ [0, 1] through a density-biasing
    inverse-CDF for the chosen ``law``.

    ``strength`` parameterises how skewed the result is. ``strength = 0``
    (and ``law = "uniform"``) recover ``u`` unchanged for every law.

    - ``"linear"``: linear-ramp density ρ(x) = 1 + s·(2x − 1) on [0, 1],
      ``s ∈ [-1, 1]``. Positive s biases samples toward x = 1.
    - ``"log"``: logarithmic CDF x = log(1 + u·(e^s − 1)) / s. The
      inverse-CDF is concave so samples skew above the diagonal —
      positive s biases toward x = 1.
    - ``"exp"``: exponential CDF x = (e^(s·u) − 1) / (e^s − 1). The
      inverse-CDF is convex so samples skew below the diagonal —
      positive s biases toward x = 0. (To bias toward x = 1 with an
      exponential curve, use ``law="log"``; to bias toward x = 0 with
      a logarithmic curve, use ``law="exp"``. The two laws are
      inverses of each other.)
    """
    if law == "uniform" or abs(strength) < 1e-12:
        return u
    if law == "linear":
        s = float(np.clip(strength, -1.0, 1.0))
        if abs(s) < 1e-12:
            return u
        disc = (1.0 - s) ** 2 + 4.0 * s * u
        return (-(1.0 - s) + np.sqrt(disc)) / (2.0 * s)
    if law == "log":
        return np.log1p(u * np.expm1(strength)) / strength
    if law == "exp":
        return np.expm1(strength * u) / np.expm1(strength)
    raise ValueError(f"Unknown density_law: {law!r}")


def generate_points(n_points, mode, *,
                    density_axis: str = "none",
                    density_law:  str = "uniform",
                    density_strength: float = 1.0):
    """Generate the lattice point cloud and its triangulation for ``mode``.

    With every keyword arg at its default the function reproduces the
    original V20 script's output byte-for-byte (regression-tested in
    ``tests/test_regression.py``).

    The optional density-gradient knobs only take effect for the random
    Delaunay modes (1, 2, 3). Grid modes (4, 5, 6) ignore them — their
    points are deterministic by construction.
    """
    bias_active = (
        density_axis != "none"
        and density_law != "uniform"
        and mode in (1, 2, 3)
    )

    if not bias_active:
        # Legacy code path — must remain byte-identical to V20.
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

    # Biased random Delaunay path (modes 1, 2, 3).
    dim = 3 if mode == 3 else 2
    pts = np.random.rand(n_points, dim)
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(density_axis, -1)
    if 0 <= axis_idx < dim:
        pts[:, axis_idx] = _density_bias_inv_cdf(
            pts[:, axis_idx], density_law, density_strength)
    return pts, Delaunay(pts)


# ---------------------------------------------------------------------------
# Edge-flip apparatus (2D only).
#
# In a 2D triangulation, each interior edge is shared by exactly two
# triangles; together those triangles form a quadrilateral with one
# diagonal (the edge). Flipping the edge swaps that diagonal for the
# other one. The flip is geometrically valid iff the quad is strictly
# convex; otherwise the result self-intersects.
#
# 3D tetrahedral flips (2-3 / 3-2 bistellar moves) are intentionally
# out of scope for M1 — the convexity book-keeping is much harder and
# not required for the kirigami pipeline yet.
# ---------------------------------------------------------------------------


class _FlippedTri:
    """Lightweight stand-in for ``scipy.spatial.Delaunay``, exposing the
    ``simplices`` attribute (the only thing the downstream geometry and
    tile-collection pipeline reads off ``tri``)."""

    __slots__ = ("simplices",)

    def __init__(self, simplices):
        self.simplices = np.asarray(simplices, dtype=np.int64)


def _cross2d(u, v) -> float:
    """Signed scalar 2D cross product ``u_x·v_y − u_y·v_x``. Used in place
    of ``np.cross`` on 2-vectors, which is deprecated in NumPy 2.0."""
    return float(u[0]) * float(v[1]) - float(u[1]) * float(v[0])


def _orient_ccw(triangle, points_2d):
    """Return ``triangle`` with vertex order swapped if needed so its
    signed area is non-negative (counter-clockwise)."""
    a, b, c = (int(v) for v in triangle)
    cross = _cross2d(points_2d[b] - points_2d[a],
                     points_2d[c] - points_2d[a])
    return [a, b, c] if cross >= 0.0 else [a, c, b]


def _quad_is_strictly_convex(pa, pb, pc, pd):
    """``(a, b)`` is the candidate diagonal; ``c`` and ``d`` are the
    apex vertices of the two adjacent triangles. The quad is strictly
    convex iff each diagonal separates the other diagonal's endpoints —
    i.e. ``c`` and ``d`` lie on opposite sides of line ``(a, b)`` AND
    ``a`` and ``b`` lie on opposite sides of line ``(c, d)``."""
    s1 = _cross2d(pb - pa, pc - pa) * _cross2d(pb - pa, pd - pa)
    s2 = _cross2d(pd - pc, pa - pc) * _cross2d(pd - pc, pb - pc)
    return s1 < 0.0 and s2 < 0.0


def flippable_edges(tri, points):
    """Return the sorted list of edges in ``tri`` that can be diagonal-flipped.

    Each edge is a tuple ``(i, j)`` with ``i < j`` referring to vertex
    indices into ``points``. Boundary edges (one adjacent triangle) and
    non-convex-quad edges are excluded. Returns ``[]`` for 3D
    triangulations or otherwise unsupported inputs.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return []
    simplices = np.asarray(tri.simplices)
    if simplices.size == 0 or simplices.shape[1] != 3:
        return []

    edge_to_opps: dict[tuple[int, int], list[int]] = {}
    for simplex in simplices:
        s = [int(v) for v in simplex]
        for k in range(3):
            a, b = s[k], s[(k + 1) % 3]
            if a > b:
                a, b = b, a
            opp = s[(k + 2) % 3]
            edge_to_opps.setdefault((a, b), []).append(opp)

    out: list[tuple[int, int]] = []
    for (a, b), opps in sorted(edge_to_opps.items()):
        if len(opps) != 2:
            continue
        c, d = opps
        if _quad_is_strictly_convex(pts[a], pts[b], pts[c], pts[d]):
            out.append((a, b))
    return out


def apply_edge_flips(tri, points, flips):
    """Apply ``flips`` (an iterable of ``(i, j)`` edges) to ``tri``.

    Flips are applied in sorted order on the *current* triangulation
    state, so a sequence of flips compounds. Edges that are not
    currently flippable (boundary, non-convex quad, or no longer present
    after earlier flips) are silently skipped.

    The returned object exposes a ``.simplices`` attribute compatible
    with the rest of the geometry pipeline; the original ``tri`` is
    returned unchanged when ``flips`` is empty.
    """
    if not flips:
        return tri
    pts = np.asarray(points, float)
    simplices_in = np.asarray(tri.simplices)
    if (pts.ndim != 2 or pts.shape[1] != 2 or
            simplices_in.size == 0 or simplices_in.shape[1] != 3):
        return tri

    simplices = simplices_in.astype(np.int64, copy=True)
    flips_sorted = sorted({tuple(sorted((int(i), int(j)))) for i, j in flips})

    for a, b in flips_sorted:
        adj_idx: list[int] = []
        for ti in range(simplices.shape[0]):
            verts = set(int(v) for v in simplices[ti])
            if a in verts and b in verts:
                adj_idx.append(ti)
        if len(adj_idx) != 2:
            continue
        t1, t2 = adj_idx
        c = next(int(v) for v in simplices[t1] if int(v) not in (a, b))
        d = next(int(v) for v in simplices[t2] if int(v) not in (a, b))
        if not _quad_is_strictly_convex(pts[a], pts[b], pts[c], pts[d]):
            continue
        simplices[t1] = _orient_ccw([a, c, d], pts)
        simplices[t2] = _orient_ccw([b, c, d], pts)

    return _FlippedTri(simplices)


def points_from_mesh_vertices(verts: np.ndarray, mode: int):
    """Build the (points, tri) pair for mesh-import modes 7, 8, 9.

    ``verts`` is the (N, 3) array of unit-cube-normalised vertices the
    user imported. For 2D / 2.5D modes (7, 8) the z-coordinate is
    dropped before deduplicating and triangulating; mode 9 keeps the
    full 3D point cloud.
    """
    arr = np.asarray(verts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"points_from_mesh_vertices: verts must be (N, 3), got {arr.shape}")

    if mode in (7, 8):
        xy = arr[:, :2]
        rounded = np.round(xy, 9)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        pts = xy[np.sort(idx)]
        if len(pts) < 3:
            raise ValueError(
                f"mode {mode} requires at least 3 distinct XY vertices "
                f"after dedup; got {len(pts)}.")
        return pts, Delaunay(pts)

    if mode == 9:
        if len(arr) < 4:
            raise ValueError(
                f"mode 9 requires at least 4 distinct vertices for 3D "
                f"Delaunay; got {len(arr)}.")
        return arr, Delaunay(arr)

    raise ValueError(
        f"points_from_mesh_vertices only supports modes 7, 8, 9; got {mode}.")


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


def extrude_polygon_solid(verts_3d, ngon_thickness=None):
    if ngon_thickness is None:
        ngon_thickness = NGON_THICKNESS
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


def hub_scale_for_tcoh(hub_center, pts_list, hub_size_factor=None):
    if hub_size_factor is None:
        hub_size_factor = HUB_SIZE_FACTOR
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
        scale                         = hub_scale_for_tcoh(hub_center, pts_list)
        verts, oct_faces, hex_faces, sq_faces = make_truncated_cuboctahedron(hub_center, scale)
        all_triangles.extend(triangles_for_convex_solid(verts))
    else:
        pts_arr = np.array(pts_list)
        ordered = convex_order_3d(pts_arr)
        if ordered is not None:
            all_triangles.extend(extrude_polygon_solid(ordered))


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


def collect_export_geometry(points_nd, tri, ratio, mode, nz_layers):
    """Build the strut curves, solid triangles, and joint positions for STL/OBJ/SCAD output."""
    strut_curves  = []
    all_triangles = []
    joint_positions = set()

    def add_strut(p0, p1):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        if np.linalg.norm(p1 - p0) > 1e-9:
            strut_curves.append(np.array([p0, p1]))

    def register_joint(pt):
        key = tuple(np.round(pt, 8))
        joint_positions.add(key)

    if mode in [3, 6, 9]:
        pts_norm = points_nd
        groups   = build_3d_groups(pts_norm, tri, ratio)

        for simplex in tri.simplices:
            tet      = pts_norm[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = np.array([(1 - ratio) * tet[i] + ratio * centroid
                                  for i in range(4)])
            all_triangles.extend(triangles_for_solid_tetrahedron(shrunk))
            for pt in shrunk:
                register_joint(pt)

        for key, pts_list in groups.items():
            if len(pts_list) == 2:
                add_strut(pts_list[0], pts_list[1])
            elif len(pts_list) >= 3:
                dispatch_hub_export(key, pts_list, all_triangles)

    else:
        points_2d = points_nd
        layers    = [0] if mode in [1, 4, 7] else list(range(nz_layers))

        for simplex in tri.simplices:
            tri_pts = points_2d[simplex]
            c2d     = tri_pts.mean(axis=0)
            t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                 for i in range(3)])
            if mode in [1, 4, 7]:
                face3 = np.hstack([t2d, np.zeros((3, 1))])
                all_triangles.extend(extrude_polygon_solid(face3))
                for pt in face3:
                    register_joint(pt)
            elif mode in [2, 5, 8]:
                for z_idx in range(nz_layers - 1):
                    zb = z_idx / (nz_layers - 1)
                    zt = (z_idx + 1) / (nz_layers - 1)
                    bp = np.hstack([t2d, np.full((3, 1), zb)])
                    tp = np.hstack([t2d, np.full((3, 1), zt)])
                    all_triangles.extend(extrude_polygon_solid(tp))
                    for pt in bp: register_joint(pt)
                    for pt in tp: register_joint(pt)
                    for i in range(3):
                        j    = (i + 1) % 3
                        quad = np.array([bp[i], bp[j], tp[j], tp[i]])
                        all_triangles.extend(extrude_polygon_solid(quad))

        for z_idx in layers:
            z_val  = 0.0 if mode in [1, 4, 7] else z_idx / (nz_layers - 1)
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
                    if mode in [1, 4, 7]:
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
                        if mode in [1, 4, 7]:
                            ring = order_hub_ring_xy(verts, z_val)
                            v3d  = np.hstack([ring, np.full((len(ring), 1), z_val)])
                            all_triangles.extend(extrude_polygon_solid(v3d))
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
                                all_triangles.extend(extrude_polygon_solid(bv))
                    except Exception:
                        pass

    return strut_curves, all_triangles, joint_positions


def collect_export_geometry_from_posed_tiles(
    tiles: list,
    tile_source: list,
    dimension: int,
):
    """Mirror of ``collect_export_geometry`` that operates on
    pre-positioned tile vertex arrays instead of a (lattice,
    triangulation) pair.

    ``tiles[i]`` is the **pose-transformed** (world-frame) vertex array
    of tile ``i``; the caller is responsible for applying the per-tile
    pose before calling this function. ``tile_source[i]`` carries the
    metadata produced by ``collect_kirigami_tiles``: ``type`` plus the
    type-specific fields used to dispatch geometry construction
    (``vertex_keys`` for the simplex-derived tiles, used by the strut
    detector).

    Returns ``(strut_curves, all_triangles, joint_positions)`` in the
    same format as ``collect_export_geometry`` so the result feeds
    directly into ``build_export_triangles`` unchanged. Solid triangles
    per tile come from the same primitives the static export path uses
    (``extrude_polygon_solid``, ``triangles_for_solid_tetrahedron``,
    ``triangles_for_convex_solid``); struts are emitted between two
    pose-transformed corners that came from the same canonical lattice
    point under exactly two tiles (the boundary case where no hub is
    built); joint positions are the union of unique pose-transformed
    tile vertex positions, rounded to 8 decimals.
    """
    if len(tiles) != len(tile_source):
        raise ValueError(
            f"tiles ({len(tiles)}) and tile_source ({len(tile_source)}) "
            f"must have the same length"
        )

    # Forward-compatibility backstop: file-loaded TileSystems lack the
    # tile-type metadata needed for type dispatch. The check fires before
    # any geometry work so the failure mode is unambiguous.
    for src in tile_source:
        if src.get('type') == 'unknown':
            raise ValueError(
                "Pose rendering requires lattice-derived TileSystem; "
                "file-loaded systems lack tile-type metadata."
            )

    def _to_3d(arr):
        a = np.asarray(arr, dtype=float)
        if dimension == 2 and a.ndim == 2 and a.shape[1] == 2:
            return np.hstack([a, np.zeros((a.shape[0], 1))])
        return a

    posed_3d = [_to_3d(t) for t in tiles]

    strut_curves    = []
    all_triangles   = []
    joint_positions = set()

    # Solid geometry per tile, dispatched on the tile type.
    for tile_idx, src in enumerate(tile_source):
        type_   = src.get('type')
        verts3d = posed_3d[tile_idx]

        if type_ == 'tri_face' or type_ == 'hub_face':
            all_triangles.extend(extrude_polygon_solid(verts3d))
        elif type_ == 'tetrahedron':
            all_triangles.extend(triangles_for_solid_tetrahedron(verts3d))
        elif type_ == 'hub_polyhedron':
            all_triangles.extend(triangles_for_convex_solid(verts3d))

        for v in verts3d:
            joint_positions.add(tuple(np.round(v, 8)))

    # Strut detection: walk simplex-derived tiles, group their vertices
    # by canonical lattice-point key (carried in ``vertex_keys`` from
    # ``collect_kirigami_tiles``). A group of exactly 2 tile vertices
    # is a strut between those two pose-transformed positions —
    # the boundary case where no hub was built.
    groups: dict = {}
    for tile_idx, src in enumerate(tile_source):
        if src.get('type') not in ('tri_face', 'tetrahedron'):
            continue
        for vert_idx, key in enumerate(src.get('vertex_keys', [])):
            groups.setdefault(key, []).append((tile_idx, vert_idx))

    for occupants in groups.values():
        if len(occupants) == 2:
            (ta, va), (tb, vb) = occupants
            p0 = posed_3d[ta][va]
            p1 = posed_3d[tb][vb]
            if np.linalg.norm(p1 - p0) > 1e-9:
                strut_curves.append(np.array([p0, p1]))

    return strut_curves, all_triangles, joint_positions


def build_export_triangles(strut_curves, all_triangles, joint_positions,
                            strut_radius=None, scad_segments=None,
                            joint_sphere_radius=None,
                            joint_sphere_rings=None, joint_sphere_segments=None,
                            verbose=True):
    """Take the raw geometry and add strut tubes + joint spheres into a final triangle list."""
    if strut_radius is None:        strut_radius = STRUT_RADIUS
    if scad_segments is None:       scad_segments = SCAD_SEGMENTS
    if joint_sphere_radius is None: joint_sphere_radius = JOINT_SPHERE_RADIUS
    if joint_sphere_rings is None:  joint_sphere_rings = JOINT_SPHERE_RINGS
    if joint_sphere_segments is None: joint_sphere_segments = JOINT_SPHERE_SEGMENTS

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

    if joint_sphere_radius > 1e-12:
        n_spheres = 0
        for pt_key in joint_positions:
            center = np.array(pt_key, float)
            result += sphere_mesh(center, joint_sphere_radius,
                                  joint_sphere_rings, joint_sphere_segments)
            n_spheres += 1
        if verbose:
            print(f"  Joint spheres: {n_spheres} placed")

    return result
