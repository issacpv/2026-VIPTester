"""Bipartite-polygon auxetic tile construction.

Inspired by the three-step recipe of *A three step recipe for designing
auxetic materials on demand* (Acuna, Gutierrez, Silva, Palza, Nunez,
During — Communications Physics **5**, 113, 2022,
doi:10.1038/s42005-022-00876-5), specialised here to the unit-cell
construction the studio uses: a 2D point cloud is triangulated, and
**each triangle independently emits four rigid polygons** —

- one **central** polygon (set B), the triangle of hinge points
  ``T_i = P_i + t·(M - P_i)`` around the centroid ``M``; and
- three **corner kites** (set A), one per corner ``P_i``, the quad
  ``[P_i, E_ij, T_i, E_ik]`` where ``E_ij``/``E_ik`` are points on the
  two triangle edges incident to ``P_i``.

The central polygon is the big triangle scaled by ``(1 - t)`` about the
centroid, so its faces are **parallel** to the big triangle's faces.

The constant size ratio ``C`` sets ``t = 1/(1+C)`` (paper step 3):

- ``t`` positions each centroid hinge along its ``corner -> centroid``
  segment (``C = 1`` -> midpoint); and
- each edge point ``E_ij`` is the **foot of the perpendicular dropped
  from the hinge ``T_i`` onto the triangle edge ``(P_i, P_j)``**. The
  kite's inner edge ``E_ij -> T_i`` is therefore perpendicular to that
  edge — and, because the central triangle's faces are parallel to the
  big triangle's, perpendicular to the corresponding central face too.
  This makes every hole between tiles a parallelogram (a rectangle in
  the symmetric case): the perfect-auxetic condition.

This reproduces the user's hand-derived tile ``(T_0, P_001, P_0,
P_002)``: ``P_0`` is the corner node, ``P_001``/``P_002`` the edge
points, ``T_0`` the centroid hinge shared with the central polygon.

Adjacent kites within a triangle are joined by **bonds** — the segments
along each triangle edge between the two perpendicular feet placed there
by the two corner kites (``BipartiteNetwork.bonds``). The central
polygon already connects to all three kites at the hinge points; the
bonds connect the kites to each other along the edges.

Fusion across triangles: two triangles sharing an edge place their feet
on that shared edge and share the corner nodes, so neighbouring tiles
meet edge-on.

``C`` semantics (matching the user's "C=1 mesh / C=0 solid" note):

- ``C -> 0`` (``t -> 1``): hinges collapse onto the centroid, kites grow
  to fill the whole triangle — a solid polygon.
- larger ``C`` (smaller ``t``): kites shrink toward the corners, opening
  up the holes between tiles — a mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BipartitePolygon:
    """One rigid polygon in an auxetic tile.

    ``node_xy`` is the polygon's anchor: the triangle centroid for a
    set-B ``central`` polygon, or the corner node for a set-A ``corner``
    kite. ``vertices`` are the polygon's corners in counter-clockwise
    order. ``triangle_index`` is the index of the source triangle (so
    callers can tell which tiles came from the same triangle); for a
    corner kite, ``corner_point_index`` is the index (into the lattice
    points) of the corner the kite is anchored at, else ``-1``.
    """

    node_xy: np.ndarray          # shape (2,)
    set_label: str               # 'A' (corner kite) or 'B' (central)
    vertices: np.ndarray         # shape (k, 2), CCW
    triangle_index: int = -1
    corner_point_index: int = -1
    hinge_index: int = -1        # index into ``vertices`` of the centroid
                                 # hinge T_i (kites only); -1 for central

    @property
    def degree(self) -> int:
        return int(self.vertices.shape[0])

    def inner_edges(self) -> list[np.ndarray]:
        """For a corner kite, the two edges incident to the hinge — the
        perpendicular inner edges that meet the central polygon. Each is
        a ``(2, 2)`` array ``[hinge_neighbour, hinge]``. Empty for the
        central polygon (no distinguished hinge)."""
        if self.hinge_index < 0:
            return []
        n = self.degree
        h = self.hinge_index
        prev_v = self.vertices[(h - 1) % n]
        next_v = self.vertices[(h + 1) % n]
        hv = self.vertices[h]
        return [np.array([prev_v, hv]), np.array([hv, next_v])]

    @property
    def kind(self) -> str:
        return "central" if self.set_label == "B" else "corner"


@dataclass(frozen=True)
class BipartiteNetwork:
    """All polygons of an auxetic tiling plus the ``C`` used to build it.

    ``bonds`` are the inter-kite connecting segments — one per triangle
    edge — each an ``(2, 2)`` array ``[[x0, y0], [x1, y1]]`` joining the
    two perpendicular feet placed on that edge by the edge's two corner
    kites.
    """

    polygons: tuple[BipartitePolygon, ...]
    C: float | np.ndarray          # scalar, or one value per triangle
    bonds: tuple[np.ndarray, ...] = ()

    @property
    def set_a(self) -> tuple[BipartitePolygon, ...]:
        """The corner kites."""
        return tuple(p for p in self.polygons if p.set_label == 'A')

    @property
    def set_b(self) -> tuple[BipartitePolygon, ...]:
        """The central polygons."""
        return tuple(p for p in self.polygons if p.set_label == 'B')

    @property
    def hinges(self) -> np.ndarray:
        """The unique centroid-hinge points (the ``T_i``) — the vertices
        of the central polygons, which are shared with the corner kites.
        Returns an ``(H, 2)`` array (deduplicated to 1e-9)."""
        if not self.set_b:
            return np.empty((0, 2), dtype=float)
        allv = np.vstack([p.vertices for p in self.set_b])
        _, idx = np.unique(np.round(allv, 9), axis=0, return_index=True)
        return allv[np.sort(idx)]


def _perp_foot(T: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Foot of the perpendicular from point ``T`` onto the line through
    ``A`` and ``B``. The segment ``foot -> T`` is perpendicular to
    ``A -> B`` by construction."""
    AB = B - A
    denom = float(np.dot(AB, AB))
    if denom < 1e-18:
        return A.copy()
    s = float(np.dot(T - A, AB)) / denom
    return A + s * AB


def _orient_ccw(verts: np.ndarray) -> np.ndarray:
    """Return ``verts`` reversed if its signed (shoelace) area is
    negative, so every polygon winds counter-clockwise."""
    v = np.asarray(verts, dtype=float)
    area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1)
                        - np.roll(v[:, 0], -1) * v[:, 1])
    return v if area >= 0.0 else v[::-1]


def build_bipartite_network(
    points: np.ndarray,
    simplices: np.ndarray,
    C: float | np.ndarray = 1.0,
    theta: float = 0.0,
    *,
    fuse_shared_feet: bool = True,
) -> BipartiteNetwork:
    """Build the auxetic tile network for a 2D triangulation.

    Parameters
    ----------
    points : (N, 2) array
        Corner positions in lattice space.
    simplices : (T, 3) int array
        Triangle vertex indices into ``points``.
    C : float or (T,) array, default 1.0
        Constant size ratio (paper step 3); ``t = 1/(1+C)``. Must be > 0.
        A scalar applies one ratio to every triangle; a length-``T`` array
        gives each triangle (``simplices[i]``) its own ratio, so composed
        tiles can carry independent ``C`` values.
    theta : float, default 0.0
        Mechanism actuation angle (radians). Each corner kite rotates
        **rigidly about its hinge** ``T_c`` by ``theta`` while the central
        polygon — whose vertices *are* the hinges — stays fixed. This is
        the rotating-units motion of the auxetic (notes p.2:
        ``P_{a,j} = R(θ)[P_a - (T_a - M)] + (T_a - M)``, i.e. rotation of
        ``P_a`` about ``T_a``). ``theta = 0`` is the rest tile.
    fuse_shared_feet : bool, default True
        When two triangles share an edge but their kite feet there don't
        coincide, the default fuses them to their midpoint *unless the two
        tiles carry different ``C``*. Fusing mismatched-``C`` neighbours
        would tilt the inner edges off-perpendicular into a locking right
        trapezoid, so each keeps its true perpendicular foot instead (the
        inter-tile hole stays a collapsible rectangle). Same-``C`` neighbours
        — including irregular and edge-flipped meshes — still fuse, exactly
        as before. Set ``False`` to force the perpendicular (no-fuse) method
        on every shared edge.

    Returns
    -------
    BipartiteNetwork
        For each triangle, one central polygon (set B) followed by its
        three corner kites (set A) — ``4·T`` polygons total.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2); got {pts.shape}")
    simps = np.asarray(simplices, dtype=np.int64)
    if simps.ndim != 2 or simps.shape[1] != 3:
        raise ValueError(f"simplices must be (T, 3); got {simps.shape}")
    # ``C`` may be a scalar (uniform size ratio) or a per-triangle array
    # — one value per simplex — so each tile can carry its own ratio. A
    # ``(T,)`` array is used as-is; a scalar broadcasts. Where adjacent
    # triangles disagree on ``C`` their shared perpendicular feet stop
    # coinciding and the foot-fusion step below snaps them to a midpoint
    # (the documented asymmetric-defect coupling), so a per-tile ``C`` is
    # a valid — if less symmetric — tiling.
    C_arr = np.broadcast_to(
        np.asarray(C, dtype=float), (simps.shape[0],)).astype(float)
    if not np.all(C_arr > 0.0):
        raise ValueError(f"C must be > 0; got {C}")

    # ``t`` positions each centroid hinge along its corner->centroid
    # segment; one per triangle so per-tile ``C`` gives per-tile hinges.
    t_arr = 1.0 / (1.0 + C_arr)
    cth, sth = np.cos(float(theta)), np.sin(float(theta))
    R = np.array([[cth, -sth], [sth, cth]])
    polygons: list[BipartitePolygon] = []
    bonds: list[np.ndarray] = []

    def _rotate_about(p, hinge):
        return R @ (np.asarray(p, dtype=float) - hinge) + hinge

    # ---- Pass 1: per-triangle centroid hinges + perpendicular feet -------
    tris: list[dict] = []
    for tri_idx in range(simps.shape[0]):
        idx = [int(v) for v in simps[tri_idx]]
        P = [pts[i] for i in idx]
        M = (P[0] + P[1] + P[2]) / 3.0
        # Centroid hinges, shared between the central polygon and kites.
        t = float(t_arr[tri_idx])
        T = [P[c] + t * (M - P[c]) for c in range(3)]
        # foot[c][o] is the foot of the perpendicular from hinge T[c] onto
        # the triangle edge (P[c], P[o]).
        foot = [[None, None, None] for _ in range(3)]
        for c in range(3):
            for o in range(3):
                if o != c:
                    foot[c][o] = _perp_foot(T[c], P[c], P[o])
        tris.append({
            "idx": idx,
            "g2l": {g: l for l, g in enumerate(idx)},
            "P": P, "M": M, "T": T, "foot": foot,
        })

    # ---- Fuse shared-edge feet (defect coupling) -------------------------
    #
    # Two triangles sharing an edge each drop a perpendicular foot near each
    # shared corner. When the adjacency is symmetric about the edge those two
    # feet COINCIDE and the kites fuse there (a shared vertex) — that fusion
    # is what couples neighbours into a single rigid mechanism (see
    # ``test_shared_corner_kites_fuse_on_the_diagonal``). When the adjacency
    # is NOT symmetric (e.g. after an edge flip, or an irregular mesh) the two
    # feet land at different points, the fusion is lost, and the shared corner
    # alone leaves a relative-rotation DOF — the neighbouring region floats
    # free of the mechanism and overlaps under actuation. Snap both feet to
    # their midpoint so they coincide again: the two triangles' edge bonds
    # rejoin into one connection (instead of sitting offset) and the
    # single-DOF coupling is restored, with no extra struts. A foot-to-foot
    # bridge bond would be collinear with the shared edge and so can't resist
    # rotation — fusing the vertices is the coupling that works. Edges whose
    # feet already coincide (symmetric tilings) and boundary edges are
    # untouched.
    edge_tris: dict[tuple[int, int], list[int]] = {}
    for r_idx, rec in enumerate(tris):
        g = rec["idx"]
        for a in range(3):
            for b in range(a + 1, 3):
                e = (g[a], g[b]) if g[a] < g[b] else (g[b], g[a])
                edge_tris.setdefault(e, []).append(r_idx)
    for (gi, gj), recs in edge_tris.items():
        if len(recs) != 2:
            continue  # boundary edge — no neighbour to fuse with
        r1, r2 = tris[recs[0]], tris[recs[1]]
        for gc, go in ((gi, gj), (gj, gi)):   # foot near each shared corner
            l1c, l1o = r1["g2l"][gc], r1["g2l"][go]
            l2c, l2o = r2["g2l"][gc], r2["g2l"][go]
            f1, f2 = r1["foot"][l1c][l1o], r2["foot"][l2c][l2o]
            if np.allclose(f1, f2, atol=1e-9):
                continue                # symmetric — already coincident
            if not fuse_shared_feet:
                continue                # caller forced the perpendicular method
            # Per-tile C MISMATCH is the reported case: when the two tiles
            # carry different C their kites meet the shared edge at different
            # perpendicular heights, so snapping both feet to their midpoint
            # tilts each kite's inner edge off-perpendicular and turns the
            # inter-tile hole into a NON-collapsible right trapezoid (the
            # magenta-trapezoid lock). Skip the snap there and keep each
            # tile's TRUE perpendicular foot, so its holes stay rectangular
            # and the lattice can still rotate; the tiles remain coupled
            # through the shared corner node. Same-C neighbours (incl.
            # irregular / edge-flipped meshes) keep the midpoint fusion that
            # couples them into one mechanism — unchanged from before.
            if abs(float(C_arr[recs[0]]) - float(C_arr[recs[1]])) > 1e-12:
                continue                # mismatched C → perpendicular method
            mid = 0.5 * (f1 + f2)
            r1["foot"][l1c][l1o] = mid
            r2["foot"][l2c][l2o] = mid

    # ---- Pass 2: build central polygons, corner kites, and edge bonds ----
    for tri_idx, rec in enumerate(tris):
        idx, P, M, T, foot = (rec["idx"], rec["P"], rec["M"],
                              rec["T"], rec["foot"])

        # Rigid rotation of a point about hinge T[c] by theta. At theta = 0
        # this is the identity, so the rest tile is unchanged.
        def about(p, c, _T=T):
            return _rotate_about(p, _T[c])

        # The central polygon is fixed (its vertices are the hinges).
        polygons.append(BipartitePolygon(
            node_xy=M.copy(),
            set_label='B',
            vertices=_orient_ccw(np.array(T)),
            triangle_index=tri_idx,
            corner_point_index=-1,
        ))

        # One kite per corner, rotated rigidly about its hinge.
        for c in range(3):
            j, k = (c + 1) % 3, (c + 2) % 3
            kite = _orient_ccw(np.array([
                about(P[c], c), about(foot[c][j], c), T[c], about(foot[c][k], c)
            ]))
            hinge_index = int(np.argmin(
                np.linalg.norm(kite - T[c], axis=1)))
            polygons.append(BipartitePolygon(
                node_xy=P[c].copy(),
                set_label='A',
                vertices=kite,
                triangle_index=tri_idx,
                corner_point_index=idx[c],
                hinge_index=hinge_index,
            ))

        # One bond per triangle edge: the segment between the two feet
        # (each carried along by its own kite's rotation).
        for a, b in ((0, 1), (1, 2), (2, 0)):
            bonds.append(np.array([about(foot[a][b], a), about(foot[b][a], b)]))

    # Preserve the caller's intent on the network: a scalar stays a
    # float; a per-triangle array is stored verbatim.
    C_store = float(C) if np.ndim(C) == 0 else np.asarray(C, dtype=float)
    return BipartiteNetwork(polygons=tuple(polygons), C=C_store,
                            bonds=tuple(bonds))


def jamming_angle(
    points: np.ndarray,
    simplices: np.ndarray,
    C: float | np.ndarray = 1.0,
) -> float:
    """Largest ``|theta|`` a kite can rotate about its hinge before an
    inner edge collides with the adjacent central-polygon face — the
    notes' jamming angle (p.2).

    For each corner ``c`` and each adjacent corner ``o`` the kite's inner
    edge ``T_c -> foot[c][o]`` swings toward the central edge
    ``T_c -> T_o``; the angle between them is how far it can turn before
    contact. The whole tiling jams at the minimum such angle over every
    corner, so that minimum (radians) is returned. Empty/degenerate input
    yields ``pi/2``.
    """
    pts = np.asarray(points, dtype=float)
    simps = np.asarray(simplices, dtype=np.int64)
    if (pts.ndim != 2 or pts.shape[1] != 2
            or simps.ndim != 2 or simps.shape[1] != 3):
        return float(np.pi / 2.0)
    # ``C`` is scalar or per-triangle (see ``build_bipartite_network``).
    try:
        C_arr = np.broadcast_to(
            np.asarray(C, dtype=float), (simps.shape[0],)).astype(float)
    except ValueError:
        return float(np.pi / 2.0)
    if simps.shape[0] == 0 or not np.all(C_arr > 0.0):
        return float(np.pi / 2.0)

    t_arr = 1.0 / (1.0 + C_arr)
    min_ang = float(np.pi / 2.0)
    for ti, tri in enumerate(simps):
        P = [pts[int(v)] for v in tri]
        M = (P[0] + P[1] + P[2]) / 3.0
        t = float(t_arr[ti])
        T = [P[c] + t * (M - P[c]) for c in range(3)]
        for c in range(3):
            for o in range(3):
                if o == c:
                    continue
                u = _perp_foot(T[c], P[c], P[o]) - T[c]
                v = T[o] - T[c]
                nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
                if nu < 1e-12 or nv < 1e-12:
                    continue
                ang = float(np.arccos(np.clip(np.dot(u, v) / (nu * nv),
                                              -1.0, 1.0)))
                min_ang = min(min_ang, ang)
    return min_ang
