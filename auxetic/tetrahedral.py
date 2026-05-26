"""3D tetrahedral auxetic — the volumetric analogue of the 2D bipartite
auxetic in :mod:`auxetic.bipartite`.

Each tetrahedron of a 3D Delaunay triangulation independently emits:

- one **internal tetrahedron** ``I`` (set B), whose four vertices are the
  hinge points ``t_i = C·S + (1-C)·P_i`` contracted toward the
  tetrahedron centroid ``S`` (Commandino's theorem: ``S`` is the mean of
  the four corners); and
- four **corner polyhedra** (set A), one per corner ``P_i``: the convex
  solid spanning the corner ``P_i``, its hinge ``t_i``, three **canonical
  edge points** (one toward each incident edge's far corner) and three
  **canonical face points** (one toward each incident face's centroid).

This is the direct 3D lift of the bipartite kite. Where the 2D corner
kite is ``[P_i, foot_ij, T_i, foot_ik]`` — corner, two edge-feet, hinge —
the 3D corner polyhedron adds the third edge point *and* three face
points, because in 3D a vertex touches three edges and three faces.

**Cross-tetra fusion (what makes it a mechanism).** Each edge / face
point is defined purely from the *shared* geometry — an edge point is
``P_i + f·(P_j − P_i)`` (depends only on that edge's two endpoints), a
face point is ``P_i + f·(M_f − P_i)`` where ``M_f`` is the face centroid
(depends only on that face's three corners), with ``f = 0.5·C``. So two
tetrahedra that share an edge or a face compute the **identical** point
there, and their corner polyhedra meet at coincident vertices: a shared
corner plus a shared face point are two common points → a revolute
hinge between the neighbouring rigid pieces. That fusion is what turns
the pieces into a single coherent low-DOF auxetic mechanism the kinematic
simulator can solve. (The first MVP used per-tile *perpendicular* feet,
which depend on each tetra's own hinge and so could **not** fuse — the
pieces touched only at single points, i.e. 3D ball joints, leaving a
~96-DOF floppy null space with no meaningful mode.)

``C`` semantics follow the tetrahedron algorithm literally
(``t_i = C·S + (1-C)·P_i``) — note this is a *different* parameterisation
from the bipartite ``t = 1/(1+C)``:

- ``C → 0``: the internal tetra equals the full tetra and every corner /
  edge / face point collapses onto its corner — a solid tetrahedron.
- ``C → 1``: the internal tetra collapses toward the centroid and the
  corner polyhedra open out — a porous mesh.

Deferred to a follow-up (see ``Tetrahedron_algorithm.txt``): the exact
neighbour-centroid ``g``/``h`` intersection construction (here we use the
simpler shared-by-construction canonical points) and the quadratic-Bézier
corner smoothing. The rest tile is what is rendered; the kirigami
mechanism (relative rotation of the rigid pieces) is the same
rotating-units motion as the 2D case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TetraPolyhedron:
    """One rigid solid of a tetrahedral auxetic cell.

    ``node_xyz`` is the solid's anchor: the tetrahedron centroid for a
    set-B ``internal`` tetra, or the corner node for a set-A ``corner``
    polyhedron. ``vertices`` are the solid's corners (a convex point
    set — the renderer takes their convex hull). ``tetra_index`` is the
    index of the source tetrahedron so callers can tell which pieces came
    from the same cell; for a corner polyhedron, ``corner_point_index``
    is the index (into the lattice points) of the corner it is anchored
    at, else ``-1``.
    """

    node_xyz: np.ndarray          # shape (3,)
    set_label: str                # 'A' (corner polyhedron) or 'B' (internal)
    vertices: np.ndarray          # shape (k, 3), convex point set
    tetra_index: int = -1
    corner_point_index: int = -1

    @property
    def degree(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def kind(self) -> str:
        return "internal" if self.set_label == "B" else "corner"


@dataclass(frozen=True)
class TetrahedralNetwork:
    """All rigid solids of a tetrahedral auxetic plus the ``C`` used to
    build it. For each tetrahedron, one internal tetra (set B) followed
    by its four corner polyhedra (set A) — ``5·T`` solids total."""

    polyhedra: tuple[TetraPolyhedron, ...]
    C: float

    @property
    def set_a(self) -> tuple[TetraPolyhedron, ...]:
        """The corner polyhedra."""
        return tuple(p for p in self.polyhedra if p.set_label == "A")

    @property
    def set_b(self) -> tuple[TetraPolyhedron, ...]:
        """The internal tetrahedra."""
        return tuple(p for p in self.polyhedra if p.set_label == "B")


# Fraction of the corner→edge-other / corner→face-centroid distance at
# which the canonical hinge points sit, expressed relative to the
# contraction ``C`` (so the actual fraction is ``_HINGE_FRACTION · C``).
# This makes the points scale with C — at C = 0 they collapse onto the
# corner (solid tetra), matching the internal-tetra contraction.
_HINGE_FRACTION = 0.5


def build_tetrahedral_network(
    points: np.ndarray,
    simplices: np.ndarray,
    C: float = 0.5,
) -> TetrahedralNetwork:
    """Build the tetrahedral auxetic solids for a 3D triangulation.

    Parameters
    ----------
    points : (N, 3) array
        Corner positions in lattice space.
    simplices : (T, 4) int array
        Tetrahedron vertex indices into ``points``.
    C : float, default 0.5
        Contraction ratio (``t_i = C·S + (1-C)·P_i``); must lie in
        ``[0, 1]``. ``C = 0`` → solid tetra; ``C → 1`` → open mesh.

    Returns
    -------
    TetrahedralNetwork
        For each tetrahedron, one internal tetra (set B) followed by its
        four corner polyhedra (set A).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {pts.shape}")
    simps = np.asarray(simplices, dtype=np.int64)
    if simps.ndim != 2 or simps.shape[1] != 4:
        raise ValueError(f"simplices must be (T, 4); got {simps.shape}")
    Cf = float(C)
    if not (0.0 <= Cf <= 1.0):
        raise ValueError(f"C must be in [0, 1]; got {C}")

    polyhedra: list[TetraPolyhedron] = []

    for tet_idx in range(simps.shape[0]):
        idx = [int(v) for v in simps[tet_idx]]
        P = [pts[i] for i in idx]
        # Tetrahedron centroid (Commandino's theorem — mean of corners).
        S = (P[0] + P[1] + P[2] + P[3]) / 4.0
        # Internal-tetra hinge vertices, shared with the corner polyhedra.
        t = [Cf * S + (1.0 - Cf) * P[c] for c in range(4)]

        # Internal tetrahedron I (set B) — the four hinge vertices.
        polyhedra.append(TetraPolyhedron(
            node_xyz=S.copy(),
            set_label="B",
            vertices=np.array(t),
            tetra_index=tet_idx,
            corner_point_index=-1,
        ))

        # One corner polyhedron per corner (set A): the corner, its hinge,
        # three canonical edge points and three canonical face points.
        #
        # Each edge / face point is defined purely from the *shared*
        # geometry (the two edge endpoints, or the three face corners),
        # so two tetrahedra that share an edge or a face compute the
        # **same** point there. Their corner polyhedra therefore meet at
        # coincident vertices: the shared corner ``P_c`` plus a shared
        # face point give two common points → a revolute hinge; the
        # shared edge point lies on the edge with ``P_c`` → a hinge about
        # the edge axis. That cross-tetra fusion is what turns the
        # otherwise free-floating pieces into a coherent low-DOF auxetic
        # mechanism (the kinematic simulator rides those hinges). Per-tile
        # perpendicular feet — used in the first MVP — could *not* fuse,
        # because each tetra's feet depend on its own hinge, leaving only
        # ball-joint point contacts (a 96-DOF floppy mess).
        f = _HINGE_FRACTION * Cf
        for c in range(4):
            j, k, l = (o for o in range(4) if o != c)
            edge_pts = [P[c] + f * (P[o] - P[c]) for o in (j, k, l)]
            face_pts = []
            for o, p in ((j, k), (j, l), (k, l)):
                m_face = (P[c] + P[o] + P[p]) / 3.0
                face_pts.append(P[c] + f * (m_face - P[c]))
            verts = np.array([P[c], t[c], *edge_pts, *face_pts])
            polyhedra.append(TetraPolyhedron(
                node_xyz=P[c].copy(),
                set_label="A",
                vertices=verts,
                tetra_index=tet_idx,
                corner_point_index=idx[c],
            ))

    return TetrahedralNetwork(polyhedra=tuple(polyhedra), C=Cf)
