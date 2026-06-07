"""Task B — per-vertex shrink ``c`` in ``centroid_tile_demo``.

Covers, bottom-up:

- ``compute_T`` accepting a scalar, a ``(M,)`` per-triangle array, and a
  ``(M, 3)`` per-vertex array, with the scalar / ``(M,)`` fast paths giving
  results identical to the per-triangle model;
- the per-vertex geometry ``T_i = C + c_i (P_i - C)``, the c_i=0 / c_i=1
  endpoints, and the breaking of the uniform-scaling identity
  ``T_i - T_j = c (P_i - P_j)`` once the three c_i differ;
- a numeric spot-check against ``CENTROID_TILE_SPEC.md`` §5.2;
- that every downstream consumer (feet, wings, links, fillet bridges, STL
  export) digests a ``(M, 3)`` c without choking;
- the ctrl-press pick-vs-drag disambiguation ``_classify_ctrl_press``,
  factored out of the GUI so it runs headlessly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import matplotlib                       # set a non-interactive backend BEFORE
if matplotlib.get_backend().lower() != "agg":   # centroid_tile_demo imports
    try:                                        # pyplot, so importing never
        matplotlib.use("Agg")                   # tries to open a window.
    except Exception:
        pass
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import centroid_tile_demo as demo
from centroid_tile_demo import (
    TilingGeometry,
    compute_T,
    compute_feet,
    _classify_ctrl_press,
    _PICK_RADIUS_PX,
)

SQRT3 = float(np.sqrt(3.0))

# Two edge-adjacent equilateral triangles (CENTROID_TILE_SPEC.md §5.2).
_TWO_TRI_PTS = np.array(
    [[0.0, 0.0], [1.0, SQRT3], [2.0, 0.0], [3.0, SQRT3]], dtype=float
)

# A single equilateral triangle (side 2): incenter == centroid exactly.
_EQ_TRI_PTS = np.array([[0.0, 0.0], [1.0, SQRT3], [2.0, 0.0]], dtype=float)

# A denser fan so M > 2 exercises the per-vertex broadcast over many triangles.
_FAN_PTS = np.array(
    [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0],
     [1.0, 1.6], [3.0, 1.6], [2.0, 3.0]], dtype=float
)


def _geom(pts):
    return TilingGeometry.build(pts, anchor="incenter")


# ---------------------------------------------------------------------------
# compute_T: shapes and fast-path equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pts", [_TWO_TRI_PTS, _FAN_PTS, _EQ_TRI_PTS])
def test_scalar_uniform_array_and_uniform_per_vertex_agree(pts):
    geom = _geom(pts)
    M = len(geom.triangles)
    Ts = compute_T(geom, 0.4)
    Tm = compute_T(geom, np.full(M, 0.4))
    Tv = compute_T(geom, np.full((M, 3), 0.4))
    assert Ts.shape == (M, 3, 2)
    assert np.allclose(Ts, Tm)          # scalar == (M,) (acceptance: identical)
    assert np.allclose(Ts, Tv)          # uniform (M,3) collapses to the same


def test_per_triangle_array_unchanged_from_main():
    # A genuinely per-triangle (M,) c must still take the old code path: row m
    # of T is C_m + c[m] * v_m, every vertex sharing c[m].
    geom = _geom(_TWO_TRI_PTS)
    M = len(geom.triangles)
    c = np.linspace(0.2, 0.7, M)
    T = compute_T(geom, c)
    for m in range(M):
        assert np.allclose(T[m], geom.C[m] + c[m] * geom.v[m])


def test_per_vertex_formula_matches_definition():
    geom = _geom(_FAN_PTS)
    M = len(geom.triangles)
    rng = np.random.default_rng(0)
    c = rng.uniform(0.1, 0.9, size=(M, 3))
    T = compute_T(geom, c)
    assert T.shape == (M, 3, 2)
    # Vectorized broadcast result...
    assert np.allclose(T, geom.C[:, None, :] + c[:, :, None] * geom.v)
    # ...and the per-(triangle, vertex) geometric definition, element by element.
    for m in range(M):
        for i in range(3):
            expect = geom.C[m] + c[m, i] * (geom.P[m, i] - geom.C[m])
            assert np.allclose(T[m, i], expect)


def test_per_vertex_endpoints_zero_and_one():
    geom = _geom(_FAN_PTS)
    M = len(geom.triangles)
    # vertex 0 -> anchor (c=0), vertex 1 -> corner (c=1), vertex 2 -> halfway.
    c = np.tile(np.array([0.0, 1.0, 0.5]), (M, 1))
    T = compute_T(geom, c)
    assert np.allclose(T[:, 0], geom.C)            # c_i = 0 collapses to C
    assert np.allclose(T[:, 1], geom.P[:, 1])      # c_i = 1 lands on P_i
    assert np.allclose(T[:, 2], 0.5 * (geom.C + geom.P[:, 2]))


def test_per_vertex_breaks_uniform_scaling_identity():
    geom = _geom(_TWO_TRI_PTS)
    M = len(geom.triangles)
    # Uniform per-vertex c: the identity T_i - T_j = c (P_i - P_j) holds.
    Tu = compute_T(geom, np.full((M, 3), 0.5))
    for m in range(M):
        assert np.allclose(Tu[m, 0] - Tu[m, 1], 0.5 * (geom.P[m, 0] - geom.P[m, 1]))
    # Distinct c_i: the identity must break for at least one edge.
    Tm = compute_T(geom, np.tile(np.array([0.3, 0.6, 0.9]), (M, 1)))
    broke = any(
        not np.allclose(Tm[m, 0] - Tm[m, 1], 0.3 * (geom.P[m, 0] - geom.P[m, 1]))
        for m in range(M)
    )
    assert broke


def test_per_vertex_change_moves_only_target_vertex():
    # Acceptance: editing one vertex's c moves only that inner vertex.
    geom = _geom(_FAN_PTS)
    M = len(geom.triangles)
    base = np.full((M, 3), 0.5)
    T0 = compute_T(geom, base)
    mod = base.copy()
    mod[0, 1] = 0.15                       # change triangle 0, local vertex 1
    T1 = compute_T(geom, mod)
    changed = ~np.all(np.isclose(T1, T0), axis=-1)     # (M, 3) bool
    assert changed.sum() == 1 and changed[0, 1]


@pytest.mark.parametrize("bad", [
    np.zeros((1, 4)),          # wrong vertex count (must be 3)
    np.zeros((99, 3)),         # wrong triangle count
    np.zeros((2, 3, 2)),       # 3-D c is unsupported
])
def test_compute_T_rejects_bad_per_vertex_shapes(bad):
    geom = _geom(_TWO_TRI_PTS)
    with pytest.raises(ValueError):
        compute_T(geom, bad)


def test_spec_section_5_2_anchor_numbers():
    # Single equilateral triangle: incenter == centroid == (1, sqrt(3)/3).
    geom = _geom(_EQ_TRI_PTS)
    assert len(geom.triangles) == 1
    C = geom.C[0]
    assert np.allclose(C, [1.0, SQRT3 / 3.0])

    # scipy may permute the local vertex order; locate the corner that is P1.
    lv_p1 = next(li for li in range(3)
                 if np.allclose(geom.P[0, li], [1.0, SQRT3]))

    # c1A = 1 -> T1A == P1; c1A = 0 -> T1A == C (spec §5.2 self-checks).
    c = np.zeros((1, 3)); c[0, lv_p1] = 1.0
    assert np.allclose(compute_T(geom, c)[0, lv_p1], [1.0, SQRT3])
    c[0, lv_p1] = 0.0
    assert np.allclose(compute_T(geom, c)[0, lv_p1], C)

    # c1A = 0.5 -> T1A == (1, 2*sqrt(3)/3) == (1, 1.154701).
    c[0, lv_p1] = 0.5
    assert np.allclose(compute_T(geom, c)[0, lv_p1], [1.0, 2.0 * SQRT3 / 3.0])


# ---------------------------------------------------------------------------
# Downstream consumers digest (M, 3) c
# ---------------------------------------------------------------------------

def test_feet_wings_links_fillets_stl_accept_per_vertex_c():
    geom = _geom(_FAN_PTS)
    M = len(geom.triangles)
    c = np.linspace(0.3, 0.8, M * 3).reshape(M, 3)

    feet = compute_feet(geom, compute_T(geom, c))
    assert feet.shape == (M, 6, 2) and np.all(np.isfinite(feet))

    for theta in (0.0, np.radians(25.0)):
        T, wings = demo.tile_state(geom, c, theta, 0)
        assert T.shape == (M, 3, 2) and np.all(np.isfinite(T))
        assert wings.shape == (M, 3, 3, 2) and np.all(np.isfinite(wings))
        links = demo.compute_links(geom, c, theta, 0)
        assert all(np.all(np.isfinite(p)) for p in links)

    inner, wing, link = demo.build_panels(geom, c, 0.0, 0)
    bridges = demo.build_joint_bridges(inner + wing + link, 0.5, len(inner))
    assert all(np.all(np.isfinite(b)) for b in bridges)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.stl")
        ntri = demo.export_stl(inner + wing + link + bridges, path, 0.1)
        assert ntri > 0
        assert os.path.getsize(path) > 84      # header(80) + count(4)


def test_per_vertex_panels_differ_from_uniform():
    # A real per-vertex override must actually change the rendered inner
    # triangle relative to the uniform-c baseline.
    geom = _geom(_TWO_TRI_PTS)
    M = len(geom.triangles)
    base = demo.compute_T(geom, np.full((M, 3), 0.5))
    c = np.full((M, 3), 0.5)
    c[0, 0] = 0.9
    mod = demo.compute_T(geom, c)
    assert not np.allclose(base, mod)
    # Only triangle 0 changed.
    assert np.allclose(base[1:], mod[1:])


# ---------------------------------------------------------------------------
# Pick-vs-drag disambiguation (_classify_ctrl_press), headless
# ---------------------------------------------------------------------------

# A triangle whose corners sit at known pixel positions, plus an extra,
# non-selected draggable point far away.
_SEL = np.array([[100.0, 100.0], [200.0, 100.0], [150.0, 200.0]])
_ALL = np.vstack([_SEL, [[400.0, 400.0]]])      # corners 0..2, far point = 3


def test_pick_returns_nearest_selected_vertex():
    kind, idx = _classify_ctrl_press((152.0, 203.0), _SEL, _ALL, _PICK_RADIUS_PX)
    assert (kind, idx) == ("vertex", 2)


def test_pick_takes_priority_over_a_closer_nonselected_point():
    # Put a non-selected point right under the cursor; the click is still
    # within threshold of selected vertex 2, so the vertex pick must win.
    allpts = np.vstack([_SEL, [[151.0, 201.0]]])
    click = (151.5, 201.5)          # ~0.7 px from allpts[3], ~2.1 px from vtx 2
    kind, idx = _classify_ctrl_press(click, _SEL, allpts, _PICK_RADIUS_PX)
    assert (kind, idx) == ("vertex", 2)


def test_far_from_selected_vertices_falls_through_to_drag():
    kind, idx = _classify_ctrl_press((402.0, 399.0), _SEL, _ALL, _PICK_RADIUS_PX)
    assert (kind, idx) == ("drag", 3)


def test_drag_still_works_on_a_selected_triangles_neighbourhood_point():
    # A draggable point that is NOT a selected-triangle corner stays draggable
    # even when a triangle is selected.
    allpts = np.vstack([_SEL, [[260.0, 100.0]]])
    kind, idx = _classify_ctrl_press((261.0, 101.0), _SEL, allpts, _PICK_RADIUS_PX)
    assert (kind, idx) == ("drag", 3)


def test_no_selection_never_picks_a_vertex():
    kind, idx = _classify_ctrl_press((101.0, 101.0), None, _ALL, _PICK_RADIUS_PX)
    assert (kind, idx) == ("drag", 0)


def test_nothing_within_threshold_returns_none():
    kind, idx = _classify_ctrl_press((1000.0, 1000.0), _SEL, _ALL, _PICK_RADIUS_PX)
    assert (kind, idx) == (None, -1)


def test_threshold_boundary_is_inclusive():
    # Exactly threshold px from selected vertex 0 -> still a hit.
    click = (_SEL[0, 0] + _PICK_RADIUS_PX, _SEL[0, 1])
    kind, idx = _classify_ctrl_press(click, _SEL, _SEL, _PICK_RADIUS_PX)
    assert (kind, idx) == ("vertex", 0)
    # Just beyond -> no vertex; and no other point near -> None.
    click = (_SEL[0, 0] + _PICK_RADIUS_PX + 0.5, _SEL[0, 1])
    kind, idx = _classify_ctrl_press(click, None, np.empty((0, 2)), _PICK_RADIUS_PX)
    assert (kind, idx) == (None, -1)
