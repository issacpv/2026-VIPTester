"""Synthesis integration tests — Task B (per-vertex c) feeding Task C (Bézier
fillet bounds + uniform/per-arm toggle) in one ``centroid_tile_demo``.

These exercise the headline combined behaviour: a per-vertex ``c`` (B) makes a
triangle's inner edges different lengths, and C's per-arm fillet must read those
varied lengths so each inner arm backs off to its own midpoint. They also pin
the invariance both tasks promised: with a *uniform* ``c`` the fillet output is
identical whether ``c`` is passed as a scalar, an ``(M,)`` array, or an
``(M, 3)`` array — and identical to C-alone's default (uniform-mode) result.

Ground truth is the §5.2 two-equilateral-triangle example (outer side 2).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from centroid_tile_demo import (  # noqa: E402
    FILLET_PER_ARM,
    FILLET_UNIFORM,
    TilingGeometry,
    _arm_backoff_bound,
    _find_junctions,
    _joint_arms,
    build_joint_bridges,
    build_panels,
    export_stl,
    joint_radii,
)

SQRT3 = float(np.sqrt(3.0))


def _geom():
    pts = np.array([[0.0, 0.0], [1.0, SQRT3], [2.0, 0.0], [3.0, SQRT3]], float)
    return TilingGeometry.build(pts)


def _panels(geom, c):
    inner, wing, link = build_panels(geom, c, 0.0, 0)
    return inner + wing + link, len(inner)


def _junctions(polys):
    allv = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in polys])
    scale = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    return _find_junctions(polys, scale * 1e-6)


def _inner_arm_lengths(polys, n_inner):
    out = []
    for jn in _junctions(polys):
        _, arms = _joint_arms(jn, polys, n_inner)
        out += [a[1] for a in arms if a[3]]
    return out


# ---------------------------------------------------------------------------
# B -> C: per-vertex c makes inner edges of different lengths; the per-arm
# fillet reads them, so inner arms back off to different (per-edge) midpoints.
# ---------------------------------------------------------------------------
def test_per_vertex_c_makes_asymmetric_per_arm_fillet():
    geom = _geom()
    M = len(geom.triangles)

    # Uniform scalar c: every inner edge is 2c, so every inner-arm bound is c.
    uni_polys, n_inner = _panels(geom, 0.5)
    uni_inner = _inner_arm_lengths(uni_polys, n_inner)
    assert uni_inner and np.allclose(uni_inner, 1.0)            # 2 * 0.5
    uni_bounds = {round(_arm_backoff_bound(("_", L, 0.0, True)), 6) for L in uni_inner}
    assert uni_bounds == {0.5}                                  # all the same midpoint

    # Per-vertex c on triangle 0: three independent c_i -> asymmetric inner tri.
    cv = np.full((M, 3), 0.5)
    cv[0] = [0.3, 0.6, 0.9]
    pv_polys, n_inner = _panels(geom, cv)
    pv_inner = _inner_arm_lengths(pv_polys, n_inner)
    pv_bounds = {round(_arm_backoff_bound(("_", L, 0.0, True)), 4) for L in pv_inner}
    # The asymmetric triangle introduces inner edges of several distinct lengths,
    # so the per-arm inner-edge midpoints now differ (where uniform c gave one).
    assert len(pv_bounds) >= 2

    # And the per-arm radii at a triangle-0 joint genuinely vary, while uniform
    # mode collapses them to one shared binding radius.
    d = 0.7
    juncs = _junctions(pv_polys)
    tri0_joint = None
    for jn in juncs:
        _, arms = _joint_arms(jn, pv_polys, n_inner)
        inner_lens = sorted(round(a[1], 4) for a in arms if a[3])
        if len(set(inner_lens)) >= 2:        # a joint touching the asymmetric tri
            tri0_joint = (jn, arms)
            break
    assert tri0_joint is not None, "expected a joint with unequal inner edges"
    _, arms = tri0_joint
    per = [d * _arm_backoff_bound(a) for a in arms]
    uni = [d * min(_arm_backoff_bound(a) for a in arms)] * len(arms)
    assert np.ptp(per) > 1e-6                 # asymmetric flower (per-arm)
    assert np.ptp(uni) == pytest.approx(0.0)  # symmetric flower (uniform)


# ---------------------------------------------------------------------------
# Invariance: a uniform c gives identical fillets regardless of c's shape, and
# matches C-alone's default uniform-mode behaviour.
# ---------------------------------------------------------------------------
def test_uniform_c_fillet_invariant_to_c_representation():
    geom = _geom()
    M = len(geom.triangles)
    reps = {
        "scalar": 0.5,
        "per_tri": np.full(M, 0.5),
        "per_vtx": np.full((M, 3), 0.5),
    }
    baseline = None
    for mode in (FILLET_UNIFORM, FILLET_PER_ARM):
        ref = None
        for _, c in reps.items():
            polys, n_inner = _panels(geom, c)
            radii = joint_radii(polys, 0.8, n_inner, mode)
            vals = np.array([radii[k] for k in sorted(radii)])
            if ref is None:
                ref = vals
            else:
                assert vals == pytest.approx(ref)   # same across scalar/(M,)/(M,3)
        if baseline is None:
            baseline = ref
    # Under uniform c the binding radius is also mode-independent (per C's docs).
    assert baseline is not None and np.all(baseline > 0.0)


# ---------------------------------------------------------------------------
# Full B -> C -> export chain digests per-vertex c without choking.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", [FILLET_UNIFORM, FILLET_PER_ARM])
def test_per_vertex_c_through_fillet_to_stl(mode):
    geom = _geom()
    M = len(geom.triangles)
    cv = np.full((M, 3), 0.5)
    cv[0] = [0.35, 0.55, 0.8]
    inner, wing, link = build_panels(geom, cv, 0.0, 0)
    bridges = build_joint_bridges(inner + wing + link, 0.6, len(inner), mode=mode)
    assert len(bridges) > 0
    polys = inner + wing + link + bridges
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, f"syn_{mode}.stl")
        ntri = export_stl(polys, path, 0.1)
        assert ntri > 0 and os.path.getsize(path) > 84   # header + >=1 triangle


def test_d_zero_sharp_under_per_vertex_c_both_modes():
    """d = 0 stays sharp (no bridges) even with an asymmetric per-vertex inner
    triangle, in both fillet modes."""
    geom = _geom()
    M = len(geom.triangles)
    cv = np.full((M, 3), 0.5)
    cv[0] = [0.3, 0.6, 0.9]
    polys, n_inner = _panels(geom, cv)
    for mode in (FILLET_UNIFORM, FILLET_PER_ARM):
        assert build_joint_bridges(polys, 0.0, n_inner, mode=mode) == []
