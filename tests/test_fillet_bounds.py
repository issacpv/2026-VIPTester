"""Task C — Bézier fillet bounds + uniform/per-arm toggle.

Pure-geometry tests (no GUI) for the new joint-fillet model in
``centroid_tile_demo``:

* the new per-arm back-off **bounds** — an inner-triangle-edge arm backs off by
  half its length (the inner-edge midpoint), a leg arm by its full length
  (replacing the old 0.25·inner / 0.5·leg uniform bounds);
* the **uniform** vs **per-arm** application modes and the back-off distances
  each produces at a known joint;
* that the new uniform fillet reaches strictly **further** than the old one;
* that ``d = 0`` still yields a **sharp** join in both modes;
* that ``joint_radii`` / the "radius vs c" read-out stay well defined (the
  binding arm radius) under both modes.

Ground truth: the two-equilateral-triangle example of ``CENTROID_TILE_SPEC.md``
§5.2 — P0=(0,0), P1=(1,√3), P2=(2,0), P3=(3,√3) — outer side 2, so at shrink
``c`` every inner-triangle edge has length ``2c`` and at ``c = 0.5`` the joints
carry inner arms of length 1.0 and leg arms of length √3/6 and √3/3.
"""

from __future__ import annotations

import os

# Force a non-interactive backend before importing the demo (which imports
# pyplot at module load) so the suite stays headless.
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
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
    _arm_radii,
    _build_joint_bridge,
    _find_junctions,
    _joint_arms,
    _joint_radius_from_arms,
    build_joint_bridges,
    build_panels,
    joint_radii,
)

SQRT3 = float(np.sqrt(3.0))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _two_triangle_panels(c: float = 0.5):
    """Panels (inner + wing + link) for the §5.2 two-triangle example."""
    pts = np.array([[0.0, 0.0], [1.0, SQRT3], [2.0, 0.0], [3.0, SQRT3]], float)
    geom = TilingGeometry.build(pts)
    inner, wing, link = build_panels(geom, c, 0.0, 0)
    return inner + wing + link, len(inner)


def _junctions(polys):
    allv = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in polys])
    scale = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    return _find_junctions(polys, scale * 1e-6)


def _nearest_junction(juncs, polys, n_inner, target):
    """The junction whose centre is closest to ``target`` (a 2-vector)."""
    target = np.asarray(target, float)
    best, best_d = None, np.inf
    for jn in juncs:
        center, _ = _joint_arms(jn, polys, n_inner)
        d = float(np.linalg.norm(center - target))
        if d < best_d:
            best_d, best = d, jn
    return best


def _old_uniform_radius(arms, d: float) -> float:
    """The PRE-Task-C uniform radius: d·min(0.5·shortest arm, 0.25·shortest inner)."""
    half_leg = 0.5 * min(a[1] for a in arms)
    inner = [a[1] for a in arms if a[3]]
    quarter_inner = 0.25 * min(inner) if inner else float("inf")
    return d * min(half_leg, quarter_inner)


# ---------------------------------------------------------------------------
# New bound VALUES
# ---------------------------------------------------------------------------
def test_bound_fractions_tt_edge_half_leg_full():
    """Per-arm bound is 0.5·len for any T-to-T edge (inner edge OR link
    cross-arm) and 1.0·len for a leg. The two-triangle tiling exercises all
    three arm kinds."""
    polys, n_inner = _two_triangle_panels(c=0.5)
    saw_inner = saw_cross = saw_leg = False
    for jn in _junctions(polys):
        _, arms = _joint_arms(jn, polys, n_inner)
        for a in arms:
            leg_len, is_inner, tt = a[1], a[3], a[4]
            if tt:
                assert _arm_backoff_bound(a) == pytest.approx(0.5 * leg_len)
            else:
                assert _arm_backoff_bound(a) == pytest.approx(1.0 * leg_len)
            if is_inner:
                saw_inner = True            # inner-triangle edge (tt, same tri)
            elif tt:
                saw_cross = True            # link cross-arm (tt, neighbouring tris)
            else:
                saw_leg = True              # perpendicular wing leg (T -> foot)
    assert saw_inner and saw_cross and saw_leg, (
        "expected inner edges, link cross-arms, and legs in the tiling")


def test_inner_bound_is_inner_triangle_midpoint():
    """At shrink c the inner-triangle edge has length 2c (outer side 2), and the
    inner-arm bound is half of it — the midpoint."""
    c = 0.5
    polys, n_inner = _two_triangle_panels(c)
    inner_arms = [
        a
        for jn in _junctions(polys)
        for a in _joint_arms(jn, polys, n_inner)[1]
        if a[3]
    ]
    assert inner_arms, "no inner-triangle-edge arms found"
    for a in inner_arms:
        assert a[1] == pytest.approx(2.0 * c)              # inner edge = 2c
        assert _arm_backoff_bound(a) == pytest.approx(c)   # midpoint = 0.5·2c


# ---------------------------------------------------------------------------
# Uniform radius formula
# ---------------------------------------------------------------------------
def test_uniform_radius_is_min_arm_bound():
    """Uniform radius = d · min over arms of each arm's own bound = d · min(
    0.5·shortest T-to-T edge, 1.0·shortest leg)."""
    d = 1.0
    polys, n_inner = _two_triangle_panels(c=0.5)
    for jn in _junctions(polys):
        _, arms = _joint_arms(jn, polys, n_inner)
        # bound: 0.5·len for a T-to-T edge (a[4]); 1.0·len for a leg.
        tt = [a[1] for a in arms if a[4]]
        legs = [a[1] for a in arms if not a[4]]
        expected = d * min(
            0.5 * min(tt) if tt else float("inf"),
            1.0 * min(legs) if legs else float("inf"),
        )
        got = _joint_radius_from_arms(arms, d, FILLET_UNIFORM)
        assert got == pytest.approx(expected)
        assert got == pytest.approx(d * min(_arm_backoff_bound(a) for a in arms))


def test_new_uniform_reaches_further_than_old():
    """Acceptance: with the new bounds the (binding) fillet reaches strictly
    further than the old 0.5·leg / 0.25·inner formula — and never less at any
    joint."""
    d = 1.0
    polys, n_inner = _two_triangle_panels(c=0.5)
    news, olds = [], []
    for jn in _junctions(polys):
        _, arms = _joint_arms(jn, polys, n_inner)
        new = _joint_radius_from_arms(arms, d, FILLET_UNIFORM)
        old = _old_uniform_radius(arms, d)
        assert new >= old - 1e-12          # never smaller
        news.append(new)
        olds.append(old)
    assert min(news) > min(olds)           # binding fillet reaches further


# ---------------------------------------------------------------------------
# Uniform vs per-arm back-off distances at a known joint
# ---------------------------------------------------------------------------
def test_backoff_distances_uniform_vs_per_arm_known_joint():
    """At the joint centred on T2 of triangle A — (1.5, √3/6) — uniform backs
    every arm off by the same binding radius; per-arm backs each off by its own
    bound (T-to-T edge → 0.5·len, leg → 1.0·len). The non-inner arms here are one
    perpendicular leg (√3/6) and one link cross-arm (√3/3); the cross-arm is now
    capped at its midpoint, not its full length."""
    d = 0.8
    polys, n_inner = _two_triangle_panels(c=0.5)
    juncs = _junctions(polys)
    jn = _nearest_junction(juncs, polys, n_inner, (1.5, SQRT3 / 6.0))
    _, arms = _joint_arms(jn, polys, n_inner)

    leg_lens = sorted(a[1] for a in arms if not a[3])      # the two non-inner arms
    inner_lens = sorted(a[1] for a in arms if a[3])
    assert leg_lens == pytest.approx([SQRT3 / 6.0, SQRT3 / 3.0])
    assert inner_lens == pytest.approx([1.0, 1.0])
    # Of the two non-inner arms, the longer (√3/3) is a link cross-arm (tt), the
    # shorter (√3/6) a perpendicular wing leg (not tt).
    cross = [a for a in arms if a[4] and not a[3]]
    perp = [a for a in arms if not a[4]]
    assert len(cross) == 1 and cross[0][1] == pytest.approx(SQRT3 / 3.0)
    assert len(perp) == 1 and perp[0][1] == pytest.approx(SQRT3 / 6.0)

    # Uniform: one shared radius = d · (binding bound) = d · √3/6.
    uni = _arm_radii(arms, d, FILLET_UNIFORM)
    assert uni == pytest.approx([d * SQRT3 / 6.0] * len(arms))
    assert np.ptp(uni) == pytest.approx(0.0)               # symmetric flower

    # Per-arm: each arm reaches d · its own bound (0.5·len if T-to-T else 1.0·len).
    per = _arm_radii(arms, d, FILLET_PER_ARM)
    for a, r in zip(arms, per):
        bound = (0.5 if a[4] else 1.0) * a[1]
        assert r == pytest.approx(d * bound)
    assert np.ptp(per) > 1e-6                              # asymmetric flower
    # The cross-arm backs off to its MIDPOINT (d·0.5·√3/3 = d·√3/6), NOT its full
    # length (d·√3/3) — that is the overlap fix.
    cross_r = next(r for a, r in zip(arms, per) if a[4] and not a[3])
    assert cross_r == pytest.approx(d * 0.5 * SQRT3 / 3.0)
    assert cross_r != pytest.approx(d * SQRT3 / 3.0)
    # inner edges back off furthest (d·0.5·1.0); the binding is d·√3/6.
    assert max(per) == pytest.approx(d * 0.5)
    assert min(per) == pytest.approx(d * SQRT3 / 6.0)


def test_per_arm_flower_differs_from_uniform_but_d0_matches():
    """The actual Bézier 'flower' geometry differs between modes for d>0, but
    both collapse identically (to nothing) at d=0."""
    polys, n_inner = _two_triangle_panels(c=0.5)
    jn = _junctions(polys)[0]
    uni = _build_joint_bridge(jn, polys, 0.7, n_inner=n_inner, mode=FILLET_UNIFORM)
    per = _build_joint_bridge(jn, polys, 0.7, n_inner=n_inner, mode=FILLET_PER_ARM)
    assert uni is not None and per is not None
    assert uni.shape == per.shape          # same arms/samples
    assert not np.allclose(uni, per)       # but different back-off radii


# ---------------------------------------------------------------------------
# d = 0 sharpness, both modes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", [FILLET_UNIFORM, FILLET_PER_ARM])
def test_d_zero_is_sharp_join(mode):
    polys, n_inner = _two_triangle_panels(c=0.5)
    # No bridges built at d = 0 (sharp single-point joins).
    assert build_joint_bridges(polys, 0.0, n_inner, mode=mode) == []
    # The single-joint builder also returns None at d = 0.
    for jn in _junctions(polys):
        assert _build_joint_bridge(jn, polys, 0.0, n_inner=n_inner, mode=mode) is None
    # A positive d does produce bridges (sanity: the toggle isn't dead).
    assert len(build_joint_bridges(polys, 0.5, n_inner, mode=mode)) > 0


# ---------------------------------------------------------------------------
# joint_radii / "radius vs c" read-out under both modes
# ---------------------------------------------------------------------------
def test_joint_radii_binding_value_consistent_and_linear_in_d():
    """Both modes report the SAME binding (min) arm radius per joint (documented),
    discover the same joints, and scale linearly with d."""
    polys, n_inner = _two_triangle_panels(c=0.5)
    r_uni = joint_radii(polys, 1.0, n_inner, FILLET_UNIFORM)
    r_per = joint_radii(polys, 1.0, n_inner, FILLET_PER_ARM)
    assert set(r_uni) == set(r_per) and len(r_uni) > 0
    for jid in r_uni:
        assert r_uni[jid] == pytest.approx(r_per[jid])     # binding == uniform
        assert r_uni[jid] > 0.0
    # linear in d
    r_half = joint_radii(polys, 0.5, n_inner, FILLET_UNIFORM)
    for jid in r_uni:
        assert r_half[jid] == pytest.approx(0.5 * r_uni[jid])


def test_default_mode_is_uniform():
    """Callers that omit `mode` get the uniform (new-bounds) behaviour."""
    polys, n_inner = _two_triangle_panels(c=0.5)
    jn = _junctions(polys)[0]
    _, arms = _joint_arms(jn, polys, n_inner)
    assert _joint_radius_from_arms(arms, 0.9) == pytest.approx(
        _joint_radius_from_arms(arms, 0.9, FILLET_UNIFORM)
    )
    assert _arm_radii(arms, 0.9) == pytest.approx(_arm_radii(arms, 0.9, FILLET_UNIFORM))
    assert joint_radii(polys, 0.9, n_inner) == pytest.approx(
        joint_radii(polys, 0.9, n_inner, FILLET_UNIFORM)
    )


# ---------------------------------------------------------------------------
# Link cross-arms: capped at the midpoint so the fillets along the purple link
# faces don't overlap at d = 1 (the reported bug).
# ---------------------------------------------------------------------------
def test_link_cross_arm_capped_at_midpoint():
    """A link cross-arm — a purple-polygon face joining two inner triangles
    (tt_edge True, is_inner_edge False) — backs off by HALF its length, not the
    full leg, so it no longer overruns its twin from the other end."""
    polys, n_inner = _two_triangle_panels(c=0.5)
    crosses = [
        a
        for jn in _junctions(polys)
        for a in _joint_arms(jn, polys, n_inner)[1]
        if a[4] and not a[3]
    ]
    assert crosses, "expected link cross-arms in the two-triangle tiling"
    for a in crosses:
        assert _arm_backoff_bound(a) == pytest.approx(0.5 * a[1])   # midpoint
        assert _arm_backoff_bound(a) != pytest.approx(1.0 * a[1])   # not the old full leg


@pytest.mark.parametrize("mode", [FILLET_UNIFORM, FILLET_PER_ARM])
def test_cross_arm_back_offs_do_not_overlap_at_d1(mode):
    """The two fillets sharing a link cross-arm must not overrun each other: the
    back-off distances from its two endpoint joints sum to <= the arm length at
    d = 1 (per-arm meets exactly at the midpoint)."""
    polys, n_inner = _two_triangle_panels(c=0.5)
    juncs = _junctions(polys)
    ca = [_joint_arms(jn, polys, n_inner) for jn in juncs]
    centers = [c for c, _ in ca]

    def cross_radius_toward(jidx, target):
        center, arms = ca[jidx]
        for a, r in zip(arms, _arm_radii(arms, 1.0, mode)):
            if a[4] and not a[3]:
                far = center + a[0] * a[1]
                if float(np.linalg.norm(far - target)) < 1e-6:
                    return r, a[1]
        return None

    checked = 0
    for ia, (center_a, arms_a) in enumerate(ca):
        for a in arms_a:
            if not (a[4] and not a[3]):
                continue
            far = center_a + a[0] * a[1]
            ib = next(
                (k for k, c in enumerate(centers)
                 if k != ia and float(np.linalg.norm(c - far)) < 1e-6),
                None,
            )
            if ib is None:
                continue
            ra, L = cross_radius_toward(ia, centers[ib])
            back = cross_radius_toward(ib, center_a)
            assert back is not None
            rb, _ = back
            assert ra + rb <= L + 1e-9                 # no overlap
            if mode == FILLET_PER_ARM:
                assert ra + rb == pytest.approx(L)     # meet exactly at the midpoint
            checked += 1
    assert checked > 0, "no cross-arm shared between two junctions was found"
