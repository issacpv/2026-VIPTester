"""Tests for Bezier-curved strut edges (task 1).

Two layers:

1. The pure curve math (``auxetic.geometry.bezier_polyline``): zero
   curvature reproduces the straight 2-point segment exactly; non-zero
   curvature produces a denser, smooth, single-bump polyline with the
   endpoints preserved.
2. The Lattice/export integration: with curves OFF, STL/OBJ/SCAD output
   is byte-for-byte identical to before; with curves ON, all three
   exporters still emit valid, denser geometry; and ``set_bezier``
   correctly invalidates the cached geometry both ways.

Pure (no GUI) — drives ``Lattice`` and ``geometry`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxetic.geometry import bezier_polyline
from auxetic.lattice import Lattice


# ---------------------------------------------------------------------------
# Curve math
# ---------------------------------------------------------------------------

def _offsets_from_line(poly, p0, p1):
    """Perpendicular distance of each polyline point from the p0->p1 line."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    dh = d / np.linalg.norm(d)
    out = []
    for p in np.asarray(poly, float):
        v = p - p0
        out.append(float(np.linalg.norm(v - np.dot(v, dh) * dh)))
    return np.array(out)


def test_zero_strength_returns_straight_segment():
    poly = bezier_polyline([0, 0, 0], [1, 2, 3], strength=0.0, segments=12)
    assert poly.shape == (2, 3)
    assert np.array_equal(poly, np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))


def test_one_segment_returns_straight_segment():
    poly = bezier_polyline([0, 0, 0], [1, 0, 0], strength=0.5, segments=1)
    assert poly.shape == (2, 3)


def test_degenerate_segment_returns_two_points():
    poly = bezier_polyline([1, 1, 1], [1, 1, 1], strength=0.5, segments=10)
    assert poly.shape == (2, 3)


@pytest.mark.parametrize("segments", [2, 4, 8, 16])
def test_nonzero_is_denser_polyline(segments):
    poly = bezier_polyline([0, 0, 0], [2, 0, 0], strength=0.3, segments=segments)
    assert poly.shape == (segments + 1, 3)
    assert len(poly) > 2


def test_endpoints_preserved_exactly():
    p0, p1 = [0.3, -1.0, 0.5], [2.2, 0.4, -1.1]
    poly = bezier_polyline(p0, p1, strength=0.4, segments=9)
    assert np.array_equal(poly[0], np.asarray(p0, float))
    assert np.array_equal(poly[-1], np.asarray(p1, float))


def test_midpoint_offset_matches_quadratic_bezier():
    # For a quadratic Bezier with control point lifted by s*L perpendicular
    # to the chord, the apex (t=0.5) sits at 0.5*s*L off the chord.
    p0, p1 = [0, 0, 0], [4, 0, 0]
    s, L = 0.25, 4.0
    poly = bezier_polyline(p0, p1, strength=s, segments=10)
    offs = _offsets_from_line(poly, p0, p1)
    assert offs.max() == pytest.approx(0.5 * s * L, rel=1e-6)


def test_offset_profile_is_a_single_smooth_bump():
    poly = bezier_polyline([0, 0, 0], [3, 0, 0], strength=0.2, segments=12)
    offs = _offsets_from_line(poly, [0, 0, 0], [3, 0, 0])
    # Endpoints flush with the chord; interior strictly lifted.
    assert offs[0] == pytest.approx(0.0)
    assert offs[-1] == pytest.approx(0.0)
    assert np.all(offs[1:-1] > 0.0)
    # Unimodal: rises to a single peak then falls (no wiggles).
    peak = int(np.argmax(offs))
    assert np.all(np.diff(offs[: peak + 1]) >= -1e-12)
    assert np.all(np.diff(offs[peak:]) <= 1e-12)
    # Symmetric apex sits at the middle for an evenly-sampled quadratic.
    assert peak == len(offs) // 2


def test_bow_dir_hint_controls_curve_plane():
    # A bow_dir along +y should bow the curve in +y; the apex y-offset is
    # positive and the z-offset stays ~0.
    poly = bezier_polyline([0, 0, 0], [2, 0, 0], strength=0.3, segments=8,
                           bow_dir=[0, 1, 0])
    apex = poly[len(poly) // 2]
    assert apex[1] > 0.05
    assert abs(apex[2]) < 1e-9


def test_bow_dir_parallel_falls_back_to_perpendicular():
    # bow_dir parallel to the segment has no perpendicular component, so a
    # deterministic perpendicular is used and the curve still bows.
    poly = bezier_polyline([0, 0, 0], [2, 0, 0], strength=0.3, segments=8,
                           bow_dir=[1, 0, 0])
    offs = _offsets_from_line(poly, [0, 0, 0], [2, 0, 0])
    assert offs.max() > 0.05


# ---------------------------------------------------------------------------
# Lattice / export integration
# ---------------------------------------------------------------------------

def _stl_triangle_count(path):
    from stl import mesh as stl_mesh
    return len(stl_mesh.Mesh.from_file(str(path)).vectors)


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("mode,kwargs", [
    (1, dict(n_points=6, seed=3)),
    (6, dict(n_points=8)),
])
def test_curves_off_is_byte_identical(tmp_path, mode, kwargs):
    """A lattice with bezier explicitly disabled exports identically to a
    default lattice (the regression suite already locks default==V20)."""
    a = Lattice(mode=mode, ratio=0.35, **kwargs)
    b = Lattice(mode=mode, ratio=0.35, bezier_enabled=False, **kwargs)
    pa, pb = tmp_path / "a.stl", tmp_path / "b.stl"
    a.to_stl(str(pa), verbose=False)
    b.to_stl(str(pb), verbose=False)
    # Skip the 80-byte STL header (timestamp/name vary), like the
    # regression test does.
    assert _read_bytes(pa)[80:] == _read_bytes(pb)[80:]


@pytest.mark.parametrize("mode,kwargs", [
    (1, dict(n_points=6, seed=3)),
    (6, dict(n_points=8)),
    (11, dict(n_points=8, seed=5)),
])
def test_curves_on_all_three_exporters(tmp_path, mode, kwargs):
    lat = Lattice(mode=mode, ratio=0.35, **kwargs)
    lat.set_bezier(enabled=True, strength=0.3, segments=8)

    stl = tmp_path / "c.stl"
    obj = tmp_path / "c.obj"
    scad = tmp_path / "c.scad"
    lat.to_stl(str(stl), verbose=False)
    lat.to_obj(str(obj), verbose=False)
    lat.to_scad(str(scad), verbose=False)

    # STL reads back as a non-empty mesh.
    assert stl.exists() and _stl_triangle_count(stl) > 0
    # OBJ has vertices and faces.
    obj_text = obj.read_text()
    assert "\nv " in "\n" + obj_text and "\nf " in "\n" + obj_text
    # SCAD parses structurally and contains curved struts.
    scad_text = scad.read_text()
    assert "union(){" in scad_text
    assert scad_text.count("polyhedron(") == 1
    assert scad_text.rstrip().endswith("}")
    for o, c in (("{", "}"), ("[", "]"), ("(", ")")):
        assert scad_text.count(o) == scad_text.count(c)


def test_curves_on_emits_more_strut_cylinders_than_off(tmp_path):
    """Each strut becomes N cylinders (one per Bezier segment), so ON must
    emit strictly more cylinders than OFF for the same lattice. Mode 6
    (3D grid) reliably has boundary struts; some random 2D lattices have
    none (every hub is a polygon)."""
    off = Lattice(mode=6, n_points=8, ratio=0.35)
    off_scad = tmp_path / "off.scad"
    off.to_scad(str(off_scad), verbose=False)
    n_off = off_scad.read_text().count("cylinder(")

    on = Lattice(mode=6, n_points=8, ratio=0.35)
    on.set_bezier(enabled=True, strength=0.3, segments=6)
    on_scad = tmp_path / "on.scad"
    on.to_scad(str(on_scad), verbose=False)
    n_on = on_scad.read_text().count("cylinder(")

    assert n_off >= 1
    assert n_on == n_off * 6  # 6 segments per strut


def test_set_bezier_round_trips_geometry(tmp_path):
    """Enabling then disabling bezier must return byte-identical geometry,
    proving the export-geometry cache is invalidated both ways."""
    lat = Lattice(mode=1, n_points=6, ratio=0.35, seed=3)
    base = tmp_path / "base.stl"
    lat.to_stl(str(base), verbose=False)
    base_bytes = _read_bytes(base)[80:]

    lat.set_bezier(enabled=True, strength=0.3, segments=8)
    on = tmp_path / "on.stl"
    lat.to_stl(str(on), verbose=False)
    assert _read_bytes(on)[80:] != base_bytes  # geometry actually changed

    lat.set_bezier(enabled=False)
    back = tmp_path / "back.stl"
    lat.to_stl(str(back), verbose=False)
    assert _read_bytes(back)[80:] == base_bytes  # cache invalidated -> identical


def test_zero_strength_with_enabled_is_still_straight(tmp_path):
    """enabled=True but strength=0 must NOT curve (defensive double-guard)."""
    a = Lattice(mode=1, n_points=6, ratio=0.35, seed=3)
    a_stl = tmp_path / "a.stl"
    a.to_stl(str(a_stl), verbose=False)

    b = Lattice(mode=1, n_points=6, ratio=0.35, seed=3)
    b.set_bezier(enabled=True, strength=0.0, segments=8)
    b_stl = tmp_path / "b.stl"
    b.to_stl(str(b_stl), verbose=False)

    assert _read_bytes(a_stl)[80:] == _read_bytes(b_stl)[80:]
