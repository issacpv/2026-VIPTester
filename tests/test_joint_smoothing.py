"""Tests for mode-11 joint smoothing: curved Bézier webs that round the
sharp kirigami pivot joints.

Pure (no GUI) — drives ``Lattice``, ``geometry`` and ``bipartite`` directly.
The webs are a render/export-only addition: enabling them adds material at
the joints and grows the export, disabling them is byte-for-byte identical to
before, and the kinematic simulation never sees them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxetic import bipartite as bip
from auxetic.geometry import joint_smoothing_webs
from auxetic.lattice import Lattice
from auxetic.tile_library import get_tile


def _composed_square() -> Lattice:
    lat = Lattice(mode=11, n_points=4, ratio=0.35)
    sq = get_tile("Square")
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.5, 0.5))
    return lat


def _stl_tris(path) -> int:
    from stl import mesh as stl_mesh
    return len(stl_mesh.Mesh.from_file(str(path)).vectors)


def _payload(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()[80:]  # skip the 80-byte STL header (timestamp/name)


# ---------------------------------------------------------------------------
# Web geometry
# ---------------------------------------------------------------------------

def test_two_webs_per_hinge_anchored_at_the_hinge():
    lat = _composed_square()
    net = bip.build_bipartite_network(
        np.asarray(lat.points), np.asarray(lat.tri.simplices),
        C=lat.C, theta=0.0)
    webs = joint_smoothing_webs(net)
    assert len(webs) == 2 * len(net.hinges)          # one web per joint notch
    assert all(np.all(np.isfinite(w)) for w in webs)
    # Each web's anchor vertex is a hinge point.
    for w in webs:
        assert float(np.min(np.linalg.norm(net.hinges - w[0], axis=1))) < 1e-7
    # Each web is a small, non-degenerate polygon (anchor + arc samples).
    assert all(len(w) >= 3 for w in webs)


def test_bigger_radius_makes_bigger_webs():
    lat = _composed_square()
    net = bip.build_bipartite_network(
        np.asarray(lat.points), np.asarray(lat.tri.simplices),
        C=lat.C, theta=0.0)
    small = joint_smoothing_webs(net, radius_frac=0.2)
    big = joint_smoothing_webs(net, radius_frac=0.45)

    def mean_extent(webs):
        return float(np.mean([
            np.max(np.linalg.norm(w - w[0], axis=1)) for w in webs]))

    assert mean_extent(big) > mean_extent(small)


# ---------------------------------------------------------------------------
# Lattice / export integration
# ---------------------------------------------------------------------------

def test_enabling_grows_export_geometry(tmp_path):
    lat = _composed_square()
    off = tmp_path / "off.stl"
    lat.to_stl(str(off), verbose=False)
    n_off = _stl_tris(off)

    lat.set_joint_smoothing(enabled=True)
    on = tmp_path / "on.stl"
    lat.to_stl(str(on), verbose=False)
    assert _stl_tris(on) > n_off          # webs add triangles


def test_off_is_byte_identical_to_untouched(tmp_path):
    """A lattice with joint smoothing explicitly toggled on then off exports
    byte-for-byte the same payload as one that never enabled it."""
    a = _composed_square()
    pa = tmp_path / "a.stl"
    a.to_stl(str(pa), verbose=False)

    b = _composed_square()
    b.set_joint_smoothing(enabled=True)
    b.set_joint_smoothing(enabled=False)
    pb = tmp_path / "b.stl"
    b.to_stl(str(pb), verbose=False)

    assert _payload(pa) == _payload(pb)


def test_all_three_exporters_with_smoothing(tmp_path):
    lat = _composed_square()
    lat.set_joint_smoothing(enabled=True, radius=0.4, segments=10)

    stl, obj, scad = tmp_path / "c.stl", tmp_path / "c.obj", tmp_path / "c.scad"
    lat.to_stl(str(stl), verbose=False)
    lat.to_obj(str(obj), verbose=False)
    lat.to_scad(str(scad), verbose=False)

    assert _stl_tris(stl) > 0
    obj_text = "\n" + obj.read_text()
    assert "\nv " in obj_text and "\nf " in obj_text
    scad_text = scad.read_text()
    assert "union(){" in scad_text
    for o, c in (("{", "}"), ("[", "]"), ("(", ")")):
        assert scad_text.count(o) == scad_text.count(c)


def test_joint_smoothing_does_not_change_the_simulation():
    """The kinematic TileSystem is built from the rigid polygons, not the
    export webs — so enabling smoothing must not change tiles or
    constraints."""
    from auxetic import TileSystem

    lat = _composed_square()
    ts_off = TileSystem.from_lattice(lat)
    lat.set_joint_smoothing(enabled=True)
    ts_on = TileSystem.from_lattice(lat)
    assert ts_off.n_tiles == ts_on.n_tiles
    assert ts_off.n_constraints == ts_on.n_constraints


@pytest.mark.parametrize("mode,kwargs", [
    (1, dict(n_points=6, seed=3)),
    (6, dict(n_points=8)),
])
def test_non_mode11_unaffected(tmp_path, mode, kwargs):
    """Joint smoothing only touches mode 11; other modes export identically
    whether the flag is on or off."""
    a = Lattice(mode=mode, ratio=0.35, **kwargs)
    pa = tmp_path / "a.stl"
    a.to_stl(str(pa), verbose=False)

    b = Lattice(mode=mode, ratio=0.35, **kwargs)
    b.set_joint_smoothing(enabled=True)
    pb = tmp_path / "b.stl"
    b.to_stl(str(pb), verbose=False)

    assert _payload(pa) == _payload(pb)
