"""Per-triangle C ratio (mode 11): each composed triangle can carry its
own bipartite size ratio C, selected by Shift+clicking it on the canvas.

Covers, bottom-up:
- the bipartite builder + jamming accepting a per-triangle C array;
- ``Lattice`` tracking stable triangle ids + ``piece_C`` overrides that
  survive later tile adds, with a scalar fast-path when nothing is
  overridden, and ``triangle_at_point`` picking on a composed mesh;
- the GUI pick -> undoable ``SetTriangleCCommand`` / ``ClearTriangleCCommand``;
- preset v8 round-trip of the overrides.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxetic.bipartite import build_bipartite_network, jamming_angle
from auxetic.lattice import Lattice
from auxetic.tile_library import get_tile


def _poly_area(v) -> float:
    v = np.asarray(v, dtype=float)
    return 0.5 * abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1)
                            - np.roll(v[:, 0], -1) * v[:, 1]))


# Two edge-adjacent triangles, reused across the builder tests.
_PTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.9], [1.5, 0.9]], dtype=float)
_SIMPS = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)


# ---------------------------------------------------------------------------
# bipartite builder + jamming: per-triangle C
# ---------------------------------------------------------------------------

def test_scalar_and_uniform_array_C_agree():
    scalar = build_bipartite_network(_PTS, _SIMPS, C=1.0)
    arr = build_bipartite_network(_PTS, _SIMPS, C=np.array([1.0, 1.0]))
    sa = sorted(_poly_area(p.vertices) for p in scalar.set_b)
    aa = sorted(_poly_area(p.vertices) for p in arr.set_b)
    assert np.allclose(sa, aa)


def test_per_triangle_C_changes_only_its_triangle():
    base = build_bipartite_network(_PTS, _SIMPS, C=1.0)
    mixed = build_bipartite_network(_PTS, _SIMPS, C=np.array([1.0, 4.0]))

    def central_by_tri(net):
        return {p.triangle_index: _poly_area(p.vertices) for p in net.set_b}

    b, m = central_by_tri(base), central_by_tri(mixed)
    # Triangle 0 (C unchanged) keeps its central area; triangle 1 (C 1->4)
    # grows its central polygon (larger C => kites shrink toward corners).
    assert m[0] == pytest.approx(b[0])
    assert m[1] > b[1]


def test_array_C_stored_on_network():
    net = build_bipartite_network(_PTS, _SIMPS, C=np.array([1.0, 4.0]))
    assert isinstance(net.C, np.ndarray)
    assert np.allclose(net.C, [1.0, 4.0])


def test_wrong_length_C_array_raises():
    with pytest.raises(ValueError):
        build_bipartite_network(_PTS, _SIMPS, C=np.array([1.0, 2.0, 3.0]))


def test_array_C_with_nonpositive_entry_raises():
    with pytest.raises(ValueError):
        build_bipartite_network(_PTS, _SIMPS, C=np.array([1.0, 0.0]))


def test_jamming_angle_accepts_array():
    j = jamming_angle(_PTS, _SIMPS, C=np.array([1.0, 4.0]))
    assert np.isfinite(j) and 0.0 < j <= np.pi / 2.0 + 1e-9


# ---------------------------------------------------------------------------
# Mismatched neighbors keep perpendicular feet (no locking trapezoid)
# ---------------------------------------------------------------------------

# A unit square split by its (0,2) diagonal — mirror-symmetric across the
# diagonal, so its shared-edge feet coincide naturally (no snap involved).
_SQ_PTS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
_SQ_SIMPS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

# An ASYMMETRIC pair sharing edge (0, 1): the two triangles have different
# heights, so with uniform C their feet on the shared edge do NOT coincide —
# the case where the midpoint-fusion (or its absence) is actually exercised.
_ASYM_PTS = np.array([[0, 0], [1, 0], [0.35, 0.9], [0.55, -0.8]], dtype=float)
_ASYM_SIMPS = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)


def _max_abs_cos_inner_edges(net, pts, simps):
    """Worst |cos(angle)| between any kite inner edge and an incident
    triangle edge — 0.0 means every inner edge is perfectly perpendicular
    (the rectangle / perfect-auxetic condition)."""
    incident = {int(v) for s in simps for v in s}
    worst = 0.0
    for k in net.set_a:
        ci = k.corner_point_index
        Pi = pts[ci]
        others = [pts[i] for i in incident if i != ci]
        for edge in k.inner_edges():
            d = edge[1] - edge[0]
            dn = d / (np.linalg.norm(d) + 1e-12)
            cosines = [abs(float(np.dot(dn, (o - Pi) / (np.linalg.norm(o - Pi) + 1e-12))))
                       for o in others]
            worst = max(worst, min(cosines))
    return worst


def _corner0_shared_vertices(net):
    """Vertices shared by the two corner-0 kites (one per triangle) —
    {corner node} alone if feet are NOT fused, {corner + foot} if fused."""
    k0 = [k for k in net.set_a if k.corner_point_index == 0]
    assert len(k0) == 2
    va = {tuple(np.round(v, 6)) for v in k0[0].vertices}
    vb = {tuple(np.round(v, 6)) for v in k0[1].vertices}
    return va & vb


def test_mismatched_C_keeps_inner_edges_perpendicular():
    # Differing per-triangle C => the two tiles meet the shared edge at
    # different heights; the builder must NOT snap the feet to the midpoint
    # (which tilts the inner edges into a locking trapezoid). Every kite
    # inner edge must stay perpendicular to its triangle edge.
    net = build_bipartite_network(_SQ_PTS, _SQ_SIMPS, C=np.array([4.0, 1.0]))
    assert _max_abs_cos_inner_edges(net, _SQ_PTS, _SQ_SIMPS) < 1e-9


def test_matched_C_still_fuses_asymmetric_feet():
    # Same C, asymmetric pair: feet don't coincide naturally, so they are
    # fused to the midpoint (unchanged behaviour) — the two kites share the
    # corner AND the fused foot.
    net = build_bipartite_network(_ASYM_PTS, _ASYM_SIMPS, C=1.0)
    assert len(_corner0_shared_vertices(net)) >= 2


def test_mismatched_C_does_not_fuse_feet():
    # Different C on the same asymmetric pair: feet are NOT fused (only the
    # corner is shared) and perpendicularity is preserved.
    net = build_bipartite_network(_ASYM_PTS, _ASYM_SIMPS, C=np.array([4.0, 1.0]))
    assert len(_corner0_shared_vertices(net)) == 1
    assert _max_abs_cos_inner_edges(net, _ASYM_PTS, _ASYM_SIMPS) < 1e-9


def test_force_no_fuse_flag_preserves_perpendicularity():
    # The explicit escape hatch forces the perpendicular method on every
    # shared edge, even for same-C neighbours that would otherwise fuse.
    net = build_bipartite_network(_ASYM_PTS, _ASYM_SIMPS, C=1.0,
                                  fuse_shared_feet=False)
    assert len(_corner0_shared_vertices(net)) == 1
    assert _max_abs_cos_inner_edges(net, _ASYM_PTS, _ASYM_SIMPS) < 1e-9


# ---------------------------------------------------------------------------
# Lattice: stable ids + piece_C overrides
# ---------------------------------------------------------------------------

def _composed_two_squares() -> Lattice:
    lat = Lattice(mode=1, n_points=6, seed=0)
    sq = get_tile("Square")
    edge = float(sq.points[:, 0].max() - sq.points[:, 0].min())
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4, 0.5))
    lat.compose_add_tile(sq.points, sq.simplices, offset=(0.4 + edge, 0.5))
    return lat


def test_no_override_uses_scalar_fast_path():
    lat = _composed_two_squares()
    c = lat._per_triangle_C()
    assert np.ndim(c) == 0 and float(c) == pytest.approx(lat.C)


def test_set_triangle_C_produces_array_and_value():
    lat = _composed_two_squares()
    n = len(lat.tri.simplices)
    lat.set_triangle_C(0, 5.0)
    c = lat._per_triangle_C()
    assert np.ndim(c) == 1 and len(c) == n
    assert c[0] == pytest.approx(5.0)
    assert lat.get_triangle_C(0) == pytest.approx(5.0)
    assert lat.has_triangle_C(0) is True
    assert lat.has_triangle_C(1) is False
    # the render actually reflects the override (central polygon differs)
    net = lat.build_bipartite()
    central = {p.triangle_index: _poly_area(p.vertices) for p in net.set_b}
    assert central[0] != pytest.approx(central[1])


def test_override_survives_a_later_tile_add():
    lat = _composed_two_squares()
    lat.set_triangle_C(0, 7.0)
    tid = lat.triangle_id_at_index(0)
    tri = get_tile("Triangle")
    lat.compose_add_tile(tri.points, tri.simplices, offset=(0.9, 0.9))
    locs = [i for i, t in enumerate(lat.tri_ids) if int(t) == tid]
    assert locs, "the overridden triangle's id should still be present"
    assert all(lat.get_triangle_C(i) == pytest.approx(7.0) for i in locs)


def test_regenerate_clears_overrides():
    lat = _composed_two_squares()
    lat.set_triangle_C(0, 9.0)
    lat.regenerate()
    assert lat.piece_C == {}
    assert lat.tri_ids is None
    assert np.ndim(lat._per_triangle_C()) == 0


def test_triangle_at_point_picks_on_composed_mesh():
    # Composed meshes use a _FlippedTri (no find_simplex) — picking must
    # still work via the in-house containment test.
    lat = _composed_two_squares()
    simp = np.asarray(lat.tri.simplices)
    for idx in range(len(simp)):
        cen = lat.points[simp[idx]].mean(axis=0)
        assert lat.triangle_at_point(cen, world=False) == idx


def test_mismatched_C_lattice_stays_collapsible():
    # The headline fix: overriding a neighbour's C must leave a collapsible
    # (rotatable) mechanism, not a locked trapezoid.
    from auxetic.simulation import Simulator, TileSystem
    lat = _composed_two_squares()
    lat.set_triangle_C(0, 5.0)
    ts = TileSystem.from_lattice(lat)
    sim = Simulator(ts, load_axis=np.array([0.0, 1.0]))
    res = sim.sweep_theta(n_steps=61, collision_stop=True)
    assert res.locked is False
    col = np.asarray(res.collision_at_theta, dtype=bool)
    th = np.asarray(res.theta_samples, dtype=float)
    assert (~col).any()
    assert float(np.max(np.abs(th[~col]))) > np.radians(20)


# ---------------------------------------------------------------------------
# GUI: pick -> undoable C command, reset, preset round-trip
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def win(qapp):
    from auxetic_studio.main_window import MainWindow
    w = MainWindow(headless_3d=True)
    yield w
    try:
        w.close()
    except Exception:
        pass


def test_pick_then_set_triangle_C_is_undoable(win):
    win._on_tile_dropped("Square", 0.4, 0.5)
    win._on_tile_dropped("Square", 0.65, 0.5)
    assert win.lattice.mode == 11
    n_cmds = win.undo_stack.count()

    win._on_triangle_picked(0)
    assert win._selected_triangle == 0
    assert win.view_2d.selected_triangle == 0
    assert win.library_panel.triangle_c_spin.isEnabled()

    win.library_panel.triangle_c_spin.setValue(5.0)   # emits change -> command
    assert win.undo_stack.count() == n_cmds + 1
    assert win.lattice.get_triangle_C(0) == pytest.approx(5.0)
    assert win.lattice.has_triangle_C(0) is True
    # the other triangles are untouched (still global C)
    assert win.lattice.has_triangle_C(1) is False

    win.undo_stack.undo()
    assert win.lattice.has_triangle_C(0) is False
    assert win.lattice.get_triangle_C(0) == pytest.approx(win.lattice.C)


def test_reset_triangle_C_clears_override(win):
    win._on_tile_dropped("Square", 0.5, 0.5)
    win._on_triangle_picked(0)
    win.library_panel.triangle_c_spin.setValue(3.0)
    assert win.lattice.has_triangle_C(0) is True
    # Reset button is enabled once an override exists; clicking it clears.
    assert win.library_panel.triangle_reset_button.isEnabled()
    win.library_panel.triangle_reset_button.click()
    assert win.lattice.has_triangle_C(0) is False
    win.undo_stack.undo()                       # undo the clear
    assert win.lattice.has_triangle_C(0) is True


def test_undo_of_tile_add_restores_overrides(win):
    win._on_tile_dropped("Square", 0.4, 0.5)
    win._on_triangle_picked(0)
    win.library_panel.triangle_c_spin.setValue(4.0)
    # add another tile, then undo it: id state must be exactly restored.
    win._on_tile_dropped("Square", 0.65, 0.5)
    win.undo_stack.undo()                       # undo the second add
    assert win.lattice.get_triangle_C(0) == pytest.approx(4.0)


def test_per_triangle_C_survives_preset_roundtrip(tmp_path):
    from auxetic_studio.preset import PRESET_VERSION, load_preset, save_preset
    assert PRESET_VERSION == 8
    lat = _composed_two_squares()
    lat.set_triangle_C(0, 6.0)
    lat.set_triangle_C(2, 0.5)
    ids_before = lat.tri_ids.copy()

    path = str(tmp_path / "ptc.json")
    save_preset(path, lat)
    loaded = load_preset(path)

    assert np.array_equal(loaded.tri_ids, ids_before)
    assert loaded.get_triangle_C(0) == pytest.approx(6.0)
    assert loaded.get_triangle_C(2) == pytest.approx(0.5)
    assert loaded.has_triangle_C(1) is False
    c = loaded._per_triangle_C()
    assert np.ndim(c) == 1
    assert c[0] == pytest.approx(6.0) and c[2] == pytest.approx(0.5)


class _FakeClick:
    """Minimal stand-in for a pyqtgraph MouseClickEvent — just the bits
    ``_on_scene_clicked`` reads before it would touch coordinates."""

    def __init__(self, *, shift: bool, left: bool = True):
        from PyQt6.QtCore import Qt
        self._mods = (Qt.KeyboardModifier.ShiftModifier if shift
                      else Qt.KeyboardModifier.NoModifier)
        self._btn = (Qt.MouseButton.LeftButton if left
                     else Qt.MouseButton.RightButton)
        self.accepted = False

    def button(self):
        return self._btn

    def modifiers(self):
        return self._mods

    def accept(self):
        self.accepted = True


def test_scene_click_gating(win):
    # The handler ignores clicks unless mode==11 AND left button AND Shift.
    # All three negative cases bail out before any coordinate mapping, so
    # this is robust without a rendered window.
    picks: list[int] = []
    win.view_2d.trianglePicked.connect(lambda i: picks.append(i))

    # Not composed (mode 1): ignored even with Shift.
    win.view_2d._on_scene_clicked(_FakeClick(shift=True))
    assert picks == []

    win._on_tile_dropped("Square", 0.5, 0.5)        # -> mode 11
    # Left-click without Shift: ignored.
    win.view_2d._on_scene_clicked(_FakeClick(shift=False))
    assert picks == []
    # Shift + non-left button: ignored.
    win.view_2d._on_scene_clicked(_FakeClick(shift=True, left=False))
    assert picks == []


def test_migration_v7_to_v8_adds_empty_overrides():
    from auxetic_studio.preset import _migrate_v7_to_v8
    v7 = {"version": 7, "compose": {"preserve_triangulation": True,
                                    "simplices": [[0, 1, 2]]}}
    out = _migrate_v7_to_v8(v7)
    assert out["version"] == 8
    assert out["compose"]["tri_ids"] is None
    assert out["compose"]["piece_C"] == {}
