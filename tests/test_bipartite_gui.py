"""GUI integration tests for mode 11 (bipartite auxetic).

Covers the wiring added on top of the ``auxetic.bipartite`` core math
(itself tested in ``tests/test_bipartite.py``):

- the inspector exposes a 2D-only "Bipartite auxetic" strategy that
  resolves to mode 11 and forces the dim combo to 2D;
- the C-ratio control replaces Ratio / Nz-layers in mode 11 only;
- changing C routes through an undoable command that does NOT re-roll
  the placed points (regenerate=False);
- ``View2D`` renders the polygon network (filled tiles + hinge bars +
  hinge dots) and tears it down when switching away from mode 11.
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

from PyQt6.QtWidgets import QApplication, QGraphicsPolygonItem


@pytest.fixture(scope="module")
def qapp():
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


# A convex rhombus → Delaunay always yields exactly two triangles,
# regardless of which diagonal it picks.
_RHOMBUS = np.array([
    [0.2, 0.5],
    [0.5, 0.85],
    [0.8, 0.5],
    [0.5, 0.15],
])


def _enter_mode_11(win):
    win.inspector.select_mode(11)
    assert win.lattice.mode == 11


# ---------------------------------------------------------------------------
# Mode selection / strategy mapping
# ---------------------------------------------------------------------------

def test_strategy_label_present(win):
    from auxetic_studio.inspector import STRATEGY_LABELS, _DIM_STRAT_TO_MODE
    assert "Bipartite auxetic" in STRATEGY_LABELS
    assert _DIM_STRAT_TO_MODE[("2D", "Bipartite auxetic")] == 11


def test_select_mode_11_forces_2d(win):
    _enter_mode_11(win)
    assert str(win.inspector.dim_combo.currentData()) == "2D"
    assert str(win.inspector.strategy_combo.currentData()) == "Bipartite auxetic"


def test_bipartite_is_editable_and_edge_flippable(win):
    from auxetic_studio.main_window import _EDITABLE_MODES, _EDGE_FLIP_MODES
    assert 11 in _EDITABLE_MODES
    assert 11 in _EDGE_FLIP_MODES


# ---------------------------------------------------------------------------
# C-ratio control visibility
# ---------------------------------------------------------------------------

def test_c_ratio_row_visible_only_in_mode_11(win):
    insp = win.inspector
    # Default mode 1: legacy Ratio shown, C hidden.
    insp.select_mode(1)
    assert insp.ratio_spin.isVisibleTo(insp) is True
    assert insp.c_ratio_spin.isVisibleTo(insp) is False

    _enter_mode_11(win)
    assert insp.c_ratio_spin.isVisibleTo(insp) is True
    assert insp.ratio_spin.isVisibleTo(insp) is False
    # Nz-layers is hidden in mode 11 (no z-extrusion).
    assert insp.nz_layers_spin.isVisibleTo(insp) is False


# ---------------------------------------------------------------------------
# C change must not re-roll points, and is undoable
# ---------------------------------------------------------------------------

def test_changing_c_preserves_points_and_is_undoable(win):
    _enter_mode_11(win)
    pts_before = win.lattice.points.copy()
    c_before = float(win.lattice.C)

    # Drive the real GUI path: setting the spin emits parameterChanged,
    # which MainWindow wraps in a (regenerate=False) command.
    win.inspector.c_ratio_spin.setValue(4.0)
    assert win.lattice.C == pytest.approx(4.0)
    assert np.allclose(win.lattice.points, pts_before), \
        "changing C must not re-roll the placed points"

    win.undo_stack.undo()
    assert win.lattice.C == pytest.approx(c_before)
    assert np.allclose(win.lattice.points, pts_before)


def test_changing_n_points_does_reroll(win):
    """Contrast check: n_points still regenerates (regenerate=True),
    proving the no-regen path is C-specific, not a blanket change."""
    _enter_mode_11(win)
    win.inspector.n_points_spin.setValue(12)
    assert win.lattice.mode == 11
    assert len(win.lattice.points) == 12


# ---------------------------------------------------------------------------
# View rendering
# ---------------------------------------------------------------------------

def test_view_renders_rhombus_network(win):
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    win.view_2d.update_lattice(win.lattice)

    items = win.view_2d._bipartite_items
    fills = [it for it in items if isinstance(it, QGraphicsPolygonItem)]
    segs  = [it for it in items if not isinstance(it, QGraphicsPolygonItem)]
    # Two triangles, each emitting one central polygon + three corner
    # kites — all filled.
    assert len(fills) == 8
    # Line segments: per triangle, 3 kites × 2 perpendicular inner edges
    # (6 blue) + 3 bonds (purple) = 9; two triangles → 18.
    assert len(segs) == 18
    # Hinge dots mark the shared centroid hinges (central-polygon
    # vertices): two triangles × three hinges = six.
    hx, _ = win.view_2d._hinge_scatter.getData()
    assert hx is not None and len(hx) == 6


def test_switching_away_clears_network(win):
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    win.view_2d.update_lattice(win.lattice)
    assert len(win.view_2d._bipartite_items) > 0

    win.inspector.select_mode(1)
    win.view_2d.update_lattice(win.lattice)
    assert len(win.view_2d._bipartite_items) == 0
    hx, _ = win.view_2d._hinge_scatter.getData()
    assert hx is None or len(hx) == 0


# ---------------------------------------------------------------------------
# 3D export + simulation (mode 11 no longer falls through to the 2.5D path)
# ---------------------------------------------------------------------------

def test_mode_11_3d_export_is_nonempty(win):
    """The "lattice not showing in 3D" bug: collect_export_geometry used
    to fall through to the 2.5D path; now it builds the kite slabs."""
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    tris = win.lattice.build_export_triangles(verbose=False)
    assert len(tris) > 0


def test_mode_11_tiles_include_bonds_single_layer(win):
    """Mode 11 is planar (one z-layer) and emits kites + central
    triangles ('tri_face') PLUS the structural bond bars that link
    adjacent kites — without those bonds the mechanism pinwheels."""
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    tiles, src, _cons = win.lattice.collect_kirigami()
    zs = {round(float(t[0][2]), 6) for t in tiles}
    assert zs == {0.0}                       # planar, no second layer
    types = [s['type'] for s in src]
    assert types.count('tri_face') == 8      # 2 central + 6 kites
    assert types.count('bond') == 6          # structural bond bars


def test_mode_11_static_views_render_rest(win):
    """The deterministic per-kite spin was replaced by the coherent
    simulator mechanism, so static rendering ignores joint_angle —
    build_bipartite() always returns the rest tile."""
    import math
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    rest = win.lattice.build_bipartite()
    win.lattice.joint_angle = math.radians(30)
    still_rest = win.lattice.build_bipartite()
    for a, b in zip(rest.polygons, still_rest.polygons):
        assert np.allclose(a.vertices, b.vertices)


def test_mode_11_slider_maps_to_physical_closure(win):
    """After Run Simulation, the slider maps to the mechanism's *physical
    closure* angle — the kite-vs-central relative rotation, i.e. what the
    eye reads as "how far the units turned": 90°→rest, 180°→jamming (full
    hole closure), 135°→half-way.

    The earlier mapping selected poses by ``max(|per-tile rotation|)``.
    Because the two tile families counter-rotate in the floppy mode, a
    single tile's rotation is only ~half the closure, so the slider ran
    ~2× too fast (135° already showed ~90° of closure) and saturated
    before reaching jamming — leaving holes open at "full compression"."""
    import numpy as np
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    panel = win.simulation_panel
    panel.run_simulation()
    assert panel._sim_result is not None

    src = panel._tile_system.tile_source
    central = [i for i, s in enumerate(src) if s.get("kind") == "central"]
    corner  = [i for i, s in enumerate(src) if s.get("kind") == "corner"]
    assert central and corner

    def closure_deg(slider):
        idx = panel._mode11_pose_index_for_slider(slider)
        rot = panel._sim_result.poses[idx][2::3]
        return float(np.degrees(np.mean(rot[corner]) - np.mean(rot[central])))

    jam = float(np.degrees(win.lattice.bipartite_jamming_angle()))
    tol = 0.12 * jam
    assert closure_deg(90.0) == pytest.approx(0.0, abs=2.0)       # rest exact
    assert closure_deg(135.0) == pytest.approx(0.5 * jam, abs=tol)  # half travel
    assert closure_deg(180.0) == pytest.approx(jam, abs=tol)        # full closure
    assert closure_deg(0.0) == pytest.approx(-jam, abs=tol)         # mirror


def test_mode_11_anchor_view_to_polygon(win):
    """Clicking a polygon in the 3D view anchors the kinematic render to
    it: the picked polygon is pinned to its rest placement (zero pose) for
    every slider position, the rest of the structure stays constraint-
    consistent in that frame, and clicking it again releases the anchor."""
    import numpy as np
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    panel = win.simulation_panel
    panel.run_simulation()
    assert panel._sim_result is not None

    # Drive to a tilted pose, then feign a click on a central polygon by
    # emitting the View3D's surface-pick signal at that polygon's centroid.
    panel._drive_pose_from_slider(157.5)
    ts, sim = panel._tile_system, panel._simulator
    central = next(i for i, s in enumerate(ts.tile_source)
                   if s.get("kind") == "central")
    c = sim._tile_world_vertices(panel._displayed_pose, central).mean(axis=0)
    win.view_3d.surfacePointPicked.emit(np.array([c[0], c[1], 0.0]))
    assert panel._anchor_tile == central

    # The anchored polygon is pinned to its rest placement at every slider
    # position, and the relativized pose still satisfies all constraints.
    s = central * sim.dofs
    for slider in (120.0, 175.0, 95.0):
        panel._drive_pose_from_slider(slider)
        dp = panel._displayed_pose
        assert abs(dp[s]) < 1e-9 and abs(dp[s + 1]) < 1e-9 and abs(dp[s + 2]) < 1e-9
        assert np.linalg.norm(sim.constraint_residual(dp)) < 1e-6

    # The anchor banner is shown, and re-clicking the same polygon releases.
    assert "anchored to polygon" in panel._compose_readout_html().lower()
    c2 = sim._tile_world_vertices(panel._displayed_pose, central).mean(axis=0)
    win.view_3d.surfacePointPicked.emit(np.array([c2[0], c2[1], 0.0]))
    assert panel._anchor_tile is None


def test_mode_11_anchor_cleared_on_lattice_change(win):
    """A lattice edit can renumber tiles, so the anchor is released when
    the simulation is marked outdated."""
    import numpy as np
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    panel = win.simulation_panel
    panel.run_simulation()
    panel._drive_pose_from_slider(150.0)
    ts, sim = panel._tile_system, panel._simulator
    central = next(i for i, s in enumerate(ts.tile_source)
                   if s.get("kind") == "central")
    c = sim._tile_world_vertices(panel._displayed_pose, central).mean(axis=0)
    win.view_3d.surfacePointPicked.emit(np.array([c[0], c[1], 0.0]))
    assert panel._anchor_tile is not None
    panel.mark_outdated()
    assert panel._anchor_tile is None


def test_mode_11_sweep_from_rest_keeps_rest_exact(win):
    """The from_rest sweep must keep the θ=0 sample at the rest pose
    (the old −θ_max-start drifted it for wide amplitudes)."""
    import numpy as np
    from auxetic import TileSystem, Simulator
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    sim = Simulator(TileSystem.from_lattice(win.lattice),
                    load_axis=np.array([0.0, -1.0]))
    res = sim.sweep_theta(n_steps=61, theta_max=4 * np.pi, from_rest=True)
    center = int(np.argmin(np.abs(res.theta_samples)))
    assert np.max(np.abs(res.poses[center])) < 1e-6   # exact rest at θ=0
    # and it reaches a large rotation at the extreme
    assert np.degrees(np.max(np.abs(res.poses[-1][2::3]))) > 60.0


def test_mode_11_simulation_is_coherent(win):
    """The wired-in mechanism: Run Simulation builds the kite + central
    + bond tile system, and the solved mode keeps every hinge connected
    (zero constraint residual) through the deformation — no pinwheel,
    no kites snapping off their shared hinge."""
    from auxetic import TileSystem, Simulator
    _enter_mode_11(win)
    win.lattice.regenerate_from_points(_RHOMBUS)
    ts = TileSystem.from_lattice(win.lattice)
    assert any(s.get('type') == 'bond' for s in ts.tile_source)
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0]))
    res = sim.sweep_theta(n_steps=31, theta_max=np.pi / 2)
    deformed = res.poses[len(res.poses) // 2 + 5]
    resid = float(np.max(np.abs(sim.constraint_residual(deformed))))
    assert resid < 1e-6


# ---------------------------------------------------------------------------
# Lattice API guard
# ---------------------------------------------------------------------------

def test_build_bipartite_rejects_non_bipartite_modes(win):
    from auxetic import Lattice
    lat = Lattice(mode=1, n_points=5, seed=0)
    with pytest.raises(RuntimeError, match="only valid for bipartite"):
        lat.build_bipartite()
