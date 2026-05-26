"""GUI integration tests for mode 12 (3D tetrahedral auxetic).

Covers the wiring added on top of the ``auxetic.tetrahedral`` core math
(itself tested in ``tests/test_tetrahedral.py``):

- the inspector exposes a 3D-only "Tetrahedral auxetic" strategy that
  resolves to mode 12 and forces the dim combo to 3D;
- the C control (relabelled "C contraction", range [0, 1]) replaces
  Ratio / Nz-layers in mode 12, and reverts to the bipartite range when
  leaving;
- picking the strategy from the combo seeds an illustrative C;
- changing C routes through an undoable command that does NOT re-roll
  the placed points (regenerate=False);
- selecting mode 12 yields a lattice that produces renderable export
  triangles end-to-end.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtWidgets import QApplication


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


def _enter_mode_12(win):
    win.inspector.select_mode(12)
    assert win.lattice.mode == 12


# ---------------------------------------------------------------------------
# Mode selection / strategy mapping
# ---------------------------------------------------------------------------

def test_strategy_label_present():
    from auxetic_studio.inspector import STRATEGY_LABELS, _DIM_STRAT_TO_MODE
    assert "Tetrahedral auxetic" in STRATEGY_LABELS
    assert _DIM_STRAT_TO_MODE[("3D", "Tetrahedral auxetic")] == 12


def test_select_mode_12_forces_3d(win):
    _enter_mode_12(win)
    assert str(win.inspector.dim_combo.currentData()) == "3D"
    assert str(win.inspector.strategy_combo.currentData()) == "Tetrahedral auxetic"
    # 3D points + tetrahedra.
    assert win.lattice.points.shape[1] == 3


def test_mode_12_not_editable_or_edge_flippable():
    from auxetic_studio.main_window import _EDITABLE_MODES, _EDGE_FLIP_MODES
    assert 12 not in _EDITABLE_MODES
    assert 12 not in _EDGE_FLIP_MODES


# ---------------------------------------------------------------------------
# C control visibility / convention
# ---------------------------------------------------------------------------

def test_c_control_swaps_in_for_mode_12(win):
    insp = win.inspector
    insp.select_mode(1)
    assert insp.ratio_spin.isVisibleTo(insp) is True
    assert insp.c_ratio_spin.isVisibleTo(insp) is False

    _enter_mode_12(win)
    assert insp.c_ratio_spin.isVisibleTo(insp) is True
    assert insp.ratio_spin.isVisibleTo(insp) is False
    assert insp.nz_layers_spin.isVisibleTo(insp) is False
    # Mode-12 contraction convention: range [0, 1], relabelled.
    assert insp.c_ratio_spin.minimum() == pytest.approx(0.0)
    assert insp.c_ratio_spin.maximum() == pytest.approx(1.0)
    assert insp._c_ratio_label.text() == "C contraction"


def test_c_range_reverts_to_bipartite_when_leaving_mode_12(win):
    _enter_mode_12(win)
    assert win.inspector.c_ratio_spin.maximum() == pytest.approx(1.0)
    # Mode 11 uses the Acuna size ratio range up to 20.
    win.inspector.select_mode(11)
    assert win.inspector.c_ratio_spin.maximum() == pytest.approx(20.0)
    assert win.inspector._c_ratio_label.text() == "C ratio"


def test_picking_strategy_from_combo_seeds_illustrative_C(win):
    """Driving the strategy combo like a real user (not the programmatic
    select_mode) flips dim→3D and seeds C into the (0, 1) sweet spot so
    the first render shows both the internal tetra and the corners."""
    insp = win.inspector
    insp.select_mode(1)                       # start somewhere with C=1.0
    assert float(win.lattice.C) >= 1.0
    si = insp.strategy_combo.findData("Tetrahedral auxetic")
    insp.strategy_combo.setCurrentIndex(si)   # fires _on_strategy_changed
    assert win.lattice.mode == 12
    assert 0.0 < float(win.lattice.C) < 1.0


# ---------------------------------------------------------------------------
# C change must not re-roll points, and is undoable
# ---------------------------------------------------------------------------

def test_changing_c_preserves_points_and_is_undoable(win):
    _enter_mode_12(win)
    pts_before = win.lattice.points.copy()
    c_before = float(win.lattice.C)

    win.inspector.c_ratio_spin.setValue(0.7)
    assert win.lattice.C == pytest.approx(0.7)
    assert (win.lattice.points == pts_before).all(), \
        "changing C must not re-roll the placed points"

    win.undo_stack.undo()
    assert win.lattice.C == pytest.approx(c_before)
    assert (win.lattice.points == pts_before).all()


# ---------------------------------------------------------------------------
# End-to-end renderable geometry
# ---------------------------------------------------------------------------

def test_mode_12_selection_yields_renderable_geometry(win):
    _enter_mode_12(win)
    win.lattice.C = 0.5
    win.lattice._clear_caches()
    tris = win.lattice.build_export_triangles(verbose=False)
    assert len(tris) > 0


def test_mode_12_simulation_inputs_build(win):
    """The 'Run Simulation' path can assemble its inputs for mode 12 —
    TileSystem.from_lattice succeeds and reports a 3D system — so the
    button is no longer blocked by the old not-supported guard. (We build
    the inputs rather than run the full 3D sweep, which is slow.)"""
    _enter_mode_12(win)
    ts, load_axis, is_mode_11, jam = win.simulation_panel._build_sim_inputs()
    assert ts.dimension == 3
    assert ts.n_tiles > 0
    assert ts.n_constraints > 0
    assert is_mode_11 is False
    assert load_axis.shape == (3,)
