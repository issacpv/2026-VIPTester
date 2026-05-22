"""Smoke tests for the auxetic_studio application shell.

Forces the Qt offscreen platform so the window can be instantiated
without a real display.
"""

import os
import sys

# Force headless Qt before importing anything that touches QtGui/QtWidgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt6.QtWidgets import QApplication


_STL_HEADER_BYTES = 80


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def main_window(qapp):
    from auxetic_studio import MainWindow
    # headless_3d skips the VTK QtInteractor under offscreen Qt — VTK's
    # render-window init can crash without a real display backend.
    win = MainWindow(headless_3d=True)
    yield win
    win.close()


def test_main_window_constructs(main_window):
    """The main window builds with menus, toolbar, central stack, inspector
    dock, and status bar — no display required."""
    win = main_window
    assert win.menuBar() is not None

    # Top-level menus: walk the menu bar's actions; each top-level menu
    # appears as a QAction whose menu() is the QMenu itself.
    actual = {a.menu().title().replace("&", "")
              for a in win.menuBar().actions()
              if a.menu() is not None}
    expected = {"File", "Edit", "View", "Simulate", "Help"}
    assert expected.issubset(actual), f"missing menus: {expected - actual}"

    assert win.statusBar() is not None
    assert win.stack.count() == 2          # 2D + 3D
    assert win.inspector is not None
    assert win._toolbar is not None


def test_inspector_binds_and_regenerates(main_window):
    """Changing inspector widgets updates lattice attributes and triggers
    regenerate() — the points reflect the new mode/n_points."""
    win = main_window
    lattice = win.lattice

    # Start state: mode=1, n_points=5 → 5 random 2D points
    assert lattice.mode == 1
    assert lattice.n_points == 5
    assert lattice.points.shape == (5, 2)

    # Flip to mode=6 (symmetric 3D grid) with n_points=8 via the inspector.
    # Order matters: mode change triggers regenerate first, but the spin
    # boxes will then re-trigger with their existing values (no-op-ish).
    insp = win.inspector
    insp.select_mode(6)
    insp.n_points_spin.setValue(8)

    # After both changes, lattice should be (8, 3)
    assert win.lattice.mode == 6
    assert win.lattice.n_points == 8
    assert win.lattice.points.shape == (8, 3)

    # And ratio binding too
    insp.ratio_spin.setValue(0.50)
    assert win.lattice.ratio == pytest.approx(0.50)


def test_tessellate_from_inspector_rebuilds_lattice(main_window):
    """Task 6d: the inspector's Tessellate control rebuilds the lattice as a
    2D equilateral-fill tessellation at the chosen density (geometry via
    Lattice.from_tessellation). End-to-end: button click → request → swap."""
    win = main_window
    insp = win.inspector
    assert hasattr(insp, "tess_n_triangles_spin")
    assert hasattr(insp, "tessellate_button")

    insp.tess_n_triangles_spin.setValue(60)
    insp.tessellate_button.click()      # → tessellateRequested → MainWindow swap

    lat = win.lattice
    assert lat.mode in (1, 2, 4, 5, 11)            # a 2D mode
    assert lat.points.ndim == 2 and lat.points.shape[1] == 2
    assert lat.points.shape[0] > 5                 # 60-triangle fill >> default 5
    assert win.undo_stack.count() == 0             # major action clears undo


def test_export_stl_matches_direct_lattice_call(main_window, tmp_path):
    """MainWindow.export_stl(path) writes the same STL payload as
    Lattice.to_stl(path) — i.e. the GUI export path doesn't add any
    transformation on top of the library."""
    from auxetic import Lattice
    from scipy.spatial.transform import Rotation

    win = main_window
    # Use a deterministic configuration that doesn't depend on RNG state.
    # Stage 5 made rotation a real concept on Lattice; spell out identity
    # rotation explicitly here so STL byte-equality is pinned to the
    # canonical (no-transform) frame.
    win.lattice = Lattice(mode=6, n_points=8, ratio=0.35, nz_layers=2, seed=42)
    win.lattice.rigid_rotation = Rotation.identity()
    win.lattice.flipped = False
    win.inspector.set_lattice(win.lattice)

    gui_path    = tmp_path / "via_gui.stl"
    direct_path = tmp_path / "via_lib.stl"

    win.export_stl(str(gui_path))

    # Independent Lattice instance, same parameters, same seed, same
    # explicit identity rotation.
    direct = Lattice(mode=6, n_points=8, ratio=0.35, nz_layers=2, seed=42)
    direct.rigid_rotation = Rotation.identity()
    direct.flipped = False
    direct.to_stl(str(direct_path), verbose=False)

    # Skip the 80-byte STL header (numpy-stl writes a timestamp + filename
    # there); compare the triangle payload.
    with open(gui_path, "rb")    as f: gui_bytes    = f.read()[_STL_HEADER_BYTES:]
    with open(direct_path, "rb") as f: direct_bytes = f.read()[_STL_HEADER_BYTES:]

    assert gui_bytes == direct_bytes, (
        f"GUI export and direct library export produced different STL "
        f"payloads ({len(gui_bytes)} vs {len(direct_bytes)} bytes)"
    )
