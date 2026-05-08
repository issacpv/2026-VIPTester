"""Tests for the rotating-cuboids 3D kirigami (mode 10).

Covers the geometry generator, its TileSystem integration, and the
end-to-end Lattice → Simulator → Poisson's ratio path. The
rotating-cubes mechanism's Poisson ratio should be ≈ -1, mirroring
the classic 2D rotating-squares result on the ν_x and ν_z lateral
directions when compressing along y.
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


# ---------------------------------------------------------------------------
# Geometry generator
# ---------------------------------------------------------------------------

def test_generate_cuboids_basic_shape():
    from auxetic.cuboid_kirigami import generate_cuboids
    points, tiles, constraints = generate_cuboids(n=3, ratio=0.35)
    assert points.shape == (27, 3)
    assert len(tiles) == 8       # (n-1)^3 cubes
    for t in tiles:
        assert t.shape == (8, 3)
    # 24 face-adjacent pairs in a 2×2×2 cube cluster, each contributing
    # 2 constraints → 48 single-vertex constraints. Wait — ``generate_cuboids``
    # counts only directed (+x, +y, +z) face neighbours so we expect
    # 12 face-adjacent pairs × 2 = 24 constraints for a 2×2×2 cluster.
    assert len(constraints) == 24


def test_generate_cuboids_corners_match_grid_when_ratio_zero():
    from auxetic.cuboid_kirigami import CUBE_CORNERS, generate_cuboids
    points, tiles, _ = generate_cuboids(n=3, ratio=0.0)
    # ratio=0 → no shrinking → tile vertices == raw lattice points
    n = 3
    def pidx(i, j, k):
        return i + n * j + n * n * k
    cell = (0, 0, 0)
    expected = points[[pidx(cell[0] + di, cell[1] + dj, cell[2] + dk)
                        for di, dj, dk in CUBE_CORNERS]]
    np.testing.assert_allclose(tiles[0], expected, atol=1e-12)


def test_generate_cuboids_shrinks_toward_centroid():
    from auxetic.cuboid_kirigami import generate_cuboids
    _, tiles_a, _ = generate_cuboids(n=3, ratio=0.0)
    _, tiles_b, _ = generate_cuboids(n=3, ratio=0.5)
    # Half-shrink should bring corners halfway to the cell centroid.
    for ta, tb in zip(tiles_a, tiles_b):
        centroid = ta.mean(axis=0)
        expected = 0.5 * ta + 0.5 * centroid
        np.testing.assert_allclose(tb, expected, atol=1e-12)


def test_generate_cuboids_invalid_n_raises():
    from auxetic.cuboid_kirigami import generate_cuboids
    with pytest.raises(ValueError, match="n must be"):
        generate_cuboids(n=1)


def test_triangles_for_cube_returns_12_triangles():
    from auxetic.cuboid_kirigami import triangles_for_cube
    verts = np.array(
        [(x, y, z) for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )
    tris = triangles_for_cube(verts)
    assert len(tris) == 12
    for tri in tris:
        assert tri.shape == (3, 3)


def test_triangles_for_cube_invalid_input_raises():
    from auxetic.cuboid_kirigami import triangles_for_cube
    with pytest.raises(ValueError):
        triangles_for_cube(np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# Constraint topology — bipartite, no orphan tiles
# ---------------------------------------------------------------------------

def test_cuboid_constraints_are_bipartite():
    """The Grima rotating-cubes scheme yields a bipartite tile graph
    (color cubes by (i+j+k) parity). The simulator's bipartite-rotation
    mode selector relies on this for coherent auxetic mode pickup."""
    from auxetic.cuboid_kirigami import generate_cuboids
    _, tiles, constraints = generate_cuboids(n=3, ratio=0.35)
    # Build adjacency
    n = len(tiles)
    adj: list[list[int]] = [[] for _ in range(n)]
    for ta, _va, tb, _vb, _ct in constraints:
        if ta != tb:
            adj[ta].append(tb)
            adj[tb].append(ta)
    # BFS 2-color from tile 0
    color = [-1] * n
    color[0] = 0
    queue = [0]
    bipartite = True
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if color[v] == -1:
                color[v] = 1 - color[u]
                queue.append(v)
            elif color[v] == color[u]:
                bipartite = False
    assert bipartite, "constraint graph must be bipartite for the auxetic mode"


# ---------------------------------------------------------------------------
# TileSystem + Simulator integration
# ---------------------------------------------------------------------------

def test_tile_system_built_from_cuboid_lattice():
    from auxetic import Lattice
    from auxetic.simulation import TileSystem
    lat = Lattice(mode=10, n_points=27, ratio=0.35, seed=42)
    ts = TileSystem.from_lattice(lat)
    assert ts.dimension == 3
    assert ts.n_tiles == 8
    assert ts.n_constraints == 24


def test_cuboid_kirigami_has_negative_poissons_ratio():
    """Textbook rotating-cubes auxetic should produce ν ≈ -1 when
    compressed along an axis (we use y here)."""
    from auxetic import Lattice
    from auxetic.simulation import Simulator, TileSystem
    lat = Lattice(mode=10, n_points=27, ratio=0.35, seed=42)
    ts = TileSystem.from_lattice(lat)
    sim = Simulator(ts, load_axis=np.array([0.0, -1.0, 0.0]))
    nu = sim.poissons_ratio()
    assert isinstance(nu, tuple)
    nu_x, nu_y, nu_z = nu
    # nu_y is on the load axis — NaN by convention.
    assert np.isnan(nu_y)
    # x and z lateral ratios should both be near -1 (auxetic).
    assert nu_x < -0.5, f"expected nu_x ≈ -1, got {nu_x}"
    assert nu_z < -0.5, f"expected nu_z ≈ -1, got {nu_z}"


# ---------------------------------------------------------------------------
# Lattice integration: round-trips through STL export
# ---------------------------------------------------------------------------

def test_cuboid_lattice_stl_export_roundtrip(tmp_path):
    from auxetic import Lattice
    lat = Lattice(mode=10, n_points=27, ratio=0.35, seed=42)
    out = tmp_path / "cuboid.stl"
    lat.to_stl(str(out), verbose=False)
    assert out.is_file()
    assert out.stat().st_size > 1000   # non-trivial STL


def test_cuboid_lattice_regenerate_is_deterministic():
    from auxetic import Lattice
    a = Lattice(mode=10, n_points=27, ratio=0.35, seed=42)
    b = Lattice(mode=10, n_points=27, ratio=0.35, seed=42)
    np.testing.assert_array_equal(a.points, b.points)
    for ta, tb in zip(a.cuboid_tiles, b.cuboid_tiles):
        np.testing.assert_array_equal(ta, tb)
    assert a.cuboid_constraints == b.cuboid_constraints


# ---------------------------------------------------------------------------
# Inspector wiring — Cuboid grid available as a strategy option
# ---------------------------------------------------------------------------

@pytest.fixture
def main_window():
    from PyQt6.QtWidgets import QApplication
    from auxetic_studio.main_window import MainWindow
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(headless_3d=True)
    yield win
    try:
        win.close()
    except Exception:
        pass


def test_inspector_strategy_combo_includes_cuboid_grid(main_window):
    insp = main_window.inspector
    options = [insp.strategy_combo.itemData(i)
               for i in range(insp.strategy_combo.count())]
    assert "Cuboid grid" in options


def test_picking_cuboid_grid_auto_switches_dim_to_3d(main_window):
    insp = main_window.inspector
    # Default lattice is mode 1 (2D random) → dim is 2D.
    assert insp.dim_combo.currentData() == "2D"
    si = insp.strategy_combo.findData("Cuboid grid")
    insp.strategy_combo.setCurrentIndex(si)
    assert insp.dim_combo.currentData() == "3D"
    assert main_window.lattice.mode == 10


def test_switching_dim_off_3d_while_cuboid_reverts_to_grid(main_window):
    """If the user picks Cuboid grid (which auto-flips dim to 3D)
    and then manually flips dim back to 2D, the strategy should
    fall back to Grid (the closest 2D equivalent) so we don't end
    up in an undefined (2D, Cuboid grid) state."""
    insp = main_window.inspector
    insp.select_mode(10)
    assert insp.dim_combo.currentData()      == "3D"
    assert insp.strategy_combo.currentData() == "Cuboid grid"
    di_2d = insp.dim_combo.findData("2D")
    insp.dim_combo.setCurrentIndex(di_2d)
    assert insp.strategy_combo.currentData() == "Grid"
    assert main_window.lattice.mode == 4   # 2D × Grid
