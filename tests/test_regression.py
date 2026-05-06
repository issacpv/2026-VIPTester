"""Regression test: refactored auxetic/ package must produce byte-identical
STL, kirigami vertices, and kirigami constraints output as the original
displayAuxeticV20.py script across every mode (1–6).
"""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pytest


REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_SCRIPT   = os.path.join(REPO_ROOT, "data", "grid", "displayAuxeticV20.py")
SEED          = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_original_module():
    spec = importlib.util.spec_from_file_location("displayAuxeticV20_orig", ORIG_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_original(orig, mode, n_points, ratio, nz_layers,
                   stl_path, vertices_path, constraints_path):
    np.random.seed(SEED)
    points, tri = orig.generate_points(n_points, mode)

    strut_curves, solid_tris, joint_positions = orig.collect_export_geometry(
        points, tri, ratio, mode, nz_layers)
    triangles = orig.build_export_triangles(strut_curves, solid_tris, joint_positions)
    orig.export_stl_direct(stl_path, triangles)

    tiles, source = orig.collect_kirigami_tiles(points, tri, ratio, mode, nz_layers)
    constraints   = orig.build_kirigami_constraints(tiles, source)
    orig.export_kirigami_vertices(vertices_path, tiles)
    orig.export_kirigami_constraints(constraints_path, constraints)


def _run_refactored(mode, n_points, ratio, nz_layers,
                     stl_path, vertices_path, constraints_path):
    from auxetic import Lattice

    lattice = Lattice(mode=mode, n_points=n_points, ratio=ratio,
                      nz_layers=nz_layers, seed=SEED)
    lattice.to_stl(stl_path, verbose=False)
    lattice.to_kirigami(vertices_path, constraints_path, verbose=False)


def _diff_bytes(a_path, b_path, skip=0):
    """Compare two files byte-by-byte, optionally skipping the first `skip` bytes."""
    with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
        a, b = fa.read()[skip:], fb.read()[skip:]
    if a == b:
        return None
    diffs = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diffs.append(skip + i)
            if len(diffs) >= 5:
                break
    return f"sizes={len(a)} vs {len(b)}; first diffs at {diffs}"


# Binary STL: bytes 0–79 are a free-form header that numpy-stl populates
# with a timestamp and the output filename — both vary between writes even
# for identical geometry. Skip the header and compare the count + triangle
# payload (bytes 80 onward), which is what determines mesh equivalence.
_STL_HEADER_BYTES = 80


# Each parametrize tuple covers a distinct code path:
#   mode=1 — random 2-D Delaunay, flat (no nz extrusion)
#   mode=2 — random 2-D Delaunay, extruded across nz_layers (most interleaved logic)
#   mode=4 — symmetric 2-D grid via triangulate_grid_symmetric (MockTri path), flat
#   mode=5 — symmetric 2-D grid, extruded across nz_layers
#   mode=6 — symmetric 3-D grid via triangulate_3d_grid_symmetric (tetrahedral)
# Mode 3 is random 3-D Delaunay; covered indirectly by mode 6, omitted because
# Delaunay's tet ordering for n=8 random pts can drift across SciPy versions.
CASES = [
    (1, 5, 0.35, 2),
    (2, 5, 0.35, 2),
    (4, 6, 0.35, 2),
    (5, 6, 0.35, 2),
    (6, 8, 0.35, 2),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,n_points,ratio,nz_layers", CASES)
def test_outputs_match(mode, n_points, ratio, nz_layers, tmp_path):
    orig = _load_original_module()

    orig_stl    = tmp_path / "orig.stl"
    orig_verts  = tmp_path / "orig_vertices.txt"
    orig_consts = tmp_path / "orig_constraints.txt"
    new_stl     = tmp_path / "new.stl"
    new_verts   = tmp_path / "new_vertices.txt"
    new_consts  = tmp_path / "new_constraints.txt"

    _run_original(orig, mode, n_points, ratio, nz_layers,
                  str(orig_stl), str(orig_verts), str(orig_consts))
    _run_refactored(mode, n_points, ratio, nz_layers,
                    str(new_stl), str(new_verts), str(new_consts))

    stl_diff = _diff_bytes(str(orig_stl), str(new_stl), skip=_STL_HEADER_BYTES)
    assert stl_diff is None, f"STL bytes differ: {stl_diff}"

    verts_diff = _diff_bytes(str(orig_verts), str(new_verts))
    assert verts_diff is None, f"Vertices bytes differ: {verts_diff}"

    consts_diff = _diff_bytes(str(orig_consts), str(new_consts))
    assert consts_diff is None, f"Constraints bytes differ: {consts_diff}"


# Allow `python tests/test_regression.py` to run a quick check outside pytest.
if __name__ == "__main__":
    sys.path.insert(0, REPO_ROOT)
    failed = 0
    for case in CASES:
        mode, n_points, ratio, nz_layers = case
        with tempfile.TemporaryDirectory() as tmp:
            orig = _load_original_module()
            orig_stl    = os.path.join(tmp, "orig.stl")
            orig_verts  = os.path.join(tmp, "orig_vertices.txt")
            orig_consts = os.path.join(tmp, "orig_constraints.txt")
            new_stl     = os.path.join(tmp, "new.stl")
            new_verts   = os.path.join(tmp, "new_vertices.txt")
            new_consts  = os.path.join(tmp, "new_constraints.txt")
            _run_original(orig, mode, n_points, ratio, nz_layers,
                          orig_stl, orig_verts, orig_consts)
            _run_refactored(mode, n_points, ratio, nz_layers,
                            new_stl, new_verts, new_consts)
            d_stl    = _diff_bytes(orig_stl,    new_stl, skip=_STL_HEADER_BYTES)
            d_verts  = _diff_bytes(orig_verts,  new_verts)
            d_consts = _diff_bytes(orig_consts, new_consts)
            ok = all(d is None for d in (d_stl, d_verts, d_consts))
            if not ok: failed += 1
            print(f"mode={mode} n={n_points} ratio={ratio}: "
                  f"stl={'OK' if d_stl is None else d_stl}  "
                  f"vertices={'OK' if d_verts is None else d_verts}  "
                  f"constraints={'OK' if d_consts is None else d_consts}")
    sys.exit(1 if failed else 0)
