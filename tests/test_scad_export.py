"""Integration tests guarding ``Lattice.to_scad`` / ``export_to_scad``.

There was previously no test exercising the SCAD writer. These tests
assert that ``to_scad`` emits structurally valid OpenSCAD for a
representative spread of modes (2D random / 2D grid / 2.5D extruded /
3D grid), and — when the ``openscad`` CLI is available — that the file
actually parses. They exist to catch regressions from the bezier-edge
work (task 1) and the tessellation generator (task 5), both of which
feed new geometry through this writer.

Pure (no GUI) — drives ``Lattice`` directly, so it is stable on the
offscreen Qt platform.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auxetic.lattice import Lattice


# (mode, kwargs) covering each branch of the export pipeline.
_MODES = [
    (1, dict(n_points=5, seed=1)),                  # 2D random Delaunay
    (4, dict(n_points=9)),                          # 2D symmetric grid
    (2, dict(n_points=6, nz_layers=2, seed=1)),     # 2.5D extruded
    (6, dict(n_points=8)),                          # 3D grid (tet + hubs)
]


def _balanced(text: str, open_ch: str, close_ch: str) -> bool:
    """True iff ``open_ch`` / ``close_ch`` are balanced and never go
    negative scanning left-to-right (so closers never precede openers)."""
    depth = 0
    for ch in text:
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _assert_valid_scad(text: str, *, mode: int, n_points: int, ratio: float) -> None:
    assert text, "SCAD output is empty"

    # Header comment carries the generation parameters.
    assert re.search(
        rf"^//\s*auxetic lattice\s+mode={mode}\s+n_points={n_points}\s+ratio={ratio}",
        text,
    ), "missing/incorrect SCAD header comment"

    # Global facet count + the render/union scaffold.
    assert re.search(r"\$fn=\d+;", text), "missing $fn declaration"
    assert "union(){" in text, "missing union() block"

    # Struts render as cylinders; the combined solid is one polyhedron.
    assert text.count("cylinder(") >= 1, "expected at least one strut cylinder"
    assert text.count("polyhedron(") == 1, "expected exactly one polyhedron"

    # Delimiters must balance and never close before opening.
    for o, c in (("{", "}"), ("[", "]"), ("(", ")")):
        assert _balanced(text, o, c), f"unbalanced '{o}{c}' in SCAD"

    # The whole thing closes the union block.
    assert text.rstrip().endswith("}"), "SCAD does not end with closing brace"

    # No non-finite coordinates leaked into the geometry.
    assert not re.search(r"\b(nan|inf|-inf)\b", text, re.IGNORECASE), \
        "SCAD contains non-finite coordinates"

    # Polyhedron face indices must reference points that exist. The writer
    # emits points=[[x,y,z],...],faces=[[i,j,k],...],convexity=...
    m = re.search(r"polyhedron\(points=\[(.+)\],faces=\[(.+)\],convexity",
                  text, re.S)
    assert m, "polyhedron points/faces block not found"
    pts_blob, faces_blob = m.group(1), m.group(2)
    n_vertices = len(re.findall(r"\[[^\[\]]+\]", pts_blob))
    face_indices = [int(t) for t in re.findall(r"-?\d+", faces_blob)]
    assert n_vertices > 0, "polyhedron has no vertices"
    assert face_indices, "polyhedron has no faces"
    assert min(face_indices) >= 0, "negative face index"
    assert max(face_indices) < n_vertices, \
        f"face index {max(face_indices)} out of range for {n_vertices} vertices"


@pytest.mark.parametrize("mode,kwargs", _MODES)
def test_to_scad_emits_valid_scad(tmp_path, mode, kwargs):
    ratio = 0.35
    lat = Lattice(mode=mode, ratio=ratio, **kwargs)
    out = tmp_path / f"lattice_mode{mode}.scad"
    lat.to_scad(str(out), verbose=False)
    assert out.exists() and out.stat().st_size > 0
    _assert_valid_scad(out.read_text(), mode=mode,
                       n_points=kwargs["n_points"], ratio=ratio)


def test_to_scad_struts_have_positive_radius(tmp_path):
    """Every emitted cylinder must have a finite length and positive
    radius — a degenerate strut would print as an invalid/zero solid."""
    lat = Lattice(mode=1, n_points=6, ratio=0.35, seed=2)
    out = tmp_path / "struts.scad"
    lat.to_scad(str(out), verbose=False)
    text = out.read_text()
    cyls = re.findall(r"cylinder\(h=([-\d.eE]+),r=([-\d.eE]+),", text)
    assert cyls, "no cylinders found"
    for h, r in cyls:
        assert float(h) > 0.0, f"non-positive cylinder height {h}"
        assert float(r) > 0.0, f"non-positive cylinder radius {r}"


@pytest.mark.skipif(shutil.which("openscad") is None,
                    reason="openscad CLI not installed")
def test_to_scad_parses_in_openscad(tmp_path):
    """When the OpenSCAD CLI is present, the emitted file must parse and
    render without error (the strongest 'valid output' check)."""
    lat = Lattice(mode=1, n_points=5, ratio=0.35, seed=1)
    scad = tmp_path / "parse.scad"
    lat.to_scad(str(scad), verbose=False)
    stl = tmp_path / "parse.stl"
    proc = subprocess.run(
        ["openscad", "-o", str(stl), str(scad)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"openscad failed:\n{proc.stderr}"
    assert stl.exists() and stl.stat().st_size > 0
