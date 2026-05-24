"""Mesh I/O helpers used by ``Lattice.from_mesh`` for modes 7/8/9.

The M1 strategy is "use mesh vertices directly": no surface sampling,
no voxel-fill. The user's STL or OBJ is read, vertices are deduplicated
to remove the per-face duplicates STL stores, and (optionally) decimated
down to a manageable count. ``normalize_to_unit_cube`` maps the result
into the same [0, 1]^d coordinate frame the rest of the geometry
pipeline assumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


_VERTEX_DEDUP_DECIMALS = 9


def read_mesh_vertices(path: str) -> np.ndarray:
    """Return an ``(N, 3)`` array of unique vertices from an STL or OBJ file.

    Vertices are deduplicated at ~1e-9 tolerance — STL stores three
    copies of every shared corner (one per adjacent face) and a naive
    Delaunay over those would be degenerate. The returned coordinates
    are in the source file's native frame; pass them through
    ``normalize_to_unit_cube`` before handing them to ``Lattice``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".stl":
        verts = _read_stl_vertices(str(p))
    elif suffix == ".obj":
        verts = _read_obj_vertices(str(p))
    else:
        raise ValueError(f"Unsupported mesh extension: {suffix!r} (want .stl or .obj)")

    if verts.size == 0:
        return np.zeros((0, 3), dtype=float)

    rounded = np.round(verts, _VERTEX_DEDUP_DECIMALS)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return verts[np.sort(idx)]


def _read_stl_vertices(path: str) -> np.ndarray:
    try:
        from stl import mesh as stl_mesh
    except ImportError as exc:
        raise ImportError(
            "Reading STL files requires the 'numpy-stl' package "
            "(`pip install numpy-stl`)."
        ) from exc
    m = stl_mesh.Mesh.from_file(path)
    # m.vectors has shape (n_triangles, 3, 3); flatten to (n_triangles*3, 3).
    return m.vectors.reshape(-1, 3).astype(float)


def _read_obj_vertices(path: str) -> np.ndarray:
    """Parse the ``v X Y Z`` lines of a Wavefront OBJ file.

    No third-party dependency: the OBJ vertex format is line-oriented
    and the first three numerals after ``v`` are X, Y, Z. ``vt``
    (texture) and ``vn`` (normal) lines are ignored.
    """
    verts: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.asarray(verts, dtype=float) if verts else np.zeros((0, 3), dtype=float)


def normalize_to_unit_cube(verts: np.ndarray) -> np.ndarray:
    """Affinely map ``verts`` into ``[0, 1]^d`` per-axis.

    Axes whose extent is below 1e-12 (a flat or single-point axis) are
    set to 0.5 — the same centroid the rest of the lattice pipeline
    assumes for collapsed dimensions.
    """
    arr = np.asarray(verts, dtype=float)
    if arr.size == 0:
        return arr
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    extent = hi - lo
    out = arr - lo
    safe = extent > 1e-12
    out[:, safe] /= extent[safe]
    out[:, ~safe] = 0.5
    return out


def decimate_uniform(verts: np.ndarray, n: int, seed: Optional[int] = None) -> np.ndarray:
    """Reduce ``verts`` to at most ``n`` rows via uniform random sampling
    without replacement. ``seed`` makes the choice deterministic.

    If ``verts`` already has ``≤ n`` rows it is returned unchanged.
    """
    arr = np.asarray(verts)
    if arr.shape[0] <= n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[np.sort(idx)]
