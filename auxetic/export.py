"""File-format writers: STL, OBJ, SCAD (mesh) and kirigami vertices/constraints (text)."""

import os
import numpy as np

from . import geometry as _geom


def _fmt(v):
    return f"[{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}]"


def export_to_scad(scad_path, strut_curves, triangles,
                    mode=None, n_points=None, ratio=None,
                    strut_radius=None, scad_segments=None,
                    verbose=True):
    if strut_radius is None:  strut_radius = _geom.STRUT_RADIUS
    if scad_segments is None: scad_segments = _geom.SCAD_SEGMENTS

    def scad_cylinder(p0, p1, radius, fn):
        d   = p1 - p0
        L   = np.linalg.norm(d)
        if L < 1e-10: return ""
        dh  = d / L
        dot = np.clip(dh[2], -1, 1)
        ang = np.degrees(np.arccos(dot))
        ax  = np.cross([0, 0, 1], dh)
        an  = np.linalg.norm(ax)
        ax  = ax / an if an > 1e-10 else np.array([1, 0, 0])
        return (f"translate({_fmt(p0)})\n"
                f"  rotate(a={ang:.6f},v={_fmt(ax)})\n"
                f"    cylinder(h={L:.6f},r={radius:.6f},$fn={fn});\n")

    lines = [f"// auxetic lattice  mode={mode}  n_points={n_points}  ratio={ratio}\n",
             f"$fn={scad_segments};\nrender(convexity=10)\nunion(){{\n",
             "  // struts\n"]
    for pts in strut_curves:
        pts = np.asarray(pts, float)
        # A straight strut is a 2-point polyline -> one cylinder (output
        # unchanged). A bezier strut is an N-point polyline -> one
        # cylinder per consecutive segment so the arc renders in SCAD too.
        for i in range(len(pts) - 1):
            lines.append(scad_cylinder(pts[i], pts[i + 1], strut_radius, scad_segments))
    if triangles:
        all_v, all_f, vi = [], [], 0
        for tri in triangles:
            a, b, c = (np.asarray(tri[k], float) for k in range(3))
            all_v  += [a, b, c]
            all_f.append([vi, vi + 1, vi + 2])
            vi     += 3
        vs = ", ".join(_fmt(v) for v in all_v)
        fs = ", ".join("[" + ",".join(str(x) for x in f) + "]" for f in all_f)
        lines.append(f"polyhedron(points=[{vs}],faces=[{fs}],convexity=10);\n")
    lines.append("}\n")
    with open(scad_path, "w") as f:
        f.writelines(lines)
    if verbose:
        print(f"  SCAD: {scad_path}  ({len(strut_curves)} struts, {len(triangles)} tris)")


def export_stl_direct(stl_path, triangles, verbose=True):
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        print("  numpy-stl not installed — skipping STL"); return
    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        for j in range(3):
            m.vectors[i][j] = np.asarray(tri[j], float)
    m.save(stl_path)
    if verbose:
        print(f"  STL: {stl_path}  ({os.path.getsize(stl_path) // 1024} KB)")


def export_obj_direct(obj_path, triangles, verbose=True):
    lines   = ["# auxetic lattice\n"]
    v_count = 0
    n_count = 0
    for tri in triangles:
        a, b, c = (np.asarray(tri[k], float) for k in range(3))
        nv      = np.cross(b - a, c - a)
        nn      = np.linalg.norm(nv)
        if nn < 1e-14: continue
        nv      = nv / nn
        vb      = v_count + 1
        for p in (a, b, c):
            lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        v_count += 3
        n_count += 1
        lines.append(f"vn {nv[0]:.9g} {nv[1]:.9g} {nv[2]:.9g}\n")
        lines.append(f"f {vb}//{n_count} {vb+1}//{n_count} {vb+2}//{n_count}\n")
    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    if verbose:
        print(f"  OBJ: {obj_path}  ({os.path.getsize(obj_path) // 1024} KB, {n_count} tris)")


def export_kirigami_vertices(path, tiles, verbose=True):
    lines = []
    for tile in tiles:
        coords = []
        for vert in tile:
            coords.extend([f"{vert[0]:.9g}", f"{vert[1]:.9g}", f"{vert[2]:.9g}"])
        lines.append(" ".join(coords) + "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    if verbose:
        print(f"  Vertices: {path}  ({len(tiles)} tiles, "
              f"{os.path.getsize(path) // 1024} KB)")


def export_kirigami_constraints(path, constraints, verbose=True):
    lines = []
    for (t1, v1, t2, v2, ctype) in constraints:
        lines.append(f"{t1} {v1} {t2} {v2} {ctype}\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    if verbose:
        print(f"  Constraints: {path}  ({len(constraints)} connections, "
              f"{os.path.getsize(path) // 1024} KB)")
