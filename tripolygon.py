#!/usr/bin/env python3
"""
Triangle Negative/Positive Space Builder
=========================================
Standalone desktop app (matplotlib).  Run from a terminal:

    python3 tripolygon.py

Headless render for verification (flags also work interactively as defaults):

    python3 tripolygon.py --snapshot out.png [--preset obtuse] [--c 0.3] [--theta 0]

Construction
------------
  * 3 vertices P0,P1,P2 form a triangle (presets: equilateral / right /
    isosceles / scalene-acute / scalene-obtuse).
  * Center M = centroid or incenter (toggle).
  * Inner triangle:  Ti = Pi + c*(M - Pi)   (c=0 at vertex, c=1 at center).
  * For each Ti, drop a perpendicular onto the two outer edges touching Pi.
    The foot is the named point  P{i}{a}{b}  (P001, P101, ...).
  * Reflected point P' = Ti mirrored across the edge (the "imaginary point");
    the foot is the midpoint Ti<->P', matching the handwritten t_001 formula.
  * Positive space (solid material) = inner triangle + 3 corner pieces.
    Negative space (voids)          = the 3 edge rectangles/arms.

STL scaffolding is included (thickness is a parameter only -- no UI yet).
"""

import sys, os, json, warnings
import numpy as np
import matplotlib
warnings.filterwarnings("ignore", message="Mean of empty slice")

SNAP = "--snapshot" in sys.argv
if SNAP:
    matplotlib.use("Agg")            # must precede pyplot import

def _argval(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
THETA0  = _argval("--theta", 0.0)    # initial theta (mainly for headless snapshots)
PRESET0 = _argval("--preset", "obtuse")
# --c accepts a scalar (linked) or "c0,c1,c2" (unlinked per-corner)
_craw = sys.argv[sys.argv.index("--c") + 1] if "--c" in sys.argv else "0.3"
if "," in _craw:
    C_LIST = ([float(x) for x in _craw.split(",")] + [0.3, 0.3, 0.3])[:3]; LINK0 = False
else:
    C_LIST = [float(_craw)] * 3; LINK0 = True
C0 = C_LIST[0]
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Polygon as MplPoly
from matplotlib.ticker import MultipleLocator

# =====================================================================
# Geometry  (pure functions -- this is the STL-ready layer)
# =====================================================================
PRESETS = {
    "equilateral": [[0, 0], [2, 0], [1, 1.732]],
    "right":       [[0, 0], [2, 0], [0, 1.6]],
    "isosceles":   [[0, 0], [2, 0], [1, 2.2]],
    "acute":       [[0, 0], [2.2, 0], [1.4, 1.7]],
    "obtuse":      [[0, 0], [2.6, 0], [2.0, 1.1]],
}
EDGES = [(0, 1), (1, 2), (0, 2)]

def centroid(P):
    return P.mean(axis=0)

def incenter(P):
    a = np.linalg.norm(P[1] - P[2])   # opposite P0
    b = np.linalg.norm(P[0] - P[2])   # opposite P1
    c = np.linalg.norm(P[0] - P[1])   # opposite P2
    return (a * P[0] + b * P[1] + c * P[2]) / (a + b + c)

def foot(Q, A, B):
    """Foot of perpendicular from Q onto the line through A,B."""
    AB = B - A
    t = np.dot(Q - A, AB) / np.dot(AB, AB)
    return A + t * AB

def reflect(Q, A, B):
    """Q mirrored across line A,B -- the 'imaginary point' P'."""
    return 2 * foot(Q, A, B) - Q

def reflect_across(pts, A, B):
    """Mirror an array of points across the line through A,B (vectorized).
    Kept for the future multi-point / neighbor (rhombus) tab."""
    pts = np.asarray(pts, dtype=float)
    AB = B - A
    t = ((pts - A) @ AB) / (AB @ AB)
    feet = A + np.outer(t, AB)
    return 2 * feet - pts

def rotmat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def angle_at(O, A, B):
    """Angle (degrees) at O between rays O->A and O->B."""
    u, v = A - O, B - O
    cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

def _cc(c):
    """Normalize c to a 3-tuple (per-corner)."""
    return tuple(float(x) for x in c) if np.ndim(c) else (float(c),) * 3

def compute(preset, center_mode, c, theta_deg=0.0, with_jam=True):
    # --- rest pose (theta = 0) -----------------------------------------
    P0 = np.array(PRESETS[preset], dtype=float)
    M = incenter(P0) if center_mode == "incenter" else centroid(P0)
    cc = _cc(c)                                          # per-corner contraction
    T = np.array([P0[i] + cc[i] * (M - P0[i]) for i in range(3)])  # middle (FIXED frame)

    feet = {}
    for (a, b) in EDGES:
        for i in (a, b):
            name = f"{i}{a}{b}"
            feet[name] = {"pt": foot(T[i], P0[a], P0[b]),
                          "refl": reflect(T[i], P0[a], P0[b]),
                          "tIdx": i, "edge": (a, b)}

    # --- rotation: each corner rotates RIGIDLY about its hinge T_a ------
    # Middle triangle stays put.  Notes specify clockwise -> negative angle.
    # No fold clamp: rotation is free to continue past 90 deg.
    th = -np.radians(max(theta_deg, 0.0))
    R = rotmat(th)
    def rot_about(pt, piv):
        return piv + R @ (pt - piv)

    P = P0.copy()
    for a in range(3):
        P[a] = rot_about(P0[a], T[a])                    # outer vertex swings about hinge
    for f in feet.values():
        a = f["tIdx"]
        f["ptr"]   = rot_about(f["pt"],   T[a])          # rotated foot
        f["reflr"] = rot_about(f["refl"], T[a])          # rotated ghost

    # positive space (rotated): inner triangle (fixed) + corner quads
    corners = []
    for i in range(3):
        adj = [e for e in EDGES if i in e]
        f1 = feet[f"{i}{adj[0][0]}{adj[0][1]}"]["ptr"]
        f2 = feet[f"{i}{adj[1][0]}{adj[1][1]}"]["ptr"]
        corners.append({"vertex": i, "pts": np.array([P[i], f1, T[i], f2])})

    # negative space (rotated feet): edge rectangles / arms
    arms = []
    for (a, b) in EDGES:
        fa = feet[f"{a}{a}{b}"]["ptr"]
        fb = feet[f"{b}{a}{b}"]["ptr"]
        arms.append({"edge": (a, b), "pts": np.array([T[a], fa, fb, T[b]])})

    jam = collision_jam(preset, center_mode, cc) if with_jam else None
    return dict(P=P, P0=P0, M=M, center=center_mode, c=cc, theta=theta_deg, jam=jam,
                T=T, feet=feet,
                positive=dict(inner=T.copy(), corners=corners),
                negative=dict(arms=arms))

# =====================================================================
# Collision-based jamming: smallest theta at which solid pieces overlap.
# Checks BOTH corner-vs-corner (edge test; corners share no vertices) AND
# corner-vs-middle-triangle (AREA test, since a corner is hinged to the
# middle at Ti and the shared vertex must not count as a collision).
# The middle is a solid too, and corners hit it earliest -- that's the
# real range-of-motion limit (~ where the pores close).
# Cached per (preset, center, c) so dragging theta stays cheap.
# =====================================================================
_JAM_CACHE = {}

_EPS = 1e-7
def _opp(u, v):                                   # strictly opposite signs (not near-zero)
    return (u > _EPS and v < -_EPS) or (u < -_EPS and v > _EPS)

def _seg_cross(a, b, c, d):
    # bounding-box reject first (kills near-collinear false positives)
    if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
        return False
    if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
        return False
    def ccw(p, q, r): return (r[1]-p[1])*(q[0]-p[0]) - (q[1]-p[1])*(r[0]-p[0])
    return _opp(ccw(c, d, a), ccw(c, d, b)) and _opp(ccw(a, b, c), ccw(a, b, d))

def _pt_in(p, poly):
    x, y = p; inside = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-12) + xi):
            inside = not inside
        j = i
    return inside

def _poly_overlap(A, B):
    # polygon-level bounding-box reject
    if A[:, 0].max() < B[:, 0].min() or B[:, 0].max() < A[:, 0].min():
        return False
    if A[:, 1].max() < B[:, 1].min() or B[:, 1].max() < A[:, 1].min():
        return False
    nA, nB = len(A), len(B)
    for i in range(nA):
        for j in range(nB):
            if _seg_cross(A[i], A[(i+1) % nA], B[j], B[(j+1) % nB]):
                return True
    return _pt_in(A[0], B) or _pt_in(B[0], A)

def _clip_area_tri(subject, tri, eps=1e-12):
    """Area of `subject` polygon clipped to the convex triangle `tri`
    (Sutherland-Hodgman).  Used for corner-vs-middle overlap: an AREA test
    cleanly ignores the shared hinge vertex (a point has zero area)."""
    T = [list(p) for p in tri]
    if (T[1][0]-T[0][0])*(T[2][1]-T[0][1]) - (T[1][1]-T[0][1])*(T[2][0]-T[0][0]) < 0:
        T = T[::-1]                                   # ensure CCW
    def inside(p, a, b): return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -eps
    def inter(p1, p2, a, b):
        d1 = (b[0]-a[0])*(p1[1]-a[1]) - (b[1]-a[1])*(p1[0]-a[0])
        d2 = (b[0]-a[0])*(p2[1]-a[1]) - (b[1]-a[1])*(p2[0]-a[0])
        t = d1/(d1-d2) if (d1-d2) != 0 else 0.0
        return [p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1])]
    poly = [list(p) for p in subject]
    for i in range(3):
        a, b = T[i], T[(i+1) % 3]; new = []
        for j in range(len(poly)):
            cur, nxt = poly[j], poly[(j+1) % len(poly)]
            if inside(cur, a, b):
                new.append(cur)
                if not inside(nxt, a, b): new.append(inter(cur, nxt, a, b))
            elif inside(nxt, a, b):
                new.append(inter(cur, nxt, a, b))
        poly = new
        if not poly: return 0.0
    p = np.array(poly); x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def collision_jam(preset, center, c, step=1.0, limit=180.0):
    cc = _cc(c)
    key = (preset, center, tuple(round(x, 3) for x in cc))
    if key in _JAM_CACHE:
        return _JAM_CACHE[key]
    res, th = None, step
    while th <= limit:
        G = compute(preset, center, cc, th, with_jam=False)
        cs = [cor["pts"] for cor in G["positive"]["corners"]]
        mid = G["positive"]["inner"]
        corner_corner = (_poly_overlap(cs[0], cs[1]) or _poly_overlap(cs[1], cs[2])
                         or _poly_overlap(cs[0], cs[2]))
        corner_middle = any(_clip_area_tri(c2, mid) > 1e-3 for c2 in cs)
        if corner_corner or corner_middle:
            res = th; break
        th += step
    _JAM_CACHE[key] = res
    return res

# =====================================================================
# Poisson's ratio readout
# ---------------------------------------------------------------------
# Per direction i:  axial strain = change in apex-height (P_i -> midpoint
# of opposite edge);  transverse strain = change in that opposite edge's
# width.  nu_i = -(transverse strain)/(axial strain).  Undefined (nan) at
# rest (theta=0) where both strains are zero.
# =====================================================================
_NU_CACHE = {}
NU_CAP = 8.0   # |ν| above this is an asymptote artifact (ε_axial → 0); mask it

def _cap_nu(a):
    """Mask out asymptote spikes (|ν| > NU_CAP) so curves/metrics stay readable."""
    a = np.asarray(a, dtype=float)
    return np.where(np.abs(a) > NU_CAP, np.nan, a)

def poisson(G):
    P0, Pd = G["P0"], G["P"]
    nus = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        m0 = (P0[j] + P0[k]) / 2.0; md = (Pd[j] + Pd[k]) / 2.0
        w0 = np.linalg.norm(P0[j] - P0[k]); wd = np.linalg.norm(Pd[j] - Pd[k])
        h0 = np.linalg.norm(P0[i] - m0);    hd = np.linalg.norm(Pd[i] - md)
        et = (wd - w0) / w0 if w0 > 1e-12 else 0.0     # transverse strain
        ea = (hd - h0) / h0 if h0 > 1e-12 else 0.0     # axial strain
        nus.append(-et / ea if abs(ea) > 1e-6 else np.nan)
    return nus

def nu_curve(preset, center, c, step=2.0, limit=180.0):
    cc = _cc(c)
    key = (preset, center, tuple(round(x, 3) for x in cc))
    if key in _NU_CACHE:
        return _NU_CACHE[key]
    th = np.arange(step, limit + 1e-9, step)
    arr = np.full((len(th), 3), np.nan)
    for idx, t in enumerate(th):
        arr[idx] = poisson(compute(preset, center, cc, t, with_jam=False))
    _NU_CACHE[key] = (th, arr)
    return _NU_CACHE[key]

# =====================================================================
# Surface-area readout
# ---------------------------------------------------------------------
# solid = positive material (rigid -> constant);  void = negative space
# (shrinks with theta);  bbox = axis-aligned bounding box of the solid
# shape (its footprint -- shrinks/rotates as the pinwheel compresses).
# =====================================================================
_AREA_CACHE = {}

def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def material_points(G):
    return np.vstack([G["positive"]["inner"]] + [c["pts"] for c in G["positive"]["corners"]])

def _convex_hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 9))))
    if len(pts) <= 2:
        return np.array(pts, dtype=float)
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)

def min_area_rect(points):
    """Minimum-area oriented bounding box: returns (4x2 corners, area)."""
    h = _convex_hull(points)
    if len(h) < 3:
        x0, y0 = points.min(axis=0); x1, y1 = points.max(axis=0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), (x1-x0)*(y1-y0)
    best = None
    n = len(h)
    for i in range(n):
        e = h[(i+1) % n] - h[i]
        ang = np.arctan2(e[1], e[0])
        ca, sa = np.cos(-ang), np.sin(-ang)
        R = np.array([[ca, -sa], [sa, ca]])
        rot = (h - h[i]) @ R.T
        mn = rot.min(axis=0); mx = rot.max(axis=0)
        area = (mx[0]-mn[0]) * (mx[1]-mn[1])
        if best is None or area < best[0]:
            local = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
            cb, sb = np.cos(ang), np.sin(ang)
            Rb = np.array([[cb, -sb], [sb, cb]])
            best = (area, local @ Rb.T + h[i])
    return best[1], best[0]

def areas(G):
    solid = poly_area(G["positive"]["inner"]) + \
            sum(poly_area(c["pts"]) for c in G["positive"]["corners"])
    void = sum(poly_area(a["pts"]) for a in G["negative"]["arms"])
    total = solid + void
    obb, obb_area = min_area_rect(material_points(G))
    return dict(solid=solid, void=void, phi=(void / total if total > 1e-12 else 0.0),
                obb=obb, obb_area=obb_area)

def porosity_curve(preset, center, c, step=2.0, limit=180.0):
    cc = _cc(c)
    key = (preset, center, tuple(round(x, 3) for x in cc))
    if key in _AREA_CACHE:
        return _AREA_CACHE[key]
    th = np.arange(0.0, limit + 1e-9, step)
    phi = np.empty(len(th)); vd = np.empty(len(th))
    for i, t in enumerate(th):
        A = areas(compute(preset, center, cc, t, with_jam=False))
        phi[i] = A["phi"]; vd[i] = A["void"]
    _AREA_CACHE[key] = (th, phi, vd)
    return _AREA_CACHE[key]

# =====================================================================
# STL scaffolding  (2D solid -> 3D mesh).  Thickness is a PARAMETER ONLY
# for now -- no UI control yet, by design.  Wire it up later.
# =====================================================================
STL_THICKNESS = 1.0   # TODO: replace with a UI variable later

def solid_polygons(G):
    return [G["positive"]["inner"]] + [c["pts"] for c in G["positive"]["corners"]]

def polygons_to_stl(polys, thickness, name="triangle_part"):
    out = [f"solid {name}"]
    def facet(a, b, c):
        n = np.cross(b - a, c - a)
        L = np.linalg.norm(n) or 1.0
        n = n / L
        out.append(f"  facet normal {n[0]} {n[1]} {n[2]}")
        out.append("    outer loop")
        for p in (a, b, c):
            out.append(f"      vertex {p[0]} {p[1]} {p[2]}")
        out.append("    endloop")
        out.append("  endfacet")
    for poly in polys:
        m = len(poly)
        top = np.column_stack([poly[:, 0], poly[:, 1], np.full(m, thickness)])
        bot = np.column_stack([poly[:, 0], poly[:, 1], np.zeros(m)])
        for i in range(1, m - 1):
            facet(top[0], top[i], top[i + 1])         # top cap
        for i in range(1, m - 1):
            facet(bot[0], bot[i + 1], bot[i])         # bottom cap (reversed)
        for i in range(m):                            # side walls
            j = (i + 1) % m
            facet(bot[i], bot[j], top[j])
            facet(bot[i], top[j], top[i])
    out.append(f"endsolid {name}")
    return "\n".join(out) + "\n"

# =====================================================================
# Deep analysis: Centroid vs Incenter comparison dashboard
# =====================================================================
def _sweep_center(preset, cc, center, thetas):
    nus = np.empty((len(thetas), 3)); phis = np.empty(len(thetas))
    for i, t in enumerate(thetas):
        G = compute(preset, center, cc, t, with_jam=False)
        nus[i] = poisson(G); phis[i] = areas(G)["phi"]
    return nus, phis, collision_jam(preset, center, cc)

def _phi_grid(preset, center, cs, thetas):
    Z = np.empty((len(cs), len(thetas)))
    for i, cv in enumerate(cs):
        for j, t in enumerate(thetas):
            Z[i, j] = areas(compute(preset, center, cv, t, with_jam=False))["phi"]
    return Z

def per_corner_sweep(preset, center, cc, theta, var, csw):
    """Vary ONE corner's c across csw (holding the other two at cc), per corner.
    Returns array (3, len(csw)) of the chosen variable ('nu' mean, or 'phi')."""
    out = np.full((3, len(csw)), np.nan)
    for i in range(3):
        for k, cv in enumerate(csw):
            ccx = list(cc); ccx[i] = cv
            G = compute(preset, center, ccx, theta, with_jam=False)
            out[i, k] = _cap_nu(np.nanmean(poisson(G))) if var == "nu" else areas(G)["phi"]
    return out

def _center_metrics(thetas, nus, phis, jam):
    mean_nu = _cap_nu(np.nanmean(nus, axis=1))        # drop asymptote spikes
    mask = thetas >= 5.0                              # avoid the theta~0 blow-up
    mvals = mean_nu[mask & np.isfinite(mean_nu)]
    aux = thetas[mask & np.isfinite(mean_nu)][mvals < 0] if mvals.size else np.array([])
    auxwin = (aux.min(), aux.max()) if aux.size else None
    if mvals.size:
        sub_t = thetas[mask & np.isfinite(mean_nu)]
        k = int(np.argmin(mvals)); numin = float(mvals[k]); th_numin = float(sub_t[k])
    else:
        numin, th_numin = np.nan, np.nan
    jp = int(np.argmin(phis))
    return dict(phi_rest=float(phis[0]), phi_min=float(phis[jp]), th_phimin=float(thetas[jp]),
                stroke=float(phis[0] - phis[jp]), numin=numin, th_numin=th_numin,
                auxwin=auxwin, jam=jam)

def write_analysis_csv(preset, cc, path):
    thetas = np.arange(0.0, 180.0001, 2.0)
    cn, pc, _ = _sweep_center(preset, cc, "centroid", thetas)
    ci, pi, _ = _sweep_center(preset, cc, "incenter", thetas)
    lines = ["theta,nu0_cen,nu1_cen,nu2_cen,phi_cen,nu0_inc,nu1_inc,nu2_inc,phi_inc"]
    for k, t in enumerate(thetas):
        lines.append(f"{t:.1f},{cn[k,0]:.5f},{cn[k,1]:.5f},{cn[k,2]:.5f},{pc[k]:.5f},"
                     f"{ci[k,0]:.5f},{ci[k,1]:.5f},{ci[k,2]:.5f},{pi[k]:.5f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

def build_analysis_figure(preset, cc, theta=0.0, cur_center="centroid"):
    thetas = np.arange(0.0, 180.0001, 2.0)
    centers = [("centroid", "#2f6fdb", "-"), ("incenter", "#d9534f", "--")]
    data = {nm: _sweep_center(preset, cc, nm, thetas) for nm, _, _ in centers}
    met = {nm: _center_metrics(thetas, *data[nm]) for nm, _, _ in centers}

    figA = plt.figure(figsize=(15, 9))
    cstr = "/".join(f"{x:.2f}" for x in cc)
    figA.suptitle(f"Deep Analysis — {preset} triangle, c = {cstr}   ·   Centroid vs Incenter",
                  fontsize=13, fontweight="bold")

    # 1) porosity vs theta
    ax1 = figA.add_subplot(2, 3, 1)
    for nm, col, ls in centers:
        nus, phis, jam = data[nm]
        ax1.plot(thetas, phis, color=col, ls=ls, label=nm)
        if jam is not None: ax1.axvline(jam, color=col, ls=":", lw=0.8, alpha=0.6)
        m = met[nm]; ax1.plot(m["th_phimin"], m["phi_min"], "o", color=col, ms=4)
    ax1.axvline(theta, color="black", lw=1.1, alpha=0.7)        # you-are-here (current θ)
    ax1.set_xlabel("θ (deg)"); ax1.set_ylabel("porosity φ")
    ax1.set_title(f"Porosity vs θ   (○ pore-closed · ▏θ={theta:.0f}°)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 2) mean nu vs theta
    ax2 = figA.add_subplot(2, 3, 2)
    allm = []
    for nm, col, ls in centers:
        nus, phis, jam = data[nm]; m = _cap_nu(np.nanmean(nus, axis=1))
        ax2.plot(thetas, m, color=col, ls=ls, label=nm); allm.append(m)
        if jam is not None: ax2.axvline(jam, color=col, ls=":", lw=0.8, alpha=0.6)
    ax2.axhline(0, color="#9aa3b2", lw=0.8)
    fin = np.concatenate([m[np.isfinite(m)] for m in allm]) if allm else np.array([0.0])
    if fin.size:
        lo, hi = np.percentile(fin, [3, 97])
        lo, hi = min(lo, -1.2), max(hi, 0.2)
        ax2.set_ylim(lo, hi); ax2.axhspan(lo, 0, color="#eaf3ea", zorder=0)
    ax2.axvline(theta, color="black", lw=1.1, alpha=0.7)        # current θ
    ax2.set_xlabel("θ (deg)"); ax2.set_ylabel("mean ν")
    ax2.set_title("Poisson ν vs θ  (green = auxetic)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    if abs(met["centroid"]["numin"] + 1) < 1e-6 and abs(met["incenter"]["numin"] + 1) < 1e-6:
        ax2.text(0.5, 0.5, "uniform c → ν = −1 for both\n(unlink c0/c1/c2 to see a center effect)",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=8, color="#777",
                 bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    # 3) phi-nu tradeoff (parametric in theta)
    ax3 = figA.add_subplot(2, 3, 3)
    for nm, col, ls in centers:
        nus, phis, jam = data[nm]; m = _cap_nu(np.nanmean(nus, axis=1))
        ax3.plot(m, phis, color=col, ls=ls, label=nm)
        Gc = compute(preset, nm, cc, theta, with_jam=False)        # current-θ point
        ax3.plot(_cap_nu(np.nanmean(poisson(Gc))), areas(Gc)["phi"], "o",
                 color=col, ms=7, mec="black", mew=0.6, zorder=5)
    ax3.set_xlim(-NU_CAP, NU_CAP)
    ax3.axvline(0, color="#9aa3b2", lw=0.8)
    ax3.set_xlabel("mean ν"); ax3.set_ylabel("porosity φ")
    ax3.set_title("φ–ν tradeoff (param θ; ● = current θ)"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # 4,5) per-corner c impact: vary one corner's c (others held), at current θ
    csw = np.linspace(0.02, 0.98, 30)
    Gcur = compute(preset, cur_center, cc, theta, with_jam=False)
    nu_cur = _cap_nu(np.nanmean(poisson(Gcur))); phi_cur = areas(Gcur)["phi"]
    cols = ["#d9534f", "#3a9d4a", "#2f6fdb"]

    ax4 = figA.add_subplot(2, 3, 4)
    NU = per_corner_sweep(preset, cur_center, cc, theta, "nu", csw)
    for i, col in enumerate(cols):
        ax4.plot(csw, NU[i], color=col, label=f"vary c{i} (now {cc[i]:.2f})")
        ax4.plot(cc[i], nu_cur, "o", color=col, ms=5, mec="black", mew=0.5, zorder=5)
    ax4.axhline(0, color="#9aa3b2", lw=0.8)
    fin = NU[np.isfinite(NU)]
    if fin.size:
        lo, hi = np.percentile(fin, [2, 98]); ax4.set_ylim(min(lo, -1.2), max(hi, 0.3))
    else:
        ax4.text(0.5, 0.5, "ν is undefined at θ = 0°\n(no deformation yet —\nraise θ and re-Analyze)",
                 transform=ax4.transAxes, ha="center", va="center", fontsize=9, color="#777",
                 bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax4.set_xlabel("corner c"); ax4.set_ylabel("mean ν")
    ax4.set_title(f"ν vs each corner c  (θ={theta:.0f}°, {cur_center})")
    ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

    ax5 = figA.add_subplot(2, 3, 5)
    PH = per_corner_sweep(preset, cur_center, cc, theta, "phi", csw)
    for i, col in enumerate(cols):
        ax5.plot(csw, PH[i], color=col, label=f"vary c{i} (now {cc[i]:.2f})")
        ax5.plot(cc[i], phi_cur, "o", color=col, ms=5, mec="black", mew=0.5, zorder=5)
    ax5.set_xlabel("corner c"); ax5.set_ylabel("porosity φ")
    ax5.set_title(f"φ vs each corner c  (θ={theta:.0f}°, {cur_center})")
    ax5.legend(fontsize=7); ax5.grid(alpha=0.3)

    # 6) summary table
    ax6 = figA.add_subplot(2, 3, 6); ax6.axis("off")
    def fz(v, f):
        return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f.format(v)
    rows = [("metric", "centroid", "incenter")]
    for label, key, f in [("φ at rest", "phi_rest", "{:.3f}"),
                          ("min φ", "phi_min", "{:.3f}"),
                          ("pore-closed θ", "th_phimin", "{:.0f}"),
                          ("valve stroke Δφ", "stroke", "{:.3f}"),
                          ("ν_min (mean)", "numin", "{:+.2f}"),
                          ("θ at ν_min", "th_numin", "{:.0f}"),
                          ("jam angle", "jam", "{:.0f}")]:
        rows.append((label, fz(met["centroid"][key], f), fz(met["incenter"][key], f)))
    wstr = lambda w: "none" if w is None else f"{w[0]:.0f}-{w[1]:.0f}"
    rows.append(("auxetic window", wstr(met["centroid"]["auxwin"]), wstr(met["incenter"]["auxwin"])))
    txt = "\n".join(f"{a:<16}{b:>11}{c:>11}" for a, b, c in rows)
    ax6.text(0.0, 0.95, txt, family="monospace", fontsize=10, va="top", transform=ax6.transAxes)
    ax6.set_title("Centroid vs Incenter — summary", fontsize=10, loc="left")

    figA.tight_layout(rect=[0, 0, 1, 0.96])
    return figA

def pair_grid(preset, center, cc, theta, var, pair, n=24):
    """var ('nu' mean or 'phi') over a grid of two corners' c (third held at cc)."""
    a, b = pair
    cs = np.linspace(0.05, 0.95, n)
    Z = np.full((n, n), np.nan)
    for ja, ca in enumerate(cs):                 # rows  = corner a (y)
        for jb, cb in enumerate(cs):             # cols  = corner b (x)
            ccx = list(cc); ccx[a] = ca; ccx[b] = cb
            G = compute(preset, center, ccx, theta, with_jam=False)
            Z[ja, jb] = _cap_nu(np.nanmean(poisson(G))) if var == "nu" else areas(G)["phi"]
    return cs, Z

def build_cmap_figure(preset, cc, theta=0.0, cur_center="centroid"):
    pairs = [(0, 1), (1, 2), (0, 2)]
    figC = plt.figure(figsize=(15, 9))
    cstr = "/".join(f"{x:.2f}" for x in cc)
    figC.suptitle(f"Corner-c interaction maps — {preset}, c = {cstr}, θ = {theta:.0f}°, "
                  f"{cur_center}   (★ = current; ν: blue=auxetic)", fontsize=12, fontweight="bold")
    for col, (a, b) in enumerate(pairs):
        cs, ZN = pair_grid(preset, cur_center, cc, theta, "nu", (a, b))
        axn = figC.add_subplot(2, 3, 1 + col)
        cmap_nu = plt.get_cmap("coolwarm").copy(); cmap_nu.set_bad("#e6e6e6")  # nan = gray
        im = axn.imshow(ZN, origin="lower", extent=[cs[0], cs[-1], cs[0], cs[-1]],
                        aspect="auto", cmap=cmap_nu, vmin=-NU_CAP, vmax=NU_CAP)
        axn.plot(cc[b], cc[a], "*", color="yellow", ms=14, mec="black", mew=0.6)
        axn.set_xlabel(f"c{b}"); axn.set_ylabel(f"c{a}"); axn.set_title(f"ν  over (c{a}, c{b})")
        figC.colorbar(im, ax=axn, fraction=0.046, pad=0.04)
        if not np.isfinite(ZN).any():
            axn.text(0.5, 0.5, "ν undefined at θ = 0°\n(raise θ and re-Analyze)",
                     transform=axn.transAxes, ha="center", va="center", fontsize=10,
                     color="#555", bbox=dict(boxstyle="round", fc="white", ec="#bbb"))

        cs, ZP = pair_grid(preset, cur_center, cc, theta, "phi", (a, b))
        axp = figC.add_subplot(2, 3, 4 + col)
        im2 = axp.imshow(ZP, origin="lower", extent=[cs[0], cs[-1], cs[0], cs[-1]],
                         aspect="auto", cmap="viridis", vmin=0, vmax=0.6)
        axp.plot(cc[b], cc[a], "*", color="yellow", ms=14, mec="black", mew=0.6)
        axp.set_xlabel(f"c{b}"); axp.set_ylabel(f"c{a}"); axp.set_title(f"φ  over (c{a}, c{b})")
        figC.colorbar(im2, ax=axp, fraction=0.046, pad=0.04)
    figC.tight_layout(rect=[0, 0, 1, 0.96])
    return figC

# =====================================================================
# App state
# =====================================================================
state = dict(
    preset=PRESET0, center="centroid", c_arr=list(C_LIST), link=LINK0, theta=THETA0,
    show=dict(pos=True, neg=True, perp=True, spokes=True, refl=False, labels=True,
              bbox=True, rest=True),
)
G_current = None

def eff_c():
    """Effective per-corner c: master (c0) for all when linked, else the 3 values."""
    a = state["c_arr"]
    return (a[0], a[0], a[0]) if state["link"] else tuple(a)

POS_FILL = (0.78, 0.86, 0.98, 0.85)

# =====================================================================
# Figure + widgets
# =====================================================================
fig = plt.figure(figsize=(13, 8))
if not SNAP:
    try: fig.canvas.manager.set_window_title("Triangle Space Builder")
    except Exception: pass

ax       = fig.add_axes([0.06, 0.08, 0.55, 0.86])
ax_table = fig.add_axes([0.82, 0.06, 0.17, 0.42]); ax_table.axis("off")

ax_preset = fig.add_axes([0.64, 0.72, 0.15, 0.20])
_ORDER = ("equilateral", "right", "isosceles", "acute", "obtuse")
rb_preset = RadioButtons(ax_preset, _ORDER, active=_ORDER.index(PRESET0))
ax_preset.set_title("Triangle", fontsize=9, loc="left")

ax_center = fig.add_axes([0.81, 0.79, 0.14, 0.12])
rb_center = RadioButtons(ax_center, ("centroid", "incenter"), active=0)
ax_center.set_title("Center", fontsize=9, loc="left")

# per-corner c (c0 acts as master when "link" is on)
ax_c0 = fig.add_axes([0.665, 0.665, 0.21, 0.020]); s_c0 = Slider(ax_c0, "c0", 0.0, 1.0, valinit=C_LIST[0], valstep=0.005)
ax_c1 = fig.add_axes([0.665, 0.630, 0.21, 0.020]); s_c1 = Slider(ax_c1, "c1", 0.0, 1.0, valinit=C_LIST[1], valstep=0.005)
ax_c2 = fig.add_axes([0.665, 0.595, 0.21, 0.020]); s_c2 = Slider(ax_c2, "c2", 0.0, 1.0, valinit=C_LIST[2], valstep=0.005)
ax_link = fig.add_axes([0.905, 0.60, 0.085, 0.072]); chk_link = CheckButtons(ax_link, ["link"], [LINK0])

ax_th = fig.add_axes([0.665, 0.540, 0.21, 0.020])
s_th  = Slider(ax_th, "θ°", 0.0, 180.0, valinit=THETA0, valstep=0.5, color="#d9534f")

ax_chk = fig.add_axes([0.625, 0.265, 0.17, 0.215])
CHK_LABELS = ["positive", "negative", "perpendiculars", "spokes", "reflected P'",
              "labels", "bbox", "rest overlay"]
chk = CheckButtons(ax_chk, CHK_LABELS,
                   [True, True, True, True, False, True, True, True])
ax_chk.set_title("Display", fontsize=9, loc="left")

ax_json = fig.add_axes([0.625, 0.240, 0.052, 0.038]); b_json = Button(ax_json, "JSON")
ax_stl  = fig.add_axes([0.683, 0.240, 0.045, 0.038]); b_stl  = Button(ax_stl,  "STL")
ax_an   = fig.add_axes([0.734, 0.240, 0.062, 0.038]); b_an   = Button(ax_an,  "Analyze")

ax_nu = fig.add_axes([0.625, 0.135, 0.165, 0.060])     # nu-vs-theta mini plot
ax_ar = fig.add_axes([0.625, 0.045, 0.165, 0.060])     # area-vs-theta mini plot

CHK_KEY = {"positive": "pos", "negative": "neg", "perpendiculars": "perp",
           "spokes": "spokes", "reflected P'": "refl", "labels": "labels",
           "bbox": "bbox", "rest overlay": "rest"}

# =====================================================================
# Rendering
# =====================================================================
def update_table(G):
    ax_table.cla(); ax_table.axis("off")
    ax_table.set_title("Points involved", fontsize=9, loc="left")
    rows = []
    for i, p in enumerate(G["P"]):  rows.append((f"P{i}", p))
    rows.append(("I" if state["center"] == "incenter" else "G", G["M"]))
    for i, t in enumerate(G["T"]):  rows.append((f"T{i}", t))
    for name, f in G["feet"].items():  rows.append((f"P{name}", f["ptr"]))
    if state["show"]["refl"]:
        for name, f in G["feet"].items():  rows.append((f"P'{name}", f["reflr"]))
    txt = f"{'pt':<7}{'x':>8}{'y':>8}\n" + "\n".join(
        f"{n:<7}{p[0]:>8.3f}{p[1]:>8.3f}" for n, p in rows)
    ax_table.text(0, 1, txt, family="monospace", fontsize=7, va="top",
                  transform=ax_table.transAxes)

def redraw(_=None):
    global G_current
    cc = eff_c()
    G = compute(state["preset"], state["center"], cc, state["theta"])
    G_current = G
    sh = state["show"]
    P, M, T = G["P"], G["M"], G["T"]
    G0 = compute(state["preset"], state["center"], cc, 0.0, with_jam=False)  # rest pose
    jam = G["jam"]
    jammed = (jam is not None) and (state["theta"] >= jam - 1e-6)
    jam_txt = f"jam {jam:.0f}°" if jam is not None else "jam none<180°"

    ax.cla()
    ax.set_aspect("equal")
    ax.set_facecolor("#fdecec" if jammed else "white")
    base = G["P0"]                                   # stable limits from rest pose
    mn = base.min(axis=0) - 0.5; mx = base.max(axis=0) + 0.5
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(True, color="#edf0f4", lw=1)
    ax.tick_params(labelsize=9, colors="#5b626c")
    a = state["c_arr"]
    cstr = f"{a[0]:.3f}" if state["link"] else f"[{a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}]"
    ax.set_title(f"{state['preset']}   ·   c = {cstr}   ·   {state['center']}   ·   "
                 f"θ = {state['theta']:.1f}° ({jam_txt})"
                 f"{'  — JAMMED' if jammed else ''}   ·   n = {len(P)}",
                 loc="left", fontsize=11)

    # rest-pose overlay (theta = 0) so the rotation is visible against it
    if sh["rest"]:
        ax.add_patch(MplPoly(G0["P"], closed=True, fill=False, edgecolor="#c2c5cb",
                             lw=1.0, ls=(0, (4, 3)), zorder=1))
        ax.add_patch(MplPoly(G0["positive"]["inner"], closed=True, fill=False,
                             edgecolor="#ccced3", lw=0.8, ls=(0, (2, 2)), zorder=1))
        for cor in G0["positive"]["corners"]:
            ax.add_patch(MplPoly(cor["pts"], closed=True, fill=False,
                                 edgecolor="#ccced3", lw=0.8, ls=(0, (2, 2)), zorder=1))

    # outer triangle
    ax.add_patch(MplPoly(P, closed=True, fill=False, edgecolor="#c2c8d0", lw=1.4))

    # spokes
    if sh["spokes"]:
        for p in P:
            ax.plot([p[0], M[0]], [p[1], M[1]], color="#e3e7ee", lw=1, zorder=1)

    # negative space (hatched)
    if sh["neg"]:
        for arm in G["negative"]["arms"]:
            ax.add_patch(MplPoly(arm["pts"], closed=True, facecolor="none",
                                 hatch="////", edgecolor="#9aa3b2", lw=0.8, zorder=2))

    # positive space (solid)
    if sh["pos"]:
        for cor in G["positive"]["corners"]:
            ax.add_patch(MplPoly(cor["pts"], closed=True, facecolor=POS_FILL,
                                 edgecolor="#2f6fdb", lw=1.5, zorder=3))
        ax.add_patch(MplPoly(G["positive"]["inner"], closed=True, facecolor=POS_FILL,
                             edgecolor="#2f6fdb", lw=2, zorder=3))

    # oriented (min-area) bounding box of the rotating solid shape
    A = areas(G)
    if sh["bbox"]:
        def _box(corners, color, ls, lw, alpha=1.0):
            c = np.vstack([corners, corners[0]])
            ax.plot(c[:, 0], c[:, 1], color=color, ls=ls, lw=lw, alpha=alpha, zorder=2.5)
        if sh["rest"]:
            _box(areas(G0)["obb"], "#caa6e0", (0, (4, 3)), 1.0)    # rest footprint
        _box(A["obb"], "#8e44ad", "-", 1.4)                        # current footprint

    # perpendiculars + feet
    if sh["perp"]:
        for f in G["feet"].values():
            t, pt = T[f["tIdx"]], f["ptr"]
            ax.plot([t[0], pt[0]], [t[1], pt[1]], color="#3a9d4a", lw=1.5, zorder=4)
            ax.plot(pt[0], pt[1], "o", color="#3a9d4a", ms=4, zorder=5)

    # reflected ghost points P'
    if sh["refl"]:
        for f in G["feet"].values():
            t, r = T[f["tIdx"]], f["reflr"]
            ax.plot([t[0], r[0]], [t[1], r[1]], color="#969da8", lw=1,
                    ls=(0, (3, 3)), alpha=0.6, zorder=3)
            ax.plot(r[0], r[1], "o", mfc="white", mec="#aab0ba", ms=5, zorder=4)

    # vertices / center / T
    for p in P:           ax.plot(p[0], p[1], "o", color="#1d1f23", ms=7, zorder=6)
    ax.plot(M[0], M[1], "o", color="#d9534f", ms=6, zorder=6)
    for t in T:           ax.plot(t[0], t[1], "o", color="#2f6fdb", ms=5, zorder=6)

    # labels (mathtext subscripts)
    if sh["labels"]:
        for i, p in enumerate(P):
            ax.annotate(f"$P_{i}$", p, textcoords="offset points", xytext=(6, 6), fontsize=11)
        for i, t in enumerate(T):
            ax.annotate(f"$T_{i}$", t, textcoords="offset points", xytext=(5, 5),
                        color="#2f6fdb", fontsize=10)
        ax.annotate("$I$" if state["center"] == "incenter" else "$G$", M,
                    textcoords="offset points", xytext=(6, -11), color="#d9534f", fontsize=10)
        if sh["perp"]:
            for name, f in G["feet"].items():
                ax.annotate(f"$P_{{{name}}}$", f["ptr"], textcoords="offset points",
                            xytext=(4, -12), color="#3a9d4a", fontsize=8)
        if sh["refl"]:
            for name, f in G["feet"].items():
                ax.annotate(f"$P'_{{{name}}}$", f["reflr"], textcoords="offset points",
                            xytext=(4, 4), color="#9aa1ac", fontsize=8)

    # --- Poisson's ratio: numeric readout on the plot ------------------
    nus = poisson(G)
    fmt = lambda v: "—" if np.isnan(v) else f"{v:+.2f}"
    valid = [v for v in nus if not np.isnan(v)]
    meanv = np.mean(valid) if valid else np.nan
    ax.text(0.015, 0.03,
            f"$\\nu_0$={fmt(nus[0])}    $\\nu_1$={fmt(nus[1])}    "
            f"$\\nu_2$={fmt(nus[2])}    mean={fmt(meanv)}",
            transform=ax.transAxes, fontsize=10, color="#1d1f23",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d9dde3", alpha=0.9))

    # --- porosity readout ----------------------------------------------
    A0 = areas(G0)
    dphi = (A["phi"] - A0["phi"]) * 100.0          # change in percentage points
    ax.text(0.015, 0.105,
            f"void area={A['void']:.3f}    porosity φ={A['phi']*100:.1f}%"
            f"    (Δφ {dphi:+.1f} pp vs θ0)",
            transform=ax.transAxes, fontsize=9, color="#1d1f23",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d9dde3", alpha=0.9))

    # --- Poisson's ratio: nu-vs-theta mini plot ------------------------
    ax_nu.cla()
    th, arr = nu_curve(state["preset"], state["center"], cc)
    arr = _cap_nu(arr)                                  # mask asymptote spikes
    for i, col in enumerate(["#d9534f", "#3a9d4a", "#2f6fdb"]):
        ax_nu.plot(th, arr[:, i], color=col, lw=1.0)
    ax_nu.axhline(0, color="#9aa3b2", lw=0.8, ls=":")
    ax_nu.axvline(state["theta"], color="black", lw=0.8, alpha=0.5)
    ax_nu.set_title("ν vs θ", fontsize=7, pad=2)
    ax_nu.tick_params(labelsize=6); ax_nu.set_xticklabels([])
    ax_nu.set_xlim(0, 180)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = np.percentile(finite, [2, 98])
        ax_nu.set_ylim(min(lo, -1.0), max(hi, 1.0))

    # --- porosity vs theta mini plot -----------------------------------
    ax_ar.cla()
    tha, phi, vd = porosity_curve(state["preset"], state["center"], cc)
    ax_ar.plot(tha, phi, color="#8e44ad", lw=1.2)         # porosity phi
    ax_ar.axvline(state["theta"], color="black", lw=0.8, alpha=0.5)
    ax_ar.set_title("porosity φ vs θ", fontsize=7, pad=2)
    ax_ar.tick_params(labelsize=6)
    ax_ar.set_xlim(0, 180); ax_ar.set_ylim(0, max(0.6, phi.max() * 1.1))

    update_table(G)
    if not SNAP:
        fig.canvas.draw_idle()

# =====================================================================
# Callbacks
# =====================================================================
def on_preset(v): state["preset"] = v; redraw()
def on_center(v): state["center"] = v; redraw()
def on_c0(v):     state["c_arr"][0] = float(v); redraw()
def on_c1(v):     state["c_arr"][1] = float(v); redraw()
def on_c2(v):     state["c_arr"][2] = float(v); redraw()
def on_link(_):   state["link"] = not state["link"]; redraw()
def on_theta(v):  state["theta"] = float(v); redraw()
def on_chk(label):
    k = CHK_KEY[label]
    state["show"][k] = not state["show"][k]
    redraw()

def _here(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

def on_json(_):
    G = G_current
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict):       return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):       return [conv(v) for v in o]
        return o
    path = _here("triangle_geometry.json")
    with open(path, "w") as fh:
        json.dump(conv(G), fh, indent=2)
    print("Wrote", path)

def on_stl(_):
    path = _here("triangle_part.stl")
    with open(path, "w") as fh:
        fh.write(polygons_to_stl(solid_polygons(G_current), STL_THICKNESS))
    print("Wrote", path, f"(thickness={STL_THICKNESS})")

def on_analysis(_):
    cc = eff_c()
    csv = _here("analysis_sweep.csv")
    write_analysis_csv(state["preset"], cc, csv)
    print("Wrote", csv)
    build_analysis_figure(state["preset"], cc, state["theta"], state["center"]).show()
    build_cmap_figure(state["preset"], cc, state["theta"], state["center"]).show()

rb_preset.on_clicked(on_preset)
rb_center.on_clicked(on_center)
s_c0.on_changed(on_c0); s_c1.on_changed(on_c1); s_c2.on_changed(on_c2)
chk_link.on_clicked(on_link)
s_th.on_changed(on_theta)
chk.on_clicked(on_chk)
b_json.on_clicked(on_json)
b_stl.on_clicked(on_stl)
b_an.on_clicked(on_analysis)

# =====================================================================
# Go
# =====================================================================
redraw()
if SNAP:
    out = sys.argv[sys.argv.index("--snapshot") + 1]
    if "--cmap" in sys.argv:
        build_cmap_figure(state["preset"], eff_c(), state["theta"], state["center"]).savefig(out, dpi=110)
    elif "--analysis" in sys.argv:
        build_analysis_figure(state["preset"], eff_c(), state["theta"], state["center"]).savefig(out, dpi=110)
    else:
        fig.savefig(out, dpi=110)
    print("saved", out)
else:
    plt.show()
