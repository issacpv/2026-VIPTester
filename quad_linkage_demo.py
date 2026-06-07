"""Flexing-quadrilateral (four-bar linkage) demo.

A triangle with fixed side lengths is *rigid* -- the three lengths pin every
angle. A quadrilateral with four fixed side lengths is not: it keeps a single
internal degree of freedom and can flex through a one-parameter family of
shapes. This script takes the four side lengths and animates that motion.

Model
-----
Label the corners A, B, C, D so the sides, in order, are

    AB, BC, CD, DA  =  the four input lengths.

* AB is pinned to the x-axis as the fixed "ground" link: A = (0, 0),
  B = (AB, 0).
* The single degree of freedom is the angle ``theta`` of link BC about B:
        C = B + BC * (cos theta, sin theta).
* With C placed, D must keep both remaining lengths fixed, so it lies at an
  intersection of two circles:
        circle(A, DA)   n   circle(C, CD).
  There are up to two such points (the two "assembly modes"); we follow one
  continuously so the mechanism moves smoothly instead of snapping between them.

Constraints honoured
--------------------
* Only angles where the two circles actually meet are kept -- i.e. angles at
  which the four lengths can still close into a quadrilateral. The caller is
  trusted to supply lengths that close at *some* angle, as per the brief.
* Sides may not pass through one another: any state whose quadrilateral is
  self-intersecting (a non-adjacent edge pair crosses) is discarded, so the
  animation only ever shows *simple* quadrilaterals.

Run
---
    python quad_linkage_demo.py                 # built-in example lengths
    python quad_linkage_demo.py 4 2 3.5 3       # AB BC CD DA
"""
from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def circle_intersections(
    P0: np.ndarray, r0: float, P1: np.ndarray, r1: float,
) -> list[np.ndarray]:
    """Intersection point(s) of circle(P0, r0) and circle(P1, r1): 0, 1 or 2."""
    delta = P1 - P0
    dist = float(np.hypot(*delta))
    if dist == 0.0 or dist > r0 + r1 + 1e-9 or dist < abs(r0 - r1) - 1e-9:
        return []                                  # too far apart, nested, or concentric
    a = (r0 * r0 - r1 * r1 + dist * dist) / (2.0 * dist)
    h = np.sqrt(max(r0 * r0 - a * a, 0.0))
    mid = P0 + a * delta / dist                    # foot on the line of centres
    if h < 1e-9:
        return [mid]                               # circles tangent -> single point
    perp = np.array([-delta[1], delta[0]]) / dist
    return [mid + h * perp, mid - h * perp]


def _cross(o: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """z-component of (p - o) x (q - o); the signed area test."""
    return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])


def _segments_cross(p1, p2, p3, p4) -> bool:
    """True iff open segment p1p2 *properly* crosses open segment p3p4.

    Endpoints touching (collinear / shared) do not count -- so adjacent sides,
    which legitimately meet at a corner, are never flagged.
    """
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return d1 * d2 < 0.0 and d3 * d4 < 0.0


def is_simple(A, B, C, D) -> bool:
    """A quadrilateral A-B-C-D is simple iff its two non-adjacent edge pairs
    (AB vs CD, and BC vs DA) do not cross -- i.e. no side passes through another.
    """
    return not (_segments_cross(A, B, C, D) or _segments_cross(B, C, D, A))


def linkage_states(
    AB: float, BC: float, CD: float, DA: float, n: int = 720,
) -> tuple[list[tuple[np.ndarray, ...]], bool]:
    """Sweep the degree of freedom and collect every valid, simple quadrilateral.

    Returns ``(frames, full_circle)`` where ``frames`` is the longest run of
    consecutive valid states (each a 4-tuple A, B, C, D) traced while following
    one assembly mode continuously, and ``full_circle`` is True when link BC
    sweeps a complete 360 deg rotation (letting the animation loop forward
    rather than rock back and forth).
    """
    A = np.array([0.0, 0.0])
    B = np.array([AB, 0.0])
    runs: list[list] = []
    current: list = []
    prev_D: np.ndarray | None = None
    for theta in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False):
        C = B + BC * np.array([np.cos(theta), np.sin(theta)])
        candidates = circle_intersections(A, DA, C, CD)
        if not candidates:                         # lengths can't close at this angle
            prev_D = None
            if current:
                runs.append(current)
                current = []
            continue
        # Stay on one assembly mode: pick the solution nearest the previous D.
        D = (candidates[0] if prev_D is None
             else min(candidates, key=lambda P: float(np.hypot(*(P - prev_D)))))
        prev_D = D
        if is_simple(A, B, C, D):
            current.append((A.copy(), B.copy(), C.copy(), D.copy()))
        elif current:                              # would self-intersect -> end run
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return [], False
    best = max(runs, key=len)
    return best, len(best) == n


def main() -> None:
    args = sys.argv[1:]
    if len(args) >= 4:
        AB, BC, CD, DA = (float(x) for x in args[:4])
    else:
        AB, BC, CD, DA = 4.0, 2.0, 3.5, 3.0        # a Grashof crank-rocker

    frames, full_circle = linkage_states(AB, BC, CD, DA)
    if not frames:
        print("No valid, non-self-intersecting quadrilateral for those lengths.")
        return
    # Loop forward through a full rotation; otherwise rock back and forth.
    seq = frames if full_circle else frames + frames[::-1]
    print(f"{len(frames)} distinct states "
          f"({'full 360 deg rotation' if full_circle else 'rocking arc'}).")

    pts = np.array([v for f in frames for v in f])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.1 * float(np.max(hi - lo)) + 1e-9

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_title(f"Flexing quadrilateral   "
                 f"AB={AB:g}  BC={BC:g}  CD={CD:g}  DA={DA:g}")

    # Faint loci of the two moving corners C and D -- the family of states swept.
    C_tr = np.array([f[2] for f in frames])
    D_tr = np.array([f[3] for f in frames])
    ax.plot(C_tr[:, 0], C_tr[:, 1], lw=0.7, color="tab:blue", alpha=0.3)
    ax.plot(D_tr[:, 0], D_tr[:, 1], lw=0.7, color="tab:red", alpha=0.3)

    quad = plt.Polygon(np.array(frames[0]), closed=True,
                       fc="0.8", ec="black", lw=2.0, alpha=0.9)
    ax.add_patch(quad)
    # Emphasise the fixed ground link AB.
    A0, B0 = frames[0][0], frames[0][1]
    ax.plot([A0[0], B0[0]], [A0[1], B0[1]],
            color="black", lw=3.5, solid_capstyle="round", zorder=1)
    corners, = ax.plot([], [], "o", color="black", zorder=3)
    labels = [ax.text(0.0, 0.0, s, fontsize=12, fontweight="bold") for s in "ABCD"]

    def update(i: int):
        verts = seq[i]
        quad.set_xy(np.array(verts))
        corners.set_data([p[0] for p in verts], [p[1] for p in verts])
        for txt, p in zip(labels, verts):
            txt.set_position((p[0] + 0.15 * pad, p[1] + 0.15 * pad))
        return quad, corners, *labels

    # Hold a reference so the animation isn't garbage-collected before it plays.
    fig._anim = FuncAnimation(fig, update, frames=len(seq), interval=20, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
