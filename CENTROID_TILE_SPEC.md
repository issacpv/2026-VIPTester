# Centroid-Tile Demo — Geometry & Math Spec

> **Scope.** This document is the shared context for work on
> [`centroid_tile_demo.py`](centroid_tile_demo.py): three concurrent *coding*
> features (Tasks A/B/C) and eight *mathematical* derivations (M1–M8).
>
> It is deliberately self-contained so a fresh Claude Code session can derive
> the math **before** the code is updated with the new features. Where the
> current code and the intended model differ, the **intended model** is what the
> math must target; the code is for geometric grounding only (incenter, feet,
> foot naming, Bézier sampling).
>
> This is **not** the project-wide `SPEC.md` (that belongs to "Auxetic Lattice
> Studio"). This file governs only the standalone demo.
>
> **Status of §6 (the d0/d1/d2 fillet framework): PROPOSED.** It is a
> self-consistent starting point. Every math session must restate the exact
> definitions it uses at the top of its answer and flag any disagreement before
> deriving (per the user's "independent proportions / restate-first" decision).

---

## 1. The pipeline at a glance

```
points (N×2)
  → Delaunay triangulation              scipy.spatial.Delaunay
  → per-triangle anchor C               _incenter (default) or centroid
  → inner-triangle vertices  T_i        T_i = C + c·(P_i − C)
  → feet (perp. projections)            foot of T_i on each adjacent edge
  → wings (quad panels)                 (T_i, foot_a, P_i, foot_b)
  → links (joint polygons)              one per shared Delaunay vertex
  → actuation θ                         rigid kirigami opening
  → Bézier joint bridges (fillets)      rounded "flowers" welding the joints
  → STL extrusion                       thin slab per panel
```

Everything downstream of `points` is recomputed on demand. Static per-triangle
geometry is cached in `TilingGeometry`; each `(c, θ)` evaluation is a few
einsums. See the module docstring in `centroid_tile_demo.py` for the canonical
description.

---

## 2. Core definitions and notation

Let a triangle have vertices `P0, P1, P2` (CCW), indices `i ∈ {0,1,2}`.

### 2.1 Anchor point `C`

- **Incenter (default, `anchor="incenter"`).** Weighted average of vertices by
  the length of the **opposite** edge:
  ```
  a   = |P1 − P2|      (edge opposite P0)
  b   = |P0 − P2|      (edge opposite P1)
  c_w = |P0 − P1|      (edge opposite P2)
  C   = (a·P0 + b·P1 + c_w·P2) / (a + b + c_w)
  ```
  Equidistant from the three edges; strictly interior.
- **Centroid (`anchor="centroid"`).** `C = (P0 + P1 + P2)/3`.
- For an **equilateral** triangle the two coincide. The canonical two-triangle
  example (§5) is equilateral, so there incenter = centroid.

### 2.2 Inner-triangle vertices `T_i` and the shrink `c`

```
v_i = P_i − C
T_i = C + c·v_i = C + c·(P_i − C)
```

- `c = 0` → all `T_i` collapse to `C`.
- `c = 1` → `T_i = P_i` (inner triangle equals the outer triangle).
- `c ∈ (0,1)` → a scaled copy of the triangle about `C`.
- The inner triangle `T0 T1 T2` is `c·` the size of the outer triangle and is
  a **translate-free central scaling** about `C`, so `T_i − T_j = c·(P_i − P_j)`.

`c` may be **scalar** (global), **per-triangle** (length-M array; see
`compute_T`), or — once **Task B** lands — **per-vertex** `c_i` so that
`T_i = C + c_i·(P_i − C)` with the three `c_i` independent. With per-vertex `c`
the identity `T_i − T_j = c·(P_i − P_j)` no longer holds; the inner triangle is
no longer a uniform scaling and its edges change length independently.

### 2.3 Feet (perpendicular projections) and the `Pabc` naming

The **foot** of a point `T` on the line through `A, B`:
```
d    = B − A
t    = ((T − A)·d)/(d·d)
foot = A + t·d
```

Each `T_i` is projected onto its **two adjacent** outer edges. The six feet, in
canonical order, are named `Pabc` where:

> **`Pabc` = foot of `T_a` projected onto the edge `(P_b, P_c)`.**

| name   | owner `T_a` | edge `(P_b,P_c)` | belongs to wing |
|--------|-------------|------------------|-----------------|
| `P001` | T0          | (P0,P1)          | wing T0         |
| `P101` | T1          | (P0,P1)          | wing T1         |
| `P112` | T1          | (P1,P2)          | wing T1         |
| `P212` | T2          | (P1,P2)          | wing T2         |
| `P002` | T0          | (P0,P2)          | wing T0         |
| `P202` | T2          | (P0,P2)          | wing T2         |

(Implementation: `_FOOT_NAMES`, `_T_FOR_FOOT`, `_EDGE_FOR_FOOT`, `_EDGES`.)

The segment `T_a → Pabc` is **perpendicular** to edge `(P_b,P_c)` — this is the
**"perpendicular leg."**

### 2.4 Wings

Each `T_i` owns a quadrilateral wing panel `(T_i, foot_a, P_i, foot_b)`:
```
wing(T0) = (T0, P001, P0, P002)
wing(T1) = (T1, P101, P1, P112)
wing(T2) = (T2, P212, P2, P202)
```

### 2.5 Links / joints

A **joint** is a Delaunay vertex `v` shared by `K ≥ 2` triangles. `compute_links`
emits one polygon per joint:
- `K = 2` (two triangles across one shared edge): 6-corner hexagon
  `[T_m, F_m, P, P, F_n, T_n]` — two inner-triangle `T`s, two non-shared-edge
  feet, and the shared Delaunay vertex `P` listed twice (the kirigami hinge, a
  zero-length edge).
- `K ≥ 3` interior: the convex `K`-gon of the incident `T`s.
- `K ≥ 3` boundary (hull vertex): `K` `T`s + 2 outer feet + central `P`.

### 2.6 Actuation `θ` (current, uniform-`c` model)

Rigid kirigami opening, currently parameterized by a single angle `θ`:
- Every wing rotates by `+θ` about its own `T_i`.
- Each link rotates rigidly by `+θ` about its pivot on the **upstream**
  (BFS-parent) inner triangle.
- The **root** inner triangle stays fixed; every other inner triangle is
  **translated** (orientation preserved) so its pivot `T` stays attached.
- This closes up exactly **only under uniform `c`**, because for a shared edge
  `(A,B)` the auxetic identity `T_n_A − T_n_B = c·(P_A − P_B) = T_m_A − T_m_B`
  makes both link translations agree. With per-triangle/per-vertex `c`, the
  identity breaks and the current code is only *approximately* rigid — this is
  exactly what **Task A** addresses.

### 2.7 Bézier joint bridges (fillets) — current behaviour

At each joint the incident edges are rounded by a quadratic-Bézier "flower":
1. Collect the joint's **arms** — every edge leaving the joint centre. Each arm
   carries `(unit_dir, leg_len, angle, is_inner_edge, tt_edge)`. `is_inner_edge`
   is True iff the source polygon is one of the inner triangles (`pi < n_inner`).
   `tt_edge` ("T-to-T") is True iff **both** endpoints are inner-triangle
   vertices (T's) — that is, an inner-triangle edge **or** a link **cross-arm**
   joining two T's of neighbouring triangles (a face of a purple link polygon).
   Anything else is a **leg** (a perpendicular wing leg `T → foot`, etc.).
2. Back every arm off from the centre by a distance bounded by a fraction of its
   **own** length, set by the edge type (Task C bounds, now implemented):
   - **T-to-T edge** (inner edge **or** link cross-arm) → `0.5 · (its length)`
     (its **midpoint**). Both ends of a T-to-T edge are joints, so the midpoint
     cap makes the two fillets meet there instead of **overrunning** each other.
   - **leg** → `1.0 · (its full length)`.

   How the per-arm bounds combine is set by the **fillet-mode** toggle
   (`uniform` | `per-arm`):
   ```
   uniform:  radius = d · min( 0.5 · (shortest T-to-T edge) , 1.0 · (shortest leg) )
             (one shared radius → symmetric flower)
   per-arm:  radius_k = d · bound_k                       (each arm its own bound
             → asymmetric flower; back-off points at different radii)
   ```
   where `d ∈ [0,1]` is the fillet slider. (These replace the previous
   `0.5·leg / 0.25·inner` uniform bounds. The leg bound now reaches the full
   leg; every T-to-T edge — not just same-triangle inner edges — is capped at
   its midpoint, which is what stops the per-arm/uniform flowers from
   overlapping along the purple link faces at `d = 1`.)
3. Take the back-off points in angular order; join consecutive ones with a
   quadratic Bézier whose **control point is the joint centre**: arc
   `D_i → centre → D_{i+1}`.

Sampling: `_quadratic_bezier(p0, p1, p2, n) = (1−t)²p0 + 2(1−t)t·p1 + t²p2`,
`t ∈ [0,1]` linspace. Relevant functions: `_joint_arms`, `_inner_vertex_keys`,
`_arm_backoff_bound`, `_arm_radii`, `_joint_radius_from_arms`,
`_build_joint_bridge`, `build_joint_bridges`, `joint_radii` (all take a `mode` of
`"uniform"` | `"per-arm"`). The "radius vs c" plot reports, per joint, the
**binding (min)** arm radius — which equals the uniform radius — and labels the
active mode.

> **Task C (new bounds + uniform/per-arm toggle) is implemented — see §4.3.**

### 2.8 STL export

Each rendered polygon (inner triangles, wings, links, Bézier bridges) is
extruded into a slab of thickness 5% of the bounding-box diagonal and welded
into a binary STL. Not central to the math here.

---

## 3. Interactive viewer (current controls)

Sliders: `c` (global or selected-triangle shrink), `theta` (deg), `spin` (deg,
whole-structure view rotation), `fillet` (`d`). Radio: `anchor`. Button:
`radius vs c`.

Mouse / keys (see `on_press`):
- **Shift-click** a triangle → select it (`state["sel_tri"]`); the `c` slider
  then edits that triangle's `c` (`state["c_over"]`), and it becomes the BFS
  root. Shift-click empty space → deselect.
- **Ctrl-click-drag** a point → move it; re-triangulates live.
- **Alt-click** → add a point.
- `c` key → dump spun coordinates; `e` key → export STL.

State lives in the `state` dict inside `show()`. `effective_c()` resolves the
per-triangle override array; `redraw()` rebuilds all artists.

---

## 4. Planned features (intended end-state)

These three are implemented in **separate worktrees** and synthesized later
(§7). Each must preserve current behaviour in the **uniform-`c`, default-mode**
case and add tests.

### 4.1 Task A — flexing trapezoid (four-bar linkage actuation)

**Problem.** With a per-triangle (or per-vertex) `c` override, two neighbours
across a shared edge have inner edges of **different lengths**
(`2c_m` vs `2c_n` for the canonical equilateral pair). The negative space
between them is a **trapezoid**, and the current translate-only propagation no
longer closes — the mechanism appears locked.

**Decision (chosen): four-bar linkage.** Treat each trapezoidal gap as a
**1-DOF four-bar linkage** (see [`quad_linkage_demo.py`](quad_linkage_demo.py)
for the exact recipe). Drive the actuation; let the inner triangles **rotate**
as the linkage demands (not just translate); and **stop at the first
configuration** where either:
- the bars can no longer close (the circle–circle intersection that places the
  next corner vanishes), **or**
- any panel pair **self-intersects** (reuse `_segments_cross`-style tests).

That stopping configuration *is* "can no longer rotate."

**References to mine:**
- `quad_linkage_demo.py` — `circle_intersections`, `linkage_states`,
  `is_simple`, continuous assembly-mode following. This is the kinematic core.
- `auxetic/bipartite.py::jamming_angle` and
  `auxetic/simulation.py::Simulator.sweep_theta(collision_stop=True)` — a
  *working reference solver* for a sibling structure (constraint-Jacobian /
  collision-stop). It solved the trapezoid the **opposite** way (kept feet
  perpendicular to avoid the trapezoid); here we *embrace* the trapezoid, but
  the collision-stop machinery and `res.collision_at_theta` pattern are a useful
  model.

### 4.2 Task B — per-vertex `c`

**Decision (chosen): proximity disambiguation.** After **shift-click**
selecting a triangle, a **ctrl-click near one of that triangle's three
vertices** selects that vertex as the c-edit target; the `c` slider then changes
**only that vertex's** `c_i`, leaving the other two in place. **Ctrl-drag
elsewhere still moves points** (disambiguate by proximity to a selected-triangle
vertex; below a pixel threshold → vertex-pick, otherwise → drag/move).

**Model.** Generalise `c` to per-vertex: `T_i = C + c_i·(P_i − C)`. The override
store becomes per-`(triangle, local-vertex)` (or a `(M,3)` array). `compute_T`
must accept a `(M,3)` `c`. Visual feedback for the picked vertex. Preserve the
scalar/`(M,)` fast paths.

### 4.3 Task C — Bézier bounds + uniform/per-arm toggle

**Decision (chosen): support BOTH, toggle-able in the UI.** Replace the current
bounds with the **new bounds**, and add a UI toggle (radio/checkbox) between two
application modes:

New per-arm bounds:
- inner-edge arm bound = **½ · (its inner-triangle edge length)** = the
  **midpoint of the inner triangle** edge.
- leg arm bound = **1 · (its full perpendicular leg length)**.

Modes:
- **Uniform** (toggle = uniform): one radius for all arms,
  `radius = d · min( 0.5·(shortest inner edge) , 1.0·(shortest leg) )`.
- **Per-arm** (toggle = per-arm): each arm backs off independently by
  `d · (its own bound)` — inner arms by `d·½·innerlen`, leg arms by
  `d·1.0·leglen`. Endpoints may then differ per arm; this is the basis for the
  asymmetric d0/d1/d2 thickness work (§6).

Touch `_joint_radius_from_arms` / `_build_joint_bridge` (and `joint_radii` for
the radius-vs-c plot). Keep `d = 0` → sharp join.

---

## 5. Canonical examples (ground truth for the math)

### 5.1 Single triangle (the file's `__main__`)

`P0=(0,0), P1=(1, 1.73), P2=(2,0)` — near-equilateral.

### 5.2 Two neighbouring equilateral triangles (used by M1)

```
P0 = (0, 0)
P1 = (1, √3)      ≈ (1, 1.7320508)
P2 = (2, 0)
P3 = (3, √3)      ≈ (3, 1.7320508)
```
- **Triangle A** = `(P0, P1, P2)` — equilateral, side 2. Shared edge `P1–P2`.
- **Triangle B** = `(P1, P2, P3)` — equilateral, side 2. Shared edge `P1–P2`.
- Anchors (incenter = centroid, equilateral):
  ```
  C_A = (1,  √3/3) ≈ (1, 0.5773503)
  C_B = (2, 2√3/3) ≈ (2, 1.1547005)
  ```

Per-vertex inner vertices at the shared edge (with per-vertex `c`):
```
v at P1 in A:  P1 − C_A = (0,  2√3/3)
v at P1 in B:  P1 − C_B = (−1,  √3/3)
v at P2 in A:  P2 − C_A = (1, −√3/3)
v at P2 in B:  P2 − C_B = (0, −2√3/3)

T1A(c1A) = (1,            (√3/3)(1 + 2·c1A))
T1B(c1B) = (2 − c1B,      (√3/3)(2 +   c1B))
T2A(c2A) = (1 + c2A,      (√3/3)(1 −   c2A))
T2B(c2B) = (2,            (2√3/3)(1 −  c2B))
```

**Numeric anchors for self-checking (do not skip your own derivation; use these
only to verify):**
- `c1A = 0 → T1A = (1, 0.577350)`, `c1A = 1 → T1A = (1, 1.732051) = P1`. ✓
- `c1B = 0 → T1B = (2, 1.154701) = C_B`, `c1B = 1 → T1B = (1, 1.732051) = P1`. ✓
- At `c1A = c1B = 0.5`:
  `T1A = (1, 1.154701)`, `T1B = (1.5, 1.443376)`;
  displacement `T1B − T1A = (0.5, 0.288675)`;
  midpoint `= (1.25, 1.299038)`.

For **uniform `c`** in each triangle, the inner edge along the shared edge has
length `2c_A` (resp. `2c_B`), both parallel to `P1–P2` → the trapezoid of §4.1.

---

## 6. PROPOSED fillet re-parameterization: d0 / d1 / d2 (confirm per session)

> The user is **exploring** a richer fillet model than the single slider `d`.
> Decision: **d0/d1/d2 are independent proportion parameters.** The notation
> below is a proposal; **each math session must restate the exact definitions it
> uses and confirm/refine with the user before deriving.**

### 6.1 Two naming layers

- **Design proportions** `d0, d1, d2 ∈ [0,1]` — the knobs being explored.
- **Concrete per-arm back-off proportion** `d⟨abc⟩⟨s⟩` — the proportion applied
  to the *specific* arm at foot/vertex name `abc` on side-type `s`:
  - `s = 0` → the arm runs along a **perpendicular leg**; bound = full leg
    length → back-off distance `= d⟨abc⟩0 · leglen`.
  - `s = 1` → the arm runs along an **inner-triangle edge**; bound = **½** the
    inner edge (its **midpoint**) → back-off distance `= d⟨abc⟩1 · ½ · innerlen`.

  Examples from the brief: `d0010` = `P001`, leg; `d1011` = `P101`, inner;
  `d2021` = `P202`, inner.
  **⚠ `d0120` decodes to "`P012`, leg" — but `P012` is owner `T0` on edge
  `(P1,P2)`, which is `T0`'s *opposite* edge and has no foot. This is almost
  certainly a typo (intended `P002` → `d0020`, or `P112` → `d1120`). Confirm the
  intended foot at session start.**

### 6.2 Proposed roles of d0, d1, d2

A fillet endpoint at a joint is placed by backing off along one arm. Two
directions matter at a shared-edge joint:
- **`d0` — "between the inner triangles."** Proportion of the displacement
  *between neighbouring inner triangles* (across the gap / the leg-side, `s=0`)
  at which the Bézier **start** point sits. (This is what M2 formalises.)
- **`d1` — "to the inner-triangle midpoint."** Proportion of the displacement
  *toward the midpoint of the inner-triangle edge* (the inner-side, `s=1`) at
  which the Bézier **start** point sits. (This is what M4 formalises.)
- **`d2` — second inner/opposite proportion** used when locating the **thinnest
  neck** (the user pairs `d0` with `d2` in M6). Exact role **to be confirmed**;
  candidate: the inner-side proportion on the *other* panel meeting the joint,
  so the neck width depends on `d0` (one side) and `d2` (the other).

### 6.3 "Thickness" and "thinnest point" of the Bézier

A curve has no thickness; the quantity of interest is the **width of the
material neck** the fillet leaves at a joint — the distance between the two
opposing fillet curves that bound a rounded strut/web. The user wants this:
- as a function of the back-off proportions on **both sides** of a joint —
  `d0 & d1` for the general thickness (M5), `d0 & d2` for the thinnest point
  (M6); and
- as a function of the **Bézier parameter `t`** along a single arc — the
  thickness profile `w(t)` (M7) and its minimiser `t*` (M8).

Each math session must **pin the precise definition of "thickness"** (which two
curves / which perpendicular) before deriving, state it explicitly, and keep it
consistent with the quadratic-Bézier form in §2.7.

---

## 7. Synthesis / merge hotspots (for combining A+B+C)

All three tasks touch the same hot path, so expect conflicts here — call them
out in each PR/notes:
- `redraw()` and `build_panels()` — render path used by A (kinematics), C
  (fillet build), and reads the `c` model from B.
- `compute_T` / `effective_c()` / the `state` dict — B changes `c` from
  scalar/`(M,)` to per-vertex `(M,3)`; A consumes `c` for the linkage; C is
  largely independent of `c` shape but reads inner-edge lengths.
- `on_press` / mouse handling — B adds vertex-pick (ctrl-click proximity); A may
  add UI to drive/clamp `θ`; C adds the uniform/per-arm toggle.
- `_bfs_propagate` — A replaces/augments it with the linkage solve.

**Guidance for each worktree:** implement your task **independently** (do not
depend on the other two), keep the diff localized, and in your final notes list
exactly which shared functions you changed and how, so synthesis is mechanical.

---

## 8. Conventions for the math derivations (M1–M8)

Every math prompt requires, in this order:
1. **Restate the setup & notation** you will use (symbols, frames, the d0/d1/d2
   definitions from §6), and **flag any inconsistency** with this spec (e.g. the
   `P012` typo) before proceeding.
2. **Derive symbolically, showing all work** — no skipped steps. Define every
   symbol on first use. Keep `c` per-vertex where relevant.
3. Give the **final closed-form** parametric equation(s): the **displacement
   vector** and the **location/coordinates** where the quantity occurs, as
   explicit functions of the stated parameters.
4. **Numerically sanity-check** against the canonical example in §5 (report the
   numbers; confirm they match the provided anchors where applicable).
5. **Map to code:** name the functions in `centroid_tile_demo.py` the result
   corresponds to or should modify, and note any discrepancy between the
   *current* code and the *intended* model this result assumes.

Ground the geometry in the real code (incenter, feet, foot naming, Bézier form),
but derive for the **intended** model (per-vertex `c`; new fillet bounds;
d0/d1/d2), which may not yet exist in the file.
