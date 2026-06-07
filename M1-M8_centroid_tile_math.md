# Centroid-Tile Math Derivations M1–M8 — Consolidated Reference

> **Purpose.** Self-contained record of the eight mathematical derivations
> (M1–M8) for [`centroid_tile_demo.py`](../../centroid_tile_demo.py), to be used
> by the **synthesis session** that merges Tasks A/B/C and wires the
> d0/d1/d2 fillet model into code. Governing spec:
> [`CENTROID_TILE_SPEC.md`](../../CENTROID_TILE_SPEC.md). Source prompts:
> [`prompts/MATH_M1…M8`](../).
>
> **Model targeted.** The **intended** model (per-vertex `c`; new fillet bounds;
> the d0/d1/d2 framework) — which may run ahead of what the file currently
> implements. Every closed form below was **numerically verified against the
> live code** (`_incenter` / `compute_T`, and the `_build_joint_bridge` back-off
> path) at the canonical §5.2 configuration.
>
> **Math math-notation.** Written in plain text + code blocks (no LaTeX), so it
> renders in a terminal and in any markdown viewer. `√3/3 = 1/√3 ≈ 0.5773503`,
> `√3/6 ≈ 0.2886751`, `√3/2 ≈ 0.8660254`.

---

## 0. Conventions and flags (read first)

Per §8 every result restates its setup, derives symbolically, gives closed
forms, sanity-checks against §5.2, and maps to code. Cross-cutting flags that
the synthesis must honour:

1. **`P012` typo (§6.1).** `d0120` decodes to "foot `P012`, leg," but `P012` is
   owner `T0` on edge `(P1,P2)` — `T0`'s **opposite** edge, which has **no foot**.
   Resolved to **`P002` → `d0020`** (foot of `T0` on edge `(P0,P2)`, a real
   wing-`T0` leg). Alternative `P112 → d1120`. **Use `d0020`.**
2. **`s=0`/"leg" label vs the cross-arm.** §6.2 calls `d0` the "leg-side, `s=0`"
   proportion, but the inter-triangle displacement `Δ = T1B − T1A` is the link
   **cross-arm** — a ½-capped **T-to-T** edge (`tt_edge=True`), *not* a
   full-length perpendicular leg. M2/M5 use `d0` along `Δ` regardless of the
   imperfect label.
3. **Per-vertex `c` is the intended model.** `compute_T` already supports the
   `(M,3)` per-vertex branch (Task B, synthesized). The single-triangle helper
   `construct_triangle_tile` **still takes a scalar `c`** — it cannot express
   `c1A ≠ c2A` and must be generalized to a length-3 `c` to realize any
   per-vertex result below.
4. **Fillet model is PROPOSED (§6).** The exact panel/strut topology for the
   "neck/thickness" is not pinned in code. The thickness definition used in
   M5–M8 ("distance between two opposing fillet curves") is a **stated modeling
   choice**, not a unique geometric consequence — see §M5 and the flag in §9.
5. **`d2` role (§6.2 "to be confirmed").** M6 adopts `d2` = the **opposing
   gap-side (cross-arm) proportion at `T1B`**, because §6.2's literal "inner-side
   on the other panel" leaves the fracture neck **uncontrolled** (verified
   numerically). Flagged for user confirmation.
6. **Current code couples the proportions to the global slider:** the cross-arm
   back-off is `d0 = d·½` and the inner-edge back-off is `d1 = d` (one slider
   `d`). The intended model **frees them per-arm** (`per-arm` fillet mode).

---

## 1. Shared setup and canonical example (§2, §5.2)

**Per-vertex inner vertices** (§2.2): with anchor `C` (incenter = centroid for
equilateral) and `v_i = P_i − C`,
```
T_i = C + c_i·(P_i − C),     c_i ∈ [0,1] independent per vertex.
```

**Canonical two-triangle equilateral pair (§5.2):**
```
P0=(0,0)  P1=(1,√3)  P2=(2,0)  P3=(3,√3)
A=(P0,P1,P2)   B=(P1,P2,P3)   shared edge P1–P2
C_A = (1,  √3/3) ≈ (1, 0.577350)      C_B = (2, 2√3/3) ≈ (2, 1.154701)
```
`T1A`,`T1B` = the two inner images of the shared corner `P1` (knobs `c1A`,`c1B`);
`T2A`,`T2B` = the inner images of the shared corner `P2` (knobs `c2A`,`c2B`).
```
v1A = P1−C_A = (0,  2√3/3)     v1B = P1−C_B = (−1, √3/3)
v2A = P2−C_A = (1, −√3/3)      v2B = P2−C_B = (0, −2√3/3)
```

**Canonical numeric joint** (`c1A=c1B=c2A=c2B=½`), used by M5–M8 with the
joint centre / Bézier control point `Q = T1A`:
```
T1A=(1, 1.154701)   T2A=(1.5, 0.288675)   T1B=(1.5, 1.443376)   T2B=(2, 0.577350)
Δ  = T1B − T1A = (0.5, 0.288675),   |Δ| = 1/√3 ≈ 0.577350        [M1]
b  = Mᴬ − T1A = ½(T2A − T1A) = (0.25, −0.433013),   |b| = 0.5    [M4]
Δ·b = 0.125 − 0.125 = 0     (cross-arm ⊥ inner half-vector at c=½)
```

---

## M1 — Displacement & midpoint of `T1A`, `T1B`

```
T1A(c1A) = ( 1,        (√3/3)(1 + 2·c1A) )
T1B(c1B) = ( 2 − c1B,  (√3/3)(2 +   c1B) )

Δ(c1A,c1B) = ( 1 − c1B,  (√3/3)(1 − 2·c1A + c1B) )
M(c1A,c1B) = ( (3 − c1B)/2,  (√3/6)(3 + 2·c1A + c1B) )

|Δ| = √[ (1−c1B)² + (1/3)(1−2·c1A+c1B)² ]
Δ/|Δ| = ( √3(1−c1B), 1−2·c1A+c1B ) / √[ 3(1−c1B)² + (1−2·c1A+c1B)² ]
```
**M-locus** (affine image of the unit square): `M = M0 + c1A·a + c1B·b` with
`M0=(3/2, √3/2)`, `a=½v1A=(0, √3/3)`, `b=½v1B=(−1/2, √3/6)` — a **rhombus**,
side `1/√3`, interior angle `60°`. Hold `c1B` fixed → vertical segment; hold
`c1A` fixed → line of slope `−1/√3` (−30°).
**Equal-c diagonal** `c1A=c1B=c`: `Δ = (1−c)(C_B−C_A)`, direction fixed at +30°.

**Numbers.** `(0.5,0.5)`→ `Δ=(0.5,0.288675)`, `M=(1.25,1.299038)`.
`(0,0)`→ `Δ=(1,0.577350)=C_B−C_A`, `M=(1.5,0.866025)`.
`(1,1)`→ `Δ=0`, `M=P1=(1,1.732051)`.

---

## M2 — `d0`: Bézier start along the gap `Δ`

`d0 ∈ [0,1]`, measured from `T1A` along `Δ`. `S0(d0) = T1A + d0·Δ`.
```
ρ0(d0) = d0·Δ = ( d0(1−c1B),  d0(√3/3)(1−2·c1A+c1B) )
S0(d0) = ( 1 + d0(1−c1B),  (√3/3)[ 1 + d0 + 2·c1A(1−d0) + d0·c1B ] )
S0(0)=T1A,   S0(½)=M=(1.25,1.299038)@c=½,   S0(1)=T1B
```
**Code:** current back-off on the cross-arm is `T1A + (d·½)·Δ` ⇒ **`d0 = d·½`**
(slider `d` sweeps `d0∈[0,½]`, hitting the midpoint `M` at `d=1`).

---

## M3 — General `Ti`,`Tj` (arbitrary triangles, incenter anchor)

Incenter (§2.1): `C = (a·P0 + b·P1 + cw·P2)/(a+b+cw)`, `a=|P1−P2|`, `b=|P0−P2|`,
`cw=|P0−P1|`. With `v_i=P_i−C_i`, `T_i=C_i+c_i·v_i`:
```
Δ(ci,cj) = (Cj − Ci) − ci·(Pi − Ci) + cj·(Pj − Cj)        [affine, no ci·cj term]
M(ci,cj) = ½(Ci + Cj) + ½·ci·(Pi − Ci) + ½·cj·(Pj − Cj)
```
**M-locus parallelogram:** base `½(Ci+Cj)`, edges `½vi` & `½vj`, far corner
`½(Pi+Pj)`. Specializing `Pi=Pj=P1`, `Ci=C_A`, `Cj=C_B` recovers M1 exactly.

**Scalene check** (pts `(0,0),(6,0),(1,2),(7,3)`, shared corner `(6,0)`):
`Ci=(1.425452,0.880978)`, `Cj=(5.287349,1.536551)`; at `(ci,cj)=(0.4,0.7)`
→ `Δ=(2.530934,−0.067621)`, `M=(4.520738,0.494776)` (matches `compute_T`).

---

## M4 — `d1`: Bézier start along the inner edge to its midpoint

`d1 ∈ [0,1]`, from `T1A` toward the inner-edge midpoint `Mᴬ = ½(T1A+T2A)`.
`S1(d1) = T1A + d1·(Mᴬ − T1A)`.
```
T2A(c2A) = ( 1 + c2A,  (√3/3)(1 − c2A) )
Mᴬ(c1A,c2A) = ( 1 + c2A/2,  (√3/6)(2 + 2·c1A − c2A) )
Mᴬ − T1A = ½(T2A − T1A) = ( c2A/2,  −(√3/6)(2·c1A + c2A) )

ρ1(d1) = d1·(Mᴬ − T1A)
S1(d1) = ( 1 + d1·c2A/2,  (√3/6)[ 2 + 2·c1A(2 − d1) − d1·c2A ] )
S1(0)=T1A,   S1(1)=Mᴬ        |Mᴬ − T1A| = ½·innerlen
```
**Numbers** (`c=½`): `T2A=(1.5,0.288675)`, `Mᴬ=(1.25,0.721688)`, `innerlen=1`,
`|Mᴬ−T1A|=0.5`. **Code:** current back-off `T1A+(d·½)(T2A−T1A)=T1A+d·(Mᴬ−T1A)`
⇒ **`d1 = d`** (hits the midpoint `Mᴬ` at `d=1`). Contrast M2: `d0` spans the
**full** cross-arm (midpoint at `d0=½`), `d1` spans **to** the inner-edge
midpoint (midpoint at `d1=1`); along the inner edge `d1 = 2·d0ᴬ`.

---

## Thickness chain — shared definitions (M5–M8)

```
Q  = T1A (joint centre = Bézier control point)
S0 = Q + d0·Δ    [M2]        S1 = Q + d1·b   [M4],   b = ½(T2A−T1A)
B(t) = (1−t)²·S0 + 2(1−t)t·Q + t²·S1            (fillet arc, §2.7)
B(t) − Q = (1−t)²·g0 + t²·g1,   g0 = d0·Δ,  g1 = d1·b
```
**Pinned neck definition (identical in M5/M6/M7/M8).** Neck = distance between
the **two opposing fillet curves** that bound a web (§6.3). For one rounded
corner the opposing curve is `B(t)` reflected through `Q`, giving
```
w(t) = N(t) = 2·|B(t) − Q| = 2·| (1−t)²·g0 + t²·g1 |.
```
Scalar thickness / throat `W = min_t w(t)`. **(Modeling choice — see flag 4/§9.)**

---

## M5 — Thickness `W(d0,d1)`

```
w(t)² = 4[ (1−t)⁴·G00 + 2(1−t)²t²·G01 + t⁴·G11 ]
   G00 = d0²|Δ|²,   G11 = d1²|b|²,   G01 = d0·d1·(Δ·b)

W(d0,d1) = min_t w(t) = w(t*)         (t* = interior cubic root, see M8)
  equidistant case d0|Δ| = d1|b| (⇔ G00=G11):  t* = ½,  W = ½·|d0·Δ + d1·b|
```
**Monotonicity** (canonical, `Δ·b=0`): `∂W/∂d0 = ½ d0|Δ|²/|g0+g1| > 0`,
`∂W/∂d1 = ½ d1|b|²/|g0+g1| > 0`. Increasing **either** proportion **thickens**
the web. **Thickest** at `d0=d1=1`; neck **→ 0** as either `d → 0`.
(Cross-arm "meet-don't-overrun" cap of §2.7 limits `d0 ≤ ½`.)

**Numbers.** `W(0.5,0.5)=0.189632 @ t*=0.52396` (endpoints `w(0)=0.577350`,
`w(1)=0.5`). Equidistant `W(0.5,0.577)=0.204124 @ t*=½`.

---

## M6 — Thinnest point from `d0` and `d2`

`d2` := **opposing gap-side (cross-arm) proportion at `T1B`** (flag 5):
`S2 = T1B − d2·Δ = T1A + (1−d2)·Δ`. The shared cross-arm strut `T1A→T1B` is
carved to `S0` from one end and `S2` from the other; the remaining solid
**ligament** is `[S0, S2]`:
```
W(d0,d2) = max( 0, 1 − d0 − d2 ) · |Δ|        (remaining neck width)
throat at τ* = (d0 + 1 − d2)/2  →  point  P* = T1A + τ*·Δ
```
- **(a) geometry:** narrowest at the ligament midpoint; **severs** (pinches to a
  point) when `d0+d2 = 1` — exactly the §2.7 midpoint cap.
- **(b) parameters:** `W` is linear, strictly decreasing in each; the min is on
  the boundary `d0+d2=1` (`W=0`), the max `W=|Δ|=0.577350` at `d0=d2=0`.
- **Reconciles with M8:** symmetric `d0=d2` → `τ*=½` → `P* = T1A+½Δ =
  M = (1.25, 1.299038)` — the same throat point M8 finds at `t*=½`.

**Numbers.** `(0.3,0.3)→W=0.230940 @ M`; `(0.5,0.5)→W=0` (severed at `M`).

---

## M7 — Thickness profile `w(t)`

```
B'(t) = 2(1−t)(Q − S0) + 2t(S1 − Q)                       (tangent)
w(t)  = 2·√[ (1−t)⁴·G00 + 2(1−t)²t²·G01 + t⁴·G11 ]        (w² is QUARTIC in t)

w(0) = 2·d0|Δ| = 0.577350      w(1) = 2·d1|b| = 0.5
w(½) = ½·|g0 + g1| = 0.190940
```
**Profile:** a single **interior throat** (one minimum) — `w` falls from `w(0)`
to the throat at `t*`, then rises to `w(1)` (the unimodal *dip* dual to a
quadratic-Bézier offset bump). Verified on a 200k `t`-grid (`t*=0.52396`,
`w(t*)=0.189632`). At the canonical joint `G01 = 0` (since `Δ·b=0`).

---

## M8 — Thinnest part in `t` (minimize `w(t)`)

```
d/dt[ w(t)² ] = 0   ⇒   cubic:
  (G00+2G01+G11)·t³ − 3(G00+G01)·t² + (3G00+G01)·t − G00 = 0

t* = ½   iff   G00 = G11   (⇔ d0|Δ| = d1|b|, equidistant back-offs);
otherwise t* = the unique root in (0,1), biased toward the nearer back-off.

w(t*) = 2·| (1−t*)²·d0·Δ + t*²·d1·b |
B(t*) = T1A + (1−t*)²·d0·Δ + t*²·d1·b      (world coords of the thinnest point)
```
**Min vs endpoint.** `f''(t*)>0` and `w(t*) < w(0),w(1)` ⇒ interior minimum,
present whenever `h := dist(Q, chord S0S1) > 0` (generic `d0,d1>0`). Degenerates
to an endpoint only when `S0,Q,S1` are collinear (`d0=0`, `d1=0`, or `Δ∥b`).

**Numbers** (`d0=d1=0.5`): `t*=0.52396`, `w(t*)=0.189632` (= M5 throat),
`B(t*)≈(1.0910,1.1280)`. Equidistant `d0=0.5,d1=0.577` → `t*=½`, `w=0.204124`.
**Reconcile M6:** symmetric `d`'s → `t*=½` → `B(½)` on the cross-arm = `M`;
same neck point as M6's `τ*=½`. (They measure different widths there — M8 the
perpendicular web `2|B−Q|`, M6 the along-strut ligament — but the fracture
*location* agrees, and both vanish in the sever limit.)

---

## 8. Closed-form cheat sheet

```
M1  T1A=(1,(√3/3)(1+2c1A))            T1B=(2−c1B,(√3/3)(2+c1B))
    Δ =(1−c1B,(√3/3)(1−2c1A+c1B))     M =((3−c1B)/2,(√3/6)(3+2c1A+c1B))
M2  S0(d0)=T1A+d0·Δ
M3  Δ=(Cj−Ci)−ci(Pi−Ci)+cj(Pj−Cj)    M=½(Ci+Cj)+½ci(Pi−Ci)+½cj(Pj−Cj)
M4  S1(d1)=T1A+d1·(Mᴬ−T1A)           Mᴬ=½(T1A+T2A)
M5  W(d0,d1)=½|d0·Δ+d1·b| (equidist.)         (b=½(T2A−T1A))
M6  W(d0,d2)=max(0,1−d0−d2)·|Δ|       throat at T1A+½Δ when d0=d2
M7  w(t)=2√[(1−t)⁴d0²|Δ|²+2(1−t)²t²d0d1(Δ·b)+t⁴d1²|b|²]
M8  (G00+2G01+G11)t³−3(G00+G01)t²+(3G00+G01)t−G00=0,  t*=½ iff d0|Δ|=d1|b|
```

## 8a. Canonical numeric anchors (`c=½` joint)

| quantity | value |
|---|---|
| `T1A,T2A,T1B,T2B` | `(1,1.154701) (1.5,0.288675) (1.5,1.443376) (2,0.577350)` |
| `Δ`, `\|Δ\|` | `(0.5,0.288675)`, `0.577350` |
| `b=Mᴬ−T1A`, `\|b\|` | `(0.25,−0.433013)`, `0.5`; `Δ·b=0` |
| M1 `Δ,M` @(½,½) | `(0.5,0.288675)`, `(1.25,1.299038)` |
| M2 `S0(½)` | `M=(1.25,1.299038)` |
| M4 `Mᴬ`, `innerlen` | `(1.25,0.721688)`, `1.0` |
| M5 `W(½,½)`,`t*` | `0.189632`, `0.52396` |
| M6 `W(0.3,0.3)` | `0.230940` at `M` |
| M7 `w(0),w(½),w(1)` | `0.577350, 0.190940, 0.5` |
| M8 `t*`,`w(t*)`,`B(t*)` | `0.52396`, `0.189632`, `≈(1.0910,1.1280)` |

---

## 9. Code-mapping table (synthesis targets)

| symbol / result | function (`centroid_tile_demo.py`) | notes |
|---|---|---|
| `C` anchor (M1,M3) | `_incenter` [:552], `_anchor_points` [:570] | general incenter; = centroid for equilateral |
| `T_i = C+c_i v_i` (all) | `compute_T` [:806], `(M,3)` branch [:827] | **intended per-vertex model already here** |
| single-tile `T` | `construct_triangle_tile` [:673] (`T=C+c*(P−C)` [:691]) | **scalar `c` only — generalize to length-3** |
| `T1A,T1B` joint, cross-arm `Δ` (M2,M6) | `compute_links` [:915] K=2 hexagon `[T1A,F_A,P1,P1,F_B,T1B]` | `Δ` = the closing T-to-T cross-arm |
| arms, `tt_edge` (M5–M8) | `_joint_arms` [:182] | classifies cross-arm / inner edge vs leg |
| ½ midpoint cap (M2,M4,M6) | `_arm_backoff_bound` [:259]; `_FILLET_TT_FRAC=0.5` [:251] | T-to-T → ½·len; leg → 1.0·len |
| `S0,S1,Q,B(t)` (M5,M7,M8) | `_build_joint_bridge` [:318], back-off [:362] | `center=Q`; per-arm mode → independent `d0,d1` |
| `B(t)` sampling, `t`-grid (M7,M8) | `_quadratic_bezier` [:133] | `np.linspace(0,1,n)` ↔ `t` |
| min-neck / fracture diagnostic (M6,M8) | `joint_radii` [:410] | binding (min) arm radius; extend to min-ligament |

---

## 10. Open items the synthesis must decide

1. **Thickness definition** (flag 4): confirm "two opposing curves = reflection
   through `Q`" (M5/M7/M8) and "shared cross-arm ligament" (M6), or substitute a
   different opposing boundary (arc-to-chord sagitta; literal adjacent-hole arc).
   All machinery above is reusable — only the distance functional changes.
2. **`d2` semantics** (flag 5): cross-arm gap-side (used here, controls the neck)
   vs §6.2's inner-side-other-panel (does not).
3. **`P012` → `P002`/`d0020`** (flag 1) in any d-notation that lands in code.
4. **Free the proportions** (flag 6): replace the single slider's `d0=d·½`,
   `d1=d` coupling with independent per-arm `d0,d1,(d2)` in `per-arm` mode.
5. **Generalize `construct_triangle_tile`** to a length-3 `c` (flag 3) so the
   per-vertex results are reachable through the single-tile path too.

— End of M1–M8 reference. Each section corresponds to `prompts/MATH_M*.md`;
numbers verified against the live code at the §5.2 canonical configuration.
