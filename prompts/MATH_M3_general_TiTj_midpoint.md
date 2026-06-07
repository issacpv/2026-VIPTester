# M3 — General midpoint & displacement between Tᵢ and Tⱼ at shrinks cᵢ, cⱼ

> Math-derivation session. First read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md)
> (§2.1 incenter, §2.2, §5, §8). Obey §8. This **generalises M1** off the
> equilateral special case to arbitrary triangles and a general anchor.

## Setup

Two triangles `Ti`-tri and `Tj`-tri (a neighbouring pair across a shared edge,
or any two inner-triangle vertices of interest). For each, the anchor is the
**incenter** (general formula, §2.1):
```
C = (a·P0 + b·P1 + c_w·P2)/(a+b+c_w),   a=|P1−P2|, b=|P0−P2|, c_w=|P0−P1|
T = C + c·(P − C)
```
Let `Tᵢ = Cᵢ + cᵢ·(Pᵢ − Cᵢ)` and `Tⱼ = Cⱼ + cⱼ·(Pⱼ − Cⱼ)`, where `Pᵢ, Pⱼ` are
the chosen corners (e.g. the two copies of a shared Delaunay vertex, as in M1).

## Derive (show ALL work)

1. Restate the incenter formula and write `Cᵢ, Cⱼ` symbolically in terms of the
   triangles' vertices (keep general; do **not** assume equilateral).
2. **Displacement vector** `Δ(cᵢ, cⱼ) = Tⱼ − Tᵢ` in closed form. Group it into a
   constant part (the `cᵢ=cⱼ=0` anchor-to-anchor vector `Cⱼ − Cᵢ`) plus the
   `cᵢ`- and `cⱼ`-linear parts. Show `Δ` is **affine** (linear) in `(cᵢ, cⱼ)`.
3. **Midpoint** `M(cᵢ, cⱼ) = ½(Tᵢ + Tⱼ)` in closed form; identify the plane
   region/locus it sweeps as `(cᵢ, cⱼ)` range over `[0,1]²` (an affine image of
   the unit square → a parallelogram; give its corners).
4. Specialise to the §5.2 equilateral pair (shared corner `P1`) and confirm you
   recover M1's `Δ` and `M` exactly.

## Verify

Reproduce the M1 numeric anchors (`c=0.5,0.5 → Δ=(0.5,0.288675)`,
`M=(1.25,1.299038)`) as the equilateral specialisation. Then give one
**non-equilateral** worked example of your choosing (state the vertices) with
numbers, so the general formula is exercised.

## Map to code

Relate to `_incenter`, `compute_T`, and the per-vertex `c` model. Note this is
the building block for the thickness/thinnest work (M5–M8), where displacements
between specific `T`s and feet drive the neck width.

## Deliverable

General closed forms `Δ(cᵢ,cⱼ)`, `M(cᵢ,cⱼ)`, the affine/parallelogram locus, the
equilateral check, a non-equilateral example, and the code mapping.
