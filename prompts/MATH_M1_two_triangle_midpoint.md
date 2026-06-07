# M1 — Midpoint & displacement between two neighbouring inner triangles (T1A↔T1B)

> Math-derivation session. Run in the repo so you can read the code, but derive
> for the **intended** model. First read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md)
> — especially §2 (definitions), §5.2 (this exact example), §6 (d-notation),
> and §8 (conventions). Obey §8 in full.

## Setup (do not change without flagging)

Two neighbouring **equilateral** triangles sharing edge `P1–P2`, with
```
P0 = (0, 0),  P1 = (1, √3),  P2 = (2, 0),  P3 = (3, √3)
A = (P0,P1,P2)   B = (P1,P2,P3)
```
Anchor = incenter = centroid (equilateral). Inner vertices use **per-vertex `c`**
(Task B's model): `T_i = C + c_i·(P_i − C)`.

`T1A` is triangle A's inner vertex at the shared corner `P1`, controlled by
`c1A`. `T1B` is triangle B's inner vertex at `P1`, controlled by `c1B`.

## Derive (show ALL work)

1. Restate the notation and confirm the §5.2 coordinates for `C_A, C_B`,
   `P1−C_A`, `P1−C_B` from first principles.
2. Write `T1A(c1A)` and `T1B(c1B)` explicitly.
3. The **displacement vector** `Δ(c1A, c1B) = T1B − T1A` as a closed-form
   parametric function of `c1A, c1B`. Simplify fully.
4. The **midpoint** `M(c1A, c1B) = ½(T1A + T1B)` as a closed-form parametric
   function. Describe the **line/locus** it traces as each of `c1A`, `c1B` varies
   (hold one fixed, vary the other): direction, and the point it passes through.
5. Give `|Δ|` (the gap length) and the unit direction `Δ/|Δ|` as functions of
   `c1A, c1B`.

## Verify

Plug in `c1A = c1B = 0.5` and confirm `Δ = (0.5, 0.288675)`,
`M = (1.25, 1.299038)` (from §5.2). Also check the degenerate limits
`c1A=c1B=1` (both → `P1`, so `Δ→0`, `M→P1`) and `c1•=0`.

## Map to code

Relate `T1A/T1B` to `compute_T` / `construct_triangle_tile` and the per-vertex
`c` generalisation (§4.2). Note that current code uses a single `c` per triangle,
so this per-vertex result is for the **intended** model.

## Deliverable

A rigorous derivation with the final closed forms for `Δ(c1A,c1B)` and
`M(c1A,c1B)`, the locus description, the numeric checks, and the code mapping.
This is the foundation for **M2** (d0) — keep your notation reusable.
