# M2 — d0: the Bézier start point as a proportion of the inter-inner-triangle displacement

> Math-derivation session. First read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md)
> (§2, §5.2, §6, §8). Obey §8. **Restate the d0 definition and confirm it before
> deriving** (per the user's "independent proportions / restate-first" rule).

## Definition to formalize

`d0 ∈ [0,1]` is the **proportion of the displacement between the two
neighbouring inner triangles** at which the Bézier curve **start point** sits.
Use the displacement `Δ = T1B − T1A` (and its midpoint `M`) from **M1** as the
"displacement between the inner triangles." (If you believe the relevant
displacement is a different pair — e.g. the inner-edge endpoints `T2A↔T2B`, or
foot-to-foot — state your interpretation, justify it, and proceed; flag for the
user.)

## Derive (show ALL work)

1. State precisely from which endpoint `d0` is measured (e.g. `d0=0` at `T1A`,
   `d0=1` at `T1B`, or measured outward from the midpoint `M`). Pick the
   convention that matches "start point of the Bézier" and justify it.
2. **Parametric equation for the displacement vector** that locates the start
   point: i.e. the vector from the chosen origin to the start point, as a
   function of `d0` (and `c1A, c1B`). Show it is `d0·Δ` (or your justified
   variant) and expand using M1's closed form.
3. **Parametric equation for where it occurs** — the absolute **coordinates** of
   the start point `S0(d0; c1A, c1B)` in the plane. Expand fully.
4. Give the special points: `d0 = 0`, `d0 = ½` (should coincide with `M` if
   measured along `Δ`), `d0 = 1`.

## Verify

At `c1A=c1B=0.5`, `d0=0.5`: confirm `S0 = M = (1.25, 1.299038)`. Check `d0=0`
and `d0=1` give `T1A`, `T1B` respectively (under the `0→T1A, 1→T1B` convention).

## Map to code

This is the **intended** generalisation of the back-off start point in
`_build_joint_bridge` (§2.7, §4.3). Relate `d0` to the per-arm back-off
proportion `d⟨abc⟩0` on the **leg / inter-triangle (`s=0`)** side from §6.1, and
note how it would replace the uniform `radius` placement.

## Deliverable

Closed forms for the start-point displacement vector and its coordinates
`S0(d0; c1A,c1B)`, special points, numeric checks, code mapping. Reuse M1.
