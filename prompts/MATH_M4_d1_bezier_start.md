# M4 — d1: the Bézier start point as a proportion of the displacement to the inner-triangle midpoint

> Math-derivation session. First read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md)
> (§2.2–§2.4, §5.2, §6, §8). Obey §8. **Restate the d1 definition and confirm it
> before deriving.**

## Definition to formalize

`d1 ∈ [0,1]` is the **proportion of the displacement to the midpoint of the
inner-triangle edge** at which the Bézier curve **start point** sits — i.e. the
**inner-side (`s=1`)** back-off, measured along an inner-triangle edge from a
corner `T` toward that edge's **midpoint** (recall the Task C inner bound is the
edge **midpoint**, §4.3).

Pick a concrete inner edge to anchor the derivation: in the §5.2 example, use
triangle A's inner edge `T1A–T2A` along the shared edge, with `T1A` the corner
and the edge midpoint `Mᴬ = ½(T1A + T2A)`. (State your choice; if you think the
relevant "inner triangle" is the *other* panel, justify and proceed.)

## Derive (show ALL work)

1. Write `T1A(c1A)`, `T2A(c2A)` (from §5.2) and the inner-edge midpoint
   `Mᴬ = ½(T1A + T2A)` in closed form. Note that under **uniform** `c`
   (`c1A=c2A=c_A`) this edge is `c_A·(P1−P2)` and `Mᴬ` is the midpoint of a
   length-`2c_A` segment.
2. **Parametric equation for the displacement vector** locating the start point:
   the vector from `T1A` toward `Mᴬ`, scaled by `d1` — i.e. `d1·(Mᴬ − T1A)` —
   expanded in closed form (as a function of `d1, c1A, c2A`).
3. **Parametric equation for where it occurs** — absolute coordinates of the
   start point `S1(d1; c1A, c2A) = T1A + d1·(Mᴬ − T1A)`. Expand fully.
4. Special points `d1 = 0` (`→ T1A`), `d1 = 1` (`→ Mᴬ`, the edge midpoint).
   Relate to the §4.3 inner bound (`½·innerlen`) — show `d1=1` reaches exactly
   the midpoint.

## Verify

At `c1A=c2A=0.5` (uniform), compute `T1A, T2A, Mᴬ`, then `S1` for `d1=0,½,1`.
Confirm `d1=1` lands on the inner-edge midpoint and `|Mᴬ − T1A| = ½·innerlen`.

## Map to code

Relate `d1` to the per-arm back-off proportion `d⟨abc⟩1` on the **inner (`s=1`)**
side (§6.1), and to the new inner bound in `_joint_radius_from_arms` /
`_build_joint_bridge` (§4.3). Contrast with **M2**'s `d0` (the leg/`s=0` side).

## Deliverable

Closed forms for the start-point displacement vector and coordinates
`S1(d1; c1A,c2A)`, special points, numeric checks, code mapping. Keep notation
consistent with M2 so M5 can combine `d0` and `d1`.
