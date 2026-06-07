# M6 — Thinnest point of the Bézier from d0 and d2 on both sides of a joint

> Math-derivation session. First read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md)
> (§2.7, §6, §8) and the results of **M2 (d0)** and **M5 (thickness W)**. Obey §8.
> **Before deriving, restate the d-notation and explicitly define `d2`, then
> confirm with the user** (per §6.2, `d2`'s exact role is to-be-confirmed).

## What to pin down first

- **Define `d2`.** Candidate (from §6.2): the inner-side (`s=1`) proportion on
  the **other** panel meeting the joint, so the neck width depends on `d0` (one
  side) and `d2` (the opposite side). State your definition precisely; if you
  adopt a different one, justify it and flag for the user.
- Reuse the **thickness/neck definition** you pinned in M5 (state it again).

## Derive (show ALL work)

1. Write the two governing back-off points in terms of `d0` and `d2` (closed
   form, via M2 and your `d2` definition), plus the joint centre `Q`.
2. Form the relevant neck-width function `W(d0, d2)` (or the distance between the
   two opposing arcs parameterized appropriately).
3. **Find the thinnest point**: minimize the neck width. Do it two ways and
   reconcile:
   - **(a) over the geometry** — the location along the strut/arc where the
     opposing boundaries are closest (the structural neck), and
   - **(b) over the parameters** — how `min` width depends on `(d0, d2)`, and the
     `(d0, d2)` that minimize it within `[0,1]²`.
   Give the **coordinates** of the thinnest point and the **minimum width
   value** in closed form. Use calculus (set derivative = 0), show the stationary
   condition, and confirm it's a minimum (second-order / boundary check).

## Verify

Numerically locate the thinnest point on the §5.2 joint for a chosen `(d0,d2)`;
confirm the analytic minimizer matches a fine sampling of the width.

## Map to code

Explain what the program would compute to **report the minimum neck thickness**
at each joint (the fracture-critical quantity) under the per-arm fillet (§4.3),
and how it relates to the "radius vs c" diagnostic plot.

## Deliverable

The `d2` definition, closed-form neck width, the thinnest-point coordinates and
minimum value, the optimizing `(d0,d2)`, numeric verification, and code mapping.
