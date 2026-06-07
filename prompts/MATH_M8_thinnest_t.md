# M8 — Thinnest part of the Bézier in the parameter t (minimize w(t))

> Math-derivation session. First read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md)
> (§2.7, §6.3, §8) and the **M7** result `w(t)`. Obey §8. Use the **same**
> thickness definition as M7 (restate it).

## Derive (show ALL work)

1. Take `w(t)` (or, to avoid square roots, the squared width `w(t)²`) from M7 and
   find its **minimum over `t ∈ [0,1]`**:
   - Compute `d/dt [w(t)²] = 0`; solve for the stationary `t*` in closed form
     (state the polynomial and its roots; a quadratic Bézier width is low-degree,
     so this should be explicit).
   - Check the endpoints `t=0, 1` against interior stationary points; apply the
     second-derivative (or sign) test to confirm `t*` is the **minimum**.
2. Give `t*` and the **minimum thickness** `w(t*)` in **closed form** as
   functions of `S0, S1, Q` (hence of the `c`'s and the `d`'s). Also give the
   **coordinates** `B(t*)` of the thinnest point on the curve.
3. State the conditions under which the minimum is at an **endpoint** vs the
   **interior** (degenerate cases), and what `t*` means geometrically (the neck).
4. Reconcile with **M6**: the parameter-space thinnest (`M6`, over `d0,d2`) and
   the curve-parameter thinnest (`M8`, over `t`) should agree on the located neck
   point for the same configuration — show they do (or explain the difference).

## Verify

For the §5.2 joint (same `c`'s and `d`'s as M7), confirm `t*` matches the
argmin of a fine `t`-sampling of `w(t)`, and that `w(t*)` matches M6's minimum
width for the corresponding `(d0,d2)`.

## Map to code

Explain how the program would compute and mark the thinnest point per joint
(both the `t*` and the world coordinates), and how this is the fracture-critical
read-out for the fillet design.

## Deliverable

Closed-form `t*`, `w(t*)`, and `B(t*)`; endpoint-vs-interior conditions; the M6
reconciliation; numeric verification; and code mapping.
