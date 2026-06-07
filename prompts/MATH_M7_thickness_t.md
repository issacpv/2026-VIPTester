# M7 — Bézier thickness as a function of the parameter t

> Math-derivation session. First read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md)
> (§2.7, §6.3, §8) and the **M5** thickness definition. Obey §8. Restate the
> thickness definition before deriving and keep it identical to M5/M6.

## Setup

A single joint fillet arc is the quadratic Bézier
```
B(t) = (1−t)²·S0 + 2(1−t)t·Q + t²·S1,   t ∈ [0,1]
```
with `S0, S1` the two back-off (start/end) points and `Q` the joint centre
(control point) — §2.7. Use the closed forms for `S0, S1, Q` from M2/M4/M5.

## Derive (show ALL work)

1. Pick the thickness/neck definition from M5 and express the local width as a
   function of the Bézier parameter, **`w(t)`**. Be explicit about what is
   measured at parameter `t`: e.g. the distance from `B(t)` to the opposing
   boundary curve (another Bézier `B̃(t̃)`, the panel edge, or the strut axis) —
   state it and the matching/projection rule between the two curves.
2. Compute `B(t)`, the tangent `B'(t) = 2(1−t)(Q−S0) + 2t(S1−Q)`, and (if your
   width uses normal offset) the unit normal `n(t)`. Assemble `w(t)` in closed
   form.
3. Simplify `w(t)` to a polynomial/rational function of `t` (state its degree)
   with coefficients in terms of `S0, S1, Q` (hence in `c`'s and the `d`'s).
4. Tabulate `w(0)`, `w(½)`, `w(1)` and describe the profile shape (monotone?
   single bump? — connect to the unimodal-bump behaviour of a quadratic Bézier
   offset).

## Verify

For the §5.2 joint with chosen `c`'s and `d`'s, evaluate `w(t)` on a `t`-grid and
plot/print the profile; confirm endpoints and shape match the closed form.

## Map to code

Relate `t` to `_quadratic_bezier`'s `np.linspace(0,1,n)` sampling and explain
how the program could output a per-`t` thickness profile for a joint.

## Deliverable

Closed-form `w(t)` (with stated degree and coefficients), the endpoint/midpoint
values, the profile description, numeric verification, and code mapping. Sets up
**M8** (minimize `w(t)`).
