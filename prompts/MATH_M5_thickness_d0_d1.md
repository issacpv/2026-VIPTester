# M5 — Bézier thickness from d0 and d1 on both sides of a joint

> Math-derivation session. First read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md)
> (§2.3, §2.7, §6, §8) and the results of **M2 (d0)** and **M4 (d1)**. Obey §8.
> **Before deriving, restate the full d-notation and the "thickness" definition,
> and flag the `P012` typo (§6.1) — confirm the intended foot.**

## Notation to pin down first

- Per-arm back-off proportion `d⟨abc⟩⟨s⟩`, `s=0` leg (bound = full leg),
  `s=1` inner (bound = ½·inner edge). Examples: `d0010`=P001·leg,
  `d1011`=P101·inner, `d2021`=P202·inner, and **`d0120`** (decodes to P012·leg,
  which is impossible — resolve to the intended foot, likely `P002`→`d0020` or
  `P112`→`d1120`).
- A single fillet arc bridges **two consecutive arms** of a joint — "both sides":
  one **leg** side governed by a `d⟨·⟩0` (the `d0` family from M2) and one
  **inner** side governed by a `d⟨·⟩1` (the `d1` family from M4). State which two
  feet/arms you pair (use a concrete joint from §5.2).
- **Define "thickness" precisely.** A curve has no width; define the material
  **neck width** the fillet leaves — pick two specific opposing boundary curves
  (e.g. the back-to-back Bézier arcs of two panels meeting along a strut, or the
  arc vs the panel edge) and the direction in which width is measured
  (perpendicular to the strut axis). Write the definition explicitly and keep it
  fixed.

## Derive (show ALL work)

1. Using M2/M4, write the two **back-off (start) points** of the arc in closed
   form: the leg-side point `S0` (from `d0`/`d⟨·⟩0`) and the inner-side point
   `S1` (from `d1`/`d⟨·⟩1`), as functions of the relevant `c`'s and the two
   proportions.
2. Write the quadratic Bézier for the arc with control point = joint centre
   `Q`: `B(t) = (1−t)²S0 + 2(1−t)t·Q + t²S1`, `t∈[0,1]` (§2.7).
3. From your thickness definition, derive **thickness as a function of `d0` and
   `d1`** (the two sides): `W(d0, d1)` in closed form. Show how each proportion
   widens/narrows the neck; give `∂W/∂d0`, `∂W/∂d1` and their signs.
4. Identify the `(d0, d1)` that **maximize** material (thickest) and that drive
   the neck toward zero, within `[0,1]²`.

## Verify

Evaluate `W` on the §5.2 joint at a couple of `(d0,d1)` pairs with numbers;
sanity-check limits (`d0 or d1 → 0`, and `→1`) against geometric intuition.

## Map to code

Tie `S0/S1/Q` to `_build_joint_bridge` / `_joint_arms` and the **per-arm** mode
of Task C (§4.3). State what the program would compute to report this thickness.

## Deliverable

The pinned thickness definition, closed-form `W(d0,d1)`, its monotonicity,
extremal `(d0,d1)`, numeric checks, and code mapping. Sets up **M6** (thinnest,
with `d2`) and **M7** (thickness vs `t`).
