# Task C — Bézier fillet bounds + uniform/per-arm toggle

> Paste this into a fresh Claude Code session in its own worktree. Implement
> independently of Tasks A and B.

## Context

Work in [`centroid_tile_demo.py`](centroid_tile_demo.py). First read
[`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md) (§2.3, §2.7, §4.3, §6, §7).
Relevant code: `_joint_arms`, `_joint_radius_from_arms`, `_build_joint_bridge`,
`build_joint_bridges`, `joint_radii`, `_quadratic_bezier`, and the `fillet`
slider + `redraw` + `_plot_radius_vs_c` in `show()`.

## The problem

Today every joint arm backs off by a single **uniform** radius
```
radius = d · min( 0.5·(shortest leg) , 0.25·(shortest inner edge) )     (CURRENT)
```
The user wants different bounds, and the ability to back off **per-arm**.

## What to build

**Replace the bounds** with:
- inner-edge arm bound = **½ · (its inner-triangle edge length)** — the
  **midpoint of the inner triangle** (was ¼).
- leg arm bound = **1 · (its full perpendicular leg length)** (was ½).

**Add a UI toggle** (e.g. a `RadioButtons` "fillet mode: uniform | per-arm",
or a checkbox) selecting how the bounds are applied:
- **Uniform:** one radius for all arms at the joint,
  `radius = d · min( 0.5·(shortest inner edge) , 1.0·(shortest leg) )`.
- **Per-arm:** each arm backs off **independently** by `d · (its own bound)` —
  inner arms by `d·½·innerlen`, leg arms by `d·1.0·leglen`. The back-off points
  may then sit at different radii, so the Bézier "flower" is asymmetric.

In **per-arm** mode, `_build_joint_bridge` must place each back-off point at its
own arm's radius (not a shared one) before stitching consecutive points with the
quadratic Bézier (control point = joint centre), exactly as today otherwise.

Keep `d = 0` → sharp single-point join in both modes.

## Constraints

- **`is_inner_edge` semantics unchanged:** an arm is an inner-triangle edge iff
  its source polygon index `< n_inner` (the first `n_inner` polys are the inner
  triangles). Legs = perpendicular wing legs and link cross-arms.
- Update `joint_radii` and the **"radius vs c"** plot to be consistent with the
  selected mode (in per-arm mode the per-joint "radius" is no longer a single
  number — decide and document: e.g. report the binding/min arm radius, and say
  so in the plot title).
- Don't change the `c` model or actuation (Tasks B/A own those).

## Acceptance criteria

- [ ] New bounds in effect: with the toggle on **uniform**, radius uses
      `min(0.5·inner, 1.0·leg)`; the fillet visibly reaches further than before.
- [ ] Toggle to **per-arm**: inner and leg arms back off by different amounts
      (asymmetric flower); switching back to uniform restores the symmetric one.
- [ ] `d = 0` still yields a sharp join in both modes.
- [ ] "radius vs c" plot and `joint_radii` behave sensibly under both modes
      (documented).
- [ ] New tests in `tests/` cover: the new bound values, uniform-vs-per-arm
      back-off distances at a known joint, and `d=0` sharpness.

## Deliverable

Implement it, add tests, run them. In your final message list **exactly which
shared functions you changed** (`_joint_radius_from_arms`, `_build_joint_bridge`,
`build_joint_bridges`, `joint_radii`, `redraw`, the slider/toggle setup) and how
— this feeds the later synthesis with Tasks A and B (see
`CENTROID_TILE_SPEC.md` §7). Do **not** pull in Task A/B changes.
