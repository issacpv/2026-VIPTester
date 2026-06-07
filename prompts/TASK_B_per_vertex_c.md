# Task B — Per-vertex `c` (edit one inner-triangle vertex)

> Paste this into a fresh Claude Code session in its own worktree. Implement
> independently of Tasks A and C.

## Context

Work in [`centroid_tile_demo.py`](centroid_tile_demo.py). First read
[`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md) (§2.2, §3, §4.2, §7). Relevant
existing code: `compute_T`, `effective_c()`, the `state` dict, `on_press` /
`on_motion` / `on_release`, `redraw`, `refresh_static`, the `c` slider callback
`on_c`. (For reference on stable per-triangle `c` and undo, see
`tests/test_per_triangle_c.py`, though this demo has no undo stack.)

## The problem

Today `c` is global or per-**triangle** (`state["c_over"]`, applied to all three
`T_i` of a triangle). The user wants to shrink **one** inner-triangle vertex
independently:
```
T_i = C + c_i·(P_i − C)      with c0, c1, c2 independent
```

## What to build

1. **Per-vertex model.** Generalise the override store so a selected triangle
   can carry independent `c0, c1, c2`. Extend `compute_T` to accept a `(M, 3)`
   per-vertex array (broadcast: `C[:,None,:] + c[:,:,None]·v`). **Preserve the
   scalar and `(M,)` fast paths** so untouched triangles/global `c` are
   unchanged.
2. **Selection flow (decision: proximity disambiguation).**
   - **Shift-click** a triangle → select it (unchanged).
   - **Ctrl-click NEAR one of the selected triangle's three vertices**
     (within a pixel threshold, like the existing 15px point-pick in
     `on_press`) → set that vertex as the **c-edit target**.
   - **Ctrl-drag elsewhere** (not near a selected-triangle vertex) → **still
     moves the point**, exactly as today.
   - The `c` slider, when a vertex target is active, edits **only that vertex's**
     `c_i`; with a triangle selected but no vertex target, it edits all three
     (current per-triangle behaviour); with nothing selected, it edits global
     `c`.
3. **Feedback.** Highlight the picked vertex (e.g. a marker / colored dot) and
   reflect the active target in the title text. Re-triangulation (move/add
   point) must reset the vertex target along with `sel_tri`/`c_over`.

## Constraints

- **Disambiguation must be robust:** a ctrl-press that is near a selected-
  triangle vertex picks that vertex (no drag); otherwise it falls through to the
  existing nearest-point drag. Document the threshold.
- Wings, feet, links, fillets, STL export must all consume per-vertex `c`
  correctly (they already read `T`; just make sure nothing assumes
  `T_i − T_j = c·(P_i − P_j)`).
- Default/global and per-triangle behaviour unchanged when no vertex is picked.

## Acceptance criteria

- [ ] Shift-select a triangle, ctrl-click one of its corners, drag the `c`
      slider → only that inner vertex moves along `C→P_i`; the other two stay.
- [ ] Ctrl-drag on a point not belonging to the current vertex-pick still moves
      the point and re-triangulates.
- [ ] `compute_T` handles scalar, `(M,)`, and `(M,3)` `c`; scalar/`(M,)` paths
      give identical results to current `main`.
- [ ] Picked-vertex visual feedback; target resets on re-triangulation.
- [ ] New tests in `tests/` cover `compute_T` per-vertex math and the
      pick-vs-drag disambiguation logic (factor the hit-test so it's testable
      headlessly).

## Deliverable

Implement it, add tests, run them. In your final message list **exactly which
shared functions/state you changed** (`compute_T`, `effective_c`, the `state`
dict keys, `on_press`/`on_motion`, `on_c`, `redraw`) and how — this feeds the
later synthesis with Tasks A and C (see `CENTROID_TILE_SPEC.md` §7). Do **not**
pull in Task A/C changes.
