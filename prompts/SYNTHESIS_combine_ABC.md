# Synthesis — combine Tasks A + B + C into one `centroid_tile_demo.py`

> Use this once Tasks A, B, and C are each implemented to your satisfaction in
> their own worktrees. Paste this prompt and attach/point to all three versions.

## Inputs

Three diverged copies of [`centroid_tile_demo.py`](centroid_tile_demo.py):
- **A** — flexing-trapezoid four-bar linkage actuation (rotates inner triangles;
  jamming-angle stop). Touched: `_bfs_propagate`, `tile_state`/`compute_wings`,
  `compute_links`, `redraw`, theta-slider clamp.
- **B** — per-vertex `c` (`T_i = C + c_i(P_i−C)`); ctrl-click-near-vertex pick.
  Touched: `compute_T`, `effective_c`, `state` dict, `on_press`/`on_motion`,
  `on_c`, `redraw`, vertex highlight.
- **C** — new Bézier bounds (½·inner, 1·leg) + uniform/per-arm UI toggle.
  Touched: `_joint_radius_from_arms`, `_build_joint_bridge`,
  `build_joint_bridges`, `joint_radii`, `redraw`, fillet mode toggle.

Read [`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md) §7 (merge hotspots) first.

## Goal

Produce one coherent `centroid_tile_demo.py` with all three features working
**together**, plus a merged test suite that passes.

## Method

1. **Diff each version against the common base** (the `main` they branched from)
   to get three clean, isolated change-sets. Prefer 3-way merge per function over
   hand-copying.
2. **Resolve the shared hot path deliberately** (these are the expected
   conflicts — see §7):
   - `compute_T` must take per-vertex `(M,3)` `c` (**B**), and **A**'s linkage
     and **C**'s inner-edge lengths must read from the resulting `T` — confirm A
     and C consume per-vertex `c` correctly (the trapezoid is now driven by
     per-vertex `c1A,c2A,…`, which is exactly the M-series math model).
   - `redraw()` is edited by all three — reassemble it so it: resolves per-vertex
     `c` (B) → runs the linkage actuation with jamming clamp (A) → builds fillets
     in the selected uniform/per-arm mode with new bounds (C).
   - `on_press` — B's vertex-pick (ctrl-near-vertex) and any A/C UI must coexist;
     keep the proximity disambiguation so ctrl-drag still moves points.
   - State dict — union the new keys (vertex target from B, fillet mode from C,
     any theta-range cache from A) without collisions.
3. **Interactions to verify explicitly:**
   - Per-vertex `c` (B) that creates mismatched neighbours must feed A's
     four-bar linkage (the headline combined behaviour).
   - C's fillet (esp. per-arm) must render correctly on A's rotated inner
     triangles and B's asymmetric inner triangles.
4. **Merge the tests** from all three; add at least one **integration test**
   exercising B→A→C together (per-vertex `c` → flex to jamming → per-arm fillet).
5. Run the full `tests/` suite; fix regressions. Confirm uniform-`c`,
   default-mode output still matches base where each task promised invariance.

## Deliverable

The merged file, the merged+integration tests (passing), and a short report:
what conflicted, how you resolved it, and any behaviour that changed versus the
individual versions.
