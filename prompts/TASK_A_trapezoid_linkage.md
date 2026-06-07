# Task A — Flexing trapezoid (four-bar linkage actuation)

> Paste this into a fresh Claude Code session in its own worktree. Implement
> independently of Tasks B and C.

## Context

Work in [`centroid_tile_demo.py`](centroid_tile_demo.py). First read
[`CENTROID_TILE_SPEC.md`](CENTROID_TILE_SPEC.md) (§2, §2.6, §4.1, §7) and study
two reference files:
- [`quad_linkage_demo.py`](quad_linkage_demo.py) — the four-bar linkage core:
  `circle_intersections`, `linkage_states`, `is_simple`, and the trick of
  following **one assembly mode continuously** so the mechanism moves smoothly.
- `auxetic/simulation.py` (`Simulator.sweep_theta(collision_stop=True)`,
  `res.collision_at_theta`) and `auxetic/bipartite.py` (`jamming_angle`) — a
  working reference solver for a sibling structure, for the collision-stop /
  jamming-angle pattern only.

## The problem

When a triangle's `c` differs from a neighbour's (via the existing per-triangle
`c` override — shift-click a triangle, move the `c` slider), the two inner-
triangle edges along their shared Delaunay edge have **different lengths**, so
the negative space between the inner triangles is a **trapezoid**. The current
actuation (`_bfs_propagate`, which only **translates** inner triangles and
assumes the uniform-`c` auxetic identity) no longer closes — the structure
looks locked or tears.

## What to build

Make the `theta` actuation **physically simulate the flexing trapezoid as a
four-bar linkage**, and let it rotate until it **cannot rotate any further**:

1. **Inner triangles may rotate**, not just translate. Replace/augment
   `_bfs_propagate` so that, when neighbours have mismatched `c`, each
   downstream inner triangle's pose (rotation + translation) is solved from the
   four-bar linkage formed by the trapezoidal gap, following one assembly mode
   continuously across the `θ` sweep (à la `linkage_states`). Under **uniform
   `c`** the solution must reduce **exactly** to today's behaviour (verify the
   reduction).
2. **Stop condition = "can no longer rotate":** the first `θ` (in the swept
   direction) at which **either** the linkage cannot close (the circle–circle
   intersection placing the next corner vanishes) **or** any panel pair
   **self-intersects** (reuse the `_segments_cross` / `_quad_fixed_lead_edge`
   machinery already in the file). Report/limit `θ` to the achievable range.
3. **UI:** clamp the `theta` slider to the achievable `[θ_min, θ_max]` for the
   current `c` configuration (or visually indicate the jamming limit), so the
   user can sweep "until they are no longer able to rotate." Keep `spin`,
   `fillet`, `anchor`, export, etc. working.

## Constraints

- **Uniform-`c` regression:** with all `c` equal, output (T's, wings, links) and
  the rendered figure must be **unchanged** vs. current `main` for representative
  `θ`. Add a test asserting this.
- Keep the math vectorized where reasonable; the per-joint linkage solve can be
  a clear Python loop if needed (correctness first).
- Don't change the `c` model's shape contract beyond what you need (Task B owns
  per-vertex `c`); assume scalar/`(M,)` `c`.

## Acceptance criteria

- [ ] Mismatched-`c` neighbours produce a **rotatable** mechanism that visibly
      flexes the trapezoid and **halts at a well-defined jamming angle** (no
      tearing, no panel overlap past the limit).
- [ ] Uniform-`c` behaviour is byte-for-byte/coordinate-for-coordinate identical
      to current `main` (regression test passes).
- [ ] The achievable `θ` range is computed and the slider respects it.
- [ ] New tests in `tests/` cover: linkage closure, the self-intersection stop,
      continuous assembly-mode following, and the uniform-`c` reduction.

## Deliverable

Implement it, add tests, run them. In your final message list **exactly which
shared functions you changed** (`_bfs_propagate`, `tile_state`/`compute_wings`,
`compute_links`, `redraw`, slider setup) and how — this feeds the later
synthesis with Tasks B and C (see `CENTROID_TILE_SPEC.md` §7). Do **not** pull
in Task B/C changes.
