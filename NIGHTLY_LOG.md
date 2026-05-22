# Nightly Autonomous Work Log

**Branch:** `nightly/auto-2026-05-22`
**Started:** 2026-05-22
**Engineer:** Claude (autonomous overnight run)

This file is the cross-iteration state. Read it FIRST every iteration.

---

## Task checklist

| # | Task | Status |
|---|------|--------|
| 3 | Zoom-to-cursor in the 3D view | DONE (commit) |
| 2 | SCAD intact — integration test guarding to_scad | IN PROGRESS |
| 1 | Bezier-curve edges before export | PENDING |
| 5 | Tessellation generator | PENDING |
| 4 | Generalized Poisson's ratio on triangle edge vectors | PENDING |

Working order: 3 → 2 → 1 → 5 → 4 (risk-managed, per prompt).

---

## Decisions & assumptions

- **Baseline.** Branched from `claude/epic-greider-02752d`, which had substantial
  *uncommitted* WIP (mode-11 bipartite, coordinates panel, preset v5, edge-flip GUI).
  Verified green before touching anything: `test_regression.py` 5/5,
  `test_bipartite + test_preset_v5 + test_coordinates_panel` 47/47. Committed that
  pre-existing WIP as a baseline checkpoint commit so my own increments diff cleanly
  on top. Excluded from commits: `.claude/` (local config — added to .gitignore) and
  the loose root scratch presets `preset*.json`/`presetEq*.json` (unreferenced by any
  test/code — left untracked, not mine to commit).
- **Env.** Python 3.14.3, pyvista 0.47.3, vtk 9.6.1, PyQt6. GUI tests need
  `QT_QPA_PLATFORM=offscreen`.
- **Task 3 wiring approach.** No existing wheel handling in `views.py`. Will add a
  high-priority (10.0) observer on the raw VTK interactor for
  `MouseWheelForward/BackwardEvent` and abort the default center-dolly via
  `interactor.GetCommand(tag).SetAbortFlag(1)` (verified both APIs exist in vtk 9.6).
  Pure, testable core = `dolly_toward_cursor(cam_pos, focal, target, factor)` which
  uniformly scales camera position + focal point toward the cursor world-point
  (fixed-point of the scaling => cursor point stays put). Also scales parallel-scale
  by 1/factor when the camera is in parallel projection so it works in both modes.
  All VTK wiring guarded behind `self.interactor is not None` (headless-safe).

---

## Per-iteration notes

### Iteration 1 (2026-05-22)
- Read CLAUDE.md + SPEC.md end-to-end. Mapped auxetic/ + auxetic_studio/ + tests/.
- Established green baseline; created branch + this log.
- Studied export.py / geometry.py / lattice.py / views.py for the task surface.
- Probed VTK abort + override mechanisms (all available).
- **Next step:** commit baseline checkpoint, then implement task 3.

### Iteration 2 (2026-05-22) — Task 3 zoom-to-cursor
- Extracted pure math into new `auxetic_studio/camera_controls.py`
  (`dolly_toward_cursor`, `ZOOM_WHEEL_STEP`) — Qt/VTK-free so it unit-tests
  without the GUI stack. Re-exported by views.py and used by View3D.
- Wired `View3D._install_zoom_to_cursor` (high-prio wheel observers on the raw
  VTK interactor, abort default dolly via `GetCommand(tag).SetAbortFlag(1)`),
  `_zoom_to_cursor`, `_cursor_world_point` (display→world on focal plane),
  `_on_wheel_forward/backward`, `_abort_event`. All guarded on
  `self.interactor is None`. Parallel-projection handled by also scaling
  ParallelScale by 1/factor.
- Test `tests/test_zoom_to_cursor.py`: 18 pure-math tests (identity, fixed
  point, distance scaling, view-direction preservation, centred-zoom =
  preset-safe, in/out round-trip, bad-factor/non-finite no-ops, no aliasing).
- **Important env finding (recorded for later GUI tests):** constructing a
  pyvistaqt `View3D` in a *small* offscreen pytest session reliably aborts the
  process at teardown (EXIT 127, no junit xml) — it crashes even when a test
  only reads a long-standing attribute like `has_grid`, so it is NOT my code.
  `test_view3d_navigation.py` survives only because of its 6-test composition.
  Decision: keep new test files **pure / GUI-stack-free** wherever possible;
  do widget-level GUI assertions only by extending an already-stable GUI test
  module or via `MainWindow(headless_3d=True)` with enough tests in the file.
  I dropped the (supplementary) headless-widget no-op tests for task 3 rather
  than risk suite flakiness; the guards are trivial early-returns.
- Verified: zoom file 18/18 (solo x3 + in company), regression 5/5, load-bearing
  `test_app` GUI STL-diff + nav 9/9 — all EXIT 0.
- **Next step:** Task 2 — add `tests/test_scad_export.py` integration test that
  `to_scad` emits valid OpenSCAD for a representative mode (assert structure:
  `union(){`, `cylinder(`, `polyhedron(points=[...],faces=[...]`, balanced
  braces, nonzero strut+triangle counts). Must guard the bezier work in task 1.

---

## Final summary
_(written at the end of the run)_
