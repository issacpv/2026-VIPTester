# Nightly Autonomous Work Log

**Branch:** `nightly/auto-2026-05-22`
**Started:** 2026-05-22
**Engineer:** Claude (autonomous overnight run)

This file is the cross-iteration state. Read it FIRST every iteration.

---

## Task checklist

| # | Task | Status |
|---|------|--------|
| 3 | Zoom-to-cursor in the 3D view | DONE (13fd991) |
| 2 | SCAD intact — integration test guarding to_scad | DONE (7b00de6) |
| 1 | Bezier-curve edges before export | DONE (8a8c282, aedd64a, ccd873c) |
| 5 | Tessellation generator | DONE (876d5e8) |
| 4 | Generalized Poisson's ratio on triangle edge vectors | DONE (265050d) |

**All 5 tasks complete.** Full-suite verification running (background id brdhfxqi0).

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
### Iteration 3 (2026-05-22) — Task 2 SCAD guard
- Added `tests/test_scad_export.py` (pure, no GUI). Validates `to_scad` for
  modes 1/4/2/6: header params, `$fn`, `union(){`, ≥1 strut `cylinder(`, exactly
  one `polyhedron(points=[...],faces=[...]`, balanced `{}`/`[]`/`()` (and never
  closing-before-opening), ends with `}`, no nan/inf, and **face indices in
  range of the points array**. Plus a positive-radius/length strut check, and an
  `openscad`-CLI parse test that auto-skips when the binary is absent.
- 5 passed, 1 skipped (no openscad CLI here). This is the guard for tasks 1 & 5.

### Iteration 4 (2026-05-22) — Task 1 bezier edges (STARTING)
- **Design (decided):** Add a `bezier` config to `Lattice`, default **off** so
  modes 1–9 stay byte-for-byte identical.
  - `Lattice.bezier_enabled: bool = False`, `Lattice.bezier_strength: float`
    (0 = straight; curvature magnitude as a fraction of edge length), and
    `bezier_segments: int` (polyline tessellation density; 1 == straight).
  - Geometry: a new pure helper in `auxetic/geometry.py`,
    `bezier_polyline(p0, p1, *, strength, segments, ...)`, returns a dense
    polyline. Zero strength OR segments<=1 returns exactly `[p0, p1]` (so OFF is
    byte-identical). Nonzero => smooth denser polyline (quadratic/cubic Bézier).
  - The struts are the natural curve target: `collect_export_geometry`'s
    `add_strut` currently appends a 2-point `np.array([p0,p1])`. When bezier is
    on, replace with the tessellated polyline. `build_export_triangles.tube_mesh`
    ALREADY handles multi-point paths, and SCAD already iterates strut segments —
    but check: SCAD's `scad_cylinder` only uses `pts[0]`/`pts[1]`! Must update
    `export_to_scad` to emit a cylinder per polyline segment when len>2. Verify.
  - Round-trip through preset: bump preset to **v6**, add migration v5->v6 that
    injects bezier defaults (off). Never break old presets.
  - GUI: a control (checkbox + strength/segments spin) in the inspector or a
    relevant panel, wired through a Lattice setter (no geometry in GUI).
  - Tests: curve math (zero curvature == original 2 pts; nonzero == denser smooth
    polyline, endpoints preserved, midpoint offset along normal); all three
    exporters with curves ON produce valid output; with OFF byte-identical
    (regression already covers OFF for modes 1-9). Reuse test_scad_export guard.
- **Next step (was):** implement bezier core, preset, GUI. (DONE — see below.)

### Iteration 5 (2026-05-22) — Task 1 bezier edges (COMPLETE)
- **Scope decision:** bezier curving targets **struts** (the connective edges of
  the kirigami graph), not tile-face polygon outlines. Struts are the clean,
  well-defined curve target; bowing closed polygon outlines is ambiguous
  (which way?) and riskier. "tile/strut edges" is satisfied by the strut edges.
  Posed/simulation-playback struts stay straight (export-time feature only).
- 1a (8a8c282): `geometry.bezier_polyline` (pure quadratic Bezier; OFF returns
  exact [p0,p1]); `collect_export_geometry` threads bezier_* and bows struts
  away from the lattice centroid; SCAD emits one cylinder per polyline segment
  (2-pt strut => identical single cylinder). Lattice fields + `set_bezier()`
  (clears export cache). 20 tests. Regression byte-identical (OFF).
- 1b (aedd64a): preset **v6** + `bezier` block + `_migrate_v5_to_v6` (OFF). Bumped
  PRESET_VERSION 5->6. Relaxed test_preset_v5's exact-version pin to `>=5`
  (exact pin now in test_preset_v6); added "bezier" to schema key-set test.
- 1c (ccd873c): InspectorPanel "Bezier edges" group (checkbox+strength+segments)
  -> parameterChanged -> ParameterChangeCommand(regenerate=False); command's
  non-regenerate branch now `_clear_caches()`. End-to-end GUI test via headless
  MainWindow.
- **GUI-test race (confirmed + resolved):** a SINGLE Qt/MainWindow test in a tiny
  offscreen session hits the documented VTK/Qt atexit teardown race (EXIT 127,
  no junit xml) — even `coordinates_panel`'s 18-test file passes alone but an
  8-test file may not; it's nondeterministic/ordering-sensitive, NOT a real
  failure (no assertion ever fails). PROOF my tests are correct: running
  `test_coordinates_panel + test_bezier_gui` => 26/0/0 EXIT 0, and a 6-file
  suite-ordered batch => 64/0/0 EXIT 0. In the full suite `test_app.py` precedes
  my files alphabetically so they're never the leading Qt module. Rule for
  future GUI tests: never rely on running a GUI test file alone; verify in
  company / via the full suite.
- Broad regression batch (touched modules + new tests) running in background
  (id bwgitajt9) — verifying inspector/commands/main_window edits didn't
  regress rotation/edit/bipartite/predictor.

### Iteration 6 (2026-05-22) — Task 5 tessellation generator (STARTING)
- **Plan:** new `auxetic/tessellation.py`. Given n boundary points defining a
  region (polygon) + a target tile density (edge length or count), fill the
  interior with a near-equilateral triangular lattice and close the boundary.
  - Approach: build a triangular (equilateral) point grid covering the region's
    bbox at spacing `h`; keep interior points (inside polygon, with margin);
    add the boundary polygon vertices (optionally resampled at ~h). Delaunay
    the union, then clip triangles whose centroid is outside the polygon.
    Interior triangles from the triangular grid are ~equilateral; boundary
    triangles (grid-to-boundary) are isosceles/scalene closers.
  - Public API (typed, dataclass result): e.g.
    `generate_tessellation(boundary_pts, density, *, margin=...) ->
    TessellationResult(points, simplices)`. Density param = target edge length
    OR triangle count; pick edge-length (`target_edge`) as primary, derive from
    count if needed.
  - Integrate with Lattice: a `from_tessellation(...)` classmethod or a mode/
    points injection so it can drive kirigami/bipartite + export. Cleanest:
    build points + (optionally) set them via `regenerate_from_points` on a 2D
    mode, OR add a thin classmethod. Keep geometry in auxetic/.
  - Tests: coverage (every region sub-area covered; triangulation spans the
    polygon), interior near-equilaterality (interior triangles' angles within
    tolerance of 60°), boundary closure (no gaps; union of triangles ≈ polygon
    area within tolerance).
- **Next step (was):** write tessellation + Lattice entry point. (DONE.)

### Iteration 6 (2026-05-22) — Task 5 tessellation (COMPLETE, 876d5e8)
- `auxetic/tessellation.py`: `generate_tessellation(boundary, target_edge |
  n_triangles)` -> `TessellationResult(points, simplices, boundary, n_boundary)`
  with `interior_triangle_mask()`. Equilateral grid + margin-clip interior +
  resampled boundary + Delaunay + centroid-in-polygon clip (carves concavities).
  Helpers: polygon_area, points_in_polygon, distance_to_polygon, resample_polygon,
  equilateral_grid, triangle_angles, equilateral_deviation, edge_from_triangle_count.
- `Lattice.from_tessellation(boundary, target_edge|n_triangles, mode=1, ...)`:
  uniform-normalizes to unit square (`_normalize_to_unit_square`), installs the
  clipped triangulation via `_geom._FlippedTri` (correct for concave; re-Delaunays
  on edit/reset — documented). Verified it drives kirigami (67 tiles), bipartite
  mode 11 (120 polys), STL/OBJ/SCAD. 25 tests. Interior tiles exactly equilateral
  (dev 0.000), concave coverage exact (L area 3.0 not hull 3.5).

### Iteration 7 (2026-05-22) — Task 4 generalized Poisson (STARTING)
- **Existing baseline:** `simulation.py::Simulator.poissons_ratio` (SPEC §7.4)
  uses bbox lateral/axial strain. Task 4 is the EDGE-VECTOR alternative — keep
  both; do NOT modify the bbox one.
- **Math (worked out):** For one triangle (corners P_0..2, centroid M, hinges
  `T_c = P_c + t(M-P_c)`, `t=1/(1+C)`), actuation rotates each corner kite rigidly
  about its hinge: `Q_c(θ) = R(θ)(P_c - T_c) + T_c`. The CORNER triangle
  Q_0,Q_1,Q_2 deforms with θ. Build the (exact, 3-point) affine deformation
  gradient `A` mapping rest edges [P1-P0 | P2-P0] -> deformed [Q1-Q0 | Q2-Q0]:
  `A = E_def @ inv(E_rest)`. Small-strain `ε = (A+Aᵀ)/2 - I`. Generalized
  Poisson: principal (eigen) strains ε1,ε2 (|ε1|>=|ε2|) -> `ν = -ε2/ε1`; OR a
  directional version `ν(axis) = -ε_perp/ε_axial`. Equilateral+symmetric C is
  isotropic => ε1==ε2 => ν=-1 (known test case). θ=0 => ε=0 => return nan/0
  (guard 0/0).
- **Plan:**
  - New `auxetic/edge_poisson.py` (numpy only): `triangle_strain_tensor(tri, C,
    theta)`, `generalized_poisson_ratio(tri, C, theta, axis=None)`,
    `sweep_poisson(triangles, C_values, theta)` -> (nT, nC) array, plus shape
    helpers `apex_triangle(apex_x, apex_y)` and `morph_triangle(s)`
    (equilateral s=0 -> scalene s=1). Reuse hinge math from bipartite (t=1/(1+C));
    keep it self-contained / import the t formula.
  - Tests `tests/test_edge_poisson.py`: equilateral -> ν≈-1 across C and θ;
    θ=0 -> nan/guard; strain tensor symmetric; isotropic dilation -> ν=-1;
    a uniaxial/known case -> expected sign; sweep returns right shape and
    equilateral column ≈ -1; degenerate triangle guarded.
  - Predictor panel: surface ν as a read-only computed metric IF low-risk
    (read predictor_panel.py first; GUI tests are fragile — prefer a pure
    metric method on Lattice/sim that the panel can call, tested without GUI,
    and only a tiny display hook). If risky, compute+expose via a Lattice
    method and note panel wiring as optional.
### Iteration 7 (2026-05-22) — Task 4 generalized Poisson (COMPLETE, 265050d)
- **Key result (documented in the module):** a single triangle's actuated kite
  *corners* deform by a shape-INDEPENDENT gradient `B = t·R(θ)+(1-t)I`, so their
  strain `ε = t(cosθ-1)I` is isotropic and a corner-based ν is identically -1 for
  every shape/C. The shape/C dependence the task asks for therefore had to come
  from a non-affine edge measure: the per-edge **bond midpoints** (each carried by
  a kite rotating about a different hinge). The triangle of those midpoints
  deforms non-affinely → its principal-strain ratio IS shape/C-dependent.
- `auxetic/edge_poisson.py` (numpy only): `generalized_poisson_ratio(tri, C, theta,
  axis=None)` (principal or directional), `triangle_strain_tensor`,
  `edge_midpoint_triangle`, `actuated_corners` (kept to demonstrate the isotropy),
  `sweep_poisson` / `sweep_shape_and_C` (+ `PoissonSweep`), shape helpers
  (`equilateral_triangle`, `apex_triangle`, `morph_triangle`). Equilateral → -1
  across all C/θ; scalene drifts toward 0/positive; θ=0 → nan (guarded).
- `Lattice.edge_vector_poisson_ratio(theta=0.1)` = mean over the lattice's 2D
  triangles (nan for 3D). Surfaced read-only in the Predictor panel's new
  "Geometry metrics" box; refreshed live via `MainWindow._refresh_state ->
  predictor_panel.refresh_metrics()`.
- 31 pure tests + 2 GUI tests (panel label ties to the Lattice method). Regression
  green. Committed task 4 (265050d); also committed the baseline's intended
  `.gitignore` `.claude/` line that had been left unstaged (a8f0513).
- **Next:** await full-suite result (background brdhfxqi0); if green, finalize the
  summary below and STOP. If any failure, fix (my code is wrong) and re-verify.

---

## Final summary
_(written at the end of the run)_
