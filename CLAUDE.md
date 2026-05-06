# Auxetic Lattice Studio — Project Conventions

## Source of truth
The geometry pipeline lives in the `auxetic/` package:
- `auxetic/geometry.py` — point generation, triangulation, hub detection, mesh assembly
- `auxetic/tiles.py` — kirigami tile collection and constraint building
- `auxetic/export.py` — all writers
- `auxetic/lattice.py` — the `Lattice` class
- `auxetic/simulation.py` — quasi-static kinematic solver (SPEC §7;
  null-space mode identification, projection, Poisson's ratio, locking
  criterion). Use this for *what equilibrium kirigami mode looks like*.
- `auxetic/dynamics.py` (M2) — Newtonian rigid-body simulator with
  user-defined forces, ground-plane contact, and pre-rotation. Built
  on the same tile/constraint topology as `simulation.py` but adds
  mass, time integration, and contact. Use this for *how the lattice
  responds to a load over time*. The two modules are sibling concerns
  and share constraint primitives — `simulation.py` is **not** a
  superset of `dynamics.py` and vice versa.

The original script `data/grid/displayAuxeticV20.py` is preserved as a
reference for regression testing only. Do not modify it. Do not import
from it in new code. If you need to understand a convention, read the
package; if the package and the script disagree, the script is correct
and the package is the bug.

Always run `pytest tests/test_regression.py` before committing changes
to anything in `auxetic/`.

This script is the source of truth for:
- Point generation and triangulation (modes 1–6)
- Hub detection (`is_central_hub`) and truncated cuboctahedron sizing
- Tile ordering, convex ring ordering, Newell normals
- Kirigami tile collection and constraint building
- STL/OBJ/SCAD export

A refactor into an `auxetic/` package is planned but has not happened yet.
Until it does, treat `displayAuxeticV20.py` as the canonical reference.

## Before touching geometry code
Always read `data/grid/displayAuxeticV20.py` end-to-end before writing
or modifying anything related to the systems listed above. Do not
reimplement these conventions from scratch. If existing logic seems
wrong, flag it and propose a change — do not silently rewrite with
different conventions.

When the refactor happens, update this file to point at the new module
paths instead.

## Coordinate conventions (from displayAuxeticV20.py)
- Points are stored in lattice space, range [0, 1] per axis
- 2D modes (1, 4, 7) store points as Nx2; 3D modes (3, 6, 9) store as Nx3
- Modes 2, 5, and 8 are 2D points extruded into `nz_layers` z-layers at render time
- A "hub" is a lattice point; "tiles" are the polygonal faces collected around it
- Central hubs (the `is_central_hub` test, requiring all 8 octants populated)
  get a truncated cuboctahedron solid; other hubs get extruded polygons
- Module-level constants at the top of the script (`mode`, `n_points`, `ratio`,
  `nz_layers`, etc.) are the current configuration interface — preserve these
  names if you wrap the script in a class

### Mode taxonomy (M1)
The single `mode` integer encodes a (dimensionality × strategy) pair.
The `auxetic_studio` UI exposes this as two dropdowns (Dimensionality
and Strategy); the package keeps the integer for back-compat with the
original V20 script and the regression test harness.

| dim \ strategy | random | grid | mesh-import |
|----------------|--------|------|-------------|
| 2D             | 1      | 4    | 7           |
| 2.5D           | 2      | 5    | 8           |
| 3D             | 3      | 6    | 9           |

Modes 7 / 8 / 9 (M1 additions) take their points from an STL or OBJ
file's vertices via `Lattice.from_mesh`. They run the same Delaunay /
extrusion / 3D-Delaunay pipeline as 1 / 2 / 3 — only the source of the
points differs. `mesh_path` and `mesh_vertices` (normalised to the
unit cube) are stored on the lattice and round-tripped through preset
v3.

### Physical scale (M1)
`Lattice.unit_scale_cm` (default 1.0) maps lattice units to centimetres.
The geometry pipeline is unit-agnostic — only the M2 dynamic simulator
will read this when computing default masses and force ranges.

## Two distinct rotation concepts — do not conflate
1. **Rigid lattice rotation** — rotates the entire structure in world
   space relative to gravity/load direction. Used for orientation
   optimization. Includes the `flipped` field (a special case = 180°
   about X), which the UI exposes as the **Mirror** button. Do *not*
   call this "flip" in new code — that name now means edge flips
   (below).
2. **Joint rotation** — internal degree of freedom of the kirigami
   mechanism (the 0°/45°/90°/135° bistable sweep). Used for the
   compression simulation.
3. **Edge flip** (M1) — choosing the *other* diagonal in a Delaunay
   triangulation quad. This is a triangulation choice, not a rotation,
   but lives in the user's mental model of "rearranging the lattice"
   alongside the two rotations above. Stored as a set of `(i, j)`
   tuples in `lattice.edge_flips`; applied via
   `geometry.apply_edge_flips` after every (re)triangulation. 2D-only
   in M1; 3D tetrahedral 2-3 / 3-2 flips are deferred.

These must live in separate fields and must not be combined into a
single "rotation" parameter.

## Regression safety
`displayAuxeticV20.py` produces working STL and kirigami output today.
Before any refactor lands, write a regression test that runs the script
with `mode=1, n_points=5, ratio=0.35` and `mode=6, n_points=8, ratio=0.35`
and hashes the resulting `.stl`, `vertices.txt`, and `constraints.txt`.
The refactored code must reproduce those hashes exactly.

## Regression test policy
`tests/test_regression.py` covers modes 1, 2, 4, 5, 6 against the
preserved original script. Mode 3 is intentionally skipped due to
Qhull's cross-version tetrahedron ordering instability — do not
add it without first solving the determinism problem (likely via
invariant comparison rather than byte equality). Modes 7, 8, 9 are
not in the regression set — the V20 reference script doesn't have a
mesh-import path to compare against; correctness for those modes is
covered by `tests/test_mesh_import.py` integration tests instead.

The M1 generation extensions (density gradient, edge flips,
mesh import) all default to behavior-preserving values, so
`generate_points(n_points, mode)` with no kwargs reproduces the V20
output byte-for-byte. Any change that modifies the legacy code path
must keep `tests/test_regression.py` passing without exception.

If a regression test fails, the package is wrong, not the test.
Do not update goldens to make tests pass without explicit approval.

## Stack (planned)
- PySide6 for GUI, PyVistaQt for 3D view, pyqtgraph for 2D editor
- `auxetic/` package: numpy / scipy only for numerics — no torch,
  no jax. M2 added Newtonian dynamics in pure numpy/scipy
  (semi-implicit Euler with Baumgarte-stabilized soft constraints) —
  do not pull in PyBullet, MuJoCo, or any other native physics engine.
- The forthcoming `auxetic_ml/` package (M3) is allowed to import
  PyTorch; the geometry / dynamics packages above are not.
- Python 3.11+

## Style
- Type hints on all new public functions
- No bare `except:` — catch specific exceptions
- Prefer dataclasses over dicts for structured data
- Match existing naming in `displayAuxeticV20.py` when extending it

## GUI architecture
The application shell lives in `auxetic_studio/`. It is a thin layer
over the `auxetic/` package — it owns no geometry logic. Every numerical
or geometric operation must go through a `Lattice` method.

If you find yourself writing geometry code in `auxetic_studio/`, stop:
it belongs in `auxetic/` and should be exposed via the `Lattice` API.
The GUI is allowed to know about Qt, PyVista, and pyqtgraph; the
package is not. This separation must hold.

Test policy: `tests/test_app.py` covers the GUI shell. The test that
exports an STL through the GUI and diffs it against the direct
`Lattice.to_stl()` call is load-bearing — it proves the GUI hasn't
silently forked the geometry pipeline. Do not delete or weaken this
test.

## Actual stack
- PyQt6 (not PySide6 as originally specified — see SPEC.md §11.2)
- PyVistaQt for 3D viewport
- pyqtgraph for 2D viewport
- numpy / scipy / numpy-stl for numerics

## Preset format
Presets are versioned. Current version: see
`auxetic_studio/preset.py` (M1: bumped to v3 to add the `generation`
block — density gradient, edge flips, mesh import, `unit_scale_cm`).
When changing the preset schema, bump the version and add a migration
function from the previous version. Never break old presets silently.

## Conventions worth knowing

The joint angle θ has two parameterizations — see SPEC §6.2. The
simulator uses θ ∈ [-π/2, π/2] with rest at 0; the GUI and spec
language use θ ∈ [0°, 180°] with rest at 90°. The conversion happens
only at the GUI boundary. Don't introduce a third convention.

The locking criterion's "mode direction" is the bbox-change direction
of the structure under mode perturbation, not the mean per-vertex
displacement. The mean-displacement formulation is a documented dead
end (see SPEC §7.5); don't reintroduce it.