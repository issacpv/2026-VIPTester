# Auxetic Lattice Studio — Specification (DRAFT)

> **Status:** Draft. Edit freely. Items marked `TODO(you)` need product decisions
> from Zack. Items marked `TODO(physics)` need a mathematical definition before
> any code in that area can be written. Items marked `TODO(later)` are
> deliberately deferred to a later milestone.

---

## 1. Purpose

A desktop application for designing, editing, simulating, and exporting
auxetic kirigami lattices. The application replaces the current
configure-and-run script (`data/grid/displayAuxeticV20.py`) with an
interactive GUI while preserving its geometry pipeline and export formats.

The application supports two design domains:

- **2D lattices** — top-down planar kirigami structures, optionally
  extruded into z-layers (current modes 1, 2, 4, 5).
- **3D lattices** — fully three-dimensional tetrahedral lattices with
  truncated-cuboctahedron hubs (current modes 3, 6).

Both domains share the same data model, edit flow, simulation engine,
and export pipeline.

---

## 2. Data Model

### 2.1 The `Lattice` object

The `Lattice` is the root document type. One open document = one `Lattice`.

```
Lattice:
    mode: int               # 1, 2, 3, 4, 5, 6 (matches existing modes)
    n_points: int           # target point count (used only on regenerate)
    ratio: float            # shrink ratio toward triangle/tet centroid (0..1)
    nz_layers: int          # used only in modes 2, 5
    points: np.ndarray      # Nx2 (2D modes) or Nx3 (3D modes), in [0,1]
    triangulation: object   # Delaunay or MockTri from existing script

    # Shape parameters (currently module-level constants)
    ngon_thickness: float
    hub_size_factor: float
    joint_sphere_radius: float
    joint_sphere_rings: int
    joint_sphere_segments: int
    strut_radius: float
    face_thickness: float

    # View / orientation state (NEW — see §6)
    rigid_rotation: Rotation     # rotation of whole lattice in world space
    flipped: bool                # mirror about X axis when viewed top-down

    # Simulation state (NEW — see §7)
    joint_angle: float           # internal kirigami DOF, radians
```

The `points` array is the single source of truth. The triangulation is
recomputed from `points` whenever points change. All derived geometry
(struts, hubs, tiles, exports) is recomputed on demand.

### 2.2 Invariants

- `points.shape[1] == 2` iff `mode in {1, 2, 4, 5}`
- `points.shape[1] == 3` iff `mode in {3, 6}`
- `0 < ratio < 1`
- `nz_layers >= 2` when `mode in {2, 5}`, else ignored
- All point coordinates in `[0, 1]` after normalization

### 2.3 Coordinate conventions (preserved from existing script)

- Lattice space: `[0, 1]` per axis
- 2D points are projected to `z = 0` for rendering and extruded for modes 2/5
- A *hub* is a lattice point in `points`
- A *tile* is a polygonal face collected around a hub or a shrunken simplex face
- Central hubs (all 8 octants populated, `is_central_hub` test) render as
  truncated cuboctahedra; non-central hubs render as extruded convex polygons

---

## 3. View Modes

The application has two viewports, switched via a toolbar toggle.

### 3.1 2D Top-Down View

- Background: a square grid (configurable spacing, snap optional)
- Foreground: the lattice projected to the XY plane
  - In modes 1, 4: rendered as-is
  - In modes 2, 5: rendered as the projection of the *bottom* z-layer,
    with upper layers shown as faded outlines `TODO(you): confirm`
  - In modes 3, 6: rendered as the projection of all points, with
    z-depth indicated by `TODO(you): color ramp? size? toggle off?`
- Origin marker at `(0, 0)` and axis indicators at `+X`, `+Y`
- Pan: middle-mouse-drag or space+drag
- Zoom: scroll wheel
- The 2D view is the **only** view in which point editing is allowed

### 3.2 3D View

- Standard orbit camera (left-drag rotate, right-drag pan, scroll zoom)
- Renders the full STL geometry (hubs + struts + joint spheres)
- Shows the lattice's bounding box and the current load axis as an arrow
- Shading: matte, single directional light + ambient
- Edit mode is **disabled** in 3D view `TODO(you): confirm or allow 3D drag?`

### 3.3 What's shared between views

Both views reflect the same `Lattice` object. Changing parameters in the
inspector panel updates both. Switching views does not modify state.

---

## 4. Edit Operations

### 4.1 Edit Mode

Edit mode is a toggle. While ON:

- Each lattice point is rendered as a draggable handle in the 2D view
- Hovering a point highlights it; clicking selects it
- Drag-to-move applies live; on mouse-release, the lattice regenerates
  (re-triangulates and rebuilds all derived geometry)
- Hold Shift while dragging to snap to grid
- Multi-select via marquee or Shift+click `TODO(you): needed for v1?`
- Delete key removes selected points (must leave ≥ 3 points)
- Right-click on empty grid: "Add point here" `TODO(you): needed for v1?`

### 4.2 Undo / Redo

- All edits go through a `QUndoStack`
- Granularity: one undo step per drag-release (not per-pixel)
- Parameter changes (ratio, n_points, mode) are also undoable
- `Ctrl+Z` / `Ctrl+Shift+Z` bindings

### 4.3 Regenerate

The "Regenerate" button re-rolls random points based on the current `mode`
and `n_points`. This **discards** any manual edits and replaces `points`
with a fresh random sample. A confirmation dialog appears if there are
unsaved manual edits. `TODO(you): confirm flow`

### 4.4 Reset to original

A "Reset" command returns `points` to the values they had immediately after
the last regenerate or load. Implemented by caching `points_original`
alongside `points`.

---

## 5. Preset / Save System

### 5.1 File format

Presets are JSON files with extension `.auxlat`. Schema:

```json
{
  "version": 1,
  "mode": 1,
  "n_points": 5,
  "ratio": 0.35,
  "nz_layers": 2,
  "points": [[x, y], ...],
  "shape_params": {
    "ngon_thickness": 0.03,
    "hub_size_factor": 0.75,
    "joint_sphere_radius": 0.015,
    "strut_radius": 0.02
  },
  "view_state": {
    "rigid_rotation_quat": [w, x, y, z],
    "flipped": false,
    "joint_angle_deg": 0.0
  },
  "metadata": {
    "name": "my lattice",
    "created": "2026-04-29T00:00:00Z",
    "modified": "2026-04-29T00:00:00Z",
    "notes": ""
  }
}
```

### 5.2 Save / Load behavior

- `Ctrl+S` saves to the current file path; `Ctrl+Shift+S` is Save As
- `Ctrl+O` opens a file dialog
- Loading a preset **replaces** the entire `Lattice` state
- The application maintains a recent-presets list of the last 10 files
- Default preset directory: `~/.auxetic_studio/presets/`
- Presets opened from outside this directory still work; they are
  added to the recent list

### 5.3 Versioning

The `version` field is mandatory. The loader rejects unknown versions
with a clear error. When the schema changes, bump the version and
write a migration function from the previous version.

### 5.4 What is *not* saved in a preset

- Window layout / camera position `TODO(you): save these too?`
- Undo history
- Simulation cache results

---

## 6. Orientation & Rotation Model

> **Critical:** Two distinct rotation concepts. Do not collapse them
> into a single value. They mean different things and are used at
> different stages of the pipeline.

### 6.1 Rigid lattice rotation

A rotation applied to the entire lattice as a rigid body, in world space.
This is what you mean by "flip the structure upside down" or "rotate the
whole thing 30° before compressing."

- Stored as a `scipy.spatial.transform.Rotation` (quaternion-backed)
- For 2D modes, only rotation about the Z axis is meaningful (single scalar)
- The "Flip" button is a special case: `Rotation.from_euler('x', 180, degrees=True)`
- This rotation does **not** modify `points`. It is applied at render time
  and passed to the simulator as the orientation of the load frame
- UI: a rotation gizmo in the 3D view, plus numeric entry in inspector

### 6.2 Joint rotation (kirigami DOF)

The internal angle θ that parameterizes the kirigami mechanism's
single soft mode — the one that takes the structure between its
two compressed states through the rest pose.

- Stored as a single scalar `joint_angle` in radians
- Affects the *positions of tiles* via the kinematic constraint solver
  (see §7.3), not the rigid orientation of the lattice

The joint angle θ has two distinct parameterizations in the application:

- **Simulator (mathematical):** θ ∈ [-π/2, π/2] radians, with rest at θ=0
  and the two compressed states at the endpoints. This is the natural
  parameterization for the kinematic mode (`pose = θ * mode`), and it places
  rest where Taylor expansions and Jacobian linearizations want it.

- **GUI / spec language (physical):** θ ∈ [0°, 180°] degrees, with rest at
  90° and compressed states at 0° and 180°. This matches the bistable
  cycle described in user-facing language: "compressed-A → unstable
  equilibrium at 45° → expanded rest at 90° → unstable at 135° →
  compressed-B at 180°."

The mapping between them is `θ_physical_deg = degrees(θ_simulator) + 90`.
The GUI's joint angle slider operates in physical degrees; the conversion
happens at the boundary in `auxetic_studio/simulation_panel.py` (Stage 6c).
The simulator and `SimResult` never see physical degrees.

### 6.3 Why both must exist independently

A structure compresses well *only when its soft kinematic mode is
aligned with the load axis*. The rigid rotation chooses where the load
hits; the joint rotation describes the resulting motion. The
optimization in §8 searches over rigid rotation; the simulation in §7
sweeps joint rotation.

---

## 7. Simulation Model

> This is the section with the most open physics questions. Resolve
> these before writing simulation code.

### 7.1 Goal

Given a lattice and a load direction, predict:

1. Whether the structure compresses meaningfully or locks up
2. How much it compresses (compression depth as a fraction of original height)
3. The path it takes through joint-angle space
4. Whether it gets stuck at an unstable equilibrium

### 7.2 Bodies and constraints

- Each *tile* (as collected by `collect_kirigami_tiles`) is a rigid body
  with 6 DOF in 3D, 3 DOF in 2D
- Each *constraint* (as built by `build_kirigami_constraints`) pins
  vertex `v_a` of tile A to vertex `v_b` of tile B (a 3-DOF or 2-DOF
  point coincidence, depending on dimension)
- The constraint set defines a kinematic manifold of allowed
  configurations, parameterized by the remaining DOFs

### 7.3 Quasi-static kinematic solver

Because the structure is (by design) a mechanism with a small number of
soft modes, a quasi-static solver is sufficient — no inertia, no
dynamics, no time integration of mass.

Algorithm sketch:

1. Start from the rest configuration (all tiles at their as-built poses)
2. Identify the kinematic DOFs by computing the null space of the
   constraint Jacobian at the rest pose
3. Apply a small perturbation along one null-space direction (the
   "kirigami mode")
4. Project back onto the constraint manifold via constrained
   least squares (Gauss-Newton on the constraint residual)
5. Record the resulting bounding-box height under the load direction
6. Repeat for θ from 0° to 180° in 1° increments

`TODO(physics): How is the kirigami mode identified when there are
multiple null-space directions? For a single-mechanism unit cell the
null space is 1D and unambiguous. For a periodic lattice with multiple
soft modes, you need a selection rule — e.g., the mode whose
displacement field has the largest Y-component (axial compliance).`

### 7.4 Defining "compresses well"

`TODO(physics): Pin this down. Candidate definitions:`

**Option A (kinematic):** The fraction by which the bounding-box
height in the load direction decreases as θ sweeps from its
expanded-state value (e.g. 90°) to its compressed-state value (e.g. 0°
or 180°). Higher = better. This ignores forces entirely.

**Option B (compliance):** The minimum eigenvalue of the stiffness
matrix projected onto the load axis, evaluated at the rest pose.
Smaller = softer = compresses well under the same load. Requires a
constitutive model for the joints (linear torsional spring at each ball
joint, with stiffness `k_joint`).

**Option C (energy barrier):** The height of the energy barrier between
the rest pose and the compressed state, as θ sweeps. Low barrier =
compresses well. Same constitutive model as B.

Recommendation: start with Option A — it's parameter-free and matches
the intuition in the user narrative ("nowhere else to distribute the
force"). Add Options B and C later if needed.

### 7.5 Locking detection

A lattice "locks" under a given load direction when the projection of
the kirigami mode onto the load axis is approximately zero — the soft
mode exists but does no useful work against the load.

**Mode-projection criterion:** the kirigami mode's bounding-box change
direction (a unit vector indicating which world-space direction the
structure as a whole moves along when the mode is perturbed) projected
onto the load axis. If `|bbox_change_direction · load_axis| < 0.05`, the
mode does not produce axial motion under the applied load and the
structure is reported as locked.

Note: an earlier draft of this section defined "mode direction" as the
mean per-vertex displacement. That definition is incorrect for symmetric
auxetic mechanisms (rotating squares is the canonical example: vertices
move in equal-and-opposite pairs and the mean cancels exactly), making
every well-formed auxetic appear locked. The bbox-change definition
captures the physically meaningful quantity in all cases.

### 7.6 Simulation output

The simulator returns a `SimResult` object:

```
SimResult:
    theta_samples: np.ndarray      # joint angles tested, radians
    heights: np.ndarray            # bounding-box height at each theta
    compression_ratio: float       # max(heights) - min(heights) / max(heights)
    locked: bool                   # True if compression_ratio < threshold
    energy_barrier: float | None   # only populated for Option B/C
    trajectory: list[np.ndarray]   # tile poses at each theta sample
```

### 7.7 Visualization

- A "Play" button animates the lattice through `theta_samples`
- A 2D plot shows `heights vs theta` with markers for the current θ
- Locked configurations are shown with a red badge in the inspector

---

## 8. Optimization

### 8.1 Problem statement

Find the rigid rotation `R*` (and optionally the initial joint angle `θ₀*`)
that maximizes the compression metric defined in §7.4, subject to the
load being applied along the world `-Y` axis.

### 8.2 Search space

- 2D modes: `R` is a single scalar in `[0°, 360°)`; `θ₀` in `[0°, 180°)`
- 3D modes: `R` parameterized as Euler angles or as a sample from a
  spherical Fibonacci lattice over orientations; `θ₀` in `[0°, 180°)`

`TODO(you): For 3D, how dense should the orientation sampling be?
24 cube symmetries? 60? 600? Brute-force runtime grows linearly.`

### 8.3 Algorithm (v1: brute force)

1. For each candidate rigid rotation `R_i`:
   a. Apply `R_i` to the lattice
   b. Run the simulator (§7.3) over θ ∈ [0°, 180°] in 1° increments
   c. Record the compression metric
2. Return the `R_i` that maximizes the metric
3. Store the top 5 candidates so the user can flip between them

### 8.4 Algorithm (v2: gradient descent) — `TODO(later)`

Replace the outer brute-force loop with gradient descent on `R` using
finite differences over a coarse grid as the initialization. Defer
until v1 is working and the metric is validated.

### 8.5 UI

- "Optimize Orientation" button in the simulation panel
- Progress bar during the search
- On completion: lattice snaps to the best `R*`, plot updates, top-5
  alternatives appear as clickable thumbnails

---

## 9. Export

The application preserves all existing export formats from
`displayAuxeticV20.py`:

- STL (`.stl`) — for 3D printing
- OBJ (`.obj`) — for general 3D tools
- SCAD (`.scad`) — for OpenSCAD round-tripping
- Kirigami vertices (`vertices.txt`)
- Kirigami constraints (`constraints.txt`)

Export is reachable from `File → Export…` with format checkboxes.
The export pipeline reuses the existing functions verbatim — see
the regression test requirement in §11.

`TODO(you): Should exports apply the rigid rotation? Probably yes for
STL/OBJ (so the printed object matches the optimized orientation) and
no for kirigami files (which describe the mechanism in its canonical
frame). Confirm.`

---

## 10. UI Layout

```
┌────────────────────────────────────────────────────────────────┐
│ File   Edit   View   Simulate   Help                           │  menu bar
├──┬──────────────────────────────────────────────────┬──────────┤
│T │                                                  │ Inspector│
│o │                                                  │ ─────────│
│o │              Central viewport                    │ Mode: ▼  │
│l │              (2D or 3D, toggled)                 │ n_points │
│b │                                                  │ ratio    │
│a │                                                  │ ...      │
│r │                                                  │          │
│  │                                                  │ Sim panel│
│  │                                                  │ θ slider │
│  │                                                  │ Optimize │
├──┴──────────────────────────────────────────────────┴──────────┤
│  Status bar: mode | tile count | last action | unsaved marker │
└────────────────────────────────────────────────────────────────┘
```

Toolbar buttons (top-to-bottom):
- View toggle (2D / 3D)
- Edit mode toggle
- Regenerate (re-roll points)
- Reset to original
- Save preset
- Load preset
- Flip
- Rotation gizmo

---

## 11. Engineering Constraints

### 11.1 Refactor before GUI

Before any GUI code is written, `displayAuxeticV20.py` must be refactored
into a package `auxetic/` with submodules:

- `auxetic/geometry.py` — point generation, triangulation, hub detection
- `auxetic/tiles.py` — kirigami tile collection, constraint building
- `auxetic/export.py` — all writers (STL/OBJ/SCAD/kirigami)
- `auxetic/lattice.py` — the `Lattice` class

A regression test must lock in current behavior:

- Run with `mode=1, n_points=5, ratio=0.35`
- Run with `mode=6, n_points=8, ratio=0.35`
- Hash the resulting `.stl`, `vertices.txt`, `constraints.txt`
- The refactor must reproduce these hashes exactly

### 11.2 Stack

- Python 3.11+
- PySide6 for GUI
- PyVistaQt for 3D viewport
- pyqtgraph for 2D viewport
- numpy / scipy for numerics
- numpy-stl for STL I/O (already a dependency)
- No torch, no jax, no tensorflow

### 11.3 Performance targets

- Edit-drag → regenerate → render: < 100 ms for `n_points ≤ 30`
- 2D simulation sweep (180 θ samples): < 500 ms
- 3D simulation sweep: < 5 s
- Optimization in 2D: < 30 s for 360 rotation samples × 180 θ samples
- Optimization in 3D: `TODO(you): acceptable budget? minutes? hours?`

### 11.4 Platform

- Primary: Linux + macOS
- Windows: `TODO(you): required?`

---

## 12. Out of Scope (v1)

- Multi-cell tiling / periodic lattices (only unit cells in v1)
- Material properties beyond uniform tile rigidity
- FEA / continuum mechanics
- GPU rendering
- Cloud sync of presets
- Collaborative editing
- Plugin system
- Undo for orientation/rotation gizmo manipulation `TODO(you): or include?`

---

## 13. Open Questions Index

Every `TODO(you)` and `TODO(physics)` marker, collected here for tracking:

**Product decisions (`TODO(you)`):**
- §3.1: How are upper z-layers shown in 2D top-down view for modes 2/5?
- §3.1: How is z-depth indicated in 2D top-down view for 3D modes?
- §3.2: Is point editing allowed in 3D view, or 2D-only?
- §4.1: Multi-select / add-point-on-empty needed for v1?
- §4.3: Confirm the regenerate-discards-edits flow
- §5.4: Save window layout and camera position in presets?
- §7.7: `TODO(you): UI for inspecting which tile pairs are constrained?`
- §8.2: Density of 3D orientation sampling
- §9: Should exports apply the rigid rotation?
- §11.3: Acceptable optimization runtime in 3D
- §11.4: Windows support required?
- §12: Undo for rotation gizmo manipulation?

**Physics decisions (`TODO(physics)`):**
- §7.3: How is the kirigami mode selected from a multi-D null space?
- §7.4: Which definition of "compresses well"? (Options A / B / C)
- §7.5: Is dot product of mode direction and load axis the right
  locking metric?

---

## 14. Glossary

- **Auxetic** — material/structure with negative Poisson's ratio:
  contracts laterally when compressed axially (or vice versa)
- **Kirigami** — the design paradigm where a structure deforms via
  rotation of rigid panels around point-like hinges
- **Hub** — a lattice point where multiple tiles meet
- **Tile** — a rigid polygonal/polyhedral body in the kirigami mesh
- **Strut** — a thin cylindrical connector between two tile vertices
- **Joint angle (θ)** — internal kinematic DOF of the mechanism
- **Rigid rotation (R)** — orientation of the whole lattice in world space
- **Locking** — configuration in which the soft mode does not project
  onto the load axis, so the structure cannot deform meaningfully
- **Compression metric** — scalar quantifying how well a structure
  collapses under load (definition `TODO(physics)`, see §7.4)
