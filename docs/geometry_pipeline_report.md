# Geometry pipeline — descriptive report (Stage 6c.5 prep)

Purpose: document what the four functions on the geometry path
**actually do** today. No code changes. No fix proposals.

Empirical examples come from `Lattice(mode=1, n_points=4,
ratio=0.35, seed=1)` — small enough to enumerate, big enough to
exercise both tri-face and hub-face tile types and the strut branch.

```
mode=1, n_points=4, ratio=0.35, seed=1
points.shape         = (4, 2)
tri.simplices.shape  = (3, 3)         # 3 Delaunay triangles
```

---

## 1. `auxetic.tiles.collect_kirigami_tiles`

```python
def collect_kirigami_tiles(points_nd, tri, ratio, mode, nz_layers):
    return tiles, tile_source
    # tiles:       list[np.ndarray of shape (N_i, 3)]
    # tile_source: list[dict]   (parallel to tiles)
```

**What it does** (from the body): enumerates Delaunay simplices and
emits one **shrunken-toward-centroid** tile per simplex (3D modes
get a 4-vertex tetrahedron; 2D modes get a 3-vertex triangle).
After the simplex loop, walks per-lattice-point groups (every
shrunken corner that landed at this lattice point) and emits one
**hub** tile per lattice point that has ≥ 3 incident shrunken
corners — convex-ordered ring (`hub_face`) or truncated
cuboctahedron (`hub_polyhedron`, when `is_central_hub` is true).

**I/O:**

- `points_nd`: `(N, 2)` for 2D modes, `(N, 3)` for 3D.
- `tri`: object with `.simplices` of shape `(M, 3)` (2D) or `(M, 4)`
  (3D).
- `tiles[i]`: `np.ndarray` of shape `(N_i, 3)`. **Always 3D**, even
  for 2D modes (z=0 for modes 1/4, layer-z for 2/5). Vertices are
  in **canonical lattice space** (no world transform applied).
- `tile_source[i]`: dict with `'type' ∈ {'tri_face', 'tetrahedron',
  'hub_face', 'hub_polyhedron'}` and per-type metadata
  (`'simplex_idx'`, `'hub_key'`, `'hub_center'`, `'z_idx'`,
  `'z_val'`).

**Tile order is load-bearing:** all simplex-derived tiles first
(in `tri.simplices` order), then hub tiles in iteration order over
the per-lattice-point groups. For 2D extruded modes (2, 5) this
sequence repeats per z-layer.

**Module state:** `TOL_KEY = 9` (decimal places for matching
shrunken corners back to lattice points). Helpers from
`auxetic.geometry`: `is_central_hub`, `hub_scale_for_tcoh`,
`make_truncated_cuboctahedron`, `convex_order_3d`,
`order_hub_ring_xy`, `build_3d_groups`. Pure (output depends only
on inputs).

**Empirical:**

```
collect_kirigami_tiles -> 4 tiles
  tile 0: shape=(3, 3) type=tri_face z=0.000
  tile 1: shape=(3, 3) type=tri_face z=0.000
  tile 2: shape=(3, 3) type=tri_face z=0.000
  tile 3: shape=(3, 3) type=hub_face z=0.000
```

3 tri_faces (one per Delaunay triangle) + 1 hub_face for the lattice
point shared by all 3 triangles. The other 3 lattice points are
boundary points with only 1 incident triangle → no hub_face.

Constraints from this tile set:
```
3 constraints
  (0, 0, 3, 0, 1)   tile 0 vert 0  ↔  tile 3 vert 0
  (1, 1, 3, 1, 1)   tile 1 vert 1  ↔  tile 3 vert 1
  (2, 0, 3, 2, 1)   tile 2 vert 0  ↔  tile 3 vert 2
```

Each pins one tri_face's corner to the corresponding hub_face
corner — the canonical-frame coincidence at the shared shrunken
position.

---

## 2. `auxetic.geometry.collect_export_geometry`

```python
def collect_export_geometry(points_nd, tri, ratio, mode, nz_layers):
    return strut_curves, all_triangles, joint_positions
    # strut_curves:    list[np.ndarray of shape (2, 3)]
    # all_triangles:   list[list[np.ndarray of shape (3,)]]   # 3 verts/tri
    # joint_positions: set[tuple[float, float, float]]
```

**What it does:** the same shrunken-corner construction as
`collect_kirigami_tiles`, but emits **mesh** geometry — extruded
solid triangles for tile faces, line segments for inter-tile
struts, and de-duplicated point positions for joint spheres.

The crucial structural similarity: it builds
`groups[lattice_point]` populated with the same shrunken corners
that the kirigami pipeline emits as tile vertices.

- `len(pts_list) == 2` → 2 corners join a **strut** (the boundary
  case where 2 tiles share a lattice point but no hub is built).
- `len(pts_list) >= 3` → hub geometry is built from the
  **shrunken corners themselves**, not from `points_nd`.
  - 2D path: convex-hull-then-extrude
    (`order_hub_ring_xy → extrude_polygon_solid`).
  - 3D non-central path: `convex_order_3d → extrude_polygon_solid`
    on the shrunken corners.
  - 3D central path: `make_truncated_cuboctahedron(hub_center,
    scale)` — `hub_center` reads the **lattice-point key
    directly**, only `scale` is derived from the shrunken corners.

`joint_positions` is rounded to 8 decimals via the inner
`register_joint`; iteration order over the resulting `set` is not
insertion-stable.

**Module state:** none directly. Calls helpers that use
`NGON_THICKNESS`, `HUB_SIZE_FACTOR` (extrusion thickness, hub
sizing). Pure.

**Empirical:**

```
3 struts, 32 solid tris, 9 joints
  first strut shape:    (2, 3), dtype=float64
  first triangle shape: 3 vertices, each (3,)
  first joint position: (0.15993436, 0.31097486, 0.0)
```

3 struts = the three boundary edges (each connecting 2 shrunken
corners with no hub). 32 solid triangles = extruded geometry for
3 tri_faces + 1 hub_face. 9 joint positions = 3 boundary corners +
3 interior corners + 3 hub-face vertices.

---

## 3. `auxetic.geometry.build_export_triangles`

```python
def build_export_triangles(strut_curves, all_triangles, joint_positions,
                            strut_radius=None, scad_segments=None,
                            joint_sphere_radius=None,
                            joint_sphere_rings=None, joint_sphere_segments=None,
                            verbose=True):
    return result   # list[list[np.ndarray of shape (3,)]]
```

**What it does:** consumes the **raw geometry collections** from
`collect_export_geometry` (NOT a Lattice) and adds two layers of
detail: each strut curve is extruded to a cylindrical tube of
triangles via the inner `tube_mesh`, and each joint position gets
a UV-sphere via `sphere_mesh`. The output subsumes the input
`all_triangles`.

**I/O:** inputs are the three outputs of `collect_export_geometry`
plus optional shape parameters defaulting to module constants
(`STRUT_RADIUS`, `SCAD_SEGMENTS`, `JOINT_SPHERE_RADIUS`,
`JOINT_SPHERE_RINGS`, `JOINT_SPHERE_SEGMENTS`). Output length =
`len(all_triangles) + 2 * len(strut_curves) * scad_segments` (tube
body+caps) `+ len(joint_positions) * sphere_tris`. Pure.

**Empirical:**

```
build_export_triangles -> 848 triangles total
  (was 32 solid tris; added 816 from struts + spheres)
```

**Call sites:** `Lattice.build_export_triangles` only — which is
called from `Lattice.to_stl` / `to_obj` / `to_scad`. No GUI code
calls either of these geometry functions.

---

## 4. `auxetic_studio.views.View3D._build_pose_mesh_triangles`

```python
@staticmethod
def _build_pose_mesh_triangles(tile_system, pose):
    return triangles   # list[np.ndarray of shape (3, 3)]
```

**What it does:** for each tile in `tile_system.tiles`, applies the
per-tile pose `[tx, ty, θ]` (2D) or `[tx, ty, tz, rx, ry, rz]`
(3D, axis-angle), and **fans the resulting polygon into triangles
from vertex 0** (an N-gon yields N-2 triangles). 2D output is
lifted to z=0.

**I/O:** `tile_system.tiles[i]` is `(N_i, 2)` or `(N_i, 3)` —
already in world frame per `TileSystem.from_lattice`. `pose` is
flat shape `(n_tiles * dofs,)`. Returns a list of `(3, 3)` arrays
in world frame.

**Helpers from `auxetic.geometry`: NONE.** Pose application and
fan triangulation are inline; only `numpy` and
`scipy.spatial.transform.Rotation` are imported. Pure.

**Empirical (mode=1, n=4, pose = zeros(12)):** 4 tiles each `(3, 3)`
→ 1 triangle per tile (vertex 0 fan of a triangle is itself) →
4 triangles total. Compare against Section 3's 848 triangles for
the full-detail render — that's the visual gap.

---

## Specific questions

### Q1. Does `build_export_triangles` accept tile vertex arrays, or only a Lattice?

**Raw geometry collections, not a Lattice.** Signature is
`(strut_curves, all_triangles, joint_positions, *shape_params)`.
There is no Lattice dependency inside the body — only the three
collections and the shape constants are read.

`Lattice.build_export_triangles` is a thin wrapper that hands in
its cached `_strut_curves`, `_solid_triangles`, `_joint_positions`
(produced by `collect_export_geometry`).

**Refactor implication:** any caller that can produce
`(strut_curves, all_triangles, joint_positions)` from
pose-transformed inputs can hand those to the existing
`build_export_triangles` unchanged.

### Q2. Are hub positions derived from `lattice.points` or from tile vertex arrays?

**From the shrunken tile corners**, in almost every case — see
Section 2 for the full case-split.

The single exception is the 3D-mode central-hub path
(`make_truncated_cuboctahedron(hub_center, scale)`): the
truncated-cuboctahedron is centered at the **lattice-point key
directly**. The mean-distance scale comes from the shrunken
corners, but the center does not. For mode 6 with interior
central hubs, this means a pose-transformed render of those hubs
would need either (a) a Stage 6c.5 decision about how to map
"central hub center" through pose space (the hub itself is one
of the simulator's tiles, so its rigid-body pose IS available),
or (b) an explicit assertion that the test corpus only exercises
non-central hubs (which Stage 6a's mode-6 n=8 test does — none
of those 8 lattice points qualify as central).

For modes 1, 2, 4, 5 and non-central 3D hubs (i.e. everything
the existing test corpus actually hits): hub geometry is fully
derived from shrunken corners, so feeding pose-transformed
corners suffices.

### Q3. Does `_build_pose_mesh_triangles` reuse helpers, or roll its own?

**Rolls its own.** No imports from `auxetic.geometry`, no calls
to `extrude_polygon_solid`, `collect_export_geometry`, or
`build_export_triangles`. Pose application and fan triangulation
are inline (~25 lines).

**Was this deliberate?** The Stage 6c summary explicitly noted
the choice: "Build a temporary mesh from the transformed
vertices (you can reuse the STL-mesh assembly... or write a
simpler tile-only mesh builder — whichever is cleaner)." The
simpler path was chosen to validate the slider→pose→view
plumbing without committing to a full lattice render. The
Stage 6c diagnostic confirms the tile-only path is
constraint-satisfying; the visual gaps come from missing
hub/strut/joint geometry that the full pipeline would supply.

### Q4. Are `TileSystem.tiles[i]` indices the same as `collect_kirigami_tiles` indices?

**Yes, identity-stable.** Walking through `TileSystem.from_lattice`:

1. `tile_arrays_3d, source = collect_kirigami_tiles(...)` — produces
   the canonical-frame tile list in collect-order.
2. `constraint_tuples = build_kirigami_constraints(tile_arrays_3d,
   source)` — references the same indices.
3. World transform applied element-wise: `[ _apply_4x4_to_points(M,
   t) for t in tile_arrays_3d ]` — preserves list order.
4. 2D-planar slicing (`tiles = [t[:, :2].copy() for t in
   tile_arrays_3d]`) — preserves list order.
5. `cls(dimension, tiles, constraints)` stores them; constraints
   become `Constraint` dataclasses carrying the original
   `(tile_a, vert_a, tile_b, vert_b, ctype)` tuple verbatim.

Empirical:
```
ts.constraints[0] = Constraint(tile_a=0, vert_a=0, tile_b=3, vert_b=0, ctype=1)
raw tuple [0]     = (0, 0, 3, 0, 1)
```

One subtle-but-irrelevant note: constraints are computed from
canonical-frame coincidences (step 2 happens before the
world-transform in step 3). Coincidence is preserved by isometries,
so this doesn't affect correctness — only the world coordinates
shift. The indices are stable.
