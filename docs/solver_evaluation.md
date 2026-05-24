# Constraint solver — library evaluation

**Purpose:** Pick (or rule out) an existing Python library for the
quasi-static kinematic solver described in [SPEC.md](../SPEC.md) §7.3.
This document is the deliverable; no solver code lands in this stage.

**Status:** Recommendation at the bottom. Candidate sections in
between document the five evaluation dimensions called out in the
prompt:

1. Does it solve our problem? (rigid tiles + point-coincidence
   constraints + null-space identification + projection back to the
   manifold)
2. What does the API look like for our use case?
3. Active maintenance — last commit ≤ 18 months from today
   (2026-04-30), no glaring open correctness issues.
4. License compatibility — avoid GPL where possible; not yet
   committed to a license ourselves.
5. Dependency footprint — current stack is numpy / scipy / PyQt6 /
   pyvistaqt / pyqtgraph / numpy-stl. Pure-Python deps are fine;
   adding JAX / PyTorch / heavyweight C++ toolchains is not.

---

## 1. Problem we're solving

Restating from SPEC §7.2–7.3 so the comparisons below are anchored:

- **Bodies.** Each kirigami tile (as collected by
  `auxetic.tiles.collect_kirigami_tiles`) is a rigid body — 6 DOF in
  3D, 3 DOF in 2D.
- **Constraints.** Each entry in `constraints.txt` pins vertex `v_a`
  of tile A to vertex `v_b` of tile B (3-DOF or 2-DOF point
  coincidence).
- **Manifold.** The constraint set defines a kinematic manifold of
  allowed configurations, parameterized by the remaining DOFs (the
  kirigami "soft mode").
- **Solver loop.**
  1. Identify kinematic DOFs by computing the **null space of the
     constraint Jacobian** at the rest pose.
  2. Pick the kirigami mode (1-D null space ⇒ unambiguous; multi-D
     ⇒ pick by axial-compliance heuristic, deferred per
     `TODO(physics)`).
  3. Step a small distance along that null-space direction.
  4. **Project back onto the manifold** via constrained least
     squares (Gauss-Newton on the constraint residual).
  5. Record bounding-box height, advance θ.
- **Throughput targets** (SPEC §11.3): 180 θ-samples in <500 ms in
  2D, <5 s in 3D. The hot loop is steps 3–5.

The two algorithmic primitives we need are therefore:

- **Null space of a sparse rectangular matrix** (Jacobian), at sizes
  governed by `N_tiles × DOF_per_tile` columns and `N_constraints
  × dim` rows.
- **Constrained least-squares projection** that minimizes constraint
  residual under a small-step prior — i.e. Gauss-Newton with a fixed
  Jacobian.

Both already exist in SciPy (`scipy.linalg.null_space`,
`scipy.optimize.least_squares` with a callable `jac`). What does
**not** exist in SciPy is the modeling layer that turns "tile A's
vertex 3 coincides with tile B's vertex 7" into a constraint
function and Jacobian block. That's the part we'd be picking up
from a library — or writing.

---

## 2. PyKirigami — `andy-qhjiang/PyKirigami`

The library whose `vertices.txt` + `constraints.txt` format
`auxetic.export.export_kirigami_*` already writes to.

**Last activity.** Most recent commit 2026-04-10 (≤ 1 month). Active.
[Commits page.](https://github.com/andy-qhjiang/PyKirigami/commits/main)

**License.** Apache-2.0. Compatible with anything we'd choose.

**Scope vs. our problem.**
- The repo's entry point is `run_sim.py` — a CLI with `--model`,
  `--brick_thickness`, `--spring_stiffness`, `--gravity`,
  `--force_damping`, etc. It is functionally a CLI tool with
  subroutines, not a library with a public Python API.
- Algorithmically, PyKirigami simulates kirigami deployment via
  **PyBullet** — a soft-real-time rigid-body physics engine with
  inertia, springs, damping, and (optionally) gravity and a ground
  plane.
- This is **fundamentally different from SPEC §7.3.** Our spec asks
  for a *quasi-static kinematic* solver: no time integration, no
  inertia, no spring tuning. PyBullet does exactly the things we
  said we don't want.
- The arXiv writeup
  ([2508.15753v1](https://arxiv.org/html/2508.15753v1)) confirms the
  dynamics model — Algorithm 2 applies vertex forces through spring
  stiffness and damping; there is no null-space extraction, no
  constraint Jacobian, no projection step.

**Our use case in API form.** None — the natural integration would
be `subprocess.run(["python", "run_sim.py", "--vertices_file",
"a.txt", ...])` plus parsing whatever PyBullet logs out. That's a
CLI handoff, not a library use.

**Dependencies.** numpy + pybullet (the README installs them via
conda-forge). PyBullet is ~25 MB compiled; would meaningfully
expand our wheel footprint.

**Verdict.** **Wrong algorithmic family.** PyKirigami answers
"given these tiles + connectivity, what does PyBullet do over time?"
SPEC §7.3 asks "given these tiles + connectivity, what is the
kinematic null space and how does the bounding-box height change as
I sweep the soft mode?" The answers diverge precisely where they
matter — locking detection (§7.5) and compression-metric definition
(§7.4) both depend on the kinematic interpretation.

PyKirigami remains useful as a **future verification baseline** — once
our solver runs, exporting to its file format and comparing
trajectories is a sanity check we already get for free. As the
solver itself it's the wrong tool.

> Sibling note: `lliu12/kirigami_sim` solves the same problem with
> Pymunk (2D physics) instead of PyBullet. Same disposition for the
> same reason.

---

## 3. `scipy.optimize.least_squares` (+ `scipy.linalg.null_space`)

**Already in our dependency list.** Stage 1's `auxetic.geometry`
imports `scipy.spatial.Delaunay` and Stage 5's `auxetic.lattice`
imports `scipy.spatial.transform.Rotation` — `scipy.optimize` and
`scipy.linalg` would land at zero install-time cost.

**License.** BSD-3-Clause. Already governs everything we touch.

**Scope vs. our problem.**
`least_squares` is the projection primitive: it takes a residual
function `fun(x) → r`, an optional callable Jacobian
`jac(x) → J`, an initial guess `x0`, and runs trust-region or
Levenberg-Marquardt to drive `||r||² → 0`. With a sparse Jacobian
(`jac_sparsity=`) it scales to large problems and uses LSMR
internally.
[API reference.](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

`scipy.linalg.null_space(A)` returns an orthonormal basis for the
null space via SVD — that's the kinematic-DOF identification step.

**API for our use case.** Schematically:

```python
# Pseudo-API only — illustrative, not part of this stage's deliverable.
# pose = (positions + rotations) for every tile, packed into a vector.
def constraint_residual(pose):
    r = np.empty(num_constraints * dim)
    for k, (tile_a, v_a, tile_b, v_b) in enumerate(constraints):
        p_a = world_position_of_vertex(pose, tile_a, v_a)
        p_b = world_position_of_vertex(pose, tile_b, v_b)
        r[k*dim : (k+1)*dim] = p_a - p_b
    return r

def constraint_jacobian(pose):
    # Block-sparse: each constraint touches 12 (3D) or 6 (2D) DOFs
    # per pair of tiles. Assemble as scipy.sparse matrix.
    ...

# 1. Null-space of J at rest pose:
J_rest = constraint_jacobian(rest_pose)
modes  = scipy.linalg.null_space(J_rest.toarray())

# 2. Step + project loop (per θ):
target = rest_pose + step_size * modes[:, kirigami_mode_idx]
result = least_squares(constraint_residual, target,
                        jac=constraint_jacobian,
                        method="trf",
                        jac_sparsity=sparsity_pattern)
new_pose = result.x
```

**What scipy doesn't give us.**
- The *modeling* layer: turning the `(tile_a, v_a, tile_b, v_b)`
  records in `constraints.txt` into the residual + Jacobian
  callables.
- The kinematic-mode-selection rule when the null space is multi-
  dimensional (this is the §7.3 `TODO(physics)` anyway — not
  something a generic library would solve for us).

That modeling layer is roughly:
- a `Pose` packing/unpacking routine (tile pose ↔ flat vector);
- the residual function above (~30 LOC);
- the analytic Jacobian (∂residual / ∂pose) — straightforward
  because each constraint is linear in tile translation and
  bilinear-ish in tile rotation (use the standard `R · r ↦
  -[R · r]_×` derivative). ~80 LOC for 3D, ~30 LOC for 2D.

Total estimated solver shim: 200–400 LOC. SciPy does the heavy
lifting (SVD, trust-region steps, LSMR).

**Maintenance / dependencies.** SciPy 1.17.1 already in our env;
maintenance is not a question.

**Verdict.** **Provides everything below the modeling layer and
nothing above it.** That's actually the right level of
abstraction — the modeling layer is short, problem-specific, and
where we get to encode the SPEC §7.3 mode-selection rule on our
own terms. No vendored copy of someone else's joint-type taxonomy
to fight against.

---

## 4. `pymanopt` — Riemannian manifold optimization

**Last release.** 2.2.1 on 2024-09-20 — 19 months ago today
(2026-04-30). Just over the prompt's 18-month threshold. No
post-2.2.1 commits visible on the main branch as of this writing.

**License.** BSD-3-Clause. Compatible.
[GitHub](https://github.com/pymanopt/pymanopt).

**Scope vs. our problem.**
Pymanopt minimizes a cost function over a *named* Riemannian
manifold — Stiefel, Grassmann, sphere, SE(3), product manifolds,
etc. Documentation at
[pymanopt.org/docs](https://pymanopt.org/docs/stable/manifolds.html)
explicitly does **not** include a "level-set" or "constraint-
defined" manifold. The page even invites users to ask the
maintainers about new manifold types.

For our problem, the configuration space is the level set of the
constraint residual — there is no off-the-shelf Pymanopt manifold
that represents it. We'd have to subclass
`pymanopt.manifolds.Manifold` and implement: `random_point`,
`projection_to_tangent`, `retraction`, `log`/`exp`, an inner
product, and parallel transport. Each of those is a chunk of code,
and the retraction in particular is the projection-onto-manifold
step we'd be writing for SciPy anyway — except now we also have
to provide it in the form Pymanopt expects.

**API for our use case.** Even the optimistic version requires us
to write the custom manifold first; only then can we call:

```python
problem = pymanopt.Problem(manifold=KirigamiConstraintManifold(...),
                           cost=lambda pose: -bbox_height(pose))
solver  = pymanopt.optimizers.SteepestDescent()
solver.run(problem)
```

This is a poor fit because:
- Our hot loop isn't "minimize a cost on the manifold." It's
  "step along a known direction, then project back." That's a
  retraction, not a minimization.
- The cost-minimization framing forces an objective-gradient
  callback Pymanopt would call repeatedly — overhead we don't
  need.

**Dependencies.** numpy + (optionally) one of autograd / jax /
pytorch / theano for autodiff. Without an autodiff backend
Pymanopt is fine; with one, the footprint balloons. Our analytic
Jacobian means we wouldn't need autodiff anyway.

**Verdict.** **Wrong shape of API for the work we have to do.** It
optimizes *over* a manifold, but the manifold class is the part
we'd be writing — so Pymanopt isn't saving us the modeling work,
just adding an outer optimizer we don't need.

If the §7.4 compression metric eventually becomes a real
optimization problem (search over rigid rotation `R` in §8 — and
that is rotation, not constraint manifold), Pymanopt's SE(3) /
SO(3) manifolds *would* fit cleanly. Worth keeping in mind as
Stage 7+ work.

---

## 5. `compas_fd` — force density form-finding

**Last release.** 0.5.4 on 2024-11-10. Active enough.
[GitHub](https://github.com/blockresearchgroup/compas_fd).

**License.** Visible LICENSE file; per the COMPAS project README it's
MIT — permissive.

**Scope vs. our problem.**
The force density method (FDM) finds equilibrium shapes for
**axial-force networks**: cable nets, membranes, gridshells. Each
edge has a force-to-length ratio; vertices are placed by solving
a linear system that balances the force-density-weighted edge
sums. It is *not* a rigid-body solver — there is no notion of a
rigid tile, no rotational DOF, no point-coincidence between
distinct rigid bodies, and the constraints it supports
(fixed-vertex, plane, line) live on the network nodes themselves.

The COMPAS-FoFin documentation
([blockresearchgroup.gitbook.io/compas-fofin](https://blockresearchgroup.gitbook.io/compas-fofin))
confirms the domain: "form finding and structural design ... with
axial force members in tension and compression."

**Verdict.** **Wrong problem.** Adjacent in vibe (constrained
geometric solver in Python, computational-design-flavored), but the
mathematical model has nothing in common with rigid-tile kirigami.

---

## 6. Additional candidates surfaced during research

Briefly, since the prompt asked for an open search:

### `pymadcad` — CAD library with a kinematic solver
[GitHub](https://github.com/jimy-byerley/pymadcad). LGPL-3.0 / GPL-3.0
dual license. Has a real kinematic system with named joints (Ball,
Revolute, Planar, Cylindrical, Prismatic, Weld) — the closest
domain match. **But:**
- 14% Rust + GLSL by lines, plus a `moderngl` + Qt rendering stack
  it expects you to use. Heavy install.
- Built for CAD assemblies (handful of solids) rather than lattices
  with 50–500 tiles; performance characteristics for our N are
  unverified.
- LGPL-3.0 is weak copyleft (dynamic linking is fine), but if we
  later want to ship a permissively-licensed wheel, the LGPL bit
  forces a "user can replace the pymadcad library" provision.
- Worth a second look in Stage 6.5+ if our hand-rolled solver hits a
  correctness wall on a specific joint type. Not the right
  starting point.

### `pydy` — symbolic multibody dynamics via SymPy
[GitHub](https://github.com/pydy/pydy). BSD-3-Clause. Last release
2026-04-26, very active. Generates symbolic equations of motion via
SymPy; you then numerically integrate. **But:**
- Symbolic EOM generation is the inverse of what SPEC §7.3 wants —
  the spec explicitly says "no time integration of mass."
- The output is a system of ODEs, not a constraint-projection
  primitive.
- Pulling in SymPy is fine (BSD, pure Python) but unnecessary.

### EXUDYN — C++/Python multibody dynamics
[GitHub](https://github.com/jgerstmayr/EXUDYN). Has a static solver
and constraint support. **But:** C++ core, prebuilt wheels per
Python version (Python 3.13 is current upper bound — we're on
3.14, so we'd be building from source). Heavy industrial
package; overkill for the scale we run at.

### `pinocchio`, `PositionBasedDynamics` (pyPBD)
Robotics-grade rigid-body dynamics; both have C++ cores. Wrong
scale and wrong algorithmic family (forward dynamics + integrators,
not quasi-static constraint projection).

### `python-constraint`, `ericPrince/constraint-solver`
Generic CSP / 2D drafting-style geometric solvers. Not rigid-body.
Hobby-scale, no relevant correctness pedigree.

---

## 7. Recommendation: **build from scratch, on top of `scipy`**

The constraint solver should be implemented in `auxetic/solver.py`
(new file in a future stage) using SciPy's existing `linalg.null_space`
and `optimize.least_squares` as primitives.

**Why no library wins outright:**

| Criterion (per prompt) | PyKirigami | scipy alone | pymanopt | compas_fd | pymadcad |
|---|---|---|---|---|---|
| Solves our problem | ✗ (PyBullet dynamics) | partial — primitives only | ✗ (no level-set manifold OOTB) | ✗ (axial-force network) | ~ (CAD assemblies, small N) |
| Reasonable API | ✗ (CLI tool) | ✓ (callable + jac) | ✗ (write custom manifold first) | n/a | ~ (heavy) |
| Maintained | ✓ (Apr 2026) | ✓ (Apr 2026) | borderline (Sep 2024) | ✓ (Nov 2024) | ✓ |
| License | ✓ Apache-2.0 | ✓ BSD-3 | ✓ BSD-3 | ✓ MIT | ✗ LGPL/GPL-3 |
| Footprint | + PyBullet | already in tree | + autodiff if used | already-light | + Rust + Qt + GL |

**Closest contenders, and where they fall short:**

1. **`scipy.optimize.least_squares` + `scipy.linalg.null_space`** —
   already in our tree, BSD-licensed, exposes the exact two
   primitives SPEC §7.3 calls for. The gap is just the modeling
   layer (constraint residual + analytic Jacobian assembled from
   `constraints.txt`), which is ~200–400 LOC and *intrinsically*
   problem-specific. Recommended.

2. **`pymadcad`** — has the right domain primitives (rigid bodies +
   named joints + a real kinematic solver). Falls short on three
   independent axes: GPL-3.0 component restricts our future
   licensing options; Rust + Qt + GL install footprint is heavy
   for a project that currently stops at numpy/scipy/Qt; designed
   for low-N CAD assemblies rather than the hundreds of tiles a
   non-trivial lattice produces. Worth re-evaluating later if the
   from-scratch solver hits a problem-class wall it can't paper
   over.

3. **`pymanopt`** — closest "this is exactly the math" match
   conceptually (constraint manifold ≡ Riemannian manifold), but
   in practice we'd have to write a custom `Manifold` subclass,
   which is the same modeling-layer code we're avoiding. The
   outer minimization loop it provides isn't the operation our
   hot path actually needs (we want a retraction, not a
   cost-minimization). Net: not a saving.

**Why "build from scratch" isn't reckless:**

- The two algorithmic primitives we need (null-space identification,
  constrained-LS projection) are SciPy library calls. The novel
  code is exclusively in the modeling layer — and that layer is
  short, well-defined, and tightly coupled to our existing
  `vertices.txt` / `constraints.txt` schema.
- Analytic Jacobians for point-coincidence-of-rigid-bodies are
  textbook (the `[ω]_×` cross-product matrix for the rotation
  partial). No autodiff dependency required.
- Sparsity is trivially exploitable: each constraint row touches
  exactly two tiles' DOF blocks. SciPy's sparse Jacobian path will
  hit our perf budget (180 θ-samples in <500 ms / <5 s) without
  drama.
- The §7.3 mode-selection `TODO(physics)` is an open question we
  *want* in our own code, where we can iterate on heuristics
  alongside the rest of the simulation work — not buried inside a
  third-party API we don't control.

**Concrete next-stage shape (for whoever picks up Stage 6+):**

- New `auxetic/solver.py` with:
  - `pose_pack(tiles) → ndarray` and the inverse;
  - `constraint_residual(pose, constraints) → ndarray`;
  - `constraint_jacobian(pose, constraints) → scipy.sparse.csr`;
  - `null_space_at(pose) → (modes, sigma)` via
    `scipy.linalg.null_space`;
  - `project_to_manifold(pose, target_step) →
    pose` via `scipy.optimize.least_squares` (method `trf`,
    `jac_sparsity` from constraint pattern).
- New `auxetic/sim.py` with the θ-sweep loop and `SimResult`
  (already typed in SPEC §7.6).
- Tests follow the same pattern as Stage 1's regression suite —
  hand-build a minimal lattice (e.g. 4 tiles, 4 constraints) where
  the kinematic mode is known analytically and verify the solver
  recovers it.

---

## Sources

- PyKirigami:
  [GitHub](https://github.com/andy-qhjiang/PyKirigami) /
  [arXiv 2508.15753](https://arxiv.org/html/2508.15753v1) /
  [commits](https://github.com/andy-qhjiang/PyKirigami/commits/main)
- `lliu12/kirigami_sim`: [GitHub](https://github.com/lliu12/kirigami_sim)
- SciPy `least_squares`:
  [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
- Pymanopt:
  [pymanopt.org](https://pymanopt.org/) /
  [GitHub](https://github.com/pymanopt/pymanopt) /
  [manifolds list](https://pymanopt.org/docs/stable/manifolds.html)
- compas_fd:
  [GitHub](https://github.com/blockresearchgroup/compas_fd) /
  [COMPAS-FoFin docs](https://blockresearchgroup.gitbook.io/compas-fofin)
- pymadcad: [GitHub](https://github.com/jimy-byerley/pymadcad)
- PyDy: [GitHub](https://github.com/pydy/pydy) /
  [pydy.org](https://pydy.org/)
- EXUDYN: [GitHub](https://github.com/jgerstmayr/EXUDYN)
