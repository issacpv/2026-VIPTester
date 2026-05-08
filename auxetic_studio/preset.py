"""Preset save/load for Auxetic Studio.

Schema is SPEC §5.1 v2 — flat top-level fields:

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
        "name": "...",
        "created": "ISO-8601",
        "modified": "ISO-8601",
        "notes": "..."
      }
    }

Notes:
- ``points_original`` is **not** stored in presets. After load,
  ``points_original`` is set to the loaded points so "Reset to
  Original" returns to the as-loaded state.
- ``seed`` is intentionally not stored either — once explicit
  ``points`` are saved the seed is no longer load-bearing for
  reproducibility, and ``regenerate()`` is meant to discard the saved
  points (SPEC §4.3) so a stale seed would be misleading.
- The Stage 2 preset format wrote ``version: 1`` with a *different*
  schema (everything nested under ``"lattice"``). That collision is
  why ``_is_legacy`` does both a version check and a structural check
  for the legacy ``"lattice"`` dict — the prompt's stated rule
  ("version missing or 0") covers fresh legacy files but not Stage 2
  saves already on disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict

import numpy as np

from auxetic import Lattice
from auxetic import geometry as _geom


PRESET_VERSION = 4  # M2 — adds the "dynamics" block (forces, ground
                     # face, pre-rotation overrides, fixed tiles, dt /
                     # stiffness config). v3 → v4 migration just fills
                     # defaults; geometry payload is unchanged.


# ---------------------------------------------------------------------------
# Stub builders (used both for legacy migration and for empty defaults)
# ---------------------------------------------------------------------------

def _stub_view_state() -> Dict[str, Any]:
    """Identity view_state per SPEC §5.1. Stage 5+ will populate these
    from the live camera / flip toggle / θ slider."""
    return {
        "rigid_rotation_quat": [1.0, 0.0, 0.0, 0.0],
        "flipped":             False,
        "joint_angle_deg":     0.0,
    }


def _stub_metadata() -> Dict[str, Any]:
    return {
        "name":     "",
        "created":  "",
        "modified": "",
        "notes":    "",
    }


def _default_shape_params() -> Dict[str, float]:
    """SPEC §5.1's four shape_params, defaulted from
    ``auxetic.geometry`` constants so a missing block round-trips to
    the same rendering as before."""
    return {
        "ngon_thickness":      _geom.NGON_THICKNESS,
        "hub_size_factor":     _geom.HUB_SIZE_FACTOR,
        "joint_sphere_radius": _geom.JOINT_SPHERE_RADIUS,
        "strut_radius":        _geom.STRUT_RADIUS,
    }


def _stub_generation() -> Dict[str, Any]:
    """v3 ``generation`` block defaults — every field set to the value
    that produces V20-equivalent output. Used both by ``save_preset``
    when the lattice's M1 fields haven't been touched and by the v2→v3
    migration when loading older files."""
    return {
        "density_axis":     "none",
        "density_law":      "uniform",
        "density_strength": 1.0,
        "edge_flips":       [],
        "mesh_path":        None,
        "mesh_vertices":    None,
        "unit_scale_cm":    1.0,
    }


def _stub_dynamics() -> Dict[str, Any]:
    """v4 ``dynamics`` block defaults. Mirrors
    ``Lattice.dynamics_state`` in-memory defaults: a 5 N piston
    compression load case as the ready-to-go workflow. Set
    ``piston_force_n`` to 0 to fall back to the manual ground-face
    + force-table path."""
    return {
        "piston_force_n":      5.0,
        "forces":              [],
        "ground_face":         None,
        "pre_rotation_quat":   None,
        "pre_joint_angle_deg": None,
        "fixed_tiles":         [],
        "config": {
            "dt":                         1.0e-3,
            "duration":                   1.0,
            "joint_stiffness":            5.0e2,
            "joint_damping":              5.0e0,
            "gravity_cm_per_s2":          [0.0, 0.0, 0.0],
            "convergence_kinetic_thresh": 1.0e-5,
        },
    }


def _default_now() -> str:
    """ISO-8601 UTC with microseconds — sub-second precision lets two
    saves in the same second still produce different ``modified``
    strings, which is what
    ``test_metadata_created_preserved_across_saves`` expects."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_preset(
    path: str,
    lattice: Lattice,
    *,
    _now_fn: Callable[[], str] | None = None,
) -> None:
    """Write ``lattice`` to ``path`` as a v1 preset.

    Side-effect: updates ``lattice.metadata['created']`` (only if not
    previously set) and ``lattice.metadata['modified']`` (always) so
    repeated saves on the same in-memory lattice see the timestamps
    advance correctly.

    ``_now_fn`` is an injection seam for tests — production code should
    let it default to UTC ISO-8601 with microseconds."""
    now = _now_fn or _default_now

    # ---- metadata: preserve created, advance modified --------------------
    md = dict(lattice.metadata) if lattice.metadata else _stub_metadata()
    if not md.get("created"):
        md["created"] = now()
    md["modified"] = now()
    md.setdefault("name",  "")
    md.setdefault("notes", "")

    # Reflect the just-saved state back onto the lattice so a follow-up
    # save sees ``created`` already populated.
    lattice.metadata.update({
        "created":  md["created"],
        "modified": md["modified"],
        "name":     md["name"],
        "notes":    md["notes"],
    })

    # ---- M1 generation block (v3) ---------------------------------------
    # Edge flips: serialise as sorted list of [i, j] pairs (JSON has no
    # set type). Mesh vertices: serialise inline so the preset is
    # self-contained even if the source file moves; ``mesh_path`` is
    # informational only.
    edge_flips_list = sorted(
        [list(map(int, e)) for e in getattr(lattice, "edge_flips", set())]
    )
    mesh_verts = getattr(lattice, "mesh_vertices", None)
    generation = {
        "density_axis":     str(getattr(lattice, "density_axis", "none")),
        "density_law":      str(getattr(lattice, "density_law", "uniform")),
        "density_strength": float(getattr(lattice, "density_strength", 1.0)),
        "edge_flips":       edge_flips_list,
        "mesh_path":        getattr(lattice, "mesh_path", None),
        "mesh_vertices":    (np.asarray(mesh_verts).tolist()
                             if mesh_verts is not None else None),
        "unit_scale_cm":    float(getattr(lattice, "unit_scale_cm", 1.0)),
    }

    # ---- M2 dynamics block (v4) -----------------------------------------
    # Lattice.dynamics_state already mirrors the v4 schema 1:1, so just
    # deep-copy. Missing-or-shaped-wrong dicts get filled with defaults.
    dynamics_dict = dict(getattr(lattice, "dynamics_state", None) or _stub_dynamics())
    for k, v in _stub_dynamics().items():
        dynamics_dict.setdefault(k, v)
    # Nested config block — fill any missing inner key.
    cfg = dict(dynamics_dict.get("config") or {})
    for k, v in _stub_dynamics()["config"].items():
        cfg.setdefault(k, v)
    dynamics_dict["config"] = cfg

    payload: Dict[str, Any] = {
        "version":   PRESET_VERSION,
        "mode":      int(lattice.mode),
        "n_points":  int(lattice.n_points),
        "ratio":     float(lattice.ratio),
        "nz_layers": int(lattice.nz_layers),
        # tolist() preserves dimensionality automatically (Nx2 / Nx3).
        "points":    np.asarray(lattice.points).tolist(),
        "shape_params": {
            "ngon_thickness":      float(lattice.ngon_thickness),
            "hub_size_factor":     float(lattice.hub_size_factor),
            "joint_sphere_radius": float(lattice.joint_sphere_radius),
            "strut_radius":        float(lattice.strut_radius),
        },
        "generation": generation,
        "dynamics":   dynamics_dict,
        "view_state": dict(lattice.view_state) if lattice.view_state else _stub_view_state(),
        "metadata":   md,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _is_legacy(data: Dict[str, Any]) -> bool:
    """Detect Stage 2 / pre-v1 presets.

    The prompt-stated rule is ``version missing or 0``. We also detect
    the Stage 2 stub structurally (a nested ``"lattice"`` dict) because
    Stage 2 set ``version: 1`` despite not conforming to §5.1, so
    existing Stage 2 saves on disk wouldn't be caught by the version
    rule alone."""
    if data.get("version", 0) == 0:
        return True
    return isinstance(data.get("lattice"), dict) and "mode" not in data


def load_preset(path: str) -> Lattice:
    """Read ``path`` and return a fully-populated ``Lattice``.

    Migration chain: legacy/Stage-2 → v1 → v2 → load. Each step is a
    pure dict transform; the loader proper never branches on version.

    Raises ``ValueError`` if the file's ``version`` is newer than this
    application supports."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if _is_legacy(data):
        # Legacy / Stage 2 stub → v1-shaped dict.
        data = _migrate_legacy(data)

    version = int(data.get("version", PRESET_VERSION))
    if version > PRESET_VERSION:
        raise ValueError(
            f"Preset version {version} is newer than this application supports."
        )

    if version == 1:
        # v1 → v2: fill identity rotation/flip/joint defaults for any
        # field the v1 file didn't carry.
        data = _migrate_v1_to_v2(data)

    if int(data.get("version", PRESET_VERSION)) == 2:
        data = _migrate_v2_to_v3(data)

    if int(data.get("version", PRESET_VERSION)) == 3:
        data = _migrate_v3_to_v4(data)

    # ---- M1 generation block (v3) ---------------------------------------
    # Pull these out first so they can be passed to the Lattice
    # constructor. Mesh-import modes (7, 8, 9) need ``mesh_vertices``
    # available BEFORE ``__init__`` calls ``regenerate``.
    gen = dict(data.get("generation") or _stub_generation())
    mesh_vertices_list = gen.get("mesh_vertices")
    mesh_vertices = (np.asarray(mesh_vertices_list, dtype=float)
                     if mesh_vertices_list else None)
    edge_flips_raw = gen.get("edge_flips") or []
    edge_flips = {tuple(sorted((int(e[0]), int(e[1])))) for e in edge_flips_raw}

    # ---- construct the Lattice with saved scalars -----------------------
    lattice = Lattice(
        mode      = int(data.get("mode", 1)),
        n_points  = int(data.get("n_points", 5)),
        ratio     = float(data.get("ratio", 0.35)),
        nz_layers = int(data.get("nz_layers", 2)),
        density_axis     = str(gen.get("density_axis",     "none")),
        density_law      = str(gen.get("density_law",      "uniform")),
        density_strength = float(gen.get("density_strength", 1.0)),
        edge_flips       = edge_flips,
        mesh_path        = gen.get("mesh_path"),
        mesh_vertices    = mesh_vertices,
        unit_scale_cm    = float(gen.get("unit_scale_cm", 1.0)),
    )

    # ---- install saved points (and freeze them as the new "original") --
    saved_points = data.get("points") or []
    if saved_points:
        pts = np.asarray(saved_points, dtype=float)
        lattice.regenerate_from_points(pts)
        # SPEC §5.1 doesn't store points_original; the as-loaded points
        # ARE the new original, so Reset → returns to load state.
        lattice.points_original = lattice.points.copy()
    # else: __init__'s regenerate() already set points + points_original
    # to a fresh roll; missing-points presets fall back to that.

    # ---- shape_params ---------------------------------------------------
    sp = data.get("shape_params") or {}
    if "ngon_thickness"      in sp: lattice.ngon_thickness      = float(sp["ngon_thickness"])
    if "hub_size_factor"     in sp: lattice.hub_size_factor     = float(sp["hub_size_factor"])
    if "joint_sphere_radius" in sp: lattice.joint_sphere_radius = float(sp["joint_sphere_radius"])
    if "strut_radius"        in sp: lattice.strut_radius        = float(sp["strut_radius"])

    # ---- M2 dynamics block (v4) -----------------------------------------
    dyn = dict(data.get("dynamics") or _stub_dynamics())
    # Fill any missing top-level keys so the lattice always sees the
    # full v4 schema even if the on-disk file pre-dated a sub-field.
    for k, v in _stub_dynamics().items():
        dyn.setdefault(k, v)
    inner_cfg = dict(dyn.get("config") or {})
    for k, v in _stub_dynamics()["config"].items():
        inner_cfg.setdefault(k, v)
    dyn["config"] = inner_cfg
    lattice.dynamics_state = dyn

    # ---- view_state + metadata (stored, not yet acted on) ---------------
    lattice.view_state = dict(data.get("view_state") or _stub_view_state())
    lattice.metadata   = dict(data.get("metadata")   or _stub_metadata())

    return lattice


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------

def _migrate_legacy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the Stage 2 stub (``{"version": 1, "lattice": {...},
    "view_state": {...}, "metadata": {...}}``) — and any older
    version-less file — into a **v1-shaped** dict. The v1→v2 migration
    runs as a separate step so this function doesn't need to know about
    rotation defaults.

    Missing fields get defaults from the current ``Lattice`` class /
    geometry constants. Missing ``points`` becomes ``[]``, which routes
    the loader to a fresh ``regenerate()`` rather than fabricating
    coordinates."""
    legacy_lattice = data.get("lattice") or {}

    md = dict(data.get("metadata") or {})
    # Stage 2's metadata dict had {name, author, notes, created}; v1
    # drops author and adds modified. Preserve created if present.
    md_v1 = _stub_metadata()
    md_v1["name"]     = str(md.get("name", ""))
    md_v1["notes"]    = str(md.get("notes", ""))
    md_v1["created"]  = str(md.get("created", "")) or _default_now()
    md_v1["modified"] = _default_now()

    return {
        "version":   1,  # _migrate_v1_to_v2 then promotes this
        "mode":      int(legacy_lattice.get("mode", 1)),
        "n_points":  int(legacy_lattice.get("n_points", 5)),
        "ratio":     float(legacy_lattice.get("ratio", 0.35)),
        "nz_layers": int(legacy_lattice.get("nz_layers", 2)),
        "points":    [],  # forces __init__ regenerate
        "shape_params": _default_shape_params(),
        "view_state":   _stub_view_state(),
        "metadata":     md_v1,
    }


def _migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Promote a v1 preset to v2: fills identity rotation, flipped=False,
    and joint_angle_deg=0 for any view_state field the v1 file is
    missing. Stage 4 v1 files already wrote stub identity values for
    these — this migration is a safety net for hand-rolled or
    third-party v1 files that omit them."""
    out = dict(data)
    out["version"] = 2

    vs = dict(out.get("view_state") or {})
    vs.setdefault("rigid_rotation_quat", [1.0, 0.0, 0.0, 0.0])
    vs.setdefault("flipped",             False)
    vs.setdefault("joint_angle_deg",     0.0)
    out["view_state"] = vs

    return out


def _migrate_v2_to_v3(data: Dict[str, Any]) -> Dict[str, Any]:
    """Promote a v2 preset to v3: adds the ``generation`` block with
    behavior-preserving defaults (no density bias, no edge flips, no
    mesh import, unit scale 1 cm). v2 files load with the same
    geometry they would in the v2 codebase."""
    out = dict(data)
    out["version"] = 3
    gen = dict(out.get("generation") or {})
    defaults = _stub_generation()
    for key, value in defaults.items():
        gen.setdefault(key, value)
    out["generation"] = gen
    return out


def _migrate_v3_to_v4(data: Dict[str, Any]) -> Dict[str, Any]:
    """Promote a v3 preset to v4: adds the ``dynamics`` block with
    empty defaults (no forces, no ground face, default integrator
    config). v3 files load with no behaviour change."""
    out = dict(data)
    out["version"] = 4
    dyn = dict(out.get("dynamics") or {})
    defaults = _stub_dynamics()
    for key, value in defaults.items():
        dyn.setdefault(key, value)
    inner_cfg = dict(dyn.get("config") or {})
    for k, v in defaults["config"].items():
        inner_cfg.setdefault(k, v)
    dyn["config"] = inner_cfg
    out["dynamics"] = dyn
    return out
