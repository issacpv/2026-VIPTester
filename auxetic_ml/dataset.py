"""Sample-generation pipeline for the M3 predictor.

For each ``(starting lattice, action)`` pair we run the M2 dynamic
sim and record the (compression efficiency, stability score) outcome.
The resulting :class:`Sample` records can be batched into a training
set and consumed by either the MLP head (via
:func:`auxetic_ml.features.lattice_features`) or the GNN head.

Storage:
    ``save_samples(samples, dir)`` writes one ``sample_NNNN.npz``
    per sample plus a top-level ``manifest.json`` listing the files.
    Sticking with ``.npz`` per sample (rather than a single HDF5
    file with compound types) keeps the pipeline portable — no
    h5py dep — and makes it trivial to drop in / swap out individual
    samples by hand.

Action space:
    - ``ground_face`` : one of ``"+x" / "-x" / "+y" / "-y" / "+z" / "-z"``
      or ``None`` (no anchored face).
    - ``edge_flips``  : a subset of the lattice's flippable edges
      (uniform random subset, controlled by a probability per edge).
    - ``pre_rotation_rad`` : starting joint angle, drawn uniformly
      from ``[-π, π]`` (matches the M2.8 extended sweep range).

A :class:`SampleGenerator` orchestrates: build the configured
lattice, sample N actions, run dynamics for each, pack into Samples.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


GROUND_FACES: Tuple[Optional[str], ...] = (
    None, "+x", "-x", "+y", "-y", "+z", "-z",
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """One action sampled from the predictor's action space."""
    ground_face:      Optional[str]   = None
    edge_flips:       Tuple[Tuple[int, int], ...] = ()
    pre_rotation_rad: float           = 0.0


@dataclass
class SampleLabel:
    """The simulator's outcome on a (lattice, action) pair."""
    compression_efficiency: float
    stability_score:        float
    converged:              bool
    final_kinetic_energy:   float


@dataclass
class Sample:
    """Single training sample: features + action + label.

    ``features`` is the fixed-size handcrafted feature vector
    (``auxetic_ml.features.lattice_features``); ``action`` is the
    (face, edge-flip subset, pre-rotation) tuple; ``label`` carries
    the dynamic-sim outcome.
    """
    features: np.ndarray
    action:   Action
    label:    SampleLabel
    # Optional: original lattice scalar metadata for diagnostics /
    # later GNN feature extraction.
    lattice_meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

def _faces_for_dim(dim: int) -> Tuple[Optional[str], ...]:
    """Filter ``GROUND_FACES`` to those that make sense for a lattice
    of the given dimensionality. For 2D, drop the z-faces (the
    dynamic solver raises on a z-face request when dim < 3)."""
    if dim >= 3:
        return GROUND_FACES
    return tuple(f for f in GROUND_FACES
                 if f is None or not f.endswith("z"))


def sample_action(rng: np.random.Generator,
                   flippable_edges: Sequence[Tuple[int, int]],
                   *,
                   flip_probability:        float = 0.3,
                   pre_rotation_range_rad:  Tuple[float, float] = (-np.pi, np.pi),
                   dim:                     int = 3,
                   ) -> Action:
    """Draw one random :class:`Action`.

    - ``ground_face`` is uniform over ``GROUND_FACES`` (including
      ``None``), filtered to faces that match ``dim`` so we don't ask
      a 2D lattice to anchor a z-face (which the dynamic solver
      rejects).
    - Each flippable edge is included independently with probability
      ``flip_probability`` (Bernoulli). Yields ``2^N`` patterns weighted
      toward the empty pattern when ``flip_probability < 0.5``.
    - ``pre_rotation_rad`` is uniform over ``pre_rotation_range_rad``.
    """
    faces = _faces_for_dim(dim)
    face = faces[int(rng.integers(0, len(faces)))]
    flips = tuple(
        tuple(e) for e in flippable_edges
        if float(rng.random()) < flip_probability
    )
    pre_rot = float(rng.uniform(*pre_rotation_range_rad))
    return Action(
        ground_face=face, edge_flips=flips, pre_rotation_rad=pre_rot,
    )


# ---------------------------------------------------------------------------
# Sample generator
# ---------------------------------------------------------------------------

LatticeFactory = Callable[[], Any]   # returns a fresh Lattice each call


def _apply_action(lattice, action: Action) -> None:
    """Mutate ``lattice`` in place to install ``action``."""
    lattice.dynamics_state["ground_face"] = action.ground_face
    lattice.edge_flips = set(action.edge_flips)
    lattice.joint_angle = float(action.pre_rotation_rad)
    # Re-triangulate so edge flips take effect on the live tri.
    if lattice.points is not None:
        lattice.regenerate_from_points(lattice.points)


def _label_from_lattice(lattice, *,
                         duration: float = 0.1,
                         dt:       float = 1.0e-3) -> SampleLabel:
    """Run the dynamic sim with the current ``lattice.dynamics_state``
    and reduce the trajectory to a :class:`SampleLabel`.

    Compression efficiency: ``DynamicsResult.final_compression`` —
    fractional change in axial bbox extent. Positive = compressed.
    Stability score: ``1 / (1 + final_kinetic_energy)`` — higher when
    the sim settles cleanly.
    """
    # Local import so the module stays usable even if the dynamics
    # solver evolves; cycles are avoided.
    from auxetic.dynamics import build_dynamics_simulator_from_lattice

    # Cap duration to keep sample-gen tractable; the user can tune.
    saved_dt       = lattice.dynamics_state["config"].get("dt")
    saved_duration = lattice.dynamics_state["config"].get("duration")
    lattice.dynamics_state["config"]["dt"]       = float(dt)
    lattice.dynamics_state["config"]["duration"] = float(duration)
    try:
        ds  = build_dynamics_simulator_from_lattice(lattice)
        res = ds.simulate()
    finally:
        lattice.dynamics_state["config"]["dt"]       = saved_dt
        lattice.dynamics_state["config"]["duration"] = saved_duration

    final_ke = float(res.energy_trace["kinetic"][-1]) if res.energy_trace.get(
        "kinetic", np.array([0.0])).size else 0.0
    if not np.isfinite(final_ke):
        final_ke = float("inf")
    stability = 1.0 / (1.0 + max(0.0, final_ke))
    comp = float(res.final_compression)
    if not np.isfinite(comp):
        comp = 0.0
    return SampleLabel(
        compression_efficiency=comp,
        stability_score=float(stability),
        converged=bool(res.converged),
        final_kinetic_energy=final_ke,
    )


def generate_samples(lattice_factory: LatticeFactory,
                      n_samples: int,
                      *,
                      seed: int = 0,
                      duration: float = 0.1,
                      dt:       float = 1.0e-3,
                      ) -> List[Sample]:
    """Generate ``n_samples`` Samples for the given lattice family.

    ``lattice_factory`` is called once per sample to produce a fresh
    :class:`auxetic.Lattice`; this lets callers vary mode / n_points /
    seed across the dataset (e.g., generate samples for several
    different lattice configs in one call).

    The same RNG ``seed`` produces the same sample sequence — useful
    for regression tests and reproducible runs.
    """
    from auxetic.geometry import flippable_edges as _flippable_edges
    from auxetic_ml.features import lattice_features

    rng = np.random.default_rng(seed)
    out: List[Sample] = []
    for _ in range(int(n_samples)):
        lattice = lattice_factory()
        # Compute flippable edges on the canonical (action-free) tri.
        edges = _flippable_edges(lattice.tri, lattice.points)
        # Lattice dimensionality (2 for 2D modes incl. extruded, 3 for
        # native 3D modes) — used to filter the ground_face options so
        # we never ask a 2D lattice to anchor a z-face.
        dim = 3 if int(lattice.mode) in (3, 6, 9) else 2
        action = sample_action(rng, edges, dim=dim)
        _apply_action(lattice, action)
        feats = lattice_features(lattice)
        label = _label_from_lattice(lattice, duration=duration, dt=dt)
        meta = {
            "mode":      int(lattice.mode),
            "n_points":  int(lattice.n_points),
            "ratio":     float(lattice.ratio),
            "nz_layers": int(lattice.nz_layers),
        }
        out.append(Sample(
            features=feats, action=action, label=label, lattice_meta=meta,
        ))
    return out


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

_MANIFEST_NAME = "manifest.json"


def save_samples(samples: Iterable[Sample], dir_path: str | Path) -> Path:
    """Write samples to ``dir_path`` as one ``sample_NNNN.npz`` per
    sample plus a ``manifest.json`` listing them.

    Returns the directory path so callers can chain.
    """
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    file_names: List[str] = []
    for i, s in enumerate(samples):
        name = f"sample_{i:04d}.npz"
        np.savez(
            p / name,
            features=s.features,
            edge_flips=np.asarray([list(e) for e in s.action.edge_flips],
                                   dtype=int) if s.action.edge_flips
                       else np.zeros((0, 2), dtype=int),
            ground_face=np.asarray(
                "" if s.action.ground_face is None else s.action.ground_face,
                dtype=str),
            pre_rotation_rad=np.asarray(s.action.pre_rotation_rad, dtype=float),
            compression_efficiency=np.asarray(
                s.label.compression_efficiency, dtype=float),
            stability_score=np.asarray(s.label.stability_score, dtype=float),
            converged=np.asarray(s.label.converged, dtype=bool),
            final_kinetic_energy=np.asarray(s.label.final_kinetic_energy,
                                              dtype=float),
        )
        file_names.append(name)
    manifest = {
        "feature_version": 1,
        "n_samples":       len(file_names),
        "files":           file_names,
        "lattice_meta_per_sample": [
            s.lattice_meta for s in samples
        ] if hasattr(samples, "__len__") else None,
    }
    with open(p / _MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return p


def load_samples(dir_path: str | Path) -> List[Sample]:
    """Inverse of :func:`save_samples`."""
    p = Path(dir_path)
    manifest_path = p / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    out: List[Sample] = []
    for i, name in enumerate(manifest["files"]):
        with np.load(p / name, allow_pickle=False) as data:
            features = np.asarray(data["features"])
            edge_flips_arr = np.asarray(data["edge_flips"]).reshape(-1, 2)
            edge_flips = tuple(
                tuple(int(v) for v in row) for row in edge_flips_arr
            )
            face_str = str(data["ground_face"])
            ground_face = None if face_str == "" else face_str
            action = Action(
                ground_face=ground_face,
                edge_flips=edge_flips,
                pre_rotation_rad=float(data["pre_rotation_rad"]),
            )
            label = SampleLabel(
                compression_efficiency=float(data["compression_efficiency"]),
                stability_score=float(data["stability_score"]),
                converged=bool(data["converged"]),
                final_kinetic_energy=float(data["final_kinetic_energy"]),
            )
        meta_list = manifest.get("lattice_meta_per_sample") or []
        meta = meta_list[i] if i < len(meta_list) else {}
        out.append(Sample(
            features=features, action=action, label=label, lattice_meta=meta,
        ))
    return out
