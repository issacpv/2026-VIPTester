"""Lattice → fixed-size feature vector.

Used by the MLP head of the M3 predictor (the GNN head will extract
its own graph features in :mod:`auxetic_ml.model`). The feature
schema is intentionally compact and interpretable — every component
has a documented meaning so a regression-tree baseline could be
trained on the same vector for sanity checks.

Schema (see :data:`FEATURE_NAMES` for the live order):

- ``mode_one_hot``     (9) — one-hot of ``lattice.mode`` (modes 1..9)
- ``n_points``         (1) — number of lattice points
- ``ratio``            (1) — kirigami shrink ratio
- ``unit_scale_cm``    (1) — physical scale (cm per lattice unit)
- ``n_edge_flips``     (1) — count of currently-flipped edges
- ``cov_eigvals``      (3) — eigenvalues of the point-cloud
                              covariance matrix (z-padded to 3 in 2D)
- ``bbox_aspect``      (3) — bbox extent ratios (x/y/z normalised by
                              the largest extent; smallest = 1.0)
- ``rigid_quat_wxyz``  (4) — current rigid_rotation quaternion
- ``flipped``          (1) — 1.0 if the mirror toggle is on, else 0
- ``joint_angle_rad``  (1) — current joint angle in radians

Total: 25 elements. Pad/extend by appending new fields and bumping
:data:`FEATURE_VERSION`; existing models trained on an older version
should reject mismatched feature dimensions.
"""

from __future__ import annotations

from typing import List

import numpy as np


FEATURE_VERSION = 1
N_MODES = 9   # modes 1 through 9 (M1 expanded the taxonomy to 9)

FEATURE_NAMES: List[str] = (
    [f"mode_{i}" for i in range(1, N_MODES + 1)]
    + ["n_points", "ratio", "unit_scale_cm", "n_edge_flips"]
    + ["cov_eig_0", "cov_eig_1", "cov_eig_2"]
    + ["bbox_aspect_x", "bbox_aspect_y", "bbox_aspect_z"]
    + ["rigid_quat_w", "rigid_quat_x", "rigid_quat_y", "rigid_quat_z"]
    + ["flipped", "joint_angle_rad"]
)
FEATURE_DIM = len(FEATURE_NAMES)
assert FEATURE_DIM == 25, f"FEATURE_DIM bookkeeping: expected 25, got {FEATURE_DIM}"


def lattice_features(lattice) -> np.ndarray:
    """Return a deterministic ``(FEATURE_DIM,)`` float64 feature vector
    for the given :class:`auxetic.Lattice`.

    Same lattice → same vector (modulo float drift from the underlying
    geometry pipeline). Used for both training-set generation and
    model inference.
    """
    out = np.zeros(FEATURE_DIM, dtype=float)
    base = 0

    # Mode one-hot ----------------------------------------------------
    mode = int(getattr(lattice, "mode", 1))
    if 1 <= mode <= N_MODES:
        out[base + (mode - 1)] = 1.0
    base += N_MODES

    # Scalar lattice params -------------------------------------------
    out[base + 0] = float(getattr(lattice, "n_points", 0))
    out[base + 1] = float(getattr(lattice, "ratio", 0.0))
    out[base + 2] = float(getattr(lattice, "unit_scale_cm", 1.0))
    edge_flips = getattr(lattice, "edge_flips", set()) or set()
    out[base + 3] = float(len(edge_flips))
    base += 4

    # Point-cloud covariance eigenvalues (sorted descending) ---------
    pts = np.asarray(getattr(lattice, "points", None), dtype=float)
    if pts is not None and pts.ndim == 2 and pts.shape[0] >= 2:
        # Pad to 3D so the eigenvalue field is fixed-shape across modes.
        if pts.shape[1] == 2:
            pts3 = np.hstack([pts, np.zeros((pts.shape[0], 1))])
        else:
            pts3 = pts
        cov = np.cov(pts3.T)
        try:
            eigs = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            eigs = np.zeros(3)
        eigs = np.sort(eigs)[::-1]   # descending
        out[base:base + 3] = eigs[:3]
    base += 3

    # Bbox aspect ratios (normalise by the largest extent) -----------
    # NumPy 2.0 dropped ``ndarray.ptp()``; use the function form.
    if pts is not None and pts.ndim == 2 and pts.shape[0] >= 2:
        if pts.shape[1] == 2:
            extents = np.array(
                [np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 0.0])
        else:
            extents = np.array(
                [np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])])
        max_ext = float(extents.max())
        if max_ext > 1e-12:
            out[base:base + 3] = extents / max_ext
    base += 3

    # Rigid rotation quaternion (w, x, y, z) -------------------------
    rr = getattr(lattice, "rigid_rotation", None)
    if rr is not None:
        try:
            xyzw = rr.as_quat()  # scipy default ordering
            out[base + 0] = float(xyzw[3])  # w
            out[base + 1] = float(xyzw[0])  # x
            out[base + 2] = float(xyzw[1])  # y
            out[base + 3] = float(xyzw[2])  # z
        except Exception:
            out[base + 0] = 1.0  # identity quat
    base += 4

    # Flipped + joint angle -------------------------------------------
    out[base + 0] = 1.0 if bool(getattr(lattice, "flipped", False)) else 0.0
    out[base + 1] = float(getattr(lattice, "joint_angle", 0.0))
    base += 2

    assert base == FEATURE_DIM, (
        f"feature offset bookkeeping wrong: ended at {base}, expected {FEATURE_DIM}"
    )
    return out
