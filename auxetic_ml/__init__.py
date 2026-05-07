"""Machine-learning helpers for the auxetic project.

Per :file:`CLAUDE.md`, the geometry packages (``auxetic/`` and the
``auxetic/dynamics.py`` Newtonian solver) are numpy/scipy-only — no
torch, no native physics deps. ``auxetic_ml/`` is the carve-out where
PyTorch and other ML libraries are allowed: model architectures,
training loops, datasets, and inference helpers all live here.

Current contents (M3.1 — pipeline scaffold, no PyTorch yet):

- :mod:`auxetic_ml.features` — handcrafted lattice → fixed-size
  feature vector. Used by the MLP/feature-based predictor head; the
  GNN head will use a separate per-node / per-edge graph extractor.
- :mod:`auxetic_ml.dataset` — sample generation pipeline. Sweeps a
  lattice family through random ``(face, edge-flip pattern,
  pre-rotation)`` action triples, runs each through the M2 dynamic
  simulator, and records the result. Saves and loads from a
  directory of ``.npz`` files plus a ``manifest.json`` (HDF5 was
  considered; ``.npz``-per-sample is simpler for the small datasets
  we'll start with and avoids the h5py dependency).

The PyTorch model and training loop land in subsequent commits as
``auxetic_ml/model.py`` and ``auxetic_ml/train.py``.
"""

from __future__ import annotations
