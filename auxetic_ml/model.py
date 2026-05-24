"""PyTorch MLP predictor for the M3 milestone.

Architecture
------------
Input:  ``(lattice_features ⊕ action_features)`` — the 25-dim
        handcrafted lattice vector concatenated with a small
        action-features block.
Trunk:  ``Linear → GELU → Dropout → Linear → GELU → Linear → 2``.
Output: dual regression heads (compression_efficiency,
        stability_score). Both are unbounded floats; the training
        loss is plain MSE.

Why MLP first
-------------
The plan in :file:`docs/.../i-have-this-code-crystalline-knuth.md`
mentions a GNN as the "more expressive" option, but a plain MLP on
the handcrafted feature vector is the right baseline:

- Pipeline / training loop / inference path can be exercised end-to-
  end without graph-batching machinery.
- Sets a quantitative floor for any GNN to beat.
- Smaller code surface: easier to debug pose / scaling issues.

The GNN can drop in later as a sibling :class:`torch.nn.Module` —
the dataset, training loop, and predictor panel are model-agnostic.

Action features
---------------
:func:`encode_action` builds a fixed-size action vector:

- Ground face one-hot (7) — ``[None, +x, -x, +y, -y, +z, -z]``.
- Number of edge flips (1) — scalar count, since the predictor
  consumes lattice features that already record this.
- Pre-rotation as ``(sin θ, cos θ)`` (2) — keeps the rotation
  representation continuous across the ±π wrap-around.

Total action dim: ``ACTION_DIM = 10``. Combined with the 25-dim
lattice vector, the model input is ``35``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover — torch is a hard M3 dep
    raise ImportError(
        "auxetic_ml.model requires PyTorch. "
        "Install with `pip install torch` (CPU-only is fine)."
    ) from exc

from .features import FEATURE_DIM
from .dataset import Action, GROUND_FACES, Sample


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

ACTION_DIM = 7 + 1 + 2   # face one-hot + n_flips + (sin, cos) pre-rotation


def encode_action(action: Action) -> np.ndarray:
    """Convert an :class:`Action` to a fixed-size feature vector."""
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    # Face one-hot — index by GROUND_FACES order so it round-trips
    # with the dataset module.
    try:
        face_idx = GROUND_FACES.index(action.ground_face)
    except ValueError:
        face_idx = 0
    out[face_idx] = 1.0
    # n_flips
    out[7] = float(len(action.edge_flips))
    # Pre-rotation as (sin, cos)
    theta = float(action.pre_rotation_rad)
    out[8] = math.sin(theta)
    out[9] = math.cos(theta)
    return out


def encode_actions(actions: Iterable[Action]) -> np.ndarray:
    """Stack :func:`encode_action` over an iterable into ``(N, ACTION_DIM)``."""
    arr = np.stack([encode_action(a) for a in actions], axis=0)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

INPUT_DIM = FEATURE_DIM + ACTION_DIM   # 25 + 10 = 35


class PredictorMLP(nn.Module):
    """Two-headed MLP predicting ``(compression_efficiency,
    stability_score)`` from a ``(lattice ⊕ action)`` feature vector.

    Trunk: Linear(35→128) → GELU → Dropout(0.1) → Linear(128→128)
           → GELU → Linear(128→2). Total: ~21k params.
    """

    OUTPUT_DIM = 2   # compression + stability

    def __init__(self,
                 input_dim:  int = INPUT_DIM,
                 hidden_dim: int = 128,
                 dropout:    float = 0.1):
        super().__init__()
        self.input_dim  = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape: ``(batch, INPUT_DIM)``. Returns ``(batch, 2)``."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.trunk(x)


# ---------------------------------------------------------------------------
# Sample → tensors helper
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    """Collated tensors for one minibatch."""
    inputs: torch.Tensor   # (N, INPUT_DIM)
    labels: torch.Tensor   # (N, 2)


def samples_to_tensors(samples: List[Sample]) -> Batch:
    """Concatenate lattice + action features and stack labels for a
    list of :class:`Sample` records. Returns a :class:`Batch`."""
    if not samples:
        return Batch(
            inputs=torch.zeros((0, INPUT_DIM), dtype=torch.float32),
            labels=torch.zeros((0, PredictorMLP.OUTPUT_DIM),
                                dtype=torch.float32),
        )
    lattice_feats = np.stack(
        [np.asarray(s.features, dtype=np.float32) for s in samples], axis=0,
    )
    action_feats = encode_actions(s.action for s in samples)
    inputs_np = np.concatenate([lattice_feats, action_feats], axis=1)
    labels_np = np.stack(
        [
            np.array(
                [s.label.compression_efficiency, s.label.stability_score],
                dtype=np.float32,
            )
            for s in samples
        ],
        axis=0,
    )
    return Batch(
        inputs=torch.from_numpy(inputs_np),
        labels=torch.from_numpy(labels_np),
    )


# ---------------------------------------------------------------------------
# Inference: pick the action that maximises predicted compression
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """Result of :func:`predict_best_action`."""
    action:                Action
    predicted_compression: float
    predicted_stability:   float
    confidence:            float    # top-1 minus runner-up compression score


@torch.no_grad()
def score_actions(model: PredictorMLP,
                  lattice_feats: np.ndarray,
                  actions: List[Action],
                  ) -> np.ndarray:
    """Score every action in ``actions`` against the same lattice.
    Returns ``(N, 2)`` numpy array of ``(compression, stability)``
    predictions."""
    if len(actions) == 0:
        return np.zeros((0, PredictorMLP.OUTPUT_DIM), dtype=np.float32)
    lattice_t = np.broadcast_to(lattice_feats, (len(actions), FEATURE_DIM))
    inputs = np.concatenate([lattice_t, encode_actions(actions)], axis=1)
    x = torch.from_numpy(inputs.astype(np.float32))
    model.eval()
    out = model(x).cpu().numpy()
    return out


def predict_best_action(model: PredictorMLP,
                         lattice_feats: np.ndarray,
                         candidate_actions: List[Action],
                         ) -> Optional[Recommendation]:
    """Run ``score_actions`` and return the top-1 action.

    The objective optimised at inference is the ``compression_efficiency``
    head — the predictor's primary metric. ``confidence`` is the score
    margin between the top-1 and top-2 action.
    """
    if not candidate_actions:
        return None
    scores = score_actions(model, lattice_feats, candidate_actions)
    comp = scores[:, 0]
    order = np.argsort(comp)[::-1]
    top = int(order[0])
    margin = (
        float(comp[order[0]] - comp[order[1]]) if len(order) >= 2 else 0.0
    )
    return Recommendation(
        action=candidate_actions[top],
        predicted_compression=float(scores[top, 0]),
        predicted_stability=float(scores[top, 1]),
        confidence=float(margin),
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model: PredictorMLP, path: str) -> None:
    """Save model weights + architecture metadata to ``path`` (.pt)."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim":  model.input_dim,
            "hidden_dim": model.hidden_dim,
            "feature_dim": FEATURE_DIM,
            "action_dim":  ACTION_DIM,
        },
        path,
    )


def load_checkpoint(path: str) -> PredictorMLP:
    """Inverse of :func:`save_checkpoint`. Reconstructs the model with
    the same architecture and loads the weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PredictorMLP(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
