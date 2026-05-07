"""Tests for ``auxetic_ml.model`` and ``auxetic_ml.train``.

Skipped if PyTorch is not installed (the rest of the package keeps
working — features and dataset don't depend on torch).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from auxetic_ml.dataset import (
    Action,
    Sample,
    SampleLabel,
    generate_samples,
)
from auxetic_ml.features import FEATURE_DIM
from auxetic_ml.model import (
    ACTION_DIM,
    INPUT_DIM,
    PredictorMLP,
    Recommendation,
    encode_action,
    load_checkpoint,
    predict_best_action,
    samples_to_tensors,
    save_checkpoint,
    score_actions,
)
from auxetic_ml.train import TrainConfig, train


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

def test_action_dim_and_input_dim_are_consistent():
    assert ACTION_DIM == 10
    assert INPUT_DIM == FEATURE_DIM + ACTION_DIM


def test_encode_action_face_one_hot_layout():
    # GROUND_FACES = (None, +x, -x, +y, -y, +z, -z) — 7 entries.
    a = Action(ground_face=None, edge_flips=(), pre_rotation_rad=0.0)
    v = encode_action(a)
    assert v.shape == (ACTION_DIM,)
    # None is index 0 in the one-hot.
    assert v[0] == 1.0
    assert v[7] == 0.0   # n_flips
    # (sin 0, cos 0) at the end.
    assert v[8] == pytest.approx(0.0)
    assert v[9] == pytest.approx(1.0)


def test_encode_action_pre_rotation_uses_sin_cos():
    a = Action(ground_face=None, edge_flips=(),
                pre_rotation_rad=float(np.pi / 2))
    v = encode_action(a)
    assert v[8] == pytest.approx(1.0, abs=1e-6)
    assert v[9] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# PredictorMLP shape contract
# ---------------------------------------------------------------------------

def test_model_forward_shape():
    model = PredictorMLP()
    x = torch.zeros(4, INPUT_DIM)
    y = model(x)
    assert y.shape == (4, PredictorMLP.OUTPUT_DIM)


def test_model_forward_handles_unbatched_input():
    model = PredictorMLP()
    x = torch.zeros(INPUT_DIM)
    y = model(x)
    assert y.shape == (1, PredictorMLP.OUTPUT_DIM)


# ---------------------------------------------------------------------------
# samples_to_tensors
# ---------------------------------------------------------------------------

def test_samples_to_tensors_concatenates_lattice_and_action():
    s = Sample(
        features=np.zeros(FEATURE_DIM, dtype=float),
        action=Action(ground_face="+y", edge_flips=((0, 1),), pre_rotation_rad=0.5),
        label=SampleLabel(compression_efficiency=0.4, stability_score=0.8,
                           converged=True, final_kinetic_energy=0.1),
    )
    batch = samples_to_tensors([s, s])
    assert batch.inputs.shape == (2, INPUT_DIM)
    assert batch.labels.shape == (2, 2)
    assert batch.labels[0, 0].item() == pytest.approx(0.4)
    assert batch.labels[0, 1].item() == pytest.approx(0.8)


def test_samples_to_tensors_empty_list_returns_empty_tensors():
    batch = samples_to_tensors([])
    assert batch.inputs.shape  == (0, INPUT_DIM)
    assert batch.labels.shape  == (0, 2)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def test_predict_best_action_returns_recommendation():
    model = PredictorMLP()
    feats = np.zeros(FEATURE_DIM, dtype=float)
    actions = [
        Action(ground_face="+y", edge_flips=(),
               pre_rotation_rad=float(t))
        for t in (-1.0, 0.0, +1.0)
    ]
    rec = predict_best_action(model, feats, actions)
    assert isinstance(rec, Recommendation)
    assert rec.action in actions
    assert rec.confidence >= 0.0


def test_predict_best_action_empty_returns_none():
    model = PredictorMLP()
    feats = np.zeros(FEATURE_DIM, dtype=float)
    assert predict_best_action(model, feats, []) is None


def test_score_actions_shape():
    model = PredictorMLP()
    feats = np.zeros(FEATURE_DIM, dtype=float)
    actions = [
        Action(ground_face=None, edge_flips=(), pre_rotation_rad=0.0)
        for _ in range(5)
    ]
    scores = score_actions(model, feats, actions)
    assert scores.shape == (5, 2)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    """Save then load → identical eval-mode predictions. Dropout must
    be off for both passes (load_checkpoint installs eval mode by
    default; we mirror that on the original model so the comparison
    is apples-to-apples)."""
    model = PredictorMLP()
    model.eval()
    x = torch.randn(3, INPUT_DIM)
    with torch.no_grad():
        pred_before = model(x).cpu().numpy()
    save_checkpoint(model, str(tmp_path / "ckpt.pt"))
    loaded = load_checkpoint(str(tmp_path / "ckpt.pt"))
    with torch.no_grad():
        pred_after = loaded(x).cpu().numpy()
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-7)


# ---------------------------------------------------------------------------
# Training loop — smoke + overfit
# ---------------------------------------------------------------------------

def _synthetic_samples(n: int = 16, seed: int = 0) -> list[Sample]:
    """Build an n-sample synthetic dataset whose labels are a simple
    deterministic function of the inputs so a small MLP can fit it
    easily. Used by the overfit sanity check."""
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    for _ in range(n):
        feats = rng.standard_normal(FEATURE_DIM).astype(float)
        action = Action(
            ground_face=None, edge_flips=(),
            pre_rotation_rad=float(rng.uniform(-np.pi, np.pi)),
        )
        # Label is a deterministic function of feats[0] + action.pre_rotation
        comp = float(0.3 * feats[0] + 0.5 * np.sin(action.pre_rotation_rad))
        stab = float(0.5 + 0.2 * feats[1])
        samples.append(Sample(
            features=feats,
            action=action,
            label=SampleLabel(comp, stab, True, 0.0),
        ))
    return samples


def test_train_runs_one_epoch_without_error():
    samples = _synthetic_samples(n=8)
    cfg = TrainConfig(epochs=1, batch_size=4, learning_rate=1e-3, seed=0)
    result = train(samples, config=cfg)
    assert len(result.train_losses) == 1
    assert isinstance(result.model, PredictorMLP)


def test_train_overfits_tiny_synthetic_dataset():
    """Sanity: the MLP should drive train loss meaningfully down on
    a small deterministic dataset. Doesn't require crossing a strict
    threshold (that depends on init / hardware) — only that the final
    loss is < 50% of the first-epoch loss."""
    samples = _synthetic_samples(n=32, seed=0)
    cfg = TrainConfig(epochs=120, batch_size=8, learning_rate=3e-3, seed=0)
    result = train(samples, config=cfg)
    assert result.final_train_loss < 0.5 * result.train_losses[0], (
        f"loss didn't go down: first={result.train_losses[0]:.4f}, "
        f"last={result.final_train_loss:.4f}"
    )


def test_train_records_per_epoch_loss_trace():
    samples = _synthetic_samples(n=4)
    cfg = TrainConfig(epochs=5, batch_size=2, seed=0)
    result = train(samples, config=cfg)
    assert len(result.train_losses) == 5
    assert all(isinstance(v, float) for v in result.train_losses)


def test_train_with_validation_split_records_val_losses():
    samples = _synthetic_samples(n=16)
    cfg = TrainConfig(epochs=3, batch_size=4, val_fraction=0.25, seed=0)
    result = train(samples, config=cfg)
    assert len(result.val_losses)   == 3
    assert len(result.train_losses) == 3


def test_train_calls_progress_callback_per_epoch():
    samples = _synthetic_samples(n=4)
    cfg = TrainConfig(epochs=3, batch_size=2, seed=0)
    calls = []
    def cb(epoch, tl, vl):
        calls.append((epoch, tl, vl))
    train(samples, config=cfg, progress=cb)
    assert len(calls) == 3
    assert [c[0] for c in calls] == [0, 1, 2]


def test_train_with_no_samples_raises():
    with pytest.raises(ValueError, match="at least one sample"):
        train([])


# ---------------------------------------------------------------------------
# End-to-end: real (tiny) dataset → trained model → inference
# ---------------------------------------------------------------------------

def test_end_to_end_real_dataset_train_and_predict():
    """Generate a small real dataset via the M3 pipeline, train for
    a few epochs, predict on a held-out lattice, verify the model
    produces sensible numbers (no NaNs, no crashes)."""
    from auxetic import Lattice

    def make():
        return Lattice(mode=4, n_points=9, ratio=0.35, seed=42)

    samples = generate_samples(make, n_samples=6, seed=0, duration=0.02)
    cfg = TrainConfig(epochs=10, batch_size=3, seed=0)
    result = train(samples, config=cfg)
    # Score a candidate-action set against a fresh lattice.
    fresh_lattice = make()
    from auxetic_ml.features import lattice_features
    feats = lattice_features(fresh_lattice)
    candidates = [s.action for s in samples]
    rec = predict_best_action(result.model, feats, candidates)
    assert rec is not None
    assert np.isfinite(rec.predicted_compression)
    assert np.isfinite(rec.predicted_stability)
