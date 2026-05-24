"""Training loop for :class:`auxetic_ml.model.PredictorMLP`.

Pure-PyTorch, no Lightning / accelerate / etc — keeps the dep
footprint to torch + the package itself. The loop fits comfortably
in a single function and is exercised end-to-end by tests on a
deterministic tiny dataset (the "overfit a few samples" sanity
check).

CLI: ``python -m auxetic_ml.train --data <dir> --epochs 100 --out
ckpt.pt`` — runs against a saved dataset directory (output of
:func:`auxetic_ml.dataset.save_samples`) and writes the final
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:   # pragma: no cover
    raise ImportError(
        "auxetic_ml.train requires PyTorch. "
        "Install with `pip install torch`."
    ) from exc

from .dataset import Sample, load_samples
from .model import PredictorMLP, samples_to_tensors


# ---------------------------------------------------------------------------
# Config + result records
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs:        int   = 100
    batch_size:    int   = 32
    learning_rate: float = 1.0e-3
    weight_decay:  float = 1.0e-5
    val_fraction:  float = 0.0       # 0 = use all data for training
    seed:          int   = 0
    log_every:     int   = 10        # print loss every N epochs


@dataclass
class TrainResult:
    """Per-epoch loss trace and the trained model. ``train_losses``
    and ``val_losses`` are lists of floats; ``val_losses`` is ``[]``
    when ``val_fraction == 0``."""
    model:          PredictorMLP
    train_losses:   List[float] = field(default_factory=list)
    val_losses:     List[float] = field(default_factory=list)
    final_train_loss: float     = 0.0
    final_val_loss:   float     = 0.0


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def _split(samples: List[Sample], val_fraction: float, seed: int
           ) -> tuple[List[Sample], List[Sample]]:
    """Deterministic train/val split."""
    n = len(samples)
    n_val = int(round(val_fraction * n))
    if n_val <= 0:
        return list(samples), []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = set(int(i) for i in perm[:n_val])
    train = [samples[i] for i in range(n) if i not in val_idx]
    val   = [samples[i] for i in range(n) if i     in val_idx]
    return train, val


def _make_loader(samples: List[Sample], batch_size: int,
                  *, shuffle: bool = True) -> DataLoader:
    batch = samples_to_tensors(samples)
    ds = TensorDataset(batch.inputs, batch.labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(samples: List[Sample],
          *,
          model:    Optional[PredictorMLP] = None,
          config:   Optional[TrainConfig]  = None,
          progress: Optional[Callable[[int, float, float], None]] = None,
          ) -> TrainResult:
    """Train a predictor on ``samples``.

    ``progress(epoch, train_loss, val_loss)`` is called at the end of
    every epoch (val_loss is NaN when no validation set). Used by the
    GUI training dialog to update a progress bar without coupling the
    model package to Qt.

    Returns a :class:`TrainResult` with the trained model and full
    per-epoch loss trace.
    """
    cfg = config or TrainConfig()
    if not samples:
        raise ValueError("train: at least one sample required")

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    train_samples, val_samples = _split(samples, cfg.val_fraction, cfg.seed)
    train_loader = _make_loader(train_samples, cfg.batch_size, shuffle=True)
    val_loader   = (_make_loader(val_samples, max(1, cfg.batch_size),
                                   shuffle=False)
                    if val_samples else None)

    model = model or PredictorMLP()
    optim = torch.optim.Adam(model.parameters(),
                              lr=cfg.learning_rate,
                              weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_losses: List[float] = []
    val_losses:   List[float] = []
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        for batch_inputs, batch_labels in train_loader:
            optim.zero_grad()
            preds = model(batch_inputs)
            loss = loss_fn(preds, batch_labels)
            loss.backward()
            optim.step()
            bs = int(batch_inputs.shape[0])
            epoch_loss += float(loss.item()) * bs
            n_seen += bs
        train_loss = epoch_loss / max(1, n_seen)
        train_losses.append(float(train_loss))

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, loss_fn)
            val_losses.append(float(val_loss))

        if progress is not None:
            progress(epoch, train_loss, val_loss)

    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        final_train_loss=train_losses[-1] if train_losses else 0.0,
        final_val_loss=val_losses[-1]   if val_losses   else float("nan"),
    )


@torch.no_grad()
def _eval_loss(model: PredictorMLP, loader: DataLoader,
                loss_fn: nn.Module) -> float:
    model.eval()
    total = 0.0
    n_seen = 0
    for inputs, labels in loader:
        preds = model(inputs)
        loss = loss_fn(preds, labels)
        bs = int(inputs.shape[0])
        total  += float(loss.item()) * bs
        n_seen += bs
    return total / max(1, n_seen)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:   # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(
        description="Train the auxetic_ml MLP predictor on a saved dataset.")
    p.add_argument("--data",       required=True,
                   help="Dataset directory (output of save_samples).")
    p.add_argument("--out",        default="predictor.pt",
                   help="Where to write the final checkpoint.")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val",        type=float, default=0.0,
                   help="Validation fraction in [0, 1).")
    p.add_argument("--seed",       type=int, default=0)
    args = p.parse_args()

    samples = load_samples(args.data)
    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, val_fraction=args.val, seed=args.seed,
    )

    def _progress(epoch: int, tl: float, vl: float) -> None:
        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            extra = f"  val_loss={vl:.6f}" if not np.isnan(vl) else ""
            print(f"epoch {epoch + 1:4d}/{cfg.epochs}  train_loss={tl:.6f}{extra}")

    result = train(samples, config=cfg, progress=_progress)
    from .model import save_checkpoint
    save_checkpoint(result.model, args.out)
    print(f"Saved trained predictor to {args.out}")


if __name__ == "__main__":   # pragma: no cover
    _cli()
