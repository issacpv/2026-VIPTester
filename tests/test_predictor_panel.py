"""GUI tests for ``auxetic_studio.predictor_panel.PredictorPanel`` and
the M3 ``RecommendationApplyCommand`` it pushes to the undo stack.

Skipped if PyTorch isn't installed.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def main_window():
    from PyQt6.QtWidgets import QApplication
    from auxetic_studio.main_window import MainWindow
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(headless_3d=True)
    yield win
    try:
        win.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Construction + initial state
# ---------------------------------------------------------------------------

def test_panel_attached_to_main_window(main_window):
    assert hasattr(main_window, "predictor_panel")
    p = main_window.predictor_panel
    assert p.train_button.text() == "Train new model"


def test_predict_disabled_when_no_model_loaded(main_window):
    p = main_window.predictor_panel
    assert p.model is None
    assert p.predict_button.isEnabled() is False
    assert p.apply_button.isEnabled()   is False


def test_save_disabled_when_no_model_loaded(main_window):
    p = main_window.predictor_panel
    assert p.save_button.isEnabled() is False


# ---------------------------------------------------------------------------
# Edge-vector Poisson metric (task 4)
# ---------------------------------------------------------------------------

def test_panel_shows_edge_poisson_metric(main_window):
    p = main_window.predictor_panel
    assert hasattr(p, "edge_poisson_label")
    # Default lattice is 2D mode 1 -> a finite metric is displayed.
    expected = f"{main_window.lattice.edge_vector_poisson_ratio():+.3f}"
    assert expected in p.edge_poisson_label.text()


def test_edge_poisson_metric_matches_lattice_method(main_window):
    import math
    p = main_window.predictor_panel
    p.refresh_metrics()
    val = main_window.lattice.edge_vector_poisson_ratio()
    assert math.isfinite(val)
    assert f"{val:+.3f}" in p.edge_poisson_label.text()


# ---------------------------------------------------------------------------
# Inject a tiny trained model and exercise predict + apply
# ---------------------------------------------------------------------------

def _inject_tiny_model(panel):
    """Train a 5-epoch synthetic model and stash it on the panel.
    Avoids running the panel's own training thread (which would
    require the QThread event loop to spin to completion in tests)."""
    import numpy as np
    from auxetic_ml.dataset import Action, Sample, SampleLabel
    from auxetic_ml.features import FEATURE_DIM
    from auxetic_ml.train import TrainConfig, train

    rng = np.random.default_rng(0)
    samples = []
    for _ in range(8):
        feats = rng.standard_normal(FEATURE_DIM).astype(float)
        action = Action(ground_face=None, edge_flips=(),
                        pre_rotation_rad=0.0)
        samples.append(Sample(
            features=feats,
            action=action,
            label=SampleLabel(0.1, 0.9, True, 0.0),
        ))
    cfg = TrainConfig(epochs=5, batch_size=4, seed=0)
    result = train(samples, config=cfg)
    panel._model = result.model
    panel._refresh_state_dependent_ui()
    return result.model


def test_predict_button_enabled_after_model_injected(main_window):
    p = main_window.predictor_panel
    assert p.predict_button.isEnabled() is False
    _inject_tiny_model(p)
    assert p.predict_button.isEnabled() is True
    assert p.save_button.isEnabled()    is True


def test_predict_clicked_populates_recommendation(main_window):
    p = main_window.predictor_panel
    _inject_tiny_model(p)
    # Reduce candidate count for test speed.
    p.n_candidates_spin.setValue(20)
    p.predict_button.click()
    assert p._recommendation is not None
    assert p.apply_button.isEnabled() is True


def test_predict_text_includes_face_and_compression_fields(main_window):
    p = main_window.predictor_panel
    _inject_tiny_model(p)
    p.n_candidates_spin.setValue(20)
    p.predict_button.click()
    text = p.recommendation_label.text()
    assert "Best face" in text
    assert "Pred. compression" in text
    assert "Confidence" in text


# ---------------------------------------------------------------------------
# Apply recommendation goes through the undo stack
# ---------------------------------------------------------------------------

def test_apply_pushes_one_undoable_command(main_window):
    p = main_window.predictor_panel
    _inject_tiny_model(p)
    p.n_candidates_spin.setValue(20)
    p.predict_button.click()
    initial = main_window.undo_stack.count()
    p.apply_button.click()
    assert main_window.undo_stack.count() == initial + 1


def test_apply_then_undo_restores_original_state(main_window):
    p = main_window.predictor_panel
    lat = main_window.lattice
    # Snapshot the pre-apply state of the three fields the command
    # touches.
    old_face  = lat.dynamics_state.get("ground_face")
    old_flips = set(lat.edge_flips)
    old_theta = float(lat.joint_angle)

    _inject_tiny_model(p)
    p.n_candidates_spin.setValue(20)
    p.predict_button.click()
    p.apply_button.click()

    main_window.undo_stack.undo()

    assert lat.dynamics_state.get("ground_face") == old_face
    assert set(lat.edge_flips) == old_flips
    assert float(lat.joint_angle) == pytest.approx(old_theta)


def test_apply_with_no_recommendation_does_nothing(main_window):
    p = main_window.predictor_panel
    initial = main_window.undo_stack.count()
    p.apply_button.click()
    assert main_window.undo_stack.count() == initial


# ---------------------------------------------------------------------------
# Save / load round-trip via the panel buttons (using monkeypatched dialogs)
# ---------------------------------------------------------------------------

def test_save_then_load_round_trip(main_window, tmp_path, monkeypatch):
    p = main_window.predictor_panel
    _inject_tiny_model(p)
    ckpt_path = str(tmp_path / "panel_ckpt.pt")

    # Stub the QFileDialog calls so tests don't pop a real dialog.
    from PyQt6.QtWidgets import QFileDialog
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (ckpt_path, "")))
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (ckpt_path, "")))

    p._on_save_clicked()
    assert Path(ckpt_path).is_file()

    # Drop the model and reload from disk via the panel's own load path.
    p._model = None
    p._refresh_state_dependent_ui()
    assert p.predict_button.isEnabled() is False
    p._on_load_clicked()
    assert p.model is not None
    assert p.predict_button.isEnabled() is True


# ---------------------------------------------------------------------------
# Re-binding the panel to a new lattice keeps the model
# ---------------------------------------------------------------------------

def test_set_lattice_preserves_model(main_window):
    """Models are not lattice-specific — they consume features. After
    File→Open of a different lattice the user shouldn't have to
    retrain."""
    p = main_window.predictor_panel
    _inject_tiny_model(p)
    from auxetic import Lattice
    new_lat = Lattice(mode=4, n_points=9, ratio=0.35, seed=42)
    p.set_lattice(new_lat)
    assert p.model is not None
    assert p.predict_button.isEnabled() is True
