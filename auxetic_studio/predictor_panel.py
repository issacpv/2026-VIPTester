"""Predictor dock — wraps :mod:`auxetic_ml` for in-app training and
recommendation (M3 final piece).

Layout:

    Model status: <text>
    [Train new model]   [Save model]   [Load model]
    Train config: n_samples / n_epochs (compact spinboxes)
    [Predict optimal action]
        Candidates: N
    Recommendation:
        face / flips / pre-rotation / predicted scores
    [Apply recommendation]

Training runs in a :class:`QThread` so the GUI stays responsive
even on multi-second training runs. The worker emits per-epoch
``progress`` signals that update the panel's progress bar; when
the run completes the trained model is handed back via the
``finished`` signal.

Apply-recommendation goes through a single
:class:`RecommendationApplyCommand` so undo restores the exact
prior (ground_face, edge_flips, joint_angle) triple in one click.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import (
    Qt,
    QObject,
    QThread,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QProgressBar,
    QFileDialog,
    QMessageBox,
)


# ---------------------------------------------------------------------------
# Background worker for training (keeps the GUI responsive)
# ---------------------------------------------------------------------------

class _TrainerWorker(QObject):
    """Runs sample generation + training in a background thread.

    Emits:
    - ``progress(epoch, train_loss)`` — once per epoch
    - ``finished(model)``             — when training completes
    - ``failed(message)``             — on any uncaught exception
    """
    progress = pyqtSignal(int, float)
    finished = pyqtSignal(object)
    failed   = pyqtSignal(str)

    def __init__(self, lattice, n_samples: int, n_epochs: int,
                  duration: float, parent: Optional[QObject] = None):
        super().__init__(parent)
        # Capture the lattice's geometry knobs at construction time so
        # the background thread can build fresh lattices without
        # touching the live one (QObject lattice isn't thread-safe).
        self._mode      = int(lattice.mode)
        self._n_points  = int(lattice.n_points)
        self._ratio     = float(lattice.ratio)
        self._nz_layers = int(lattice.nz_layers)
        self._n_samples = int(n_samples)
        self._n_epochs  = int(n_epochs)
        self._duration  = float(duration)

    def run(self) -> None:
        """Slot connected to ``QThread.started``."""
        try:
            from auxetic import Lattice
            from auxetic_ml.dataset import generate_samples
            from auxetic_ml.train import TrainConfig, train

            mode      = self._mode
            n_points  = self._n_points
            ratio     = self._ratio
            nz_layers = self._nz_layers

            def factory():
                return Lattice(
                    mode=mode, n_points=n_points,
                    ratio=ratio, nz_layers=nz_layers,
                    seed=42,
                )

            samples = generate_samples(
                factory, n_samples=self._n_samples, seed=0,
                duration=self._duration,
            )
            cfg = TrainConfig(epochs=self._n_epochs, batch_size=8, seed=0)
            result = train(
                samples, config=cfg,
                progress=lambda ep, tl, _vl: self.progress.emit(ep, float(tl)),
            )
            self.finished.emit(result.model)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{tb}")


# ---------------------------------------------------------------------------
# Predictor panel
# ---------------------------------------------------------------------------

class PredictorPanel(QDockWidget):
    """Right-side dock holding the train / load / predict / apply
    workflow."""

    # Emitted when the user clicks Apply on a recommendation. Carries
    # ``(ground_face, edge_flips_tuple, joint_angle_rad)`` — MainWindow
    # wraps in a :class:`RecommendationApplyCommand` so undo works.
    applyRecommendationRequested = pyqtSignal(object, object, float)

    def __init__(self, lattice, parent=None):
        super().__init__("Predictor", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._lattice = lattice
        self._model   = None     # trained / loaded PredictorMLP, or None
        self._recommendation = None
        self._train_thread = None
        self._train_worker = None

        body  = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(8, 8, 8, 8)

        self._build_status_row(outer)
        self._build_metrics_box(outer)
        self._build_train_box(outer)
        self._build_predict_box(outer)
        outer.addStretch(1)
        self.setWidget(body)
        self._refresh_state_dependent_ui()
        self.refresh_metrics()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_status_row(self, outer):
        self.status_label = QLabel("Model: <i>none loaded</i>", self)
        self.status_label.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(self.status_label)

    def _build_metrics_box(self, outer):
        """Read-only geometry metrics for the current lattice. Currently
        the edge-vector generalized Poisson's ratio (task 4) — a local
        per-triangle auxetic metric, averaged over the lattice, distinct
        from the simulator's bounding-box Poisson ratio."""
        box = QGroupBox("Geometry metrics", self)
        form = QFormLayout(box)
        self.edge_poisson_label = QLabel("—", box)
        self.edge_poisson_label.setTextFormat(Qt.TextFormat.RichText)
        self.edge_poisson_label.setToolTip(
            "Mean generalized Poisson's ratio from how each triangle's "
            "edge connection points deform under a small actuation of the "
            "rotating-units mechanism (uses the lattice's C). "
            "Equilateral tiles give -1 (isotropic auxetic).")
        form.addRow(QLabel("Edge-vector ν:"), self.edge_poisson_label)
        outer.addWidget(box)

    def _build_train_box(self, outer):
        box = QGroupBox("Train / load model", self)
        v = QVBoxLayout(box)

        # Compact row of spinboxes for the two key training knobs.
        cfg_form = QFormLayout()
        self.n_samples_spin = QSpinBox(box)
        self.n_samples_spin.setRange(2, 5000)
        self.n_samples_spin.setValue(40)
        self.n_samples_spin.setToolTip(
            "Number of (action, label) samples to generate from the "
            "current lattice family for training.")
        self.n_epochs_spin = QSpinBox(box)
        self.n_epochs_spin.setRange(1, 5000)
        self.n_epochs_spin.setValue(60)
        self.n_epochs_spin.setToolTip(
            "Adam optimizer passes over the dataset.")
        cfg_form.addRow(QLabel("N samples"), self.n_samples_spin)
        cfg_form.addRow(QLabel("N epochs"),  self.n_epochs_spin)
        v.addLayout(cfg_form)

        self.train_button = QPushButton("Train new model", box)
        self.train_button.clicked.connect(self._on_train_clicked)
        v.addWidget(self.train_button)

        self.train_progress = QProgressBar(box)
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setVisible(False)
        v.addWidget(self.train_progress)

        # Save / Load row
        sl_row = QWidget(box)
        sl_layout = QHBoxLayout(sl_row); sl_layout.setContentsMargins(0, 0, 0, 0)
        self.save_button = QPushButton("Save model…", sl_row)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.load_button = QPushButton("Load model…", sl_row)
        self.load_button.clicked.connect(self._on_load_clicked)
        sl_layout.addWidget(self.save_button)
        sl_layout.addWidget(self.load_button)
        v.addWidget(sl_row)

        outer.addWidget(box)

    def _build_predict_box(self, outer):
        box = QGroupBox("Predict optimal action", self)
        v = QVBoxLayout(box)

        cfg = QFormLayout()
        self.n_candidates_spin = QSpinBox(box)
        self.n_candidates_spin.setRange(1, 5000)
        self.n_candidates_spin.setValue(200)
        self.n_candidates_spin.setToolTip(
            "How many random (face, flips, pre-rotation) tuples to "
            "score before picking the best.")
        cfg.addRow(QLabel("Candidate actions"), self.n_candidates_spin)
        v.addLayout(cfg)

        self.predict_button = QPushButton("Predict optimal action", box)
        self.predict_button.clicked.connect(self._on_predict_clicked)
        v.addWidget(self.predict_button)

        self.recommendation_label = QLabel(
            "<i>Train or load a model, then click "
            "&laquo;Predict optimal action&raquo; to see a "
            "recommendation.</i>", box)
        self.recommendation_label.setTextFormat(Qt.TextFormat.RichText)
        self.recommendation_label.setWordWrap(True)
        v.addWidget(self.recommendation_label)

        self.apply_button = QPushButton("Apply recommendation", box)
        self.apply_button.clicked.connect(self._on_apply_clicked)
        v.addWidget(self.apply_button)

        outer.addWidget(box)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lattice(self, lattice) -> None:
        """Re-bind to a new Lattice (e.g. after File→Open). The
        loaded model is preserved across binds because models aren't
        lattice-specific (they consume features, not the lattice
        directly)."""
        self._lattice = lattice
        # Don't drop the model on lattice change — predictor is generic.
        self._refresh_state_dependent_ui()
        self.refresh_metrics()

    def refresh_metrics(self) -> None:
        """Recompute the read-only geometry metrics for the current
        lattice and update their labels. Called on construction, on
        lattice rebind, and from MainWindow after every lattice change."""
        import math
        try:
            nu = float(self._lattice.edge_vector_poisson_ratio())
        except Exception:
            nu = float("nan")
        if math.isnan(nu):
            self.edge_poisson_label.setText("<i>n/a (3D mode)</i>")
        else:
            self.edge_poisson_label.setText(f"<b>{nu:+.3f}</b>")

    @property
    def model(self):
        return self._model

    # ------------------------------------------------------------------
    # Train workflow
    # ------------------------------------------------------------------

    def _on_train_clicked(self) -> None:
        if self._train_thread is not None:
            QMessageBox.information(
                self, "Already training",
                "A training run is already in progress; please wait.")
            return
        n_samples = int(self.n_samples_spin.value())
        n_epochs  = int(self.n_epochs_spin.value())
        # Short per-sample sim duration so training stays under a minute.
        duration  = 0.05

        self.train_progress.setRange(0, max(1, n_epochs))
        self.train_progress.setValue(0)
        self.train_progress.setVisible(True)
        self._set_buttons_enabled(False)
        self.status_label.setText(
            f"<b>Training</b> ({n_samples} samples × {n_epochs} epochs)…")

        thread = QThread(self)
        worker = _TrainerWorker(
            self._lattice, n_samples, n_epochs, duration,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_train_progress)
        worker.finished.connect(self._on_train_finished)
        worker.failed.connect(self._on_train_failed)
        # Cleanup chain — when worker is done, quit the thread, then
        # delete both objects.
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_train_thread_finished)

        self._train_thread = thread
        self._train_worker = worker
        thread.start()

    def _on_train_progress(self, epoch: int, train_loss: float) -> None:
        self.train_progress.setValue(epoch + 1)
        self.status_label.setText(
            f"<b>Training</b> (epoch {epoch + 1}, loss={train_loss:.4f})")

    def _on_train_finished(self, model) -> None:
        self._model = model
        self.status_label.setText(
            "<b>Trained model loaded.</b> "
            "Click &laquo;Predict optimal action&raquo; to use it.")
        self._refresh_state_dependent_ui()

    def _on_train_failed(self, message: str) -> None:
        self._model = None
        self.status_label.setText(
            "<span style='color:#b00020'><b>Training failed.</b></span>")
        QMessageBox.critical(self, "Training failed", message)
        self._refresh_state_dependent_ui()

    def _on_train_thread_finished(self) -> None:
        # Thread + worker are scheduled for deletion via deleteLater
        # already; just clear our refs and re-enable UI.
        self._train_thread = None
        self._train_worker = None
        self.train_progress.setVisible(False)
        self._set_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _on_save_clicked(self) -> None:
        if self._model is None:
            QMessageBox.information(
                self, "No model", "Train or load a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save predictor model", "predictor.pt",
            "PyTorch checkpoint (*.pt);;All Files (*)")
        if not path:
            return
        try:
            from auxetic_ml.model import save_checkpoint
            save_checkpoint(self._model, str(path))
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed", f"{type(exc).__name__}: {exc}")
            return
        self.status_label.setText(
            f"<b>Saved model</b> to {Path(path).name}")

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load predictor model", "",
            "PyTorch checkpoint (*.pt);;All Files (*)")
        if not path:
            return
        try:
            from auxetic_ml.model import load_checkpoint
            self._model = load_checkpoint(str(path))
        except Exception as exc:
            QMessageBox.critical(
                self, "Load failed", f"{type(exc).__name__}: {exc}")
            return
        self.status_label.setText(
            f"<b>Loaded model</b> from {Path(path).name}")
        self._refresh_state_dependent_ui()

    # ------------------------------------------------------------------
    # Predict / Apply
    # ------------------------------------------------------------------

    def _candidate_actions(self, n: int) -> List:
        """Sample ``n`` random actions appropriate for the current
        lattice's dimensionality."""
        import numpy as np
        from auxetic.geometry import flippable_edges
        from auxetic_ml.dataset import sample_action

        edges = flippable_edges(self._lattice.tri, self._lattice.points)
        dim = 3 if int(self._lattice.mode) in (3, 6, 9) else 2
        rng = np.random.default_rng(0)
        return [sample_action(rng, edges, dim=dim) for _ in range(int(n))]

    def _on_predict_clicked(self) -> None:
        if self._model is None:
            QMessageBox.information(
                self, "No model", "Train or load a model first.")
            return
        try:
            from auxetic_ml.features import lattice_features
            from auxetic_ml.model import predict_best_action

            feats = lattice_features(self._lattice)
            candidates = self._candidate_actions(self.n_candidates_spin.value())
            rec = predict_best_action(self._model, feats, candidates)
        except Exception as exc:
            QMessageBox.critical(
                self, "Prediction failed",
                f"{type(exc).__name__}: {exc}")
            return
        if rec is None:
            self.recommendation_label.setText(
                "<i>No candidate actions to score.</i>")
            self._recommendation = None
            self._refresh_state_dependent_ui()
            return
        self._recommendation = rec
        self._render_recommendation(rec)
        self._refresh_state_dependent_ui()

    def _render_recommendation(self, rec) -> None:
        flips_str = ", ".join(f"{a}-{b}" for a, b in rec.action.edge_flips) or "—"
        face_str  = rec.action.ground_face if rec.action.ground_face else "none"
        import math as _math
        self.recommendation_label.setText(
            "<table style='border-spacing:0;'>"
            f"<tr><td><b>Best face:</b></td><td>{face_str}</td></tr>"
            f"<tr><td><b>Best flips:</b></td><td>{flips_str}</td></tr>"
            f"<tr><td><b>Pre-rotation:</b></td>"
            f"<td>{_math.degrees(rec.action.pre_rotation_rad):+.1f}°</td></tr>"
            f"<tr><td><b>Pred. compression:</b></td>"
            f"<td>{rec.predicted_compression:+.4f}</td></tr>"
            f"<tr><td><b>Pred. stability:</b></td>"
            f"<td>{rec.predicted_stability:.4f}</td></tr>"
            f"<tr><td><b>Confidence:</b></td>"
            f"<td>{rec.confidence:.4f}</td></tr>"
            "</table>"
        )

    def _on_apply_clicked(self) -> None:
        if self._recommendation is None:
            QMessageBox.information(
                self, "No recommendation",
                "Click &laquo;Predict optimal action&raquo; first.")
            return
        rec = self._recommendation
        self.applyRecommendationRequested.emit(
            rec.action.ground_face,
            tuple(rec.action.edge_flips),
            float(rec.action.pre_rotation_rad),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for b in (self.train_button, self.save_button, self.load_button,
                  self.predict_button, self.apply_button):
            b.setEnabled(enabled)
        # Re-apply finer-grained gating below.
        self._refresh_state_dependent_ui()

    def _refresh_state_dependent_ui(self) -> None:
        has_model = self._model is not None
        has_rec   = self._recommendation is not None
        # Save / predict only meaningful with a model.
        self.save_button.setEnabled(has_model)
        self.predict_button.setEnabled(has_model)
        self.apply_button.setEnabled(has_rec)

    def shutdown(self) -> None:
        """Stop any running trainer thread cleanly. Called by
        ``MainWindow.closeEvent`` to avoid leaving a worker thread
        spinning past app exit (mirrors the kinematic
        SimulationPanel.shutdown pattern)."""
        thread = self._train_thread
        if thread is not None:
            try:
                thread.quit()
                thread.wait(2000)
            except Exception:
                pass
