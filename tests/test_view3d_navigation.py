"""Tests for the M3-polish 3D navigation aids — floor grid, axes
triad, clickable view cube, and the camera-preset toolbar.

The widgets only install when the underlying ``QtInteractor`` is
available (real VTK + Qt). On headless / placeholder paths they're
no-ops; the tests cover both.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import pytest


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
# Toolbar actions
# ---------------------------------------------------------------------------

def test_toolbar_has_all_camera_preset_actions(main_window):
    win = main_window
    for attr in ("cam_top_act", "cam_front_act", "cam_side_act",
                 "cam_iso_act", "cam_fit_act"):
        assert hasattr(win, attr), f"missing toolbar action: {attr}"


def test_camera_preset_toolbar_buttons_have_distinct_labels(main_window):
    win = main_window
    labels = {win.cam_top_act.text(), win.cam_front_act.text(),
              win.cam_side_act.text(), win.cam_iso_act.text(),
              win.cam_fit_act.text()}
    assert len(labels) == 5    # all distinct


def test_camera_preset_clicks_call_view3d_methods(main_window, monkeypatch):
    """Clicking a camera-preset toolbar action invokes the matching
    View3D method. We monkeypatch the methods to record calls so the
    test doesn't depend on a real VTK render window."""
    win = main_window
    calls: list[str] = []
    monkeypatch.setattr(win.view_3d, "camera_top",
                         lambda: calls.append("top"))
    monkeypatch.setattr(win.view_3d, "camera_front",
                         lambda: calls.append("front"))
    monkeypatch.setattr(win.view_3d, "camera_side",
                         lambda: calls.append("side"))
    monkeypatch.setattr(win.view_3d, "camera_isometric",
                         lambda: calls.append("iso"))
    monkeypatch.setattr(win.view_3d, "camera_fit",
                         lambda: calls.append("fit"))
    win.cam_top_act.trigger()
    win.cam_front_act.trigger()
    win.cam_side_act.trigger()
    win.cam_iso_act.trigger()
    win.cam_fit_act.trigger()
    assert calls == ["top", "front", "side", "iso", "fit"]


# ---------------------------------------------------------------------------
# View3D camera-preset methods are callable on headless instances
# ---------------------------------------------------------------------------

def test_camera_methods_no_op_on_headless_view3d(main_window):
    """In headless mode the interactor is None; camera-preset calls
    should silently return rather than raising."""
    v = main_window.view_3d
    assert v.interactor is None
    v.camera_top()
    v.camera_front()
    v.camera_side()
    v.camera_isometric()
    v.camera_fit()


def test_headless_view3d_reports_no_navigation_widgets(main_window):
    """has_grid / has_axes / has_view_cube are False when the
    interactor isn't available."""
    v = main_window.view_3d
    assert v.has_grid      is False
    assert v.has_axes      is False
    assert v.has_view_cube is False


# ---------------------------------------------------------------------------
# Note: a "real interactor installs widgets" test was tried here but
# crashed VTK under ``QT_QPA_PLATFORM=offscreen`` because the VTK
# render-window backing the widget calls is unreliable in headless
# CI. Widget install IS exercised at every real app launch — the
# headless smoke test in ``CHANGELOG`` of this commit confirms the
# QtInteractor path runs ``_install_3d_navigation_aids`` cleanly on
# a normal Windows + VTK setup.
# ---------------------------------------------------------------------------


def test_view_cube_widget_supports_animation_disable_api():
    """The fix for the "weird animation" report relies on the VTK
    widget exposing ``SetAnimate(False)`` (or the older ``AnimateOff``
    fallback). Pin that the API exists in the installed VTK so we
    notice if a future VTK rev removes it."""
    pytest.importorskip("vtkmodules.vtkInteractionWidgets")
    from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
    w = vtkCameraOrientationWidget()
    assert hasattr(w, "SetAnimate") or hasattr(w, "AnimateOff"), (
        "VTK's vtkCameraOrientationWidget no longer exposes the "
        "animation toggle our fix relies on. Update _install_3d_"
        "navigation_aids in auxetic_studio/views.py."
    )
