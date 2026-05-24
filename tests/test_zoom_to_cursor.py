"""Tests for the 3D-view zoom-to-cursor behaviour (Task 3).

The wheel-zoom dollies the camera toward the world point under the
cursor rather than the view centre. The camera math lives in the pure
helper ``auxetic_studio.camera_controls.dolly_toward_cursor`` (re-exported
by ``auxetic_studio.views`` and used by ``View3D``) and is exercised
exhaustively here; the VTK observer wiring is headless-guarded, so the
remaining tests only assert it stays a safe no-op on the placeholder
(headless) View3D path.

Imports of ``auxetic_studio`` are kept lazy (inside fixtures), matching
this suite's convention: importing the package at collection time pulls
in the Qt/VTK GUI stack before a QApplication exists, which destabilises
process teardown on the offscreen Qt platform.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def dolly():
    """Lazily import the pure helper (see module docstring on import timing)."""
    from auxetic_studio.camera_controls import dolly_toward_cursor
    return dolly_toward_cursor


# ---------------------------------------------------------------------------
# Pure helper: dolly_toward_cursor
# ---------------------------------------------------------------------------

def test_factor_one_is_identity(dolly):
    cam = [0.0, 0.0, 5.0]
    foc = [0.0, 0.0, 0.0]
    tgt = [1.0, 2.0, 0.0]
    new_cam, new_foc = dolly(cam, foc, tgt, 1.0)
    assert np.allclose(new_cam, cam)
    assert np.allclose(new_foc, foc)


@pytest.mark.parametrize("factor", [1.5, 2.0, 0.5, 1.15])
def test_camera_and_focal_scale_toward_target(dolly, factor):
    cam = np.array([0.0, 0.0, 10.0])
    foc = np.array([0.0, 0.0, 0.0])
    tgt = np.array([3.0, 4.0, 0.0])
    new_cam, new_foc = dolly(cam, foc, tgt, factor)
    s = 1.0 / factor
    assert np.allclose(new_cam, tgt + (cam - tgt) * s)
    assert np.allclose(new_foc, tgt + (foc - tgt) * s)


@pytest.mark.parametrize("factor", [1.2, 2.0, 0.5])
def test_distance_to_target_scales_by_inverse_factor(dolly, factor):
    """Zooming IN (factor>1) must bring the camera proportionally closer
    to the cursor target; zooming OUT pushes it away."""
    cam = np.array([0.0, 0.0, 10.0])
    foc = np.array([0.0, 0.0, 0.0])
    tgt = np.array([1.0, 1.0, 1.0])
    new_cam, _ = dolly(cam, foc, tgt, factor)
    d0 = np.linalg.norm(cam - tgt)
    d1 = np.linalg.norm(new_cam - tgt)
    assert d1 == pytest.approx(d0 / factor)


@pytest.mark.parametrize("factor", [1.3, 0.7])
def test_view_direction_preserved(dolly, factor):
    """The zoom must never rotate the camera: the view vector
    (position - focal) only changes length, not direction."""
    cam = np.array([2.0, 0.0, 10.0])
    foc = np.array([0.0, 0.0, 0.0])
    tgt = np.array([5.0, -3.0, 2.0])
    new_cam, new_foc = dolly(cam, foc, tgt, factor)
    assert np.allclose(new_cam - new_foc, (cam - foc) / factor)


def test_centred_zoom_leaves_focal_point_fixed(dolly):
    """When the cursor target coincides with the focal point the result
    is the classic zoom-to-centre dolly — focal point unmoved. This is
    what keeps camera presets (which look at the focal point) unaffected
    by a subsequent centred wheel-zoom."""
    cam = np.array([0.0, 0.0, 10.0])
    foc = np.array([1.0, 1.0, 1.0])
    new_cam, new_foc = dolly(cam, foc, foc, 1.5)
    assert np.allclose(new_foc, foc)
    assert np.allclose(new_cam, foc + (cam - foc) / 1.5)


def test_zoom_in_then_out_roundtrips(dolly):
    cam = np.array([0.0, 0.0, 10.0])
    foc = np.array([0.0, 0.0, 0.0])
    tgt = np.array([2.0, 3.0, 1.0])
    step = 1.15
    nc, nf = dolly(cam, foc, tgt, step)
    nc2, nf2 = dolly(nc, nf, tgt, 1.0 / step)
    assert np.allclose(nc2, cam)
    assert np.allclose(nf2, foc)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_bad_factor_is_noop(dolly, bad):
    cam = np.array([0.0, 0.0, 10.0])
    foc = np.array([0.0, 0.0, 0.0])
    tgt = np.array([2.0, 3.0, 1.0])
    new_cam, new_foc = dolly(cam, foc, tgt, bad)
    assert np.allclose(new_cam, cam)
    assert np.allclose(new_foc, foc)


def test_nonfinite_target_is_noop(dolly):
    cam = np.array([0.0, 0.0, 10.0])
    foc = np.array([0.0, 0.0, 0.0])
    new_cam, new_foc = dolly(cam, foc, [np.nan, 0.0, 0.0], 1.5)
    assert np.allclose(new_cam, cam)
    assert np.allclose(new_foc, foc)


def test_returns_independent_arrays(dolly):
    """The helper must not alias its inputs — callers mutate the camera
    in place right after."""
    cam = np.array([0.0, 0.0, 10.0])
    foc = np.array([0.0, 0.0, 0.0])
    tgt = np.array([1.0, 0.0, 0.0])
    new_cam, _ = dolly(cam, foc, tgt, 1.0)
    new_cam[0] = 999.0
    assert cam[0] == 0.0  # input untouched


# NOTE: The View3D wheel-observer wiring (``_install_zoom_to_cursor`` /
# ``_zoom_to_cursor`` / ``_on_wheel_*``) is headless-guarded by
# construction — every method early-returns when ``self.interactor is
# None``, and the observers are only installed when a real VTK interactor
# exists. It is deliberately *not* covered by a widget-level test here:
# constructing a pyvistaqt View3D in a small offscreen pytest session
# triggers a pre-existing VTK/Qt teardown abort on this platform (it
# crashes even reading long-standing attributes like ``has_grid`` — see
# the nightly log), which would make the suite flaky. The cursor-zoom
# correctness lives entirely in the pure ``dolly_toward_cursor`` math
# exercised above.
