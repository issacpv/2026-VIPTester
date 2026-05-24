"""Stage 5 rotation-model tests.

Maps to the prompt:

1.  ``test_rigid_rotation_does_not_modify_points``
2.  ``test_world_transform_combines_flip_and_rotation``
3.  ``test_world_transform_rotates_around_centroid``
4.  ``test_joint_angle_storage_in_radians``
5.  ``test_rotation_change_command_undoable``
6.  ``test_flip_command_undoable``
7.  ``test_v1_preset_loads_with_default_rotation``
8.  ``test_v2_preset_round_trips_rotation``
9.  ``test_rejects_v3_preset``
10. ``test_stl_export_differs_with_rotation``
11. ``test_kirigami_export_applies_rotation``
12. ``test_preset_orientation_buttons``
13. ``test_2d_view_inverse_transform_for_edit``

Per SPEC §6.3 the three rotation concepts (rigid rotation, flip, joint
angle) are kept in distinct fields and exercised independently here.
"""

import json
import math
import os
import sys

# Force headless Qt before importing anything that touches QtGui/QtWidgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt6.QtWidgets import QApplication

from auxetic import Lattice
from auxetic_studio.commands import (
    RotationChangeCommand,
    FlipCommand,
    JointAngleChangeCommand,
)
from auxetic_studio.preset import (
    save_preset,
    load_preset,
    PRESET_VERSION,
)


# Binary STL header (numpy-stl writes timestamp + filename here).
_STL_HEADER_BYTES = 80


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def main_window(qapp):
    from auxetic_studio import MainWindow
    win = MainWindow(headless_3d=True)
    yield win
    win.close()


def _quat_close(a: Rotation, b: Rotation, tol: float = 1e-9) -> bool:
    qa = a.as_quat()
    qb = b.as_quat()
    return bool(np.linalg.norm(qa - qb) < tol or np.linalg.norm(qa + qb) < tol)


# ---------------------------------------------------------------------------
# 1. Rigid rotation does not mutate canonical points (SPEC §6.1)
# ---------------------------------------------------------------------------

def test_rigid_rotation_does_not_modify_points():
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    canonical = L.points.copy()

    L.rigid_rotation = Rotation.from_euler("z", 47, degrees=True)
    np.testing.assert_array_equal(L.points, canonical)

    # Even after computing the world transform / transformed_points
    # the canonical array is untouched.
    _ = L.world_transform()
    _ = L.transformed_points()
    np.testing.assert_array_equal(L.points, canonical)


# ---------------------------------------------------------------------------
# 2. world_transform combines flip + rotation
# ---------------------------------------------------------------------------

def test_world_transform_combines_flip_and_rotation():
    """Hand-computed: pivot is centroid (0.5, 0.5, 0.5).
    Take p = (0.7, 0.6, 0.4):
      p - c   = (0.2, 0.1, -0.1)
      flip X  = (0.2, -0.1, 0.1)        # 180° about X: (x, -y, -z)
      rot 90Z = (0.1, 0.2, 0.1)         # 90° about Z: (-y, x, z)
      + c     = (0.6, 0.7, 0.6)
    """
    L = Lattice(mode=6, n_points=8, ratio=0.35, seed=42)
    L.flipped = True
    L.rigid_rotation = Rotation.from_euler("z", 90, degrees=True)

    M = L.world_transform()
    p = np.array([0.7, 0.6, 0.4, 1.0])
    out = M @ p
    expected = np.array([0.6, 0.7, 0.6, 1.0])
    np.testing.assert_allclose(out, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Rotation pivots about (0.5, 0.5, 0.5) — centroid is fixed
# ---------------------------------------------------------------------------

def test_world_transform_rotates_around_centroid():
    L = Lattice(mode=6, n_points=8, ratio=0.35, seed=42)
    centroid = np.array([0.5, 0.5, 0.5, 1.0])

    for rot in [
        Rotation.from_euler("xyz", [10, 20, 30], degrees=True),
        Rotation.from_euler("z", 47, degrees=True),
        Rotation.from_rotvec([0.3, 0.4, 0.5]),
    ]:
        L.rigid_rotation = rot
        for flipped in (False, True):
            L.flipped = flipped
            M = L.world_transform()
            np.testing.assert_allclose(
                M @ centroid, centroid, atol=1e-12,
                err_msg=f"centroid moved under rot={rot.as_rotvec()}, flipped={flipped}"
            )


# ---------------------------------------------------------------------------
# 4. Joint angle is stored in radians (slider works in degrees)
# ---------------------------------------------------------------------------

def test_joint_angle_storage_in_radians(main_window):
    """Verify joint_angle is stored in radians, going through the
    SPEC §6.2 slider-degrees↔simulator-radians conversion at the
    panel boundary (degrees on the slider, radians in the field).

    Stage 6c implemented the proper §6.2 mapping (slider 90° = rest =
    0 rad, 0° = -π/2, 180° = +π/2). Earlier drafts in Stage 5 used a
    direct degrees→radians mapping that pre-dated the §6.2
    reconciliation; this test reads through the panel's own
    conversion helpers so it stays robust to that history."""
    from auxetic_studio.simulation_panel import slider_to_simulator_theta

    win = main_window
    panel = win.simulation_panel

    # Slider at 45° → simulator radians via §6.2 mapping.
    panel.spin.setValue(45.0)
    panel._on_spin_committed()
    assert win.lattice.joint_angle == pytest.approx(slider_to_simulator_theta(45.0))

    # Slider at 90° → simulator radians = 0 (rest).
    panel._press_angle_rad = win.lattice.joint_angle
    panel.slider.setValue(int(round(90.0 * panel.SLIDER_SCALE)))
    panel._on_slider_released()
    assert win.lattice.joint_angle == pytest.approx(slider_to_simulator_theta(90.0))
    assert win.lattice.joint_angle == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. RotationChangeCommand is undoable with exact restoration
# ---------------------------------------------------------------------------

def test_rotation_change_command_undoable():
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    L.rigid_rotation = Rotation.from_euler("z", 30, degrees=True)
    old = L.rigid_rotation
    new = Rotation.from_euler("xyz", [12.0, 34.0, 56.0], degrees=True)

    cmd = RotationChangeCommand(L, old, new)
    cmd.redo()
    assert _quat_close(L.rigid_rotation, new)

    cmd.undo()
    assert _quat_close(L.rigid_rotation, old)


# ---------------------------------------------------------------------------
# 6. FlipCommand is undoable
# ---------------------------------------------------------------------------

def test_flip_command_undoable():
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    assert L.flipped is False

    cmd = FlipCommand(L, False, True)
    cmd.redo()
    assert L.flipped is True

    cmd.undo()
    assert L.flipped is False


# ---------------------------------------------------------------------------
# 7. v1 preset loads with default rotation values
# ---------------------------------------------------------------------------

def test_v1_preset_loads_with_default_rotation(tmp_path):
    """Hand-rolled v1 preset with no rotation fields. The v1→v2
    migration fills identity defaults."""
    v1 = {
        "version":  1,
        "mode":     1,
        "n_points": 5,
        "ratio":    0.35,
        "nz_layers": 2,
        "points":   [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]],
        "shape_params": {},
        # view_state intentionally omitted — migration must fill defaults.
        "metadata": {},
    }
    path = str(tmp_path / "v1_no_view_state.auxlat")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(v1, f)

    L = load_preset(path)

    # Identity rotation: magnitude (rotation angle in radians) is 0.
    assert L.rigid_rotation.magnitude() == pytest.approx(0.0, abs=1e-12)
    assert L.flipped is False
    assert L.joint_angle == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. v2 preset round-trips rotation / flip / joint angle
# ---------------------------------------------------------------------------

def test_v2_preset_round_trips_rotation(tmp_path):
    L = Lattice(mode=1, n_points=5, ratio=0.35, seed=42)
    L.rigid_rotation = Rotation.from_euler("xyz", [11.5, 22.5, 33.5], degrees=True)
    L.flipped        = True
    L.joint_angle    = math.radians(72.5)

    path = str(tmp_path / "v2_rot.auxlat")
    save_preset(path, L)
    dst = load_preset(path)

    # Quaternion equality up to 1e-9 (scipy normalizes; both ends do).
    assert _quat_close(dst.rigid_rotation, L.rigid_rotation, tol=1e-9)
    assert dst.flipped is True
    assert dst.joint_angle == pytest.approx(L.joint_angle, abs=1e-9)


# ---------------------------------------------------------------------------
# 9. Future-version preset is rejected
# ---------------------------------------------------------------------------

def test_rejects_future_preset_version(tmp_path):
    """A preset one version newer than the app supports must be rejected.
    Pinned to ``PRESET_VERSION + 1`` so it stays correct across schema
    bumps (was hard-coded to 5 when v4 was current; v5 added mode-11 C)."""
    payload = {
        "version":  PRESET_VERSION + 1,
        "mode":     1,
        "n_points": 5,
        "ratio":    0.35,
        "nz_layers": 2,
        "points":   [],
        "shape_params": {},
        "generation":   {},
        "dynamics":     {},
        "view_state":   {},
        "metadata":     {},
    }
    path = str(tmp_path / "future.auxlat")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(ValueError, match="newer than this application supports"):
        load_preset(path)


# ---------------------------------------------------------------------------
# 10. STL export reflects the rotation
# ---------------------------------------------------------------------------

def test_stl_export_differs_with_rotation(tmp_path):
    """SPEC §9 export rotation policy: STL output reflects the oriented
    frame. Compare bytes 80+ (skip numpy-stl's header timestamp) — a
    rotation must produce a different triangle payload."""
    L = Lattice(mode=6, n_points=8, ratio=0.35, seed=42)

    p_id  = str(tmp_path / "id.stl")
    p_rot = str(tmp_path / "rot.stl")

    L.rigid_rotation = Rotation.identity()
    L.to_stl(p_id, verbose=False)

    L.rigid_rotation = Rotation.from_euler("z", 90, degrees=True)
    L.to_stl(p_rot, verbose=False)

    with open(p_id,  "rb") as f: a = f.read()[_STL_HEADER_BYTES:]
    with open(p_rot, "rb") as f: b = f.read()[_STL_HEADER_BYTES:]
    assert a != b, "STL payload should differ when rotation is applied"


# ---------------------------------------------------------------------------
# 11. Kirigami export rotates vertices but leaves constraints unchanged
# ---------------------------------------------------------------------------

def test_kirigami_export_applies_rotation(tmp_path):
    """SPEC §9: kirigami vertices reflect the oriented frame, but
    constraints reference (tile, vertex) indices — connectivity is
    unaffected by rotation."""
    L = Lattice(mode=6, n_points=8, ratio=0.35, seed=42)

    v_id_path = str(tmp_path / "v_id.txt")
    c_id_path = str(tmp_path / "c_id.txt")
    v_rot_path = str(tmp_path / "v_rot.txt")
    c_rot_path = str(tmp_path / "c_rot.txt")

    L.rigid_rotation = Rotation.identity()
    L.to_kirigami(v_id_path, c_id_path, verbose=False)

    rot = Rotation.from_euler("z", 90, degrees=True)
    L.rigid_rotation = rot
    L.to_kirigami(v_rot_path, c_rot_path, verbose=False)

    # Constraints must be byte-identical (connectivity only).
    with open(c_id_path, "rb") as f: c_id = f.read()
    with open(c_rot_path, "rb") as f: c_rot = f.read()
    assert c_id == c_rot, "constraint connectivity must not change with rotation"

    # Vertices in the rotated export should be the rotated version of
    # the identity-export vertices.
    def _read_tiles(path: str) -> list[np.ndarray]:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                vals = [float(v) for v in line.split()]
                arr = np.array(vals, dtype=float).reshape(-1, 3)
                out.append(arr)
        return out

    tiles_id  = _read_tiles(v_id_path)
    tiles_rot = _read_tiles(v_rot_path)
    assert len(tiles_id) == len(tiles_rot)

    M = L.world_transform()  # rotation matrix, currently set on L
    for t_id, t_rot in zip(tiles_id, tiles_rot):
        assert t_id.shape == t_rot.shape
        homo = np.hstack([t_id, np.ones((len(t_id), 1))])
        expected = (M @ homo.T).T[:, :3]
        np.testing.assert_allclose(t_rot, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 12. Inspector preset orientation buttons set canonical orientations
# ---------------------------------------------------------------------------

def test_preset_orientation_buttons(main_window):
    """Programmatically trigger Top, Front, Side, Reset and verify the
    documented canonical orientation lands on the lattice."""
    win = main_window
    # Switch to a 3D mode so the preset row is visible / connected.
    insp = win.inspector
    insp.select_mode(6)

    expected = {
        "Top":   Rotation.identity(),
        "Front": Rotation.from_euler("x", 90, degrees=True),
        "Side":  Rotation.from_euler("y", 90, degrees=True),
        "Reset": Rotation.identity(),
    }
    for name, target in expected.items():
        # Set lattice off-target first so the button has work to do.
        win.lattice.rigid_rotation = Rotation.from_euler("xyz", [5, 6, 7], degrees=True)
        insp.preset_buttons[name].click()
        assert _quat_close(win.lattice.rigid_rotation, target, tol=1e-9), (
            f"preset {name!r} did not land on the canonical orientation"
        )


# ---------------------------------------------------------------------------
# 13. View2D inverse-transforms drag end position back to canonical
# ---------------------------------------------------------------------------

def test_2d_view_inverse_transform_for_edit(main_window):
    """When a non-identity rotation is in effect, the 2D view renders
    points in world space. A drag in the view delivers a world-space
    target — View2D must inverse-transform it before the lattice
    stores the canonical coordinate."""
    win = main_window
    win.lattice.rigid_rotation = Rotation.from_euler("z", 90, degrees=True)
    win.view_2d.update_lattice(win.lattice)

    # Pick an arbitrary world-space drop position.
    x_world, y_world = 0.7, 0.5

    canonical = win.view_2d.world_to_canonical_2d(x_world, y_world)

    # Hand-check: M_inv applied to (0.7, 0.5, 0, 1) should equal canonical.
    M_inv = np.linalg.inv(win.lattice.world_transform())
    expected = M_inv @ np.array([x_world, y_world, 0.0, 1.0])
    np.testing.assert_allclose(canonical, expected[:2], atol=1e-12)
