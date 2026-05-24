"""Pure camera-math helpers for the 3D viewport.

Deliberately free of Qt / VTK / pyvista imports so the math can be
unit-tested without dragging in the GUI rendering stack (which, on the
offscreen Qt platform used in CI, destabilises process teardown). The
VTK observer wiring that *uses* these helpers lives in
``auxetic_studio.views.View3D``.
"""

from __future__ import annotations

import numpy as np


# Fraction the camera distance changes per mouse-wheel notch. 0.15 -> a
# notch in dollies to ~87% of the previous distance, a notch out to ~115%.
ZOOM_WHEEL_STEP = 0.15


def dolly_toward_cursor(camera_position, focal_point, cursor_world_point,
                        factor):
    """Compute the new ``(camera_position, focal_point)`` for a wheel-zoom
    that dollies toward ``cursor_world_point`` instead of the view centre.

    ``factor`` > 1 zooms **in** (camera moves closer to the cursor point);
    ``factor`` < 1 zooms **out**. The cursor world point is the fixed
    point of the transform — it stays under the cursor across the zoom —
    because both the camera position and the focal point are scaled
    toward it by ``s = 1 / factor``::

        new = cursor + (old - cursor) * s

    Two consequences worth noting:

    - With ``factor == 1`` (``s == 1``) nothing moves.
    - The view *direction* is preserved: ``new_cam - new_foc ==
      (cam - foc) * s``, so the zoom never rotates the camera. When the
      cursor point coincides with the focal point this degenerates to the
      classic zoom-to-centre dolly, so a centred wheel-zoom leaves camera
      presets looking exactly where they were.

    Pure (no VTK / Qt). Returns two length-3 ``numpy`` arrays. A
    non-finite or non-positive ``factor`` (or a non-finite target) is
    treated as a no-op so a bad event can never NaN-out the camera.
    """
    cam = np.asarray(camera_position, dtype=float).copy()
    foc = np.asarray(focal_point, dtype=float).copy()
    tgt = np.asarray(cursor_world_point, dtype=float)
    f = float(factor)
    if not np.isfinite(f) or f <= 0.0 or not np.all(np.isfinite(tgt)):
        return cam, foc
    s = 1.0 / f
    new_cam = tgt + (cam - tgt) * s
    new_foc = tgt + (foc - tgt) * s
    return new_cam, new_foc
