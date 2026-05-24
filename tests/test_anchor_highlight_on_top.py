"""Pure unit tests for the anchor-ring "always on top" styling.

The reference-polygon highlight is lifted just above the structure, so
from below the opaque mesh used to occlude it (visible from the top, not
the bottom — the user-reported bug). ``_force_actor_on_top`` pushes the
ring's rasterized depth toward the near plane via large negative
coincident-topology offsets so it renders over the mesh from any angle.

Pixel visibility can't be asserted headlessly (no render window), so we
lock in the *intent*: the helper drives the actor's VTK mapper with the
expected always-on-top calls. These tests use recording fakes and never
construct a Qt widget, so they're immune to the GUI teardown race.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeMapper:
    """Records the coincident-topology calls the helper makes."""

    def __init__(self):
        self.calls: list[str] = []
        self.resolve_mode = None
        self.polygon = None
        self.line = None
        self.point = None

    def SetResolveCoincidentTopologyToPolygonOffset(self):
        self.resolve_mode = "polygon_offset"
        self.calls.append("resolve")

    def SetRelativeCoincidentTopologyPolygonOffsetParameters(self, factor, units):
        self.polygon = (factor, units)
        self.calls.append("polygon")

    def SetRelativeCoincidentTopologyLineOffsetParameters(self, factor, units):
        self.line = (factor, units)
        self.calls.append("line")

    def SetRelativeCoincidentTopologyPointOffsetParameter(self, units):
        self.point = units
        self.calls.append("point")


class _FakeActor:
    def __init__(self, mapper):
        self._mapper = mapper

    def GetMapper(self):
        return self._mapper


def test_force_on_top_applies_all_offsets():
    from auxetic_studio.views import _force_actor_on_top
    m = _FakeMapper()
    ok = _force_actor_on_top(_FakeActor(m))
    assert ok is True
    assert m.resolve_mode == "polygon_offset"
    # Every primitive offset is pushed toward the camera (negative units)
    # so the ring wins the depth test against the solid mesh from below.
    assert m.polygon is not None and m.polygon[1] < 0
    assert m.line is not None and m.line[1] < 0
    assert m.point is not None and m.point < 0
    assert m.calls == ["resolve", "polygon", "line", "point"]


def test_force_on_top_uses_mapper_attr_fallback():
    """Actors that expose a ``.mapper`` (pyvista) rather than GetMapper()
    are still handled."""
    from auxetic_studio.views import _force_actor_on_top

    class _PvActor:
        def __init__(self, mapper):
            self.mapper = mapper

    m = _FakeMapper()
    assert _force_actor_on_top(_PvActor(m)) is True
    assert m.calls == ["resolve", "polygon", "line", "point"]


def test_force_on_top_no_mapper_is_safe():
    from auxetic_studio.views import _force_actor_on_top

    class _NoMapper:
        pass

    assert _force_actor_on_top(_NoMapper()) is False


def test_force_on_top_swallows_mapper_errors():
    """A mapper missing the coincident-topology API (a future/older VTK
    build) must not raise — the ring just falls back to the lifted-only
    look rather than crashing the render."""
    from auxetic_studio.views import _force_actor_on_top

    class _BadMapper:
        def SetResolveCoincidentTopologyToPolygonOffset(self):
            raise RuntimeError("no such method on this VTK build")

    class _Actor:
        def GetMapper(self):
            return _BadMapper()

    assert _force_actor_on_top(_Actor()) is False
