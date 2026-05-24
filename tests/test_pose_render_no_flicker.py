"""Headless smoke test for the simulation-playback flicker fix.

During playback, ``View3D.show_pose`` swapped the mesh actor with
``remove_actor`` (default ``render=True``) then ``add_mesh`` — so VTK
rendered a frame with no mesh actor in the gap, strobing the lattice in
and out at 30 fps. ``_swap_mesh_actor`` now issues both ops with
``render=False`` and the caller renders once after the scene is rebuilt.

On-screen flicker can't be measured headlessly, but the invariant that
prevents it can: a pose step keeps exactly one mesh actor and never asks
VTK to render while the mesh is removed. A recording fake interactor
stands in for the real QtInteractor — no Qt widget is built, so this is
immune to the GUI teardown race.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeInteractor:
    """Records add/remove/render ops and tracks the live actor set."""

    def __init__(self):
        self.live: list[str] = []
        self.ops: list[tuple[str, bool]] = []   # (op, render_flag)
        self._n = 0

    def remove_actor(self, actor, render=True):
        self.ops.append(("remove", render))
        if actor in self.live:
            self.live.remove(actor)

    def add_mesh(self, mesh, render=True, **kw):
        self.ops.append(("add", render))
        self._n += 1
        actor = f"actor{self._n}"
        self.live.append(actor)
        return actor

    def render(self):
        self.ops.append(("render", True))


def test_swap_keeps_exactly_one_actor_across_steps():
    from auxetic_studio.views import _swap_mesh_actor
    fi = _FakeInteractor()

    actor = None
    for _ in range(5):                       # simulate a 5-frame sweep
        actor = _swap_mesh_actor(
            fi, actor, "mesh",
            color="lightsteelblue", show_edges=False, smooth_shading=True,
        )
        assert actor is not None
        assert len(fi.live) == 1             # never zero, never leaking

    # First frame only adds; each later frame removes then adds.
    assert [op for op, _ in fi.ops] == [
        "add",
        "remove", "add",
        "remove", "add",
        "remove", "add",
        "remove", "add",
    ]


def test_swap_never_renders_an_empty_frame():
    """remove + add both go through render=False, and the helper itself
    never renders — so VTK is never asked to draw with the mesh gone."""
    from auxetic_studio.views import _swap_mesh_actor
    fi = _FakeInteractor()
    a = _swap_mesh_actor(fi, None, "m")
    _swap_mesh_actor(fi, a, "m2")
    assert all(flag is False for op, flag in fi.ops if op in ("remove", "add"))
    assert ("render", True) not in fi.ops    # the caller owns the render


def test_swap_returns_none_when_add_fails():
    from auxetic_studio.views import _swap_mesh_actor

    class _Boom:
        def remove_actor(self, actor, render=True):
            pass

        def add_mesh(self, mesh, render=True, **kw):
            raise RuntimeError("VTK add failed")

    assert _swap_mesh_actor(_Boom(), "old", "mesh") is None
