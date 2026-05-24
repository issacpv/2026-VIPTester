"""Shared pytest infrastructure.

This file is fixture infrastructure, not a test — and is therefore not
covered by the "do not modify existing tests" rule.

Background: on Windows + Python 3.14 + PyQt6 + pyvistaqt + the
``offscreen`` Qt platform, the test session can hit a fatal access
violation on the boundary between two consecutive Qt-using test
modules (e.g. ``test_app.py`` → ``test_edit.py``). The crash is in
session/module fixture teardown, not in any individual test — every
test passes its assertions before the abort. Reproducer: spinning up
N MainWindows in a loop in a single Python process is fine; the same
N split across two pytest modules is not.

The trigger is a stale Qt/VTK event-loop state surviving the boundary.
A ``gc.collect()`` + ``QApplication.processEvents()`` after each test
forces dangling QObjects to actually disappear before the next test
re-uses the application singleton, which avoids the crash. The
overhead is microseconds per test.
"""

from __future__ import annotations

import gc
import os

import pytest


# Force offscreen before any test module imports Qt — module-level so
# it runs before the first import, not per-fixture.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _flush_qt_state_between_tests():
    """Run after every test: force Python GC, drain the Qt event queue,
    and explicitly schedule any straggling top-level widgets for
    deletion so widgets created during the test are fully torn down
    before the next test (or fixture teardown) runs.

    Stage 5 added enough GUI surface area that the Stage 4 hook (just
    ``gc.collect() + processEvents()``) was no longer enough on
    Windows + Python 3.14 + PyQt6 + pyvistaqt + offscreen — the suite
    started aborting mid-run. Walking ``topLevelWidgets()`` and
    ``deleteLater()``-ing whatever's left between tests keeps the
    accumulated widget count bounded across the full suite.

    Stage 6c added matplotlib FigureCanvases inside the
    SimulationPanel. Even with parent-child Qt cleanup, pyplot's
    global figure registry holds strong refs that survive the
    parent's deletion, leaving FigureCanvas widgets queued for VTK's
    atexit racer. Closing all pyplot figures here before the GC
    settles avoids the late-stage abort that otherwise hits when
    QTest.qWait is called downstream."""
    yield
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
    except Exception:
        pass
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        return
    gc.collect()
    app = QApplication.instance()
    if app is None:
        return
    app.processEvents()
    for w in list(app.topLevelWidgets()):
        try:
            w.deleteLater()
        except Exception:
            pass
    app.processEvents()
    gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """Drain Qt's event queue and force the QApplication singleton
    to release before Python interpreter shutdown.

    Without this hook, every test passes but the interpreter aborts in
    VTK's atexit handler racing with Qt's. Flushing here ensures the
    process exits cleanly with the exit status pytest computed.

    Also closes any matplotlib figures Stage 6c may have left behind.
    Pyplot's global figure registry holds strong refs to FigureCanvas
    Qt widgets; if we don't close them explicitly, they outlive the
    QApplication and the dtor races with the app's teardown path."""
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
    except Exception:
        pass

    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        return
    gc.collect()
    app = QApplication.instance()
    if app is None:
        return
    app.processEvents()
    # Drop any remaining top-level widgets so their dtors run while
    # the application is still alive.
    for w in list(app.topLevelWidgets()):
        try:
            w.close()
            w.deleteLater()
        except Exception:
            pass
    app.processEvents()
    gc.collect()
