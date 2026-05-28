#!/bin/bash
# Auxetic Lattice Studio - macOS launcher (double-click to run).
# First run sets up the Python environment; later runs just launch the app.

cd "$(dirname "$0")" || exit 1

# --- locate a suitable Python (3.11+) ---
PY=""
for c in python3.12 python3.11 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done

if [ -z "$PY" ]; then
  osascript -e 'display alert "Python not found" message "Please install Python 3.11 or newer from https://www.python.org/downloads/ (double-click the installer and click through it), then double-click this file again."'
  exit 1
fi

MAJOR="$("$PY" -c 'import sys;print(sys.version_info[0])')"
MINOR="$("$PY" -c 'import sys;print(sys.version_info[1])')"
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]; }; then
  osascript -e "display alert \"Python too old\" message \"Found Python $MAJOR.$MINOR. Please install Python 3.11 or newer from python.org, then try again.\""
  exit 1
fi

# --- first-run setup: create venv + install dependencies ---
if [ ! -d ".venv" ]; then
  echo "First-time setup - this downloads a few hundred MB and may take several minutes..."
  "$PY" -m venv .venv || { osascript -e 'display alert "Setup failed" message "Could not create the Python environment."'; exit 1; }
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install PyQt6 pyqtgraph pyvista pyvistaqt numpy scipy numpy-stl
  if [ $? -ne 0 ]; then
    rm -rf .venv
    osascript -e 'display alert "Install failed" message "Dependency installation failed. Check your internet connection, then double-click again."'
    exit 1
  fi
fi

# --- launch ---
echo "Launching Auxetic Lattice Studio..."
exec ./.venv/bin/python -m auxetic_studio
