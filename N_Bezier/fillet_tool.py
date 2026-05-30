"""
STL Bezier Fillet Tool
======================
A desktop app for smoothing the sharp corners of 2.5D extruded STL models
(the kind you get from extruding a 2D sketch) so they print more reliably.

The pipeline:
  1. Load an STL.
  2. Take a horizontal cross-section to recover the 2D outline.
  3. Walk every vertex of every polygon; at each "sharp enough" corner,
     back off along both adjacent edges by `radius` and replace the corner
     with a quadratic Bezier curve (control point = original corner).
  4. Re-extrude back to the original height.
  5. Render before/after side-by-side and let the user export.

Run:
    python fillet_tool.py
"""

import os
import sys

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSlider, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

import pyvista as pv
from pyvistaqt import QtInteractor

# VTK's trackball-camera style is what we subclass to remap the right button.
try:
    from vtkmodules.vtkInteractionStyle import (
        vtkInteractorStyleTrackballCamera as _TrackballCamera,
    )
except Exception:  # older vtk packaging
    from vtk import vtkInteractorStyleTrackballCamera as _TrackballCamera


class _RotatePanZoomStyle(_TrackballCamera):
    """Camera controls: left-drag rotates, right-drag moves, scroll zooms.

    Subclasses VTK's trackball-camera style and remaps only the right mouse
    button from its default dolly (zoom) to a pan. We do this with observer
    callbacks rather than by overriding OnRightButtonDown/Up, because VTK does
    not reliably dispatch overridden virtual methods to a Python subclass,
    whereas observer callbacks always fire. Registering an observer for these
    button events also suppresses the built-in handler, so the default zoom no
    longer runs. Left-drag (rotate) and the scroll wheel (zoom) are untouched.
    """

    def __init__(self):
        self.AddObserver("RightButtonPressEvent", self._on_right_press)
        self.AddObserver("RightButtonReleaseEvent", self._on_right_release)

    def _on_right_press(self, obj, event):
        self.StartPan()

    def _on_right_release(self, obj, event):
        self.EndPan()


# ---------------------------------------------------------------------------
# Geometry: find where separate 2.5D pieces meet at a single point,
# then add Bezier-curved bridge material to bind them together.
# ---------------------------------------------------------------------------

def _quadratic_bezier(p0, p1, p2, n_samples=20):
    """Sample `n_samples` points along the quadratic Bezier P0 -> P1 -> P2."""
    t = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    p0, p1, p2 = np.asarray(p0), np.asarray(p1), np.asarray(p2)
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _section_components_to_polygons(mesh, z_mid):
    """
    Section every connected component of `mesh` at z=z_mid and return a flat
    list of shapely Polygons in world-XY coordinates.

    Splitting into components first matters: when pieces only touch at points,
    trimesh's polygon extractor gets confused if you section the whole soup at
    once and produces garbage. Sectioning each piece in isolation avoids this.
    """
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        components = [mesh]

    polygons = []
    for comp in components:
        sec = comp.section(plane_origin=[0, 0, z_mid], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        if hasattr(sec, "to_2D"):
            section_2d, T = sec.to_2D()
        else:
            section_2d, T = sec.to_planar()
        T = np.asarray(T)

        def to_world(xy):
            v = T @ np.array([xy[0], xy[1], 0.0, 1.0])
            return (v[0], v[1])

        for poly in section_2d.polygons_full:
            ext = [to_world(c) for c in list(poly.exterior.coords)[:-1]]
            holes = [[to_world(c) for c in list(h.coords)[:-1]]
                     for h in poly.interiors]
            wp = Polygon(ext, holes=holes)
            if not wp.is_valid:
                wp = wp.buffer(0)
                if wp.is_empty:
                    continue
            # buffer(0) can return MultiPolygon; flatten.
            for sp in ([wp] if isinstance(wp, Polygon) else list(wp.geoms)):
                if not sp.is_empty:
                    polygons.append(sp)
    return polygons


def _find_junctions(polygons, tol):
    """
    Greedy-cluster every polygon vertex by 2D position. A cluster is a
    "junction" if it contains vertices from at least two different polygons.

    Returns a list of clusters, each a list of (poly_idx, vert_idx, position).
    """
    verts = []
    for pi, poly in enumerate(polygons):
        for vi, v in enumerate(list(poly.exterior.coords)[:-1]):
            verts.append((pi, vi, np.array(v, dtype=float)))

    junctions = []
    used = [False] * len(verts)
    for i in range(len(verts)):
        if used[i]:
            continue
        cluster = [verts[i]]
        used[i] = True
        for j in range(i + 1, len(verts)):
            if used[j]:
                continue
            if np.linalg.norm(verts[i][2] - verts[j][2]) < tol:
                cluster.append(verts[j])
                used[j] = True
        if len({c[0] for c in cluster}) >= 2:
            junctions.append(cluster)
    return junctions


def _build_bridge_polygon(junction, polygons, radius, samples=12):
    """
    Build a single Bezier-curved polygon that fills the space around a
    junction point so the meeting shapes get welded together with curves.

    For every edge ending at the junction (collected from all participating
    polygons), back off along the edge by `radius` to get a "stop point".
    Sort these stop points by angle around the junction and connect each
    consecutive pair with a quadratic Bezier whose control point is the
    junction itself. The control point pulls the curve toward the junction,
    so the resulting polygon sits like a flower around it, overlapping the
    polygon corners and filling the gaps between them.
    """
    center = np.mean([c[2] for c in junction], axis=0)

    edges = []
    for pi, vi, _ in junction:
        coords = list(polygons[pi].exterior.coords)[:-1]
        n = len(coords)
        for ni in [(vi - 1) % n, (vi + 1) % n]:
            d = np.array(coords[ni], dtype=float) - center
            L = float(np.linalg.norm(d))
            if L > 1e-9:
                edges.append({
                    "dir": d / L,
                    "len": L,
                    "angle": float(np.arctan2(d[1], d[0])),
                })
    if len(edges) < 2:
        return None
    edges.sort(key=lambda e: e["angle"])

    backoffs = [center + e["dir"] * min(radius, e["len"] * 0.45) for e in edges]

    pts = []
    for i in range(len(backoffs)):
        bez = _quadratic_bezier(backoffs[i], center,
                                backoffs[(i + 1) % len(backoffs)],
                                n_samples=samples)
        pts.extend(bez[:-1])  # avoid duplicating endpoints

    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly


def fillet_25d_mesh(mesh, fillet_radius, junction_tol=None):
    """
    Find points where separate 2.5D pieces meet at a single vertex and
    bind them with Bezier-curved bridge material. Individual polygon corners
    that aren't junctions are left alone.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input. Assumed to be extruded along +Z.
    fillet_radius : float
        How far back along each edge meeting at a junction the curve starts.
        Bigger = thicker, more aggressive bridge.
    junction_tol : float or None
        Two vertices count as "the same junction point" if they're within
        this distance. Defaults to 10% of `fillet_radius`.
    """
    if fillet_radius <= 1e-6:
        return mesh.copy()

    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    height = z_max - z_min
    if height <= 1e-9:
        raise ValueError("Mesh has zero height along Z — is it really 2.5D?")
    z_mid = (z_min + z_max) / 2.0

    polygons = _section_components_to_polygons(mesh, z_mid)
    if not polygons:
        return mesh.copy()

    if junction_tol is None:
        junction_tol = max(fillet_radius * 0.1, 1e-3)

    junctions = _find_junctions(polygons, junction_tol)

    bridges = []
    for jn in junctions:
        bridge = _build_bridge_polygon(jn, polygons, fillet_radius)
        if bridge is not None:
            bridges.append(bridge)

    combined = unary_union(polygons + bridges)
    if combined.geom_type == "Polygon":
        result_polys = [combined]
    elif combined.geom_type == "MultiPolygon":
        result_polys = list(combined.geoms)
    else:
        result_polys = [g for g in getattr(combined, "geoms", [])
                        if g.geom_type == "Polygon"]

    new_meshes = []
    for poly in result_polys:
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty:
                continue
        new_meshes.append(trimesh.creation.extrude_polygon(poly, height=height))

    if not new_meshes:
        return mesh.copy()

    out = trimesh.util.concatenate(new_meshes) if len(new_meshes) > 1 else new_meshes[0]
    out.apply_translation([0, 0, z_min])
    return out


# ---------------------------------------------------------------------------
# A single 3D viewer panel (title + embedded PyVista plotter).
# ---------------------------------------------------------------------------

class MeshViewer(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-weight: 600; padding: 4px; color: #1f2937;")
        layout.addWidget(self.label)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        # Controls: left-drag rotates, right-drag moves (pans), scroll zooms.
        self._style = _RotatePanZoomStyle()
        try:
            self.plotter.iren.interactor.SetInteractorStyle(self._style)
        except Exception:
            # Fallback for other pyvista/pyvistaqt versions.
            self.plotter.interactor.GetRenderWindow().GetInteractor() \
                .SetInteractorStyle(self._style)
        self.plotter.interactor.setToolTip(
            "Left-drag: rotate   •   Right-drag: move   •   Scroll: zoom"
        )
        layout.addWidget(self.plotter.interactor, stretch=1)

        self._actor = None

    def show_mesh(self, mesh, color="#cbd5e1", reset_camera=True):
        if self._actor is not None:
            self.plotter.remove_actor(self._actor)
        self._actor = self.plotter.add_mesh(
            pv.wrap(mesh),
            color=color,
            show_edges=True,
            edge_color="#475569",
            line_width=0.4,
            smooth_shading=True,
        )
        if reset_camera:
            self.plotter.reset_camera()

    def clear(self):
        if self._actor is not None:
            self.plotter.remove_actor(self._actor)
            self._actor = None


# ---------------------------------------------------------------------------
# Main window.
# ---------------------------------------------------------------------------

class FilletApp(QMainWindow):
    EXPORT_FORMATS = ["STL", "OBJ", "PLY", "GLB", "3MF", "DAE", "F3D (Fusion 360)"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Bezier Fillet Tool")
        self.resize(1280, 760)

        self.original_mesh = None
        self.processed_mesh = None
        self.current_path = None

        self._build_ui()
        self.stack.setCurrentWidget(self.home_widget)

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_home())
        self.stack.addWidget(self._build_editor())
        root.addWidget(self.stack)

    def _build_home(self):
        self.home_widget = QWidget()
        layout = QVBoxLayout(self.home_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        title = QLabel("STL Bezier Fillet Tool")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: bold; color: #0f172a;")

        subtitle = QLabel("Smooth sharp corners on 2.5D extruded STL models for 3D printing.")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #475569; padding-bottom: 20px;")

        load_btn = QPushButton("Choose an STL file…")
        load_btn.setMinimumSize(260, 52)
        load_btn.setCursor(Qt.PointingHandCursor)
        load_btn.setStyleSheet(
            "QPushButton { font-size: 15px; background-color: #2563eb; color: white;"
            " border: none; border-radius: 8px; padding: 10px 20px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
            "QPushButton:pressed { background-color: #1e40af; }"
        )
        load_btn.clicked.connect(self._open_file)

        hint = QLabel("Works best on models that are a straight extrusion of a 2D shape.")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #94a3b8; font-size: 12px; padding-top: 30px;")

        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(load_btn, alignment=Qt.AlignCenter)
        layout.addWidget(hint)
        layout.addStretch()
        return self.home_widget

    def _build_editor(self):
        self.editor_widget = QWidget()
        layout = QVBoxLayout(self.editor_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Top row: filename + reload
        top_row = QHBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #475569; padding: 4px 0;")
        change_btn = QPushButton("Load different file…")
        change_btn.clicked.connect(self._open_file)
        top_row.addWidget(self.file_label)
        top_row.addStretch()
        top_row.addWidget(change_btn)
        layout.addLayout(top_row)

        # Side-by-side viewers
        viewers = QSplitter(Qt.Horizontal)
        self.before_viewer = MeshViewer("Before")
        self.after_viewer = MeshViewer("After")
        viewers.addWidget(self.before_viewer)
        viewers.addWidget(self.after_viewer)
        viewers.setSizes([600, 600])
        layout.addWidget(viewers, stretch=1)

        # Bottom bar: slider + export
        bottom = QHBoxLayout()

        slider_col = QVBoxLayout()
        slider_header = QHBoxLayout()
        slider_header.addWidget(QLabel("Fillet thickness:"))
        self.slider_value_label = QLabel("0.00")
        self.slider_value_label.setStyleSheet("font-weight: bold; color: #0f172a;")
        slider_header.addStretch()
        slider_header.addWidget(self.slider_value_label)
        slider_col.addLayout(slider_header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._reprocess)
        slider_col.addWidget(self.slider)
        bottom.addLayout(slider_col, stretch=1)

        bottom.addSpacing(20)

        export_col = QVBoxLayout()
        export_col.addWidget(QLabel("Export format:"))
        export_row = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(self.EXPORT_FORMATS)
        export_btn = QPushButton("Export")
        export_btn.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white;"
            " border: none; border-radius: 6px; padding: 6px 18px; font-weight: 600; }"
            "QPushButton:hover { background-color: #15803d; }"
        )
        export_btn.clicked.connect(self._export)
        export_row.addWidget(self.format_combo)
        export_row.addWidget(export_btn)
        export_col.addLayout(export_row)
        bottom.addLayout(export_col)

        layout.addLayout(bottom)
        self._link_viewers()
        return self.editor_widget

    def _link_viewers(self):
        """Share one camera between the Before and After views so rotating,
        panning, or zooming either view moves the other in lockstep."""
        a = self.before_viewer.plotter
        b = self.after_viewer.plotter

        # Make both renderers drive the same underlying camera object.
        try:
            b.camera = a.camera
        except Exception:
            try:
                b.renderer.SetActiveCamera(a.renderer.GetActiveCamera())
            except Exception:
                return  # camera linking unsupported on this build; leave independent

        # The shared camera means an interaction updates it for both, but only
        # the interacted window repaints automatically — so nudge the other.
        def repaint_b(*_):
            b.render()

        def repaint_a(*_):
            a.render()

        for event in ("InteractionEvent", "EndInteractionEvent"):
            try:
                a.iren.add_observer(event, repaint_b)
                b.iren.add_observer(event, repaint_a)
            except Exception:
                pass

    # -- Actions -------------------------------------------------------------

    def _slider_to_radius(self, value):
        # The slider is integer 0..1000; we scale it to a per-model range
        # set in _open_file (max ≈ 25% of the model's largest XY dimension).
        if not hasattr(self, "_radius_max"):
            return value / 100.0
        return (value / 1000.0) * self._radius_max

    def _on_slider_changed(self, value):
        r = self._slider_to_radius(value)
        self.slider_value_label.setText(f"{r:.3f}")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open STL file", "", "STL files (*.stl);;All files (*)"
        )
        if not path:
            return
        try:
            mesh = trimesh.load(path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                raise ValueError("File did not contain a usable mesh.")

            self.original_mesh = mesh
            self.current_path = path

            # Set a sensible slider range based on model size.
            largest_xy = max(mesh.extents[0], mesh.extents[1])
            self._radius_max = max(largest_xy * 0.25, 1e-3)
            self.slider.blockSignals(True)
            self.slider.setValue(int(1000 * 0.2))  # start at 20% -> a visible fillet
            self.slider.blockSignals(False)
            self._on_slider_changed(self.slider.value())

            self.file_label.setText(
                f"Loaded: <b>{os.path.basename(path)}</b>  "
                f"<span style='color:#94a3b8'>"
                f"({mesh.extents[0]:.1f} × {mesh.extents[1]:.1f} × {mesh.extents[2]:.1f})"
                f"</span>"
            )
            self.before_viewer.show_mesh(mesh, color="#cbd5e1")
            self._reprocess()
            self.stack.setCurrentWidget(self.editor_widget)

        except Exception as exc:
            QMessageBox.critical(self, "Load failed",
                                 f"Could not load file:\n\n{exc}")

    def _reprocess(self):
        if self.original_mesh is None:
            return
        try:
            r = self._slider_to_radius(self.slider.value())
            self.processed_mesh = fillet_25d_mesh(self.original_mesh, r)
            self.after_viewer.show_mesh(self.processed_mesh, color="#93c5fd",
                                        reset_camera=False)
        except Exception as exc:
            QMessageBox.warning(self, "Processing error", str(exc))

    def _export(self):
        if self.processed_mesh is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Load a model first.")
            return

        fmt = self.format_combo.currentText()

        if fmt.startswith("F3D"):
            QMessageBox.information(
                self, "About F3D export",
                "F3D is Autodesk Fusion 360's proprietary format and isn't writable "
                "outside of Fusion itself.\n\n"
                "To get this model into Fusion as .f3d:\n"
                "  1. Export here as STL, OBJ, or 3MF.\n"
                "  2. In Fusion: File → Open → pick the file → "
                "File → Save As → choose '.f3d'."
            )
            return

        ext = fmt.lower()
        default_name = os.path.splitext(
            os.path.basename(self.current_path or "model"))[0] + f"_filleted.{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as", default_name, f"{ext.upper()} files (*.{ext})"
        )
        if not path:
            return
        try:
            self.processed_mesh.export(path)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FilletApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
