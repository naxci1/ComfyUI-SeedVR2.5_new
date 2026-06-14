"""Left-side project / file navigator panel."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from PySide6.QtCore import Qt, Signal, QObject, QThread, QSize
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..theme import Colors, Dims, Fonts
from .button3d import Button3D

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

_FILTER = "Media (*.mp4 *.mov *.mkv *.avi *.webm *.png *.jpg *.jpeg *.tif *.tiff *.exr *.dpx);;All files (*)"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v"}


def _extract_metadata(path: str):
    """Return (thumbnail QImage|None, meta_string)."""
    if cv2 is None or np is None or not os.path.isfile(path):
        return None, ""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in _IMAGE_EXTS:
            raw = np.fromfile(path, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                return None, ""
            h, w = frame.shape[:2]
            meta = f"{w}×{h}"
        else:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                return None, ""
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return None, f"{w}×{h}"
            meta = f"{w}×{h} • {fps:.0f} fps"
        thumb = cv2.resize(frame, (64, 36))
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, 64, 36, 3 * 64, QImage.Format_RGB888).copy()
        return img, meta
    except Exception:
        return None, ""


class _ThumbWorker(QObject):
    done = Signal(str, object, str)  # path, QImage|None, meta

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def run(self) -> None:
        img, meta = _extract_metadata(self._path)
        self.done.emit(self._path, img, meta)


class _FileItemWidget(QWidget):
    """List item widget showing filename + remove (×) button."""

    remove_requested = Signal(str)

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._label = QLabel(os.path.basename(path), self)
        self._label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        layout.addWidget(self._label, 1)

        self._icon_lbl = QLabel(self)
        self._icon_lbl.setFixedSize(64, 36)
        layout.insertWidget(0, self._icon_lbl)

        btn = Button3D("✕", variant="danger", parent=self)
        btn.setFixedSize(28, 28)
        btn.setToolTip("Remove from list")
        btn.clicked.connect(lambda: self.remove_requested.emit(self._path))
        layout.addWidget(btn)

    def set_icon(self, pixmap: QPixmap) -> None:
        self._icon_lbl.setPixmap(
            pixmap.scaled(64, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def set_meta(self, meta: str) -> None:
        name = os.path.basename(self._path)
        self._label.setText(f"{name}\n{meta}" if meta else name)


class ProjectPanel(QWidget):
    """Project navigator listing imported media with thumbnails."""

    file_selected = Signal(str)
    file_removed = Signal(str)
    input_folder_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(Dims.PANEL_WIDTH_LEFT)
        self._threads: List[QThread] = []
        self._workers: list = []  # keep worker refs alive until thread finishes
        self._output_dir: str = ""
        self._item_widgets: dict = {}  # path → (QListWidgetItem, _FileItemWidget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("FILES")
        header.setStyleSheet(
            f"background-color: {Colors.BG_LIGHTER}; color: {Colors.TEXT_ACCENT};"
            f" font-weight: {Fonts.WEIGHT_BOLD}; font-size: {Fonts.SIZE_SMALL}px;"
            f" padding: {Dims.PADDING_SM}px {Dims.PADDING_MD}px;"
        )
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setIconSize(QSize(64, 36))
        self._list.setSpacing(2)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list, 1)

        self._browse = Button3D("＋ Browse Files", variant="default")
        self._browse.clicked.connect(self._on_browse)
        self._browse_folder = Button3D("📂 Input Folder", variant="default")
        self._browse_folder.clicked.connect(self._on_browse_folder)
        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setContentsMargins(Dims.PADDING_SM, Dims.PADDING_SM, Dims.PADDING_SM, Dims.PADDING_SM)
        wrap_layout.setSpacing(Dims.PADDING_SM)
        wrap_layout.addWidget(self._browse)
        wrap_layout.addWidget(self._browse_folder)
        layout.addWidget(wrap)

    # ---------------------------------------------------------------- api
    def set_output_dir(self, path: str) -> None:
        self._output_dir = path or ""

    def add_file(self, path: str, select: bool = True) -> None:
        # Avoid duplicates.
        if path in self._item_widgets:
            if select:
                item, _ = self._item_widgets[path]
                self._list.setCurrentItem(item)
            return
        item = QListWidgetItem()
        item.setData(Qt.UserRole, path)
        item.setSizeHint(QSize(Dims.PANEL_WIDTH_LEFT - 12, 54))
        self._list.addItem(item)

        widget = _FileItemWidget(path, self._list)
        widget.remove_requested.connect(self._on_remove_file)
        self._list.setItemWidget(item, widget)
        self._item_widgets[path] = (item, widget)

        if select:
            self._list.setCurrentItem(item)
        self._load_thumb(path, widget)

    def add_files(self, paths: Iterable[str], select_last: bool = True) -> None:
        files = [path for path in paths if path]
        for index, path in enumerate(files):
            self.add_file(path, select=select_last and index == len(files) - 1)

    def _on_remove_file(self, path: str) -> None:
        if path not in self._item_widgets:
            return
        item, _ = self._item_widgets.pop(path)
        row = self._list.row(item)
        self._list.takeItem(row)
        self.file_removed.emit(path)

    def _load_thumb(self, path: str, widget: _FileItemWidget) -> None:
        if cv2 is None:
            return
        thread = QThread()
        worker = _ThumbWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _apply(p, img, meta, _widget=widget) -> None:
            if img is not None and not img.isNull():
                _widget.set_icon(QPixmap.fromImage(img))
            if meta:
                _widget.set_meta(meta)
            thread.quit()

        worker.done.connect(_apply)

        def _cleanup(t=thread, w=worker) -> None:
            if t in self._threads:
                self._threads.remove(t)
            if w in self._workers:
                self._workers.remove(w)

        thread.finished.connect(_cleanup)
        self._threads.append(thread)
        self._workers.append(worker)
        thread.start()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.file_selected.emit(path)

    def _on_browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Browse media", "", _FILTER)
        self.add_files(paths)
        if paths:
            self.file_selected.emit(paths[-1])

    def _on_browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select input folder", "")
        if not folder:
            return
        video_files = [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, name))
            and os.path.splitext(name)[1].lower() in _VIDEO_EXTS
        ]
        self.add_files(video_files)
        if video_files:
            self.file_selected.emit(video_files[0])
        self.input_folder_selected.emit(folder)

    def cleanup(self) -> None:
        for t in list(self._threads):
            t.quit()
            t.wait(200)
        self._threads.clear()
        self._workers.clear()
