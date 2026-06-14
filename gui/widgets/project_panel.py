"""Left-side project / file navigator panel."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable, List, Optional

from PySide6.QtCore import Qt, Signal, QObject, QThread, QSize
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
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

_FILTER = "Media (*.mp4 *.mov *.mkv *.avi *.webm *.png *.jpg *.jpeg *.tif *.tiff *.exr *.dpx);;All files (*)"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v"}


def _extract_metadata(path: str):
    """Return (thumbnail QImage|None, meta_string)."""
    if cv2 is None or not os.path.isfile(path):
        return None, ""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in _IMAGE_EXTS:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
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


class ProjectPanel(QWidget):
    """Project navigator listing imported media with thumbnails."""

    file_selected = Signal(str)
    input_folder_selected = Signal(str)
    output_folder_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(Dims.PANEL_WIDTH_LEFT)
        self._threads: List[QThread] = []
        self._output_dir: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("PROJECT")
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
        self._output_btn = Button3D("📁 Output Folder", variant="ghost")
        self._output_btn.clicked.connect(self._on_open_output_folder)
        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setContentsMargins(Dims.PADDING_SM, Dims.PADDING_SM, Dims.PADDING_SM, Dims.PADDING_SM)
        wrap_layout.setSpacing(Dims.PADDING_SM)
        wrap_layout.addWidget(self._browse)
        wrap_layout.addWidget(self._browse_folder)
        wrap_layout.addWidget(self._output_btn)
        layout.addWidget(wrap)

    # ---------------------------------------------------------------- api
    def set_output_dir(self, path: str) -> None:
        self._output_dir = path or ""

    def add_file(self, path: str, select: bool = True) -> None:
        # Avoid duplicates.
        for i in range(self._list.count()):
            if self._list.item(i).data(Qt.UserRole) == path:
                if select:
                    self._list.setCurrentRow(i)
                return
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.UserRole, path)
        item.setSizeHint(QSize(Dims.PANEL_WIDTH_LEFT - 12, 44))
        self._list.addItem(item)
        if select:
            self._list.setCurrentItem(item)
        self._load_thumb(path, item)

    def add_files(self, paths: Iterable[str], select_last: bool = True) -> None:
        files = [path for path in paths if path]
        for index, path in enumerate(files):
            self.add_file(path, select=select_last and index == len(files) - 1)

    def _load_thumb(self, path: str, item: QListWidgetItem) -> None:
        if cv2 is None:
            return
        thread = QThread()
        worker = _ThumbWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _apply(p, img, meta, _item=item) -> None:
            if img is not None and not img.isNull():
                _item.setIcon(QIcon(QPixmap.fromImage(img)))
            if meta:
                _item.setText(f"{os.path.basename(p)}\n{meta}")
            thread.quit()

        worker.done.connect(_apply)
        thread.finished.connect(lambda t=thread: self._threads.remove(t) if t in self._threads else None)
        self._threads.append(thread)
        thread.start()

    def _on_open_output_folder(self) -> None:
        folder = self._output_dir or os.getcwd()
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception:
            pass
        self.output_folder_requested.emit()

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
