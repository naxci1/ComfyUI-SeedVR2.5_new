"""Central video / image preview widget backed by ``cv2.VideoCapture`` with audio via QMediaPlayer."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, QRectF, QUrl
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from ..theme import Colors, Dims, Fonts

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 optional at import time
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput  # type: ignore
    _HAS_MULTIMEDIA = True
except Exception:  # pragma: no cover
    _HAS_MULTIMEDIA = False  # type: ignore


def _bgr_to_qimage(frame) -> QImage:
    """Convert a BGR/BGRA numpy frame to a QImage (copied, owns its memory)."""
    if frame is None:
        return QImage()
    if frame.ndim == 2:
        h, w = frame.shape
        return QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, ch = frame.shape
    if ch == 4:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        return QImage(rgb.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class VideoPreviewWidget(QWidget):
    """Frame-accurate video / image preview with playback controls."""

    frame_changed = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 200)
        self.setFocusPolicy(Qt.StrongFocus)

        self._cap = None
        self._is_image = False
        self._pixmap: Optional[QPixmap] = None
        self._frame_count = 0
        self._fps = 0.0
        self._frame_w = 0
        self._frame_h = 0
        self._current = 0
        self._path = ""

        # Zoom (1.0 = fit-to-widget, >1.0 = zoomed in)
        self._zoom = 1.0
        self._zoom_min = 0.1
        self._zoom_max = 8.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._playing = False

        # Audio via QMediaPlayer (optional)
        self._media_player: Optional[object] = None
        self._audio_output: Optional[object] = None
        if _HAS_MULTIMEDIA:
            try:
                self._audio_output = QAudioOutput(self)
                self._media_player = QMediaPlayer(self)
                self._media_player.setAudioOutput(self._audio_output)
            except Exception:
                self._media_player = None
                self._audio_output = None

    # ---------------------------------------------------------------- load
    def load_file(self, path: str) -> bool:
        self.cleanup()
        self._zoom = 1.0
        if cv2 is None or not path or not os.path.isfile(path):
            self.update()
            return False
        self._path = path
        ext = os.path.splitext(path)[1].lower()
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        if ext in image_exts:
            frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                return False
            if frame.dtype != "uint8" and np is not None:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            self._is_image = True
            self._frame_count = 1
            self._fps = 0.0
            self._frame_h, self._frame_w = frame.shape[:2]
            self._pixmap = QPixmap.fromImage(_bgr_to_qimage(frame))
            self._current = 0
            self.update()
            self.frame_changed.emit(0)
            return True

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        self._is_image = False
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Set up audio player.
        if self._media_player is not None:
            try:
                self._media_player.setSource(QUrl.fromLocalFile(path))
            except Exception:
                pass
        self.seek_frame(0)
        return True

    # ---------------------------------------------------------------- info
    def get_frame_count(self) -> int:
        return self._frame_count

    def get_fps(self) -> float:
        return self._fps

    def get_frame_width(self) -> int:
        return self._frame_w

    def get_frame_height(self) -> int:
        return self._frame_h

    def current_frame(self) -> int:
        return self._current

    def current_frame_image(self) -> QImage:
        if self._pixmap is None or self._pixmap.isNull():
            return QImage()
        return self._pixmap.toImage()

    # ---------------------------------------------------------------- playback
    def play(self) -> None:
        if self._is_image or self._cap is None:
            return
        fps = self._fps if self._fps > 0 else 25.0
        self._timer.start(int(1000 / fps))
        self._playing = True
        # Start audio player synced to current position.
        if self._media_player is not None:
            try:
                pos_ms = int(self._current * 1000 / fps)
                self._media_player.setPosition(pos_ms)
                self._media_player.play()
            except Exception:
                pass

    def pause(self) -> None:
        self._timer.stop()
        self._playing = False
        if self._media_player is not None:
            try:
                self._media_player.pause()
            except Exception:
                pass

    def stop(self) -> None:
        self.pause()
        if self._media_player is not None:
            try:
                self._media_player.stop()
            except Exception:
                pass
        self.seek_frame(0)

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def _advance(self) -> None:
        if self._current + 1 >= self._frame_count:
            self.pause()
            return
        self.seek_frame(self._current + 1)

    def step_forward(self) -> None:
        self.seek_frame(min(self._current + 1, max(0, self._frame_count - 1)))

    def step_backward(self) -> None:
        self.seek_frame(max(self._current - 1, 0))

    def seek_frame(self, idx: int) -> None:
        if self._is_image:
            self._current = 0
            self.frame_changed.emit(0)
            self.update()
            return
        if self._cap is None or self._frame_count <= 0:
            return
        idx = max(0, min(idx, self._frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if ok:
            self._pixmap = QPixmap.fromImage(_bgr_to_qimage(frame))
            self._current = idx
            self.frame_changed.emit(idx)
            self.update()

    # ---------------------------------------------------------------- events
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Zoom in/out on the preview with the mouse wheel."""
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self._zoom = max(self._zoom_min, min(self._zoom_max, self._zoom * factor))
        self.update()
        event.accept()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(Colors.PREVIEW_BG))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(Colors.TEXT_SECONDARY))
            font = painter.font()
            font.setFamily(Fonts.FAMILY_PRIMARY)
            font.setPointSize(Fonts.SIZE_H2)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "No media loaded")
            painter.end()
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Fit-to-widget base size, then apply zoom.
        base = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        zoomed_w = int(base.width() * self._zoom)
        zoomed_h = int(base.height() * self._zoom)
        scaled = self._pixmap.scaled(zoomed_w, zoomed_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

        font = painter.font()
        font.setFamily(Fonts.FAMILY_MONO)
        font.setPointSize(Fonts.SIZE_SMALL)
        painter.setFont(font)
        painter.setPen(QColor(Colors.TEXT_PRIMARY))

        # Frame counter (top-right).
        if not self._is_image:
            counter = f"{self._current + 1} / {self._frame_count}"
            painter.drawText(
                QRectF(0, 8, self.width() - 12, 18), Qt.AlignRight | Qt.AlignTop, counter
            )
        # Resolution (bottom-right).
        res = f"{self._frame_w}×{self._frame_h}"
        if self._zoom != 1.0:
            res += f"  {self._zoom:.1f}×"
        painter.drawText(
            QRectF(0, self.height() - 26, self.width() - 12, 18),
            Qt.AlignRight | Qt.AlignBottom,
            res,
        )
        painter.end()

    # ---------------------------------------------------------------- cleanup
    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Display an externally provided pixmap (e.g. a preview result)."""
        self._pixmap = pixmap
        if pixmap is not None and not pixmap.isNull():
            self._frame_w = pixmap.width()
            self._frame_h = pixmap.height()
        self.update()

    def cleanup(self) -> None:
        self.pause()
        if self._media_player is not None:
            try:
                self._media_player.stop()
                self._media_player.setSource(QUrl())
            except Exception:
                pass
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._pixmap = None
        self._is_image = False
        self._frame_count = 0
        self._fps = 0.0
        self._frame_w = 0
        self._frame_h = 0
        self._current = 0
        self._zoom = 1.0
