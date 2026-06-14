"""IN/OUT trim selection timeline with a cv2-extracted thumbnail strip."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, QObject, QThread, QRectF, QPoint
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import QWidget

from ..theme import Colors, Dims, Fonts

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


class _ThumbnailWorker(QObject):
    """Extracts one thumbnail every ~2 seconds of video, off the GUI thread."""

    ready = Signal(list)  # list[QImage]

    def __init__(self, path: str, fps: float, frame_count: int) -> None:
        super().__init__()
        self._path = path
        self._fps = fps if fps > 0 else 25.0
        self._frame_count = frame_count

    def run(self) -> None:
        thumbs: List[QImage] = []
        if cv2 is not None and self._frame_count > 0:
            try:
                cap = cv2.VideoCapture(self._path)
                if cap.isOpened():
                    step = max(1, int(self._fps * 2))
                    idx = 0
                    while idx < self._frame_count:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = cv2.resize(frame, (80, 45))
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = QImage(rgb.data, 80, 45, 3 * 80, QImage.Format_RGB888).copy()
                        thumbs.append(img)
                        idx += step
                cap.release()
            except Exception:
                thumbs = []
        self.ready.emit(thumbs)


class TrimTimeline(QWidget):
    """Visual IN/OUT trim timeline with draggable handles and a playhead."""

    in_point_changed = Signal(int)
    out_point_changed = Signal(int)
    playhead_moved = Signal(int)

    _HANDLE_W = 14
    _HANDLE_H = 24
    _GRAB = 20

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(Dims.TRIM_HEIGHT + 24)
        self.setMouseTracking(True)

        self._frame_count = 0
        self._fps = 0.0
        self._in = 0
        self._out = 0
        self._playhead = 0
        self._thumbs: List[QImage] = []
        self._drag: Optional[str] = None  # "in" | "out" | "playhead"

        self._thumb_thread: Optional[QThread] = None
        self._thumb_worker: Optional[_ThumbnailWorker] = None

    # ---------------------------------------------------------------- load
    def load_video(self, path: str) -> None:
        self._thumbs = []
        self._frame_count = 0
        self._fps = 0.0
        if cv2 is not None and path:
            try:
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                cap.release()
            except Exception:
                pass
        self._in = 0
        self._out = max(0, self._frame_count - 1)
        self._playhead = 0
        self.update()
        self._start_thumbnails(path)

    def _start_thumbnails(self, path: str) -> None:
        if cv2 is None or self._frame_count <= 0:
            return
        self._stop_thumbnails()
        self._thumb_thread = QThread()
        self._thumb_worker = _ThumbnailWorker(path, self._fps, self._frame_count)
        self._thumb_worker.moveToThread(self._thumb_thread)
        self._thumb_thread.started.connect(self._thumb_worker.run)
        self._thumb_worker.ready.connect(self._on_thumbs_ready)
        self._thumb_worker.ready.connect(self._thumb_thread.quit)
        self._thumb_thread.start()

    def _stop_thumbnails(self) -> None:
        if self._thumb_thread is not None:
            self._thumb_thread.quit()
            self._thumb_thread.wait(200)
            self._thumb_thread = None
            self._thumb_worker = None

    def _on_thumbs_ready(self, thumbs: list) -> None:
        self._thumbs = thumbs
        self.update()

    # ---------------------------------------------------------------- api
    def set_in_point(self, frame: int) -> None:
        frame = max(0, min(frame, self._out))
        if frame != self._in:
            self._in = frame
            self.in_point_changed.emit(frame)
            self.update()

    def set_out_point(self, frame: int) -> None:
        frame = max(self._in, min(frame, max(0, self._frame_count - 1)))
        if frame != self._out:
            self._out = frame
            self.out_point_changed.emit(frame)
            self.update()

    def set_playhead(self, frame: int) -> None:
        frame = max(0, min(frame, max(0, self._frame_count - 1)))
        if frame != self._playhead:
            self._playhead = frame
            self.playhead_moved.emit(frame)
            self.update()

    def set_full_range(self) -> None:
        self.set_in_point(0)
        self.set_out_point(max(0, self._frame_count - 1))

    def get_selected_range(self) -> Tuple[int, int]:
        return self._in, self._out

    def get_selected_frame_count(self) -> int:
        return max(0, self._out - self._in + 1)

    def get_selected_duration(self) -> float:
        if self._fps <= 0:
            return 0.0
        return self.get_selected_frame_count() / self._fps

    def is_full_range(self) -> bool:
        return self._in == 0 and self._out == max(0, self._frame_count - 1)

    def frame_to_timecode(self, frame: int) -> str:
        fps = self._fps if self._fps > 0 else 25.0
        total_seconds = frame / fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        f = int(frame % round(fps)) if fps else 0
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"
        return f"{m:02d}:{s:02d}:{f:02d}"

    # ---------------------------------------------------------------- geometry
    def _track_rect(self) -> QRectF:
        m = self._HANDLE_W
        return QRectF(m, 20, self.width() - 2 * m, Dims.TRIM_HEIGHT)

    def _frame_from_x(self, x: float) -> int:
        track = self._track_rect()
        if track.width() <= 0 or self._frame_count <= 1:
            return 0
        frac = (x - track.left()) / track.width()
        frac = max(0.0, min(1.0, frac))
        return int(round(frac * (self._frame_count - 1)))

    def _x_from_frame(self, idx: int) -> float:
        track = self._track_rect()
        if self._frame_count <= 1:
            return track.left()
        return track.left() + track.width() * (idx / (self._frame_count - 1))

    # ---------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self._frame_count <= 0:
            return
        x = event.position().x()
        in_x = self._x_from_frame(self._in)
        out_x = self._x_from_frame(self._out)
        if abs(x - in_x) <= self._GRAB:
            self._drag = "in"
        elif abs(x - out_x) <= self._GRAB:
            self._drag = "out"
        else:
            self._drag = "playhead"
            self.set_playhead(self._frame_from_x(x))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is None or self._frame_count <= 0:
            return
        frame = self._frame_from_x(event.position().x())
        if self._drag == "in":
            self.set_in_point(frame)
        elif self._drag == "out":
            self.set_out_point(frame)
        else:
            self.set_playhead(frame)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = None

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        track = self._track_rect()

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(Colors.SCRUB_TRACK))
        painter.drawRoundedRect(track, 4, 4)

        # Thumbnail strip.
        if self._thumbs:
            painter.save()
            painter.setClipRect(track)
            tw = track.width() / len(self._thumbs)
            for i, img in enumerate(self._thumbs):
                pix = QPixmap.fromImage(img).scaled(
                    int(tw) + 1, int(track.height()),
                    Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation,
                )
                painter.drawPixmap(int(track.left() + i * tw), int(track.top()), pix)
            painter.restore()

        in_x = self._x_from_frame(self._in)
        out_x = self._x_from_frame(self._out)

        # Darken regions outside selection.
        dark = QColor(0, 0, 0, 120)
        painter.setBrush(dark)
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(track.left(), track.top(), in_x - track.left(), track.height()))
        painter.drawRect(QRectF(out_x, track.top(), track.right() - out_x, track.height()))

        # Selection overlay.
        painter.setBrush(QColor(Colors.ACCENT_GLOW))
        painter.drawRect(QRectF(in_x, track.top(), out_x - in_x, track.height()))

        # IN handle (green).
        self._draw_handle(painter, in_x, QColor(Colors.TRIM_HANDLE_IN))
        # OUT handle (red).
        self._draw_handle(painter, out_x, QColor(Colors.TRIM_HANDLE_OUT))

        # Playhead.
        ph_x = self._x_from_frame(self._playhead)
        painter.setPen(QPen(QColor(Colors.TEXT_PRIMARY), 2))
        painter.drawLine(int(ph_x), int(track.top()), int(ph_x), int(track.bottom()))
        tri = QPolygon([
            QPoint(int(ph_x) - 5, int(track.top()) - 6),
            QPoint(int(ph_x) + 5, int(track.top()) - 6),
            QPoint(int(ph_x), int(track.top())),
        ])
        painter.setBrush(QColor(Colors.TEXT_PRIMARY))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(tri)

        # Timecode markers below track.
        painter.setPen(QColor(Colors.TEXT_SECONDARY))
        font = painter.font()
        font.setFamily(Fonts.FAMILY_MONO)
        font.setPointSize(Fonts.SIZE_TINY)
        painter.setFont(font)
        marks = 6
        for i in range(marks + 1):
            frac = i / marks
            frame = int(frac * max(0, self._frame_count - 1))
            x = self._x_from_frame(frame)
            painter.drawText(
                QRectF(x - 24, track.bottom() + 2, 48, 14),
                Qt.AlignCenter,
                self.frame_to_timecode(frame),
            )
        painter.end()

    def _draw_handle(self, painter: QPainter, x: float, color: QColor) -> None:
        track = self._track_rect()
        rect = QRectF(x - self._HANDLE_W / 2, track.top() - 2, self._HANDLE_W, self._HANDLE_H)
        glow = QColor(color)
        glow.setAlpha(80)
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawRoundedRect(rect.adjusted(-2, -2, 2, 2), 4, 4)
        painter.setBrush(color)
        painter.drawRoundedRect(rect, 3, 3)

    def cleanup(self) -> None:
        self._stop_thumbnails()
