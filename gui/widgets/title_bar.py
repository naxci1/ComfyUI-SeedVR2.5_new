"""Custom frameless-window title bar with drag-to-move and window controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ..theme import Colors, Dims, Fonts

_APP_TITLE = "1-Click SeedVR2.5 v.1.8b (by Naxci1)"
_APP_VERSION = "v1.8b"


class CustomTitleBar(QWidget):
    """A draggable frameless title bar with min / max / close buttons."""

    def __init__(self, parent_window: QWidget, title: str = _APP_TITLE, version: str = _APP_VERSION) -> None:
        super().__init__(parent_window)
        self._window = parent_window
        self._drag_pos: QPoint | None = None
        self.setFixedHeight(Dims.HEADER_HEIGHT)
        self.setObjectName("titleBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Dims.PADDING_MD, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_SM)

        logo = QLabel("◆")
        logo.setStyleSheet(f"color: {Colors.ACCENT}; font-size: {Fonts.SIZE_H1}px;")
        layout.addWidget(logo)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_H2}px;"
            f" font-weight: {Fonts.WEIGHT_SEMIBOLD};"
        )
        layout.addWidget(title_label)

        badge = QLabel(version)
        badge.setStyleSheet(
            f"color: {Colors.TEXT_ACCENT}; background-color: {Colors.ACCENT_SUBTLE};"
            f" border-radius: {Dims.CORNER_RADIUS_SM}px; padding: 1px 6px;"
            f" font-size: {Fonts.SIZE_TINY}px;"
        )
        layout.addWidget(badge)
        layout.addStretch(1)

        self._min_btn = self._make_button("—", self._on_minimize)
        self._max_btn = self._make_button("▢", self._on_maximize)
        self._close_btn = self._make_button("✕", self._on_close, close=True)
        layout.addWidget(self._min_btn)
        layout.addWidget(self._max_btn)
        layout.addWidget(self._close_btn)

    def _make_button(self, text: str, slot, close: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(Dims.HEADER_HEIGHT, Dims.HEADER_HEIGHT)
        hover = Colors.DANGER if close else Colors.SURFACE_HOVER
        btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {Colors.TEXT_SECONDARY};"
            f" font-size: {Fonts.SIZE_H2}px; }}"
            f"QPushButton:hover {{ background-color: {hover}; color: {Colors.WHITE}; }}"
        )
        btn.clicked.connect(slot)
        return btn

    # ---------------------------------------------------------------- slots
    def _on_minimize(self) -> None:
        self._window.showMinimized()

    def _on_maximize(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def _on_close(self) -> None:
        self._window.close()

    # ---------------------------------------------------------------- drag
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            if self._window.isMaximized():
                self._window.showNormal()
            self._window.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._on_maximize()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(Colors.BG_DARKEST))
        painter.setPen(QColor(Colors.BORDER))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.end()
