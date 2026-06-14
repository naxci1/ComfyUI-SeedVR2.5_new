"""Runtime error dialog with collapsible traceback details."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import Anim, Colors, Dims
from .button3d import Button3D


class ErrorDialog(QDialog):
    """Shows a short error message with optional full traceback details."""

    def __init__(self, parent=None, message: str = "Runtime Error", details: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Runtime Error")
        self.setModal(True)
        self.resize(720, 420)

        self._details_expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Dims.PADDING_XL, Dims.PADDING_LG, Dims.PADDING_XL, Dims.PADDING_LG)
        layout.setSpacing(Dims.PADDING_MD)

        self.message_label = QLabel(message, self)
        self.message_label.setWordWrap(True)
        self.message_label.setProperty("role", "h1")
        layout.addWidget(self.message_label)

        self.summary_label = QLabel("An unexpected error occurred. You can copy the full traceback below.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(self.summary_label)

        self.toggle_btn = Button3D("Show Details", variant="default", parent=self)
        self.toggle_btn.clicked.connect(self._toggle_details)
        layout.addWidget(self.toggle_btn)

        self.details_container = QWidget(self)
        self.details_layout = QVBoxLayout(self.details_container)
        self.details_layout.setContentsMargins(0, 0, 0, 0)
        self.details_layout.setSpacing(0)
        self.details_text = QTextEdit(self.details_container)
        self.details_text.setReadOnly(True)
        self.details_text.setPlainText(details)
        self.details_container.setMaximumHeight(0)
        self.details_layout.addWidget(self.details_text)
        layout.addWidget(self.details_container)

        self._details_anim = QPropertyAnimation(self.details_container, b"maximumHeight", self)
        self._details_anim.setDuration(Anim.NORMAL)
        self._details_anim.setEasingCurve(QEasingCurve.OutCubic)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.copy_btn = Button3D("Copy to Clipboard", variant="default", parent=self)
        self.close_btn = Button3D("Close", variant="primary", parent=self)
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        self.close_btn.clicked.connect(self.accept)
        buttons.addWidget(self.copy_btn)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    def _toggle_details(self) -> None:
        self._details_expanded = not self._details_expanded
        self.toggle_btn.setText("Hide Details" if self._details_expanded else "Show Details")
        self._details_anim.stop()
        self._details_anim.setStartValue(self.details_container.maximumHeight())
        self._details_anim.setEndValue(220 if self._details_expanded else 0)
        self._details_anim.start()

    def _copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.details_text.toPlainText())

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        try:
            import winsound

            for _ in range(3):
                winsound.Beep(1000, 250)
        except Exception:
            pass
