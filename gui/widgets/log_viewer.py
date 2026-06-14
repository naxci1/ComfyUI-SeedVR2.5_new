"""Modeless streaming log viewer dialog."""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPlainTextEdit, QVBoxLayout

from ..theme import Dims
from .button3d import Button3D


class LogViewer(QDialog):
    """Non-blocking log window with a bounded line buffer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Process Log")
        self.setModal(False)
        self.resize(860, 520)
        self._lines: deque[str] = deque(maxlen=5000)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Dims.PADDING_XL, Dims.PADDING_LG, Dims.PADDING_XL, Dims.PADDING_LG)
        layout.setSpacing(Dims.PADDING_MD)

        self.editor = QPlainTextEdit(self)
        self.editor.setReadOnly(True)
        layout.addWidget(self.editor, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.copy_btn = Button3D("Copy All", variant="default", parent=self)
        self.clear_btn = Button3D("Clear", variant="default", parent=self)
        self.close_btn = Button3D("Close", variant="primary", parent=self)
        self.copy_btn.clicked.connect(lambda: self.editor.selectAll() or self.editor.copy())
        self.clear_btn.clicked.connect(self.clear)
        self.close_btn.clicked.connect(self.close)
        buttons.addWidget(self.copy_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    @Slot(str)
    def append_line(self, text: str) -> None:
        lines = text.splitlines() or [text]
        changed = False
        for line in lines:
            if line == "" and not text:
                continue
            self._lines.append(line)
            changed = True
        if not changed:
            return
        if len(self._lines) >= 5000:
            self.editor.setPlainText("\n".join(self._lines))
        else:
            for line in lines:
                if line != "" or text.endswith("\n"):
                    self.editor.appendPlainText(line)
        scrollbar = self.editor.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self) -> None:
        self._lines.clear()
        self.editor.clear()
