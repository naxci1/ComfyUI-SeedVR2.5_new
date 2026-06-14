"""Playback transport controls row."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QWidget

from .button3d import Button3D


class PlaybackControls(QWidget):
    """Compact playback and trim transport bar."""

    play_pause_toggled = Signal(bool)
    prev_frame_requested = Signal()
    next_frame_requested = Signal()
    mute_toggled = Signal(bool)
    trim_in_requested = Signal()
    trim_out_requested = Signal()
    trim_clear_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._playing = False
        self._muted = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.play_pause_btn = Button3D("▶", variant="default", parent=self)
        self.prev_btn = Button3D("⏮", variant="ghost", parent=self)
        self.next_btn = Button3D("⏭", variant="ghost", parent=self)
        self.mute_btn = Button3D("🔊", variant="ghost", parent=self)
        self.trim_in_btn = Button3D("[", variant="ghost", parent=self)
        self.trim_in_btn.setToolTip("IN [")
        self.trim_clear_btn = Button3D("✕", variant="ghost", parent=self)
        self.trim_clear_btn.setToolTip("Clear IN/OUT — reset to full range")
        self.trim_out_btn = Button3D("]", variant="ghost", parent=self)
        self.trim_out_btn.setToolTip("OUT ]")

        self.play_pause_btn.clicked.connect(self._toggle_play_pause)
        self.prev_btn.clicked.connect(self.prev_frame_requested.emit)
        self.next_btn.clicked.connect(self.next_frame_requested.emit)
        self.mute_btn.clicked.connect(self._toggle_mute)
        self.trim_in_btn.clicked.connect(self.trim_in_requested.emit)
        self.trim_clear_btn.clicked.connect(self.trim_clear_requested.emit)
        self.trim_out_btn.clicked.connect(self.trim_out_requested.emit)

        for button in (
            self.play_pause_btn,
            self.prev_btn,
            self.next_btn,
            self.mute_btn,
            self.trim_in_btn,
            self.trim_clear_btn,
            self.trim_out_btn,
        ):
            layout.addWidget(button)
        layout.addStretch(1)

    def _toggle_play_pause(self) -> None:
        self.set_playing(not self._playing)
        self.play_pause_toggled.emit(self._playing)

    def _toggle_mute(self) -> None:
        self.set_muted(not self._muted)
        self.mute_toggled.emit(self._muted)

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self.play_pause_btn.setText("⏸" if playing else "▶")

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        self.mute_btn.setText("🔇" if muted else "🔊")
