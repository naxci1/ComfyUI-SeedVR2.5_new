"""Application path configuration dialog."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import Dims
from .button3d import Button3D
from .toggle_switch import ToggleSwitch

try:
    from gui.config_manager import load_config, save_config
except ImportError:  # pragma: no cover - direct-script execution fallback
    from config_manager import load_config, save_config  # type: ignore[no-redef]


class SettingsDialog(QDialog):
    """Edits runtime paths stored by ``config_manager``."""

    config_saved = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.setModal(True)
        self.setMinimumWidth(560)

        self._config = load_config()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Dims.PADDING_XL, Dims.PADDING_LG, Dims.PADDING_XL, Dims.PADDING_LG)
        layout.setSpacing(Dims.PADDING_MD)

        title = QLabel("Runtime Paths")
        title.setProperty("role", "h1")
        layout.addWidget(title)

        form_widget = QWidget(self)
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(Dims.PADDING_MD)

        self.python_edit = QLineEdit(self._config.get("python_exe", ""), self)
        self.seedvr2_edit = QLineEdit(self._config.get("seedvr2_folder", ""), self)
        self.ffmpeg_edit = QLineEdit(self._config.get("ffmpeg_path", ""), self)
        self.models_edit = QLineEdit(self._config.get("models_dir", ""), self)
        self.temp_edit = QLineEdit(self._config.get("temp_dir", ""), self)
        alarm_enabled = str(self._config.get("alarm_enabled", "true")).strip().lower() not in {"0", "false", "no", "off"}
        self.alarm_toggle = ToggleSwitch("", alarm_enabled, self)

        form.addRow("Python executable", self._browse_row(self.python_edit, True))
        form.addRow("SeedVR2 folder", self._browse_row(self.seedvr2_edit, False))
        form.addRow("FFmpeg binary", self._browse_row(self.ffmpeg_edit, True))
        form.addRow("Models directory", self._browse_row(self.models_edit, False))
        form.addRow("Temp directory", self._browse_row(self.temp_edit, False))
        form.addRow("Alarm sounds", self.alarm_toggle)
        layout.addWidget(form_widget)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.save_btn = Button3D("Save", variant="primary", parent=self)
        self.cancel_btn = Button3D("Cancel", variant="default", parent=self)
        self.save_btn.clicked.connect(self._save)
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)

    def _browse_row(self, line_edit: QLineEdit, file_mode: bool) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_SM)
        button = Button3D("Browse", variant="default", parent=row)
        button.clicked.connect(lambda: self._browse(line_edit, file_mode))
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return row

    def _browse(self, target: QLineEdit, file_mode: bool) -> None:
        if file_mode:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", target.text() or "")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select folder", target.text() or "")
        if path:
            target.setText(path)

    def _save(self) -> None:
        self._config.update(
            {
                "python_exe": self.python_edit.text().strip(),
                "seedvr2_folder": self.seedvr2_edit.text().strip(),
                "ffmpeg_path": self.ffmpeg_edit.text().strip(),
                "models_dir": self.models_edit.text().strip(),
                "temp_dir": self.temp_edit.text().strip(),
                "alarm_enabled": "true" if self.alarm_toggle.isChecked() else "false",
            }
        )
        save_config(self._config)
        self.config_saved.emit(dict(self._config))
        self.accept()
