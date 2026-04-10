"""
SeedVR2 GUI – Paths & Configuration Settings Window.

A persistent (non-modal) QDialog that owns all path/directory fields.
MainWindow opens it via the "⚙ Settings" button and reads the field values
via the public widget attributes (python_exe_edit, seedvr2_folder_edit, etc.).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from gui.worker import DEFAULT_PYTHON_EXE
except ImportError:
    from worker import DEFAULT_PYTHON_EXE  # type: ignore[no-redef]


_SETTINGS_ORG = "SeedVR2GUI"
_SETTINGS_APP = "SeedVR2_GUI"


# ---------------------------------------------------------------------------
# Helpers (local)
# ---------------------------------------------------------------------------

def _make_group(title: str) -> tuple[QGroupBox, QFormLayout]:
    box = QGroupBox(title)
    layout = QFormLayout()
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    layout.setHorizontalSpacing(12)
    layout.setVerticalSpacing(6)
    layout.setContentsMargins(10, 6, 10, 10)
    box.setLayout(layout)
    return box, layout


def _wrap(h_layout: QHBoxLayout) -> QWidget:
    w = QWidget()
    w.setLayout(h_layout)
    return w


# ---------------------------------------------------------------------------
# SettingsWindow
# ---------------------------------------------------------------------------

class SettingsWindow(QDialog):
    """Persistent settings / paths dialog.

    Attributes exposed for MainWindow to read
    -----------------------------------------
    python_exe_edit      QLineEdit
    seedvr2_folder_edit  QLineEdit
    input_mode_combo     QComboBox  ("File" / "Folder")
    input_edit           QLineEdit
    output_edit          QLineEdit
    model_dir_edit       QLineEdit
    """

    # Emitted when the user picks an input FILE (not folder) via Browse.
    # MainWindow connects this to its _load_preview() slot.
    input_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paths & Configuration")
        self.setMinimumWidth(620)
        # Persistent, non-modal – stays open alongside the main window
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowTitleHint
        )
        self._build_ui()
        self.load_settings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Environment ────────────────────────────────────────────────
        g, f = _make_group("Environment")

        self.python_exe_edit = QLineEdit()
        self.python_exe_edit.setPlaceholderText(DEFAULT_PYTHON_EXE)
        browse_py_btn = QPushButton("Browse…")
        browse_py_btn.clicked.connect(self._browse_python)
        py_row = QHBoxLayout()
        py_row.addWidget(self.python_exe_edit)
        py_row.addWidget(browse_py_btn)
        f.addRow("Python Executable:", _wrap(py_row))

        self.seedvr2_folder_edit = QLineEdit()
        self.seedvr2_folder_edit.setPlaceholderText("Folder containing inference_cli.py…")
        browse_sv_btn = QPushButton("Browse…")
        browse_sv_btn.clicked.connect(self._browse_seedvr2_folder)
        sv_row = QHBoxLayout()
        sv_row.addWidget(self.seedvr2_folder_edit)
        sv_row.addWidget(browse_sv_btn)
        f.addRow("SeedVR2 Folder:", _wrap(sv_row))

        layout.addWidget(g)

        # ── Paths ──────────────────────────────────────────────────────
        g2, f2 = _make_group("Paths")

        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItems(["File", "Folder"])
        self.input_mode_combo.setMaximumWidth(72)
        self.input_mode_combo.setToolTip(
            "File: single video/image  |  Folder: batch-process all videos"
        )
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Path to video, image, or directory…")
        browse_input_btn = QPushButton("📂")
        browse_input_btn.setFixedWidth(32)
        browse_input_btn.setToolTip("Browse for input file or folder")
        browse_input_btn.setAccessibleName("Browse input")
        browse_input_btn.clicked.connect(self._browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_mode_combo)
        input_row.addWidget(self.input_edit)
        input_row.addWidget(browse_input_btn)
        f2.addRow("Input:", _wrap(input_row))

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Optional – leave blank for auto")
        browse_out_btn = QPushButton("Browse…")
        browse_out_btn.clicked.connect(self._browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse_out_btn)
        f2.addRow("Output Path:", _wrap(out_row))

        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("Optional – defaults to models/SEEDVR2/")
        browse_md_btn = QPushButton("Browse…")
        browse_md_btn.clicked.connect(self._browse_model_dir)
        fp8_btn = QPushButton("FP8/FP16")
        fp8_btn.setToolTip("Download FP8 / FP16 models from HuggingFace")
        fp8_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://huggingface.co/numz/SeedVR2_comfyUI/tree/main")
            )
        )
        gguf_btn = QPushButton("GGUF")
        gguf_btn.setToolTip("Download GGUF models from HuggingFace")
        gguf_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://huggingface.co/AInVFX/SeedVR2_comfyUI/tree/main")
            )
        )
        md_row = QHBoxLayout()
        md_row.addWidget(self.model_dir_edit)
        md_row.addWidget(browse_md_btn)
        md_row.addWidget(fp8_btn)
        md_row.addWidget(gguf_btn)
        f2.addRow("Model Directory:", _wrap(md_row))

        layout.addWidget(g2)

        # ── Hint label ─────────────────────────────────────────────────
        hint = QLabel(
            "These settings are saved automatically when you click "
            "<b>Save &amp; Close</b>."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        layout.addWidget(hint)

        # ── Button row ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        save_btn = QPushButton("Save && Close")
        save_btn.setObjectName("primary_button")
        save_btn.clicked.connect(self._save_and_close)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel)

        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    def _save_and_close(self) -> None:
        self.save_settings()
        self.hide()

    def _cancel(self) -> None:
        self.load_settings()  # discard unsaved edits
        self.hide()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Discard unsaved changes when the user closes via the X button."""
        self.load_settings()
        event.accept()

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_python(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Python Executable",
            "",
            "Executables (*.exe python python3);;All Files (*)",
        )
        if path:
            self.python_exe_edit.setText(path)

    def _browse_seedvr2_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select SeedVR2 Folder (containing inference_cli.py)", ""
        )
        if path:
            self.seedvr2_folder_edit.setText(path)

    def _browse_input(self) -> None:
        if self.input_mode_combo.currentText() == "Folder":
            path = QFileDialog.getExistingDirectory(self, "Select Input Folder", "")
            if path:
                self.input_edit.setText(path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Input File",
                "",
                "Videos & Images "
                "(*.mp4 *.avi *.mov *.mkv *.webm *.png *.jpg *.jpeg *.bmp *.tiff)"
                ";;All Files (*)",
            )
            if path:
                self.input_edit.setText(path)
                self.input_changed.emit(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if path:
            self.output_edit.setText(path)

    def _browse_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory", "")
        if path:
            self.model_dir_edit.setText(path)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def load_settings(self) -> None:
        """Restore fields from persistent QSettings storage."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self.python_exe_edit.setText(s.value("python_exe", DEFAULT_PYTHON_EXE, type=str))
        self.seedvr2_folder_edit.setText(s.value("seedvr2_folder", "", type=str))
        self.input_edit.setText(s.value("input_path", "", type=str))
        saved_mode: str = s.value("input_mode", "File", type=str)
        idx = self.input_mode_combo.findText(saved_mode)
        if idx >= 0:
            self.input_mode_combo.setCurrentIndex(idx)
        self.output_edit.setText(s.value("output_path", "", type=str))
        saved_md: str = s.value("model_dir", "", type=str)
        if saved_md:
            self.model_dir_edit.setText(saved_md)
        elif self.seedvr2_folder_edit.text():
            default_md = str(
                Path(self.seedvr2_folder_edit.text()) / "models" / "SEEDVR2"
            )
            self.model_dir_edit.setPlaceholderText(default_md)

    def save_settings(self) -> None:
        """Write current field values to persistent QSettings storage."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue("python_exe", self.python_exe_edit.text().strip())
        s.setValue("seedvr2_folder", self.seedvr2_folder_edit.text().strip())
        s.setValue("input_path", self.input_edit.text().strip())
        s.setValue("input_mode", self.input_mode_combo.currentText())
        s.setValue("output_path", self.output_edit.text().strip())
        s.setValue("model_dir", self.model_dir_edit.text().strip())
