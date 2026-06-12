"""
SeedVR2 GUI – Paths & Configuration Settings Window.

A persistent (non-modal) QDialog that owns all *system-level* path fields
(Python executable, SeedVR2 folder, FFmpeg, and Models directory).

The dialog is reorganised into a "Directory Setup" tab so the user can see
and browse all four paths at once, exactly as required by the workspace
directive (Task 4 – Settings / Directory Setup tab).

Input / Output paths have moved to the separate FoldersDialog, which is
opened via the "📁 Folders" button on the main screen.

Persistence is now handled by config.json (via config_manager) in addition
to the legacy QSettings store, ensuring portability across folder moves.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSettings, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from gui.config_manager import load_config, save_config, DEFAULT_PATHS
except ImportError:
    from config_manager import load_config, save_config, DEFAULT_PATHS  # type: ignore[no-redef]

try:
    from gui.worker import DEFAULT_PYTHON_EXE
except ImportError:
    from worker import DEFAULT_PYTHON_EXE  # type: ignore[no-redef]


_SETTINGS_ORG = "SeedVR2GUI"
_SETTINGS_APP = "SeedVR2_GUI"


# ---------------------------------------------------------------------------
# Helpers
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
    """Persistent system-paths / configuration dialog.

    Attributes exposed for MainWindow to read
    -----------------------------------------
    python_exe_edit      QLineEdit   - Python interpreter path
    seedvr2_folder_edit  QLineEdit   - Folder containing inference_cli.py
    model_dir_edit       QLineEdit   - Models directory (SEEDVR2)
    ffmpeg_path_edit     QLineEdit   - FFmpeg executable path
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paths & Configuration")
        self.setMinimumWidth(660)
        # Persistent, non-modal – stays open alongside the main window.
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
        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(12, 12, 12, 12)

        # ── Tab widget ─────────────────────────────────────────────────
        tabs = QTabWidget()
        root_layout.addWidget(tabs)

        # ── Tab 1: Directory Setup ─────────────────────────────────────
        dir_tab = QWidget()
        dir_layout = QVBoxLayout(dir_tab)
        dir_layout.setSpacing(10)
        dir_layout.setContentsMargins(8, 8, 8, 8)

        g, f = _make_group("Directory Setup")

        # Python Executable
        self.python_exe_edit = QLineEdit()
        self.python_exe_edit.setPlaceholderText(DEFAULT_PYTHON_EXE)
        browse_py_btn = QPushButton("Browse...")
        browse_py_btn.clicked.connect(self._browse_python)
        py_row = QHBoxLayout()
        py_row.addWidget(self.python_exe_edit)
        py_row.addWidget(browse_py_btn)
        f.addRow("Python Executable:", _wrap(py_row))

        # SeedVR2 Script Folder
        self.seedvr2_folder_edit = QLineEdit()
        self.seedvr2_folder_edit.setPlaceholderText(
            "Folder containing inference_cli.py..."
        )
        browse_sv_btn = QPushButton("Browse...")
        browse_sv_btn.clicked.connect(self._browse_seedvr2_folder)
        sv_row = QHBoxLayout()
        sv_row.addWidget(self.seedvr2_folder_edit)
        sv_row.addWidget(browse_sv_btn)
        f.addRow("SeedVR2 Script Folder:", _wrap(sv_row))

        # FFmpeg Executable
        self.ffmpeg_path_edit = QLineEdit()
        self.ffmpeg_path_edit.setPlaceholderText(
            DEFAULT_PATHS.get("ffmpeg_path", "Path to ffmpeg executable...")
        )
        browse_ff_btn = QPushButton("Browse...")
        browse_ff_btn.clicked.connect(self._browse_ffmpeg)
        ff_row = QHBoxLayout()
        ff_row.addWidget(self.ffmpeg_path_edit)
        ff_row.addWidget(browse_ff_btn)
        f.addRow("FFmpeg Executable:", _wrap(ff_row))

        # Models Directory
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText(
            DEFAULT_PATHS.get("models_dir", "models/SEEDVR2/")
        )
        browse_md_btn = QPushButton("Browse...")
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
        f.addRow("Models Directory:", _wrap(md_row))

        dir_layout.addWidget(g)

        # Root-dir info label
        try:
            from gui.config_manager import ROOT_DIR, CONFIG_PATH
        except ImportError:
            from config_manager import ROOT_DIR, CONFIG_PATH  # type: ignore[no-redef]

        info = QLabel(
            "<b>Installation root:</b> {}<br>"
            "<b>config.json:</b> {}".format(ROOT_DIR, CONFIG_PATH)
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#888; font-size:10px; padding:4px 0;")
        dir_layout.addWidget(info)
        dir_layout.addStretch(1)

        tabs.addTab(dir_tab, "Directory Setup")

        # ── Hint label ─────────────────────────────────────────────────
        hint = QLabel(
            "Settings are saved to <b>config.json</b> when you click "
            "<b>Save &amp; Close</b>.<br>"
            "Input / Output paths are managed via the <b>Folders</b> button "
            "on the main screen."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        root_layout.addWidget(hint)

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
        root_layout.addLayout(btn_row)

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
        start = self.python_exe_edit.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Python Executable",
            start,
            "Executables (*.exe python python3);;All Files (*)",
        )
        if path:
            self.python_exe_edit.setText(path)

    def _browse_seedvr2_folder(self) -> None:
        start = self.seedvr2_folder_edit.text().strip() or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select SeedVR2 Folder (containing inference_cli.py)", start
        )
        if path:
            self.seedvr2_folder_edit.setText(path)

    def _browse_ffmpeg(self) -> None:
        start = self.ffmpeg_path_edit.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FFmpeg Executable",
            start,
            "Executables (ffmpeg ffmpeg.exe *.exe);;All Files (*)",
        )
        if path:
            self.ffmpeg_path_edit.setText(path)

    def _browse_model_dir(self) -> None:
        start = self.model_dir_edit.text().strip() or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Models Directory", start
        )
        if path:
            self.model_dir_edit.setText(path)

    # ------------------------------------------------------------------
    # Settings persistence (config.json + QSettings for back-compat)
    # ------------------------------------------------------------------

    def load_settings(self) -> None:
        """Restore fields from config.json (with QSettings fallback)."""
        cfg = load_config()
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

        def _get(cfg_key: str, qs_key: str, default: str = "") -> str:
            v = cfg.get(cfg_key, "")
            if not v:
                v = s.value(qs_key, default, type=str)
            return v

        self.python_exe_edit.setText(
            _get("python_exe", "python_exe", DEFAULT_PYTHON_EXE)
        )
        self.seedvr2_folder_edit.setText(
            _get("seedvr2_folder", "seedvr2_folder", "")
        )
        self.ffmpeg_path_edit.setText(
            _get("ffmpeg_path", "ffmpeg_path", "")
        )
        self.model_dir_edit.setText(
            _get("models_dir", "model_dir", "")
        )

    def save_settings(self) -> None:
        """Write current field values to config.json (and QSettings)."""
        cfg = load_config()
        cfg["python_exe"] = self.python_exe_edit.text().strip()
        cfg["seedvr2_folder"] = self.seedvr2_folder_edit.text().strip()
        cfg["ffmpeg_path"] = self.ffmpeg_path_edit.text().strip()
        cfg["models_dir"] = self.model_dir_edit.text().strip()
        save_config(cfg)

        # Keep QSettings in sync for any code that still reads from it.
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue("python_exe", cfg["python_exe"])
        s.setValue("seedvr2_folder", cfg["seedvr2_folder"])
        s.setValue("ffmpeg_path", cfg["ffmpeg_path"])
        s.setValue("model_dir", cfg["models_dir"])
