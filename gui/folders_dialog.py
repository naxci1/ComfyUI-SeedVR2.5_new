"""
SeedVR2 GUI – Input / Output Folders Dialog.

A small modal dialog (opened via the "📁 Folders" button on the main screen)
that manages the Input path and Output path.  These controls were previously
embedded in the global Settings window; moving them here keeps the Settings
window focused on system-level Directory Setup (Task 4 of the workspace
directive).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
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
    from gui.config_manager import load_config, save_config
except ImportError:
    from config_manager import load_config, save_config  # type: ignore[no-redef]


def _wrap(h: QHBoxLayout) -> QWidget:
    w = QWidget()
    w.setLayout(h)
    return w


class FoldersDialog(QDialog):
    """Modal dialog for managing Input / Output paths.

    Attributes exposed for MainWindow
    ----------------------------------
    input_edit       QLineEdit
    output_edit      QLineEdit
    input_mode_combo QComboBox  ("File" / "Folder")

    Signals
    -------
    input_changed(str)
        Emitted when the user selects a new input FILE via Browse.
        MainWindow connects this to its ``_load_preview`` slot.
    """

    input_changed = pyqtSignal(str)

    MEDIA_FILTER = (
        "Supported Media "
        "(*.mp4 *.mov *.mkv *.avi *.webm *.mpeg *.mpg *.m4v *.wmv *.flv *.mts *.m2ts "
        "*.png *.tif *.tiff *.jpg *.jpeg *.dpx *.exr)"
        ";;All Files (*)"
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Input / Output Paths")
        self.setMinimumWidth(560)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._build_ui()
        self.load_io_paths()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        box = QGroupBox("Input / Output Folders")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setContentsMargins(10, 6, 10, 10)
        box.setLayout(form)

        # ── Input row ──────────────────────────────────────────────────
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
        form.addRow("Input:", _wrap(input_row))

        # ── Output row ─────────────────────────────────────────────────
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Optional – leave blank for auto")

        browse_out_btn = QPushButton("Browse…")
        browse_out_btn.clicked.connect(self._browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse_out_btn)
        form.addRow("Output Path:", _wrap(out_row))

        layout.addWidget(box)

        hint = QLabel(
            "Tip: leave Output blank to auto-name the result next to the input file."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        layout.addWidget(hint)

        # ── Button row ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("primary_button")
        ok_btn.clicked.connect(self._accept)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel)

        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_input(self) -> None:
        start = self.input_edit.text().strip() or ""
        if self.input_mode_combo.currentText() == "Folder":
            path = QFileDialog.getExistingDirectory(self, "Select Input Folder", start)
            if path:
                self.input_edit.setText(path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Input File", start, self.MEDIA_FILTER
            )
            if path:
                self.input_edit.setText(path)
                self.input_changed.emit(path)

    def _browse_output(self) -> None:
        start = self.output_edit.text().strip() or ""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", start)
        if path:
            self.output_edit.setText(path)

    # ------------------------------------------------------------------
    # Accept / Cancel
    # ------------------------------------------------------------------

    def _accept(self) -> None:
        self.save_io_paths()
        self.accept()

    def _cancel(self) -> None:
        self.load_io_paths()  # discard unsaved edits
        self.reject()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.load_io_paths()  # discard unsaved edits on window-X close
        event.accept()

    # ------------------------------------------------------------------
    # Persistence (config.json)
    # ------------------------------------------------------------------

    def load_io_paths(self) -> None:
        """Restore fields from config.json."""
        cfg = load_config()
        self.input_edit.setText(cfg.get("input_path", ""))
        self.output_edit.setText(cfg.get("output_path", ""))
        mode = cfg.get("input_mode", "File")
        idx = self.input_mode_combo.findText(mode)
        if idx >= 0:
            self.input_mode_combo.setCurrentIndex(idx)

    def save_io_paths(self) -> None:
        """Write current fields back to config.json."""
        cfg = load_config()
        cfg["input_path"] = self.input_edit.text().strip()
        cfg["output_path"] = self.output_edit.text().strip()
        cfg["input_mode"] = self.input_mode_combo.currentText()
        save_config(cfg)
