"""
SeedVR2 GUI – Main Window
Topaz-style dark-mode wrapper around inference_cli.py.
"""

from __future__ import annotations

import ctypes
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt, QRectF, QSettings, QUrl, QEvent, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImageReader,
    QPixmap,
    QPainter,
    QPen,
    QStandardItem,
    QStandardItemModel,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QButtonGroup,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QSplitterHandle,
    QStackedWidget,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaMetaData, QMediaPlayer, QVideoFrame, QVideoSink
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    _MULTIMEDIA_AVAILABLE = True
except ImportError:
    _MULTIMEDIA_AVAILABLE = False

if _MULTIMEDIA_AVAILABLE:
    try:
        from gui.split_view import SplitViewWidget
    except ImportError:
        from split_view import SplitViewWidget  # type: ignore[no-redef]
else:
    SplitViewWidget = None  # type: ignore[assignment,misc]

try:
    from gui.settings_window import SettingsWindow
except ImportError:
    from settings_window import SettingsWindow  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Resource path helper (PyInstaller-compatible)
# ---------------------------------------------------------------------------

def get_resource_path(relative_path: str) -> str:
    """Return absolute path to *relative_path*, working both in development
    and when frozen with PyInstaller (where data files land in sys._MEIPASS)."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui", relative_path)


# ---------------------------------------------------------------------------
# GPU auto-detection
# ---------------------------------------------------------------------------

# Win32 flag that prevents a console window from flashing when spawning child
# processes.  Equals 0 on non-Windows so the constant is always safe to use.
_CREATE_NO_WINDOW: int = 0x08000000 if sys.platform == "win32" else 0

# Populated by _detect_gpus(); read by MainWindow to emit a startup console note.
_GPU_INIT_MSG: str = ""


def _detect_gpus() -> list[str]:
    """Return GPU entries for the checkable GPU ComboBox.

    Format: ``["Auto", "CPU", "GPU 0: NVIDIA GeForce RTX 5070 Ti", …]``

    Strategy
    --------
    1. **Primary – Windows wmic**: run
       ``wmic path win32_VideoController get name`` via ``subprocess.check_output``
       with ``shell=True``.  This is hardware-level and works regardless of whether
       torch or CUDA is installed in the GUI Python.
    2. **Enhancement – torch.cuda**: if torch is importable and reports GPUs, its
       device names overwrite the wmic names (they are more precise and correctly
       ordered by CUDA index).
    3. **Fallback**: if neither source yields GPU entries, return ``["Auto", "CPU"]``
       and emit a warning.

    Side-effect: sets module-level ``_GPU_INIT_MSG`` for startup console display.
    """
    global _GPU_INIT_MSG  # noqa: PLW0603
    entries = ["Auto", "CPU"]
    gpu_entries: list[str] = []
    wmic_names: list[str] = []

    # ── Primary: wmic (Windows-native) ─────────────────────────────────────
    try:
        flags = _CREATE_NO_WINDOW
        raw = subprocess.check_output(
            "wmic path win32_VideoController get name",
            shell=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
            creationflags=flags,
        )
        for line in raw.decode(errors="replace").splitlines():
            name = line.strip()
            if name and name.lower() != "name":
                wmic_names.append(name)

        cuda_idx = 0
        for name in wmic_names:
            if "nvidia" in name.lower():
                gpu_entries.append(f"GPU {cuda_idx}: {name}")
                cuda_idx += 1

        if gpu_entries:
            _GPU_INIT_MSG = (
                f"✅  Detected {len(gpu_entries)} CUDA GPU(s) via wmic "
                f"(all adapters: {', '.join(wmic_names)})."
            )
        elif wmic_names:
            _GPU_INIT_MSG = (
                f"⚠  wmic found adapter(s) ({', '.join(wmic_names)}) "
                "but none are NVIDIA/CUDA-capable."
            )
        else:
            _GPU_INIT_MSG = "⚠  wmic returned no video controller names."

    except FileNotFoundError:
        _GPU_INIT_MSG = "ℹ  wmic not available (non-Windows system or PATH issue)."
    except subprocess.TimeoutExpired:
        _GPU_INIT_MSG = "⚠  wmic timed out during GPU scan."
    except Exception as exc:  # noqa: BLE001
        _GPU_INIT_MSG = f"⚠  wmic GPU scan error: {exc}"
        print(f"[SeedVR2 GPU] {_GPU_INIT_MSG}", flush=True)

    # ── Enhancement: torch.cuda (overwrites wmic names with CUDA-runtime names) ─
    try:
        import torch  # noqa: PLC0415
        try:
            torch.cuda.init()
        except Exception:
            pass
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                gpu_entries = [
                    f"GPU {i}: {torch.cuda.get_device_name(i)}"
                    for i in range(count)
                ]
                _GPU_INIT_MSG = (
                    f"✅  Detected {count} CUDA device(s) via torch "
                    f"(wmic adapters: {', '.join(wmic_names) if wmic_names else 'none'})."
                )
    except ImportError:
        pass  # torch not in GUI env – wmic result stands
    except Exception:
        pass  # best-effort; keep wmic result

    if not gpu_entries:
        _GPU_INIT_MSG = (
            "⚠  No CUDA-capable GPUs found – defaulting to Auto.  " + _GPU_INIT_MSG
        ).strip()

    entries.extend(gpu_entries)
    return entries

try:
    from gui.styles import DARK_STYLESHEET
    from gui.worker import create_worker_thread, resolve_paths, DEFAULT_PYTHON_EXE
except ImportError:
    from styles import DARK_STYLESHEET  # type: ignore[no-redef]
    from worker import create_worker_thread, resolve_paths, DEFAULT_PYTHON_EXE  # type: ignore[no-redef]

SUPPORTED_VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg",
    ".m4v", ".wmv", ".flv", ".mts", ".m2ts",
}
SUPPORTED_IMAGE_EXTS = {
    ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dpx", ".exr",
}
MIN_SEQUENCE_FRAMES = 5

INPUT_DIALOG_FILTER = (
    "Supported Media (*.mp4 *.mov *.mkv *.avi *.webm *.mpeg *.mpg *.m4v *.wmv *.flv *.mts *.m2ts "
    "*.png *.tif *.tiff *.jpg *.jpeg *.dpx *.exr);;All Files (*)"
)

EXPORT_CODEC_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "MP4": {
        "H.264 High (8-bit)": {"ffmpeg": ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main (8-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main10 (10-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main10", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
        "AV1 (8-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "AV1 (10-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
    },
    "MOV": {
        "ProRes 422 Proxy": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le"], "is_10bit": True},
        "ProRes 422 LT": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "1", "-pix_fmt", "yuv422p10le"], "is_10bit": True},
        "ProRes 422": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"], "is_10bit": True},
        "ProRes 422 HQ": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], "is_10bit": True},
        "ProRes 4444 XQ": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "5", "-pix_fmt", "yuva444p12le"], "is_10bit": True},
        "QuickTime Animation (Alpha)": {"ffmpeg": ["-c:v", "qtrle", "-pix_fmt", "argb"], "is_10bit": False},
        "Uncompressed RGB (R210)": {"ffmpeg": ["-c:v", "r210"], "is_10bit": True},
    },
    "MKV": {
        "H.264 High (8-bit)": {"ffmpeg": ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main (8-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main10 (10-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main10", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
        "AV1 (8-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "AV1 (10-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
        "FFV1 (Lossless 8/10/12-bit)": {"ffmpeg": ["-c:v", "ffv1", "-level", "3"], "is_10bit": True},
        "Uncompressed YUV (V210)": {"ffmpeg": ["-c:v", "v210"], "is_10bit": True},
    },
    "WEBM": {
        "VP9 (Good)": {"ffmpeg": ["-c:v", "libvpx-vp9", "-deadline", "good"], "is_10bit": False},
        "VP9 (Best)": {"ffmpeg": ["-c:v", "libvpx-vp9", "-deadline", "best"], "is_10bit": False},
        "AV1 (8-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "AV1 (10-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
    },
}

IMAGE_SEQUENCE_PROFILES: dict[str, dict[str, Any]] = {
    "TIFF (8-bit)": {"ext": ".tiff", "ffmpeg": ["-f", "image2", "-pix_fmt", "rgb24"], "is_10bit": False},
    "TIFF (16-bit)": {"ext": ".tiff", "ffmpeg": ["-f", "image2", "-pix_fmt", "rgb48le"], "is_10bit": True},
    "PNG (8-bit)": {"ext": ".png", "ffmpeg": ["-f", "image2", "-pix_fmt", "rgb24"], "is_10bit": False},
    "PNG (16-bit)": {"ext": ".png", "ffmpeg": ["-f", "image2", "-pix_fmt", "rgb48le"], "is_10bit": True},
    "JPEG (8-bit)": {"ext": ".jpg", "ffmpeg": ["-f", "image2", "-pix_fmt", "yuvj420p"], "is_10bit": False},
    "DPX (10-bit)": {"ext": ".dpx", "ffmpeg": ["-f", "image2", "-pix_fmt", "gbrp10le"], "is_10bit": True},
    "DPX (12-bit)": {"ext": ".dpx", "ffmpeg": ["-f", "image2", "-pix_fmt", "gbrp12le"], "is_10bit": True},
    "EXR": {"ext": ".exr", "ffmpeg": ["-f", "image2", "-pix_fmt", "gbrpf32le"], "is_10bit": True},
}

AUDIO_PROFILES: dict[str, list[str]] = {
    "Copy Audio": ["-c:a", "copy"],
    "AAC": ["-c:a", "aac", "-b:a", "192k"],
    "PCM": ["-c:a", "pcm_s24le"],
    "AC3": ["-c:a", "ac3", "-b:a", "448k"],
    "FLAC": ["-c:a", "flac"],
    "No Audio": ["-an"],
}


# ---------------------------------------------------------------------------
# Resource path helper
# ---------------------------------------------------------------------------

def _resource_path(relative: str) -> str:
    """Resolve *relative* path whether running as PyInstaller bundle or from source."""
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return str(base / relative)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _form_row(label_text: str, widget: QWidget) -> QHBoxLayout:
    """Return an HBoxLayout with a right-aligned label (min 160 px) + widget."""
    lbl = QLabel(label_text)
    lbl.setMinimumWidth(160)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    row = QHBoxLayout()
    row.addWidget(lbl)
    row.addWidget(widget)
    return row



# ---------------------------------------------------------------------------
# Styled splitter handle – thick, coloured, with a ⇔ arrow indicator
# ---------------------------------------------------------------------------

class _StyledSplitterHandle(QSplitterHandle):
    """A 10 px wide splitter handle painted in #11abda with a centred ⇔ icon."""

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Background
        painter.fillRect(self.rect(), QColor("#11abda"))
        # Arrow label
        painter.setPen(QPen(QColor("white"), 1))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "⇔")
        painter.end()


class _StyledSplitter(QSplitter):
    """QSplitter that returns a coloured, thicker handle."""

    def createHandle(self) -> QSplitterHandle:  # type: ignore[override]
        return _StyledSplitterHandle(Qt.Orientation.Horizontal, self)


# ---------------------------------------------------------------------------
# Group-box / form-layout factory
# ---------------------------------------------------------------------------

def _make_group(title: str) -> tuple[QGroupBox, QFormLayout]:
    """Create a titled QGroupBox with an inner QFormLayout."""
    box = QGroupBox(title)
    layout = QFormLayout()
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setHorizontalSpacing(12)
    layout.setVerticalSpacing(6)
    layout.setContentsMargins(5, 10, 5, 10)
    box.setLayout(layout)
    return box, layout


# ---------------------------------------------------------------------------
# Zoomable image preview widget (QGraphicsView-based)
# ---------------------------------------------------------------------------

class _FullscreenWindow(QWidget):
    """A top-level window that shows a fullscreen comparison view.

    For image inputs a *copy* SplitViewWidget is created (no re-parenting of
    the live widget so the main window keeps working while fullscreen is open).
    For video inputs the live split_view is re-parented here temporarily; it
    is returned to the caller via the ``restore_widget`` signal on close.
    """

    restore_widget = pyqtSignal()

    def __init__(
        self,
        split_view: "SplitViewWidget",
        *,
        image_mode: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Full Screen  –  ESC to exit")
        self.setStyleSheet("background:#0d0d0d;")

        self._owned_view: Optional["SplitViewWidget"] = None  # lives here; destroyed on close
        self._borrowed_view: Optional["SplitViewWidget"] = None  # returned on close

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if image_mode:
            # Create a fresh SplitViewWidget and mirror the current images.
            copy_view = SplitViewWidget(self)
            if split_view._input_image and not split_view._input_image.isNull():
                copy_view.set_input_image(split_view._input_image)
            if split_view._output_image and not split_view._output_image.isNull():
                copy_view.set_output_image(split_view._output_image)
            copy_view._split_ratio = split_view._split_ratio
            copy_view._zoom = split_view._zoom
            copy_view._pan_offset = split_view._pan_offset
            self._owned_view = copy_view
            layout.addWidget(copy_view)
        else:
            # Video mode: re-parent the live split_view temporarily.
            self._borrowed_view = split_view
            layout.addWidget(split_view)

        close_btn = QPushButton("✕  Exit Full Screen  (ESC)")
        close_btn.setFixedHeight(32)
        close_btn.setStyleSheet(
            "background:#1a1a1a; color:#ccc; border:none; font-size:12px;"
            "border-top:1px solid #333;"
        )
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._borrowed_view is not None:
            # Remove from our layout WITHOUT destroying – the main window will re-adopt it.
            self.layout().removeWidget(self._borrowed_view)
            self._borrowed_view = None
            self.restore_widget.emit()
        super().closeEvent(event)


class _ZoomableImageView(QGraphicsView):
    """QGraphicsView that shows a static image and supports mouse-wheel zoom."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMinimumHeight(300)
        self.setStyleSheet(
            "background:#0d0d0d; border:1px solid #2a2a2a; border-radius:4px;"
        )

    def set_pixmap(self, pix: QPixmap) -> None:
        self._scene.clear()
        self._pix_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if event.angleDelta().y() > 0:
            self.scale(1.15, 1.15)
        elif event.angleDelta().y() < 0:
            self.scale(0.85, 0.85)


# ---------------------------------------------------------------------------
# CheckableComboBox – multi-select GPU picker
# ---------------------------------------------------------------------------

class CheckableComboBox(QComboBox):
    """A QComboBox where every item carries a checkbox for multi-selection.

    Mutual-exclusion rules
    ----------------------
    * Checking **"Auto"** unchecks all other items.
    * Checking **"CPU"** unchecks all other items.
    * Checking any **GPU N** item unchecks "Auto" and "CPU".

    The closed-combo display shows a comma-separated summary of checked items.
    Call :meth:`checkedTexts` to retrieve the current selection as a list.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        # Make the combo editable so we can write a summary into the line-edit.
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Clicking the (read-only) line-edit should open the popup.
        self.lineEdit().installEventFilter(self)
        self.view().pressed.connect(self._on_item_pressed)

    # ------------------------------------------------------------------
    # Event filter – open popup when the read-only line-edit is clicked

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if obj is self.lineEdit() and event.type() == QEvent.Type.MouseButtonRelease:
            self.showPopup()
            return True
        return super().eventFilter(obj, event)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Item helpers

    def _item(self, row: int) -> QStandardItem:
        return self._model.item(row)

    # ------------------------------------------------------------------
    # Public API – mirroring QComboBox where needed

    def addItem(self, text: str, userData: object = None) -> None:  # type: ignore[override]
        item = QStandardItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        self._model.appendRow(item)
        self._refresh_label()

    def addItems(self, texts: list[str]) -> None:  # type: ignore[override]
        for t in texts:
            self.addItem(t)

    def checkedTexts(self) -> list[str]:
        """Return all currently checked item texts in row order."""
        return [
            self._item(i).text()
            for i in range(self._model.rowCount())
            if self._item(i).checkState() == Qt.CheckState.Checked
        ]

    def setCheckedTexts(self, texts: list[str]) -> None:
        """Check all items in *texts*, preserving Auto/CPU exclusivity rules."""
        wanted = set(texts)
        for i in range(self._model.rowCount()):
            it = self._item(i)
            it.setCheckState(
                Qt.CheckState.Checked
                if it.text() in wanted
                else Qt.CheckState.Unchecked
            )

        # Re-apply exclusivity in deterministic order
        for i in range(self._model.rowCount()):
            if self._item(i).checkState() == Qt.CheckState.Checked:
                self._enforce_exclusion(i)
        self._refresh_label()

    def setCurrentText(self, text: str) -> None:  # type: ignore[override]
        """Check the single item matching *text*, uncheck everything else."""
        for i in range(self._model.rowCount()):
            it = self._item(i)
            if it.text() == text:
                it.setCheckState(Qt.CheckState.Checked)
                self._enforce_exclusion(i)
            else:
                it.setCheckState(Qt.CheckState.Unchecked)
        self._refresh_label()

    def currentText(self) -> str:  # type: ignore[override]
        """Return the first checked item text (for single-select compatibility)."""
        texts = self.checkedTexts()
        return texts[0] if texts else ""

    # ------------------------------------------------------------------
    # Internal logic

    def _on_item_pressed(self, index: object) -> None:
        item = self._model.itemFromIndex(index)  # type: ignore[arg-type]
        new_state = (
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        item.setCheckState(new_state)
        if new_state == Qt.CheckState.Checked:
            self._enforce_exclusion(index.row())  # type: ignore[union-attr]
        self._refresh_label()

    def _enforce_exclusion(self, row: int) -> None:
        """Apply mutual-exclusion rules when the item at *row* has just been checked.

        Auto / CPU selected → uncheck everything else.
        GPU X selected      → uncheck Auto and CPU only; other GPUs may co-exist.
        """
        model = self._model
        item = model.item(row)
        if not item:
            return
        txt = item.text().strip()
        if txt in ("Auto", "CPU"):
            if item.checkState() == Qt.CheckState.Checked:
                for i in range(model.rowCount()):
                    if i != row:
                        model.item(i).setCheckState(Qt.CheckState.Unchecked)
        else:
            if item.checkState() == Qt.CheckState.Checked:
                for i in range(model.rowCount()):
                    it_txt = model.item(i).text().strip()
                    if it_txt in ("Auto", "CPU"):
                        model.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _refresh_label(self) -> None:
        sel = self.checkedTexts()
        self.lineEdit().setText(", ".join(sel) if sel else "— none —")

    def hidePopup(self) -> None:
        self._refresh_label()
        super().hidePopup()


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window for SeedVR2 GUI."""

    _SETTINGS_ORG = "SeedVR2GUI"
    _SETTINGS_APP = "SeedVR2_GUI"

    def __init__(self) -> None:
        # Windows: set AppUserModelID so taskbar icon matches the window icon
        try:
            myappid = "naxci1.seedvr.upscaler.2.5"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

        super().__init__()
        self.setWindowTitle("SeedVR2.5 GUI by HB2k v.1.4 beta")
        self.resize(1100, 900)

        # Create settings window first – it loads saved paths in its __init__
        self._settings_win = SettingsWindow(self)
        self._settings_win.input_changed.connect(self._load_preview)

        self._thread = None
        self._worker = None
        self._input_meta_text: str = ""   # last "Input: …" string for dual metadata
        self._preview_temp_path: Optional[str] = None  # temp file for Preview runs
        self._is_preview_run: bool = False  # flag: switch to Split View on finish
        self._run_started_at: Optional[float] = None
        self._latest_output_path: Optional[Path] = None
        self._queue_jobs: list[dict[str, Any]] = []
        self._queue_running: bool = False
        self._queue_entry_counter: int = 0
        self._force_exit: bool = False
        self._tray_tip_shown: bool = False
        self._drop_highlight_count: int = 0

        self._build_ui()
        self._build_menu_bar()

        # Load window icon (PyInstaller-compatible path)
        icon_path = Path(get_resource_path("assets/icon.ico"))
        if not icon_path.exists():
            icon_path = Path(get_resource_path("icon.ico"))
        self.setWindowIcon(QIcon(str(icon_path)))
        self._enable_global_drop_targets()
        self._setup_system_tray()
        self._persistable_widgets = self._build_persistable_widget_map()
        self._update_export_controls()
        self._load_model_settings()
        self._prompt_load_last_preset()

        self._set_running(False)

        # Emit the GPU scan result into the console so the user can see it on startup.
        if _GPU_INIT_MSG:
            self._on_log(_GPU_INIT_MSG)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(6)
        self._drop_overlay = QLabel("Drop supported media anywhere to import", root)
        self._drop_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_overlay.setStyleSheet(
            "QLabel {"
            "background-color: rgba(17,18,20,230);"
            "border: 2px dashed #0052CC;"
            "border-radius: 14px;"
            "color: #E3E4E6;"
            "font-size: 18px;"
            "font-weight: 600;"
            "padding: 18px;"
            "}"
        )
        self._drop_overlay.hide()

        # ── 1. Header ──────────────────────────────────────────────────
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        title_col = QWidget()
        title_vlayout = QVBoxLayout(title_col)
        title_vlayout.setContentsMargins(0, 0, 0, 0)
        title_vlayout.setSpacing(2)
        title_lbl = QLabel("SeedVR2.5 GUI by HB2k v.1.4 beta")
        title_lbl.setObjectName("header_label")
        sub_lbl = QLabel("Powered by SeedVR2 Diffusion Models")
        sub_lbl.setObjectName("subheader_label")
        title_vlayout.addWidget(title_lbl)
        title_vlayout.addWidget(sub_lbl)

        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setToolTip("Open Paths & Configuration settings")
        settings_btn.setMinimumWidth(110)
        settings_btn.clicked.connect(self._open_settings)

        github_btn = QPushButton("GitHub")
        github_btn.setToolTip("Open project on GitHub")
        github_btn.setMinimumWidth(80)
        github_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/naxci1/ComfyUI-SeedVR2.5_new"))
        )

        header_layout.addWidget(title_col, stretch=1)
        header_layout.addWidget(github_btn)
        header_layout.addWidget(settings_btn)
        root_layout.addWidget(header_widget)

        # ── 2. Main splitter ───────────────────────────────────────────
        splitter = _StyledSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(10)
        splitter.setChildrenCollapsible(True)
        root_layout.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # ── 3. Bottom controls bar ─────────────────────────────────────
        root_layout.addWidget(self._build_bottom_bar())
        self._drop_overlay.raise_()
        self._sync_drop_overlay_geometry()

    # ── Left panel (comparison player) ────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(520)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        # ── Mode selector buttons ──────────────────────────────────────
        mode_bar_w = QWidget()
        mode_bar = QHBoxLayout(mode_bar_w)
        mode_bar.setContentsMargins(0, 0, 0, 0)
        mode_bar.setSpacing(4)

        self._mode_input_btn = QPushButton("Input")
        self._mode_input_btn.setCheckable(True)
        self._mode_input_btn.setChecked(True)
        self._mode_output_btn = QPushButton("Output")
        self._mode_output_btn.setCheckable(True)
        self._mode_split_btn = QPushButton("⊢  Split View")
        self._mode_split_btn.setCheckable(True)

        for btn in (self._mode_input_btn, self._mode_output_btn, self._mode_split_btn):
            btn.setEnabled(_MULTIMEDIA_AVAILABLE)
            btn.setMinimumWidth(80)

        self._mode_btn_group = QButtonGroup(panel)
        self._mode_btn_group.setExclusive(True)
        self._mode_btn_group.addButton(self._mode_input_btn, 0)
        self._mode_btn_group.addButton(self._mode_output_btn, 1)
        self._mode_btn_group.addButton(self._mode_split_btn, 2)
        self._mode_btn_group.idToggled.connect(self._on_mode_button)

        mode_bar.addWidget(self._mode_input_btn)
        mode_bar.addWidget(self._mode_output_btn)
        mode_bar.addWidget(self._mode_split_btn)
        mode_bar.addStretch(1)

        # "📂 Open" – quick input-file picker aligned to the far right of the mode bar
        self._open_input_btn = QPushButton("📂 Open")
        self._open_input_btn.setToolTip("Open a supported input video or image file")
        self._open_input_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._open_input_btn.clicked.connect(self._browse_input_for_player)
        mode_bar.addWidget(self._open_input_btn)

        # "Full Screen" – opens split view fullscreen
        self._fullscreen_btn = QPushButton("Full Screen")
        self._fullscreen_btn.setToolTip("Open Split View in full screen (ESC to exit)")
        self._fullscreen_btn.setMinimumWidth(80)
        self._fullscreen_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._fullscreen_btn.clicked.connect(self._open_fullscreen)
        mode_bar.addWidget(self._fullscreen_btn)

        layout.addWidget(mode_bar_w)

        # ── Viewer stack ───────────────────────────────────────────────
        self._viewer_stack = QStackedWidget()
        self._player_mode = "input"

        if _MULTIMEDIA_AVAILABLE:
            # page 0 – Input solo
            self._solo_input_vw = QVideoWidget()
            self._solo_input_vw.setMinimumHeight(300)
            self._viewer_stack.addWidget(self._solo_input_vw)

            # page 1 – Output solo
            self._solo_output_vw = QVideoWidget()
            self._solo_output_vw.setMinimumHeight(300)
            self._viewer_stack.addWidget(self._solo_output_vw)

            # page 2 – Overlay Split View (SplitViewWidget)
            self._split_view = SplitViewWidget()
            self._viewer_stack.addWidget(self._split_view)

            # page 3 – Static image preview (for image inputs)
            # page 3 – Zoomable static image preview
            self._image_view = _ZoomableImageView()
            self._viewer_stack.addWidget(self._image_view)

            # Media players
            self._input_player = QMediaPlayer()
            self._input_audio = QAudioOutput()
            self._input_player.setAudioOutput(self._input_audio)
            self._input_player.setVideoOutput(self._solo_input_vw)

            self._output_player = QMediaPlayer()
            self._output_audio = QAudioOutput()
            self._output_player.setAudioOutput(self._output_audio)
            self._output_player.setVideoOutput(self._solo_output_vw)

            # Output player is the master: drives seek bar, time label, play state.
            self._output_player.durationChanged.connect(self._on_player_duration)
            self._output_player.positionChanged.connect(self._on_player_position)
            self._output_player.playbackStateChanged.connect(self._on_player_state)

            # Input-only mode: also connect input player so seek/time work when
            # no output is loaded yet.
            self._input_player.durationChanged.connect(self._on_player_duration)
            self._input_player.positionChanged.connect(self._on_player_position)
            self._input_player.playbackStateChanged.connect(self._on_player_state)

            # Slave input position strictly to output (master→slave sync).
            self._output_player.positionChanged.connect(self._sync_input_to_output)

            # Metadata signal – update the file-info label after source loads.
            self._input_player.metaDataChanged.connect(self._on_input_meta_changed)
            self._input_player.durationChanged.connect(self._on_input_meta_changed)

            # Initial volume (70 %)
            self._input_audio.setVolume(0.70)
            self._output_audio.setVolume(0.70)
        else:
            self._input_player = None   # type: ignore[assignment]
            self._output_player = None  # type: ignore[assignment]
            self._input_audio = None    # type: ignore[assignment]
            self._output_audio = None   # type: ignore[assignment]
            placeholder = QLabel(
                "PyQt6.QtMultimedia not available\n\n"
                "Video preview is disabled.\n"
                "Install PyQt6-Qt6-Multimedia to enable it."
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setMinimumHeight(300)
            placeholder.setStyleSheet(
                "background:#0d0d0d; border:1px solid #2a2a2a;"
                " border-radius:4px; color:#555;"
            )
            self._viewer_stack.addWidget(placeholder)

        layout.addWidget(self._viewer_stack, stretch=1)

        # ── File metadata label ────────────────────────────────────────
        self._current_input_is_image: bool = False
        self._meta_label = QLabel("")
        self._meta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meta_label.setStyleSheet("color:#888; font-size:11px; padding:2px 0;")
        layout.addWidget(self._meta_label)

        # ── Seek slider ────────────────────────────────────────────────
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        layout.addWidget(self._seek_slider)

        # ── Control bar ────────────────────────────────────────────────
        ctrl = QHBoxLayout()

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._play_btn.clicked.connect(self._on_play_pause)

        self._stop_btn = QPushButton("⏹")
        self._stop_btn.setFixedWidth(36)
        self._stop_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._stop_btn.clicked.connect(self._on_stop)

        self._time_lbl = QLabel("0:00 / 0:00")
        self._time_lbl.setMinimumWidth(100)
        self._time_lbl.setStyleSheet("color:#888; font-size:11px;")

        self._mute_btn = QPushButton("\U0001f50a")  # 🔊
        self._mute_btn.setFixedWidth(36)
        self._mute_btn.setCheckable(True)
        self._mute_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._mute_btn.toggled.connect(self._on_mute_toggled)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(70)
        self._volume_slider.setMaximumWidth(100)
        self._volume_slider.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)

        self._open_output_btn = QPushButton("Open Output…")
        self._open_output_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._open_output_btn.clicked.connect(self._browse_output_video)

        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._stop_btn)
        ctrl.addWidget(self._time_lbl)
        ctrl.addStretch(1)
        ctrl.addWidget(self._mute_btn)
        ctrl.addWidget(self._volume_slider)
        ctrl.addWidget(self._open_output_btn)
        layout.addLayout(ctrl)

        return panel

    # ── Right panel ────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setMinimumWidth(100)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(8)

        # ── Presets ────────────────────────────────────────────────────
        preset_bar = QHBoxLayout()
        self.save_preset_btn = QPushButton("Save Preset…")
        self.save_preset_btn.setToolTip("Save model/processing settings to a JSON preset")
        self.save_preset_btn.clicked.connect(self._save_preset_dialog)
        self.load_preset_btn = QPushButton("Load Preset…")
        self.load_preset_btn.setToolTip("Load model/processing settings from a JSON preset")
        self.load_preset_btn.clicked.connect(self._load_preset_dialog)
        preset_bar.addWidget(self.save_preset_btn)
        preset_bar.addWidget(self.load_preset_btn)
        preset_bar.addStretch(1)
        container_layout.addLayout(preset_bar)

        # ── AI Model ───────────────────────────────────────────────────
        g, f = _make_group("AI Model")
        self.dit_model_combo = QComboBox()
        self.dit_model_combo.addItems([
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "seedvr2_ema_3b-Q4_K_M.gguf",
            "seedvr2_ema_3b-Q8_0.gguf",
            "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_7b-Q4_K_M.gguf",
            "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
            "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_sharp_fp16.safetensors",
        ])
        _idx = self.dit_model_combo.findText("seedvr2_ema_3b-Q8_0.gguf")
        if _idx >= 0:
            self.dit_model_combo.setCurrentIndex(_idx)
        f.addRow("DiT Model:", self.dit_model_combo)
        container_layout.addWidget(g)

        # ── Output Settings ────────────────────────────────────────────
        g, f = _make_group("Output Settings")
        self.container_combo = QComboBox()
        self.container_combo.addItems(list(EXPORT_CODEC_PROFILES.keys()))
        self.container_combo.currentTextChanged.connect(self._update_export_controls)
        f.addRow("Container:", self.container_combo)

        self.video_codec_combo = QComboBox()
        f.addRow("Video Codec:", self.video_codec_combo)

        self.export_image_sequence_check = QCheckBox()
        self.export_image_sequence_check.toggled.connect(self._update_export_controls)
        f.addRow("Export as Image Sequence:", self.export_image_sequence_check)

        self.image_sequence_format_combo = QComboBox()
        self.image_sequence_format_combo.addItems(list(IMAGE_SEQUENCE_PROFILES.keys()))
        f.addRow("Image Sequence Format:", self.image_sequence_format_combo)

        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItems(list(AUDIO_PROFILES.keys()))
        f.addRow("Audio:", self.audio_mode_combo)

        self.video_backend_combo = QComboBox()
        self.video_backend_combo.addItems(["opencv", "ffmpeg"])
        f.addRow("Video Backend:", self.video_backend_combo)

        self.use_10bit_check = QCheckBox()
        f.addRow("10-bit Output:", self.use_10bit_check)

        self.color_correction_combo = QComboBox()
        self.color_correction_combo.addItems(["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
        f.addRow("Color Correction:", self.color_correction_combo)
        container_layout.addWidget(g)

        # ── Enhancement (Upscaling) ────────────────────────────────────
        g, f = _make_group("Enhancement (Upscaling)")
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(128, 7680)
        self.resolution_spin.setValue(720)
        f.addRow("Resolution:", self.resolution_spin)

        self.max_resolution_spin = QSpinBox()
        self.max_resolution_spin.setRange(0, 7680)
        self.max_resolution_spin.setValue(0)
        self.max_resolution_spin.setToolTip("0 = no limit")
        f.addRow("Max Resolution:", self.max_resolution_spin)

        f.addRow("Batch Size:", self._build_batch_stepper())

        self.uniform_batch_check = QCheckBox()
        f.addRow("Uniform Batch Size:", self.uniform_batch_check)

        self.temporal_overlap_spin = QSpinBox()
        self.temporal_overlap_spin.setRange(0, 100)
        self.temporal_overlap_spin.setValue(0)
        f.addRow("Temporal Overlap:", self.temporal_overlap_spin)

        self.prepend_frames_spin = QSpinBox()
        self.prepend_frames_spin.setRange(0, 100)
        self.prepend_frames_spin.setValue(0)
        f.addRow("Prepend Frames:", self.prepend_frames_spin)
        container_layout.addWidget(g)

        # ── Processing ─────────────────────────────────────────────────
        g, f = _make_group("Processing")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(313)
        f.addRow("Seed:", self.seed_spin)

        self.skip_first_frames_spin = QSpinBox()
        self.skip_first_frames_spin.setRange(0, 99999)
        self.skip_first_frames_spin.setValue(0)
        f.addRow("Skip First Frames:", self.skip_first_frames_spin)

        self.load_cap_spin = QSpinBox()
        self.load_cap_spin.setRange(0, 99999)
        self.load_cap_spin.setValue(0)
        self.load_cap_spin.setToolTip("0 = load all frames")
        f.addRow("Load Cap:", self.load_cap_spin)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(0, 99999)
        self.chunk_size_spin.setValue(0)
        self.chunk_size_spin.setToolTip("0 = process all at once")
        f.addRow("Chunk Size:", self.chunk_size_spin)
        container_layout.addWidget(g)

        # ── Device Management ──────────────────────────────────────────
        g, f = _make_group("Device Management")
        self.gpu_device_combo = CheckableComboBox()
        self.gpu_device_combo.addItems(_detect_gpus())
        self.gpu_device_combo.setCurrentText("Auto")  # default: Auto checked
        self.gpu_device_combo.setToolTip(
            "Select one or more GPUs.\n"
            "Auto/CPU are exclusive; multiple GPU N items may be checked together\n"
            "for multi-GPU inference (--cuda_device 0,1,…)."
        )
        f.addRow("GPU Device:", self.gpu_device_combo)

        self.dit_offload_combo = QComboBox()
        self.dit_offload_combo.addItems(["none", "cpu"])
        self.dit_offload_combo.setCurrentText("cpu")
        f.addRow("DiT Offload:", self.dit_offload_combo)

        self.vae_offload_combo = QComboBox()
        self.vae_offload_combo.addItems(["none", "cpu"])
        self.vae_offload_combo.setCurrentText("cpu")
        f.addRow("VAE Offload:", self.vae_offload_combo)

        self.tensor_offload_combo = QComboBox()
        self.tensor_offload_combo.addItems(["none", "cpu"])
        f.addRow("Tensor Offload:", self.tensor_offload_combo)
        container_layout.addWidget(g)

        # ── Memory (BlockSwap) ─────────────────────────────────────────
        g, f = _make_group("Memory (BlockSwap)")
        self.blocks_to_swap_spin = QSpinBox()
        self.blocks_to_swap_spin.setRange(0, 36)
        self.blocks_to_swap_spin.setValue(0)
        f.addRow("Blocks to Swap:", self.blocks_to_swap_spin)

        self.swap_io_check = QCheckBox()
        f.addRow("Swap I/O Components:", self.swap_io_check)
        container_layout.addWidget(g)

        # ── VAE Tiling ─────────────────────────────────────────────────
        g, f = _make_group("VAE Tiling")
        self.vae_encode_tiled_check = QCheckBox()
        f.addRow("Encode Tiled:", self.vae_encode_tiled_check)

        self.vae_encode_tile_size_spin = QSpinBox()
        self.vae_encode_tile_size_spin.setRange(128, 4096)
        self.vae_encode_tile_size_spin.setValue(1024)
        f.addRow("Encode Tile Size:", self.vae_encode_tile_size_spin)

        self.vae_encode_tile_overlap_spin = QSpinBox()
        self.vae_encode_tile_overlap_spin.setRange(0, 512)
        self.vae_encode_tile_overlap_spin.setValue(128)
        f.addRow("Encode Tile Overlap:", self.vae_encode_tile_overlap_spin)

        self.vae_decode_tiled_check = QCheckBox()
        f.addRow("Decode Tiled:", self.vae_decode_tiled_check)

        self.vae_decode_tile_size_spin = QSpinBox()
        self.vae_decode_tile_size_spin.setRange(128, 4096)
        self.vae_decode_tile_size_spin.setValue(1024)
        f.addRow("Decode Tile Size:", self.vae_decode_tile_size_spin)

        self.vae_decode_tile_overlap_spin = QSpinBox()
        self.vae_decode_tile_overlap_spin.setRange(0, 512)
        self.vae_decode_tile_overlap_spin.setValue(128)
        f.addRow("Decode Tile Overlap:", self.vae_decode_tile_overlap_spin)

        self.tile_debug_combo = QComboBox()
        self.tile_debug_combo.addItems(["false", "encode", "decode"])
        f.addRow("Tile Debug:", self.tile_debug_combo)
        container_layout.addWidget(g)

        # ── Performance ────────────────────────────────────────────────
        g, f = _make_group("Performance")
        self.attention_mode_combo = QComboBox()
        self.attention_mode_combo.addItems([
            "sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"
        ])
        _attn_idx = self.attention_mode_combo.findText("sageattn_3")
        if _attn_idx >= 0:
            self.attention_mode_combo.setCurrentIndex(_attn_idx)
        f.addRow("Attention Mode:", self.attention_mode_combo)

        self.compile_dit_check = QCheckBox()
        f.addRow("Compile DiT:", self.compile_dit_check)

        self.compile_vae_check = QCheckBox()
        f.addRow("Compile VAE:", self.compile_vae_check)

        self.compile_backend_combo = QComboBox()
        self.compile_backend_combo.addItems(["inductor", "cudagraphs"])
        f.addRow("Compile Backend:", self.compile_backend_combo)

        self.compile_mode_combo = QComboBox()
        self.compile_mode_combo.addItems([
            "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        ])
        f.addRow("Compile Mode:", self.compile_mode_combo)

        self.compile_fullgraph_check = QCheckBox()
        f.addRow("Full Graph:", self.compile_fullgraph_check)

        self.compile_dynamic_check = QCheckBox()
        f.addRow("Dynamic:", self.compile_dynamic_check)

        self.dynamo_cache_spin = QSpinBox()
        self.dynamo_cache_spin.setRange(1, 1000)
        self.dynamo_cache_spin.setValue(64)
        f.addRow("Dynamo Cache Limit:", self.dynamo_cache_spin)

        self.dynamo_recompile_spin = QSpinBox()
        self.dynamo_recompile_spin.setRange(1, 1000)
        self.dynamo_recompile_spin.setValue(128)
        f.addRow("Dynamo Recompile Limit:", self.dynamo_recompile_spin)
        container_layout.addWidget(g)

        # ── Model Cache ────────────────────────────────────────────────
        g, f = _make_group("Model Cache")
        self.cache_dit_check = QCheckBox()
        f.addRow("Cache DiT:", self.cache_dit_check)

        self.cache_vae_check = QCheckBox()
        f.addRow("Cache VAE:", self.cache_vae_check)
        container_layout.addWidget(g)

        # ── Debug ──────────────────────────────────────────────────────
        g, f = _make_group("Debug")
        self.debug_check = QCheckBox()
        f.addRow("Verbose Debug:", self.debug_check)
        container_layout.addWidget(g)

        # ── Job Queue ──────────────────────────────────────────────────
        qg = QGroupBox("Job Queue")
        qg.setCheckable(True)
        qg.setChecked(False)
        qv = QVBoxLayout(qg)
        qv.setContentsMargins(8, 8, 8, 8)
        qv.setSpacing(6)

        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(120)
        qv.addWidget(self.queue_list)

        queue_btns_row = QHBoxLayout()
        self.queue_add_btn = QPushButton("Add Current")
        self.queue_add_btn.clicked.connect(self._queue_add_current_job)
        self.queue_remove_btn = QPushButton("Remove Selected")
        self.queue_remove_btn.clicked.connect(self._queue_remove_selected)
        self.queue_clear_btn = QPushButton("Clear")
        self.queue_clear_btn.clicked.connect(self._queue_clear_all)
        queue_btns_row.addWidget(self.queue_add_btn)
        queue_btns_row.addWidget(self.queue_remove_btn)
        queue_btns_row.addWidget(self.queue_clear_btn)
        qv.addLayout(queue_btns_row)

        self.queue_run_btn = QPushButton("Run Queue")
        self.queue_run_btn.clicked.connect(self._queue_run)
        qv.addWidget(self.queue_run_btn)

        container_layout.addWidget(qg)

        container_layout.addStretch(1)

        # Constrain input widgets and set flexible size policy to prevent overflow
        container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
        )
        # Make input widgets fill the available horizontal space
        for _cw in container.findChildren((QComboBox, QSpinBox, QLineEdit)):
            _cw.setMaximumWidth(16777215)  # Qt default – no cap
            _cw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        scroll.setWidget(container)
        return scroll

    # ── Bottom bar ─────────────────────────────────────────────────────

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        # Progress bars (outer: global frame/chunk, inner: batch step)
        self.global_progress = QProgressBar()
        self.global_progress.setRange(0, 100)
        self.global_progress.setValue(0)
        self.global_progress.setFormat("Total Progress: idle")
        layout.addWidget(self.global_progress)

        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("Batch Progress: idle")
        layout.addWidget(self.batch_progress)

        # Status + Run / Abort / Copy All / Clear row
        btn_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        self.fps_label = QLabel("0.0 fps")
        self.fps_label.setMinimumWidth(80)
        self.run_btn = QPushButton("▶  Export Video")
        self.run_btn.setObjectName("primary_button")
        self.run_btn.clicked.connect(self._run)

        self.preview_btn = QPushButton("⚡ Preview")
        self.preview_btn.setToolTip(
            "Upscale the current video frame as a single-image preview.\n"
            "Batch size will be set to 1 automatically."
        )
        self.preview_btn.clicked.connect(self._preview_run)

        self.abort_btn = QPushButton("⏹  Abort")
        self.abort_btn.setObjectName("danger_button")
        self.abort_btn.clicked.connect(self._abort)

        self.copy_log_btn = QPushButton("Copy All")
        self.copy_log_btn.setToolTip("Copy all log output to clipboard")
        self.copy_log_btn.clicked.connect(self._copy_log)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(lambda: self.console.clear())

        self.open_output_folder_btn = QPushButton("Open Output Folder")
        self.open_output_folder_btn.setEnabled(False)
        self.open_output_folder_btn.clicked.connect(self._open_output_folder)

        btn_row.addWidget(self.status_label)
        btn_row.addWidget(self.fps_label)
        btn_row.addStretch(1)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.abort_btn)
        btn_row.addWidget(self.open_output_folder_btn)
        btn_row.addSpacing(12)
        btn_row.addWidget(self.copy_log_btn)
        btn_row.addWidget(self.clear_log_btn)
        layout.addLayout(btn_row)

        # Row 4 – Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(120)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.console.setFont(mono)
        layout.addWidget(self.console)

        return bar

    # ------------------------------------------------------------------
    # Settings window
    # ------------------------------------------------------------------

    def _open_settings(self) -> None:
        """Show (or raise) the persistent Paths & Configuration window."""
        self._settings_win.show()
        self._settings_win.raise_()
        self._settings_win.activateWindow()

    def _build_menu_bar(self) -> None:
        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About SeedVR2 GUI", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

        github_action = QAction("Open GitHub", self)
        github_action.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://github.com/naxci1/ComfyUI-SeedVR2.5_new")
            )
        )
        help_menu.addAction(github_action)

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About SeedVR2 GUI",
            (
                "<b>SeedVR2.5 GUI by HB2k</b><br>"
                "Version: v1.4 beta<br><br>"
                "Topaz-style wrapper for SeedVR2 inference_cli.py.<br>"
                "License: Apache-2.0<br><br>"
                '<a href="https://github.com/naxci1/ComfyUI-SeedVR2.5_new">'
                "GitHub Repository</a>"
            ),
        )

    def _setup_system_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray_icon = None
            return

        icon = self.windowIcon()
        if icon.isNull():
            path = Path(get_resource_path("assets/icon.ico"))
            if not path.exists():
                path = Path(get_resource_path("icon.ico"))
            icon = QIcon(str(path))
        self._tray_icon = QSystemTrayIcon(icon, self)
        self._tray_icon.setToolTip("SeedVR2.5 GUI")
        self._tray_icon.activated.connect(self._on_tray_activated)

        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self._restore_from_tray)
        exit_action = tray_menu.addAction("Exit")
        exit_action.triggered.connect(self._exit_from_tray)
        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.show()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore_from_tray()

    def _restore_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _exit_from_tray(self) -> None:
        self._force_exit = True
        self.close()

    def _build_persistable_widget_map(self) -> dict[str, QWidget]:
        return {
            "dit_model_combo": self.dit_model_combo,
            "container_combo": self.container_combo,
            "video_codec_combo": self.video_codec_combo,
            "export_image_sequence_check": self.export_image_sequence_check,
            "image_sequence_format_combo": self.image_sequence_format_combo,
            "audio_mode_combo": self.audio_mode_combo,
            "video_backend_combo": self.video_backend_combo,
            "use_10bit_check": self.use_10bit_check,
            "color_correction_combo": self.color_correction_combo,
            "resolution_spin": self.resolution_spin,
            "max_resolution_spin": self.max_resolution_spin,
            "batch_size_spin": self.batch_size_spin,
            "uniform_batch_check": self.uniform_batch_check,
            "temporal_overlap_spin": self.temporal_overlap_spin,
            "prepend_frames_spin": self.prepend_frames_spin,
            "seed_spin": self.seed_spin,
            "skip_first_frames_spin": self.skip_first_frames_spin,
            "load_cap_spin": self.load_cap_spin,
            "chunk_size_spin": self.chunk_size_spin,
            "gpu_device_combo": self.gpu_device_combo,
            "dit_offload_combo": self.dit_offload_combo,
            "vae_offload_combo": self.vae_offload_combo,
            "tensor_offload_combo": self.tensor_offload_combo,
            "blocks_to_swap_spin": self.blocks_to_swap_spin,
            "swap_io_check": self.swap_io_check,
            "vae_encode_tiled_check": self.vae_encode_tiled_check,
            "vae_encode_tile_size_spin": self.vae_encode_tile_size_spin,
            "vae_encode_tile_overlap_spin": self.vae_encode_tile_overlap_spin,
            "vae_decode_tiled_check": self.vae_decode_tiled_check,
            "vae_decode_tile_size_spin": self.vae_decode_tile_size_spin,
            "vae_decode_tile_overlap_spin": self.vae_decode_tile_overlap_spin,
            "tile_debug_combo": self.tile_debug_combo,
            "attention_mode_combo": self.attention_mode_combo,
            "compile_dit_check": self.compile_dit_check,
            "compile_vae_check": self.compile_vae_check,
            "compile_backend_combo": self.compile_backend_combo,
            "compile_mode_combo": self.compile_mode_combo,
            "compile_fullgraph_check": self.compile_fullgraph_check,
            "compile_dynamic_check": self.compile_dynamic_check,
            "dynamo_cache_spin": self.dynamo_cache_spin,
            "dynamo_recompile_spin": self.dynamo_recompile_spin,
            "cache_dit_check": self.cache_dit_check,
            "cache_vae_check": self.cache_vae_check,
            "debug_check": self.debug_check,
        }

    def _update_export_controls(self, *_: object) -> None:
        container = self.container_combo.currentText()
        prev_codec = self.video_codec_combo.currentText()
        self.video_codec_combo.blockSignals(True)
        self.video_codec_combo.clear()
        self.video_codec_combo.addItems(list(EXPORT_CODEC_PROFILES.get(container, {}).keys()))
        keep_idx = self.video_codec_combo.findText(prev_codec)
        self.video_codec_combo.setCurrentIndex(keep_idx if keep_idx >= 0 else 0)
        self.video_codec_combo.blockSignals(False)

        exporting_sequence = self.export_image_sequence_check.isChecked()
        self.video_codec_combo.setEnabled(not exporting_sequence)
        self.container_combo.setEnabled(not exporting_sequence)
        self.image_sequence_format_combo.setEnabled(exporting_sequence)

    def _selected_export_extension(self) -> str:
        if self.export_image_sequence_check.isChecked():
            fmt = self.image_sequence_format_combo.currentText()
            profile = IMAGE_SEQUENCE_PROFILES.get(fmt, {})
            return str(profile.get("ext", ".png"))
        container = self.container_combo.currentText().strip().lower()
        return "." + container if container else ".mp4"

    def _selected_export_profile_to_ffmpeg_args(self) -> dict[str, Any]:
        """Return a backend mapping of current export choices to FFmpeg arguments."""
        if self.export_image_sequence_check.isChecked():
            image_fmt = self.image_sequence_format_combo.currentText()
            image_profile = IMAGE_SEQUENCE_PROFILES.get(image_fmt, {})
            return {
                "mode": "image_sequence",
                "container": "image2",
                "image_format": image_fmt,
                "video_args": image_profile.get("ffmpeg", []),
                "audio_args": ["-an"],
            }

        container = self.container_combo.currentText()
        codec = self.video_codec_combo.currentText()
        codec_profile = EXPORT_CODEC_PROFILES.get(container, {}).get(codec, {})
        audio = self.audio_mode_combo.currentText()
        audio_args = AUDIO_PROFILES.get(audio, AUDIO_PROFILES["Copy Audio"])
        return {
            "mode": "video",
            "container": container,
            "codec": codec,
            "video_args": codec_profile.get("ffmpeg", []),
            "audio_mode": audio,
            "audio_args": audio_args,
        }

    def _selected_profile_is_10bit(self) -> bool:
        if self.export_image_sequence_check.isChecked():
            fmt = self.image_sequence_format_combo.currentText()
            profile = IMAGE_SEQUENCE_PROFILES.get(fmt, {})
            return bool(profile.get("is_10bit", False))
        container = self.container_combo.currentText()
        codec = self.video_codec_combo.currentText()
        profile = EXPORT_CODEC_PROFILES.get(container, {}).get(codec, {})
        return bool(profile.get("is_10bit", False))

    def _serialize_model_settings(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, widget in self._persistable_widgets.items():
            if isinstance(widget, CheckableComboBox):
                data[key] = widget.checkedTexts()
            elif isinstance(widget, QComboBox):
                data[key] = widget.currentText()
            elif isinstance(widget, QSpinBox):
                data[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                data[key] = widget.isChecked()
        return data

    def _apply_model_settings(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            widget = self._persistable_widgets.get(key)
            if widget is None:
                continue
            if isinstance(widget, CheckableComboBox) and isinstance(value, list):
                widget.setCheckedTexts([str(v) for v in value])
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QSpinBox):
                try:
                    widget.setValue(int(value))
                except (TypeError, ValueError):
                    continue
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
        self._update_export_controls()

    def _save_model_settings(self) -> None:
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        s.setValue("model_settings_json", json.dumps(self._serialize_model_settings()))

    def _load_model_settings(self) -> None:
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        raw = s.value("model_settings_json", "", type=str)
        if not raw:
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            self._apply_model_settings(data)

    def _save_preset_dialog(self) -> None:
        start = ""
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        last = s.value("last_preset_path", "", type=str)
        if last:
            start = str(Path(last).parent)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GUI Preset",
            start,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        payload = {"version": 1, "model_settings": self._serialize_model_settings()}
        try:
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            self._on_log(f"❌  Failed to save preset: {exc}")
            return
        s.setValue("last_preset_path", str(p))
        self._on_log(f"✅  Preset saved: {p}")

    def _load_preset_dialog(self) -> None:
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        start = s.value("last_preset_path", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load GUI Preset",
            start,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        self._load_preset_from_path(Path(path), write_last_path=True)

    def _load_preset_from_path(self, path: Path, write_last_path: bool = False) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._on_log(f"❌  Failed to load preset: {exc}")
            return
        if not isinstance(payload, dict):
            self._on_log("❌  Preset format is invalid.")
            return
        model_settings = payload.get("model_settings", {})
        if not isinstance(model_settings, dict):
            self._on_log("❌  Preset model_settings section is invalid.")
            return
        self._apply_model_settings(model_settings)
        self._save_model_settings()
        if write_last_path:
            s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
            s.setValue("last_preset_path", str(path))
        self._on_log(f"✅  Preset loaded: {path}")

    def _prompt_load_last_preset(self) -> None:
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        last = s.value("last_preset_path", "", type=str).strip()
        if not last:
            return
        p = Path(last)
        if not p.is_file():
            return
        answer = QMessageBox.question(
            self,
            "Load last preset?",
            f"Load the last used preset?\n\n{p}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._load_preset_from_path(p, write_last_path=False)

    def _queue_add_current_job(self) -> None:
        if not self._settings_win.input_edit.text().strip():
            self._on_log("⚠  Queue: set an input file/folder first.")
            return
        self._queue_entry_counter += 1
        job = {
            "id": self._queue_entry_counter,
            "paths": {
                "python_exe": self._settings_win.python_exe_edit.text().strip(),
                "seedvr2_folder": self._settings_win.seedvr2_folder_edit.text().strip(),
                "input_mode": self._settings_win.input_mode_combo.currentText(),
                "input_path": self._settings_win.input_edit.text().strip(),
                "output_path": self._settings_win.output_edit.text().strip(),
                "model_dir": self._settings_win.model_dir_edit.text().strip(),
            },
            "model_settings": self._serialize_model_settings(),
        }
        self._queue_jobs.append(job)
        self.queue_list.addItem(
            f"#{job['id']}  {job['paths']['input_path']}  →  {job['paths']['output_path'] or 'auto'}"
        )

    def _queue_remove_selected(self) -> None:
        row = self.queue_list.currentRow()
        if row < 0 or row >= len(self._queue_jobs):
            return
        self._queue_jobs.pop(row)
        self.queue_list.takeItem(row)

    def _queue_clear_all(self) -> None:
        self._queue_jobs.clear()
        self.queue_list.clear()
        self._queue_running = False

    def _queue_run(self) -> None:
        if not self._queue_jobs:
            self._on_log("⚠  Queue is empty.")
            return
        if self.abort_btn.isEnabled():
            self._on_log("⚠  Cannot start queue while a job is running.")
            return
        self._queue_running = True
        self._run_next_queued_job()

    def _run_next_queued_job(self) -> None:
        if not self._queue_jobs:
            self._queue_running = False
            self._on_log("✅  Queue finished.")
            return
        job = self._queue_jobs[0]
        paths = job["paths"]
        self._settings_win.python_exe_edit.setText(str(paths.get("python_exe", "")))
        self._settings_win.seedvr2_folder_edit.setText(str(paths.get("seedvr2_folder", "")))
        self._settings_win.input_edit.setText(str(paths.get("input_path", "")))
        self._settings_win.output_edit.setText(str(paths.get("output_path", "")))
        self._settings_win.model_dir_edit.setText(str(paths.get("model_dir", "")))
        mode = str(paths.get("input_mode", "File"))
        idx = self._settings_win.input_mode_combo.findText(mode)
        if idx >= 0:
            self._settings_win.input_mode_combo.setCurrentIndex(idx)
        model_settings = job.get("model_settings", {})
        if isinstance(model_settings, dict):
            self._apply_model_settings(model_settings)
        inp = self._settings_win.input_edit.text().strip()
        if inp and Path(inp).is_file():
            self._load_preview(inp)
        self._on_log(f"▶  Queue job #{job['id']} started.")
        self._run()
        # Remove the started job from visual queue list
        self._queue_jobs.pop(0)
        self.queue_list.takeItem(0)
        if self._worker is None:
            self._on_log(f"⚠  Queue job #{job['id']} could not start, skipping.")
            self._run_next_queued_job()

    # ------------------------------------------------------------------
    # Preview / video loading
    # ------------------------------------------------------------------

    def _load_preview(self, path: str) -> None:
        """Load *path* into the preview area and update the metadata label.

        Image files are shown in the zoomable image view (page 3) and also
        pre-loaded into the SplitViewWidget for image comparison.
        Video files are fed to the input player (page 0).
        """
        _IMAGE_SUFFIXES = set(SUPPORTED_IMAGE_EXTS) | {".bmp", ".webp", ".gif"}
        suffix = Path(path).suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            self._current_input_is_image = True
            # Auto-set batch size to 1 for single image inputs
            self.batch_size_spin.setValue(1)
            # Dimensions via QImageReader (no full decode needed)
            reader = QImageReader(path)
            size = reader.size()
            if size.isValid():
                self._input_meta_text = f"Input: {size.width()}×{size.height()} px"
                self._meta_label.setText(self._input_meta_text)
            else:
                self._input_meta_text = ""
                self._meta_label.setText("")
            pix = QPixmap(path)
            if not pix.isNull():
                self._image_view.set_pixmap(pix)
                if _MULTIMEDIA_AVAILABLE and self._split_view is not None:
                    self._split_view.set_input_image(pix.toImage())
            # Stay in split view if the user is already there; otherwise show the
            # standalone image viewer.
            if self._player_mode == "split" and self._split_view is not None:
                self._viewer_stack.setCurrentIndex(2)
            else:
                self._viewer_stack.setCurrentIndex(3)
        else:
            self._current_input_is_image = False
            self._meta_label.setText("Loading…")
            self._load_input_video(path)
            self._viewer_stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Metadata update for video sources
    # ------------------------------------------------------------------

    def _on_input_meta_changed(self) -> None:
        """Update the metadata label from QMediaPlayer metadata + duration."""
        if not _MULTIMEDIA_AVAILABLE or not self._input_player:
            return
        if self._current_input_is_image:
            return  # image meta already set in _load_preview
        parts: list[str] = []
        try:
            meta = self._input_player.metaData()
            res = meta.value(QMediaMetaData.Key.Resolution)
            if res is not None:
                parts.append(f"{res.width()}×{res.height()} px")
            fps = meta.value(QMediaMetaData.Key.VideoFrameRate)
            if fps is not None:
                try:
                    parts.append(f"{float(fps):.0f} fps")
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass
        dur_ms = self._input_player.duration()
        if dur_ms > 0:
            secs = dur_ms // 1000
            parts.append(f"{secs // 60:02d}:{secs % 60:02d} min")
        if parts:
            self._input_meta_text = "Input: " + ", ".join(parts)
            self._meta_label.setText(self._input_meta_text)
        elif self._meta_label.text() == "Loading…":
            pass  # keep "Loading…" until metadata arrives

    def _on_output_meta_changed(self) -> None:
        """Update the metadata label with output video info (combined with input)."""
        if not _MULTIMEDIA_AVAILABLE or not self._output_player:
            return
        parts: list[str] = []
        try:
            meta = self._output_player.metaData()
            res = meta.value(QMediaMetaData.Key.Resolution)
            if res is not None:
                parts.append(f"{res.width()}×{res.height()} px")
            fps = meta.value(QMediaMetaData.Key.VideoFrameRate)
            if fps is not None:
                try:
                    parts.append(f"{float(fps):.0f} fps")
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass
        dur_ms = self._output_player.duration()
        if dur_ms > 0:
            secs = dur_ms // 1000
            parts.append(f"{secs // 60:02d}:{secs % 60:02d} min")
        if parts:
            out_info = "Output: " + ", ".join(parts)
            combined = f"{self._input_meta_text}  |  {out_info}" if self._input_meta_text else out_info
            self._meta_label.setText(combined)

    def _build_args(self) -> list[str]:
        args: list[str] = []

        # positional input
        inp = self._settings_win.input_edit.text().strip()
        args.append(inp)

        # output
        out = self._settings_win.output_edit.text().strip()
        if out:
            args += ["--output", out]

        # model dir
        md = self._settings_win.model_dir_edit.text().strip()
        if md:
            args += ["--model_dir", md]

        # output format mapping for SeedVR2 CLI:
        # - video export modes map to CLI output_format=mp4
        # - image sequence modes map to CLI output_format=png (CLI image-sequence path)
        if self.export_image_sequence_check.isChecked():
            args += ["--output_format", "png"]
        else:
            args += ["--output_format", "mp4"]

        # video backend
        vb = self.video_backend_combo.currentText()
        if vb != "opencv":
            args += ["--video_backend", vb]

        # 10-bit: either explicitly requested or implied by selected export profile
        if self.use_10bit_check.isChecked() or self._selected_profile_is_10bit():
            args.append("--10bit")

        # dit model
        args += ["--dit_model", self.dit_model_combo.currentText()]

        # resolution – always emit so the CLI uses the GUI value regardless of its own default
        res = self.resolution_spin.value()
        args += ["--resolution", str(res)]

        max_res = self.max_resolution_spin.value()
        if max_res != 0:
            args += ["--max_resolution", str(max_res)]

        batch = self.batch_size_spin.value()
        if batch != 5:
            args += ["--batch_size", str(batch)]

        if self.uniform_batch_check.isChecked():
            args.append("--uniform_batch_size")

        # seed – always emit so the CLI uses the GUI value regardless of its own default
        seed = self.seed_spin.value()
        args += ["--seed", str(seed)]

        skip = self.skip_first_frames_spin.value()
        if skip:
            args += ["--skip_first_frames", str(skip)]

        load_cap = self.load_cap_spin.value()
        if load_cap:
            args += ["--load_cap", str(load_cap)]

        chunk = self.chunk_size_spin.value()
        if chunk:
            args += ["--chunk_size", str(chunk)]

        prepend = self.prepend_frames_spin.value()
        if prepend:
            args += ["--prepend_frames", str(prepend)]

        temporal = self.temporal_overlap_spin.value()
        if temporal:
            args += ["--temporal_overlap", str(temporal)]

        # color correction
        cc = self.color_correction_combo.currentText()
        if cc != "lab":
            args += ["--color_correction", cc]

        # device – read all checked items from the CheckableComboBox
        checked_gpus = self.gpu_device_combo.checkedTexts()
        if not checked_gpus or "Auto" in checked_gpus:
            # Auto or nothing: let the backend choose device 0
            cuda_dev = "0"
        elif "CPU" in checked_gpus:
            cuda_dev = "cpu"
        else:
            # One or more "GPU N: Name" entries – extract the numeric CUDA indices
            # and join with commas for multi-GPU support.
            indices: list[str] = []
            for sel in checked_gpus:
                try:
                    # "GPU 0: NVIDIA GeForce …" → split on ":" → "GPU 0" → last token
                    indices.append(sel.split(":")[0].split()[-1].strip())
                except (IndexError, ValueError):
                    print(
                        f"[SeedVR2] Warning: could not parse GPU index from {sel!r}",
                        flush=True,
                    )
            cuda_dev = ",".join(indices) if indices else "0"
        args += ["--cuda_device", cuda_dev]

        dit_offload = self.dit_offload_combo.currentText()
        if dit_offload != "none":
            args += ["--dit_offload_device", dit_offload]

        vae_offload = self.vae_offload_combo.currentText()
        if vae_offload != "none":
            args += ["--vae_offload_device", vae_offload]

        tensor_offload = self.tensor_offload_combo.currentText()
        if tensor_offload != "none":
            args += ["--tensor_offload_device", tensor_offload]

        # blockswap
        bswap = self.blocks_to_swap_spin.value()
        if bswap:
            args += ["--blocks_to_swap", str(bswap)]

        if self.swap_io_check.isChecked():
            args.append("--swap_io_components")

        # vae tiling
        if self.vae_encode_tiled_check.isChecked():
            args.append("--vae_encode_tiled")
            enc_sz = self.vae_encode_tile_size_spin.value()
            if enc_sz != 1024:
                args += ["--vae_encode_tile_size", str(enc_sz)]
            enc_ov = self.vae_encode_tile_overlap_spin.value()
            if enc_ov != 128:
                args += ["--vae_encode_tile_overlap", str(enc_ov)]

        if self.vae_decode_tiled_check.isChecked():
            args.append("--vae_decode_tiled")
            dec_sz = self.vae_decode_tile_size_spin.value()
            if dec_sz != 1024:
                args += ["--vae_decode_tile_size", str(dec_sz)]
            dec_ov = self.vae_decode_tile_overlap_spin.value()
            if dec_ov != 128:
                args += ["--vae_decode_tile_overlap", str(dec_ov)]

        tile_dbg = self.tile_debug_combo.currentText()
        if tile_dbg != "false":
            args += ["--tile_debug", tile_dbg]

        # performance
        attn = self.attention_mode_combo.currentText()
        if attn != "sdpa":
            args += ["--attention_mode", attn]

        if self.compile_dit_check.isChecked():
            args.append("--compile_dit")

        if self.compile_vae_check.isChecked():
            args.append("--compile_vae")

        cb = self.compile_backend_combo.currentText()
        if cb != "inductor":
            args += ["--compile_backend", cb]

        cm = self.compile_mode_combo.currentText()
        if cm != "default":
            args += ["--compile_mode", cm]

        if self.compile_fullgraph_check.isChecked():
            args.append("--compile_fullgraph")

        if self.compile_dynamic_check.isChecked():
            args.append("--compile_dynamic")

        dc = self.dynamo_cache_spin.value()
        if dc != 64:
            args += ["--compile_dynamo_cache_size_limit", str(dc)]

        dr = self.dynamo_recompile_spin.value()
        if dr != 128:
            args += ["--compile_dynamo_recompile_limit", str(dr)]

        # cache
        if self.cache_dit_check.isChecked():
            args.append("--cache_dit")

        if self.cache_vae_check.isChecked():
            args.append("--cache_vae")

        # debug
        if self.debug_check.isChecked():
            args.append("--debug")

        return args

    # ------------------------------------------------------------------
    # Run / Abort
    # ------------------------------------------------------------------

    def _run(self) -> None:
        inp = self._settings_win.input_edit.text().strip()
        if not inp:
            self._on_log("❌  Please specify an input file or directory (⚙ Settings).")
            return

        python_exe = self._settings_win.python_exe_edit.text().strip() or DEFAULT_PYTHON_EXE
        seedvr2_folder = self._settings_win.seedvr2_folder_edit.text().strip()

        if not seedvr2_folder:
            self._on_log("❌  Please select the SeedVR2 folder in ⚙ Settings.")
            return

        cli_script = str(Path(seedvr2_folder) / "inference_cli.py")
        if not os.path.isfile(cli_script):
            self._on_log(
                f"❌  inference_cli.py not found in: {seedvr2_folder}\n"
                "    Please select the correct SeedVR2 installation folder in ⚙ Settings."
            )
            return

        if not os.path.isfile(python_exe):
            self._on_log(
                f"❌  Python executable not found: {python_exe}\n"
                "    Please check the Python Executable path in ⚙ Settings."
            )
            return

        # Persist the current paths
        self._settings_win.save_settings()
        self._save_model_settings()
        self._set_latest_output_path(None)

        args = self._build_args()
        ffmpeg_profile = self._selected_export_profile_to_ffmpeg_args()
        self._on_log(f"🎬  Export Profile: {json.dumps(ffmpeg_profile, ensure_ascii=False)}")

        self._thread, self._worker = create_worker_thread(cli_script, args, python_exe)
        self._worker.log_line.connect(self._on_log)
        self._worker.progress_update.connect(self._on_global_progress)
        self._worker.batch_progress_update.connect(self._on_batch_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.started_signal.connect(lambda: self._set_running(True))

        self._run_started_at = time.time()
        self._reset_progress_bars()
        self._set_running(True)
        self.status_label.setText("Starting…")
        self._thread.start()

    def _abort(self) -> None:
        if self._worker:
            self._worker.request_abort()
        self.status_label.setText("Aborting…")

    def _preview_run(self) -> None:
        """Capture the currently displayed video frame, save to a temp PNG,
        set batch size=1 and input to that file, then start an upscale run."""
        frame_img = None

        if _MULTIMEDIA_AVAILABLE and self._input_player is not None:
            # Step 1: pause the video so the frame buffer is stable
            self._input_player.pause()
            if self._output_player:
                self._output_player.pause()

            # Step 2: process pending Qt events so the video sink flushes its frame
            QApplication.processEvents()

            # Step 3: capture frame from the input player's video sink
            try:
                sink = self._input_player.videoSink()
                if sink is not None:
                    vframe = sink.videoFrame()
                    if vframe is not None and vframe.isValid():
                        frame_img = vframe.toImage()
                        if frame_img is not None and frame_img.isNull():
                            frame_img = None
            except Exception:
                frame_img = None

        # Fallback: try grabbing from the split view's input sink (when in split mode)
        if frame_img is None and _MULTIMEDIA_AVAILABLE and self._split_view is not None:
            try:
                vframe = self._split_view.input_sink.videoFrame()
                if vframe is not None and vframe.isValid():
                    frame_img = vframe.toImage()
                    if frame_img is not None and frame_img.isNull():
                        frame_img = None
            except Exception:
                frame_img = None

        # Fallback: grab the input widget visually (may be black for GPU-rendered video,
        # but works fine for image inputs displayed in the zoomable viewer)
        if frame_img is None and _MULTIMEDIA_AVAILABLE:
            for player_widget in (
                getattr(self, "_solo_input_vw", None),
                getattr(self, "_solo_output_vw", None),
            ):
                if player_widget is not None:
                    pix = player_widget.grab()
                    if not pix.isNull() and pix.width() > 4:
                        frame_img = pix.toImage()
                        break

        # Fallback: use whatever the zoomable image view is showing (image input case)
        if frame_img is None:
            img_view = getattr(self, "_image_view", None)
            if img_view is not None and img_view._pix_item is not None:
                scene_pix = img_view._pix_item.pixmap()
                if not scene_pix.isNull():
                    frame_img = scene_pix.toImage()

        if frame_img is None or frame_img.isNull():
            self._on_log("⚠  Preview: no frame available – play or pause a video first.")
            return

        # Save to a temporary PNG file (persist across the upscale run)
        if self._preview_temp_path and os.path.isfile(self._preview_temp_path):
            try:
                os.remove(self._preview_temp_path)
            except OSError:
                pass
        tmp = tempfile.NamedTemporaryFile(suffix="_preview.png", delete=False)
        tmp.close()
        self._preview_temp_path = tmp.name
        if not frame_img.save(self._preview_temp_path, "PNG"):
            self._on_log(f"⚠  Preview: could not save temp frame to {self._preview_temp_path}")
            return

        self._on_log(f"ℹ  Preview: captured frame → {self._preview_temp_path}")

        # Point input to the temp file, set batch size to 1, mark as preview run
        self._settings_win.input_edit.setText(self._preview_temp_path)
        self.batch_size_spin.setValue(1)
        self._is_preview_run = True

        # Refresh the preview display with the captured frame
        self._load_preview(self._preview_temp_path)

        # Now launch the normal run
        self._run()

    # ------------------------------------------------------------------
    # Player – mode switching
    # ------------------------------------------------------------------

    def _on_mode_button(self, btn_id: int, checked: bool) -> None:
        """Switch the viewer stack page and reassign video outputs."""
        if not checked or not _MULTIMEDIA_AVAILABLE:
            return
        modes = ["input", "output", "split"]
        if btn_id >= len(modes):
            return
        new_mode = modes[btn_id]
        # For video modes skip redundant switches; always allow image-tab refresh.
        if new_mode == self._player_mode and not self._current_input_is_image:
            return
        self._player_mode = new_mode
        if new_mode == "input":
            if self._current_input_is_image:
                # Always re-load input pixmap so the tab shows the original image even
                # if the previous tab had swapped _image_view to display the output.
                inp_path = self._settings_win.input_edit.text().strip()
                if inp_path:
                    pix = QPixmap(inp_path)
                    if not pix.isNull():
                        self._image_view.set_pixmap(pix)
                self._viewer_stack.setCurrentIndex(3)
            else:
                self._input_player.setVideoOutput(self._solo_input_vw)
                self._viewer_stack.setCurrentIndex(0)
        elif new_mode == "output":
            if self._current_input_is_image:
                # Use whatever output image is already held by the split view – it was
                # populated by _try_auto_load_output() when upscaling finished.
                # Do NOT call _try_auto_load_output() here: that function calls
                # _on_mode_button(split) internally which would corrupt _player_mode.
                out_img = self._split_view._output_image if self._split_view else None
                if out_img and not out_img.isNull():
                    self._image_view.set_pixmap(QPixmap.fromImage(out_img))
                # Always show the image-view page (output image if available, else
                # the viewer remains blank with the dark background).
                self._viewer_stack.setCurrentIndex(3)
            else:
                self._output_player.setVideoOutput(self._solo_output_vw)
                self._viewer_stack.setCurrentIndex(1)
        else:  # split
            if self._current_input_is_image:
                # Image input: feed directly into SplitViewWidget; no video sink needed.
                inp_path = self._settings_win.input_edit.text().strip()
                if inp_path:
                    pix = QPixmap(inp_path)
                    if not pix.isNull():
                        self._split_view.set_input_image(pix.toImage())
                # Do NOT connect either player to the split view sinks – images only.
            else:
                self._input_player.setVideoOutput(self._split_view.input_sink)
                self._output_player.setVideoOutput(self._split_view.output_sink)
            self._viewer_stack.setCurrentIndex(2)

    # ------------------------------------------------------------------
    # Fullscreen
    # ------------------------------------------------------------------

    def _open_fullscreen(self) -> None:
        """Open the current view in a dedicated fullscreen window."""
        if self._split_view is None:
            return
        is_img = self._current_input_is_image

        if not is_img:
            # Video mode: ensure split view is active before re-parenting it.
            if not _MULTIMEDIA_AVAILABLE:
                return
            if self._player_mode != "split":
                self._mode_split_btn.setChecked(True)
                self._on_mode_button(2, True)

        self._fs_window = _FullscreenWindow(
            self._split_view, image_mode=is_img
        )
        if not is_img:
            # Video mode: the live split_view was borrowed; restore it when closed.
            self._fs_window.restore_widget.connect(self._on_fullscreen_closed)
        self._fs_window.showFullScreen()

    def _on_fullscreen_closed(self) -> None:
        """Re-insert split_view back into the viewer stack after video-fullscreen exits."""
        if self._split_view is None:
            return
        self._viewer_stack.insertWidget(2, self._split_view)
        self._viewer_stack.setCurrentIndex(2)

    # ------------------------------------------------------------------
    # Player – unified controls
    # ------------------------------------------------------------------

    def _active_player(self):
        """Return the primary player driving the seek slider."""
        if not _MULTIMEDIA_AVAILABLE:
            return None
        return self._output_player if self._player_mode == "output" else self._input_player

    def _on_play_pause(self) -> None:
        p = self._active_player()
        if not p:
            return
        if p.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._input_player.pause()
            self._output_player.pause()
        else:
            self._output_player.play()
            self._input_player.play()

    def _on_stop(self) -> None:
        if self._input_player:
            self._input_player.stop()
        if self._output_player:
            self._output_player.stop()

    def _on_seek(self, pos: int) -> None:
        if self._input_player:
            self._input_player.setPosition(pos)
        if self._output_player:
            self._output_player.setPosition(pos)

    def _on_player_duration(self, _duration: int) -> None:
        p = self._active_player()
        if p:
            self._seek_slider.setRange(0, p.duration())
        self._update_time_label()

    def _on_player_position(self, position: int) -> None:
        if not self._seek_slider.isSliderDown():
            self._seek_slider.setValue(position)
        self._update_time_label()

    def _on_player_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("\u23f8" if playing else "\u25b6")  # ⏸ / ▶

    def _sync_input_to_output(self, pos: int) -> None:
        """Slave the input player's position strictly to the output player (master)."""
        if self._input_player and abs(self._input_player.position() - pos) > 200:
            self._input_player.setPosition(pos)

    def _update_time_label(self) -> None:
        p = self._active_player()
        if not p:
            return
        def _fmt(ms: int) -> str:
            s = ms // 1000
            return f"{s // 60}:{s % 60:02d}"
        self._time_lbl.setText(f"{_fmt(p.position())} / {_fmt(p.duration())}")

    # ------------------------------------------------------------------
    # Player – volume / mute
    # ------------------------------------------------------------------

    def _on_volume_changed(self, val: int) -> None:
        v = val / 100.0
        if self._input_audio:
            self._input_audio.setVolume(v)
        if self._output_audio:
            self._output_audio.setVolume(v)

    def _on_mute_toggled(self, muted: bool) -> None:
        self._mute_btn.setText("\U0001f507" if muted else "\U0001f50a")  # 🔇 / 🔊
        if self._input_audio:
            self._input_audio.setMuted(muted)
        if self._output_audio:
            self._output_audio.setMuted(muted)

    # ------------------------------------------------------------------
    # Video source loading
    # ------------------------------------------------------------------

    def _load_input_video(self, path: str) -> None:
        if self._input_player:
            self._input_player.setSource(QUrl.fromLocalFile(path))

    def _load_output_video(self, path: str) -> None:
        if self._output_player:
            self._output_player.setSource(QUrl.fromLocalFile(path))
            # Connect metadata update so we can show "Input … | Output …" once ready
            try:
                self._output_player.metaDataChanged.disconnect(self._on_output_meta_changed)
            except Exception:
                pass
            self._output_player.metaDataChanged.connect(self._on_output_meta_changed)

    def _browse_output_video(self) -> None:
        start_dir = self._settings_win.output_edit.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Output Video", start_dir,
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if path:
            self._load_output_video(path)
            # Switch to output view automatically
            self._mode_output_btn.setChecked(True)

    def _browse_input_for_player(self) -> None:
        """Open a file picker from the mode bar; sets the input path and loads the preview."""
        start_dir = self._settings_win.input_edit.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", start_dir,
            INPUT_DIALOG_FILTER,
        )
        if path:
            if not self._apply_dropped_path(Path(path)):
                self._on_log(f"⚠  Unsupported input format: {path}")

    def _try_auto_load_output(self) -> None:
        out = self._settings_win.output_edit.text().strip()
        if not out:
            return
        out_path = Path(out)

        _image_exts = set(SUPPORTED_IMAGE_EXTS) | {".bmp", ".webp"}
        _video_exts = set(SUPPORTED_VIDEO_EXTS)

        # ── Image input: construct output path directly (mirrors inference_cli.generate_output_path) ──
        if self._current_input_is_image and self._split_view is not None:
            inp = self._settings_win.input_edit.text().strip()
            if not inp:
                return
            inp_path = Path(inp)
            inp_stem = inp_path.stem

            # Determine expected extension from current export profile.
            _preferred_ext: Optional[str] = (
                self._selected_export_extension()
                if self.export_image_sequence_check.isChecked()
                else None
            )

            def _find_image_output() -> Optional[Path]:
                # Build ordered list of extensions to try: preferred ext first, then all others.
                _all_img_exts = [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"]
                if _preferred_ext:
                    ordered_exts = [_preferred_ext] + [
                        e for e in _all_img_exts if e != _preferred_ext
                    ]
                else:
                    ordered_exts = _all_img_exts

                # --- Direct path construction (mirrors inference_cli.generate_output_path) ---
                if out_path.is_dir():
                    # User specified output dir: <output_dir>/<stem><ext> (no _upscaled suffix)
                    for ext in ordered_exts:
                        candidate = out_path / f"{inp_stem}{ext}"
                        if candidate.is_file():
                            return candidate
                elif not out_path.exists():
                    # No output dir set → same dir as input with _upscaled suffix
                    for ext in ordered_exts:
                        candidate = inp_path.parent / f"{inp_stem}_upscaled{ext}"
                        if candidate.is_file():
                            return candidate

                # --- Fallback: newest image in folder matching input stem ---
                search_dir = out_path if out_path.is_dir() else inp_path.parent
                candidates = sorted(
                    (f for f in search_dir.iterdir() if f.suffix.lower() in _image_exts),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                preferred = [f for f in candidates if f.stem.startswith(inp_stem)]
                if preferred:
                    return preferred[0]
                return candidates[0] if candidates else None

            result = _find_image_output()
            if result is not None:
                img = QPixmap(str(result))
                if not img.isNull():
                    self._split_view.set_output_image(img.toImage())
                    self._set_latest_output_path(result)
                    # Update metadata label with combined input | output info
                    reader = QImageReader(str(result))
                    size = reader.size()
                    if size.isValid():
                        out_info = f"Output: {size.width()}×{size.height()} px"
                        combined = f"{self._input_meta_text}  |  {out_info}" if self._input_meta_text else out_info
                        self._meta_label.setText(combined)
                    self._mode_split_btn.setChecked(True)
                    self._on_mode_button(2, True)
            return

        # ── Video input: load into output player as before ──
        if not _MULTIMEDIA_AVAILABLE:
            return
        if out_path.is_file() and out_path.suffix.lower() in _video_exts:
            self._load_output_video(str(out_path))
            self._set_latest_output_path(out_path)
            self._mode_output_btn.setChecked(True)
        elif out_path.is_dir():
            candidates = sorted(
                (f for f in out_path.iterdir() if f.suffix.lower() in _video_exts),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                self._load_output_video(str(candidates[0]))
                self._set_latest_output_path(candidates[0])
                self._mode_output_btn.setChecked(True)

    # ------------------------------------------------------------------
    # Batch size constraint (4n+1) – stepper widget + snap validator
    # ------------------------------------------------------------------

    def _build_batch_stepper(self) -> QWidget:
        """Return a [−] QSpinBox [+] widget for batch size (4k+1 rule)."""
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        _btn_font = QFont()
        _btn_font.setBold(True)
        _btn_font.setPointSize(12)

        _stepper_ss = (
            "QPushButton { background-color: #222222; border: 1px solid #444; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2e2e2e; }"
            "QPushButton:pressed { background-color: #1a1a1a; }"
        )

        minus_btn = QPushButton("-")
        minus_btn.setFixedWidth(34)
        minus_btn.setFont(_btn_font)
        minus_btn.setToolTip("Decrease batch size by 4")
        minus_btn.setStyleSheet(_stepper_ss)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10001)
        self.batch_size_spin.setValue(5)
        # NoButtons: the custom ± buttons handle ±4; manual typing snaps on commit
        self.batch_size_spin.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.batch_size_spin.setToolTip(
            "Must be 4k+1: 1, 5, 9, 13, … (automatically snapped when typed)"
        )
        self.batch_size_spin.valueChanged.connect(self._snap_batch_size)

        plus_btn = QPushButton("+")
        plus_btn.setFixedWidth(34)
        plus_btn.setFont(_btn_font)
        plus_btn.setToolTip("Increase batch size by 4")
        plus_btn.setStyleSheet(_stepper_ss)

        # ±4 step – result is always 4k+1 if starting from a valid value
        minus_btn.clicked.connect(
            lambda: self.batch_size_spin.setValue(
                max(1, self.batch_size_spin.value() - 4)
            )
        )
        plus_btn.clicked.connect(
            lambda: self.batch_size_spin.setValue(self.batch_size_spin.value() + 4)
        )

        row.addWidget(minus_btn)
        row.addWidget(self.batch_size_spin, stretch=1)
        row.addWidget(plus_btn)
        return wrapper

    def _snap_batch_size(self, val: int) -> None:
        """Snap the batch_size spinbox to the nearest 4n+1 value (1, 5, 9, 13, …).

        Rounds to nearest valid value so that incrementing (5→6) advances to 9
        and decrementing (9→8) goes back to 5.
        """
        if (val - 1) % 4 == 0:
            return  # already valid
        # Nearest lower and upper valid values
        lower = max(1, 1 + 4 * ((val - 1) // 4))
        upper = lower + 4
        snapped = lower if (val - lower) <= (upper - val) else upper
        self.batch_size_spin.blockSignals(True)
        self.batch_size_spin.setValue(snapped)
        self.batch_size_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _copy_log(self) -> None:
        """Copy all console output to the system clipboard."""
        QApplication.clipboard().setText(self.console.toPlainText())

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_log(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{ts}] {line}")

    def _reset_progress_bars(self) -> None:
        self.global_progress.setRange(0, 100)
        self.global_progress.setValue(0)
        self.global_progress.setFormat("Total Progress: idle")
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("Batch Progress: idle")
        self.fps_label.setText("0.0 fps")

    def _format_seconds(self, seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60}:{total % 60:02d}"

    def _on_global_progress(self, cur: int, tot: int) -> None:
        if tot <= 0:
            return
        self.global_progress.setRange(0, tot)
        self.global_progress.setValue(max(0, min(cur, tot)))
        self.global_progress.setFormat(f"Total Progress: {cur}/{tot}")

        elapsed = 0.0 if self._run_started_at is None else (time.time() - self._run_started_at)
        eta_text = "estimating…"
        fps_text = "0.0 fps"
        if cur > 0:
            remaining = (elapsed / cur) * max(0, tot - cur)
            eta_text = f"≈ {self._format_seconds(remaining)} remaining"
            if elapsed > 0:
                fps_text = f"{(cur / elapsed):.1f} fps"
        self.fps_label.setText(fps_text)
        self.status_label.setText(
            f"Processing {cur}/{tot}  |  {self._format_seconds(elapsed)} elapsed  |  {eta_text}"
        )

    def _on_batch_progress(self, cur: int, tot: int) -> None:
        if tot <= 0:
            return
        self.batch_progress.setRange(0, tot)
        self.batch_progress.setValue(max(0, min(cur, tot)))
        self.batch_progress.setFormat(f"Batch Progress: {cur}/{tot}")

    def _set_latest_output_path(self, path: Optional[Path]) -> None:
        self._latest_output_path = path
        self.open_output_folder_btn.setEnabled(path is not None)

    def _open_output_folder(self) -> None:
        if self._latest_output_path is None:
            return
        folder = (
            self._latest_output_path
            if self._latest_output_path.is_dir()
            else self._latest_output_path.parent
        )
        if not folder.exists():
            self._on_log(f"⚠  Output folder does not exist: {folder}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            self._on_log(f"❌  Failed to open output folder: {exc}")

    def _on_finished(self, success: bool, msg: str) -> None:
        self._set_running(False)
        self._run_started_at = None
        if success:
            self.status_label.setText(f"✅  {msg}")
            self._try_auto_load_output()
            # After a Preview run, automatically switch to Split View for comparison
            if self._is_preview_run and _MULTIMEDIA_AVAILABLE:
                self._mode_split_btn.setChecked(True)
                self._on_mode_button(2, True)
        else:
            self.status_label.setText(f"⚠  {msg}")
        if self.batch_progress.maximum() == self.batch_progress.value():
            self.batch_progress.setFormat("Batch Progress: complete")
        if self.global_progress.maximum() == self.global_progress.value():
            self.global_progress.setFormat("Total Progress: complete")
        if getattr(self, "_tray_icon", None) is not None:
            self._tray_icon.showMessage(
                "SeedVR2 GUI",
                f"Processing finished: {msg}",
                QSystemTrayIcon.MessageIcon.Information if success else QSystemTrayIcon.MessageIcon.Warning,
                4000,
            )
        self._is_preview_run = False
        self._worker = None
        self._thread = None

        if self._queue_running:
            self._run_next_queued_job()

    # ------------------------------------------------------------------
    # Settings persistence (delegated to SettingsWindow)
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        """Reload settings into the SettingsWindow (called on startup)."""
        self._settings_win.load_settings()

    def _save_settings(self) -> None:
        """Persist current path values via SettingsWindow."""
        self._settings_win.save_settings()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.preview_btn.setEnabled(not running)
        self.abort_btn.setEnabled(running)
        if hasattr(self, "queue_run_btn"):
            self.queue_run_btn.setEnabled(not running)

    def _enable_global_drop_targets(self) -> None:
        self.setAcceptDrops(True)
        self.installEventFilter(self)
        cw = self.centralWidget()
        if cw is not None:
            cw.setAcceptDrops(True)
            cw.installEventFilter(self)
            for w in cw.findChildren(QWidget):
                w.setAcceptDrops(True)
                w.installEventFilter(self)

    def _sync_drop_overlay_geometry(self) -> None:
        cw = self.centralWidget()
        if cw is None:
            return
        margin = 18
        self._drop_overlay.setGeometry(
            margin,
            margin,
            max(100, cw.width() - (2 * margin)),
            max(80, cw.height() - (2 * margin)),
        )

    def _set_drop_overlay_visible(self, visible: bool) -> None:
        if visible:
            self._sync_drop_overlay_geometry()
            self._drop_overlay.raise_()
            self._drop_overlay.show()
        else:
            self._drop_overlay.hide()

    def _extract_first_local_path(self, event: Any) -> Optional[Path]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile():
                return Path(url.toLocalFile())
        return None

    def _sequence_frame_candidates(self, folder: Path) -> list[Path]:
        try:
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        except OSError:
            return []
        return sorted(files, key=lambda p: p.name.lower())

    def _is_supported_drop_path(self, path: Path) -> bool:
        if path.is_file():
            return path.suffix.lower() in SUPPORTED_VIDEO_EXTS | SUPPORTED_IMAGE_EXTS
        if path.is_dir():
            return len(self._sequence_frame_candidates(path)) >= MIN_SEQUENCE_FRAMES
        return False

    def _apply_dropped_path(self, path: Path) -> bool:
        if not self._is_supported_drop_path(path):
            return False
        if path.is_dir():
            frames = self._sequence_frame_candidates(path)
            self._settings_win.input_mode_combo.setCurrentText("Folder")
            self._settings_win.input_edit.setText(str(path))
            if frames:
                self._load_preview(str(frames[0]))
                self._mode_input_btn.setChecked(True)
            self._on_log(f"📂  Loaded image-sequence folder ({len(frames)} frames): {path}")
            return True
        self._settings_win.input_mode_combo.setCurrentText("File")
        self._settings_win.input_edit.setText(str(path))
        self._load_preview(str(path))
        self._mode_input_btn.setChecked(True)
        self._on_log(f"📂  Input file set from drag-drop: {path}")
        return True

    def eventFilter(self, watched: object, event: object) -> bool:
        if isinstance(event, QEvent) and event.type() == QEvent.Type.DragMove:
            path = self._extract_first_local_path(event)  # type: ignore[arg-type]
            if path is not None and self._is_supported_drop_path(path):
                self._set_drop_overlay_visible(True)
                event.acceptProposedAction()  # type: ignore[attr-defined]
                return True
            event.ignore()  # type: ignore[attr-defined]
            return True
        if isinstance(event, (QDragEnterEvent, QDropEvent)):
            if isinstance(event, QDragEnterEvent):
                path = self._extract_first_local_path(event)
                if path is not None and self._is_supported_drop_path(path):
                    self._drop_highlight_count += 1
                    self._set_drop_overlay_visible(True)
                    event.acceptProposedAction()
                    return True
                self._set_drop_overlay_visible(False)
                event.ignore()
                return True
            if isinstance(event, QDropEvent):
                path = self._extract_first_local_path(event)
                self._drop_highlight_count = 0
                self._set_drop_overlay_visible(False)
                if path is not None and self._apply_dropped_path(path):
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True

        if isinstance(event, QDragLeaveEvent):
            self._drop_highlight_count = max(0, self._drop_highlight_count - 1)
            if self._drop_highlight_count == 0:
                self._set_drop_overlay_visible(False)
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        path = self._extract_first_local_path(event)
        if path is not None and self._is_supported_drop_path(path):
            self._set_drop_overlay_visible(True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self._set_drop_overlay_visible(False)
        path = self._extract_first_local_path(event)
        if path is not None and self._apply_dropped_path(path):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # type: ignore[override]
        self._set_drop_overlay_visible(False)
        super().dragLeaveEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_drop_overlay_geometry()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_model_settings()
        if getattr(self, "_tray_icon", None) is not None and not self._force_exit:
            self.hide()
            event.ignore()
            if not self._tray_tip_shown:
                self._tray_icon.showMessage(
                    "SeedVR2 GUI",
                    "SeedVR2 GUI is still running in the system tray.",
                    QSystemTrayIcon.MessageIcon.Information,
                    3500,
                )
                self._tray_tip_shown = True
            return
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helpers (module-level, used inside this file only)
# ---------------------------------------------------------------------------

def _wrap(layout: QHBoxLayout) -> QWidget:
    """Wrap a QHBoxLayout in a plain QWidget so it can be added to a QFormLayout."""
    w = QWidget()
    w.setLayout(layout)
    return w
