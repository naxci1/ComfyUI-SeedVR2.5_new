"""
SeedVR2 GUI – Main Window
Topaz-style dark-mode wrapper around inference_cli.py.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt, QPoint, QRectF, QSettings, QUrl, QEvent, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QImageReader,
    QImageWriter,
    QPixmap,
    QPainter,
    QPen,
    QPolygon,
    QStandardItem,
    QStandardItemModel,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QButtonGroup,
    QFileDialog,
    QFrame,
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
    QDoubleSpinBox,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QSplitterHandle,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import winsound as _winsound
    _WINSOUND_AVAILABLE = True
except ImportError:
    _winsound = None  # type: ignore[assignment]
    _WINSOUND_AVAILABLE = False

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

try:
    from gui.folders_dialog import FoldersDialog
except ImportError:
    from folders_dialog import FoldersDialog  # type: ignore[no-redef]

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

    # ── Intel XPU (torch.xpu) ─────────────────────────────────────────────
    if not gpu_entries:
        try:
            import torch as _torch  # noqa: PLC0415
            if hasattr(_torch, 'xpu') and _torch.xpu.is_available():
                xpu_count = _torch.xpu.device_count()
                gpu_entries = [f"GPU {i}: Intel XPU {i}" for i in range(xpu_count)]
                if not _GPU_INIT_MSG.startswith("✅"):
                    _GPU_INIT_MSG = f"✅  Detected {xpu_count} Intel XPU device(s) via torch.xpu."
        except Exception:
            pass

    # ── AMD ROCm / HIP (torch.cuda with ROCm backend) ────────────────────
    if not gpu_entries:
        try:
            import torch as _torch  # noqa: PLC0415
            if hasattr(_torch.version, 'hip') and _torch.version.hip is not None:
                count = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
                if count > 0:
                    gpu_entries = [
                        f"GPU {i}: AMD {_torch.cuda.get_device_name(i)}"
                        for i in range(count)
                    ]
                    if not _GPU_INIT_MSG.startswith("✅"):
                        _GPU_INIT_MSG = f"✅  Detected {count} AMD ROCm device(s)."
        except Exception:
            pass

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
        "H.264 (NVIDIA NVENC)": {"ffmpeg": ["-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main (8-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main10 (10-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main10", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
        "AV1 (8-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-strict", "experimental", "-cpu-used", "4", "-row-mt", "1"], "is_10bit": False},
        "AV1 (10-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p10le", "-strict", "experimental", "-cpu-used", "4", "-row-mt", "1"], "is_10bit": True},
        "AV1 (NVIDIA NVENC)": {"ffmpeg": ["-c:v", "av1_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p"], "is_10bit": False},
    },
    "MOV": {
        "ProRes 422 Proxy": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le"], "is_10bit": True},
        "ProRes 422 LT": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "1", "-pix_fmt", "yuv422p10le"], "is_10bit": True},
        "ProRes 422": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"], "is_10bit": True},
        "ProRes 422 HQ": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], "is_10bit": True},
        "ProRes 4444 XQ": {"ffmpeg": ["-c:v", "prores_ks", "-profile:v", "5", "-pix_fmt", "yuva444p12le"], "is_10bit": True},
        "H.264 High (8-bit)": {"ffmpeg": ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.264 (NVIDIA NVENC)": {"ffmpeg": ["-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main (8-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main10 (10-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main10", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
        "Uncompressed YUV (V210)": {"ffmpeg": ["-c:v", "v210"], "is_10bit": True},
        "Uncompressed RGB (R210)": {"ffmpeg": ["-c:v", "r210"], "is_10bit": True},
        "QuickTime Animation (Alpha)": {"ffmpeg": ["-c:v", "qtrle", "-pix_fmt", "argb"], "is_10bit": False},
    },
    "MKV": {
        "H.264 High (8-bit)": {"ffmpeg": ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.264 (NVIDIA NVENC)": {"ffmpeg": ["-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main (8-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "H.265 (HEVC) Main10 (10-bit)": {"ffmpeg": ["-c:v", "libx265", "-profile:v", "main10", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
        "AV1 (8-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-strict", "experimental", "-cpu-used", "4", "-row-mt", "1"], "is_10bit": False},
        "AV1 (10-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p10le", "-strict", "experimental", "-cpu-used", "4", "-row-mt", "1"], "is_10bit": True},
        "AV1 (NVIDIA NVENC)": {"ffmpeg": ["-c:v", "av1_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p"], "is_10bit": False},
        "VP9 (Good)": {"ffmpeg": ["-c:v", "libvpx-vp9", "-deadline", "good"], "is_10bit": False},
        "VP9 (Best)": {"ffmpeg": ["-c:v", "libvpx-vp9", "-deadline", "best"], "is_10bit": False},
        "FFV1 (Lossless 8/10/12-bit)": {"ffmpeg": ["-c:v", "ffv1", "-level", "3"], "is_10bit": True},
        "Uncompressed YUV (V210)": {"ffmpeg": ["-c:v", "v210"], "is_10bit": True},
    },
    "WEBM": {
        "VP9 (Good)": {"ffmpeg": ["-c:v", "libvpx-vp9", "-deadline", "good"], "is_10bit": False},
        "VP9 (Best)": {"ffmpeg": ["-c:v", "libvpx-vp9", "-deadline", "best"], "is_10bit": False},
        "AV1 (8-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-strict", "experimental", "-cpu-used", "4", "-row-mt", "1"], "is_10bit": False},
        "AV1 (10-bit)": {"ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p10le", "-strict", "experimental", "-cpu-used", "4", "-row-mt", "1"], "is_10bit": True},
    },
}

IMAGE_SEQUENCE_PROFILES: dict[str, dict[str, Any]] = {
    ".png":  {"ext": ".png",  "ffmpeg": ["-f", "image2", "-pix_fmt", "rgb24"],     "is_10bit": False},
    ".tif":  {"ext": ".tif",  "ffmpeg": ["-f", "image2", "-pix_fmt", "rgb48le"],   "is_10bit": True},
    ".tiff": {"ext": ".tiff", "ffmpeg": ["-f", "image2", "-pix_fmt", "rgb48le"],   "is_10bit": True},
    ".jpg":  {"ext": ".jpg",  "ffmpeg": ["-f", "image2", "-pix_fmt", "yuvj420p"],  "is_10bit": False},
    ".jpeg": {"ext": ".jpeg", "ffmpeg": ["-f", "image2", "-pix_fmt", "yuvj420p"],  "is_10bit": False},
    ".dpx":  {"ext": ".dpx",  "ffmpeg": ["-f", "image2", "-pix_fmt", "gbrp10le"],  "is_10bit": True},
    ".exr":  {"ext": ".exr",  "ffmpeg": ["-f", "image2", "-pix_fmt", "gbrpf32le"], "is_10bit": True},
}

# Per-format bit-depth options for image-sequence export.
# Each entry is an ordered list of {label, pix_fmt, is_10bit} dicts.
# The UI populates image_bit_depth_combo from this map whenever the file-type changes.
IMAGE_BIT_DEPTHS: dict[str, list[dict[str, Any]]] = {
    ".png":  [{"label": "8-bit",  "pix_fmt": "rgb24",     "is_10bit": False}],
    ".tif":  [{"label": "8-bit",  "pix_fmt": "rgb24",     "is_10bit": False},
              {"label": "16-bit", "pix_fmt": "rgb48le",   "is_10bit": True}],
    ".tiff": [{"label": "8-bit",  "pix_fmt": "rgb24",     "is_10bit": False},
              {"label": "16-bit", "pix_fmt": "rgb48le",   "is_10bit": True}],
    ".jpg":  [{"label": "8-bit",  "pix_fmt": "yuvj420p",  "is_10bit": False}],
    ".jpeg": [{"label": "8-bit",  "pix_fmt": "yuvj420p",  "is_10bit": False}],
    ".dpx":  [{"label": "10-bit", "pix_fmt": "gbrp10le",  "is_10bit": True},
              {"label": "12-bit", "pix_fmt": "gbrp12le",  "is_10bit": True}],
    ".exr":  [{"label": "16-bit", "pix_fmt": "gbrpf16le", "is_10bit": True},
              {"label": "32-bit", "pix_fmt": "gbrpf32le", "is_10bit": True}],
}

# Unified flat codec list shown when Video export mode is active.
# Each entry maps display name → {container, ffmpeg args, is_10bit}.
# The container is automatically applied to the backing container_combo.
UNIFIED_VIDEO_CODEC_PROFILES: dict[str, dict[str, Any]] = {
    # MOV container – ProRes family
    "ProRes 422 HQ":  {"container": "MOV", "ffmpeg": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],  "is_10bit": True},
    "ProRes 4444 XQ": {"container": "MOV", "ffmpeg": ["-c:v", "prores_ks", "-profile:v", "5", "-pix_fmt", "yuva444p12le"], "is_10bit": True},
    # MP4 container – H.264 / H.265 / AV1
    "H.264 High":     {"container": "MP4", "ffmpeg": ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"],     "is_10bit": False},
    "H.265 Main":     {"container": "MP4", "ffmpeg": ["-c:v", "libx265", "-profile:v", "main", "-pix_fmt", "yuv420p"],     "is_10bit": False},
    "H.265 Main10":   {"container": "MP4", "ffmpeg": ["-c:v", "libx265", "-profile:v", "main10", "-pix_fmt", "yuv420p10le"], "is_10bit": True},
    "AV1":            {"container": "MP4", "ffmpeg": ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-strict", "experimental", "-cpu-used", "4", "-row-mt", "1"], "is_10bit": False},
    # MKV container – VP9
    "VP9":            {"container": "MKV", "ffmpeg": ["-c:v", "libvpx-vp9", "-deadline", "good"],                          "is_10bit": False},
}

# Codecs available per container for the UI filter
CONTAINER_CODECS: dict[str, list[str]] = {
    "MOV": ["ProRes 422 HQ", "ProRes 4444 XQ"],
    "MP4": ["H.264 High", "H.265 Main", "H.265 Main10", "AV1"],
    "MKV": ["VP9"],
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


class CircularProgressWidget(QWidget):
    """Compact circular progress indicator with title and value text."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._value = 0.0
        self._text = "0%"
        self.setMinimumSize(122, 122)
        self.setMaximumSize(150, 150)

    def set_progress(self, value: float) -> None:
        self._value = max(0.0, min(1.0, value))
        self.update()

    def set_text(self, text: str) -> None:
        self._text = text
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(10, 16, -10, -10)
        circle_rect = QRectF(rect.left(), rect.top() + 12, rect.width(), rect.width())

        bg_pen = QPen(QColor("#2E3338"), 9)
        painter.setPen(bg_pen)
        painter.drawArc(circle_rect, 0, 360 * 16)

        fg_pen = QPen(QColor("#11abda"), 9)
        painter.setPen(fg_pen)
        start_angle = 90 * 16
        span_angle = int(-360 * 16 * self._value)
        painter.drawArc(circle_rect, start_angle, span_angle)

        painter.setPen(QColor("#E3E4E6"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(rect.left(), rect.top() - 2, rect.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            self._title,
        )
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(circle_rect, Qt.AlignmentFlag.AlignCenter, self._text)
        painter.end()


class _NoOpProgressIndicator:
    """Compatibility shim for removed circular progress widgets."""

    def set_progress(self, _value: float) -> None:
        pass

    def set_text(self, _text: str) -> None:
        pass



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
# Trim-aware timeline slider
# ---------------------------------------------------------------------------

class _TrimSlider(QSlider):
    """QSlider subclass that draws a coloured In/Out region and triangle markers."""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None) -> None:
        super().__init__(orientation, parent)
        self._in_frac: Optional[float] = None   # [0, 1] fraction of total range
        self._out_frac: Optional[float] = None

    def set_trim_fractions(self, in_frac: Optional[float], out_frac: Optional[float]) -> None:
        self._in_frac = in_frac
        self._out_frac = out_frac
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self._in_frac is None and self._out_frac is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        # Small inset so markers align with the actual track groove
        pad = 8
        track_w = w - 2 * pad
        track_cy = h // 2

        in_frac = self._in_frac if self._in_frac is not None else 0.0
        out_frac = self._out_frac if self._out_frac is not None else 1.0
        in_x = pad + int(in_frac * track_w)
        out_x = pad + int(out_frac * track_w)

        # Highlight selected range
        region_color = QColor("#11abda")
        region_color.setAlpha(60)
        painter.fillRect(in_x, track_cy - 3, max(1, out_x - in_x), 6, region_color)

        marker_top = track_cy - 11

        # In-point: green downward triangle ▼
        if self._in_frac is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#00cc66"))
            painter.drawPolygon(
                QPolygon([
                    QPoint(in_x - 4, marker_top),
                    QPoint(in_x + 4, marker_top),
                    QPoint(in_x, track_cy - 2),
                ])
            )

        # Out-point: red downward triangle ▼
        if self._out_frac is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ff4444"))
            painter.drawPolygon(
                QPolygon([
                    QPoint(out_x - 4, marker_top),
                    QPoint(out_x + 4, marker_top),
                    QPoint(out_x, track_cy - 2),
                ])
            )

        painter.end()


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
            "background:#0d0d0d; border:1px solid #2a2a2a; border-radius:6px;"
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
        self.setWindowTitle("1Click SeedVR2.5 ver. 1.7b (by Naxci1)")
        self.resize(1100, 900)

        # Create settings window first – it loads saved paths in its __init__
        self._settings_win = SettingsWindow(self)

        # Folders dialog manages Input / Output paths (separate from system paths)
        self._folders_dlg = FoldersDialog(self)
        self._folders_dlg.input_changed.connect(self._load_preview)

        self._thread = None
        self._worker = None
        self._input_meta_text: str = ""   # last "Input: …" string for dual metadata
        self._preview_temp_path: Optional[str] = None  # preview input frame path
        self._preview_output_path: Optional[str] = None  # tracked preview upscale output (TIFF) path
        self._is_preview_run: bool = False  # flag: switch to Split View on finish
        self._run_started_at: Optional[float] = None
        self._latest_output_path: Optional[Path] = None
        self._queue_jobs: list[dict[str, Any]] = []
        self._queue_running: bool = False
        self._queue_entry_counter: int = 0
        self._active_file_status: str = ""
        self._progress_status: str = ""
        self._batch_status: str = ""
        self._force_exit: bool = False
        self._tray_tip_shown: bool = False
        self._drop_highlight_count: int = 0
        self._current_file_path: str = ""
        self._current_file_total_frames: int = 0
        self._current_file_processed_frames: int = 0
        self._queue_files_total: int = 0
        self._queue_files_completed: int = 0
        self._queue_file_frame_counts: dict[str, int] = {}
        self._queue_ordered_files: list[str] = []
        self._active_queue_index: int = -1
        self._preview_original_input_path: Optional[str] = None
        self._preview_original_input_mode: str = "File"
        self._preview_original_position: int = 0
        self._preview_compare_active: bool = False
        self._preview_saved_batch_size: int = 81
        self._preview_saved_load_cap: Optional[int] = None  # restored after video-mode preview
        # True when the current/last preview run processed a VIDEO clip (not a PNG frame)
        self._is_preview_video_mode: bool = False
        self._advanced_mode_enabled: bool = False
        self._last_batch_cur: int = 0
        self._last_batch_tot: int = 0
        self._frozen_elapsed_seconds: Optional[float] = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed_progress_ui)
        # In/Out trim point state (milliseconds in the media timeline)
        self._in_point_ms: Optional[int] = None
        self._out_point_ms: Optional[int] = None
        self._current_fps: float = 0.0  # cached from QMediaPlayer metadata
        self._simple_defaults: dict[str, Any] = {
            "pre_downscale": "1:1",
            "resolution_mode": "Pixel",
            "resolution": 720,
            "batch_size": 81,
            "enable_video_chunking": False,
            "split_minutes": 3,
            "vae_tiling": False,
            "debug": False,
        }

        self._build_ui()
        self._build_menu_bar()
        self.menuBar().hide()

        # Load window icon (PyInstaller-compatible path)
        icon_path = Path(get_resource_path("assets/logo.png"))
        if not icon_path.exists():
            icon_path = Path(get_resource_path("assets/icon.ico"))
        if not icon_path.exists():
            icon_path = Path(get_resource_path("icon.ico"))
        self.setWindowIcon(QIcon(str(icon_path)))
        self._enable_global_drop_targets()
        self._persistable_widgets = self._build_persistable_widget_map()
        self._on_container_changed()
        self._load_model_settings()
        self._prompt_load_last_preset()
        self._apply_mode_visibility()

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
        header_widget.setObjectName("top_bar")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(6, 2, 6, 6)
        header_layout.setSpacing(10)

        help_btn = QPushButton("About")
        help_btn.setFlat(True)
        help_btn.clicked.connect(self._show_about_dialog)

        github_btn = QPushButton("GitHub")
        github_btn.setFlat(True)
        github_btn.setToolTip("Open project on GitHub")
        github_btn.setMinimumWidth(80)
        github_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/naxci1/1Click_SeedVR2.5"))
        )

        self.advanced_mode_btn = QPushButton("Simple Mode")
        self.advanced_mode_btn.setCheckable(True)
        self.advanced_mode_btn.setToolTip("Toggle between Simple Mode and Advanced Mode")
        self.advanced_mode_btn.clicked.connect(self._toggle_advanced_mode)

        settings_btn = QPushButton("Settings")
        settings_btn.setToolTip("Open Paths & Configuration settings")
        settings_btn.setMinimumWidth(84)
        settings_btn.clicked.connect(self._open_settings)

        header_layout.addWidget(help_btn)
        header_layout.addWidget(github_btn)
        header_layout.addStretch(1)
        header_layout.addWidget(self.advanced_mode_btn)
        header_layout.addWidget(settings_btn)
        root_layout.addWidget(header_widget)

        # ── 2. Main splitter ───────────────────────────────────────────
        splitter = _StyledSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(10)
        splitter.setChildrenCollapsible(True)
        root_layout.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_right_panel())
        splitter.addWidget(self._build_left_panel())
        splitter.setStretchFactor(0, 30)
        splitter.setStretchFactor(1, 70)
        splitter.setSizes([300, 700])

        # ── 3. Bottom controls bar ─────────────────────────────────────
        root_layout.addWidget(self._build_bottom_bar())
        root_layout.setStretch(0, 0)
        root_layout.setStretch(1, 14)
        root_layout.setStretch(2, 3)
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

        # "📁 Folders" – opens the Input/Output Folders dialog
        self._folders_btn = QPushButton("📁 Folders")
        self._folders_btn.setToolTip("Manage Input / Output folder paths")
        self._folders_btn.setMinimumWidth(80)
        self._folders_btn.clicked.connect(self._open_folders_dialog)
        mode_bar.addWidget(self._folders_btn)

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

        # ── Thin separator between viewer and under-viewer controls ───
        _under_sep = QWidget()
        _under_sep.setFixedHeight(1)
        _under_sep.setStyleSheet("background:#272A2D;")
        layout.addWidget(_under_sep)

        # ── Single strict horizontal row: all controls under preview ───
        # Contains: ⏮ ◀ ⏸ ⏭ | timecode | seek slider | Preview | Export | Split | Open Output
        self._current_input_is_image: bool = False

        under_row = QHBoxLayout()
        under_row.setContentsMargins(0, 4, 0, 4)
        under_row.setSpacing(3)

        _btn_css = (
            "QPushButton {"
            "  min-width: 34px; min-height: 26px;"
            "  padding: 2px 4px;"
            "  font-size: 14px;"
            "  color: #e8e8e8;"
            "  background: #3a3a3a;"
            "  border: 1px solid #555;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:hover  { background: #505050; border-color: #888; }"
            "QPushButton:pressed { background: #222; }"
            "QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #3a3a3a; }"
        )

        self._frame_back_btn = QPushButton("⏮")
        self._frame_back_btn.setToolTip("Previous frame")
        self._frame_back_btn.setStyleSheet(_btn_css)
        self._frame_back_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._frame_back_btn.clicked.connect(lambda: self._step_frame(-1))

        self._play_btn = QPushButton("▶")
        self._play_btn.setToolTip("Play")
        self._play_btn.setStyleSheet(_btn_css)
        self._play_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._play_btn.clicked.connect(self._on_play_pause)

        self._pause_btn = QPushButton("⏸")
        self._pause_btn.setToolTip("Pause")
        self._pause_btn.setStyleSheet(_btn_css)
        self._pause_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._pause_btn.clicked.connect(self._pause_playback)

        self._frame_forward_btn = QPushButton("⏭")
        self._frame_forward_btn.setToolTip("Next frame")
        self._frame_forward_btn.setStyleSheet(_btn_css)
        self._frame_forward_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._frame_forward_btn.clicked.connect(lambda: self._step_frame(1))

        self._time_lbl = QLabel("0:00/0:00")
        self._time_lbl.setMinimumWidth(72)
        self._time_lbl.setStyleSheet("color:#aaa; font-size:10px; padding: 0 2px;")

        # Detailed timecode label: "HH:MM:SS | F:N" — updated on every position change
        self._timecode_lbl = QLabel("00:00:00 | F:0")
        self._timecode_lbl.setToolTip(
            "Current position: HH:MM:SS | Frame index\n"
            "Press [ to set In-Point, ] to set Out-Point"
        )
        self._timecode_lbl.setStyleSheet("color:#11abda; font-size:10px; padding: 0 2px;")
        self._timecode_lbl.setEnabled(_MULTIMEDIA_AVAILABLE)

        self._seek_slider = _TrimSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._seek_slider.sliderMoved.connect(self._on_seek)

        self._split_toggle = QCheckBox("⊣⊢")
        self._split_toggle.setToolTip("Split View")
        self._split_toggle.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._split_toggle.setMaximumWidth(22)
        self._split_toggle.toggled.connect(self._on_split_toggle_changed)

        _out_css = _btn_css.replace("min-width: 34px", "min-width: 60px")
        self._open_output_btn = QPushButton("📂 Out")
        self._open_output_btn.setToolTip("Select Output Video…")
        self._open_output_btn.setStyleSheet(_out_css)
        self._open_output_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._open_output_btn.clicked.connect(self._browse_output_video)

        # Trim control buttons: [ Set In | X Clear | ] Set Out
        _trim_css = (
            _btn_css
            + "QPushButton { min-width: 24px; font-weight: bold; }"
        )
        self._trim_in_btn = QPushButton("[")
        self._trim_in_btn.setToolTip("Set In-Point to current frame  (shortcut: [)")
        self._trim_in_btn.setStyleSheet(_trim_css)
        self._trim_in_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._trim_in_btn.clicked.connect(self._set_in_point)

        self._trim_clear_btn = QPushButton("✕")
        self._trim_clear_btn.setToolTip("Clear In/Out trim range")
        self._trim_clear_btn.setStyleSheet(_trim_css)
        self._trim_clear_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._trim_clear_btn.clicked.connect(self._clear_trim_range)

        self._trim_out_btn = QPushButton("]")
        self._trim_out_btn.setToolTip("Set Out-Point to current frame  (shortcut: ])")
        self._trim_out_btn.setStyleSheet(_trim_css)
        self._trim_out_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._trim_out_btn.clicked.connect(self._set_out_point)

        under_row.addWidget(self._frame_back_btn)
        under_row.addWidget(self._play_btn)
        under_row.addWidget(self._pause_btn)
        under_row.addWidget(self._frame_forward_btn)
        under_row.addWidget(self._trim_in_btn)
        under_row.addWidget(self._trim_clear_btn)
        under_row.addWidget(self._trim_out_btn)
        under_row.addWidget(self._time_lbl)
        under_row.addWidget(self._timecode_lbl)
        under_row.addWidget(self._seek_slider, stretch=1)
        under_row.addWidget(self.preview_btn)
        under_row.addWidget(self._split_toggle)
        under_row.addWidget(self._open_output_btn)
        layout.addLayout(under_row)

        # ── File metadata label ────────────────────────────────────────
        self._meta_label = QLabel("")
        self._meta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meta_label.setStyleSheet("color:#888; font-size:11px; padding:1px 0;")
        layout.addWidget(self._meta_label)
        layout.setStretch(0, 0)
        layout.setStretch(1, 14)
        layout.setStretch(2, 0)
        layout.setStretch(3, 0)
        layout.setStretch(4, 0)

        return panel

    # ── Right panel ────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        # ── Outer container (never scrolls itself) ──────────────────────
        outer = QWidget()
        outer.setMinimumWidth(560)
        outer.setMaximumWidth(700)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── Tab header bar (Adjustments | Codec settings) ───────────────
        _TAB_SS = (
            "QPushButton{"
            "  background:transparent; border:none;"
            "  border-bottom:2px solid transparent;"
            "  color:#888; padding:6px 14px; font-size:13px;"
            "}"
            "QPushButton:checked{ color:#E3E4E6; border-bottom:2px solid #0052CC; }"
            "QPushButton:hover:!checked{ color:#BFC3CA; }"
        )
        tab_header = QWidget()
        tab_header.setObjectName("tab_bar")
        tab_header.setFixedHeight(38)
        th_layout = QHBoxLayout(tab_header)
        th_layout.setContentsMargins(6, 0, 6, 0)
        th_layout.setSpacing(0)

        self._tab_adj_btn = QPushButton("Adjustments")
        self._tab_adj_btn.setCheckable(True)
        self._tab_adj_btn.setChecked(True)
        self._tab_adj_btn.setStyleSheet(_TAB_SS)

        self._tab_codec_btn = QPushButton("Codec settings")
        self._tab_codec_btn.setCheckable(True)
        self._tab_codec_btn.setStyleSheet(_TAB_SS)

        _tab_grp = QButtonGroup(outer)
        _tab_grp.setExclusive(True)
        _tab_grp.addButton(self._tab_adj_btn, 0)
        _tab_grp.addButton(self._tab_codec_btn, 1)
        _tab_grp.idToggled.connect(
            lambda idx, checked: self._switch_right_tab(idx) if checked else None
        )

        th_layout.addWidget(self._tab_adj_btn)
        th_layout.addWidget(self._tab_codec_btn)
        th_layout.addStretch(1)

        tab_sep = QWidget()
        tab_sep.setFixedHeight(1)
        tab_sep.setStyleSheet("background:#272A2D;")

        outer_layout.addWidget(tab_header)
        outer_layout.addWidget(tab_sep)

        # ── Single scroll area that holds BOTH panes ────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Host: stacks the two panes vertically; only one is visible at a time
        _host = QWidget()
        _host_layout = QVBoxLayout(_host)
        _host_layout.setContentsMargins(0, 0, 0, 0)
        _host_layout.setSpacing(0)

        # ── Adjustments pane ───────────────────────────────────────────
        self._adj_pane = QWidget()
        adj_layout = QVBoxLayout(self._adj_pane)
        adj_layout.setContentsMargins(10, 10, 15, 10)
        adj_layout.setSpacing(8)

        # Presets bar
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
        adj_layout.addLayout(preset_bar)

        # AI Model
        g, f = _make_group("AI Model")
        self.dit_model_combo = QComboBox()
        _dit_model_names = [
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
        ]
        for _name in _dit_model_names:
            self.dit_model_combo.addItem(_name, _name)
        _idx = self.dit_model_combo.findData("seedvr2_ema_3b-Q8_0.gguf")
        if _idx >= 0:
            self.dit_model_combo.setCurrentIndex(_idx)
        f.addRow("DiT Model:", self.dit_model_combo)
        adj_layout.addWidget(g)

        # Processing Settings
        g, f = _make_group("Processing Settings")
        self._processing_settings_group = g

        # --- Pre-Downscale ---
        self.pre_downscale_combo = QComboBox()
        self.pre_downscale_combo.addItems(["1:1", "2:1", "3:1"])
        self.pre_downscale_combo.setToolTip(
            "Downscale the input before upscaling.\n"
            "1:1 = no downscale (passthrough)\n"
            "2:1 = halve input dimensions via Lanczos before feeding the model\n"
            "3:1 = reduce to 1/3 input dimensions via Lanczos"
        )
        f.addRow("Pre-Downscale:", self.pre_downscale_combo)

        # --- Resolution Mode (X Times / Pixel / Standard) + value control ---
        _res_mode_row = QHBoxLayout()
        _res_mode_row.setSpacing(4)
        self.resolution_mode_combo = QComboBox()
        self.resolution_mode_combo.addItems(["Pixel", "X Times", "Standard"])
        self.resolution_mode_combo.setToolTip(
            "Pixel: target the output at an exact pixel height.\n"
            "X Times: multiply the (pre-downscaled) input height by this factor.\n"
            "Standard: choose from common resolution presets (480/720/1080/1440/2160)."
        )
        _res_mode_row.addWidget(self.resolution_mode_combo)

        # Pixel mode: plain spinbox
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(128, 7680)
        self.resolution_spin.setValue(720)
        self.resolution_spin.setSingleStep(1)

        # X Times mode: combo with preset multipliers
        self.resolution_times_combo = QComboBox()
        self.resolution_times_combo.addItems(["1x", "2x", "3x", "4x", "5x"])
        self.resolution_times_combo.setCurrentText("2x")

        # Standard mode: named presets (numeric value is extracted for the CLI)
        self.resolution_standard_combo = QComboBox()
        self.resolution_standard_combo.addItems(["480", "720 (HD)", "1080 (FHD)", "1440 (2K)", "2160 (4K)"])
        self.resolution_standard_combo.setCurrentText("720 (HD)")

        _res_mode_row.addWidget(self.resolution_spin, stretch=1)
        _res_mode_row.addWidget(self.resolution_times_combo, stretch=1)
        _res_mode_row.addWidget(self.resolution_standard_combo, stretch=1)

        # Wire up visibility
        def _update_res_mode(text: str) -> None:
            self.resolution_spin.setVisible(text == "Pixel")
            self.resolution_times_combo.setVisible(text == "X Times")
            self.resolution_standard_combo.setVisible(text == "Standard")

        self.resolution_mode_combo.currentTextChanged.connect(_update_res_mode)
        _update_res_mode(self.resolution_mode_combo.currentText())

        _res_mode_container = QWidget()
        _res_mode_container.setLayout(_res_mode_row)
        f.addRow("Resolution:", _res_mode_container)

        self.max_resolution_spin = QSpinBox()
        self.max_resolution_spin.setRange(0, 7680)
        self.max_resolution_spin.setValue(0)
        self.max_resolution_spin.setToolTip("0 = no limit")
        self._max_resolution_label = QLabel("Max Resolution:")
        f.addRow(self._max_resolution_label, self.max_resolution_spin)

        # Batch Size – custom ±4 stepper enforces strict 4k+1 values
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

        # --- Video Chunking ---
        # Kept in the always-visible Processing Settings group so the control is
        # available (visible and functional) in both Simple Mode and Advanced
        # Mode. Enabling it bounds RAM usage to a single chunk during streaming.
        self.enable_video_chunking_check = QCheckBox()
        f.addRow("Enable Video Chunking:", self.enable_video_chunking_check)

        self.chunk_duration_minutes_label = QLabel("Chunk Duration (Minutes):")
        self.split_size_minutes_spin = QSpinBox()
        self.split_size_minutes_spin.setRange(1, 120)
        self.split_size_minutes_spin.setValue(3)
        self.split_size_minutes_spin.setSuffix(" min")
        self.split_size_minutes_spin.setToolTip("Chunk duration in minutes. Converted at runtime via minutes × 60 × source FPS.")
        self.chunk_size_spin = self.split_size_minutes_spin  # backwards-compatible persistence key
        f.addRow(self.chunk_duration_minutes_label, self.split_size_minutes_spin)
        self.enable_video_chunking_check.toggled.connect(self._update_chunking_visibility)
        self._update_chunking_visibility(self.enable_video_chunking_check.isChecked())
        adj_layout.addWidget(g)

        # Preview & Processing
        g, f = _make_group("Preview & Processing")
        self._preview_processing_group = g

        self.skip_first_frames_spin = QSpinBox()
        self.skip_first_frames_spin.setRange(0, 99999)
        self.skip_first_frames_spin.setValue(0)
        f.addRow("Skip First Frames:", self.skip_first_frames_spin)

        self.load_cap_spin = QSpinBox()
        self.load_cap_spin.setRange(0, 99999)
        self.load_cap_spin.setValue(0)
        self.load_cap_spin.setToolTip("0 = load all frames; Preview auto-sets this to 81 and restores it when done")
        f.addRow("Load Cap Frames:", self.load_cap_spin)

        self.only_frames_spin = QSpinBox()
        self.only_frames_spin.setRange(0, 99999)
        self.only_frames_spin.setValue(0)
        self.only_frames_spin.setToolTip("0 = no limit; limits the maximum number of frames processed per VAE decode chunk to prevent OOM.")
        f.addRow("Only Frames:", self.only_frames_spin)

        adj_layout.addWidget(g)

        # Device Management
        g, f = _make_group("Device Management")
        self.gpu_device_combo = CheckableComboBox()
        self.gpu_device_combo.addItems(_detect_gpus())
        self.gpu_device_combo.setCurrentText("Auto")
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
        adj_layout.addWidget(g)

        # Memory (BlockSwap)
        g, f = _make_group("Memory (BlockSwap)")
        self._memory_blockswap_group = g
        self.blocks_to_swap_spin = QSpinBox()
        self.blocks_to_swap_spin.setRange(0, 36)
        self.blocks_to_swap_spin.setValue(0)
        f.addRow("Blocks to Swap:", self.blocks_to_swap_spin)

        self.swap_io_check = QCheckBox()
        f.addRow("Swap I/O Components:", self.swap_io_check)
        adj_layout.addWidget(g)

        # VAE Tiling
        g, f = _make_group("VAE Tiling")
        self._vae_tiling_group = g
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
        adj_layout.addWidget(g)

        # Performance
        g, f = _make_group("Performance")
        self._performance_group = g
        self.attention_mode_combo = QComboBox()
        self.attention_mode_combo.addItems([
            "sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"
        ])
        _attn_idx = self.attention_mode_combo.findText("sageattn_3")
        if _attn_idx >= 0:
            self.attention_mode_combo.setCurrentIndex(_attn_idx)
        f.addRow("Attention Mode:", self.attention_mode_combo)

        adj_layout.addWidget(g)

        # Quality Control (noise injection — advanced only)
        g, f = _make_group("Quality Control")
        self._quality_control_group = g

        self.input_noise_scale_spin = QDoubleSpinBox()
        self.input_noise_scale_spin.setRange(0.0, 1.0)
        self.input_noise_scale_spin.setValue(0.0)
        self.input_noise_scale_spin.setSingleStep(0.05)
        self.input_noise_scale_spin.setDecimals(2)
        self.input_noise_scale_spin.setToolTip(
            "Input noise injection scale. Reduces artifacts at high resolutions."
        )
        f.addRow("Input Noise Scale:", self.input_noise_scale_spin)

        self.latent_noise_scale_spin = QDoubleSpinBox()
        self.latent_noise_scale_spin.setRange(0.0, 1.0)
        self.latent_noise_scale_spin.setValue(0.0)
        self.latent_noise_scale_spin.setSingleStep(0.05)
        self.latent_noise_scale_spin.setDecimals(2)
        self.latent_noise_scale_spin.setToolTip(
            "Latent space noise scale. Softens details if needed."
        )
        f.addRow("Latent Noise Scale:", self.latent_noise_scale_spin)
        adj_layout.addWidget(g)

        # Model Cache
        g, f = _make_group("Model Cache")
        self._model_cache_group = g
        self.cache_dit_check = QCheckBox()
        f.addRow("Cache DiT:", self.cache_dit_check)

        self.cache_vae_check = QCheckBox()
        f.addRow("Cache VAE:", self.cache_vae_check)
        adj_layout.addWidget(g)

        # Debug
        g, f = _make_group("Debug")
        self._debug_group = g
        self.auto_tune_check = QCheckBox()
        f.addRow("Auto Tune:", self.auto_tune_check)
        self.debug_check = QCheckBox()
        f.addRow("Verbose Debug:", self.debug_check)
        self.enable_audio_notifications_check = QCheckBox()
        self.enable_audio_notifications_check.setToolTip(
            "Play a Windows system sound when processing finishes (success) or errors."
        )
        f.addRow("Sound Notifications:", self.enable_audio_notifications_check)
        adj_layout.addWidget(g)

        adj_layout.addStretch(1)

        # ── Codec settings pane ────────────────────────────────────────
        self._codec_pane = QWidget()
        self._codec_pane.setVisible(False)
        codec_layout = QVBoxLayout(self._codec_pane)
        codec_layout.setContentsMargins(10, 10, 15, 10)
        codec_layout.setSpacing(8)

        # ── Output Type toggle (first item in Codec pane) ──────────────
        # "Video" → show video export group; "Image Seq." → show image sequence group.
        _ff_row = QHBoxLayout()
        _ff_row.setContentsMargins(0, 0, 0, 4)
        _ff_row.setSpacing(8)
        _ff_lbl = QLabel("Output Type:")
        _ff_lbl.setStyleSheet("color:#B0B3B8; font-size:12px;")
        _ff_row.addWidget(_ff_lbl)
        self.file_format_combo = QComboBox()
        self.file_format_combo.addItems(["Video", "Image Seq."])
        self.file_format_combo.setCurrentText("Video")
        self.file_format_combo.currentTextChanged.connect(self._on_file_format_changed)
        _ff_row.addWidget(self.file_format_combo, stretch=1)
        codec_layout.addLayout(_ff_row)

        # Invisible backing state — read by _build_args, _selected_export_extension,
        # _selected_export_profile_to_ffmpeg_args, and preset load/save.
        # Driven by the file_format_combo; also accepts setChecked() from preset restore.
        self.export_image_sequence_check = QCheckBox()
        self.export_image_sequence_check.toggled.connect(self._update_export_controls)
        self.export_image_sequence_check.toggled.connect(self._on_image_sequence_toggled)

        # ── Video export group (visible when Video Mode is active) ──────
        self._video_export_group, vf = _make_group("Video Export")
        # Container selection drives the codec list
        self.container_combo = QComboBox()
        self.container_combo.addItems(list(CONTAINER_CODECS.keys()))
        vf.addRow("Container:", self.container_combo)

        self.video_codec_combo = QComboBox()
        vf.addRow("Video Codec:", self.video_codec_combo)
        # Container change filters codec list; codec change syncs container
        self.container_combo.currentIndexChanged.connect(self._on_container_changed)
        self.video_codec_combo.currentIndexChanged.connect(self._update_export_controls)

        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItems(list(AUDIO_PROFILES.keys()))
        vf.addRow("Audio:", self.audio_mode_combo)

        # ── Bitrate / Quality Mode ────────────────────────────────────────
        self.bitrate_mode_combo = QComboBox()
        self.bitrate_mode_combo.addItems(["Dynamic (VBR/CRF)", "Constant (CBR)"])
        vf.addRow("Bitrate Mode:", self.bitrate_mode_combo)

        # Quality level row – visible only in Dynamic mode
        self._quality_row_lbl = QLabel("Quality Level:")
        self.quality_level_combo = QComboBox()
        self.quality_level_combo.addItems(["Max", "High", "Medium", "Low"])
        vf.addRow(self._quality_row_lbl, self.quality_level_combo)

        # Target bitrate row – visible only in Constant mode
        self._bitrate_row_lbl = QLabel("Target Bitrate (Mbps):")
        self.target_bitrate_combo = QComboBox()
        self.target_bitrate_combo.addItems([
            "1", "2.5", "4", "5", "7.5", "8", "12", "16", "24", "40", "60", "120", "180"
        ])
        self.target_bitrate_combo.setCurrentText("8")
        vf.addRow(self._bitrate_row_lbl, self.target_bitrate_combo)

        # Wire up dynamic visibility; initialise to Dynamic mode
        self.bitrate_mode_combo.currentTextChanged.connect(self._on_bitrate_mode_changed)
        self._on_bitrate_mode_changed(self.bitrate_mode_combo.currentText())

        self.video_backend_combo = QComboBox()
        self.video_backend_combo.addItems(["ffmpeg", "opencv"])
        self.video_backend_combo.setToolTip(
            "ffmpeg: high-quality encoding via FFmpeg (recommended)\n"
            "opencv: fallback OpenCV VideoWriter (mp4/avi only, no 10-bit)"
        )
        vf.addRow("Video Backend:", self.video_backend_combo)
        codec_layout.addWidget(self._video_export_group)

        # ── Image sequence group (visible when Image Sequence Mode is active) ─
        self._image_export_group, imgf = _make_group("Image Sequence Export")
        self.image_sequence_format_combo = QComboBox()
        self.image_sequence_format_combo.addItems(list(IMAGE_SEQUENCE_PROFILES.keys()))
        imgf.addRow("File Type:", self.image_sequence_format_combo)

        # Bit depth dropdown — options are populated dynamically by _on_image_format_changed
        self.image_bit_depth_combo = QComboBox()
        imgf.addRow("Bit Depth:", self.image_bit_depth_combo)

        # Populate initial bit depth options from the default format
        self.image_sequence_format_combo.currentTextChanged.connect(self._on_image_format_changed)
        self._on_image_format_changed(self.image_sequence_format_combo.currentText())

        self._image_export_group.setVisible(False)
        codec_layout.addWidget(self._image_export_group)

        # ── Common output options (always visible) ──────────────────────
        g, f = _make_group("Output Options")
        self.use_10bit_check = QCheckBox()
        f.addRow("10-bit Output:", self.use_10bit_check)

        self.color_correction_combo = QComboBox()
        self.color_correction_combo.addItems(["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
        f.addRow("Color Correction:", self.color_correction_combo)
        codec_layout.addWidget(g)
        codec_layout.addStretch(1)

        # Pack both panes into the scroll host
        _host_layout.addWidget(self._adj_pane)
        _host_layout.addWidget(self._codec_pane)

        # Constrain input widgets: comboboxes are capped at 180 px (~20 chars) so
        # they never push past the right border of the panel.  Spinboxes are capped
        # at 80 px.  Checkboxes are kept at a compact indicator-only width.
        for _cw in _host.findChildren(QComboBox):
            _cw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            _cw.setMaximumWidth(180)
        for _cw in _host.findChildren(QSpinBox):
            _cw.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            _cw.setMaximumWidth(80)
        for _cw in _host.findChildren(QDoubleSpinBox):
            _cw.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            _cw.setMaximumWidth(80)
        for _cw in _host.findChildren(QCheckBox):
            _cw.setMaximumWidth(18)
            _cw.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Per-widget width overrides (applied after the global loop above).
        # DiT Model: closed button ≈ 30 chars (~240 px); popup auto-fits full names.
        self.dit_model_combo.setMaximumWidth(240)
        self.dit_model_combo.view().setMinimumWidth(460)
        self.dit_model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        # Pre-Downscale: compact ≈ 10 chars (~80 px).
        self.pre_downscale_combo.setMaximumWidth(80)
        self.pre_downscale_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        # Resolution container and Batch stepper: cap to 280 px so they stay compact.
        _res_mode_container.setMaximumWidth(280)
        self._batch_stepper_widget.setMaximumWidth(280)

        scroll.setWidget(_host)
        outer_layout.addWidget(scroll, stretch=1)

        # Preview and Export buttons are added to the left panel (under viewer) in _build_left_panel.
        # Create them here as instance attributes so they exist before left panel is built.
        self.preview_btn = QPushButton("⚡ Preview")
        self.preview_btn.setMinimumWidth(90)
        self.preview_btn.setToolTip(
            "Capture the current frame from the timeline, upscale it as a single PNG,\n"
            "and display original vs upscaled side-by-side in Split View."
        )
        self.preview_btn.clicked.connect(self._preview_run)

        self.run_btn = QPushButton("▶  Export")
        self.run_btn.setObjectName("primary_button")
        self.run_btn.setMinimumWidth(110)
        self.run_btn.clicked.connect(self._run)

        return outer

    def _switch_right_tab(self, idx: int) -> None:
        """Show the Adjustments pane (idx=0) or Codec settings pane (idx=1)."""
        self._adj_pane.setVisible(idx == 0)
        self._codec_pane.setVisible(idx == 1)

    def _on_bitrate_mode_changed(self, mode: str) -> None:
        """Toggle Quality Level / Target Bitrate rows based on selected Bitrate Mode."""
        is_dynamic = "Dynamic" in mode
        self._quality_row_lbl.setVisible(is_dynamic)
        self.quality_level_combo.setVisible(is_dynamic)
        self._bitrate_row_lbl.setVisible(not is_dynamic)
        self.target_bitrate_combo.setVisible(not is_dynamic)

    def _on_image_format_changed(self, fmt: str) -> None:
        """Repopulate image_bit_depth_combo when the file-type changes.

        Looks up available bit-depth options from IMAGE_BIT_DEPTHS and
        restores the previously selected label where possible.
        """
        if not hasattr(self, "image_bit_depth_combo"):
            return
        depths = IMAGE_BIT_DEPTHS.get(fmt, [{"label": "8-bit", "pix_fmt": "rgb24", "is_10bit": False}])
        prev = self.image_bit_depth_combo.currentText()
        self.image_bit_depth_combo.blockSignals(True)
        self.image_bit_depth_combo.clear()
        self.image_bit_depth_combo.addItems([d["label"] for d in depths])
        keep = self.image_bit_depth_combo.findText(prev)
        self.image_bit_depth_combo.setCurrentIndex(keep if keep >= 0 else 0)
        self.image_bit_depth_combo.blockSignals(False)

    def _on_file_format_changed(self, fmt: str) -> None:
        """Handle Output Type dropdown changes, coupling codec group and backing state."""
        if getattr(self, "_updating_output_mode", False):
            return
        self._updating_output_mode = True
        try:
            if fmt == "Video":
                self.export_image_sequence_check.setChecked(False)
                self._video_export_group.setVisible(True)
                self._image_export_group.setVisible(False)
            elif fmt == "Image Seq.":
                self.export_image_sequence_check.setChecked(True)
                self._video_export_group.setVisible(False)
                self._image_export_group.setVisible(True)
        finally:
            self._updating_output_mode = False
        self._update_export_controls()

    def _update_output_mode(self, idx: int) -> None:
        """Switch between Video Mode (idx=0) and Image Sequence Mode (idx=1).

        Drives the invisible ``export_image_sequence_check`` backing widget that
        all downstream code (``_build_args``, ``_selected_export_extension``, etc.)
        reads.  Uses a guard flag so the ``_on_image_sequence_toggled`` callback
        does not create a cycle.
        """
        is_image_seq = (idx == 1)
        self._updating_output_mode = True
        try:
            self.export_image_sequence_check.setChecked(is_image_seq)
            if hasattr(self, "file_format_combo"):
                self.file_format_combo.blockSignals(True)
                target = "Image Seq." if is_image_seq else "Video"
                fidx = self.file_format_combo.findText(target)
                if fidx >= 0:
                    self.file_format_combo.setCurrentIndex(fidx)
                self.file_format_combo.blockSignals(False)
        finally:
            self._updating_output_mode = False
        self._video_export_group.setVisible(not is_image_seq)
        self._image_export_group.setVisible(is_image_seq)
        self._update_export_controls()

    def _on_image_sequence_toggled(self, is_seq: bool) -> None:
        """Sync the file_format_combo when the backing checkbox is set externally.

        This is called when preset load/save restores ``export_image_sequence_check``
        (via the persistable widget map) so the Output Type combo stays in sync
        even without user interaction.
        """
        if getattr(self, "_updating_output_mode", False):
            return
        # Sync file_format_combo to "Image Seq." or "Video"
        if hasattr(self, "file_format_combo"):
            self.file_format_combo.blockSignals(True)
            target = "Image Seq." if is_seq else "Video"
            fidx = self.file_format_combo.findText(target)
            if fidx >= 0:
                self.file_format_combo.setCurrentIndex(fidx)
            self.file_format_combo.blockSignals(False)
        # Sync group visibility
        self._video_export_group.setVisible(not is_seq)
        self._image_export_group.setVisible(is_seq)


    # ── Bottom bar ─────────────────────────────────────────────────────

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        # Status row
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setObjectName("danger_button")
        self.abort_btn.clicked.connect(self._abort)

        self.copy_log_btn = QPushButton("Copy All")
        self.copy_log_btn.setToolTip("Copy all log output to clipboard")
        self.copy_log_btn.clicked.connect(self._copy_log)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(lambda: self.console.clear())

        self.open_output_folder_btn = QPushButton("Open Output Folder")
        self.open_output_folder_btn.setEnabled(True)
        self.open_output_folder_btn.clicked.connect(self._open_output_folder)

        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        self.run_btn.setText("Export")

        # Minimal progress panel
        self.progress_panel = QFrame()
        self.progress_panel.setObjectName("progress_panel")
        self.progress_panel.setStyleSheet(
            "QFrame#progress_panel {"
            "background:#1A1D21;"
            "border:1px solid #2E3338;"
            "border-radius:8px;"
            "padding:2px;"
            "}"
        )
        panel_layout = QVBoxLayout(self.progress_panel)
        panel_layout.setContentsMargins(8, 4, 8, 4)
        panel_layout.setSpacing(3)

        self.batch_progress_circle = _NoOpProgressIndicator()
        self.phase_progress_circle = _NoOpProgressIndicator()
        self.eta_progress_circle = _NoOpProgressIndicator()
        self.queue_progress_circle = _NoOpProgressIndicator()
        self.elapsed_progress_circle = _NoOpProgressIndicator()

        self.current_file_progress_label = QLabel("Current File: -")
        self.batch_progress_label = QLabel("Overall Batch Progress | Completed: 0/0")
        self.video_proc_time_label = QLabel("Video Processing Time: 00:00")
        panel_layout.addWidget(self.current_file_progress_label)
        panel_layout.addWidget(self.batch_progress_label)
        panel_layout.addWidget(self.video_proc_time_label)
        layout.addWidget(self.progress_panel)

        # Console + controls (70/30 split)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(96)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.console.setFont(mono)

        controls_widget = QWidget()
        controls_widget.setFixedWidth(170)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        for btn in (
            self.open_output_folder_btn,
            self.copy_log_btn,
            self.clear_log_btn,
            self.run_btn,
            self.abort_btn,
        ):
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            controls_layout.addWidget(btn)
        controls_layout.addStretch(1)

        log_row = QHBoxLayout()
        log_row.setContentsMargins(0, 0, 0, 0)
        log_row.setSpacing(8)
        log_row.addWidget(self.console, 7)
        log_row.addWidget(controls_widget)
        layout.addLayout(log_row, 1)
        layout.setStretch(0, 0)
        layout.setStretch(1, 0)
        layout.setStretch(2, 1)

        return bar

    # ------------------------------------------------------------------
    # Settings window
    # ------------------------------------------------------------------

    def _open_settings(self) -> None:
        """Show (or raise) the persistent Paths & Configuration window."""
        self._settings_win.show()
        self._settings_win.raise_()
        self._settings_win.activateWindow()

    def _open_folders_dialog(self) -> None:
        """Open the Input / Output Folders modal dialog."""
        self._folders_dlg.load_io_paths()
        self._folders_dlg.exec()

    def _toggle_advanced_mode(self, checked: bool) -> None:
        self._advanced_mode_enabled = checked
        self._apply_mode_visibility()

    def _apply_mode_visibility(self) -> None:
        advanced = bool(self._advanced_mode_enabled)
        self.advanced_mode_btn.setText("Advanced Mode" if advanced else "Simple Mode")

        if not advanced:
            self.pre_downscale_combo.setCurrentText(str(self._simple_defaults["pre_downscale"]))
            self.resolution_mode_combo.setCurrentText(str(self._simple_defaults["resolution_mode"]))
            self.batch_size_spin.setValue(int(self._simple_defaults["batch_size"]))
            self.enable_video_chunking_check.setChecked(bool(self._simple_defaults["enable_video_chunking"]))
            self.split_size_minutes_spin.setValue(int(self._simple_defaults["split_minutes"]))
            self.vae_encode_tiled_check.setChecked(bool(self._simple_defaults["vae_tiling"]))
            self.vae_decode_tiled_check.setChecked(bool(self._simple_defaults["vae_tiling"]))
            self.debug_check.setChecked(bool(self._simple_defaults["debug"]))

        # Simple mode: show Processing Settings, VAE Tiling, Debug; hide Memory, Performance, Model Cache.
        # Advanced mode: show everything.
        self._preview_processing_group.setVisible(advanced)
        self._memory_blockswap_group.setVisible(advanced)
        self._performance_group.setVisible(advanced)
        self._quality_control_group.setVisible(advanced)
        self._model_cache_group.setVisible(advanced)
        # Max Resolution is hidden in Simple Mode to avoid user confusion.
        self._max_resolution_label.setVisible(advanced)
        self.max_resolution_spin.setVisible(advanced)
        if not advanced:
            self.max_resolution_spin.setValue(0)
        # These are always visible (simple or advanced):
        # _processing_settings_group, _vae_tiling_group, _debug_group

    def _update_chunking_visibility(self, enabled: bool) -> None:
        self.chunk_duration_minutes_label.setVisible(enabled)
        self.split_size_minutes_spin.setVisible(enabled)

    # ------------------------------------------------------------------
    # In/Out point keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        """Handle global hotkeys: [ sets In-Point, ] sets Out-Point."""
        key = event.key()
        if key == Qt.Key.Key_BracketLeft:
            self._set_in_point()
            return
        if key == Qt.Key.Key_BracketRight:
            self._set_out_point()
            return
        super().keyPressEvent(event)

    def _build_menu_bar(self) -> None:
        settings_menu = self.menuBar().addMenu("Settings")
        open_settings_action = QAction("Open Settings", self)
        open_settings_action.triggered.connect(self._open_settings)
        settings_menu.addAction(open_settings_action)

        help_menu = self.menuBar().addMenu("About")
        about_action = QAction("About 1Click SeedVR2.5", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

        github_action = QAction("Open GitHub", self)
        github_action.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://github.com/naxci1/1Click_SeedVR2.5")
            )
        )
        help_menu.addAction(github_action)

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About 1Click SeedVR2.5 ver. 1.7b (by Naxci1)",
            (
                "<b>1Click SeedVR2.5 ver. 1.7b (by Naxci1)</b><br>"
                "Version: v1.7b<br><br>"
                "Professional video upscaler powered by SeedVR2.<br>"
                "License: Apache-2.0<br><br>"
                '<a href="https://github.com/naxci1/1Click_SeedVR2.5">'
                "GitHub Repository</a>"
            ),
        )

    def _build_persistable_widget_map(self) -> dict[str, QWidget]:
        return {
            "dit_model_combo": self.dit_model_combo,
            "container_combo": self.container_combo,
            "video_codec_combo": self.video_codec_combo,
            "export_image_sequence_check": self.export_image_sequence_check,
            "image_sequence_format_combo": self.image_sequence_format_combo,
            "image_bit_depth_combo": self.image_bit_depth_combo,
            "audio_mode_combo": self.audio_mode_combo,
            "video_backend_combo": self.video_backend_combo,
            "bitrate_mode_combo": self.bitrate_mode_combo,
            "quality_level_combo": self.quality_level_combo,
            "target_bitrate_combo": self.target_bitrate_combo,
            "use_10bit_check": self.use_10bit_check,
            "color_correction_combo": self.color_correction_combo,
            "pre_downscale_combo": self.pre_downscale_combo,
            "resolution_mode_combo": self.resolution_mode_combo,
            "resolution_times_combo": self.resolution_times_combo,
            "resolution_standard_combo": self.resolution_standard_combo,
            "resolution_spin": self.resolution_spin,
            "max_resolution_spin": self.max_resolution_spin,
            "batch_size_spin": self.batch_size_spin,
            "uniform_batch_check": self.uniform_batch_check,
            "temporal_overlap_spin": self.temporal_overlap_spin,
            "prepend_frames_spin": self.prepend_frames_spin,
            "skip_first_frames_spin": self.skip_first_frames_spin,
            "load_cap_spin": self.load_cap_spin,
            "only_frames_spin": self.only_frames_spin,
            "enable_video_chunking_check": self.enable_video_chunking_check,
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
            "input_noise_scale_spin": self.input_noise_scale_spin,
            "latent_noise_scale_spin": self.latent_noise_scale_spin,
            "cache_dit_check": self.cache_dit_check,
            "cache_vae_check": self.cache_vae_check,
            "auto_tune_check": self.auto_tune_check,
            "debug_check": self.debug_check,
            "enable_audio_notifications_check": self.enable_audio_notifications_check,
            "file_format_combo": self.file_format_combo,
        }

    # Codecs that have fixed/required container associations (codec name → required container key)
    _CODEC_LOCKED_CONTAINER: dict[str, str] = {
        "ProRes 422 Proxy": "MOV",
        "ProRes 422 LT": "MOV",
        "ProRes 422": "MOV",
        "ProRes 422 HQ": "MOV",
        "ProRes 4444 XQ": "MOV",
        "QuickTime Animation (Alpha)": "MOV",
        "Uncompressed RGB (R210)": "MOV",
        "FFV1 (Lossless 8/10/12-bit)": "MKV",
    }

    def _on_container_changed(self, _index: int = 0) -> None:
        """Repopulate the codec combo to show only codecs compatible with the selected container."""
        container = self.container_combo.currentText()
        codecs = CONTAINER_CODECS.get(container, [])
        prev = self.video_codec_combo.currentText()
        self.video_codec_combo.blockSignals(True)
        self.video_codec_combo.clear()
        self.video_codec_combo.addItems(codecs)
        keep_idx = self.video_codec_combo.findText(prev)
        self.video_codec_combo.setCurrentIndex(keep_idx if keep_idx >= 0 else 0)
        self.video_codec_combo.blockSignals(False)
        self._update_export_controls()

    def _update_export_controls(self, *_: object) -> None:
        exporting_sequence = self.export_image_sequence_check.isChecked()
        self.video_codec_combo.setEnabled(not exporting_sequence)
        self.image_sequence_format_combo.setEnabled(exporting_sequence)

        if exporting_sequence:
            return

        # Ensure codec list is filtered for the current container (no-op if already correct)
        container = self.container_combo.currentText()
        expected_codecs = CONTAINER_CODECS.get(container, [])
        current_codecs = [self.video_codec_combo.itemText(i) for i in range(self.video_codec_combo.count())]
        if current_codecs != expected_codecs:
            prev_codec = self.video_codec_combo.currentText()
            self.video_codec_combo.blockSignals(True)
            self.video_codec_combo.clear()
            self.video_codec_combo.addItems(expected_codecs)
            keep_idx = self.video_codec_combo.findText(prev_codec)
            self.video_codec_combo.setCurrentIndex(keep_idx if keep_idx >= 0 else 0)
            self.video_codec_combo.blockSignals(False)

        # Keep file_format_combo showing "Video" when in video mode
        if hasattr(self, "file_format_combo") and not getattr(self, "_updating_output_mode", False):
            if self.file_format_combo.currentText() != "Video":
                self.file_format_combo.blockSignals(True)
                fidx = self.file_format_combo.findText("Video")
                if fidx >= 0:
                    self.file_format_combo.setCurrentIndex(fidx)
                self.file_format_combo.blockSignals(False)

    def _selected_export_extension(self) -> str:
        if self.export_image_sequence_check.isChecked():
            fmt = self.image_sequence_format_combo.currentText()
            profile = IMAGE_SEQUENCE_PROFILES.get(fmt, {})
            return str(profile.get("ext", ".png"))
        codec = self.video_codec_combo.currentText()
        container = UNIFIED_VIDEO_CODEC_PROFILES.get(codec, {}).get("container", "MP4")
        return "." + container.lower() if container else ".mp4"

    @staticmethod
    def _extract_encoder_from_args(ffmpeg_args: list[str], codec_fallback: str = "") -> str:
        encoder = ""
        for i, tok in enumerate(ffmpeg_args):
            if tok == "-c:v" and i + 1 < len(ffmpeg_args):
                encoder = str(ffmpeg_args[i + 1]).strip()
                break
        if not encoder:
            encoder = codec_fallback.strip().lower().replace(" ", "_")
        return encoder or "libx264"

    @staticmethod
    def _strip_quality_flags(ffmpeg_args: list[str]) -> list[str]:
        flags_with_value = {
            "-preset",
            "-crf",
            "-cq",
            "-qp",
            "-q:v",
            "-qscale:v",
            "-global_quality",
            "-profile:v",
            "-b:v",
            "-maxrate",
            "-bufsize",
        }
        stripped: list[str] = []
        skip_next = False
        for tok in ffmpeg_args:
            if skip_next:
                skip_next = False
                continue
            if tok in flags_with_value:
                skip_next = True
                continue
            stripped.append(tok)
        return stripped

    def _ui_selected_bitrate(self) -> str:
        mode = self.bitrate_mode_combo.currentText()
        if "Constant" in mode:
            return f"{self.target_bitrate_combo.currentText()}M"
        dynamic_quality = self.quality_level_combo.currentText()
        dynamic_map = {
            "Max": "120M",
            "High": "40M",
            "Medium": "20M",
            "Low": "8M",
        }
        return dynamic_map.get(dynamic_quality, "20M")

    def _max_quality_flags_for_encoder(self, encoder: str, bitrate: str) -> list[str]:
        enc = encoder.lower()
        # Family detection is capability-based, not per-codec hardcoding.
        if "nvenc" in enc:
            return ["-preset", "p7", "-cq", "16", "-b:v", bitrate]
        if "prores" in enc:
            return ["-profile:v", "3"]
        # Generic software/other encoders: apply aggressive quality-first defaults.
        if "libx26" in enc or "libaom" in enc or "svt" in enc or "rav1e" in enc or "vpx" in enc:
            return ["-preset", "veryslow", "-crf", "12", "-b:v", bitrate]
        return ["-crf", "12", "-b:v", bitrate]

    def _image_seq_bit_depth_info(self) -> dict[str, Any]:
        """Return the {label, pix_fmt, is_10bit} dict for the currently selected image format + bit depth."""
        fmt = self.image_sequence_format_combo.currentText()
        depths = IMAGE_BIT_DEPTHS.get(fmt, [{"label": "8-bit", "pix_fmt": "rgb24", "is_10bit": False}])
        label = self.image_bit_depth_combo.currentText() if hasattr(self, "image_bit_depth_combo") else ""
        return next((d for d in depths if d["label"] == label), depths[0])

    def _selected_export_profile_to_ffmpeg_args(self) -> dict[str, Any]:
        """Return a backend mapping of current export choices to FFmpeg arguments."""
        if self.export_image_sequence_check.isChecked():
            image_fmt = self.image_sequence_format_combo.currentText()
            depth_info = self._image_seq_bit_depth_info()
            pix_fmt = depth_info["pix_fmt"]
            return {
                "mode": "image_sequence",
                "container": "image2",
                "image_format": image_fmt,
                "video_args": ["-f", "image2", "-pix_fmt", pix_fmt],
                "audio_args": ["-an"],
            }

        codec = self.video_codec_combo.currentText()
        codec_profile = UNIFIED_VIDEO_CODEC_PROFILES.get(codec, {})
        container = codec_profile.get("container", "MP4")
        audio = self.audio_mode_combo.currentText()
        audio_args = AUDIO_PROFILES.get(audio, AUDIO_PROFILES["Copy Audio"])

        base_video_args = list(codec_profile.get("ffmpeg", []))
        encoder = self._extract_encoder_from_args(base_video_args, codec_fallback=codec)

        if "prores" in encoder.lower():
            # ProRes: -profile:v selects the codec variant (Proxy/LT/Standard/HQ/4444 XQ).
            # Preserve the base args exactly — overriding -profile:v would silently change
            # which ProRes tier gets encoded, causing unexpected quality/compatibility.
            final_video_args = base_video_args
        else:
            # Generic path: strip per-codec quality settings, then re-apply universal
            # max-quality flags appropriate for the encoder family.
            sanitized_video_args = self._strip_quality_flags(base_video_args)
            bitrate = self._ui_selected_bitrate()
            quality_flags = self._max_quality_flags_for_encoder(encoder, bitrate)
            if "-c:v" not in sanitized_video_args:
                sanitized_video_args = ["-c:v", encoder] + sanitized_video_args
            final_video_args = sanitized_video_args + quality_flags

        return {
            "mode": "video",
            "container": container,
            "codec": codec,
            "video_args": final_video_args,
            "audio_mode": audio,
            "audio_args": audio_args,
        }

    def _selected_profile_is_10bit(self) -> bool:
        if self.export_image_sequence_check.isChecked():
            return bool(self._image_seq_bit_depth_info().get("is_10bit", False))
        codec = self.video_codec_combo.currentText()
        profile = UNIFIED_VIDEO_CODEC_PROFILES.get(codec, {})
        return bool(profile.get("is_10bit", False))

    @staticmethod
    def _save_qimage_as_tiff16(image: "QImage", file_path: str) -> bool:
        """Save *image* as an uncompressed 16-bit TIFF.

        The source QImage is promoted to a 64-bit (16-bit-per-channel) format so
        the on-disk TIFF carries 16-bit depth, avoiding the precision loss of
        8-bit PNG for preview and intermediate frames. Returns True on success.
        """
        try:
            img16 = image.convertToFormat(QImage.Format.Format_RGBX64)
            if img16.isNull():
                img16 = image
            writer = QImageWriter(file_path, b"tiff")
            # 0 = no compression (uncompressed TIFF).
            writer.setCompression(0)
            return bool(writer.write(img16))
        except Exception:
            # Fall back to Qt's default TIFF writer if 16-bit promotion fails.
            try:
                return bool(image.save(file_path, "TIFF"))
            except Exception:
                return False

    def _resolve_export_output_dir(self) -> Path:
        """Return the base output directory for all generated files.

        The output (and temp) location is always *exactly* the parent directory
        of the input file (``os.path.dirname(input_file)``) for every operation
        (Preview, Upscale, Split). No subfolders or nested directories are ever
        created, and configured/user-selected output paths are intentionally
        ignored so that every generated file lands directly beside the input.
        """
        inp = self._folders_dlg.input_edit.text().strip()
        if inp:
            inp_path = Path(inp)
            # A directory input is itself the base; a file input uses its parent.
            return inp_path if inp_path.is_dir() else inp_path.parent
        return Path.cwd()

    @staticmethod
    def _ensure_unique_file_path(path: Path) -> Path:
        """Return a non-colliding file path by appending _N before suffix."""
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    @staticmethod
    def _seedvr_prefixed_stem(stem: str) -> str:
        clean_stem = stem.strip()
        return clean_stem if clean_stem.startswith("seedvr2_") else f"seedvr2_{clean_stem}"

    @classmethod
    def _generate_export_output_path(cls, ext: str, output_dir: Path, part_idx: int = 1) -> Path:
        """Return a unique output path using padded numerical indexing.

        Format: ``seedvr_output_part_NNN_MMMMM<ext>``.
        *part_idx* is the chunk/part number (1-based); the file counter increments
        until a non-existing path is found.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_idx in range(1, 100_000):
            stem = cls._seedvr_prefixed_stem(f"output_part_{part_idx:03d}_{file_idx:05d}")
            candidate = output_dir / f"{stem}{ext}"
            if not candidate.exists():
                return candidate
        return output_dir / f"{cls._seedvr_prefixed_stem(f'output_part_{part_idx:03d}_99999')}{ext}"

    def _serialize_model_settings(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, widget in self._persistable_widgets.items():
            if isinstance(widget, CheckableComboBox):
                data[key] = widget.checkedTexts()
            elif isinstance(widget, QComboBox):
                # Prefer item data (full value) over display text for combos using addItem(display, data)
                item_data = widget.currentData()
                data[key] = item_data if item_data is not None else widget.currentText()
            elif isinstance(widget, QSpinBox):
                data[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
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
                # Try matching by item data (full value) first, then by display text
                idx = widget.findData(str(value))
                if idx < 0:
                    idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QSpinBox):
                try:
                    widget.setValue(int(value))
                except (TypeError, ValueError):
                    continue
            elif isinstance(widget, QDoubleSpinBox):
                try:
                    widget.setValue(float(value))
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
        if not self._folders_dlg.input_edit.text().strip():
            self._on_log("⚠  Queue: set an input file/folder first.")
            return
        self._queue_entry_counter += 1
        job = {
            "id": self._queue_entry_counter,
            "paths": {
                "python_exe": self._settings_win.python_exe_edit.text().strip(),
                "seedvr2_folder": self._settings_win.seedvr2_folder_edit.text().strip(),
                "input_mode": self._folders_dlg.input_mode_combo.currentText(),
                "input_path": self._folders_dlg.input_edit.text().strip(),
                "output_path": self._folders_dlg.output_edit.text().strip(),
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
        self._folders_dlg.input_edit.setText(str(paths.get("input_path", "")))
        self._folders_dlg.output_edit.setText(str(paths.get("output_path", "")))
        self._settings_win.model_dir_edit.setText(str(paths.get("model_dir", "")))
        mode = str(paths.get("input_mode", "File"))
        idx = self._folders_dlg.input_mode_combo.findText(mode)
        if idx >= 0:
            self._folders_dlg.input_mode_combo.setCurrentIndex(idx)
        model_settings = job.get("model_settings", {})
        if isinstance(model_settings, dict):
            self._apply_model_settings(model_settings)
        inp = self._folders_dlg.input_edit.text().strip()
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
        if path != self._preview_temp_path:
            self._preview_compare_active = False
        _IMAGE_SUFFIXES = set(SUPPORTED_IMAGE_EXTS) | {".bmp", ".webp", ".gif"}
        suffix = Path(path).suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            self._current_input_is_image = True
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
            # Reset trim markers whenever a new video source is loaded
            self._in_point_ms = None
            self._out_point_ms = None
            self._current_fps = 0.0
            if hasattr(self, "_seek_slider"):
                self._seek_slider.set_trim_fractions(None, None)
            # Extract and display the first frame immediately for instant preview
            self._try_extract_first_frame(path)
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
        self._update_input_frame_counter_label(self._input_player.position())

    def _update_input_frame_counter_label(self, position_ms: Optional[int] = None) -> None:
        """Update input metadata as: Input: <file> | <current>/<total> frame."""
        if not _MULTIMEDIA_AVAILABLE or not self._input_player or self._current_input_is_image:
            return
        input_path = self._folders_dlg.input_edit.text().strip()
        if not input_path:
            return
        if position_ms is None:
            position_ms = self._input_player.position()

        fps_val: Optional[float] = None
        try:
            meta = self._input_player.metaData()
            fps = meta.value(QMediaMetaData.Key.VideoFrameRate)
            if fps is not None:
                fps_val = float(fps)
        except Exception:
            fps_val = None

        # Cache FPS for In/Out point calculations
        if fps_val is not None and fps_val > 0:
            self._current_fps = fps_val

        duration_ms = self._input_player.duration()
        total_frames = 0
        current_frame = 0
        if fps_val is not None and fps_val > 0 and duration_ms > 0:
            total_frames = max(1, int(round((duration_ms / 1000.0) * fps_val)))
            current_frame = max(1, int((max(0, position_ms) / 1000.0) * fps_val) + 1)
            current_frame = min(current_frame, total_frames)

        filename = Path(input_path).name
        self._input_meta_text = f"Input: {filename} | {current_frame}/{total_frames} frame"
        self._meta_label.setText(self._input_meta_text)

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
        inp = self._folders_dlg.input_edit.text().strip()
        input_mode = self._folders_dlg.input_mode_combo.currentText()
        input_is_directory = (
            not self._is_preview_run
            and (
                input_mode == "Folder"
                or (inp and Path(inp).is_dir())
            )
        )
        args.append(inp)

        # output
        # The output is always forced to the input file's parent directory
        # (resolved via _resolve_export_output_dir). User-selected, auto-derived
        # and restored output paths are all ignored so that every generated file
        # lands directly beside the input with no subfolders.
        out = ""
        if self._is_preview_run:
            if self._is_preview_video_mode:
                # Video-mode preview: short clip in the selected container.
                export_dir = self._resolve_export_output_dir()
                container_ext = self._selected_export_extension()
                preview_out = self._generate_export_output_path(container_ext, export_dir)
            else:
                # Image-mode preview: force the upscaled frame to a direct file
                # path located strictly inside the original input file's parent
                # directory, sitting right beside the original video. No
                # automatic prefixing or folder-creation helpers are used so the
                # output never lands in a nested subfolder.
                orig = self._preview_original_input_path or inp
                orig_path = Path(orig) if orig else None
                if orig_path is not None and str(orig_path.parent) not in ("", "."):
                    parent_dir = orig_path.parent
                else:
                    parent_dir = self._resolve_export_output_dir()
                base_stem = orig_path.stem if (orig_path and orig_path.stem) else "preview"
                preview_out = self._ensure_unique_file_path(
                    parent_dir / f"{base_stem}_preview_upscaled.tiff"
                )
            # Track the exact output path so _on_finished can load it reliably
            # from disk without depending on any UI text field.
            self._preview_output_path = str(preview_out)
            self._folders_dlg.output_edit.setText(str(preview_out))
            args += ["--output", str(preview_out)]
        elif input_is_directory:
            # Directory-mode should mirror native CLI usage:
            #   python inference_cli.py <folder> --output <directory>
            if out:
                args += ["--output", out]
            else:
                args += ["--output", str(self._resolve_export_output_dir())]
        elif out:
            out_path = Path(out)
            if out_path.suffix:
                # Force-enforce the correct extension based on the selected export
                # format BEFORE launching the subprocess.  A user-supplied path may
                # carry a stale extension (e.g. ".png") that does not match the
                # chosen container/image-sequence format.  Reconcile it here so the
                # GUI never sends a mismatched extension (such as ".png" when MP4 is
                # selected) to inference_cli.py.
                correct_ext = self._selected_export_extension()
                if correct_ext and out_path.suffix.lower() != correct_ext.lower():
                    out_path = out_path.with_suffix(correct_ext)
                args += ["--output", str(self._ensure_unique_file_path(out_path))]
            else:
                args += ["--output", out]
        else:
            # No output path set: derive from input filename when processing a single file,
            # otherwise fall back to padded numerical naming convention.
            export_dir = self._resolve_export_output_dir()
            auto_ext = self._selected_export_extension()
            inp_path = Path(inp) if inp else None
            if inp_path and inp_path.is_file():
                # Use seedvr2_<original_stem> as the output name, deduplicating as needed.
                auto_stem = f"seedvr2_{inp_path.stem}"
                auto_out = self._ensure_unique_file_path(export_dir / f"{auto_stem}{auto_ext}")
            else:
                auto_out = self._generate_export_output_path(auto_ext, export_dir)
            args += ["--output", str(auto_out)]

        # model dir
        md = self._settings_win.model_dir_edit.text().strip()
        if md:
            args += ["--model_dir", md]

        # output format mapping for SeedVR2 CLI:
        # - image-mode preview ALWAYS maps to 16-bit TIFF (single frame) — highest priority
        # - image sequence modes map to CLI output_format=<actual ext without dot>
        # - normal video export maps to the selected container
        _is_image_preview = self._is_preview_run and not self._is_preview_video_mode
        if _is_image_preview:
            # Image-mode preview: always a single high-fidelity 16-bit TIFF frame,
            # regardless of any export settings.  This must take priority over the
            # image-sequence checkbox so that the CLI never interprets the
            # single-frame capture as a video pipeline run.
            args += ["--output_format", "tiff"]
        elif self.export_image_sequence_check.isChecked():
            # Pass the real extension so the CLI writes TIFF/DPX/EXR/JPEG correctly,
            # not always PNG.  Use the profile ext but strip the leading dot.
            img_fmt = self.image_sequence_format_combo.currentText()
            img_ext = IMAGE_SEQUENCE_PROFILES.get(img_fmt, {}).get("ext", ".png")
            # The CLI treats "png" as an image sequence trigger; for other formats
            # we pass the bare extension name (e.g. "tiff", "jpg", "dpx", "exr").
            cli_fmt = img_ext.lstrip(".") if img_ext else "png"
            args += ["--output_format", cli_fmt]
        else:
            container = self.container_combo.currentText().lower()
            args += ["--output_format", container or "mp4"]

        # video backend – user-selectable (ffmpeg or opencv)
        video_backend = self.video_backend_combo.currentText() or "ffmpeg"
        args += ["--video_backend", video_backend]
        # Emit ffmpeg_video_args only when ffmpeg backend is selected.
        # Skip for image-mode preview TIFF runs and image sequence exports.
        if not _is_image_preview and not self.export_image_sequence_check.isChecked() and video_backend == "ffmpeg":
            profile = self._selected_export_profile_to_ffmpeg_args()
            video_codec_args = profile.get("video_args", [])
            if video_codec_args:
                import json as _json
                args += ["--ffmpeg_video_args", _json.dumps(video_codec_args)]

        # 10-bit: either explicitly requested or implied by selected export profile
        if self.use_10bit_check.isChecked() or self._selected_profile_is_10bit():
            args.append("--10bit")

        # dit model
        _dit_model_val = self.dit_model_combo.currentData()
        args += ["--dit_model", _dit_model_val if _dit_model_val is not None else self.dit_model_combo.currentText()]

        # pre-downscale (preprocessing factor before upscaling)
        pre_ds_text = self.pre_downscale_combo.currentText()  # "1:1", "2:1", "3:1"
        pre_ds_factor = int(pre_ds_text.split(":")[0])  # 1, 2, or 3
        if pre_ds_factor > 1:
            args += ["--pre_downscale", str(pre_ds_factor)]

        # resolution – compute final target from mode + pre-downscale factor
        res_mode = self.resolution_mode_combo.currentText()  # "Pixel", "X Times", or "Standard"
        if res_mode == "X Times":
            # Multiplier applied to the (already pre-downscaled) input height.
            # We don't know the actual input dimension at arg-build time, so we
            # pass a special combined flag that the CLI interprets: negative value
            # signals "X times" mode.  We encode as --resolution_mode xtimes
            # and --resolution_scale <N> for the CLI to interpret.
            times_text = self.resolution_times_combo.currentText()  # "1x".."5x"
            times_val = int(times_text.rstrip("x"))
            args += ["--resolution_mode", "xtimes", "--resolution_scale", str(times_val)]
        elif res_mode == "Standard":
            # Standard presets: extract the leading numeric value ("720 (HD)" → 720)
            std_text = self.resolution_standard_combo.currentText()
            std_val = int(std_text.split()[0])
            args += ["--resolution", str(std_val)]
        else:
            # Pixel mode: direct target resolution — always read from the widget
            args += ["--resolution", str(self.resolution_spin.value())]

        max_res = self.max_resolution_spin.value()
        if max_res != 0:
            args += ["--max_resolution", str(max_res)]

        args += ["--batch_size", str(self.batch_size_spin.value())]

        if self.uniform_batch_check.isChecked():
            args.append("--uniform_batch_size")

        # seed – always hardcoded so the CLI uses a fixed deterministic value
        args += ["--seed", "313"]

        skip = self.skip_first_frames_spin.value()
        if skip:
            args += ["--skip_first_frames", str(skip)]

        # For preview runs, hard-cap to 1 frame so the CLI never feeds more than one
        # frame through the pipeline (prevents accidental video-pipeline activation).
        if _is_image_preview:
            args += ["--load_cap", "1"]
        else:
            load_cap = self.load_cap_spin.value()
            if load_cap:
                args += ["--load_cap", str(load_cap)]

        only_frames = self.only_frames_spin.value()
        if only_frames:
            args += ["--only_frames", str(only_frames)]

        chunking_enabled = self.enable_video_chunking_check.isChecked()
        if chunking_enabled:
            split_minutes = self.split_size_minutes_spin.value()
            # Convert minutes → frames using input video FPS so --chunk_size receives
            # the pre-calculated frame count, which is what the CLI expects.
            # Fall back to --chunk_duration_minutes when FPS cannot be determined.
            fps: float = self._current_fps
            if fps <= 0 and cv2 is not None:
                inp_path = self._folders_dlg.input_edit.text().strip()
                if inp_path:
                    try:
                        cap = cv2.VideoCapture(inp_path)
                        if cap.isOpened():
                            fps = float(cap.get(cv2.CAP_PROP_FPS))
                        cap.release()
                    except Exception:
                        fps = 0.0
            if fps > 0:
                chunk_frames = max(1, int(round(split_minutes * 60.0 * fps)))
                args += ["--chunk_size", str(chunk_frames)]
            else:
                args += ["--chunk_duration_minutes", str(split_minutes)]

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
        if self._advanced_mode_enabled and self.vae_encode_tiled_check.isChecked():
            args.append("--vae_encode_tiled")
            enc_sz = self.vae_encode_tile_size_spin.value()
            if enc_sz != 1024:
                args += ["--vae_encode_tile_size", str(enc_sz)]
            enc_ov = self.vae_encode_tile_overlap_spin.value()
            if enc_ov != 128:
                args += ["--vae_encode_tile_overlap", str(enc_ov)]

        if self._advanced_mode_enabled and self.vae_decode_tiled_check.isChecked():
            args.append("--vae_decode_tiled")
            dec_sz = self.vae_decode_tile_size_spin.value()
            if dec_sz != 1024:
                args += ["--vae_decode_tile_size", str(dec_sz)]
            dec_ov = self.vae_decode_tile_overlap_spin.value()
            if dec_ov != 128:
                args += ["--vae_decode_tile_overlap", str(dec_ov)]

        tile_dbg = self.tile_debug_combo.currentText()
        if self._advanced_mode_enabled and tile_dbg != "false":
            args += ["--tile_debug", tile_dbg]

        # performance
        attn = self.attention_mode_combo.currentText()
        if attn != "sdpa":
            args += ["--attention_mode", attn]

        # quality control – noise injection scales (advanced only; pass when > 0)
        if self._advanced_mode_enabled:
            input_noise = self.input_noise_scale_spin.value()
            if input_noise > 0.0:
                args += ["--input_noise_scale", f"{input_noise:.2f}"]
            latent_noise = self.latent_noise_scale_spin.value()
            if latent_noise > 0.0:
                args += ["--latent_noise_scale", f"{latent_noise:.2f}"]

        # cache
        if self.cache_dit_check.isChecked():
            args.append("--cache_dit")

        if self.cache_vae_check.isChecked():
            args.append("--cache_vae")

        # auto tune
        if self.auto_tune_check.isChecked():
            args.append("--auto_tune")
            # Force tile_overlap to 32 when Auto Tune is active
            if self._advanced_mode_enabled:
                self.vae_encode_tile_overlap_spin.setValue(32)
                self.vae_decode_tile_overlap_spin.setValue(32)

        # debug – works in both simple and advanced mode
        if self.debug_check.isChecked():
            args.append("--debug")

        return args

    # ------------------------------------------------------------------
    # Run / Abort
    # ------------------------------------------------------------------

    def _run(self) -> None:
        inp = self._folders_dlg.input_edit.text().strip()
        if not inp:
            self._on_log("❌  Please specify an input file or directory (⚙ Settings).")
            return
        if (
            not self._is_preview_run
            and self._folders_dlg.input_mode_combo.currentText() == "Folder"
            and not Path(inp).is_dir()
        ):
            self._on_log(f"❌  Folder mode requires a directory path, got: {inp}")
            return
        if not self._is_preview_run:
            self._preview_compare_active = False

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
        self._folders_dlg.save_io_paths()
        self._save_model_settings()
        self._set_latest_output_path(None)

        args = self._build_args()
        ffmpeg_profile = self._selected_export_profile_to_ffmpeg_args()
        self._on_log(f"🎬  Export Profile: {json.dumps(ffmpeg_profile, ensure_ascii=False)}")

        bitrate_mode_text = self.bitrate_mode_combo.currentText()
        if "Constant" in bitrate_mode_text:
            bitrate_text = self.target_bitrate_combo.currentText().strip() or "8"
        else:
            dynamic_map = {"Max": "80", "High": "40", "Medium": "20", "Low": "8"}
            bitrate_text = dynamic_map.get(self.quality_level_combo.currentText(), "20")
        worker_env = {"SEEDVR2_DEFAULT_BITRATE": f"{bitrate_text}M"}

        self._thread, self._worker = create_worker_thread(cli_script, args, python_exe, env=worker_env)
        self._worker.log_line.connect(self._on_log)
        self._worker.progress_update.connect(self._on_global_progress)
        self._worker.batch_progress_update.connect(self._on_batch_progress)
        self._worker.queue_status_update.connect(self._on_queue_status_update)
        self._worker.finished.connect(self._on_finished)
        self._worker.started_signal.connect(lambda: self._set_running(True))

        self._reset_progress_bars()
        # Reset timer state for new export; video_proc_time_label shows 00:00 until first tick
        self._frozen_elapsed_seconds = None
        self.video_proc_time_label.setText("Video Processing Time: 00:00")
        self._run_started_at = time.time()
        self._elapsed_timer.start()
        self._prepare_queue_progress_context()
        self._update_current_file_progress_ui()
        self._update_batch_progress_ui()
        self._update_elapsed_progress_ui()
        self._set_running(True)
        self.status_label.setText("Starting…")
        self._thread.start()

    def _abort(self) -> None:
        if self._worker:
            self._worker.request_abort()
        self.status_label.setText("Aborting…")

    def _resume_original_video_after_preview(self, *, switch_mode: bool = True) -> bool:
        original = self._preview_original_input_path
        if not original or Path(original).suffix.lower() not in SUPPORTED_VIDEO_EXTS:
            return False
        self._preview_compare_active = False
        self._folders_dlg.input_mode_combo.setCurrentText(self._preview_original_input_mode)
        self._folders_dlg.input_edit.setText(original)
        self._current_input_is_image = False
        self._input_player.setVideoOutput(self._solo_input_vw)
        if switch_mode:
            self._mode_input_btn.setChecked(True)
            self._on_mode_button(0, True)
        self._load_input_video(original)
        self._input_player.setPosition(self._preview_original_position)
        if self._output_player:
            self._output_player.pause()
            self._output_player.setPosition(0)
        return True

    def _release_cuda_resources(self) -> None:
        python_exe = self._settings_win.python_exe_edit.text().strip() or DEFAULT_PYTHON_EXE
        if not os.path.isfile(python_exe):
            return
        try:
            subprocess.run(
                [
                    python_exe,
                    "-c",
                    (
                        "import gc; "
                        "gc.collect(); "
                        "import torch; "
                        "torch.cuda.empty_cache() if torch.cuda.is_available() else None; "
                        "torch.cuda.ipc_collect() if torch.cuda.is_available() else None"
                    ),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=_CREATE_NO_WINDOW,
            )
        except Exception:
            pass

    def _force_shutdown(self) -> None:
        self._force_exit = True
        try:
            if self._worker:
                self._worker.force_kill()
        except Exception:
            pass
        try:
            if self._thread:
                self._thread.quit()
                self._thread.wait(3000)
        except Exception:
            pass
        self._pause_playback()
        self._release_cuda_resources()
        app = QApplication.instance()
        if app is not None:
            app.quit()
        sys.exit(0)

    def _preview_run(self) -> None:
        """Run a short preview of the current input.

        Video input + video export mode (BUG 1 fix)
        ─────────────────────────────────────────────
        Do NOT capture a single PNG frame.  Instead run the real pipeline on
        the first 81 frames of the source video, outputting a short clip in
        the selected container/codec.  The resulting video is shown in Split
        View so the user can compare original vs upscaled.  Clicking Play
        afterwards resumes the full original video (BUG 2 fix).

        Image input or image-sequence export mode (legacy)
        ───────────────────────────────────────────────────
        Capture the currently displayed frame as a PNG, upscale it as a
        single image, and display the result in Split View.
        """
        # Save the caller's state so _on_finished can restore it.
        self._preview_original_input_path = self._folders_dlg.input_edit.text().strip()
        self._preview_original_input_mode = self._folders_dlg.input_mode_combo.currentText()
        self._preview_original_output_path = self._folders_dlg.output_edit.text().strip()
        self._preview_saved_batch_size = self.batch_size_spin.value()
        if self._input_player is not None:
            self._preview_original_position = self._input_player.position()
        self._preview_compare_active = False

        # ── Always use single-frame PNG capture (never trigger video pipeline) ──
        self._is_preview_video_mode = False

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

        # Save preview input frame directly inside the input file's directory as a
        # 16-bit TIFF so the source and upscaled frames stay aligned at full
        # fidelity for side-by-side comparison.
        export_dir = self._resolve_export_output_dir()
        export_dir.mkdir(parents=True, exist_ok=True)
        preview_input = self._ensure_unique_file_path(
            export_dir / "preview_input_frame_001.tiff"
        )
        self._preview_temp_path = str(preview_input)
        if not self._save_qimage_as_tiff16(frame_img, self._preview_temp_path):
            self._on_log(f"⚠  Preview: could not save input frame to {self._preview_temp_path}")
            return

        self._on_log(f"ℹ  Preview: captured frame → {self._preview_temp_path}")

        # Point input to the temp file, set batch size to 1, mark as preview run.
        # Also set an explicit 16-bit TIFF output path so the CLI never tries to
        # create a video from a single-image input (which caused the cv2.imwrite
        # crash on .mp4 extension).
        self._folders_dlg.input_edit.setText(self._preview_temp_path)
        # Derive preview output name from the original input filename.
        _orig_for_preview = Path(self._preview_original_input_path) if self._preview_original_input_path else None
        if _orig_for_preview and _orig_for_preview.stem:
            _preview_out_stem = f"seedvr2_{_orig_for_preview.stem}"
        else:
            _preview_out_stem = "seedvr2_preview_frame"
        preview_out_tiff = str(
            self._ensure_unique_file_path(export_dir / f"{_preview_out_stem}.tiff")
        )
        self._folders_dlg.output_edit.setText(preview_out_tiff)
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
            if self._preview_compare_active and self._preview_original_input_path:
                self._resume_original_video_after_preview(switch_mode=False)
            self._split_toggle.blockSignals(True)
            self._split_toggle.setChecked(False)
            self._split_toggle.blockSignals(False)
            if self._current_input_is_image:
                # Always re-load input pixmap so the tab shows the original image even
                # if the previous tab had swapped _image_view to display the output.
                inp_path = self._folders_dlg.input_edit.text().strip()
                if inp_path:
                    pix = QPixmap(inp_path)
                    if not pix.isNull():
                        self._image_view.set_pixmap(pix)
                self._viewer_stack.setCurrentIndex(3)
            else:
                self._input_player.setVideoOutput(self._solo_input_vw)
                self._viewer_stack.setCurrentIndex(0)
        elif new_mode == "output":
            self._split_toggle.blockSignals(True)
            self._split_toggle.setChecked(False)
            self._split_toggle.blockSignals(False)
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
            self._split_toggle.blockSignals(True)
            self._split_toggle.setChecked(True)
            self._split_toggle.blockSignals(False)
            if self._preview_compare_active:
                # Output player always shows the preview clip on the right side.
                if self._output_player is not None:
                    self._output_player.setVideoOutput(self._split_view.output_sink)
                # BUG 2 fix: for video-mode preview also route the input player to the
                # split view so the left side shows the original video for comparison.
                if self._is_preview_video_mode and self._input_player is not None:
                    self._input_player.setVideoOutput(self._split_view.input_sink)
            elif self._current_input_is_image:
                # Image input: feed directly into SplitViewWidget; no video sink needed.
                inp_path = self._folders_dlg.input_edit.text().strip()
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
        if self._player_mode in {"output", "split"}:
            return self._output_player if self._output_player and self._output_player.source().isValid() else self._input_player
        return self._input_player

    def _on_play_pause(self) -> None:
        if self._preview_compare_active and self._resume_original_video_after_preview():
            if self._input_player:
                self._input_player.play()
            return
        p = self._active_player()
        if not p:
            return
        if p.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._input_player.pause()
            self._output_player.pause()
        else:
            self._output_player.play()
            self._input_player.play()

    def _pause_playback(self) -> None:
        if self._input_player:
            self._input_player.pause()
        if self._output_player:
            self._output_player.pause()

    def _on_stop(self) -> None:
        if self._input_player:
            self._input_player.stop()
        if self._output_player:
            self._output_player.stop()

    def _step_frame(self, direction: int) -> None:
        p = self._active_player()
        if not p:
            return
        self._pause_playback()
        frame_ms = int(1000 / 24)
        try:
            meta = p.metaData()
            fps = meta.value(QMediaMetaData.Key.VideoFrameRate)
            if fps:
                frame_ms = max(1, int(1000 / float(fps)))
        except Exception:
            pass
        new_pos = max(0, p.position() + (direction * frame_ms))
        self._on_seek(new_pos)

    def _on_split_toggle_changed(self, checked: bool) -> None:
        if checked:
            self._mode_split_btn.setChecked(True)
            self._on_mode_button(2, True)
        else:
            self._mode_input_btn.setChecked(True)
            self._on_mode_button(0, True)

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
        self._update_input_frame_counter_label(position)
        self._update_timecode_label(position)

    def _on_player_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("\u23f8" if playing else "\u25b6")  # ⏸ / ▶

    def _sync_input_to_output(self, pos: int) -> None:
        """Slave the input player's position strictly to the output player (master)."""
        if self._input_player and abs(self._input_player.position() - pos) > 200:
            self._input_player.setPosition(pos)

    @staticmethod
    def _ms_to_hhmmss(ms: int) -> str:
        """Format *ms* milliseconds as HH:MM:SS."""
        s = ms // 1000
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def _ms_to_frame(self, ms: int) -> int:
        """Convert *ms* milliseconds to a 0-based frame index using the cached FPS."""
        if self._current_fps > 0:
            return max(0, int(ms / 1000.0 * self._current_fps))
        return 0

    def _update_timecode_label(self, position_ms: Optional[int] = None) -> None:
        """Refresh the HH:MM:SS | F:N timecode label next to the seek slider."""
        if not hasattr(self, "_timecode_lbl"):
            return
        p = self._active_player()
        if p is None:
            return
        pos = position_ms if position_ms is not None else p.position()
        frame = self._ms_to_frame(pos)
        self._timecode_lbl.setText(f"{self._ms_to_hhmmss(pos)} | F:{frame}")

    def _update_trim_slider(self) -> None:
        """Refresh the In/Out fraction overlays on the seek slider."""
        p = self._active_player()
        dur = p.duration() if p else 0
        if dur <= 0:
            self._seek_slider.set_trim_fractions(None, None)
            return
        in_frac = (self._in_point_ms / dur) if self._in_point_ms is not None else None
        out_frac = (self._out_point_ms / dur) if self._out_point_ms is not None else None
        self._seek_slider.set_trim_fractions(in_frac, out_frac)

    def _set_in_point(self) -> None:
        """Set the In-Point to the current playback position and update the spinbox."""
        p = self._active_player()
        if not p:
            return
        pos_ms = p.position()
        self._in_point_ms = pos_ms
        in_frame = self._ms_to_frame(pos_ms)
        self.skip_first_frames_spin.setValue(in_frame)
        # Recompute load_cap if out-point is already set
        if self._out_point_ms is not None and self._out_point_ms >= pos_ms:
            out_frame = self._ms_to_frame(self._out_point_ms)
            self.load_cap_spin.setValue(max(1, out_frame - in_frame + 1))
        self._update_trim_slider()
        self._on_log(
            f"[  In-Point set → frame {in_frame} ({self._ms_to_hhmmss(pos_ms)}) "
            f"→ Skip First Frames: {in_frame}"
        )

    def _set_out_point(self) -> None:
        """Set the Out-Point to the current playback position and update the spinbox."""
        p = self._active_player()
        if not p:
            return
        pos_ms = p.position()
        self._out_point_ms = pos_ms
        out_frame = self._ms_to_frame(pos_ms)
        in_frame = self._ms_to_frame(self._in_point_ms) if self._in_point_ms is not None else 0
        total = max(1, out_frame - in_frame + 1)
        self.load_cap_spin.setValue(total)
        self._update_trim_slider()
        self._on_log(
            f"]  Out-Point set → frame {out_frame} ({self._ms_to_hhmmss(pos_ms)}) "
            f"→ Load Cap: {total} frames"
        )

    def _clear_trim_range(self) -> None:
        """Clear In/Out markers, remove the blue timeline highlight, and reset spinboxes."""
        self._in_point_ms = None
        self._out_point_ms = None
        self._seek_slider.set_trim_fractions(None, None)
        self.skip_first_frames_spin.setValue(0)
        self.load_cap_spin.setValue(0)
        self._on_log("✕  Trim range cleared → Skip First Frames and Load Cap reset to 0")

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

    def _try_extract_first_frame(self, video_path: str) -> None:
        """Extract the first frame from *video_path* via ffmpeg and display it immediately.

        Falls back silently if ffmpeg is unavailable or extraction fails.
        The static frame is also loaded into the SplitViewWidget input side so
        Split View mode has an immediate preview without waiting for the player.
        """
        import tempfile
        import os as _os
        try:
            # Force the temporary working directory to the absolute parent
            # directory of the input file, overriding the default system temp
            # path. Fall back to the system temp dir only if that location is
            # unavailable/unwritable.
            try:
                temp_dir = str(Path(video_path).resolve().parent)
                if not (temp_dir and _os.path.isdir(temp_dir) and _os.access(temp_dir, _os.W_OK)):
                    temp_dir = None
            except Exception:
                temp_dir = None
            fd, tmp_path = tempfile.mkstemp(
                suffix="_frame0.png", prefix="seedvr2_preview_", dir=temp_dir
            )
            _os.close(fd)  # close fd so ffmpeg can write to the path
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vframes", "1",
                    "-an",
                    tmp_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                creationflags=_CREATE_NO_WINDOW,
            )
            if result.returncode == 0 and Path(tmp_path).is_file():
                pix = QPixmap(tmp_path)
                if not pix.isNull():
                    self._image_view.set_pixmap(pix)
                    if _MULTIMEDIA_AVAILABLE and self._split_view is not None:
                        self._split_view.set_input_image(pix.toImage())
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
        except Exception:
            pass  # ffmpeg unavailable or extraction failed – normal video loading continues

    def _load_input_video(self, path: str) -> None:
        if self._input_player:
            self._input_player.setSource(QUrl.fromLocalFile(path))
            # Force the decoder to render the first frame immediately so the
            # viewer shows a static preview instead of a black rectangle.
            self._input_player.play()
            QTimer.singleShot(150, self._input_player.pause)

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
        start_dir = self._folders_dlg.output_edit.text().strip() or ""
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
        start_dir = self._folders_dlg.input_edit.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", start_dir,
            INPUT_DIALOG_FILTER,
        )
        if path:
            if not self._apply_dropped_path(Path(path)):
                self._on_log(f"⚠  Unsupported input format: {path}")

    def _try_auto_load_output(self) -> None:
        out = self._folders_dlg.output_edit.text().strip()
        if not out:
            return
        out_path = Path(out)

        _image_exts = set(SUPPORTED_IMAGE_EXTS) | {".bmp", ".webp"}
        _video_exts = set(SUPPORTED_VIDEO_EXTS)

        # ── Image input: construct output path directly (mirrors inference_cli.generate_output_path) ──
        if self._current_input_is_image and self._split_view is not None:
            inp = self._folders_dlg.input_edit.text().strip()
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
                # Explicit file path already points to image output.
                if out_path.is_file() and out_path.suffix.lower() in _image_exts:
                    return out_path

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

            # Preview clips can still render as video output (e.g. load_cap previews).
            search_dir = out_path if out_path.is_dir() else inp_path.parent
            video_candidates = sorted(
                (f for f in search_dir.iterdir() if f.suffix.lower() in _video_exts),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            preferred_videos = [
                f for f in video_candidates
                if f.stem.startswith(inp_stem) or f.stem.startswith(self._seedvr_prefixed_stem(inp_stem))
            ]
            selected_video = preferred_videos[0] if preferred_videos else (video_candidates[0] if video_candidates else None)
            if selected_video is not None:
                self._load_output_video(str(selected_video))
                self._set_latest_output_path(selected_video)
                self._preview_compare_active = True
                self._mode_split_btn.setChecked(True)
                self._on_mode_button(2, True)
                if self._output_player:
                    self._output_player.pause()
                    self._output_player.setPosition(0)
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
            "QPushButton { background-color: #222222; border: 1px solid #444; border-radius: 6px; }"
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
        self.batch_size_spin.setValue(81)
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
        self._batch_stepper_widget = wrapper
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
        if line.startswith("__GUI_FATAL__|"):
            self._show_execution_error_modal("Execution Crash", line.split("|", 1)[1])
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{ts}] {line}")

    def _reset_progress_bars(self) -> None:
        self._active_file_status = ""
        self._progress_status = ""
        self._batch_status = ""
        self._current_file_path = ""
        self._current_file_total_frames = 0
        self._current_file_processed_frames = 0
        self._queue_files_total = 0
        self._queue_files_completed = 0
        self._queue_file_frame_counts = {}
        self._queue_ordered_files = []
        self._active_queue_index = -1
        self.status_label.setToolTip("")
        self.status_label.setText("Ready")
        self.current_file_progress_label.setText("Current File: -")
        self.batch_progress_label.setText("Overall Batch Progress | Completed: 0/0")
        self._last_batch_cur = 0
        self._last_batch_tot = 0
        self.batch_progress_circle.set_progress(0.0)
        self.batch_progress_circle.set_text("0/0")
        self.phase_progress_circle.set_progress(0.0)
        self.phase_progress_circle.set_text("1/4")
        self.eta_progress_circle.set_progress(0.0)
        self.eta_progress_circle.set_text("00:00:00")
        self.queue_progress_circle.set_progress(0.0)
        self.queue_progress_circle.set_text("0/0")
        self.elapsed_progress_circle.set_progress(0.0)
        self.elapsed_progress_circle.set_text("00:00:00")
        self._elapsed_timer.stop()

    def _format_seconds(self, seconds: float) -> str:
        total = max(0, int(seconds))
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_mmss(self, seconds: float) -> str:
        """Format elapsed seconds as MM:SS for the Video Processing Time label."""
        total = max(0, int(seconds))
        minutes = total // 60
        secs = total % 60
        return f"{minutes:02d}:{secs:02d}"

    def _estimated_processing_fps(self) -> float:
        return 1.8

    def _count_frames_for_file(self, path: Path) -> int:
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTS:
            return 1
        if suffix not in SUPPORTED_VIDEO_EXTS or cv2 is None:
            return 0
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return 0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return max(0, frames)
        except Exception:
            return 0
        finally:
            cap.release()

    def _collect_input_files_for_estimation(self) -> list[Path]:
        inp = self._folders_dlg.input_edit.text().strip()
        if not inp:
            return []
        input_path = Path(inp)
        if self._folders_dlg.input_mode_combo.currentText() == "Folder" and input_path.is_dir():
            try:
                files = [
                    p for p in sorted(input_path.iterdir(), key=lambda x: x.name.lower())
                    if p.is_file() and p.suffix.lower() in (SUPPORTED_VIDEO_EXTS | SUPPORTED_IMAGE_EXTS)
                ]
                return files
            except OSError:
                return []
        if input_path.is_file():
            return [input_path]
        return []

    def _prepare_queue_progress_context(self) -> None:
        input_files = self._collect_input_files_for_estimation()
        self._queue_ordered_files = [str(p.resolve()) for p in input_files]
        self._queue_file_frame_counts = {
            str(p.resolve()): self._count_frames_for_file(p) for p in input_files
        }
        self._queue_files_total = len(self._queue_ordered_files)
        self._queue_files_completed = 0
        self._active_queue_index = 0 if self._queue_files_total > 0 else -1
        if self._queue_ordered_files:
            self._current_file_path = self._queue_ordered_files[0]
            self._current_file_total_frames = max(
                0, self._queue_file_frame_counts.get(self._current_file_path, 0)
            )
        else:
            self._current_file_path = ""
            self._current_file_total_frames = 0
        self._current_file_processed_frames = 0

    def _update_current_file_progress_ui(self) -> None:
        total_frames = max(0, self._current_file_total_frames)
        current_frames = max(0, min(self._current_file_processed_frames, total_frames)) if total_frames > 0 else max(0, self._current_file_processed_frames)
        file_name = Path(self._current_file_path).name if self._current_file_path else "-"
        self.current_file_progress_label.setText(
            f"Current File: {file_name} | Batches: {current_frames}/{total_frames}"
        )
        ratio = (current_frames / total_frames) if total_frames > 0 else 0.0
        self.eta_progress_circle.set_progress(max(0.0, min(1.0, ratio)))
        phase = 1 if total_frames <= 0 else min(4, max(1, int(ratio * 4) + 1))
        self.phase_progress_circle.set_progress(phase / 4.0)
        self.phase_progress_circle.set_text(f"{phase}/4")

    def _update_batch_progress_ui(self) -> None:
        if self._queue_files_total <= 0:
            self.batch_progress_label.setText(
                "Overall Batch Progress | Completed: 0/0"
            )
            self.batch_progress_circle.set_progress(0.0)
            self.batch_progress_circle.set_text("0/0")
            self.queue_progress_circle.set_progress(0.0)
            self.queue_progress_circle.set_text("0/0")
            return

        current_index = max(0, self._active_queue_index)

        completed_files = min(self._queue_files_total, max(self._queue_files_completed, current_index))
        self.batch_progress_label.setText(
            f"Overall Batch Progress | Completed: {completed_files}/{self._queue_files_total}"
        )

        processed_current = max(0, min(self._current_file_processed_frames, self._current_file_total_frames))
        processed_frames = processed_current
        for idx in range(min(current_index, self._queue_files_total)):
            processed_frames += max(0, self._queue_file_frame_counts.get(self._queue_ordered_files[idx], 0))
        total_frames = sum(max(0, v) for v in self._queue_file_frame_counts.values())
        ratio = (processed_frames / total_frames) if total_frames > 0 else (completed_files / self._queue_files_total)
        pct = max(0.0, min(1.0, ratio))
        self.batch_progress_circle.set_progress(pct)
        self.batch_progress_circle.set_text(f"{self._last_batch_cur}/{self._last_batch_tot}" if self._last_batch_tot > 0 else f"{completed_files}/{self._queue_files_total}")
        queue_ratio = completed_files / max(1, self._queue_files_total)
        self.queue_progress_circle.set_progress(max(0.0, min(1.0, queue_ratio)))
        self.queue_progress_circle.set_text(f"{completed_files}/{self._queue_files_total}")

    def _update_elapsed_progress_ui(self) -> None:
        if self._run_started_at is None:
            self.elapsed_progress_circle.set_progress(0.0)
            self.elapsed_progress_circle.set_text("00:00:00")
            # Show frozen elapsed time if available; do not reset to 00:00
            if self._frozen_elapsed_seconds is not None:
                self.video_proc_time_label.setText(
                    f"Video Processing Time: {self._format_mmss(self._frozen_elapsed_seconds)}"
                )
            return
        elapsed = max(0.0, time.time() - self._run_started_at)
        self._frozen_elapsed_seconds = elapsed
        self.elapsed_progress_circle.set_text(self._format_seconds(elapsed))
        self.video_proc_time_label.setText(f"Video Processing Time: {self._format_mmss(elapsed)}")
        total_frames = sum(max(0, v) for v in self._queue_file_frame_counts.values())
        if total_frames <= 0:
            self.elapsed_progress_circle.set_progress(0.0)
            return
        processed_current = max(0, min(self._current_file_processed_frames, self._current_file_total_frames))
        processed_frames = processed_current
        current_index = max(0, self._active_queue_index)
        for idx in range(min(current_index, self._queue_files_total)):
            processed_frames += max(0, self._queue_file_frame_counts.get(self._queue_ordered_files[idx], 0))
        self.elapsed_progress_circle.set_progress(max(0.0, min(1.0, processed_frames / total_frames)))

    def _on_global_progress(self, cur: int, tot: int) -> None:
        if tot <= 0:
            return
        self._current_file_processed_frames = max(0, cur)
        self._current_file_total_frames = max(0, tot)
        self._update_current_file_progress_ui()
        self._update_batch_progress_ui()
        self.status_label.setText("Processing…")

    def _on_batch_progress(self, cur: int, tot: int) -> None:
        if tot <= 0:
            return
        self._last_batch_cur = max(0, cur)
        self._last_batch_tot = max(0, tot)
        self._batch_status = f"Batch {cur}/{tot}"
        self.batch_progress_circle.set_text(self._batch_status.replace("Batch ", ""))
        self.status_label.setToolTip(self._batch_status)

    def _on_queue_status_update(
        self, file_path: str, current: int, total: int, done: int, remaining: int
    ) -> None:
        if total > 0 and file_path:
            resolved = str(Path(file_path).resolve())
            self._active_file_status = f"Processing: {resolved}"
            self._queue_files_total = total
            self._queue_files_completed = done
            self._current_file_path = resolved
            self._active_queue_index = max(0, current - 1)
            if resolved in self._queue_file_frame_counts:
                self._current_file_total_frames = max(0, self._queue_file_frame_counts[resolved])
            elif self._current_file_total_frames <= 0:
                self._current_file_total_frames = 0
        else:
            self._active_file_status = ""
        self._progress_status = ""
        self._update_current_file_progress_ui()
        self._update_batch_progress_ui()
        self.status_label.setText("Processing…")
        self.status_label.setToolTip(self._active_file_status if self._active_file_status else "")

    def _set_latest_output_path(self, path: Optional[Path]) -> None:
        self._latest_output_path = path
        # Button stays enabled at all times so the user can always open the output folder.

    def _open_output_folder(self) -> None:
        # Prefer the most-recently produced output path; fall back to the configured dir.
        if self._latest_output_path is not None:
            folder = (
                self._latest_output_path
                if self._latest_output_path.is_dir()
                else self._latest_output_path.parent
            )
        else:
            # No output produced yet — open the configured output directory.
            folder = self._resolve_export_output_dir()

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

    def _show_execution_error_modal(self, title: str, text: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText("A processing error occurred. The application is still running.")
        box.setDetailedText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        copy_btn = box.addButton("Copy to Clipboard", QMessageBox.ButtonRole.ActionRole)
        box.exec()
        if box.clickedButton() == copy_btn:
            QApplication.clipboard().setText(text)

    def _on_finished(self, success: bool, msg: str) -> None:
        was_preview = self._is_preview_run
        self._set_running(False)
        # Freeze the elapsed processing time before nulling run_started_at
        if self._run_started_at is not None:
            self._frozen_elapsed_seconds = max(0.0, time.time() - self._run_started_at)
            self.video_proc_time_label.setText(
                f"Video Processing Time: {self._format_mmss(self._frozen_elapsed_seconds)}"
            )
        self._elapsed_timer.stop()
        self._run_started_at = None
        self._active_file_status = ""
        self._progress_status = ""
        self._batch_status = ""
        if success:
            self.status_label.setText(f"✅  {msg}")
            self.status_label.setToolTip(msg)
            # For non-preview runs use the heuristic auto-loader; preview runs
            # load the two generated TIFF files explicitly below.
            if not was_preview:
                self._try_auto_load_output()
            # Feed completed output path into the split tracking context so that
            # multi-part chunk compilation can resolve input→output mappings.
            if not was_preview and self._latest_output_path is not None:
                inp_raw = self._folders_dlg.input_edit.text().strip()
                if inp_raw:
                    self._on_log(
                        f"📂  Output: {self._latest_output_path.resolve()}"
                    )
            # Play success notification if enabled
            if (
                _WINSOUND_AVAILABLE
                and _winsound is not None
                and hasattr(self, "enable_audio_notifications_check")
                and self.enable_audio_notifications_check.isChecked()
            ):
                try:
                    _winsound.MessageBeep(_winsound.MB_ICONASTERISK)
                except Exception:
                    pass
            # As soon as the preview upscale finishes, automatically trigger and
            # open the split-screen comparison view (SplitViewWidget) using the
            # two generated TIFF files: the captured preview input frame and the
            # upscaled output – both living directly in the input file's dir.
            if was_preview and _MULTIMEDIA_AVAILABLE and self._split_view is not None:
                # Bypass the UI text fields entirely: load the raw captured input
                # TIFF and the newly generated upscale output TIFF directly from
                # their tracked file paths on disk.
                preview_in = getattr(self, "_preview_temp_path", None)
                preview_out = getattr(self, "_preview_output_path", None)
                loaded_in = False
                loaded_out = False
                # Left side: captured original frame (image-mode preview only).
                if preview_in and not self._is_preview_video_mode and Path(preview_in).is_file():
                    try:
                        pix_in = QPixmap(preview_in)
                        if not pix_in.isNull():
                            self._split_view.set_input_image(pix_in.toImage())
                            loaded_in = True
                    except Exception:
                        pass
                # Right side: upscaled TIFF written by the CLI.
                if preview_out and Path(preview_out).is_file():
                    try:
                        pix_out = QPixmap(preview_out)
                        if not pix_out.isNull():
                            self._split_view.set_output_image(pix_out.toImage())
                            self._set_latest_output_path(Path(preview_out))
                            loaded_out = True
                    except Exception:
                        pass
                # Fallback to the heuristic loader (e.g. video-mode previews or
                # when a tracked file could not be loaded from disk).
                if not loaded_out or (not self._is_preview_video_mode and not loaded_in):
                    self._try_auto_load_output()
                self._preview_compare_active = self._latest_output_path is not None
                if self._preview_original_input_path:
                    self._folders_dlg.input_mode_combo.setCurrentText(self._preview_original_input_mode)
                    self._folders_dlg.input_edit.setText(self._preview_original_input_path)
                # Programmatically switch the UI to the split-screen view layout.
                self._mode_split_btn.setChecked(True)
                self._on_mode_button(2, True)
        else:
            self.status_label.setText(f"⚠  {msg}")
            self.status_label.setToolTip(msg)
            # Play error notification if enabled
            if (
                _WINSOUND_AVAILABLE
                and _winsound is not None
                and hasattr(self, "enable_audio_notifications_check")
                and self.enable_audio_notifications_check.isChecked()
            ):
                try:
                    _winsound.MessageBeep(_winsound.MB_ICONHAND)
                except Exception:
                    pass
            console_text = self.console.toPlainText()
            tail = console_text[-12000:] if len(console_text) > 12000 else console_text
            if "out of memory" in tail.lower() or "cuda out of memory" in tail.lower():
                self._show_execution_error_modal("Out of Memory", tail or msg)
            else:
                self._show_execution_error_modal("Execution Error", tail or msg)
        self._is_preview_run = False
        # Reset video-mode preview flag AFTER _on_mode_button has had a chance to read it.
        self._is_preview_video_mode = False
        if was_preview and self.batch_size_spin.value() == 1 and self._preview_saved_batch_size:
            self.batch_size_spin.setValue(self._preview_saved_batch_size)
        # Restore load_cap that was temporarily set to 81 during video-mode preview.
        if was_preview and self._preview_saved_load_cap is not None:
            self.load_cap_spin.setValue(self._preview_saved_load_cap)
            self._preview_saved_load_cap = None
        # Restore output path that was overridden during preview (but keep the input field
        # pointing at the original path so the user still sees the correct input displayed).
        if was_preview and hasattr(self, "_preview_original_output_path"):
            self._folders_dlg.output_edit.setText(self._preview_original_output_path)
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
        self._preview_compare_active = False
        if path.is_dir():
            frames = self._sequence_frame_candidates(path)
            self._folders_dlg.input_mode_combo.setCurrentText("Folder")
            self._folders_dlg.input_edit.setText(str(path))
            if frames:
                self._load_preview(str(frames[0]))
                self._mode_input_btn.setChecked(True)
            self._on_log(f"📂  Loaded image-sequence folder ({len(frames)} frames): {path}")
            return True
        self._folders_dlg.input_mode_combo.setCurrentText("File")
        self._folders_dlg.input_edit.setText(str(path))
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
        if self._force_exit:
            try:
                if self._worker:
                    self._worker.force_kill()
            except Exception:
                pass
            self._release_cuda_resources()
            event.accept()
            return
        event.ignore()
        self._force_shutdown()


# ---------------------------------------------------------------------------
# Helpers (module-level, used inside this file only)
# ---------------------------------------------------------------------------

def _wrap(layout: QHBoxLayout) -> QWidget:
    """Wrap a QHBoxLayout in a plain QWidget so it can be added to a QFormLayout."""
    w = QWidget()
    w.setLayout(layout)
    return w
