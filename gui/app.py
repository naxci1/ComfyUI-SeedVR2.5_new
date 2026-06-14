#!/usr/bin/env python3
"""PySide6 application entry-point for 1-Click SeedVR2.5 v.1.8b (by Naxci1)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import traceback
import webbrowser
import json
import tempfile
from pathlib import Path
from typing import Optional

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_APP_DIR)
for _p in (_APP_ROOT, _APP_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QImage, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


def _custom_exception_hook(exctype, value, tb):
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    try:
        with open("crash_log.txt", "a", encoding="utf-8") as f:
            f.write(error_msg + "\n")
    except Exception:
        pass
    # Show OOM-specific advice.
    oom_hint = ""
    if "out of memory" in error_msg.lower() or "cuda oom" in error_msg.lower():
        oom_hint = (
            "\n\n⚠ Out of Memory suggestions:\n"
            "• Reduce batch size\n• Enable VAE tiling\n• Enable DiT/VAE offload\n"
            "• Lower resolution\n• Use a smaller model"
        )
    app = QApplication.instance()
    if app:
        from gui.widgets.error_dialog import ErrorDialog
        dlg = ErrorDialog(None, "Unhandled Exception", error_msg + oom_hint)
        dlg.exec()


sys.excepthook = _custom_exception_hook

try:
    from gui.config_manager import load_config, save_config
    from gui.export_encoder import build_ffmpeg_command
    from gui.theme import Colors, Dims, Fonts, generate_stylesheet
    from gui.widgets import (
        AnimatedProgressBar,
        Button3D,
        DropZone,
        ErrorDialog,
        ExportDialog,
        LogViewer,
        PlaybackControls,
        ProjectPanel,
        SettingsDialog,
        SettingsPanel,
        SplitViewWidget,
        Toast,
        TrimTimeline,
        VideoPreviewWidget,
    )
    from gui.workers import create_worker_thread, resolve_paths
except ImportError:  # pragma: no cover - direct-script execution fallback
    from config_manager import load_config, save_config  # type: ignore[no-redef]
    from export_encoder import build_ffmpeg_command  # type: ignore
    from theme import Colors, Dims, Fonts, generate_stylesheet  # type: ignore
    from widgets import (  # type: ignore
        AnimatedProgressBar,
        Button3D,
        DropZone,
        ErrorDialog,
        ExportDialog,
        LogViewer,
        PlaybackControls,
        ProjectPanel,
        SettingsDialog,
        SettingsPanel,
        Toast,
        TrimTimeline,
        VideoPreviewWidget,
    )
    from workers import create_worker_thread, resolve_paths  # type: ignore
    try:
        from widgets import SplitViewWidget  # type: ignore
    except ImportError:
        SplitViewWidget = None  # type: ignore

APP_NAME = "1-Click SeedVR2.5 v.1.8b (by Naxci1)"
GITHUB_URL = "https://github.com/naxci1/1Click_SeedVR2.5"


def _seedvr_temp_dir() -> str:
    if os.name == "nt":
        return os.path.join("C:\\1Click_SeedVR2.5", "temp")
    return os.path.join(tempfile.gettempdir(), "1Click_SeedVR2.5", "temp")


TEMP_DIR = _seedvr_temp_dir()
os.makedirs(TEMP_DIR, exist_ok=True)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore
    _HAS_NVML = False


class _HardwareProbe(QObject):
    detected = Signal(str, str)

    def run(self) -> None:
        name, vram = "CPU", "—"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                detected_name = torch.cuda.get_device_name(idx)
                lowered = detected_name.lower()
                if any(token in lowered for token in ("nvidia", "rtx", "geforce", "quadro", "rtx pro")):
                    name = detected_name
                else:
                    name = f"NVIDIA {detected_name}"
                total = torch.cuda.get_device_properties(idx).total_memory
                vram = f"{total / (1024 ** 3):.1f} GB"
        except Exception:
            pass
        self.detected.emit(name, vram)


class _CodecProbe(QObject):
    detected = Signal(dict)

    def __init__(self, ffmpeg_path: str = "") -> None:
        super().__init__()
        self._ffmpeg_path = ffmpeg_path

    def run(self) -> None:
        result = {"nvenc": False, "qsv": False, "amf": False}
        ffmpeg = self._ffmpeg_path if self._ffmpeg_path and Path(self._ffmpeg_path).exists() else shutil.which("ffmpeg")
        if ffmpeg:
            try:
                out = subprocess.run(
                    [ffmpeg, "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    check=False,
                    creationflags=0x08000000 if sys.platform == "win32" else 0,
                ).stdout.lower()
                result["nvenc"] = "nvenc" in out
                result["qsv"] = "_qsv" in out
                result["amf"] = "_amf" in out
            except Exception:
                pass
        self.detected.emit(result)


class MainWindow(QMainWindow):
    """Main application window."""

    export_requested = Signal(dict)
    preview_requested = Signal(dict)
    cancel_requested = Signal()

    _PRESET_TO_RESOLUTION = {
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
        "1440p": 1440,
        "2K": 2048,
        "4K": 2160,
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
        self.setAcceptDrops(True)

        self._config = load_config()
        self._current_file = ""
        self._processing = False
        self._thread: Optional[QThread] = None
        self._worker = None
        self._active_mode = ""
        self._active_output_path = ""
        self._snapshot_fallback_path = ""
        self._preview_source_frame_path = ""
        self._last_processed_preview_path = ""
        self._codec_cache = {"nvenc": False, "qsv": False, "amf": False}
        self._gpu_name = "CPU"
        self._gpu_vram = "—"
        self._log_viewer: Optional[LogViewer] = None
        self._preview_mode = "single"
        self._current_phase_index = 0
        self._fs_win = None  # fullscreen overlay window reference
        self._chunk_current = 0
        self._chunk_total = 0
        self._batch_current = 0
        self._batch_total = 0
        self._phase_name = "idle"
        self._device_timer = QTimer(self)
        self._device_timer.setInterval(1500)
        self._device_timer.timeout.connect(self._update_device_info)

        self._build_ui()
        self._connect_signals()
        self.input_path_edit.setText(self._config.get("input_path", ""))
        self.output_path_edit.setText(self._config.get("output_path", ""))
        self._start_hardware_probe()
        self._start_codec_probe()
        self._update_status_summary()
        self._device_timer.start()
        self._update_device_info()

        self.export_requested.connect(self._spawn_worker)
        self.preview_requested.connect(self._spawn_preview)
        self.cancel_requested.connect(self._abort_worker)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.project_panel = ProjectPanel(self)
        root.addWidget(self.project_panel)

        center = QWidget(self)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_MD)
        center_layout.setSpacing(Dims.PADDING_MD)
        root.addWidget(center, 1)

        center_layout.addWidget(self._build_header())
        self.io_bar = self._build_io_bar()
        self.io_bar.setVisible(False)  # Hidden — paths tracked internally
        center_layout.addWidget(self.io_bar)
        self.view_mode_bar = self._build_view_mode_bar()
        center_layout.addWidget(self.view_mode_bar)

        self.center_stack = QStackedWidget(self)
        self.drop_zone = DropZone(self)
        self.preview_widget = VideoPreviewWidget(self)
        self.split_view_widget = SplitViewWidget(self)
        self.center_stack.addWidget(self.drop_zone)       # index 0
        self.center_stack.addWidget(self.preview_widget)  # index 1
        self.center_stack.addWidget(self.split_view_widget)  # index 2
        center_layout.addWidget(self.center_stack, 1)

        # Video info bar below preview (fix #14).
        self.video_info_bar = self._build_video_info_bar()
        center_layout.addWidget(self.video_info_bar)

        self.trim_timeline = TrimTimeline(self)
        center_layout.addWidget(self.trim_timeline)

        center_layout.addWidget(self._build_trim_bar())

        self.playback_controls = PlaybackControls(self)
        center_layout.addWidget(self.playback_controls)

        center_layout.addWidget(self._build_action_bar())

        self.settings_panel = SettingsPanel(self)
        root.addWidget(self.settings_panel)

        self._build_status_bar()
        self._build_shortcuts()

    def _build_header(self) -> QWidget:
        header = QWidget(self)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_SM)

        title = QLabel(APP_NAME, header)
        title.setProperty("role", "h1")
        layout.addWidget(title)

        layout.addStretch(1)
        self.show_log_btn = Button3D("Show Log", variant="ghost", parent=header)
        self.settings_btn = Button3D("Settings", variant="ghost", parent=header)
        self.about_btn = Button3D("About", variant="ghost", parent=header)
        self.github_btn = Button3D("GitHub", variant="ghost", parent=header)
        self.update_btn = Button3D("Update", variant="ghost", parent=header)
        for button in (self.show_log_btn, self.settings_btn, self.about_btn, self.github_btn, self.update_btn):
            layout.addWidget(button)
        return header

    def _build_view_mode_bar(self) -> QWidget:
        """Three view mode buttons: Single View / Split View / Side by Side (fix #9)."""
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_SM)

        self.view_single_btn = Button3D("Single View", variant="primary", parent=widget)
        self.view_split_btn = Button3D("Split View", variant="default", parent=widget)
        self.view_sidebyside_btn = Button3D("Side by Side", variant="default", parent=widget)
        self.view_fullscreen_btn = Button3D("⛶ Full Screen", variant="ghost", parent=widget)
        self.view_fullscreen_btn.setToolTip("Show split view fullscreen")

        self.view_single_btn.clicked.connect(lambda: self._set_preview_mode("single"))
        self.view_split_btn.clicked.connect(lambda: self._set_preview_mode("split"))
        self.view_sidebyside_btn.clicked.connect(lambda: self._set_preview_mode("sidebyside"))
        self.view_fullscreen_btn.clicked.connect(self._toggle_split_fullscreen)

        layout.addWidget(self.view_single_btn)
        layout.addWidget(self.view_split_btn)
        layout.addWidget(self.view_sidebyside_btn)
        layout.addWidget(self.view_fullscreen_btn)
        layout.addStretch(1)
        return widget

    def _build_video_info_bar(self) -> QWidget:
        """Detailed video info below preview (fix #14)."""
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_MD)

        style = f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}px;"
        self.info_resolution_lbl = QLabel("—", widget)
        self.info_fps_lbl = QLabel("—", widget)
        self.info_duration_lbl = QLabel("—", widget)
        self.info_frames_lbl = QLabel("—", widget)
        self.info_filesize_lbl = QLabel("—", widget)
        for lbl in (
            self.info_resolution_lbl, self.info_fps_lbl,
            self.info_duration_lbl, self.info_frames_lbl, self.info_filesize_lbl,
        ):
            lbl.setStyleSheet(style)
            layout.addWidget(lbl)
        layout.addStretch(1)
        return widget

    def _update_video_info(self) -> None:
        """Populate the video info bar from the loaded file."""
        path = self._current_file
        w = self.preview_widget.get_frame_width()
        h = self.preview_widget.get_frame_height()
        fps = self.preview_widget.get_fps()
        fc = self.preview_widget.get_frame_count()
        dur = (fc / fps) if fps > 0 else 0.0
        hours = int(dur // 3600)
        mins = int((dur % 3600) // 60)
        secs = int(dur % 60)
        dur_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
        file_size = ""
        if path and os.path.isfile(path):
            sz = os.path.getsize(path)
            if sz >= 1 << 30:
                file_size = f"{sz / (1 << 30):.2f} GB"
            elif sz >= 1 << 20:
                file_size = f"{sz / (1 << 20):.1f} MB"
            else:
                file_size = f"{sz / 1024:.0f} KB"
        self.info_resolution_lbl.setText(f"{w}×{h}" if w else "—")
        self.info_fps_lbl.setText(f"{fps:.3f} fps" if fps else "—")
        self.info_duration_lbl.setText(dur_str if fc else "—")
        self.info_frames_lbl.setText(f"{fc} frames" if fc else "—")
        self.info_filesize_lbl.setText(file_size or "—")

    def _set_preview_mode(self, mode: str) -> None:
        """Switch between Single / Split / Side-by-Side view modes."""
        self._preview_mode = mode
        # Update button states.
        self.view_single_btn.setProperty("variant", "primary" if mode == "single" else "default")
        self.view_split_btn.setProperty("variant", "primary" if mode == "split" else "default")
        self.view_sidebyside_btn.setProperty("variant", "primary" if mode == "sidebyside" else "default")
        for btn in (self.view_single_btn, self.view_split_btn, self.view_sidebyside_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if mode == "single":
            if self._current_file:
                # Reload the original input so Single View always shows the source.
                self.preview_widget.load_file(self._current_file)
                self.center_stack.setCurrentWidget(self.preview_widget)
            else:
                self.center_stack.setCurrentWidget(self.drop_zone)
        elif mode in ("split", "sidebyside"):
            self._refresh_comparison_view()

    def _toggle_split_fullscreen(self) -> None:
        """Show the split view widget in a fullscreen window with an Exit button."""
        if self._fs_win is not None:
            self._fs_win.close()
            return
        if not self.split_view_widget.has_images():
            Toast.show(self, "No split view to show fullscreen", "warning")
            return

        fs_win = QMainWindow(self)
        self._fs_win = fs_win
        fs_win.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        fs_win.setStyleSheet(f"background: {Colors.PREVIEW_BG};")
        container = QWidget(fs_win)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        fs_split = SplitViewWidget(container)
        if self.split_view_widget._original and self.split_view_widget._processed:
            fs_split.set_images(
                self.split_view_widget._original,
                self.split_view_widget._processed,
            )
        layout.addWidget(fs_split, 1)

        exit_btn = Button3D("✕  Exit Full Screen", variant="danger", parent=container)
        exit_btn.setFixedHeight(36)
        exit_btn.clicked.connect(fs_win.close)
        layout.addWidget(exit_btn)

        fs_win.setCentralWidget(container)
        fs_win.destroyed.connect(lambda: setattr(self, "_fs_win", None))
        fs_win.showFullScreen()

    def _refresh_comparison_view(self) -> None:
        """Refresh split or side-by-side view with current images."""
        mode = getattr(self, "_preview_mode", "single")
        if mode not in ("split", "sidebyside"):
            return
        original_path = self._preview_source_frame_path or self._snapshot_fallback_path
        processed_path = self._last_processed_preview_path
        if not processed_path:
            if self._active_output_path and os.path.isfile(self._active_output_path):
                processed_path = self._active_output_path
        if not original_path or not processed_path:
            Toast.show(self, "Need both original and processed frames", "warning")
            return

        orig = QPixmap(original_path)
        proc = QPixmap(processed_path)
        if orig.isNull() or proc.isNull():
            Toast.show(self, "Could not load comparison images", "warning")
            return

        if mode == "split":
            self.split_view_widget.set_images(orig, proc)
            self.center_stack.setCurrentWidget(self.split_view_widget)
        else:
            # Side-by-side: combine into single pixmap.
            self._show_comparison(original_path, processed_path)

    def _build_io_bar(self) -> QWidget:
        """Visible I/O path bar above preview."""
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_SM)

        self.input_path_edit = QLineEdit(widget)
        self.output_path_edit = QLineEdit(widget)
        self.input_path_edit.setPlaceholderText("Input path")
        self.output_path_edit.setPlaceholderText("Output path")
        self.browse_input_btn = Button3D("Browse", variant="default", parent=widget)
        self.browse_output_btn = Button3D("Browse", variant="default", parent=widget)
        self.split_btn = Button3D("Split", variant="default", parent=widget)

        layout.addWidget(QLabel("Input", widget))
        layout.addWidget(self.input_path_edit, 2)
        layout.addWidget(self.browse_input_btn)
        layout.addWidget(QLabel("Output", widget))
        layout.addWidget(self.output_path_edit, 2)
        layout.addWidget(self.browse_output_btn)
        layout.addWidget(self.split_btn)
        return widget

    def _build_trim_bar(self) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_MD)

        self.in_label = QLabel("IN 00:00:00")
        self.in_label.setStyleSheet(f"color: {Colors.TRIM_HANDLE_IN};")
        self.out_label = QLabel("OUT 00:00:00")
        self.out_label.setStyleSheet(f"color: {Colors.TRIM_HANDLE_OUT};")
        self.duration_label = QLabel("Duration 00:00:00 (0 frames)")
        self.duration_label.setStyleSheet(f"color: {Colors.TEXT_ACCENT};")

        layout.addWidget(self.in_label)
        layout.addWidget(self.duration_label)
        layout.addWidget(self.out_label)
        layout.addStretch(1)
        return widget

    def _build_action_bar(self) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_MD)

        self.preview_btn = Button3D("Preview", variant="default", parent=widget)
        self.export_btn = Button3D("Export", variant="primary", parent=widget)
        self.cancel_btn = Button3D("Cancel", variant="danger", parent=widget)
        self.cancel_btn.setEnabled(False)
        self.progress = AnimatedProgressBar(widget)

        layout.addWidget(self.preview_btn)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.progress, 1)
        layout.addWidget(self.cancel_btn)
        return widget

    def _build_status_bar(self) -> None:
        status = QStatusBar(self)
        self.setStatusBar(status)
        self.status_label = QLabel("Ready", self)
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self.gpu_label = QLabel("GPU: detecting…", self)
        self.gpu_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self.vram_label = QLabel("VRAM: —", self)
        self.vram_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self.codec_label = QLabel("Encoders: probing…", self)
        self.codec_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self.settings_summary_label = QLabel("", self)
        self.settings_summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        status.addWidget(self.status_label, 1)
        status.addPermanentWidget(self.gpu_label)
        status.addPermanentWidget(self.vram_label)
        status.addPermanentWidget(self.codec_label)
        status.addPermanentWidget(self.settings_summary_label)

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence("I"), self, activated=self._set_trim_in)
        QShortcut(QKeySequence("O"), self, activated=self._set_trim_out)

    def _connect_signals(self) -> None:
        self.project_panel.file_selected.connect(self.load_file)
        self.project_panel.file_removed.connect(self._on_file_removed)
        self.project_panel.input_folder_selected.connect(self._on_input_folder_selected)
        self.drop_zone.file_dropped.connect(self._on_file_dropped)

        self.trim_timeline.in_point_changed.connect(lambda _=0: self._update_trim_labels())
        self.trim_timeline.out_point_changed.connect(lambda _=0: self._update_trim_labels())
        self.trim_timeline.playhead_moved.connect(self.preview_widget.seek_frame)
        self.trim_timeline.playhead_moved.connect(self._update_current_timecode)
        self.preview_widget.frame_changed.connect(self._on_preview_frame_changed)

        self.playback_controls.play_pause_toggled.connect(self._toggle_playback)
        self.playback_controls.prev_frame_requested.connect(self.preview_widget.step_backward)
        self.playback_controls.next_frame_requested.connect(self.preview_widget.step_forward)
        self.playback_controls.mute_toggled.connect(self._on_mute_toggled)
        self.playback_controls.trim_in_requested.connect(self._set_trim_in)
        self.playback_controls.trim_out_requested.connect(self._set_trim_out)
        self.playback_controls.trim_clear_requested.connect(self._clear_trim)

        self.preview_btn.clicked.connect(self._request_preview)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)

        self.show_log_btn.clicked.connect(self._show_log_viewer)
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        self.about_btn.clicked.connect(self._show_about_dialog)
        self.github_btn.clicked.connect(lambda: webbrowser.open(GITHUB_URL))
        self.update_btn.clicked.connect(lambda: webbrowser.open(f"{GITHUB_URL}/releases"))

        self.settings_panel.settings_changed.connect(self._update_status_summary)

        self.browse_input_btn.clicked.connect(self._browse_input_path)
        self.browse_output_btn.clicked.connect(self._browse_output_path)
        self.split_btn.clicked.connect(self._show_split_comparison)
        self.input_path_edit.editingFinished.connect(self._on_input_path_edited)
        self.output_path_edit.editingFinished.connect(self._on_output_path_edited)

    # ---------------------------------------------------------------- drag & drop on main window
    _VALID_DROP_EXTS = {
        ".mp4", ".mov", ".mkv", ".avi", ".webm",
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".dpx", ".bmp",
    }

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        last_path = None
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.splitext(path)[1].lower() in self._VALID_DROP_EXTS:
                self.project_panel.add_file(path, select=False)
                last_path = path
        if last_path:
            self.load_file(last_path)
        event.acceptProposedAction()

    def _clear_trim(self) -> None:
        """Reset IN/OUT to full range (fix #8 — X clear button)."""
        self.trim_timeline.set_full_range()
        self._update_trim_labels()

    @staticmethod
    def _normalize_audio_mode(value: str) -> str:
        text = (value or "").strip().lower()
        if text in {"none", "no audio"}:
            return "none"
        if "copy" in text or "passthrough" in text:
            return "copy"
        if "aac" in text:
            digits = "".join(ch for ch in text if ch.isdigit())
            return f"aac_{digits}kbps" if digits else "aac_192kbps"
        if "flac" in text:
            return "flac"
        return "copy"

    def _build_export_ffmpeg_args(self, export_settings: dict, settings: dict) -> list[str]:
        codec = str(export_settings.get("codec", "H265")).strip()
        full_cmd = build_ffmpeg_command(
            input_path=str(self._current_file or ""),
            output_path=str(export_settings.get("output_path", "")),
            codec=codec,
            profile=str(export_settings.get("profile", "")).strip(),
            quality_level=str(export_settings.get("quality_level", "medium")).lower(),
            bitrate_mode=str(export_settings.get("bitrate_mode", "dynamic")).lower(),
            bitrate_mbps=float(export_settings.get("bitrate_mbps", 0) or 0),
            audio_mode=self._normalize_audio_mode(str(export_settings.get("audio_mode", "copy"))),
            source_fps=float(export_settings.get("source_fps", self.preview_widget.get_fps() or 30.0)),
            use_10bit=bool(settings.get("use_10bit", True)),
            trim_in=int(export_settings.get("trim_in", -1)),
            trim_out=int(export_settings.get("trim_out", -1)),
        )
        # CLI writer accepts only ffmpeg video args (not input/output/audio flags).
        try:
            cvidx = full_cmd.index("-c:v")
        except ValueError:
            return []
        return full_cmd[cvidx:-1]

    def _on_input_folder_selected(self, folder: str) -> None:
        if not folder or self._processing:
            return
        payload = {
            "mode": "batch_folder",
            "input": folder,
            "settings": self.settings_panel.get_all_settings(),
            "processing_list_file": str(Path(folder) / "seedvr2_processing_list.txt"),
        }
        self.export_requested.emit(payload)

    def _resolve_output_dir(self) -> Path:
        raw = self.output_path_edit.text().strip() if hasattr(self, "output_path_edit") else ""
        if raw:
            out = Path(raw)
            if out.is_file():
                out = out.parent
            out.mkdir(parents=True, exist_ok=True)
            return out
        if self._current_file:
            return Path(self._current_file).parent
        return Path.cwd()

    def _browse_input_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select input media", "", "Media (*.*)")
        if not path:
            return
        self.input_path_edit.setText(path)
        self.load_file(path)

    def _browse_output_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_path_edit.text().strip() or "")
        if not path:
            return
        self.output_path_edit.setText(path)
        self._config["output_path"] = path
        save_config(self._config)

    def _on_input_path_edited(self) -> None:
        path = self.input_path_edit.text().strip()
        if path and os.path.isfile(path):
            self.load_file(path)

    def _on_output_path_edited(self) -> None:
        raw = self.output_path_edit.text().strip()
        self._config["output_path"] = raw
        save_config(self._config)

    def _start_hardware_probe(self) -> None:
        self._hw_thread = QThread(self)
        self._hw_worker = _HardwareProbe()
        self._hw_worker.moveToThread(self._hw_thread)
        self._hw_thread.started.connect(self._hw_worker.run)
        self._hw_worker.detected.connect(self._on_hardware_detected)
        self._hw_worker.detected.connect(self._hw_thread.quit)
        self._hw_thread.start()

    def _start_codec_probe(self) -> None:
        ffmpeg_path = self._config.get("ffmpeg_path", "")
        self._codec_thread = QThread(self)
        self._codec_worker = _CodecProbe(ffmpeg_path)
        self._codec_worker.moveToThread(self._codec_thread)
        self._codec_thread.started.connect(self._codec_worker.run)
        self._codec_worker.detected.connect(self._on_codec_detected)
        self._codec_worker.detected.connect(self._codec_thread.quit)
        self._codec_thread.start()

    def _on_hardware_detected(self, name: str, vram: str) -> None:
        self._gpu_name = name
        self._gpu_vram = vram
        self.gpu_label.setText(f"GPU: {name}")
        self.vram_label.setText(f"VRAM: {vram}")
        self.settings_panel.set_device_info_lines([
            f"GPU: {name}",
            "GPU Load: —  |  —°C",
            f"VRAM: — / {vram}" if vram and vram != "—" else "VRAM: — / —",
            "Shared VRAM: — / —",
            "CPU: —  |  —°C",
            "RAM: — / —",
            "Temp: CPU —°C | GPU —°C",
        ])

    def _on_codec_detected(self, result: dict) -> None:
        self._codec_cache = result
        available = [name.upper() for name, enabled in result.items() if enabled]
        self.codec_label.setText(
            f"Encoders: {', '.join(available)}" if available else "Encoders: software"
        )

    def _update_status_summary(self) -> None:
        settings = self.settings_panel.get_all_settings()
        summary = (
            f"Mode {settings['resolution_mode']} • Batch {settings['batch_size']} • "
            f"Auto Tune {'On' if settings['auto_tune'] else 'Off'}"
        )
        self.settings_summary_label.setText(summary)

    def _update_device_info(self) -> None:
        cpu_pct = 0.0
        cpu_temp = None
        ram_used_gb = 0.0
        ram_total_gb = 0.0
        ram_pct = 0.0
        if psutil is not None:
            try:
                cpu_pct = float(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                ram_used_gb = float(vm.used) / (1024 ** 3)
                ram_total_gb = float(vm.total) / (1024 ** 3)
                ram_pct = float(vm.percent)
                temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
                if temps:
                    for entries in temps.values():
                        if entries:
                            cpu_temp = getattr(entries[0], "current", None)
                            if cpu_temp is not None:
                                break
            except Exception:
                pass

        gpu_name = self._gpu_name
        gpu_util = None
        gpu_temp = None
        vram_used_gb = None
        vram_total_gb = None
        vram_shared_used_gb = None
        vram_shared_total_gb = None
        if _HAS_NVML and pynvml is not None:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8", errors="replace")
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util.gpu)
                gpu_temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used_gb = float(mem.used) / (1024 ** 3)
                vram_total_gb = float(mem.total) / (1024 ** 3)
                try:
                    mem2 = pynvml.nvmlDeviceGetMemoryInfo_v2(handle)
                    reserved = float(getattr(mem2, "reserved", 0.0))
                    vram_shared_used_gb = reserved / (1024 ** 3)
                    vram_shared_total_gb = max(0.0, (vram_total_gb or 0.0) - (vram_used_gb or 0.0))
                except Exception:
                    pass
            except Exception:
                pass

        gpu_line = (
            f"GPU Load: {gpu_util:.0f}%  |  {gpu_temp:.0f}°C"
            if gpu_util is not None and gpu_temp is not None
            else "GPU Load: —  |  —°C"
        )
        if vram_used_gb is not None and vram_total_gb and vram_total_gb > 0:
            vram_pct = (vram_used_gb / vram_total_gb) * 100.0
            vram_line = f"VRAM: {vram_used_gb:.1f} / {vram_total_gb:.1f} GB ({vram_pct:.0f}%)"
        else:
            vram_line = f"VRAM: — / {self._gpu_vram}" if self._gpu_vram and self._gpu_vram != "—" else "VRAM: — / —"
        if vram_shared_used_gb is not None and vram_shared_total_gb is not None:
            shared_line = f"Shared VRAM: {vram_shared_used_gb:.1f} / {vram_shared_total_gb:.1f} GB"
        else:
            shared_line = "Shared VRAM: — / —"
        cpu_line = f"CPU: {cpu_pct:.0f}%  |  {cpu_temp:.0f}°C" if cpu_temp is not None else f"CPU: {cpu_pct:.0f}%  |  —°C"
        if ram_total_gb > 0:
            ram_line = f"RAM: {ram_used_gb:.1f} / {ram_total_gb:.1f} GB ({ram_pct:.0f}%)"
        else:
            ram_line = "RAM: — / —"
        temp_line = (
            f"Temp: CPU {cpu_temp:.0f}°C | GPU {gpu_temp:.0f}°C"
            if cpu_temp is not None and gpu_temp is not None
            else "Temp: CPU —°C | GPU —°C"
        )
        self.settings_panel.set_device_info_lines([
            f"GPU: {gpu_name}",
            gpu_line,
            vram_line,
            shared_line,
            cpu_line,
            ram_line,
            temp_line,
        ])

    def _on_file_dropped(self, path: str) -> None:
        self.project_panel.add_file(path)
        self.load_file(path)

    def _on_file_removed(self, path: str) -> None:
        """Handle X-button removal of a file from the project panel."""
        if path == self._current_file:
            self._current_file = ""
            self.preview_widget.clear()
            self.input_path_edit.setText("")
            self._preview_source_frame_path = ""
            self._last_processed_preview_path = ""

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def load_file(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            return
        self._current_file = path
        self.input_path_edit.setText(path)
        if not self.output_path_edit.text().strip():
            self.output_path_edit.setText(str(Path(path).parent))
        self.preview_widget.load_file(path)
        self.trim_timeline.load_video(path)
        self.settings_panel.set_trim_range(0, 0, self.preview_widget.get_frame_count(), False)
        self._set_preview_mode("single")
        # Fix 4 — auto-set image mode for image files
        if os.path.splitext(path)[1].lower() in self._IMAGE_EXTS:
            self.settings_panel.set_image_mode()
        self._update_trim_labels()
        self._update_video_info()
        out_dir = self.output_path_edit.text().strip() or str(Path(path).parent)
        self.project_panel.set_output_dir(out_dir)
        self.status_label.setText(f"Loaded {os.path.basename(path)}")
        self._config["input_path"] = path
        self._config["output_path"] = self.output_path_edit.text().strip()
        save_config(self._config)

    def _on_preview_frame_changed(self, frame: int) -> None:
        self.trim_timeline.set_playhead(frame)
        self._update_current_timecode(frame)

    def _update_current_timecode(self, frame: int) -> None:
        self.status_label.setText(f"Frame {frame} • {self.trim_timeline.frame_to_timecode(frame)}")

    def _update_trim_labels(self) -> None:
        in_frame, out_frame = self.trim_timeline.get_selected_range()
        self.settings_panel.set_trim_range(
            in_frame,
            out_frame,
            self.preview_widget.get_frame_count(),
            not self.trim_timeline.is_full_range(),
        )
        self.in_label.setText(f"IN {self.trim_timeline.frame_to_timecode(in_frame)}")
        self.out_label.setText(f"OUT {self.trim_timeline.frame_to_timecode(out_frame)}")
        count = self.trim_timeline.get_selected_frame_count()
        self.duration_label.setText(
            f"Duration {self.trim_timeline.frame_to_timecode(count)} ({count} frames)"
        )

    def _set_trim_in(self) -> None:
        self.trim_timeline.set_in_point(self.preview_widget.current_frame())

    def _set_trim_out(self) -> None:
        self.trim_timeline.set_out_point(self.preview_widget.current_frame())

    def _toggle_playback(self, playing: bool) -> None:
        if playing:
            self.preview_widget.play()
        else:
            self.preview_widget.pause()

    def _on_mute_toggled(self, muted: bool) -> None:
        """Propagate mute state to the video preview widget's audio output."""
        ao = getattr(self.preview_widget, "_audio_output", None)
        if ao is not None:
            try:
                ao.setMuted(muted)
            except Exception:
                pass

    def _request_preview(self) -> None:
        if not self._current_file:
            Toast.show(self, "Import a file first", "warning")
            return
        self.preview_requested.emit(
            {
                "mode": "preview",
                "input": self._current_file,
                "settings": self.settings_panel.get_all_settings(),
                "frame_index": self.preview_widget.current_frame(),
            }
        )

    def _on_export_clicked(self) -> None:
        if not self._current_file:
            Toast.show(self, "Import a file first", "warning")
            return
        input_path = Path(self._current_file)
        default_dir = str(input_path.parent)
        default_name = input_path.stem
        in_frame, out_frame = self.trim_timeline.get_selected_range()
        if self.trim_timeline.is_full_range():
            in_frame, out_frame = 0, 0
        dialog = ExportDialog(
            self,
            default_dir=default_dir,
            default_name=default_name,
            frame_count=self.preview_widget.get_frame_count(),
            width=self.preview_widget.get_frame_width(),
            height=self.preview_widget.get_frame_height(),
            fps=self.preview_widget.get_fps(),
            trim_in=in_frame,
            trim_out=out_frame,
        )
        dialog.export_confirmed.connect(self._on_export_confirmed)
        dialog.exec()

    def _on_export_confirmed(self, export_settings: dict) -> None:
        export_settings["source_fps"] = self.preview_widget.get_fps()
        payload = {
            "mode": "export",
            "input": self._current_file,
            "settings": self.settings_panel.get_all_settings(),
            "export": export_settings,
        }
        self.export_requested.emit(payload)

    def _config_alarm_enabled(self) -> bool:
        raw = str(self._config.get("alarm_enabled", "true")).strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _build_worker_env(self) -> dict:
        env: dict = {}
        ffmpeg_path = self._config.get("ffmpeg_path", "")
        if ffmpeg_path:
            ffmpeg_dir = str(Path(ffmpeg_path).parent)
            env["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return env

    def _get_output_path(self, input_path: str, suffix: str = "_sv2", ext: Optional[str] = None) -> str:
        """Output always goes to the same directory as the input file."""
        inp = Path(input_path)
        if ext is None:
            ext = inp.suffix
        output = inp.parent / f"{inp.stem}{suffix}{ext}"
        return str(output)

    def _with_sv2_suffix(self, output_path: str) -> str:
        if not output_path:
            return output_path
        path = Path(output_path)
        if path.suffix:
            stem = path.stem if path.stem.endswith("_sv2") else f"{path.stem}_sv2"
            return str(path.with_name(stem + path.suffix))
        name = path.name if path.name.endswith("_sv2") else f"{path.name}_sv2"
        return str(path.with_name(name))

    def _preview_output_path(self) -> str:
        return str(Path(TEMP_DIR) / "preview_result.tiff")

    def _build_cli_args(self, payload: dict) -> list[str]:
        settings = payload.get("settings", {})
        args: list[str] = [payload["input"]]

        output_path = payload.get("output_path") or payload.get("export", {}).get("output_path", "")
        if output_path:
            args += ["--output", output_path]
        output_format = payload.get("output_format")
        if not output_format:
            export = payload.get("export", {})
            if export:
                if export.get("output_type") == "image_sequence":
                    output_format = str(export.get("image_format", "TIFF")).lower()
                else:
                    output_format = str(export.get("container", "MP4")).lower()
        if output_format:
            args += ["--output_format", output_format]
        source_fps = float(payload.get("export", {}).get("source_fps", 0.0) or 0.0)
        if source_fps > 0:
            args += ["--source_fps", f"{source_fps:.6f}"]
        processing_list_file = str(payload.get("processing_list_file", "")).strip()
        if processing_list_file:
            args += ["--processing_list_file", processing_list_file, "--video_only_directory"]

        resolution_mode = settings.get("resolution_mode", "pixel")
        if resolution_mode == "xtimes":
            args += ["--resolution_mode", "xtimes", "--resolution_scale", str(settings.get("resolution_scale", "2"))]
        else:
            resolution = int(settings.get("resolution", 720))
            if resolution_mode == "presets":
                resolution = self._PRESET_TO_RESOLUTION.get(settings.get("resolution_presets", "720p (HD)"), resolution)
            args += ["--resolution_mode", "pixel", "--resolution", str(resolution)]

        args += ["--max_resolution", str(int(settings.get("max_resolution", 3840)))]
        args += ["--pre_downscale", str(settings.get("pre_downscale", "1"))]
        args += ["--dit_model", str(settings.get("dit_model", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"))]
        args += ["--attention_mode", str(settings.get("attention_mode", "sage_attn_3")).replace("sage_attn", "sageattn")]
        args += ["--batch_size", str(int(settings.get("batch_size", 81)))]
        args += ["--load_cap", str(int(settings.get("load_cap", 0)))]
        args += ["--skip_first_frames", str(int(settings.get("skip_first_frames", 0)))]
        args += ["--vae_encode_tile_size", str(int(settings.get("vae_encode_tile_size", 1024)))]
        args += ["--vae_encode_tile_overlap", str(int(settings.get("vae_encode_tile_overlap", 64)))]
        args += ["--vae_decode_tile_size", str(int(settings.get("vae_decode_tile_size", 1024)))]
        args += ["--vae_decode_tile_overlap", str(int(settings.get("vae_decode_tile_overlap", 64)))]
        args += ["--color_correction", str(settings.get("color_correction", "none"))]
        args += ["--input_noise_scale", f"{float(settings.get('input_noise_scale', 0.0)):.2f}"]
        args += ["--latent_noise_scale", f"{float(settings.get('latent_noise_scale', 0.0)):.2f}"]
        args += ["--temporal_overlap", str(int(settings.get("temporal_overlap", 8)))]
        args += ["--prepend_frames", str(int(settings.get("prepend_frames", 4)))]
        args += ["--dit_offload_device", str(settings.get("dit_offload_device", "none"))]
        args += ["--vae_offload_device", str(settings.get("vae_offload_device", "cpu"))]
        args += ["--tensor_offload_device", str(settings.get("tensor_offload_device", "none"))]
        args += ["--blocks_to_swap", str(int(settings.get("blocks_to_swap", 0)))]
        args += ["--seed", str(int(settings.get("seed", 313)))]
        args += ["--video_backend", str(settings.get("video_backend", "ffmpeg"))]
        args += ["--tile_debug", str(settings.get("tile_debug", "false"))]

        chunk_minutes = int(settings.get("chunk_minutes", 0))
        if chunk_minutes > 0:
            args += ["--chunk_duration_minutes", str(chunk_minutes)]

        only_frames = str(settings.get("only_frames", "")).strip()
        if only_frames:
            args += ["--only_frames", only_frames]
        cuda_device = str(settings.get("cuda_device", "")).strip()
        if cuda_device:
            args += ["--cuda_device", cuda_device]
        model_dir = self._config.get("models_dir", "")
        if model_dir:
            args += ["--model_dir", model_dir]

        if settings.get("uniform_batch_size"):
            args.append("--uniform_batch_size")
        if settings.get("auto_tune"):
            args.append("--auto_tune")
        if settings.get("cache_dit"):
            args.append("--cache_dit")
        if settings.get("cache_vae"):
            args.append("--cache_vae")
        if settings.get("use_10bit"):
            args.append("--10bit")
        if settings.get("debug"):
            args.append("--debug")
        if settings.get("vae_encode_tiled"):
            args.append("--vae_encode_tiled")
        if settings.get("vae_decode_tiled"):
            args.append("--vae_decode_tiled")
        if settings.get("swap_io_components"):
            args.append("--swap_io_components")
        if settings.get("compile_dit"):
            args.append("--compile_dit")
        if settings.get("compile_vae"):
            args.append("--compile_vae")
        if settings.get("compile_backend"):
            args += ["--compile_backend", str(settings.get("compile_backend"))]
        if settings.get("compile_mode"):
            args += ["--compile_mode", str(settings.get("compile_mode"))]
        export = payload.get("export", {})
        if export and export.get("output_type") == "video":
            ffmpeg_args = self._build_export_ffmpeg_args(export, settings)
            if ffmpeg_args:
                args += ["--ffmpeg_video_args", json.dumps(ffmpeg_args)]

        return args

    def _spawn_worker(self, payload: dict) -> None:
        if self._processing:
            Toast.show(self, "A job is already running", "warning")
            return
        self._config = load_config()
        python_exe, cli_script = resolve_paths(
            self._config.get("seedvr2_folder", ""),
            self._config.get("python_exe", ""),
        )
        if not os.path.isfile(cli_script):
            Toast.show(self, "inference_cli.py not found", "error")
            return

        export_settings = dict(payload.get("export", {}))
        # Always write output to the same directory as the input file (Fix #6 / P1).
        if self._current_file:
            source = Path(self._current_file)
            if export_settings.get("output_type") == "image_sequence":
                container = ""
                ext = "." + str(export_settings.get("image_format", "tiff")).lower()
            else:
                container = str(export_settings.get("container", "mp4")).lower()
                ext = f".{container}"
            output_path = self._get_output_path(self._current_file, "_sv2", ext)
        else:
            output_path = self._with_sv2_suffix(str(export_settings.get("output_path", "")))
        export_settings["output_path"] = output_path
        payload = dict(payload)
        payload["export"] = export_settings
        payload["output_path"] = output_path
        if export_settings:
            payload["output_format"] = (
                str(export_settings.get("image_format", "TIFF")).lower()
                if export_settings.get("output_type") == "image_sequence"
                else str(export_settings.get("container", "MP4")).lower()
            )

        # Fix 5 — capture snapshot of current frame for split view after export
        frame_image = self.preview_widget.current_frame_image()
        if not frame_image.isNull():
            snap_path = str(Path(TEMP_DIR) / "export_snapshot.tiff")
            self._save_qimage_as_tiff16(frame_image, snap_path)
            self._snapshot_fallback_path = snap_path
        else:
            self._snapshot_fallback_path = ""

        args = self._build_cli_args(payload)
        self._start_worker(cli_script, python_exe, args, "export", output_path)

    def _spawn_preview(self, payload: dict) -> None:
        if self._processing:
            Toast.show(self, "A job is already running", "warning")
            return
        if not self._current_file:
            return
        self._config = load_config()
        python_exe, cli_script = resolve_paths(
            self._config.get("seedvr2_folder", ""),
            self._config.get("python_exe", ""),
        )
        if not os.path.isfile(cli_script):
            Toast.show(self, "inference_cli.py not found", "error")
            return

        frame_image = self.preview_widget.current_frame_image()
        if frame_image.isNull():
            Toast.show(self, "No preview frame available", "warning")
            return

        os.makedirs(TEMP_DIR, exist_ok=True)
        input_frame_path = Path(TEMP_DIR) / "preview_source.tiff"
        preview_output = Path(self._preview_output_path())

        self._save_qimage_as_tiff16(frame_image, str(input_frame_path))
        self._preview_source_frame_path = str(input_frame_path)

        preview_settings = dict(payload.get("settings", {}))
        preview_settings["skip_first_frames"] = 0
        preview_settings["load_cap"] = 0
        preview_settings["batch_size"] = 1
        preview_settings["uniform_batch_size"] = False
        preview_payload = {
            "mode": "preview",
            "input": str(input_frame_path),
            "settings": preview_settings,
            "output_path": str(preview_output),
            "output_format": "tiff",
        }
        self._snapshot_fallback_path = str(
            Path(TEMP_DIR) / "preview_fallback.tiff"
        )
        self._save_qimage_as_tiff16(frame_image, self._snapshot_fallback_path)
        args = self._build_cli_args(preview_payload)
        self._start_worker(cli_script, python_exe, args, "preview", preview_payload["output_path"])

    def _start_worker(self, cli_script: str, python_exe: str, args: list[str], mode: str, output_path: str) -> None:
        self._active_mode = mode
        self._active_output_path = output_path
        self._thread, self._worker = create_worker_thread(
            cli_script,
            args,
            python_exe,
            self._build_worker_env(),
        )
        self._worker.alarm_enabled = self._config_alarm_enabled()
        self._worker.log_line.connect(self._on_log_line)
        self._worker.progress_update.connect(self._on_progress)
        self._worker.batch_progress_update.connect(self._on_batch_progress)
        self._worker.chunk_status_update.connect(self._on_chunk_status)
        self._worker.queue_status_update.connect(self._on_queue_status)
        self._worker.phase_update.connect(self._on_phase_update)
        self._worker.oom_detected.connect(self._on_oom_detected)
        self._worker.started_signal.connect(lambda: self._set_processing(True))
        self._worker.finished.connect(self._on_finished)
        self.progress.reset()
        self._current_phase_index = 0
        self._chunk_current = 0
        self._chunk_total = 0
        self._batch_current = 0
        self._batch_total = 0
        self._phase_name = "idle"
        self._update_progress_status_text()
        self._thread.start()

    def _set_processing(self, processing: bool) -> None:
        self._processing = processing
        self.preview_btn.setEnabled(not processing)
        self.export_btn.setEnabled(not processing)
        self.cancel_btn.setEnabled(processing)
        self.settings_panel.set_enabled_state(not processing)
        self.status_label.setText("Processing…" if processing else "Ready")

    def _abort_worker(self) -> None:
        if self._worker is not None:
            self._worker.request_abort()
            self.status_label.setText("Cancelling…")

    def _on_log_line(self, line: str) -> None:
        message = line.strip()
        if message:
            self.status_label.setText(message[:120])
        if self._log_viewer is not None:
            self._log_viewer.append_line(line)

    def _on_progress(self, current: int, total: int) -> None:
        """Process (current phase) progress — right bar."""
        if total > 0:
            self.progress.setValue(100.0 * current / total)
            self._batch_current = current
            self._batch_total = total
            self._update_progress_status_text()

    def _on_batch_progress(self, current: int, total: int) -> None:
        """Legacy per-step progress updates from CLI."""
        if total > 0:
            self.progress.setValue(100.0 * current / total)
            self._batch_current = current
            self._batch_total = total
            self._update_progress_status_text()

    def _on_chunk_status(
        self,
        phase_name: str,
        chunk_current: int,
        chunk_total: int,
        batch_current: int,
        batch_total: int,
    ) -> None:
        self._phase_name = (phase_name or self._phase_name or "encoding").strip().lower()
        if chunk_total > 0:
            self._chunk_current = chunk_current
            self._chunk_total = chunk_total
        if batch_total > 0:
            self._batch_current = batch_current
            self._batch_total = batch_total
        self._update_progress_status_text()

    def _on_phase_update(self, phase_name: str, current: int, total: int, phase_progress: float = 0.0) -> None:
        if total <= 0 or current <= 0:
            return
        self._phase_name = phase_name
        self._current_phase_index = current
        bounded_phase_progress = max(0.0, min(1.0, float(phase_progress)))
        self.progress.setTotalLabel(f"{current}/{total} {phase_name}")
        # Fix 8 — jump total bar to phase start immediately when phase begins
        self.progress.setTotalValue(100.0 * current / total)
        self.progress.setValue(100.0 * bounded_phase_progress)
        self.status_label.setText(phase_name)
        self._update_progress_status_text()

    def _update_progress_status_text(self) -> None:
        phase = str(self._phase_name or "idle").upper()
        chunk_cur = self._chunk_current if self._chunk_current > 0 else 0
        chunk_tot = self._chunk_total if self._chunk_total > 0 else 0
        batch_cur = self._batch_current if self._batch_current > 0 else 0
        batch_tot = self._batch_total if self._batch_total > 0 else 0
        self.progress.setStatusText(
            f"CHUNK {chunk_cur}/{chunk_tot} | BATCH {batch_cur}/{batch_tot} | {phase}"
        )

    def _on_queue_status(self, file_path: str, current: int, total: int, done: int, remaining: int) -> None:
        name = os.path.basename(file_path) if file_path else "queue"
        self.status_label.setText(f"Processing {current}/{total}: {name}")

    def _on_oom_detected(self, retry_count: int, max_retries: int, new_batch_size: int) -> None:
        self.status_label.setText(
            f"CUDA OOM detected • retry {retry_count}/{max_retries} • suggested batch {new_batch_size}"
        )
        Toast.show(
            self,
            f"CUDA OOM: retry {retry_count}/{max_retries}, batch {new_batch_size}",
            "warning",
            4000,
        )

    def _load_preview_outputs(self) -> Optional[Path]:
        candidates: list[Path] = []
        output = Path(self._active_output_path) if self._active_output_path else None
        if output is not None:
            if output.exists():
                candidates.append(output)
            parent = output.parent
            stem = output.stem if output.suffix else output.name
            for pattern in (f"{stem}*.tif", f"{stem}*.tiff", f"{stem}*.png", f"{stem}*.jpg", f"{stem}*.jpeg"):
                candidates.extend(sorted(parent.glob(pattern)))
        for extra in (self._snapshot_fallback_path,):
            if extra and Path(extra).exists():
                candidates.append(Path(extra))

        seen = set()
        for candidate in candidates:
            if candidate in seen or not candidate.exists():
                continue
            seen.add(candidate)
            pixmap = QPixmap(str(candidate))
            if pixmap.isNull():
                image = QImage(str(candidate))
                if image.isNull():
                    continue
                pixmap = QPixmap.fromImage(image)
            self.preview_widget.set_pixmap(pixmap)
            self.center_stack.setCurrentWidget(self.preview_widget)
            self.status_label.setText(f"Preview loaded: {candidate.name}")
            self._last_processed_preview_path = str(candidate)
            return candidate
        return None

    def _show_comparison(self, original_path: str, processed_path: str) -> None:
        if not original_path or not processed_path:
            return
        orig = QPixmap(original_path)
        proc = QPixmap(processed_path)
        if orig.isNull() or proc.isNull():
            return
        target_h = max(1, max(orig.height(), proc.height()))
        orig_s = orig.scaledToHeight(target_h, Qt.SmoothTransformation)
        proc_s = proc.scaledToHeight(target_h, Qt.SmoothTransformation)
        combined = QPixmap(orig_s.width() + proc_s.width(), target_h)
        combined.fill(Qt.transparent)
        painter = QPainter(combined)
        painter.drawPixmap(0, 0, orig_s)
        painter.drawPixmap(orig_s.width(), 0, proc_s)
        painter.end()
        self.preview_widget.set_pixmap(combined)
        self.center_stack.setCurrentWidget(self.preview_widget)
        self.status_label.setText("Comparison shown: original vs processed")

    def _show_split_comparison(self) -> None:
        processed = self._last_processed_preview_path
        if not processed and self._active_output_path and Path(self._active_output_path).exists():
            processed = self._active_output_path
        original = self._preview_source_frame_path or self._snapshot_fallback_path
        if not processed or not Path(processed).exists():
            Toast.show(self, "No processed preview found yet", "warning")
            return
        if not original or not Path(original).exists():
            Toast.show(self, "No source frame available for split view", "warning")
            return
        # Use SplitViewWidget for proper draggable split comparison.
        orig = QPixmap(original)
        proc = QPixmap(processed)
        if orig.isNull() or proc.isNull():
            self._show_comparison(original, processed)
            return
        self.split_view_widget.set_images(orig, proc)
        self.center_stack.setCurrentWidget(self.split_view_widget)
        self.status_label.setText("Split view: drag divider to compare")

    def _on_finished(self, success: bool, message: str) -> None:
        self._set_processing(False)
        self.playback_controls.set_playing(False)
        if success:
            self.progress.setValue(100.0)
            self.progress.setTotalValue(100.0)
            self._phase_name = "done"
            self._batch_current = max(self._batch_current, self._batch_total)
            self._chunk_current = max(self._chunk_current, self._chunk_total)
            self._update_progress_status_text()
            self._play_success_sound()
            Toast.show(self, "Completed successfully", "success")
            loaded = self._load_preview_outputs()
            original = self._preview_source_frame_path or self._snapshot_fallback_path
            if loaded is not None and original and Path(original).exists():
                # Fix 5 — auto-add to split view after both preview and export
                self._preview_source_frame_path = original
                self._last_processed_preview_path = str(loaded)
                self._set_preview_mode("split")
        else:
            Toast.show(self, message or "Processing failed", "error")
            if message and message != "Cancelled.":
                # Add OOM hint if applicable.
                oom_hint = ""
                if "out of memory" in message.lower():
                    oom_hint = (
                        "\n\n⚠ Recovery suggestions:\n"
                        "• Reduce batch size\n• Enable VAE tiling\n"
                        "• Lower resolution\n• Enable offload"
                    )
                dlg = ErrorDialog(self, "Processing failed", message + oom_hint)
                dlg.show()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(500)
        self._thread = None
        self._worker = None
        self._active_mode = ""

    def _play_success_sound(self) -> None:
        if not self._config_alarm_enabled():
            return
        try:
            import winsound

            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception:
            pass

    def _save_qimage_as_tiff16(self, image: QImage, path: str) -> None:
        if image.isNull() or not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            converted = image.convertToFormat(QImage.Format_RGB888)
            width = converted.width()
            height = converted.height()
            buffer = converted.bits()
            array = np.frombuffer(buffer, dtype=np.uint8, count=converted.sizeInBytes())
            array = array.reshape((height, width, 3))
            rgb16 = array.astype(np.uint16) * 257
            bgr16 = cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
            _ext = os.path.splitext(str(target))[1]
            _success, _buf = cv2.imencode(_ext, bgr16)
            if _success:
                _buf.tofile(str(target))
        except Exception:
            image.save(str(target), "TIFF")

    def _show_log_viewer(self) -> None:
        if self._log_viewer is None:
            self._log_viewer = LogViewer(self)
        self._log_viewer.show()
        self._log_viewer.raise_()
        self._log_viewer.activateWindow()

    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec():
            self._config = load_config()
            self.settings_panel.reload_models()
            self._start_codec_probe()
            Toast.show(self, "Settings saved", "success")

    def _show_about_dialog(self) -> None:
        QMessageBox.information(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME}\n\nPySide6 GUI rebuild for SeedVR2 processing and export.",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            if self._worker is not None:
                self._worker.request_abort()
            self.preview_widget.cleanup()
            self.trim_timeline.cleanup()
            self.project_panel.cleanup()
        except Exception:
            pass
        super().closeEvent(event)


def _find_app_icon() -> QIcon:
    candidates = [
        Path(_APP_DIR) / "assets" / "icon.ico",
        Path(_APP_DIR) / "assets" / "icon.png",
        Path(_APP_DIR) / "assets" / "logo.png",
        Path(_APP_ROOT) / "icon.ico",
        Path(_APP_ROOT) / "icon.png",
    ]
    for path in candidates:
        if path.exists():
            icon = QIcon(str(path))
            if not icon.isNull():
                return icon
    return QIcon()


def main() -> int:
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    if platform.system() == "Windows":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "naxci1.seedvr.upscaler.25"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("SeedVR2")
    icon = _find_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    app.setStyleSheet(generate_stylesheet())

    window = MainWindow()
    if not icon.isNull():
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
