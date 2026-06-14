#!/usr/bin/env python3
"""
1Click_SeedVR2.5 — PySide6 GUI (Topaz-style dark video enhancement UI).

Entry point and :class:`MainWindow`.  Run directly::

    python gui/app.py

or as a package::

    python -m gui.app
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Runtime sys.path bootstrap
# ---------------------------------------------------------------------------
# When frozen into an executable (PyInstaller / cx_Freeze) the import system
# may not know about the ``gui`` package or its sibling modules (``theme`` …).
# Explicitly add this script's own directory and its parent (the project root)
# to ``sys.path`` so ``import gui.app`` and the legacy ``import theme`` both
# resolve regardless of how the executable is launched.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_APP_DIR)
for _p in (_APP_ROOT, _APP_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

try:
    from gui.theme import Colors, Dims, Fonts, generate_stylesheet
    from gui.widgets import (
        AnimatedProgressBar,
        Button3D,
        DropZone,
        ExportDialog,
        FrameScrubber,
        ProjectPanel,
        SettingsPanel,
        Toast,
        TrimTimeline,
        VideoPreviewWidget,
    )
    from gui.workers import create_worker_thread, resolve_paths
except ImportError:  # pragma: no cover - direct-script execution fallback
    from theme import Colors, Dims, Fonts, generate_stylesheet  # type: ignore
    from widgets import (  # type: ignore
        AnimatedProgressBar,
        Button3D,
        DropZone,
        ExportDialog,
        FrameScrubber,
        ProjectPanel,
        SettingsPanel,
        Toast,
        TrimTimeline,
        VideoPreviewWidget,
    )
    from workers import create_worker_thread, resolve_paths  # type: ignore


# ---------------------------------------------------------------------------
# Hardware detection (background thread, never blocks the GUI)
# ---------------------------------------------------------------------------

class _HardwareProbe(QObject):
    detected = Signal(str, str)  # gpu_name, vram_text

    def run(self) -> None:
        name, vram = "CPU", "—"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                total = torch.cuda.get_device_properties(idx).total_memory
                vram = f"{total / (1024 ** 3):.1f} GB"
        except Exception:
            pass
        self.detected.emit(name, vram)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "muted")
    return lbl


class MainWindow(QMainWindow):
    """Three-column main window matching the Topaz Video AI layout."""

    export_requested = Signal(dict)
    preview_requested = Signal(dict)
    cancel_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("1Click SeedVR2.5")
        self.setMinimumSize(1100, 750)
        self.resize(1280, 850)

        self._current_file: str = ""
        self._thread: Optional[QThread] = None
        self._worker = None
        self._processing = False

        self._build_ui()
        self._connect_signals()
        self._start_hardware_probe()

        # Self-wire the high-level signals to the worker pipeline so the app is
        # usable standalone while keeping the decoupled signal contract.
        self.export_requested.connect(self._spawn_worker)
        self.preview_requested.connect(self._spawn_worker)
        self.cancel_requested.connect(self._abort_worker)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left panel.
        self.project_panel = ProjectPanel()
        root.addWidget(self.project_panel)

        # Center column.
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_MD)
        center_layout.setSpacing(Dims.PADDING_MD)
        root.addWidget(center, 1)

        # Stacked: drop zone (no file) vs. preview (file loaded).
        self.center_stack = QStackedWidget()
        self.drop_zone = DropZone()
        self.preview = VideoPreviewWidget()
        self.center_stack.addWidget(self.drop_zone)
        self.center_stack.addWidget(self.preview)
        center_layout.addWidget(self.center_stack, 1)

        # Frame scrubber.
        self.scrubber = FrameScrubber()
        center_layout.addWidget(self.scrubber)

        # Trim timeline.
        self.trim_timeline = TrimTimeline()
        center_layout.addWidget(self.trim_timeline)

        # Trim info bar.
        center_layout.addLayout(self._build_trim_info_bar())

        # Playback bar.
        center_layout.addLayout(self._build_playback_bar())

        # Action bar.
        center_layout.addLayout(self._build_action_bar())

        # Right panel.
        self.settings_panel = SettingsPanel()
        root.addWidget(self.settings_panel)

        self._build_status_bar()

    def _build_trim_info_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.in_label = QLabel("IN 00:00:00")
        self.in_label.setStyleSheet(f"color: {Colors.TRIM_HANDLE_IN};")
        self.out_label = QLabel("OUT 00:00:00")
        self.out_label.setStyleSheet(f"color: {Colors.TRIM_HANDLE_OUT};")
        self.duration_label = QLabel("Duration 00:00:00 (0 frames)")
        self.duration_label.setStyleSheet(f"color: {Colors.TEXT_ACCENT};")

        self.set_in_btn = Button3D("Set In [I]", variant="default")
        self.set_out_btn = Button3D("Set Out [O]", variant="default")
        self.full_range_btn = Button3D("Full Range", variant="ghost")

        bar.addWidget(self.in_label)
        bar.addWidget(self.duration_label)
        bar.addWidget(self.out_label)
        bar.addStretch(1)
        bar.addWidget(self.set_in_btn)
        bar.addWidget(self.set_out_btn)
        bar.addWidget(self.full_range_btn)
        return bar

    def _build_playback_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.prev_btn = Button3D("◀◀", variant="ghost")
        self.play_btn = Button3D("▶", variant="ghost")
        self.next_btn = Button3D("▶▶", variant="ghost")
        self.timecode_label = QLabel("00:00:00")
        self.timecode_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-family: '{Fonts.FAMILY_MONO}';"
        )
        self.preview5s_btn = Button3D("▶ Preview 5s", variant="default")

        for b in (self.prev_btn, self.play_btn, self.next_btn):
            b.setFixedWidth(48)
        bar.addWidget(self.prev_btn)
        bar.addWidget(self.play_btn)
        bar.addWidget(self.next_btn)
        bar.addWidget(self.timecode_label)
        bar.addStretch(1)
        bar.addWidget(self.preview5s_btn)
        return bar

    def _build_action_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.export_btn = Button3D("Export", variant="primary")
        self.export_btn.setMinimumHeight(Dims.BUTTON_HEIGHT_LG)
        self.cancel_btn = Button3D("Cancel", variant="danger")
        self.cancel_btn.setEnabled(False)
        self.progress = AnimatedProgressBar()

        bar.addWidget(self.export_btn)
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.cancel_btn)
        return bar

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)

        self.queue_label = QLabel("Preview Queue  •  Sources  •  Export Queue")
        self.queue_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        status.addWidget(self.queue_label)

        self.gpu_label = QLabel("GPU: detecting…")
        self.vram_label = QLabel("VRAM: —")
        self.status_text = QLabel("Ready")
        for lbl in (self.gpu_label, self.vram_label, self.status_text):
            lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
            status.addPermanentWidget(lbl)

    # ------------------------------------------------------------------ wiring
    def _connect_signals(self) -> None:
        self.project_panel.file_selected.connect(self.load_file)
        self.drop_zone.file_dropped.connect(self._on_dropped)

        self.trim_timeline.in_point_changed.connect(self._on_trim_changed)
        self.trim_timeline.out_point_changed.connect(self._on_trim_changed)
        self.trim_timeline.playhead_moved.connect(self._on_playhead_moved)

        self.scrubber.frame_changed.connect(self.preview.seek_frame)
        self.preview.frame_changed.connect(self._on_preview_frame)

        self.prev_btn.clicked.connect(self.preview.step_backward)
        self.next_btn.clicked.connect(self.preview.step_forward)
        self.play_btn.clicked.connect(self.preview.toggle_play)

        self.set_in_btn.clicked.connect(
            lambda: self.trim_timeline.set_in_point(self.preview.current_frame())
        )
        self.set_out_btn.clicked.connect(
            lambda: self.trim_timeline.set_out_point(self.preview.current_frame())
        )
        self.full_range_btn.clicked.connect(self.trim_timeline.set_full_range)

        self.preview5s_btn.clicked.connect(self._on_preview5s)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)

        self.settings_panel.settings_changed.connect(self.settings_panel.update_vram_estimate)

    def _start_hardware_probe(self) -> None:
        self._hw_thread = QThread()
        self._hw_probe = _HardwareProbe()
        self._hw_probe.moveToThread(self._hw_thread)
        self._hw_thread.started.connect(self._hw_probe.run)
        self._hw_probe.detected.connect(self._on_hardware_detected)
        self._hw_probe.detected.connect(self._hw_thread.quit)
        self._hw_thread.start()

    def _on_hardware_detected(self, name: str, vram: str) -> None:
        self.gpu_label.setText(f"GPU: {name}")
        self.vram_label.setText(f"VRAM: {vram}")

    # ------------------------------------------------------------------ file
    def _on_dropped(self, path: str) -> None:
        self.project_panel.add_file(path)
        self.load_file(path)

    def load_file(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            return
        self._current_file = path
        self.preview.load_file(path)
        self.trim_timeline.load_video(path)
        self.scrubber.set_frame_count(self.preview.get_frame_count())
        self.center_stack.setCurrentWidget(self.preview)
        self._update_trim_labels()
        self.status_text.setText(f"Loaded {os.path.basename(path)}")

    # ------------------------------------------------------------------ trim
    def _on_trim_changed(self, _frame: int) -> None:
        self._update_trim_labels()

    def _on_playhead_moved(self, frame: int) -> None:
        self.preview.seek_frame(frame)
        self.scrubber.set_frame(frame)
        self.timecode_label.setText(self.trim_timeline.frame_to_timecode(frame))

    def _on_preview_frame(self, frame: int) -> None:
        self.scrubber.set_frame(frame)
        self.timecode_label.setText(self.trim_timeline.frame_to_timecode(frame))

    def _update_trim_labels(self) -> None:
        in_f, out_f = self.trim_timeline.get_selected_range()
        self.in_label.setText(f"IN {self.trim_timeline.frame_to_timecode(in_f)}")
        self.out_label.setText(f"OUT {self.trim_timeline.frame_to_timecode(out_f)}")
        count = self.trim_timeline.get_selected_frame_count()
        dur = self.trim_timeline.frame_to_timecode(count)
        self.duration_label.setText(f"Duration {dur} ({count} frames)")

    # ------------------------------------------------------------------ run
    def _on_preview5s(self) -> None:
        if not self._current_file:
            Toast.show(self, "Import a file first", "warning")
            return
        settings = self.settings_panel.get_all_settings()
        fps = self.preview.get_fps() or 25.0
        start = self.preview.current_frame()
        payload = {
            "mode": "preview",
            "input": self._current_file,
            "settings": settings,
            "start_frame": start,
            "frame_cap": int(fps * 5),
        }
        self.preview_requested.emit(payload)

    def _on_export_clicked(self) -> None:
        if not self._current_file:
            Toast.show(self, "Import a file first", "warning")
            return
        in_f, out_f = self.trim_timeline.get_selected_range()
        # Treat 0 -> last_frame as "full video" (no trim).
        if self.trim_timeline.is_full_range():
            in_f, out_f = 0, 0
        default_dir = os.path.dirname(self._current_file)
        default_name = f"seedvr2_{os.path.splitext(os.path.basename(self._current_file))[0]}"

        dialog = ExportDialog(
            self,
            default_dir=default_dir,
            default_name=default_name,
            frame_count=self.preview.get_frame_count(),
            width=self.preview.get_frame_width(),
            height=self.preview.get_frame_height(),
            fps=self.preview.get_fps(),
            trim_in=in_f,
            trim_out=out_f,
        )
        dialog.export_confirmed.connect(self._on_export_confirmed)
        dialog.exec()

    def _on_export_confirmed(self, export_settings: dict) -> None:
        payload = {
            "mode": "export",
            "input": self._current_file,
            "settings": self.settings_panel.get_all_settings(),
            "export": export_settings,
        }
        self.export_requested.emit(payload)

    # ------------------------------------------------------------------ worker
    def _build_cli_args(self, payload: dict) -> list:
        s = payload.get("settings", {})
        args: list = [payload["input"]]

        export = payload.get("export", {})
        if payload.get("mode") == "export" and export.get("output_path"):
            args += ["--output", export["output_path"]]
            if export.get("output_type") == "image_sequence":
                args += ["--output_format", export.get("image_format", "tiff").lower()]
            else:
                args += ["--output_format", export.get("container", "mp4").lower()]

        # Resolution.
        if s.get("resolution_mode") == "X-Times":
            scale = {"50%": 1, "75%": 1, "100%": 1, "150%": 2, "200%": 2,
                     "300%": 3, "400%": 4}.get(s.get("scale", "100%"), 2)
            args += ["--resolution_mode", "xtimes", "--resolution_scale", str(scale)]
        else:
            short = {"480p": 480, "720p": 720, "1080p": 1080,
                     "1440p": 1440, "2K": 1440, "4K": 2160}.get(s.get("target_short_side", "1080p"), 1080)
            args += ["--resolution", str(short)]

        if s.get("pre_downscale", "").startswith("2"):
            args += ["--pre_downscale", "2"]

        args += ["--batch_size", str(s.get("batch_size", 1))]
        if s.get("uniform_batch_size"):
            args.append("--uniform_batch_size")
        args += ["--seed", str(s.get("seed", 313))]

        if s.get("temporal_overlap"):
            args += ["--temporal_overlap", str(s["temporal_overlap"])]

        cc = s.get("color_correction", "None").lower()
        if cc and cc != "none":
            args += ["--color_correction", cc]

        attn = {"Auto Best": "sdpa", "SDPA Safe": "sdpa",
                "Flash Attn 2": "flash_attn", "Flash Attn 3": "flash_attn"}.get(
            s.get("attention", "Auto Best"), "sdpa")
        if attn != "sdpa":
            args += ["--attention_mode", attn]

        args += ["--dit_model", s.get("dit_model", "SeedVR2 3B Q8")]

        if s.get("ten_bit_output"):
            args.append("--10bit")
        if s.get("cache_dit"):
            args.append("--cache_dit")
        if s.get("cache_vae"):
            args.append("--cache_vae")
        if s.get("auto_tune"):
            args.append("--auto_tune")
        if s.get("debug_mode"):
            args.append("--debug")

        if s.get("enable_tiling"):
            args.append("--vae_encode_tiled")
            args += ["--vae_encode_tile_size", str(s.get("encode_tile_size", 1024))]
            args += ["--vae_encode_tile_overlap", str(s.get("encode_overlap", 64))]
            args.append("--vae_decode_tiled")
            args += ["--vae_decode_tile_size", str(s.get("decode_tile_size", 1024))]
            args += ["--vae_decode_tile_overlap", str(s.get("decode_overlap", 64))]

        if s.get("input_noise"):
            args += ["--input_noise_scale", f"{s['input_noise'] / 100.0:.2f}"]
        if s.get("latent_noise"):
            args += ["--latent_noise_scale", f"{s['latent_noise'] / 100.0:.2f}"]

        # Preview: cap frames + start offset.
        if payload.get("mode") == "preview":
            if payload.get("start_frame"):
                args += ["--skip_first_frames", str(payload["start_frame"])]
            args += ["--load_cap", str(payload.get("frame_cap", 120))]

        return args

    def _spawn_worker(self, payload: dict) -> None:
        if self._processing:
            Toast.show(self, "A job is already running", "warning")
            return
        python_exe, cli_script = resolve_paths()
        if not os.path.isfile(cli_script):
            Toast.show(self, "inference_cli.py not found", "error")
            return
        args = self._build_cli_args(payload)
        self._thread, self._worker = create_worker_thread(cli_script, args, python_exe)
        self._worker.log_line.connect(self._on_log)
        self._worker.progress_update.connect(self._on_progress)
        self._worker.finished.connect(
            lambda ok, msg: self.on_export_finished(ok, msg)
        )
        self._worker.started_signal.connect(lambda: self._set_processing(True))
        self.progress.reset()
        self._thread.start()

    def _abort_worker(self) -> None:
        if self._worker is not None:
            self._worker.request_abort()
            self.status_text.setText("Cancelling…")

    def _on_log(self, line: str) -> None:
        # Surface the latest log line in the status bar.
        self.status_text.setText(line.strip()[:80])

    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.update_progress(100.0 * current / total)

    def _set_processing(self, processing: bool) -> None:
        self._processing = processing
        self.settings_panel.set_enabled_state(not processing)
        self.export_btn.setEnabled(not processing)
        self.preview5s_btn.setEnabled(not processing)
        self.cancel_btn.setEnabled(processing)
        self.status_text.setText("Processing…" if processing else "Ready")

    # ------------------------------------------------------------------ public callbacks
    def update_progress(self, value: float, eta: str = "", status: str = "") -> None:
        self.progress.setValue(value, eta)
        if status:
            self.status_text.setText(status)

    def on_export_finished(self, success: bool, message: str) -> None:
        self._set_processing(False)
        if success:
            self.update_progress(100.0)
            self.status_text.setText("Export complete")
            Toast.show(self, "Export complete", "success")
        else:
            self.status_text.setText(message)
            Toast.show(self, message or "Export failed", "error")
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(500)
        self._thread = None
        self._worker = None

    def on_preview_ready(self, frames_tensor) -> None:
        """Display preview output frames in the preview widget."""
        try:
            import numpy as np  # type: ignore
            from PySide6.QtGui import QImage, QPixmap

            frame = frames_tensor
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            h, w = frame.shape[:2]
            img = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.preview.set_pixmap(QPixmap.fromImage(img))
        except Exception:
            pass

    # ------------------------------------------------------------------ close
    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.preview.cleanup()
            self.trim_timeline.cleanup()
            self.project_panel.cleanup()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    if platform.system() == "Windows":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "naxci1.seedvr.upscaler.25"
            )
        except Exception:
            pass

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # MUST be set before the stylesheet.
    app.setApplicationName("1Click SeedVR2.5")
    app.setOrganizationName("SeedVR2")
    app.setStyleSheet(generate_stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
