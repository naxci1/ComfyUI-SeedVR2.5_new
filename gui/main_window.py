"""
SeedVR2 GUI – Main Window
Topaz-style dark-mode wrapper around inference_cli.py.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from gui.styles import DARK_STYLESHEET
    from gui.worker import create_worker_thread, resolve_paths
except ImportError:
    from styles import DARK_STYLESHEET  # type: ignore[no-redef]
    from worker import create_worker_thread, resolve_paths  # type: ignore[no-redef]


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


def _make_group(title: str) -> tuple[QGroupBox, QFormLayout]:
    """Create a titled QGroupBox with an inner QFormLayout."""
    box = QGroupBox(title)
    layout = QFormLayout()
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    layout.setHorizontalSpacing(12)
    layout.setVerticalSpacing(6)
    layout.setContentsMargins(10, 6, 10, 10)
    box.setLayout(layout)
    return box, layout


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window for SeedVR2 GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SeedVR2 Upscaler")
        self.resize(1280, 860)

        self._thread = None
        self._worker = None

        self._build_ui()
        self._set_running(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(6)

        # ── 1. Header ──────────────────────────────────────────────────
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)

        title_lbl = QLabel("SeedVR2 Upscaler")
        title_lbl.setObjectName("header_label")
        sub_lbl = QLabel("Powered by SeedVR2 Diffusion Models")
        sub_lbl.setObjectName("subheader_label")

        header_layout.addWidget(title_lbl)
        header_layout.addWidget(sub_lbl)
        root_layout.addWidget(header_widget)

        # ── 2. Main splitter ───────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([460, 560])

        # ── 3. Bottom controls bar ─────────────────────────────────────
        root_layout.addWidget(self._build_bottom_bar())

    # ── Left panel ─────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(8)

        # Preview group
        preview_group = QGroupBox("Preview")
        pg_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel("No video selected")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(220)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview_label.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 4px; color: #555555;"
        )
        pg_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)

        # Input file
        input_box, input_form = _make_group("Input / Output")

        self.seedvr2_folder_edit = QLineEdit()
        self.seedvr2_folder_edit.setPlaceholderText("Folder containing inference_cli.py…")
        browse_seedvr2_btn = QPushButton("Browse…")
        browse_seedvr2_btn.clicked.connect(self._browse_seedvr2_folder)
        seedvr2_row = QHBoxLayout()
        seedvr2_row.addWidget(self.seedvr2_folder_edit)
        seedvr2_row.addWidget(browse_seedvr2_btn)
        input_form.addRow("SeedVR2 Folder:", _wrap(seedvr2_row))

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Path to video, image, or directory…")
        browse_input_btn = QPushButton("Browse…")
        browse_input_btn.clicked.connect(self._browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit)
        input_row.addWidget(browse_input_btn)
        input_form.addRow("Input File:", _wrap(input_row))

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Optional – leave blank for auto")
        browse_output_btn = QPushButton("Browse…")
        browse_output_btn.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(browse_output_btn)
        input_form.addRow("Output Path:", _wrap(output_row))

        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("Optional – defaults to models/SEEDVR2/")
        browse_model_dir_btn = QPushButton("Browse…")
        browse_model_dir_btn.clicked.connect(self._browse_model_dir)
        model_dir_row = QHBoxLayout()
        model_dir_row.addWidget(self.model_dir_edit)
        model_dir_row.addWidget(browse_model_dir_btn)
        input_form.addRow("Model Directory:", _wrap(model_dir_row))

        layout.addWidget(input_box)
        layout.addStretch(1)
        return panel

    # ── Right panel ────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setMinimumWidth(360)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(4, 0, 0, 0)
        container_layout.setSpacing(8)

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
        self.dit_model_combo.setCurrentIndex(0)
        f.addRow("DiT Model:", self.dit_model_combo)
        container_layout.addWidget(g)

        # ── Output Settings ────────────────────────────────────────────
        g, f = _make_group("Output Settings")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["Auto-detect", "mp4", "png"])
        f.addRow("Output Format:", self.output_format_combo)

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
        self.resolution_spin.setValue(1080)
        f.addRow("Resolution:", self.resolution_spin)

        self.max_resolution_spin = QSpinBox()
        self.max_resolution_spin.setRange(0, 7680)
        self.max_resolution_spin.setValue(0)
        self.max_resolution_spin.setToolTip("0 = no limit")
        f.addRow("Max Resolution:", self.max_resolution_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(5)
        f.addRow("Batch Size:", self.batch_size_spin)

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
        self.seed_spin.setValue(42)
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
        self.cuda_device_edit = QLineEdit()
        self.cuda_device_edit.setPlaceholderText('e.g. "0"  or  "0,1,2"')
        f.addRow("CUDA Device:", self.cuda_device_edit)

        self.dit_offload_combo = QComboBox()
        self.dit_offload_combo.addItems(["none", "cpu"])
        f.addRow("DiT Offload:", self.dit_offload_combo)

        self.vae_offload_combo = QComboBox()
        self.vae_offload_combo.addItems(["none", "cpu"])
        f.addRow("VAE Offload:", self.vae_offload_combo)

        self.tensor_offload_combo = QComboBox()
        self.tensor_offload_combo.addItems(["cpu", "none"])
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

        container_layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    # ── Bottom bar ─────────────────────────────────────────────────────

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        # Row 1 – Progress + status
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        prog_row.addWidget(self.progress_bar, stretch=1)
        prog_row.addWidget(self.status_label)
        layout.addLayout(prog_row)

        # Row 3 – Run / Abort / Clear
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setObjectName("primary_button")
        self.run_btn.clicked.connect(self._run)

        self.abort_btn = QPushButton("⏹  Abort")
        self.abort_btn.setObjectName("danger_button")
        self.abort_btn.clicked.connect(self._abort)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(lambda: self.console.clear())

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.abort_btn)
        btn_row.addStretch(1)
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
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_seedvr2_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select SeedVR2 Folder (containing inference_cli.py)", ""
        )
        if path:
            self.seedvr2_folder_edit.setText(path)

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            "",
            "Videos & Images (*.mp4 *.avi *.mov *.mkv *.webm *.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)",
        )
        if path:
            self.input_edit.setText(path)
            self._load_preview(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if path:
            self.output_edit.setText(path)

    def _browse_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory", "")
        if path:
            self.model_dir_edit.setText(path)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _load_preview(self, path: str) -> None:
        if not _CV2_AVAILABLE:
            self.preview_label.setText("cv2 not available – no preview")
            return

        try:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                self.preview_label.setText("Could not read frame")
                return

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(pixmap)
        except Exception as exc:
            self.preview_label.setText(f"Preview error: {exc}")

    # ------------------------------------------------------------------
    # Argument builder
    # ------------------------------------------------------------------

    def _build_args(self) -> list[str]:
        args: list[str] = []

        # positional input
        inp = self.input_edit.text().strip()
        args.append(inp)

        # output
        out = self.output_edit.text().strip()
        if out:
            args += ["--output", out]

        # model dir
        md = self.model_dir_edit.text().strip()
        if md:
            args += ["--model_dir", md]

        # output format
        fmt = self.output_format_combo.currentText()
        if fmt != "Auto-detect":
            args += ["--output_format", fmt]

        # video backend
        vb = self.video_backend_combo.currentText()
        if vb != "opencv":
            args += ["--video_backend", vb]

        # 10-bit
        if self.use_10bit_check.isChecked():
            args.append("--10bit")

        # dit model
        args += ["--dit_model", self.dit_model_combo.currentText()]

        # resolution
        res = self.resolution_spin.value()
        if res != 1080:
            args += ["--resolution", str(res)]

        max_res = self.max_resolution_spin.value()
        if max_res != 0:
            args += ["--max_resolution", str(max_res)]

        batch = self.batch_size_spin.value()
        if batch != 5:
            args += ["--batch_size", str(batch)]

        if self.uniform_batch_check.isChecked():
            args.append("--uniform_batch_size")

        seed = self.seed_spin.value()
        if seed != 42:
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

        # device
        cuda_dev = self.cuda_device_edit.text().strip()
        if cuda_dev:
            args += ["--cuda_device", cuda_dev]

        dit_offload = self.dit_offload_combo.currentText()
        if dit_offload != "none":
            args += ["--dit_offload_device", dit_offload]

        vae_offload = self.vae_offload_combo.currentText()
        if vae_offload != "none":
            args += ["--vae_offload_device", vae_offload]

        tensor_offload = self.tensor_offload_combo.currentText()
        if tensor_offload != "cpu":
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
        inp = self.input_edit.text().strip()
        if not inp:
            self._on_log("❌  Please specify an input file or directory.")
            return

        seedvr2_folder = self.seedvr2_folder_edit.text().strip()
        python_exe, cli_script = resolve_paths(seedvr2_folder)

        if not os.path.isfile(cli_script):
            self._on_log(f"❌  inference_cli.py not found at: {cli_script}")
            if not hasattr(sys, "_MEIPASS"):
                self._on_log("    Please select your SeedVR2 installation folder.")
            return

        if not os.path.isfile(python_exe):
            self._on_log(f"❌  Python executable not found: {python_exe}")
            if not hasattr(sys, "_MEIPASS"):
                self._on_log("    Please check your SeedVR2 installation folder.")
            return

        args = self._build_args()

        self._thread, self._worker = create_worker_thread(cli_script, args, python_exe)
        self._worker.log_line.connect(self._on_log)
        self._worker.progress_update.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.started_signal.connect(lambda: self._set_running(True))

        self._set_running(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting…")
        self._thread.start()

    def _abort(self) -> None:
        if self._worker:
            self._worker.request_abort()
        self.status_label.setText("Aborting…")

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_log(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{ts}] {line}")

    def _on_progress(self, cur: int, tot: int) -> None:
        if tot > 0:
            pct = int(cur / tot * 100)
            self.progress_bar.setValue(pct)
            self.status_label.setText(f"Processing {cur}/{tot}")

    def _on_finished(self, success: bool, msg: str) -> None:
        self._set_running(False)
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText(f"✅  {msg}")
        else:
            self.status_label.setText(f"⚠  {msg}")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.abort_btn.setEnabled(running)


# ---------------------------------------------------------------------------
# Helpers (module-level, used inside this file only)
# ---------------------------------------------------------------------------

def _wrap(layout: QHBoxLayout) -> QWidget:
    """Wrap a QHBoxLayout in a plain QWidget so it can be added to a QFormLayout."""
    w = QWidget()
    w.setLayout(layout)
    return w
