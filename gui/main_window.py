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

from PyQt6.QtCore import Qt, QSettings, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
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
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    _MULTIMEDIA_AVAILABLE = True
except ImportError:
    _MULTIMEDIA_AVAILABLE = False

# ---------------------------------------------------------------------------
# GPU auto-detection
# ---------------------------------------------------------------------------

def _detect_gpus() -> list[str]:
    """Return a list of GPU entries suitable for a QComboBox.

    Format: ``["Auto", "0: NVIDIA GeForce RTX 5070 Ti", "1: …", …]``
    Falls back to ``["Auto"]`` when torch is unavailable or no CUDA GPUs exist.
    """
    entries = ["Auto"]
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                entries.append(f"{i}: {torch.cuda.get_device_name(i)}")
    except Exception:  # torch not installed in GUI's Python – that's fine
        pass
    return entries

try:
    from gui.styles import DARK_STYLESHEET
    from gui.worker import create_worker_thread, resolve_paths, DEFAULT_PYTHON_EXE
except ImportError:
    from styles import DARK_STYLESHEET  # type: ignore[no-redef]
    from worker import create_worker_thread, resolve_paths, DEFAULT_PYTHON_EXE  # type: ignore[no-redef]


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

    _SETTINGS_ORG = "SeedVR2GUI"
    _SETTINGS_APP = "SeedVR2_GUI"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SeedVR2 Upscaler")
        self.resize(1400, 960)

        self._thread = None
        self._worker = None

        self._build_ui()
        self._load_settings()
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
        splitter.setSizes([580, 620])

        # ── 3. Bottom controls bar ─────────────────────────────────────
        root_layout.addWidget(self._build_bottom_bar())

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

            # page 2 – Split view (QSplitter as the draggable divider)
            split_container = QWidget()
            split_hlayout = QHBoxLayout(split_container)
            split_hlayout.setContentsMargins(0, 0, 0, 0)
            split_hlayout.setSpacing(0)
            self._split_splitter = QSplitter(Qt.Orientation.Horizontal)
            self._split_input_vw = QVideoWidget()
            self._split_output_vw = QVideoWidget()
            self._split_splitter.addWidget(self._split_input_vw)
            self._split_splitter.addWidget(self._split_output_vw)
            self._split_splitter.setSizes([1, 1])
            split_hlayout.addWidget(self._split_splitter)
            self._viewer_stack.addWidget(split_container)

            # Media players
            self._input_player = QMediaPlayer()
            self._input_audio = QAudioOutput()
            self._input_player.setAudioOutput(self._input_audio)
            self._input_player.setVideoOutput(self._solo_input_vw)
            self._input_player.durationChanged.connect(self._on_player_duration)
            self._input_player.positionChanged.connect(self._on_player_position)
            self._input_player.playbackStateChanged.connect(self._on_player_state)

            self._output_player = QMediaPlayer()
            self._output_audio = QAudioOutput()
            self._output_player.setAudioOutput(self._output_audio)
            self._output_player.setVideoOutput(self._solo_output_vw)

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
        scroll.setMinimumWidth(360)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(4, 0, 0, 0)
        container_layout.setSpacing(8)

        # ── Paths & Configuration ──────────────────────────────────────
        g, f = _make_group("Paths & Configuration")

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

        # Input: File / Folder toggle + path
        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItems(["File", "Folder"])
        self.input_mode_combo.setMaximumWidth(72)
        self.input_mode_combo.setToolTip("File: single video/image  |  Folder: batch-process all videos")
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Path to video, image, or directory…")
        browse_input_btn = QPushButton("Browse…")
        browse_input_btn.clicked.connect(self._browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_mode_combo)
        input_row.addWidget(self.input_edit)
        input_row.addWidget(browse_input_btn)
        f.addRow("Input:", _wrap(input_row))

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Optional – leave blank for auto")
        browse_out_btn = QPushButton("Browse…")
        browse_out_btn.clicked.connect(self._browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse_out_btn)
        f.addRow("Output Path:", _wrap(out_row))

        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("Optional – defaults to models/SEEDVR2/")
        browse_md_btn = QPushButton("Browse…")
        browse_md_btn.clicked.connect(self._browse_model_dir)
        fp8_btn = QPushButton("FP8/FP16")
        fp8_btn.setToolTip("Download FP8 / FP16 models from HuggingFace")
        fp8_btn.clicked.connect(self._open_fp8_url)
        gguf_btn = QPushButton("GGUF")
        gguf_btn.setToolTip("Download GGUF models from HuggingFace")
        gguf_btn.clicked.connect(self._open_gguf_url)
        md_row = QHBoxLayout()
        md_row.addWidget(self.model_dir_edit)
        md_row.addWidget(browse_md_btn)
        md_row.addWidget(fp8_btn)
        md_row.addWidget(gguf_btn)
        f.addRow("Model Directory:", _wrap(md_row))

        container_layout.addWidget(g)

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
        self.batch_size_spin.setRange(1, 1001)
        self.batch_size_spin.setValue(5)
        self.batch_size_spin.setSingleStep(4)
        self.batch_size_spin.setToolTip("Must be 4n+1: 1, 5, 9, 13, … (automatically snapped)")
        self.batch_size_spin.valueChanged.connect(self._snap_batch_size)
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
        self.gpu_device_combo = QComboBox()
        self.gpu_device_combo.addItems(_detect_gpus())
        f.addRow("GPU Device:", self.gpu_device_combo)

        self.dit_offload_combo = QComboBox()
        self.dit_offload_combo.addItems(["none", "cpu"])
        f.addRow("DiT Offload:", self.dit_offload_combo)

        self.vae_offload_combo = QComboBox()
        self.vae_offload_combo.addItems(["none", "cpu"])
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
        self.attention_mode_combo.setCurrentIndex(
            self.attention_mode_combo.findText("sageattn_3")
        )
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

        # Row 1 – Batch progress (current segment)
        batch_row = QHBoxLayout()
        batch_lbl = QLabel("Batch:")
        batch_lbl.setMinimumWidth(50)
        batch_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        batch_lbl.setStyleSheet("color:#888; font-size:11px;")
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setMaximumHeight(14)
        self.batch_progress_bar.setToolTip("Current batch / segment progress")
        batch_row.addWidget(batch_lbl)
        batch_row.addWidget(self.batch_progress_bar, stretch=1)
        layout.addLayout(batch_row)

        # Row 2 – Total progress + status
        prog_row = QHBoxLayout()
        total_lbl = QLabel("Total:")
        total_lbl.setMinimumWidth(50)
        total_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        total_lbl.setStyleSheet("color:#888; font-size:11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumHeight(14)
        self.progress_bar.setToolTip("Total video / folder progress")
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        prog_row.addWidget(total_lbl)
        prog_row.addWidget(self.progress_bar, stretch=1)
        prog_row.addWidget(self.status_label)
        layout.addLayout(prog_row)

        # Row 3 – Run / Abort / Copy All / Clear
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setObjectName("primary_button")
        self.run_btn.clicked.connect(self._run)

        self.abort_btn = QPushButton("⏹  Abort")
        self.abort_btn.setObjectName("danger_button")
        self.abort_btn.clicked.connect(self._abort)

        self.copy_log_btn = QPushButton("Copy All")
        self.copy_log_btn.setToolTip("Copy all log output to clipboard")
        self.copy_log_btn.clicked.connect(self._copy_log)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(lambda: self.console.clear())

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.abort_btn)
        btn_row.addStretch(1)
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
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Input File",
                "",
                "Videos & Images (*.mp4 *.avi *.mov *.mkv *.webm *.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)",
            )
        if path:
            self.input_edit.setText(path)
            if self.input_mode_combo.currentText() == "File":
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
    # Preview / video loading
    # ------------------------------------------------------------------

    def _load_preview(self, path: str) -> None:
        """Load *path* into the Input Preview player."""
        self._load_input_video(path)

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
        gpu_sel = self.gpu_device_combo.currentText()
        if gpu_sel == "Auto":
            cuda_dev = "0"
        else:
            # Format is "0: NVIDIA GeForce RTX 5070 Ti" – extract the index
            cuda_dev = gpu_sel.split(":")[0].strip()
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

        python_exe = self.python_exe_edit.text().strip() or DEFAULT_PYTHON_EXE
        seedvr2_folder = self.seedvr2_folder_edit.text().strip()

        if not seedvr2_folder:
            self._on_log("❌  Please select the SeedVR2 folder (containing inference_cli.py).")
            return

        cli_script = str(Path(seedvr2_folder) / "inference_cli.py")
        if not os.path.isfile(cli_script):
            self._on_log(
                f"❌  inference_cli.py not found in: {seedvr2_folder}\n"
                "    Please select the correct SeedVR2 installation folder."
            )
            return

        if not os.path.isfile(python_exe):
            self._on_log(
                f"❌  Python executable not found: {python_exe}\n"
                "    Please check the Python Executable path."
            )
            return

        # Persist the current paths so the user doesn't have to re-enter them
        self._save_settings()

        args = self._build_args()

        self._thread, self._worker = create_worker_thread(cli_script, args, python_exe)
        self._worker.log_line.connect(self._on_log)
        self._worker.progress_update.connect(self._on_progress)
        self._worker.batch_progress_update.connect(self._on_batch_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.started_signal.connect(lambda: self._set_running(True))

        self._set_running(True)
        self.progress_bar.setValue(0)
        self.batch_progress_bar.setValue(0)
        self.status_label.setText("Starting…")
        self._thread.start()

    def _abort(self) -> None:
        if self._worker:
            self._worker.request_abort()
        self.status_label.setText("Aborting…")

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
        if new_mode == self._player_mode:
            return
        self._player_mode = new_mode
        if new_mode == "input":
            self._input_player.setVideoOutput(self._solo_input_vw)
            self._viewer_stack.setCurrentIndex(0)
        elif new_mode == "output":
            self._output_player.setVideoOutput(self._solo_output_vw)
            self._viewer_stack.setCurrentIndex(1)
        else:  # split
            self._input_player.setVideoOutput(self._split_input_vw)
            self._output_player.setVideoOutput(self._split_output_vw)
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
            if self._player_mode == "split":
                self._output_player.setPosition(self._input_player.position())
                self._input_player.play()
                self._output_player.play()
            elif self._player_mode == "output":
                self._output_player.play()
            else:
                self._input_player.play()

    def _on_stop(self) -> None:
        if self._input_player:
            self._input_player.stop()
        if self._output_player:
            self._output_player.stop()

    def _on_seek(self, pos: int) -> None:
        if self._player_mode == "output":
            if self._output_player:
                self._output_player.setPosition(pos)
        else:
            if self._input_player:
                self._input_player.setPosition(pos)
            if self._player_mode == "split" and self._output_player:
                self._output_player.setPosition(pos)

    def _on_player_duration(self, duration: int) -> None:
        self._seek_slider.setRange(0, duration)
        self._update_time_label()

    def _on_player_position(self, position: int) -> None:
        if not self._seek_slider.isSliderDown():
            self._seek_slider.setValue(position)
        self._update_time_label()
        # Keep output player in sync during split mode
        if self._player_mode == "split" and self._output_player:
            if abs(self._output_player.position() - position) > 500:
                self._output_player.setPosition(position)

    def _on_player_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("\u23f8" if playing else "\u25b6")  # ⏸ / ▶

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

    def _browse_output_video(self) -> None:
        start_dir = self.output_edit.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Output Video", start_dir,
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if path:
            self._load_output_video(path)
            # Switch to output view automatically
            self._mode_output_btn.setChecked(True)

    def _try_auto_load_output(self) -> None:
        if not _MULTIMEDIA_AVAILABLE:
            return
        out = self.output_edit.text().strip()
        if not out:
            return
        out_path = Path(out)
        if out_path.is_file():
            self._load_output_video(str(out_path))
            self._mode_output_btn.setChecked(True)
        elif out_path.is_dir():
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            candidates = sorted(
                (f for f in out_path.iterdir() if f.suffix.lower() in video_exts),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                self._load_output_video(str(candidates[0]))
                self._mode_output_btn.setChecked(True)

    # ------------------------------------------------------------------
    # Resource link openers
    # ------------------------------------------------------------------

    def _open_fp8_url(self) -> None:
        QDesktopServices.openUrl(QUrl("https://huggingface.co/numz/SeedVR2_comfyUI/tree/main"))

    def _open_gguf_url(self) -> None:
        QDesktopServices.openUrl(QUrl("https://huggingface.co/AInVFX/SeedVR2_comfyUI/tree/main"))

    # ------------------------------------------------------------------
    # Batch size constraint (4n+1)
    # ------------------------------------------------------------------

    def _snap_batch_size(self, val: int) -> None:
        """Snap the batch_size spinbox to the nearest 4n+1 value (1, 5, 9, …)."""
        if val < 1:
            snapped = 1
        elif (val - 1) % 4 != 0:
            snapped = max(1, 1 + 4 * ((val - 1) // 4))
        else:
            return  # already valid
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

    def _on_progress(self, cur: int, tot: int) -> None:
        if tot > 0:
            pct = int(cur / tot * 100)
            self.progress_bar.setValue(pct)
            self.status_label.setText(f"Processing {cur}/{tot}")

    def _on_batch_progress(self, cur: int, tot: int) -> None:
        if tot > 0:
            self.batch_progress_bar.setValue(int(cur / tot * 100))

    def _on_finished(self, success: bool, msg: str) -> None:
        self._set_running(False)
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText(f"✅  {msg}")
            self._try_auto_load_output()
        else:
            self.status_label.setText(f"⚠  {msg}")

    # ------------------------------------------------------------------
    # Settings persistence (QSettings)
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        """Restore previously saved paths from persistent storage."""
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        self.python_exe_edit.setText(
            s.value("python_exe", DEFAULT_PYTHON_EXE, type=str)
        )
        self.seedvr2_folder_edit.setText(
            s.value("seedvr2_folder", "", type=str)
        )
        self.input_edit.setText(s.value("input_path", "", type=str))
        saved_input_mode: str = s.value("input_mode", "File", type=str)
        idx = self.input_mode_combo.findText(saved_input_mode)
        if idx >= 0:
            self.input_mode_combo.setCurrentIndex(idx)
        self.output_edit.setText(s.value("output_path", "", type=str))
        saved_model_dir: str = s.value("model_dir", "", type=str)
        if saved_model_dir:
            self.model_dir_edit.setText(saved_model_dir)
        elif self.seedvr2_folder_edit.text():
            default_md = str(
                Path(self.seedvr2_folder_edit.text()) / "models" / "SEEDVR2"
            )
            self.model_dir_edit.setPlaceholderText(default_md)

    def _save_settings(self) -> None:
        """Persist current path values to storage."""
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        s.setValue("python_exe", self.python_exe_edit.text().strip())
        s.setValue("seedvr2_folder", self.seedvr2_folder_edit.text().strip())
        s.setValue("input_path", self.input_edit.text().strip())
        s.setValue("input_mode", self.input_mode_combo.currentText())
        s.setValue("output_path", self.output_edit.text().strip())
        s.setValue("model_dir", self.model_dir_edit.text().strip())

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
