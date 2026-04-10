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
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
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
# GPU auto-detection
# ---------------------------------------------------------------------------

# Populated by _detect_gpus(); read by MainWindow to emit a startup console note.
_GPU_INIT_MSG: str = ""


def _detect_gpus() -> list[str]:
    """Return a list of GPU entries suitable for a QComboBox.

    Format: ``["Auto", "CPU", "GPU 0: NVIDIA GeForce RTX 5070 Ti", "GPU 1: …", …]``
    Falls back to ``["Auto", "CPU"]`` when torch is unavailable or no CUDA GPUs exist.
    Side-effect: sets the module-level ``_GPU_INIT_MSG`` for console display.
    """
    global _GPU_INIT_MSG  # noqa: PLW0603
    entries = ["Auto", "CPU"]
    try:
        import torch  # noqa: PLC0415
        # Explicitly initialise CUDA so device names are always resolvable.
        try:
            torch.cuda.init()
        except Exception:
            pass
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count == 0:
                _GPU_INIT_MSG = (
                    "⚠  torch.cuda.is_available() returned True but "
                    "device_count() is 0 – no CUDA GPUs detected."
                )
            else:
                for i in range(count):
                    entries.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                _GPU_INIT_MSG = f"✅  Detected {count} CUDA device(s)."
        else:
            _GPU_INIT_MSG = (
                "⚠  torch.cuda.is_available() returned False – "
                "CUDA is not accessible from this Python environment.  "
                "GPU inference requires the SeedVR2 Python (python_embeded)."
            )
    except ImportError:
        _GPU_INIT_MSG = (
            "ℹ  torch is not installed in the GUI Python environment – "
            "GPU list limited to Auto / CPU."
        )
    except Exception as exc:
        _GPU_INIT_MSG = f"⚠  GPU scan error: {exc}"
    return entries

try:
    from gui.styles import DARK_STYLESHEET
    from gui.worker import create_worker_thread, resolve_paths, DEFAULT_PYTHON_EXE
except ImportError:
    from styles import DARK_STYLESHEET  # type: ignore[no-redef]
    from worker import create_worker_thread, resolve_paths, DEFAULT_PYTHON_EXE  # type: ignore[no-redef]


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
        self.setWindowTitle("SeedVR2.5 Upscaler by HB2k v.1.3 beta")
        self.resize(1400, 960)

        # Create settings window first – it loads saved paths in its __init__
        self._settings_win = SettingsWindow(self)
        self._settings_win.input_changed.connect(self._load_preview)

        self._thread = None
        self._worker = None

        self._build_ui()

        # Load window icon – try .ico first (Windows), fall back to .png
        for _icon_rel in ("icon.ico", "assets/icon.ico", "assets/icon.png"):
            _icon = _resource_path(_icon_rel)
            if os.path.isfile(_icon):
                self.setWindowIcon(QIcon(_icon))
                break

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

        # ── 1. Header ──────────────────────────────────────────────────
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        title_col = QWidget()
        title_vlayout = QVBoxLayout(title_col)
        title_vlayout.setContentsMargins(0, 0, 0, 0)
        title_vlayout.setSpacing(2)
        title_lbl = QLabel("SeedVR2.5 Upscaler by HB2k v.1.3 beta")
        title_lbl.setObjectName("header_label")
        sub_lbl = QLabel("Powered by SeedVR2 Diffusion Models")
        sub_lbl.setObjectName("subheader_label")
        title_vlayout.addWidget(title_lbl)
        title_vlayout.addWidget(sub_lbl)

        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setToolTip("Open Paths & Configuration settings")
        settings_btn.setMinimumWidth(110)
        settings_btn.clicked.connect(self._open_settings)

        header_layout.addWidget(title_col, stretch=1)
        header_layout.addWidget(settings_btn)
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

        # "📂 Open" – quick input-file picker aligned to the far right of the mode bar
        self._open_input_btn = QPushButton("📂 Open")
        self._open_input_btn.setToolTip("Open an input video or image file")
        self._open_input_btn.setEnabled(_MULTIMEDIA_AVAILABLE)
        self._open_input_btn.clicked.connect(self._browse_input_for_player)
        mode_bar.addWidget(self._open_input_btn)

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
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems([
            "Auto-detect",
            # ── Video codecs ──────────────────────
            "H.264 / mp4 (AVC)",
            "H.265 / mp4 (HEVC)",
            "AV1 / mp4",
            # ── Image sequences ───────────────────
            "PNG",
            "JPG",
            "WEBP",
            "TIFF",
        ])
        self.output_format_combo.setToolTip(
            "H.265 automatically enables the --10bit flag for x265 encoding.\n"
            "JPG / WEBP / TIFF map to 'png' container in the current CLI."
        )
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
        self.gpu_device_combo = QComboBox()
        self.gpu_device_combo.addItems(_detect_gpus())
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
    # Settings window
    # ------------------------------------------------------------------

    def _open_settings(self) -> None:
        """Show (or raise) the persistent Paths & Configuration window."""
        self._settings_win.show()
        self._settings_win.raise_()
        self._settings_win.activateWindow()

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

        # output format + codec mapping
        # Maps UI display name → (--output_format value, force_10bit)
        # JPG/WEBP/TIFF map to "png" because the current CLI only accepts "mp4"/"png".
        _FMT_MAP: dict[str, tuple[str, bool]] = {
            "Auto-detect":       ("",    False),
            "H.264 / mp4 (AVC)": ("mp4", False),
            "H.265 / mp4 (HEVC)": ("mp4", True),   # force --10bit for x265
            "AV1 / mp4":         ("mp4", False),
            "PNG":               ("png", False),
            "JPG":               ("png", False),
            "WEBP":              ("png", False),
            "TIFF":              ("png", False),
        }
        fmt_val, fmt_10bit = _FMT_MAP.get(self.output_format_combo.currentText(), ("", False))
        if fmt_val:
            args += ["--output_format", fmt_val]
        # H.265 forces 10-bit encoding; only append here if the checkbox hasn't
        # already done so (avoids duplicate --10bit flags).
        if fmt_10bit and not self.use_10bit_check.isChecked():
            args.append("--10bit")

        # video backend
        vb = self.video_backend_combo.currentText()
        if vb != "opencv":
            args += ["--video_backend", vb]

        # 10-bit
        if self.use_10bit_check.isChecked():
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

        # device
        gpu_sel = self.gpu_device_combo.currentText()
        if gpu_sel == "CPU":
            cuda_dev = "cpu"
        elif gpu_sel == "Auto":
            cuda_dev = "0"
        else:
            # Format is "GPU 0: NVIDIA GeForce RTX 5070 Ti" – extract the numeric index
            # "GPU 0" is before the colon; split on space to get "0"
            cuda_dev = gpu_sel.split(":")[0].split()[-1].strip()
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
            self._input_player.setVideoOutput(self._split_view.input_sink)
            self._output_player.setVideoOutput(self._split_view.output_sink)
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
            "Videos & Images "
            "(*.mp4 *.avi *.mov *.mkv *.webm *.png *.jpg *.jpeg *.bmp *.tiff)"
            ";;All Files (*)",
        )
        if path:
            self._settings_win.input_edit.setText(path)
            self._load_preview(path)
            self._mode_input_btn.setChecked(True)

    def _try_auto_load_output(self) -> None:
        if not _MULTIMEDIA_AVAILABLE:
            return
        out = self._settings_win.output_edit.text().strip()
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

        minus_btn = QPushButton("-")
        minus_btn.setFixedWidth(34)
        minus_btn.setFont(_btn_font)
        minus_btn.setToolTip("Decrease batch size by 4")
        minus_btn.setStyleSheet(
            "QPushButton { background-color: #222222; border: 1px solid #444; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2e2e2e; }"
            "QPushButton:pressed { background-color: #1a1a1a; }"
        )

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
        plus_btn.setStyleSheet(
            "QPushButton { background-color: #222222; border: 1px solid #444; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2e2e2e; }"
            "QPushButton:pressed { background-color: #1a1a1a; }"
        )

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
        self.abort_btn.setEnabled(running)


# ---------------------------------------------------------------------------
# Helpers (module-level, used inside this file only)
# ---------------------------------------------------------------------------

def _wrap(layout: QHBoxLayout) -> QWidget:
    """Wrap a QHBoxLayout in a plain QWidget so it can be added to a QFormLayout."""
    w = QWidget()
    w.setLayout(layout)
    return w
