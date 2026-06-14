"""Right-side settings panel with 1:1 parity to inference_cli.py arguments."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..theme import Colors, Dims, Fonts
from .toggle_switch import ToggleSwitch

try:
    from gui.config_manager import load_config
except ImportError:  # pragma: no cover - direct-script execution fallback
    from config_manager import load_config  # type: ignore[no-redef]


class _FrameListLineEdit(QLineEdit):
    """Sanitises comma-separated frame indices when focus leaves the field."""

    validated = Signal()

    def focusOutEvent(self, event) -> None:  # noqa: N802
        raw = self.text().strip()
        invalid = False
        if raw:
            cleaned: List[str] = []
            for part in raw.split(","):
                token = part.strip()
                if not token:
                    continue
                if token.isdigit():
                    cleaned.append(str(int(token)))
                else:
                    invalid = True
            self.setText(",".join(cleaned))
        self.setProperty("error", "true" if invalid else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.validated.emit()
        super().focusOutEvent(event)


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setProperty("role", "separator")
    line.setFixedHeight(1)
    line.setStyleSheet(f"background-color: {Colors.BORDER};")
    return line


class SettingsPanel(QWidget):
    """Scrollable CLI-parity settings panel."""

    settings_changed = Signal()

    _PRESET_RESOLUTIONS = {
        "720p (HD)": 720,
        "1080p (FHD)": 1080,
        "2K (1440p)": 1440,
        "4K (2160p)": 2160,
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(max(Dims.PANEL_WIDTH_RIGHT, 280))
        self._model_fallback = [
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_3b-Q8_0.gguf",
            "seedvr2_ema_3b-Q4_K_M.gguf",
            "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b-Q4_K_M.gguf",
            "seedvr2_ema_7b_sharp_fp16.safetensors",
            "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
        ]

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        body = QWidget()
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_MD)
        self._layout.setSpacing(Dims.PADDING_MD)
        scroll.setWidget(body)

        self._build_resolution_group()
        self._build_model_group()
        self._build_runtime_group()
        self._build_batch_group()
        self._build_vae_group()
        self._build_quality_group()
        self._build_device_group()
        self._build_vram_group()
        self._layout.addStretch(1)

        self._vram_timer = QTimer(self)
        self._vram_timer.setSingleShot(True)
        self._vram_timer.setInterval(200)
        self._vram_timer.timeout.connect(self.update_vram_estimate)

        self._update_resolution_mode()
        self._update_vae_controls()
        self.update_vram_estimate()

    def _group(self, title: str) -> QFormLayout:
        box = QGroupBox(title, self)
        form = QFormLayout(box)
        form.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_LG, Dims.PADDING_MD, Dims.PADDING_MD)
        form.setSpacing(Dims.PADDING_SM)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._layout.addWidget(box)
        return form

    def _label(self, text: str) -> QLabel:
        return QLabel(text)

    def _spin(self, minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
        spin = QSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(self._emit_settings_changed)
        return spin

    def _combo(self, items: List[str], current_text: str) -> QComboBox:
        combo = QComboBox(self)
        combo.addItems(items)
        if current_text in items:
            combo.setCurrentText(current_text)
        combo.currentIndexChanged.connect(self._emit_settings_changed)
        return combo

    def _toggle(self, checked: bool) -> ToggleSwitch:
        toggle = ToggleSwitch("", checked, self)
        toggle.toggled.connect(lambda _=False: self._emit_settings_changed())
        return toggle

    def _check(self, checked: bool) -> QCheckBox:
        check = QCheckBox(self)
        check.setChecked(checked)
        check.toggled.connect(lambda _=False: self._emit_settings_changed())
        return check

    def _slider_row(self, value: int) -> tuple[QWidget, QSlider, QLabel]:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_SM)
        slider = QSlider(Qt.Horizontal, container)
        slider.setRange(0, 100)
        slider.setValue(value)
        label = QLabel(f"{value / 100.0:.2f}", container)
        label.setMinimumWidth(40)
        layout.addWidget(slider, 1)
        layout.addWidget(label)
        slider.valueChanged.connect(lambda v, lbl=label: lbl.setText(f"{v / 100.0:.2f}"))
        slider.valueChanged.connect(self._emit_settings_changed)
        return container, slider, label

    def _populate_cuda_devices(self) -> None:
        self.cuda_device_list.clear()
        items = ["0"]
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                items = [str(index) for index in range(max(1, count))]
        except Exception:
            items = ["0"]
        for value in items:
            item = QListWidgetItem(f"GPU {value}")
            item.setData(Qt.UserRole, value)
            self.cuda_device_list.addItem(item)
        if self.cuda_device_list.count() > 0:
            self.cuda_device_list.item(0).setSelected(True)

    def _discover_models(self) -> List[str]:
        discovered: List[str] = []
        try:
            cfg = load_config()
            models_dir = cfg.get("models_dir", "")
            if models_dir and os.path.isdir(models_dir):
                for name in sorted(os.listdir(models_dir)):
                    if name.lower().endswith((".safetensors", ".gguf", ".pt", ".pth", ".bin")):
                        discovered.append(name)
        except Exception:
            discovered = []
        merged = discovered + [item for item in self._model_fallback if item not in discovered]
        return merged or list(self._model_fallback)

    def reload_models(self) -> None:
        current = self.dit_model_combo.currentText()
        self.dit_model_combo.blockSignals(True)
        self.dit_model_combo.clear()
        self.dit_model_combo.addItems(self._discover_models())
        if current and self.dit_model_combo.findText(current) >= 0:
            self.dit_model_combo.setCurrentText(current)
        self.dit_model_combo.blockSignals(False)
        self._emit_settings_changed()

    def _build_resolution_group(self) -> None:
        form = self._group("RESOLUTION")
        self.resolution_mode_combo = self._combo(["pixel", "xtimes", "presets"], "pixel")
        self.resolution_mode_combo.currentTextChanged.connect(self._update_resolution_mode)
        self.resolution_spin = self._spin(128, 7680, 720, 8)
        self.resolution_scale_combo = self._combo(["2", "3", "4", "5"], "2")
        self.resolution_presets_combo = self._combo(list(self._PRESET_RESOLUTIONS.keys()), "720p (HD)")
        self.max_resolution_spin = self._spin(128, 7680, 3840, 8)
        self.pre_downscale_combo = self._combo(["1", "2"], "1")

        self._resolution_label = self._label("Resolution")
        self._resolution_scale_label = self._label("Scale")
        self._resolution_presets_label = self._label("Preset")

        form.addRow(self._label("Mode"), self.resolution_mode_combo)
        form.addRow(self._resolution_label, self.resolution_spin)
        form.addRow(self._resolution_scale_label, self.resolution_scale_combo)
        form.addRow(self._resolution_presets_label, self.resolution_presets_combo)
        form.addRow(self._label("Max resolution"), self.max_resolution_spin)
        form.addRow(self._label("Pre-downscale"), self.pre_downscale_combo)

    def _build_model_group(self) -> None:
        form = self._group("MODEL & PERFORMANCE")
        self.dit_model_combo = self._combo(self._discover_models(), "seedvr2_ema_3b_fp8_e4m3fn.safetensors")
        self.attention_mode_combo = self._combo(
            ["sdpa", "flash_attn_2", "flash_attn_3", "sage_attn_2", "sage_attn_3"],
            "sage_attn_3",
        )
        self.auto_tune_toggle = self._toggle(True)
        self.cache_dit_toggle = self._toggle(True)
        self.cache_vae_toggle = self._toggle(True)
        self.use_10bit_toggle = self._toggle(True)
        self.debug_toggle = self._toggle(False)

        form.addRow(self._label("DiT model"), self.dit_model_combo)
        form.addRow(self._label("Attention"), self.attention_mode_combo)
        form.addRow(self._label("Auto tune"), self.auto_tune_toggle)
        form.addRow(self._label("Cache DiT"), self.cache_dit_toggle)
        form.addRow(self._label("Cache VAE"), self.cache_vae_toggle)
        form.addRow(self._label("10-bit output"), self.use_10bit_toggle)
        form.addRow(self._label("Debug"), self.debug_toggle)

    def _build_runtime_group(self) -> None:
        form = self._group("RUNTIME")
        self.uniform_batch_toggle = self._toggle(True)
        self.batch_size_spin = self._spin(1, 999999, 81, 4)
        self.batch_size_spin.valueChanged.connect(self._snap_batch_size)
        self.load_cap_spin = self._spin(0, 999999, 0)
        self.skip_first_frames_spin = self._spin(0, 999999, 0)
        self.only_frames_edit = _FrameListLineEdit(self)
        self.only_frames_edit.setPlaceholderText("e.g. 1,5,9")
        self.only_frames_edit.textChanged.connect(self._emit_settings_changed)
        self.only_frames_edit.validated.connect(self._emit_settings_changed)

        form.addRow(self._label("Uniform batch"), self.uniform_batch_toggle)
        form.addRow(self._label("Batch size"), self.batch_size_spin)
        form.addRow(self._label("Load cap"), self.load_cap_spin)
        form.addRow(self._label("Skip first frames"), self.skip_first_frames_spin)
        form.addRow(self._label("Only frames"), self.only_frames_edit)

    def _build_batch_group(self) -> None:
        form = self._group("QUALITY & CHUNKING")
        self.color_correction_combo = self._combo(
            ["none", "lab", "wavelet", "wavelet_adaptive", "hsv", "adain"],
            "none",
        )
        input_row, self.input_noise_slider, self.input_noise_label = self._slider_row(0)
        latent_row, self.latent_noise_slider, self.latent_noise_label = self._slider_row(0)
        self.temporal_overlap_spin = self._spin(0, 64, 8)
        self.prepend_frames_spin = self._spin(0, 16, 4)
        self.chunk_size_spin = self._spin(0, 999999, 0)
        self.chunk_duration_combo = self._combo([str(index) for index in range(6)], "0")

        form.addRow(self._label("Color correction"), self.color_correction_combo)
        form.addRow(self._label("Input noise"), input_row)
        form.addRow(self._label("Latent noise"), latent_row)
        form.addRow(self._label("Temporal overlap"), self.temporal_overlap_spin)
        form.addRow(self._label("Prepend frames"), self.prepend_frames_spin)
        form.addRow(self._label("Chunk size"), self.chunk_size_spin)
        form.addRow(self._label("Chunk minutes"), self.chunk_duration_combo)

    def _build_vae_group(self) -> None:
        form = self._group("VAE TILING")
        self.vae_encode_tiled_toggle = self._toggle(True)
        self.vae_encode_tiled_toggle.toggled.connect(self._update_vae_controls)
        self.vae_encode_tile_size_spin = self._spin(128, 4096, 1024, 128)
        self.vae_encode_tile_overlap_spin = self._spin(0, 512, 64, 16)
        self.vae_decode_tiled_toggle = self._toggle(True)
        self.vae_decode_tiled_toggle.toggled.connect(self._update_vae_controls)
        self.vae_decode_tile_size_spin = self._spin(128, 4096, 1024, 128)
        self.vae_decode_tile_overlap_spin = self._spin(0, 512, 64, 16)

        form.addRow(self._label("Encode tiled"), self.vae_encode_tiled_toggle)
        form.addRow(self._label("Encode tile size"), self.vae_encode_tile_size_spin)
        form.addRow(self._label("Encode overlap"), self.vae_encode_tile_overlap_spin)
        form.addRow(_separator())
        form.addRow(self._label("Decode tiled"), self.vae_decode_tiled_toggle)
        form.addRow(self._label("Decode tile size"), self.vae_decode_tile_size_spin)
        form.addRow(self._label("Decode overlap"), self.vae_decode_tile_overlap_spin)

    def _build_quality_group(self) -> None:
        form = self._group("OFFLOAD & BLOCKSWAP")
        self.dit_offload_device_combo = self._combo(["none", "cpu"], "none")
        self.vae_offload_device_combo = self._combo(["none", "cpu"], "cpu")
        self.tensor_offload_device_combo = self._combo(["none", "cpu"], "none")
        self.blocks_to_swap_spin = self._spin(0, 200, 0)
        self.swap_io_components_check = self._check(False)

        form.addRow(self._label("DiT offload"), self.dit_offload_device_combo)
        form.addRow(self._label("VAE offload"), self.vae_offload_device_combo)
        form.addRow(self._label("Tensor offload"), self.tensor_offload_device_combo)
        form.addRow(self._label("Blocks to swap"), self.blocks_to_swap_spin)
        form.addRow(self._label("Swap I/O components"), self.swap_io_components_check)

    def _build_device_group(self) -> None:
        form = self._group("COMPILATION & DEVICES")
        self.compile_dit_check = self._check(False)
        self.compile_vae_check = self._check(False)
        self.compile_backend_combo = self._combo(["inductor", "eager"], "inductor")
        self.compile_mode_combo = self._combo(["reduce-overhead", "default", "max-autotune"], "reduce-overhead")
        self.cuda_device_list = QListWidget(self)
        self.cuda_device_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.cuda_device_list.setMaximumHeight(96)
        self.cuda_device_list.itemSelectionChanged.connect(self._emit_settings_changed)
        self._populate_cuda_devices()

        form.addRow(self._label("Compile DiT"), self.compile_dit_check)
        form.addRow(self._label("Compile VAE"), self.compile_vae_check)
        form.addRow(self._label("Compile backend"), self.compile_backend_combo)
        form.addRow(self._label("Compile mode"), self.compile_mode_combo)
        form.addRow(self._label("CUDA device(s)"), self.cuda_device_list)

    def _build_vram_group(self) -> None:
        box = QGroupBox("ESTIMATED VRAM", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_LG, Dims.PADDING_MD, Dims.PADDING_MD)
        self.vram_value_label = QLabel("0.0 GB", box)
        self.vram_value_label.setStyleSheet(
            f"font-size: {Fonts.SIZE_H1 + 6}px; font-weight: {Fonts.WEIGHT_BOLD}; color: {Colors.SUCCESS};"
        )
        self.vram_status_label = QLabel("Balanced defaults selected", box)
        self.vram_status_label.setWordWrap(True)
        self.vram_status_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}px;"
        )
        layout.addWidget(self.vram_value_label)
        layout.addWidget(self.vram_status_label)
        self._layout.addWidget(box)

    def _snap_batch_size(self, value: int) -> None:
        snapped = round((value - 1) / 4) * 4 + 1
        snapped = max(1, snapped)
        if snapped != value:
            self.batch_size_spin.blockSignals(True)
            self.batch_size_spin.setValue(snapped)
            self.batch_size_spin.blockSignals(False)
        self._emit_settings_changed()

    def _update_resolution_mode(self) -> None:
        mode = self.resolution_mode_combo.currentText()
        self._resolution_label.setVisible(mode == "pixel")
        self.resolution_spin.setVisible(mode == "pixel")
        self._resolution_scale_label.setVisible(mode == "xtimes")
        self.resolution_scale_combo.setVisible(mode == "xtimes")
        self._resolution_presets_label.setVisible(mode == "presets")
        self.resolution_presets_combo.setVisible(mode == "presets")
        self._emit_settings_changed()

    def _update_vae_controls(self) -> None:
        encode_enabled = self.vae_encode_tiled_toggle.isChecked()
        decode_enabled = self.vae_decode_tiled_toggle.isChecked()
        self.vae_encode_tile_size_spin.setEnabled(encode_enabled)
        self.vae_encode_tile_overlap_spin.setEnabled(encode_enabled)
        self.vae_decode_tile_size_spin.setEnabled(decode_enabled)
        self.vae_decode_tile_overlap_spin.setEnabled(decode_enabled)
        self._emit_settings_changed()

    def _emit_settings_changed(self) -> None:
        self.settings_changed.emit()
        self._vram_timer.start()

    def update_vram_estimate(self) -> None:
        settings = self.get_all_settings()
        resolution = settings["resolution"]
        if settings["resolution_mode"] == "presets":
            resolution = self._PRESET_RESOLUTIONS.get(settings["resolution_presets"], resolution)
        elif settings["resolution_mode"] == "xtimes":
            resolution = int(settings["resolution_scale"]) * 720

        batch = settings["batch_size"]
        est = 3.0 + (resolution / 720.0) * 1.4 + ((batch - 1) / 4.0) * 0.08
        if "7b" in settings["dit_model"].lower():
            est += 3.8
        if settings["cache_dit"]:
            est += 0.5
        if settings["cache_vae"]:
            est += 0.3
        if settings["vae_encode_tiled"] or settings["vae_decode_tiled"]:
            est *= 0.82
        if settings["compile_dit"] or settings["compile_vae"]:
            est += 0.4

        if est < 8:
            color = Colors.SUCCESS
            text = "Comfortable — current settings should fit well"
        elif est < 14:
            color = Colors.WARNING
            text = "Elevated — Auto Tune recommended"
        else:
            color = Colors.DANGER
            text = "High risk — reduce resolution or batch size"

        self.vram_value_label.setText(f"{est:.1f} GB")
        self.vram_value_label.setStyleSheet(
            f"font-size: {Fonts.SIZE_H1 + 6}px; font-weight: {Fonts.WEIGHT_BOLD}; color: {color};"
        )
        self.vram_status_label.setText(text)

    def set_enabled_state(self, enabled: bool) -> None:
        self.setEnabled(enabled)

    def _selected_cuda_devices(self) -> str:
        values = []
        for index in range(self.cuda_device_list.count()):
            item = self.cuda_device_list.item(index)
            if item.isSelected():
                values.append(str(item.data(Qt.UserRole)))
        return ",".join(values)

    def get_all_settings(self) -> Dict[str, Any]:
        return {
            "resolution_mode": self.resolution_mode_combo.currentText(),
            "resolution": self.resolution_spin.value(),
            "resolution_scale": self.resolution_scale_combo.currentText(),
            "resolution_presets": self.resolution_presets_combo.currentText(),
            "max_resolution": self.max_resolution_spin.value(),
            "pre_downscale": self.pre_downscale_combo.currentText(),
            "dit_model": self.dit_model_combo.currentText(),
            "attention_mode": self.attention_mode_combo.currentText(),
            "auto_tune": self.auto_tune_toggle.isChecked(),
            "cache_dit": self.cache_dit_toggle.isChecked(),
            "cache_vae": self.cache_vae_toggle.isChecked(),
            "use_10bit": self.use_10bit_toggle.isChecked(),
            "debug": self.debug_toggle.isChecked(),
            "uniform_batch_size": self.uniform_batch_toggle.isChecked(),
            "batch_size": self.batch_size_spin.value(),
            "load_cap": self.load_cap_spin.value(),
            "skip_first_frames": self.skip_first_frames_spin.value(),
            "only_frames": self.only_frames_edit.text().strip(),
            "vae_encode_tiled": self.vae_encode_tiled_toggle.isChecked(),
            "vae_encode_tile_size": self.vae_encode_tile_size_spin.value(),
            "vae_encode_tile_overlap": self.vae_encode_tile_overlap_spin.value(),
            "vae_decode_tiled": self.vae_decode_tiled_toggle.isChecked(),
            "vae_decode_tile_size": self.vae_decode_tile_size_spin.value(),
            "vae_decode_tile_overlap": self.vae_decode_tile_overlap_spin.value(),
            "color_correction": self.color_correction_combo.currentText(),
            "input_noise_scale": self.input_noise_slider.value() / 100.0,
            "latent_noise_scale": self.latent_noise_slider.value() / 100.0,
            "temporal_overlap": self.temporal_overlap_spin.value(),
            "prepend_frames": self.prepend_frames_spin.value(),
            "chunk_size": self.chunk_size_spin.value(),
            "chunk_duration_minutes": self.chunk_duration_combo.currentText(),
            "dit_offload_device": self.dit_offload_device_combo.currentText(),
            "vae_offload_device": self.vae_offload_device_combo.currentText(),
            "tensor_offload_device": self.tensor_offload_device_combo.currentText(),
            "blocks_to_swap": self.blocks_to_swap_spin.value(),
            "swap_io_components": self.swap_io_components_check.isChecked(),
            "compile_dit": self.compile_dit_check.isChecked(),
            "compile_vae": self.compile_vae_check.isChecked(),
            "compile_backend": self.compile_backend_combo.currentText(),
            "compile_mode": self.compile_mode_combo.currentText(),
            "cuda_device": self._selected_cuda_devices(),
            "seed": 313,
            "video_backend": "ffmpeg",
            "tile_debug": "false",
        }
