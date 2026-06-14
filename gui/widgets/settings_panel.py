"""Right-side enhancement settings panel with a live VRAM estimate."""

from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..theme import Colors, Dims, Fonts
from .toggle_switch import ToggleSwitch


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setProperty("role", "separator")
    line.setFixedHeight(1)
    line.setStyleSheet(f"background-color: {Colors.BORDER};")
    return line


class SettingsPanel(QWidget):
    """Enhancement settings with a real-time VRAM estimate box."""

    settings_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(Dims.PANEL_WIDTH_RIGHT)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        body = QWidget()
        self._v = QVBoxLayout(body)
        self._v.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_MD)
        self._v.setSpacing(Dims.PADDING_MD)
        scroll.setWidget(body)

        self._build_resolution_group()
        self._build_model_group()
        self._build_options_group()
        self._build_batch_group()
        self._build_tiling_group()
        self._build_color_export_group()
        self._build_vram_box()
        self._v.addStretch(1)

        # Debounced VRAM recompute.
        self._vram_timer = QTimer(self)
        self._vram_timer.setSingleShot(True)
        self._vram_timer.setInterval(300)
        self._vram_timer.timeout.connect(self.update_vram_estimate)

        self.update_vram_estimate()

    # ---------------------------------------------------------------- helpers
    def _emit(self) -> None:
        self.settings_changed.emit()
        self._vram_timer.start()

    def _group(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_LG, Dims.PADDING_MD, Dims.PADDING_MD)
        form.setSpacing(Dims.PADDING_SM)
        form.setLabelAlignment(Qt.AlignLeft)
        self._v.addWidget(box)
        return form

    def _spin(self, lo, hi, val, step=1, suffix="") -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        s.valueChanged.connect(self._emit)
        return s

    def _combo(self, items, current=0) -> QComboBox:
        c = QComboBox()
        c.addItems(items)
        c.setCurrentIndex(current)
        c.currentIndexChanged.connect(self._emit)
        return c

    def _toggle(self, label, checked) -> ToggleSwitch:
        t = ToggleSwitch(label, checked)
        t.toggled.connect(lambda _=False: self._emit())
        return t

    def _check(self, label, checked) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(checked)
        cb.toggled.connect(lambda _=False: self._emit())
        return cb

    # ---------------------------------------------------------------- groups
    def _build_resolution_group(self) -> None:
        form = self._group("OUTPUT RESOLUTION")
        self.width_spin = self._spin(128, 7680, 1920, 1, " px")
        self.height_spin = self._spin(128, 4320, 1080, 1, " px")
        self.scale_combo = self._combo(["50%", "75%", "100%", "150%", "200%", "300%", "400%"], 2)
        self.short_side_combo = self._combo(["480p", "720p", "1080p", "1440p", "2K", "4K"], 2)
        self.res_mode_combo = self._combo(["Pixel", "X-Times"], 0)
        self.lock_aspect_check = self._check("Lock Aspect Ratio", True)
        self.pre_downscale_combo = self._combo(["1:1 None", "2:1 Half"], 0)

        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("Scale", self.scale_combo)
        form.addRow("Target short-side", self.short_side_combo)
        form.addRow("Mode", self.res_mode_combo)
        form.addRow(self.lock_aspect_check)
        form.addRow("Pre-Downscale", self.pre_downscale_combo)

    def _build_model_group(self) -> None:
        form = self._group("AI MODEL")
        self.dit_model_combo = self._combo(
            ["SeedVR2 3B Q8", "3B FP8", "7B Q4", "7B Q8", "7B Sharp Q4", "7B FP8"], 0
        )
        self.attention_combo = self._combo(
            ["Auto Best", "SDPA Safe", "Flash Attn 2", "Flash Attn 3"], 0
        )
        form.addRow("DiT Model", self.dit_model_combo)
        form.addRow("Attention", self.attention_combo)

    def _build_options_group(self) -> None:
        form = self._group("OPTIONS")
        self.auto_tune_toggle = self._toggle("Auto Tune", True)
        self.cache_dit_toggle = self._toggle("Cache DiT", True)
        self.cache_vae_toggle = self._toggle("Cache VAE", True)
        self.ten_bit_toggle = self._toggle("10-bit Output", True)
        self.debug_toggle = self._toggle("Debug Mode", False)
        for t in (self.auto_tune_toggle, self.cache_dit_toggle, self.cache_vae_toggle,
                  self.ten_bit_toggle, self.debug_toggle):
            form.addRow(t)

    def _build_batch_group(self) -> None:
        form = self._group("BATCH PROCESSING")
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.batch_slider = QSlider(Qt.Horizontal)
        self.batch_slider.setRange(1, 81)
        self.batch_slider.setValue(1)
        self.batch_value_label = QLabel("1")
        self.batch_value_label.setFixedWidth(28)
        self.batch_slider.valueChanged.connect(self._on_batch_changed)
        h.addWidget(self.batch_slider, 1)
        h.addWidget(self.batch_value_label)
        form.addRow("Batch Size", row)
        self.uniform_batch_check = self._check("Uniform Batch Size", True)
        form.addRow(self.uniform_batch_check)

    def _on_batch_changed(self, value: int) -> None:
        # Snap to 4n+1.
        snapped = round((value - 1) / 4) * 4 + 1
        snapped = max(1, min(81, snapped))
        if snapped != value:
            self.batch_slider.blockSignals(True)
            self.batch_slider.setValue(snapped)
            self.batch_slider.blockSignals(False)
        self.batch_value_label.setText(str(snapped))
        self._emit()

    def _build_tiling_group(self) -> None:
        form = self._group("VAE TILING")
        self.enable_tiling_toggle = self._toggle("Enable Tiling", True)
        self.enable_tiling_toggle.toggled.connect(self._update_tiling_enabled)
        form.addRow(self.enable_tiling_toggle)

        self.encode_tile_spin = self._spin(128, 2048, 1024, 128, " px")
        self.encode_overlap_spin = self._spin(0, 512, 64, 1, " px")
        self.decode_tile_spin = self._spin(128, 2048, 1024, 128, " px")
        self.decode_overlap_spin = self._spin(0, 512, 64, 1, " px")

        form.addRow("Encode Tile Size", self.encode_tile_spin)
        form.addRow("Encode Overlap", self.encode_overlap_spin)
        form.addRow(_separator())
        form.addRow("Decode Tile Size", self.decode_tile_spin)
        form.addRow("Decode Overlap", self.decode_overlap_spin)
        self._tiling_controls = [
            self.encode_tile_spin, self.encode_overlap_spin,
            self.decode_tile_spin, self.decode_overlap_spin,
        ]
        self._update_tiling_enabled()

    def _update_tiling_enabled(self) -> None:
        enabled = self.enable_tiling_toggle.isChecked()
        for c in self._tiling_controls:
            c.setEnabled(enabled)

    def _build_color_export_group(self) -> None:
        form = self._group("COLOR & EXPORT")
        self.color_correction_combo = self._combo(["None", "Simple", "Advanced", "Histogram"], 0)
        self.input_noise_slider = QSlider(Qt.Horizontal)
        self.input_noise_slider.setRange(0, 100)
        self.input_noise_slider.valueChanged.connect(self._emit)
        self.latent_noise_slider = QSlider(Qt.Horizontal)
        self.latent_noise_slider.setRange(0, 100)
        self.latent_noise_slider.valueChanged.connect(self._emit)
        self.temporal_overlap_spin = self._spin(0, 32, 8, 1, " frames")
        self.seed_spin = self._spin(0, 999999, 313, 1, "")

        form.addRow("Color Correction", self.color_correction_combo)
        form.addRow("Input Noise", self.input_noise_slider)
        form.addRow("Latent Noise", self.latent_noise_slider)
        form.addRow("Temporal Overlap", self.temporal_overlap_spin)
        form.addRow("Seed", self.seed_spin)

    def _build_vram_box(self) -> None:
        box = QGroupBox("ESTIMATED VRAM USAGE")
        v = QVBoxLayout(box)
        v.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_LG, Dims.PADDING_MD, Dims.PADDING_MD)
        self.vram_value_label = QLabel("0.0 GB")
        self.vram_value_label.setStyleSheet(
            f"font-size: {Fonts.SIZE_H1 + 6}px; font-weight: {Fonts.WEIGHT_BOLD};"
            f" color: {Colors.SUCCESS};"
        )
        self.vram_status_label = QLabel("Comfortable — should run smoothly")
        self.vram_status_label.setWordWrap(True)
        self.vram_status_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}px;"
        )
        v.addWidget(self.vram_value_label)
        v.addWidget(self.vram_status_label)
        self._v.addWidget(box)

    # ---------------------------------------------------------------- vram
    def update_vram_estimate(self) -> None:
        """Recalculate and display a rough VRAM estimate from current settings."""
        # Heuristic: base model footprint + resolution + batch + tiling effects.
        model = self.dit_model_combo.currentText()
        base = 6.0 if "7B" in model else 3.0
        if "Q4" in model:
            base *= 0.6
        elif "FP8" in model or "Q8" in model:
            base *= 0.8

        pixels = self.width_spin.value() * self.height_spin.value()
        res_factor = pixels / (1920 * 1080)
        batch = self.batch_slider.value()

        est = base + res_factor * 1.8 + (batch - 1) * 0.05
        if self.enable_tiling_toggle.isChecked():
            est *= 0.7  # tiling reduces peak VRAM
        if self.cache_dit_toggle.isChecked():
            est += 0.4
        if self.cache_vae_toggle.isChecked():
            est += 0.3

        self.vram_value_label.setText(f"{est:.1f} GB")
        if est < 8:
            color = Colors.SUCCESS
            status = "Comfortable — should run smoothly"
        elif est < 16:
            color = Colors.WARNING
            status = "High — close other GPU apps"
        else:
            color = Colors.DANGER
            status = "Too high — reduce resolution or batch size"
        self.vram_value_label.setStyleSheet(
            f"font-size: {Fonts.SIZE_H1 + 6}px; font-weight: {Fonts.WEIGHT_BOLD};"
            f" color: {color};"
        )
        self.vram_status_label.setText(status)

    # ---------------------------------------------------------------- api
    def set_enabled_state(self, enabled: bool) -> None:
        self.setEnabled(enabled)

    def get_all_settings(self) -> Dict[str, Any]:
        return {
            # Resolution.
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "scale": self.scale_combo.currentText(),
            "target_short_side": self.short_side_combo.currentText(),
            "resolution_mode": self.res_mode_combo.currentText(),
            "lock_aspect_ratio": self.lock_aspect_check.isChecked(),
            "pre_downscale": self.pre_downscale_combo.currentText(),
            # Model.
            "dit_model": self.dit_model_combo.currentText(),
            "attention": self.attention_combo.currentText(),
            # Options.
            "auto_tune": self.auto_tune_toggle.isChecked(),
            "cache_dit": self.cache_dit_toggle.isChecked(),
            "cache_vae": self.cache_vae_toggle.isChecked(),
            "ten_bit_output": self.ten_bit_toggle.isChecked(),
            "debug_mode": self.debug_toggle.isChecked(),
            # Batch.
            "batch_size": self.batch_slider.value(),
            "uniform_batch_size": self.uniform_batch_check.isChecked(),
            # Tiling.
            "enable_tiling": self.enable_tiling_toggle.isChecked(),
            "encode_tile_size": self.encode_tile_spin.value(),
            "encode_overlap": self.encode_overlap_spin.value(),
            "decode_tile_size": self.decode_tile_spin.value(),
            "decode_overlap": self.decode_overlap_spin.value(),
            # Color & export.
            "color_correction": self.color_correction_combo.currentText(),
            "input_noise": self.input_noise_slider.value(),
            "latent_noise": self.latent_noise_slider.value(),
            "temporal_overlap": self.temporal_overlap_spin.value(),
            "seed": self.seed_spin.value(),
        }
