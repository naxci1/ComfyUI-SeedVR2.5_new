"""Right-side settings panel with 1:1 parity to inference_cli.py arguments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import QSize, Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..theme import Colors, Dims, Fonts
from .button3d import Button3D
from .toggle_switch import ToggleSwitch

try:
    from gui.config_manager import load_config
except ImportError:  # pragma: no cover - direct-script execution fallback
    from config_manager import load_config  # type: ignore[no-redef]

# ------------------------------------------------------------------ model name mapping
# Maps short display names → actual filenames (order determines default priority).
_MODEL_DISPLAY_MAP: Dict[str, str] = {
    "3B Q8": "seedvr2_ema_3b-Q8_0.gguf",
    "3B FP8": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "3B FP16": "seedvr2_ema_3b_fp16.safetensors",
    "3B Q4": "seedvr2_ema_3b-Q4_K_M.gguf",
    "7B Q4": "seedvr2_ema_7b-Q4_K_M.gguf",
    "7B Q8": "seedvr2_ema_7b-Q8_0.gguf",
    "7B FP8": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "7B FP16": "seedvr2_ema_7b_fp16.safetensors",
    "7B Sharp Q4": "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
    "7B Sharp FP8": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "7B Sharp FP16": "seedvr2_ema_7b_sharp_fp16.safetensors",
}
# Reverse map: filename → short display name.
_MODEL_FILENAME_TO_DISPLAY: Dict[str, str] = {v: k for k, v in _MODEL_DISPLAY_MAP.items()}


def _display_name(filename: str) -> str:
    """Return short display name for a model filename."""
    return _MODEL_FILENAME_TO_DISPLAY.get(filename, filename)


def _filename_for_display(display: str) -> str:
    """Return actual filename for a short display name."""
    return _MODEL_DISPLAY_MAP.get(display, display)


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


class _ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class NumericSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(
        self,
        minimum: int,
        maximum: int,
        value: int,
        step: int = 1,
        formatter=None,
        parser=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self._step = max(1, int(step))
        self._formatter = formatter or (lambda v: str(v))
        self._parser = parser or (lambda text: int(round(float(text.strip()))))
        self._editing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Dims.PADDING_SM)

        self._slider = QSlider(Qt.Horizontal, self)
        self._slider.setRange(0, max(0, (self._maximum - self._minimum) // self._step))
        self._label = _ClickableLabel("", self)
        self._label.setMinimumWidth(58)
        self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._label.setCursor(Qt.PointingHandCursor)
        self._editor = QLineEdit(self)
        self._editor.setMinimumWidth(58)
        self._editor.setAlignment(Qt.AlignRight)
        self._editor.hide()

        layout.addWidget(self._slider, 1)
        layout.addWidget(self._label)
        layout.addWidget(self._editor)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._label.clicked.connect(self._begin_edit)
        self._editor.returnPressed.connect(self._commit_edit)
        self._editor.editingFinished.connect(self._commit_edit)
        self.setValue(value)

    def _snap(self, value: int) -> int:
        snapped = self._minimum + round((value - self._minimum) / self._step) * self._step
        return max(self._minimum, min(self._maximum, snapped))

    def _value_from_position(self, position: int) -> int:
        return self._snap(self._minimum + position * self._step)

    def _position_from_value(self, value: int) -> int:
        return max(0, min(self._slider.maximum(), (self._snap(value) - self._minimum) // self._step))

    def _update_label(self, value: int) -> None:
        self._label.setText(self._formatter(value))

    def _on_slider_changed(self, position: int) -> None:
        value = self._value_from_position(position)
        self._update_label(value)
        if not self.signalsBlocked():
            self.valueChanged.emit(value)

    def _begin_edit(self) -> None:
        if not self.isEnabled():
            return
        self._editing = True
        self._editor.setText(self._formatter(self.value()))
        self._label.hide()
        self._editor.show()
        self._editor.selectAll()
        self._editor.setFocus()

    def _commit_edit(self) -> None:
        if not self._editing:
            return
        self._editing = False
        try:
            parsed = self._parser(self._editor.text())
            self.setValue(parsed)
        except Exception:
            self._update_label(self.value())
        self._editor.hide()
        self._label.show()

    def value(self) -> int:
        return self._value_from_position(self._slider.value())

    def setValue(self, value: int) -> None:  # noqa: N802
        snapped = self._snap(int(value))
        position = self._position_from_value(snapped)
        changed = snapped != self.value()
        self._slider.blockSignals(True)
        self._slider.setValue(position)
        self._slider.blockSignals(False)
        self._update_label(snapped)
        if changed and not self.signalsBlocked():
            self.valueChanged.emit(snapped)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        self._label.setEnabled(enabled)
        self._editor.setEnabled(enabled)


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
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
        "1440p": 1440,
        "2K": 2048,
        "4K": 2160,
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(max(Dims.PANEL_WIDTH_RIGHT + 96, 356))
        self._trim_in_frame = 0
        self._trim_out_frame = 0
        self._trim_frame_count = 0
        self._trim_active = False
        self._simple_mode = False
        self._forms: List[QFormLayout] = []
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

        self._vram_timer = QTimer(self)
        self._vram_timer.setSingleShot(True)
        self._vram_timer.setInterval(300)
        self._vram_timer.timeout.connect(self._update_vram_prediction)
        self.settings_changed.connect(self._vram_timer.start)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, Dims.PADDING_LG, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        body = QWidget()
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_MD, Dims.PADDING_LG, Dims.PADDING_MD)
        self._layout.setSpacing(Dims.PADDING_MD)
        scroll.setWidget(body)

        self._build_presets_group()
        self._build_resolution_group()
        self._build_model_group()
        self._build_runtime_group()
        self._build_batch_group()
        self._build_vae_group()
        self._build_quality_group()
        self._build_device_group()
        self._build_vram_group()
        self._layout.addStretch(1)

        self._update_resolution_mode()
        self._update_vae_controls()
        self._update_vram_prediction()

    # ---------------------------------------------------------------- simple mode
    def set_simple_mode(self, simple: bool) -> None:
        """Show only resolution preset + quality chunk controls when simple=True."""
        self._simple_mode = simple
        advanced = not simple
        for widget in self._advanced_only_widgets:
            widget.setVisible(advanced)
        self._preset_group_box.setVisible(advanced)
        self._resolution_group_box.setVisible(True)
        self._batch_group_box.setVisible(True)

        self._set_field_visible(self.resolution_mode_combo, advanced)
        self._set_field_visible(self.resolution_spin, advanced and self.resolution_mode_combo.currentText() == "pixel")
        self._set_field_visible(self.resolution_scale_combo, advanced and self.resolution_mode_combo.currentText() == "xtimes")
        self._set_field_visible(self.resolution_presets_combo, True)
        self._set_field_visible(self.max_resolution_spin, advanced)
        self._set_field_visible(self.pre_downscale_combo, advanced)

        self._set_field_visible(self.color_correction_combo, advanced)
        self._set_field_visible(self.input_noise_slider, advanced)
        self._set_field_visible(self.latent_noise_slider, advanced)
        self._set_field_visible(self.temporal_overlap_spin, advanced)
        self._set_field_visible(self.prepend_frames_spin, advanced)
        self._set_field_visible(self.chunk_duration_combo, advanced)
        self._set_field_visible(self.chunk_size_spin, True)

        if simple:
            self.resolution_mode_combo.setCurrentText("presets")
            label = self._batch_form.labelForField(self.chunk_size_spin)
            if label is not None:
                label.setText("Quality chunk")
            res_label = self._resolution_form.labelForField(self.resolution_presets_combo)
            if res_label is not None:
                res_label.setText("Resolution")
        else:
            label = self._batch_form.labelForField(self.chunk_size_spin)
            if label is not None:
                label.setText("Chunk size")
            self._update_resolution_mode()

    def set_trim_range(self, trim_in: int, trim_out: int, frame_count: int, active: bool) -> None:
        self._trim_in_frame = max(0, int(trim_in))
        self._trim_out_frame = max(self._trim_in_frame, int(trim_out))
        self._trim_frame_count = max(0, int(frame_count))
        self._trim_active = bool(active) and self._trim_frame_count > 0

    def _group(self, title: str, advanced_only: bool = False) -> QFormLayout:
        box = QGroupBox(title, self)
        form = QFormLayout(box)
        form.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_LG, Dims.PADDING_MD, Dims.PADDING_MD)
        form.setSpacing(Dims.PADDING_SM)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._layout.addWidget(box)
        if advanced_only:
            self._advanced_only_widgets.append(box)
        self._forms.append(form)
        return form

    def _set_field_visible(self, widget: QWidget, visible: bool) -> None:
        for form in self._forms:
            label = form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
                break
        widget.setVisible(visible)

    def _build_presets_group(self) -> None:
        """Presets section: save/load named settings presets."""
        self._advanced_only_widgets: List[QWidget] = []
        self._presets_file = Path(os.path.expanduser("~")) / ".seedvr2_presets.json"
        self._presets: Dict[str, Any] = self._load_presets()

        box = QGroupBox("PRESETS", self)
        self._preset_group_box = box
        vlay = QVBoxLayout(box)
        vlay.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_LG, Dims.PADDING_MD, Dims.PADDING_MD)
        vlay.setSpacing(Dims.PADDING_SM)
        self._layout.addWidget(box)

        # Buttons row.
        btn_row = QHBoxLayout()
        self._preset_name_edit = QLineEdit(box)
        self._preset_name_edit.setPlaceholderText("Preset name…")
        self._save_preset_btn = Button3D("Save", variant="default", parent=box)
        self._save_preset_btn.clicked.connect(self._on_save_preset)
        btn_row.addWidget(self._preset_name_edit, 1)
        btn_row.addWidget(self._save_preset_btn)
        vlay.addLayout(btn_row)

        self._preset_list = QListWidget(box)
        self._preset_list.setViewMode(QListView.IconMode)
        self._preset_list.setFlow(QListView.LeftToRight)
        self._preset_list.setWrapping(True)
        self._preset_list.setResizeMode(QListView.Adjust)
        self._preset_list.setMovement(QListView.Static)
        self._preset_list.setSpacing(Dims.PADDING_XS)
        self._preset_list.setMaximumHeight(110)
        self._preset_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._preset_list.customContextMenuRequested.connect(self._on_preset_context_menu)
        self._preset_list.itemDoubleClicked.connect(self._on_load_preset)
        vlay.addWidget(self._preset_list)

        del_lbl = QLabel("(double-click to load • right-click to edit)", box)
        del_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_TINY}px;")
        vlay.addWidget(del_lbl)

        self._refresh_preset_list()

    def _load_presets(self) -> Dict[str, Any]:
        try:
            if self._presets_file.exists():
                with open(self._presets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
        except Exception:
            pass
        return {}

    def _save_presets_to_disk(self) -> None:
        try:
            with open(self._presets_file, "w", encoding="utf-8") as f:
                json.dump(self._presets, f, indent=2)
        except Exception:
            pass

    def _refresh_preset_list(self) -> None:
        self._preset_list.clear()
        for name in sorted(self._presets.keys()):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            item.setSizeHint(QSize(92, 28))
            self._preset_list.addItem(item)

    def _on_save_preset(self, _event=None) -> None:
        name = self._preset_name_edit.text().strip()
        if not name:
            return
        try:
            self._presets[name] = self.get_all_settings()
        except Exception:
            return
        self._save_presets_to_disk()
        self._refresh_preset_list()
        self._preset_name_edit.clear()

    def _on_preset_context_menu(self, pos) -> None:
        item = self._preset_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        edit_action = menu.addAction("Edit")
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self._preset_list.viewport().mapToGlobal(pos))
        if action in (edit_action, rename_action):
            self._rename_preset(item)
        elif action == delete_action:
            self._delete_preset(item)

    def _rename_preset(self, item: QListWidgetItem) -> None:
        current_name = str(item.data(Qt.UserRole))
        new_name, ok = QInputDialog.getText(self, "Rename preset", "Preset name:", text=current_name)
        new_name = new_name.strip()
        if not ok or not new_name or new_name == current_name:
            return
        self._presets[new_name] = self._presets.pop(current_name)
        self._save_presets_to_disk()
        self._refresh_preset_list()

    def _delete_preset(self, item: QListWidgetItem) -> None:
        name = str(item.data(Qt.UserRole))
        if QMessageBox.question(self, "Delete preset", f"Delete preset '{name}'?") != QMessageBox.Yes:
            return
        self._presets.pop(name, None)
        self._save_presets_to_disk()
        self._refresh_preset_list()

    def _on_load_preset(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.UserRole)
        settings = self._presets.get(name)
        if not settings or not isinstance(settings, dict):
            return
        # Apply all available settings from the preset.
        try:
            if "resolution_mode" in settings:
                self.resolution_mode_combo.setCurrentText(str(settings["resolution_mode"]))
            if "resolution" in settings:
                self.resolution_spin.setValue(int(settings["resolution"]))
            if "resolution_scale" in settings:
                self.resolution_scale_combo.setCurrentText(str(settings["resolution_scale"]))
            if "resolution_presets" in settings:
                self.resolution_presets_combo.setCurrentText(str(settings["resolution_presets"]))
            if "batch_size" in settings:
                self.batch_size_spin.setValue(int(settings["batch_size"]))
            if "auto_tune" in settings:
                self.auto_tune_toggle.setChecked(bool(settings["auto_tune"]))
            if "cache_dit" in settings:
                self.cache_dit_toggle.setChecked(bool(settings["cache_dit"]))
            if "cache_vae" in settings:
                self.cache_vae_toggle.setChecked(bool(settings["cache_vae"]))
            if "temporal_overlap" in settings:
                self.temporal_overlap_spin.setValue(int(settings["temporal_overlap"]))
            if "color_correction" in settings:
                self.color_correction_combo.setCurrentText(str(settings["color_correction"]))
        except Exception:
            pass
        self._emit_settings_changed()

    def _label(self, text: str) -> QLabel:
        return QLabel(text)

    def _spin(self, minimum: int, maximum: int, value: int, step: int = 1, formatter=None, parser=None) -> NumericSlider:
        spin = NumericSlider(minimum, maximum, value, step=step, formatter=formatter, parser=parser)
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

    def _slider_row(self, value: int) -> NumericSlider:
        return self._spin(
            0,
            100,
            value,
            formatter=lambda v: f"{v / 100.0:.2f}",
            parser=lambda text: int(round(float(text.strip()) * 100.0)),
        )

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
        """Return list of short display names for discovered + fallback models."""
        discovered: List[str] = []
        try:
            cfg = load_config()
            models_dir = cfg.get("models_dir", "")
            if models_dir and os.path.isdir(models_dir):
                for name in sorted(os.listdir(models_dir)):
                    lowered = name.lower()
                    if (
                        lowered.startswith("seedvr2_ema_")
                        and lowered.endswith((".safetensors", ".gguf"))
                    ):
                        discovered.append(name)
        except Exception:
            discovered = []
        merged = discovered + [item for item in self._model_fallback if item not in discovered]
        if not merged:
            merged = list(self._model_fallback)
        # Convert filenames to display names; keep unknown names as-is.
        return [_display_name(fn) for fn in merged]

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
        form = self._group("RESOLUTION")  # always visible
        self._resolution_form = form
        self._resolution_group_box = form.parentWidget()
        self.resolution_mode_combo = self._combo(["pixel", "xtimes", "presets"], "pixel")
        self.resolution_mode_combo.currentTextChanged.connect(self._update_resolution_mode)
        self.resolution_spin = self._spin(128, 7680, 720, 1)
        self.resolution_scale_combo = self._combo(["2", "3", "4", "5"], "2")
        self.resolution_presets_combo = self._combo(list(self._PRESET_RESOLUTIONS.keys()), "720p")
        self.max_resolution_spin = self._spin(128, 7680, 3840, 1)
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
        form = self._group("MODEL & PERFORMANCE", advanced_only=True)
        models = self._discover_models()
        default_display = _display_name("seedvr2_ema_3b_fp8_e4m3fn.safetensors")
        self.dit_model_combo = self._combo(models, default_display if default_display in models else (models[0] if models else ""))
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
        form = self._group("RUNTIME", advanced_only=True)
        self.uniform_batch_toggle = self._toggle(True)
        self.batch_size_spin = self._spin(1, 200, 81, 4)
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
        form = self._group("QUALITY & CHUNKING")  # always visible — chunk_size = "Quality/chunk"
        self._batch_form = form
        self._batch_group_box = form.parentWidget()
        self.color_correction_combo = self._combo(
            ["none", "lab", "wavelet", "wavelet_adaptive", "hsv", "adain"],
            "none",
        )
        self.input_noise_slider = self._slider_row(0)
        self.latent_noise_slider = self._slider_row(0)
        self.temporal_overlap_spin = self._spin(0, 64, 8)
        self.prepend_frames_spin = self._spin(0, 16, 4)
        self.chunk_size_spin = self._spin(0, 999999, 0)
        self.chunk_duration_combo = self._combo([str(index) for index in range(6)], "0")

        form.addRow(self._label("Color correction"), self.color_correction_combo)
        form.addRow(self._label("Input noise"), self.input_noise_slider)
        form.addRow(self._label("Latent noise"), self.latent_noise_slider)
        form.addRow(self._label("Temporal overlap"), self.temporal_overlap_spin)
        form.addRow(self._label("Prepend frames"), self.prepend_frames_spin)
        form.addRow(self._label("Chunk size"), self.chunk_size_spin)
        form.addRow(self._label("Chunk minutes"), self.chunk_duration_combo)

    def _build_vae_group(self) -> None:
        form = self._group("VAE TILING", advanced_only=True)
        self.vae_encode_tiled_toggle = self._toggle(True)
        self.vae_encode_tiled_toggle.toggled.connect(self._update_vae_controls)
        self.vae_encode_tile_size_spin = self._spin(128, 4096, 1024, 128)
        self.vae_encode_tile_overlap_spin = self._spin(0, 512, 64)
        self.vae_decode_tiled_toggle = self._toggle(True)
        self.vae_decode_tiled_toggle.toggled.connect(self._update_vae_controls)
        self.vae_decode_tile_size_spin = self._spin(128, 4096, 1024, 128)
        self.vae_decode_tile_overlap_spin = self._spin(0, 512, 64)

        form.addRow(self._label("Encode tiled"), self.vae_encode_tiled_toggle)
        form.addRow(self._label("Encode tile size"), self.vae_encode_tile_size_spin)
        form.addRow(self._label("Encode overlap"), self.vae_encode_tile_overlap_spin)
        form.addRow(_separator())
        form.addRow(self._label("Decode tiled"), self.vae_decode_tiled_toggle)
        form.addRow(self._label("Decode tile size"), self.vae_decode_tile_size_spin)
        form.addRow(self._label("Decode overlap"), self.vae_decode_tile_overlap_spin)

    def _build_quality_group(self) -> None:
        form = self._group("OFFLOAD & BLOCKSWAP", advanced_only=True)
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
        form = self._group("COMPILATION & DEVICES", advanced_only=True)
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
        self._advanced_only_widgets.append(box)

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
        if self._simple_mode:
            self._set_field_visible(self.resolution_mode_combo, False)
            self._set_field_visible(self.resolution_spin, False)
            self._set_field_visible(self.resolution_scale_combo, False)
            self._set_field_visible(self.resolution_presets_combo, True)
        else:
            self._set_field_visible(self.resolution_spin, mode == "pixel")
            self._set_field_visible(self.resolution_scale_combo, mode == "xtimes")
            self._set_field_visible(self.resolution_presets_combo, mode == "presets")
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

    def _update_vram_prediction(self) -> None:
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

    def update_vram_estimate(self) -> None:
        self._update_vram_prediction()

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
        skip_first_frames = 0
        load_cap = 0
        if self._trim_active:
            skip_first_frames = self._trim_in_frame
            load_cap = max(0, self._trim_out_frame - self._trim_in_frame)
        return {
            "resolution_mode": self.resolution_mode_combo.currentText(),
            "resolution": self.resolution_spin.value(),
            "resolution_scale": self.resolution_scale_combo.currentText(),
            "resolution_presets": self.resolution_presets_combo.currentText(),
            "max_resolution": self.max_resolution_spin.value(),
            "pre_downscale": self.pre_downscale_combo.currentText(),
            "dit_model": _filename_for_display(self.dit_model_combo.currentText()),
            "attention_mode": self.attention_mode_combo.currentText(),
            "auto_tune": self.auto_tune_toggle.isChecked(),
            "cache_dit": self.cache_dit_toggle.isChecked(),
            "cache_vae": self.cache_vae_toggle.isChecked(),
            "use_10bit": self.use_10bit_toggle.isChecked(),
            "debug": self.debug_toggle.isChecked(),
            "uniform_batch_size": self.uniform_batch_toggle.isChecked(),
            "batch_size": self.batch_size_spin.value(),
            "load_cap": load_cap,
            "skip_first_frames": skip_first_frames,
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
