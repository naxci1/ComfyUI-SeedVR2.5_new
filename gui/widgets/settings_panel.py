"""Right-side settings panel with 1:1 parity to inference_cli.py arguments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PySide6.QtCore import QEvent, QSize, Qt, Signal, QTimer
from PySide6.QtGui import QCursor
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
    QToolTip,
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
        self._tooltip_widgets: Set[QWidget] = set()
        self._tooltip_target: Optional[QWidget] = None
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
        # Debounce last_used saves so we don't write on every slider tick.
        self._last_used_timer = QTimer(self)
        self._last_used_timer.setSingleShot(True)
        self._last_used_timer.setInterval(1000)
        self._last_used_timer.timeout.connect(self.save_last_used)
        self.settings_changed.connect(self._last_used_timer.start)
        self._tooltip_timer = QTimer(self)
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.setInterval(700)
        self._tooltip_timer.timeout.connect(self._show_delayed_tooltip)

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
        self._build_chunk_group()
        self._build_vae_group()
        self._build_quality_group()
        self._build_device_group()
        self._build_vram_group()
        self._layout.addStretch(1)

        self._update_resolution_mode()
        self._update_pre_downscale_visibility()
        self._update_vae_controls()
        self._update_vram_prediction()

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

    @staticmethod
    def _presets_dir() -> Path:
        """Return the directory where preset JSON files are stored."""
        import sys as _sys
        if _sys.platform == "win32":
            base = Path("C:/1Click_SeedVR2.5/presets")
        else:
            base = Path(os.path.expanduser("~")) / ".seedvr2" / "presets"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _build_presets_group(self) -> None:
        """Presets section: save/load named settings presets."""
        self._advanced_only_widgets: List[QWidget] = []
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
        self._set_setting_tooltip(
            self._preset_name_edit,
            "Name for a reusable settings profile. Save to quickly recall the same configuration.",
        )
        self._set_setting_tooltip(
            self._save_preset_btn,
            "Save current settings to the preset name entered on the left.",
        )
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
        self._set_setting_tooltip(
            self._preset_list,
            "Saved presets. Double-click to load. Right-click to update, rename, or delete.",
        )
        vlay.addWidget(self._preset_list)

        del_lbl = QLabel("(double-click to load • right-click to update)", box)
        del_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_TINY}px;")
        vlay.addWidget(del_lbl)

        self._refresh_preset_list()
        self._apply_last_used_settings()

    def _load_presets(self) -> Dict[str, Any]:
        """Load all presets from the presets directory (one JSON file each)."""
        presets: Dict[str, Any] = {}
        try:
            d = self._presets_dir()
            for f in d.glob("*.json"):
                if f.stem == "last_used":
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        if isinstance(data, dict):
                            presets[f.stem] = data
                except Exception:
                    pass
        except Exception:
            pass
        return presets

    def _save_preset_to_disk(self, name: str, settings: Dict[str, Any]) -> None:
        """Save a single preset as {name}.json in the presets directory."""
        try:
            d = self._presets_dir()
            # Sanitise name for use as a filename.
            safe = "".join(c if c.isalnum() or c in " _-." else "_" for c in name).strip()
            if not safe:
                safe = "preset"
            path = d / f"{safe}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(settings, fh, indent=2)
        except Exception:
            pass

    def _delete_preset_from_disk(self, name: str) -> None:
        """Remove the preset JSON file for *name*."""
        try:
            d = self._presets_dir()
            safe = "".join(c if c.isalnum() or c in " _-." else "_" for c in name).strip()
            path = d / f"{safe}.json"
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def save_last_used(self) -> None:
        """Persist current settings as last_used.json."""
        try:
            d = self._presets_dir()
            with open(d / "last_used.json", "w", encoding="utf-8") as fh:
                json.dump(self.get_all_settings(), fh, indent=2)
        except Exception:
            pass

    def _apply_last_used_settings(self) -> None:
        """Restore settings from last_used.json if it exists."""
        try:
            d = self._presets_dir()
            last = d / "last_used.json"
            if not last.exists():
                return
            with open(last, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                # Reuse the preset load logic.
                class _FakeItem:
                    def data(self, _role):
                        return "__last_used__"
                self._presets["__last_used__"] = data
                self._on_load_preset(_FakeItem())  # type: ignore[arg-type]
                self._presets.pop("__last_used__", None)
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
            settings = self.get_all_settings()
            self._presets[name] = settings
            self._save_preset_to_disk(name, settings)
        except Exception:
            return
        self._refresh_preset_list()
        self._preset_name_edit.clear()

    def _on_preset_context_menu(self, pos) -> None:
        item = self._preset_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        update_action = menu.addAction("Update")
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self._preset_list.viewport().mapToGlobal(pos))
        if action == update_action:
            self._update_preset(item)
        elif action == rename_action:
            self._rename_preset(item)
        elif action == delete_action:
            self._delete_preset(item)

    def _update_preset(self, item: QListWidgetItem) -> None:
        name = str(item.data(Qt.UserRole))
        if not name:
            return
        try:
            settings = self.get_all_settings()
            self._presets[name] = settings
            self._save_preset_to_disk(name, settings)
        except Exception:
            return
        self._refresh_preset_list()

    def _rename_preset(self, item: QListWidgetItem) -> None:
        current_name = str(item.data(Qt.UserRole))
        new_name, ok = QInputDialog.getText(self, "Rename preset", "Preset name:", text=current_name)
        new_name = new_name.strip()
        if not ok or not new_name or new_name == current_name:
            return
        settings = self._presets.pop(current_name)
        self._delete_preset_from_disk(current_name)
        self._presets[new_name] = settings
        self._save_preset_to_disk(new_name, settings)
        self._refresh_preset_list()

    def _delete_preset(self, item: QListWidgetItem) -> None:
        name = str(item.data(Qt.UserRole))
        if QMessageBox.question(self, "Delete preset", f"Delete preset '{name}'?") != QMessageBox.Yes:
            return
        self._presets.pop(name, None)
        self._delete_preset_from_disk(name)
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
            if "max_resolution" in settings:
                max_res = int(settings["max_resolution"])
                if max_res > 0:
                    self.max_resolution_toggle.setChecked(True)
                    self.max_resolution_spin.setValue(max_res)
                else:
                    self.max_resolution_toggle.setChecked(False)
            if "pre_downscale" in settings:
                raw_downscale = str(settings["pre_downscale"]).strip()
                if raw_downscale in {"0", "1", "1:1"}:
                    self.pre_downscale_toggle.setChecked(False)
                else:
                    ratio = f"1:{raw_downscale}" if ":" not in raw_downscale else raw_downscale
                    if ratio in {"1:2", "1:3", "1:4", "1:5"}:
                        self.pre_downscale_combo.setCurrentText(ratio)
                        self.pre_downscale_toggle.setChecked(True)
            if "dit_model" in settings:
                display = _display_name(str(settings["dit_model"]))
                idx = self.dit_model_combo.findText(display)
                if idx >= 0:
                    self.dit_model_combo.setCurrentIndex(idx)
                else:
                    self.dit_model_combo.setCurrentText(display)
            if "attention_mode" in settings:
                self.attention_mode_combo.setCurrentText(str(settings["attention_mode"]))
            if "auto_tune" in settings:
                self.auto_tune_toggle.setChecked(bool(settings["auto_tune"]))
            if "cache_dit" in settings:
                self.cache_dit_toggle.setChecked(bool(settings["cache_dit"]))
            if "cache_vae" in settings:
                self.cache_vae_toggle.setChecked(bool(settings["cache_vae"]))
            if "use_10bit" in settings:
                self.use_10bit_toggle.setChecked(bool(settings["use_10bit"]))
            if "debug" in settings:
                self.debug_toggle.setChecked(bool(settings["debug"]))
            if "uniform_batch_size" in settings:
                self.uniform_batch_toggle.setChecked(bool(settings["uniform_batch_size"]))
            if "batch_size" in settings:
                self.batch_size_spin.setValue(int(settings["batch_size"]))
            if "only_frames" in settings:
                self.only_frames_edit.setText(str(settings["only_frames"]))
            if "vae_encode_tiled" in settings:
                self.vae_encode_tiled_toggle.setChecked(bool(settings["vae_encode_tiled"]))
            if "vae_encode_tile_size" in settings:
                self.vae_encode_tile_size_spin.setValue(int(settings["vae_encode_tile_size"]))
            if "vae_encode_tile_overlap" in settings:
                self.vae_encode_tile_overlap_spin.setValue(int(settings["vae_encode_tile_overlap"]))
            if "vae_decode_tiled" in settings:
                self.vae_decode_tiled_toggle.setChecked(bool(settings["vae_decode_tiled"]))
            if "vae_decode_tile_size" in settings:
                self.vae_decode_tile_size_spin.setValue(int(settings["vae_decode_tile_size"]))
            if "vae_decode_tile_overlap" in settings:
                self.vae_decode_tile_overlap_spin.setValue(int(settings["vae_decode_tile_overlap"]))
            if "color_correction" in settings:
                self.color_correction_combo.setCurrentText(str(settings["color_correction"]))
            if "input_noise_scale" in settings:
                self.input_noise_slider.setValue(int(round(float(settings["input_noise_scale"]) * 100)))
            if "latent_noise_scale" in settings:
                self.latent_noise_slider.setValue(int(round(float(settings["latent_noise_scale"]) * 100)))
            if "temporal_overlap" in settings:
                self.temporal_overlap_spin.setValue(int(settings["temporal_overlap"]))
            if "prepend_frames" in settings:
                self.prepend_frames_spin.setValue(int(settings["prepend_frames"]))
            if "dit_offload_device" in settings:
                self.dit_offload_device_combo.setCurrentText(str(settings["dit_offload_device"]))
            if "vae_offload_device" in settings:
                self.vae_offload_device_combo.setCurrentText(str(settings["vae_offload_device"]))
            if "tensor_offload_device" in settings:
                self.tensor_offload_device_combo.setCurrentText(str(settings["tensor_offload_device"]))
            if "blocks_to_swap" in settings:
                self.blocks_to_swap_spin.setValue(int(settings["blocks_to_swap"]))
            if "swap_io_components" in settings:
                self.swap_io_components_check.setChecked(bool(settings["swap_io_components"]))
            if "compile_dit" in settings:
                self.compile_dit_check.setChecked(bool(settings["compile_dit"]))
            if "compile_vae" in settings:
                self.compile_vae_check.setChecked(bool(settings["compile_vae"]))
            if "compile_backend" in settings:
                self.compile_backend_combo.setCurrentText(str(settings["compile_backend"]))
            if "compile_mode" in settings:
                self.compile_mode_combo.setCurrentText(str(settings["compile_mode"]))
        except Exception:
            pass
        self._emit_settings_changed()

    def _label(self, text: str) -> QLabel:
        return QLabel(text)

    def _set_setting_tooltip(self, widget: QWidget, tooltip: str, label: Optional[QLabel] = None) -> None:
        if not tooltip:
            return
        widget.setToolTip(tooltip)
        if widget not in self._tooltip_widgets:
            widget.installEventFilter(self)
            self._tooltip_widgets.add(widget)
        if label is not None:
            label.setToolTip(tooltip)
            if label not in self._tooltip_widgets:
                label.installEventFilter(self)
                self._tooltip_widgets.add(label)

    def _add_row(self, form: QFormLayout, label_text: str, field: QWidget, tooltip: str) -> QLabel:
        label = self._label(label_text)
        form.addRow(label, field)
        self._set_setting_tooltip(field, tooltip, label=label)
        return label

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

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if isinstance(obj, QWidget) and obj in self._tooltip_widgets:
            event_type = event.type()
            if event_type in (QEvent.Enter, QEvent.FocusIn):
                self._tooltip_target = obj
                self._tooltip_timer.start()
            elif event_type in (QEvent.Leave, QEvent.FocusOut, QEvent.Hide):
                if self._tooltip_target is obj:
                    self._tooltip_target = None
                    self._tooltip_timer.stop()
                QToolTip.hideText()
            elif event_type == QEvent.ToolTip:
                return True
        return super().eventFilter(obj, event)

    def _show_delayed_tooltip(self) -> None:
        target = self._tooltip_target
        if target is None:
            return
        text = target.toolTip()
        if not text:
            return
        if not target.underMouse():
            return
        QToolTip.showText(QCursor.pos(), text, target)

    def _populate_cuda_devices(self) -> None:
        self.cuda_device_list.clear()
        gpu_ids = ["0"]
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                gpu_ids = [str(index) for index in range(max(1, count))]
        except Exception:
            gpu_ids = ["0"]
        for value in gpu_ids:
            item = QListWidgetItem(f"GPU {value}")
            item.setData(Qt.UserRole, value)
            self.cuda_device_list.addItem(item)
        # Add CPU option
        cpu_item = QListWidgetItem("CPU (slow, for testing)")
        cpu_item.setData(Qt.UserRole, "cpu")
        self.cuda_device_list.addItem(cpu_item)
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
        self.max_resolution_toggle = self._toggle(True)
        self.max_resolution_toggle.toggled.connect(self._update_resolution_mode)
        self.max_resolution_spin = self._spin(128, 7680, 3840, 1)
        self.pre_downscale_toggle = self._toggle(False)
        self.pre_downscale_toggle.toggled.connect(self._update_pre_downscale_visibility)
        self.pre_downscale_combo = self._combo(["1:2", "1:3", "1:4", "1:5"], "1:2")

        self._add_row(
            form,
            "Mode",
            self.resolution_mode_combo,
            "How output size is defined: fixed pixels, scale multiplier, or preset resolution.",
        )
        self._resolution_label = self._add_row(
            form,
            "Resolution",
            self.resolution_spin,
            "Target short-side resolution in pixels when Mode is pixel.",
        )
        self._resolution_scale_label = self._add_row(
            form,
            "Scale",
            self.resolution_scale_combo,
            "Upscale multiplier when Mode is xtimes.",
        )
        self._resolution_presets_label = self._add_row(
            form,
            "Preset",
            self.resolution_presets_combo,
            "Common output resolution presets.",
        )
        self._add_row(
            form,
            "Max resolution",
            self.max_resolution_toggle,
            "Enable a hard cap for output resolution regardless of mode.",
        )
        self._add_row(
            form,
            "Max px",
            self.max_resolution_spin,
            "Maximum allowed output short-side pixels when Max resolution is enabled.",
        )
        self._add_row(
            form,
            "Pre-downscale",
            self.pre_downscale_toggle,
            "Enable pre-downscaling before processing to reduce VRAM usage.",
        )
        self._pre_downscale_ratio_label = self._add_row(
            form,
            "Downscale ratio",
            self.pre_downscale_combo,
            "Pre-downscale ratio when enabled. Larger reduction lowers VRAM use but can lose detail.",
        )

    def _build_model_group(self) -> None:
        form = self._group("MODEL & PERFORMANCE", advanced_only=True)
        models = self._discover_models()
        default_display = "3B Q8"
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

        self._add_row(
            form,
            "DiT model",
            self.dit_model_combo,
            "AI model for upscaling. 3B is faster; 7B can improve detail quality.",
        )
        self._add_row(
            form,
            "Attention",
            self.attention_mode_combo,
            "Attention backend implementation. sage_attn_3 is generally the recommended default.",
        )
        self._add_row(
            form,
            "Auto tune",
            self.auto_tune_toggle,
            "Automatic OOM recovery by reducing workload and retrying when GPU memory is exceeded.",
        )
        self._add_row(
            form,
            "Cache DiT",
            self.cache_dit_toggle,
            "Keep DiT resources cached between phases for speed at the cost of extra VRAM.",
        )
        self._add_row(
            form,
            "Cache VAE",
            self.cache_vae_toggle,
            "Keep VAE resources cached between phases for speed at the cost of extra VRAM.",
        )
        self._add_row(
            form,
            "10-bit output",
            self.use_10bit_toggle,
            "Prefer 10-bit output formats when available to reduce banding artifacts.",
        )
        self._add_row(
            form,
            "Debug",
            self.debug_toggle,
            "Enable verbose debug logging and diagnostics output.",
        )

    def _build_runtime_group(self) -> None:
        form = self._group("RUNTIME", advanced_only=True)
        self.uniform_batch_toggle = self._toggle(True)
        self.batch_size_spin = self._spin(1, 200, 81, 4)
        self.load_cap_spin = self._spin(0, 1000, 0)
        self.skip_first_frames_spin = self._spin(0, 1000, 0)
        self.only_frames_edit = _FrameListLineEdit(self)
        self.only_frames_edit.setPlaceholderText("e.g. 1,5,9")
        self.only_frames_edit.textChanged.connect(self._emit_settings_changed)
        self.only_frames_edit.validated.connect(self._emit_settings_changed)

        self._add_row(
            form,
            "Uniform batch",
            self.uniform_batch_toggle,
            "Keep all processing batches the same size for predictable VRAM usage.",
        )
        self._add_row(
            form,
            "Batch size",
            self.batch_size_spin,
            "Number of frames processed simultaneously. Higher is faster but uses more VRAM.",
        )
        self._add_row(
            form,
            "Load cap",
            self.load_cap_spin,
            "Maximum number of frames to process from the source (0 means no cap).",
        )
        self._add_row(
            form,
            "Skip first frames",
            self.skip_first_frames_spin,
            "Number of initial frames to skip before processing begins.",
        )
        self._add_row(
            form,
            "Only frames",
            self.only_frames_edit,
            "Comma-separated list of exact frame indices to process.",
        )

    def _build_batch_group(self) -> None:
        form = self._group("QUALITY")
        self.color_correction_combo = self._combo(
            ["none", "lab", "wavelet", "wavelet_adaptive", "hsv", "adain"],
            "none",
        )
        self.input_noise_slider = self._slider_row(0)
        self.latent_noise_slider = self._slider_row(0)
        self.temporal_overlap_spin = self._spin(0, 64, 8)
        self.prepend_frames_spin = self._spin(0, 16, 4)

        self._add_row(
            form,
            "Color correction",
            self.color_correction_combo,
            "Color matching mode for output frames. lab usually preserves source color most accurately.",
        )
        self._add_row(
            form,
            "Input noise",
            self.input_noise_slider,
            "Noise injected into input frames before processing. 0 disables extra input noise.",
        )
        self._add_row(
            form,
            "Latent noise",
            self.latent_noise_slider,
            "Noise scale applied in latent space during generation.",
        )
        self._add_row(
            form,
            "Temporal overlap",
            self.temporal_overlap_spin,
            "Overlap between adjacent batch segments to improve temporal consistency.",
        )
        self._add_row(
            form,
            "Prepend frames",
            self.prepend_frames_spin,
            "Number of preceding frames used as temporal context for each segment.",
        )

    def _build_chunk_group(self) -> None:
        form = self._group("CHUNK PROCESSING")
        self.chunk_enabled_toggle = self._toggle(False)
        self.chunk_minutes_spin = self._spin(1, 5, 3)
        self.chunk_enabled_toggle.toggled.connect(self._update_chunk_visibility)

        self._add_row(
            form,
            "Enable chunks",
            self.chunk_enabled_toggle,
            "Split long videos into timed segments and process them one at a time.",
        )
        self._chunk_minutes_label = self._add_row(
            form,
            "Chunk size (min)",
            self.chunk_minutes_spin,
            "Duration of each chunk in minutes (1-5). Frames per chunk = minutes × 60 × FPS.",
        )
        self._update_chunk_visibility()

    def _update_chunk_visibility(self) -> None:
        enabled = self.chunk_enabled_toggle.isChecked()
        self._set_field_visible(self.chunk_minutes_spin, enabled)

    def _build_vae_group(self) -> None:
        form = self._group("VAE TILING", advanced_only=True)
        self.vae_encode_tiled_toggle = self._toggle(True)
        self.vae_encode_tiled_toggle.toggled.connect(self._update_vae_controls)
        self.vae_encode_tile_size_spin = self._spin(64, 8192, 1024, 1)
        self.vae_encode_tile_overlap_spin = self._spin(0, 8192, 64, 1)
        self.vae_decode_tiled_toggle = self._toggle(True)
        self.vae_decode_tiled_toggle.toggled.connect(self._update_vae_controls)
        self.vae_decode_tile_size_spin = self._spin(64, 8192, 1024, 1)
        self.vae_decode_tile_overlap_spin = self._spin(0, 8192, 64, 1)

        self._add_row(
            form,
            "Encode tiled",
            self.vae_encode_tiled_toggle,
            "Enable tiled VAE encoding to reduce VRAM usage on high-resolution frames.",
        )
        self._add_row(
            form,
            "Encode tile size",
            self.vae_encode_tile_size_spin,
            "Tile size for VAE encoding. Smaller tiles reduce VRAM but can be slower.",
        )
        self._add_row(
            form,
            "Encode overlap",
            self.vae_encode_tile_overlap_spin,
            "Overlap size between encoding tiles to reduce seam artifacts.",
        )
        form.addRow(_separator())
        self._add_row(
            form,
            "Decode tiled",
            self.vae_decode_tiled_toggle,
            "Enable tiled VAE decoding to lower peak VRAM during frame reconstruction.",
        )
        self._add_row(
            form,
            "Decode tile size",
            self.vae_decode_tile_size_spin,
            "Tile size for VAE decoding. Smaller values use less VRAM and may run slower.",
        )
        self._add_row(
            form,
            "Decode overlap",
            self.vae_decode_tile_overlap_spin,
            "Overlap size between decoding tiles to reduce visible tile boundaries.",
        )

    def _build_quality_group(self) -> None:
        form = self._group("OFFLOAD & BLOCKSWAP", advanced_only=True)
        self.dit_offload_device_combo = self._combo(["none", "cpu"], "none")
        self.vae_offload_device_combo = self._combo(["none", "cpu"], "cpu")
        self.tensor_offload_device_combo = self._combo(["none", "cpu"], "none")
        self.blocks_to_swap_spin = self._spin(0, 36, 0)
        self.swap_io_components_check = self._check(False)

        self._add_row(
            form,
            "DiT offload",
            self.dit_offload_device_combo,
            "Move DiT model blocks to CPU when possible to reduce GPU memory pressure.",
        )
        self._add_row(
            form,
            "VAE offload",
            self.vae_offload_device_combo,
            "Move VAE model data to CPU between phases to save VRAM.",
        )
        self._add_row(
            form,
            "Tensor offload",
            self.tensor_offload_device_combo,
            "Offload intermediate tensors to CPU memory to lower VRAM usage.",
        )
        self._add_row(
            form,
            "Blocks to swap",
            self.blocks_to_swap_spin,
            "Number of model blocks swapped to CPU. Higher saves VRAM but can reduce speed.",
        )
        self._add_row(
            form,
            "Swap I/O components",
            self.swap_io_components_check,
            "Also swap model input/output components when block swapping is enabled.",
        )

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

        self._add_row(
            form,
            "Compile DiT",
            self.compile_dit_check,
            "Enable torch.compile for DiT to improve runtime speed after warmup.",
        )
        self._add_row(
            form,
            "Compile VAE",
            self.compile_vae_check,
            "Enable torch.compile for VAE for potential speed gains after initial compile time.",
        )
        self._add_row(
            form,
            "Compile backend",
            self.compile_backend_combo,
            "torch.compile backend implementation to use for graph execution.",
        )
        self._add_row(
            form,
            "Compile mode",
            self.compile_mode_combo,
            "Compilation optimization profile balancing compile time and runtime speed.",
        )
        self._add_row(
            form,
            "GPU Device(s)",
            self.cuda_device_list,
            "Select one or more GPU devices used for processing. Select CPU for CPU-only processing.",
        )

    def _build_vram_group(self) -> None:
        box = QGroupBox("DEVICE INFO", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(Dims.PADDING_MD, Dims.PADDING_LG, Dims.PADDING_MD, Dims.PADDING_MD)
        box.setStyleSheet(
            f"QGroupBox {{ color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_SMALL + 1}px; "
            f"font-weight: {Fonts.WEIGHT_BOLD}; border: 1px solid {Colors.BORDER}; "
            f"border-radius: {Dims.CORNER_RADIUS_MD}px; margin-top: 8px; padding-top: 8px; }}"
        )
        self._device_info_labels: List[QLabel] = []
        for _ in range(7):
            lbl = QLabel("—", box)
            lbl.setStyleSheet(
                f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_SMALL + 1}px; "
                f"font-weight: {Fonts.WEIGHT_MEDIUM};"
            )
            layout.addWidget(lbl)
            self._device_info_labels.append(lbl)
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
        self._set_field_visible(self.resolution_spin, mode == "pixel")
        self._set_field_visible(self.resolution_scale_combo, mode == "xtimes")
        self._set_field_visible(self.resolution_presets_combo, mode == "presets")
        self._set_field_visible(self.max_resolution_spin, self.max_resolution_toggle.isChecked())
        self._update_pre_downscale_visibility(emit=False)
        self._emit_settings_changed()

    def _update_pre_downscale_visibility(self, _checked: bool = False, emit: bool = True) -> None:
        enabled = self.pre_downscale_toggle.isChecked()
        self._set_field_visible(self.pre_downscale_combo, enabled)
        if emit:
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
        return

    def update_vram_estimate(self) -> None:
        return

    def set_device_info_lines(self, lines: List[str]) -> None:
        if not hasattr(self, "_device_info_labels"):
            return
        padded = list(lines[:len(self._device_info_labels)])
        while len(padded) < len(self._device_info_labels):
            padded.append("—")
        for lbl, text in zip(self._device_info_labels, padded):
            lbl.setText(str(text))

    def set_image_mode(self) -> None:
        """Force settings appropriate for single-image processing."""
        self.batch_size_spin.blockSignals(True)
        self.batch_size_spin.setValue(1)
        self.batch_size_spin.blockSignals(False)
        self.load_cap_spin.blockSignals(True)
        self.load_cap_spin.setValue(0)
        self.load_cap_spin.blockSignals(False)
        self.skip_first_frames_spin.blockSignals(True)
        self.skip_first_frames_spin.setValue(0)
        self.skip_first_frames_spin.blockSignals(False)
        self._emit_settings_changed()

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
            "max_resolution": self.max_resolution_spin.value() if self.max_resolution_toggle.isChecked() else 0,
            "pre_downscale": self.pre_downscale_combo.currentText().split(":", 1)[1] if self.pre_downscale_toggle.isChecked() else "1",
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
            "chunk_enabled": self.chunk_enabled_toggle.isChecked(),
            "chunk_minutes": self.chunk_minutes_spin.value() if self.chunk_enabled_toggle.isChecked() else 0,
        }
