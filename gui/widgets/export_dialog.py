"""Modal export-settings dialog (Topaz-style).  No LUT functionality."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..theme import Colors, Dims, Fonts
from ..export_encoder import get_export_extension
from .button3d import Button3D

# ---------------------------------------------------------------------------
# Codec / container / profile configuration
# ---------------------------------------------------------------------------

CODECS: List[str] = [
    "ProRes",
    "DNxHR",
    "H264",
    "H265",
    "AV1",
    "FFV1",
    "VP9",
    "QuickTime V210",
    "QuickTime R210",
    "QuickTime Animation",
]

# container_options, locked (single forced container)
CODEC_CONTAINERS: Dict[str, tuple] = {
    "ProRes": (["MOV"], True),
    "DNxHR": (["MOV"], True),
    "H264": (["MP4", "MKV", "AVI"], False),
    "H265": (["MP4", "MKV"], False),
    "AV1": (["MP4", "MKV", "WebM"], False),
    "FFV1": (["MKV", "AVI"], False),
    "VP9": (["WebM", "MKV"], False),
    "QuickTime V210": (["MOV"], True),
    "QuickTime R210": (["MOV"], True),
    "QuickTime Animation": (["MOV"], True),
}

CODEC_PROFILES: Dict[str, List[str]] = {
    "H264": ["Baseline", "Main", "High"],
    "H265": ["Main", "Main10", "Main12"],
    "ProRes": ["Proxy", "LT", "Standard", "HQ", "4444", "4444 XQ"],
    "DNxHR": ["LB", "SQ", "HQ", "HQX", "444"],
    # The following hide the profile row entirely.
    "AV1": [],
    "FFV1": [],
    "VP9": [],
    "QuickTime V210": [],
    "QuickTime R210": [],
    "QuickTime Animation": [],
}

AUDIO_MODES: List[str] = [
    "None",
    "Copy from source (passthrough)",
    "AAC 128 kbps",
    "AAC 192 kbps",
    "AAC 256 kbps",
    "AAC 320 kbps",
    "FLAC (lossless)",
]

IMAGE_FORMATS: List[str] = ["TIFF", "PNG", "JPEG", "EXR", "DPX"]

IMAGE_BIT_DEPTHS: Dict[str, List[str]] = {
    "TIFF": ["8-bit", "16-bit"],
    "PNG": ["8-bit", "16-bit"],
    "JPEG": ["8-bit"],
    "EXR": ["16-bit (half)", "32-bit (float)"],
    "DPX": ["8-bit", "10-bit", "16-bit"],
}

# Quality-level values per codec family.  Keyed by (family, hw).
QUALITY_VALUES = {
    ("H264", "NVENC"): ("CQ", 28, 23, 18),
    ("H265", "NVENC"): ("CQ", 28, 23, 18),
    ("H264", "QSV"): ("ICQ", 28, 23, 18),
    ("H265", "QSV"): ("ICQ", 28, 23, 18),
    ("H264", "AMF"): ("QP", 28, 23, 18),
    ("H265", "AMF"): ("QP", 28, 23, 18),
    ("H264", "SW"): ("CRF", 28, 23, 18),
    ("H265", "SW"): ("CRF", 28, 23, 18),
    ("AV1", "SW"): ("CRF", 35, 25, 18),
    ("VP9", "SW"): ("CRF", 35, 25, 18),
}

QUALITY_LABELS = {
    "ProRes": ("Proxy", "HQ", "4444"),
    "DNxHR": ("LB", "SQ", "HQX"),
    "FFV1": ("Level 1", "Level 3", "Level 4"),
}


# ---------------------------------------------------------------------------
# Hardware encoder detection (run once, cached)
# ---------------------------------------------------------------------------

_HW_CACHE: Optional[Dict[str, bool]] = None


def detect_hardware_encoders() -> Dict[str, bool]:
    """Detect available ffmpeg hardware encoders once and cache the result."""
    global _HW_CACHE
    if _HW_CACHE is not None:
        return _HW_CACHE

    result = {
        "h264_nvenc": False, "hevc_nvenc": False,
        "h264_qsv": False, "hevc_qsv": False,
        "h264_amf": False, "hevc_amf": False,
    }
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            flags = 0x08000000 if sys.platform == "win32" else 0
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=False, creationflags=flags,
            ).stdout
            for enc in result:
                if enc in out:
                    result[enc] = True
        except Exception:
            pass
    _HW_CACHE = result
    return result


def best_hw_variant(codec: str) -> str:
    """Return the best available HW variant label for H264/H265 (or 'SW')."""
    hw = detect_hardware_encoders()
    fam = "h264" if codec == "H264" else "hevc"
    if hw.get(f"{fam}_nvenc"):
        return "NVENC"
    if hw.get(f"{fam}_qsv"):
        return "QSV"
    if hw.get(f"{fam}_amf"):
        return "AMF"
    return "SW"


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background-color: {Colors.BORDER};")
    return line


class ExportDialog(QDialog):
    """Modal export-settings dialog emitting ``export_confirmed(dict)``."""

    export_confirmed = Signal(dict)

    def __init__(
        self,
        parent=None,
        default_dir: str = "",
        default_name: str = "seedvr2_output",
        frame_count: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        trim_in: int = 0,
        trim_out: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Settings")
        self.setModal(True)
        self.setMinimumWidth(440)

        self._default_dir = default_dir
        self._frame_count = frame_count
        self._width = width
        self._height = height
        self._fps = fps if fps > 0 else 30.0
        self._trim_in = trim_in
        self._trim_out = trim_out

        root = QVBoxLayout(self)
        root.setContentsMargins(Dims.PADDING_XL, Dims.PADDING_LG, Dims.PADDING_XL, Dims.PADDING_LG)
        root.setSpacing(Dims.PADDING_MD)

        title = QLabel("📤  Export Settings")
        title.setStyleSheet(
            f"font-size: {Fonts.SIZE_H1}px; font-weight: {Fonts.WEIGHT_BOLD};"
            f" color: {Colors.TEXT_PRIMARY};"
        )
        root.addWidget(title)

        # ----- OUTPUT -----
        root.addWidget(self._section_label("OUTPUT"))
        root.addWidget(_hline())

        self.filename_edit = QLineEdit(default_name)
        self.filename_edit.textChanged.connect(self._update_summary)
        root.addLayout(self._row("Filename", self.filename_edit))

        self.saveto_combo = QComboBox()
        self.saveto_combo.addItems(["Original Folder", "Custom path…"])
        self.saveto_combo.currentIndexChanged.connect(self._on_saveto_changed)
        root.addLayout(self._row("Save to", self.saveto_combo))

        self.custom_path_edit = QLineEdit(default_dir)
        self.browse_btn = Button3D("Browse", variant="default")
        self.browse_btn.clicked.connect(self._on_browse)
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_path_edit, 1)
        custom_row.addWidget(self.browse_btn)
        self._custom_row_widget = QWidget()
        self._custom_row_widget.setLayout(custom_row)
        root.addWidget(self._custom_row_widget)
        self._custom_row_widget.setVisible(False)

        # Output type.
        root.addWidget(self._section_label("Output type"))
        type_row = QHBoxLayout()
        self.type_video_radio = QRadioButton("Video")
        self.type_image_radio = QRadioButton("Image seq.")
        self.type_video_radio.setChecked(True)
        self._type_group = QButtonGroup(self)
        self._type_group.addButton(self.type_video_radio)
        self._type_group.addButton(self.type_image_radio)
        self.type_video_radio.toggled.connect(self._on_type_changed)
        type_row.addWidget(self.type_video_radio)
        type_row.addWidget(self.type_image_radio)
        type_row.addStretch(1)
        root.addLayout(type_row)

        # ----- VIDEO MODE -----
        self._video_widget = QWidget()
        vlay = QVBoxLayout(self._video_widget)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(Dims.PADDING_SM)

        self.codec_combo = QComboBox()
        self.codec_combo.addItems(CODECS)
        self.codec_combo.setCurrentText("H265")
        self.codec_combo.currentIndexChanged.connect(self._on_codec_changed)
        vlay.addLayout(self._row("Codec", self.codec_combo))

        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._update_summary)
        self._profile_row = self._row("Profile", self.profile_combo)
        vlay.addLayout(self._profile_row)

        # Bitrate mode.
        self._bitrate_widget = QWidget()
        br_inner = QVBoxLayout(self._bitrate_widget)
        br_inner.setContentsMargins(0, 0, 0, 0)
        br_inner.setSpacing(Dims.PADDING_SM)
        vlay.addWidget(self._section_label("Bitrate"))
        br_row = QHBoxLayout()
        self.br_dynamic_radio = QRadioButton("Dynamic")
        self.br_constant_radio = QRadioButton("Constant")
        self.br_dynamic_radio.setChecked(True)
        self._br_group = QButtonGroup(self)
        self._br_group.addButton(self.br_dynamic_radio)
        self._br_group.addButton(self.br_constant_radio)
        self.br_dynamic_radio.toggled.connect(self._on_bitrate_mode_changed)
        br_row.addWidget(self.br_dynamic_radio)
        br_row.addWidget(self.br_constant_radio)
        br_row.addStretch(1)
        br_inner.addLayout(br_row)

        # Constant bitrate input.
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(1, 500)
        self.bitrate_spin.setValue(20)
        self.bitrate_spin.setSuffix(" Mbps")
        self.bitrate_spin.valueChanged.connect(self._update_summary)
        self._bitrate_row = self._row("Target bitrate", self.bitrate_spin)
        br_inner.addLayout(self._bitrate_row)
        vlay.addWidget(self._bitrate_widget)

        # Quality level.
        self._quality_label = self._section_label("Quality level")
        vlay.addWidget(self._quality_label)
        q_row = QHBoxLayout()
        self.q_low_radio = QRadioButton("Low")
        self.q_med_radio = QRadioButton("Medium")
        self.q_high_radio = QRadioButton("High")
        self.q_med_radio.setChecked(True)
        self._q_group = QButtonGroup(self)
        for b in (self.q_low_radio, self.q_med_radio, self.q_high_radio):
            self._q_group.addButton(b)
            b.toggled.connect(self._update_summary)
        self._quality_row_widget = QWidget()
        self._quality_row_widget.setLayout(q_row)
        q_row.addWidget(self.q_low_radio)
        q_row.addWidget(self.q_med_radio)
        q_row.addWidget(self.q_high_radio)
        q_row.addStretch(1)
        vlay.addWidget(self._quality_row_widget)

        self.audio_combo = QComboBox()
        self.audio_combo.addItems(AUDIO_MODES)
        self.audio_combo.setCurrentText("Copy from source (passthrough)")
        self.audio_combo.currentIndexChanged.connect(self._update_summary)
        vlay.addLayout(self._row("Audio mode", self.audio_combo))

        self.container_combo = QComboBox()
        self.container_combo.currentIndexChanged.connect(self._update_summary)
        vlay.addLayout(self._row("Container", self.container_combo))

        root.addWidget(self._video_widget)

        # ----- IMAGE SEQ MODE -----
        self._image_widget = QWidget()
        ilay = QVBoxLayout(self._image_widget)
        ilay.setContentsMargins(0, 0, 0, 0)
        ilay.setSpacing(Dims.PADDING_SM)

        self.start_number_spin = QSpinBox()
        self.start_number_spin.setRange(0, 9999999)
        self.start_number_spin.setValue(0)
        self.start_number_spin.valueChanged.connect(self._update_summary)
        ilay.addLayout(self._row("Start #", self.start_number_spin))

        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(IMAGE_FORMATS)
        self.image_format_combo.currentIndexChanged.connect(self._on_image_format_changed)
        ilay.addLayout(self._row("File type", self.image_format_combo))

        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.currentIndexChanged.connect(self._update_summary)
        ilay.addLayout(self._row("Bit depth", self.bit_depth_combo))

        root.addWidget(self._image_widget)
        self._image_widget.setVisible(False)

        # ----- SUMMARY -----
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            f"background-color: {Colors.BG_LIGHT}; border: 1px solid {Colors.BORDER};"
            f" border-radius: {Dims.CORNER_RADIUS_SM}px; padding: {Dims.PADDING_MD}px;"
            f" color: {Colors.TEXT_SECONDARY};"
        )
        root.addWidget(self.summary_label)

        # ----- BUTTONS -----
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.cancel_btn = Button3D("Cancel", variant="ghost")
        self.cancel_btn.clicked.connect(self.reject)
        self.export_btn = Button3D("▶ Start Export", variant="primary")
        self.export_btn.clicked.connect(self._on_start_export)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.export_btn)
        root.addLayout(btn_row)

        # Initialise dependent widgets.
        self._on_codec_changed()
        self._on_image_format_changed()
        self._on_bitrate_mode_changed()
        self._update_summary()

    # ---------------------------------------------------------------- ui helpers
    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}px;"
            f" font-weight: {Fonts.WEIGHT_BOLD};"
        )
        return lbl

    @staticmethod
    def _row(label: str, widget) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(96)
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return row

    @staticmethod
    def _set_row_visible(row: QHBoxLayout, visible: bool) -> None:
        for i in range(row.count()):
            item = row.itemAt(i).widget()
            if item is not None:
                item.setVisible(visible)

    # ---------------------------------------------------------------- slots
    def _on_saveto_changed(self) -> None:
        self._custom_row_widget.setVisible(self.saveto_combo.currentIndex() == 1)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", self._default_dir)
        if path:
            self.custom_path_edit.setText(path)
        self._update_summary()

    def _on_type_changed(self) -> None:
        is_video = self.type_video_radio.isChecked()
        self._video_widget.setVisible(is_video)
        self._image_widget.setVisible(not is_video)
        self._update_summary()

    def _on_codec_changed(self) -> None:
        codec = self.codec_combo.currentText()

        # Profiles.
        profiles = CODEC_PROFILES.get(codec, [])
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        if profiles:
            self.profile_combo.addItems(profiles)
            if codec == "H264":
                self.profile_combo.setCurrentText("Main")
            elif codec == "H265":
                self.profile_combo.setCurrentText("Main")
            elif codec == "ProRes":
                self.profile_combo.setCurrentText("Standard")
            elif codec == "DNxHR":
                self.profile_combo.setCurrentText("SQ")
        self.profile_combo.blockSignals(False)
        self._set_row_visible(self._profile_row, bool(profiles))

        # Containers.
        containers, locked = CODEC_CONTAINERS.get(codec, (["MP4"], False))
        self.container_combo.blockSignals(True)
        self.container_combo.clear()
        self.container_combo.addItems(containers)
        self.container_combo.setCurrentIndex(0)
        self.container_combo.setEnabled(not locked)
        self.container_combo.blockSignals(False)

        # For ProRes and DNxHR: quality is determined solely by the profile preset.
        # Hide bitrate controls entirely.
        # For FFV1 and uncompressed QT: hide bitrate and CRF quality.
        # For everything else: show dynamic/constant bitrate selector.
        profile_only = codec in ("ProRes", "DNxHR")
        uncompressed = codec in ("QuickTime V210", "QuickTime R210", "QuickTime Animation")
        ffv1 = codec == "FFV1"
        show_bitrate_section = not profile_only and not uncompressed and not ffv1
        self._bitrate_widget.setVisible(show_bitrate_section)

        # Quality labels (depend on codec + hardware).
        self._refresh_quality_labels()
        self._update_summary()

    def _refresh_quality_labels(self) -> None:
        codec = self.codec_combo.currentText()
        # ProRes / DNxHR: quality is the profile — no CRF quality row needed.
        # Uncompressed QT codecs: no quality control at all.
        # FFV1: shows level as quality.
        profile_only = codec in ("ProRes", "DNxHR")
        uncompressed = codec in ("QuickTime V210", "QuickTime R210", "QuickTime Animation")
        ffv1 = codec == "FFV1"

        show_quality = (
            not profile_only
            and not uncompressed
            and (ffv1 or self.br_dynamic_radio.isChecked())
        )
        self._quality_label.setVisible(show_quality)
        self._quality_row_widget.setVisible(show_quality)

        if codec in QUALITY_LABELS:
            low, med, high = QUALITY_LABELS[codec]
        else:
            hw = best_hw_variant(codec) if codec in ("H264", "H265") else "SW"
            key = (codec, hw)
            if key in QUALITY_VALUES:
                prefix, lo, me, hi = QUALITY_VALUES[key]
                low, med, high = f"Low ({prefix} {lo})", f"Medium ({prefix} {me})", f"High ({prefix} {hi})"
            else:
                low, med, high = "Low", "Medium", "High"
        self.q_low_radio.setText(low)
        self.q_med_radio.setText(med)
        self.q_high_radio.setText(high)

    def _on_bitrate_mode_changed(self) -> None:
        dynamic = self.br_dynamic_radio.isChecked()
        self._set_row_visible(self._bitrate_row, not dynamic)
        self._refresh_quality_labels()
        self._update_summary()

    def _on_image_format_changed(self) -> None:
        fmt = self.image_format_combo.currentText()
        depths = IMAGE_BIT_DEPTHS.get(fmt, ["8-bit"])
        self.bit_depth_combo.blockSignals(True)
        self.bit_depth_combo.clear()
        self.bit_depth_combo.addItems(depths)
        self.bit_depth_combo.blockSignals(False)
        self._update_summary()

    # ---------------------------------------------------------------- summary
    def _codec_display(self) -> str:
        codec = self.codec_combo.currentText()
        if codec in ("H264", "H265"):
            variant = best_hw_variant(codec)
            if variant != "SW":
                return f"{codec} ({variant})"
        return codec

    def _quality_level_name(self) -> str:
        if self.q_low_radio.isChecked():
            return "Low"
        if self.q_high_radio.isChecked():
            return "High"
        return "Medium"

    def _quality_value(self) -> str:
        codec = self.codec_combo.currentText()
        idx = {"Low": 0, "Medium": 1, "High": 2}[self._quality_level_name()]
        if codec in QUALITY_LABELS:
            return QUALITY_LABELS[codec][idx]
        hw = best_hw_variant(codec) if codec in ("H264", "H265") else "SW"
        key = (codec, hw)
        if key in QUALITY_VALUES:
            prefix, lo, me, hi = QUALITY_VALUES[key]
            return f"{prefix} {(lo, me, hi)[idx]}"
        return self._quality_level_name()

    def _frames_to_export(self) -> int:
        if self._trim_out > self._trim_in:
            return self._trim_out - self._trim_in + 1
        return self._frame_count

    def _update_summary(self) -> None:
        frames = self._frames_to_export()
        est_minutes = max(1, int(frames / max(1.0, self._fps) / 4)) if frames else 0
        if self.type_video_radio.isChecked():
            codec = self._codec_display()
            container = self.container_combo.currentText()
            quality = self._quality_level_name()
            self.summary_label.setText(
                f"{frames} frames → {codec} {container} @ {quality} quality | "
                f"{self._width}×{self._height} | ~{est_minutes} min estimated"
            )
        else:
            fmt = self.image_format_combo.currentText()
            depth = self.bit_depth_combo.currentText()
            self.summary_label.setText(
                f"{frames} frames → {fmt} sequence ({depth}) | "
                f"{self._width}×{self._height} | starting at #{self.start_number_spin.value()}"
            )

    # ---------------------------------------------------------------- export
    def _output_dir(self) -> str:
        if self.saveto_combo.currentIndex() == 1:
            return self.custom_path_edit.text().strip()
        return self._default_dir

    def _on_start_export(self) -> None:
        out_dir = self._output_dir()
        name = self.filename_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Invalid filename", "Please enter a filename.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Invalid folder", "Output directory does not exist.")
            return
        if not os.access(out_dir, os.W_OK):
            QMessageBox.warning(self, "Not writable", "Output directory is not writable.")
            return

        codec = self.codec_combo.currentText()
        if codec in ("H264", "H265"):
            # If user expects HW but none available, warn (SW fallback is allowed).
            variant = best_hw_variant(codec)
            if variant == "SW" and not shutil.which("ffmpeg"):
                QMessageBox.critical(
                    self, "Encoder unavailable",
                    "ffmpeg was not found and no hardware encoder is available.",
                )
                return

        is_video = self.type_video_radio.isChecked()
        if is_video:
            ext = get_export_extension(codec, self.container_combo.currentText().lower()).lstrip(".")
            output_path = os.path.join(out_dir, f"{name}.{ext}")
            if os.path.exists(output_path):
                if QMessageBox.question(
                    self, "Overwrite?",
                    f"{os.path.basename(output_path)} already exists. Overwrite?",
                ) != QMessageBox.Yes:
                    return
        else:
            output_path = os.path.join(out_dir, name)

        payload = {
            "output_path": output_path,
            "output_type": "video" if is_video else "image_sequence",
            "codec": codec,
            "codec_display": self._codec_display(),
            "profile": self.profile_combo.currentText() if self.profile_combo.count() else "",
            "bitrate_mode": "dynamic" if self.br_dynamic_radio.isChecked() else "constant",
            "quality_level": self._quality_level_name(),
            "quality_value": self._quality_value(),
            "bitrate_mbps": self.bitrate_spin.value(),
            "audio_mode": self.audio_combo.currentText(),
            "container": self.container_combo.currentText(),
            "start_number": self.start_number_spin.value(),
            "image_format": self.image_format_combo.currentText(),
            "bit_depth": self.bit_depth_combo.currentText(),
            "trim_in": self._trim_in,
            "trim_out": self._trim_out,
        }
        self.export_confirmed.emit(payload)
        self.accept()
