"""Live hardware utilization panel."""

from __future__ import annotations

import psutil
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from ..theme import Colors, Dims, Fonts

try:
    import pynvml

    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]
    HAS_NVML = False


class DeviceInfoPanel(QWidget):
    """Real-time device monitoring panel."""

    def __init__(self, gpu_index: int = 0, parent=None):
        super().__init__(parent)
        self._gpu_index = gpu_index

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("DEVICE INFO", self)
        group.setStyleSheet(
            f"""
            QGroupBox {{
                font-size: {Fonts.SIZE_SMALL}px;
                font-weight: bold;
                color: {Colors.TEXT_ACCENT};
                border: 1px solid {Colors.BORDER};
                border-radius: {Dims.CORNER_RADIUS_MD}px;
                margin-top: 12px;
                padding: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }}
            """
        )
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        self.label_gpu = self._make_label("GPU: --")
        self.label_gpu_load = self._make_label("GPU Load: --%  |  --°C")
        self.label_vram = self._make_label("VRAM: -- / -- GB")
        self.label_shared_vram = self._make_label("Shared VRAM: -- / -- GB")
        self.label_cpu = self._make_label("CPU: --%  |  --°C")
        self.label_ram = self._make_label("RAM: -- / -- GB")

        for label in (
            self.label_gpu,
            self.label_gpu_load,
            self.label_vram,
            self.label_shared_vram,
            self.label_cpu,
            self.label_ram,
        ):
            group_layout.addWidget(label)

        layout.addWidget(group)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_info)
        self._timer.start()
        self._update_info()

    @staticmethod
    def _make_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(QFont(Fonts.FAMILY_PRIMARY, Fonts.SIZE_SMALL))
        label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; padding: 2px;")
        return label

    def _update_info(self) -> None:
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_temp = self._get_cpu_temp()
            self.label_cpu.setText(f"CPU: {cpu_percent:.0f}%  |  {cpu_temp}")

            ram = psutil.virtual_memory()
            self.label_ram.setText(
                f"RAM: {ram.used / 1e9:.1f} / {ram.total / 1e9:.1f} GB ({ram.percent:.0f}%)"
            )

            if HAS_NVML and pynvml is not None:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                self.label_gpu.setText(f"GPU: {name}")
                self.label_gpu_load.setText(f"GPU Load: {util.gpu}%  |  {temp}°C")
                self.label_vram.setText(
                    f"VRAM: {mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB"
                )
                shared = getattr(mem, "shared", None)
                if shared is None:
                    self.label_shared_vram.setText("Shared VRAM: N/A")
                else:
                    self.label_shared_vram.setText(
                        f"Shared VRAM: {shared / 1e9:.2f} GB"
                    )
            else:
                self.label_gpu.setText("GPU: N/A (pynvml not installed)")
                self.label_gpu_load.setText("GPU Load: --%  |  --°C")
                self.label_vram.setText("VRAM: -- / -- GB")
                self.label_shared_vram.setText("Shared VRAM: N/A")
        except Exception:
            pass

    @staticmethod
    def _get_cpu_temp() -> str:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for entries in temps.values():
                    for entry in entries:
                        if getattr(entry, "current", 0) > 0:
                            return f"{entry.current:.0f}°C"
        except Exception:
            pass
        return "N/A"
