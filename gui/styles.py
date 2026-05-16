"""
Dark Mode Stylesheet – Topaz-like premium theme for SeedVR2 GUI.
"""

DARK_STYLESHEET = """
QWidget {
    background-color: #111214;
    color: #E3E4E6;
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #111214;
}

QMenuBar, QMenu {
    background-color: #1A1C1E;
    color: #E3E4E6;
    border: 1px solid #272A2D;
}
QMenuBar::item:selected, QMenu::item:selected {
    background-color: #0052CC;
    color: #FFFFFF;
    border-radius: 5px;
}

QSplitter::handle {
    background-color: #272A2D;
}
QSplitter::handle:hover {
    background-color: #0052CC;
}

QGroupBox {
    background-color: #1A1C1E;
    border: 1px solid #2A2D31;
    border-radius: 8px;
    margin-top: 14px;
    padding: 8px 10px;
    font-weight: 600;
    color: #BFD2F5;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -2px;
    padding: 0 4px;
    color: #BFD2F5;
}

QScrollArea {
    border: 1px solid #25282D;
    border-radius: 8px;
    background-color: #17191C;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #141619;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #2D3238;
    border-radius: 7px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background-color: #0052CC;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0px;
    height: 0px;
}

QLabel {
    color: #E3E4E6;
    background-color: transparent;
}
QLabel#header_label {
    font-size: 18px;
    font-weight: 700;
    color: #F5F6F7;
}
QLabel#subheader_label {
    color: #AAB0B7;
    font-size: 11px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #141619;
    border: 1px solid #2B2E33;
    border-radius: 7px;
    padding: 6px 9px;
    color: #E3E4E6;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #0052CC;
}
QComboBox::drop-down {
    border-left: 1px solid #2B2E33;
    width: 22px;
}
QComboBox QAbstractItemView {
    background-color: #141619;
    border: 1px solid #2B2E33;
    selection-background-color: #0052CC;
    selection-color: #FFFFFF;
}

QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3A3F45;
    border-radius: 4px;
    background-color: #141619;
}
QCheckBox::indicator:checked {
    background-color: #0052CC;
    border-color: #0052CC;
}

QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #2A2D31;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #0052CC;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #D7DBE0;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QPushButton {
    background-color: #23272D;
    border: 1px solid #2E3339;
    border-radius: 8px;
    padding: 7px 14px;
    color: #E3E4E6;
    min-height: 28px;
}
QPushButton:hover {
    border-color: #0052CC;
    background-color: #2A2F36;
}
QPushButton:pressed {
    background-color: #20242A;
}
QPushButton:disabled {
    background-color: #1A1D21;
    color: #70757C;
    border-color: #24282D;
}
QPushButton#primary_button {
    background-color: #0052CC;
    border-color: #0052CC;
    color: #FFFFFF;
    font-weight: 700;
    min-height: 36px;
}
QPushButton#primary_button:hover {
    background-color: #1B65D6;
    border-color: #1B65D6;
}
QPushButton#danger_button {
    background-color: #8A2B2B;
    border-color: #8A2B2B;
    color: #FFFFFF;
}
QPushButton[flat="true"] {
    background: transparent;
    border: none;
    color: #B8BDC4;
    padding: 4px 8px;
}
QPushButton[flat="true"]:hover {
    background: #1A1C1E;
    border: none;
    color: #FFFFFF;
}

QProgressBar {
    border: 1px solid #2A2D31;
    border-radius: 7px;
    background-color: #141619;
    color: #E3E4E6;
    text-align: center;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #0052CC;
    border-radius: 7px;
}

QTextEdit {
    background-color: #101215;
    border: 1px solid #25292E;
    border-radius: 7px;
    color: #C8E7C8;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}

QVideoWidget, QGraphicsView, QStackedWidget {
    background-color: #0F1012;
    border: 1px solid #25282D;
    border-radius: 8px;
}

QToolTip {
    background-color: #1A1C1E;
    color: #E3E4E6;
    border: 1px solid #2A2D31;
    border-radius: 6px;
    padding: 4px 8px;
}
"""
