"""
Dark Mode Stylesheet – Topaz Video AI inspired theme for SeedVR2 GUI.
"""

DARK_STYLESHEET = """
QWidget {
    background-color: #1a1a1a;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #141414;
}

/* ── Menu bar ──────────────────────────────────────────────────────────── */
QMenuBar {
    background-color: #1f1f1f;
    color: #cccccc;
    border-bottom: 1px solid #2a2a2a;
    padding: 2px 4px;
}
QMenuBar::item:selected {
    background-color: #2d2d2d;
    border-radius: 4px;
}
QMenu {
    background-color: #252525;
    border: 1px solid #3a3a3a;
}
QMenu::item:selected {
    background-color: #00b4d8;
    color: #ffffff;
}

/* ── Splitter ───────────────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #2a2a2a;
}
QSplitter::handle:hover {
    background-color: #00b4d8;
}

/* ── Group box ──────────────────────────────────────────────────────────── */
QGroupBox {
    background-color: #1f1f1f;
    border: 1px solid #2e2e2e;
    border-radius: 6px;
    margin-top: 14px;
    padding: 6px 8px;
    font-weight: 600;
    color: #a0c4ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -2px;
    padding: 0 4px;
    color: #a0c4ff;
}

/* ── Scroll area ────────────────────────────────────────────────────────── */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: #1a1a1a;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background-color: #3a3a3a;
    border-radius: 6px;
    min-height: 40px;
}
QScrollBar::handle:vertical:hover {
    background-color: #00b4d8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background-color: #1a1a1a;
    height: 12px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background-color: #3a3a3a;
    border-radius: 6px;
    min-width: 40px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #00b4d8;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ── Labels ─────────────────────────────────────────────────────────────── */
QLabel {
    color: #cccccc;
    background-color: transparent;
}
QLabel#header_label {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
}
QLabel#subheader_label {
    color: #888888;
    font-size: 11px;
}
QLabel#section_label {
    color: #a0c4ff;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Line edit / Spin box ───────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #252525;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
    selection-background-color: #00b4d8;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #00b4d8;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #1e1e1e;
    color: #555555;
    border-color: #2a2a2a;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #2d2d2d;
    border-left: 1px solid #3a3a3a;
    border-radius: 0 4px 0 0;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #2d2d2d;
    border-left: 1px solid #3a3a3a;
    border-radius: 0 0 4px 0;
}

/* ── Combo box ──────────────────────────────────────────────────────────── */
QComboBox {
    background-color: #252525;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
    min-height: 24px;
}
QComboBox:focus {
    border: 1px solid #00b4d8;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #3a3a3a;
    border-radius: 0 4px 4px 0;
}
QComboBox::down-arrow {
    image: none;
    width: 10px;
    height: 10px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #888888;
}
QComboBox QAbstractItemView {
    background-color: #252525;
    border: 1px solid #3a3a3a;
    selection-background-color: #00b4d8;
    selection-color: #ffffff;
    outline: none;
}

/* ── Check box ──────────────────────────────────────────────────────────── */
QCheckBox {
    spacing: 6px;
    color: #cccccc;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    background-color: #252525;
}
QCheckBox::indicator:checked {
    background-color: #00b4d8;
    border-color: #00b4d8;
}
QCheckBox::indicator:hover {
    border-color: #00b4d8;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 5px;
    padding: 6px 14px;
    color: #e0e0e0;
    font-weight: 500;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #333333;
    border-color: #00b4d8;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #222222;
}
QPushButton:disabled {
    background-color: #1e1e1e;
    color: #555555;
    border-color: #2a2a2a;
}
QPushButton#primary_button {
    background-color: #00b4d8;
    border-color: #00b4d8;
    color: #ffffff;
    font-weight: 600;
    font-size: 14px;
    min-height: 36px;
    border-radius: 6px;
}
QPushButton#primary_button:hover {
    background-color: #0095b3;
    border-color: #0095b3;
}
QPushButton#primary_button:pressed {
    background-color: #007a94;
}
QPushButton#primary_button:disabled {
    background-color: #1e4a55;
    color: #555555;
    border-color: #1e4a55;
}
QPushButton#danger_button {
    background-color: #8b1a1a;
    border-color: #8b1a1a;
    color: #ffffff;
    font-weight: 600;
    min-height: 36px;
    border-radius: 6px;
}
QPushButton#danger_button:hover {
    background-color: #a52020;
    border-color: #a52020;
}
QPushButton#danger_button:disabled {
    background-color: #2a1a1a;
    color: #555555;
    border-color: #2a1a1a;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
QProgressBar {
    border: 1px solid #2e2e2e;
    border-radius: 5px;
    background-color: #1a1a1a;
    text-align: center;
    color: #e0e0e0;
    font-size: 12px;
    min-height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0077a8, stop:1 #00b4d8);
    border-radius: 5px;
}

/* ── Text edit (console) ────────────────────────────────────────────────── */
QTextEdit {
    background-color: #101010;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    color: #c5f5c5;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    selection-background-color: #00b4d8;
}

/* ── Tab bar ────────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #2a2a2a;
    background-color: #1a1a1a;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #1f1f1f;
    border: 1px solid #2a2a2a;
    padding: 5px 14px;
    color: #888888;
}
QTabBar::tab:selected {
    background-color: #252525;
    color: #00b4d8;
    border-bottom: 2px solid #00b4d8;
}
QTabBar::tab:hover {
    color: #cccccc;
}

/* ── Tool tip ───────────────────────────────────────────────────────────── */
QToolTip {
    background-color: #252525;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 4px 8px;
    opacity: 240;
}

/* ── Separator ──────────────────────────────────────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #2e2e2e;
}
"""
