"""
Centralised theme for the 1Click_SeedVR2.5 GUI.

ALL colours, fonts and dimensions live here.  No other module should hard-code
a colour value — import the :class:`Colors`, :class:`Fonts`, :class:`Dims` and
:class:`Anim` constant classes instead.

``generate_stylesheet()`` builds a complete Qt Style Sheet (QSS) string from the
constants below so the whole application can be themed from a single source of
truth.
"""

from __future__ import annotations


class Colors:
    """Application colour palette (Topaz-style dark theme)."""

    BG_DARKEST = "#0b0b10"
    BG_DARK = "#10121a"
    BG_MEDIUM = "#161822"
    BG_LIGHT = "#1c1f2e"
    BG_LIGHTER = "#242738"
    SURFACE = "#1e2133"
    SURFACE_HOVER = "#282c42"
    SURFACE_ACTIVE = "#303550"
    ACCENT = "#5b7fff"
    ACCENT_HOVER = "#7a99ff"
    ACCENT_PRESSED = "#4565d9"
    ACCENT_GLOW = "#5b7fff40"
    ACCENT_SUBTLE = "#5b7fff12"
    SUCCESS = "#34d399"
    SUCCESS_HOVER = "#4ade80"
    SUCCESS_PRESSED = "#22b07a"
    WARNING = "#fbbf24"
    DANGER = "#f87171"
    DANGER_HOVER = "#fca5a5"
    DANGER_PRESSED = "#dc2626"
    TEXT_PRIMARY = "#e6eaf2"
    TEXT_SECONDARY = "#8490a8"
    TEXT_MUTED = "#4a5568"
    TEXT_ACCENT = "#5b7fff"
    BORDER = "#262a3d"
    BORDER_FOCUS = "#5b7fff"
    BORDER_ERROR = "#f87171"
    SHADOW_DARK = "#00000080"
    PREVIEW_BG = "#08080c"
    SCRUB_TRACK = "#1a1d2a"
    SCRUB_FILL = "#5b7fff"
    TRIM_REGION = "#5b7fff25"
    TRIM_HANDLE_IN = "#34d399"
    TRIM_HANDLE_OUT = "#f87171"
    WHITE = "#ffffff"
    BLACK = "#000000"
    SUCCESS_TEXT = "#06281d"


class Fonts:
    """Font families, sizes and weights."""

    FAMILY_PRIMARY = "Segoe UI"
    FAMILY_MONO = "Cascadia Code"

    SIZE_H1 = 16
    SIZE_H2 = 13
    SIZE_BODY = 11
    SIZE_SMALL = 10
    SIZE_TINY = 9

    WEIGHT_LIGHT = 300
    WEIGHT_NORMAL = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700


class Dims:
    """Pixel dimensions used throughout the UI."""

    CORNER_RADIUS_SM = 5
    CORNER_RADIUS_MD = 8
    CORNER_RADIUS_LG = 12

    PADDING_XS = 3
    PADDING_SM = 6
    PADDING_MD = 10
    PADDING_LG = 14
    PADDING_XL = 20

    BUTTON_HEIGHT_SM = 28
    BUTTON_HEIGHT_MD = 34
    BUTTON_HEIGHT_LG = 44

    INPUT_HEIGHT = 32
    PANEL_WIDTH_LEFT = 200
    PANEL_WIDTH_RIGHT = 260
    HEADER_HEIGHT = 42
    FOOTER_HEIGHT = 32
    SCRUBBER_HEIGHT = 36
    TRIM_HEIGHT = 56
    BORDER_WIDTH = 1


class Anim:
    """Animation durations in milliseconds."""

    FAST = 120
    NORMAL = 200
    SLOW = 350


def generate_stylesheet() -> str:
    """Build the complete application QSS from the theme constants."""

    c = Colors
    f = Fonts
    d = Dims

    return f"""
/* ===================== Base ===================== */
QMainWindow, QDialog {{
    background-color: {c.BG_DARK};
}}
QWidget {{
    background-color: transparent;
    color: {c.TEXT_PRIMARY};
    font-family: "{f.FAMILY_PRIMARY}";
    font-size: {f.SIZE_BODY}px;
    font-weight: {f.WEIGHT_NORMAL};
}}

/* ===================== Labels ===================== */
QLabel {{
    background: transparent;
    color: {c.TEXT_PRIMARY};
}}
QLabel[role="h1"] {{
    font-size: {f.SIZE_H1}px;
    font-weight: {f.WEIGHT_BOLD};
}}
QLabel[role="h2"] {{
    font-size: {f.SIZE_H2}px;
    font-weight: {f.WEIGHT_SEMIBOLD};
}}
QLabel[role="muted"] {{
    color: {c.TEXT_SECONDARY};
    font-size: {f.SIZE_SMALL}px;
}}
QLabel[role="accent"] {{
    color: {c.TEXT_ACCENT};
}}

/* ===================== Buttons ===================== */
QPushButton {{
    background-color: {c.SURFACE};
    color: {c.TEXT_PRIMARY};
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_MD}px;
    padding: {d.PADDING_SM}px {d.PADDING_LG}px;
    min-height: {d.BUTTON_HEIGHT_SM}px;
    font-weight: {f.WEIGHT_MEDIUM};
}}
QPushButton:hover {{
    background-color: {c.SURFACE_HOVER};
    border-color: {c.ACCENT};
}}
QPushButton:pressed {{
    background-color: {c.SURFACE_ACTIVE};
}}
QPushButton:disabled {{
    color: {c.TEXT_MUTED};
    background-color: {c.BG_MEDIUM};
    border-color: {c.BORDER};
}}
QPushButton[variant="primary"] {{
    background-color: {c.ACCENT};
    border-color: {c.ACCENT};
    color: #ffffff;
}}
QPushButton[variant="primary"]:hover {{
    background-color: {c.ACCENT_HOVER};
}}
QPushButton[variant="primary"]:pressed {{
    background-color: {c.ACCENT_PRESSED};
}}
QPushButton[variant="danger"] {{
    background-color: {c.DANGER};
    border-color: {c.DANGER};
    color: #ffffff;
}}
QPushButton[variant="danger"]:hover {{
    background-color: {c.DANGER_HOVER};
}}
QPushButton[variant="success"] {{
    background-color: {c.SUCCESS};
    border-color: {c.SUCCESS};
    color: #06281d;
}}
QPushButton[variant="ghost"] {{
    background-color: transparent;
    border-color: transparent;
}}
QPushButton[variant="ghost"]:hover {{
    background-color: {c.SURFACE_HOVER};
}}

/* ===================== Inputs ===================== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {c.BG_LIGHT};
    color: {c.TEXT_PRIMARY};
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_SM}px;
    padding: {d.PADDING_XS}px {d.PADDING_SM}px;
    min-height: {d.INPUT_HEIGHT - 8}px;
    selection-background-color: {c.ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {c.BORDER_FOCUS};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {c.TEXT_MUTED};
    background-color: {c.BG_MEDIUM};
}}
QLineEdit[error="true"] {{
    border-color: {c.BORDER_ERROR};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {c.SURFACE};
    border: none;
    width: 16px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {c.SURFACE_HOVER};
}}

QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox::down-arrow {{
    width: 8px;
    height: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {c.BG_LIGHTER};
    color: {c.TEXT_PRIMARY};
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_SM}px;
    selection-background-color: {c.ACCENT};
    selection-color: #ffffff;
    outline: none;
}}

/* ===================== Sliders ===================== */
QSlider::groove:horizontal {{
    height: 4px;
    background: {c.SCRUB_TRACK};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {c.ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {c.TEXT_PRIMARY};
    border: 2px solid {c.ACCENT};
    width: 12px;
    height: 12px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {c.ACCENT_HOVER};
}}

/* ===================== CheckBox ===================== */
QCheckBox {{
    spacing: {d.PADDING_SM}px;
    color: {c.TEXT_PRIMARY};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_SM}px;
    background-color: {c.BG_LIGHT};
}}
QCheckBox::indicator:hover {{
    border-color: {c.ACCENT};
}}
QCheckBox::indicator:checked {{
    background-color: {c.ACCENT};
    border-color: {c.ACCENT};
}}
QCheckBox:disabled {{
    color: {c.TEXT_MUTED};
}}

/* ===================== RadioButton ===================== */
QRadioButton {{
    spacing: {d.PADDING_SM}px;
    color: {c.TEXT_PRIMARY};
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: 9px;
    background-color: {c.BG_LIGHT};
}}
QRadioButton::indicator:checked {{
    background-color: {c.ACCENT};
    border-color: {c.ACCENT};
}}

/* ===================== GroupBox ===================== */
QGroupBox {{
    background-color: {c.BG_MEDIUM};
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_MD}px;
    margin-top: 16px;
    padding: {d.PADDING_MD}px;
    font-weight: {f.WEIGHT_SEMIBOLD};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {d.PADDING_MD}px;
    top: 2px;
    padding: 0 {d.PADDING_SM}px;
    color: {c.TEXT_SECONDARY};
    font-size: {f.SIZE_SMALL}px;
    font-weight: {f.WEIGHT_BOLD};
}}

/* ===================== Tabs ===================== */
QTabWidget::pane {{
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_MD}px;
    background-color: {c.BG_MEDIUM};
}}
QTabBar::tab {{
    background-color: {c.BG_LIGHT};
    color: {c.TEXT_SECONDARY};
    padding: {d.PADDING_SM}px {d.PADDING_LG}px;
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-bottom: none;
    border-top-left-radius: {d.CORNER_RADIUS_SM}px;
    border-top-right-radius: {d.CORNER_RADIUS_SM}px;
}}
QTabBar::tab:selected {{
    background-color: {c.SURFACE};
    color: {c.TEXT_PRIMARY};
}}
QTabBar::tab:hover {{
    color: {c.TEXT_PRIMARY};
}}

/* ===================== ProgressBar ===================== */
QProgressBar {{
    background-color: {c.SCRUB_TRACK};
    border: none;
    border-radius: {d.CORNER_RADIUS_SM}px;
    text-align: center;
    color: {c.TEXT_PRIMARY};
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {c.ACCENT};
    border-radius: {d.CORNER_RADIUS_SM}px;
}}

/* ===================== ScrollBars ===================== */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c.SURFACE_ACTIVE};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {c.ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {c.SURFACE_ACTIVE};
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {c.ACCENT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ===================== ToolTip ===================== */
QToolTip {{
    background-color: {c.BG_LIGHTER};
    color: {c.TEXT_PRIMARY};
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_SM}px;
    padding: {d.PADDING_XS}px {d.PADDING_SM}px;
}}

/* ===================== Menu ===================== */
QMenu {{
    background-color: {c.BG_LIGHTER};
    color: {c.TEXT_PRIMARY};
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_SM}px;
    padding: {d.PADDING_XS}px;
}}
QMenu::item {{
    padding: {d.PADDING_SM}px {d.PADDING_LG}px;
    border-radius: {d.CORNER_RADIUS_SM}px;
}}
QMenu::item:selected {{
    background-color: {c.ACCENT};
    color: #ffffff;
}}

/* ===================== StatusBar ===================== */
QStatusBar {{
    background-color: {c.BG_DARKEST};
    color: {c.TEXT_SECONDARY};
    border-top: {d.BORDER_WIDTH}px solid {c.BORDER};
}}
QStatusBar::item {{
    border: none;
}}

/* ===================== ScrollArea / Frame ===================== */
QScrollArea {{
    background-color: transparent;
    border: none;
}}
QFrame[role="card"] {{
    background-color: {c.BG_MEDIUM};
    border: {d.BORDER_WIDTH}px solid {c.BORDER};
    border-radius: {d.CORNER_RADIUS_MD}px;
}}
QFrame[role="separator"] {{
    background-color: {c.BORDER};
    max-height: 1px;
    border: none;
}}

/* ===================== ListWidget ===================== */
QListWidget {{
    background-color: {c.BG_DARK};
    border: none;
    outline: none;
}}
QListWidget::item {{
    color: {c.TEXT_PRIMARY};
    border-radius: {d.CORNER_RADIUS_SM}px;
    padding: {d.PADDING_XS}px;
}}
QListWidget::item:hover {{
    background-color: {c.SURFACE_HOVER};
}}
QListWidget::item:selected {{
    background-color: {c.ACCENT_SUBTLE};
    color: {c.TEXT_PRIMARY};
}}
"""
