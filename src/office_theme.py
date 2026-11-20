"""Office 风格浅色主题。

参照 Microsoft Office 桌面版的视觉语言：白底、蓝灰细边框、
Word 蓝 (#2b579a) 作唯一强调色、分组框带浅色标题、按钮 hover 才显色。

注意：三维视图画布仍由 ``PointCloudRenderer.bg`` 自己填充，
不受这里的 ``QWidget`` 白底影响——点云在深底上对比度更高，
这也和 Office 里嵌入的 3D/图表视窗一致（可切换画布底色）。
"""

from __future__ import annotations

# ---------------------------------------------------------------- 配色 --
BLUE = "#2b579a"            # Office / Word 主色
BLUE_DARK = "#1e3f75"
BLUE_LIGHT = "#e8f0fb"
BLUE_LIGHTER = "#f4f8fd"
BORDER = "#c9ced5"
BORDER_SOFT = "#dfe3e8"
TEXT = "#24292f"
MUTED = "#5a6472"
SURFACE = "#ffffff"
SURFACE_ALT = "#f3f5f8"
TIP_BG = "#ffffff"

# 供像素级回归断言直接用（与 QToolTip 的 background-color 保持一致）
TIP_BG_RGB = (255, 255, 255)

# 三维画布可选底色（裁剪页「画布」下拉；名字 → RGB）
CANVAS_COLORS = {
    "深色（默认）": (32, 36, 42),
    "浅灰": (242, 244, 247),
    "白色": (255, 255, 255),
}


OFFICE_QSS = f"""
/* ---------- 基础 ---------- */
QWidget {{
    background: {SURFACE};
    color: {TEXT};
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
    font-size: 13px;
}}
QMainWindow, QStackedWidget {{ background: {SURFACE_ALT}; }}
QFrame[frameShape="4"] {{ color: {BORDER_SOFT}; }}          /* HLine */
QFrame[frameShape="5"] {{ color: {BORDER_SOFT}; }}          /* VLine */
QSplitter::handle {{ background: {BORDER_SOFT}; }}
QSplitter::handle:horizontal {{ background: {BORDER_SOFT}; width: 6px; }}
QSplitter::handle:vertical {{ background: {BORDER_SOFT}; height: 6px; }}
QScrollArea {{ border: 0; background: transparent; }}

/* ---------- 顶部导航条（蓝底选项卡） ---------- */
QToolBar#navBar {{
    background: {BLUE};
    border: 0;
    spacing: 2px;
    padding: 0px 10px;
    min-height: 42px;
}}
QToolBar#navBar QLabel {{
    background: transparent;
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
    padding: 0px 14px 0px 4px;
}}
QToolBar#navBar QLabel#navVer {{
    color: #c8daf3;
    font-size: 11px;
    font-weight: 400;
    padding: 0px 18px 0px 0px;
}}
QToolButton#navTab {{
    background: transparent;
    color: #dce8f8;
    border: 0px;
    border-bottom: 3px solid transparent;
    border-radius: 0px;
    padding: 10px 20px 8px 20px;
    font-size: 14px;
    font-weight: 600;
}}
QToolButton#navTab:hover {{ background: #3663a8; color: #ffffff; }}
QToolButton#navTab:checked {{
    background: #ffffff;
    color: {BLUE};
    border-bottom: 3px solid #ffffff;
}}

/* ---------- 功能区（Ribbon 工具行） ---------- */
QToolBar {{
    background: {SURFACE};
    border: 0;
    border-bottom: 1px solid {BORDER_SOFT};
    spacing: 5px;
    padding: 5px 8px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 6px 10px;
    color: {TEXT};
}}
QToolBar QToolButton:hover {{ background: {BLUE_LIGHTER}; border-color: {BORDER}; }}
QToolBar QToolButton:checked {{ background: {BLUE_LIGHT}; border-color: {BLUE}; color: {BLUE_DARK}; }}
QToolBar QLabel {{ background: transparent; color: {MUTED}; }}
QToolBar QToolBar#ribbonSep {{ background: {BORDER_SOFT}; max-width: 1px; }}

/* ---------- 通用控件 ---------- */
QToolButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 9px;
    color: {TEXT};
}}
QToolButton:hover {{ background: {BLUE_LIGHTER}; border-color: {BLUE}; }}
QToolButton:checked {{
    background: {BLUE}; border: 1px solid {BLUE_DARK};
    color: #ffffff; font-weight: 700;
}}
QToolButton:!checked {{ font-weight: 400; }}
QToolButton#tool {{ padding: 5px 4px; }}

QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 13px;
    color: {TEXT};
    min-height: 18px;
}}
QPushButton:hover {{ background: {BLUE_LIGHTER}; border-color: {BLUE}; }}
QPushButton:pressed {{ background: {BLUE_LIGHT}; }}
QPushButton:disabled {{ background: {SURFACE_ALT}; color: #9aa2ac; border-color: {BORDER_SOFT}; }}
QPushButton#primary {{ background: {BLUE}; border-color: {BLUE_DARK}; color: #ffffff; font-weight: 600; }}
QPushButton#primary:hover {{ background: #3663a8; }}
QPushButton#primary:disabled {{ background: #a7b8d0; border-color: #a7b8d0; color: #f0f3f8; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {BLUE_LIGHT};
    selection-color: {TEXT};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QPlainTextEdit:focus {{
    border-color: {BLUE};
}}
QComboBox::drop-down {{ border: 0; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {BORDER};
    selection-background-color: {BLUE_LIGHT}; selection-color: {BLUE_DARK};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{ background: {SURFACE_ALT}; border: 0; }}

QListWidget {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px;
}}
QListWidget::item {{ padding: 4px 6px; border-radius: 3px; }}
QListWidget::item:hover {{ background: {BLUE_LIGHTER}; }}
QListWidget::item:selected {{ background: {BLUE_LIGHT}; color: {BLUE_DARK}; }}

QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {BLUE_DARK};
}}

/* 浅色底上的勾选/单选：Office 蓝实心，未选为白底灰框 */
QCheckBox, QRadioButton {{ color: {TEXT}; spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 3px;
    border: 1px solid #8b949f; background: {SURFACE};
}}
QCheckBox::indicator:hover {{ border-color: {BLUE}; background: {BLUE_LIGHTER}; }}
QCheckBox::indicator:checked {{ background: {BLUE}; border: 1px solid {BLUE_DARK}; }}
QRadioButton::indicator {{
    width: 16px; height: 16px; border-radius: 8px;
    border: 1px solid #8b949f; background: {SURFACE};
}}
QRadioButton::indicator:hover {{ border-color: {BLUE}; }}
QRadioButton::indicator:checked {{ background: {BLUE}; border: 4px solid {SURFACE}; outline: 1px solid {BLUE}; }}
QCheckBox:checked, QRadioButton:checked {{ font-weight: 600; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER_SOFT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; margin: -6px 0; border-radius: 7px;
    background: {BLUE}; border: 1px solid {BLUE_DARK};
}}

QProgressBar {{
    background: {SURFACE_ALT}; border: 1px solid {BORDER};
    border-radius: 4px; text-align: center; height: 16px;
}}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 3px; }}

QStatusBar {{
    background: {SURFACE_ALT};
    color: {MUTED};
    border-top: 1px solid {BORDER_SOFT};
}}
QStatusBar QLabel {{ background: transparent; }}

QMenuBar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER_SOFT}; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
QMenuBar::item:selected {{ background: {BLUE_LIGHT}; color: {BLUE_DARK}; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {BLUE_LIGHT}; color: {BLUE_DARK}; }}

QTabWidget::pane {{ border: 1px solid {BORDER_SOFT}; background: {SURFACE}; }}
QTabBar::tab {{
    background: {SURFACE_ALT}; border: 1px solid {BORDER_SOFT};
    padding: 6px 14px; color: {MUTED};
}}
QTabBar::tab:selected {{ background: {SURFACE}; color: {BLUE_DARK}; border-bottom-color: {SURFACE}; }}

/* 悬浮说明气泡：Office 浅色卡片 + 细边框，深色文字 */
QToolTip {{
    background-color: {TIP_BG};
    color: {TEXT};
    border: 1px solid #b9c2ce;
    border-radius: 4px;
    padding: 6px 9px;
    font-size: 12px;
}}

QLabel#hint {{ color: {MUTED}; font-size: 12px; background: transparent; }}
QLabel#title {{ color: {BLUE_DARK}; font-size: 15px; font-weight: 700; background: transparent; }}
QLabel#section {{ color: {BLUE_DARK}; font-weight: 700; background: transparent; }}
"""
