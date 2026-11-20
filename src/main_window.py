"""主窗口：菜单 / 工具栏 / 三维视图 / 标注面板 / 状态栏。"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QRect, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QActionGroup, QColor, QFont, QIcon, QKeySequence,
                           QPalette, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressDialog, QPushButton, QRadioButton, QScrollArea,
                               QSizePolicy, QSlider, QSpinBox, QSplitter, QStackedWidget,
                               QStatusBar, QToolBar, QToolButton, QVBoxLayout, QWidget,
                               QWidgetAction, QButtonGroup)

from annotation import AnnotationSet, LABEL_NAMES, LABEL_COLORS
from cropping import CropBox, apply_masks, compound_mask, scan_bounds
from crop_dialog import CropDialog
from crop_page import CropPage
import cloudio
from e57io import E57Document, StructureError
from instance_seg import SegmentParams, segment_points
from label_edit_dialog import LabelEditDialog
from office_theme import (BLUE, BLUE_DARK, BLUE_LIGHT, BLUE_LIGHTER, BORDER,
                          BORDER_SOFT, CANVAS_COLORS, OFFICE_QSS, TEXT)
from renderer import PALETTE
from segment_dialog import SegmentDialog
from view3d import (PointCloudView, MODE_NAV, MODE_BRUSH, MODE_ERASE,
                    MODE_BOX, MODE_CROP, MODE_INST, Box3D, instance_box_color)

APP_NAME = "作物点云处理软件"
APP_VER = "2.0"
MODULE_ANNOT = "器官标注"
MODULE_CROP = "点云裁剪"
# 内部英文标识：QSettings 组织名 / e57 库标识沿用旧值，避免用户配置丢失
APP_ID = "PointCloudAnnotator"

# Office 风格浅色主题定义在 office_theme.py；旧名 DARK_QSS 保留为别名，
# 避免外部脚本/回归用例引用失效。改配色只需改那一个文件。
DARK_QSS = OFFICE_QSS


def make_swatch(color) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(QColor(*color))
    return QIcon(pm)


# 每个器官类别按钮的悬浮说明（LabelButton 用）
LABEL_TIPS = {
    1: "穗 —— 麦穗。标完可用「穗实例分割」自动拆成一根根独立麦穗并编号，\n"
       "分错时用「实例」工具框选合并/拆分。",
    2: "叶 —— 叶片。实例模式下每落一笔生成一个叶片实例编号。",
    3: "茎 —— 茎秆（含叶鞘、穗下节间）。",
    4: "其他 —— 不属于穗/叶/茎的东西：芒、支撑杆、地面碎片、杂株等。",
}


class ScanList(QListWidget):
    """扫描/实例列表。

    ``QListWidget`` 默认 sizeHint 高达 256 px，放进侧栏滚动区会把整块内容
    撑出窗口，导致 1080p 也要滚动。「想要的初始高度」与实际高度分开：
    这里给一个小的 sizeHint，有富余空间时由 stretch 自动撑满。
    """

    def sizeHint(self) -> QSize:
        return QSize(220, 120)


class LabelButton(QToolButton):
    """器官类别按钮：选中时整块填充该类别颜色，未选中为 Office 白底按钮。

    必须用**内联 QSS**：全局 ``QToolButton:checked`` 是蓝色，会把类别色覆盖掉。
    """
    def __init__(self, label_id: int, parent=None):
        super().__init__(parent)
        self.label_id = label_id
        txt = LABEL_NAMES[label_id]
        self.setText(txt)
        self.setIcon(make_swatch(LABEL_COLORS[label_id]))
        self.setCheckable(True)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        r, g, b = LABEL_COLORS[label_id]
        self.setStyleSheet(
            f"QToolButton {{ background: #ffffff; border: 1px solid {BORDER};"
            f" border-radius: 4px; padding: 5px 4px; color: {TEXT}; }}"
            f"QToolButton:hover {{ background: {BLUE_LIGHTER}; border-color: {BLUE}; }}"
            f"QToolButton:checked {{ background: rgb({r},{g},{b});"
            f" border: 2px solid {BLUE_DARK}; color: #12151a; font-weight: 700; }}"
        )
        self.setToolTip(f"{LABEL_TIPS.get(label_id, '')}\n"
                        f"当前类别：{txt}（选中后按钮整块填充该类别颜色）")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VER}")
        # 启动由 app.py 调 showMaximized()；这里只是"还原/取消最大化"后的大小，
        # 并给一个足够小的下限，保证 1024×640 的老笔记本上也能完整操作
        self.resize(1500, 940)
        self.setMinimumSize(1000, 640)
        self._split = None
        self._panel_dragged = False    # 用户手动拖过分隔条后就不再自动调宽
        self._page = 0                # 0 = 器官标注，1 = 点云裁剪
        self._canvas_name = "深色（默认）"
        self.setStyleSheet(OFFICE_QSS)

        self.doc: Optional[E57Document] = None
        self.ann: Optional[AnnotationSet] = None
        self.path: Optional[str] = None
        self._dirty = False
        self._undo: List[tuple] = []
        self._selected_sid = 0
        self._queue: List[str] = []
        self._queue_src: List[str] = []
        self.apply_pose = True          # 是否把扫描的 <pose> 烘焙进坐标
        self.ann_mode = "instance"      # 标注方式：instance 实例 / semantic 语义
        self.settings = QSettings(APP_ID, "Main")

        self._restore_last_rect()
        self._build_ui()
        self._build_actions()
        self._update_actions()
        self._apply_page_state()
        self._update_title()
        QTimer.singleShot(60, self._maybe_open_from_args)

    # ------------------------------------------------------- 窗口几何 --

    def _restore_last_rect(self) -> None:
        """恢复上次"非最大化"时的窗口大小（启动后仍会被最大化覆盖）。"""
        r = self.settings.value("last_rect")
        if isinstance(r, QRect) and r.isValid() and r.width() >= 1000 and r.height() >= 640:
            self.setGeometry(r)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._adapt_panel_width()
        self._adapt_density()

    def _adapt_density(self) -> None:
        """窗口不够高时收紧侧栏：藏掉说明文字、减小间距，尽量少让人滚动。"""
        if not getattr(self, "_side_hints", None):
            return
        compact = self.height() < 1180
        if compact == getattr(self, "_compact", None):
            return
        self._compact = compact
        for w in self._side_hints:
            w.setVisible(not compact)
        self._side_lay.setSpacing(6 if compact else 10)
        self.lst.setMinimumHeight(110 if compact else 180)

    def _adapt_panel_width(self) -> None:
        """按窗口宽度给侧栏分一档合适的宽度（用户拖过分隔条后不再干预）。"""
        if self._split is None or self._panel_dragged:
            return
        w = self._split.width() or self.width()
        if w <= 0:
            return
        want = int(min(520, max(320, round(w * 0.21))))
        sizes = self._split.sizes()
        if len(sizes) == 2 and abs(sizes[1] - want) <= 16:
            return                     # 差别太小就不动，避免来回抖
        self._split.setSizes([max(w - want, 1), want])

    def _on_split_moved(self, *_a) -> None:
        self._panel_dragged = True

    # ------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        self.view = PointCloudView(self)
        self.view.annotateRequest.connect(self._on_annotate)
        self.view.hoverInfo.connect(self._on_hover)
        self.view.viewChanged.connect(self._on_view_changed)
        self.view.cropRequest.connect(self._on_crop_request)
        self.view.instanceRequest.connect(self._on_instance_box)
        self.view.escapePressed.connect(lambda: self.set_mode(MODE_NAV))

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.view)
        split.addWidget(self._build_side_panel())
        split.setStretchFactor(0, 1)      # 窗口变大时多出来的宽度给三维视图
        split.setStretchFactor(1, 0)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(6)
        split.setSizes([1140, 360])
        split.setStyleSheet("QSplitter::handle:horizontal { background: #dfe3e8; }")
        split.splitterMoved.connect(self._on_split_moved)
        self._split = split

        # 两个功能页共用一个主窗口，但数据与操作完全隔离：
        #   页 0 = 器官标注（原「三维视图 + 侧栏」）
        #   页 1 = 点云裁剪（CropPage 自带文件列表、视图、参数与保存）
        self.crop_page = CropPage()
        self._pages = QStackedWidget()
        self._pages.addWidget(split)
        self._pages.addWidget(self.crop_page)
        self.setCentralWidget(self._pages)
        # 导航条要先于功能区 addToolBar，才会排在功能区上方（Office 选项卡行）
        self._build_nav_bar()
        # 侧栏按钮：文字宽度定下限（必须在窗口样式表生效后再算，否则 QSS 内边距没算进去）
        self._fix_button_widths(self.btn_nav, self.btn_brush, self.btn_erase,
                               self.btn_box, self.btn_inst, self.btn_crop,
                               self.btn_semantic, self.btn_instance,
                               *self._label_btns.values())
        self._adapt_panel_width()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_file = QLabel("未打开文件")
        # 路径可能很长：允许被压缩，完整信息放 tooltip，别把右侧的渲染耗时挤出去
        self.lbl_file.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_file.setToolTip("当前文件")
        self.lbl_hover = QLabel("")
        self.lbl_perf = QLabel("")
        self.status.addWidget(self.lbl_file, 0)
        self.status.addPermanentWidget(self.lbl_hover, 1)
        self.status.addPermanentWidget(self.lbl_perf, 0)
        self.crop_page.statusMessage.connect(self.status.showMessage)

        self._on_depth_mode()          # 把侧栏里的拾取深度同步给渲染器

    # ------------------------------------------------------- 功能页导航 --

    def _build_nav_bar(self) -> None:
        """Office 风格顶部选项卡行：蓝底白字标题 + 两个功能页按钮。"""
        nav = QToolBar("功能导航")
        nav.setObjectName("navBar")
        nav.setMovable(False)
        nav.setFloatable(False)
        nav.setToolButtonStyle(Qt.ToolButtonTextOnly)
        # QToolBar 默认不横向撑满，蓝色只会在有控件的那一小段出现；
        # 必须显式 Expanding，顶部选项卡行才是一条通栏蓝带。
        nav.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.addToolBar(nav)

        logo = QLabel(APP_NAME)
        # 导航条是蓝底：这几个标签必须显式透明 + 白字，否则会被全局
        # `QWidget { background: #ffffff }` 刷成白块，蓝色只在有控件的地方露出来
        logo.setStyleSheet("color:#ffffff; background:transparent; font-size:15px;"
                           " font-weight:700; padding:0px 14px 0px 4px;")
        ver = QLabel(f"v{APP_VER}")
        ver.setStyleSheet("color:#c8daf3; background:transparent; font-size:11px;"
                          " padding:0px 18px 0px 0px;")
        nav.addWidget(logo)
        nav.addWidget(ver)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self.btn_page_annot = QToolButton()
        self.btn_page_crop = QToolButton()
        for b, text, tip in (
                (self.btn_page_annot, MODULE_ANNOT,
                 "器官标注：给穗 / 叶 / 茎打类别与实例编号，可分割单穗、修改已有标注。\n"
                 "已有标注的点云用这一页处理；标注数据写进 annotationLabel / annotationInstance。"),
                (self.btn_page_crop, MODULE_CROP,
                 "点云裁剪：加载文件夹、按矩形或任意多边形圈区域、绕轴旋转、另存或覆盖保存。\n"
                 "这一页只改几何与文件，不写标注；也可调用配准脚本用 PLY 参考裁剪 E57。")):
            b.setText(text)
            b.setObjectName("navTab")
            b.setCheckable(True)
            b.setFixedHeight(42)
            b.setToolTip(tip)
            self._nav_group.addButton(b)
            nav.addWidget(b)
        self.btn_page_annot.setChecked(True)
        self.btn_page_annot.clicked.connect(lambda: self.switch_page(0))
        self.btn_page_crop.clicked.connect(lambda: self.switch_page(1))

        spacer = QWidget()
        spacer.setStyleSheet("background: transparent;")
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        nav.addWidget(spacer)
        hint = QLabel("两个页面各自管理自己的文件与未保存改动")
        hint.setStyleSheet("color:#c8daf3; background:transparent; font-size:11px;"
                           " padding-right:14px;")
        nav.addWidget(hint)

    def _build_crop_ribbon(self) -> None:
        """裁剪页功能行。刻意用 QToolButton 而非 QAction：

        标注页已有 Ctrl+S / Ctrl+Shift+S / Ctrl+Z 等 ApplicationShortcut，
        这里若再注册同名 QAction 会抢事件；裁剪页的快捷键由 CropPage 内部的
        QShortcut（WidgetWithChildrenShortcut）+ 页面切换时启停来隔离。
        """
        tb = QToolBar("裁剪功能")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.addToolBar(tb)
        self._tb_crop = tb
        cp = self.crop_page
        for text, fn, tip in (
                ("打开文件夹…", cp.open_folder,
                 "选择一个文件夹，递归列出所有支持的点云文件。\n批量裁剪前先在这里定位数据目录。"),
                ("加载选中", cp._load_selected,
                 "把文件列表里选中的点云载入三维视图。\n等价于双击列表项，会先询问是否丢弃未保存改动。"),
                ("适合窗口", cp.fit_view,
                 "让整份点云刚好填满视图。\n旋转或缩放过头后用来复位。"),
                ("应用裁剪", cp.apply_crop,
                 "按当前区域与保留方式真正删点，可用 Ctrl+Z 撤销。\n区域画好后点这里。"),
                ("撤销", cp.undo,
                 "退回上一次裁剪或旋转之前的状态。\n一次操作算一步，可连续撤销。"),
                ("覆盖保存", cp.save_overwrite,
                 "按原格式直接写回当前文件（Ctrl+S），执行前有一次确认。\n会替换磁盘原文件，想留原件请用另存为。"),
                ("另存为…", cp.save_as,
                 "另存为新文件（Ctrl+Shift+S），格式由扩展名决定。\n原文件不变，支持 e57 / ply / pcd / xyz / csv。"),
                ("运行配准裁剪", cp.run_align,
                 "调用 align_crop_pointcloud.py，用点云1(PLY) 对点云2(E57) 配准裁剪。\n"
                 "需先在左侧「配准裁剪」页选好两个文件与 Python 环境。")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            tb.addWidget(b)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.setVisible(False)

    def switch_page(self, index: int) -> None:
        """在「器官标注」与「点云裁剪」之间切换。"""
        idx = 1 if index else 0
        if idx == self._page and self._pages.currentIndex() == idx:
            return
        self._page = idx
        self._pages.setCurrentIndex(idx)
        self._apply_page_state()
        self._update_title()
        self.status.showMessage(
            f"已切换到「{MODULE_CROP if idx else MODULE_ANNOT}」", 2500)

    def _apply_page_state(self) -> None:
        """页面显隐 + 快捷键隔离（同一个 QMainWindow 不能有两套同名快捷键）。"""
        annot = self._page == 0
        if hasattr(self, "_tb_annot"):
            self._tb_annot.setVisible(annot)
        if hasattr(self, "_tb_crop"):
            self._tb_crop.setVisible(not annot)
        self.btn_page_annot.setChecked(annot)
        self.btn_page_crop.setChecked(not annot)
        self.crop_page.set_shortcuts_enabled(not annot)
        if annot:
            for a in self._plain_actions():
                a.setEnabled(True)
            self._update_actions()
        else:
            # 只禁用「挂在动作上的」标注页 QAction（含 ApplicationShortcut）。
            # 绝不能连 QWidgetAction 一起禁用：QToolBar.addWidget() 会把每个
            # 裁剪功能区按钮、以及顶部导航选项卡都包成 QWidgetAction，
            # 一旦禁用，裁剪页的按钮全部变灰、连页签都点不动（2.0 曾踩过）。
            for a in self._plain_actions():
                a.setEnabled(False)

    def _plain_actions(self) -> List[QAction]:
        """本窗口里所有「纯 QAction」（排除 QToolBar/QMenu 承载控件的 QWidgetAction）。"""
        return [a for a in self.findChildren(QAction)
                if not isinstance(a, QWidgetAction)]

    def _build_side_panel(self) -> QWidget:
        # 侧栏内容较高（约 1100 px），小屏或系统缩放 150% 时会被窗口裁掉，
        # 因此整块放进滚动区：高度够时列表自动撑满，不够时出现竖向滚动条。
        body = QWidget()
        body.setMinimumWidth(300)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
        self._side_lay = lay
        self._side_hints = []          # 小窗口时自动隐藏的说明文字

        # --- 标注工具 ---
        gb_tool = QGroupBox("标注工具")
        v = QVBoxLayout(gb_tool)
        v.setSpacing(6)
        row = QHBoxLayout()
        self.btn_nav = QToolButton(); self.btn_nav.setText("浏览")
        self.btn_nav.setToolTip(
            "浏览：左键拖动旋转，右键/中键拖动或 Shift+左键拖动平移，滚轮缩放。\n"
            "此模式不改动任何标注，按 Esc 也能随时切回。")
        self.btn_brush = QToolButton(); self.btn_brush.setText("笔刷")
        self.btn_brush.setToolTip(
            "笔刷：左键涂抹当前类别的器官，右键结束这一笔（一笔 = 一个撤销步）。\n"
            "`[` 缩小 / `]` 放大笔刷，或在下方「笔刷大小」里调；涂错切「橡皮」擦回。")
        self.btn_erase = QToolButton(); self.btn_erase.setText("橡皮")
        self.btn_erase.setToolTip(
            "橡皮：左键擦除笔刷范围内的标注，恢复为「未标注」（类别与实例编号一起清掉），\n"
            "右键结束这一笔，Ctrl+Z 可整笔撤销。")
        self.btn_box = QToolButton(); self.btn_box.setText("框选")
        self.btn_box.setToolTip(
            "框选：拖出一个矩形，松手即把框内点标为当前类别。\n"
            "默认只标最前面「容差」cm 厚的一层（不会误标背面）；\n"
            "按住 Ctrl 拖拽＝穿透全部深度，整株/整片一次标完。")
        self.btn_inst = QToolButton(); self.btn_inst.setText("实例")
        self.btn_inst.setToolTip(
            "手动穗实例：框住一根穗的穗点，松手即归为一个新实例编号。\n"
            "框内只有「穗」类别的点会被编号，其他类别不受影响；\n"
            "把自动分割分错的两部分一起框住即可合并，框住一部分即可拆分。")
        self.btn_crop = QToolButton(); self.btn_crop.setText("裁剪")
        self.btn_crop.setToolTip(
            "裁剪：拖出矩形，松手弹出裁剪对话框，范围已按框选反算好。\n"
            "可保留框内的点（截单株）或删除框内的点（去地面/支架）；\n"
            "也可直接按世界坐标输入范围。一次裁剪可用 Ctrl+Z 撤销。")
        self._tool_group = QButtonGroup(self)
        for b in (self.btn_nav, self.btn_brush, self.btn_erase,
                  self.btn_box, self.btn_inst, self.btn_crop):
            b.setObjectName("tool")
            b.setCheckable(True)
            b.setMinimumHeight(30)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._tool_group.addButton(b)
            row.addWidget(b)
        row.setSpacing(5)
        self.btn_nav.setChecked(True)
        self.btn_nav.clicked.connect(lambda: self.set_mode(MODE_NAV))
        self.btn_brush.clicked.connect(lambda: self.set_mode(MODE_BRUSH))
        self.btn_erase.clicked.connect(lambda: self.set_mode(MODE_ERASE))
        self.btn_box.clicked.connect(lambda: self.set_mode(MODE_BOX))
        self.btn_inst.clicked.connect(lambda: self.set_mode(MODE_INST))
        self.btn_crop.clicked.connect(lambda: self.set_mode(MODE_CROP))
        v.addLayout(row)

        v.addWidget(self._sep())
        # 「器官类别」标题行右侧放语义/实例开关：只改类别 vs 同时发实例号
        hrow = QHBoxLayout()
        hrow.setSpacing(5)
        hrow.addWidget(QLabel("器官类别"))
        hrow.addStretch(1)
        self.btn_semantic = QToolButton(); self.btn_semantic.setText("语义")
        self.btn_semantic.setToolTip(
            "语义标注：只写器官类别，不分配实例编号。\n"
            "适合只做「穗/叶/茎/其他」的像素级分类，或修正导入文件里的类别；\n"
            "被覆盖的点原有实例编号会被清掉（该实例仍然存在，只是少这些点）。")
        self.btn_instance = QToolButton(); self.btn_instance.setText("实例")
        self.btn_instance.setToolTip(
            "实例标注（默认）：每落一笔为本站生成一个新实例编号，\n"
            "用于数穗、数叶，列表里按「类别 #编号 (点数)」逐条列出。")
        self._ann_mode_group = QButtonGroup(self)
        for b in (self.btn_semantic, self.btn_instance):
            b.setObjectName("tool")
            b.setCheckable(True)
            b.setMinimumHeight(24)
            self._ann_mode_group.addButton(b)
            hrow.addWidget(b)
        self.btn_semantic.clicked.connect(lambda: self.set_ann_mode("semantic"))
        self.btn_instance.clicked.connect(lambda: self.set_ann_mode("instance"))
        v.addLayout(hrow)
        # 沿用上次的标注方式（构造期状态栏还没建，quiet 一下）
        self.set_ann_mode(str(self.settings.value("ann_mode", "instance")), quiet=True)

        crow = QHBoxLayout()
        self._label_btns = {}
        self._label_group = QButtonGroup(self)
        for lid in (1, 2, 3, 4):
            b = LabelButton(lid)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._label_group.addButton(b)
            self._label_btns[lid] = b
            crow.addWidget(b)
        self._label_btns[1].setChecked(True)
        for lid, b in self._label_btns.items():
            b.clicked.connect(lambda _=False, x=lid: self.set_label(x))
        v.addLayout(crow)

        v.addWidget(self._sep())
        h = QHBoxLayout()
        h.addWidget(QLabel("笔刷大小"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(6, 180)
        self.spin_brush.setValue(self.view.brush_radius)
        self.spin_brush.setToolTip(
            "笔刷大小：屏幕上笔刷的像素半径，越大一笔覆盖越广（快捷键 `[` 缩小 / `]` 放大）。\n"
            "数值按屏幕像素算，缩放视图后同样的值覆盖的实际范围会变。")
        self.spin_brush.valueChanged.connect(self._on_brush_size)
        h.addWidget(self.spin_brush, 1)
        v.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("拾取深度"))
        self.cmb_depth = QComboBox()
        self.cmb_depth.addItem("表面层", "surface")
        self.cmb_depth.addItem("前 25%", "front25")
        self.cmb_depth.addItem("前 50%", "front50")
        self.cmb_depth.addItem("全部穿透", "all")
        self.cmb_depth.setToolTip(
            "框/笔刷选中的点按相机深度再筛一层：\n"
            "  表面层 —— 只取最前面「容差」cm 厚的一层，避免误标背面的植株\n"
            "  前 25%/50% —— 框内点按深度取前面一段\n"
            "  全部穿透 —— 框内所有点都标注（整株/整片一次性标注时用）\n"
            "任何模式下按住 Ctrl 拖拽都临时等于「全部穿透」。")
        self.cmb_depth.currentIndexChanged.connect(self._on_depth_mode)
        h.addWidget(self.cmb_depth, 1)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.0, 500.0)
        self.spin_tol.setDecimals(1)
        self.spin_tol.setSingleStep(0.5)
        self.spin_tol.setValue(5.0)
        self.spin_tol.setSuffix(" cm")
        self.spin_tol.setMaximumWidth(90)
        self.spin_tol.setToolTip(
            "表面层模式的厚度：只标注距相机最近这一层内的点，比目标器官略厚一点即可。\n"
            "太大会把背面植株一起标上，太小会漏掉器官边缘。")
        self.spin_tol.valueChanged.connect(self._on_depth_tol)
        h.addWidget(self.spin_tol)
        v.addLayout(h)
        # 沿用上次的拾取深度设置
        i = self.cmb_depth.findData(str(self.settings.value("depth_mode", "surface")))
        self.cmb_depth.setCurrentIndex(max(i, 0))
        self.spin_tol.setValue(float(self.settings.value("depth_tol_cm", 5.0)))

        h2 = QHBoxLayout()
        h2.setSpacing(5)
        b_undo = QPushButton("撤销")
        b_undo.setToolTip(
            "撤销上一步（Ctrl+Z），可连续撤销。\n"
            "一笔涂抹、一次框选、一次裁剪、一次分割各算一步（跨多站也只算一步）。")
        b_undo.clicked.connect(self.undo)
        b_clear = QPushButton("清空扫描")
        b_clear.setToolTip(
            "清空当前聚焦扫描的全部标注，可撤销。\n"
            "只清列表里选中的那一站，其他站不动；要按类别清除或整文件清空请用「改标注…」。")
        b_clear.clicked.connect(self.clear_scan)
        self.btn_label_edit = QPushButton("改标注…")
        self.btn_label_edit.setToolTip(
            "查看/修改文件里已有的标注：改类别、去掉实例编号（转语义）、\n"
            "合并实例、删除选中点、按类别整体清除、清空扫描或全部（均可撤销）。")
        self.btn_label_edit.clicked.connect(self.label_edit_dialog)
        for b in (b_undo, b_clear, self.btn_label_edit):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            h2.addWidget(b)
        v.addLayout(h2)
        lay.addWidget(gb_tool)

        # --- 穗实例分割 ---
        gb_seg = QGroupBox("穗实例分割")
        vs2 = QVBoxLayout(gb_seg)
        vs2.setSpacing(6)
        self.btn_seg = QPushButton("穗实例分割…")
        self.btn_seg.setObjectName("primary")
        self.btn_seg.setMinimumHeight(30)
        self.btn_seg.setToolTip(
            "把已标注为「穗」的点按空间连通性自动拆成一个个独立麦穗，\n"
            "并给每一穗分配实例编号（Ctrl+Z 可整体撤销）")
        self.btn_seg.clicked.connect(self.segment_spikes)
        vs2.addWidget(self.btn_seg)
        seg_tip = QLabel("先标注「穗」，再一键分穗；参数可试算后再执行。")
        seg_tip.setObjectName("hint")
        seg_tip.setWordWrap(True)
        vs2.addWidget(seg_tip)
        self._side_hints.append(seg_tip)
        lay.addWidget(gb_seg)

        # --- 扫描与标注列表（放在"显示"之前：标注时最常看的是它，
        #     小屏滚动时先看到它，而不是被挤到最下面） ---
        gb_list = QGroupBox("扫描 / 实例")
        vv = QVBoxLayout(gb_list)
        self.lst = ScanList()
        self.lst.setAlternatingRowColors(False)
        self.lst.setMinimumHeight(180)
        self.lst.setToolTip(
            "扫描 / 实例列表：按站列出每个实例的点数，加粗行是站号。\n"
            "点某一行即「聚焦」该站——实例框、清空扫描都跟着它走；\n"
            "「未分实例」行是只有类别、没有实例编号的点（语义标注的结果）。")
        self.lst.currentItemChanged.connect(self._on_list_sel)
        vv.addWidget(self.lst, 1)
        self.lbl_stat = QLabel("—")
        self.lbl_stat.setObjectName("hint")
        self.lbl_stat.setWordWrap(True)
        vv.addWidget(self.lbl_stat)
        lay.addWidget(gb_list, 1)

        # --- 显示 ---
        gb_view = QGroupBox("显示")
        f = QFormLayout(gb_view)
        f.setLabelAlignment(Qt.AlignLeft)
        f.setSpacing(8)
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["真彩色", "高度", "强度", "标注"])
        self.cmb_color.setToolTip(
            "着色方式：\n"
            "  真彩色 —— 相机 RGB，最真实，适合认器官\n"
            "  高度 —— 按 Z 值渐变，看株高与分层\n"
            "  强度 —— 按回波强度，弱反射（芒、细茎）也能看出轮廓\n"
            "  标注 —— 按器官类别上色（穗红/叶绿/茎蓝/其他黄），标注时最常用")
        self.cmb_color.currentIndexChanged.connect(self._on_color_mode)
        # 着色与点大小并成一行，给下面的标注选项省出高度
        row_color = QHBoxLayout()
        row_color.setSpacing(6)
        row_color.addWidget(self.cmb_color, 3)
        lbl_size = QLabel("点大小")
        lbl_size.setObjectName("hint")
        row_color.addWidget(lbl_size)
        self.sld_size = QSlider(Qt.Horizontal)
        self.sld_size.setRange(8, 40)
        self.sld_size.setValue(14)
        self.sld_size.setToolTip(
            "点大小：每个点在屏幕上的像素直径。点云稀疏时调大更容易看清，\n"
            "标细节（芒、小穗）时调小一点，避免相邻点糊成一片。")
        self.sld_size.valueChanged.connect(self._on_point_size)
        row_color.addWidget(self.sld_size, 4)
        f.addRow("显示", row_color)

        self.cmb_canvas = QComboBox()
        self.cmb_canvas.addItems(list(CANVAS_COLORS) + ["自定义…"])
        self.cmb_canvas.setToolTip(
            "三维视图的画布底色：\n"
            "  深色（默认）—— 真彩色点云对比度最高，长时间标注最舒服\n"
            "  浅灰 / 白色 —— 出图、贴到文档里时用\n"
            "  自定义… —— 打开取色器挑任意颜色")
        self.cmb_canvas.currentTextChanged.connect(self._on_canvas_changed)
        f.addRow("画布", self.cmb_canvas)

        row2 = QHBoxLayout()
        self.chk_all_scans = QCheckBox("显示全部扫描")
        self.chk_all_scans.setChecked(True)
        self.chk_all_scans.setToolTip(
            "勾选（默认）：把文件里所有扫描叠加显示，看整株/整小区。\n"
            "取消勾选：只显示列表里选中的那一站，近距标注时画面更干净、拾取更准。")
        self.chk_all_scans.stateChanged.connect(self._on_scan_filter)
        row2.addWidget(self.chk_all_scans)
        row2.addStretch(1)
        f.addRow(row2)

        row3 = QHBoxLayout()
        self.chk_axes = QCheckBox("坐标轴")
        self.chk_axes.setChecked(True)
        self.chk_axes.setToolTip("在视图右下角显示 X/Y/Z 方向指示器，旋转后也能判断当前朝向")
        self.chk_axes.stateChanged.connect(self._on_overlay)
        self.chk_bbox = QCheckBox("包围盒")
        self.chk_bbox.setToolTip(
            "用红色虚线框出点云的世界包围盒，确认裁剪范围、检查位姿是否拼接正常时用。")
        self.chk_bbox.stateChanged.connect(self._on_overlay)
        self.chk_ibox = QCheckBox("实例框")
        self.chk_ibox.setToolTip(
            "为列表里聚焦的那一站中，每个已编号的穗画彩色立方体框并显示编号。\n"
            "在左侧列表点别的扫描，框会跟着切到那一站。\n"
            "穗分割对话框里点「试算」时，会用虚线框预览将生成的穗。")
        self.chk_ibox.stateChanged.connect(self._on_overlay)
        row3.addWidget(self.chk_axes)
        row3.addWidget(self.chk_bbox)
        row3.addWidget(self.chk_ibox)
        row3.addStretch(1)
        f.addRow(row3)

        # 隐藏：只看剩下的点，便于继续标注/分割其余植株（不改变标注数据）
        row4 = QHBoxLayout()
        self.chk_hide_marked = QCheckBox("隐藏已标注")
        self.chk_hide_marked.setToolTip(
            "把已有类别标签的点全部藏起来，只剩未标注的点——\n"
            "标完一片就勾上，接着标剩下的，不会互相遮挡（不改变标注，只影响显示）。")
        self.chk_hide_marked.stateChanged.connect(self._on_hidden)
        self.chk_hide_cur = QCheckBox("隐藏当前类别")
        self.chk_hide_cur.setToolTip(
            "把当前选中类别（如「穗」）的点藏起来，其余类别照常显示。\n"
            "切换类别按钮时，隐藏的对象会跟着换。")
        self.chk_hide_cur.stateChanged.connect(self._on_hidden)
        row4.addWidget(self.chk_hide_marked)
        row4.addWidget(self.chk_hide_cur)
        row4.addStretch(1)
        f.addRow(row4)
        self.chk_hide_marked.setChecked(
            bool(self.settings.value("hide_marked", False, type=bool)))
        self.chk_hide_cur.setChecked(
            bool(self.settings.value("hide_cur", False, type=bool)))

        self.chk_pose = QCheckBox("应用 E57 位姿（多站拼接）")
        self.chk_pose.setChecked(True)
        self.chk_pose.setToolTip(
            "按文件内每个扫描的 <pose> 把点变换到统一坐标系，与 CloudCompare 一致。\n"
            "关掉则按扫描自身的局部坐标显示（用于排查位姿错误）。")
        self.chk_pose.stateChanged.connect(self._on_pose_toggle)
        f.addRow(self.chk_pose)
        lay.addWidget(gb_view)

        # --- 保存 ---
        gb_save = QGroupBox("保存")
        vs = QVBoxLayout(gb_save)
        row3 = QHBoxLayout()
        self.btn_save = QPushButton("保存为标注点云…")
        self.btn_save.setObjectName("primary")
        self.btn_save.setToolTip(
            "另存为一个新文件，当前打开的文件不受影响（Ctrl+S）。\n"
            "格式由保存对话框里的扩展名决定：e57 / ply / pcd / xyz / csv，\n"
            "标注都会一起写进去（附加属性 label / instance），重开能看到类别与编号。")
        self.btn_save.clicked.connect(self.save_as)
        self.btn_save2 = QPushButton("覆盖保存")
        self.btn_save2.setToolTip(
            "把标注写回当前打开的文件，按原格式保存（Ctrl+Shift+S）。\n"
            "会覆盖原文件——它已被打开修改过，建议先确认原始数据有备份。")
        self.btn_save2.clicked.connect(self.save_overwrite)
        row3.addWidget(self.btn_save, 2)
        row3.addWidget(self.btn_save2, 1)
        vs.addLayout(row3)
        self.btn_close = QPushButton("关闭当前文件 (Ctrl+W)")
        self.btn_close.setToolTip(
            "关闭当前文件（Ctrl+W）：清空视图、标注、撤销栈与队列，回到刚启动时的空白界面。\n"
            "不退出程序，也不删除磁盘上的文件；有未保存标注会先提示。")
        self.btn_close.clicked.connect(self.close_file)
        vs.addWidget(self.btn_close)
        tip = QLabel("支持 e57 / ply / pcd / xyz / csv；标注随几何一起写入，\n"
                     "坐标、颜色、强度完整保留。")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        vs.addWidget(tip)
        self._side_hints.append(tip)

        # 上半部分可滚动，底部「保存」组固定——小屏/高缩放下按钮永远够得着
        scroll = QScrollArea()
        scroll.setObjectName("sideScroll")
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea#sideScroll { background: transparent; border: 0; }"
            "QScrollArea#sideScroll > QWidget > QWidget { background: transparent; }")

        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll, 1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(10, 4, 10, 10)
        fl.addWidget(gb_save)
        outer.addWidget(foot, 0)
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(560)
        self.side_scroll = scroll
        return panel

    @staticmethod
    def _fix_button_widths(*buttons) -> None:
        """按文字实际宽度给按钮设下限，侧栏变窄时不会把文字挤成「…」。"""
        for b in buttons:
            b.ensurePolished()                     # 先把 QSS 算进去再取 sizeHint
            b.setMinimumWidth(max(b.sizeHint().width(), b.minimumWidth()))

    @staticmethod
    def _sep() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color:{BORDER_SOFT};")
        return line

    def _build_actions(self) -> None:
        tb = QToolBar("标注功能")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        tb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # QMainWindow 默认把多个工具栏**并排**放同一行；这里断一次行，
        # 变成「蓝色选项卡行 / 功能区行」两行，才是 Office 的层次。
        self.addToolBarBreak()
        self.addToolBar(tb)
        self._tb_annot = tb

        a_open = QAction("打开点云…", self)
        a_open.setShortcut(QKeySequence.Open)
        a_open.setToolTip(
            "打开点云文件（Ctrl+O）：支持 e57 / ply / pcd / xyz / txt / csv。\n"
            "当前文件有未保存标注时会先提示。")
        a_open.triggered.connect(self.open_file)
        tb.addAction(a_open)

        a_open_dir = QAction("打开文件夹…", self)
        a_open_dir.setShortcut("Ctrl+Shift+O")
        a_open_dir.setToolTip(
            "载入一个文件夹（含子目录）内的全部点云文件，按文件名顺序排成队列逐个处理（Ctrl+Shift+O）。")
        a_open_dir.triggered.connect(self.open_folder)
        tb.addAction(a_open_dir)

        a_close = QAction("关闭文件", self)
        a_close.setShortcut("Ctrl+W")
        a_close.setToolTip("关闭当前点云回到空白界面（不退出程序）；未保存会先提示")
        a_close.triggered.connect(self.close_file)
        tb.addAction(a_close)
        self.act_close = a_close

        a_save = QAction("保存为…", self)
        a_save.setShortcut(QKeySequence.Save)
        a_save.setToolTip(
            "保存为标注点云（Ctrl+S）：另存为新文件，格式可选 e57 / ply / pcd / xyz / csv。\n"
            "不覆盖当前文件——想写回原文件请用「覆盖保存」（Ctrl+Shift+S）。")
        a_save.triggered.connect(self.save_as)
        tb.addAction(a_save)
        self.act_save = a_save

        a_save_ov = QAction("覆盖保存", self)
        a_save_ov.setShortcut("Ctrl+Shift+S")
        a_save_ov.setToolTip("按原格式写回当前打开的文件（Ctrl+Shift+S），会覆盖原文件")
        a_save_ov.triggered.connect(self.save_overwrite)
        tb.addAction(a_save_ov)
        self.act_save_ov = a_save_ov
        tb.addSeparator()

        a_crop = QAction("裁剪…", self)
        a_crop.setShortcut("C")
        a_crop.setToolTip(
            "框选裁剪（快捷键 C），也可直接按世界坐标输入范围。\n"
            "保留框内＝截出单株；删除框内＝去掉地面、支架等干扰。")
        a_crop.triggered.connect(self.crop_box_dialog)
        tb.addAction(a_crop)
        self.act_crop = a_crop

        a_seg = QAction("穗分割…", self)
        a_seg.setShortcut("S")
        a_seg.setToolTip("把已标注为「穗」的点自动拆成独立麦穗并编号（Ctrl+Z 可撤销）")
        a_seg.triggered.connect(self.segment_spikes)
        tb.addAction(a_seg)
        self.act_seg = a_seg

        a_edit = QAction("标注管理…", self)
        a_edit.setShortcut("M")
        a_edit.setToolTip("查看并修改文件里已有的标注：改类别、转语义、合并、删除、按类别清除")
        a_edit.triggered.connect(self.label_edit_dialog)
        tb.addAction(a_edit)
        self.act_label_edit = a_edit
        tb.addSeparator()

        a_fit = QAction("适合窗口", self)
        a_fit.setShortcut("F")
        a_fit.setToolTip("把整块点云缩放到刚好铺满视图（F）；裁剪或换文件后常用")
        a_fit.triggered.connect(self.view.reset_view)
        tb.addAction(a_fit)

        a_top = QAction("俯视", self)
        a_top.setToolTip("俯视：从正上方往下看（XY 平面），看小区内的植株分布与分蘖")
        a_top.triggered.connect(lambda: self._set_view(90, 89))
        tb.addAction(a_top)
        a_side = QAction("侧视", self)
        a_side.setToolTip("侧视：从侧面平视（XZ 平面），看株高与层间遮挡")
        a_side.triggered.connect(lambda: self._set_view(0, 8))
        tb.addAction(a_side)
        a_iso = QAction("等轴", self)
        a_iso.setToolTip("等轴测：45° 斜视，兼顾分布与株型，日常标注最常用")
        a_iso.triggered.connect(lambda: self._set_view(38, 28))
        tb.addAction(a_iso)
        tb.addSeparator()

        a_undo = QAction("撤销", self)
        a_undo.setShortcut(QKeySequence.Undo)
        a_undo.setToolTip(
            "撤销上一步：一笔标注、一次框选、一次裁剪或一次分割（Ctrl+Z），可连续撤销。\n"
            "一次操作跨多个扫描也只算一步。")
        a_undo.triggered.connect(self.undo)
        tb.addAction(a_undo)
        self.act_undo = a_undo
        tb.addSeparator()

        a_next = QAction("下一个文件", self)
        a_next.setShortcut("Ctrl+Right")
        a_next.setToolTip("处理文件夹队列中的下一个文件（Ctrl+Right）；当前文件未保存会先提示")
        a_next.triggered.connect(self.next_in_queue)
        tb.addAction(a_next)
        self.act_next = a_next

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        # 导航条已经有大标题，这里再放一个只用灰色小字标注软件名，
        # 避免上半屏出现两个大小相同的蓝色标题。
        self.lbl_title = QLabel(APP_NAME)
        self.lbl_title.setObjectName("hint")
        tb.addWidget(self.lbl_title)
        self._build_crop_ribbon()

    def _maybe_open_from_args(self) -> None:
        import sys
        for a in sys.argv[1:]:
            if cloudio.is_supported(a) and os.path.isfile(a):
                self.load_file(a)
                return

    def open_file(self) -> None:
        start = self.settings.value("last_dir", r"D:\Data\PC\ResearchPlots")
        path, _ = QFileDialog.getOpenFileName(
            self, "打开点云", start, cloudio.OPEN_FILTER)
        if path:
            self._queue = []
            self.load_file(path)

    def open_folder(self) -> None:
        start = self.settings.value("last_dir", r"D:\Data\PC\ResearchPlots")
        folder = QFileDialog.getExistingDirectory(
            self, "选择含点云的文件夹（含子目录）", start)
        if not folder:
            return
        files = []
        for root, _dirs, names in os.walk(folder):
            for nm in sorted(names):
                if cloudio.is_supported(nm):
                    files.append(os.path.join(root, nm))
        files.sort()
        if not files:
            QMessageBox.information(
                self, APP_NAME,
                "该文件夹内没有找到点云文件。\n支持：" + " / ".join(cloudio.READ_EXTS))
            return
        self.settings.setValue("last_dir", folder)
        # 第一项立即载入，其余进入队列
        self._queue = files[1:]
        self._queue_src = files
        self.load_file(files[0])

    def next_in_queue(self) -> None:
        """处理文件夹队列中的下一个文件。"""
        if not self._queue:
            self.status.showMessage("队列已处理完毕", 2500)
            return
        if self._dirty and not self._confirm_discard():
            return
        nxt = self._queue.pop(0)
        self.load_file(nxt)

    def _queue_status(self) -> str:
        if not self._queue:
            return ""
        total = len(getattr(self, "_queue_src", [])) or (len(self._queue) + 1)
        done = total - len(self._queue)
        return f"　队列 {done}/{total}　剩余 {len(self._queue)}"

    def close_file(self) -> None:
        """关闭当前点云，回到空白界面（不退出程序、不改动磁盘上的文件）。"""
        if self.doc is None and self.ann is None and self.path is None:
            self.status.showMessage("当前没有打开的文件", 1500)
            return
        if self._dirty and not self._confirm_discard():
            return
        self.doc = None
        self.ann = None
        self.path = None
        self._dirty = False
        self._undo.clear()
        self._queue = []
        self._queue_src = []
        self._selected_sid = 0
        self.view.clear_document()
        self.set_mode(MODE_NAV)
        self.cmb_color.setCurrentIndex(0)          # 着色回到真彩色
        self._refresh_list()
        self._update_title()
        self._update_actions()
        self.lbl_file.setText("未打开文件")
        self.lbl_hover.setText("")
        self.lbl_perf.setText("")
        self.status.showMessage("已关闭文件，点「打开点云…」或「打开文件夹…」重新开始", 3000)

    def _update_actions(self) -> None:
        """没有打开文件时把依赖文档的按钮置灰。"""
        has = self.doc is not None
        for w in (getattr(self, "act_close", None), getattr(self, "act_save", None),
                  getattr(self, "act_save_ov", None),
                  getattr(self, "act_crop", None), getattr(self, "act_undo", None),
                  getattr(self, "act_next", None), getattr(self, "act_seg", None),
                  getattr(self, "act_label_edit", None),
                  getattr(self, "btn_save", None), getattr(self, "btn_save2", None),
                  getattr(self, "btn_close", None), getattr(self, "btn_seg", None),
                  getattr(self, "btn_label_edit", None)):
            if w is not None:
                w.setEnabled(has)
        n_scans = len(self.doc.scans) if has else 0
        if hasattr(self, "chk_all_scans"):
            self.chk_all_scans.setEnabled(n_scans > 1)
        # 切到裁剪页时，标注页所有动作（含 ApplicationShortcut）必须保持禁用，
        # 否则 Ctrl+S / Ctrl+Z 会两个页面同时响应。同样只动纯 QAction，
        # 不碰承载裁剪功能区/导航选项卡的 QWidgetAction。
        if getattr(self, "_page", 0) != 0:
            for a in self._plain_actions():
                a.setEnabled(False)

    def load_file(self, path: str, force: bool = False) -> None:
        if self._dirty and not self._confirm_discard():
            return
        dlg = QProgressDialog("正在读取点云…", "取消", 0, 100, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(200)
        dlg.setValue(10)
        QApplication.processEvents()
        try:
            t0 = time.perf_counter()
            is_e57 = cloudio.ext_of(path) == ".e57"
            if is_e57:
                doc = E57Document.read(path, apply_pose=self.apply_pose, force=force)
            else:
                doc = cloudio.load(path)        # ply / pcd / xyz / txt / csv
            dlg.setValue(60)
            QApplication.processEvents()
            if not doc.scans or doc.total_points == 0:
                dlg.close()
                QMessageBox.warning(self, APP_NAME, "这个文件里没有可用的三维点。")
                return
            ann = AnnotationSet.from_scans(doc.scans)
            dlg.setValue(80)
            QApplication.processEvents()
            self.doc, self.ann, self.path = doc, ann, path
            self._selected_sid = 0
            self.view.set_document(doc, ann)
            self._undo.clear()
            dlg.setValue(100)
            dlg.close()
            self.view.fit_and_render()
            self._update_instance_boxes()
            self._apply_hidden()
            self._refresh_list()
            self._update_title()
            el = time.perf_counter() - t0
            self.lbl_file.setText(
                f"{os.path.basename(path)}   ·   {doc.total_points:,} 点   ·   {len(doc.scans)} 扫描"
                + self._queue_status())
            self.lbl_file.setToolTip(f"{path}\n格式：{cloudio.format_name(path)}")
            self.lbl_perf.setText(f"读取 {el*1000:.0f} ms")
            self.settings.setValue("last_dir", os.path.dirname(path))
            self.settings.setValue("last_ext", cloudio.ext_of(path))
            self._dirty = ann.total_marked() > 0
            self.chk_all_scans.setEnabled(len(doc.scans) > 1)
            self.chk_all_scans.setChecked(True)
            # 位姿只对 e57 有意义（其它格式读进来就是已烘焙的坐标）
            self.chk_pose.setEnabled(is_e57)
            self._update_actions()
            if ann.total_marked() > 0:
                # 带标注的文件打开后直接切到"标注"着色，一眼看到哪些点已标
                self.cmb_color.setCurrentIndex(3)
                self.status.showMessage(
                    f"这个文件里已有 {ann.total_marked():,} 个标注点，"
                    f"着色已切到「标注」，右侧列表按站列出实例", 6000)
            if getattr(doc, "warnings", None):
                self.status.showMessage(
                    "已强制打开：结构自检有告警（" + doc.warnings[0] + "），请核对点数与坐标", 8000)
        except StructureError as e:
            try:
                dlg.close()
            except Exception:
                pass
            if self._ask_force_open(e):
                self.load_file(path, force=True)
        except Exception as e:  # noqa: BLE001
            try:
                dlg.close()
            except Exception:
                pass
            QMessageBox.critical(self, APP_NAME, f"读取失败：\n{e}")

    def _ask_force_open(self, err: "StructureError") -> bool:
        """自检没通过时给一个"仍然打开"的出口，避免体检误报把用户堵死。"""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(APP_NAME)
        box.setText("这个 e57 没通过结构自检")
        box.setInformativeText(
            "发现的问题：\n" + "\n".join(f"· {p}" for p in err.problems)
            + "\n\n可以「仍然打开」：能读出多少算多少，标注后可另存为新文件。\n"
              "体检规则偶尔会误判（例如文件由更早的版本写出），强制打开通常没问题；\n"
              "若确实损坏，读出来的点数会明显偏少或坐标错乱，这时请回到原始 e57 重新标注。")
        b_open = box.addButton("仍然打开", QMessageBox.AcceptRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(b_open)
        box.exec()
        return box.clickedButton() is b_open

    def _confirm_discard(self) -> bool:
        r = QMessageBox.question(self, APP_NAME, "当前标注尚未保存，是否放弃？",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            # e57 是"标注文件"，直接写回；ply/pcd/xyz 往往是原始数据，
            # 覆盖它会毁掉原始点云，所以走「另存为」让人挑路径与格式。
            if cloudio.ext_of(self.path or "") == ".e57":
                self.save_overwrite()
            else:
                self.save_as()
            return not self._dirty
        return r == QMessageBox.Discard

    # --------------------------------------------------------- 保存 --

    def save_as(self) -> None:
        """另存为标注点云：格式由保存对话框里的扩展名决定。"""
        if self.doc is None:
            return
        src_ext = cloudio.ext_of(self.path or "")
        ext = src_ext if src_ext in cloudio.WRITE_EXTS else ".e57"
        if ext == ".pts":
            ext = ".xyz"
        base = os.path.splitext(self.path or "")[0]
        stem = base if base.endswith("_labeled") else (base + "_labeled" if base else "")
        d = (stem + ext) if stem else ""
        title = "保存为标注点云"
        path, chosen = QFileDialog.getSaveFileName(self, title, d, cloudio.SAVE_FILTER)
        if not path:
            return
        # 用户可能没写扩展名，或选了别的过滤器：以最终扩展名为准
        if cloudio.ext_of(path) not in cloudio.WRITE_EXTS:
            guess = self._ext_from_filter(chosen) or ext
            path += guess
        self._do_save(path)

    @staticmethod
    def _ext_from_filter(chosen: str) -> str:
        """从 "PLY 点云 (*.ply)" 里取出 .ply。"""
        if "(*" not in chosen:
            return ""
        inner = chosen.split("(*", 1)[1].split(")", 1)[0]
        first = inner.split()[0] if inner.split() else ""
        return first[1:] if first.startswith("*") else first

    def save_overwrite(self) -> None:
        """覆盖保存：按当前文件的**原格式**写回。"""
        if self.doc is None:
            return
        if not self.path:
            self.save_as()
            return
        self._do_save(self.path)

    def _do_save(self, path: str) -> None:
        ext = cloudio.ext_of(path)
        if ext not in cloudio.WRITE_EXTS:
            path += ".e57"
            ext = ".e57"
        is_e57 = ext == ".e57"
        what = "e57" if is_e57 else cloudio.format_name(path)
        dlg = QProgressDialog(f"正在写入 {what}…", "取消", 0, 100, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(150)
        dlg.setValue(15)
        QApplication.processEvents()
        try:
            # 把标注写进扫描字段（不污染其它字段）；e57 之外由 cloudio 从 ann 取
            if self._ensure_annotation_sync():
                self._refresh_list()
            if is_e57:
                for i, s in enumerate(self.doc.scans):
                    self._attach_annotation(s, self.ann.labels[i], self.ann.instances[i])
            dlg.setValue(45)
            QApplication.processEvents()
            t0 = time.perf_counter()
            if is_e57:
                self.doc.save(path)
            else:
                cloudio.save(path, self.doc, self.ann)
            el = time.perf_counter() - t0
            dlg.setValue(100)
            dlg.close()
            self.path = path
            self._dirty = False
            self._update_title()
            self._update_actions()
            self.lbl_file.setText(
                f"{os.path.basename(path)}   ·   {self.doc.total_points:,} 点   ·   "
                f"{len(self.doc.scans)} 扫描" + self._queue_status())
            self.lbl_file.setToolTip(f"{path}\n格式：{cloudio.format_name(path)}")
            self.lbl_perf.setText(f"写入 {el:.1f} s")
            extra = ""
            if not is_e57:
                extra = ("\n\n该格式把标注写在附加属性/列里（label / instance），"
                         "本软件重开、CloudCompare 打开都能看到；\n"
                         "若要长期归档，建议同时留一份 e57。")
            QMessageBox.information(self, APP_NAME,
                                    f"已保存（{what}）：\n{path}\n\n"
                                    f"点云 {self.doc.total_points:,} 个，"
                                    f"标注 {self.ann.total_marked():,} 个点。" + extra)
        except Exception as e:  # noqa: BLE001
            try:
                dlg.close()
            except Exception:
                pass
            QMessageBox.critical(self, APP_NAME, f"保存失败：\n{e}")

    @staticmethod
    def _attach_annotation(scan, labels: np.ndarray, instances: np.ndarray) -> None:
        """把标注数组以 e57 原生 Integer 字段附加到扫描上。"""
        n = scan.count
        scan.points["annotationLabel"] = np.asarray(labels, np.uint8)[:n]
        scan.points["annotationInstance"] = np.asarray(instances, np.int32)[:n]
        have = {f.name for f in scan.fields}
        from e57io import FieldDef
        if "annotationLabel" not in have:
            scan.fields.append(FieldDef(name="annotationLabel", kind="Integer",
                                        minimum=0, maximum=255, byte_size=1))
        if "annotationInstance" not in have:
            scan.fields.append(FieldDef(name="annotationInstance", kind="Integer",
                                        minimum=0, maximum=65535, byte_size=2))

    # --------------------------------------------------------- 裁剪 --

    def crop_box_dialog(self) -> None:
        """打开裁剪对话框（默认范围为整个点云）。"""
        if self.doc is None:
            return
        b = scan_bounds(self.doc.scans)
        if b is None:
            return
        self._open_crop_dialog(CropBox.from_bounds(b[0], b[1]))

    def _on_crop_request(self, rect) -> None:
        """屏幕框选 -> 投影出世界包围盒 -> 弹对话框微调。"""
        if self.doc is None:
            return
        full = scan_bounds(self.doc.scans)
        if full is None:
            return
        w, h = self.view.width(), self.view.height()
        sel = self.view.renderer.world_bounds_of_screen_rect(
            rect.left(), rect.top(), rect.right(), rect.bottom(), w, h)
        base = CropBox.from_bounds(full[0], full[1])
        if sel is None:
            self.status.showMessage("框内没有点，已按整个点云打开裁剪范围", 2500)
            self._open_crop_dialog(base)
            return
        box = CropBox.from_bounds(sel[0], sel[1]).clamp(base)
        self._open_crop_dialog(box, fallback=base)

    def _open_crop_dialog(self, box: CropBox, fallback: Optional[CropBox] = None) -> None:
        total = self.doc.total_points
        sel_n = 0
        if box.is_valid():
            xyz_all = self.view.renderer.xyz
            if len(xyz_all):
                sel_n = int(box.mask(xyz_all.astype(np.float64)).sum())
        dlg = CropDialog(box, total, sel_n, self)
        if dlg.exec() != QDialog.Accepted:
            self.set_mode(MODE_NAV)
            return
        out = dlg.result_box()
        if not out.is_valid():
            QMessageBox.warning(self, APP_NAME, "裁剪范围无效（最小值需小于最大值）。")
            self.set_mode(MODE_NAV)
            return
        self.apply_crop(out, invert=dlg.invert())
        self.set_mode(MODE_NAV)

    def apply_crop(self, box: CropBox, invert: bool = False) -> None:
        """按包围盒裁剪当前文档（会改变点数，可撤销）。"""
        if self.doc is None or self.ann is None:
            return
        before = self.doc.total_points
        masks = compound_mask(self.doc, box, invert)
        if all(m is None for m in masks):
            QMessageBox.information(self, APP_NAME, "该范围内没有点，未做修改。")
            return
        # 快照：原始扫描对象列表 + 各扫描点数据 + 标注数组（裁剪是破坏性的，
        # 且可能整段移除某个扫描，撤销时必须能把扫描列表本身放回去）
        snap = (list(self.doc.scans),
                [dict(p) for p in (s.points for s in self.doc.scans)],
                [l.copy() for l in self.ann.labels],
                [i.copy() for i in self.ann.instances],
                dict(self.ann.instance_info), self.ann.next_instance)
        # apply_masks 会移除被裁空的扫描，扫描索引会整体前移；交给 rebuild 的
        # 掩膜必须同步剔除这些扫描，否则标注数组会与扫描错位（表现为
        # "各列长度不一致 / boolean index did not match"）。
        keep = [np.asarray(m) for m in masks
                if m is not None and bool(np.asarray(m).any())]
        try:
            before_n, after_n = apply_masks(self.doc, masks)
            counts = [s.count for s in self.doc.scans]
            if len(keep) != len(counts):
                raise RuntimeError(
                    f"裁剪后扫描数 {len(counts)} 与保留掩膜数 {len(keep)} 不一致")
            self.ann.rebuild(keep, counts)
        except Exception as e:  # noqa: BLE001
            self._restore_snapshot(snap)      # 回滚，绝不留下 doc/ann 错位
            QMessageBox.critical(self, APP_NAME, f"裁剪失败：\n{e}")
            return
        if before_n == after_n and len(self.doc.scans) == len(snap[1]):
            self.status.showMessage("该范围内没有可裁剪的点", 2500)
            return
        self._crop_undo = snap
        self._selected_sid = min(self._selected_sid, max(len(self.doc.scans) - 1, 0))
        self.view.set_document(self.doc, self.ann)
        self.view.fit_and_render()
        self._update_instance_boxes()
        self._apply_hidden()
        self._refresh_list()
        self._dirty = True
        self._update_title()
        self.lbl_file.setText(
            f"{os.path.basename(self.path) if self.path else '未保存'}"
            f"   ·   {self.doc.total_points:,} 点   ·   {len(self.doc.scans)} 扫描"
            + self._queue_status())
        self.status.showMessage(
            f"裁剪完成：{before_n:,} → {after_n:,} 点（移除 {before_n - after_n:,}）", 4000)

    def _restore_snapshot(self, snap) -> None:
        """把裁剪快照整体放回（扫描列表、点数据、标注）。"""
        scans, points, labels, instances, info, nxt = snap
        for s, p in zip(scans, points):
            s.points = {k: np.asarray(v) for k, v in p.items()}
        self.doc.scans = scans
        self.ann.labels = [l.copy() for l in labels]
        self.ann.instances = [i.copy() for i in instances]
        self.ann.instance_info = dict(info)
        self.ann.next_instance = nxt
        self.ann.n_scans = len(self.ann.labels)

    def _ensure_annotation_sync(self) -> bool:
        """保证 ann 与 doc 逐扫描等长；不一致时按较短长度对齐并返回 True。

        任何"裁剪/撤销/换文件"路径出岔子都可能让两者错位，与其在标注或保存时
        抛 IndexError，不如就地修好并提示一次。
        """
        if self.doc is None or self.ann is None:
            return False
        fixed = False
        while len(self.ann.labels) < len(self.doc.scans):
            self.ann.labels.append(np.zeros(0, np.uint8))
            self.ann.instances.append(np.zeros(0, np.uint16))
            fixed = True
        if len(self.ann.labels) > len(self.doc.scans):
            del self.ann.labels[len(self.doc.scans):]
            del self.ann.instances[len(self.doc.scans):]
            fixed = True
        for i, s in enumerate(self.doc.scans):
            n = s.count
            lab = self.ann.labels[i]
            ins = self.ann.instances[i]
            if len(lab) != n or len(ins) != n:
                new_lab = np.zeros(n, np.uint8)
                new_ins = np.zeros(n, np.uint16)
                k = min(len(lab), n)
                new_lab[:k] = lab[:k]
                new_ins[:k] = ins[:k] if len(ins) >= k else 0
                self.ann.labels[i] = new_lab
                self.ann.instances[i] = new_ins
                fixed = True
        if fixed:
            self.ann.n_scans = len(self.ann.labels)
            alive = {int(v) for ins in self.ann.instances if len(ins)
                     for v in np.unique(ins) if v != 0}
            self.ann.instance_info = {k: v for k, v in self.ann.instance_info.items()
                                      if k in alive}
        return fixed

    def undo_crop(self) -> bool:
        """撤销最近一次裁剪，返回是否执行。"""
        snap = getattr(self, "_crop_undo", None)
        if not snap or self.doc is None or self.ann is None:
            return False
        self._restore_snapshot(snap)
        self._crop_undo = None
        self._selected_sid = min(self._selected_sid, max(len(self.doc.scans) - 1, 0))
        self.view.set_document(self.doc, self.ann)
        self.view.fit_and_render()
        self._update_instance_boxes()
        self._apply_hidden()
        self._refresh_list()
        self._update_title()
        self.status.showMessage("已撤销裁剪", 2500)
        return True

    # --------------------------------------------------------- 交互 --

    def set_mode(self, mode: str) -> None:
        self.view.set_mode(mode)
        for b, m in ((self.btn_nav, MODE_NAV), (self.btn_brush, MODE_BRUSH),
                     (self.btn_erase, MODE_ERASE), (self.btn_box, MODE_BOX),
                     (self.btn_inst, MODE_INST), (self.btn_crop, MODE_CROP)):
            b.setChecked(m == mode)
        self.status.showMessage(
            {MODE_NAV: "浏览：左键旋转 · 右键/中键平移 · 滚轮缩放",
             MODE_BRUSH: "笔刷：左键涂抹 · 右键结束一笔",
             MODE_ERASE: "橡皮：左键擦除 · 右键结束一笔",
             MODE_BOX: "框选：左键拖出矩形，松手即标注（Ctrl+拖拽 = 穿透全部深度）",
             MODE_INST: "穗实例：框住一根穗的穗点，松手即归为新实例（Ctrl+拖拽 = 穿透全部深度）",
             MODE_CROP: "裁剪：左键拖出矩形，松手后选择保留/删除"}[mode], 4000)

    def set_label(self, lid: int) -> None:
        self.view.current_label = lid
        self._label_btns[lid].setChecked(True)
        if getattr(self, "chk_hide_cur", None) is not None:
            self._apply_hidden()      # 「隐藏当前类别」跟着换成新类别

    def set_ann_mode(self, mode: str, quiet: bool = False) -> None:
        """切换语义标注 / 实例标注。"""
        mode = "semantic" if str(mode) == "semantic" else "instance"
        self.ann_mode = mode
        self.btn_semantic.setChecked(mode == "semantic")
        self.btn_instance.setChecked(mode == "instance")
        self.settings.setValue("ann_mode", mode)
        if not quiet and getattr(self, "status", None) is not None:
            self.status.showMessage(
                "语义标注：只写器官类别，不分配实例编号"
                if mode == "semantic" else
                "实例标注：每落一笔生成一个新实例编号", 3500)

    def _on_brush_size(self, v: int) -> None:
        self.view.brush_radius = v

    def _on_depth_mode(self, _i: int = 0) -> None:
        mode = self.cmb_depth.currentData() or "surface"
        self.view.renderer.depth_mode = mode
        self.spin_tol.setEnabled(mode == "surface")
        self.settings.setValue("depth_mode", mode)
        if getattr(self, "status", None) is not None:      # 构造期状态栏还没建
            self.status.showMessage(f"拾取深度：{self._depth_text()}", 2500)

    def _on_depth_tol(self, v: float) -> None:
        self.view.renderer.depth_tol = float(v) / 100.0
        self.settings.setValue("depth_tol_cm", float(v))
        if self.cmb_depth.currentData() == "surface" and getattr(self, "status", None):
            self.status.showMessage(f"拾取深度：{self._depth_text()}", 1500)

    def _depth_text(self) -> str:
        mode = self.view.renderer.depth_mode
        if mode == "all":
            return "全部穿透（框内所有点）"
        if mode == "surface":
            return f"表面 {self.view.renderer.depth_tol * 100:.1f} cm"
        return "前 50% 深度" if mode == "front50" else "前 25% 深度"

    def _on_point_size(self, v: int) -> None:
        self.view.set_point_size(v / 10.0)

    def _on_color_mode(self, i: int) -> None:
        self.view.set_color_mode(["rgb", "height", "intensity", "label"][i])

    def _on_scan_filter(self, _st) -> None:
        self._rebuild_scene()

    def _rebuild_scene(self) -> None:
        """按"显示全部扫描"开关重建渲染数据（不改动标注数组）。"""
        if self.doc is None or self.ann is None:
            return
        if self.chk_all_scans.isChecked() or len(self.doc.scans) == 1:
            self.view.renderer.set_data(self.doc, self.ann)
        else:
            # 单扫描模式：只看当前选中扫描，便于近距离标注
            idx = self._current_scan()
            self.view.renderer.set_data(self.doc, self.ann, only_scans={idx})
        self.view.reset_view()
        self.view.fit_and_render()
        self._update_instance_boxes()
        self._apply_hidden()

    def _current_scan(self) -> int:
        """当前聚焦的扫描序号（列表里离选中行最近的扫描标题）。"""
        cur = self.lst.currentRow()
        if cur < 0:
            return 0
        for row in range(cur, -1, -1):
            it = self.lst.item(row)
            if it is None:
                continue
            v = it.data(Qt.UserRole)
            if isinstance(v, int):
                return v
        return 0

    def _on_view_changed(self) -> None:
        pass

    def _on_hover(self, text: str) -> None:
        self.lbl_hover.setText(text)
        self.lbl_perf.setText(f"渲染 {self.view.renderer._last_ms:.0f} ms")

    def _set_view(self, pitch: float, yaw: float) -> None:
        cam = self.view.renderer.camera
        cam.pitch, cam.yaw = pitch, yaw
        self.view.reset_view()

    def _on_overlay(self, _st=None) -> None:
        self.view.show_axes = self.chk_axes.isChecked()
        self.view.show_bbox = self.chk_bbox.isChecked()
        self.view.show_instance_boxes = self.chk_ibox.isChecked()
        if self.view.show_instance_boxes:
            self._update_instance_boxes()
        else:
            self.view.set_instance_boxes([])

    def _on_canvas_changed(self, name: str) -> None:
        """标注页画布底色：预设直接切，「自定义…」弹取色器。"""
        if name == "自定义…":
            c = QColorDialog.getColor(QColor(*self.view.renderer.bg), self,
                                      "选择画布颜色")
            if c.isValid():
                self.view.renderer.bg = (c.red(), c.green(), c.blue())
                self._canvas_name = "自定义"
                self.view.redraw()
            else:
                self.cmb_canvas.blockSignals(True)
                self.cmb_canvas.setCurrentText(self._canvas_name)
                self.cmb_canvas.blockSignals(False)
            return
        self._canvas_name = name
        self.view.renderer.bg = CANVAS_COLORS.get(name, (26, 29, 34))
        self.view.redraw()
        self.view.update()

    def _on_hidden(self, _st=None) -> None:
        self.settings.setValue("hide_marked", self.chk_hide_marked.isChecked())
        self.settings.setValue("hide_cur", self.chk_hide_cur.isChecked())
        self._apply_hidden()
        if getattr(self, "status", None) is not None and self.doc is not None:
            r = self.view.renderer
            self.status.showMessage(
                f"显示 {r.n_visible():,} / {len(r.xyz):,} 点"
                f"（隐藏只影响显示，不改标注，也不影响存盘）", 3500)

    def _apply_hidden(self) -> None:
        """按「隐藏已标注 / 隐藏当前类别」重算渲染器的隐藏掩膜。

        隐藏只作用于显示与拾取，``scan_ids``/``local_index`` 不动，
        所以标注仍然写回正确的扫描与点序号。
        """
        r = self.view.renderer
        n = len(r.xyz)
        if n == 0:
            r.set_hidden(None)
            return
        mask = np.zeros(n, bool)
        lab = r.labels
        if lab.size == n:
            if self.chk_hide_marked.isChecked():
                mask |= lab > 0
            if self.chk_hide_cur.isChecked():
                mask |= lab == int(self.view.current_label)
        r.set_hidden(mask)
        self.view.redraw()

    def _focused_scan(self) -> int:
        """列表里聚焦的扫描序号（实例框只画这一站，避免多站叠加糊成一片）。"""
        if self.doc is None:
            return 0
        return max(0, min(self._current_scan(), len(self.doc.scans) - 1))

    def _update_instance_boxes(self) -> None:
        """为当前聚焦扫描的已编号穗实例生成彩色立方体框（世界坐标）。

        未开启「实例框」时直接返回（大文件逐实例求包络有开销），
        开启时 `_on_overlay` 会补算。只画聚焦的那一站：实例编号本来就是
        按站独立编号，多站同时画会叠出上百个框、编号互相压住读不出来。
        """
        if not self.view.show_instance_boxes:
            return
        if self.doc is None or self.ann is None:
            self.view.set_instance_boxes([])
            return
        boxes: List[Box3D] = []
        sid = self._focused_scan()
        scan = self.doc.scans[sid]
        lab = np.asarray(self.ann.labels[sid])[:scan.count]
        ins = np.asarray(self.ann.instances[sid])[:scan.count]
        m = (lab == 1) & (ins > 0)
        if m.any():
            pts = scan.points
            xyz = np.column_stack([pts["cartesianX"], pts["cartesianY"],
                                   pts["cartesianZ"]]).astype(np.float64)[:len(m)][m]
            ids = ins[m]
            for uid in np.unique(ids):
                sub = xyz[ids == uid]
                uid = int(uid)
                boxes.append(Box3D(sub.min(axis=0), sub.max(axis=0),
                                   f"#{uid}", instance_box_color(uid)))
                if len(boxes) >= 600:      # 防御：太多框会拖慢绘制
                    break
        self.view.set_instance_boxes(boxes)

    def _on_pose_toggle(self, _st=None) -> None:
        """切换"应用 E57 位姿"后需要重新读取文件（坐标会整体改变）。"""
        want = self.chk_pose.isChecked()
        if want == self.apply_pose:
            return
        if self.path is None:
            self.apply_pose = want
            return
        if self._dirty and not self._confirm_discard():
            self.chk_pose.blockSignals(True)
            self.chk_pose.setChecked(self.apply_pose)
            self.chk_pose.blockSignals(False)
            return
        self.apply_pose = want
        self.load_file(self.path)

    # --------------------------------------------------------- 标注 --

    def _on_annotate(self, glob_idx: np.ndarray, label: int, _action: int) -> None:
        if self.ann is None or glob_idx is None or len(glob_idx) == 0:
            return
        if self._ensure_annotation_sync():
            self.status.showMessage("检测到标注与点云错位，已自动对齐", 2000)
        r = self.view.renderer
        scan_ids = r.scan_ids[glob_idx]
        local = r.local_index[glob_idx]
        # 按扫描分组，一次笔画在同一扫描内用同一实例编号
        total = 0
        group: List[tuple] = []          # 一次操作 = 一个撤销步（可跨多个扫描）
        semantic = label != 0 and self.ann_mode == "semantic"
        for sid in np.unique(scan_ids):
            m = scan_ids == sid
            li = local[m]
            mask = np.zeros(self.doc.scans[int(sid)].count, bool)
            mask[li] = True
            group.append((int(sid), self.ann.undoable_snapshot(int(sid))))
            if label == 0:
                self.ann.labels[int(sid)][mask] = 0
                self.ann.instances[int(sid)][mask] = 0
                total += int(mask.sum())
            elif semantic:
                # 语义标注：只写类别，不分配实例编号；原有编号一并清掉
                self.ann.instances[int(sid)][mask] = 0
                self.ann.labels[int(sid)][mask] = label
                total += int(mask.sum())
            else:
                total += self.ann.apply(int(sid), mask, label)
        if label == 0 or semantic:
            for sid, _snap in group:
                self.ann.prune_instance_info(int(sid))
        self._undo.append(group)
        if len(self._undo) > 60:
            self._undo = self._undo[-60:]
        r.refresh_labels(self.ann)
        self._update_instance_boxes()
        self._apply_hidden()
        self._dirty = True
        self._refresh_list()
        self._update_title()
        if label:
            name = LABEL_NAMES.get(label, "?")
            extra = f"（跨 {len(group)} 个扫描）" if len(group) > 1 else ""
            if not self.chk_all_scans.isChecked() and len(self.doc.scans) > 1:
                extra += f"（当前只显示扫描 #{int(group[0][0])}，其余站未参与）"
            mode_txt = "语义标注，不分配实例编号" if semantic else "实例标注"
            self.status.showMessage(
                f"已标注 {total:,} 个点为「{name}」· {mode_txt}{extra}"
                f" · 深度 {self._depth_text()} · 点数偏少就把「拾取深度」调宽或按 Ctrl 拖拽", 4000)
        else:
            self.status.showMessage(
                f"已擦除 {total:,} 个点 · 深度 {self._depth_text()}", 2500)

    def _on_instance_box(self, glob_idx: np.ndarray) -> None:
        """手动框选穗实例：框内「穗」点归为一个新实例（可整体撤销）。

        只动 label==1 的点：叶/茎/其他类别的实例不受影响。
        框住两段自动分割的碎片 = 合并；框住一根穗的一部分 = 拆分。
        """
        if self.ann is None or self.doc is None or glob_idx is None or len(glob_idx) == 0:
            return
        if self._ensure_annotation_sync():
            self.status.showMessage("检测到标注与点云错位，已自动对齐", 2000)
        r = self.view.renderer
        scan_ids = r.scan_ids[glob_idx]
        local = r.local_index[glob_idx]
        group: List[tuple] = []          # 一次框选 = 一个撤销步（可跨扫描）
        made: List[tuple] = []           # (sid, uid, n)
        for sid in np.unique(scan_ids):
            sid = int(sid)
            if self.ann.next_instance > 65535:      # annotationInstance 是 2 字节
                self.status.showMessage(
                    "实例编号已达上限 65535，无法再分配新实例", 4000)
                break
            m = scan_ids == sid
            li = local[m]
            lab = self.ann.labels[sid]
            ins = self.ann.instances[sid]
            sel = np.zeros(lab.shape[0], bool)
            sel[li] = True
            sel &= (lab == 1)           # 只给穗点编号
            n = int(sel.sum())
            if n == 0:
                continue
            uid = self.ann.new_instance(sid, 1)
            group.append((sid, self.ann.undoable_snapshot(sid)))
            ins[sel] = uid
            made.append((sid, uid, n))
            # 旧实例若已被这次框选整体覆盖，清掉它的登记
            alive = {int(v) for v in np.unique(ins) if v != 0}
            self.ann.instance_info = {
                k: v for k, v in self.ann.instance_info.items()
                if k in alive or v[1] != sid}
        if not group:
            self.status.showMessage(
                "框内没有穗点：先用「框选/笔刷」把这些点标成「穗」，再分实例"
                f"（框内 {len(glob_idx):,} 个点里没有穗类别）", 5000)
            return
        self._undo.append(group)
        if len(self._undo) > 60:
            self._undo = self._undo[-60:]
        r.refresh_labels(self.ann)
        self._dirty = True
        self._refresh_list()
        self._update_title()
        if not self.chk_ibox.isChecked():
            self.chk_ibox.setChecked(True)   # 直接开实例框，框选结果立等可见
        else:
            self._update_instance_boxes()
        self._apply_hidden()
        parts = [f"穗实例 #{uid}（{n:,} 点，扫描 #{sid + 1}）"
                 for sid, uid, n in made]
        skipped = len(glob_idx) - sum(n for _, _, n in made)
        extra = f" · 另有 {skipped:,} 个非穗点未动" if skipped else ""
        self.status.showMessage(
            "已建 " + "；".join(parts) + extra + " · 框错 Ctrl+Z 撤销", 5000)

    # ------------------------------------------------- 标注管理（改已有） --

    def label_edit_dialog(self) -> None:
        """打开「修改已有标注」对话框。"""
        if self.doc is None or self.ann is None:
            self.status.showMessage("请先打开点云文件", 2000)
            return
        if self._ensure_annotation_sync():
            self.status.showMessage("检测到标注与点云错位，已自动对齐", 2000)
        LabelEditDialog(self, self).exec()
        self._refresh_list()
        self._update_title()

    def _key_mask(self, sid: int, kind: str, val: int) -> np.ndarray:
        """把标注管理对话框里的一行转成该扫描内的布尔掩膜。"""
        lab = self.ann.labels[sid]
        ins = self.ann.instances[sid]
        if kind == "inst":
            return ins == int(val)
        # "sem"：该类别的点里没有实例编号的那部分
        return (lab == int(val)) & (ins == 0)

    def edit_labels(self, action: str, keys=None, label: int = 0) -> int:
        """修改已有标注的统一入口；一个动作 = 一个撤销步。

        ``action``：set_label / to_semantic / merge / delete /
        clear_scan / clear_category / clear_all。
        ``keys`` 为 ``(kind, sid, value)`` 列表，kind ∈ {inst, sem, scan}。
        """
        if self.doc is None or self.ann is None:
            return 0
        keys = list(keys or [])
        if action != "clear_all" and action != "clear_category" and not keys:
            return 0
        if self._ensure_annotation_sync():
            self.status.showMessage("检测到标注与点云错位，已自动对齐", 2000)

        if action in ("clear_all", "clear_category"):
            sids = list(range(len(self.doc.scans)))
        else:
            sids = sorted({int(k[1]) for k in keys})
        group = [(sid, self.ann.undoable_snapshot(sid)) for sid in sids]
        total = 0

        if action == "clear_all":
            total = sum(int((l > 0).sum()) for l in self.ann.labels)
            self.ann.clear_all()
        elif action == "clear_category":
            for sid in sids:
                lab = self.ann.labels[sid]
                ins = self.ann.instances[sid]
                m = lab == int(label)
                total += int(m.sum())
                lab[m] = 0
                ins[m] = 0
        elif action == "clear_scan":
            for sid in sids:
                total += int((self.ann.labels[sid] > 0).sum())
                self.ann.clear_scan(sid)
        elif action == "merge":
            for sid in sids:
                m = np.zeros(self.ann.labels[sid].size, bool)
                for kind, ks, val in keys:
                    if int(ks) == sid:
                        m |= self._key_mask(sid, kind, val)
                n = int(m.sum())
                if n == 0:
                    continue
                lab = self.ann.labels[sid]
                vals, cnts = np.unique(lab[m], return_counts=True)
                lb = int(vals[cnts.argmax()]) if len(vals) else 1
                uid = self.ann.new_instance(sid, lb)
                lab[m] = lb
                self.ann.instances[sid][m] = uid
                total += n
        else:
            for kind, sid, val in keys:
                m = self._key_mask(sid, kind, val)
                if not m.any():
                    continue
                total += int(m.sum())
                if action == "set_label":
                    self.ann.labels[sid][m] = int(label)
                    if kind == "inst":
                        self.ann.instance_info[int(val)] = (int(label), sid)
                elif action == "to_semantic":
                    self.ann.instances[sid][m] = 0
                elif action == "delete":
                    self.ann.labels[sid][m] = 0
                    self.ann.instances[sid][m] = 0
                else:
                    raise ValueError(f"未知的标注修改动作：{action}")

        for sid in sids:
            self.ann.prune_instance_info(sid)
        self.ann.ensure_next_instance()
        self._undo.append(group)
        if len(self._undo) > 60:
            self._undo = self._undo[-60:]
        self.view.refresh_labels(self.ann)
        self._update_instance_boxes()
        self._apply_hidden()
        self._dirty = True
        self._refresh_list()
        self._update_title()
        self.status.showMessage(f"标注已修改（{total:,} 个点）· Ctrl+Z 可撤销", 4000)
        return total

    def undo(self) -> None:
        if self.undo_crop():
            return
        if not self._undo or self.ann is None:
            self.status.showMessage("没有可撤销的操作", 1500)
            return
        group = self._undo.pop()
        for sid, snap in reversed(group):
            self.ann.restore(sid, snap)
        self.ann.ensure_next_instance()   # 撤销「清空全部」后编号不能低于在用最大值
        self.view.refresh_labels(self.ann)
        self._update_instance_boxes()
        self._apply_hidden()
        self._dirty = True
        self._refresh_list()
        self._update_title()
        self.status.showMessage("已撤销", 1500)

    def clear_scan(self) -> None:
        """清空当前聚焦扫描的标注（整步可撤销）。"""
        if self.ann is None or self.doc is None:
            return
        sid = max(0, min(self._current_scan(), len(self.doc.scans) - 1))
        if int((self.ann.labels[sid] > 0).sum()) == 0:
            self.status.showMessage("当前扫描没有标注", 1500)
            return
        self._undo.append([(sid, self.ann.undoable_snapshot(sid))])
        self.ann.clear_scan(sid)
        self.view.refresh_labels(self.ann)
        self._update_instance_boxes()
        self._apply_hidden()
        self._dirty = True
        self._refresh_list()
        self._update_title()

    # ------------------------------------------------- 穗实例分割 --

    def _seg_params(self) -> SegmentParams:
        """从 QSettings 读上次用的分割参数（缺省 0.30 cm / 26 邻域 / 10 点）。"""
        try:
            return SegmentParams(
                voxel_cm=float(self.settings.value("seg_voxel_cm", 0.30)),
                link=int(self.settings.value("seg_link", 1)),
                min_points=int(self.settings.value("seg_min_points", 10)))
        except Exception:                             # noqa: BLE001
            return SegmentParams()

    def _save_seg_params(self, p: SegmentParams) -> None:
        self.settings.setValue("seg_voxel_cm", float(p.voxel_cm))
        self.settings.setValue("seg_link", int(p.link))
        self.settings.setValue("seg_min_points", int(p.min_points))

    def _segment_targets(self, scope: str) -> List[int]:
        """返回要处理的扫描索引：当前扫描，或全部扫描（逐站）。"""
        if self.doc is None:
            return []
        if scope == "current":
            return [max(0, min(self._current_scan(), len(self.doc.scans) - 1))]
        return list(range(len(self.doc.scans)))

    def _spike_selection(self, sid: int, overwrite: bool):
        """取某个扫描待分割的穗点：返回 (mask, xyz)。

        ``overwrite=False`` 时只取「已标穗但还没有实例编号」的点，
        便于先做一版自动分穗、再手工补标后增量补分。
        """
        lab = self.ann.labels[sid]
        ins = self.ann.instances[sid]
        m = lab == 1
        if not overwrite:
            m = m & (ins == 0)
        pts = self.doc.scans[sid].points
        xyz = np.column_stack([pts["cartesianX"], pts["cartesianY"],
                               pts["cartesianZ"]]).astype(np.float64)
        return m, xyz

    def _unassigned_spikes(self) -> int:
        """全文件范围内「已标穗但还没有实例编号」的点数。"""
        if self.ann is None:
            return 0
        n = 0
        for lab, ins in zip(self.ann.labels, self.ann.instances):
            n += int(((lab == 1) & (ins == 0)).sum())
        return n

    def _segment_probe(self, scope: str, params: SegmentParams,
                       overwrite: bool) -> str:
        """试算：只统计不改标注，供对话框实时显示效果。

        同时把「将生成的穗」以虚线立方体框画进三维视图（编号按执行后
        实际会分配的实例号推算），分割前就能预览效果；对话框关闭时清掉。
        """
        if self.doc is None or self.ann is None:
            return "请先打开点云文件。"
        lines, tot_c, tot_p, tot_rej, tot_comp = [], 0, 0, 0, 0
        focus = self._focused_scan()           # 预览框只画聚焦的这一站
        prev_boxes: List[Box3D] = []
        base = int(self.ann.next_instance)     # 与 apply_segment 同口径的编号推算
        for sid in self._segment_targets(scope):
            m, xyz = self._spike_selection(sid, overwrite)
            n = int(m.sum())
            if n == 0:
                have = int((self.ann.labels[sid] == 1).sum())
                if have and not overwrite:
                    lines.append(f"扫描 #{sid + 1}：{have:,} 个穗点都已有实例编号"
                                 f"（勾选「覆盖已有实例编号」可重新分割）")
                else:
                    lines.append(f"扫描 #{sid + 1}：没有穗点")
                continue
            t0 = time.perf_counter()
            r = segment_points(xyz[m], params)
            ms = (time.perf_counter() - t0) * 1000
            tot_c += r.n_clusters
            tot_p += r.n_assigned
            tot_rej += r.n_rejected
            tot_comp += r.n_components
            # 预览框只画聚焦扫描；其它站的统计仍在上面文字里
            if sid == focus and r.n_clusters and len(prev_boxes) < 600:
                sel_xyz = xyz[m]
                for cid in range(1, r.n_clusters + 1):
                    sub = sel_xyz[r.cluster == cid]
                    if len(sub) == 0:
                        continue
                    uid = base + cid - 1
                    prev_boxes.append(Box3D(sub.min(axis=0), sub.max(axis=0),
                                            f"#{uid}", instance_box_color(uid),
                                            dashed=True))
            base += r.n_clusters
            if r.n_clusters:
                med = int(np.median(r.sizes))
                lines.append(f"扫描 #{sid + 1}：{n:,} 穗点 → {r.n_clusters} 个穗"
                             f"（单穗 {int(r.sizes.min()):,}–{int(r.sizes.max()):,} 点，"
                             f"中位 {med:,}）　{ms:.0f} ms")
            else:
                lines.append(f"扫描 #{sid + 1}：{n:,} 穗点 → 0 个穗"
                             f"（连通块 {r.n_components} 个，都少于最小穗点数）")
        self.view.set_preview_boxes(prev_boxes)
        if not lines:
            return "没有可处理的扫描。"
        lines.append(f"合计：{tot_c} 个穗实例，{tot_p:,} 点已编号；"
                     f"连通块 {tot_comp} 个，其中 {tot_rej:,} 点判为碎片未编号。")
        if tot_c == 0:
            if tot_comp == 0 and not overwrite:
                lines.append("提示：这些穗点已有实例编号；勾选「覆盖已有实例编号」可重新分割。")
            else:
                lines.append("提示：把体素边长调小、连通方式收紧，或调低最小穗点数。")
        elif tot_rej > tot_p * 0.3:
            lines.append("提示：碎片偏多，可适当调低「最小穗点数」。")
        if prev_boxes:
            lines.append(f"视图里已用虚线框预览扫描 #{focus + 1} 的穗"
                         f"（框顶编号 = 执行后的实例号）。")
        elif tot_c:
            lines.append(f"提示：预览框只画当前聚焦的扫描 #{focus + 1}，"
                         f"该站没有可预览的穗；统计仍覆盖所选范围。")
        return "\n".join(lines)

    def segment_spikes(self) -> None:
        """打开穗实例分割对话框。"""
        if self.doc is None or self.ann is None:
            return
        if self._ensure_annotation_sync():
            self._refresh_list()
        total_spike = sum(int((l == 1).sum()) for l in self.ann.labels)
        if total_spike == 0:
            QMessageBox.information(
                self, APP_NAME,
                "这个文件里还没有「穗」点。\n\n"
                "请先用框选 / 笔刷把麦穗标成「穗」类别，再回来做实例分割。")
            return
        dlg = SegmentDialog(len(self.doc.scans), self._current_scan(),
                            self._seg_params(), probe=self._segment_probe,
                            default_overwrite=(self._unassigned_spikes() == 0),
                            parent=self)
        try:
            ok = dlg.exec() == QDialog.Accepted
        finally:
            self.view.set_preview_boxes([])     # 对话框一关就清掉预览框
        if not ok:
            return
        params = dlg.params()
        self._save_seg_params(params)
        self.apply_segment(params, dlg.scope(), dlg.overwrite())

    def apply_segment(self, params: SegmentParams, scope: str = "current",
                      overwrite: bool = False) -> int:
        """执行穗实例分割，返回新增的穗实例数（可整体撤销）。"""
        if self.doc is None or self.ann is None:
            return 0
        self.view.set_preview_boxes([])    # 执行后预览框就过期了
        group: List[tuple] = []            # 一次分割 = 一个撤销步（可跨扫描）
        added = 0
        total_pts = 0
        limited = False
        for sid in self._segment_targets(scope):
            m, xyz = self._spike_selection(sid, overwrite)
            if not m.any():
                continue
            r = segment_points(xyz[m], params)
            if r.n_clusters == 0:
                continue
            group.append((sid, self.ann.undoable_snapshot(sid)))
            idx = np.nonzero(m)[0]
            lab = self.ann.labels[sid]
            ins = self.ann.instances[sid]
            base = self.ann.next_instance
            if base + r.n_clusters - 1 > 65535:       # annotationInstance 是 2 字节
                leftover = 65535 - base + 1
                if leftover <= 0:
                    limited = True
                    group.pop()
                    break
                r.n_clusters = int(leftover)
                limited = True
            # 簇号 1..K 连续，且按点数降序 —— 直接平移成实例编号（cluster 1 → base），
            # 免逐簇筛点；cluster 0 是碎片，保持不编号。
            uids = (base + r.cluster - 1).astype(np.uint16)
            keep = r.cluster > 0
            if limited:
                keep &= r.cluster <= r.n_clusters
                uids = np.where(keep, uids, 0)
            if overwrite:
                # 覆盖模式：这批穗点的旧编号一律作废（包括被判为碎片的点），
                # 否则碎片会带着旧号留下来，执行结果的框数跟试算预览对不上。
                ins[idx] = 0
            sel = idx[keep]
            ins[sel] = uids[keep]
            lab[sel] = 1
            for k in range(r.n_clusters):
                self.ann.instance_info[base + k] = (1, sid)
            self.ann.next_instance = base + r.n_clusters
            added += r.n_clusters
            total_pts += int(sel.size)
            # 旧实例若已被完全覆盖，清掉它的登记
            alive = {int(v) for v in np.unique(ins) if v != 0}
            self.ann.instance_info = {
                k: v for k, v in self.ann.instance_info.items()
                if k in alive or v[1] != sid}
            if limited:
                break
        if not group:
            QMessageBox.information(
                self, APP_NAME,
                "没有生成任何穗实例。\n\n"
                "可能原因：待分割的穗点太少（可勾选「覆盖已有实例编号」），\n"
                "或最小穗点数设得太大。")
            return 0
        self._undo.append(group)
        if len(self._undo) > 60:
            self._undo = self._undo[-60:]
        self.view.refresh_labels(self.ann)
        if not self.chk_ibox.isChecked():
            self.chk_ibox.setChecked(True)      # 分割完自动开实例框，编号一眼可见
        self._update_instance_boxes()
        self._apply_hidden()
        if self.cmb_color.currentIndex() != 3:
            self.cmb_color.setCurrentIndex(3)     # 切到"标注"着色，结果一眼可见
        self._dirty = True
        self._refresh_list()
        self._update_title()
        msg = (f"穗实例分割完成：新增 {added} 个穗实例，{total_pts:,} 点已编号"
               f"（体素 {params.voxel_cm:.2f} cm · 最小 {params.min_points} 点）")
        if limited:
            msg += "　实例编号已达上限 65535，余下扫描未处理"
        self.status.showMessage(msg + "　·　Ctrl+Z 可整体撤销", 6000)
        return added

    # --------------------------------------------------------- 列表 --

    def _refresh_list(self) -> None:
        self.lst.clear()
        if self.doc is None or self.ann is None:
            self.lbl_stat.setText("—")
            return
        counts = self.ann.label_counts()
        insts = self.ann.instance_counts()
        prev_sid = self._selected_sid
        for i, s in enumerate(self.doc.scans):
            marked = int((self.ann.labels[i] > 0).sum())
            ids = self.ann.instance_ids(i)
            title = f"扫描 {i + 1}/{len(self.doc.scans)}    {s.count:,} 点"
            it = QListWidgetItem(title)
            it.setData(Qt.UserRole, i)
            f = it.font(); f.setBold(True); it.setFont(f)
            self.lst.addItem(it)
            if marked == 0:
                sub = QListWidgetItem("     未标注")
                sub.setForeground(QColor(120, 126, 134))
                sub.setFlags(sub.flags() & ~Qt.ItemIsSelectable)
                self.lst.addItem(sub)
                continue
            for uid in ids:
                lb = self.ann.instance_label(i, uid)
                size = self.ann.instance_size(i, uid)
                text = f"     {LABEL_NAMES.get(lb, '?')} #{uid}   ({size:,} 点)"
                it2 = QListWidgetItem(make_swatch(LABEL_COLORS.get(lb, (150, 150, 150))), text)
                it2.setFlags(it2.flags() & ~Qt.ItemIsSelectable)
                self.lst.addItem(it2)
            # 语义标注的点（有类别、没有实例编号）汇总成一行
            sem = self.ann.label_groups(i)
            if sem:
                txt = "     未分实例：" + " / ".join(
                    f"{LABEL_NAMES.get(lb, '?')} {c:,} 点" for lb, c in sem)
                it3 = QListWidgetItem(txt)
                it3.setForeground(QColor(150, 156, 166))
                it3.setFlags(it3.flags() & ~Qt.ItemIsSelectable)
                self.lst.addItem(it3)

        # 恢复选中行
        self._selected_sid = prev_sid
        for row in range(self.lst.count()):
            if self.lst.item(row).data(Qt.UserRole) == prev_sid:
                self.lst.setCurrentRow(row)
                break

        parts = []
        for lid in (1, 2, 3, 4):
            parts.append(f"{LABEL_NAMES[lid]} {insts.get(lid, 0)} 个 / {counts.get(lid, 0):,} 点")
        total_pts = self.doc.total_points
        marked = self.ann.total_marked()
        pct = marked / total_pts * 100 if total_pts else 0
        self.lbl_stat.setText("　".join(parts) +
                              f"\n已标注 {marked:,} / {total_pts:,} 点（{pct:.1f}%）")

    def _on_list_sel(self, cur, _prev) -> None:
        if cur is None:
            return
        v = cur.data(Qt.UserRole)
        if not isinstance(v, int):
            return
        self._selected_sid = v
        # 未勾选"显示全部扫描"时，切换扫描即切换聚焦对象
        if not self.chk_all_scans.isChecked():
            self._rebuild_scene()
        self._update_instance_boxes()      # 实例框跟着切到这一站

    def _update_title(self) -> None:
        base = os.path.basename(self.path) if self.path else "未打开文件"
        star = " *" if self._dirty else ""
        module = MODULE_CROP if getattr(self, "_page", 0) else MODULE_ANNOT
        self.setWindowTitle(f"{APP_NAME}  ·  {module}  v{APP_VER}  —  {base}{star}")

    def closeEvent(self, ev):
        if self._dirty and not self._confirm_discard():
            ev.ignore()
            return
        # 记住"非最大化"时的窗口大小：下次启动仍最大化，但取消最大化后是熟悉的大小
        try:
            r = self.normalGeometry()
            if r.isValid() and r.width() >= 1000 and r.height() >= 640:
                self.settings.setValue("last_rect", r)
        except Exception:                       # noqa: BLE001
            pass
        ev.accept()
