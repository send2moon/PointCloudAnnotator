"""主窗口：菜单 / 工具栏 / 三维视图 / 标注面板 / 状态栏。"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QActionGroup, QColor, QFont, QIcon, QKeySequence,
                           QPalette, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressDialog, QPushButton, QRadioButton, QScrollArea,
                               QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar,
                               QToolBar, QToolButton, QVBoxLayout, QWidget, QButtonGroup)

from annotation import AnnotationSet, LABEL_NAMES, LABEL_COLORS
from cropping import CropBox, apply_masks, compound_mask, scan_bounds
from crop_dialog import CropDialog
from e57io import E57Document, StructureError
from renderer import PALETTE
from view3d import PointCloudView, MODE_NAV, MODE_BRUSH, MODE_ERASE, MODE_BOX, MODE_CROP

APP_NAME = "作物点云器官标注软件"
APP_VER = "1.0"
# 内部英文标识：QSettings 组织名 / e57 库标识沿用旧值，避免用户配置丢失
APP_ID = "PointCloudAnnotator"


DARK_QSS = """
QWidget { background: #1e2126; color: #d8dce2; font-size: 13px; }
QMainWindow::separator { background: #2c3037; width: 2px; height: 2px; }
QToolBar { background: #23272d; border: 0; spacing: 6px; padding: 6px; }
QToolBar QToolButton { background: transparent; border: 1px solid transparent;
    border-radius: 6px; padding: 6px 10px; color: #cfd4db; }
QToolBar QToolButton:hover { background: #2c3138; }
QToolBar QToolButton:checked { background: #34506e; border-color: #3f6a95; color: #ffffff; }
/* 侧栏"标注工具"按钮：当前模式必须一眼看出来 */
QToolButton { background: #262b31; border: 1px solid #3a4048; border-radius: 6px;
    padding: 5px 8px; color: #cfd4db; }
QToolButton:hover { background: #343a43; }
QToolButton:checked { background: #2f6db5; border: 2px solid #8fc0ff;
    color: #ffffff; font-weight: 700; }
QToolButton:!checked { font-weight: 400; }
QGroupBox { border: 1px solid #2f343c; border-radius: 8px; margin-top: 12px;
    padding: 10px 10px 10px 10px; font-weight: 600; color: #aeb6c0; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background: #2c3138; border: 1px solid #3a4048; border-radius: 6px;
    padding: 6px 12px; }
QPushButton:hover { background: #343a43; }
QPushButton:pressed { background: #262b31; }
QPushButton#primary { background: #2f6db5; border-color: #3d82d2; color: #ffffff; font-weight: 600; }
QPushButton#primary:hover { background: #3879c4; }
QComboBox, QSpinBox { background: #262b31; border: 1px solid #3a4048;
    border-radius: 6px; padding: 4px 8px; min-height: 20px; }
QComboBox::drop-down { border: 0; width: 18px; }
QComboBox QAbstractItemView { background: #262b31; selection-background-color: #34506e; }
QListWidget { background: #22262b; border: 1px solid #2f343c; border-radius: 6px;
    padding: 4px; }
QListWidget::item { padding: 3px 4px; border-radius: 4px; }
QListWidget::item:selected { background: #34506e; color: #ffffff; }
QSlider::groove:horizontal { height: 4px; background: #33383f; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -6px 0; border-radius: 7px;
    background: #6fa8dc; }
QStatusBar { background: #1a1d21; color: #98a0aa; border-top: 1px solid #2c3037; }
/* 深色底上的勾选/单选框：白色描边，选中为实心白，避免和背景糊在一起 */
QCheckBox, QRadioButton { color: #e8ebf0; spacing: 8px; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 3px;
    border: 2px solid #ffffff; background: #23272d; }
QCheckBox::indicator:hover { background: #3a4048; }
QCheckBox::indicator:checked { background: #ffffff; border: 2px solid #ffffff; }
QRadioButton::indicator { width: 16px; height: 16px; border-radius: 8px;
    border: 2px solid #ffffff; background: #23272d; }
QRadioButton::indicator:hover { background: #3a4048; }
QRadioButton::indicator:checked { background: #ffffff; border: 2px solid #ffffff; }
QRadioButton:checked, QCheckBox:checked { color: #ffffff; font-weight: 600; }
QSplitter::handle { background: #2c3037; }
QScrollArea { border: 0; }
QLabel#hint { color: #8b939d; font-size: 12px; }
QLabel#title { color: #eef1f5; font-size: 15px; font-weight: 700; }
"""


def make_swatch(color) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(QColor(*color))
    return QIcon(pm)


class LabelButton(QToolButton):
    """器官类别按钮：选中时用该类别的颜色填充 + 白描边，未选中为普通深色按钮。"""
    def __init__(self, label_id: int, parent=None):
        super().__init__(parent)
        self.label_id = label_id
        txt = LABEL_NAMES[label_id]
        self.setText(f"  {txt}")
        self.setIcon(make_swatch(LABEL_COLORS[label_id]))
        self.setCheckable(True)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setMinimumHeight(30)
        r, g, b = LABEL_COLORS[label_id]
        self.setStyleSheet(
            "QToolButton { background: #262b31; border: 1px solid #3a4048;"
            " border-radius: 6px; padding: 5px 6px; color: #cfd4db; }"
            "QToolButton:hover { background: #343a43; }"
            f"QToolButton:checked {{ background: rgb({r},{g},{b});"
            " border: 2px solid #ffffff; color: #12151a; font-weight: 700; }"
        )
        self.setToolTip(f"当前类别：{txt}（选中后按钮会高亮为该类别颜色）")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VER}")
        self.resize(1500, 940)
        self.setStyleSheet(DARK_QSS)

        self.doc: Optional[E57Document] = None
        self.ann: Optional[AnnotationSet] = None
        self.path: Optional[str] = None
        self._dirty = False
        self._undo: List[tuple] = []
        self._selected_sid = 0
        self._queue: List[str] = []
        self._queue_src: List[str] = []
        self.apply_pose = True          # 是否把扫描的 <pose> 烘焙进坐标
        self.settings = QSettings(APP_ID, "Main")

        self._build_ui()
        self._build_actions()
        self._update_actions()
        self._update_title()
        QTimer.singleShot(60, self._maybe_open_from_args)

    # ------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        self.view = PointCloudView(self)
        self.view.annotateRequest.connect(self._on_annotate)
        self.view.hoverInfo.connect(self._on_hover)
        self.view.viewChanged.connect(self._on_view_changed)
        self.view.cropRequest.connect(self._on_crop_request)
        self.view.escapePressed.connect(lambda: self.set_mode(MODE_NAV))

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.view)
        split.addWidget(self._build_side_panel())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1140, 340])
        self.setCentralWidget(split)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_file = QLabel("未打开文件")
        self.lbl_hover = QLabel("")
        self.lbl_perf = QLabel("")
        self.status.addWidget(self.lbl_file, 0)
        self.status.addPermanentWidget(self.lbl_hover, 1)
        self.status.addPermanentWidget(self.lbl_perf, 0)

        self._on_depth_mode()          # 把侧栏里的拾取深度同步给渲染器

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(400)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # --- 标注工具 ---
        gb_tool = QGroupBox("标注工具")
        v = QVBoxLayout(gb_tool)
        v.setSpacing(6)
        row = QHBoxLayout()
        self.btn_nav = QToolButton(); self.btn_nav.setText("浏览")
        self.btn_brush = QToolButton(); self.btn_brush.setText("笔刷")
        self.btn_erase = QToolButton(); self.btn_erase.setText("橡皮")
        self.btn_box = QToolButton(); self.btn_box.setText("框选")
        self.btn_crop = QToolButton(); self.btn_crop.setText("裁剪")
        self._tool_group = QButtonGroup(self)
        for b in (self.btn_nav, self.btn_brush, self.btn_erase, self.btn_box, self.btn_crop):
            b.setCheckable(True)
            b.setMinimumHeight(30)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._tool_group.addButton(b)
            row.addWidget(b)
        self.btn_nav.setChecked(True)
        self.btn_nav.clicked.connect(lambda: self.set_mode(MODE_NAV))
        self.btn_brush.clicked.connect(lambda: self.set_mode(MODE_BRUSH))
        self.btn_erase.clicked.connect(lambda: self.set_mode(MODE_ERASE))
        self.btn_box.clicked.connect(lambda: self.set_mode(MODE_BOX))
        self.btn_crop.clicked.connect(lambda: self.set_mode(MODE_CROP))
        v.addLayout(row)

        v.addWidget(self._sep())
        v.addWidget(QLabel("器官类别"))
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
        self.spin_tol.setToolTip("表面层模式的厚度：比目标器官略大一点即可")
        self.spin_tol.valueChanged.connect(self._on_depth_tol)
        h.addWidget(self.spin_tol)
        v.addLayout(h)
        # 沿用上次的拾取深度设置
        i = self.cmb_depth.findData(str(self.settings.value("depth_mode", "surface")))
        self.cmb_depth.setCurrentIndex(max(i, 0))
        self.spin_tol.setValue(float(self.settings.value("depth_tol_cm", 5.0)))

        h2 = QHBoxLayout()
        b_undo = QPushButton("撤销 (Ctrl+Z)")
        b_undo.clicked.connect(self.undo)
        b_clear = QPushButton("清空当前扫描")
        b_clear.clicked.connect(self.clear_scan)
        h2.addWidget(b_undo); h2.addWidget(b_clear)
        v.addLayout(h2)
        lay.addWidget(gb_tool)

        # --- 显示 ---
        gb_view = QGroupBox("显示")
        f = QFormLayout(gb_view)
        f.setLabelAlignment(Qt.AlignLeft)
        f.setSpacing(8)
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["真彩色", "高度", "强度", "标注"])
        self.cmb_color.currentIndexChanged.connect(self._on_color_mode)
        f.addRow("着色", self.cmb_color)

        self.sld_size = QSlider(Qt.Horizontal)
        self.sld_size.setRange(8, 40)
        self.sld_size.setValue(14)
        self.sld_size.valueChanged.connect(self._on_point_size)
        f.addRow("点大小", self.sld_size)

        row2 = QHBoxLayout()
        self.chk_all_scans = QCheckBox("显示全部扫描")
        self.chk_all_scans.setChecked(True)
        self.chk_all_scans.setToolTip("取消勾选后只显示列表中选中的扫描，便于近距标注")
        self.chk_all_scans.stateChanged.connect(self._on_scan_filter)
        row2.addWidget(self.chk_all_scans)
        row2.addStretch(1)
        f.addRow(row2)

        row3 = QHBoxLayout()
        self.chk_axes = QCheckBox("坐标轴")
        self.chk_axes.setChecked(True)
        self.chk_axes.setToolTip("在视图右下角显示 X/Y/Z 方向指示器")
        self.chk_axes.stateChanged.connect(self._on_overlay)
        self.chk_bbox = QCheckBox("包围盒")
        self.chk_bbox.setToolTip("用红色虚线框出点云的世界包围盒")
        self.chk_bbox.stateChanged.connect(self._on_overlay)
        row3.addWidget(self.chk_axes)
        row3.addWidget(self.chk_bbox)
        row3.addStretch(1)
        f.addRow(row3)

        self.chk_pose = QCheckBox("应用 E57 位姿（多站拼接）")
        self.chk_pose.setChecked(True)
        self.chk_pose.setToolTip(
            "按文件内每个扫描的 <pose> 把点变换到统一坐标系，与 CloudCompare 一致。\n"
            "关掉则按扫描自身的局部坐标显示（用于排查位姿错误）。")
        self.chk_pose.stateChanged.connect(self._on_pose_toggle)
        f.addRow(self.chk_pose)
        lay.addWidget(gb_view)

        # --- 扫描与标注列表 ---
        gb_list = QGroupBox("扫描 / 实例")
        vv = QVBoxLayout(gb_list)
        self.lst = QListWidget()
        self.lst.setAlternatingRowColors(False)
        self.lst.setMinimumHeight(180)
        self.lst.currentItemChanged.connect(self._on_list_sel)
        vv.addWidget(self.lst, 1)
        self.lbl_stat = QLabel("—")
        self.lbl_stat.setObjectName("hint")
        self.lbl_stat.setWordWrap(True)
        vv.addWidget(self.lbl_stat)
        lay.addWidget(gb_list, 1)

        # --- 保存 ---
        gb_save = QGroupBox("保存")
        vs = QVBoxLayout(gb_save)
        row3 = QHBoxLayout()
        self.btn_save = QPushButton("保存为 e57…")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self.save_as)
        self.btn_save2 = QPushButton("覆盖保存")
        self.btn_save2.clicked.connect(self.save_overwrite)
        row3.addWidget(self.btn_save, 2)
        row3.addWidget(self.btn_save2, 1)
        vs.addLayout(row3)
        self.btn_close = QPushButton("关闭当前文件 (Ctrl+W)")
        self.btn_close.setToolTip("清空视图、标注与队列，回到刚启动时的空白界面")
        self.btn_close.clicked.connect(self.close_file)
        vs.addWidget(self.btn_close)
        tip = QLabel("标注写入 e57 的 annotationLabel / annotationInstance 字段，\n"
                     "原始坐标、颜色、强度完整保留。")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        vs.addWidget(tip)
        lay.addWidget(gb_save)

        return panel

    @staticmethod
    def _sep() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#2f343c;")
        return line

    def _build_actions(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        self.addToolBar(tb)

        a_open = QAction("打开 e57…", self)
        a_open.setShortcut(QKeySequence.Open)
        a_open.triggered.connect(self.open_file)
        tb.addAction(a_open)

        a_open_dir = QAction("打开文件夹…", self)
        a_open_dir.setShortcut("Ctrl+Shift+O")
        a_open_dir.setToolTip("载入一个文件夹内的全部 e57，逐个处理")
        a_open_dir.triggered.connect(self.open_folder)
        tb.addAction(a_open_dir)

        a_close = QAction("关闭文件", self)
        a_close.setShortcut("Ctrl+W")
        a_close.setToolTip("关闭当前点云回到空白界面（不退出程序）；未保存会先提示")
        a_close.triggered.connect(self.close_file)
        tb.addAction(a_close)
        self.act_close = a_close

        a_save = QAction("保存", self)
        a_save.setShortcut(QKeySequence.Save)
        a_save.triggered.connect(self.save_overwrite)
        tb.addAction(a_save)
        self.act_save = a_save
        tb.addSeparator()

        a_crop = QAction("裁剪…", self)
        a_crop.setShortcut("C")
        a_crop.setToolTip("框选裁剪，或直接按世界坐标设定裁剪范围")
        a_crop.triggered.connect(self.crop_box_dialog)
        tb.addAction(a_crop)
        self.act_crop = a_crop
        tb.addSeparator()

        a_fit = QAction("适合窗口", self)
        a_fit.setShortcut("F")
        a_fit.triggered.connect(self.view.reset_view)
        tb.addAction(a_fit)

        a_top = QAction("俯视", self)
        a_top.triggered.connect(lambda: self._set_view(90, 89))
        tb.addAction(a_top)
        a_side = QAction("侧视", self)
        a_side.triggered.connect(lambda: self._set_view(0, 8))
        tb.addAction(a_side)
        a_iso = QAction("等轴", self)
        a_iso.triggered.connect(lambda: self._set_view(38, 28))
        tb.addAction(a_iso)
        tb.addSeparator()

        a_undo = QAction("撤销", self)
        a_undo.setShortcut(QKeySequence.Undo)
        a_undo.triggered.connect(self.undo)
        tb.addAction(a_undo)
        self.act_undo = a_undo
        tb.addSeparator()

        a_next = QAction("下一个文件", self)
        a_next.setShortcut("Ctrl+Right")
        a_next.setToolTip("处理文件夹队列中的下一个 e57")
        a_next.triggered.connect(self.next_in_queue)
        tb.addAction(a_next)
        self.act_next = a_next

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.lbl_title = QLabel(APP_NAME)
        self.lbl_title.setObjectName("title")
        tb.addWidget(self.lbl_title)

    # --------------------------------------------------------- 文件 --

    def _maybe_open_from_args(self) -> None:
        import sys
        for a in sys.argv[1:]:
            if a.lower().endswith(".e57") and os.path.isfile(a):
                self.load_file(a)
                return

    def open_file(self) -> None:
        start = self.settings.value("last_dir", r"D:\PCProjects\01wheat\FinalScripts\outputs")
        path, _ = QFileDialog.getOpenFileName(self, "打开 e57 点云", start, "E57 点云 (*.e57)")
        if path:
            self._queue = []
            self.load_file(path)

    def open_folder(self) -> None:
        start = self.settings.value("last_dir", r"D:\PCProjects\01wheat\FinalScripts\outputs")
        folder = QFileDialog.getExistingDirectory(self, "选择含 e57 的文件夹（含子目录）", start)
        if not folder:
            return
        files = []
        for root, _dirs, names in os.walk(folder):
            for nm in sorted(names):
                if nm.lower().endswith(".e57"):
                    files.append(os.path.join(root, nm))
        files.sort()
        if not files:
            QMessageBox.information(self, APP_NAME, "该文件夹内没有找到 .e57 文件。")
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
        self.status.showMessage("已关闭文件，点「打开 e57…」或「打开文件夹…」重新开始", 3000)

    def _update_actions(self) -> None:
        """没有打开文件时把依赖文档的按钮置灰。"""
        has = self.doc is not None
        for w in (getattr(self, "act_close", None), getattr(self, "act_save", None),
                  getattr(self, "act_crop", None), getattr(self, "act_undo", None),
                  getattr(self, "act_next", None), getattr(self, "btn_save", None),
                  getattr(self, "btn_save2", None), getattr(self, "btn_close", None)):
            if w is not None:
                w.setEnabled(has)
        n_scans = len(self.doc.scans) if has else 0
        if hasattr(self, "chk_all_scans"):
            self.chk_all_scans.setEnabled(n_scans > 1)

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
            doc = E57Document.read(path, apply_pose=self.apply_pose, force=force)
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
            self._refresh_list()
            self._update_title()
            el = time.perf_counter() - t0
            self.lbl_file.setText(
                f"{os.path.basename(path)}   ·   {doc.total_points:,} 点   ·   {len(doc.scans)} 扫描"
                + self._queue_status())
            self.lbl_perf.setText(f"读取 {el*1000:.0f} ms")
            self.settings.setValue("last_dir", os.path.dirname(path))
            self._dirty = ann.total_marked() > 0
            self.chk_all_scans.setEnabled(len(doc.scans) > 1)
            self.chk_all_scans.setChecked(True)
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
            self.save_overwrite()
            return not self._dirty
        return r == QMessageBox.Discard

    # --------------------------------------------------------- 保存 --

    def save_as(self) -> None:
        if self.doc is None:
            return
        base = self.path or ""
        d = os.path.splitext(base)[0] + "_labeled.e57" if base else ""
        path, _ = QFileDialog.getSaveFileName(self, "保存为 e57", d, "E57 点云 (*.e57)")
        if path:
            self._do_save(path)

    def save_overwrite(self) -> None:
        if self.doc is None:
            return
        if not self.path:
            self.save_as()
            return
        self._do_save(self.path)

    def _do_save(self, path: str) -> None:
        if not path.lower().endswith(".e57"):
            path += ".e57"
        dlg = QProgressDialog("正在写入 e57…", "取消", 0, 100, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(150)
        dlg.setValue(15)
        QApplication.processEvents()
        try:
            # 把标注写进扫描字段（不污染其它字段）
            if self._ensure_annotation_sync():
                self._refresh_list()
            for i, s in enumerate(self.doc.scans):
                self._attach_annotation(s, self.ann.labels[i], self.ann.instances[i])
            dlg.setValue(45)
            QApplication.processEvents()
            t0 = time.perf_counter()
            self.doc.save(path)
            el = time.perf_counter() - t0
            dlg.setValue(100)
            dlg.close()
            self.path = path
            self._dirty = False
            self._update_title()
            self.lbl_perf.setText(f"写入 {el:.1f} s")
            QMessageBox.information(self, APP_NAME,
                                    f"已保存：\n{path}\n\n"
                                    f"点云 {self.doc.total_points:,} 个，"
                                    f"标注 {self.ann.total_marked():,} 个点。")
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
        self._refresh_list()
        self._update_title()
        self.status.showMessage("已撤销裁剪", 2500)
        return True

    # --------------------------------------------------------- 交互 --

    def set_mode(self, mode: str) -> None:
        self.view.set_mode(mode)
        for b, m in ((self.btn_nav, MODE_NAV), (self.btn_brush, MODE_BRUSH),
                     (self.btn_erase, MODE_ERASE), (self.btn_box, MODE_BOX),
                     (self.btn_crop, MODE_CROP)):
            b.setChecked(m == mode)
        self.status.showMessage(
            {MODE_NAV: "浏览：左键旋转 · 右键/中键平移 · 滚轮缩放",
             MODE_BRUSH: "笔刷：左键涂抹 · 右键结束一笔",
             MODE_ERASE: "橡皮：左键擦除 · 右键结束一笔",
             MODE_BOX: "框选：左键拖出矩形，松手即标注（Ctrl+拖拽 = 穿透全部深度）",
             MODE_CROP: "裁剪：左键拖出矩形，松手后选择保留/删除"}[mode], 4000)

    def set_label(self, lid: int) -> None:
        self.view.current_label = lid
        self._label_btns[lid].setChecked(True)

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
        self.view.update()

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
            else:
                total += self.ann.apply(int(sid), mask, label)
        self._undo.append(group)
        if len(self._undo) > 60:
            self._undo = self._undo[-60:]
        r.refresh_labels(self.ann)
        self._dirty = True
        self._refresh_list()
        self._update_title()
        if label:
            name = LABEL_NAMES.get(label, "?")
            extra = f"（跨 {len(group)} 个扫描）" if len(group) > 1 else ""
            if not self.chk_all_scans.isChecked() and len(self.doc.scans) > 1:
                extra += f"（当前只显示扫描 #{int(group[0][0])}，其余站未参与）"
            self.status.showMessage(
                f"已标注 {total:,} 个点为「{name}」{extra} · 深度 {self._depth_text()}"
                f" · 点数偏少就把「拾取深度」调宽或按 Ctrl 拖拽", 4000)
        else:
            self.status.showMessage(
                f"已擦除 {total:,} 个点 · 深度 {self._depth_text()}", 2500)

    def undo(self) -> None:
        if self.undo_crop():
            return
        if not self._undo or self.ann is None:
            self.status.showMessage("没有可撤销的操作", 1500)
            return
        group = self._undo.pop()
        for sid, snap in reversed(group):
            self.ann.restore(sid, snap)
        self.view.refresh_labels(self.ann)
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
        self._dirty = True
        self._refresh_list()
        self._update_title()

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

    def _update_title(self) -> None:
        base = os.path.basename(self.path) if self.path else "未打开文件"
        star = " *" if self._dirty else ""
        self.setWindowTitle(f"{APP_NAME}  v{APP_VER}  —  {base}{star}")

    def closeEvent(self, ev):
        if self._dirty and not self._confirm_discard():
            ev.ignore()
            return
        ev.accept()
