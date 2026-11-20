"""点云裁剪页面：与器官标注完全独立的第二功能页。

职责
----
* 加载文件夹里的点云列表，双击打开任一文件（e57 / ply / pcd / xyz / txt / csv / pts）
* 在三维视图里用矩形或任意多边形圈出区域，选择保留区域内 / 区域外
* 俯视、前视、侧视、后视、底视、等轴等预设视角
* ``Ctrl+S`` 覆盖保存、``Ctrl+Shift+S`` 另存为（格式由扩展名决定）
* 绕 X / Y / Z 任意角度旋转（可先预览再烘焙进坐标，支持撤销）
* 调用 ``align_crop_pointcloud.py``，用点云1(PLY) 对点云2(E57) 做配准裁剪
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup,
                               QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit,
                               QPushButton, QRadioButton, QScrollArea, QSizePolicy,
                               QSlider, QSpinBox, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

import cloudio
from align_runner import (AlignRunner, DEFAULT_SCRIPT, candidate_pythons,
                          default_output, probe_python, validate_inputs, unique_output)
from annotation import AnnotationSet
from cropping import apply_masks
from crop_view import (CropView, MODE_NAV, MODE_RECT, MODE_POLY, VIEW_PRESETS)
from e57io import E57Document, StructureError
from office_theme import BLUE, CANVAS_COLORS

TWO_LINE = "做什么 + 什么时候用 / 注意什么"  # tooltip 文案约定：两行


def _tip(what: str, when: str) -> str:
    return f"{what}\n{when}"


class CropPage(QWidget):
    """裁剪页根控件。"""

    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cropPage")

        self.doc: Optional[E57Document] = None
        self.ann: Optional[AnnotationSet] = None
        self.path: Optional[str] = None
        self._dirty = False
        self._undo: List[list] = []
        self._files: List[str] = []
        self._folder = ""
        self._rot_preview = False
        self._canvas_name = "深色（默认）"

        self._build_ui()
        self._build_shortcuts()
        self._update_actions()

    # ------------------------------------------------------------ 界面 --

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_center())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([280, 900, 320])
        split.setStyleSheet("QSplitter::handle:horizontal { background: #dfe3e8; }")
        root.addWidget(split)

    # -- 左：文件列表 / 配准裁剪 -----------------------------------------

    def _build_left(self) -> QWidget:
        box = QWidget()
        box.setMinimumWidth(250)
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        tabs.addTab(self._build_files_tab(), "文件列表")
        tabs.addTab(self._build_align_tab(), "配准裁剪")
        v.addWidget(tabs)
        return box

    def _build_files_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        row = QHBoxLayout()
        self.btn_folder = QPushButton("打开文件夹…")
        self.btn_folder.setObjectName("primary")
        self.btn_folder.setToolTip(_tip(
            "选择一个文件夹，递归收集里面所有支持的点云文件并列出。",
            "批量处理时用；支持 e57 / ply / pcd / xyz / txt / csv / pts。"))
        self.btn_folder.clicked.connect(self.open_folder)
        row.addWidget(self.btn_folder)
        self.btn_rescan = QPushButton("刷新")
        self.btn_rescan.setToolTip(_tip("重新扫描当前文件夹，更新文件列表。",
                                        "往文件夹里新拷了文件后用，不用重新选目录。"))
        self.btn_rescan.clicked.connect(self._rescan)
        row.addWidget(self.btn_rescan)
        v.addLayout(row)

        self.lst_files = QListWidget()
        self.lst_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst_files.setToolTip(_tip(
            "当前文件夹里的点云文件，双击打开并载入三维视图。",
            "单击只选中，双击才加载；文件名显示的是相对路径。"))
        self.lst_files.itemDoubleClicked.connect(self._on_file_double_clicked)
        v.addWidget(self.lst_files, 1)

        self.lbl_files = QLabel("尚未选择文件夹")
        self.lbl_files.setObjectName("hint")
        v.addWidget(self.lbl_files)

        self.btn_load = QPushButton("加载选中的文件")
        self.btn_load.setToolTip(_tip("把列表里当前选中的文件载入三维视图。",
                                      "等价于双击列表项；已有未保存改动时会先询问。"))
        self.btn_load.clicked.connect(self._load_selected)
        v.addWidget(self.btn_load)
        return w

    def _build_align_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(5)

        hint = QLabel("用点云1（PLY 参考范围）对点云2（E57）做配准 + 保真裁剪")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(5)

        self.cmb_ref = QComboBox()
        self.cmb_ref.setToolTip(_tip("点云1：参考 PLY，脚本按它的点范围裁剪点云2。",
                                     "从左侧文件列表里挑一份 PLY，或用「浏览」另选。"))
        self.btn_ref = QPushButton("…")
        self.btn_ref.setFixedWidth(30)
        self.btn_ref.setToolTip(_tip("浏览选择点云1（参考 PLY）。", "文件不在当前文件夹里时用。"))
        self.btn_ref.clicked.connect(lambda: self._browse_into(self.cmb_ref, "PLY 点云 (*.ply)"))
        form.addRow("点云1 (PLY)", self._with_button(self.cmb_ref, self.btn_ref))

        self.cmb_src = QComboBox()
        self.cmb_src.setToolTip(_tip("点云2：待配准与裁剪的 E57。",
                                     "脚本会保留它全部原始字段/位姿/图像并逐字段自检。"))
        self.btn_src = QPushButton("…")
        self.btn_src.setFixedWidth(30)
        self.btn_src.setToolTip(_tip("浏览选择点云2（待裁剪 E57）。", "文件不在当前文件夹里时用。"))
        self.btn_src.clicked.connect(lambda: self._browse_into(self.cmb_src, "E57 点云 (*.e57)"))
        form.addRow("点云2 (E57)", self._with_button(self.cmb_src, self.btn_src))

        self.edt_out = QLineEdit()
        self.edt_out.setToolTip(_tip("配准裁剪结果的输出 E57 路径。",
                                     "选好点云2 后自动填 <原名>_cropped.e57；脚本不会覆盖已存在文件。"))
        self.btn_out = QPushButton("…")
        self.btn_out.setFixedWidth(30)
        self.btn_out.setToolTip(_tip("浏览选择输出位置。", "输出必须是 .e57，且不能和输入同名。"))
        self.btn_out.clicked.connect(self._browse_output)
        form.addRow("输出 E57", self._with_button(self.edt_out, self.btn_out))
        v.addLayout(form)

        # 常用参数
        g = QGroupBox("配准参数")
        gf = QFormLayout(g)
        gf.setSpacing(4)
        self.cmb_reg = QComboBox()
        self.cmb_reg.addItems(["auto", "global", "icp", "none"])
        self.cmb_reg.setToolTip(_tip(
            "配准方式：auto 自动、global 加 FPFH+RANSAC 粗配准、icp 精配准、none 只按初始位姿裁剪。",
            "两片点云已经对齐时选 none 最快；错得较多时用 auto/global。"))
        gf.addRow("方式", self.cmb_reg)

        self.spn_crop_d = QDoubleSpinBox()
        self.spn_crop_d.setDecimals(4)
        self.spn_crop_d.setRange(0.0, 1e6)
        self.spn_crop_d.setSpecialValueText("自动（参考点间距 ×3）")
        self.spn_crop_d.setToolTip(_tip(
            "点云2 的点到点云1 的最近距离小于该值才保留。",
            "0 表示按参考点间距自动估算；单位与输入坐标一致（本软件为 cm）。"))
        gf.addRow("裁剪距离", self.spn_crop_d)

        self.spn_voxel = QDoubleSpinBox()
        self.spn_voxel.setDecimals(4)
        self.spn_voxel.setRange(0.0, 1e6)
        self.spn_voxel.setSpecialValueText("自动")
        self.spn_voxel.setToolTip(_tip("配准阶段的降采样体素大小，只影响速度，最终导出不采样。",
                                       "0 表示自动；配准点数太少导致报错时把它调小。"))
        gf.addRow("体素大小", self.spn_voxel)

        self.spn_overlap = QDoubleSpinBox()
        self.spn_overlap.setDecimals(2)
        self.spn_overlap.setRange(0.0, 1.0)
        self.spn_overlap.setSingleStep(0.05)
        self.spn_overlap.setValue(0.10)
        self.spn_overlap.setToolTip(_tip("配准重叠率低于该值就判定失败并中止。",
                                         "默认 0.10；明显配歪时调高，避免导出错误结果。"))
        gf.addRow("最小重叠", self.spn_overlap)

        self.spn_workers = QSpinBox()
        self.spn_workers.setRange(1, 32)
        self.spn_workers.setValue(4)
        self.spn_workers.setToolTip(_tip("KDTree 查询的并行线程数。", "核多可以调大，默认 4。"))
        gf.addRow("线程数", self.spn_workers)
        v.addWidget(g)

        self.chk_refine = QCheckBox("颜色位姿再精配准")
        self.chk_refine.setToolTip(_tip(
            "若颜色对应点已给出可信位姿，再额外做一轮近邻精配准。",
            "默认关闭，保持经过验证的颜色位姿；数据噪声大时可打开。"))
        v.addWidget(self.chk_refine)

        adv = QGroupBox("运行环境")
        af = QFormLayout(adv)
        af.setSpacing(4)
        self.edt_python = QLineEdit()
        self.edt_python.setToolTip(_tip(
            "运行配准脚本的 Python 解释器（需装 open3d / plyfile / pye57 / scipy / matplotlib）。",
            "点「探测」自动挑一个依赖齐全的环境；脚本以外部进程运行，失败不影响本软件。"))
        self.btn_python = QPushButton("…")
        self.btn_python.setFixedWidth(30)
        self.btn_python.setToolTip(_tip("浏览选择 python.exe。", "自动探测找不到环境时手动指定。"))
        self.btn_python.clicked.connect(self._browse_python)
        af.addRow("Python", self._with_button(self.edt_python, self.btn_python))

        self.edt_script = QLineEdit(DEFAULT_SCRIPT)
        self.edt_script.setToolTip(_tip("配准裁剪脚本的完整路径。",
                                        "默认指向 align_crop_pointcloud.py；脚本换了位置就改这里。"))
        self.btn_script = QPushButton("…")
        self.btn_script.setFixedWidth(30)
        self.btn_script.setToolTip(_tip("浏览选择配准脚本。", "脚本被移动或换版本时用。"))
        self.btn_script.clicked.connect(self._browse_script)
        af.addRow("脚本", self._with_button(self.edt_script, self.btn_script))
        v.addWidget(adv)

        row2 = QHBoxLayout()
        self.btn_probe = QPushButton("探测环境")
        self.btn_probe.setToolTip(_tip("检查已填的 Python 是否装齐脚本依赖。",
                                       "首次运行前点一下，缺包会明确列出，不用等脚本报错。"))
        self.btn_probe.clicked.connect(self._probe_env)
        row2.addWidget(self.btn_probe)
        self.btn_run = QPushButton("运行配准裁剪")
        self.btn_run.setObjectName("primary")
        self.btn_run.setToolTip(_tip(
            "调用脚本：把点云2 配准到点云1，只留下落在参考范围内的点，导出保真 E57。",
            "点云1 必须是 PLY、点云2 必须是 E57；耗时取决于点数，日志在下方实时滚动。"))
        self.btn_run.clicked.connect(self.run_align)
        row2.addWidget(self.btn_run)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setToolTip(_tip("强制终止正在运行的配准任务。",
                                      "参数填错或等太久时用；已生成的临时文件不会自动清理。"))
        self.btn_stop.clicked.connect(self.stop_align)
        row2.addWidget(self.btn_stop)
        v.addLayout(row2)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(2000)
        self.txt_log.setMinimumHeight(90)
        self.txt_log.setPlaceholderText("运行日志（配准进度、点数、自检结果）")
        self.txt_log.setToolTip(_tip("脚本的标准输出/错误实时显示在这里。",
                                     "失败时先看最后几行；成功会给出保留点数与输出路径。"))
        v.addWidget(self.txt_log, 1)

        self.runner = AlignRunner(self)
        self.runner.logLine.connect(self._append_log)
        self.runner.started.connect(self._on_align_started)
        self.runner.finished.connect(self._on_align_finished)
        # 不在构造期同步 import open3d（可能卡住启动数秒）：先填首选解释器，
        # 真正运行前 AlignRunner.run() 会做依赖检查，用户也可点「探测环境」。
        cands = candidate_pythons()
        self.edt_python.setText(cands[0] if cands else sys.executable)
        self.cmb_src.currentIndexChanged.connect(self._on_source_changed)
        return w

    @staticmethod
    def _with_button(w: QWidget, b: QPushButton) -> QWidget:
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        h.addWidget(w, 1)
        h.addWidget(b)
        box = QWidget()
        box.setLayout(h)
        return box

    # -- 中：视图 + 视图工具行 -------------------------------------------

    def _build_center(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(5)

        bar = QHBoxLayout()
        bar.setSpacing(4)
        self._view_buttons = []
        for name in VIEW_PRESETS:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setToolTip(_tip(f"切换到{name}视角，便于沿该方向圈裁剪区域。",
                              "在俯视/前视/侧视下画的区域是屏幕投影范围，深度方向默认全部穿透。"))
            b.clicked.connect(lambda checked=False, n=name: self.set_view(n))
            bar.addWidget(b)
            self._view_buttons.append((name, b))
        self.btn_fit = QPushButton("适合窗口")
        self.btn_fit.setToolTip(_tip("让整份点云刚好填满视图。", "旋转或缩放过头后按 F / 点这里复位。"))
        self.btn_fit.clicked.connect(self.fit_view)
        bar.addWidget(self.btn_fit)
        bar.addStretch(1)
        v.addLayout(bar)

        self.view = CropView()
        self.view.regionDefined.connect(self._on_region)
        self.view.hoverInfo.connect(lambda s: self.statusMessage.emit(s) if s else None)
        self.view.notice.connect(lambda s: self.statusMessage.emit(s))
        v.addWidget(self.view, 1)

        self.lbl_info = QLabel("尚未载入点云")
        self.lbl_info.setObjectName("hint")
        v.addWidget(self.lbl_info)
        return w

    # -- 右：裁剪 / 旋转 / 显示 / 保存 ------------------------------------

    def _build_right(self) -> QWidget:
        outer = QWidget()
        outer.setMinimumWidth(290)
        outer.setMaximumWidth(440)
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(0, 0, 0, 0)

        # 四组内容在 1366×768 / 系统缩放 150% 下会放不下。放进滚动区，
        # 并给内容一个最小高度，避免 QGroupBox 被压成 0 高（标注页侧栏踩过这个坑）。
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(7)
        v.addWidget(self._build_crop_group())
        v.addWidget(self._build_rotate_group())
        v.addWidget(self._build_display_group())
        v.addWidget(self._build_save_group())
        v.addStretch(1)
        body.setMinimumHeight(660)
        scroll.setWidget(body)
        lay.addWidget(scroll)
        return outer

    def _build_crop_group(self) -> QGroupBox:
        g = QGroupBox("裁剪区域")
        v = QVBoxLayout(g)
        v.setSpacing(5)

        self.rb_nav = QRadioButton("浏览（拖动旋转 / 平移）")
        self.rb_nav.setToolTip(_tip("只看不画：左键拖动旋转视角，右键/中键拖动平移，滚轮缩放。",
                                    "想换个角度观察点云时选它；要圈区域再切到矩形或多边形。"))
        self.rb_rect = QRadioButton("矩形框选")
        self.rb_rect.setToolTip(_tip("在视图里左键拖出一个矩形作为裁剪区域。",
                                     "适合小区、矩形样方；松开鼠标即确定区域。"))
        self.rb_poly = QRadioButton("任意多边形（套索）")
        self.rb_poly.setToolTip(_tip("逐点点击勾出不规则轮廓，双击 / 右键 / Enter 闭合。",
                                     "适合单株、弧形边界；Esc 取消，Backspace 退一点。"))
        self.rb_inside = QRadioButton("只保留区域内的点")
        self.rb_inside.setToolTip(_tip("裁出圈中的部分，其余点删掉。",
                                       "截单株、取小区时选这个。"))
        self.rb_outside = QRadioButton("删除区域内、保留区域外")
        self.rb_outside.setToolTip(_tip("把圈中的部分挖掉，其余点保留。",
                                        "去地面、去支架、去杂株时选这个。"))

        # 「形状」和「保留方式」是两组独立的互斥选项。QRadioButton 默认
        # autoExclusive=True 且只看父控件：几个都放进同一个 QGroupBox 时会
        # 变成「全选一」，切保留方式会把形状选择也取消掉（视图悄悄回到浏览模式）。
        # 这里关掉 widget 自带的互斥，改由两个 QButtonGroup 各自管理。
        self._grp_shape = QButtonGroup(self)
        self._grp_keep = QButtonGroup(self)
        self._grp_shape.setExclusive(True)
        self._grp_keep.setExclusive(True)
        for rb in (self.rb_nav, self.rb_rect, self.rb_poly,
                   self.rb_inside, self.rb_outside):
            rb.setAutoExclusive(False)
        for rb in (self.rb_nav, self.rb_rect, self.rb_poly):
            self._grp_shape.addButton(rb)
        for rb in (self.rb_inside, self.rb_outside):
            self._grp_keep.addButton(rb)
        self.rb_nav.toggled.connect(lambda on: self.view.set_mode(MODE_NAV) if on else None)
        self.rb_rect.toggled.connect(lambda on: self.view.set_mode(MODE_RECT) if on else None)
        self.rb_poly.toggled.connect(lambda on: self.view.set_mode(MODE_POLY) if on else None)
        # 默认进「浏览」：打开文件就能用鼠标拖动旋转（用户最直接的期望）。
        # 要圈区域时再点矩形/多边形，模式切换是显式单选，不会出现
        # 「界面选着矩形、鼠标却在旋转」的错觉。
        self.rb_nav.setChecked(True)
        self.rb_inside.setChecked(True)
        v.addWidget(self.rb_nav)
        v.addWidget(self.rb_rect)
        v.addWidget(self.rb_poly)

        sep = QLabel("保留方式")
        sep.setObjectName("section")
        v.addWidget(sep)
        v.addWidget(self.rb_inside)
        v.addWidget(self.rb_outside)

        self.cmb_depth = QComboBox()
        self.cmb_depth.addItem("全部穿透（默认）", "all")
        self.cmb_depth.addItem("仅最前面一层", "surface")
        self.cmb_depth.setToolTip(_tip(
            "区域内沿视线方向的点是否全部参与：默认全部；仅表面层只动最前面一层。",
            "裁剪一般选全部穿透；只想修表面噪点才用「仅最前面一层」。"))
        # 改深度后命中数会变，标签必须同步刷新，否则显示的「命中 N 点」
        # 与「应用裁剪」实际删掉的点数对不上。
        self.cmb_depth.currentIndexChanged.connect(self._on_depth_changed)
        v.addWidget(self.cmb_depth)

        row = QHBoxLayout()
        self.btn_apply = QPushButton("应用裁剪")
        self.btn_apply.setObjectName("primary")
        self.btn_apply.setToolTip(_tip(
            "按当前区域和保留方式真正删点，可用 Ctrl+Z 整步撤销。",
            "区域画完再点；应用后视图会刷新，区域自动清空。"))
        self.btn_apply.clicked.connect(self.apply_crop)
        row.addWidget(self.btn_apply)
        self.btn_clear_region = QPushButton("清除区域")
        self.btn_clear_region.setToolTip(_tip("擦掉画好的区域轮廓，不改变点云。",
                                              "圈错了重画时用；不会撤销已应用的裁剪。"))
        self.btn_clear_region.clicked.connect(self.view.clear_region)
        row.addWidget(self.btn_clear_region)
        v.addLayout(row)

        row2 = QHBoxLayout()
        self.btn_undo = QPushButton("撤销 (Ctrl+Z)")
        self.btn_undo.setToolTip(_tip("退回上一次裁剪/旋转之前的状态。",
                                      "一次操作 = 一个撤销步；可连续撤销多步。"))
        self.btn_undo.clicked.connect(self.undo)
        row2.addWidget(self.btn_undo)
        self.btn_reload = QPushButton("重新载入")
        self.btn_reload.setToolTip(_tip("放弃所有改动，从磁盘重新读取当前文件。",
                                        "改乱了又没保存时用；会清空撤销栈。"))
        self.btn_reload.clicked.connect(self.reload)
        row2.addWidget(self.btn_reload)
        v.addLayout(row2)

        self.lbl_region = QLabel("区域：未绘制")
        self.lbl_region.setObjectName("hint")
        self.lbl_region.setWordWrap(True)
        v.addWidget(self.lbl_region)
        return g

    def _build_rotate_group(self) -> QGroupBox:
        g = QGroupBox("绕轴旋转")
        f = QFormLayout(g)
        f.setSpacing(5)
        self.spn_rx = QDoubleSpinBox()
        self.spn_ry = QDoubleSpinBox()
        self.spn_rz = QDoubleSpinBox()
        for sp, axis in ((self.spn_rx, "X"), (self.spn_ry, "Y"), (self.spn_rz, "Z")):
            sp.setDecimals(2)
            sp.setRange(-360.0, 360.0)
            sp.setSingleStep(5.0)
            sp.setSuffix(" °")
            sp.setValue(0.0)
            sp.setToolTip(_tip(f"绕 {axis} 轴旋转的角度，可填任意正负值。",
                               "三个轴可叠加（按 X→Y→Z 顺序）；先「预览」看效果，再「应用」写进坐标。"))
            sp.valueChanged.connect(self._on_rotation_changed)
        f.addRow("绕 X", self.spn_rx)
        f.addRow("绕 Y", self.spn_ry)
        f.addRow("绕 Z", self.spn_rz)

        self.chk_center = QCheckBox("绕点云质心旋转")
        self.chk_center.setChecked(True)
        self.chk_center.setToolTip(_tip("以整份点云的包围中心为旋转中心。",
                                        "默认开启，点云不会「飞走」；关掉则绕坐标原点旋转。"))
        f.addRow("", self.chk_center)

        row = QHBoxLayout()
        self.btn_rot_preview = QPushButton("预览")
        self.btn_rot_preview.setCheckable(True)
        self.btn_rot_preview.setToolTip(_tip("只改变视图里的显示姿态，不改文件坐标。",
                                             "勾选后拖动角度数值会实时更新；再点一次关闭预览。"))
        self.btn_rot_preview.toggled.connect(self._on_preview_toggled)
        row.addWidget(self.btn_rot_preview)
        self.btn_rot_apply = QPushButton("应用旋转")
        self.btn_rot_apply.setToolTip(_tip(
            "把当前角度烘焙进每个点的坐标（旋转矩阵写进坐标，不再改 E57 位姿）。",
            "应用后可用 Ctrl+Z 撤销；保存时写出的是旋转后的坐标。"))
        self.btn_rot_apply.clicked.connect(self.apply_rotation)
        row.addWidget(self.btn_rot_apply)
        self.btn_rot_reset = QPushButton("复位")
        self.btn_rot_reset.setToolTip(_tip("三个角度归零并关闭预览。",
                                           "不改动已应用的旋转；要撤回请用 Ctrl+Z。"))
        self.btn_rot_reset.clicked.connect(self.reset_rotation)
        row.addWidget(self.btn_rot_reset)
        rowbox = QWidget()
        rowbox.setLayout(row)
        f.addRow(rowbox)
        return g

    def _build_display_group(self) -> QGroupBox:
        g = QGroupBox("显示")
        f = QFormLayout(g)
        f.setSpacing(5)
        self.cmb_canvas = QComboBox()
        self.cmb_canvas.addItems(list(CANVAS_COLORS) + ["自定义…"])
        self.cmb_canvas.setToolTip(_tip("三维视图的画布底色。",
                                        "真彩色点云在深色底上对比更强；出图时可切白色；"
                                        "选「自定义…」可挑任意颜色。"))
        self.cmb_canvas.currentTextChanged.connect(self._on_canvas_changed)
        f.addRow("画布", self.cmb_canvas)

        self.cmb_color = QComboBox()
        self.cmb_color.addItem("真彩色", "rgb")
        self.cmb_color.addItem("高度", "height")
        self.cmb_color.addItem("强度", "intensity")
        self.cmb_color.addItem("标注", "label")
        self.cmb_color.setToolTip(_tip("点云着色方式。",
                                       "裁剪时一般用真彩色；想看清层次可切高度，检查已有标注切标注。"))
        self.cmb_color.currentIndexChanged.connect(
            lambda: self.view.set_color_mode(self.cmb_color.currentData()))
        f.addRow("着色", self.cmb_color)

        self.sld_size = QSlider(Qt.Horizontal)
        self.sld_size.setRange(5, 40)
        self.sld_size.setValue(14)
        self.sld_size.setToolTip(_tip("点显示大小（1.4 对应这里的 14）。",
                                      "点稀疏时调大更容易看清轮廓，不影响数据。"))
        self.sld_size.valueChanged.connect(lambda x: self.view.set_point_size(x / 10.0))
        f.addRow("点大小", self.sld_size)
        return g

    def _build_save_group(self) -> QGroupBox:
        g = QGroupBox("保存")
        v = QVBoxLayout(g)
        v.setSpacing(5)
        self.lbl_path = QLabel("当前文件：—")
        self.lbl_path.setObjectName("hint")
        self.lbl_path.setWordWrap(True)
        v.addWidget(self.lbl_path)

        self.btn_save = QPushButton("覆盖保存 (Ctrl+S)")
        self.btn_save.setObjectName("primary")
        self.btn_save.setToolTip(_tip(
            "直接按原格式写回当前文件，不弹路径选择框。",
            "会覆盖磁盘上的原文件，执行前有一次确认；想留原件请用「另存为」。"))
        self.btn_save.clicked.connect(self.save_overwrite)
        v.addWidget(self.btn_save)

        self.btn_save_as = QPushButton("另存为… (Ctrl+Shift+S)")
        self.btn_save_as.setToolTip(_tip("另存为一个新文件，格式由扩展名决定。",
                                         "支持 e57 / ply / pcd / xyz / txt / csv；原文件保持不变。"))
        self.btn_save_as.clicked.connect(self.save_as)
        v.addWidget(self.btn_save_as)
        return g

    # ---------------------------------------------------------- 快捷键 --

    def _build_shortcuts(self) -> None:
        self._shortcuts = []
        for seq, fn in (("Ctrl+S", self.save_overwrite),
                        ("Ctrl+Shift+S", self.save_as),
                        ("Ctrl+Z", self.undo)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

    def set_shortcuts_enabled(self, on: bool) -> None:
        """主窗口切换页面时调用，避免和标注页的同名快捷键抢事件。"""
        for sc in self._shortcuts:
            sc.setEnabled(on)

    # ------------------------------------------------------------ 文件 --

    def open_folder(self) -> None:
        start = self._folder or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "选择点云文件夹", start)
        if not d:
            return
        self._folder = d
        self._rescan()

    def _rescan(self) -> None:
        if not self._folder:
            self.open_folder()
            return
        files: List[str] = []
        for base, _dirs, names in os.walk(self._folder):
            for n in names:
                p = os.path.join(base, n)
                if cloudio.is_supported(p):
                    files.append(p)
        files.sort()
        self._files = files
        self.lst_files.clear()
        for p in files:
            rel = os.path.relpath(p, self._folder)
            it = QListWidgetItem(rel)
            it.setData(Qt.UserRole, p)
            it.setToolTip(p)
            self.lst_files.addItem(it)
        self.lbl_files.setText(f"{len(files)} 个点云文件  ·  {self._folder}")
        self._fill_align_combos()
        self.statusMessage.emit(f"已扫描 {len(files)} 个点云文件：{self._folder}")

    def _fill_align_combos(self) -> None:
        keep_ref = self.cmb_ref.currentData()
        keep_src = self.cmb_src.currentData()
        self.cmb_ref.clear()
        self.cmb_src.clear()
        self.cmb_ref.addItem("（浏览选择…）", "")
        self.cmb_src.addItem("（浏览选择…）", "")
        for p in self._files:
            rel = os.path.relpath(p, self._folder) if self._folder else os.path.basename(p)
            ext = Path(p).suffix.lower()
            if ext == ".ply":
                self.cmb_ref.addItem(rel, p)
            if ext == ".e57":
                self.cmb_src.addItem(rel, p)
        for cmb, keep in ((self.cmb_ref, keep_ref), (self.cmb_src, keep_src)):
            if keep:
                i = cmb.findData(keep)
                cmb.setCurrentIndex(i if i >= 0 else 0)
            elif cmb.count() > 1:
                # 文件夹里恰好有一种候选文件时自动预选，省一次点击，
                # 也顺带触发输出路径自动推导
                cmb.setCurrentIndex(1)
        self._on_source_changed()

    def _on_source_changed(self) -> None:
        p = self.cmb_src.currentData()
        if not p:
            return
        # 只接管「自动填的」输出路径；用户手改过或另选过就不再覆盖
        cur = self.edt_out.text()
        if not cur or cur == getattr(self, "_auto_out", ""):
            self._auto_out = unique_output(default_output(p))
            self.edt_out.setText(self._auto_out)

    def _on_file_double_clicked(self, item) -> None:
        p = item.data(Qt.UserRole)
        if p:
            self.load_file(p)

    def _load_selected(self) -> None:
        it = self.lst_files.currentItem()
        if it:
            self.load_file(it.data(Qt.UserRole))

    def load_file(self, path: str) -> None:
        if not self._confirm_discard():
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ext = cloudio.ext_of(path)
            if ext == ".e57":
                try:
                    doc = E57Document.read(path)
                except StructureError as e:
                    r = QMessageBox.warning(
                        self, "文件体检未通过",
                        f"{path}\n\n结构自检发现问题：\n{e}\n\n仍要尝试读取吗？",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if r != QMessageBox.Yes:
                        return
                    doc = E57Document.read(path, force=True)
            else:
                doc = cloudio.load(path)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "读取失败", f"{path}\n\n{e}")
            return
        QApplication.restoreOverrideCursor()

        self.doc = doc
        self.path = path
        self.ann = AnnotationSet.from_scans(doc.scans)
        self._undo.clear()
        self._dirty = False
        self._rot_preview = False
        # 命令行 / 配准结果载入时也要把列表选中项跟过去，否则用户看不出
        # 视图里是哪一份文件
        for i in range(self.lst_files.count()):
            it = self.lst_files.item(i)
            if os.path.normcase(it.data(Qt.UserRole) or "") == os.path.normcase(path):
                self.lst_files.setCurrentItem(it)
                break
        self.view.set_document(doc, self.ann)
        self.set_view(self.view.view_name)
        self._update_actions()
        self._refresh_info()
        self.statusMessage.emit(f"已载入 {path}  ·  {doc.total_points:,} 点")

    def reload(self) -> None:
        if self.path:
            self._dirty = False
            self.load_file(self.path)

    def _confirm_discard(self) -> bool:
        if not self._dirty or self.doc is None:
            return True
        r = QMessageBox.question(
            self, "未保存的改动",
            "当前点云有未保存的裁剪/旋转改动，载入新文件会丢弃它们。\n是否继续？",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No)
        return r == QMessageBox.Yes

    # ------------------------------------------------------------ 视图 --

    def set_view(self, name: str) -> None:
        for n, b in self._view_buttons:
            b.setChecked(n == name)
        self.view.set_view(name)

    def fit_view(self) -> None:
        self.view.fit_view()

    def _on_canvas_changed(self, name: str) -> None:
        """画布底色：预设直接切，「自定义…」弹取色器。"""
        if name == "自定义…":
            c = QColorDialog.getColor(QColor(*self.view.renderer.bg), self,
                                      "选择画布颜色")
            if c.isValid():
                self.view.set_canvas_rgb((c.red(), c.green(), c.blue()))
                self._canvas_name = "自定义"
            else:
                self.cmb_canvas.blockSignals(True)
                self.cmb_canvas.setCurrentText(self._canvas_name)
                self.cmb_canvas.blockSignals(False)
            return
        self._canvas_name = name
        self.view.set_canvas(name)

    # ------------------------------------------------------------ 区域 --

    def _region_hits(self, region) -> int:
        return int(self.view.region_mask(region, self.cmb_depth.currentData()).sum())

    def _update_region_label(self, region) -> None:
        n = self._region_hits(region)
        kind = "矩形" if region[0] == "rect" else "多边形"
        self.lbl_region.setText(f"区域：{kind}  命中 {n:,} / {self.doc.total_points:,} 点"
                                if self.doc else f"区域：{kind}")

    def _on_region(self, region) -> None:
        self._update_region_label(region)
        self._update_actions()
        kind = "矩形" if region[0] == "rect" else "多边形"
        self.statusMessage.emit(
            f"已圈定{kind}区域，命中 {self._region_hits(region):,} 点；"
            "设置保留方式后点「应用裁剪」")

    def _on_depth_changed(self, _i=None) -> None:
        region = self.view.region()
        if region is not None:
            self._update_region_label(region)

    def _drop_region_for_rotation(self) -> None:
        """预览旋转会挪动点的屏幕位置，已圈的屏幕区域随即失效。"""
        if self.view.region() is None:
            return
        self.view.clear_region()
        self.lbl_region.setText("区域：已清除（旋转改变了点的位置），请重新圈定")
        self._update_actions()

    # ------------------------------------------------------------ 裁剪 --

    def _snapshot(self) -> None:
        if self.doc is not None:
            self._undo.append(copy.deepcopy(self.doc.scans))
            if len(self._undo) > 20:
                self._undo.pop(0)

    def apply_crop(self) -> None:
        if self.doc is None:
            QMessageBox.information(self, "提示", "请先载入一个点云文件。")
            return
        region = self.view.region()
        if region is None:
            QMessageBox.information(self, "提示", "请先在视图里画一块裁剪区域。")
            return
        depth = self.cmb_depth.currentData()
        inside = self.view.region_mask(region, depth)
        if inside.size != self.doc.total_points:
            QMessageBox.warning(self, "无法裁剪",
                                "区域点数与文档点数不一致，请重新载入文件后重试。")
            return
        keep = inside if self.rb_inside.isChecked() else ~inside
        if not keep.any():
            QMessageBox.information(self, "提示", "该操作会删掉全部点，已取消。")
            return
        if keep.all():
            QMessageBox.information(self, "提示", "区域内没有需要删除的点，未做修改。")
            return

        try:
            self._snapshot()
            self._commit_preview_rotation()
            r = self.view.renderer
            masks = []
            for i, s in enumerate(self.doc.scans):
                sel = r.scan_ids == i
                m = np.zeros(s.count, bool)
                if sel.any():
                    m[r.local_index[sel]] = keep[sel]
                masks.append(m if m.any() else None)
            before, after = apply_masks(self.doc, masks)
            self.ann = AnnotationSet.from_scans(self.doc.scans)
            self._dirty = True
            self.view.set_document(self.doc, self.ann)
            self.view.set_view(self.view.view_name)
            self._update_actions()
            self._refresh_info()
            self.statusMessage.emit(f"裁剪完成：{before:,} → {after:,} 点（删除 {before - after:,}）")
        except Exception as e:  # noqa: BLE001
            self._undo.pop()
            QMessageBox.critical(self, "裁剪失败", str(e))

    def undo(self) -> None:
        if not self._undo or self.doc is None:
            return
        self.doc.scans = self._undo.pop()
        self.ann = AnnotationSet.from_scans(self.doc.scans)
        self.view.set_document(self.doc, self.ann)
        self.view.set_view(self.view.view_name)
        self._dirty = bool(self._undo)
        self._refresh_info()
        self._update_actions()
        self.statusMessage.emit("已撤销上一步")

    # ------------------------------------------------------------ 旋转 --

    @staticmethod
    def _rot_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
        a, b, c = np.radians([rx, ry, rz])
        cx, sx = np.cos(a), np.sin(a)
        cy, sy = np.cos(b), np.sin(b)
        cz, sz = np.cos(c), np.sin(c)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
        return Rz @ Ry @ Rx

    def _rotation_center(self) -> np.ndarray:
        if self.chk_center.isChecked() and self.doc is not None:
            xs = [np.asarray(s.points["cartesianX"], np.float64)
                  for s in self.doc.scans if "cartesianX" in s.points and s.count]
            if xs:
                x = np.concatenate(xs)
                ys = np.concatenate([np.asarray(s.points["cartesianY"], np.float64)
                                     for s in self.doc.scans if "cartesianY" in s.points and s.count])
                zs = np.concatenate([np.asarray(s.points["cartesianZ"], np.float64)
                                     for s in self.doc.scans if "cartesianZ" in s.points and s.count])
                return np.array([(x.min() + x.max()) / 2, (ys.min() + ys.max()) / 2,
                                 (zs.min() + zs.max()) / 2], np.float64)
        return np.zeros(3, np.float64)

    def _on_preview_toggled(self, on: bool) -> None:
        self._rot_preview = on
        self._drop_region_for_rotation()
        self._update_preview()

    def _on_rotation_changed(self, _v=None) -> None:
        if self._rot_preview:
            self._drop_region_for_rotation()
            self._update_preview()

    def _update_preview(self) -> None:
        if not self._rot_preview:
            self.view.set_preview_matrix(None)
            return
        m = self._rot_matrix(self.spn_rx.value(), self.spn_ry.value(), self.spn_rz.value())
        self.view.set_preview_matrix(m, self._rotation_center())

    def _commit_preview_rotation(self) -> None:
        """把预览旋转真正写进各扫描坐标；由 apply_rotation / apply_crop 调用。"""
        if not self._rot_preview or self.doc is None:
            return
        m = self._rot_matrix(self.spn_rx.value(), self.spn_ry.value(), self.spn_rz.value())
        c = self._rotation_center()
        for s in self.doc.scans:
            if not all(k in s.points for k in ("cartesianX", "cartesianY", "cartesianZ")):
                continue
            cols = [s.points[k] for k in ("cartesianX", "cartesianY", "cartesianZ")]
            dt = cols[0].dtype
            xyz = np.column_stack([np.asarray(k, np.float64) for k in cols])
            new = (xyz - c) @ m.T + c
            for k, j in zip(("cartesianX", "cartesianY", "cartesianZ"), range(3)):
                s.points[k] = new[:, j].astype(dt)
        self._rot_preview = False
        self.btn_rot_preview.setChecked(False)
        self.view.set_preview_matrix(None)
        self._dirty = True

    def apply_rotation(self) -> None:
        if self.doc is None:
            QMessageBox.information(self, "提示", "请先载入一个点云文件。")
            return
        if not any(abs(v) > 1e-9 for v in (self.spn_rx.value(), self.spn_ry.value(),
                                           self.spn_rz.value())):
            QMessageBox.information(self, "提示", "三个角度都是 0，没有可应用的旋转。")
            return
        self._snapshot()
        self._rot_preview = True
        try:
            self._commit_preview_rotation()
            self.view.set_document(self.doc, self.ann)
            self.view.set_view(self.view.view_name)
            self._refresh_info()
            self._update_actions()
            self.statusMessage.emit("旋转已烘焙进点坐标（可用 Ctrl+Z 撤销）")
        except Exception as e:  # noqa: BLE001
            self._undo.pop()
            QMessageBox.critical(self, "旋转失败", str(e))

    def reset_rotation(self) -> None:
        for sp in (self.spn_rx, self.spn_ry, self.spn_rz):
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
        self.btn_rot_preview.setChecked(False)
        self._rot_preview = False
        self.view.set_preview_matrix(None)

    # ------------------------------------------------------------ 保存 --

    def save_overwrite(self) -> None:
        if self.doc is None:
            return
        if not self.path:
            self.save_as()
            return
        r = QMessageBox.question(
            self, "确认覆盖保存",
            f"将按原格式直接写回：\n{self.path}\n\n"
            "磁盘上的原文件会被替换（不会保留副本），确定继续吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        self._write(self.path)

    def save_as(self) -> None:
        if self.doc is None:
            return
        ext = cloudio.ext_of(self.path or "") or ".e57"
        if ext not in cloudio.WRITE_EXTS:
            ext = ".e57"
        base = os.path.splitext(self.path or "")[0]
        d = (base + "_cropped" + ext) if base else ""
        path, chosen = QFileDialog.getSaveFileName(
            self, "另存为点云", d, cloudio.SAVE_FILTER)
        if not path:
            return
        if cloudio.ext_of(path) not in cloudio.WRITE_EXTS:
            guess = ""
            if "(*" in chosen:
                inner = chosen.split("(*", 1)[1].split(")", 1)[0]
                first = inner.split()[0] if inner.split() else ""
                guess = first[1:] if first.startswith("*") else first
            path += guess or ext
        self._write(path)
        self.path = path
        self.lbl_path.setText(f"当前文件：{os.path.basename(path)}")
        self.lbl_path.setToolTip(path)

    def _write(self, path: str) -> None:
        if self.doc is None:
            return
        if self._rot_preview:
            # 预览未应用时直接保存会把未烘焙的姿态丢掉：先把旋转写进坐标，
            # 并刷新视图，保证屏幕上看到的和写进文件的是同一份数据。
            self._commit_preview_rotation()
            self.view.set_document(self.doc, self.ann)
            self.view.set_view(self.view.view_name)
        self.ann = AnnotationSet.from_scans(self.doc.scans)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if cloudio.ext_of(path) == ".e57":
                self.doc.save(path)
            else:
                cloudio.save(path, self.doc, self.ann)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "保存失败", f"{path}\n\n{e}")
            return
        QApplication.restoreOverrideCursor()
        self._dirty = False
        self._refresh_info()
        self.statusMessage.emit(f"已保存 {path}  ·  {self.doc.total_points:,} 点")

    # ---------------------------------------------------------- 配准 --

    def _browse_into(self, cmb: QComboBox, flt: str) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "选择点云", self._folder or "", flt)
        if not p:
            return
        i = cmb.findData(p)
        if i < 0:
            cmb.addItem(os.path.basename(p), p)
            i = cmb.count() - 1
        cmb.setCurrentIndex(i)

    def _browse_output(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "输出 E57", self.edt_out.text(),
                                           "E57 点云 (*.e57)")
        if p:
            if cloudio.ext_of(p) != ".e57":
                p += ".e57"
            self.edt_out.setText(p)

    def _browse_python(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "选择 python.exe", "",
                                           "Python 解释器 (python.exe);;所有文件 (*)")
        if p:
            self.edt_python.setText(p)

    def _browse_script(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "选择配准脚本", "",
                                           "Python 脚本 (*.py);;所有文件 (*)")
        if p:
            self.edt_script.setText(p)

    def _probe_env(self) -> None:
        ok, missing = probe_python(self.edt_python.text())
        if ok:
            QMessageBox.information(self, "环境检查", "依赖齐全，可以运行配准脚本。")
        else:
            QMessageBox.warning(self, "环境检查",
                                "所选 Python 缺少依赖：\n" + "\n".join(missing) +
                                "\n\n可点「浏览」换一个已装齐依赖的解释器。")

    def _append_log(self, text: str) -> None:
        self.txt_log.moveCursor(QTextCursor.End)
        self.txt_log.insertPlainText(text)
        self.txt_log.ensureCursorVisible()

    def run_align(self) -> None:
        ref = self.cmb_ref.currentData() or ""
        src = self.cmb_src.currentData() or ""
        out = self.edt_out.text().strip()
        if not ref or not src:
            QMessageBox.warning(self, "参数不全", "请先选择点云1（PLY）与点云2（E57）。")
            return
        if not out:
            out = unique_output(default_output(src))
            self.edt_out.setText(out)
        err = validate_inputs(ref, src, out, self.edt_script.text())
        if err:
            QMessageBox.warning(self, "参数有误", err)
            return
        self.txt_log.clear()
        self.runner.run(
            self.edt_python.text(), self.edt_script.text(), ref, src, out,
            crop_distance=self.spn_crop_d.value() or None,
            voxel_size=self.spn_voxel.value() or None,
            registration=self.cmb_reg.currentText(),
            min_overlap=self.spn_overlap.value(),
            workers=self.spn_workers.value(),
            refine_color_pose=self.chk_refine.isChecked())

    def stop_align(self) -> None:
        self.runner.stop()

    def _on_align_started(self) -> None:
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusMessage.emit("配准裁剪脚本运行中…")

    def _on_align_finished(self, ok: bool, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log("\n" + msg + "\n")
        self.statusMessage.emit(msg)
        if ok and os.path.isfile(self.runner.output):
            r = QMessageBox.question(
                self, "配准裁剪完成",
                f"{msg}\n\n是否立即载入结果查看？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if r == QMessageBox.Yes:
                self._dirty = False
                self.load_file(self.runner.output)
        else:
            QMessageBox.warning(self, "配准裁剪未完成", msg)

    # ------------------------------------------------------------ 状态 --

    def _refresh_info(self) -> None:
        if self.doc is None:
            self.lbl_info.setText("尚未载入点云")
            self.lbl_path.setText("当前文件：—")
            return
        lo = np.array([np.inf] * 3)
        hi = np.array([-np.inf] * 3)
        for s in self.doc.scans:
            if s.count and "cartesianX" in s.points:
                for j, k in enumerate(("cartesianX", "cartesianY", "cartesianZ")):
                    a = np.asarray(s.points[k], np.float64)
                    lo[j] = min(lo[j], float(a.min()))
                    hi[j] = max(hi[j], float(a.max()))
        star = " *" if self._dirty else ""
        self.lbl_info.setText(
            f"{self.doc.total_points:,} 点  ·  {len(self.doc.scans)} 个扫描  ·  "
            f"范围 X[{lo[0]:.1f}, {hi[0]:.1f}]  Y[{lo[1]:.1f}, {hi[1]:.1f}]  "
            f"Z[{lo[2]:.1f}, {hi[2]:.1f}]{star}")
        self.lbl_path.setText(f"当前文件：{os.path.basename(self.path) if self.path else '—'}")
        self.lbl_path.setToolTip(self.path or "")
        if self.view.region() is None:
            self.lbl_region.setText("区域：未绘制")

    def _update_actions(self) -> None:
        has = self.doc is not None
        for w in (self.btn_apply, self.btn_clear_region, self.btn_undo,
                  self.btn_reload, self.btn_save, self.btn_save_as,
                  self.btn_rot_apply, self.btn_rot_preview, self.btn_rot_reset):
            w.setEnabled(has)
        self.btn_undo.setEnabled(has and bool(self._undo))
        self.btn_stop.setEnabled(self.runner.is_running())
        self.btn_run.setEnabled(not self.runner.is_running())
