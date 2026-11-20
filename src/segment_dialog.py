"""穗实例分割对话框：参数设置 + 一键试算（不落盘、可反复调参）。"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout)

from instance_seg import SegmentParams


class SegmentDialog(QDialog):
    """分割参数对话框。

    ``probe(scope, params, overwrite) -> str`` 由主窗口提供，用于「试算」时
    在对话框里直接给出簇数/点数统计，不必真的改标注。
    """

    def __init__(self, n_scans: int, current_sid: int, params: SegmentParams,
                 spike_by_scan: Optional[list] = None,
                 probe: Optional[Callable] = None,
                 default_overwrite: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("穗实例分割")
        self.setMinimumWidth(460)
        self._probe = probe

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        head = QLabel("把已标注为「穗」的点，按空间连通性自动拆成一个个独立麦穗，\n"
                      "并给每一穗分配实例编号（写入 annotationInstance）。")
        head.setWordWrap(True)
        lay.addWidget(head)

        form = QFormLayout()
        form.setSpacing(8)

        self.cmb_scope = QComboBox()
        self.cmb_scope.addItem(f"当前扫描（第 {current_sid + 1} 站）", "current")
        self.cmb_scope.addItem("全部扫描（逐站分别分割）", "all")
        if n_scans <= 1:
            self.cmb_scope.setEnabled(False)
        self.cmb_scope.setToolTip(
            "每个扫描是同一批植株在不同站位的一次采集，实例编号按站独立编号，\n"
            "因此跨站合成一片点再分割没有意义——一律逐站分别分割。")
        form.addRow("作用范围", self.cmb_scope)

        self.spin_voxel = QDoubleSpinBox()
        self.spin_voxel.setRange(0.05, 5.0)
        self.spin_voxel.setSingleStep(0.05)
        self.spin_voxel.setDecimals(2)
        self.spin_voxel.setSuffix(" cm")
        self.spin_voxel.setValue(float(params.voxel_cm))
        self.spin_voxel.setToolTip(
            "体素边长：越小越容易把贴在一起的相邻穗分开，也越容易把一根穗打碎。\n"
            "经验值 0.25–0.40 cm（穗径约 1 cm、穗间净距约 1–2 cm 时）。")
        form.addRow("体素边长", self.spin_voxel)

        self.cmb_link = QComboBox()
        self.cmb_link.addItem("紧邻（26 邻域，推荐）", 1)
        self.cmb_link.addItem("放宽 2 格（补碎片）", 2)
        self.cmb_link.addItem("很宽 3 格（易粘连）", 3)
        i = self.cmb_link.findData(int(params.link))
        self.cmb_link.setCurrentIndex(max(i, 0))
        self.cmb_link.setToolTip(
            "相邻体素算不算连在一起：\n"
            "  26 邻域 —— 体素间距 1 格就算连通，最贴合真实表面\n"
            "  放宽 2 格 —— 点云稀疏、穗被切碎时用，代价是相邻穗可能粘连")
        form.addRow("连通方式", self.cmb_link)

        self.spin_min = QSpinBox()
        self.spin_min.setRange(1, 100000)
        self.spin_min.setSingleStep(5)
        self.spin_min.setValue(int(params.min_points))
        self.spin_min.setSuffix(" 点")
        self.spin_min.setToolTip(
            "少于这么多点的连通块判为碎片（噪点、断枝），不分配实例。\n"
            "穗点密度约 100–500 点/穗时可设 10–30。")
        form.addRow("最小穗点数", self.spin_min)
        lay.addLayout(form)

        self.chk_overwrite = QCheckBox("覆盖已有实例编号（重新编号）")
        self.chk_overwrite.setToolTip(
            "不勾选：只给「还没有实例编号」的穗点补编号，已有实例原样保留（增量标注用）。\n"
            "勾选：范围内的穗点全部重新分割并重新编号，原来的实例编号失效。")
        # 穗点都已经有编号时，默认勾上，否则一进来"没东西可分割"
        self.chk_overwrite.setChecked(bool(default_overwrite))
        lay.addWidget(self.chk_overwrite)

        row = QHBoxLayout()
        self.btn_probe = QPushButton("试算")
        self.btn_probe.setToolTip(
            "按当前参数先跑一遍：三维视图里用虚线立方体框预览将生成的穗\n"
            "（框顶编号 = 执行后的实例号；只画列表聚焦的那一站），\n"
            "这里给出簇数与点数分布。不改动标注。")
        self.btn_probe.clicked.connect(self._do_probe)
        if self._probe is None:
            self.btn_probe.setEnabled(False)
        row.addWidget(self.btn_probe)
        row.addStretch(1)
        lay.addLayout(row)

        self.lbl_probe = QLabel("点「试算」先看看效果，满意再执行。")
        self.lbl_probe.setWordWrap(True)
        self.lbl_probe.setStyleSheet(
            "color:#24292f; background:#f4f8fd; border:1px solid #dfe3e8;"
            " border-radius:4px; padding:8px; font-size:12px;")
        lay.addWidget(self.lbl_probe)

        tip = QLabel("提示：分割结果可用 Ctrl+Z 整体撤销；\n"
                     "相邻穗紧贴时自动分割难免有粘连/切碎，可用笔刷+框选再手工修正。")
        tip.setStyleSheet("color:#5a6472; font-size:12px;")
        tip.setWordWrap(True)
        lay.addWidget(tip)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("执行分割")
        bb.button(QDialogButtonBox.Ok).setToolTip(
            "按当前参数真正写入实例编号（整次分割 = 一个撤销步，Ctrl+Z 可整体退回）")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.button(QDialogButtonBox.Cancel).setToolTip("不分割，直接关闭对话框（标注不变）")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        for w in (self.spin_voxel, self.spin_min):
            w.valueChanged.connect(self._mark_stale)
        self.cmb_link.currentIndexChanged.connect(self._mark_stale)
        self.cmb_scope.currentIndexChanged.connect(self._mark_stale)
        self.chk_overwrite.stateChanged.connect(self._mark_stale)
        # 打开时先算一遍，用户一进来就有数
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._do_probe)

    # ------------------------------------------------------------------ --

    def _mark_stale(self, *_a) -> None:
        self.lbl_probe.setText("参数已改，点「试算」刷新统计。")

    def _do_probe(self) -> None:
        if self._probe is None:
            return
        self.lbl_probe.setText("正在试算…")
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            txt = self._probe(self.scope(), self.params(), self.overwrite())
        except Exception as e:                        # noqa: BLE001
            txt = f"试算失败：{e}"
        finally:
            QApplication.restoreOverrideCursor()
        self.lbl_probe.setText(txt)

    def params(self) -> SegmentParams:
        return SegmentParams(voxel_cm=float(self.spin_voxel.value()),
                             link=int(self.cmb_link.currentData()),
                             min_points=int(self.spin_min.value()))

    def scope(self) -> str:
        return str(self.cmb_scope.currentData() or "current")

    def overwrite(self) -> bool:
        return bool(self.chk_overwrite.isChecked())
