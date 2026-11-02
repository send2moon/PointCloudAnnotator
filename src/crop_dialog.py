"""裁剪交互对话框：世界坐标包围盒微调 + 保留/删除语义选择。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QPushButton,
                               QRadioButton, QVBoxLayout, QWidget)

from cropping import CropBox


class _Spin(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(4)
        self.setRange(-1e6, 1e6)
        self.setSingleStep(0.01)
        self.setKeyboardTracking(False)
        self.setMinimumWidth(110)


class CropDialog(QDialog):
    """裁剪参数对话框。

    ``box`` 为初始包围盒（通常来自屏幕框选投影出的范围或整个点云包围盒）。
    返回时 ``result_box()`` 给出最终范围，``invert()`` 给出语义。
    """

    def __init__(self, box: CropBox, total_points: int, selected_points: int,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("点云裁剪")
        self.setMinimumWidth(420)
        self._init_box = box
        self._full = box

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        info = QLabel(f"当前点云 {total_points:,} 点　·　框内 {selected_points:,} 点")
        lay.addWidget(info)

        form = QFormLayout()
        form.setSpacing(8)
        self.spins = {}
        for axis, idx in (("X", 0), ("Y", 1), ("Z", 2)):
            row = QHBoxLayout()
            lo, hi = _Spin(), _Spin()
            lo.setValue(float(box.lo[idx]))
            hi.setValue(float(box.hi[idx]))
            row.addWidget(QLabel("最小"))
            row.addWidget(lo, 1)
            row.addWidget(QLabel("最大"))
            row.addWidget(hi, 1)
            w = QWidget()
            w.setLayout(row)
            form.addRow(axis, w)
            self.spins[axis] = (lo, hi)
        lay.addLayout(form)

        # 快捷操作
        quick = QHBoxLayout()
        b_all = QPushButton("重置为整个点云")
        b_all.clicked.connect(self._reset_all)
        quick.addWidget(b_all)
        quick.addStretch(1)
        lay.addLayout(quick)

        # 语义
        self.rb_keep = QRadioButton("保留框内的点（裁掉框外）")
        self.rb_remove = QRadioButton("删除框内的点（保留框外）")
        self.rb_keep.setChecked(True)
        # 对话框是独立窗口，这里再单独给单选框上一次样式，
        # 保证白色圆点/白描边一定和深色底分开（不依赖主窗口 QSS 的级联）。
        radio_qss = (
            "QRadioButton { color: #eef1f5; font-size: 13px; spacing: 9px;"
            " padding: 3px 2px; background: transparent; }"
            "QRadioButton::indicator { width: 16px; height: 16px; border-radius: 8px;"
            " border: 2px solid #ffffff; background: #23272d; }"
            "QRadioButton::indicator:hover { background: #3a4048; }"
            "QRadioButton::indicator:checked { background: #ffffff;"
            " border: 2px solid #ffffff; }"
            "QRadioButton:checked { color: #ffffff; font-weight: 600; }"
        )
        self.rb_keep.setStyleSheet(radio_qss)
        self.rb_remove.setStyleSheet(radio_qss)
        lay.addWidget(QLabel("裁剪方式"))
        lay.addWidget(self.rb_keep)
        lay.addWidget(self.rb_remove)

        tip = QLabel("提示：松开框选后可在上方微调数值；裁剪会改变点数，可用 Ctrl+Z 撤销。")
        tip.setStyleSheet("color:#8b939d; font-size:12px;")
        tip.setWordWrap(True)
        lay.addWidget(tip)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("执行裁剪")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _reset_all(self) -> None:
        b = self._full
        for axis, idx in (("X", 0), ("Y", 1), ("Z", 2)):
            lo, hi = self.spins[axis]
            lo.blockSignals(True); hi.blockSignals(True)
            lo.setValue(float(b.lo[idx])); hi.setValue(float(b.hi[idx]))
            lo.blockSignals(False); hi.blockSignals(False)

    def result_box(self) -> CropBox:
        vals = {}
        for axis, idx in (("X", 0), ("Y", 1), ("Z", 2)):
            lo, hi = self.spins[axis]
            vals[idx] = (lo.value(), hi.value())
        return CropBox(xmin=vals[0][0], xmax=vals[0][1],
                       ymin=vals[1][0], ymax=vals[1][1],
                       zmin=vals[2][0], zmax=vals[2][1])

    def invert(self) -> bool:
        return self.rb_remove.isChecked()
