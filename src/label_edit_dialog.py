"""标注管理对话框：查看并修改导入文件里已有的标注。

列表按扫描组织：每个实例一行（``类别 #编号 (点数)``），语义标注的点
（有类别、没有实例编号）按类别汇总成一行「未分实例」。

这里的操作都通过 ``MainWindow.edit_labels()`` 执行，因此：
* 一次操作 = 一个撤销步，``Ctrl+Z`` 可整体退回；
* 标注列表、实例框、隐藏状态、统计会一起刷新；
* 对话框本身只负责收集选择与确认，不直接改 ``AnnotationSet``。
"""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout)

from annotation import LABEL_COLORS, LABEL_NAMES


def _swatch(color) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(QColor(*color))
    return QIcon(pm)


class LabelEditDialog(QDialog):
    """按扫描列出实例与语义组，批量修改已有标注。"""

    def __init__(self, win, parent=None):
        super().__init__(parent if parent is not None else win)
        self.win = win
        self.setWindowTitle("修改已有标注")
        self.resize(600, 680)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        tip = QLabel("选中一行或多行（Ctrl / Shift 多选）后使用下面的操作；"
                     "所有修改都能用 Ctrl+Z 撤销。")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        v.addWidget(tip)

        self.lst = QListWidget()
        self.lst.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst.setToolTip(
            "文件里现有的标注，按扫描分组：\n"
            "  穗 #45 (5 点) —— 一个实例（有类别 + 实例编号）\n"
            "  穗（未分实例）(1,234 点) —— 只有类别、没有编号的点\n"
            "Ctrl 点选 / Shift 连续选，可跨扫描多选后用下面的按钮批量处理。")
        self.lst.itemSelectionChanged.connect(self._update_btns)
        v.addWidget(self.lst, 1)

        self.lbl_sel = QLabel("未选中")
        self.lbl_sel.setObjectName("hint")
        v.addWidget(self.lbl_sel)

        # --- 改类别 ---
        row = QHBoxLayout()
        row.addWidget(QLabel("改类别为"))
        self.cmb_label = QComboBox()
        for lid in (1, 2, 3, 4):
            self.cmb_label.addItem(_swatch(LABEL_COLORS[lid]), LABEL_NAMES[lid], lid)
        self.cmb_label.setToolTip("要把选中行改成哪个器官类别（只改类别，实例编号保留）")
        row.addWidget(self.cmb_label, 1)
        self.btn_set_label = QPushButton("应用")
        self.btn_set_label.setToolTip("把选中实例/语义组的点改成所选器官类别（实例编号保留）")
        self.btn_set_label.clicked.connect(self._on_set_label)
        row.addWidget(self.btn_set_label)
        v.addLayout(row)

        row2 = QHBoxLayout()
        self.btn_semantic = QPushButton("转为语义（去实例号）")
        self.btn_semantic.setToolTip(
            "保留类别，清掉实例编号。改完这些点就归入列表里的「未分实例」一行，\n"
            "适合导入文件里编号混乱、只想保留类别的情况。")
        self.btn_semantic.clicked.connect(self._on_semantic)
        self.btn_merge = QPushButton("合并为一个实例")
        self.btn_merge.setToolTip(
            "把选中的多行（可跨类别语义组）合并成一个新实例，类别取多数。\n"
            "用于把自动分割切碎的一根穗重新合成一个。")
        self.btn_merge.clicked.connect(self._on_merge)
        row2.addWidget(self.btn_semantic)
        row2.addWidget(self.btn_merge)
        v.addLayout(row2)

        self.btn_delete = QPushButton("删除选中点的标注")
        self.btn_delete.setToolTip("把选中行的点恢复为「未标注」（类别与实例编号都清空）")
        self.btn_delete.clicked.connect(self._on_delete)
        v.addWidget(self.btn_delete)

        # --- 按类别清除 ---
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("按类别清除"))
        self.cmb_clear = QComboBox()
        for lid in (1, 2, 3, 4):
            self.cmb_clear.addItem(_swatch(LABEL_COLORS[lid]), LABEL_NAMES[lid], lid)
        self.cmb_clear.setToolTip("要清除哪个类别的全部标注（跨所有扫描，含其下所有实例）")
        row3.addWidget(self.cmb_clear, 1)
        self.btn_clear_cat = QPushButton("清除全部扫描")
        self.btn_clear_cat.setToolTip("把这个类别的所有点（含各实例）清成未标注，跨全部扫描")
        self.btn_clear_cat.clicked.connect(self._on_clear_category)
        row3.addWidget(self.btn_clear_cat)
        v.addLayout(row3)

        # --- 整段清空 ---
        row4 = QHBoxLayout()
        self.btn_clear_scan = QPushButton("清空当前扫描")
        self.btn_clear_scan.setToolTip(
            "把列表里聚焦的那一站（或选中行所在的站）的标注全部清成未标注，\n"
            "其他站不动；会先弹窗确认，可撤销。")
        self.btn_clear_scan.clicked.connect(self._on_clear_scan)
        self.btn_clear_all = QPushButton("清空全部标注")
        self.btn_clear_all.setToolTip(
            "把整个文件里所有扫描的标注全部清空（含所有实例编号），\n"
            "相当于从未标注过；会先弹窗确认，可撤销。")
        self.btn_clear_all.clicked.connect(self._on_clear_all)
        row4.addWidget(self.btn_clear_scan)
        row4.addWidget(self.btn_clear_all)
        v.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.setToolTip("关闭对话框；已经应用的修改仍然保留（可回主窗口 Ctrl+Z 撤销）")
        btn_close.clicked.connect(self.accept)
        row5.addWidget(btn_close)
        v.addLayout(row5)

        self.populate()

    # ------------------------------------------------------------ 数据 --

    def populate(self) -> None:
        """按当前标注重建列表（操作后调用）。"""
        self.lst.clear()
        win = self.win
        if win.doc is None or win.ann is None:
            return
        for sid, scan in enumerate(win.doc.scans):
            lab = win.ann.labels[sid]
            marked = int((lab > 0).sum())
            head = QListWidgetItem(
                f"扫描 {sid + 1}/{len(win.doc.scans)}    {scan.count:,} 点"
                f"    已标注 {marked:,} 点")
            f = head.font(); f.setBold(True); head.setFont(f)
            head.setFlags(head.flags() & ~Qt.ItemIsSelectable)
            self.lst.addItem(head)

            ids = win.ann.instance_ids(sid)
            ids.sort(key=lambda u: -win.ann.instance_size(sid, u))
            for uid in ids:
                lb = win.ann.instance_label(sid, uid)
                size = win.ann.instance_size(sid, uid)
                it = QListWidgetItem(
                    _swatch(LABEL_COLORS.get(lb, (150, 150, 150))),
                    f"     {LABEL_NAMES.get(lb, '?')} #{uid}   ({size:,} 点)")
                it.setData(Qt.UserRole, ("inst", sid, int(uid)))
                self.lst.addItem(it)

            for lb, cnt in win.ann.label_groups(sid):
                it = QListWidgetItem(
                    _swatch(LABEL_COLORS.get(lb, (150, 150, 150))),
                    f"     {LABEL_NAMES.get(lb, '?')}（未分实例）   ({cnt:,} 点)")
                it.setData(Qt.UserRole, ("sem", sid, int(lb)))
                self.lst.addItem(it)

            if marked == 0:
                it = QListWidgetItem("     未标注")
                it.setForeground(QColor(120, 126, 134))
                it.setFlags(it.flags() & ~Qt.ItemIsSelectable)
                self.lst.addItem(it)
        self._update_btns()

    def keys(self) -> List[Tuple[str, int, int]]:
        out: List[Tuple[str, int, int]] = []
        for it in self.lst.selectedItems():
            v = it.data(Qt.UserRole)
            if isinstance(v, tuple) and len(v) == 3:
                out.append((str(v[0]), int(v[1]), int(v[2])))
        return out

    def _update_btns(self) -> None:
        n = len(self.keys())
        for b in (self.btn_set_label, self.btn_semantic, self.btn_merge, self.btn_delete):
            b.setEnabled(n > 0)
        self.lbl_sel.setText(f"已选中 {n} 项" if n else "未选中（可 Ctrl / Shift 多选）")

    # ------------------------------------------------------------ 操作 --

    def _after(self, msg: str = "") -> None:
        self.populate()
        if msg:
            self.win.status.showMessage(msg, 3000)

    def _on_set_label(self) -> None:
        keys = self.keys()
        if not keys:
            return
        lid = int(self.cmb_label.currentData())
        n = self.win.edit_labels("set_label", keys, lid)
        self._after(f"已把 {len(keys)} 项共 {n:,} 个点改为「{LABEL_NAMES[lid]}」")

    def _on_semantic(self) -> None:
        keys = self.keys()
        if not keys:
            return
        n = self.win.edit_labels("to_semantic", keys)
        self._after(f"已清除 {len(keys)} 项的实例编号（保留类别，共 {n:,} 点）")

    def _on_merge(self) -> None:
        keys = self.keys()
        if len(keys) < 2:
            QMessageBox.information(self, "修改已有标注", "请至少选中两行再合并。")
            return
        n = self.win.edit_labels("merge", keys)
        self._after(f"已把 {len(keys)} 项共 {n:,} 个点合并为一个新实例")

    def _on_delete(self) -> None:
        keys = self.keys()
        if not keys:
            return
        if QMessageBox.question(
                self, "删除标注",
                f"把选中的 {len(keys)} 项的点恢复为「未标注」？") != QMessageBox.Yes:
            return
        n = self.win.edit_labels("delete", keys)
        self._after(f"已删除 {n:,} 个点的标注")

    def _on_clear_category(self) -> None:
        lid = int(self.cmb_clear.currentData())
        if QMessageBox.question(
                self, "按类别清除",
                f"清除全部扫描里所有「{LABEL_NAMES[lid]}」标注？\n"
                f"（包括该类别下的全部实例编号）") != QMessageBox.Yes:
            return
        n = self.win.edit_labels("clear_category", [], lid)
        self._after(f"已清除 {n:,} 个「{LABEL_NAMES[lid]}」点")

    def _on_clear_scan(self) -> None:
        keys = self.keys()
        sid = keys[0][1] if keys else self.win._focused_scan()
        if QMessageBox.question(
                self, "清空扫描",
                f"清空扫描 #{sid + 1} 的全部标注？") != QMessageBox.Yes:
            return
        self.win.edit_labels("clear_scan", [("scan", sid, 0)])
        self._after(f"已清空扫描 #{sid + 1} 的标注")

    def _on_clear_all(self) -> None:
        if QMessageBox.question(
                self, "清空全部标注",
                "清空这个文件里**所有扫描**的标注？\n此操作可用 Ctrl+Z 撤销。") != QMessageBox.Yes:
            return
        n = self.win.edit_labels("clear_all", [])
        self._after(f"已清空全部标注（{n:,} 个点）")
