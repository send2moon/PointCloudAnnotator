"""Independent review page: Point-SAM prompts, preview, correction and saving."""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
import numpy as np
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget)
from annotation import LABEL_NAMES
import cloudio
from crop_view import MODE_NAV, MODE_RECT
from office_theme import CANVAS_COLORS
from point_sam_runner import DEFAULT_REPO, DEFAULT_PYTHON, DEFAULT_CHECKPOINT, PointSAMRunner
from review_session import ReviewSession
from review_view import ReviewView


class ReviewPage(QWidget):
    statusMessage = Signal(str)
    stateChanged = Signal()
    importRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        self.preview = None
        self.candidates = None
        self.result_indices = None
        self.prompts = []
        self.work = tempfile.TemporaryDirectory(prefix='point_sam_review_')
        self.request_kind = ''
        self.runner = PointSAMRunner(self)
        self.settings = QSettings('PointCloudAnnotator', 'Review')
        self._build_ui()
        self.runner.busyChanged.connect(self._busy_changed)
        self.runner.finished.connect(self._finished)
        self.runner.logLine.connect(self._log)
        self._shortcuts = []
        for key, fn in [('Ctrl+O', self.open_file), ('Ctrl+S', self.save_as),
                        ('Ctrl+Shift+S', self.save_overwrite), ('Ctrl+Z', self.undo),
                        ('Ctrl+W', self.close_file)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(fn)
            self._shortcuts.append(shortcut)
        self.set_shortcuts_enabled(False)
        self._update_actions()

    @property
    def path(self):
        return self.session.path if self.session else ''

    @property
    def dirty(self):
        return self.session is not None and self.session.dirty

    def _button(self, text, callback, tip):
        button = QPushButton(text)
        button.setToolTip(tip)
        button.clicked.connect(callback)
        return button

    def _build_ui(self):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        tabs = QTabWidget()
        tabs.setMinimumWidth(240)
        tabs.setMaximumWidth(400)
        listing = QWidget()
        left = QVBoxLayout(listing)
        self.btn_open = self._button('打开待核查点云…', self.open_file,
            '打开已有标注或未标注点云（Ctrl+O），保留原始点序、属性和分站结构。')
        self.btn_import = self._button('载入器官标注页当前结果', self.importRequested.emit,
            '复制器官标注页当前内存中的点云和标注，包含尚未保存的修改；两页之后独立编辑。')
        left.addWidget(self.btn_open)
        left.addWidget(self.btn_import)
        self.lbl_stats = QLabel('尚未载入点云')
        self.lbl_stats.setWordWrap(True)
        left.addWidget(self.lbl_stats)
        self.lst = QListWidget()
        self.lst.setToolTip('按扫描列出已有器官实例；选择实例后以青色显示，可作为修正目标。')
        self.lst.currentItemChanged.connect(self._select_item)
        left.addWidget(self.lst, 1)
        tabs.addTab(listing, '核查列表')
        environment = QWidget()
        env = QVBoxLayout(environment)
        form = QFormLayout()
        self.txt_repo = QLineEdit(self.settings.value('repo', DEFAULT_REPO))
        self.txt_python = QLineEdit(self.settings.value('python', DEFAULT_PYTHON))
        self.txt_checkpoint = QLineEdit(self.settings.value('checkpoint', DEFAULT_CHECKPOINT))
        for label, field, folder, tip in (
            ('源码目录', self.txt_repo, True, 'Point-SAM-main 目录，需包含 pc_sam 和 configs。'),
            ('Python', self.txt_python, False, '推理解释器；默认使用已指定的 conda b2l 环境。'),
            ('权重文件', self.txt_checkpoint, False, '官方 Point-SAM ViT-L 的 model.safetensors 预训练权重。')):
            field.setToolTip(tip)
            row = QHBoxLayout()
            row.addWidget(field, 1)
            row.addWidget(self._button('…', lambda _=False, f=field, d=folder: self._browse(f, d), tip))
            form.addRow(label, row)
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(['auto', 'cuda', 'cpu'])
        self.cmb_device.setToolTip('auto 优先使用 GPU；CPU 可运行但耗时较长。')
        form.addRow('推理设备', self.cmb_device)
        self.spn_samples = QSpinBox()
        self.spn_samples.setRange(256, 50000)
        self.spn_samples.setSingleStep(1000)
        self.spn_samples.setValue(10000)
        self.spn_samples.setToolTip('模型采样点数上限；保留全部提示点，结果再映射回区域内原始点。显存不足时调低。')
        form.addRow('采样上限', self.spn_samples)
        self.spn_distance = QDoubleSpinBox()
        self.spn_distance.setRange(0, 1000)
        self.spn_distance.setSuffix(' mm')
        self.spn_distance.setToolTip('采样结果向原始点传播的最大距离；0 不限制。局部点云稀疏时可限制远处扩散。')
        form.addRow('传播距离', self.spn_distance)
        env.addLayout(form)
        self.btn_probe = self._button('检测 Point-SAM 环境', self.probe,
            '检查解释器、CUDA、模型代码、依赖及权重文件；实际加载权重在首次运行分割时进行。')
        env.addWidget(self.btn_probe)
        self.lbl_env = QLabel('首次使用可先检测环境；模型仅在运行时加载。')
        self.lbl_env.setWordWrap(True)
        env.addWidget(self.lbl_env)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setToolTip('Point-SAM 独立进程日志；失败时显示具体原因。')
        env.addWidget(self.log, 1)
        tabs.addTab(environment, 'Point-SAM 设置')
        split.addWidget(tabs)

        center = QWidget()
        middle = QVBoxLayout(center)
        middle.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = []
        for text, mode, tip in (
            ('浏览', MODE_NAV, '左键旋转、右键平移、滚轮缩放；F 适合窗口。'),
            ('限定区域', MODE_RECT, '在视图中框出本次核查范围；这是推理范围，不裁掉原始点。'),
            ('正点 +', 'positive', '点击目标器官上的点，告诉 Point-SAM 要保留哪一部分。'),
            ('负点 −', 'negative', '点击误选的背景或相邻器官，帮助 Point-SAM 排除这些点。')):
            button = self._button(text, lambda _=False, m=mode: self.view.set_mode(m), tip)
            button.setCheckable(True)
            self.mode_group.addButton(button)
            self.mode_buttons.append(button)
            tools.addWidget(button)
        self.mode_buttons[0].setChecked(True)
        tools.addWidget(self._button('适合窗口', lambda: self.view.fit_view(), '将可见点云缩放到适合窗口。'))
        middle.addLayout(tools)
        self.view = ReviewView()
        self.view.regionDefined.connect(self._region_changed)
        self.view.promptPicked.connect(self.add_prompt)
        self.view.notice.connect(self.statusMessage.emit)
        middle.addWidget(self.view, 1)
        self.lbl_preview = QLabel('金色为待应用选区；青色为已有目标。预览不会修改标注。')
        self.lbl_preview.setWordWrap(True)
        middle.addWidget(self.lbl_preview)
        split.addWidget(center)

        right = QWidget()
        right.setMinimumWidth(285)
        right.setMaximumWidth(390)
        side = QVBoxLayout(right)
        side.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        controls = QVBoxLayout(body)
        scope = QGroupBox('范围与交互提示')
        f = QVBoxLayout(scope)
        self.cmb_scan = QComboBox()
        self.cmb_scan.addItem('全部扫描', -1)
        self.cmb_scan.setToolTip('选择本次核查的扫描；加入已有实例时须选择该实例所属扫描。')
        self.cmb_scan.currentIndexChanged.connect(self._scope_changed)
        f.addWidget(self.cmb_scan)
        self.lbl_scope = QLabel('范围：全部可用点')
        self.lbl_scope.setWordWrap(True)
        f.addWidget(self.lbl_scope)
        f.addWidget(self._button('清除限定区域', self.clear_region, '恢复当前扫描的全部点，清除旧提示和旧预览。'))
        self.lbl_prompts = QLabel('正点 0 · 负点 0')
        f.addWidget(self.lbl_prompts)
        self.btn_pop = self._button('撤回最后一个提示点', self.pop_prompt, '撤回最后一个正点或负点，再次运行分割可更新结果。')
        f.addWidget(self.btn_pop)
        self.btn_clear = self._button('清空提示和预览', self.clear_prompts, '清空交互提示与预测预览，保留已经应用的标注。')
        f.addWidget(self.btn_clear)
        self.chk_prior = QCheckBox('使用已有目标作初始掩膜')
        self.chk_prior.setToolTip('先在列表选择实例或语义类别，将其已有标注作为 Point-SAM 修正的起点。')
        self.chk_prior.toggled.connect(self._invalidate_result)
        f.addWidget(self.chk_prior)
        self.btn_run = self._button('运行 Point-SAM', self.run_model, '根据当前范围和正负点分割；输出先显示为金色预览，不会自动写入标注。')
        self.btn_run.setObjectName('primary')
        f.addWidget(self.btn_run)
        self.btn_cancel = self._button('取消推理', lambda: self.runner.cancel(), '终止独立模型进程，已经应用的标注不受影响。')
        f.addWidget(self.btn_cancel)
        self.btn_manual = self._button('将框选区域设为手动选区', self.use_manual_region, '直接用框选范围补标或修正，不运行模型；该选区同样先预览、后应用。')
        f.addWidget(self.btn_manual)
        self.cmb_candidate = QComboBox()
        self.cmb_candidate.setToolTip('选择 Point-SAM 候选掩膜；分数是模型预测值，不代表实际标注准确率。')
        self.cmb_candidate.currentIndexChanged.connect(self._candidate_changed)
        f.addWidget(self.cmb_candidate)
        controls.addWidget(scope)

        edit = QGroupBox('应用核查结果')
        form = QFormLayout(edit)
        self.cmb_label = QComboBox()
        for lid in (1, 2, 3, 4):
            self.cmb_label.addItem(LABEL_NAMES[lid], lid)
        self.cmb_label.setToolTip('将金色选区标为该器官类别。加入已有实例时须与其类别相同。')
        form.addRow('器官类别', self.cmb_label)
        self.cmb_action = QComboBox()
        for text, action in [('补标未标注点', 'fill'), ('覆盖修正选区', 'overwrite'),
                             ('清除选区标注', 'erase'), ('替换当前实例边界', 'replace_instance')]:
            self.cmb_action.addItem(text, action)
        self.cmb_action.setToolTip('补标仅处理未标注点；覆盖修正会修改选区旧标注；替换实例还会清除其旧边界外的点。')
        form.addRow('修正方式', self.cmb_action)
        self.cmb_instance = QComboBox()
        for text, value in [('仅语义类别', 'semantic'), ('生成新实例', 'new'), ('加入已有实例', 'existing')]:
            self.cmb_instance.addItem(text, value)
        self.cmb_instance.setToolTip('选择写入语义类别、新建器官实例，或将选区加入核查列表中选中的已有实例。')
        form.addRow('实例方式', self.cmb_instance)
        self.lbl_target = QLabel('目标：未选择')
        self.lbl_target.setWordWrap(True)
        form.addRow(self.lbl_target)
        self.btn_apply = self._button('应用当前选区', self.apply_preview, '仅在点击后写入标注；一次应用可整步撤销，修改后需保存。')
        self.btn_apply.setObjectName('primary')
        form.addRow(self.btn_apply)
        self.btn_undo = self._button('撤销上一步 (Ctrl+Z)', self.undo, '撤销上一次补标或修正，恢复类别与实例编号；保存后撤销也会重新提示未保存。')
        form.addRow(self.btn_undo)
        controls.addWidget(edit)

        display = QGroupBox('显示与对照')
        form = QFormLayout(display)
        self.cmb_color = QComboBox()
        for text, mode in [('标注', 'label'), ('真彩色', 'rgb'), ('高度', 'height'), ('强度', 'intensity')]:
            self.cmb_color.addItem(text, mode)
        self.cmb_color.setToolTip('切换标注或原始颜色检查；金色预览叠加不改变原始颜色。')
        self.cmb_color.currentIndexChanged.connect(lambda: self.view.set_color_mode(self.cmb_color.currentData()))
        form.addRow('着色', self.cmb_color)
        self.chk_prediction = QCheckBox('显示待应用预览')
        self.chk_prediction.setChecked(True)
        self.chk_prediction.setToolTip('关闭金色覆盖以对照原标注，再勾选可恢复；不会清除预测结果。')
        self.chk_prediction.toggled.connect(self._show_masks)
        form.addRow(self.chk_prediction)
        self.chk_bbox = QCheckBox('显示包围盒')
        self.chk_bbox.setToolTip('显示当前可见点云的三维包围盒及尺寸。')
        self.chk_bbox.toggled.connect(self.view.set_show_bbox)
        form.addRow(self.chk_bbox)
        row = QHBoxLayout()
        self.sld_size = QSlider(Qt.Horizontal)
        self.sld_size.setRange(5, 40)
        self.sld_size.setValue(14)
        self.sld_size.setToolTip('调整点的显示大小；旁边数值与滑块同步。')
        self.lbl_point_size = QLabel('1.4')
        self.sld_size.valueChanged.connect(self._point_size)
        row.addWidget(self.sld_size, 1)
        row.addWidget(self.lbl_point_size)
        form.addRow('点大小', row)
        controls.addWidget(display)
        controls.addStretch(1)
        scroll.setWidget(body)
        side.addWidget(scroll, 1)
        save = QGroupBox('保存与关闭')
        f = QVBoxLayout(save)
        self.lbl_path = QLabel('当前文件：—')
        self.lbl_path.setWordWrap(True)
        f.addWidget(self.lbl_path)
        self.btn_save = self._button('保存核查结果… (Ctrl+S)', self.save_as, '另存带修正标注的点云，支持 E57/PLY/PCD/文本；成功后弹窗提示。')
        self.btn_overwrite = self._button('覆盖保存 (Ctrl+Shift+S)', self.save_overwrite, '确认后按原格式覆盖当前文件，未保存提醒在写入成功后解除。')
        self.btn_close = self._button('关闭当前文件 (Ctrl+W)', self.close_file, '关闭核查文件；存在未保存修正时询问保存、放弃或取消。')
        for button in (self.btn_save, self.btn_overwrite, self.btn_close):
            f.addWidget(button)
        side.addWidget(save)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 1000, 320])

    def set_shortcuts_enabled(self, enabled):
        for shortcut in self._shortcuts:
            shortcut.setEnabled(enabled)

    def _browse(self, field, folder):
        if folder:
            value = QFileDialog.getExistingDirectory(self, '选择 Point-SAM 源码目录', field.text())
        else:
            value, _ = QFileDialog.getOpenFileName(self, '选择文件', field.text())
        if value:
            field.setText(value)

    def _config(self):
        for key, field in [('repo', self.txt_repo), ('python', self.txt_python), ('checkpoint', self.txt_checkpoint)]:
            self.settings.setValue(key, field.text().strip())
        return {'repo': self.txt_repo.text().strip(), 'checkpoint': self.txt_checkpoint.text().strip(),
                'device': self.cmb_device.currentText(), 'max_points': self.spn_samples.value(),
                'max_distance': self.spn_distance.value() / 1000.0}

    def open_file(self):
        if self.runner.busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, '打开待核查点云', self.path, cloudio.OPEN_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path):
        if self.runner.busy or not self.confirm_discard():
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            session = ReviewSession(cloudio.load(path), path)
        except Exception as exc:
            QMessageBox.critical(self, '读取失败', str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.set_session(session)
        return True

    def set_session(self, session):
        self.session = session
        self.clear_prompts()
        self.view.set_document(session.doc, session.ann)
        self.view.set_color_mode(self.cmb_color.currentData())
        self.cmb_scan.blockSignals(True)
        self.cmb_scan.clear()
        self.cmb_scan.addItem('全部扫描', -1)
        for sid, scan in enumerate(session.doc.scans):
            self.cmb_scan.addItem(f'扫描 #{sid + 1} · {scan.count:,} 点', sid)
        self.cmb_scan.blockSignals(False)
        self._refresh_list()
        self._refresh()

    def _refresh_list(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        if self.session:
            ann = self.session.ann
            for sid, scan in enumerate(self.session.doc.scans):
                item = QListWidgetItem(f'扫描 #{sid + 1} · {scan.count:,} 点')
                item.setData(Qt.UserRole, ('scan', sid, 0))
                self.lst.addItem(item)
                for uid in ann.instance_ids(sid):
                    item = QListWidgetItem(f'    {LABEL_NAMES.get(ann.instance_label(sid, uid), "其他")} #{uid} · {ann.instance_size(sid, uid):,} 点')
                    item.setData(Qt.UserRole, ('instance', sid, uid))
                    self.lst.addItem(item)
                for lid, count in ann.label_groups(sid):
                    item = QListWidgetItem(f'    语义 {LABEL_NAMES.get(lid, lid)} · {count:,} 点')
                    item.setData(Qt.UserRole, ('semantic', sid, lid))
                    self.lst.addItem(item)
        self.lst.blockSignals(False)
        self.lbl_target.setText('目标：未选择')

    def _target(self):
        item = self.lst.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def _target_mask(self):
        if self.session is None:
            return None
        target = self._target()
        if not target or target[0] == 'scan':
            return None
        kind, sid, value = target
        mask = np.zeros(self.session.doc.total_points, bool)
        sub = (self.session.ann.instances[sid] == value if kind == 'instance'
               else (self.session.ann.labels[sid] == value) & (self.session.ann.instances[sid] == 0))
        mask[self.session.offsets[sid]:self.session.offsets[sid + 1]] = sub
        return mask

    def _select_item(self, item, previous=None):
        self.clear_prompts()
        target = self._target()
        if target and self.session:
            kind, sid, value = target
            self.cmb_scan.setCurrentIndex(self.cmb_scan.findData(sid))
            if kind in ('instance', 'semantic'):
                label = self.session.ann.instance_label(sid, value) if kind == 'instance' else value
                self.cmb_label.setCurrentIndex(self.cmb_label.findData(label))
            self.lbl_target.setText(f'目标：扫描 #{sid + 1}' + (f' · 实例 #{value}' if kind == 'instance' else ' · 语义类别' if kind == 'semantic' else ''))
        else:
            self.lbl_target.setText('目标：未选择')
        self._show_masks()

    def _scope_indices(self):
        if self.session is None:
            return np.zeros(0, np.int64)
        renderer = self.view.renderer
        mask = np.isfinite(renderer.xyz).all(axis=1)
        sid = self.cmb_scan.currentData()
        if sid is not None and sid >= 0:
            mask &= renderer.scan_ids == sid
        if self.view.region() is not None:
            mask &= self.view.region_mask(depth='all')
        return np.flatnonzero(mask)

    def _scope_changed(self, *_):
        if self.session is None:
            return
        self.view.clear_region()
        self.clear_prompts()
        sid = self.cmb_scan.currentData()
        hidden = self.view.renderer.scan_ids != sid if sid is not None and sid >= 0 else None
        self.view.renderer.set_hidden(hidden)
        self.view.fit_view()
        self._update_scope_label()

    def _region_changed(self, region):
        self.clear_prompts()
        self._update_scope_label()

    def _update_scope_label(self):
        self.lbl_scope.setText(f'范围：{len(self._scope_indices()):,} 点' + ('（已锁定框选视角）' if self.view.region() else ''))

    def clear_region(self):
        self.view.clear_region()
        self.clear_prompts()
        self._update_scope_label()

    def add_prompt(self, index, label):
        if self.runner.busy or self.session is None:
            return
        scope = self._scope_indices()
        if index not in scope:
            self.statusMessage.emit('提示点在当前核查范围之外，请重新选择。')
            return
        self.prompts = [(i, value) for i, value in self.prompts if i != index]
        self.prompts.append((index, label))
        self._invalidate_result()
        self._update_prompts()

    def pop_prompt(self):
        if self.prompts:
            self.prompts.pop()
            self._invalidate_result()
            self._update_prompts()

    def clear_prompts(self):
        self.prompts = []
        self._invalidate_result()
        self._update_prompts()

    def _update_prompts(self):
        self.view.prompts = list(self.prompts)
        positive = sum(value for _, value in self.prompts)
        self.lbl_prompts.setText(f'正点 {positive} · 负点 {len(self.prompts) - positive}')
        self.view.update()

    def _invalidate_result(self, *_):
        self.preview = self.candidates = self.result_indices = None
        self.cmb_candidate.blockSignals(True)
        self.cmb_candidate.clear()
        self.cmb_candidate.blockSignals(False)
        self._show_masks()
        self._update_actions()

    def _show_masks(self, *_):
        self.view.show_prediction = self.chk_prediction.isChecked()
        self.view.show_masks(self.preview, self._target_mask())

    def use_manual_region(self):
        if self.session is None or self.view.region() is None:
            QMessageBox.information(self, '手动选区', '请先使用「限定区域」框出目标。')
            return
        self._invalidate_result()
        self.preview = np.zeros(self.session.doc.total_points, bool)
        self.preview[self._scope_indices()] = True
        self.lbl_preview.setText(f'手动框选预览：{int(self.preview.sum()):,} 点；尚未应用。')
        self._show_masks()
        self._update_actions()

    def probe(self):
        if self.runner.busy:
            return
        self.request_kind = 'probe'
        self.lbl_env.setText('正在检测环境…')
        self.runner.submit(self.txt_python.text().strip(), dict(self._config(), op='probe'))

    def run_model(self):
        if self.session is None or self.runner.busy:
            return
        if not any(label for _, label in self.prompts):
            QMessageBox.information(self, '需要正点', '请至少在目标器官上点击一个正点。')
            return
        indices = self._scope_indices()
        if not 64 <= len(indices) <= 2000000:
            QMessageBox.information(self, '调整核查范围', '请将本次区域控制在 64～2,000,000 点；大点云可先限定区域或选择单个扫描。')
            return
        positions = np.searchsorted(indices, [i for i, _ in self.prompts])
        if np.any(positions >= len(indices)) or not np.array_equal(indices[positions], [i for i, _ in self.prompts]):
            QMessageBox.warning(self, '提示点已失效', '请清空提示点并在当前核查范围重新点击。')
            return
        data = dict(xyz=self.view.renderer.xyz[indices], rgb=self.view.renderer.rgb[indices],
                    prompts=positions, labels=np.array([value for _, value in self.prompts]))
        if self.chk_prior.isChecked():
            prior = self._target_mask()
            if prior is None:
                QMessageBox.information(self, '选择已有目标', '请先在左侧列表选择要修正的实例或语义类别。')
                return
            data['prior'] = prior[indices]
        input_path = str(Path(self.work.name) / 'input.npz')
        self.output_path = str(Path(self.work.name) / 'result.npz')
        np.savez_compressed(input_path, **data)
        self._invalidate_result()
        self.result_indices = indices
        self.request_kind = 'predict'
        self.lbl_preview.setText('Point-SAM 正在运行，首次加载权重可能需要一些时间…')
        self.runner.submit(self.txt_python.text().strip(), dict(self._config(), op='predict',
                            input=input_path, output=self.output_path))

    def _finished(self, success, result):
        if not success and result == '已取消 Point-SAM 推理。':
            self.lbl_preview.setText('已取消推理；已应用的标注未改变。')
            self.statusMessage.emit(str(result))
            self._update_actions()
            return
        if not success:
            self._log(str(result))
            self.lbl_env.setText('Point-SAM 未完成，请查看日志。')
            self.lbl_preview.setText('未生成新的预览；已应用的标注未改变。')
            QMessageBox.warning(self, 'Point-SAM', str(result))
        elif self.request_kind == 'probe':
            self.lbl_env.setText(result['message'])
            self._log(result['message'])
            QMessageBox.information(self, 'Point-SAM 环境检测', result['message'])
        else:
            try:
                with np.load(self.output_path, allow_pickle=False) as data:
                    logits, scores = data['logits'], data['scores']
                if self.session is None or self.result_indices is None or logits.shape != (len(scores), len(self.result_indices)):
                    raise ValueError('结果点数与本次核查范围不一致。')
                self.candidates = logits > 0
                self.cmb_candidate.blockSignals(True)
                self.cmb_candidate.clear()
                for i, score in enumerate(scores):
                    self.cmb_candidate.addItem(f'候选 {i + 1} · 模型分数 {score:.3f}', i)
                self.cmb_candidate.setCurrentIndex(int(np.argmax(scores)))
                self.cmb_candidate.blockSignals(False)
                self._candidate_changed()
                self._log(f'采样 {result["sampled"]:,}/{result["total"]:,} 点，耗时 {result["seconds"]:.1f}s，设备 {result["device"]}')
            except Exception as exc:
                self._invalidate_result()
                QMessageBox.warning(self, '结果读取失败', str(exc))
        self._update_actions()

    def _candidate_changed(self, *_):
        index = self.cmb_candidate.currentData()
        if self.candidates is None or index is None or self.session is None:
            return
        self.preview = np.zeros(self.session.doc.total_points, bool)
        self.preview[self.result_indices] = self.candidates[index]
        self.lbl_preview.setText(f'Point-SAM 预览：{int(self.preview.sum()):,} 点；金色区域尚未应用。')
        self._show_masks()
        self._update_actions()

    def apply_preview(self):
        if self.session is None or self.preview is None or self.runner.busy:
            return
        action = self.cmb_action.currentData()
        mode = self.cmb_instance.currentData()
        target = self._target()
        instance = (target[1], target[2]) if target and target[0] == 'instance' else None
        if action != 'fill':
            answer = QMessageBox.question(self, '确认标注修正',
                f'将执行「{self.cmb_action.currentText()}」，选区 {int(self.preview.sum()):,} 点。\n'
                + ('目标实例原边界外的点会恢复为未标注。\n' if action == 'replace_instance' else '')
                + '本次操作可以整步撤销，是否应用？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        try:
            changed = self.session.apply(self.preview, self.cmb_label.currentData(), action, mode, instance)
        except ValueError as exc:
            QMessageBox.warning(self, '无法应用', str(exc))
            return
        self.clear_prompts()
        self.view.renderer.refresh_labels(self.session.ann)
        self._refresh_list()
        self.lbl_target.setText('目标：未选择')
        self._refresh()
        self.statusMessage.emit(f'已应用核查结果：修改 {changed:,} 个点；Ctrl+Z 可撤销，Ctrl+S 保存。')

    def undo(self):
        if self.session and not self.runner.busy and self.session.undo():
            self.clear_prompts()
            self.view.renderer.refresh_labels(self.session.ann)
            self._refresh_list()
            self._refresh()
            self.statusMessage.emit('已撤销上一次核查修正。')

    def _point_size(self, value):
        self.lbl_point_size.setText(f'{value / 10:.1f}')
        self.view.set_point_size(value / 10)

    def save_as(self):
        if not self.session or self.runner.busy:
            return
        ext = cloudio.ext_of(self.path)
        if ext not in cloudio.WRITE_EXTS:
            ext = '.ply'
        default = str(Path(self.path).with_suffix('')) + '_reviewed' + ext if self.path else 'reviewed.ply'
        path, chosen = QFileDialog.getSaveFileName(self, '保存核查结果', default, cloudio.SAVE_FILTER)
        if not path:
            return
        if cloudio.ext_of(path) not in cloudio.WRITE_EXTS:
            extension = chosen.split('(*', 1)[-1].split(')', 1)[0].split()[0] if '(*' in chosen else ext
            path += extension
        self._write(path)

    def save_overwrite(self):
        if not self.session or self.runner.busy:
            return
        if not self.path or cloudio.ext_of(self.path) not in cloudio.WRITE_EXTS:
            self.save_as()
            return
        answer = QMessageBox.question(self, '确认覆盖保存', f'将替换当前文件：\n{self.path}\n是否继续？',
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self._write(self.path)

    def _write(self, path):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.session.save(path)
        except Exception as exc:
            QMessageBox.critical(self, '保存失败', str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh()
        QMessageBox.information(self, '保存成功', f'核查结果已保存：\n{path}\n\n共 {self.session.doc.total_points:,} 个点。')
        return True

    def confirm_discard(self):
        if self.runner.busy:
            answer = QMessageBox.question(self, 'Point-SAM 正在运行', '推理尚未完成，是否取消推理并继续关闭？',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return False
            self.runner.cancel()
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, '核查结果尚未保存',
            f'标注核查页有未保存的改动：\n{self.path or "未命名点云"}\n是否先保存？',
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            self.save_as()
            return not self.dirty
        return answer == QMessageBox.Discard

    def close_file(self):
        if not self.confirm_discard():
            return
        self.session = None
        self.clear_prompts()
        self.lst.clear()
        self.view.clear_document()
        self.mode_buttons[0].setChecked(True)
        self.cmb_scan.blockSignals(True)
        self.cmb_scan.clear()
        self.cmb_scan.addItem('全部扫描', -1)
        self.cmb_scan.blockSignals(False)
        self.lbl_target.setText('目标：未选择')
        self._refresh()

    def _refresh(self):
        if self.session:
            counts = self.session.ann.label_counts()
            self.lbl_stats.setText(f'{self.session.doc.total_points:,} 点 · {len(self.session.doc.scans)} 个扫描\n'
                + ' / '.join(f'{LABEL_NAMES[lid]} {counts.get(lid, 0):,}' for lid in (0, 1, 2, 3, 4)))
            self.lbl_path.setText(f'{Path(self.path).name or "未命名点云"}' + (' * 未保存' if self.dirty else ''))
            self.lbl_path.setToolTip(self.path)
        else:
            self.lbl_stats.setText('尚未载入点云')
            self.lbl_path.setText('当前文件：—')
            self.lbl_path.setToolTip('')
        self._update_scope_label()
        self._show_masks()
        self._update_actions()
        self.stateChanged.emit()

    def _busy_changed(self, busy):
        self.view.interaction_enabled = not busy
        self._update_actions()

    def _update_actions(self):
        if not hasattr(self, 'btn_close'):
            return
        ready = self.session is not None and not self.runner.busy
        for widget in (self.btn_open, self.btn_import, self.btn_probe, self.txt_repo, self.txt_python,
                       self.txt_checkpoint, self.cmb_device, self.spn_samples, self.spn_distance):
            widget.setEnabled(not self.runner.busy)
        for widget in (self.cmb_scan, self.lst, self.btn_run, self.btn_manual, self.btn_clear,
                       self.btn_pop, self.btn_save, self.btn_overwrite, self.btn_close,
                       self.chk_prior, self.cmb_action, self.cmb_label, self.cmb_instance, self.cmb_candidate,
                       *self.mode_buttons):
            widget.setEnabled(ready)
        self.btn_apply.setEnabled(ready and self.preview is not None and bool(self.preview.any()))
        self.btn_undo.setEnabled(ready and bool(self.session.history))
        self.btn_cancel.setEnabled(self.runner.busy)

    def _log(self, text):
        self.log.appendPlainText(str(text).strip())

    def shutdown(self):
        self.runner.shutdown()
        self.work.cleanup()
