"""Point prompts and annotation/prediction comparison in the native 3D canvas."""
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from crop_view import CropView
from renderer import PointCloudRenderer


class ReviewRenderer(PointCloudRenderer):
    def __init__(self):
        super().__init__()
        self.preview = None
        self.target = None

    def _colors(self, idx, zmin, zmax):
        colors = super()._colors(idx, zmin, zmax).copy()
        if self.target is not None and len(self.target) == len(self.xyz):
            colors[self.target[idx]] = (55, 210, 235)
        if self.preview is not None and len(self.preview) == len(self.xyz):
            colors[self.preview[idx]] = (255, 190, 45)
        return colors


class ReviewView(CropView):
    promptPicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        background = self.renderer.bg
        self.renderer = ReviewRenderer()
        self.renderer.bg = background
        self.prompts = []
        self.show_prediction = True
        self.interaction_enabled = True

    def show_masks(self, preview=None, target=None):
        self.renderer.preview = preview if self.show_prediction else None
        self.renderer.target = target
        self.renderer.invalidate()
        self._img = None
        self._schedule()

    def mousePressEvent(self, ev):
        if not self.interaction_enabled and ev.button() == Qt.LeftButton and self.mode != 'nav':
            return
        if (self.mode in ('positive', 'negative') and ev.button() == Qt.LeftButton
                and not ev.modifiers() & Qt.ShiftModifier):
            self.setFocus()
            index = self.renderer.pick(ev.position().x(), ev.position().y(), self.width(), self.height(), 7)
            if index is not None and index >= 0:
                self.promptPicked.emit(index, int(self.mode == 'positive'))
            return
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            if self.interaction_enabled:
                self.clear_region()
                self.regionDefined.emit(None)
            return
        super().keyPressEvent(ev)

    def _draw_overlay(self, painter):
        super()._draw_overlay(painter)
        if not self.has_points():
            return
        for order, (index, positive) in enumerate(self.prompts, 1):
            if not 0 <= index < len(self.renderer.xyz) or self.renderer.hidden[index]:
                continue
            x, y, depth = self.renderer.project_points(self.renderer.xyz[index:index + 1], self.width(), self.height())
            if depth[0] <= 1e-6:
                continue
            color = QColor(60, 240, 125) if positive else QColor(255, 90, 110)
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(x[0], y[0]), 7, 7)
            painter.drawText(QPointF(x[0] + 9, y[0] - 7), f'{"+" if positive else "−"}{order}')
        painter.setPen(QColor(235, 235, 235) if sum(self.renderer.bg) < 500 else QColor(43, 87, 154))
        painter.drawText(12, 62, '青色：已有目标    金色：待应用选区（尚未修改标注）')
        if self.mode in ('positive', 'negative'):
            painter.drawText(12, self.height() - 12,
                             '左键添加正点/负点    右键平移    Shift+左键平移    切到浏览可旋转')
