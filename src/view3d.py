"""三维视图控件：相机交互 + 标注工具 + 叠加层绘制。"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPainterPath,
                           QPen, QPolygonF, QCursor)
from PySide6.QtWidgets import QWidget, QApplication

from renderer import PointCloudRenderer, PALETTE

MODE_NAV = "nav"
MODE_BRUSH = "brush"
MODE_ERASE = "erase"
MODE_BOX = "box"
MODE_CROP = "crop"


class PointCloudView(QWidget):
    """点云视图。

    交互约定
    --------
    * 左键拖动：旋转；中键/滚轮：缩放；右键拖动（或 Shift+左键）：平移
    * 工具模式下左键拖动执行标注（笔刷 / 橡皮 / 框选），右键取消当前笔画
    """

    hoverInfo = Signal(str)
    annotateRequest = Signal(object, int, int)   # (mask_bool_global, label, action)
    cropRequest = Signal(object)                 # (rect_local: QRect) 框选裁剪
    escapePressed = Signal()
    viewChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setMinimumSize(420, 320)

        self.renderer = PointCloudRenderer()
        self.mode = MODE_NAV
        self.brush_radius = 26
        self.current_label = 1
        self.fill_gaps = True
        self.single_click = False
        self.show_axes = True
        self.show_bbox = False

        self._last_pos = QPoint()
        self._drag_btn = None
        self._panning = False
        self._stroke_polys: List[List[QPoint]] = []
        self._stroke_active = False
        self._box: Optional[Tuple[QPoint, QPoint]] = None
        self._ctrl_depth = False      # 本次拖拽按住 Ctrl：临时穿透全部深度
        self._cursor = QPoint(-100, -100)
        self._img: Optional[QImage] = None
        self._preview = False
        self._pending = False

        self._repaint_timer = QTimer(self)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.setInterval(0)
        self._repaint_timer.timeout.connect(self._do_render)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._img = None
        self._schedule()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._schedule()

    def set_document(self, doc, ann) -> None:
        self.renderer.set_data(doc, ann)
        self.renderer.fit_view(max(self.width(), 32), max(self.height(), 32))
        self._img = None
        self._schedule()

    def clear_document(self) -> None:
        """关闭文件：清空点云与未完成的框选/笔画，回到空白视图。"""
        self.renderer.clear()
        self._box = None
        self._stroke_polys = []
        self._stroke_active = False
        self._panning = False
        self._drag_btn = None
        self._img = None
        self.set_mode(MODE_NAV)
        self._schedule()

    def refresh_labels(self, ann) -> None:
        self.renderer.refresh_labels(ann)
        self._schedule()

    def reset_view(self) -> None:
        self.renderer.fit_view(max(self.width(), 32), max(self.height(), 32))
        self._schedule()

    def set_view(self, which: str) -> None:
        """预设视角：top / side / iso。"""
        cam = self.renderer.camera
        if which == "top":
            cam.yaw, cam.pitch = 90.0, 89.0
        elif which == "side":
            cam.yaw, cam.pitch = 0.0, 6.0
        else:
            cam.yaw, cam.pitch = 38.0, 28.0
        self.reset_view()

    def set_color_mode(self, mode: str) -> None:
        self.renderer.color_mode = mode
        self._schedule()

    def set_point_size(self, size: float) -> None:
        self.renderer.point_size = size
        self._schedule()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setCursor(Qt.CrossCursor if mode != MODE_NAV else Qt.ArrowCursor)
        self.update()

    # ----------------------------------------------------------- 绘制 --

    def _schedule(self) -> None:
        if not self._pending:
            self._pending = True
            self._repaint_timer.start()

    def _do_render(self) -> None:
        self._pending = False
        if not hasattr(self.renderer, "xyz") or len(self.renderer.xyz) == 0:
            self._img = None
            self.update()
            return
        w = max(self.width(), 16); h = max(self.height(), 16)
        self._img = self.renderer.render(w, h)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(*self.renderer.bg))
        if self._img is not None and not self._img.isNull():
            p.drawImage(0, 0, self._img)
        elif len(self.renderer.xyz):
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignCenter, "正在渲染…")

        self._draw_overlay(p)
        p.end()

    def _draw_overlay(self, p: QPainter) -> None:
        # 笔画预览
        if self._stroke_polys:
            p.setRenderHint(QPainter.Antialiasing, True)
            col = QColor(*PALETTE[min(self.current_label, len(PALETTE) - 1)])
            if self.mode == MODE_ERASE:
                col = QColor(255, 120, 120)
            pen = QPen(col, max(1.5, self.brush_radius * 0.35))
            pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            for poly in self._stroke_polys:
                if len(poly) >= 2:
                    p.drawPolyline(QPolygonF([QPointF(q) for q in poly]))
                elif len(poly) == 1:
                    p.drawPoint(QPointF(poly[0]))

        # 框选
        if self._box is not None:
            a, b = self._box
            r = QRect(a, b).normalized()
            if self.mode == MODE_CROP:
                p.setPen(QPen(QColor(255, 190, 90), 1.6, Qt.DashLine))
                p.setBrush(QBrush(QColor(255, 190, 90, 30)))
            else:
                p.setPen(QPen(QColor(120, 200, 255), 1.4, Qt.DashLine))
                p.setBrush(QBrush(QColor(120, 200, 255, 36)))
            p.drawRect(r)

        # 坐标轴 / 包围盒叠加
        self._draw_size_tag(p)
        if self.show_bbox:
            self._draw_bbox(p)
        if self.show_axes:
            self._draw_axes(p)

        # 笔刷光标
        if self.mode in (MODE_BRUSH, MODE_ERASE) and not self._stroke_active:
            col = QColor(*PALETTE[min(self.current_label, len(PALETTE) - 1)])
            if self.mode == MODE_ERASE:
                col = QColor(255, 120, 120)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(QPen(col, 1.4))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(self._cursor),
                          float(self.brush_radius), float(self.brush_radius))

        # 左上角提示
        if self.mode in (MODE_BRUSH, MODE_ERASE):
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(QColor(210, 214, 220))
            f = QFont(); f.setPointSize(9); p.setFont(f)
            hint = "左键涂抹标注  右键结束一笔  [ / ] 调笔刷  Ctrl+Z 撤销" \
                if self.mode == MODE_BRUSH else "左键擦除  右键结束一笔  Ctrl+Z 撤销"
            p.drawText(14, self.height() - 14, hint)
        elif self.mode == MODE_CROP:
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(QColor(255, 200, 120))
            f = QFont(); f.setPointSize(9); p.setFont(f)
            p.drawText(14, self.height() - 14,
                       "框选裁剪：拖出矩形后选择保留/删除框内点  Esc 退出")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._schedule()

    # ------------------------------------------------------ 叠加层绘制 --

    AXIS_COLORS = ((238, 92, 92), (108, 214, 132), (104, 162, 255))   # X Y Z
    AXIS_NAMES = ("X", "Y", "Z")

    def _draw_axes(self, p: QPainter) -> None:
        """右下角坐标轴指示器（与 CloudCompare 的样式一致）。"""
        if len(self.renderer.xyz) == 0:
            return
        _eye, _fwd, right, up = self.renderer.camera.basis()
        r = 40.0 if self.width() > 520 else 30.0
        ring = r + 20.0
        cx = self.width() - ring - 20
        cy = self.height() - ring - 20
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(74, 80, 90), 1.0))
        p.setBrush(QBrush(QColor(22, 25, 30, 170)))
        p.drawEllipse(QPointF(cx, cy), ring, ring)

        f = QFont(); f.setPointSize(8); f.setBold(True)
        p.setFont(f)
        for name, vec, col in zip(self.AXIS_NAMES,
                                  ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                  self.AXIS_COLORS):
            v = np.asarray(vec, np.float64)
            dx = float(v @ right)
            dy = -float(v @ up)
            color = QColor(*col)
            pen = QPen(color, 2.0)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            tip = QPointF(cx + dx * r, cy + dy * r)
            p.drawLine(QPointF(cx, cy), tip)
            ang = math.atan2(tip.y() - cy, tip.x() - cx)
            for s in (1.0, -1.0):
                p.drawLine(tip, QPointF(tip.x() - 7.5 * math.cos(ang - s * 0.42),
                                        tip.y() - 7.5 * math.sin(ang - s * 0.42)))
            p.setPen(color)
            p.drawText(QRectF(cx + dx * (r + 12) - 8, cy + dy * (r + 12) - 8, 16, 16),
                       Qt.AlignCenter, name)
        p.setBrush(Qt.NoBrush)

    def _draw_bbox(self, p: QPainter) -> None:
        """世界包围盒线框（红色，借鉴 CloudCompare 的显示方式）。"""
        if len(self.renderer.xyz) == 0:
            return
        lo, hi = self.renderer.bounds()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float64)
        w, h = self.width(), self.height()
        sx, sy, zc = self.renderer.project_points(corners, w, h)
        if not np.all(zc > 1e-6):
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(220, 80, 80, 190), 1.0)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        # 顶点编号：x 为最低位，其次 y，最高位 z
        edges = [(0, 1), (2, 3), (4, 5), (6, 7),      # 沿 X
                 (0, 2), (1, 3), (4, 6), (5, 7),      # 沿 Y
                 (0, 4), (1, 5), (2, 6), (3, 7)]      # 沿 Z
        for a, b in edges:
            p.drawLine(QPointF(sx[a], sy[a]), QPointF(sx[b], sy[b]))

    def _draw_size_tag(self, p: QPainter) -> None:
        """左上角显示包围盒尺寸，便于判断尺度。"""
        if len(self.renderer.xyz) == 0:
            return
        lo, hi = self.renderer.bounds()
        d = hi - lo
        p.setRenderHint(QPainter.Antialiasing, False)
        f = QFont(); f.setPointSize(9); p.setFont(f)
        p.setPen(QColor(150, 156, 166))
        p.drawText(14, 22, f"尺寸 {d[0]:.2f} × {d[1]:.2f} × {d[2]:.2f} m"
                           f"    Z 轴向上")

    # ----------------------------------------------------------- 交互 --

    def wheelEvent(self, ev):
        d = ev.angleDelta().y()
        if d == 0:
            return
        cam = self.renderer.camera
        cam.distance *= 0.88 if d > 0 else 1.0 / 0.88
        cam.distance = max(cam.distance, 1e-3)
        self._schedule()
        self.viewChanged.emit()

    def _hit_test(self, pos: QPoint) -> int:
        return self.renderer.pick(pos.x(), pos.y(), self.width(), self.height(), 5)

    def mousePressEvent(self, ev):
        self._last_pos = ev.position().toPoint()
        self.setFocus()
        self._ctrl_depth = bool(ev.modifiers() & Qt.ControlModifier)
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._drag_btn = Qt.MiddleButton
            return
        if ev.button() == Qt.RightButton:
            if self._stroke_active:
                self._finish_stroke()
                return
            self._panning = True
            self._drag_btn = Qt.RightButton
            return
        if ev.button() == Qt.LeftButton:
            if self.mode == MODE_NAV or ev.modifiers() & Qt.ShiftModifier:
                self._panning = bool(ev.modifiers() & Qt.ShiftModifier)
                self._drag_btn = Qt.LeftButton
                return
            # 标注
            pos = ev.position().toPoint()
            if self.mode in (MODE_BOX, MODE_CROP):
                self._box = (pos, pos)
                self.update()
            else:
                self._stroke_active = True
                self._stroke_polys = [[pos]]
                self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        self._cursor = pos
        btn = ev.buttons()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        cam = self.renderer.camera

        if btn & Qt.MiddleButton or (btn & Qt.RightButton and self._panning) or \
                (btn & Qt.LeftButton and self._drag_btn == Qt.LeftButton and self._panning):
            # 平移：沿屏幕右/上向量移动中心
            eye, fwd, right, up = cam.basis()
            scale = cam.distance * math.tan(math.radians(cam.fov * 0.5)) * 2 / max(self.height(), 1)
            cam.center = cam.center + (-dx * scale) * right + (dy * scale) * up
            self._last_pos = pos
            self._schedule()
            self.viewChanged.emit()
            return

        if btn & Qt.LeftButton and self._drag_btn == Qt.LeftButton and self.mode == MODE_NAV:
            cam.yaw += dx * 0.35
            cam.pitch = max(-89.0, min(89.0, cam.pitch + dy * 0.3))
            self._last_pos = pos
            self._schedule()
            self.viewChanged.emit()
            return

        self._last_pos = pos

        if self._stroke_active and btn & Qt.LeftButton:
            last = self._stroke_polys[-1]
            if not last or (pos - last[-1]).manhattanLength() >= 2:
                last.append(QPoint(pos))
                self.update()
            return

        if self._box is not None and btn & Qt.LeftButton:
            self._box = (self._box[0], pos)
            self.update()
            return

        if self.mode in (MODE_BRUSH, MODE_ERASE):
            self.update()
        if btn == Qt.NoButton:
            idx = self._hit_test(pos)
            if idx > 0:
                r = self.renderer
                xyz = r.xyz[idx]
                self.hoverInfo.emit(
                    f"X {xyz[0]:.3f}  Y {xyz[1]:.3f}  Z {xyz[2]:.3f}    "
                    f"强度 {r.intensity[idx]:.4f}   "
                    f"RGB {r.rgb[idx,0]},{r.rgb[idx,1]},{r.rgb[idx,2]}   "
                    f"扫描 #{r.scan_ids[idx]}")
            else:
                self.hoverInfo.emit("")

    def mouseReleaseEvent(self, ev):
        pos = ev.position().toPoint()
        ctrl = self._ctrl_depth          # 按下时记录的 Ctrl：临时穿透全部深度
        self._ctrl_depth = False
        if self._box is not None:
            box = (self._box[0], pos)
            self._box = None
            if self.mode == MODE_CROP:
                r = QRect(box[0], box[1]).normalized()
                if r.width() >= 3 and r.height() >= 3:
                    self.cropRequest.emit(r)
            else:
                self._apply_box(box, self._depth_mode(ctrl))
            self.update()
            return
        if self._stroke_active:
            self._finish_stroke(self._depth_mode(ctrl))
            return
        self._panning = False
        self._drag_btn = None

    def leaveEvent(self, ev):
        self._cursor = QPoint(-100, -100)
        if self.mode in (MODE_BRUSH, MODE_ERASE):
            self.update()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_BracketLeft:
            self.brush_radius = max(6, int(self.brush_radius * 0.85))
            self.update()
        elif k == Qt.Key_BracketRight:
            self.brush_radius = min(180, int(self.brush_radius * 1.18) + 1)
            self.update()
        elif k == Qt.Key_Escape:
            self._box = None
            self._stroke_active = False
            self._stroke_polys = []
            self.escapePressed.emit()
            self.update()
        else:
            super().keyPressEvent(ev)

    # ----------------------------------------------------------- 标注 --

    def _finish_stroke(self, depth_mode=None) -> None:
        """把当前笔画栅格化并提交标注。"""
        polys = [p for p in self._stroke_polys if len(p) > 0]
        self._stroke_active = False
        self._stroke_polys = []
        if not polys:
            self.update()
            return
        w, h = self.width(), self.height()
        r = max(self.brush_radius, 2)
        x0 = min(min(q.x() for q in p) for p in polys) - r - 2
        x1 = max(max(q.x() for q in p) for p in polys) + r + 2
        y0 = min(min(q.y() for q in p) for p in polys) - r - 2
        y1 = max(max(q.y() for q in p) for p in polys) + r + 2
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(w - 1, int(x1)); y1 = min(h - 1, int(y1))
        if x1 <= x0 or y1 <= y0:
            self.update()
            return

        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        mask = np.zeros((bh, bw), np.uint8)
        # 生成一条粗线（画笔）在局部图像上的掩膜
        img = QImage(bw, bh, QImage.Format_Grayscale8)
        img.fill(0)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(255, 255, 255), r * 2.0)
        pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath()
        for poly in polys:
            path.moveTo(poly[0].x() - x0, poly[0].y() - y0)
            for q in poly[1:]:
                path.lineTo(q.x() - x0, q.y() - y0)
            if len(poly) == 1:
                path.lineTo(poly[0].x() - x0 + 0.01, poly[0].y() - y0)
        painter.drawPath(path)
        painter.end()

        ptr = img.constBits()
        arr = np.frombuffer(ptr, np.uint8, bw * bh).reshape(bh, bw)
        hit = arr > 96

        proj = self.renderer.project(w, h)
        if proj is None:
            self.update()
            return
        idx, sx, sy, zc = proj
        sel = (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)
        if not sel.any():
            self.update()
            return
        cand = np.nonzero(sel)[0]
        cx = sx[cand].astype(np.int32) - x0
        cy = sy[cand].astype(np.int32) - y0
        inside = hit[cy, cx]
        picked = cand[inside]
        if picked.size == 0:
            self.update()
            return
        # 按侧栏"拾取深度"设置过滤；按住 Ctrl 拖拽为穿透全部深度
        picked = picked[self.renderer.depth_keep(zc[picked], depth_mode)]
        glob = idx[picked]

        label = 0 if self.mode == MODE_ERASE else self.current_label
        self.annotateRequest.emit(glob, label, 0)
        self.update()

    @staticmethod
    def _depth_mode(ctrl: bool):
        """本次拾取用的深度范围：按住 Ctrl 为 "all"（穿透），否则沿用侧栏设置。"""
        return "all" if ctrl else None

    def _apply_box(self, box, depth_mode=None) -> None:
        """box 为两个 QPoint（按下点、释放点），与绘制时的 QRect 保持同一坐标系。"""
        a, b = box
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        if abs(ax - bx) < 2 and abs(ay - by) < 2:
            return
        w, h = self.width(), self.height()
        sel = self.renderer.screen_rect_indices(ax, ay, bx, by, w, h,
                                                depth_mode=depth_mode)
        if sel.size == 0:
            self.hoverInfo.emit("框内没有点")
            return
        label = 0 if self.mode == MODE_ERASE else self.current_label
        self.annotateRequest.emit(sel, label, 0)

    def fit_and_render(self) -> None:
        self._do_render()
