"""三维视图控件：相机交互 + 标注工具 + 叠加层绘制。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPolygonF, QCursor)
from PySide6.QtWidgets import QWidget, QApplication

from renderer import PointCloudRenderer, PALETTE

MODE_NAV = "nav"
MODE_BRUSH = "brush"
MODE_ERASE = "erase"
MODE_BOX = "box"
MODE_CROP = "crop"
MODE_INST = "inst"       # 手动框选穗实例


# ----------------------------------------------------------- 实例框 --

# 立方体 8 个顶点的 12 条棱（顶点编号：x 最低位，其次 y，最高位 z）
BOX_EDGES = ((0, 1), (2, 3), (4, 5), (6, 7),      # 沿 X
             (0, 2), (1, 3), (4, 6), (5, 7),      # 沿 Y
             (0, 4), (1, 5), (2, 6), (3, 7))      # 沿 Z


def instance_box_color(uid: int) -> Tuple[int, int, int]:
    """按实例编号取一个稳定的辨识色（HSV 色环按 47° 步进打散）。"""
    h = ((int(uid) * 47) % 360) / 360.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.72, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


# ------------------------------------------------- 坐标轴 + 色条（共用） --

AXIS_COLORS = ((238, 92, 92), (108, 214, 132), (104, 162, 255))   # X Y Z
AXIS_NAMES = ("X", "Y", "Z")
LABEL_NAMES = ("未标注", "穗", "叶", "茎", "其他")


def _fmt_num(v: float) -> str:
    if v is None:
        return ""
    a = abs(v)
    if a >= 100000:
        return f"{v:.3g}"
    if a < 0.005:            # 贴近 0 的浮点噪声按 0 显示，避免 4.57e-05 这种刻度
        return "0.00"
    return f"{v:.2f}"


def _draw_legend(p: QPainter, renderer, cx: float, cy: float, ring: float) -> None:
    """在坐标轴圆盘正上方画色条（仅高度/强度/标注三种着色）。"""
    leg = renderer.legend() if hasattr(renderer, "legend") else None
    if not leg:
        return
    kind, lo, hi = leg
    f = QFont(); f.setPointSize(8); p.setFont(f)
    if kind in ("height", "intensity"):
        bw, bh = 14.0, 110.0
        x0 = cx - bw / 2
        y1 = cy - ring - 16          # 色条底边（圆盘上方留 16 px）
        y0 = y1 - bh
        if y0 < 4:
            bh = max(50.0, y1 - 4); y0 = y1 - bh
        grad = QLinearGradient(x0, y1, x0, y0)   # 下=小值，上=大值
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            rgb = renderer._ramp(np.array([t]))[0]
            grad.setColorAt(t, QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        p.setPen(QPen(QColor(74, 80, 90), 1.0))
        p.setBrush(QBrush(grad))
        p.drawRect(QRectF(x0, y0, bw, bh))
        p.setBrush(Qt.NoBrush)
        p.setPen(QColor(232, 234, 238))
        unit = " m" if kind == "height" else ""
        p.drawText(QRectF(x0 + bw + 4, y0 - 8, 78, 14),
                   Qt.AlignLeft | Qt.AlignVCenter, _fmt_num(hi) + unit)
        p.drawText(QRectF(x0 + bw + 4, y1 - 6, 78, 14),
                   Qt.AlignLeft | Qt.AlignVCenter, _fmt_num(lo) + unit)
        p.setPen(QColor(200, 205, 212))
        p.drawText(QRectF(x0 - 28, y0 - 22, 90, 14),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   "高度" if kind == "height" else "强度")
    else:                             # 标注：分类色块 + 名称
        n = len(PALETTE)
        sh, bw = 18.0, 16.0
        x0 = cx - bw / 2
        y1 = cy - ring - 16
        y0 = y1 - sh * n
        f.setPointSize(7)
        p.setFont(f)
        for i in range(n):
            idx = n - 1 - i           # 顶部放编号大的，底部放「未标注」
            top = y0 + i * sh
            p.setPen(QPen(QColor(74, 80, 90), 1.0))
            p.setBrush(QBrush(QColor(*PALETTE[idx])))
            p.drawRect(QRectF(x0, top, bw, sh))
            p.setBrush(Qt.NoBrush)
            p.setPen(QColor(232, 234, 238))
            p.drawText(QRectF(x0 + bw + 4, top, 60, sh),
                       Qt.AlignLeft | Qt.AlignVCenter, LABEL_NAMES[idx])


def draw_axis_and_legend(p: QPainter, renderer, w: int, h: int,
                         with_legend: bool = True) -> None:
    """右下角坐标轴指示器 + 其正上方的色条（CloudCompare 风格）。

    两个功能页的三维视图都调用它，保证朝向指示与色条位置一致。
    """
    if len(getattr(renderer, "xyz", ())) == 0:
        return
    _eye, _fwd, right, up = renderer.camera.basis()
    r = 40.0 if w > 520 else 30.0
    ring = r + 20.0
    cx = w - ring - 20
    cy = h - ring - 20

    p.setRenderHint(QPainter.Antialiasing, True)
    if with_legend:
        _draw_legend(p, renderer, cx, cy, ring)

    p.setPen(QPen(QColor(74, 80, 90), 1.0))
    p.setBrush(QBrush(QColor(22, 25, 30, 170)))
    p.drawEllipse(QPointF(cx, cy), ring, ring)

    f = QFont(); f.setPointSize(8); f.setBold(True)
    p.setFont(f)
    for name, vec, col in zip(AXIS_NAMES,
                              ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                              AXIS_COLORS):
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


@dataclass
class Box3D:
    """世界坐标轴对齐的立方体框（用于穗实例的突出显示 / 分割预览）。"""
    lo: np.ndarray                       # (3,) 世界坐标下角
    hi: np.ndarray                       # (3,) 世界坐标上角
    text: str = ""                       # 框顶显示的编号
    color: Tuple[int, int, int] = (120, 200, 255)
    dashed: bool = False                 # True = 预览框（虚线）


class PointCloudView(QWidget):
    """点云视图。

    交互约定
    --------
    * 左键拖动：旋转；中键/滚轮：缩放；右键拖动（或 Shift+左键）：平移
    * 工具模式下左键拖动执行标注（笔刷 / 橡皮 / 框选），右键取消当前笔画
    """

    hoverInfo = Signal(str)
    annotateRequest = Signal(object, int, int)   # (mask_bool_global, label, action)
    instanceRequest = Signal(object)             # (sel_global) 手动框选穗实例
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
        self.show_instance_boxes = False      # 穗实例立方体框开关
        self.instance_boxes: List[Box3D] = [] # 已编号实例的框
        self.preview_boxes: List[Box3D] = []  # 分割试算的预览框（虚线）

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
        self.instance_boxes = []
        self.preview_boxes = []
        self.set_mode(MODE_NAV)
        self._schedule()

    def set_instance_boxes(self, boxes: List[Box3D]) -> None:
        """更新已编号实例的立方体框（世界坐标，随相机一起转动）。"""
        self.instance_boxes = list(boxes or [])
        self.update()

    def set_preview_boxes(self, boxes: List[Box3D]) -> None:
        """更新分割试算的预览框；传空列表即清除预览。"""
        self.preview_boxes = list(boxes or [])
        self.update()

    def refresh_labels(self, ann) -> None:
        self.renderer.refresh_labels(ann)
        self._schedule()

    def redraw(self) -> None:
        """可见性 / 数据变化后重新光栅化（相机不动）。"""
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
            elif self.mode == MODE_INST:
                # 穗实例框选用穗类别的红色，和普通标注区分开
                p.setPen(QPen(QColor(255, 96, 96), 1.6, Qt.DashLine))
                p.setBrush(QBrush(QColor(255, 96, 96, 40)))
            else:
                p.setPen(QPen(QColor(120, 200, 255), 1.4, Qt.DashLine))
                p.setBrush(QBrush(QColor(120, 200, 255, 36)))
            p.drawRect(r)

        # 坐标轴 / 包围盒叠加
        self._draw_size_tag(p)
        self._draw_boxes(p)
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
        elif self.mode == MODE_INST:
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(QColor(255, 140, 140))
            f = QFont(); f.setPointSize(9); p.setFont(f)
            p.drawText(14, self.height() - 14,
                       "穗实例：框住一根穗的穗点，松手即归为新实例"
                       "（Ctrl+拖拽 = 穿透全部深度，Ctrl+Z 撤销）")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._schedule()

    # ------------------------------------------------------ 叠加层绘制 --

    AXIS_COLORS = AXIS_COLORS
    AXIS_NAMES = AXIS_NAMES

    def _draw_axes(self, p: QPainter) -> None:
        """右下角坐标轴指示器 + 其正上方的色条（与 CloudCompare 的样式一致）。"""
        draw_axis_and_legend(p, self.renderer, self.width(), self.height(),
                             with_legend=True)

    def _draw_boxes(self, p: QPainter) -> None:
        """穗实例立方体框 + 编号；试算预览用虚线框，顶部加提示条。

        预览期间不再画实线框：两批框叠在一起编号会糊成一团，预览本来就
        是为了看"换成这组参数会长什么样"。
        """
        if self.preview_boxes:
            boxes = list(self.preview_boxes)
        else:
            boxes = list(self.instance_boxes) if self.show_instance_boxes else []
        if not boxes or len(self.renderer.xyz) == 0:
            return
        w, h = self.width(), self.height()
        p.setRenderHint(QPainter.Antialiasing, True)
        f = QFont(); f.setPointSize(8); f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        for b in boxes:
            lo = np.asarray(b.lo, np.float64)
            hi = np.asarray(b.hi, np.float64)
            if lo.shape != (3,) or hi.shape != (3,) or not np.all(hi >= lo):
                continue
            corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                               np.float64)
            sx, sy, zc = self.renderer.project_points(corners, w, h)
            if not np.all(zc > 1e-6):
                continue                      # 有顶点在相机后面，画出来会翻转
            if sx.max() < -40 or sx.min() > w + 40 or sy.max() < -40 or sy.min() > h + 40:
                continue                      # 整框在屏幕外
            col = QColor(*b.color)
            pen = QPen(col, 1.4)
            if b.dashed:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            for a, c in BOX_EDGES:
                p.drawLine(QPointF(sx[a], sy[a]), QPointF(sx[c], sy[c]))
            # 编号：写在框顶面中心的正上方，带底色小牌，避免与点云混在一起
            if b.text and (sx.max() - sx.min() >= 8 or sy.max() - sy.min() >= 8):
                anchor = np.array([(lo[0] + hi[0]) * 0.5,
                                   (lo[1] + hi[1]) * 0.5, hi[2]], np.float64)
                ax, ay, az = self.renderer.project_points(anchor.reshape(1, 3), w, h)
                if az[0] > 1e-6:
                    tw = fm.horizontalAdvance(b.text)
                    r = QRectF(float(ax[0]) - tw / 2 - 4, float(ay[0]) - fm.height() - 5,
                               tw + 8, fm.height() + 2)
                    p.setPen(QPen(col, 1.0))
                    p.setBrush(QBrush(QColor(22, 25, 30, 205)))
                    p.drawRoundedRect(r, 4, 4)
                    p.setPen(col)
                    p.drawText(r, Qt.AlignCenter, b.text)
        # 预览模式提示条
        if self.preview_boxes:
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(QColor(255, 205, 90))
            p.drawText(14, 46, "穗分割预览（虚线框 = 将生成的穗，编号为分配后的实例号；未写入标注）")

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
            if self.mode in (MODE_BOX, MODE_CROP, MODE_INST):
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
        if self.mode == MODE_INST:
            # 手动穗实例：选中集交给主窗口去发编号（只动 label==1 的穗点）
            self.instanceRequest.emit(sel)
            return
        label = 0 if self.mode == MODE_ERASE else self.current_label
        self.annotateRequest.emit(sel, label, 0)

    def fit_and_render(self) -> None:
        self._do_render()
