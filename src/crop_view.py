"""裁剪页专用三维视图。

与标注页 ``PointCloudView`` 刻意分开：这里不写标注、不画实例框，
只负责「看点云 + 在屏幕上圈一块区域 + 把区域翻译成点索引」。

交互约定
--------
* 浏览模式：左键拖动旋转、右键/中键拖动平移、滚轮缩放
* 矩形模式：左键拖出一个矩形，松开即确定区域
* 多边形模式：左键逐点点击，双击 / 右键 / Enter 闭合；Esc 取消、Backspace 退一点
* 区域是**屏幕投影**意义上的：落在轮廓内即算命中，深度方向默认全部穿透
  （可切「仅表面层」，避免误删被遮挡的背面点）

预览旋转只替换 ``renderer.xyz`` 供渲染/拾取使用，索引顺序完全不变，
所以「先预览旋转再圈区域」得到的掩膜仍可直接作用于原始文档。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPen,
                           QPolygonF)
from PySide6.QtWidgets import QWidget

from office_theme import BLUE, CANVAS_COLORS
from renderer import PointCloudRenderer, point_in_polygon
from view3d import draw_axis_and_legend

MODE_NAV = "nav"
MODE_RECT = "rect"
MODE_POLY = "poly"

# 名称 → (yaw, pitch)。相机沿 -forward 看向 center，Z 恒为竖直方向。
VIEW_PRESETS = {
    "俯视 (XY)": (90.0, 89.0),
    "前视 (XZ)": (-90.0, 0.0),
    "侧视 (YZ)": (0.0, 0.0),
    "后视 (XZ)": (90.0, 0.0),
    "底视 (XY)": (90.0, -89.0),
    "等轴": (38.0, 28.0),
}

_REGION_FILL = QColor(43, 87, 154, 46)
_REGION_LINE = QColor(28, 78, 148, 235)
_PENDING_LINE = QColor(255, 150, 40, 240)


# 判定统一放在 renderer.point_in_polygon（False 起算的偶奇射线法）。
# 保留旧名只是为了让已有脚本/调用点不必改动，不要再在这里另写一份实现。
_inside_polygon = point_in_polygon


class CropView(QWidget):
    """可画矩形/任意多边形裁剪区域的点云视图。"""

    regionDefined = Signal(object)     # ("rect", (x0,y0,x1,y1)) | ("polygon", [(x,y)...])
    viewChanged = Signal(str)
    hoverInfo = Signal(str)
    notice = Signal(str)               # 需要提醒用户的状态变化（接到状态栏）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setMinimumSize(420, 360)

        self.renderer = PointCloudRenderer()
        self.renderer.bg = CANVAS_COLORS["深色（默认）"]
        self.mode = MODE_NAV
        self.view_name = "等轴"

        self._base_xyz: Optional[np.ndarray] = None
        self._preview_m: Optional[np.ndarray] = None
        self._preview_center: Optional[np.ndarray] = None

        self._last_pos = QPoint()
        self._drag_btn = None
        self._panning = False
        self._box: Optional[Tuple[QPoint, QPoint]] = None
        self._poly: List[QPoint] = []
        self._region: Optional[tuple] = None
        # 区域是「屏幕投影」意义上的，画定那一刻的相机与视口尺寸必须一起
        # 锁下来。否则用户圈完框又转视角/缩放/改窗口大小，region_mask 会用
        # 新投影重算，裁掉的就不是他圈住的那批点（看起来和标记区域不一致）。
        self._region_cam = None
        self._region_warned = False
        self._img: Optional[QImage] = None
        self._pending = False

        self._repaint_timer = QTimer(self)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.setInterval(0)
        self._repaint_timer.timeout.connect(self._do_render)

    # ------------------------------------------------------------ 数据 --

    def set_document(self, doc, ann=None) -> None:
        self.renderer.set_data(doc, ann)
        self.renderer.fit_view(max(self.width(), 32), max(self.height(), 32))
        self._base_xyz = self.renderer.xyz.copy()
        self._preview_m = None
        self._preview_center = None
        self.clear_region()
        self._img = None
        self._schedule()

    def clear_document(self) -> None:
        self.renderer.clear()
        self._base_xyz = None
        self._preview_m = None
        self._preview_center = None
        self.clear_region()
        self._img = None
        self.set_mode(MODE_NAV)
        self._schedule()

    def has_points(self) -> bool:
        return self._base_xyz is not None and len(self._base_xyz) > 0

    def n_points(self) -> int:
        return 0 if self._base_xyz is None else int(len(self._base_xyz))

    def set_canvas(self, name: str) -> None:
        self.renderer.bg = CANVAS_COLORS.get(name, CANVAS_COLORS["深色（默认）"])
        self._img = None
        self._schedule()

    def set_canvas_rgb(self, rgb) -> None:
        """自定义画布颜色（0..255 三元组）。"""
        self.renderer.bg = tuple(int(v) for v in rgb)
        self._img = None
        self._schedule()

    def set_preview_matrix(self, m: Optional[np.ndarray], center=None) -> None:
        """预览刚体旋转（3×3）。传 None 清除预览，坐标回到原始值。"""
        self._preview_m = None if m is None else np.asarray(m, np.float64)
        self._preview_center = None if center is None else np.asarray(center, np.float64)
        if self._base_xyz is None:
            return
        if self._preview_m is None:
            self.renderer.xyz = self._base_xyz
        else:
            c = self._preview_center
            if c is None:
                c = self._base_xyz.mean(axis=0)
            self.renderer.xyz = ((self._base_xyz - c) @ self._preview_m.T + c).astype(np.float32)
        self.renderer.invalidate()
        self._img = None
        self._schedule()

    def has_preview(self) -> bool:
        return self._preview_m is not None

    # ------------------------------------------------------------ 视图 --

    def set_view(self, name: str) -> None:
        if name not in VIEW_PRESETS:
            return
        self._warn_region_locked()
        yaw, pitch = VIEW_PRESETS[name]
        cam = self.renderer.camera
        cam.yaw, cam.pitch = yaw, pitch
        self.view_name = name
        self.renderer.fit_view(max(self.width(), 32), max(self.height(), 32))
        self._img = None
        self._schedule()
        self.viewChanged.emit(name)

    def fit_view(self) -> None:
        self._warn_region_locked()
        self.renderer.fit_view(max(self.width(), 32), max(self.height(), 32))
        self._img = None
        self._schedule()

    def set_color_mode(self, mode: str) -> None:
        self.renderer.color_mode = mode
        self.renderer.invalidate()
        self._img = None
        self._schedule()

    def set_point_size(self, size: float) -> None:
        self.renderer.point_size = float(size)
        self.renderer.invalidate()
        self._img = None
        self._schedule()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._box = None
        self._poly = []
        self.setCursor(Qt.CrossCursor if mode != MODE_NAV else Qt.ArrowCursor)
        self.update()

    def clear_region(self) -> None:
        self._region = None
        self._region_cam = None
        self._region_warned = False
        self._box = None
        self._poly = []
        self.update()

    def region(self):
        return self._region

    def _lock_view_for_region(self) -> None:
        """画定区域时锁下相机 + 视口尺寸，让掩膜永远对应「圈中那一刻」。"""
        c = self.renderer.camera
        self._region_cam = (float(c.yaw), float(c.pitch), c.center.copy(),
                            float(c.distance), float(c.fov),
                            max(self.width(), 16), max(self.height(), 16))
        self._region_warned = False

    def _warn_region_locked(self) -> None:
        """相机被改动时提醒一次：区域仍按画定时的视角裁剪。"""
        if self._region is not None and not self._region_warned:
            self._region_warned = True
            self.notice.emit(
                "裁剪区域已按画定时的视角锁定；之后旋转/缩放视角不会改变裁剪结果")

    # --------------------------------------------------------- 区域掩膜 --

    def region_mask(self, region=None, depth: str = "all") -> np.ndarray:
        """把屏幕区域翻译成全局点索引的布尔掩膜（长度 = 当前文档点数）。

        ``depth="all"`` 穿透全部深度（裁剪的默认语义）；``"surface"`` 只取
        最前面一层，适合只想修掉表面噪点的场合。
        视口之外的点一律视为「区域外」。
        """
        region = self._region if region is None else region
        n = self.n_points()
        mask = np.zeros(n, bool)
        if region is None or n == 0:
            return mask

        # 用画定区域时的相机/视口复现投影：期间用户转视角、缩放、改窗口
        # 大小都不会让掩膜漂移，_on_region 显示的命中数 == 实际裁掉的点数。
        cam = self.renderer.camera
        snap = self._region_cam
        if snap is not None:
            w, h = snap[5], snap[6]
            bak = (cam.yaw, cam.pitch, cam.center.copy(), cam.distance, cam.fov)
            cam.yaw, cam.pitch, cam.distance, cam.fov = snap[0], snap[1], snap[3], snap[4]
            cam.center = snap[2].copy()
        else:
            w, h = max(self.width(), 16), max(self.height(), 16)
        try:
            proj = self.renderer.project(w, h)
        finally:
            if snap is not None:
                cam.yaw, cam.pitch, cam.distance, cam.fov = bak[0], bak[1], bak[3], bak[4]
                cam.center = bak[2].copy()
        if proj is None:
            return mask
        idx, sx, sy, zc = proj
        kind, data = region
        if kind == "rect":
            x0, y0, x1, y1 = data
            inside = ((sx >= min(x0, x1)) & (sx <= max(x0, x1)) &
                      (sy >= min(y0, y1)) & (sy <= max(y0, y1)))
        elif kind == "polygon":
            poly = np.asarray(data, np.float64)
            inside = _inside_polygon(sx, sy, poly) if len(poly) >= 3 else np.zeros(len(sx), bool)
        else:
            return mask
        sel = np.nonzero(inside)[0]
        if sel.size and depth == "surface":
            sel = sel[self.renderer.depth_keep(zc[sel], "surface")]
        if sel.size:
            mask[idx[sel]] = True
        return mask

    # ------------------------------------------------------------ 绘制 --

    def _schedule(self) -> None:
        if not self._pending:
            self._pending = True
            self._repaint_timer.start()

    def _do_render(self) -> None:
        self._pending = False
        if not self.has_points():
            self._img = None
            self.update()
            return
        w, h = max(self.width(), 16), max(self.height(), 16)
        self._img = self.renderer.render(w, h)
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._img = None
        self._schedule()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._schedule()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(*self.renderer.bg))
        if self._img is not None and not self._img.isNull():
            p.drawImage(0, 0, self._img)
        elif self.has_points():
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignCenter, "正在渲染…")
        else:
            p.setPen(QColor(150, 156, 164))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "从左侧列表选择一个点云文件，或点击「打开文件夹」")
        self._draw_overlay(p)
        p.end()

    def _draw_overlay(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)

        # 已确定的区域：淡蓝填充 + 蓝色虚线
        if self._region is not None:
            kind, data = self._region
            if kind == "rect":
                x0, y0, x1, y1 = data
                poly = [QPointF(x0, y0), QPointF(x1, y0),
                        QPointF(x1, y1), QPointF(x0, y1)]
            else:
                poly = [QPointF(x, y) for x, y in data]
            self._paint_polygon(p, QPolygonF(poly), _REGION_LINE, _REGION_FILL, 1.8)

        # 正在拖的框
        if self._box is not None:
            a, b = self._box
            r = QRect(a, b).normalized()
            self._paint_polygon(p, QPolygonF([
                QPointF(r.left(), r.top()), QPointF(r.right(), r.top()),
                QPointF(r.right(), r.bottom()), QPointF(r.left(), r.bottom())]),
                _PENDING_LINE, QColor(255, 170, 60, 40), 1.6)

        # 正在画的多边形：折线 + 顶点小圆
        if len(self._poly) >= 1:
            poly = self._poly
            pen = QPen(_PENDING_LINE, 1.8)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            if len(poly) >= 2:
                p.drawPolyline(QPolygonF([QPointF(q) for q in poly]))
            for q in poly:
                p.setBrush(QColor(255, 255, 255))
                p.drawEllipse(QPointF(q), 3.2, 3.2)
                p.setBrush(Qt.NoBrush)

        # 坐标轴指示器 + 其正上方的色条（右下角，两个功能页样式一致）
        draw_axis_and_legend(p, self.renderer, self.width(), self.height(),
                             with_legend=True)

        # 左上角视图名 + 底部操作提示
        p.setRenderHint(QPainter.Antialiasing, False)
        f = QFont()
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255) if sum(self.renderer.bg) < 500 else QColor(43, 87, 154))
        tag = f"{self.view_name}   |   {self.n_points():,} 点"
        p.drawText(12, 22, tag)
        f.setBold(False)
        f.setPointSize(9)
        p.setFont(f)
        hint = {
            MODE_NAV: "左键拖动旋转   右键/中键拖动平移   滚轮缩放   F 适合窗口",
            MODE_RECT: "左键拖出矩形区域   松开确定   Esc 取消",
            MODE_POLY: "左键逐点点击   双击/右键/Enter 闭合   Backspace 退一点   Esc 取消",
        }.get(self.mode, "")
        p.drawText(12, self.height() - 12, hint)

    @staticmethod
    def _paint_polygon(p: QPainter, poly: QPolygonF, line: QColor,
                       fill: QColor, width: float) -> None:
        pen = QPen(line, width, Qt.DashLine)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(QBrush(fill))
        p.drawPolygon(poly)
        p.setBrush(Qt.NoBrush)

    # ------------------------------------------------------------ 交互 --

    def wheelEvent(self, ev):
        d = ev.angleDelta().y()
        if d == 0:
            return
        self._warn_region_locked()
        cam = self.renderer.camera
        cam.distance *= 0.88 if d > 0 else 1.0 / 0.88
        cam.distance = max(cam.distance, 1e-3)
        self._schedule()

    def mousePressEvent(self, ev):
        self._last_pos = ev.position().toPoint()
        self.setFocus()
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._drag_btn = Qt.MiddleButton
            return
        if ev.button() == Qt.RightButton:
            if self.mode == MODE_POLY and len(self._poly) >= 3:
                self._finish_polygon()
                return
            self._panning = True
            self._drag_btn = Qt.RightButton
            return
        if ev.button() == Qt.LeftButton:
            if self.mode == MODE_NAV or (ev.modifiers() & Qt.ShiftModifier):
                self._panning = bool(ev.modifiers() & Qt.ShiftModifier)
                self._drag_btn = Qt.LeftButton
                return
            pos = ev.position().toPoint()
            if self.mode == MODE_RECT:
                self._box = (pos, pos)
                self.update()
            elif self.mode == MODE_POLY:
                self._poly.append(QPoint(pos))
                self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        btn = ev.buttons()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        cam = self.renderer.camera

        if btn & Qt.MiddleButton or (btn & Qt.RightButton and self._panning) or \
                (btn & Qt.LeftButton and self._drag_btn == Qt.LeftButton and self._panning):
            self._warn_region_locked()
            eye, fwd, right, up = cam.basis()
            scale = cam.distance * math.tan(math.radians(cam.fov * 0.5)) * 2 / max(self.height(), 1)
            cam.center = cam.center + (-dx * scale) * right + (dy * scale) * up
            self._last_pos = pos
            self._schedule()
            return

        if btn & Qt.LeftButton and self._drag_btn == Qt.LeftButton and self.mode == MODE_NAV:
            self._warn_region_locked()
            cam.yaw += dx * 0.35
            cam.pitch = max(-89.0, min(89.0, cam.pitch + dy * 0.3))
            self._last_pos = pos
            self._schedule()
            return

        self._last_pos = pos
        if self._box is not None and btn & Qt.LeftButton:
            self._box = (self._box[0], pos)
            self.update()
            return
        if self.mode == MODE_POLY and self._poly and btn == Qt.NoButton:
            self.update()      # 让下一段虚线跟着光标走
            return
        if btn == Qt.NoButton:
            idx = self.renderer.pick(pos.x(), pos.y(), self.width(), self.height(), 5)
            if idx is not None and idx >= 0:
                xyz = self.renderer.xyz[idx]
                self.hoverInfo.emit(
                    f"X {xyz[0]:.3f}  Y {xyz[1]:.3f}  Z {xyz[2]:.3f}   "
                    f"RGB {self.renderer.rgb[idx, 0]},{self.renderer.rgb[idx, 1]},"
                    f"{self.renderer.rgb[idx, 2]}   扫描 #{self.renderer.scan_ids[idx] + 1}")
            else:
                self.hoverInfo.emit("")

    def mouseReleaseEvent(self, ev):
        pos = ev.position().toPoint()
        if self._box is not None:
            a, b = self._box[0], pos
            self._box = None
            r = QRect(a, b).normalized()
            if r.width() >= 4 and r.height() >= 4:
                region = ("rect", (r.left(), r.top(), r.right(), r.bottom()))
                self._region = region
                self._lock_view_for_region()
                self.regionDefined.emit(region)
            self.update()
            return
        self._panning = False
        self._drag_btn = None

    def mouseDoubleClickEvent(self, ev):
        if self.mode == MODE_POLY and ev.button() == Qt.LeftButton:
            self._finish_polygon()

    def _finish_polygon(self) -> None:
        if len(self._poly) < 3:
            self.hoverInfo.emit("多边形至少需要 3 个顶点")
            return
        region = ("polygon", [(q.x(), q.y()) for q in self._poly])
        self._poly = []
        self._region = region
        self._lock_view_for_region()
        self.regionDefined.emit(region)
        self.update()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_Escape:
            self.clear_region()
            self.hoverInfo.emit("已取消裁剪区域")
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            if self.mode == MODE_POLY:
                self._finish_polygon()
            else:
                super().keyPressEvent(ev)
        elif k == Qt.Key_Backspace and self._poly:
            self._poly.pop()
            self.update()
        elif k == Qt.Key_F:
            self.fit_view()
        else:
            super().keyPressEvent(ev)
