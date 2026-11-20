"""点云三维渲染：轨道相机 + numpy 向量化软件光栅化。

设计取向
--------
* 只依赖 numpy 与 Qt 的 QImage，不引入 OpenGL/Open3D，避免部署依赖。
* 每帧把全部点投影到屏幕，用 uint32 缓冲区做"最近深度优先"的散射绘制；
  360 万点在 2560×1440 视口下约 30–60 ms，可交互。
* 颜色可切换：真彩 / 高度 / 强度 / 标注。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont

PALETTE = np.array([
    (150, 150, 150),   # 0 未标注
    (255, 96, 96),     # 1 穗
    (96, 220, 120),    # 2 叶
    (110, 170, 255),   # 3 茎
    (255, 205, 90),    # 4 其他
], dtype=np.uint8)


def point_in_polygon(sx, sy, poly) -> np.ndarray:
    """向量化偶奇（even-odd）射线法：屏幕点是否落在多边形内。

    从每个点向 **-x** 方向发射一条水平射线，统计它与多边形各边的交点数：
    穿过奇数次 → 在内部，偶数次 → 在外部。因此 ``inside`` 必须**从 False
    起算**，每命中一条边用异或翻转一次。

    .. warning::
       历史 bug：``CropView._inside_polygon`` 与 ``screen_polygon_indices``
       都曾把初值写成 ``np.ones``，结果**内外完全颠倒**——圈住的没裁、
       框外的全裁了，正是用户看到的「和标记区域不一致」。两处现已统一
       调用本函数，不要再各自复制一份。

    边按半开区间 ``[min(y1,y2), max(y1,y2))`` 参与判定，保证共享顶点只被
    计一次；与 ``QPainter.drawPolygon`` 默认的 ``OddEvenFill`` 规则一致，
    所以填充预览与命中判定对凹多边形、自交套索也保持同解。
    """
    sx = np.asarray(sx, np.float64)
    sy = np.asarray(sy, np.float64)
    p = np.asarray(poly, np.float64)
    inside = np.zeros(sx.shape, bool)          # ← 必须是 False 起算
    n = len(p)
    if n < 3:
        return inside
    for i in range(n):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % n]
        if y1 == y2:                            # 水平边不与水平射线相交
            continue
        lo, hi = (y1, y2) if y1 < y2 else (y2, y1)
        cond = (sy >= lo) & (sy < hi)
        if not cond.any():
            continue
        # 该边在 y=sy 处的交点 x；分母已排除 0（y1 != y2）
        xint = x1 + (sy - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (sx < xint)
    return inside


@dataclass
class Camera:
    """轨道相机：绕点云中心旋转，正交/透视可切。"""
    center: np.ndarray = None
    distance: float = 1.0
    yaw: float = 45.0       # 绕 Z 轴，度
    pitch: float = 25.0     # 抬升角，度
    fov: float = 35.0

    def __post_init__(self):
        if self.center is None:
            self.center = np.zeros(3, np.float32)

    def eye(self) -> np.ndarray:
        ya, pa = np.radians(self.yaw), np.radians(self.pitch)
        d = self.distance
        return self.center + np.array([
            d * np.cos(pa) * np.cos(ya),
            d * np.cos(pa) * np.sin(ya),
            d * np.sin(pa),
        ], np.float32)

    def basis(self):
        eye = self.eye()
        fwd = self.center - eye
        n = np.linalg.norm(fwd)
        if n < 1e-9:
            fwd = np.array([0, 0, -1], np.float32)
        else:
            fwd = fwd / n
        up0 = np.array([0, 0, 1], np.float32)
        if abs(float(np.dot(fwd, up0))) > 0.999:
            up0 = np.array([0, 1, 0], np.float32)
        right = np.cross(fwd, up0)
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, fwd)
        return eye.astype(np.float32), fwd.astype(np.float32), right.astype(np.float32), up.astype(np.float32)


class PointCloudRenderer:
    """管理点数据与渲染。"""

    def __init__(self) -> None:
        self.xyz = np.zeros((0, 3), np.float32)
        self.rgb = np.zeros((0, 3), np.uint8)
        self.intensity = np.zeros(0, np.float32)
        self.labels = np.zeros(0, np.uint8)
        self.scan_ids = np.zeros(0, np.int32)       # 每点所属扫描序号
        self.local_index = np.zeros(0, np.int64)    # 每点在其扫描内的序号
        self.hidden = np.zeros(0, bool)             # True = 不显示（也不参与拾取）
        self.camera = Camera()
        self.color_mode = "rgb"                     # rgb | height | intensity | label
        self.point_size = 1.4
        self.bg = (26, 29, 34)

        # 拾取（框选/笔刷）的深度范围：决定了框里哪些点会被标注
        #   surface —— 只取最前面 depth_tol 米厚的一层（默认，防误标背面）
        #   front25 / front50 —— 框内点按深度排序取前 25% / 50%
        #   all     —— 穿透，框内全部点都标注
        self.depth_mode = "surface"
        self.depth_tol = 0.05                       # 米（surface 模式下的层厚）

        self._dirty = True
        self._last_ms = 0.0
        self._img: Optional[QImage] = None
        self._proj_cache = None
        self._view_size = (0, 0)
        # 当前着色方式对应的色条信息：(kind, lo, hi)；rgb 时为 None。
        # kind ∈ {"height", "intensity", "label"}，供视图在坐标轴上方画色条。
        self._legend = None

    # ------------------------------------------------------------ 数据 --

    def set_data(self, doc, ann, only_scans=None) -> None:
        xs, ys, zs, rs, gs, bs, ins, sid, lid = [], [], [], [], [], [], [], [], []
        for i, s in enumerate(doc.scans):
            if only_scans is not None and i not in only_scans:
                continue
            n = s.count
            if n == 0:
                continue
            g = lambda k: s.points.get(k)
            x = g("cartesianX"); y = g("cartesianY"); z = g("cartesianZ")
            if x is None:
                continue
            xs.append(np.asarray(x, np.float32)); ys.append(np.asarray(y, np.float32))
            zs.append(np.asarray(z, np.float32))
            r = g("colorRed"); gg = g("colorGreen"); b = g("colorBlue")
            if r is not None and gg is not None and b is not None:
                rs.append(np.asarray(r, np.uint8)); gs.append(np.asarray(gg, np.uint8))
                bs.append(np.asarray(b, np.uint8))
            else:
                rs.append(np.full(n, 180, np.uint8)); gs.append(np.full(n, 180, np.uint8))
                bs.append(np.full(n, 180, np.uint8))
            it = g("intensity")
            ins.append(np.asarray(it, np.float32) if it is not None else np.zeros(n, np.float32))
            sid.append(np.full(n, i, np.int32))
            lid.append(np.arange(n, dtype=np.int64))

        if not xs:
            self.clear()
            return

        self.xyz = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
        self.rgb = np.column_stack([np.concatenate(rs), np.concatenate(gs), np.concatenate(bs)])
        self.intensity = np.concatenate(ins)
        self.scan_ids = np.concatenate(sid)
        self.local_index = np.concatenate(lid)
        self.hidden = np.zeros(len(self.xyz), bool)
        self.labels = self._labels_for_display(ann)
        self.reset_camera()
        self._dirty = True

    def clear(self) -> None:
        """清空渲染数据（关闭文件时回到空白视图）。"""
        self.xyz = np.zeros((0, 3), np.float32)
        self.rgb = np.zeros((0, 3), np.uint8)
        self.intensity = np.zeros(0, np.float32)
        self.labels = np.zeros(0, np.uint8)
        self.scan_ids = np.zeros(0, np.int32)
        self.local_index = np.zeros(0, np.int64)
        self.hidden = np.zeros(0, bool)
        self._img = None
        self._dirty = True

    def set_hidden(self, mask=None) -> None:
        """设置隐藏掩膜（True = 不显示）。传 None 表示全部显示。

        隐藏只影响显示与拾取：``scan_ids`` / ``local_index`` 一列不动，
        所以隐藏期间标注、存盘仍然写回正确的扫描与点序号。
        """
        n = len(self.xyz)
        if mask is None:
            self.hidden = np.zeros(n, bool)
        else:
            m = np.asarray(mask, bool)
            self.hidden = m if m.size == n else np.zeros(n, bool)
        self._dirty = True

    def visible_mask(self) -> np.ndarray:
        """当前可见点的布尔掩膜（长度等于渲染数组）。"""
        n = len(self.xyz)
        if self.hidden.size != n:
            return np.ones(n, bool)
        return ~self.hidden

    def n_visible(self) -> int:
        return int(self.visible_mask().sum())

    def reset_camera(self) -> None:
        if len(self.xyz) == 0:
            return
        w, h = self._view_size
        if w >= 16 and h >= 16:
            self.fit_view(w, h)
            return
        lo, hi = self.bounds()
        self.camera.center = ((lo + hi) / 2).astype(np.float32)
        radius = float(np.linalg.norm(hi - lo)) / 2
        self.camera.distance = max(radius / np.tan(np.radians(self.camera.fov / 2)) * 1.35, 1e-3)
        self._dirty = True

    def bounds(self):
        """当前可见点云的世界包围盒 (lo, hi)；全部隐藏时退回全部点。"""
        if len(self.xyz) == 0:
            return np.zeros(3, np.float64), np.zeros(3, np.float64)
        pts = self.xyz
        if self.hidden.size == len(pts) and self.hidden.any():
            vis = ~self.hidden
            if vis.any():
                pts = pts[vis]
        return (pts.min(axis=0).astype(np.float64),
                pts.max(axis=0).astype(np.float64))

    def fit_view(self, w: int, h: int, margin: float = 1.08) -> None:
        """让包围盒刚好填满视口（按 8 个角点的投影尺寸迭代收敛）。

        直接用包围球半径算距离会让扁平的样方看起来很小，这里按实际投影
        比例缩放，两三次迭代即可贴边。
        """
        if len(self.xyz) == 0:
            return
        lo, hi = self.bounds()
        self.camera.center = ((lo + hi) / 2).astype(np.float32)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float64)
        radius = 0.5 * float(np.linalg.norm(hi - lo))
        self.camera.distance = max(radius / np.tan(np.radians(self.camera.fov * 0.5)), 1e-3)
        for _ in range(4):
            sx, sy, zc = self.project_points(corners, w, h)
            if not (np.all(np.isfinite(sx)) and np.all(zc > 1e-6)):
                break
            ex = float(np.abs(sx - w * 0.5).max()) / (w * 0.5)
            ey = float(np.abs(sy - h * 0.5).max()) / (h * 0.5)
            # 距离与投影尺寸近似成反比：把当前占比压到 1/margin 即可
            f = max(ex, ey) * margin
            if not np.isfinite(f) or f <= 0 or abs(f - 1.0) < 0.02:
                break
            self.camera.distance = max(self.camera.distance * f, 1e-3)
        self._dirty = True

    def project_points(self, pts, w: int, h: int):
        """把任意世界坐标点投到屏幕，返回 (sx, sy, 相机深度)。"""
        eye, fwd, right, up = self.camera.basis()
        rel = np.asarray(pts, np.float64) - eye
        zc = rel @ fwd
        f = (h * 0.5) / np.tan(np.radians(self.camera.fov * 0.5))
        inv = f / np.maximum(zc, 1e-6)
        sx = (rel @ right) * inv + w * 0.5
        sy = -(rel @ up) * inv + h * 0.5
        return sx, sy, zc

    def _labels_for_display(self, ann) -> np.ndarray:
        """按当前显示的扫描/局部序号组装标签数组。

        单扫描模式（``only_scans``）下 renderer 只持有一个扫描的点，而
        ``ann.labels`` 是全部扫描的列表，直接 concatenate 会长度不符、
        着色错位甚至索引越界，这里逐扫描按 ``local_index`` 取。
        """
        n = len(self.xyz)
        if n == 0 or not getattr(ann, "labels", None):
            return np.zeros(n, np.uint8)
        out = np.zeros(n, np.uint8)
        for i, lab in enumerate(ann.labels):
            m = self.scan_ids == i
            if not m.any():
                continue
            lab = np.asarray(lab)
            if lab.size == 0:
                continue
            idx = self.local_index[m]
            if idx.size and int(idx.max()) < lab.size:
                out[m] = lab[idx]
        return out

    def refresh_labels(self, ann) -> None:
        if not getattr(ann, "labels", None):
            return
        self.labels = self._labels_for_display(ann)
        self._dirty = True

    def invalidate(self) -> None:
        self._dirty = True

    # ------------------------------------------------------------ 投影 --

    def project(self, w: int, h: int):
        """返回 (全局索引 idx, 屏幕 x, 屏幕 y, 相机空间深度 z)。

        投影矩阵按行展开手写，避免多次 (N,3)@(3,) 的大数组临时量。
        """
        eye, fwd, right, up = self.camera.basis()
        x = self.xyz[:, 0] - eye[0]
        y = self.xyz[:, 1] - eye[1]
        z = self.xyz[:, 2] - eye[2]
        zc = x * fwd[0] + y * fwd[1] + z * fwd[2]
        margin = self.camera.distance * 1e-4
        vis = zc > margin
        if not vis.any():
            return None
        f = (h * 0.5) / np.tan(np.radians(self.camera.fov * 0.5))
        inv = f / np.maximum(zc, 1e-6)
        sx = (x * right[0] + y * right[1] + z * right[2]) * inv + w * 0.5
        sy = -(x * up[0] + y * up[1] + z * up[2]) * inv + h * 0.5
        inside = vis & (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        if self.hidden.size == len(inside) and self.hidden.any():
            inside &= ~self.hidden          # 隐藏点不参与显示/拾取/框选
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            return None
        return idx, sx[idx], sy[idx], zc[idx]

    def render(self, w: int, h: int) -> QImage:
        w = max(int(w), 16); h = max(int(h), 16)
        self._view_size = (w, h)
        t0 = time.perf_counter()
        buf = np.empty((h, w), np.uint32)
        n = len(self.xyz)
        bgc = (self.bg[0] << 16) | (self.bg[1] << 8) | self.bg[2]
        if n == 0:
            buf.fill(bgc)
            self._img = self._to_qimage(buf, w, h)
            return self._img

        proj = self.project(w, h)
        if proj is None:
            buf.fill(bgc)
            self._img = self._to_qimage(buf, w, h)
            return self._img
        idx, sx, sy, zc = proj

        zmin, zmax = float(zc.min()), float(zc.max())
        rgb = self._colors(idx, zmin, zmax)
        val = (rgb[:, 0].astype(np.uint32) << 16) | \
              (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)

        # 深度量化成 8 位层号（0=最远, 255=最近），从远到近逐段散射覆盖。
        # 不做全局排序（350 万点 argsort 约 135 ms）；按 32 段区间取索引
        # 再写缓冲，同段内部先后顺序对结果影响可忽略。
        span = max(zmax - zmin, 1e-9)
        layer = 255 - np.minimum((255.0 * (zc - zmin) / span), 255.0).astype(np.int32)
        bx = np.rint(sx).astype(np.int32)
        by = np.rint(sy).astype(np.int32)

        buf.fill(bgc)
        flat = buf.reshape(-1)
        if self.point_size >= 2.6:
            offs = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2))
        elif self.point_size >= 1.9:
            offs = ((0, 0), (1, 0), (0, 1), (1, 1))
        else:
            offs = ((0, 0),)

        STEP = 32
        for lo in range(0, 256, STEP):
            sel = np.nonzero((layer >= lo) & (layer < lo + STEP))[0]
            if sel.size == 0:
                continue
            px0 = bx[sel]; py0 = by[sel]; vv = val[sel]
            for dx, dy in offs:
                px = px0 + dx; py = py0 + dy
                ok = (py >= 0) & (py < h)
                if dx:
                    ok &= (px >= 0) & (px < w)
                else:
                    ok &= (px >= 0) & (px < w)
                if ok.all():
                    flat[py * w + px] = vv
                elif ok.any():
                    flat[py[ok] * w + px[ok]] = vv[ok]

        self._last_ms = (time.perf_counter() - t0) * 1000
        self._dirty = False
        self._img = self._to_qimage(buf, w, h)
        return self._img

    @staticmethod
    def _counting_order(keys: np.ndarray, nbins: int = 256) -> np.ndarray:
        """0..nbins-1 的整型键做稳定计数排序，返回重排索引。

        单次 argsort 在 360 万点规模要约 60 ms；这里用 bincount + 顺序填充，
        实测约 8 ms。
        """
        counts = np.bincount(keys, minlength=nbins)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
        out = np.empty(keys.size, np.int64)
        # 稳定：同一层内保持原顺序（np.flatnonzero 本身按索引升序）
        for b in range(nbins):
            c = int(counts[b])
            if c:
                s = int(starts[b])
                out[s:s + c] = np.flatnonzero(keys == b)
        return out

    def _colors(self, idx: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
        """按当前着色方式取颜色，并顺手记录色条信息（``self._legend``）。

        高度/强度用**世界坐标 / 强度自身**的取值范围归一化（而不是相机深度），
        这样视图上画的色条刻度与点的实际颜色一致；真彩色时不画色条。
        """
        mode = self.color_mode
        if mode == "rgb":
            self._legend = None
            return self.rgb[idx]
        if mode == "label":
            self._legend = ("label", None, None)
            return PALETTE[np.clip(self.labels[idx], 0, len(PALETTE) - 1)]
        if mode == "intensity":
            v = self.intensity[idx]
            lo, hi = float(v.min()), float(v.max())
            self._legend = ("intensity", lo, hi)
            t = (v - lo) / max(hi - lo, 1e-9)
            return self._ramp(t)
        # height：按世界 Z 值着色
        z = self.xyz[idx, 2]
        lo, hi = float(z.min()), float(z.max())
        self._legend = ("height", lo, hi)
        t = (z - lo) / max(hi - lo, 1e-9)
        return self._ramp(t)

    def legend(self):
        """当前色条信息：``(kind, lo, hi)`` 或 None。

        kind ∈ {"height", "intensity", "label"}；label 的 lo/hi 为 None。
        未渲染过时返回 None。
        """
        return getattr(self, "_legend", None)

    @staticmethod
    def _ramp(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
        return (np.column_stack([r, g, b]) * 255).astype(np.uint8)

    @staticmethod
    def _to_qimage(buf: np.ndarray, w: int, h: int) -> QImage:
        # uint32 (0x00RRGGBB) 与 QImage Format_RGB32 的内存序一致（小端）
        img = QImage(buf.tobytes(), w, h, w * 4, QImage.Format_RGB32)
        return img.copy()

    # ----------------------------------------------------------- 拾取 --

    def pick(self, x: float, y: float, w: int, h: int, radius: int = 4):
        """返回屏幕坐标 (x,y) 附近最近点的全局索引，找不到返回 -1。"""
        proj = self.project(w, h)
        if proj is None:
            return -1
        idx, sx, sy, zc = proj
        dx = sx - int(x); dy = sy - int(y)
        m = (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
        if not m.any():
            return -1
        cand = np.nonzero(m)[0]
        best = cand[np.argmin(zc[cand])]
        return int(idx[best])

    def depth_keep(self, zs: np.ndarray, mode: str = None) -> np.ndarray:
        """按"拾取深度"设置，从一批候选点的相机深度里挑出要标注的。

        ``zs`` 是候选点的相机深度（越小越靠前），返回等长布尔掩膜。
        表面层模式用 1% 分位而不是最小值做基准，避免框里混进一个孤立的
        前方噪点时整层基准被拉走。
        """
        zs = np.asarray(zs, np.float64)
        if zs.size == 0:
            return np.zeros(0, bool)
        mode = mode or self.depth_mode
        if mode == "all":
            return np.ones(zs.size, bool)
        if mode == "surface":
            base = float(np.percentile(zs, 1))
            return zs <= base + max(float(self.depth_tol), 0.0)
        frac = 0.5 if mode == "front50" else 0.25
        return zs <= float(np.percentile(zs, frac * 100.0))

    def screen_rect_indices(self, x0, y0, x1, y1, w, h, depth_mode: str = None):
        """矩形框选：返回框内点的全局索引（深度范围由 depth_mode 决定）。"""
        proj = self.project(w, h)
        if proj is None:
            return np.zeros(0, np.int64)
        idx, sx, sy, zc = proj
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        m = (sx >= lo_x) & (sx <= hi_x) & (sy >= lo_y) & (sy <= hi_y)
        if not m.any():
            return np.zeros(0, np.int64)
        if (depth_mode or self.depth_mode) == "all":
            return idx[m]
        keep = self.depth_keep(zc[m], depth_mode)
        pos = np.nonzero(m)[0][keep]
        return idx[pos]

    def world_bounds_of_screen_rect(self, x0, y0, x1, y1, w, h):
        """屏幕矩形内所有点（含被遮挡的）的世界坐标包围盒，返回 (lo, hi)。"""
        proj = self.project(w, h)
        if proj is None:
            return None
        idx, sx, sy, _zc = proj
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        m = (sx >= lo_x) & (sx <= hi_x) & (sy >= lo_y) & (sy <= hi_y)
        if not m.any():
            return None
        sub = self.xyz[idx[m]]
        return sub.min(axis=0).astype(np.float64), sub.max(axis=0).astype(np.float64)

    def screen_polygon_indices(self, pts, w, h):
        """套索/画笔轮廓：返回多边形内点的索引。

        判定统一走 :func:`point_in_polygon`（避免早期 ``np.ones`` 起算导致
        内外颠倒的老 bug）；这里的 40% 深度分位是标注页笔刷的历史语义，
        与裁剪页的「全部穿透」无关，保持不变。
        """
        proj = self.project(w, h)
        if proj is None or len(pts) < 3:
            return np.zeros(0, np.int64)
        idx, sx, sy, zc = proj
        inside = point_in_polygon(sx, sy, pts)
        sel = np.nonzero(inside)[0]
        if sel.size:
            zs = zc[sel]
            thr = np.percentile(zs, 40)
            sel = sel[zs <= thr]
        return idx[sel]
