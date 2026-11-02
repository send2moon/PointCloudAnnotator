"""点云裁剪：按包围盒（世界坐标）或屏幕框选范围剔除点。

设计要点
--------
* 裁剪不改变剩余点的顺序与字段，`keep_mask` 直接与各字段做布尔索引，
  保证颜色/强度/结构一对一保留。
* 裁剪是**破坏性**操作（会改变点数），因此提供快照以便撤销。
* 同时支持"保留框内"与"删除框内"两种语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from e57io import Scan


@dataclass
class CropBox:
    """世界坐标轴对齐包围盒。"""
    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    zmin: float = 0.0
    zmax: float = 0.0

    @classmethod
    def from_bounds(cls, lo, hi) -> "CropBox":
        lo = np.asarray(lo, np.float64).ravel()
        hi = np.asarray(hi, np.float64).ravel()
        return cls(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]),
                   float(lo[2]), float(hi[2]))

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.zmin], np.float64)

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.xmax, self.ymax, self.zmax], np.float64)

    def clamp(self, box: "CropBox") -> "CropBox":
        """把本框夹到外框范围内，保证不越界。"""
        lo = np.maximum(self.lo, box.lo)
        hi = np.minimum(self.hi, box.hi)
        return CropBox.from_bounds(lo, hi)

    def size(self) -> np.ndarray:
        return np.maximum(self.hi - self.lo, 0.0)

    def is_valid(self, min_size: float = 1e-9) -> bool:
        return bool(np.all(self.hi - self.lo > min_size))

    def mask(self, xyz: np.ndarray) -> np.ndarray:
        """返回 xyz (N,3) 落在盒内的布尔掩膜。"""
        lo, hi = self.lo, self.hi
        return ((xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) &
                (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1]) &
                (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2]))


def scan_bounds(scans: List[Scan]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """所有扫描的联合包围盒 (lo, hi)。空数据返回 None。

    按 float64 取极值（与 ``compound_mask`` 的判据保持同一精度）：若先把列
    降成 float32 再取 min，边界上的点会被判成"盒外"而悄悄丢掉。
    """
    los, his = [], []
    for s in scans:
        x = s.points.get("cartesianX")
        if x is None or s.count == 0:
            continue
        xs = np.asarray(s.points["cartesianX"], np.float64)
        ys = np.asarray(s.points["cartesianY"], np.float64)
        zs = np.asarray(s.points["cartesianZ"], np.float64)
        if xs.size == 0:
            continue
        los.append([np.nanmin(xs), np.nanmin(ys), np.nanmin(zs)])
        his.append([np.nanmax(xs), np.nanmax(ys), np.nanmax(zs)])
    if not los:
        return None
    return np.min(los, axis=0), np.max(his, axis=0)


def compound_mask(doc, box: CropBox, invert: bool) -> List[Optional[np.ndarray]]:
    """按盒裁出每个扫描的保留掩膜；被完全裁掉的扫描返回 None。

    ``invert=True`` 表示"删除盒内、保留盒外"。
    """
    out: List[Optional[np.ndarray]] = []
    for s in doc.scans:
        n = s.count
        if n == 0:
            out.append(None)
            continue
        xyz = np.column_stack([
            np.asarray(s.points["cartesianX"], np.float64),
            np.asarray(s.points["cartesianY"], np.float64),
            np.asarray(s.points["cartesianZ"], np.float64),
        ])
        inside = box.mask(xyz)
        keep = ~inside if invert else inside
        out.append(keep if keep.any() else None)
    return out


def apply_masks(doc, masks: List[Optional[np.ndarray]], prune_empty: bool = True) -> Tuple[int, int]:
    """按掩膜就地裁剪各扫描，返回 (裁剪前点数, 裁剪后点数)。

    ``prune_empty=True`` 时移除已无点的扫描（避免空扫描写盘报错）。
    """
    before = doc.total_points
    kept_scans: List[Scan] = []
    for s, m in zip(doc.scans, masks):
        if m is None or not m.any():
            if prune_empty:
                continue
            kept_scans.append(s)
            continue
        if m.all():
            kept_scans.append(s)
            continue
        # 先定格原长度：逐列裁剪时 s.count 会随第一列变化
        n = s.count
        for k in list(s.points.keys()):
            col = np.asarray(s.points[k])
            if len(col) == n:
                s.points[k] = col[m]
        kept_scans.append(s)
    doc.scans = kept_scans
    return before, doc.total_points
