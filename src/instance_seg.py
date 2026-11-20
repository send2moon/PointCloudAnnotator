"""穗实例分割：把一片「穗」点云按空间连通性拆成一个个独立麦穗。

思路
----
1. 把点云体素化（默认 1.2 cm 边长），同一体素内的点视作相连；
2. 相邻体素（默认 26 邻域）也视作相连，容忍点云稀疏造成的空隙；
3. 在体素邻接图上求连通分量，每个分量就是一个候选穗；
4. 点数少于 ``min_points`` 的分量判为碎片（噪点、断枝），不分配实例。

只依赖 numpy；装了 scipy 时自动用 ``connected_components`` 加速，
没有 scipy 就退回自带的并查集，结果一致。

返回的簇号从 1 开始，0 表示"不属于任何保留的实例"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SegmentParams:
    """分割参数（长度单位统一为厘米，与界面一致）。"""

    voxel_cm: float = 1.2      # 体素边长
    link: int = 1              # 相邻体素的最大格距（1 = 26 邻域）
    min_points: int = 25       # 一个穗至少包含的点数，小于此值判为碎片

    def to_dict(self) -> dict:
        return {"voxel_cm": float(self.voxel_cm), "link": int(self.link),
                "min_points": int(self.min_points)}

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentParams":
        return cls(voxel_cm=float(d.get("voxel_cm", 1.2)),
                   link=int(d.get("link", 1)),
                   min_points=int(d.get("min_points", 25)))


@dataclass
class SegmentResult:
    """分割结果。``cluster`` 与输入点一一对应（0 = 未分配）。"""

    cluster: np.ndarray        # (n,) int32
    n_clusters: int            # 保留的穗实例数
    sizes: np.ndarray          # (n_clusters,) 每个实例的点数，从大到小
    n_assigned: int            # 分配到实例的点数
    n_rejected: int            # 因点数不足被丢弃的点数
    n_components: int = 0      # 连通分量总数（含被丢弃的碎片）

    def summary(self) -> str:
        if self.n_clusters == 0:
            return "没有可分割的穗点（可调小体素或最小点数）"
        s = self.sizes
        return (f"{self.n_clusters} 个穗实例：{self.n_assigned:,} 点，"
                f"单个 {int(s.min()):,}–{int(s.max()):,} 点，"
                f"中位数 {int(np.median(s)):,} 点")


# --------------------------------------------------------------------- 图 --

def _voxel_keys(xyz: np.ndarray, voxel: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (唯一体素坐标, 每点所属体素序号, 各轴尺寸)。"""
    ijk = np.floor(xyz / voxel).astype(np.int64)
    ijk -= ijk.min(axis=0)                       # 平移到非负，便于打包
    uniq, inv = np.unique(ijk, axis=0, return_inverse=True)
    dims = uniq.max(axis=0) + 1
    return uniq, inv.astype(np.int64), dims


def _pack(ijk: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """把体素坐标压成一个 int64 键；dims 决定每轴的进制。"""
    d1 = int(dims[1]) + 1
    d2 = int(dims[2]) + 1
    return (ijk[:, 0] * d1 + ijk[:, 1]) * d2 + ijk[:, 2]


def _neighbour_offsets(link: int):
    """取"半边"邻域偏移，避免重复建边。"""
    r = max(1, int(link))
    offs = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                if dx * dx + dy * dy + dz * dz > r * r * 3:
                    continue
                offs.append((dx, dy, dz))
    return offs


def _components(M: int, edges: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """连通分量标号（长度为 M，取值 0..K-1）。"""
    a, b = edges
    if a.size == 0 or M == 0:
        return np.arange(M, dtype=np.int64)
    try:                                          # scipy 更快，但不是硬依赖
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(a.size, np.int8), (a, b)), shape=(M, M))
        _, comp = connected_components(g, directed=False)
        return comp.astype(np.int64)
    except Exception:                             # noqa: BLE001
        parent = np.arange(M, dtype=np.int64)

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = int(parent[root])
            while parent[x] != root:              # 路径压缩
                parent[x], x = root, int(parent[x])
            return root

        for u, v in zip(a.tolist(), b.tolist()):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru
        # 重新标号
        roots = np.array([find(i) for i in range(M)], dtype=np.int64)
        uniq, comp = np.unique(roots, return_inverse=True)
        return comp.astype(np.int64)


def segment_points(xyz: np.ndarray, params: SegmentParams) -> SegmentResult:
    """把一批点按空间连通性分成若干个穗。

    ``xyz`` 为 (n, 3) 世界坐标（单位与 ``params.voxel_cm`` 一致：米），
    ``cluster`` 返回 (n,) 的簇号，1..n_clusters，0 表示被丢弃的碎片。
    """
    xyz = np.asarray(xyz, np.float64)
    n = len(xyz)
    if n == 0:
        return SegmentResult(np.zeros(0, np.int32), 0, np.zeros(0, np.int64), 0, 0, 0)

    voxel = max(float(params.voxel_cm) / 100.0, 1e-6)   # cm → m（点云坐标单位是米）
    uniq, inv, dims = _voxel_keys(xyz, voxel)
    M = len(uniq)
    keys = _pack(uniq, dims)                        # unique 已按字典序 → keys 有序

    ea, eb = [], []
    for dx, dy, dz in _neighbour_offsets(params.link):
        sh = uniq + np.array([dx, dy, dz], np.int64)
        inside = np.all((sh >= 0) & (sh < dims), axis=1)
        if not inside.any():
            continue
        src = np.nonzero(inside)[0]
        q = _pack(sh[src], dims)
        pos = np.searchsorted(keys, q)
        ok = pos < M
        hit = np.zeros(src.size, bool)
        hit[ok] = keys[pos[ok]] == q[ok]
        if hit.any():
            ea.append(src[hit])
            eb.append(pos[hit])

    edges = (np.concatenate(ea) if ea else np.zeros(0, np.int64),
             np.concatenate(eb) if eb else np.zeros(0, np.int64))
    comp = _components(M, edges)
    comp_pt = comp[inv]                              # 每点所属分量
    n_components = int(comp_pt.max()) + 1 if n else 0

    # 按「点数」判断碎片（体素数会因稀疏而偏小，点数更直观）
    sizes_all = np.bincount(comp_pt, minlength=n_components)
    keep = np.nonzero(sizes_all >= max(1, int(params.min_points)))[0]
    if keep.size == 0:
        return SegmentResult(np.zeros(n, np.int32), 0, np.zeros(0, np.int64),
                             0, n, n_components)

    # 按点数从大到小重排，簇号 1..K
    keep = keep[np.argsort(-sizes_all[keep], kind="stable")]
    remap = np.zeros(n_components, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    cluster = remap[comp_pt].astype(np.int32)
    sizes = sizes_all[keep]
    return SegmentResult(cluster=cluster, n_clusters=int(len(keep)), sizes=sizes,
                         n_assigned=int(sizes.sum()),
                         n_rejected=int(n - sizes.sum()),
                         n_components=n_components)
