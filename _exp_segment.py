"""调参实验：在真实标注文件上跑穗实例分割，比较不同参数的效果与耗时。"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from e57io import E57Document
from instance_seg import SegmentParams, segment_points

SRC = (sys.argv[1] if len(sys.argv) > 1 else
       r"D:/Data/PC/ResearchPlots/DT/1-FS_09_labeled.e57")

doc = E57Document.read(SRC)
print(f"载入 {doc.total_points:,} 点 / {len(doc.scans)} 扫描")

# 汇总所有扫描的「穗」点（label == 1），带扫描号
xs, sid = [], []
for i, s in enumerate(doc.scans):
    lab = s.points["annotationLabel"]
    m = lab == 1
    xyz = np.column_stack([s.points["cartesianX"][m],
                           s.points["cartesianY"][m],
                           s.points["cartesianZ"][m]])
    xs.append(xyz)
    sid.append(np.full(len(xyz), i, np.int32))
xyz = np.vstack(xs)
sid = np.concatenate(sid)
print(f"穗点合计 {len(xyz):,}，分属 {len(doc.scans)} 个扫描\n")

# 手工实例参考
manual = 0
for s in doc.scans:
    ins = s.points.get("annotationInstance")
    lab = s.points.get("annotationLabel")
    if ins is None:
        continue
    ids = {int(v) for v in np.unique(ins) if v and int(v) > 0}
    # 只统计穗实例
    for uid in ids:
        m = (ins == uid) & (lab == 1)
        if m.any():
            manual += 1
print(f"手工标注的穗实例数（参考）：{manual}\n")

print(f"{'voxel':>6} {'link':>4} {'min':>5} | {'簇数':>5} {'覆盖点':>8} {'丢弃':>7} "
      f"{'中位':>5} {'最大':>6} {'耗时':>7}")
for voxel in (0.8, 1.2, 1.6, 2.0, 2.5):
    for link in (1, 2):
        for minp in (15, 30):
            t0 = time.perf_counter()
            r = segment_points(xyz, SegmentParams(voxel_cm=voxel, link=link, min_points=minp))
            dt = (time.perf_counter() - t0) * 1000
            med = int(np.median(r.sizes)) if r.n_clusters else 0
            mx = int(r.sizes.max()) if r.n_clusters else 0
            print(f"{voxel:>6.1f} {link:>4} {minp:>5} | {r.n_clusters:>5} "
                  f"{r.n_assigned:>8,} {r.n_rejected:>7,} {med:>5} {mx:>6} {dt:>6.0f}ms")

print("\n按扫描分别分割（voxel=1.2, link=1, min=25）：")
for i, s in enumerate(doc.scans):
    lab = s.points["annotationLabel"]
    m = lab == 1
    if not m.any():
        continue
    xyz_s = np.column_stack([s.points["cartesianX"][m],
                             s.points["cartesianY"][m],
                             s.points["cartesianZ"][m]])
    r = segment_points(xyz_s, SegmentParams(1.2, 1, 25))
    print(f"  扫描 {i + 1}: {int(m.sum()):>6,} 穗点 → {r.n_clusters:>3} 个穗 "
          f"（{r.summary()}）")
