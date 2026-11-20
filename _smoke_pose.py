"""多站扫描 + pose 测试。

回答一个常被问到的问题：多站 e57 **不需要先合并**，软件按扫描逐个写出；
两种保存方式都对：
* 默认「应用位姿」——坐标已是文件坐标系，写出的 pose 为单位阵（等同 CloudCompare 的显示结果）
* 关闭「应用位姿」——保留各站局部坐标 + 原始 pose，第三方打开时自行套用
本测试验证两种方式几何等价、且位姿数值原样保留。
"""
from __future__ import annotations

import os
import struct
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from annotation import AnnotationSet
from cropping import scan_bounds
from e57io import E57Document, check_e57_structure

SRC = sys.argv[1] if len(sys.argv) > 1 else ""
if not SRC:
    for d, _s, names in os.walk(r"D:\Data\PC\ResearchPlots\DT"):
        for nm in sorted(names):
            if not nm.lower().endswith(".e57") or "labeled" in nm.lower():
                continue
            p = os.path.join(d, nm)
            if not check_e57_structure(p):   # 跳过旧版本写坏的样本
                SRC = p
                break
        if SRC:
            break

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}", flush=True)


print(f"样本 {os.path.basename(SRC)}", flush=True)

# ---------------------------------------------------------- 1. 烘焙坐标
baked = E57Document.read(SRC, apply_pose=True)
lo, hi = scan_bounds(baked.scans)
print(f"  烘焙后包围盒 {hi[0]-lo[0]:.2f} × {hi[1]-lo[1]:.2f} × {hi[2]-lo[2]:.2f} m "
      f"（{len(baked.scans)} 扫描）", flush=True)
check("多站已拼成整块", float(max(hi - lo)) < 50.0, f"max span {float(max(hi-lo)):.1f} m")
check("烘焙后 pose 归零", all(not s.has_pose() for s in baked.scans))

# ---------------------------------------------------------- 2. 保留原始位姿
local = E57Document.read(SRC, apply_pose=False)
llo, lhi = scan_bounds(local.scans)
posed = [s for s in local.scans if s.has_pose()]
print(f"  不烘焙：{len(posed)}/{len(local.scans)} 个扫描带非单位 pose，"
      f"局部包围盒 {lhi[0]-llo[0]:.2f} × {lhi[1]-llo[1]:.2f} × {lhi[2]-llo[2]:.2f} m",
      flush=True)
check("位姿被保留下来", len(posed) == len(local.scans) and len(posed) > 0)
check("局部坐标未烘焙（包围盒更大或不同）",
      not np.allclose(lhi - llo, hi - lo, atol=1e-3),
      f"{float(max(lhi-llo)):.2f} vs {float(max(hi-lo)):.2f} m")
check("点数与原文件一致", local.total_points == baked.total_points,
      f"{local.total_points:,}")

# ---------------------------------------------------------- 3. 存盘 + 回读
ann = AnnotationSet.from_scans(local.scans)
for i, s in enumerate(local.scans):
    k = max(1, s.count // 50)
    ann.labels[i][:k] = 1
    ann.instances[i][:k] = i + 1
    ann.instance_info[i + 1] = (1, i)
from main_window import MainWindow
for i, s in enumerate(local.scans):
    MainWindow._attach_annotation(s, ann.labels[i], ann.instances[i])

out = os.path.join(tempfile.gettempdir(), "pca_pose_keep.e57")
local.save(out)
check("结构体检通过", not check_e57_structure(out), str(check_e57_structure(out)))

blob = open(out, "rb").read()
phys, xlen = struct.unpack_from("<QQ", blob, 24)
lg = bytearray()
for k in range(len(blob) // 1024):
    lg += blob[k * 1024:k * 1024 + 1020]
st = (phys // 1024) * 1020 + (phys % 1024)
text = bytes(lg[st:st + xlen]).decode("utf-8", "replace")
qs = re.findall(r"<w type=\"Float\">([-\d.eE+]+)</w><x type=\"Float\">([-\d.eE+]+)</x>"
                r"<y type=\"Float\">([-\d.eE+]+)</y><z type=\"Float\">([-\d.eE+]+)</z>", text)
check("XML 里写回了原始位姿", len(qs) >= len(posed), f"{len(qs)} 个 pose")
if qs and posed:
    q0 = tuple(float(v) for v in qs[0])
    check("第一个扫描的四元数一致",
          np.allclose(q0, posed[0].rotation, atol=1e-6),
          f"{q0} vs {posed[0].rotation}")

# 第三方视角：按 pose 变换后应与烘焙版完全一致
back = E57Document.read(out, apply_pose=True)
blo, bhi = scan_bounds(back.scans)
check("回读后可拼回同一块", np.allclose(blo, lo, atol=2e-3) and np.allclose(bhi, hi, atol=2e-3),
      f"{blo} vs {lo}")
check("点数一致", back.total_points == baked.total_points, f"{back.total_points:,}")
per = [(s.count, int((s.points.get('annotationLabel', np.zeros(0)) > 0).sum()))
       for s in back.scans]
want = [(s.count, int((ann.labels[i] > 0).sum())) for i, s in enumerate(local.scans)]
check("各扫描标注都回读", per == want, f"{per[:3]} vs {want[:3]}")

try:
    os.remove(out)
except OSError:
    pass
print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
