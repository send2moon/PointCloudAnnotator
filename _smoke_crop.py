"""裁剪 + 文件夹队列的离屏测试。"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

APP = QApplication(sys.argv[:1])

from e57io import E57Document
from annotation import AnnotationSet
from cropping import CropBox, apply_masks, compound_mask, scan_bounds
from view3d import MODE_CROP
from main_window import MainWindow

SRC = r"D:\Data\PC\ResearchPlots\WS"
files = []
for root, _d, names in os.walk(SRC):
    for nm in sorted(names):
        if nm.lower().endswith(".e57"):
            files.append(os.path.join(root, nm))
files.sort()

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


# ---------------------------------------------------- 1. 纯数据层裁剪
print("1) 裁剪纯逻辑")
doc = E57Document.read(files[0])
n0 = doc.total_points
lo, hi = scan_bounds(doc.scans)
print(f"   原始 {n0:,} 点，包围盒 X[{lo[0]:.2f},{hi[0]:.2f}] "
      f"Y[{lo[1]:.2f},{hi[1]:.2f}] Z[{lo[2]:.2f},{hi[2]:.2f}]")

half = CropBox(xmin=float(lo[0]), xmax=float((lo[0] + hi[0]) / 2),
               ymin=float(lo[1]), ymax=float(hi[1]),
               zmin=float(lo[2]), zmax=float(hi[2]))
masks = compound_mask(doc, half, invert=False)
expect = 0
for s, m in zip(doc.scans, masks):
    if m is not None:
        # compound_mask 内部按 float64 判边界，这里保持同一精度，
        # 否则 float32 与 float64 在阈值上的 1ULP 差异会多数出几个点
        x = np.asarray(s.points["cartesianX"], np.float64)
        expect += int((x <= half.xmax).sum())
b, a = apply_masks(doc, masks)
print(f"   保留 X<= {half.xmax:.2f}：{b:,} -> {a:,}（预期 {expect:,}）")
check("裁剪点数正确", a == expect, f"{a} vs {expect}")
check("点数确实减少", a < b)
xs = np.concatenate([np.asarray(s.points["cartesianX"]) for s in doc.scans])
check("保留点都在框内", float(xs.max()) <= half.xmax + 1e-6, f"max={xs.max():.4f}")

# 字段完整性
s0 = doc.scans[0]
need = ("cartesianX", "cartesianY", "cartesianZ", "intensity",
        "colorRed", "colorGreen", "colorBlue")
check("裁剪后字段齐全", all(k in s0.points for k in need))
check("各列等长", len({len(s.points[k]) for k in need}) == 1)

# ---------------------------------------------------- 2. 反向裁剪
print("\n2) 删除框内")
doc2 = E57Document.read(files[0])
inner = CropBox(xmin=float((lo[0] + hi[0]) / 2), xmax=float(hi[0]),
                ymin=float((lo[1] + hi[1]) / 2), ymax=float(hi[1]),
                zmin=float(lo[2]), zmax=float(hi[2]))
m2 = compound_mask(doc2, inner, invert=True)
b2, a2 = apply_masks(doc2, m2)
print(f"   删除上方四分之一：{b2:,} -> {a2:,}")
check("反向裁剪点数减少", a2 < b2 and a2 > 0)

# ---------------------------------------------------- 3. 标注随裁剪同步
print("\n3) 标注同步")
doc3 = E57Document.read(files[0])
ann = AnnotationSet.from_scans(doc3.scans)
lab = ann.labels[0]
lab[: len(lab) // 2] = 1          # 前一半标为穗
ins = ann.instances[0]
ins[: len(ins) // 2] = 1
ann.instance_info[1] = (1, 0)
ann.next_instance = 2
before_marked = ann.total_marked()

bbox = scan_bounds(doc3.scans)
m3 = compound_mask(doc3, CropBox.from_bounds(bbox[0], bbox[1]), invert=False)
apply_masks(doc3, m3)
ann.rebuild(m3, [s.count for s in doc3.scans])
check("重建后标签长度一致",
      all(len(ann.labels[i]) == doc3.scans[i].count for i in range(len(doc3.scans))))
check("标注未越界", ann.total_marked() == before_marked,
      f"{ann.total_marked():,} vs {before_marked:,}")
check("实例表存活", 1 in ann.instance_info)

# ---------------------------------------------------- 4. 窗口裁剪 + 撤销
print("\n4) 窗口裁剪与撤销")
win = MainWindow()
win.resize(1400, 900)
win.show()
APP.processEvents()
win.load_file(files[0])
APP.processEvents()
n_before = win.doc.total_points
print(f"   载入 {n_before:,} 点")

b = scan_bounds(win.doc.scans)
box = CropBox(xmin=float(b[0][0]), xmax=float((b[0][0] + b[1][0]) / 2),
              ymin=float(b[0][1]), ymax=float(b[1][1]),
              zmin=float(b[0][2]), zmax=float(b[1][2]))
win.apply_crop(box, invert=False)
APP.processEvents()
n_after = win.doc.total_points
print(f"   裁剪后 {n_after:,} 点")
check("窗口裁剪生效", n_after < n_before)
check("列表已刷新", win.lst.count() > 0)

win.undo()
APP.processEvents()
n_undo = win.doc.total_points
print(f"   撤销后 {n_undo:,} 点")
check("撤销恢复点数", n_undo == n_before, f"{n_undo} vs {n_before}")

# ---------------------------------------------------- 5. 屏幕框选 -> 世界盒
print("\n5) 屏幕框选反算")
w, h = win.view.width(), win.view.height()
sel = win.view.renderer.world_bounds_of_screen_rect(
    w * 0.3, h * 0.3, w * 0.7, h * 0.7, w, h)
check("反算出包围盒", sel is not None)
if sel:
    print(f"   框选盒 X[{sel[0][0]:.2f},{sel[1][0]:.2f}] "
          f"Y[{sel[0][1]:.2f},{sel[1][1]:.2f}] Z[{sel[0][2]:.2f},{sel[1][2]:.2f}]")
    check("盒有效", bool(np.all(sel[1] > sel[0])))

# ---------------------------------------------------- 6. 文件夹队列
print("\n6) 文件夹队列")
folder = os.path.dirname(files[0])
win2 = MainWindow()
win2.show()
APP.processEvents()
win2.open_folder.__self__  # 存在即通过
win2._queue = files[1:4]
win2._queue_src = files[:4]
win2.load_file(files[0])
APP.processEvents()
check("队列非空", len(win2._queue) == 3)
print("   状态栏:", win2.lbl_file.text())
nxt = win2._queue[0]
win2._dirty = False
win2.next_in_queue()
APP.processEvents()
check("切换到下一个", win2.path == nxt, f"{os.path.basename(win2.path or '')}")
check("队列减少", len(win2._queue) == 2)

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
