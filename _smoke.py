"""离屏冒烟测试：加载 -> 渲染 -> 标注 -> 保存 -> 重新读取校验。

在无显示环境下用 offscreen 平台运行，输出截图与断言结果。
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

APP = QApplication(sys.argv[:1])

from e57io import E57Document
from annotation import AnnotationSet
from renderer import PointCloudRenderer
from view3d import PointCloudView, MODE_BRUSH, MODE_BOX
from main_window import MainWindow

SRC = r"D:\PCProjects\01wheat\FinalScripts\outputs\WS\FINAL_PLOTS"
if not os.path.isdir(SRC):
    SRC = r"D:\PCProjects\01wheat\FinalScripts\outputs\PLOTS\FINAL"
files = []
for root, _dirs, names in os.walk(SRC):
    for nm in sorted(names):
        if nm.lower().endswith(".e57"):
            files.append(os.path.join(root, nm))
files.sort()
print(f"数据集 {SRC}: {len(files)} 个 e57")
path = files[0]
print("测试文件:", path, f"{os.path.getsize(path)/1e6:.1f} MB")

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


# ---------------------------------------------------------------- 1. 读取
t0 = time.perf_counter()
doc = E57Document.read(path)
print(f"\n1) 读取 {time.perf_counter()-t0:.2f} s  ->  {len(doc.scans)} 扫描 / {doc.total_points:,} 点")
check("有扫描", len(doc.scans) > 0)
s0 = doc.scans[0]
print("   字段:", [f.name for f in s0.fields])
for k in ("cartesianX", "colorRed", "intensity"):
    check(f"含 {k}", k in s0.points)

# ---------------------------------------------------------------- 2. 渲染
r = PointCloudRenderer()
ann = AnnotationSet.from_scans(doc.scans)
t0 = time.perf_counter()
r.set_data(doc, ann)
print(f"\n2) 拼装 {time.perf_counter()-t0:.2f} s  ->  {len(r.xyz):,} 点")
t0 = time.perf_counter()
img = r.render(1400, 900)
print(f"   渲染 1400x900: {(time.perf_counter()-t0)*1000:.0f} ms")
check("图像非空", img is not None and not img.isNull())
img.save(os.path.join(os.path.dirname(path), "_smoke_rgb.png"))

r.color_mode = "height"; img2 = r.render(1400, 900)
r.color_mode = "label";  img3 = r.render(1400, 900)
r.color_mode = "rgb"

# 统计非背景像素，确认真的画出了东西
from PySide6.QtGui import QImage
def non_bg(im):
    im = im.convertToFormat(QImage.Format_RGB32)
    ptr = im.constBits()
    a = np.frombuffer(ptr, np.uint8, im.width() * im.height() * 4).reshape(-1, 4)
    bg = np.array([26, 29, 34], np.uint8)
    return int((np.abs(a[:, :3].astype(np.int16) - bg).sum(1) > 8).sum())

print(f"   非背景像素: {non_bg(img):,}")
check("画面有点", non_bg(img) > 5000)

# ---------------------------------------------------------------- 3. 投影 / 拾取
proj = r.project(1400, 900)
check("投影返回", proj is not None)
if proj:
    idx, sx, sy, zc = proj
    print(f"   投影可见点 {idx.size:,}  z 范围 {zc.min():.3f}~{zc.max():.3f}")
    hx = int(round(sx.mean())); hy = int(round(sy.mean()))
    p = r.pick(hx, hy, 1400, 900, 8)
    print(f"   拾取 ({hx},{hy}) -> 全局索引 {p}")

# ---------------------------------------------------------------- 4. 标注
print("\n4) 标注")
mask = np.zeros(s0.count, bool)
mask[: max(1, s0.count // 50)] = True           # 首个扫描 2% 的点
n = ann.apply(0, mask, 1)
check("写入穗", n == int(mask.sum()), f"{n} 点")
uid = ann.instances[0][mask][0]
check("实例编号有效", uid > 0, f"uid={uid}")
check("标签已落盘内存", (ann.labels[0][mask] == 1).all())

mask2 = np.zeros(s0.count, bool)
mask2[::7] = True
ann.apply(0, mask2, 2)
print("   统计:", {k: v for k, v in ann.label_counts().items() if v})
print("   实例数:", {k: v for k, v in ann.instance_counts().items() if v})

# 撤销
snap = ann.undoable_snapshot(0)
before = int((ann.labels[0] == 2).sum())
ann.restore(0, snap)
check("撤销快照可恢复", int((ann.labels[0] == 2).sum()) == before)

r.refresh_labels(ann)
r.color_mode = "label"
img4 = r.render(1400, 900)
img4.save(os.path.join(os.path.dirname(path), "_smoke_label.png"))
r.color_mode = "rgb"
print(f"   标注着色渲染完成，非背景像素 {non_bg(img4):,}")

# ---------------------------------------------------------------- 5. 保存
out = os.path.join(r"D:\PCProjects\01wheat\PointCloudAnnotator", "_smoke_out.e57")
MainWindow._attach_annotation(s0, ann.labels[0], ann.instances[0])
t0 = time.perf_counter()
doc.save(out)
print(f"\n5) 保存 {time.perf_counter()-t0:.2f} s  -> {os.path.getsize(out)/1e6:.1f} MB")

# ---------------------------------------------------------------- 6. 回读
t0 = time.perf_counter()
doc2 = E57Document.read(out)
print(f"6) 回读 {time.perf_counter()-t0:.2f} s  -> {doc2.total_points:,} 点")
allok = True
for i, (a, b) in enumerate(zip(doc.scans, doc2.scans)):
    if a.count != b.count:
        print(f"   扫描 {i} 点数不一致 {a.count} vs {b.count}"); allok = False; continue
    for k in a.points:
        if k not in b.points:
            print(f"   扫描 {i} 缺字段 {k}"); allok = False; continue
        if not np.array_equal(np.asarray(a.points[k]), np.asarray(b.points[k])):
            print(f"   扫描 {i} 字段 {k} 不一致"); allok = False
check("全字段逐位一致", allok)

ann2 = AnnotationSet.from_scans(doc2.scans)
check("标注恢复-标签", np.array_equal(ann2.labels[0], ann.labels[0]),
      f"(穗 {(ann2.labels[0]==1).sum():,} 点)")
check("标注恢复-实例", np.array_equal(ann2.instances[0], ann.instances[0]))
check("实例表重建", ann2.next_instance == ann.next_instance,
      f"next={ann2.next_instance}")

# ---------------------------------------------------------------- 7. 窗口
print("\n7) 主窗口")
win = MainWindow()
win.resize(1400, 900)
win.show()
APP.processEvents()
win.load_file(path)
APP.processEvents()
check("窗口载入", win.doc is not None)
print("   列表项:", win.lst.count(), "| 统计:", win.lbl_stat.text().replace("\n", " / "))
shot = win.grab()
shot.save(r"D:\PCProjects\01wheat\PointCloudAnnotator\_smoke_window.png")
print(f"   截图 {shot.width()}x{shot.height()}")

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
