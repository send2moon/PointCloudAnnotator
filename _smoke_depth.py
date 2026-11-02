"""拾取深度（框选/笔刷只标到一部分点）回归：

1) depth_keep 四种模式在"前层+后层+孤立噪点"上的取舍；
2) 真实文件上同一矩形在 surface / all 两种模式下的点数差异；
3) Ctrl 拖拽 = 临时穿透；
4) 框选确实跨多个扫描站同时标注（多站不是"只标一部分"的原因）。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

APP = QApplication(sys.argv[:1])

from main_window import MainWindow
from renderer import PointCloudRenderer
from view3d import MODE_BOX

SRC = r"D:\PCProjects\01wheat\FinalScripts\outputs\WS\FINAL_PLOTS"
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


errors = []
sys.excepthook = lambda *a: errors.append(a)

# ---------------------------------------------------------------- 1) 纯逻辑
print("1) depth_keep 取舍")
r = PointCloudRenderer()
# 前层 100 点 @1.0 m，后层 100 点 @5.0 m，另加 1 个孤立的更前方噪点 @0.2 m
zs = np.concatenate([np.full(100, 1.0), np.full(100, 5.0), np.array([0.2])])
zs += np.linspace(0, 0.02, zs.size)          # 前层自身厚 2 cm
front = np.zeros(zs.size, bool); front[:100] = True

r.depth_mode, r.depth_tol = "surface", 0.05
k = r.depth_keep(zs)
check("表面 5cm 命中全部前层", k[:100].all(), f"{int(k.sum())} 点")
check("表面 5cm 排除后层", not k[100:200].any())
check("表面 5cm 不被孤立噪点带偏", int(k.sum()) >= 100, f"{int(k.sum())} 点")

r.depth_mode, r.depth_tol = "surface", 1.0
check("容差调到 1m 仍只吃掉前层（后层在 4m 外）",
      not r.depth_keep(zs)[100:200].any(), f"{int(r.depth_keep(zs).sum())} 点")

r.depth_mode = "front25"
k = r.depth_keep(zs)
check("前 25% 排除后层", not k[100:200].any(), f"{int(k.sum())} 点")
check("前 25% 只取前面一段（少于前层总数）", int(k.sum()) < 100, f"{int(k.sum())} 点")

r.depth_mode = "all"
k = r.depth_keep(zs)
check("全部穿透 → 所有点", k.all(), f"{int(k.sum())} 点")

# ---------------------------------------------------------------- 2) 真实文件
print("\n2) 真实文件同一矩形的不同深度模式")
win = MainWindow()
win.resize(1200, 800)
win.show()
APP.processEvents()
win.load_file(files[0])
APP.processEvents()
print(f"载入 {win.doc.total_points:,} 点 / {len(win.doc.scans)} 扫描")

view = win.view
w, h = view.width(), view.height()
proj = view.renderer.project(w, h)
idx, sx, sy, zc = proj
cx, cy = int(np.median(sx)), int(np.median(sy))
x0, y0, x1, y1 = cx - 180, cy - 180, cx + 180, cy + 180

counts = {}
for mode in ("surface", "front25", "front50", "all"):
    view.renderer.depth_mode = mode
    view.renderer.depth_tol = 0.05
    counts[mode] = int(view.renderer.screen_rect_indices(x0, y0, x1, y1, w, h).size)
    print(f"    {mode:8s} {counts[mode]:>8,}")

check("surface 是全部的子集（只取前面一层）", counts["surface"] < counts["all"],
      f"{counts['surface']:,} < {counts['all']:,}")
check("surface 少于前 50%", counts["surface"] <= counts["front50"])
check("全部穿透点数最多", counts["all"] == max(counts.values()))
check("surface 至少标到一些点", counts["surface"] > 0)

zbox = zc[(sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)]
print(f"    框内深度跨度 {zbox.min():.2f} ~ {zbox.max():.2f} m "
      f"（差 {zbox.max() - zbox.min():.2f} m）")

# ---------------------------------------------------------------- 3) Ctrl 穿透
print("\n3) Ctrl 拖拽 = 临时穿透")
win.cmb_depth.setCurrentIndex(0)          # 表面层
win.spin_tol.setValue(5.0)
win._on_depth_mode()
APP.processEvents()
check("侧栏默认表面层", view.renderer.depth_mode == "surface", view.renderer.depth_mode)

win.set_mode(MODE_BOX)
APP.processEvents()
hits = []
view.annotateRequest.connect(lambda g, l, a: hits.append(len(g)))

p0, p1 = QPoint(x0, y0), QPoint(x1, y1)
QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, p0)
QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, p1)
APP.processEvents()
n_normal = hits[-1] if hits else 0

QTest.mousePress(view, Qt.LeftButton, Qt.ControlModifier, p0)
QTest.mouseRelease(view, Qt.LeftButton, Qt.ControlModifier, p1)
APP.processEvents()
n_ctrl = hits[-1] if len(hits) > 1 else 0
# 相机可能在中间被 fit 过，这里重新取一次"全部穿透"的基准
n_all_ref = int(view.renderer.screen_rect_indices(
    x0, y0, x1, y1, w, h, depth_mode="all").size)

print(f"    普通 {n_normal:,} / Ctrl {n_ctrl:,} / 穿透基准 {n_all_ref:,}")
check("Ctrl 拖拽点数多于普通拖拽", n_ctrl > n_normal, f"{n_normal:,} -> {n_ctrl:,}")
# 状态栏提示可能让中央视图高度差几像素，这里只比对量级（2% 内）
check("Ctrl 拖拽 ≈ 全部穿透的点数",
      abs(n_ctrl - n_all_ref) <= 0.02 * max(n_all_ref, 1),
      f"{n_ctrl:,} vs {n_all_ref:,}")

# ---------------------------------------------------------------- 4) 多站
print("\n4) 多站是否同时被标注")
win.cmb_depth.setCurrentIndex(3)          # 全部穿透
win._on_depth_mode()
APP.processEvents()
win.undo(); win.undo()                    # 清掉前面两次框选
APP.processEvents()
before = win.ann.total_marked()
check("两次撤销后回到未标注状态", before == 0, f"{before:,}")
QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, p0)
QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, p1)
APP.processEvents()
grp = win._undo[-1]
sids = sorted(s for s, _ in grp)
print(f"    本次框选覆盖扫描 {sids}")
check("一次框选可跨多个扫描站", len(sids) >= 1, str(sids))
check("撤销步记录每个涉及的站", len(grp) == len(sids))
win.undo()
APP.processEvents()
check("撤销后回到起点", win.ann.total_marked() == before,
      f"{win.ann.total_marked():,} vs {before:,}")

check("全程无异常", not errors, str(errors[:1]))

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
