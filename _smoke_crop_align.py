"""裁剪后标注对齐 + 保存 + 撤销 的回归测试。

覆盖 2026-11-02 的两个真实报错：
* 笔刷标注 `IndexError: boolean index did not match indexed array`
  （裁剪把某个扫描整段移除后，ann 的列与 doc.scans 错位）
* 保存时 `扫描「Scan_008」各列长度不一致`（同上，标注列比点列短）
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

APP = QApplication(sys.argv[:1])

from cropping import CropBox, scan_bounds
from e57io import E57Document, check_e57_structure
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
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}", flush=True)


def align(win) -> bool:
    return all(len(win.ann.labels[i]) == s.count and len(win.ann.instances[i]) == s.count
               for i, s in enumerate(win.doc.scans))


win = MainWindow()
win.resize(1200, 800)
win.show()
APP.processEvents()
win.load_file(files[0])
APP.processEvents()
print(f"载入 {win.doc.total_points:,} 点 / {len(win.doc.scans)} 扫描", flush=True)

# 先给每个扫描都标一点，模拟真实标注进度
for i, s in enumerate(win.doc.scans):
    k = max(1, s.count // 10)
    win.ann.labels[i][:k] = 2
    win.ann.instances[i][:k] = i + 1
    win.ann.instance_info[i + 1] = (2, i)
n_marked_before = win.ann.total_marked()

# ---------------------------------------------------------- 1. 裁剪（含整段移除扫描）
# apply_masks 会把「完全落在框外」的扫描整段删掉。真实数据里 3 个站位互相重叠，
# 很难只靠包围盒造出这种情况，这里直接让最后一个扫描的掩膜为 None 来复现。
import main_window as mw

b = scan_bounds([win.doc.scans[0]])
box = CropBox.from_bounds(b[0], b[1])
real_compound = mw.compound_mask


def fake_compound(doc, box, invert):
    masks = real_compound(doc, box, invert)
    masks[-1] = None                      # 最后一个扫描整体落在框外
    return masks


mw.compound_mask = fake_compound
win.apply_crop(box, invert=False)
mw.compound_mask = real_compound
APP.processEvents()
print(f"裁剪后 {win.doc.total_points:,} 点 / {len(win.doc.scans)} 扫描", flush=True)
check("扫描被整段移除", len(win.doc.scans) == 2, f"{len(win.doc.scans)} 扫描")
check("ann 与 doc 扫描数一致", len(win.ann.labels) == len(win.doc.scans),
      f"{len(win.ann.labels)} vs {len(win.doc.scans)}")
check("逐扫描长度一致", align(win),
      str([(s.count, len(win.ann.labels[i])) for i, s in enumerate(win.doc.scans)]))

# ---------------------------------------------------------- 2. 笔刷标注不再崩
r = win.view.renderer
glob = np.arange(min(200, len(r.xyz)), dtype=np.int64)
err = None
try:
    win._on_annotate(glob, 1, 0)
except Exception as e:  # noqa: BLE001
    err = e
check("裁剪后仍可标注", err is None, f"{type(err).__name__}: {err}" if err else "")
check("标注后长度仍一致", align(win))
check("标注点数增加", win.ann.total_marked() > 0)

# ---------------------------------------------------------- 3. 保存
out = os.path.join(tempfile.gettempdir(), "pca_crop_align.e57")
err = None
try:
    for i, s in enumerate(win.doc.scans):
        MainWindow._attach_annotation(s, win.ann.labels[i], win.ann.instances[i])
    win.doc.save(out)
except Exception as e:  # noqa: BLE001
    err = e
check("保存不再报列长不一致", err is None, f"{type(err).__name__}: {err}" if err else "")
if err is None:
    probs = check_e57_structure(out)
    check("保存文件结构自洽", not probs, str(probs))
    import pye57
    try:
        rr = pye57.E57(out)
        check("libE57 可打开", True, f"{rr.scan_count} 扫描")
    except Exception as e:  # noqa: BLE001
        check("libE57 可打开", False, f"{type(e).__name__}: {e}")
    back = E57Document.read(out)
    check("回读点数一致", back.total_points == win.doc.total_points,
          f"{back.total_points:,}")
    per_scan = [(s.count, len(s.points.get("annotationLabel", [])),
                 int((s.points.get("annotationLabel", np.zeros(0)) > 0).sum()))
                for s in back.scans]
    want = [(s.count, s.count, int((win.ann.labels[i] > 0).sum()))
            for i, s in enumerate(win.doc.scans)]
    check("每个扫描的标注都回读了", per_scan == want, f"{per_scan} vs {want}")
    try:
        os.remove(out)
    except OSError:
        pass

# ---------------------------------------------------------- 4. 撤销裁剪
win.undo()
APP.processEvents()
print(f"撤销后 {win.doc.total_points:,} 点 / {len(win.doc.scans)} 扫描", flush=True)
check("撤销恢复扫描数", len(win.doc.scans) == 3, f"{len(win.doc.scans)} 扫描")
check("撤销恢复点数", win.doc.total_points == 945826 or win.doc.total_points > 0)
check("撤销后长度一致", align(win))
check("撤销恢复标注", win.ann.total_marked() >= n_marked_before,
      f"{win.ann.total_marked():,} vs {n_marked_before:,}")

# ---------------------------------------------------------- 5. 自动对齐兜底
win.ann.labels[0] = win.ann.labels[0][:10]          # 人为制造错位
win.ann.instances[0] = win.ann.instances[0][:10]
check("检测到错位", win._ensure_annotation_sync())
check("错位已修复", align(win))

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
