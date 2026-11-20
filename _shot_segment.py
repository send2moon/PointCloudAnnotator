"""出图：新入口截图 + 穗分割前后对比（供人工判断效果）。"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)

APP = QApplication(sys.argv[:1])

# offscreen 平台默认字体不带中文字形，显式装一个系统字体，否则截图全是方块
from PySide6.QtGui import QFont, QFontDatabase

for f in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
          r"C:\Windows\Fonts\Deng.ttf"):
    if os.path.isfile(f):
        fid = QFontDatabase.addApplicationFont(f)
        fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
        if fams:
            APP.setFont(QFont(fams[0], 9))
            print("使用字体:", fams[0])
            break

from instance_seg import SegmentParams, segment_points
from main_window import MainWindow
from segment_dialog import SegmentDialog

REAL = r"D:\Data\PC\ResearchPlots\DT\1-FS_09_labeled.e57"
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

win = MainWindow()
win.settings = QSettings(os.path.join(OUT, "_shot.ini"), QSettings.IniFormat)
win.resize(1560, 980)
win.show()
APP.processEvents()
win.load_file(REAL)
APP.processEvents()
win.cmb_color.setCurrentIndex(3)
APP.processEvents()

# 1) 主界面（侧栏新增「穗实例分割」）
win.grab().save(os.path.join(OUT, "ui_segment_entry.png"))
print("主界面截图 ->", os.path.join(OUT, "ui_segment_entry.png"))

# 2) 分割对话框（含试算结果）
dlg = SegmentDialog(len(win.doc.scans), win._current_scan(),
                    win._seg_params(), probe=win._segment_probe,
                    default_overwrite=(win._unassigned_spikes() == 0), parent=win)
dlg.resize(520, 560)
dlg.show()
APP.processEvents()
dlg._do_probe()
APP.processEvents()
dlg.grab().save(os.path.join(OUT, "ui_segment_dialog.png"))
print("对话框截图 ->", os.path.join(OUT, "ui_segment_dialog.png"))
print("试算文本：")
print(dlg.lbl_probe.text())
dlg.close()

# 3) 分割前后对比（选穗点最多的扫描）
si = max(range(len(win.doc.scans)),
         key=lambda i: int((win.ann.labels[i] == 1).sum()))
s = win.doc.scans[si]
p = s.points
m = win.ann.labels[si] == 1
xyz = np.column_stack([p["cartesianX"][m], p["cartesianY"][m], p["cartesianZ"][m]])
manual = win.ann.instances[si][m].astype(np.int32)
r = segment_points(xyz, SegmentParams(0.3, 1, 10))
print(f"\n扫描 {si+1}: {m.sum():,} 穗点 → 自动 {r.n_clusters} 个穗；"
      f"手工 {len([v for v in np.unique(manual) if v])} 个实例")

cmap = plt.get_cmap("tab20")
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, lab, title in ((axes[0], manual, f"Manual instance = {len([v for v in np.unique(manual) if v])}"),
                       (axes[1], r.cluster, f"Auto segmentation = {r.n_clusters} spikes")):
    for k, val in enumerate(np.unique(lab)):
        q = xyz[lab == val]
        c = (0.55, 0.55, 0.55) if val == 0 else cmap((k * 7) % 20)
        ax.scatter(q[:, 0] * 100, q[:, 2] * 100, s=2.0, c=[c], linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Z (cm)")
    ax.set_aspect("equal", adjustable="datalim")
fig.suptitle(f"Scan #{si+1}: {int(m.sum()):,} spike points  "
             f"(voxel 0.3 cm, min 10 pts)", fontsize=12)
fig.tight_layout()
out = os.path.join(OUT, "segment_result.png")
fig.savefig(out, dpi=130)
print("对比图 ->", out)
