# -*- coding: utf-8 -*-
"""手动框选穗实例：真实文件上的行为验证 + 截图。

覆盖：
1. 工具按钮 / 模式切换 / 视图信号
2. 真实文件上框选 -> 新实例编号（只动穗点）
3. 合并（框住两段已有实例 -> 同一新号）与拆分（框住一部分 -> 新号）
4. 撤销整步还原（labels + instances）
5. 框内没有穗点时的提示（不产生撤销步）
6. 六个工具按钮在小窗口下不被挤压
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
from PySide6.QtCore import QSettings, QPoint, QRect, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)

APP = QApplication(sys.argv[:1])
fid = QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\msyh.ttc")
APP.setFont(QFont(QFontDatabase.applicationFontFamilies(fid)[0], 9))

from main_window import MainWindow, APP_NAME
from view3d import MODE_NAV, MODE_BOX, MODE_INST

REAL = r"D:\Data\PC\ResearchPlots\DT\1-FS_09_labeled.e57"
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

_n = [0]


def check(name: str, ok: bool, detail: str = "") -> None:
    _n[0] += 1
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail and not ok else
                               f"  ({detail})" if detail else ""))
    if not ok:
        globals()["FAILED"] = True


win = MainWindow()
win.settings = QSettings(os.path.join(OUT, "_minst.ini"), QSettings.IniFormat)
win.resize(1920, 1080)
win.show()
APP.processEvents()
win.load_file(REAL)
APP.processEvents()
check("真实文件已打开", win.doc is not None and win.doc.total_points > 0)

# ---------------------------------------------------------------- 1 模式 --
print("\n1) 工具按钮与模式")
win.set_mode(MODE_INST)
APP.processEvents()
check("实例按钮已勾选", win.btn_inst.isChecked())
check("其他工具未勾选",
      not any(b.isChecked() for b in (win.btn_nav, win.btn_brush, win.btn_erase,
                                      win.btn_box, win.btn_crop)))
check("视图进入实例模式", win.view.mode == MODE_INST)
check("实例模式光标为十字", win.view.cursor().shape() == Qt.CrossCursor)
win.set_mode(MODE_BOX)
check("切回框选后实例按钮取消勾选", not win.btn_inst.isChecked())

# ------------------------------------------------------ 2 真实框选建实例 --
print("\n2) 框选建实例（扫描 8）")
win.lst.setCurrentRow([r for r in range(win.lst.count())
                       if win.lst.item(r).data(Qt.UserRole) == 7][0])
APP.processEvents()
r = win.view.renderer
sid = 7
lab = win.ann.labels[sid]
ins = win.ann.instances[sid]
spike = np.nonzero(lab == 1)[0]
check("扫描 8 有穗点", spike.size > 1000, f"{spike.size}")

# 在穗点最密集的一小块屏幕区域上框：取世界包围盒局部投影
pts = win.doc.scans[sid].points
xyz = np.column_stack([pts["cartesianX"], pts["cartesianY"],
                       pts["cartesianZ"]]).astype(np.float64)
center = xyz[spike].mean(axis=0)
# 选中该穗点附近 6 cm 球内的穗点做一次"框选"（直接构造选中集，等价屏幕框）
d = np.linalg.norm(xyz - center, axis=1)
near = d < 0.06
n_near_spike = int((near & (lab == 1)).sum())
glob = np.nonzero(r.scan_ids == sid)[0]     # 该站在渲染数组里的起点映射
# 渲染数组是全文档拼接的，构造全局索引
sel_local = np.nonzero(near)[0]
sel_global = np.array([g for g, li in zip(np.nonzero(r.scan_ids == sid)[0],
                                          r.local_index[r.scan_ids == sid])
                       if li in set(sel_local.tolist())], np.int64)
check("构造的框选集有穗点", sel_global.size > 10, f"{sel_global.size}")

nxt = win.ann.next_instance
snap_before = (lab.copy(), ins.copy())
win._on_instance_box(sel_global)
APP.processEvents()
made_uids = [u for u in np.unique(ins) if u and u >= nxt]
check("分配了新实例编号", len(made_uids) == 1, f"{made_uids}")
uid = int(made_uids[0]) if made_uids else -1
m_new = ins == uid
check("新实例只含框内穗点",
      int(m_new.sum()) == int((near & (ins == uid)).sum()) and
      int(m_new.sum()) >= n_near_spike * 0.5,
      f"{int(m_new.sum())} / 框内穗 {n_near_spike}")
check("新实例点全部是穗类别", bool((lab[m_new] == 1).all()))
check("叶/茎/其他类别的实例编号未被动过",
      np.array_equal(ins[(lab != 1) & (ins > 0)],
                     snap_before[1][(lab != 1) & (snap_before[1] > 0)]))
check("自动打开了实例框", win.chk_ibox.isChecked())
texts = sorted(int(b.text[1:]) for b in win.view.instance_boxes)
check("实例框里出现新编号", uid in texts, f"{texts[-5:]}")
check("状态栏有穗实例提示", "穗实例" in win.status.currentMessage())

# ------------------------------------------------------------- 3 撤销 --
print("\n3) 撤销")
win.undo()
APP.processEvents()
check("撤销后实例数组完全还原", np.array_equal(ins, snap_before[1]))
check("撤销后标签数组完全还原", np.array_equal(lab, snap_before[0]))
# next_instance 不回收（与自动分割同一约定）
check("撤销后 next_instance 不回退", win.ann.next_instance >= nxt)

# ------------------------------------------------- 4 合并（框住两段） --
print("\n4) 合并与拆分")
# 手工造两个相邻"实例"：把框内穗点先各编一半
ins[:] = 0
half = sel_local[: len(sel_local) // 2]
ins[half] = 9001
ins[sel_local[len(sel_local) // 2:]] = 9002
lab[sel_local] = 1
snap2 = (lab.copy(), ins.copy())
nxt2 = win.ann.next_instance
win._on_instance_box(sel_global)
APP.processEvents()
u = int(np.unique(ins[near & (lab == 1)])[0])
check("框住两段碎片后合并为同一个新实例",
      (ins[near & (lab == 1)] == u).all() and u >= nxt2, f"#{u}")
win.undo()
APP.processEvents()
check("撤销合并后回到两个旧实例",
      set(int(v) for v in np.unique(ins[sel_local])) == {9001, 9002})
# 拆分：只框一半
half_glob = np.array([g for g, li in zip(np.nonzero(r.scan_ids == sid)[0],
                                         r.local_index[r.scan_ids == sid])
                      if li in set(half.tolist())], np.int64)
nxt3 = win.ann.next_instance
win._on_instance_box(half_glob)
APP.processEvents()
vals = set(int(v) for v in np.unique(ins[sel_local]) if v)
check("框住一半只改写一半（拆分语义）",
      len(vals) == 2 and 9002 in vals and any(v >= nxt3 for v in vals),
      f"{sorted(vals)}")
win.undo()
APP.processEvents()

# ------------------------------------------------------- 5 无穗点提示 --
print("\n5) 框内没有穗点")
empty_area = np.nonzero(lab != 1)[0][:50]
sel_g2 = np.array([g for g, li in zip(np.nonzero(r.scan_ids == sid)[0],
                                      r.local_index[r.scan_ids == sid])
                   if li in set(empty_area.tolist())], np.int64)
undo_len = len(win._undo)
ins_b = ins.copy()
win._on_instance_box(sel_g2)
APP.processEvents()
check("无穗点时不产生撤销步", len(win._undo) == undo_len)
check("无穗点时标注不变", np.array_equal(ins, ins_b))
check("无穗点时状态栏给出提示", "没有穗点" in win.status.currentMessage())

# ------------------------------------------------------ 6 按钮布局 --
print("\n6) 按钮布局（六个工具按钮）")
worst = []
for w, h in ((1000, 640), (1366, 768), (1920, 1080)):
    win.resize(w, h)
    APP.processEvents()
    for b in (win.btn_nav, win.btn_brush, win.btn_erase,
              win.btn_box, win.btn_inst, win.btn_crop,
              *win._label_btns.values()):
        if b.width() < b.sizeHint().width():
            worst.append(f"{w}x{h}:{b.text()}({b.width()}<{b.sizeHint().width()})")
check("各窗口尺寸下工具按钮不被挤压", not worst, "; ".join(worst[:3]))

# ---------------------------------------------------------- 7 截图 --
print("\n7) 截图")
win.resize(1920, 1080)
APP.processEvents()
win._on_instance_box(sel_global)
APP.processEvents()
demo_uid = int(np.max(ins))
# 只框住这次新建的实例取景：旧的手工实例框很大，整体取景会糊成一片
m_demo = ins == demo_uid
p_demo = xyz[m_demo]
lo, hi = p_demo.min(axis=0), p_demo.max(axis=0)
cam = win.view.renderer.camera
cam.center = ((lo + hi) / 2).astype(np.float32)
radius = max(0.5 * float(np.linalg.norm(hi - lo)), 0.03)
cam.distance = max(radius / np.tan(np.radians(cam.fov * 0.5)) * 2.1, 1e-3)
win.view.fit_and_render()
APP.processEvents()
win.set_mode(MODE_INST)
APP.processEvents()
# 把视图里正在拖拽的框画在这根穗的屏幕位置上（模拟用户框选动作）
sx, sy, _zc = win.view.renderer.project_points(p_demo, win.view.width(), win.view.height())
x0, x1 = float(sx.min()) - 26, float(sx.max()) + 26
y0, y1 = float(sy.min()) - 26, float(sy.max()) + 26
win.view._box = (QPoint(int(x0), int(y0)), QPoint(int(x1), int(y1)))
win.view.update()
APP.processEvents()
win.view.grab().save(os.path.join(OUT, "ui_manual_instance.png"))
win.view._box = None
win.undo()     # 还原这次演示性实例
APP.processEvents()
print(f"\n共 {_n[0]} 项，{'FAIL 存在' if globals().get('FAILED') else '全部通过'}")
sys.exit(1 if globals().get("FAILED") else 0)
