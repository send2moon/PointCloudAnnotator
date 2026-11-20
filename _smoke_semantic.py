# -*- coding: utf-8 -*-
"""语义标注 / 隐藏点云 / 修改已有标注 的回归测试。"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

APP = QApplication(sys.argv[:1])

MSGS = []
QMessageBox.information = staticmethod(
    lambda *a, **k: (MSGS.append(str(a[2]) if len(a) > 2 else ""), QMessageBox.Ok)[1])
QMessageBox.warning = staticmethod(
    lambda *a, **k: (MSGS.append(str(a[2]) if len(a) > 2 else ""), QMessageBox.Ok)[1])
QMessageBox.critical = staticmethod(
    lambda *a, **k: (MSGS.append(str(a[2]) if len(a) > 2 else ""), QMessageBox.Ok)[1])
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

from annotation import LABEL_NAMES
from label_edit_dialog import LabelEditDialog
from main_window import MainWindow

REAL = r"D:\Data\PC\ResearchPlots\DT\1-FS_09_labeled.e57"
ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


errors = []
sys.excepthook = lambda *a: errors.append(a)

tmpdir = tempfile.mkdtemp(prefix="sem_smoke_")
win = MainWindow()
win.settings = QSettings(os.path.join(tmpdir, "s.ini"), QSettings.IniFormat)
win.settings.setValue("depth_mode", "all")
win.resize(1400, 900)
win.show()
APP.processEvents()
win.load_file(REAL)
APP.processEvents()

r = win.view.renderer
n_total = len(r.xyz)
sid0 = win._focused_scan()
check("文件已打开", win.doc is not None and n_total > 0, f"{n_total:,} 点 / 扫描 #{sid0 + 1}")

g_of = {int(li): g for g, (s, li) in enumerate(zip(r.scan_ids, r.local_index))
        if int(s) == sid0}


def free_indices(count: int):
    """取若干个未标注点的全局索引（限定在聚焦扫描，且在可见范围内）。"""
    lab = win.ann.labels[sid0]
    loc = np.nonzero(lab == 0)[0]
    out = [g_of[int(li)] for li in loc if int(li) in g_of]
    return np.asarray(out[:count], np.int64)


# ------------------------------------------------------- 1) 语义 vs 实例
print("\n1) 语义标注与实例标注分开")
check("默认是实例标注", win.ann_mode == "instance" and win.btn_instance.isChecked())
win.set_ann_mode("semantic")
APP.processEvents()
check("切到语义标注", win.ann_mode == "semantic" and win.btn_semantic.isChecked())
check("两种方式互斥", not win.btn_instance.isChecked())
check("标注方式写入设置", win.settings.value("ann_mode") == "semantic")

nxt0 = win.ann.next_instance
info0 = len(win.ann.instance_info)
sel = free_indices(80)
check("取得未标注点", sel.size >= 40, f"{sel.size}")
win._on_annotate(sel, 1, 0)
APP.processEvents()
lab0 = win.ann.labels[sid0]
ins0 = win.ann.instances[sid0]
loc = r.local_index[sel[r.scan_ids[sel] == sid0]]
check("语义标注写入了类别", bool((lab0[loc] == 1).all()), f"{loc.size} 点")
check("语义标注不产生实例编号", bool((ins0[loc] == 0).all()))
check("语义标注不推进 next_instance", win.ann.next_instance == nxt0,
      f"{win.ann.next_instance} vs {nxt0}")
check("语义标注不新增实例登记", len(win.ann.instance_info) == info0)
check("列表出现「未分实例」一行",
      any("未分实例" in win.lst.item(i).text() for i in range(win.lst.count())))
check("状态栏说明是语义标注", "语义" in win.status.currentMessage())

# 实例标注：模式切回后应生成编号
win.set_ann_mode("instance")
APP.processEvents()
sel2 = free_indices(60)
nxt1 = win.ann.next_instance
win._on_annotate(sel2, 1, 0)
APP.processEvents()
loc2 = r.local_index[sel2[r.scan_ids[sel2] == sid0]]
uids = {int(v) for v in np.unique(win.ann.instances[sid0][loc2]) if v}
check("实例标注生成了新编号", len(uids) == 1 and min(uids) >= nxt1, f"{sorted(uids)}")
check("实例登记已写入", all(u in win.ann.instance_info for u in uids))

# 语义标注覆盖已有实例 → 编号被清掉
win.set_ann_mode("semantic")
APP.processEvents()
glob_inst = np.array([g_of[int(li)] for li in np.nonzero(np.isin(
    win.ann.instances[sid0], list(uids)))[0] if int(li) in g_of], np.int64)
win._on_annotate(glob_inst, 1, 0)
APP.processEvents()
check("语义覆盖后实例编号被清空",
      bool((win.ann.instances[sid0][np.isin(win.ann.instances[sid0], list(uids))] == 0).all()))
check("被覆盖的实例已注销登记", all(u not in win.ann.instance_info for u in uids))

# 撤销要能还原
win.undo()
APP.processEvents()
check("撤销后实例编号回来",
      all(u in set(int(v) for v in np.unique(win.ann.instances[sid0])) for u in uids))

# ------------------------------------------------------------- 2) 隐藏
print("\n2) 隐藏点云")
win.chk_hide_marked.setChecked(False)
win.chk_hide_cur.setChecked(False)
win._apply_hidden()
APP.processEvents()
check("默认全部可见", r.n_visible() == n_total and not r.hidden.any())

lab_all = r.labels.copy()
n_marked = int((lab_all > 0).sum())
win.chk_hide_marked.setChecked(True)
APP.processEvents()
check("隐藏已标注后可见数正确", r.n_visible() == n_total - n_marked,
      f"{r.n_visible():,} / {n_total:,}（已标注 {n_marked:,}）")
check("可见点里没有已标注点", int((r.visible_mask() & (r.labels > 0)).sum()) == 0)
proj = r.project(win.view.width(), win.view.height())
if proj is not None:
    check("投影/拾取也排除隐藏点", int((r.labels[proj[0]] > 0).sum()) == 0,
          f"{len(proj[0]):,} 点参与投影")
vis_pts = r.xyz[r.visible_mask()]
lo, hi = r.bounds()
check("包围盒按可见点计算",
      np.allclose(lo, vis_pts.min(axis=0), atol=1e-6) and
      np.allclose(hi, vis_pts.max(axis=0), atol=1e-6))

# 隐藏状态下继续标注：新标的点立刻消失
win.set_ann_mode("semantic")
sel3 = free_indices(40)
sel3 = np.array([g for g in sel3 if not r.hidden[g]], np.int64)
before_vis = r.n_visible()
win._on_annotate(sel3, 2, 0)
APP.processEvents()
loc3 = r.local_index[sel3[r.scan_ids[sel3] == sid0]]
check("隐藏已标注时新标注的点立即隐藏",
      r.n_visible() == before_vis - int(np.unique(loc3).size),
      f"{r.n_visible()} vs {before_vis}")

# 隐藏当前类别
win.chk_hide_marked.setChecked(False)
win.chk_hide_cur.setChecked(True)
win.set_label(2)
APP.processEvents()
n2 = int((r.labels == 2).sum())
check("隐藏当前类别只隐藏该类", r.n_visible() == n_total - n2, f"{r.n_visible():,} / n2={n2:,}")
win.set_label(1)
APP.processEvents()
n1 = int((r.labels == 1).sum())
check("切换类别后隐藏对象跟着换", r.n_visible() == n_total - n1)
win.chk_hide_cur.setChecked(False)
win.chk_hide_marked.setChecked(False)
APP.processEvents()
check("取消隐藏后全部恢复", r.n_visible() == n_total)

# 单站视图重建后仍保持隐藏
win.chk_all_scans.setChecked(False)
APP.processEvents()
win.chk_hide_marked.setChecked(True)
APP.processEvents()
check("单站视图下隐藏生效",
      int((win.view.renderer.visible_mask() &
           (win.view.renderer.labels > 0)).sum()) == 0)
win.chk_all_scans.setChecked(True)
win.chk_hide_marked.setChecked(False)
APP.processEvents()
check("换回全部扫描后全部可见", win.view.renderer.n_visible() == len(win.view.renderer.xyz))

# --------------------------------------------------------- 3) 修已有标注
print("\n3) 修改已有标注")
labels_before = [l.copy() for l in win.ann.labels]
ins_before = [i.copy() for i in win.ann.instances]
next_before = win.ann.next_instance
ids = win.ann.instance_ids(sid0)
check("当前扫描有实例可改", len(ids) >= 2, f"{len(ids)}")
u1, u2 = ids[0], ids[1]
m1 = win.ann.instances[sid0] == u1
old_lb = win.ann.instance_label(sid0, u1)
new_lb = 2 if old_lb != 2 else 3
total = win.edit_labels("set_label", [("inst", sid0, u1)], new_lb)
APP.processEvents()
check("改类别：命中点数正确", total == int(m1.sum()), f"{total}")
check("改类别：类别已变、实例编号保留",
      bool((win.ann.labels[sid0][m1] == new_lb).all()) and
      bool((win.ann.instances[sid0][m1] == u1).all()))
check("改类别：instance_info 同步", win.ann.instance_info[u1][0] == new_lb)
win.undo()
APP.processEvents()
check("改类别可撤销", np.array_equal(win.ann.labels[sid0], labels_before[sid0]))

n_sem_before = sum(c for _lb, c in win.ann.label_groups(sid0))
total = win.edit_labels("to_semantic", [("inst", sid0, u2)])
APP.processEvents()
check("转语义：编号清空",
      u2 not in set(int(v) for v in np.unique(win.ann.instances[sid0])))
check("转语义：类别保留（点数不变）",
      int((win.ann.labels[sid0] > 0).sum()) == int((labels_before[sid0] > 0).sum()))
check("转语义后该实例注销", u2 not in win.ann.instance_info)
check("转语义后未分实例点数增加",
      sum(c for _lb, c in win.ann.label_groups(sid0)) >= n_sem_before + total,
      f"{sum(c for _lb, c in win.ann.label_groups(sid0))}")
win.undo()
APP.processEvents()
check("转语义可撤销", np.array_equal(win.ann.instances[sid0], ins_before[sid0]))

nxt = win.ann.next_instance
total = win.edit_labels("merge", [("inst", sid0, u1), ("inst", sid0, u2)])
APP.processEvents()
uni = {int(v) for v in np.unique(win.ann.instances[sid0]) if v}
check("合并：两段归到同一个新编号",
      u1 not in uni and u2 not in uni and any(u >= nxt for u in uni), f"{sorted(uni)[-3:]}")
check("合并后旧登记注销",
      u1 not in win.ann.instance_info and u2 not in win.ann.instance_info)
win.undo()
APP.processEvents()
check("合并可撤销", np.array_equal(win.ann.instances[sid0], ins_before[sid0]))

total = win.edit_labels("delete", [("inst", sid0, u1)])
APP.processEvents()
check("删除：类别与编号都清零",
      bool((win.ann.labels[sid0][m1] == 0).all()) and
      bool((win.ann.instances[sid0][m1] == 0).all()))
win.undo()
APP.processEvents()
check("删除可撤销", np.array_equal(win.ann.labels[sid0], labels_before[sid0]))

cnt2 = int((win.ann.labels[sid0] == 2).sum())
if cnt2:
    n_all2 = sum(int((l == 2).sum()) for l in win.ann.labels)
    total = win.edit_labels("clear_category", [], 2)
    APP.processEvents()
    check("按类别清除：全部扫描里该类别归零",
          all(int((l == 2).sum()) == 0 for l in win.ann.labels) and total == n_all2,
          f"{total:,}")
    win.undo()
    APP.processEvents()
    check("按类别清除可撤销",
          sum(int((l == 2).sum()) for l in win.ann.labels) == n_all2)

# 清空全部 → 撤销后编号不能回退到会撞号的位置
n_before = sum(int((l > 0).sum()) for l in win.ann.labels)
total = win.edit_labels("clear_all", [])
APP.processEvents()
check("清空全部标注", win.ann.total_marked() == 0 and total == n_before,
      f"{total:,} / {n_before:,}")
win.undo()
APP.processEvents()
max_uid = max(int(i.max()) for i in win.ann.instances if i.size)
check("撤销清空后 next_instance 大于在用最大编号", win.ann.next_instance > max_uid,
      f"next={win.ann.next_instance} max={max_uid}")
check("撤销清空后标注完整恢复",
      all(np.array_equal(a, b) for a, b in zip(win.ann.labels, labels_before)))

# 对话框：能列出实例与语义组，操作后自动刷新
dlg = LabelEditDialog(win, win)
APP.processEvents()
rows = [dlg.lst.item(i) for i in range(dlg.lst.count())]
inst_rows = [it for it in rows if isinstance(it.data(Qt.UserRole), tuple)]
check("对话框列出了可编辑行", len(inst_rows) > 0, f"{len(inst_rows)} 行")
if len(inst_rows) >= 2:
    # 多选前两行并合并（对话框内部走 edit_labels）
    for it in inst_rows[:2]:
        it.setSelected(True)
    APP.processEvents()
    check("对话框能统计选中项", dlg.lbl_sel.text().startswith("已选中"), dlg.lbl_sel.text())
    dlg._on_merge()
    APP.processEvents()
    check("对话框合并后列表重建", dlg.lst.count() >= len(rows) - 1)
    win.undo()
    APP.processEvents()
dlg.close()

# --------------------------------------------------------- 4) 存盘回读
print("\n4) 存盘读回")
u3 = win.ann.instance_ids(sid0)[0]
win.edit_labels("to_semantic", [("inst", sid0, u3)])
sem_loc = np.nonzero((win.ann.labels[sid0] > 0) & (win.ann.instances[sid0] == 0))[0]
sem_lab = win.ann.labels[sid0][sem_loc].copy()
out = os.path.join(tmpdir, "semantic_out.e57")
win._do_save(out)
win.load_file(out)
APP.processEvents()
check("读回文件点数一致", win.doc.total_points == n_total)
check("读回后语义点仍是「有类别、无编号」",
      bool((win.ann.instances[sid0][sem_loc] == 0).all()) and
      np.array_equal(win.ann.labels[sid0][sem_loc], sem_lab),
      f"{sem_loc.size:,} 点")

check("全程无异常", not errors, str(errors[:1]))
print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
