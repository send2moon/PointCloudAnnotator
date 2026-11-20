"""穗实例分割回归：

1) 合成点云：分离正确、碎片过滤、体素过大时的粘连；
2) 并查集回退路径与 scipy 结果一致；
3) 真实文件：当前扫描分割、增量模式、覆盖模式、撤销、试算不改动标注；
4) 分割结果存盘再读回，实例编号不丢。
"""

from __future__ import annotations

import builtins
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

APP = QApplication(sys.argv[:1])

# offscreen 下真弹窗会阻塞，全部打桩（同时留存调用记录便于断言）
from PySide6.QtWidgets import QMessageBox

MSGS = []
QMessageBox.information = staticmethod(lambda *a, **k: (MSGS.append(str(a[2]) if len(a) > 2 else ""), QMessageBox.Ok)[1])
QMessageBox.warning = staticmethod(lambda *a, **k: (MSGS.append(str(a[2]) if len(a) > 2 else ""), QMessageBox.Ok)[1])
QMessageBox.critical = staticmethod(lambda *a, **k: (MSGS.append(str(a[2]) if len(a) > 2 else ""), QMessageBox.Ok)[1])
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)

from instance_seg import SegmentParams, segment_points
from main_window import MainWindow
from segment_dialog import SegmentDialog

REAL = r"D:\Data\PC\ResearchPlots\DT\1-FS_09_labeled.e57"

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


errors = []
sys.excepthook = lambda *a: errors.append(a)


def blob(center, n=5, pitch=0.001):
    """以 center 为中心生成 n³ 网格点（米）。"""
    g = (np.arange(n) - (n - 1) / 2) * pitch
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()]) + np.asarray(center)


# ------------------------------------------------------------ 1) 合成点云
print("1) 合成点云")
# 三团，团内 0.1 cm 点距、团心相距 3 cm
pts = np.vstack([blob((0, 0, 0)), blob((0.03, 0, 0)), blob((0, 0.03, 0))])
r = segment_points(pts, SegmentParams(voxel_cm=0.3, link=1, min_points=10))
check("三团分成三个实例", r.n_clusters == 3, f"{r.n_clusters} 个")
check("实例编号为 1..3", sorted(np.unique(r.cluster)) == [1, 2, 3],
      str(sorted(np.unique(r.cluster))))
check("每团点数相同（125）", list(r.sizes) == [125, 125, 125], str(list(r.sizes)))
check("没有点被判为碎片", r.n_rejected == 0, f"{r.n_rejected}")

# 加一个 3 点小团 → 被最小点数过滤
pts2 = np.vstack([pts, blob((0.06, 0, 0), n=1) * 0 + np.array([[0.06, 0, 0],
                                                               [0.0601, 0, 0],
                                                               [0.06, 0.0001, 0]])])
r2 = segment_points(pts2, SegmentParams(voxel_cm=0.3, link=1, min_points=10))
check("小团被过滤（仍是三个实例）", r2.n_clusters == 3, f"{r2.n_clusters} 个")
check("碎片点簇号为 0", (r2.cluster[-3:] == 0).all(), str(r2.cluster[-3:]))
check("碎片点数记入 n_rejected", r2.n_rejected == 3, f"{r2.n_rejected}")
check("连通块总数含碎片", r2.n_components == 4, f"{r2.n_components}")

r3 = segment_points(pts2, SegmentParams(voxel_cm=3.0, link=1, min_points=10))
check("体素放大到 3 cm 后粘连成一团", r3.n_clusters == 1, f"{r3.n_clusters} 个")

r4 = segment_points(np.zeros((0, 3)), SegmentParams())
check("空输入不崩", r4.n_clusters == 0 and r4.cluster.size == 0)

# 参数序列化
p = SegmentParams(0.42, 2, 17)
check("参数 to_dict/from_dict 往返一致",
      SegmentParams.from_dict(p.to_dict()) == p, str(p.to_dict()))

# ------------------------------------------------- 2) 并查集回退一致性
print("\n2) 无 scipy 时的并查集回退")
rng = np.random.default_rng(7)
rand_pts = rng.normal(size=(400, 3)) * 0.02
ref = segment_points(rand_pts, SegmentParams(0.4, 1, 3))

real_import = builtins.__import__


def blocked(name, *a, **k):
    if name == "scipy" or name.startswith("scipy."):
        raise ImportError("scipy blocked by smoke test")
    return real_import(name, *a, **k)


builtins.__import__ = blocked
try:
    import importlib
    import instance_seg as _is
    importlib.reload(_is)
    alt = _is.segment_points(rand_pts, SegmentParams(0.4, 1, 3))
finally:
    builtins.__import__ = real_import
    importlib.reload(_is)

check("回退路径簇数一致", alt.n_clusters == ref.n_clusters,
      f"{alt.n_clusters} vs {ref.n_clusters}")
check("回退路径每簇点数一致", list(alt.sizes) == list(ref.sizes))
check("回退路径逐点归属一致", np.array_equal(alt.cluster, ref.cluster))

# ------------------------------------------------------------ 3) 真实文件
print("\n3) 真实文件")
tmpdir = tempfile.mkdtemp(prefix="seg_smoke_")
ini = os.path.join(tmpdir, "settings.ini")
win = MainWindow()
win.settings = QSettings(ini, QSettings.IniFormat)   # 不污染用户设置
win.settings.setValue("depth_mode", "all")
win.settings.setValue("depth_tol_cm", 5.0)
win.resize(1200, 800)
win.show()
APP.processEvents()
win.load_file(REAL)
APP.processEvents()

n_spike = sum(int((l == 1).sum()) for l in win.ann.labels)
n_inst_before = len({int(v) for ins in win.ann.instances for v in np.unique(ins) if v})
print(f"   {win.doc.total_points:,} 点 / {len(win.doc.scans)} 扫描 / "
      f"{n_spike:,} 穗点 / {n_inst_before} 个已有实例")
check("文件里有穗点", n_spike > 0, f"{n_spike:,}")
check("默认参数来自缺省值", win._seg_params() == SegmentParams(0.30, 1, 10),
      str(win._seg_params().to_dict()))

# 3.1 增量模式：已有实例的点不动
sid = 0
lab_before = [l.copy() for l in win.ann.labels]
ins_before = [i.copy() for i in win.ann.instances]
nxt_before = win.ann.next_instance
# 造一个"有穗标签但没编号"的缺口：清掉当前扫描里最大的那个穗实例编号
ins0 = win.ann.instances[sid]
ids0 = [int(v) for v in np.unique(ins0[win.ann.labels[sid] == 1]) if v]
if ids0:
    big = max(ids0, key=lambda u: int((ins0 == u).sum()))
    hole = ins0 == big
    ins0[hole] = 0
    lab_before = [l.copy() for l in win.ann.labels]
    ins_before = [i.copy() for i in win.ann.instances]
n_unassigned = int(((win.ann.labels[sid] == 1) & (ins0 == 0)).sum())
added_inc = win.apply_segment(SegmentParams(0.3, 1, 10), "current", overwrite=False)
print(f"   增量模式：当前扫描还有 {n_unassigned:,} 个穗点没有实例 → 新增 {added_inc} 个")
assigned_after = int(((win.ann.labels[sid] == 1) & (win.ann.instances[sid] > 0)).sum())
check("增量模式生成了实例", added_inc >= 1, f"{added_inc}")
check("增量模式补上了原先没编号的穗点",
      assigned_after > int(((lab_before[sid] == 1) & (ins_before[sid] > 0)).sum()),
      f"{assigned_after:,} 个已编号")
keep = ins_before[sid] > 0
check("增量模式不动已有实例的编号",
      np.array_equal(win.ann.instances[sid][keep], ins_before[sid][keep]))
check("增量模式后 next_instance 只增不减", win.ann.next_instance >= nxt_before)
check("新增编号都登记在 instance_info",
      all(uid in win.ann.instance_info for uid in range(nxt_before, win.ann.next_instance)))

# 撤销回来
win.undo()
APP.processEvents()
check("撤销后实例数组完全还原",
      np.array_equal(win.ann.instances[sid], ins_before[sid]))
check("撤销后标签也还原", np.array_equal(win.ann.labels[sid], lab_before[sid]))

# 3.2 覆盖模式
added_all = win.apply_segment(SegmentParams(0.3, 1, 10), "current", overwrite=True)
spike_mask = win.ann.labels[sid] == 1
assigned = int((spike_mask & (win.ann.instances[sid] > 0)).sum())
print(f"   覆盖模式：新增 {added_all} 个实例，穗点 {int(spike_mask.sum()):,}，"
      f"其中已编号 {assigned:,}")
check("覆盖模式生成了实例", added_all > 0, f"{added_all}")
ins_new = np.asarray(win.ann.instances[sid])
new_ids = {int(v) for v in np.unique(ins_new[spike_mask]) if v}
old_spike_ids = {int(v) for v in
                 np.unique(np.asarray(ins_before[sid])[np.asarray(lab_before[sid]) == 1])
                 if v}
fresh = new_ids - old_spike_ids
check("新实例编号唯一且从旧 next 往后排",
      bool(fresh) and min(fresh) >= nxt_before,
      f"最小新编号 {min(fresh) if fresh else '-'}（旧 next={nxt_before}）")
# 覆盖模式下穗点的旧编号一律作废：碎片点归 0，不留旧号，否则执行结果会比预览多出框
check("覆盖模式后穗点不再残留旧编号", new_ids == fresh,
      f"残留 {sorted(new_ids - fresh)[:5]}")
check("被判为碎片的穗点不再带编号",
      int((spike_mask & (ins_new == 0)).sum()) > 0
      and assigned < int(spike_mask.sum()),
      f"已编号 {assigned:,} / {int(spike_mask.sum()):,}")
check("非穗点的标签没被改动",
      np.array_equal(win.ann.labels[sid][~spike_mask], lab_before[sid][~spike_mask]))
check("写入的是 uint16 能表达的范围", win.ann.next_instance <= 65535,
      f"next={win.ann.next_instance}")

# 列表与统计同步
win._refresh_list()
txt = win.lbl_stat.text()
check("统计栏把穗实例数算进去了", "穗" in txt, txt.replace("\n", " | ")[:80])

# 3.3 撤销整步
win.undo()
APP.processEvents()
check("覆盖模式可整体撤销",
      np.array_equal(win.ann.instances[sid], ins_before[sid])
      and np.array_equal(win.ann.labels[sid], lab_before[sid]))
check("撤销后与分割前逐点一致",
      all(np.array_equal(a, b) for a, b in zip(win.ann.labels, lab_before)))

# 3.4 试算不动标注
probe_txt = win._segment_probe("current", SegmentParams(0.3, 1, 10), True)
check("试算返回统计文本", "扫描" in probe_txt and "个穗" in probe_txt,
      probe_txt.splitlines()[0])
check("试算不改动标注",
      all(np.array_equal(a, b) for a, b in zip(win.ann.labels, lab_before)))

# 3.5 对话框
dlg = SegmentDialog(len(win.doc.scans), 0, SegmentParams(0.3, 1, 10),
                    probe=win._segment_probe, parent=win)
dlg._do_probe()
check("对话框试算有结果", "穗" in dlg.lbl_probe.text(), dlg.lbl_probe.text()[:60])
dlg.spin_voxel.setValue(0.45)
dlg.spin_min.setValue(20)
dlg.cmb_link.setCurrentIndex(0)
check("对话框参数正确回传",
      dlg.params() == SegmentParams(0.45, 1, 20), str(dlg.params().to_dict()))
check("默认范围是当前扫描", dlg.scope() == "current")
check("默认不覆盖已有实例", dlg.overwrite() is False)
dlg.close()

# 3.6 实例框与试算预览
win.chk_ibox.setChecked(True)
APP.processEvents()
sid_f = win._focused_scan()
lab_f = win.ann.labels[sid_f]
ins_f = np.asarray(win.ann.instances[sid_f])
ids_f = sorted({int(v) for v in np.unique(ins_f[lab_f == 1]) if v})
box_txt = sorted(int(b.text[1:]) for b in win.view.instance_boxes)
check("实例框只画聚焦站且编号一致",
      box_txt == ids_f and len(win.view.instance_boxes) == len(ids_f),
      f"{len(box_txt)} / {len(ids_f)}")
check("实例框是实线", all(not b.dashed for b in win.view.instance_boxes))

nxt0 = int(win.ann.next_instance)
snap_ins0 = [np.asarray(i).copy() for i in win.ann.instances]
win._segment_probe("current", SegmentParams(0.3, 1, 10), True)
APP.processEvents()
prev = win.view.preview_boxes
check("试算生成虚线预览框", len(prev) > 0 and all(b.dashed for b in prev),
      f"{len(prev)} 个")
check("预览编号从 next_instance 起连续排",
      sorted(int(b.text[1:]) for b in prev) == list(range(nxt0, nxt0 + len(prev))))
check("试算后标注仍未改动",
      all(np.array_equal(a, b) for a, b in zip(win.ann.instances, snap_ins0)))

# 覆盖模式下：执行结果应与预览的框数一致（碎片旧编号要清掉，否则会多出框）
n_prev = len(prev)
win.apply_segment(SegmentParams(0.3, 1, 10), "current", overwrite=True)
APP.processEvents()
check("覆盖模式执行后的框数与预览一致",
      len(win.view.instance_boxes) == n_prev,
      f"执行 {len(win.view.instance_boxes)} / 预览 {n_prev}")
check("执行后预览框被清空", win.view.preview_boxes == [])
win.undo()
APP.processEvents()
win.view.set_preview_boxes([])
win.chk_ibox.setChecked(False)
APP.processEvents()
check("关闭实例框后不再画框", win.view.instance_boxes == [])

# 3.7 存盘 → 读回
win.apply_segment(SegmentParams(0.3, 1, 10), "current", overwrite=True)
saved = [i.copy() for i in win.ann.instances]
saved_lab = [l.copy() for l in win.ann.labels]
out = os.path.join(tmpdir, "seg_out.e57")
win._do_save(out)
win.load_file(out)
APP.processEvents()
check("读回文件的实例编号与存盘前一致",
      all(np.array_equal(a, b) for a, b in zip(win.ann.instances, saved)),
      f"max={int(max(np.max(i) for i in win.ann.instances))}")
check("读回文件的标签与存盘前一致",
      all(np.array_equal(a, b) for a, b in zip(win.ann.labels, saved_lab)))

# 3.8 手动框选穗实例（实例工具 + 合并 / 撤销 / 空框提示）
print("\n3.8) 手动框选穗实例")
from view3d import MODE_INST

sid = win._focused_scan()
r = win.view.renderer
lab = win.ann.labels[sid]
ins = win.ann.instances[sid]
ids = [int(v) for v in np.unique(ins[lab == 1]) if v]
check("当前扫描有可框选的穗实例", len(ids) >= 2, f"{len(ids)} 个")
g_of = {int(li): g for g, (s, li) in enumerate(zip(r.scan_ids, r.local_index))
        if int(s) == sid}

win.set_mode(MODE_INST)
APP.processEvents()
check("实例按钮选中且视图进入实例模式",
      win.btn_inst.isChecked() and win.view.mode == MODE_INST)
check("其他工具按钮未勾选",
      not any(b.isChecked() for b in (win.btn_nav, win.btn_brush, win.btn_erase,
                                      win.btn_box, win.btn_crop)))

# (a) 框住两个已有实例 → 合并为一个新编号
pair = ids[:2]
sel_local = np.nonzero(np.isin(ins, pair))[0]
sel_glob = np.array([g_of[int(li)] for li in sel_local if int(li) in g_of], np.int64)
nxt0 = win.ann.next_instance
ins_pre = ins.copy()
win._on_instance_box(sel_glob)
APP.processEvents()
merged = {int(v) for v in np.unique(ins[sel_local])}
check("框住两段实例后合并为一个新编号",
      len(merged) == 1 and min(merged) >= nxt0, f"{sorted(merged)}（旧 next={nxt0}）")
outside = np.ones(ins.size, bool)
outside[sel_local] = False
check("框外点的实例编号未动",
      np.array_equal(ins[outside], ins_pre[outside]))
check("新实例的点全部是穗类别", bool((lab[ins == min(merged)] == 1).all()))
check("旧编号已从 instance_info 注销",
      all(u not in win.ann.instance_info for u in pair))
check("合并后自动打开实例框", win.chk_ibox.isChecked())
check("实例框里出现新编号",
      any(b.text == f"#{min(merged)}" for b in win.view.instance_boxes),
      f"{[b.text for b in win.view.instance_boxes][:6]}")
win.undo()
APP.processEvents()
check("撤销后两个旧编号还原",
      set(int(v) for v in np.unique(ins[sel_local])) == set(pair))

# (b) 框内没有穗点：不产生撤销步、标注不变、给出提示
other = np.nonzero(lab != 1)[0]
if other.size:
    glob2 = np.array([g_of[int(li)] for li in other[:40] if int(li) in g_of], np.int64)
    n_undo = len(win._undo)
    before = ins.copy()
    win._on_instance_box(glob2)
    APP.processEvents()
    check("框内无穗点时不产生撤销步", len(win._undo) == n_undo)
    check("框内无穗点时标注不变", np.array_equal(ins, before))
    check("无穗点时给出提示", "没有穗点" in win.status.currentMessage())
else:
    print("   （该文件没有非穗点，跳过空框用例）")

check("全程无异常", not errors, str(errors[:1]))

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
