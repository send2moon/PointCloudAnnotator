# -*- coding: utf-8 -*-
"""实例立方体框 + 分割预览：验证与截图（真实文件）。"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
APP = QApplication(sys.argv[:1])
fid = QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\msyh.ttc")
APP.setFont(QFont(QFontDatabase.applicationFontFamilies(fid)[0], 9))

from main_window import MainWindow
from instance_seg import SegmentParams

REAL = r"D:\Data\PC\ResearchPlots\DT\1-FS_09_labeled.e57"
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
win = MainWindow()
win.settings = QSettings(os.path.join(OUT, "_probe.ini"), QSettings.IniFormat)
win.resize(1920, 1080)
win.show(); APP.processEvents()
win.load_file(REAL); APP.processEvents()

ok = fail = 0
def check(name, cond, extra=""):
    global ok, fail
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{extra}]" if extra else ""))
    ok, fail = ok + bool(cond), fail + (not cond)

def focus_ids() -> list:
    """聚焦扫描里已编号穗实例的编号。"""
    sid = win._focused_scan()
    lab = win.ann.labels[sid]
    ins = np.asarray(win.ann.instances[sid])
    return sorted({int(v) for v in np.unique(ins[lab == 1]) if v})

def frame_boxes(pad: float = 1.7) -> None:
    """把相机拉到"刚好框住所有实例框"的位置，截图才看得清编号。"""
    boxes = win.view.instance_boxes or win.view.preview_boxes
    if not boxes:
        return
    lo = np.min([b.lo for b in boxes], axis=0)
    hi = np.max([b.hi for b in boxes], axis=0)
    cam = win.view.renderer.camera
    cam.center = ((lo + hi) / 2).astype(np.float32)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    cam.distance = max(radius / np.tan(np.radians(cam.fov * 0.5)) * pad, 1e-3)
    win.view.fit_and_render()
    APP.processEvents()

def select_scan(sid: int) -> None:
    """在左侧列表里点中某一站（与人工操作同一条路径）。"""
    for row in range(win.lst.count()):
        if win.lst.item(row).data(Qt.UserRole) == sid:
            win.lst.setCurrentRow(row)
            APP.processEvents()
            return
    raise RuntimeError(f"列表里找不到扫描 {sid}")

# 单站显示 + 聚焦到实例最多的一站（扫描 8），框才不会糊成一片
win.chk_all_scans.setChecked(False); APP.processEvents()
select_scan(7)
ids_focus = focus_ids()

# --- 1. 默认关闭，不画框 ---
check("默认不开实例框时视图无框", win.view.instance_boxes == [])

# --- 2. 打开实例框：框数与编号对得上聚焦的那一站 ---
win.chk_ibox.setChecked(True); APP.processEvents()
boxes = win.view.instance_boxes
check("打开后框数 = 聚焦站的穗实例数", len(boxes) == len(ids_focus),
      f"{len(boxes)} / {len(ids_focus)}")
texts = sorted(int(b.text[1:]) for b in boxes)
check("框顶编号与实例编号一致", texts == ids_focus, f"{texts[:6]}...")
check("框是实线（非预览）", all(not b.dashed for b in boxes))
b0 = boxes[0]
uid0 = int(b0.text[1:])
sid0 = win._focused_scan()
ins0 = np.asarray(win.ann.instances[sid0])
pts = win.doc.scans[sid0].points
sub = np.column_stack([pts["cartesianX"], pts["cartesianY"],
                       pts["cartesianZ"]])[ins0 == uid0]
check("框的包络与实例点一致",
      np.allclose(np.asarray(b0.lo), sub.min(0)) and
      np.allclose(np.asarray(b0.hi), sub.max(0)))
frame_boxes()
win.view.grab().save(os.path.join(OUT, "ui_instance_boxes.png"))

# --- 3. 切到别的一站，框跟着换 ---
select_scan(0)
ids_other = focus_ids()
other_texts = sorted(int(b.text[1:]) for b in win.view.instance_boxes)
check("切扫描后框跟着换站", other_texts == ids_other and ids_other != ids_focus,
      f"{len(other_texts)} 框")
select_scan(7); APP.processEvents()

# --- 4. 分割预览（覆盖模式）：虚线框 + 编号从 next_instance 起 ---
snap_ins = [np.asarray(i).copy() for i in win.ann.instances]
snap_lab = [np.asarray(l).copy() for l in win.ann.labels]
nxt = int(win.ann.next_instance)
win._segment_probe("current", SegmentParams(0.30, 1, 10), True)
APP.processEvents()
prev = win.view.preview_boxes
check("试算后出现预览框", len(prev) > 0, f"{len(prev)} 个")
check("预览框全部虚线", all(b.dashed for b in prev))
ptexts = sorted(int(b.text[1:]) for b in prev)
check("预览编号从 next_instance 连续排",
      ptexts == list(range(nxt, nxt + len(ptexts))), f"{ptexts[:5]}...")
check("试算不改标注",
      all(np.array_equal(a, b) for a, b in zip(win.ann.instances, snap_ins))
      and all(np.array_equal(a, b) for a, b in zip(win.ann.labels, snap_lab)))
frame_boxes()
win.view.grab().save(os.path.join(OUT, "ui_segment_preview.png"))

# --- 5. 清预览、执行分割后框自动更新 ---
win.view.set_preview_boxes([])
win.apply_segment(SegmentParams(0.30, 1, 10), "current", True)
APP.processEvents()
check("执行分割后实例框自动打开并刷新",
      win.chk_ibox.isChecked() and len(win.view.instance_boxes) > 0,
      f"{len(win.view.instance_boxes)} 框")
ids_after = focus_ids()
new_texts = sorted(int(b.text[1:]) for b in win.view.instance_boxes)
check("分割后框编号与新实例一致", new_texts == ids_after)
check("覆盖模式执行结果与试算预览一致", new_texts == ptexts,
      f"执行 {len(new_texts)} / 预览 {len(ptexts)}")
frame_boxes()
win.view.grab().save(os.path.join(OUT, "ui_instance_after_seg.png"))

# --- 6. 撤销后框跟着回退 ---
win.undo(); APP.processEvents()
back = sorted(int(b.text[1:]) for b in win.view.instance_boxes)
check("撤销后框回到分割前", back == ids_focus)

# --- 7. 关闭实例框即清空 ---
win.chk_ibox.setChecked(False); APP.processEvents()
check("关闭实例框后视图无框", win.view.instance_boxes == [])
win.chk_ibox.setChecked(True); APP.processEvents()

# --- 8. 关闭文件清框 ---
win.close_file(); APP.processEvents()
check("关闭文件后框清空",
      win.view.instance_boxes == [] and win.view.preview_boxes == [])

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
