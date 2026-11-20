"""框选（BOO 模式）真实鼠标事件回归：防止 _apply_box 解包再次回归。"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

APP = QApplication(sys.argv[:1])

from main_window import MainWindow
from view3d import MODE_BOX, MODE_CROP

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


errors = []
sys.excepthook = lambda *a: errors.append(a)


def _patched(*a):
    errors.append(a)


win = MainWindow()
win.resize(1200, 800)
win.show()
APP.processEvents()
win.load_file(files[0])
APP.processEvents()
print(f"载入 {win.doc.total_points:,} 点，{len(win.doc.scans)} 扫描")

view = win.view
w, h = view.width(), view.height()
print(f"视图 {w}x{h}")

# ---------------------------------------------------------------- 框选
lanes = []
view.annotateRequest.connect(lambda g, l, a: lanes.append((len(g), l)))
win._on_annotate_orig = win._on_annotate

# 直接触发 Qt 事件，走 mousePressEvent / mouseReleaseEvent 真实路径
win.set_mode(MODE_BOX)
APP.processEvents()
check("进入框选模式", view.mode == MODE_BOX, view.mode)

before = win.ann.total_marked()
win.current_label = 1
p0 = QPoint(int(w * 0.30), int(h * 0.25))
p1 = QPoint(int(w * 0.62), int(h * 0.70))
QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, p0)
QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, p1)
APP.processEvents()

check("无异常抛出", not errors, str(errors[:1]))
check("框选产生标注请求", len(lanes) == 1, str(lanes))
check("一次操作 = 一个撤销步", len(win._undo) == 1,
      f"{len(win._undo)} 步 / 覆盖 {len(win._undo[-1]) if win._undo else 0} 扫描")
check("框内点数 > 0", lanes and lanes[0][0] > 0)
check("label 为穗", lanes and lanes[0][1] == 1)
after = win.ann.total_marked()
check("标注点数增加", after > before, f"{before} -> {after}")

# 撤销
win.undo()
APP.processEvents()
check("撤销后恢复", win.ann.total_marked() == before,
      f"{win.ann.total_marked()} vs {before}")

# ---------------------------------------------------------------- 反向拖拽
lanes.clear()
QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, p1)
QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, p0)
APP.processEvents()
check("反向拖拽等价", lanes and lanes[0][0] > 0, str(lanes))
win.undo()
APP.processEvents()

# ---------------------------------------------------------------- 空框
lanes.clear()
QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, QPoint(5, 5))
QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, QPoint(6, 6))
APP.processEvents()
check("过小框被忽略", not lanes, str(lanes))

# ---------------------------------------------------------------- 裁剪框
win.set_mode(MODE_CROP)
APP.processEvents()
crops = []
win._open_crop_dialog = lambda *a, **k: crops.append(a)   # 避免弹出模态框
QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, p0)
QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, p1)
APP.processEvents()
check("裁剪请求非空", len(crops) == 1, str(len(crops)))
check("裁剪盒有效", crops and crops[0][0].is_valid(),
      f"{crops[0][0]}" if crops else "")

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
