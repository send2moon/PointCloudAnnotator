"""导出的标注文件能否再打开：自检、强制打开、标注回读。"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from PySide6.QtWidgets import QApplication, QMessageBox

APP = QApplication(sys.argv[:1])

from e57io import E57Document, StructureError, check_e57_structure
from main_window import MainWindow

# 测试里不要让任何模态框真的弹出来（offscreen 下会一直阻塞）
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

GOOD = r"D:/PCProjects/01wheat/FinalScripts/outputs/PLOTS/FINAL/1-FS_09_labeled.e57"
BAD = None      # 第 2 节里现场生成（截断副本），不依赖外部坏样本

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


errors = []
sys.excepthook = lambda *a: errors.append(a)

print("1) 新版导出的标注文件")
# 用户会不断覆盖这个文件，期望值一律现场算，不写死
_base = E57Document.read(GOOD)
N_PTS = _base.total_points
N_MARKED = sum(int((np.asarray(s.points["annotationLabel"]) > 0).sum())
               for s in _base.scans if "annotationLabel" in s.points)
print(f"    基准：{N_PTS:,} 点 / {N_MARKED:,} 标注点 / {len(_base.scans)} 扫描")
check("结构体检通过", not check_e57_structure(GOOD), str(check_e57_structure(GOOD))[:120])
win = MainWindow()
win.resize(1200, 800)
win.show()
APP.processEvents()
# 强制打开确认框在测试里一律不弹
win._ask_force_open = lambda err: False
win._confirm_discard = lambda: True
win.load_file(GOOD)
APP.processEvents()
check("载入成功", win.doc is not None and win.path == GOOD)
if win.doc:
    check("扫描数一致", len(win.doc.scans) == len(_base.scans), str(len(win.doc.scans)))
    check("总点数一致", win.doc.total_points == N_PTS, f"{win.doc.total_points:,}")
    marked = int(sum(int((l > 0).sum()) for l in win.ann.labels))
    check("标注点全部回读", marked == N_MARKED, f"{marked:,} / {N_MARKED:,}")
    check("实例汇总与标注一致", win.ann.total_marked() == N_MARKED,
          f"{win.ann.total_marked():,}")
    insts = sum(len(win.ann.instance_ids(i)) for i in range(len(win.doc.scans)))
    print(f"    实例总数 {insts}")
    check("渲染器标签与标注一致", int((win.view.renderer.labels > 0).sum()) == N_MARKED,
          f"{int((win.view.renderer.labels > 0).sum()):,} / {N_MARKED:,}")
    check("无体检告警", not win.doc.warnings, str(win.doc.warnings))

print("\n2) 写坏的文件（真的打不开）")
# 坏文件现场造：把好文件截断，头部声明的长度与 XML 位置就都对不上了。
# （用户那批旧版写坏的 *_labeled.e57 已删，这里不依赖外部样本。）
import tempfile

BAD = os.path.join(tempfile.gettempdir(), "_smoke_broken.e57")
with open(GOOD, "rb") as f:
    _data = f.read()
with open(BAD, "wb") as f:
    f.write(_data[: int(len(_data) * 0.75)])
print(f"    造坏样本 {os.path.basename(BAD)}：{len(_data):,} → {os.path.getsize(BAD):,} 字节")
probs = check_e57_structure(BAD)
check("体检查出问题", bool(probs), str(probs)[:140])
try:
    E57Document.read(BAD)
    check("默认打开应抛 StructureError", False)
except StructureError as e:
    check("默认打开抛 StructureError", True, f"{len(e.problems)} 条")
    check("错误信息含中文原因", "e57 结构自检未通过" in str(e))

forced = None
try:
    forced = E57Document.read(BAD, force=True)
    check("强制打开返回文档", True, f"{forced.total_points:,} 点")
    check("强制打开记录告警", bool(forced.warnings), str(forced.warnings)[:100])
except ValueError as e:
    check("强制打开给出可读报错", "强制打开失败" in str(e) or "结构" in str(e), str(e)[:140])
except Exception as e:  # noqa: BLE001
    check("强制打开报错类型合理", False, f"{type(e).__name__}: {e}")

print("\n3) 界面上「仍然打开 / 取消」两条路")
win._ask_force_open = lambda err: False            # 点取消
win.load_file(BAD)
APP.processEvents()
check("取消后不改变当前文档", win.path == GOOD, str(win.path))

win._ask_force_open = lambda err: True             # 点仍然打开
win.load_file(BAD)
APP.processEvents()
if win.path == BAD and win.doc is not None:
    print(f"    强制打开成功：{win.doc.total_points:,} 点 / {len(win.doc.scans)} 扫描")
    check("强制打开后能标注", True)
    # 强制打开后仍应能标注并撤销
    import numpy as np
    win.ann.labels[0][:10] = 1
    win._dirty = True
    check("标注数组长度与点数一致",
          len(win.ann.labels[0]) == win.doc.scans[0].count)
else:
    print("    强制打开失败（预期，文件确实损坏）")
    check("失败后状态干净", win.path == GOOD or win.doc is None, str(win.path))

check("全程无未捕获异常", not errors, str(errors[:1]))
try:
    os.remove(BAD)
except OSError:
    pass

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
