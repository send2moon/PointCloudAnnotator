"""界面细节回归：关闭文件、选中态高亮、裁剪对话框白色单选框。"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from PySide6.QtWidgets import QApplication, QMessageBox

APP = QApplication(sys.argv[:1])
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

from crop_dialog import CropDialog
from cropping import CropBox
from main_window import APP_ID, APP_NAME, DARK_QSS, MainWindow
from view3d import MODE_BOX, MODE_BRUSH, MODE_CROP, MODE_ERASE, MODE_NAV

GOOD = r"D:/PCProjects/01wheat/FinalScripts/outputs/PLOTS/FINAL/1-FS_09_labeled.e57"

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


errors = []
sys.excepthook = lambda *a: errors.append(a)

print("0) 软件名称")
check("APP_NAME 为新名称", APP_NAME == "作物点云器官标注软件", APP_NAME)
check("内部标识保持不变（配置不丢）", APP_ID == "PointCloudAnnotator", APP_ID)
_app_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "src", "app.py"), encoding="utf-8").read()
check("应用显示名已更新", 'setApplicationDisplayName("作物点云器官标注软件")' in _app_src)
check("控制台标题由宽字符 API 设置", "SetConsoleTitleW" in _app_src)
check("run.bat 仍为纯 ASCII", all(ord(c) < 128 for c in
      open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.bat"),
           encoding="ascii").read()))

print("\n1) 样式表")
check("侧栏工具按钮有选中态样式", "QToolButton:checked" in DARK_QSS)
check("单选框白描边", "QRadioButton::indicator" in DARK_QSS
      and "2px solid #ffffff" in DARK_QSS)
check("复选框白描边", "QCheckBox::indicator" in DARK_QSS)
check("选中态白底", "background: #ffffff" in DARK_QSS)

print("\n2) 关闭文件按钮")
win = MainWindow()
win.resize(1200, 800)
win.show()
APP.processEvents()
win._ask_force_open = lambda err: False
win._confirm_discard = lambda: True
check("工具栏有「关闭文件」动作", hasattr(win, "act_close")
      and win.act_close.text() == "关闭文件")
check("侧栏有关闭按钮", hasattr(win, "btn_close"))
check("无文件时按钮置灰", not win.act_close.isEnabled()
      and not win.btn_save.isEnabled() and not win.btn_close.isEnabled())
check("初始标题为未打开文件", "未打开文件" in win.windowTitle(), win.windowTitle())
check("窗口标题显示新名称", "作物点云器官标注软件" in win.windowTitle()
      and "点云器官标注工具" not in win.windowTitle())
check("工具栏右上角显示新名称", win.lbl_title.text() == APP_NAME)

win.load_file(GOOD)
APP.processEvents()
n0 = win.doc.total_points if win.doc else 0
check("打开后按钮可用", win.act_close.isEnabled() and win.btn_save.isEnabled())
check("打开后有点", n0 > 0 and len(win.view.renderer.xyz) == n0,
      f"{len(win.view.renderer.xyz):,} / 文档 {n0:,}")

print("\n3) 选中态高亮")
win.set_mode(MODE_BOX)
APP.processEvents()
check("框选按钮已勾选", win.btn_box.isChecked())
check("其他工具按钮未勾选",
      not any(b.isChecked() for b in (win.btn_nav, win.btn_brush,
                                      win.btn_erase, win.btn_crop)))
win.set_label(2)
APP.processEvents()
check("「叶」按钮已勾选", win._label_btns[2].isChecked())
check("其他类别按钮未勾选",
      not any(win._label_btns[i].isChecked() for i in (1, 3, 4)))
ss = win._label_btns[2].styleSheet()
check("类别按钮选中态用类别色填充", "rgb(96,220,120)" in ss and "QToolButton:checked" in ss)
check("类别按钮选中态白描边", "2px solid #ffffff" in ss)

print("\n4) 裁剪对话框单选框")
box = CropBox(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=0, zmax=2)
dlg = CropDialog(box, 100, 30, win)
rs = dlg.rb_keep.styleSheet()
check("单选框样式含白描边", "2px solid #ffffff" in rs)
check("单选框选中为实心白", "QRadioButton::indicator:checked" in rs
      and "background: #ffffff" in rs)
check("默认选中「保留框内」", dlg.rb_keep.isChecked() and not dlg.rb_remove.isChecked())
check("切换语义生效", (dlg.rb_remove.setChecked(True) or dlg.invert()))
dlg.close()

print("\n5) 关闭文件")
win.close_file()
APP.processEvents()
check("文档已清空", win.doc is None and win.ann is None and win.path is None)
check("渲染数据已清空", len(win.view.renderer.xyz) == 0)
check("撤销栈已清空", len(win._undo) == 0)
check("队列已清空", not win._queue)
check("按钮重新置灰", not win.act_close.isEnabled()
      and not win.btn_save.isEnabled() and not win.btn_close.isEnabled())
check("标题回到未打开文件", "未打开文件" in win.windowTitle(), win.windowTitle())
check("列表已清空", win.lst.count() == 0)
check("模式回到浏览", win.view.mode == MODE_NAV)
check("着色回到真彩色", win.cmb_color.currentIndex() == 0)

win.close_file()          # 重复关闭不应报错
APP.processEvents()

print("\n6) 关闭后可重新打开")
win.load_file(GOOD)
APP.processEvents()
check("重新打开成功", win.doc is not None and win.doc.total_points == n0,
      f"{win.doc.total_points if win.doc else 0:,} / {n0:,}")
check("按钮恢复可用", win.act_close.isEnabled())

check("全程无未捕获异常", not errors, str(errors[:1]))
print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
