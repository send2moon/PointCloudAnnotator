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
from view3d import (MODE_BOX, MODE_BRUSH, MODE_CROP, MODE_ERASE, MODE_INST,
                    MODE_NAV)

GOOD = r"D:/Data/PC/ResearchPlots/DT/1-FS_09_labeled.e57"

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


errors = []
sys.excepthook = lambda *a: errors.append(a)

print("0) 软件名称")
check("APP_NAME 为新名称", APP_NAME == "作物点云处理软件", APP_NAME)
check("内部标识保持不变（配置不丢）", APP_ID == "PointCloudAnnotator", APP_ID)
_app_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "src", "app.py"), encoding="utf-8").read()
check("应用显示名已更新", 'setApplicationDisplayName("作物点云处理软件")' in _app_src)
check("控制台标题由宽字符 API 设置", "SetConsoleTitleW" in _app_src)
check("run.bat 仍为纯 ASCII", all(ord(c) < 128 for c in
      open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.bat"),
           encoding="ascii").read()))

print("\n1) 样式表")
check("侧栏工具按钮有选中态样式", "QToolButton:checked" in DARK_QSS)
check("单选框有独立样式", "QRadioButton::indicator" in DARK_QSS)
check("复选框白描边", "QCheckBox::indicator" in DARK_QSS)
check("Office 蓝为主色且白底", "#2b579a" in DARK_QSS and "#ffffff" in DARK_QSS)

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
check("窗口标题显示新名称", "作物点云处理软件" in win.windowTitle()
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
      not any(b.isChecked() for b in (win.btn_nav, win.btn_brush, win.btn_erase,
                                      win.btn_inst, win.btn_crop)))
win.set_mode(MODE_INST)
APP.processEvents()
check("实例按钮已勾选", win.btn_inst.isChecked() and win.view.mode == MODE_INST)
check("切到实例模式后框选按钮取消勾选", not win.btn_box.isChecked())
check("工具行共 6 个按钮（浏览/笔刷/橡皮/框选/实例/裁剪）",
      len(win._tool_group.buttons()) == 6,
      str([b.text() for b in win._tool_group.buttons()]))
win.set_mode(MODE_BOX)
APP.processEvents()
win.set_label(2)
APP.processEvents()
check("「叶」按钮已勾选", win._label_btns[2].isChecked())
check("其他类别按钮未勾选",
      not any(win._label_btns[i].isChecked() for i in (1, 3, 4)))
ss = win._label_btns[2].styleSheet()
check("类别按钮选中态用类别色填充", "rgb(96,220,120)" in ss and "QToolButton:checked" in ss)
check("类别按钮选中态蓝描边", "2px solid #1e3f75" in ss)

print("\n4) 裁剪对话框单选框")
box = CropBox(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=0, zmax=2)
dlg = CropDialog(box, 100, 30, win)
rs = dlg.rb_keep.styleSheet()
check("单选框样式含灰描边", "1px solid #8b949f" in rs)
check("单选框选中为 Office 蓝实心", "QRadioButton::indicator:checked" in rs
      and "background: #2b579a" in rs)
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

print("\n7) 启动最大化与自适应布局")
from PySide6.QtWidgets import QGroupBox, QScrollArea, QSizePolicy

check("app.py 启动即最大化", "showMaximized()" in _app_src)
check("窗口有合理的最小尺寸", win.minimumWidth() >= 960 and win.minimumHeight() >= 600,
      f"{win.minimumWidth()}x{win.minimumHeight()}")

side = win._split.widget(1)
check("侧栏用滚动区包裹（小屏不截断）", isinstance(win.side_scroll, QScrollArea))
check("滚动区横向不出滚动条",
      win.side_scroll.horizontalScrollBarPolicy() == win.side_scroll.horizontalScrollBarPolicy()
      and win.side_scroll.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff")

win.resize(1200, 800)
APP.processEvents()
w_narrow = side.width()
win.resize(2200, 1300)
APP.processEvents()
w_wide = side.width()
print(f"   侧栏宽度：1200 宽窗口 → {w_narrow}px，2200 宽窗口 → {w_wide}px")
check("窗口变宽时侧栏跟着变宽", w_wide > w_narrow, f"{w_narrow} -> {w_wide}")
check("侧栏宽度有上限（不会吃掉视图）", w_wide <= 560, f"{w_wide}px")

# 用户手动拖过分隔条后，不再自动改宽度
win._split.setSizes([1500, 470])
win._on_split_moved()
APP.processEvents()
win.resize(1400, 900)
APP.processEvents()
w_manual = side.width()
check("手动拖过分隔条后不再被自动改宽", w_manual >= 400, f"{w_manual}px")

# 矮窗口隐藏说明文字、收紧间距
win.resize(1600, 1200)
APP.processEvents()
tall_hints = all(h.isVisible() for h in win._side_hints)
win.resize(1600, 700)
APP.processEvents()
short_hints = any(h.isVisible() for h in win._side_hints)
check("高窗口显示说明文字", tall_hints)
check("矮窗口自动收起说明文字", not short_hints)
groups = win.side_scroll.widget().findChildren(QGroupBox)
squashed = [g.title() for g in groups if g.height() < 40]
check("矮窗口下各分组没被挤没", not squashed, str(squashed))

# 矮窗口：内容超出时出现滚动条而不是被裁掉
body_h = win.side_scroll.widget().height()
vp_h = win.side_scroll.viewport().height()
print(f"   1600x700：侧栏内容 {body_h}px / 可见 {vp_h}px")
check("放不下时可滚动", body_h >= vp_h)

check("状态栏文件名可被压缩（不挤掉右侧耗时）",
      win.lbl_file.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored)

win.resize(1200, 800)
APP.processEvents()
win.showMaximized()
APP.processEvents()
check("showMaximized 后处于最大化状态", win.isMaximized())

print("\n8) 语义/实例切换 · 隐藏 · 改标注入口")
check("有语义 / 实例两种标注方式",
      hasattr(win, "btn_semantic") and hasattr(win, "btn_instance"))
win.set_ann_mode("semantic")
APP.processEvents()
check("语义模式选中且与实例互斥",
      win.btn_semantic.isChecked() and not win.btn_instance.isChecked())
check("标注方式写入配置", win.settings.value("ann_mode") == "semantic")
win.set_ann_mode("instance")
APP.processEvents()
check("能切回实例标注", win.btn_instance.isChecked() and win.ann_mode == "instance")
check("有隐藏已标注 / 隐藏当前类别选项",
      hasattr(win, "chk_hide_marked") and hasattr(win, "chk_hide_cur"))
check("有修改已有标注入口（按钮 + M 快捷键）",
      hasattr(win, "btn_label_edit") and hasattr(win, "act_label_edit") and
      win.act_label_edit.shortcut().toString() == "M")
check("扫描列表不再用 QListWidget 的超大默认高度",
      win.lst.sizeHint().height() <= 150, f"{win.lst.sizeHint().height()}")

print("\n9) 功能按钮悬浮说明（tooltip）")
from PySide6.QtWidgets import (QAbstractButton, QComboBox, QDoubleSpinBox, QSlider,
                               QSpinBox)

from label_edit_dialog import LabelEditDialog
from segment_dialog import SegmentDialog
from instance_seg import SegmentParams


def _interactive(root):
    """收集归属于该窗口的可交互控件（按钮/下拉/滑杆/数值框），去重。

    只取 ``window() is root`` 的控件：挂在主窗口下的对话框（父子关系上属于
    win，但视觉上是独立窗口）由各自的检查覆盖，tooltip 也各自统计。
    """
    seen = {}
    for cls in (QAbstractButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox):
        for w in root.findChildren(cls):
            seen[id(w)] = w
    return [w for w in seen.values()
            if w.objectName() not in ("qt_toolbar_ext_button", "ScrollLeftButton",
                                    "ScrollRightButton") and w.window() is root]


def _no_tip(root, skip=()):
    out = []
    for w in _interactive(root):
        if w in skip:
            continue
        if not w.toolTip().strip():
            out.append(f"{type(w).__name__}({w.text() if hasattr(w, 'text') else ''})")
    return out


check("样式表里有 tooltip 气泡样式（浅色 + 圆角）",
      "QToolTip {" in DARK_QSS and "border-radius" in DARK_QSS.split("QToolTip {")[1][:200])
missing_main = _no_tip(win)
check("主窗口所有按钮/下拉/数值框都有悬浮说明", not missing_main, str(missing_main[:4]))
_short = [f"{type(w).__name__}({getattr(w, 'text', lambda: '')()})"
          for w in _interactive(win) if len(w.toolTip().strip()) < 12]
check("没有敷衍的一句话说明（都 ≥ 12 字）", not _short, str(_short[:4]))

# 关键按钮的说明要有实质内容（不只是重复按钮文字）
for name, w in (("浏览", win.btn_nav), ("笔刷", win.btn_brush), ("橡皮", win.btn_erase),
                ("框选", win.btn_box), ("实例(工具)", win.btn_inst), ("裁剪", win.btn_crop),
                ("语义", win.btn_semantic), ("实例(模式)", win.btn_instance),
                ("穗实例分割", win.btn_seg), ("改标注", win.btn_label_edit),
                ("覆盖保存", win.btn_save2), ("保存为", win.btn_save),
                ("关闭文件", win.btn_close)):
    t = w.toolTip()
    check(f"「{name}」说明足够具体（≥ 20 字且含换行或多句）",
          len(t) >= 20 and ("\n" in t or "。" in t), f"{len(t)} 字")

for name, a in (("打开", win.act_close), ("保存", win.act_save), ("裁剪", win.act_crop),
                ("穗分割", win.act_seg), ("标注管理", win.act_label_edit),
                ("撤销", win.act_undo), ("下一个文件", win.act_next)):
    check(f"工具栏「{name}」有悬浮说明", bool(a.toolTip().strip()), a.toolTip()[:24])
check("工具栏「打开点云…」有悬浮说明",
      any(a.toolTip().strip() for a in win.findChildren(type(win.act_undo))
          if a.text() == "打开点云…"))
_named = {a.text(): a.toolTip() for a in win.findChildren(type(win.act_undo)) if a.text()}
_miss_act = [k for k in ("打开点云…", "打开文件夹…", "关闭文件", "保存为…", "覆盖保存",
                         "裁剪…", "穗分割…", "标注管理…", "适合窗口", "俯视", "侧视",
                         "等轴", "撤销", "下一个文件") if not _named.get(k, "").strip()]
check("工具栏每一个按钮都有悬浮说明", not _miss_act, str(_miss_act))
check("类别按钮说明带类别含义",
      all(len(b.toolTip()) > 15 for b in win._label_btns.values()))

# 对话框里的按钮也要有说明
dlg_crop = CropDialog(CropBox(-50, 50, -50, 50, 0, 100), 1000, 400, win)
check("裁剪对话框控件都有说明", not _no_tip(dlg_crop), str(_no_tip(dlg_crop)[:4]))
dlg_seg = SegmentDialog(len(win.doc.scans), 0, SegmentParams(), None, None, False, win)
check("穗分割对话框控件都有说明", not _no_tip(dlg_seg), str(_no_tip(dlg_seg)[:4]))
dlg_lab = LabelEditDialog(win, win)
check("标注管理对话框控件都有说明", not _no_tip(dlg_lab), str(_no_tip(dlg_lab)[:4]))

check("全程无未捕获异常", not errors, str(errors[:1]))
print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
