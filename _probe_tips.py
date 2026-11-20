"""悬浮说明探针：真正触发 tooltip 并截图，确认气泡样式与文案。

用法（offscreen 也能跑，因为 QTipLabel 是普通窗口部件）::

    python _probe_tips.py

产物：``outputs/tip_<名字>.png`` —— 单个气泡的原图；
      ``outputs/tips_all.png`` —— 把几个气泡拼到侧栏截图旁边，便于一眼看全。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QFontDatabase, QHelpEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

APP = QApplication(sys.argv[:1])
# offscreen 下没有系统字体回退，中文会变方块
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(_f):
        QFontDatabase.addApplicationFont(_f)

QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

from crop_dialog import CropDialog          # noqa: E402
from cropping import CropBox                # noqa: E402
from main_window import MainWindow          # noqa: E402
from view3d import MODE_BOX                 # noqa: E402

GOOD = r"D:/Data/PC/ResearchPlots/DT/1-FS_09_labeled.e57"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


def find_tip() -> QWidget | None:
    """Qt 的 tooltip 是个带 Qt::ToolTip 窗口标志的顶层 QLabel（QTipLabel）。

    注意不能直接 ``flags & Qt.ToolTip`` 判断真假：ToolTip 的枚举值本身含
    ``Window`` 位，普通窗口与它按位与也会非零，必须比「窗口类型掩码」。
    """
    for w in APP.topLevelWidgets():
        wtype = w.windowFlags() & Qt.WindowType_Mask
        if wtype == Qt.ToolTip and w.isVisible() and w.width() > 20:
            return w
    return None


def show_tip(widget, name: str) -> QWidget | None:
    """给控件发一个 ToolTip 事件，返回弹出的气泡窗口。"""
    from PySide6.QtCore import QTimer
    pt = QPoint(widget.width() // 2, widget.height() // 2)
    ev = QHelpEvent(QEvent.ToolTip, pt, widget.mapToGlobal(pt))
    QApplication.sendEvent(widget, ev)
    for _ in range(6):
        APP.processEvents()
    tip = find_tip()
    if tip is None:                     # 有的平台是异步弹的，再等一拍
        loop = QTimer(); loop.setSingleShot(True); loop.start(60)
        from PySide6.QtCore import QEventLoop
        e = QEventLoop(); loop.timeout.connect(e.quit); e.exec()
        APP.processEvents()
        tip = find_tip()
    if tip is not None:
        tip.grab().save(os.path.join(OUT, f"tip_{name}.png"))
    return tip


print("1) 打开文件")
win = MainWindow()
win.resize(1600, 1000)
win.show()
APP.processEvents()
win.load_file(GOOD)
APP.processEvents()
check("文件已打开", win.doc is not None and win.doc.total_points > 0,
      f"{win.doc.total_points if win.doc else 0:,} 点")

print("\n2) 逐个触发悬浮说明")
targets = [
    ("浏览", win.btn_nav), ("笔刷", win.btn_brush), ("框选", win.btn_box),
    ("实例", win.btn_inst), ("裁剪", win.btn_crop),
    ("语义", win.btn_semantic), ("穗", win._label_btns[1]),
    ("隐藏已标注", win.chk_hide_marked),
    ("拾取深度", win.cmb_depth), ("保存为", win.btn_save),
    ("关闭文件", win.btn_close),
]
tips = {}
for name, w in targets:
    tip = show_tip(w, name)
    tips[name] = tip
    got = tip is not None
    txt = tip.text() if tip is not None else ""
    check(f"「{name}」弹出气泡", got, f"{tip.width()}x{tip.height()}" if got else "")
    if got:
        check(f"「{name}」气泡文字与 tooltip 一致", txt == w.toolTip().replace("&", ""),
              txt.splitlines()[0][:24] if txt else "")
        # Office 浅色卡片 + 深色字：取左上角像素判断接近纯白
        img = tip.grab().toImage()
        c = img.pixelColor(min(4, img.width() - 1), min(4, img.height() - 1))
        check(f"「{name}」气泡为 Office 浅色底", c.lightness() > 200,
              f"rgb({c.red()},{c.green()},{c.blue()})")
    from PySide6.QtWidgets import QToolTip
    QToolTip.hideText()
    APP.processEvents()

print("\n3) 对话框里的按钮")
dfile = os.path.join(OUT, "ui_main_for_tips.png")
win.grab().save(dfile)
dlg = CropDialog(CropBox(-50, 50, -50, 50, 0, 100), 42194, 1200, win)
dlg.show()
APP.processEvents()
tip = show_tip(dlg.rb_remove, "裁剪_删除框内")
check("对话框单选框也能弹说明", tip is not None,
      tip.text().splitlines()[0] if tip is not None else "")
if tip is not None:
    c = tip.grab().toImage().pixelColor(4, 4)
    check("对话框气泡同样是浅色底（样式表级联到子窗口）", c.lightness() > 200,
          f"rgb({c.red()},{c.green()},{c.blue()})")
dlg.close()

print("\n4) 合成示意：侧栏按钮 + 真实气泡")
try:
    from PIL import Image
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QToolTip

    win.activateWindow()
    APP.processEvents()
    shot = win.grab().toImage()
    shot.save(os.path.join(OUT, "ui_tooltip_base.png"))

    pairs = []
    for name, w in (("笔刷", win.btn_brush), ("框选", win.btn_box),
                    ("保存为", win.btn_save)):
        tip = show_tip(w, name)
        APP.processEvents()
        if tip is None:
            continue
        tp = os.path.join(OUT, f"tip_{name}.png")
        p = w.mapTo(win, QPoint(0, 0))
        pairs.append((tp, p.x(), p.y(), tip.width(), tip.height()))
        QToolTip.hideText()
        APP.processEvents()

    base = Image.open(os.path.join(OUT, "ui_tooltip_base.png")).convert("RGB")
    side_left = win.side_scroll.mapTo(win, QPoint(0, 0)).x()
    for tp, x, y, tw, th in pairs[:1]:          # 只叠一个，避免糊成一片
        timg = Image.open(tp).convert("RGB")
        # 摆在侧栏左侧、与按钮垂直居中：既盖住视图（不遮按钮），又贴着被说明的控件
        px = max(4, side_left - timg.width - 12)
        py = max(4, min(y + (th - timg.height) // 2, base.height - timg.height - 4))
        base.paste(timg, (px, py))
    base.save(os.path.join(OUT, "ui_tooltip_demo.png"))
    check("合成了「按钮 + 悬浮气泡」示意图", os.path.exists(
        os.path.join(OUT, "ui_tooltip_demo.png")), "outputs/ui_tooltip_demo.png")
except Exception as e:                    # noqa: BLE001  Pillow 不在就跳过
    print(f"   (跳过合成：{e})")

print("\n5) 拼接总览图")
try:
    from PIL import Image
    shots = [("tip_" + n + ".png", n) for n in
             ("笔刷", "框选", "穗", "保存为", "关闭文件")]
    imgs = [(n, Image.open(os.path.join(OUT, f))) for f, n in shots
            if os.path.exists(os.path.join(OUT, f))]
    if imgs:
        pad, label_h = 16, 0
        W = max(im.width for _n, im in imgs) + pad * 2
        H = sum(im.height + pad for _n, im in imgs) + pad
        canvas = Image.new("RGB", (W, H), (30, 33, 38))
        y = pad
        for n, im in imgs:
            canvas.paste(im, (pad, y))
            y += im.height + pad
        canvas.save(os.path.join(OUT, "tips_all.png"))
        print(f"   -> outputs/tips_all.png  {W}x{H}")
        ok = True
except Exception as e:                    # noqa: BLE001  Pillow 不在就跳过
    print(f"   (跳过拼接：{e})")

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
