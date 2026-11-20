"""生成 2.0 Office 风格界面截图（两个功能页各一张）。

    python _shot_office.py
产物：outputs/office/ui_annot_page.png、outputs/office/ui_crop_page.png
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

APP = QApplication([])
QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")

from main_window import MainWindow

OUT = os.path.join(HERE, "outputs", "office")
os.makedirs(OUT, exist_ok=True)

ANNOT_FILE = r"D:\Data\PC\ResearchPlots\DT\1-FS_09_labeled.e57"
CROP_FILE = r"D:\Data\PC\Labeled\5.13FS\5.13DT4 - Cloud_scaled_labeled_full.ply"

MODE = sys.argv[1] if len(sys.argv) > 1 else "annot"   # annot | crop

win = MainWindow()
win.resize(1920, 1080)
win.show()
APP.processEvents()

if MODE == "annot":
    if os.path.isfile(ANNOT_FILE):
        win._ask_force_open = lambda err: False
        win.load_file(ANNOT_FILE)
        win.set_label(1)
        APP.processEvents()
    win.grab().save(os.path.join(OUT, "ui_annot_page.png"))
    print("annot page ->", os.path.join(OUT, "ui_annot_page.png"))
else:
    win.switch_page(1)
    APP.processEvents()
    cp = win.crop_page
    cp._folder = os.path.dirname(CROP_FILE)
    cp._rescan()
    APP.processEvents()
    if not os.path.isfile(CROP_FILE):
        print("WARN 裁剪页样本不存在：", CROP_FILE)
    if os.path.isfile(CROP_FILE):
        cp.load_file(CROP_FILE)
        cp.set_view("等轴")
        APP.processEvents()
        # 选中「任意多边形」，让右侧选中态与实际画出的区域一致
        cp.rb_poly.setChecked(True)
        APP.processEvents()
        # 画一块示例多边形，直观展示「任意形状裁剪区域」
        w, h = cp.view.width(), cp.view.height()
        poly = ("polygon", [(int(w * .30), int(h * .28)), (int(w * .67), int(h * .24)),
                            (int(w * .74), int(h * .52)), (int(w * .62), int(h * .76)),
                            (int(w * .33), int(h * .70))])
        cp.view._region = poly
        cp._on_region(poly)
        cp.view.update()
        APP.processEvents()
    win.grab().save(os.path.join(OUT, "ui_crop_page.png"))
    print("crop page  ->", os.path.join(OUT, "ui_crop_page.png"))
# 截图脚本跑在 offscreen 非交互环境：关闭前清掉未保存标记，
# 否则 closeEvent 会弹「是否放弃改动」的模态框把进程挂住。
win._dirty = False
win.close()
sys.exit(0)
