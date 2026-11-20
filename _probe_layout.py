"""布局核对 + 出图：各档窗口尺寸下侧栏宽度、滚动、分组是否完整。"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QGroupBox, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
APP = QApplication(sys.argv[:1])

for f in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.isfile(f):
        fid = QFontDatabase.addApplicationFont(f)
        fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
        if fams:
            APP.setFont(QFont(fams[0], 9))
            break

from main_window import MainWindow

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
REAL = r"D:\Data\PC\ResearchPlots\DT\1-FS_09_labeled.e57"

for w, h, name in ((2560, 1440, "max2560"), (1920, 1080, "max1920"),
                   (1366, 768, "small1366"), (1000, 640, "tiny1000")):
    win = MainWindow()
    win.settings = QSettings(os.path.join(OUT, "_probe.ini"), QSettings.IniFormat)
    win.resize(w, h)
    win.show()
    APP.processEvents()
    win.load_file(REAL)
    APP.processEvents()
    panel = win._split.widget(1)
    scroll = win.side_scroll
    body, vp = scroll.widget(), scroll.viewport()
    groups = body.findChildren(QGroupBox)
    squashed = [g.title() for g in groups if g.height() < 40]
    print(f"{name:9s} {w}x{h}: 侧栏 {panel.width()}px（占 {panel.width()/max(w,1)*100:.0f}%）| "
          f"内容 {body.height()} / 可见 {vp.height()} | "
          f"{'需滚动' if body.height() > vp.height() else '无需滚动'} | "
          f"挤没的分组 {squashed or '无'}")
    win.grab().save(os.path.join(OUT, f"ui_layout_{name}.png"))
    win.close()
