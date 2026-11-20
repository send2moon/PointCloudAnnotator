"""程序入口。

用法
----
    python app.py                    # 空白启动
    python app.py path/to/a.e57      # 直接打开文件
"""

from __future__ import annotations

import os
import sys
import traceback


def _excepthook(exc_type, exc, tb) -> None:
    """未捕获异常时弹窗提示，避免 GUI 静默退出。"""
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    sys.stderr.write(text)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "程序异常", text[-1800:])
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    # 必须在 QApplication 之前设置
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(argv)
    # 英文标识只用于配置路径（QSettings），改掉会丢失 last_dir / 拾取深度等偏好
    app.setApplicationName("PointCloudAnnotator")
    app.setOrganizationName("PointCloudAnnotator")
    app.setApplicationDisplayName("作物点云处理软件")

    # run.bat 必须保持纯 ASCII（cmd 按 GBK 读批处理，中文会被乱码解析），
    # 所以启动器控制台的中文标题由这里用宽字符 API 设置。
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW("作物点云处理软件")
        except Exception:  # noqa: BLE001  无控制台（双击启动）时忽略
            pass

    sys.excepthook = _excepthook

    from main_window import MainWindow
    win = MainWindow()
    # 启动即最大化：标注是"看大图"的活，窗口越宽越容易定位到单株。
    # 取消最大化后的窗口大小取自上次关闭时记录（MainWindow._restore_last_rect）。
    win.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
