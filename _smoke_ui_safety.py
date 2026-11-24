"""离屏回归：保存提示、双页退出保护、裁剪页关闭/包围盒、点大小数值。"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent, QFontDatabase, QImage, QPainter
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

import cloudio
from main_window import MainWindow


def run():
    with tempfile.TemporaryDirectory(prefix='pointcloud_ui_safety_') as folder:
        root = Path(folder)
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, folder)
        app = QApplication.instance() or QApplication([])
        for font_name in ('msyh.ttc', 'msyhbd.ttc', 'segoeui.ttf'):
            font_path = Path('C:/Windows/Fonts') / font_name
            if font_path.is_file():
                QFontDatabase.addApplicationFont(str(font_path))
        messages = []
        errors = []
        with patch.object(QMessageBox, 'information', side_effect=lambda *a, **k: messages.append(a)), \
             patch.object(QMessageBox, 'critical', side_effect=lambda *a, **k: errors.append(a)):
            window = MainWindow()
            window.resize(1600, 1000)
            window.show()
            app.processEvents()
            crop = window.crop_page
            checks = 0

            def check(name, condition):
                nonlocal checks
                assert condition, name
                checks += 1
                print('PASS:', name)

            fixture = root / 'input.ply'
            xyz = np.array([[x, y, z] for x in np.linspace(-1, 1, 10)
                            for y in np.linspace(-1, 1, 10) for z in (0., .5)], np.float32)
            fixture.write_text('ply\nformat ascii 1.0\nelement vertex 200\n'
                               'property float x\nproperty float y\nproperty float z\nend_header\n'
                               + '\n'.join(' '.join(map(str, row)) for row in xyz), encoding='ascii')

            for page in (window, crop):
                check('initial point size 1.4', page.lbl_point_size.text() == '1.4')
                for value in (page.sld_size.minimum(), 27, page.sld_size.maximum()):
                    page.sld_size.setValue(value)
                    check('slider/value/renderer agree',
                          page.lbl_point_size.text() == f'{value / 10:.1f}'
                          and page.view.renderer.point_size == value / 10)
            check('close disabled without document', not crop.btn_close.isEnabled())
            crop.load_file(str(fixture))
            crop._folder = folder
            crop._rescan()
            check('close enabled after loading', crop.btn_close.isEnabled())
            check('crop document starts clean', not crop.has_unsaved_changes())

            window.switch_page(1)
            app.processEvents()
            for width, height in ((1920, 1080), (1366, 768), (1000, 640)):
                window.resize(width, height)
                app.processEvents()
                corner = crop.btn_close.mapTo(window, crop.btn_close.rect().bottomRight())
                check(f'close button stays visible at {width}x{height}', window.rect().contains(corner))
            window.resize(1600, 1000)
            app.processEvents()
            crop.chk_bbox.setChecked(True)
            check('bbox toggle updates view', crop.view.show_bbox)
            painted = QImage(crop.view.size(), QImage.Format_ARGB32)
            painted.fill(Qt.transparent)
            painter = QPainter(painted)
            with patch.object(crop.view, '_draw_bbox', wraps=crop.view._draw_bbox) as draw:
                crop.view._draw_overlay(painter)
                check('bbox actually drawn', draw.call_count == 1)
                crop.chk_bbox.setChecked(False)
                crop.view._draw_overlay(painter)
                check('bbox hidden when unchecked', draw.call_count == 1)
            painter.end()
            window.switch_page(0)

            # 退出必须检查隐藏页面。取消时保留文件和几何。
            crop.spn_rz.setValue(25)
            crop.apply_rotation()
            original = crop.doc.merged_xyz().copy()
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Cancel) as ask:
                event = QCloseEvent()
                window.closeEvent(event)
                check('hidden crop page blocks exit on cancel', not event.isAccepted() and ask.call_count == 1)
                check('prompt identifies crop file', str(fixture) in ask.call_args.args[2])
                crop.btn_close.click()
                check('cancel close preserves data', crop.doc is not None and np.array_equal(original, crop.doc.merged_xyz()))
                crop.reload()
                check('cancel reload preserves changes', crop._dirty and np.array_equal(original, crop.doc.merged_xyz()))

            # 选择保存但取消路径、写入失败，都不能退出或改变当前路径。
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Save), \
                 patch.object(QFileDialog, 'getSaveFileName', return_value=('', '')):
                event = QCloseEvent()
                window.closeEvent(event)
                check('cancel save dialog blocks exit', not event.isAccepted() and crop._dirty)
            failed_path = str(root / 'failed.ply')
            before_path = crop.path
            before_messages = len(messages)
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Save), \
                 patch.object(QFileDialog, 'getSaveFileName', return_value=(failed_path, 'PLY 点云 (*.ply)')), \
                 patch.object(cloudio, 'save', side_effect=OSError('simulated write error')):
                event = QCloseEvent()
                window.closeEvent(event)
                check('failed save blocks exit and retains original path',
                      not event.isAccepted() and crop._dirty and crop.path == before_path)
                check('failed save has error but no success popup', bool(errors) and len(messages) == before_messages)

            saved = str(root / 'saved.ply')
            check('write succeeds', crop._write(saved))
            check('save popup contains success/path/point count',
                  messages[-1][1] == '保存成功' and saved in messages[-1][2] and '200' in messages[-1][2])
            check('successful save updates file and clears dirty state', crop.path == saved and not crop.has_unsaved_changes())
            reread = cloudio.load(saved)
            check('saved geometry can be reopened', np.allclose(reread.merged_xyz(), original))

            # 保存后撤销到旧状态同样属于未保存，不能因撤销栈空而漏报。
            crop.undo()
            check('undo after save is dirty even with empty stack', crop._dirty and not crop._undo)
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Cancel):
                event = QCloseEvent()
                window.closeEvent(event)
                check('undo after save still blocks exit', not event.isAccepted())

            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Discard):
                crop.btn_close.click()
            check('close clears document, path, view, history and labels',
                  crop.doc is None and crop.ann is None and crop.path is None
                  and not crop._undo and not crop.view.has_points()
                  and not crop.btn_close.isEnabled() and crop.lbl_path.toolTip() == '')
            check('close preserves folder list', crop.lst_files.count() > 0)
            crop.load_file(str(fixture))
            check('can reopen after closing', crop.doc.total_points == 200)

            # 没点应用的旋转预览也会保存进文件，退出前需要提醒。
            crop.spn_rx.setValue(30)
            crop.btn_rot_preview.setChecked(True)
            check('unapplied preview detected as unsaved', crop.has_unsaved_changes() and not crop._dirty)
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Cancel):
                event = QCloseEvent()
                window.closeEvent(event)
                check('rotation preview blocks exit', not event.isAccepted())
            preview_file = str(root / 'preview_saved.ply')
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Save), \
                 patch.object(QFileDialog, 'getSaveFileName', return_value=(preview_file, 'PLY 点云 (*.ply)')):
                event = QCloseEvent()
                window.closeEvent(event)
                check('saving preview allows exit', event.isAccepted() and Path(preview_file).is_file()
                      and not crop.has_unsaved_changes())

            # 标注页保存成功提示，以及两页同时有改动时分别检查。
            window.load_file(str(fixture))
            window.ann.labels[0][0] = 1
            window._dirty = True
            ann_saved = str(root / 'annotated.ply')
            window._do_save(ann_saved)
            check('annotation save displays success popup',
                  '保存成功' in messages[-1][2] and ann_saved in messages[-1][2] and not window._dirty)
            window.load_file(ann_saved)
            check('opening an annotated file is clean', not window._dirty and window.ann.total_marked() == 1)
            window._dirty = crop._dirty = True
            with patch.object(QMessageBox, 'question', side_effect=[QMessageBox.Discard, QMessageBox.Cancel]) as ask:
                event = QCloseEvent()
                window.closeEvent(event)
                check('both documents checked; second cancel blocks exit', not event.isAccepted() and ask.call_count == 2)
            with patch.object(QMessageBox, 'question', side_effect=[QMessageBox.Discard, QMessageBox.Discard]):
                event = QCloseEvent()
                window.closeEvent(event)
                check('explicitly discarding both allows exit', event.isAccepted())
            window._dirty = crop._dirty = False
            crop.reset_rotation()
            window.close()
            app.processEvents()
            print(f'All {checks} checks passed.')


if __name__ == '__main__':
    run()
