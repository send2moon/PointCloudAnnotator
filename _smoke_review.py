"""Review regression with real Qt interaction and optional real upstream inference.

Run with --model to also test the downloaded checkpoint and persistent GPU worker.
"""
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import numpy as np
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtGui import QCloseEvent, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from annotation import AnnotationSet
import cloudio
from e57io import E57Document
from main_window import MainWindow
from review_session import ReviewSession
from point_sam_worker import sample_indices


def run():
    checks = 0

    def check(name, condition):
        nonlocal checks
        assert condition, name
        checks += 1
        print('PASS:', name, flush=True)

    with tempfile.TemporaryDirectory(prefix='review_smoke_') as directory:
        root = Path(directory)
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, directory)
        app = QApplication([])
        for font in ('msyh.ttc', 'msyhbd.ttc', 'segoeui.ttf'):
            QFontDatabase.addApplicationFont('C:/Windows/Fonts/' + font)
        rng = np.random.default_rng(42)
        doc = E57Document()
        for sid in range(2):
            xyz = rng.uniform([-1 + sid * 3, -1, 0], [1 + sid * 3, 1, 1], (512, 3)).astype(np.float32)
            scan = cloudio._scan_from_columns({'x':xyz[:, 0], 'y':xyz[:, 1], 'z':xyz[:, 2],
                'r':np.full(512, 70), 'g':np.full(512, 170), 'b':np.full(512, 60)},
                {'nx': np.full(512, .25, np.float64)}, {'nx': 'f8'}, f'Scan{sid}')
            doc.scans.append(scan)
        ann = AnnotationSet.empty([512, 512])
        ann.labels[0][:50], ann.instances[0][:50] = 1, 1
        ann.labels[1][:30], ann.instances[1][:30] = 2, 2
        ann.instance_info = {1:(1, 0), 2:(2, 1)}
        ann.next_instance = 3
        original = doc.merged_xyz().copy()
        fixture = str(root / 'labeled.ply')
        cloudio.save(fixture, doc, ann)
        session = ReviewSession.copy_from(doc, fixture, ann)
        mask = np.zeros(1024, bool)
        mask[25:75] = True
        check('fill modifies only unannotated points', session.apply(mask, 2) == 25)
        check('existing labels and instances preserved by fill',
              np.all(session.ann.labels[0][:50] == 1) and np.all(session.ann.instances[0][:50] == 1))
        check('fill undo restores clean state', session.undo() and not session.dirty and session.ann.total_marked() == 80)
        check('overwrite revises category and clears old instance in semantic mode',
              session.apply(mask, 3, 'overwrite') == 50 and np.all(session.ann.labels[0][25:75] == 3)
              and not session.ann.instances[0][25:75].any())
        session.undo()
        check('replace instance clears old outside and includes new boundary',
              session.apply(mask, 1, 'replace_instance', target=(0, 1)) == 50
              and np.all(session.ann.instances[0][25:75] == 1)
              and not session.ann.labels[0][:25].any())
        session.undo()
        mask[600:650] = True
        try:
            session.apply(mask, 1, 'overwrite', 'existing', (0, 1))
            rejected = False
        except ValueError:
            rejected = True
        check('cross-scan instance rejected without partial modification', rejected and not session.dirty)
        session.apply(mask, 4, 'overwrite', 'new')
        check('new IDs unique across scans', len(set(np.unique(session.ann.instances[0][25:75]))) == 1
              and session.ann.instances[0][25] != session.ann.instances[1][100])
        corrected = str(root / 'corrected.ply')
        session.save(corrected)
        read = cloudio.load(corrected)
        restored = AnnotationSet.from_scans(read.scans)
        check('PLY roundtrip keeps coordinates and extra dtype', np.array_equal(read.merged_xyz(), original)
              and read.scans[0].points['nx'].dtype == np.float64)
        check('saved annotation labels and instances roundtrip',
              all(np.array_equal(a, b) for a, b in zip(restored.labels, session.ann.labels))
              and all(np.array_equal(a, b) for a, b in zip(restored.instances, session.ann.instances)))
        check('undo after saving is unsaved', session.undo() and session.dirty)
        e57 = str(root / 'corrected.e57')
        session.save(e57)
        re57 = cloudio.load(e57)
        check('E57 annotation roundtrip keeps scan structure', len(re57.scans) == 2
              and AnnotationSet.from_scans(re57.scans).total_marked() == 80)
        check('sampling retains all prompts with stable original indices',
              set([10, 900]).issubset(set(sample_indices(1000, 256, [10, 900])))
              and np.array_equal(sample_indices(1000, 256, [10, 900]), sample_indices(1000, 256, [10, 900])))
        messages, errors = [], []
        with patch.object(QMessageBox, 'information', side_effect=lambda *a, **k: messages.append(a)), \
             patch.object(QMessageBox, 'warning', side_effect=lambda *a, **k: errors.append(a)), \
             patch.object(QMessageBox, 'critical', side_effect=lambda *a, **k: errors.append(a)):
            win = MainWindow()
            win.resize(1920, 1080)
            win.show()
            app.processEvents()
            page = win.review_page
            check('three functional pages and review navigation exist', win._pages.count() == 3 and win.btn_page_review.text() == '标注核查')
            win.load_file(fixture)
            win._dirty = True
            win._import_for_review()
            check('import switches to review and copies unsaved source', win._page == 2 and page.dirty)
            page.session.ann.labels[0][100] = 4
            check('review edits do not mutate annotation page', win.ann.labels[0][100] == 0)
            page.session.ann.labels[0][100] = 0
            check('shortcut isolation across three pages',
                  all(s.isEnabled() for s in page._shortcuts)
                  and not any(s.isEnabled() for s in win.crop_page._shortcuts)
                  and not win.act_save.isEnabled())
            win._dirty = False
            page.session.saved_revision = 0
            for row in range(page.lst.count()):
                if page.lst.item(row).data(Qt.UserRole) == ('instance', 0, 1):
                    page.lst.setCurrentRow(row)
                    break
            app.processEvents()
            check('selecting instance restricts scan and highlights original target',
                  page.cmb_scan.currentData() == 0 and page._target_mask().sum() == 50)
            x, y, z = page.view.renderer.project_points(page.view.renderer.xyz, page.view.width(), page.view.height())
            candidates = np.flatnonzero((x > 15) & (x < page.view.width() - 15)
                & (y > 15) & (y < page.view.height() - 15) & (z > 0)
                & (page.view.renderer.scan_ids == 0))
            page.mode_buttons[2].click()
            idx = candidates[0]
            QTest.mouseClick(page.view, Qt.LeftButton, Qt.NoModifier, QPoint(int(x[idx]), int(y[idx])))
            check('real mouse positive prompt reaches original point index', len(page.prompts) == 1
                  and page.prompts[0][1] == 1 and 0 <= page.prompts[0][0] < 512)
            check('adding prompts does not alter annotations', page.session.ann.total_marked() == 80 and not page.dirty)
            page.add_prompt(int(candidates[-1]), 0)
            check('positive and negative prompts supported', len(page.prompts) == 2 and page.prompts[-1][1] == 0)
            page.pop_prompt()
            check('undo prompt invalidates outdated prediction', len(page.prompts) == 1 and page.preview is None)
            page.clear_prompts()
            w, h = page.view.width(), page.view.height()
            page.view._region = ('rect', (w * .25, h * .25, w * .75, h * .75))
            page.view._lock_view_for_region()
            page.use_manual_region()
            check('manual region is preview only', page.preview is not None and page.preview.any() and not page.dirty)
            page.cmb_label.setCurrentIndex(page.cmb_label.findData(3))
            page.apply_preview()
            check('applying fill marks review dirty with undo record', page.dirty and bool(page.session.history))
            page.undo()
            check('UI undo restores loaded annotation and clean state', not page.dirty and page.session.ann.total_marked() == 80)
            page.clear_region()
            if '--model' in sys.argv:
                # Real upstream checkpoint, subprocess IPC, original-index propagation.
                page.add_prompt(int(candidates[candidates >= 100][0]), 1)
                page.add_prompt(int(candidates[-1]), 0)
                page.spn_samples.setValue(256)
                page.run_model()
                started = time.monotonic()
                while page.runner.busy and time.monotonic() - started < 180:
                    app.processEvents()
                    QTest.qWait(20)
                check('real Point-SAM subprocess returns preview', not page.runner.busy and page.preview is not None)
                check('candidate masks remain within selected scan', not page.preview[512:].any()
                      and page.candidates.shape[1] == 512)
                check('real output respects clicked positive/negative points',
                      page.preview[page.prompts[0][0]] and not page.preview[page.prompts[-1][0]])
                check('inference alone does not edit labels', not page.dirty and page.session.ann.total_marked() == 80)
                print('GPU inference log:', page.log.toPlainText()[-400:], flush=True)
                model_mask = page.preview.copy()
                previous_labels = page.session.flat_labels().copy()
                expected = previous_labels.copy()
                expected[model_mask & (previous_labels == 0)] = 3
                page.apply_preview()
                check('real predicted mask application updates only original selected indices',
                      page.dirty and np.array_equal(page.session.flat_labels(), expected))
                ui_output = str(root / 'ui_model_result.ply')
                page._write(ui_output)
                check('UI model workflow saves and reopens without geometry change',
                      np.array_equal(cloudio.load(ui_output).merged_xyz(), original)
                      and '保存成功' in messages[-1][1])
                page.add_prompt(int(candidates[candidates >= 100][0]), 1)
                page.run_model()
                page.runner.cancel()
                app.processEvents()
                check('cancel releases worker and cannot apply stale prediction',
                      not page.runner.busy and page.runner.proc is None and page.preview is None)
                check('busy detection does not overwrite request type', page.request_kind == 'predict')
            # Confirm cancel/save failure/close handling without touching source files.
            page.preview = np.zeros(1024, bool)
            page.preview[200:220] = True
            page.apply_preview()
            win.switch_page(0)
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Cancel):
                event = QCloseEvent()
                win.closeEvent(event)
                check('hidden dirty review page prevents software exit', not event.isAccepted())
            old_path = page.path
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Save), \
                 patch.object(QFileDialog, 'getSaveFileName', return_value=('', '')):
                check('cancel save blocks discard', not page.confirm_discard() and page.dirty)
            with patch.object(cloudio, 'save', side_effect=OSError('simulated save failure')):
                check('failed save keeps original path and dirty state',
                      not page._write(str(root / 'failure.ply')) and page.path == old_path and page.dirty)
            win.switch_page(2)
            for width, height in ((1920, 1080), (1366, 768), (1000, 640)):
                win.resize(width, height)
                app.processEvents()
                check(f'save/close accessible at {width}x{height}', win.rect().contains(page.btn_close.mapTo(win, page.btn_close.rect().bottomRight())))
            page.sld_size.setValue(27)
            check('review point size value stays synchronized', page.lbl_point_size.text() == '2.7' and page.view.renderer.point_size == 2.7)
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Discard):
                page.close_file()
            check('review close clears view and independent state', page.session is None and not page.view.has_points())
            win._dirty = False
            win.close()
            app.processEvents()
        print(f'All {checks} review checks passed.', flush=True)


if __name__ == '__main__':
    run()
