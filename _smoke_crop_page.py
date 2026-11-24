"""点云裁剪页回归测试（离屏）。

覆盖：双页面结构 / Office 主题 / 文件夹列表 / 六种视角 /
矩形与多边形裁剪（保留内外、深度、撤销、重载）/ 旋转预览与应用 /
覆盖保存与另存为 / 配准脚本参数校验 / 全部控件悬浮说明。

运行：
    python _smoke_crop_page.py
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QFontDatabase, QPainterPath, QPolygonF
from PySide6.QtWidgets import (QApplication, QAbstractButton, QComboBox,
                               QDoubleSpinBox, QFileDialog, QMessageBox,
                               QSlider, QSpinBox, QToolButton, QWidgetAction)

APP = QApplication([])
QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")

import cloudio
from annotation import AnnotationSet
from crop_view import VIEW_PRESETS
from e57io import E57Document, Scan
from renderer import point_in_polygon
from main_window import APP_ID, APP_NAME, MODULE_ANNOT, MODULE_CROP, DARK_QSS, MainWindow
from office_theme import CANVAS_COLORS, OFFICE_QSS, TIP_BG_RGB

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


def make_ply(path: str, n: int = 4000) -> E57Document:
    """合成一块 100×100×20 的点云（含颜色、强度、标注列）。"""
    rng = np.random.default_rng(20261118)
    x = rng.uniform(-50, 50, n)
    y = rng.uniform(-50, 50, n)
    z = rng.uniform(0, 20, n)
    doc = E57Document()
    s = Scan(name="Synthetic")
    s.points = {
        "cartesianX": x.astype(np.float32),
        "cartesianY": y.astype(np.float32),
        "cartesianZ": z.astype(np.float32),
        "colorRed": rng.integers(0, 256, n, dtype=np.uint8),
        "colorGreen": rng.integers(0, 256, n, dtype=np.uint8),
        "colorBlue": rng.integers(0, 256, n, dtype=np.uint8),
        "intensity": rng.random(n).astype(np.float32),
    }
    doc.scans = [s]
    ann = AnnotationSet.empty([n])
    ann.labels[0][: n // 5] = 2
    cloudio.save(path, doc, ann)
    return doc


tmp = tempfile.mkdtemp(prefix="crop_page_smoke_")
P_PLY = os.path.join(tmp, "plot_A.ply")
P_SRC_E57 = os.path.join(tmp, "src_B.e57")
make_ply(P_PLY, 4000)
# 再放一个 E57：验证输出路径自动推导 + 点云2 下拉
cloudio.load(P_PLY).save(P_SRC_E57)
P_E57 = os.path.join(tmp, "plot_A.e57")

print("0) 软件名 / 主题 / 双页面结构")
check("软件名已改为作物点云处理软件", APP_NAME == "作物点云处理软件", APP_NAME)
check("内部标识保持不变（QSettings 不丢）", APP_ID == "PointCloudAnnotator", APP_ID)
check("主题换成 Office 浅色", "#2b579a" in OFFICE_QSS and "#1e2126" not in OFFICE_QSS)
check("深色 QSS 旧名仍是别名", DARK_QSS == OFFICE_QSS)
check("tooltip 气泡为浅色", f"background-color: rgb{TIP_BG_RGB}" in OFFICE_QSS.replace("(", "(")
      or f"background-color: #{TIP_BG_RGB[0]:02x}{TIP_BG_RGB[1]:02x}{TIP_BG_RGB[2]:02x}" in OFFICE_QSS)

win = MainWindow()
win.resize(1600, 1000)
win.show()
APP.processEvents()
check("中心部件是三页容器", win._pages.count() == 3)
check("默认停在标注页", win._pages.currentIndex() == 0)
check("导航条有两个功能页按钮",
      win.btn_page_annot.text() == MODULE_ANNOT and win.btn_page_crop.text() == MODULE_CROP)
check("窗口标题带模块名", MODULE_ANNOT in win.windowTitle(), win.windowTitle())
check("裁剪页快捷键初始禁用",
      not any(s.isEnabled() for s in win.crop_page._shortcuts))

win.switch_page(1)
APP.processEvents()
cp = win.crop_page
check("切到裁剪页", win._pages.currentIndex() == 1 and MODULE_CROP in win.windowTitle())
check("功能区按页面切换", not win._tb_annot.isVisible() and win._tb_crop.isVisible())
check("裁剪页快捷键已启用", all(s.isEnabled() for s in cp._shortcuts))
# 只统计「纯 QAction」：QToolBar.addWidget 会把裁剪功能区按钮与导航选项卡
# 包成 QWidgetAction，它们必须保持可用（否则裁剪页按钮全灰、页签点不动）。
_plain = [a for a in win.findChildren(type(win.act_undo))
          if not isinstance(a, QWidgetAction)]
check("标注页动作在裁剪页全禁用", all(not a.isEnabled() for a in _plain),
      f"{sum(a.isEnabled() for a in _plain)} 个仍可用")
check("裁剪功能区按钮未被误禁用（QWidgetAction 不参与禁用）",
      all(b.isEnabled() for b in win._tb_crop.findChildren(QToolButton) if b.text()))
check("顶部两个页签仍可点（能切回去）",
      win.btn_page_annot.isEnabled() and win.btn_page_crop.isEnabled())
win.switch_page(0)
APP.processEvents()
check("切回标注页恢复功能区", win._tb_annot.isVisible() and not win._tb_crop.isVisible())
check("切回标注页关闭裁剪页快捷键",
      not any(s.isEnabled() for s in cp._shortcuts))
check("标注页也有画布颜色下拉", hasattr(win, "cmb_canvas") and win.cmb_canvas.count() >= 4,
      str(win.cmb_canvas.count()))
win._on_canvas_changed("浅灰")
check("标注页画布可切换", tuple(win.view.renderer.bg) == tuple(CANVAS_COLORS["浅灰"]),
      str(win.view.renderer.bg))
win._on_canvas_changed("深色（默认）")
win.switch_page(1)
APP.processEvents()

print("\n1) 文件夹列表")
cp._folder = tmp
cp._rescan()
check("列表收集到两个点云文件", cp.lst_files.count() == 2, str(cp.lst_files.count()))
check("点云1 下拉只列 PLY", cp.cmb_ref.count() == 2 and
      cp.cmb_ref.itemData(1).endswith(".ply"))
check("点云2 下拉只列 E57", cp.cmb_src.count() == 2 and
      cp.cmb_src.itemData(1).endswith(".e57"))
check("输出路径由点云2 自动推导", cp.edt_out.text().endswith("src_B_cropped.e57"),
      cp.edt_out.text())

cp.load_file(P_PLY)
APP.processEvents()
d0 = cp.doc.total_points
check("文件已载入", d0 == 4000 and cp.view.n_points() == d0, f"{d0:,}")
check("已有标注被识别", int(cp.ann.labels[0].sum()) > 0)
check("状态信息显示点数", "4,000" in cp.lbl_info.text(), cp.lbl_info.text())
check("列表选中项同步", cp.lst_files.currentItem() is not None)

print("\n2) 六种预设视角")
for name, (yaw, pitch) in VIEW_PRESETS.items():
    cp.set_view(name)
    APP.processEvents()
    cam = cp.view.renderer.camera
    check(f"{name} 相机角度正确",
          abs(cam.yaw - yaw) < 1e-6 and abs(cam.pitch - pitch) < 1e-6,
          f"yaw={cam.yaw} pitch={cam.pitch}")
cp.set_view("俯视 (XY)")
APP.processEvents()

print("\n2b) 浏览模式 / 坐标轴 / 色条 / 画布颜色")
check("默认进入浏览模式（左键拖动可旋转）",
      cp.view.mode == "nav" and cp.rb_nav.isChecked(), cp.view.mode)
for mode, want in (("height", "height"), ("intensity", "intensity"), ("label", "label")):
    cp.view.set_color_mode(mode)
    cp.view._do_render()
    leg = cp.view.renderer.legend()
    if want == "label":
        check("标注着色提供分类色条", leg and leg[0] == "label", str(leg))
    else:
        check(f"{want} 着色提供色条范围",
              leg and leg[0] == want and leg[1] < leg[2], str(leg))
cp.view.set_color_mode("rgb")
cp.view._do_render()
check("真彩色不画色条", cp.view.renderer.legend() is None)
cp.view.set_canvas_rgb((12, 34, 56))
check("裁剪页画布可自定义颜色", tuple(cp.view.renderer.bg) == (12, 34, 56),
      str(cp.view.renderer.bg))
cp.view.set_canvas("深色（默认）")
cp.rb_rect.setChecked(True)
APP.processEvents()
check("切到矩形画框模式", cp.view.mode == "rect", cp.view.mode)

print("\n3) 矩形裁剪 / 保留方式 / 深度 / 撤销")
w, h = cp.view.width(), cp.view.height()
rect = ("rect", (int(w * 0.30), int(h * 0.30), int(w * 0.70), int(h * 0.70)))
cp.view._region = rect
cp._on_region(rect)
m = cp.view.region_mask(rect, "all")
check("矩形区域命中一部分点", 0 < m.sum() < d0, f"{int(m.sum()):,} / {d0:,}")
check("区域标签显示命中数", "命中" in cp.lbl_region.text(), cp.lbl_region.text())

cp.rb_inside.setChecked(True)
cp.apply_crop()
APP.processEvents()
n_in = cp.doc.total_points
check("只保留区域内 → 点数变少", 0 < n_in < d0, f"{d0:,} → {n_in:,}")
check("裁剪后标注列同步缩短", len(cp.ann.labels[0]) == n_in)
check("裁剪后视图点数同步", cp.view.n_points() == n_in)
check("裁剪后区域自动清空", cp.view.region() is None)
check("裁剪后标记为未保存", cp._dirty)

cp.undo()
APP.processEvents()
check("撤销回到原始点数", cp.doc.total_points == d0)

# 保留区域外
cp.view._region = rect
cp._on_region(rect)
cp.rb_outside.setChecked(True)
cp.apply_crop()
APP.processEvents()
n_out = cp.doc.total_points
check("删除区域内、保留区域外 → 点数变少", 0 < n_out < d0, f"{d0:,} → {n_out:,}")
check("两种保留方式互补（含视口外点）", n_in + n_out == d0,
      f"{n_in:,} + {n_out:,} vs {d0:,}")
check("切换保留方式不影响形状选择",
      cp.rb_rect.isChecked() and cp.view.mode == "rect", cp.view.mode)
cp.undo()
APP.processEvents()
check("第二次撤销回到原始点数", cp.doc.total_points == d0)

# 3b) 区域几何一致性 —— 专防「多边形裁剪和标记区域不一致」。
# 早期 _inside_polygon 用 np.ones 起算，内外完全颠倒，但旧断言只检查
# 「命中一部分点」所以能蒙混过关。这里用 Qt 的 OddEvenFill 填充规则做
# 独立参照，逐点比对，内外颠倒 / 边界错位都会被直接抓出来。
print("\n3b) 多边形区域与掩膜一致性（Qt OddEvenFill 参照）")


def _qt_inside(pts, poly_pts):
    path = QPainterPath()
    path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in poly_pts]))
    return np.array([path.contains(QPointF(float(x), float(y)))
                     for x, y in pts], bool)


rng = np.random.default_rng(20261119)
probe_pts = rng.uniform(-3, 13, (1500, 2))
for name, shape in (("正方形逆时针", [(0, 0), (10, 0), (10, 10), (0, 10)]),
                    ("正方形顺时针", [(0, 0), (0, 10), (10, 10), (10, 0)]),
                    ("凹 L 形", [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]),
                    ("自交蝴蝶结", [(0, 0), (10, 10), (0, 10), (10, 0)])):
    a = point_in_polygon(probe_pts[:, 0], probe_pts[:, 1], np.array(shape, float))
    b = _qt_inside(probe_pts, shape)
    check(f"判定与 Qt 填充同解：{name}", np.array_equal(a, b),
          f"差异 {int((a != b).sum())} 点")

x0, y0, x1, y1 = int(w * .30), int(h * .30), int(w * .70), int(h * .70)
box = ("rect", (x0, y0, x1, y1))
boxpoly = ("polygon", [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
m_rect = cp.view.region_mask(box, "all")
m_boxpoly = cp.view.region_mask(boxpoly, "all")
check("等价矩形与多边形掩膜完全一致", np.array_equal(m_rect, m_boxpoly),
      f"{int(m_rect.sum())} vs {int(m_boxpoly.sum())}")

r = cp.view.renderer
sx, sy, zc = r.project_points(r.xyz, w, h)
inview = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h) & (zc > 0)
ref = _qt_inside(np.column_stack([sx, sy]), boxpoly[1]) & inview
check("掩膜 == 标记区域内 ∩ 视口", np.array_equal(m_boxpoly, ref),
      f"不一致 {int((m_boxpoly != ref).sum())} 点")

# 画定区域后转视角/缩放，掩膜必须仍对应「圈中那一刻」
cam = r.camera
bak = (cam.yaw, cam.pitch, cam.center.copy(), cam.distance, cam.fov)
cp.view._region = boxpoly
cp.view._lock_view_for_region()
m_locked = cp.view.region_mask(boxpoly, "all")
cam.yaw += 33
cam.distance *= 1.6
APP.processEvents()
m_moved = cp.view.region_mask(boxpoly, "all")
check("视角改变后掩膜不漂移（区域锁定）", np.array_equal(m_locked, m_moved),
      f"{int(m_locked.sum())} vs {int(m_moved.sum())}")
cam.yaw, cam.pitch, cam.distance, cam.fov = bak[0], bak[1], bak[3], bak[4]
cam.center = bak[2].copy()
cp.view.clear_region()

# 多边形
poly = ("polygon", [(int(w * .35), int(h * .35)), (int(w * .68), int(h * .32)),
                    (int(w * .72), int(h * .66)), (int(w * .38), int(h * .70))])
mp = cp.view.region_mask(poly, "all")
check("多边形区域命中一部分点", 0 < mp.sum() < d0, f"{int(mp.sum()):,}")
sx2, sy2, zc2 = r.project_points(r.xyz, w, h)
inview2 = (sx2 >= 0) & (sx2 < w) & (sy2 >= 0) & (sy2 < h) & (zc2 > 0)
ref2 = _qt_inside(np.column_stack([sx2, sy2]), poly[1]) & inview2
check("多边形掩膜 == 标记区域内 ∩ 视口", np.array_equal(mp, ref2),
      f"不一致 {int((mp != ref2).sum())} 点")

cp.view._region = poly
cp._on_region(poly)
label_all = cp.lbl_region.text()
n_hit = int(mp.sum())
cp.cmb_depth.setCurrentIndex(1)
label_surf = cp.lbl_region.text()
check("切换深度后命中数标签同步刷新", label_all != label_surf,
      f"{label_all} -> {label_surf}")
cp.cmb_depth.setCurrentIndex(0)

cp.rb_inside.setChecked(True)
cp.apply_crop()
APP.processEvents()
check("多边形裁剪生效", 0 < cp.doc.total_points < d0, f"{cp.doc.total_points:,}")
check("留下的点数 == 圈定时显示的命中数", cp.doc.total_points == n_hit,
      f"{cp.doc.total_points:,} vs {n_hit:,}")
cp.undo()
APP.processEvents()
check("多边形裁剪可撤销", cp.doc.total_points == d0)

# 保留内 / 保留外对多边形也必须严格互补。每次都先 fit_view 把相机复位，
# 否则上一次裁剪后 set_document 会按缩小后的点云重新适配，两次投影不一致。
cp.view.fit_view(); APP.processEvents()
cp.view._region = poly
cp.rb_inside.setChecked(True)
cp.apply_crop(); APP.processEvents(); pn_in = cp.doc.total_points
cp.undo(); APP.processEvents()
cp.view.fit_view(); APP.processEvents()
cp.view._region = poly
cp.rb_outside.setChecked(True)
cp.apply_crop(); APP.processEvents(); pn_out = cp.doc.total_points
cp.undo(); APP.processEvents()
check("多边形保留内/外严格互补", pn_in + pn_out == d0,
      f"{pn_in:,} + {pn_out:,} vs {d0:,}")
cp.rb_inside.setChecked(True)

# 表面层深度
m_all = cp.view.region_mask(rect, "all")
m_surf = cp.view.region_mask(rect, "surface")
check("仅表面层命中数少于全部穿透", 0 < m_surf.sum() < m_all.sum(),
      f"{int(m_surf.sum()):,} < {int(m_all.sum()):,}")
cp.cmb_depth.setCurrentIndex(1)
cp.view._region = rect
cp.apply_crop()
APP.processEvents()
check("表面层裁剪同样生效", cp.doc.total_points < d0,
      f"{cp.doc.total_points:,}")
cp.cmb_depth.setCurrentIndex(0)
cp.undo()
APP.processEvents()

print("\n4) 绕轴旋转")
base = np.column_stack([cp.doc.scans[0].points["cartesianX"],
                        cp.doc.scans[0].points["cartesianY"],
                        cp.doc.scans[0].points["cartesianZ"]]).copy()
cp.spn_rx.setValue(0)
cp.spn_ry.setValue(0)
cp.spn_rz.setValue(30.0)
check("旋转预览未开启时文档坐标不变",
      np.array_equal(cp.doc.scans[0].points["cartesianX"], base[:, 0]))
cp.btn_rot_preview.setChecked(True)
APP.processEvents()
check("预览只改视图坐标，不改文档",
      cp.view.has_preview() and
      np.array_equal(cp.doc.scans[0].points["cartesianX"], base[:, 0]) and
      not np.allclose(cp.view.renderer.xyz[:, 0], base[:, 0]))
check("预览矩阵作用于 renderer.xyz", cp.view.renderer.xyz.shape == base.shape)

cp.apply_rotation()
APP.processEvents()
rot = np.column_stack([cp.doc.scans[0].points["cartesianX"],
                       cp.doc.scans[0].points["cartesianY"],
                       cp.doc.scans[0].points["cartesianZ"]])
check("应用旋转后坐标已改变", not np.allclose(rot[:, 0], base[:, 0]))
check("应用旋转后预览关闭", not cp.view.has_preview())
check("旋转是刚体（点间距离不变）",
      abs(float(np.linalg.norm(rot[1] - rot[0]) -
                np.linalg.norm(base[1] - base[0]))) < 1e-4)
cp.undo()
APP.processEvents()
check("旋转可整步撤销",
      np.allclose(cp.doc.scans[0].points["cartesianX"], base[:, 0], atol=1e-5))
cp.reset_rotation()
check("复位把三个角度归零",
      cp.spn_rx.value() == 0 and cp.spn_ry.value() == 0 and cp.spn_rz.value() == 0)

print("\n5) 保存（覆盖 / 另存为 / e57）")
# 先裁一点，确保写出去的点数不是原始的
cp.view._region = rect
cp.rb_inside.setChecked(True)
cp.apply_crop()
APP.processEvents()
n_keep = cp.doc.total_points

_orig_question = QMessageBox.question
_orig_save = QFileDialog.getSaveFileName
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
try:
    P_ALT = os.path.join(tmp, "plot_A_alt.ply")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (P_ALT, "PLY 点云 (*.ply)"))
    cp.save_as()
    APP.processEvents()
    check("另存为生成新文件", os.path.isfile(P_ALT))
    check("另存后当前路径切换", cp.path == P_ALT, str(cp.path))
    back = cloudio.load(P_ALT)
    check("另存文件点数一致", back.total_points == n_keep,
          f"{back.total_points:,} / {n_keep:,}")
    check("另存后未保存标记清除", not cp._dirty)

    # 覆盖保存：再裁一次再写回
    cp.view._region = ("rect", (int(w * .10), int(h * .10),
                                int(w * .90), int(h * .90)))
    cp.rb_inside.setChecked(True)
    cp.apply_crop()
    APP.processEvents()
    n_ov = cp.doc.total_points
    cp.save_overwrite()
    APP.processEvents()
    back2 = cloudio.load(P_ALT)
    check("覆盖保存按原格式写回", back2.total_points == n_ov,
          f"{back2.total_points:,} / {n_ov:,}")
    check("覆盖保存后无未保存标记", not cp._dirty)

    # 转存 e57
    P_E57_OUT = os.path.join(tmp, "plot_A_crop.e57")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (P_E57_OUT, "E57 点云 (*.e57)"))
    cp.save_as()
    APP.processEvents()
    check("裁剪结果可另存为 e57", os.path.isfile(P_E57_OUT))
    e57 = E57Document.read(P_E57_OUT)
    check("e57 读回点数一致", e57.total_points == n_ov,
          f"{e57.total_points:,} / {n_ov:,}")
finally:
    QMessageBox.question = _orig_question
    QFileDialog.getSaveFileName = _orig_save

# 重新载入：确认改动可丢弃
cp._dirty = False
cp.reload()
APP.processEvents()
check("重新载入回到磁盘内容", cp.doc.total_points == n_ov,
      f"{cp.doc.total_points:,} / {n_ov:,}")

print("\n6) 配准脚本接入（参数校验 / 环境 / 自动命名）")
from align_runner import (DEFAULT_SCRIPT, candidate_pythons, default_output,
                          probe_python, unique_output, validate_inputs)
check("脚本路径可配置且默认指向参考脚本",
      cp.edt_script.text() == DEFAULT_SCRIPT and DEFAULT_SCRIPT.endswith("align_crop_pointcloud.py"))
check("Python 路径已填且真实存在", os.path.isfile(cp.edt_python.text()), cp.edt_python.text())
check("自动命名避开已存在文件",
      unique_output(P_E57) != P_E57 or not os.path.isfile(P_E57))
check("点云1 非 PLY 会被拒绝",
      validate_inputs(P_E57_OUT, P_E57_OUT, os.path.join(tmp, "o.e57"), DEFAULT_SCRIPT) is not None)
check("输出与输入同名会被拒绝",
      validate_inputs(P_PLY, P_PLY, P_PLY, DEFAULT_SCRIPT) is not None)
check("缺文件时给出可读错误",
      validate_inputs(os.path.join(tmp, "no.ply"), os.path.join(tmp, "no.e57"),
                      os.path.join(tmp, "o.e57"), DEFAULT_SCRIPT) is not None)
cands = candidate_pythons()
check("能找到候选解释器", len(cands) >= 1, str(cands[:1]))
ok_env, missing = probe_python(cands[0])
check("首选解释器依赖齐全（open3d/plyfile/pye57）", ok_env, str(missing))
check("默认输出名由点云2 推导",
      os.path.basename(default_output(P_E57_OUT)).endswith("_cropped.e57"))
check("运行按钮/停止按钮/日志框都在",
      all(hasattr(cp, x) for x in ("btn_run", "btn_stop", "txt_log", "runner")))
check("参数面板齐全",
      all(hasattr(cp, x) for x in ("cmb_reg", "spn_crop_d", "spn_voxel",
                                  "spn_overlap", "spn_workers", "chk_refine")))

print("\n7) 裁剪页全部控件悬浮说明")
# QTabWidget 会自动生成左右滚动按钮（ScrollLeftButton / ScrollRightButton），
# 它们是 Qt 内部控件，既不可见也不该要求 tooltip。
skip = {"qt_toolbar_ext_button", "ScrollLeftButton", "ScrollRightButton"}


def _interactive(root):
    seen = {}
    for cls in (QAbstractButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox):
        for wgt in root.findChildren(cls):
            seen[id(wgt)] = wgt
    return [wgt for wgt in seen.values()
            if wgt.objectName() not in skip and wgt.window() is win]


no_tip = [f"{type(x).__name__}({getattr(x, 'text', lambda: '')()})"
          for x in _interactive(cp) if not x.toolTip().strip()]
check("裁剪页所有按钮/下拉/数值框都有说明", not no_tip, str(no_tip[:6]))
short = [f"{type(x).__name__}({getattr(x, 'text', lambda: '')()})"
         for x in _interactive(cp) if len(x.toolTip().strip()) < 12]
check("没有敷衍的一句话说明（≥12 字）", not short, str(short[:6]))
check("导航页签也有说明",
      bool(win.btn_page_annot.toolTip().strip()) and bool(win.btn_page_crop.toolTip().strip()))
check("主窗口（含两页）无空说明",
      not [x for x in _interactive(win)
           if x.objectName() not in skip and not x.toolTip().strip()])

print("\n8) 清理")
win.close()
print("临时目录：", tmp)
print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
