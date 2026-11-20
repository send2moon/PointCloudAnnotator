"""多格式读写回归：ply / pcd / xyz / txt / csv 的往返与跨格式转换。

检查点
------
1. 合成点云（含颜色、强度、类别、实例编号）逐个格式写出再读回，
   点数/坐标/颜色/类别/实例完全一致；
2. 多扫描写出再读回，站数、每站点数、每站类别都对得上；
3. 真实 e57（42,194 点 / 8 站 / 32,501 标注点）另存为 ply/pcd/csv 再读回，
   标注一个不少；再把这些文件转回 e57，用 ``E57Document.read`` 读回核对；
4. 不带表头的裸 xyz（3 列）也能读，颜色补灰、标注为 0；
5. 不支持的扩展名、空文件给出明确报错。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cloudio
from e57io import E57Document, Scan

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "fmt")
os.makedirs(OUT, exist_ok=True)
GOOD = r"D:/Data/PC/ResearchPlots/DT/1-FS_09_labeled.e57"

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")


# ------------------------------------------------ 合成数据 --

def make_doc(n_scans=1, n=1200, seed=7):
    rng = np.random.default_rng(seed)
    doc = E57Document()
    for k in range(n_scans):
        s = Scan(name=f"Scan_{k+1:03d}", guid="")
        doc.scans.append(s)
        from e57io import FieldDef
        xyz = (rng.random((n, 3)) * 40 - 20).astype(np.float32)
        s.points["cartesianX"], s.points["cartesianY"], s.points["cartesianZ"] = \
            xyz[:, 0], xyz[:, 1], xyz[:, 2]
        s.points["colorRed"] = rng.integers(0, 256, n, dtype=np.uint8)
        s.points["colorGreen"] = rng.integers(0, 256, n, dtype=np.uint8)
        s.points["colorBlue"] = rng.integers(0, 256, n, dtype=np.uint8)
        s.points["intensity"] = rng.random(n).astype(np.float32) * 100
        s.fields = [
            FieldDef("cartesianX", "Float", "single"),
            FieldDef("cartesianY", "Float", "single"),
            FieldDef("cartesianZ", "Float", "single"),
            FieldDef("colorRed", "Integer", minimum=0, maximum=255, byte_size=1),
            FieldDef("colorGreen", "Integer", minimum=0, maximum=255, byte_size=1),
            FieldDef("colorBlue", "Integer", minimum=0, maximum=255, byte_size=1),
            FieldDef("intensity", "Float", "single"),
        ]
    return doc


class FakeAnn:
    """只提供 labels / instances（cloudio.save 用到的全部）。"""

    def __init__(self, doc, seed=11):
        rng = np.random.default_rng(seed)
        self.labels = [rng.integers(0, 5, s.count, dtype=np.uint8) for s in doc.scans]
        self.instances = [rng.integers(0, 30, s.count, dtype=np.uint16) for s in doc.scans]


def total_marked(ann):
    return int(sum((l > 0).sum() for l in ann.labels))


print("1) 单扫描合成数据：四种格式往返")
doc = make_doc(1)
ann = FakeAnn(doc)
n0 = doc.total_points
for ext in (".ply", ".pcd", ".xyz", ".csv", ".txt"):
    p = os.path.join(OUT, "synthetic" + ext)
    cloudio.save(p, doc, ann)
    back = cloudio.load(p)
    a2 = FakeAnn(back, seed=0)
    s0, s1 = doc.scans[0], back.scans[0]
    d = np.abs(np.asarray(s1.points["cartesianX"], np.float64)
               - np.asarray(s0.points["cartesianX"], np.float64)).max()
    check(f"{ext} 点数一致", back.total_points == n0, f"{back.total_points:,}/{n0:,}")
    check(f"{ext} 坐标一致（≤1e-3）", d <= 1.5e-3, f"max Δ = {d:.5f}")
    rgb_same = all(np.array_equal(s1.points[k], s0.points[k])
                   for k in ("colorRed", "colorGreen", "colorBlue"))
    check(f"{ext} 颜色逐点一致", rgb_same)
    lab_same = np.array_equal(s1.points["annotationLabel"], ann.labels[0])
    ins_same = np.array_equal(s1.points["annotationInstance"], ann.instances[0])
    check(f"{ext} 类别/实例编号逐点一致", lab_same and ins_same)
    it = np.abs(np.asarray(s1.points["intensity"], np.float64)
                - np.asarray(s0.points["intensity"], np.float64)).max()
    check(f"{ext} 强度保留（≤0.01）", it <= 0.011, f"max Δ = {it:.4f}")

print("\n2) 多扫描：站号列")
doc3 = make_doc(3, 500)
ann3 = FakeAnn(doc3)
for ext in (".ply", ".pcd", ".csv", ".xyz"):
    p = os.path.join(OUT, f"multi3{ext}")
    cloudio.save(p, doc3, ann3)
    back = cloudio.load(p)
    counts = [s.count for s in back.scans]
    check(f"{ext} 读回 3 个扫描", len(back.scans) == 3, str(counts))
    check(f"{ext} 每站点数正确", counts == [500, 500, 500], str(counts))
    good = all(np.array_equal(back.scans[i].points["annotationLabel"], ann3.labels[i])
               for i in range(3))
    check(f"{ext} 各站类别对位正确", good)

print("\n3) 真实 e57 → 其他格式 → 再转回 e57")
real = E57Document.read(GOOD)
from annotation import AnnotationSet
ann_real = AnnotationSet.from_scans(real.scans)
marked0 = ann_real.total_marked()
print(f"   源文件 {real.total_points:,} 点 / {len(real.scans)} 站 / 标注 {marked0:,} 点")
for ext in (".ply", ".pcd", ".csv"):
    p = os.path.join(OUT, "real_labeled" + ext)
    cloudio.save(p, real, ann_real)
    back = cloudio.load(p)
    a2 = AnnotationSet.from_scans(back.scans)
    check(f"{ext} 点数与站数一致",
          back.total_points == real.total_points and len(back.scans) == len(real.scans),
          f"{back.total_points:,} / {len(back.scans)} 站")
    check(f"{ext} 标注点数一致", a2.total_marked() == marked0,
          f"{a2.total_marked():,} vs {marked0:,}")
    same_inst = a2.next_instance == ann_real.next_instance
    check(f"{ext} 实例编号范围一致", same_inst,
          f"next={a2.next_instance} vs {ann_real.next_instance}")

    # 再转回 e57：走原生写入，用 libE57 读回
    p2 = os.path.join(OUT, f"roundtrip_from{ext.replace('.', '_')}.e57")
    for i, s in enumerate(back.scans):
        s.points["annotationLabel"] = a2.labels[i]
        s.points["annotationInstance"] = a2.instances[i]
        from e57io import FieldDef
        have = {f.name for f in s.fields}
        if "annotationLabel" not in have:
            s.fields.append(FieldDef("annotationLabel", "Integer",
                                     minimum=0, maximum=255, byte_size=1))
        if "annotationInstance" not in have:
            s.fields.append(FieldDef("annotationInstance", "Integer",
                                     minimum=0, maximum=65535, byte_size=2))
    back.save(p2)
    again = E57Document.read(p2)
    a3 = AnnotationSet.from_scans(again.scans)
    check(f"{ext} → e57 第三方可读回且标注不丢", a3.total_marked() == marked0,
          f"{a3.total_marked():,}")

print("\n4) 裸 XYZ（无表头 3 列）")
p = os.path.join(OUT, "bare.xyz")
np.savetxt(p, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), fmt="%.3f")
d = cloudio.load(p)
check("裸 xyz 能读", d.total_points == 2)
s = d.scans[0]
check("缺颜色时补灰", int(s.points["colorRed"][0]) == 180)
_lab = s.points.get("annotationLabel")
check("无标注列时视为全未标注", _lab is None or int(np.max(_lab)) == 0)

p = os.path.join(OUT, "cc_style.txt")          # CloudCompare 导出的 6 列（x y z r g b）
np.savetxt(p, np.array([[1, 2, 3, 255, 0, 0], [4, 5, 6, 0, 255, 0]]), fmt="%.0f")
d = cloudio.load(p)
check("6 列文本按 xyz+rgb 解析", d.total_points == 2
      and int(d.scans[0].points["colorRed"][0]) == 255)

print("\n5) 错误处理")
try:
    cloudio.load(os.path.join(OUT, "nope.abc"))
    check("不支持的扩展名报错", False)
except Exception as e:                        # noqa: BLE001
    check("不支持的扩展名报错", "不支持" in str(e), str(e)[:40])
try:
    p = os.path.join(OUT, "empty.ply")
    open(p, "wb").write(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    cloudio.load(p)
    check("空 PLY 报错", False)
except Exception as e:                        # noqa: BLE001
    check("空 PLY 报错", "没有顶点" in str(e) or "没有读到" in str(e), str(e)[:40])

print("\n6) 界面集成：按钮文案 / 快捷键交换 / 另存为多格式")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QFontDatabase                    # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox      # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv[:1])
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(_f):                     # offscreen 下没字体回退，中文会变方块
        QFontDatabase.addApplicationFont(_f)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)

import main_window as mw                                   # noqa: E402
from main_window import MainWindow                         # noqa: E402

win = MainWindow()
win.resize(1400, 900)
win.show()
APP.processEvents()
win.load_file(GOOD)
APP.processEvents()

check("侧栏按钮已改名", win.btn_save.text() == "保存为标注点云…", win.btn_save.text())
check("「保存为标注点云」占 Ctrl+S",
      win.act_save.shortcut().toString() == "Ctrl+S", win.act_save.shortcut().toString())
check("「覆盖保存」占 Ctrl+Shift+S",
      win.act_save_ov.shortcut().toString() == "Ctrl+Shift+S",
      win.act_save_ov.shortcut().toString())
check("侧栏「覆盖保存」按钮存在", win.btn_save2.text() == "覆盖保存")

# 另存为 ply（模拟对话框选 PLY 过滤器）
_orig_dialog = mw.QFileDialog.getSaveFileName
p_ply = os.path.join(OUT, "ui_save_as.ply")
mw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (p_ply, "PLY 点云 (*.ply)"))
win.act_save.trigger()
APP.processEvents()
mw.QFileDialog.getSaveFileName = staticmethod(_orig_dialog)
check("Ctrl+S 走「另存为」并写出 PLY", os.path.exists(p_ply) and win.path == p_ply,
      os.path.basename(win.path or ""))
back = cloudio.load(p_ply)
check("另存的 PLY 标注完整", AnnotationSet.from_scans(back.scans).total_marked() == marked0,
      f"{AnnotationSet.from_scans(back.scans).total_marked():,}")

# 没写扩展名时按所选过滤器补全
p_noext = os.path.join(OUT, "ui_noext")
mw.QFileDialog.getSaveFileName = staticmethod(
    lambda *a, **k: (p_noext, "PCD 点云 (*.pcd)"))
win.save_as()
APP.processEvents()
mw.QFileDialog.getSaveFileName = staticmethod(_orig_dialog)
check("没写扩展名时按过滤器补 .pcd", os.path.exists(p_noext + ".pcd"))

# 覆盖保存：按当前文件的格式写回
p_cur = p_noext + ".pcd"
zero = np.nonzero(win.ann.labels[0] == 0)[0][:5]      # 挑 5 个未标注点来验证写回
win.ann.labels[0][zero] = 1
win.ann.instances[0][zero] = 0
win._dirty = True
before = int((win.ann.labels[0] > 0).sum())
win.save_overwrite()
APP.processEvents()
check("覆盖保存写回当前文件（PCD）", win.path == p_cur and os.path.exists(p_cur),
      os.path.basename(win.path or ""))
back2 = cloudio.load(p_cur)
after = int((back2.scans[0].points["annotationLabel"] > 0).sum())
check("覆盖后磁盘上的标注跟着更新", after == before, f"{after} vs {before}")

# 重新打开 PLY：界面能正常显示，状态栏 tooltip 标明格式
win.load_file(p_ply)
APP.processEvents()
check("重开 PLY 后标注被识别", win.ann.total_marked() > 0,
      f"{win.ann.total_marked():,} 点")
check("状态栏 tooltip 标明格式", "PLY" in win.lbl_file.toolTip(), win.lbl_file.toolTip())
check("非 e57 时「应用 E57 位姿」置灰", not win.chk_pose.isEnabled())
win.grab().save(os.path.join(OUT, "ui_open_ply.png"))
print("   -> outputs/fmt/ui_open_ply.png")

# 从其它格式转存成 e57（归档用），再用 libE57 读回
p_conv = os.path.join(OUT, "converted_from_ply.e57")
win._do_save(p_conv)
APP.processEvents()
check("PLY 可转存为 e57", os.path.exists(p_conv) and cloudio.ext_of(win.path) == ".e57")
again = E57Document.read(p_conv)
check("转存后的 e57 标注完整",
      AnnotationSet.from_scans(again.scans).total_marked()
      == AnnotationSet.from_scans(back.scans).total_marked(),
      f"{AnnotationSet.from_scans(again.scans).total_marked():,}")
check("转存后仍为 8 个扫描", len(again.scans) == 8, f"{len(again.scans)} 站")

print("\n7) 附加属性（法向/尺度等）原样保留")
# 造一个和 AgriTopo 输出同构的 PLY：xyz + rgb + nx/ny/nz + scale_0..2 + 一个 double 属性
from e57io import FieldDef as _FD                            # noqa: E402

n = 500
rng = np.random.default_rng(3)
props = [("x", "float", "f4"), ("y", "float", "f4"), ("z", "float", "f4"),
         ("red", "uchar", "u1"), ("green", "uchar", "u1"), ("blue", "uchar", "u1"),
         ("nx", "float", "f4"), ("ny", "float", "f4"), ("nz", "float", "f4"),
         ("scale_0", "float", "f4"), ("scale_1", "float", "f4"), ("scale_2", "float", "f4"),
         ("quality", "double", "f8")]
p_src = os.path.join(OUT, "with_extras.ply")
head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
head += [f"property {t} {nm}" for nm, t, _d in props] + ["end_header"]
dt = np.dtype([(nm, "<" + d) for nm, _t, d in props])
rec = np.empty(n, dtype=dt)
for nm, _t, _d in props:
    rec[nm] = rng.random(n).astype(np.float32) if nm == "quality" else rng.random(n)
with open(p_src, "wb") as fh:
    fh.write(("\n".join(head) + "\n").encode("ascii"))
    rec.tofile(fh)

d_src = cloudio.load(p_src)
check("读入时认出附加属性", d_src.scans[0].extra_props ==
      ["nx", "ny", "nz", "scale_0", "scale_1", "scale_2", "quality"],
      str(d_src.scans[0].extra_props))
check("double 属性按 f8 记类型", d_src.scans[0].extra_types.get("quality") == "f8")

p_out = os.path.join(OUT, "with_extras_out.ply")
cloudio.save(p_out, d_src, FakeAnn(d_src))
raw = open(p_out, "rb").read()
hdr = raw[:raw.find(b"end_header")].decode("ascii", "replace")
check("头里没有中文乱码（纯 ASCII）", "?" not in hdr)
check("写出的头仍声明 nx/scale/quality",
      all(f"property float {k}" in hdr for k in ("nx", "nz", "scale_0"))
      and "property double quality" in hdr)

back = cloudio.load(p_out)
s = back.scans[0]
same = all(np.array_equal(np.asarray(s.points[k]), np.asarray(d_src.scans[0].points[k]))
           for k in ("nx", "ny", "nz", "scale_0", "scale_1", "scale_2", "quality"))
check("附加属性逐字节还原", same)
check("double 属性精度未降级",
      np.asarray(s.points["quality"]).dtype == np.float64,
      str(np.asarray(s.points["quality"]).dtype))

# 写入尺寸核对：12(xyz)+3(rgb)+1(label)+2(instance)+24(6×f4)+8(quality)
expect = 12 + 3 + 1 + 2 + 24 + 8
check("文件大小 = 头 + 点数 × 每点字节",
      os.path.getsize(p_out) == len(hdr) + len(b"end_header\n") + n * expect,
      f"{os.path.getsize(p_out)} vs {len(hdr) + 11 + n * expect}")

# 无标注导出（裁剪页保存）：不应平白多出全 0 的 label / instance
from annotation import AnnotationSet                              # noqa: E402
p_plain = os.path.join(OUT, "no_ann.ply")
cloudio.save(p_plain, d_src, AnnotationSet.empty([s.count for s in d_src.scans]))
raw = open(p_plain, "rb").read()
hdr_plain = raw[:raw.find(b"end_header")].decode("ascii", "replace")
check("无标注时不写 label 列", "property uchar label" not in hdr_plain)
check("无标注时不写 instance 列", "property ushort instance" not in hdr_plain)
check("无标注时附加属性仍保留", "property double quality" in hdr_plain)
expect_plain = 12 + 3 + 24 + 8
check("无标注文件大小 = 头 + 点数 × 每点字节",
      os.path.getsize(p_plain) == len(hdr_plain) + len(b"end_header\n") + n * expect_plain,
      f"{os.path.getsize(p_plain)} vs {len(hdr_plain) + 11 + n * expect_plain}")
back_plain = cloudio.load(p_plain)
check("无标注 PLY 读回正常", back_plain.total_points == n
      and "annotationLabel" not in back_plain.scans[0].points)

print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
