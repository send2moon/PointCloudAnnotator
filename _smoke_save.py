"""带标注保存 + 第三方（libE57 / CloudCompare 同源）兼容性测试。

检查三件事：
1. 原型声明的每记录字节数 == 实际写入的每记录字节数（宽度声明自洽）；
2. 结构体检通过（段长、包链、XML 偏移全部衔接）；
3. pye57 能打开并读回标准列与源文件一致。
"""
from __future__ import annotations

import os
import re
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from annotation import AnnotationSet
from e57io import E57Document, check_e57_structure

def _first_e57(root: str) -> str:
    for d, _sub, names in os.walk(root):
        for nm in sorted(names):
            if nm.lower().endswith(".e57"):
                return os.path.join(d, nm)
    return ""


SRC = (sys.argv[1] if len(sys.argv) > 1 else "") or _first_e57(
    r"D:\PCProjects\01wheat\FinalScripts\outputs\WS\FINAL_PLOTS")
if not SRC:
    print("找不到样本 e57，跳过"); sys.exit(1)
ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}", flush=True)


doc = E57Document.read(SRC)
ann = AnnotationSet.from_scans(doc.scans)
for i, s in enumerate(doc.scans):
    k = max(1, s.count // 20)
    ann.labels[i][:k] = 1
    ann.instances[i][:k] = i + 1
    ann.instance_info[i + 1] = (1, i)
n_marked = ann.total_marked()

from main_window import MainWindow  # 与 GUI 保存走同一套字段附加逻辑
for i, s in enumerate(doc.scans):
    MainWindow._attach_annotation(s, ann.labels[i], ann.instances[i])
    check(f"扫描{i} 列等长", all(len(v) == s.count for v in s.points.values()),
          f"count={s.count:,}")

out = os.path.join(tempfile.gettempdir(), "pca_annot_smoke.e57")
doc.save(out)
size = os.path.getsize(out)
print(f"  写入 {size:,} 字节，标注 {n_marked:,} 点", flush=True)

# ---------------------------------------------------------- 1. 结构自洽
probs = check_e57_structure(out)
check("结构体检通过", not probs, str(probs))

# ---------------------------------------------------------- 2. 宽度自洽
blob = open(out, "rb").read()
LOG, PH = 1020, 1024
lg = bytearray()
for i in range(len(blob) // PH):
    lg += blob[i * PH: i * PH + LOG]
phys, xlen = struct.unpack_from("<QQ", blob, 24)
st = (phys // PH) * LOG + (phys % PH)
text = bytes(lg[st:st + xlen]).split(b"</e57Root>")[0].decode("utf-8", "replace")
proto = re.search(r"<prototype[^>]*>(.*?)</prototype>", text, re.S).group(1)
declared = 0
for fm in re.finditer(r'<([A-Za-z][\w]*)\s+type="(Float|Integer|ScaledInteger)"([^>]*)/>', proto):
    kind, attrs = fm.group(2), fm.group(3)
    if kind == "Float":
        declared += 8 if 'precision="double"' in attrs else 4
    else:
        lo = int(float(re.search(r'minimum="([-\d.eE+]+)"', attrs).group(1)))
        hi = int(float(re.search(r'maximum="([-\d.eE+]+)"', attrs).group(1)))
        if lo >= 0:
            declared += 1 if hi <= 255 else (2 if hi <= 65535 else 4)
        else:
            declared += 1 if (lo >= -128 and hi <= 127) else (2 if (lo >= -32768 and hi <= 32767) else 4)
written = sum(fd.itemsize for fd in doc.scans[0].fields if fd.name in doc.scans[0].points)
check("原型每记录字节 == 实际写入", declared == written, f"声明 {declared} / 实写 {written}")
check("标注字段宽度正确",
      [ (f.name, f.itemsize) for f in doc.scans[0].fields
        if f.name in ("annotationLabel", "annotationInstance") ]
      == [("annotationLabel", 1), ("annotationInstance", 2)])
# 写入 e57 的库版本字段也应是新软件名（UTF-8 XML，第三方照样能读）
check("e57LibraryVersion 为新软件名", "作物点云器官标注软件 1.0" in text,
      text[text.find("e57LibraryVersion"):text.find("e57LibraryVersion") + 90])
check("不再写入旧名称", "PointCloudAnnotator 1.0" not in text)

# ---------------------------------------------------------- 3. libE57 读回
import pye57
try:
    r = pye57.E57(out)
    arr = r.read_scan_raw(0, ignore_unsupported_fields=True)
    check("pye57 打开成功", True, f"{r.scan_count} 扫描")
    src0 = doc.scans[0]
    check("标准列点数一致", len(arr["cartesianX"]) == src0.count,
          f"{len(arr['cartesianX']):,}")
    for k in ("cartesianX", "cartesianY", "cartesianZ", "colorRed", "colorBlue"):
        a = np.asarray(arr[k], np.float64)
        b = np.asarray(src0.points[k], np.float64)
        check(f"{k} 数值一致", np.array_equal(a, b),
              f"maxdiff={np.abs(a - b).max():.3g}")
except Exception as e:  # noqa: BLE001
    check("pye57 打开成功", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------- 4. 本软件可续编辑
back = E57Document.read(out)
check("回读扫描数一致", len(back.scans) == len(doc.scans))
check("回读总点数一致", back.total_points == doc.total_points,
      f"{back.total_points:,} vs {doc.total_points:,}")
per = [(s.count, int((s.points.get("annotationLabel", np.zeros(0)) > 0).sum()))
       for s in back.scans]
want = [(s.count, int((ann.labels[i] > 0).sum())) for i, s in enumerate(doc.scans)]
check("全部扫描的标注都回来了", per == want, f"{per} vs {want}")

try:                        # pye57 还握着文件句柄，删不掉就算了
    os.remove(out)
except OSError:
    pass
print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok else 1)
