"""把「标注文件」里的类别/实例合回「带附加属性的原始 PLY」，补回 nx/ny/nz、scale 等列。

背景：早期版本的「保存为标注点云」只写 xyz+rgb+label+instance，会把 PLY 里
未识别的顶点属性（AgriTopo 的 nx/ny/nz、scale_0..2）丢掉，文件因此"变小"。
新版本已改为原样保留；本脚本用来**修补已经生成过的标注文件**。

用法::

    python _merge_ply_extras.py <带属性的原始.ply> <标注文件.ply> [-o 输出.ply]

要求两个文件点数一致、点顺序一致（同一个点云派生出来的），否则拒绝执行。
不指定 ``-o`` 时输出 ``<标注文件>_full.ply``。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cloudio
from annotation import AnnotationSet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="把标注合回带附加属性的 PLY")
    ap.add_argument("original", help="带法向/尺度等属性的原始 PLY")
    ap.add_argument("labeled", help="本软件导出的标注 PLY")
    ap.add_argument("-o", "--out", default="", help="输出路径（默认 <labeled>_full.ply）")
    a = ap.parse_args(argv)

    src = cloudio.load(a.original)
    lab = cloudio.load(a.labeled)
    if len(src.scans) != len(lab.scans):
        print(f"[拒绝] 站数不同：{len(src.scans)} vs {len(lab.scans)}")
        return 2
    for i, (s0, s1) in enumerate(zip(src.scans, lab.scans)):
        if s0.count != s1.count:
            print(f"[拒绝] 第 {i+1} 站点数不同：{s0.count} vs {s1.count}")
            return 2
        dx = float(np.abs(np.asarray(s0.points["cartesianX"], np.float64)
                          - np.asarray(s1.points["cartesianX"], np.float64)).max())
        if dx > 1e-4:
            print(f"[拒绝] 第 {i+1} 站坐标不一致（最大差 {dx:.4f}），点顺序可能不同")
            return 2

    ann = AnnotationSet.from_scans(lab.scans)
    out = a.out or (os.path.splitext(a.labeled)[0] + "_full.ply")
    cloudio.save(out, src, ann)

    # 读回自检
    back = cloudio.load(out)
    s = back.scans[0]
    a2 = AnnotationSet.from_scans(back.scans)
    print(f"写入 {out}")
    print(f"  点数 {s.count:,}（原 {src.scans[0].count:,}）")
    print(f"  附加属性 {s.extra_props}")
    print(f"  标注 {a2.total_marked():,} 点（源 {ann.total_marked():,}）")
    nx_ok = np.array_equal(s.points["nx"], np.asarray(src.scans[0].points["nx"]))
    print(f"  nx 逐点一致：{nx_ok}")
    print(f"  文件大小 {os.path.getsize(out)/1e6:.2f} MB"
          f"（原 {os.path.getsize(a.original)/1e6:.2f} MB，"
          f"旧标注 {os.path.getsize(a.labeled)/1e6:.2f} MB）")
    return 0 if (a2.total_marked() == ann.total_marked() and nx_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
