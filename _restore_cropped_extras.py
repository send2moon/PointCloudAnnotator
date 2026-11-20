"""把「原始 PLY」的附加属性按坐标补回「裁剪后点云」（E57 或 PLY）。

背景：裁剪页把带私有列的 PLY（``nidar_views`` / ``nx,ny,nz`` / ``scale_*`` 等）
另存为 E57 时，E57 标准字段表只承载 ``XYZ + RGB + intensity``，私有列会整列丢失。
与 ``_merge_ply_extras.py`` 不同，本脚本允许**点数不同**（裁剪删了点）：用坐标
最近邻把原始列映射到裁剪结果上。裁剪过程必须只删点、不改坐标，否则拒绝执行。

用法::

    python _restore_cropped_extras.py <原始.ply> <裁剪后.e57|ply> [-o 输出.ply]

不指定 ``-o`` 时输出 ``<裁剪后>_full.ply``。
输出 PLY 的字段结构与原始 PLY 一致（不额外写 label/instance 列）。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cloudio
from e57io import E57Document


def _load_any(path: str):
    """e57 走原生解析器，其余格式走 cloudio；都返回 E57Document。"""
    if path.lower().endswith(".e57"):
        return E57Document.read(path)
    return cloudio.load(path)


def _stack(doc, keys):
    """把多站同名列拼成一维数组；缺列的站补 0。"""
    parts = []
    for s in doc.scans:
        if s.count == 0:
            continue
        col = s.points.get(keys[0]) if len(keys) == 1 else None
        if col is None:
            for k in keys:
                if k in s.points:
                    col = s.points[k]
                    break
        parts.append(np.asarray(col) if col is not None
                     else np.zeros(s.count, np.float32))
    return np.concatenate(parts) if parts else np.zeros(0)


def _ply_type(arr) -> str:
    tag = cloudio._np_type_tag(np.asarray(arr).dtype)
    return cloudio._PLY_OF_TAG.get(tag, "float")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按坐标把原始 PLY 的附加列补回裁剪结果")
    ap.add_argument("original", help="带附加属性的原始 PLY")
    ap.add_argument("cropped", help="裁剪后的点云（.e57 / .ply / .pcd ...）")
    ap.add_argument("-o", "--out", default="", help="输出 PLY（默认 <cropped>_full.ply）")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="坐标匹配容差（默认 1e-5，裁剪只删点时偏差为 0）")
    a = ap.parse_args(argv)

    src = cloudio.load(a.original)
    dst = _load_any(a.cropped)
    extras = []
    for s in src.scans:
        for nm in getattr(s, "extra_props", []) or []:
            if nm not in extras:
                extras.append(nm)
    if not extras:
        print("[拒绝] 原始 PLY 没有附加属性，无需补救")
        return 2

    sx = np.column_stack([_stack(src, ("cartesianX", "x")),
                          _stack(src, ("cartesianY", "y")),
                          _stack(src, ("cartesianZ", "z"))]).astype(np.float64)
    dx = np.column_stack([_stack(dst, ("cartesianX", "x")),
                          _stack(dst, ("cartesianY", "y")),
                          _stack(dst, ("cartesianZ", "z"))]).astype(np.float64)
    if len(sx) < len(dx):
        print(f"[拒绝] 裁剪结果点数({len(dx):,}) 多于原始({len(sx):,})")
        return 2

    from scipy.spatial import cKDTree
    dist, idx = cKDTree(sx).query(dx, k=1)
    if float(dist.max()) > a.tol:
        print(f"[拒绝] 坐标最大偏差 {float(dist.max()):.6g} > {a.tol:g}；"
              "裁剪过程改过坐标（配准/旋转），无法按坐标补列")
        return 2
    if len(np.unique(idx)) != len(idx):
        print("[拒绝] 存在重复匹配点，裁剪结果可能不是原始点云的子集")
        return 2

    # 基础列全部取自裁剪结果，保证几何/颜色/强度与当前文件一致
    cols = [
        ("x", np.asarray(_stack(dst, ("cartesianX", "x")), np.float32)),
        ("y", np.asarray(_stack(dst, ("cartesianY", "y")), np.float32)),
        ("z", np.asarray(_stack(dst, ("cartesianZ", "z")), np.float32)),
        ("red", np.asarray(_stack(dst, ("colorRed", "red")), np.uint8)),
        ("green", np.asarray(_stack(dst, ("colorGreen", "green")), np.uint8)),
        ("blue", np.asarray(_stack(dst, ("colorBlue", "blue")), np.uint8)),
        ("intensity", np.asarray(_stack(dst, ("intensity",)), np.float32)),
    ]
    src_extras = {nm: _stack(src, (nm,)) for nm in extras}
    for nm in extras:
        cols.append((nm, np.asarray(src_extras[nm])[idx]))

    n = len(cols[0][1])
    out = a.out or (os.path.splitext(a.cropped)[0] + "_full.ply")
    header = ["ply", "format binary_little_endian 1.0",
              "comment extras restored from original PLY by coordinate match",
              f"element vertex {n}"]
    for nm, arr in cols:
        header.append(f"property {_ply_type(arr)} {nm}")
    header.append("end_header")
    buf = np.empty(n, dtype=[(nm, arr.dtype.newbyteorder("<")) for nm, arr in cols])
    for nm, arr in cols:
        buf[nm] = arr
    with open(out, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(buf.tobytes())

    # 读回自检
    back = cloudio.load(out)
    s = back.scans[0]
    print(f"写入 {out}")
    print(f"  点数 {s.count:,}（裁剪结果 {len(dx):,}，原始 {len(sx):,}）")
    print(f"  补回附加属性 {s.extra_props}  类型 {s.extra_types}")
    ok = (s.count == n and s.extra_props == extras)
    for nm in extras:
        same = np.array_equal(np.asarray(s.points[nm]), np.asarray(src_extras[nm])[idx])
        print(f"  {nm} 逐点一致：{same}")
        ok &= same
    print(f"  文件大小 {os.path.getsize(out)/1e6:.3f} MB"
          f"（原始 {os.path.getsize(a.original)/1e6:.3f} MB，"
          f"裁剪后 {os.path.getsize(a.cropped)/1e6:.3f} MB）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
