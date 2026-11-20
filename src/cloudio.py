"""多格式点云读写：e57 / ply / pcd / xyz / txt / csv。

界面上只暴露两个动作：打开（按扩展名自动选解析器）与「保存为标注点云…」
（按保存对话框里选的扩展名写）。非 e57 格式统一包装成 **单扫描或多扫描的
``E57Document``**，这样裁剪、标注、实例分割、撤销等上层逻辑一行都不用改。

标注的落盘方式（保证换格式后重开还能看到类别与实例编号）：

===========  ==========================================================
PLY          额外顶点属性 ``label``(uchar) / ``instance``(ushort)，
             与 ``x y z red green blue intensity`` 同行；CloudCompare
             打开会当成标量场，可按 label 着色。**没有任何标注时不写这两列**
             （裁剪/纯几何导出要与源文件字段一致，不留全 0 的占位标量场）。
PCD          额外字段 ``label``(U1) / ``instance``(U2)，``rgb`` 打包成 float，
             与 PCL/CloudCompare 的约定一致；同样无标注时不写。
CSV          表头行 ``x,y,z,r,g,b,label,instance,intensity,scan``。
XYZ / TXT    首行 ``#`` 注释写明列名，逐行数值；多站时多一列 ``scan``。
===========  ==========================================================

写入文本类格式时，坐标保留 3 位小数（单位 cm，分辨率 10 µm）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from e57io import E57Document, FieldDef, Scan

APP_TAG = "作物点云处理软件"
LABEL_FIELD = "annotationLabel"
INSTANCE_FIELD = "annotationInstance"


@dataclass
class _Cloud:
    """解析结果：标准列 + 未识别的附加属性（法向、尺度等）。

    ``extras`` 里的列原样保留（含类型），写回同族格式时逐字节还原；
    写 e57 时丢弃（e57 没有对应概念），避免给第三方软件塞一堆私有字段。
    """

    cols: Dict[str, np.ndarray] = field(default_factory=dict)
    extras: Dict[str, np.ndarray] = field(default_factory=dict)
    types: Dict[str, str] = field(default_factory=dict)      # 名 → f4/f8/u1/u2/i4/u4

# 扩展名 → 人类可读名称
FORMAT_NAMES = {
    ".e57": "E57",
    ".ply": "PLY",
    ".pcd": "PCD",
    ".xyz": "XYZ 文本",
    ".txt": "TXT 文本",
    ".pts": "PTS 文本",
    ".csv": "CSV 表格",
}

# 可打开 / 可另存
READ_EXTS = tuple(FORMAT_NAMES)
WRITE_EXTS = (".e57", ".ply", ".pcd", ".xyz", ".txt", ".csv")

OPEN_FILTER = (
    "点云文件 (*.e57 *.ply *.pcd *.xyz *.txt *.pts *.csv);;"
    "E57 点云 (*.e57);;"
    "PLY 点云 (*.ply);;"
    "PCD 点云 (*.pcd);;"
    "文本点云 (*.xyz *.txt *.pts);;"
    "CSV 表格 (*.csv);;"
    "所有文件 (*.*)"
)

# 「保存为标注点云」的过滤器，顺序即默认格式顺序
SAVE_FILTER = (
    "E57 点云 (*.e57);;"
    "PLY 点云 (*.ply);;"
    "PCD 点云 (*.pcd);;"
    "XYZ 文本 (*.xyz);;"
    "CSV 表格 (*.csv);;"
    "TXT 文本 (*.txt)"
)


def ext_of(path: str) -> str:
    return os.path.splitext(str(path))[1].lower()


def is_supported(path: str) -> bool:
    return ext_of(path) in FORMAT_NAMES


def format_name(path: str) -> str:
    return FORMAT_NAMES.get(ext_of(path), ext_of(path).lstrip(".").upper() or "未知")


# --------------------------------------------------------------------- 读 --

def load(path: str) -> E57Document:
    """读取非 e57 点云（e57 走 ``E57Document.read``，需要体检与静默修复）。"""
    ext = ext_of(path)
    if ext == ".e57":
        return E57Document.read(path)
    if ext == ".ply":
        cloud = _read_ply(path)
    elif ext == ".pcd":
        cloud = _read_pcd(path)
    elif ext in (".xyz", ".txt", ".pts", ".csv"):
        cloud = _read_text(path)
    else:
        raise ValueError(f"不支持的格式：{ext or '（无扩展名）'}")
    return _doc_from_columns(cloud, path)


def _doc_from_columns(cloud: _Cloud, path: str) -> E57Document:
    """把 {x,y,z,r,g,b,intensity,label,instance,scan} + 附加属性组装成 E57Document。"""
    cols = cloud.cols
    n = len(cols["x"])
    if n == 0:
        raise ValueError("文件里没有读到任何点")
    doc = E57Document()
    doc.source_path = path
    doc.warnings = []
    scan_col = cols.get("scan")
    if scan_col is None:
        groups = [(0, np.arange(n))]
    else:
        uniq = np.unique(scan_col)
        groups = [(int(u), np.nonzero(scan_col == u)[0]) for u in uniq]
    for k, (sid, idx) in enumerate(groups):
        sub = {kk: vv[idx] for kk, vv in cols.items() if vv is not None}
        extra = {kk: vv[idx] for kk, vv in cloud.extras.items()}
        name = f"Scan_{sid + 1:03d}" if len(groups) > 1 else "Scan_001"
        doc.scans.append(_scan_from_columns(sub, extra, cloud.types, name))
        doc.scan_order.append(k)
    return doc


def _scan_from_columns(c: Dict[str, np.ndarray], extras: Dict[str, np.ndarray],
                       types: Dict[str, str], name: str) -> Scan:
    s = Scan(name=name, guid="")
    xyz = np.column_stack([c["x"], c["y"], c["z"]]).astype(np.float32)
    s.points["cartesianX"] = xyz[:, 0]
    s.points["cartesianY"] = xyz[:, 1]
    s.points["cartesianZ"] = xyz[:, 2]
    s.fields = [
        FieldDef(name="cartesianX", kind="Float", precision="single"),
        FieldDef(name="cartesianY", kind="Float", precision="single"),
        FieldDef(name="cartesianZ", kind="Float", precision="single"),
    ]
    # 颜色统一补齐：没有颜色的点云按渲染器的默认灰（180）写入，
    # 这样后续无论存成哪种格式、颜色列都齐整，不必到处判空。
    n = len(xyz)
    for key, nm in (("r", "colorRed"), ("g", "colorGreen"), ("b", "colorBlue")):
        col = c.get(key)
        s.points[nm] = (col.astype(np.uint8) if col is not None
                        else np.full(n, 180, np.uint8))
        s.fields.append(FieldDef(name=nm, kind="Integer", minimum=0,
                                 maximum=255, byte_size=1))
    if c.get("intensity") is not None:
        s.points["intensity"] = c["intensity"].astype(np.float32)
        s.fields.append(FieldDef(name="intensity", kind="Float", precision="single"))
    if c.get("label") is not None:
        s.points[LABEL_FIELD] = c["label"].astype(np.uint8)
        s.fields.append(FieldDef(name=LABEL_FIELD, kind="Integer", minimum=0,
                                 maximum=255, byte_size=1))
    if c.get("instance") is not None:
        s.points[INSTANCE_FIELD] = c["instance"].astype(np.uint16)
        s.fields.append(FieldDef(name=INSTANCE_FIELD, kind="Integer", minimum=0,
                                 maximum=65535, byte_size=2))
    # 未识别的附加属性：进 points（长度一致，不影响裁剪/渲染），
    # 但**不进 fields**——这样 e57 不会写出私有字段，ply/pcd 写回时再带上。
    for nm, col in extras.items():
        arr = np.asarray(col)
        if arr.shape[0] != n:
            continue
        s.points[nm] = arr
        s.extra_props.append(nm)
        s.extra_types[nm] = types.get(nm, _np_type_tag(arr.dtype))
    return s


def _np_type_tag(dt) -> str:
    """numpy dtype → f4/f8/u1/u2/i4/u4 这类短标签（写 ply/pcd 用）。"""
    dt = np.dtype(dt)
    if dt.kind == "f":
        return "f8" if dt.itemsize >= 8 else "f4"
    if dt.kind == "u":
        return {1: "u1", 2: "u2", 4: "u4"}.get(dt.itemsize, "u4")
    return {1: "i1", 2: "i2", 8: "i8"}.get(dt.itemsize, "i4")


# ---- PLY ----

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}

# 常见别名 → 内部列名
_ALIASES = {
    "x": "x", "coordx": "x",
    "y": "y", "coordy": "y",
    "z": "z", "coordz": "z",
    "red": "r", "r": "r", "diffusered": "r",
    "green": "g", "g": "g", "diffusegreen": "g",
    "blue": "b", "b": "b", "diffuseblue": "b",
    "intensity": "intensity", "scalarintensity": "intensity", "i": "intensity",
    "reflectance": "intensity",
    "label": "label", "annotationlabel": "label", "scalarlabel": "label",
    "class": "label", "semantic": "label",
    "instance": "instance", "annotationinstance": "instance",
    "scalarinstance": "instance", "object": "instance",
    "scan": "scan", "scanindex": "scan", "scanid": "scan",
}


def _map_columns(names: List[str], data: np.ndarray, tags: Optional[List[str]] = None,
                 raw: Optional[np.ndarray] = None) -> _Cloud:
    """按属性名挑出 x/y/z/rgb/…；认不出的列原样收进 extras。

    ``raw`` 是结构化数组（二进制格式直接按原 dtype 取，保证附加属性逐字节还原）；
    文本格式没有 raw，附加属性按 ``tags`` 或默认 float32 记录。
    """
    cloud = _Cloud()
    cols = cloud.cols
    for i, nm in enumerate(names):
        key = _ALIASES.get(nm.strip().lower().replace("_", "").replace("-", ""))
        if key:
            if key not in cols:
                cols[key] = data[:, i]
            continue
        if raw is not None and raw.dtype.names and nm in raw.dtype.names:
            col = np.array(raw[nm])
        else:
            col = np.asarray(data[:, i]).astype(np.float64)
        if nm in cloud.extras:                 # 重名列只留第一份
            continue
        cloud.extras[nm] = col
        tag = tags[i] if tags and i < len(tags) and tags[i] else None
        if tag is None:
            if raw is None:                    # 文本：没有原始宽度，按整/浮自动定
                tag = "i4" if np.all(np.rint(col) == col) else "f4"
            else:
                tag = _np_type_tag(col.dtype)
        cloud.types[nm] = tag
    for k in ("x", "y", "z"):
        if k not in cols:
            raise ValueError(f"点云里缺少 {k.upper()} 坐标列（现有列：{', '.join(names[:8])}）")
    # 颜色：float 0..1 或 0..255、uint16 0..65535 都归一化到 0..255
    if "r" in cols and "g" in cols and "b" in cols:
        for k in ("r", "g", "b"):
            cols[k] = _to_u8(cols[k])
    else:
        for k in ("r", "g", "b"):
            cols.pop(k, None)
    if "label" in cols:
        cols["label"] = np.clip(np.rint(cols["label"]), 0, 255).astype(np.uint8)
    if "instance" in cols:
        cols["instance"] = np.clip(np.rint(cols["instance"]), 0, 65535).astype(np.uint16)
    return cloud


def _to_u8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind == "f":
        mx = float(np.nanmax(a)) if a.size else 0.0
        if mx <= 1.0001:                     # 0..1 浮点颜色
            a = a * 255.0
    elif a.dtype == np.uint16:
        a = a / 257.0                        # 0..65535 颜色
    return np.clip(np.rint(np.nan_to_num(a)), 0, 255).astype(np.uint8)


def _read_ply(path: str) -> _Cloud:
    with open(path, "rb") as fh:
        if fh.readline().strip().lower() != b"ply":
            raise ValueError("不是 PLY 文件（缺少 ply 魔数）")
        fmt = ""
        nvert = 0
        props: List[Tuple[str, Optional[str]]] = []
        in_vertex = False
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError("PLY 头不完整（缺少 end_header）")
            t = raw.decode("ascii", "replace").strip()
            low = t.lower()
            if low.startswith("format"):
                fmt = low.split()[1]
            elif low.startswith("element"):
                p = low.split()
                in_vertex = len(p) > 1 and p[1] == "vertex"
                if in_vertex:
                    nvert = int(p[2])
            elif low.startswith("property") and in_vertex:
                p = low.split()
                if p[1] == "list":           # 顶点上的 list 属性不支持，跳过
                    props.append((p[-1], None))
                else:
                    props.append((p[2], _PLY_TYPES.get(p[1])))
            elif low == "end_header":
                break
        if nvert <= 0:
            raise ValueError("PLY 里没有顶点（element vertex 0）")
        if any(dt is None for _n, dt in props):
            raise ValueError("PLY 顶点的 list 属性暂不支持，请用 CloudCompare 转成普通顶点")
        names = [n for n, _dt in props]
        dts = [dt for _n, dt in props]
        if fmt == "ascii":
            arr = np.loadtxt(fh, dtype=np.float64, ndmin=2, max_rows=nvert)
            data = np.atleast_2d(arr)
            if data.shape[0] != nvert:
                raise ValueError(f"PLY 声明 {nvert} 个点，实际读到 {data.shape[0]} 行")
            return _map_columns(names, data, tags=dts)
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if fmt.endswith("little_endian") else ">"
            dt = np.dtype([(n, endian + t) for n, t in zip(names, dts)])
            buf = fh.read(dt.itemsize * nvert)
            if len(buf) < dt.itemsize * nvert:
                raise ValueError("PLY 数据段不足（文件可能被截断）")
            rec = np.frombuffer(buf, dtype=dt)
            data = np.empty((nvert, len(names)), np.float64)
            for i, n in enumerate(names):
                data[:, i] = rec[n]
            # 传 raw：附加属性按原 dtype 取出，写回时逐字节还原（法向/尺度不缩水）
            return _map_columns(names, data, tags=dts, raw=rec)
        else:
            raise ValueError(f"不支持的 PLY 编码：{fmt}")


# ---- PCD ----

_PCD_TYPES = {"F": "f", "U": "u", "I": "i"}


def _read_pcd(path: str) -> _Cloud:
    header: Dict[str, List[str]] = {}
    with open(path, "rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError("PCD 头不完整（缺少 DATA 行）")
            t = raw.decode("ascii", "replace").strip()
            if not t or t.startswith("#"):
                continue
            p = t.split()
            header[p[0].upper()] = p[1:]
            if p[0].upper() == "DATA":
                break
        fields = header.get("FIELDS") or header.get("FIELD")
        if not fields:
            raise ValueError("PCD 缺少 FIELDS 行")
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        sizes = [int(v) for v in header.get("SIZE", ["4"] * len(fields))]
        types = [v.upper() for v in header.get("TYPE", ["F"] * len(fields))]
        npts = int((header.get("POINTS") or ["0"])[0])
        width = int((header.get("WIDTH") or ["0"])[0])
        if npts <= 0:
            npts = width
        mode = (header.get("DATA") or ["ascii"])[0].lower()
        if mode == "binary_compressed":
            raise ValueError("PCD 的 binary_compressed 编码暂不支持，"
                             "请用 CloudCompare 另存为 binary 或 ascii")
        if any(v != 1 for v in counts):
            raise ValueError("PCD 里存在 COUNT>1 的复合字段（如 histogram），暂不支持")
        names = [nm for i, nm in enumerate(fields) if counts[i] == 1]
        dts = [_PCD_TYPES.get(types[i], "f") + str(sizes[i])
               for i in range(len(fields)) if counts[i] == 1]
        if not names:
            raise ValueError("PCD 里没有可用的单值字段")
        rec = None
        if mode == "ascii":
            data = np.atleast_2d(np.loadtxt(fh, dtype=np.float64, ndmin=2))
        else:
            endian = "<"
            dt = np.dtype([(nm, endian + t) for nm, t in zip(names, dts)])
            buf = fh.read(dt.itemsize * npts)
            if len(buf) < dt.itemsize * npts:
                raise ValueError("PCD 数据段不足（文件可能被截断）")
            rec = np.frombuffer(buf, dtype=dt)
            data = np.empty((len(rec), len(names)), np.float64)
            for i, nm in enumerate(names):
                data[:, i] = rec[nm]
    # rgb 打包在 4 字节里（float 位模式或 uint32 都按位解释）
    low = [n.lower() for n in names]
    for tag in ("rgb", "rgba"):
        if tag in low:
            j = low.index(tag)
            u = data[:, j].astype(np.float32).view(np.uint32)
            data = np.column_stack([
                data,
                ((u >> 16) & 0xFF).astype(np.float64),
                ((u >> 8) & 0xFF).astype(np.float64),
                (u & 0xFF).astype(np.float64),
            ])
            names = names + ["red", "green", "blue"]
            dts = list(dts) + [_np_type_tag(np.uint8)] * 3
            break
    return _map_columns(names, data, tags=dts, raw=rec)


# ---- 文本 / CSV ----

def _read_text(path: str) -> _Cloud:
    """读 xyz / txt / pts / csv：支持注释列名、CSV 表头、PTS 首行点数。"""
    ext = ext_of(path)
    comments: List[str] = []
    names: Optional[List[str]] = None
    skip = 0
    first_data = ""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for i in range(200):
            ln = fh.readline()
            if not ln:
                break
            t = ln.strip()
            if not t:
                skip = i + 1
                continue
            if t.startswith("#") or t.startswith("//"):
                comments.append(t.lstrip("#/ ").strip())
                skip = i + 1
                continue
            toks = [x for x in _split_tokens(t) if x]
            if i == 0 and len(toks) == 1 and toks[0].isdigit():
                skip = 1                    # Leica PTS：首行是点数
                continue
            if toks and all(_is_number(x) for x in toks):
                first_data = t
                skip = i                    # 数据从这一行开始
                break
            names = toks or None            # 非数值行 = 表头
            skip = i + 1
            break

    if "," in first_data or ext == ".csv":
        delim: Optional[str] = ","
    elif ";" in first_data:
        delim = ";"
    else:
        delim = None

    # 注释里的列名（本软件写出的 xyz/txt 第三行就是列名）
    if names is None:
        for c in comments:
            toks = [x for x in _split_tokens(c) if x]
            if len(toks) >= 3 and all(x.lower() in _ALIASES for x in toks):
                names = toks
                break

    data = np.atleast_2d(np.loadtxt(path, dtype=np.float64, delimiter=delim,
                                    skiprows=skip, ndmin=2))
    if data.size == 0:
        raise ValueError("文本里没有读到数值行")
    ncol = data.shape[1]
    if not names or len(names) != ncol:
        names = _guess_names(ncol)
    return _map_columns(names, data)


def _split_tokens(t: str) -> List[str]:
    for d in (",", ";", "\t"):
        if d in t:
            return [x.strip() for x in t.split(d)]
    return t.split()


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _guess_names(n: int) -> List[str]:
    """没有表头时按列数猜：xyz [rgb] [label instance] [intensity]。"""
    if n == 3:
        return ["x", "y", "z"]
    if n == 4:
        return ["x", "y", "z", "intensity"]
    if n == 6:
        return ["x", "y", "z", "red", "green", "blue"]
    if n == 7:
        return ["x", "y", "z", "red", "green", "blue", "label"]
    if n == 8:
        return ["x", "y", "z", "red", "green", "blue", "label", "instance"]
    if n >= 9:
        return ["x", "y", "z", "red", "green", "blue", "label", "instance",
                "intensity"] + [f"c{i}" for i in range(9, n)]
    return ["x", "y", "z"][:n] + [f"c{i}" for i in range(3, n)]


# --------------------------------------------------------------------- 写 --

def save(path: str, doc: E57Document, ann) -> None:
    """按扩展名写出；标注随几何一起写入（e57 由 MainWindow 走原生写入）。"""
    ext = ext_of(path)
    cloud = _collect(doc, ann)
    if ext == ".ply":
        _write_ply(path, cloud)
    elif ext == ".pcd":
        _write_pcd(path, cloud)
    elif ext in (".xyz", ".txt", ".csv", ".pts"):
        _write_text(path, cloud.cols, ext)
    elif ext == ".e57":
        raise ValueError("e57 请走 E57Document.save")
    else:
        raise ValueError(f"不支持的保存格式：{ext or '（无扩展名）'}")


def _collect(doc: E57Document, ann) -> _Cloud:
    """把多扫描合并成一份列字典；扫描号只在多于一个扫描时写。

    附加属性（法向、scale 等）取各站的并集，缺列的站补 0——
    这样 PLY→PLY 往返不会把原文件里的 nx/ny/nz/scale_* 弄丢。
    """
    xs, ys, zs, rs, gs, bs, its, labs, inss, sids = ([] for _ in range(10))
    used = [s for s in doc.scans if s.count > 0]
    multi = len(used) > 1
    for i, s in enumerate(doc.scans):
        n = s.count
        if n == 0:
            continue
        g = s.points.get
        if g("cartesianX") is None:
            continue
        xs.append(np.asarray(g("cartesianX"), np.float32))
        ys.append(np.asarray(g("cartesianY"), np.float32))
        zs.append(np.asarray(g("cartesianZ"), np.float32))
        has_rgb = all(g(k) is not None for k in ("colorRed", "colorGreen", "colorBlue"))
        rs.append(np.asarray(g("colorRed"), np.uint8) if has_rgb else np.full(n, 180, np.uint8))
        gs.append(np.asarray(g("colorGreen"), np.uint8) if has_rgb else np.full(n, 180, np.uint8))
        bs.append(np.asarray(g("colorBlue"), np.uint8) if has_rgb else np.full(n, 180, np.uint8))
        it = g("intensity")
        its.append(np.asarray(it, np.float32) if it is not None else np.zeros(n, np.float32))
        lab = ann.labels[i] if i < len(ann.labels) else None
        ins = ann.instances[i] if i < len(ann.instances) else None
        labs.append(np.asarray(lab, np.uint8) if lab is not None else np.zeros(n, np.uint8))
        inss.append(np.asarray(ins, np.uint16) if ins is not None else np.zeros(n, np.uint16))
        sids.append(np.full(n, i + 1, np.int32))
    if not xs:
        raise ValueError("没有任何点可写入")
    cloud = _Cloud(cols={
        "x": np.concatenate(xs), "y": np.concatenate(ys), "z": np.concatenate(zs),
        "r": np.concatenate(rs), "g": np.concatenate(gs), "b": np.concatenate(bs),
        "intensity": np.concatenate(its),
        "label": np.concatenate(labs), "instance": np.concatenate(inss),
    })
    if multi:
        cloud.cols["scan"] = np.concatenate(sids)

    order: List[str] = []
    for s in used:
        for nm in getattr(s, "extra_props", []) or []:
            if nm not in order:
                order.append(nm)
    for nm in order:
        parts = []
        for s in used:
            col = s.points.get(nm)
            if col is None:
                parts.append(np.zeros(s.count, np.float32))
            else:
                parts.append(np.asarray(col))
                cloud.types[nm] = getattr(s, "extra_types", {}).get(
                    nm, _np_type_tag(np.asarray(col).dtype))
        cloud.extras[nm] = np.concatenate(parts)
        cloud.types.setdefault(nm, "f4")
    return cloud


# PLY / PCD 的类型名 ↔ numpy 短标签
_PLY_OF_TAG = {"f4": "float", "f8": "double", "u1": "uchar", "u2": "ushort",
               "u4": "uint", "i1": "char", "i2": "short", "i4": "int", "i8": "int64"}
_PCD_OF_TAG = {"f4": ("F", 4), "f8": ("F", 8), "u1": ("U", 1), "u2": ("U", 2),
               "u4": ("U", 4), "i1": ("I", 1), "i2": ("I", 2), "i4": ("I", 4),
               "i8": ("I", 8)}


def _write_ply(path: str, cloud: _Cloud) -> None:
    c = cloud.cols
    n = len(c["x"])
    has_s = "scan" in c
    # 强度全 0 时不写这一列（多数模型的 PLY 没有强度，省 4 B/点）
    has_i = bool(np.any(c["intensity"] != 0))
    # 没有任何标注时不写 label/instance：裁剪、纯几何导出应与源文件字段一致，
    # 否则 CloudCompare 里会平白多出两个全 0 的标量场。label 与 instance 成对写。
    has_ann = bool(np.any(c["label"] != 0) or np.any(c["instance"] != 0))
    props = [("x", "float", "f4"), ("y", "float", "f4"), ("z", "float", "f4"),
             ("red", "uchar", "u1"), ("green", "uchar", "u1"), ("blue", "uchar", "u1")]
    if has_i:
        props.append(("intensity", "float", "f4"))
    if has_ann:
        props += [("label", "uchar", "u1"), ("instance", "ushort", "u2")]
    if has_s:
        props.append(("scan", "int", "i4"))
    for nm in cloud.extras:
        tag = cloud.types.get(nm, "f4")
        props.append((nm, _PLY_OF_TAG.get(tag, "float"), tag))
    head = ["ply", "format binary_little_endian 1.0",
            "comment PointCloudAnnotator"]
    if has_ann:
        head += ["comment label: 0=unlabelled 1=spike 2=leaf 3=stem 4=other",
                 "comment instance: 0 = semantic label (no instance id)"]
    head.append(f"element vertex {n}")
    head += [f"property {t} {nm}" for nm, t, _d in props]
    head += ["end_header"]
    dt = np.dtype([(nm, "<" + d) for nm, _t, d in props])
    rec = np.empty(n, dtype=dt)
    rec["x"], rec["y"], rec["z"] = c["x"], c["y"], c["z"]
    rec["red"], rec["green"], rec["blue"] = c["r"], c["g"], c["b"]
    if has_i:
        rec["intensity"] = c["intensity"]
    if has_ann:
        rec["label"] = c["label"]
        rec["instance"] = c["instance"]
    if has_s:
        rec["scan"] = c["scan"]
    for nm in cloud.extras:
        rec[nm] = cloud.extras[nm]
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii", "replace"))
        rec.tofile(fh)


def _write_pcd(path: str, cloud: _Cloud) -> None:
    c = cloud.cols
    n = len(c["x"])
    has_s = "scan" in c
    has_i = bool(np.any(c["intensity"] != 0))
    has_ann = bool(np.any(c["label"] != 0) or np.any(c["instance"] != 0))
    rgb = ((c["r"].astype(np.uint32) << 16) | (c["g"].astype(np.uint32) << 8)
           | c["b"].astype(np.uint32)).view(np.float32)
    fields = ["x", "y", "z", "rgb"]
    sizes = [4, 4, 4, 4]
    types = ["F", "F", "F", "F"]
    tags = ["f4", "f4", "f4", "f4"]
    if has_i:
        fields.append("intensity"); sizes.append(4); types.append("F"); tags.append("f4")
    if has_ann:
        fields += ["label", "instance"]
        sizes += [1, 2]
        types += ["U", "U"]
        tags += ["u1", "u2"]
    if has_s:
        fields.append("scan"); sizes.append(4); types.append("I"); tags.append("i4")
    for nm, col in cloud.extras.items():
        t, sz = _PCD_OF_TAG.get(cloud.types.get(nm, "f4"), ("F", 4))
        fields.append(nm); sizes.append(sz); types.append(t)
        tags.append(cloud.types.get(nm, "f4"))
    head = ["# .PCD v0.7 - Point Cloud Data file format"]
    if has_ann:
        head.append("# PointCloudAnnotator: label 0=unlabelled 1=spike 2=leaf 3=stem 4=other")
    head += [
        "VERSION 0.7",
        "FIELDS " + " ".join(fields),
        "SIZE " + " ".join(str(v) for v in sizes),
        "TYPE " + " ".join(types),
        "COUNT " + " ".join("1" for _ in fields),
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA binary",
    ]
    dt = np.dtype([(nm, "<" + tag) for nm, tag in zip(fields, tags)])
    rec = np.empty(n, dtype=dt)
    rec["x"], rec["y"], rec["z"] = c["x"], c["y"], c["z"]
    rec["rgb"] = rgb
    if has_i:
        rec["intensity"] = c["intensity"]
    if has_ann:
        rec["label"] = c["label"]
        rec["instance"] = c["instance"]
    if has_s:
        rec["scan"] = c["scan"]
    for nm in cloud.extras:
        rec[nm] = cloud.extras[nm]
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii", "replace"))
        rec.tofile(fh)


def _write_text(path: str, c: Dict[str, np.ndarray], ext: str) -> None:
    n = len(c["x"])
    has_s = "scan" in c
    cols = ["x", "y", "z", "r", "g", "b", "label", "instance", "intensity"]
    if has_s:
        cols.append("scan")
    sep = "," if ext == ".csv" else " "
    head = []
    if ext == ".csv":
        head.append(",".join(cols))
    else:
        head.append(f"# {APP_TAG}")
        head.append("# label 0 未标注 / 1 穗 / 2 叶 / 3 茎 / 4 其他；instance 0 = 语义标注")
        head.append("# " + " ".join(cols))
    x, y, z = c["x"], c["y"], c["z"]
    r, g, b = c["r"], c["g"], c["b"]
    lab, ins, it = c["label"], c["instance"], c["intensity"]
    scan = c.get("scan")
    block = 4096
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(head) + "\n")
        for i in range(0, n, block):
            j = min(i + block, n)
            lines = []
            for k in range(i, j):
                parts = [f"{x[k]:.3f}", f"{y[k]:.3f}", f"{z[k]:.3f}",
                         str(int(r[k])), str(int(g[k])), str(int(b[k])),
                         str(int(lab[k])), str(int(ins[k])), f"{float(it[k]):.2f}"]
                if has_s:
                    parts.append(str(int(scan[k])))
                lines.append(sep.join(parts))
            fh.write("\n".join(lines) + "\n")
