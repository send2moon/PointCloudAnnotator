"""
e57io.py —— 自实现 ASTM E57 点云读写（无第三方依赖，仅 numpy）

E57 文件物理布局
----------------
    +0   48 字节  文件头：魔数 "ASTM-E57" / 版本 / 文件长度 / XML 偏移与长度 / 页大小
    +48  若干数据页：每页 1024 字节 = 32 字节页头 + 992/1020 字节载荷
         页头 = CRC32C(4,大端) | physLength(4,小端) | logicalLength(4,小端) | 20 字节保留
    +X   XML 段（明文，尾部允许填充）
    +Y   XML 段末尾若有十字节对齐填充，忽略

CompressedVector 的逐点数据按"列分包"（section）排列：每个 section 连续存放 n 个点的
同一字段。本项目统一不压缩（codecs 留空），读取时兼容该情况。

对外 API
--------
E57Document.read(path)     -> 读取全部扫描
E57Document.save(path)     -> 写回（保留字段/颜色/强度/位姿）
"""

from __future__ import annotations

import os
import re
import struct
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FILE_MAGIC = b"ASTM-E57"
PHYS_PAGE = 1024
LOGICAL_PAGE = PHYS_PAGE - 4
FILE_HEADER_LEN = 48
DATA_SECTION_LEN = 32
XML_NS = "http://www.astm.org/COMMIT/E57/2010-e57-v1.0"

# 自定义标号字段（本软件写入用）
LABEL_FIELD = "annotationLabel"
INSTANCE_FIELD = "annotationInstance"
# 内部序号字段，用于把标注结果映射回原始点序；不写入文件
SOURCE_INDEX_FIELD = "sourceIndex"


# --------------------------------------------------------------- CRC32C ----

class _CRC:
    def __init__(self) -> None:
        self.table = [0] * 256
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ 0x82F63B78 if c & 1 else (c >> 1)
            self.table[i] = c

    def compute(self, data: bytes) -> int:
        c = 0xFFFFFFFF
        t = self.table
        for b in data:
            c = (c >> 8) ^ t[(c ^ b) & 0xFF]
        return c ^ 0xFFFFFFFF


_CRC = _CRC()


def crc32c(data: bytes) -> int:
    return _CRC.compute(data)


# ------------------------------------------------------------- 数据结构 ----

_FLOAT_DT = {"single": np.dtype("<f4"), "double": np.dtype("<f8")}
_INT_DT = {1: np.dtype("<i1"), 2: np.dtype("<i2"), 4: np.dtype("<i4"), 8: np.dtype("<i8")}


@dataclass
class FieldDef:
    name: str
    kind: str                    # 'Float' | 'Integer' | 'ScaledInteger'
    precision: str = "single"
    minimum: float = 0.0
    maximum: float = 0.0
    scale: float = 1.0
    offset: float = 0.0
    byte_size: int = 4

    @property
    def dtype(self) -> np.dtype:
        if self.kind == "Float":
            return _FLOAT_DT[self.precision]
        return _INT_DT.get(self.byte_size, np.dtype("<i4"))

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize


@dataclass
class Scan:
    """一个扫描（一家传感器的完整采集）。points 里每个键是一列数组。"""
    name: str = "Scan"
    guid: str = ""
    fields: List[FieldDef] = field(default_factory=list)
    points: Dict[str, np.ndarray] = field(default_factory=dict)
    # 注意：E57 的四元数顺序是 (w, x, y, z)，单位元为 w=1；
    # 写成 (0,0,0,1) 会被解释成"绕 Z 轴 180°"，整份点云会镜像。
    rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def count(self) -> int:
        if not self.points:
            return 0
        return int(len(next(iter(self.points.values()))))

    def column(self, name: str) -> Optional[np.ndarray]:
        return self.points.get(name)

    def ensure_field(self, name: str, kind: str = "Integer", default: int = 0,
                     byte_size: int = 2) -> FieldDef:
        """确保存在某字段并返回其定义（不存在则补一个）。"""
        for f in self.fields:
            if f.name == name:
                if name not in self.points:
                    self.points[name] = np.full(self.count, default, dtype=f.dtype)
                return f
        fd = FieldDef(name=name, kind=kind, byte_size=byte_size,
                      minimum=0, maximum=255 if byte_size == 1 else 65535)
        self.fields.append(fd)
        self.points[name] = np.full(self.count, default, dtype=fd.dtype)
        return fd

    # ----------------------------------------------------------- pose --

    def has_pose(self, tol: float = 1e-9) -> bool:
        """是否带有非单位 pose（需要把局部坐标变换到文件坐标系）。"""
        w, x, y, z = self.rotation
        identity_q = (abs(abs(w) - 1.0) < 1e-6 and abs(x) < 1e-6
                      and abs(y) < 1e-6 and abs(z) < 1e-6)
        identity_t = all(abs(v) < tol for v in self.translation)
        return not (identity_q and identity_t)

    def apply_pose(self) -> bool:
        """把局部坐标变换到文件坐标系：p' = R·p + t。

        E57 的 ``<pose>`` 描述的是"扫描局部坐标系 → 文件坐标系"的刚体变换，
        libE57/pye57 读出来的原始列是**未变换**的局部坐标，必须自行应用；
        CloudCompare 就是按此规则显示，所以同一文件在本软件里会散成一圈
        而在 CloudCompare 里是完整的一株小区。
        """
        if not self.has_pose():
            return False
        x = self.points.get("cartesianX")
        y = self.points.get("cartesianY")
        z = self.points.get("cartesianZ")
        if x is None or y is None or z is None:
            return False
        if len(x) == 0:
            return False
        m = min(len(x), len(y), len(z))
        rot = _quat_matrix(self.rotation)
        pts = np.empty((m, 3), np.float64)
        pts[:, 0] = x[:m]
        pts[:, 1] = y[:m]
        pts[:, 2] = z[:m]
        out = pts @ rot.T
        out += np.asarray(self.translation, np.float64)
        dtype = x.dtype if x.dtype.kind == "f" else np.float32
        for k, name in enumerate(("cartesianX", "cartesianY", "cartesianZ")):
            col = self.points[name]
            if not col.flags.writeable:      # frombuffer 得到的是只读视图
                col = np.array(col)
                self.points[name] = col
            col[:m] = out[:, k].astype(dtype)
        # 已经烘焙进坐标，重置为单位 pose，避免写回时被二次变换
        self.rotation = (1.0, 0.0, 0.0, 0.0)
        self.translation = (0.0, 0.0, 0.0)
        return True


def _quat_matrix(q: Sequence[float]) -> np.ndarray:
    """四元数 (w,x,y,z) → 3×3 旋转矩阵。"""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float64)


# 本软件自定义写入的字段类型声明（libE57 不认识这些字段，读取时需自行补读）
_ALL_COLUMNS = {
    LABEL_FIELD: FieldDef(name=LABEL_FIELD, kind="Integer",
                          minimum=0, maximum=255, byte_size=1),
    INSTANCE_FIELD: FieldDef(name=INSTANCE_FIELD, kind="Integer",
                             minimum=0, maximum=65535, byte_size=2),
}

# 记录包内最多容纳的记录数（与写入端一致）
RECORDS_PER_PACKET = 1024

# 已知字段的实际存储宽度（字节/点）。E57 的 Integer 只声明 min/max，
# 存储宽度可以是 1/2/4/8 任意一档，规则不足以还原，故对已知字段强制指定。
_KNOWN_WIDTHS = {
    "colorRed": 1, "colorGreen": 1, "colorBlue": 1,
    "cartesianInvalidState": 1, "isColorInvalid": 1, "isIntensityInvalid": 1,
    LABEL_FIELD: 1,
    INSTANCE_FIELD: 2,
    "rowIndex": 4, "columnIndex": 4, "returnIndex": 4, "returnState": 4,
}

# e57 原型声明的 min/max 只决定"能装下"的最小宽度，但 libE57 允许按 2/4/8 字节
# 存储 Integer；这里记录本软件写入时实际使用的每点字节数，读取时优先采用。
_PARSER_COLUMN_BYTES = {
    LABEL_FIELD: 1,
    INSTANCE_FIELD: 2,
}


def _retype_fieldset(fields: List["FieldDef"], per_point: int) -> List["FieldDef"]:
    """把字段集的每点总字节数调整为 ``per_point``（只改 Integer 的宽度）。"""
    out = [fd for fd in fields]
    guess = sum(_PARSER_COLUMN_BYTES.get(fd.name, fd.itemsize) for fd in out)
    base = [FieldDef(**{**fd.__dict__}) for fd in out]
    for i, fd in enumerate(base):
        if fd.kind == "Integer":
            w = _PARSER_COLUMN_BYTES.get(fd.name, fd.itemsize)
            if w == fd.itemsize:
                base[i] = FieldDef(name=fd.name, kind=fd.kind, precision=fd.precision,
                                   minimum=fd.minimum, maximum=fd.maximum,
                                   scale=fd.scale, offset=fd.offset, byte_size=w)
    guess = sum(fd.itemsize for fd in base)
    delta = per_point - guess
    if delta == 0:
        return base
    # 差额逐个摊给 Integer 字段（每次 1 字节），直到对齐或无可调整
    for want in ((1, 2, 4, 8) if delta > 0 else (2, 1)):
        for i, fd in enumerate(base):
            if delta == 0:
                return base
            if fd.kind != "Integer" or fd.itemsize == want:
                continue
            if want not in (1, 2, 4, 8):
                continue
            signed_max = {1: 127, 2: 32767, 4: 2 ** 31 - 1, 8: 2 ** 63 - 1}[want]
            base[i] = FieldDef(name=fd.name, kind=fd.kind, precision=fd.precision,
                               minimum=min(fd.minimum, 0),
                               maximum=max(fd.maximum, signed_max),
                               scale=fd.scale, offset=fd.offset, byte_size=want)
            delta += fd.itemsize - want
    return base


class StructureError(ValueError):
    """结构自检没通过（可能是真损坏，也可能是体检规则过严）。"""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__(
            "e57 结构自检未通过，文件可能已损坏：\n- " + "\n- ".join(self.problems))


@dataclass
class E57Document:
    scans: List[Scan] = field(default_factory=list)
    coordinate_metadata: str = ""
    source_path: str = ""
    scan_order: List[int] = field(default_factory=list)   # 各扫描在文件中的原始序号
    warnings: List[str] = field(default_factory=list)     # 强制打开时记录的体检问题

    @property
    def total_points(self) -> int:
        return sum(s.count for s in self.scans)

    # ------------------------------------------------------------ 读取 --

    @classmethod
    def read(cls, path: str, scan_indices: Optional[Sequence[int]] = None,
             apply_pose: bool = True, force: bool = False) -> "E57Document":
        """用 pye57(libE57) 读取扫描数据，保证与主流软件一致。

        ``apply_pose=True`` 时按 E57 规范把每个扫描的 ``<pose>`` 烘焙进坐标
        （CloudCompare 默认也是这么显示的），否则多站位扫描会保持局部坐标、
        互相错开。读取失败时回落到内置解析器（_Parser）。

        ``force=True`` 跳过结构体检强行读（自检误报或想抢救坏文件时用），
        读出来的问题记在 ``doc.warnings`` 里。
        """
        doc = cls()
        problems = check_e57_structure(path)
        if problems and not force:
            raise StructureError(problems)
        warn = list(problems)
        try:
            import pye57  # 延迟导入：未安装时仍可用内置解析
            if hasattr(pye57, "ScanHeader"):
                reader = pye57.E57(path)
                n = reader.scan_count
                order: List[int] = []
                for i in range(n):
                    if scan_indices is not None and i not in scan_indices:
                        continue
                    hdr = reader.get_header(i)
                    arr = reader.read_scan_raw(i)
                    try:
                        sname = str(hdr.name) if getattr(hdr, "name", None) else f"Scan_{i:03d}"
                    except Exception:
                        sname = f"Scan_{i:03d}"
                    sd = Scan(name=sname, guid=str(getattr(hdr, "guid", "") or ""))
                    # pose 一律从 XML 读：pye57 的 ScanHeader.rotation 分量顺序
                    # 是 (x,y,z,w)，与 E57 的 (w,x,y,z) 相反，容易把单位阵读成
                    # 绕 Z 180°，因此这里不采用。
                    for fname in hdr.point_fields:
                        if fname not in arr:
                            continue
                        col = np.asarray(arr[fname])
                        kind = "Integer" if col.dtype.kind in "iub" else "Float"
                        prec = "single" if col.dtype.itemsize <= 4 else "double"
                        bs = col.dtype.itemsize if kind == "Integer" else 4
                        sd.fields.append(FieldDef(name=fname, kind=kind, precision=prec,
                                                  byte_size=bs))
                        sd.points[fname] = col
                    doc.scans.append(sd)
                    order.append(i)
                if doc.scans:
                    cls._attach_poses(doc, path, apply_pose, order)
                    doc.source_path = path
                    doc.warnings = warn
                    return doc
        except Exception:
            doc = cls()      # 中途失败则丢弃半成品，整体回落

        # 自带标注字段（annotationLabel/annotationInstance）libE57 不认识，
        # read_scan_raw 会直接抛 ValueError。此时改用自带解析器补齐剩余字段：
        # 先让 pye57 忽略未知字段读标准列，再由内置解析器补读标注列。
        try:
            doc = cls._read_with_unknown_fields(path, scan_indices)
            if doc.scans:
                cls._attach_poses(doc, path, apply_pose, doc.scan_order)
                doc.source_path = path
                doc.warnings = warn
                return doc
        except Exception:
            doc = cls()

        with open(path, "rb") as fh:
            blob = fh.read()
        try:
            doc = _Parser().parse(blob, scan_indices)
        except Exception as e:            # noqa: BLE001
            if warn:                      # 强制打开时给出"体检问题 + 读取器报错"的合并说明
                raise ValueError("强制打开失败：\n- " + "\n- ".join(warn)
                                 + f"\n\n内置解析器也报错：{e}") from e
            raise
        if not doc.scans:
            if warn:
                raise ValueError("强制打开失败：\n- " + "\n- ".join(warn)
                                 + "\n\n文件里的扫描结构已不可读（多段长度对不上），"
                                   "请回到原始 e57 重新标注后另存。")
            raise ValueError("这个文件里没有可解析的扫描")
        cls._attach_poses(doc, path, apply_pose)
        doc.source_path = path
        doc.warnings = warn
        return doc

    @classmethod
    def _attach_poses(cls, doc: "E57Document", path: str, apply_pose: bool,
                      order: Optional[Sequence[int]] = None) -> None:
        """从 XML 读出每个扫描的 pose；pye57 的 ScanHeader 不一定给出 rotation/translation。

        ``apply_pose=False`` 时**照样保留 pose**，只是不烘焙进坐标——
        这样存盘时原文件的各站位姿能原样写回去。
        """
        try:
            poses = _read_poses(path)
        except Exception:
            poses = []
        for k, sd in enumerate(doc.scans):
            src = order[k] if order is not None and k < len(order) else k
            if src < len(poses):
                q, t = poses[src]
                sd.rotation = tuple(float(v) for v in q)
                sd.translation = tuple(float(v) for v in t)
            if apply_pose:
                sd.apply_pose()

    @classmethod
    def _read_with_unknown_fields(cls, path: str,
                                  scan_indices: Optional[Sequence[int]] = None) -> "E57Document":
        """pye57 忽略未识别字段读标准列，再用内置解析器补读未知列。"""
        import pye57

        reader = pye57.E57(path)
        doc = cls()
        n = reader.scan_count
        wanted = [i for i in range(n) if scan_indices is None or i in scan_indices]
        for i in wanted:
            hdr = reader.get_header(i)
            try:
                arr = reader.read_scan_raw(i, ignore_unsupported_fields=True)
            except TypeError:                       # 旧版 pye57 无此参数
                arr = reader.read_scan_raw(i)
            sname = str(getattr(hdr, "name", "") or f"Scan_{i:03d}")
            sd = Scan(name=sname, guid=str(getattr(hdr, "guid", "") or ""))
            for fname, col in arr.items():
                col = np.asarray(col)
                kind = "Integer" if col.dtype.kind in "iub" else "Float"
                prec = "single" if col.dtype.itemsize <= 4 else "double"
                bs = col.dtype.itemsize if kind == "Integer" else 4
                sd.fields.append(FieldDef(name=fname, kind=kind, precision=prec,
                                          byte_size=bs))
                sd.points[fname] = col
            doc.scans.append(sd)
        doc.scan_order = list(wanted)

        # 补读 libE57 不认识的列
        known = {fname for sd in doc.scans for fname in sd.points}
        try:
            raw = _Parser().parse_columns(path)
        except Exception:
            raw = {}
        for k, cols in raw.items():
            if k in known or k not in _ALL_COLUMNS:
                continue
            for si, sd in enumerate(doc.scans):
                if si < len(cols) and len(cols[si]) == sd.count:
                    sd.points[k] = cols[si]
                    sd.fields.append(_ALL_COLUMNS[k])
        return doc


    # ------------------------------------------------------------ 写入 --

    def save(self, path: str, name_suffix: str = "") -> None:
        _Writer().write(self, path, name_suffix)

    # ------------------------------------------------------- 便捷访问 --

    def merged_xyz(self) -> np.ndarray:
        """把所有扫描叠成 (N,3) float32 数组。"""
        parts = []
        for s in self.scans:
            x, y, z = s.points.get("cartesianX"), s.points.get("cartesianY"), s.points.get("cartesianZ")
            if x is None:
                continue
            parts.append(np.column_stack((x, y, z)).astype(np.float32))
        if not parts:
            return np.zeros((0, 3), np.float32)
        return np.vstack(parts)


# --------------------------------------------------------------- XML 段 ----

def _depage(buf: bytes, base: int, off: int, length: int) -> bytes:
    """从物理缓冲中取出逻辑连续段：每 1024 字节物理页只取前 1020 字节载荷。

    ``base`` 是 ``buf[0]`` 对应的文件偏移；``off`` 为段的物理起点（不一定页对齐）。
    """
    out = bytearray()
    pos = off
    while len(out) < length:
        page = (pos // PHYS_PAGE) * PHYS_PAGE
        take = min(page + LOGICAL_PAGE - pos, length - len(out))
        if take <= 0:
            break
        a = pos - base
        if a < 0 or a >= len(buf):
            break                     # 源数据已读完（文件被截断），否则会死循环
        chunk = buf[a:a + take]
        if not chunk:
            break
        out += chunk
        pos = page + PHYS_PAGE
    return bytes(out)


def _xml_section(blob: bytes) -> bytes:
    """按物理页取出 XML 段，并剔除每页尾部的 4 字节 CRC32C。

    ``xmlPhysicalOffset`` 不一定页对齐（本数据集是页内偏移 212），所以首段
    只能取到该页载荷末尾，之后每页取满 1020 字节。若直接把
    ``blob[off:off+len]`` 当连续文本，CRC 字节会插进数值中间，
    例如 ``2.066765e+01`` 被污染成 ``2.0N66765e+01``，进而丢失整个扫描。
    """
    if blob[:8] != FILE_MAGIC:
        raise ValueError("不是合法的 e57 文件（文件头魔数不匹配）")
    off, length = struct.unpack_from("<QQ", blob, 24)
    if length <= 0:
        raise ValueError("e57 文件头中的 XML 长度为 0")
    return _depage(blob, 0, off, length)


def _read_logical_file(fh, start: int, need: int) -> bytes:
    """从文件里读逻辑连续的一段（跳过每 1020 字节后的 4 字节 CRC）。"""
    out = bytearray()
    pos = start
    while len(out) < need:
        in_page = pos % LOGICAL_PAGE
        phys = (pos // LOGICAL_PAGE) * PHYS_PAGE + in_page
        take = min(LOGICAL_PAGE - in_page, need - len(out))
        if take <= 0:
            break
        fh.seek(phys)
        chunk = fh.read(take)
        if not chunk:
            break
        out += chunk
        pos += len(chunk)
    return bytes(out)


def check_e57_structure(path: str) -> List[str]:
    """打开前的结构体检：头部、XML 段、各扫描段是否互相自洽。

    在 libE57 的报错（``inconsistent internal state``）之前先给中文原因。
    返回问题列表，空列表表示没问题。
    """
    problems: List[str] = []
    size = os.path.getsize(path)
    if size < PHYS_PAGE:
        return ["文件太小，不是合法的 e57"]
    logical_len = (size // PHYS_PAGE) * LOGICAL_PAGE
    with open(path, "rb") as fh:
        head = fh.read(FILE_HEADER_LEN)
        if head[:8] != FILE_MAGIC:
            return ["不是合法的 e57 文件（文件头魔数不匹配）"]
        phys_len, xml_phys, xml_len, page_size = struct.unpack_from("<QQQQ", head, 16)
        if page_size != PHYS_PAGE:
            problems.append(f"页大小声明为 {page_size}，应为 1024")
        if size % PHYS_PAGE:
            problems.append(f"文件长度 {size:,} 不是页大小 1024 的整数倍")
        if phys_len != size:
            problems.append(f"头部声明文件长 {phys_len:,} 字节，实际 {size:,} 字节")
        xml_log = (xml_phys // PHYS_PAGE) * LOGICAL_PAGE + (xml_phys % PHYS_PAGE)
        if xml_log + xml_len > logical_len:
            problems.append(
                f"XML 段（逻辑偏移 {xml_log:,}，长 {xml_len:,}）超出文件末尾 "
                f"{logical_len:,}（越界 {xml_log + xml_len - logical_len:,} 字节）")
            return problems          # 偏移已不可信，后面的检查没意义
        raw = _read_logical_file(fh, xml_log, xml_len)
        text = raw.split(b"</e57Root>")[0].decode("utf-8", "replace")
        hits = re.findall(
            r'<points type="CompressedVector" fileOffset="(\d+)" recordCount="(\d+)"', text)
        if not hits and "vectorChild" in text:
            problems.append("XML 里找不到任何扫描的 points 节点")
        end = FILE_HEADER_LEN
        for i, (fo, rc) in enumerate(hits):
            off = (int(fo) // PHYS_PAGE) * LOGICAL_PAGE + (int(fo) % PHYS_PAGE)
            if off + DATA_SECTION_LEN > logical_len:
                problems.append(f"第 {i + 1} 个扫描的段头偏移 {off:,} 超出文件末尾")
                break
            sh = _read_logical_file(fh, off, DATA_SECTION_LEN)
            if len(sh) < DATA_SECTION_LEN or sh[0] != 1:
                problems.append(f"第 {i + 1} 个扫描的段头无效（偏移 {off:,}）")
                break
            sec_len = struct.unpack_from("<Q", sh, 8)[0]
            if sec_len <= 0 or off + sec_len > logical_len:
                problems.append(f"第 {i + 1} 个扫描的段长 {sec_len:,} 越出文件")
                break
            if off != end:
                problems.append(f"第 {i + 1} 个扫描的段起始 {off:,} 与上一段末尾 {end:,} 不衔接")
                break
            end = off + sec_len
        else:
            if hits and end != xml_log:
                problems.append(
                    f"点数据段结束于逻辑 {end:,}，但 XML 声明在 {xml_log:,}"
                    f"（相差 {xml_log - end:+,} 字节）")
    return problems


def _read_poses(path: str):
    """返回每个 data3D 子节点的 (rotation(w,x,y,z), translation(x,y,z))。

    只读文件头与 XML 所在页，不把整份点云读进内存。
    """
    with open(path, "rb") as fh:
        head = fh.read(FILE_HEADER_LEN)
        if head[:8] != FILE_MAGIC:
            raise ValueError("不是合法的 e57 文件")
        off, length = struct.unpack_from("<QQ", head, 24)
        if length <= 0:
            return []
        start = (off // PHYS_PAGE) * PHYS_PAGE
        pages = length // LOGICAL_PAGE + 3          # 首段常不满页，多留两页余量
        fh.seek(start)
        buf = fh.read(pages * PHYS_PAGE)
    text = _Parser._clean_xml(_depage(buf, start, off, length)).decode("utf-8", "replace")
    blocks = re.findall(r"<vectorChild[^>]*>(.*?)</vectorChild>", text, re.S)
    poses = []
    for b in blocks:
        m = re.search(r"<pose[^>]*>(.*?)</pose>", b, re.S)
        if not m:
            poses.append(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            continue
        body = m.group(1)
        rm = re.search(r"<rotation[^>]*>(.*?)</rotation>", body, re.S)
        tm = re.search(r"<translation[^>]*>(.*?)</translation>", body, re.S)
        q = tuple(_sub_float(rm.group(1), k) for k in "wxyz") if rm else (1.0, 0.0, 0.0, 0.0)
        t = tuple(_sub_float(tm.group(1), k) for k in "xyz") if tm else (0.0, 0.0, 0.0)
        poses.append((q, t))
    return poses


# --------------------------------------------------------------- 读取器 ----

class _Parser:
    def parse(self, blob: bytes, scan_indices: Optional[Sequence[int]] = None) -> E57Document:
        if blob[:8] != FILE_MAGIC:
            raise ValueError("不是合法的 e57 文件（文件头魔数不匹配）")
        raw = _xml_section(blob)
        end = raw.rfind(b">")
        if end > 0:
            raw = raw[:end + 1]
        text = self._clean_xml(raw).decode("utf-8", "replace")

        doc = E57Document()
        cm = re.search(r"<coordinateMetadata[^>]*>(.*?)</coordinateMetadata>", text, re.S)
        if cm:
            doc.coordinate_metadata = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", cm.group(1), flags=re.S).strip()

        # 每个扫描都是 data3D 下的一个 vectorChild
        chunks = re.findall(r"<vectorChild[^>]*>(.*?)</vectorChild>", text, re.S)
        if not chunks:
            chunks = re.findall(r"<(?:vectorChild|Structure)[^>]*>((?:(?!</vectorChild).)*)</vectorChild>", text, re.S)
        for idx, chunk in enumerate(chunks):
            if scan_indices is not None and idx not in scan_indices:
                continue
            doc.scans.append(self._scan(blob, chunk))
        self._validate(doc)
        return doc

    @staticmethod
    def _validate(doc: E57Document) -> None:
        """内置解析器只支持未经压缩的按列分包（本软件自己写的文件）。

        libE57 写出的文件是位压缩编码，硬解会得到天文数字般的坐标。
        与其把垃圾点画出来，不如明确报错，让用户去装 pye57。
        """
        for sd in doc.scans:
            x = sd.points.get("cartesianX")
            if x is None or len(x) == 0:
                continue
            arr = np.asarray(x, np.float64)
            if not np.all(np.isfinite(arr)) or float(np.abs(arr).max()) > 1e7:
                raise ValueError(
                    "内置解析器读不出该文件的压缩点数据（数值异常）。\n"
                    "请安装 pye57：pip install pye57")

    @staticmethod
    def _clean_xml(raw: bytes) -> bytes:
        """剔除 XML 文本中混入的二进制残字节。

        libE57 / pye57 写头时会插入随机字节，例如
        ``<colorGreenMaximum typ\\x9c\\r\\nte="Integer">``。这些字节不承载信息，
        按"只保留合法字符 + 可完整解码的 UTF-8 序列"过滤即可。
        """
        out = bytearray()
        i, n = 0, len(raw)
        while i < n:
            b = raw[i]
            if b in (0x09, 0x0A, 0x0D) or 0x20 <= b <= 0x7E:
                out.append(b)
                i += 1
            elif b >= 0xC0:
                length = 2 if b < 0xE0 else (3 if b < 0xF0 else 4)
                seq = raw[i:i + length]
                try:
                    seq.decode("utf-8")
                    out += seq
                    i += length
                except UnicodeDecodeError:
                    i += 1
            else:
                i += 1
        text = bytes(out).decode("utf-8", errors="replace")
        return _Parser._repair(text).encode("utf-8")

    # 已知 E57 标签名与属性名，用于修复被脏字节污染的标签
    _TAGS = (
        "e57Root formatName guid versionMajor versionMinor e57LibraryVersion "
        "coordinateMetadata creationDateTime dateTimeValue isAtomicClockReferenced "
        "data3D vectorChild name description indexBounds rowMinimum rowMaximum "
        "columnMinimum columnMaximum returnMinimum returnMaximum "
        "intensityLimits intensityMinimum intensityMaximum "
        "colorLimits colorRedMinimum colorRedMaximum colorGreenMinimum colorGreenMaximum "
        "colorBlueMinimum colorBlueMaximum cartesianBounds xMinimum xMaximum yMinimum "
        "yMaximum zMinimum zMaximum sphericalBounds rangeMinimum rangeMaximum "
        "elevationMinimum elevationMaximum azimuthStart azimuthEnd pose rotation w x y z "
        "translation acquisitionStart acquisitionEnd points prototype codecs "
        "cartesianX cartesianY cartesianZ intensity colorRed colorGreen colorBlue "
        "cartesianInvalidState sphericalRange sphericalAzimuth sphericalElevation "
        "rowIndex columnIndex returnIndex returnState timeStamp isIntensityInvalid "
        "isColorInvalid isCartesianInvalidState images2D vectorImage imageSize "
        "imageWidth imageHeight pixelWidth pixelHeight imageMaskType imageVisualType"
    ).split()
    _ATTRS = ("type name precision minimum maximum scale offset fileOffset recordCount "
              "allowHeterogeneousChildren").split()

    @staticmethod
    def _repair(text: str) -> str:
        """修复被二进制残字节打断的 XML 标签。

        原始头部的脏数据表现为标签名或属性名中间被插入随机字节，例如
        ``<colorGreenMaximum typ\x9c\r\nte="Integer">``、``<r<eturnMinimum .../>``。
        这些残片不承载信息：按已知标签名匹配并重建 ``<name attr="value">`` 即可。
        """
        out = []
        pos = 0
        # 属性名被换行截断（"typ\n  te="）时先合并成一行
        cleaned = []
        for line in text.split("\n"):
            stripped = line.lstrip()
            if stripped and re.match(r"^[A-Za-z][\w-]*\s*=", stripped) and cleaned:
                cleaned[-1] = cleaned[-1] + " " + stripped
            else:
                cleaned.append(line)
        text = "\n".join(cleaned)
        # 先删掉所有"不成标签"的裸 '<'（如 "<r<returnMinimum" 里的第一个 <）
        text = re.sub(r"<(?![A-Za-z/!?])", "", text)
        # 标签名被截断后，形如 "<r<returnMinimum" 的两个 '<' 仍会连在一起：
        # 把中间那个多余 '<' 去掉
        text = re.sub(r"<([A-Za-z]{1,3})(?=<[A-Za-z])", "", text)
        for m in re.finditer(r"<[^<>]*>", text):
            out.append(text[pos:m.start()])
            pos = m.end()
            tok = m.group(0)
            # 前一段可能残留裸 '<'（如 "<r<eturnMinimum"），只从最后一个 '<' 起算标签
            lead = tok[:-1].rfind("<")
            if lead >= 0:
                out.append(tok[:lead])
                tok = tok[lead:]
            out.append(_Parser._fix_tag(tok))
        out.append(text[pos:])
        return "".join(out)

    @staticmethod
    def _fix_tag(tok: str) -> str:
        tags, attrs = _Parser._TAGS, _Parser._ATTRS
        inner = tok[1:-1].strip()
        if not inner or inner[0] in "!?":
            return tok
        closing = inner.startswith("/")
        if closing:
            inner = inner[1:].strip()
        selfclose = inner.endswith("/")
        if selfclose:
            inner = inner[:-1].strip()

        parts = re.split(r"\s+", inner, maxsplit=1)
        head = parts[0] if parts else ""
        key = re.sub(r"[^A-Za-z0-9_:-]", "", head)

        # 标签名：精确命中已知标签则用规范名；否则保留原名（脏字节只污染少数标签，
        # 盲目模糊匹配会把正常名字改坏）。仅当长度差 1 且编辑距离 1 时才纠正。
        name = key
        lower_map = {t.lower(): t for t in tags}
        if key.lower() in lower_map:
            name = lower_map[key.lower()]
        else:
            best, best_d = None, 99
            for t in tags:
                if len(t) < 4 or abs(len(t) - len(key)) > 1:
                    continue
                d = _edit_distance(key.lower(), t.lower())
                if d < best_d:
                    best, best_d = t, d
            if best is not None and best_d <= 1 and key.isalpha():
                name = best
        if not name:
            name = "unknown"

        kept = []
        for a in attrs:
            mm = re.search(a + r'\s*=\s*"([^"]*)"', inner)
            if mm:
                kept.append(a + '="' + mm.group(1) + '"')
        body = name + (" " + " ".join(kept) if kept else "")
        if closing:
            return "</" + name + ">"
        return "<" + body + ("/>" if selfclose else ">")

    def _scan(self, blob: bytes, chunk: str) -> Scan:
        sd = Scan()
        el = re.search(r"<name[^>]*>(.*?)</name>", chunk, re.S)
        sd.name = _cdata(el.group(1)) if el else "Scan"
        el = re.search(r"<guid[^>]*>(.*?)</guid>", chunk, re.S)
        sd.guid = _cdata(el.group(1)) if el else ""

        pose = re.search(r"<pose[^>]*>(.*?)</pose>", chunk, re.S)
        if pose:
            p = pose.group(1)
            rot = re.search(r"<rotation[^>]*>(.*?)</rotation>", p, re.S)
            if rot:
                sd.rotation = tuple(_sub_float(rot.group(1), k) for k in "wxyz")
            tr = re.search(r"<translation[^>]*>(.*?)</translation>", p, re.S)
            if tr:
                sd.translation = tuple(_sub_float(tr.group(1), k) for k in "xyz")

        pts = re.search(r"<points[^>]*>", chunk)
        if pts is None:
            return sd
        tag = pts.group(0)
        mo = re.search(r'recordCount="(\d+)"', tag)
        mf = re.search(r'fileOffset="(\d+)"', tag)
        record_count = int(mo.group(1)) if mo else 0
        file_off = int(mf.group(1)) if mf else 0

        proto = re.search(r"<prototype[^>]*>(.*?)</prototype>", chunk, re.S)
        if proto is None or record_count <= 0:
            return sd
        for fm in re.finditer(r"<([A-Za-z][\w]*)\s+type=\"(Float|Integer|ScaledInteger)\"([^>]*)>",
                              proto.group(1)):
            sd.fields.append(self._field(fm.group(1), fm.group(2), fm.group(3)))
        if not sd.fields:
            return sd

        record_len = sum(fd.itemsize for fd in sd.fields)
        raw = self._read_stream(blob, file_off, record_count * record_len)

        pos = 0
        for fd in sd.fields:
            vals = np.frombuffer(raw, dtype=fd.dtype, count=record_count, offset=pos)
            if fd.kind == "ScaledInteger":
                vals = vals.astype(np.float32) * np.float32(fd.scale) + np.float32(fd.offset)
            sd.points[fd.name] = vals
            pos += record_count * fd.itemsize
        return sd

    @staticmethod
    def _field(name: str, kind: str, attrs: str) -> FieldDef:
        fd = FieldDef(name=name, kind=kind)
        if kind == "Float":
            prec = re.search(r'precision="(\w+)"', attrs)
            fd.precision = "double" if prec and prec.group(1) == "double" else "single"
            fd.minimum = _attr_float(attrs, "minimum")
            fd.maximum = _attr_float(attrs, "maximum")
        elif kind == "Integer":
            fd.minimum = _attr_float(attrs, "minimum", 0.0)
            fd.maximum = _attr_float(attrs, "maximum", 255.0)
            lo, hi = fd.minimum, fd.maximum
            fd.byte_size = 1 if (-128 <= lo and hi <= 127) else (
                2 if (-32768 <= lo and hi <= 32767) else 4)
        else:
            fd.minimum = _attr_float(attrs, "minimum")
            fd.maximum = _attr_float(attrs, "maximum")
            fd.scale = _attr_float(attrs, "scale", 1.0)
            fd.offset = _attr_float(attrs, "offset", 0.0)
            need = max(abs(fd.minimum), abs(fd.maximum))
            fd.byte_size = 1 if need < 128 else (2 if need < 32768 else 4)
        return fd

    @staticmethod
    def _read_stream(blob: bytes, pos: int, need: int) -> bytes:
        """读取一个扫描的全部列数据，自动识别两种物理布局。

        布局 A（本写入器 + libE57 标准）：
            页 = 1020 字节载荷 + 4 字节 CRC32C，``pos`` 指向首字节载荷。
            后续页按 1024 递增，跳过每页末尾 4 字节。
        布局 B（旧式 32 字节页头）：
            页 = 32 字节头 + <=992 字节载荷，``pos`` 指向页头。
        """
        n = len(blob)
        # 先用布局 A 试读，读不满再用布局 B 重试
        out = bytearray()
        l0, l1 = pos, pos + LOGICAL_PAGE
        if l1 <= n:
            out += blob[l0:l1]
        else:
            out += blob[l0:n]
        page = pos + PHYS_PAGE
        while len(out) < need and page + LOGICAL_PAGE <= n:
            out += blob[page:page + LOGICAL_PAGE]
            page += PHYS_PAGE
        if len(out) >= need:
            return bytes(out[:need])

        out = bytearray()
        page = pos
        while len(out) < need:
            if page + DATA_SECTION_LEN > n:
                break
            _plen, llen = struct.unpack_from("<II", blob, page + 4)
            if not (0 < llen <= LOGICAL_PAGE):
                break
            out += blob[page + DATA_SECTION_LEN: page + DATA_SECTION_LEN + llen]
            page += DATA_SECTION_LEN + llen
        if len(out) < need:
            out += b"\x00" * (need - len(out))
        return bytes(out[:need])

    # ------------------------------------------------- 补读未识别字段 --

    @staticmethod
    def _logical_read(blob: bytes, start: int, need: int) -> bytes:
        """按逻辑字节流读 need 字节（跳过每 1020 字节后的 4 字节 CRC）。"""
        out = bytearray()
        pos = start
        n = len(blob)
        while len(out) < need and pos < n:
            phys = (pos // LOGICAL_PAGE) * PHYS_PAGE + (pos % LOGICAL_PAGE)
            take = min(LOGICAL_PAGE - (pos % LOGICAL_PAGE), need - len(out))
            out += blob[phys:phys + take]
            pos += take
        return bytes(out)

    @staticmethod
    def _to_logical(physical: int) -> int:
        return (physical // PHYS_PAGE) * LOGICAL_PAGE + (physical % PHYS_PAGE)

    @staticmethod
    def _pin_width(fd: FieldDef) -> FieldDef:
        """已知字段强制使用约定宽度，避免 min/max 推断错宽度导致列错位。"""
        if fd.kind != "Integer":
            return fd
        w = _KNOWN_WIDTHS.get(fd.name)
        if w is None or w == fd.itemsize:
            return fd
        return FieldDef(name=fd.name, kind=fd.kind, precision=fd.precision,
                        minimum=fd.minimum, maximum=fd.maximum,
                        scale=fd.scale, offset=fd.offset, byte_size=w)

    def parse_columns(self, path: str) -> Dict[str, List[np.ndarray]]:
        """解析 XML 原型并逐个记录包解码，返回 {字段名: [各扫描的列数组]}。

        用于 pye57 拒绝的字段（如自定义的 annotationLabel）。
        ``fileOffset`` 指向首个记录包（section 头 +32 字节之后），包内按
        字段顺序连续存放 1024 个记录；包按 4 字节对齐补齐。
        """
        with open(path, "rb") as fh:
            blob = fh.read()
        if blob[:8] != FILE_MAGIC:
            raise ValueError("不是合法的 e57 文件")
        raw = _xml_section(blob)
        end = raw.rfind(b">")
        if end > 0:
            raw = raw[:end + 1]
        text = self._clean_xml(raw).decode("utf-8", "replace")
        chunks = re.findall(r"<vectorChild[^>]*>(.*?)</vectorChild>", text, re.S)

        out: Dict[str, List[np.ndarray]] = {}
        for chunk in chunks:
            pts = re.search(r"<points[^>]*>", chunk)
            proto = re.search(r"<prototype[^>]*>(.*?)</prototype>", chunk, re.S)
            if pts is None or proto is None:
                continue
            mo = re.search(r'recordCount="(\d+)"', pts.group(0))
            mf = re.search(r'fileOffset="(\d+)"', pts.group(0))
            if not mo or not mf:
                continue
            count = int(mo.group(1))
            sect = int(mf.group(1))
            if count <= 0 or sect <= 0:
                continue
            # fileOffset 指向 section 头（32 字节），记录包紧随其后。
            # 注意：XML 里的 fileOffset 是**物理**偏移（页尾带 4 字节 CRC），
            # 而 _logical_read 收的是逻辑偏移，必须先换算，否则除第一个扫描
            # （物理偏移恰在首页、两者相等）之外全部读错，标注字段会丢。
            sect_logical = self._to_logical(sect)
            sh = self._logical_read(blob, sect_logical, DATA_SECTION_LEN)
            phys_first = struct.unpack_from("<Q", sh, 16)[0] if len(sh) >= 24 else 0
            # section 头里存的是物理偏移，换回逻辑偏移再顺序扫描记录包
            pkt = (self._to_logical(phys_first) if phys_first > 0
                   else sect_logical + DATA_SECTION_LEN)
            fields = [self._field(m.group(1), m.group(2), m.group(3))
                      for m in re.finditer(
                          r"<([A-Za-z][\w]*)\s+type=\"(Float|Integer|ScaledInteger)\"([^>]*)>",
                          proto.group(1))]
            if not fields:
                continue
            fields = [self._pin_width(fd) for fd in fields]
            cols = {fd.name: [] for fd in fields}
            got = 0
            guard = 0
            while got < count and guard < 100000:
                guard += 1
                head = self._logical_read(blob, pkt, 6 + 2 * len(fields))
                if len(head) < 6 + 2 * len(fields):
                    break
                if head[0] & 0x01 == 0:          # 非记录包（如索引包），按长度跳过
                    blen = struct.unpack_from("<H", head, 2)[0] + 1
                    if blen <= 4:
                        break
                    pkt += blen
                    continue
                take = min(RECORDS_PER_PACKET, count - got)
                sizes = [struct.unpack_from("<H", head, 6 + 2 * i)[0] for i in range(len(fields))]
                body = self._logical_read(blob, pkt + 6 + 2 * len(fields), sum(sizes))
                off = 0
                for fd, sz in zip(fields, sizes):
                    npts = sz // fd.itemsize
                    if npts != take or off + sz > len(body):
                        off += sz
                        continue
                    if fd.kind == "ScaledInteger":
                        vals = np.frombuffer(body, fd.dtype, npts, off).astype(np.float32)
                        vals = vals * np.float32(fd.scale) + np.float32(fd.offset)
                    else:
                        vals = np.array(np.frombuffer(body, fd.dtype, npts, off))
                    cols[fd.name].append(vals)
                    off += sz
                blen = struct.unpack_from("<H", head, 2)[0] + 1
                pkt += blen
                got += take
            for fd in fields:
                if cols[fd.name]:
                    out.setdefault(fd.name, []).append(
                        np.concatenate(cols[fd.name]) if len(cols[fd.name]) > 1
                        else cols[fd.name][0])
        return out



def _cdata(text: str) -> str:
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.S)
    return text.strip()


def _attr_float(attrs: str, name: str, default: float = 0.0) -> float:
    m = re.search(name + r'="([^"]*)"', attrs)
    if not m:
        return default
    try:
        return float(m.group(1))
    except ValueError:
        return default


def _sub_float(block: str, tag: str) -> float:
    m = re.search(r"<" + tag + r"(?:\s[^>]*)?>(.*?)</" + tag + r">", block, re.S)
    if not m:
        return 0.0
    try:
        return float(_cdata(m.group(1)))
    except ValueError:
        return 0.0


def _f(node: Optional[ET.Element], default: float = 0.0) -> float:
    if node is None or not node.text:
        return default
    try:
        return float(node.text)
    except ValueError:
        return default


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return 99
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
        prev = cur
    return prev[lb]


def _try_float(text: Optional[str], default: float = 0.0) -> float:
    if text is None:
        return default
    try:
        return float(text)
    except ValueError:
        return default


# --------------------------------------------------------------- 写入器 ----

class _Writer:
    """E57 写入器（libE57 兼容）。

    布局要点（与 libE57Format 一致）：
      * 文件 48 字节头（魔数/版本/文件长度/XML 偏移与长度/页大小）
      * 每个扫描 = 32 字节 section 头 + 若干记录包
          section 头：+0 version=1 | +8 段总长 | +16 首个记录包物理偏移
          记录包：+0 flags=1 | +2 包长-1 | +4 字段数 | +6.. 每字段字节数(u16)
                  随后按字段顺序连续存放该包的点数据
      * 页 = 1020 字节载荷 + 4 字节 CRC32C（大端，覆盖 1020 载荷）
    逻辑偏移到物理偏移需按页换算：phys = (l // 1020) * 1024 + (l % 1020)
    """

    RECORDS_PER_PACKET = 1024

    def write(self, doc: E57Document, path: str, name_suffix: str = "") -> None:
        scans = [s for s in doc.scans if s.count > 0]
        if not scans:
            raise ValueError("没有任何点可写入")

        # 每个扫描的字段布局（保持原始顺序，保证颜色/强度/标注都不丢）
        # 顺带把 Integer 字段的存储宽度与实际数值对齐：E57 原型只声明 min/max，
        # 第三方（libE57 / CloudCompare）按 min/max 反推每点字节数，声明与写入
        # 不一致会被判成"内部状态不一致"而直接拒绝打开。
        layouts = []
        for sd in scans:
            self._check_columns(sd)
            layouts.append(self._fit_integer_fields(sd))

        # 先算逻辑偏移（XML 里要写 fileOffset），再落盘。
        # 注意：逻辑流与物理文件的长度差 = 逻辑流长度 // 1020（每页多 4 字节 CRC）。
        section_lens, logical_starts = [], []
        logical = FILE_HEADER_LEN
        for sd, fields in zip(scans, layouts):
            section_lens.append(DATA_SECTION_LEN + self._packets_bytes(sd, fields))
            logical_starts.append(logical)
            logical += section_lens[-1]
        logical_end = logical

        offsets = [self._to_physical(x) for x in logical_starts]
        xml_off = self._to_physical(logical_end)
        xml = self._xml(doc, scans, layouts, offsets, name_suffix)

        pages = -(-(logical_end + len(xml)) // LOGICAL_PAGE)   # 向上取整
        total = pages * PHYS_PAGE

        with open(path, "w+b") as fh:
            w = _PageWriter(fh)
            header = bytearray(FILE_HEADER_LEN)
            header[0:8] = FILE_MAGIC
            struct.pack_into("<I", header, 8, 1)
            struct.pack_into("<I", header, 12, 0)
            struct.pack_into("<Q", header, 16, total)
            struct.pack_into("<Q", header, 24, xml_off)
            struct.pack_into("<Q", header, 32, len(xml))
            struct.pack_into("<Q", header, 40, PHYS_PAGE)
            w.write(bytes(header))

            for i, (sd, fields) in enumerate(zip(scans, layouts)):
                sh = bytearray(DATA_SECTION_LEN)
                sh[0] = 1
                struct.pack_into("<Q", sh, 8, section_lens[i])
                struct.pack_into("<Q", sh, 16,
                                 self._to_physical(logical_starts[i] + DATA_SECTION_LEN))
                w.write(bytes(sh))
                self._write_packets(w, sd, fields)

            # 段长是按布局算出来的，实际写入必须分毫不差；否则各段偏移会在
            # 文件里整体错位，XML 里的 fileOffset 全部失效（CloudCompare 会报
            # "inconsistent internal state"）。宁可在写的时候就炸掉。
            if w.written != logical_end:
                self._abort(path, f"点数据实写 {w.written:,} 字节，与声明的 "
                                  f"{logical_end:,} 字节不一致（某列长度与扫描点数不同）")

            w.write(xml)
            w.finish()
            fh.flush()
            real = os.path.getsize(path)
            if real != total:
                self._abort(path, f"文件实写 {real:,} 字节，与声明的 {total:,} 字节不一致")
        doc.source_path = path

    @staticmethod
    def _abort(path: str, msg: str) -> None:
        """写出不完整时删掉半成品，避免留下打不开的文件。"""
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        raise RuntimeError("保存失败：" + msg)

    @staticmethod
    def _check_columns(sd: Scan) -> None:
        """所有列必须等长，否则记录包会写不满声明长度。"""
        lens = {name: len(col) for name, col in sd.points.items()}
        if len(set(lens.values())) > 1:
            detail = "、".join(f"{k}={v:,}" for k, v in lens.items())
            raise ValueError(f"扫描「{sd.name}」各列长度不一致，无法写出合法的 e57：{detail}")

    @staticmethod
    def _fit_integer_fields(sd: Scan) -> List[FieldDef]:
        """按实际数值确定 Integer 字段的存储宽度，并同步到 XML 声明。

        libE57 读文件时只看原型里的 ``minimum``/``maximum`` 来推每点字节数，
        声明 0..255（1 字节）却写了 2 字节，第三方读取时整条记录都会错位。
        """
        out: List[FieldDef] = []
        for fd in sd.fields:
            col = sd.points.get(fd.name)
            if col is None:
                continue
            if fd.kind != "Integer":
                out.append(fd)
                continue
            col = np.asarray(col)
            lo = int(col.min()) if col.size else 0
            hi = int(col.max()) if col.size else 0

            def cap(width: int) -> Tuple[int, int]:
                """该宽度能表示的区间（有负值才用有符号区间）。"""
                if width == 1:
                    return (0, 255) if lo >= 0 else (-128, 127)
                if width == 2:
                    return (0, 65535) if lo >= 0 else (-32768, 32767)
                return (0, 2147483647) if lo >= 0 else (-2147483648, 2147483647)

            want = _KNOWN_WIDTHS.get(fd.name)
            if want is None:
                want = 1 if (cap(1)[0] <= lo and hi <= cap(1)[1]) else (
                    2 if (cap(2)[0] <= lo and hi <= cap(2)[1]) else 4)
            elif not (cap(want)[0] <= lo and hi <= cap(want)[1]):
                want = 2 if (cap(2)[0] <= lo and hi <= cap(2)[1]) else 4
            new_fd = FieldDef(name=fd.name, kind="Integer", byte_size=want,
                              minimum=float(lo), maximum=float(hi))
            if fd.itemsize != want:
                sd.points[fd.name] = col.astype(new_fd.dtype)
            out.append(new_fd)
        sd.fields = out
        return out


    @staticmethod
    def _patch_head(fh, header: bytes) -> None:
        """重写首页 48 字节头并刷新首页 CRC（首页载荷 = 头 + 后续内容）。"""
        fh.seek(0)
        page = bytearray(fh.read(LOGICAL_PAGE))
        page[0:FILE_HEADER_LEN] = header
        fh.seek(0)
        fh.write(bytes(page))
        fh.seek(LOGICAL_PAGE)
        fh.write(struct.pack(">I", crc32c(bytes(page))))
        fh.flush()

    # ------------------------------------------------------------ 分包 --

    def _packets_bytes(self, sd: Scan, fields: List[FieldDef]) -> int:
        n = sd.count
        total = 0
        remaining = n
        while remaining > 0:
            take = min(self.RECORDS_PER_PACKET, remaining)
            total += self._packet_len(fields, take)
            remaining -= take
        return total

    @staticmethod
    def _packet_len(fields: List[FieldDef], take: int) -> int:
        raw = 6 + 2 * len(fields) + sum(fd.itemsize * take for fd in fields)
        pad = (4 - raw % 4) % 4
        return raw + pad

    def _write_packets(self, w: "_PageWriter", sd: Scan, fields: List[FieldDef]) -> None:
        n = sd.count
        idx = 0
        while idx < n:
            take = min(self.RECORDS_PER_PACKET, n - idx)
            pkt_len = self._packet_len(fields, take)
            pad = pkt_len - (6 + 2 * len(fields) + sum(fd.itemsize * take for fd in fields))

            head = bytearray(6 + 2 * len(fields))
            head[0] = 1
            struct.pack_into("<H", head, 2, pkt_len - 1)
            struct.pack_into("<H", head, 4, len(fields))
            cur = 6
            for fd in fields:
                struct.pack_into("<H", head, cur, fd.itemsize * take)
                cur += 2
            w.write(bytes(head))

            for fd in fields:
                col = np.asarray(sd.points[fd.name][idx:idx + take])
                w.write(np.ascontiguousarray(col.astype(fd.dtype)).tobytes())
            if pad:
                w.write(b"\x00" * pad)
            idx += take

    @staticmethod
    def _to_physical(logical: int) -> int:
        return (logical // LOGICAL_PAGE) * PHYS_PAGE + (logical % LOGICAL_PAGE)

    @staticmethod
    def _int_range(byte_size: int, lo: int, hi: int) -> Tuple[int, int]:
        """按实际存储宽度给出**能反推出同一宽度**的声明区间。

        libE57 只看 ``minimum``/``maximum`` 决定每点字节数，因此声明区间必须是
        **该宽度的满量程**（1 字节 → 0..255，2 字节 → 0..65535，4 字节 →
        0..2147483647），只写实际值域（例如 0..5）会被反推成 1 字节而与写入
        宽度不符。满量程同时覆盖真实数据，不会触发 ErrorValueOutOfBounds。
        """
        if hi < lo:
            lo, hi = hi, lo
        if byte_size == 1:
            width = (-128, 127) if lo < 0 else (0, 255)
        elif byte_size == 2:
            width = (-32768, 32767) if lo < 0 else (0, 65535)
        else:
            width = (-2147483648, 2147483647) if lo < 0 else (0, 2147483647)
        if not (width[0] <= lo and hi <= width[1]):
            raise ValueError(f"整数取值 {lo}..{hi} 超出 {byte_size} 字节可表示范围")
        return width

    # -------------------------------------------------------------- XML --

    def _xml(self, doc: E57Document, scans: List[Scan], layouts,
             offsets: List[int], name_suffix: str) -> bytes:
        L: List[str] = []
        a = L.append
        a('<?xml version="1.0" encoding="UTF-8"?>\n')
        a(f'<e57Root type="Structure" xmlns="{XML_NS}">\n')
        a('  <formatName type="String">ASTM E57 3D Imaging Data File</formatName>\n')
        a(f'  <guid type="String">{{{uuid.uuid4()}}}</guid>\n')
        a('  <versionMajor type="Integer">1</versionMajor>\n')
        a('  <versionMinor type="Integer">0</versionMinor>\n')
        a('  <e57LibraryVersion type="String">作物点云器官标注软件 1.0</e57LibraryVersion>\n')
        a('  <coordinateMetadata type="String">')
        if doc.coordinate_metadata:
            a(doc.coordinate_metadata.replace("]]>", ""))
        a('</coordinateMetadata>\n')
        a('  <data3D type="Vector" allowHeterogeneousChildren="1">\n')

        for si, (sd, fields) in enumerate(zip(scans, layouts)):
            n = sd.count
            name = (sd.name + name_suffix).replace("<", "").replace(">", "")
            a('    <vectorChild type="Structure">\n')
            a(f'      <guid type="String">{{{uuid.uuid4()}}}</guid>\n')
            a(f'      <name type="String">{name}</name>\n')

            inten = sd.points.get("intensity")
            if inten is not None and n:
                imin = float(np.min(inten)); imax = float(np.max(inten))
            else:
                imin = imax = 0.0
            a('      <intensityLimits type="Structure">\n')
            a(f'        <intensityMinimum type="Float">{imin!r}</intensityMinimum>\n')
            a(f'        <intensityMaximum type="Float">{imax!r}</intensityMaximum>\n')
            a('      </intensityLimits>\n')

            a('      <colorLimits type="Structure">\n')
            for tag in ("Red", "Green", "Blue"):
                a(f'        <color{tag}Minimum type="Integer">0</color{tag}Minimum>\n')
                a(f'        <color{tag}Maximum type="Integer">255</color{tag}Maximum>\n')
            a('      </colorLimits>\n')

            a('      <cartesianBounds type="Structure">\n')
            for ax in "xyz":
                col = sd.points.get("cartesian" + ax.upper())
                lo = float(np.min(col)) if col is not None and n else 0.0
                hi = float(np.max(col)) if col is not None and n else 0.0
                a(f'        <{ax}Minimum type="Float">{lo!r}</{ax}Minimum>\n')
                a(f'        <{ax}Maximum type="Float">{hi!r}</{ax}Maximum>\n')
            a('      </cartesianBounds>\n')

            r, t = sd.rotation, sd.translation
            a('      <pose type="Structure">\n')
            a('        <rotation type="Structure">\n')
            a(f'          <w type="Float">{float(r[0])!r}</w><x type="Float">{float(r[1])!r}</x>'
              f'<y type="Float">{float(r[2])!r}</y><z type="Float">{float(r[3])!r}</z>\n')
            a('        </rotation>\n')
            a('        <translation type="Structure">\n')
            a(f'          <x type="Float">{float(t[0])!r}</x><y type="Float">{float(t[1])!r}</y>'
              f'<z type="Float">{float(t[2])!r}</z>\n')
            a('        </translation>\n')
            a('      </pose>\n')

            a(f'      <points type="CompressedVector" fileOffset="{offsets[si]}" recordCount="{n}">\n')
            a('        <prototype type="Structure">\n')
            for fd in fields:
                col = sd.points[fd.name]
                if fd.kind == "Float":
                    a(f'          <{fd.name} type="Float" precision="{fd.precision}" '
                      f'minimum="{float(np.min(col))!r}" maximum="{float(np.max(col))!r}"/>\n')
                elif fd.kind == "ScaledInteger":
                    a(f'          <{fd.name} type="ScaledInteger" minimum="{fd.minimum!r}" '
                      f'maximum="{fd.maximum!r}" scale="{fd.scale!r}" offset="{fd.offset!r}"/>\n')
                else:
                    # libE57 按 minimum/maximum 反推存储宽度，且逐样本校验取值范围。
                    # 这里声明与实际存储宽度一致的整数区间（1 字节 → 0..255，
                    # 2 字节 → 0..65535 …），既不会越界也不会让记录错位。
                    lo, hi = self._int_range(fd.itemsize, int(fd.minimum), int(fd.maximum))
                    a(f'          <{fd.name} type="Integer" minimum="{lo}" maximum="{hi}"/>\n')
            a('        </prototype>\n')
            a('        <codecs type="Vector" allowHeterogeneousChildren="1">\n')
            a('        </codecs>\n')
            a('      </points>\n')
            a('    </vectorChild>\n')

        a('  </data3D>\n')
        a('  <images2D type="Vector" allowHeterogeneousChildren="0">\n')
        a('  </images2D>\n')
        a('</e57Root>\n')
        return "".join(L).encode("utf-8")


class _PageWriter:
    """把逻辑字节流按 1020 字节切页，页尾追加 4 字节 CRC32C。

    实现要点：内部缓冲区只用不可变 ``bytes``，且每输出一页就整体替换为新对象。
    CPython 对 ``bytearray += bytes`` 会走 “hybrid buffer” 原地扩容，此时先前的
    切片会指向同一块内存，后续写入会把已输出的页内容覆盖（表现为每 1024 字节
    出现固定偏移的 4 字节脏数据）。用纯切片拼接可彻底规避。
    """

    def __init__(self, fh) -> None:
        self.fh = fh
        self.pending = b""
        self.written = 0          # 已提交的逻辑字节数（不含页尾 CRC）

    def write(self, data) -> None:
        data = bytes(data)
        self.written += len(data)
        self.pending = self.pending + data
        while len(self.pending) >= LOGICAL_PAGE:
            page = self.pending[:LOGICAL_PAGE]
            self.pending = self.pending[LOGICAL_PAGE:]
            self._emit(page)

    def _emit(self, page: bytes) -> None:
        self.fh.write(page)
        self.fh.write(struct.pack(">I", crc32c(page)))

    def finish(self) -> None:
        if self.pending:
            page = self.pending + b"\x00" * (LOGICAL_PAGE - len(self.pending))
            self.pending = b""
            self._emit(page)
        self.fh.flush()


def _placeholders(n: int) -> List[int]:
    return [-1 - i * 7919 for i in range(max(1, n))]


def _unused_pack_pages(stream: bytes) -> bytes:
    """按 992/1020 字节切载荷，逐页补 32 字节页头（含 CRC32C）。

    注意：这是早期按“32 字节页头”理解的实现，仅留作参考，不参与写入。
    """
    out = bytearray()
    total = len(stream)
    pos = 0
    while pos < total:
        chunk = stream[pos:pos + LOGICAL_PAGE]
        llen = len(chunk)
        hdr = bytearray(DATA_SECTION_LEN)
        struct.pack_into("<II", hdr, 4, llen + DATA_SECTION_LEN, llen)
        struct.pack_into(">I", hdr, 0, crc32c(bytes(hdr[4:])))
        out += hdr
        out += chunk
        pos += len(chunk)
    return bytes(out)
