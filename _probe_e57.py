"""E57 结构自检：头部字段、XML 段、各扫描段头与包链是否自洽。

用法: python _probe_e57.py <file.e57> [...]
"""
from __future__ import annotations

import os
import re
import struct
import sys

LOGICAL_PAGE = 1020
PHYS_PAGE = 1024


def to_logical(phys: int) -> int:
    return (phys // PHYS_PAGE) * LOGICAL_PAGE + (phys % PHYS_PAGE)


def to_phys(logical: int) -> int:
    return (logical // LOGICAL_PAGE) * PHYS_PAGE + (logical % LOGICAL_PAGE)


def check(path: str) -> bool:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        blob = fh.read()
    print(f"\n=== {os.path.basename(path)}  {size:,} 字节 ===", flush=True)
    sig = blob[0:8]
    major, minor = struct.unpack_from("<II", blob, 8)
    phys_len, xml_phys, xml_len, page_size = struct.unpack_from("<QQQQ", blob, 16)
    print(f"签名 {sig!r} 版本 {major}.{minor} 头内物理长 {phys_len:,}（实际 {size:,}）"
          f"  页大小 {page_size}", flush=True)

    lg = bytearray()
    for i in range(size // PHYS_PAGE):
        lg += blob[i * PHYS_PAGE: i * PHYS_PAGE + LOGICAL_PAGE]
    xml_logical = to_logical(xml_phys)
    print(f"XML: 物理 {xml_phys:,} → 逻辑 {xml_logical:,} 长度 {xml_len:,}；"
          f"逻辑流长 {len(lg):,}", flush=True)

    ok = True
    if sig != b"ASTM-E57":
        print("  !! 魔数错误"); ok = False
    if phys_len != size:
        print("  !! 头内物理长度 != 实际大小"); ok = False
    if page_size != PHYS_PAGE:
        print("  !! 页大小错误"); ok = False
    if xml_logical + xml_len > len(lg):
        print(f"  !! XML 段越过文件末尾（越界 {xml_logical + xml_len - len(lg):+,} 逻辑字节）")
        ok = False

    text = bytes(lg[xml_logical: xml_logical + xml_len]).split(b"</e57Root>")[0]
    text = text.decode("utf-8", "replace")
    if "<e57Root" not in text:
        print("  !! XML 段里找不到 e57Root"); ok = False
    print(f"XML 含 {text.count('<vectorChild')} 个扫描节点", flush=True)

    hits = list(re.finditer(
        r'<points type="CompressedVector" fileOffset="(\d+)" recordCount="(\d+)"', text))
    protos = re.findall(r"<prototype[^>]*>(.*?)</prototype>", text, re.S)
    for k, m in enumerate(hits):
        fo, rc = int(m.group(1)), int(m.group(2))
        off = to_logical(fo)
        ver = lg[off]
        sec_len, pkt_phys = struct.unpack_from("<QQ", lg, off + 8)
        sec_end = to_phys(off + sec_len)
        # 原型声明的每记录字节数（Integer 宽度按 min/max 推断）
        rec = 0
        for fm in re.finditer(
                r'<([A-Za-z][\w]*)\s+type="(Float|Integer|ScaledInteger)"([^>]*)/?>',
                protos[k] if k < len(protos) else ""):
            kind, attrs = fm.group(2), fm.group(3)
            if kind == "Float":
                rec += 8 if 'precision="double"' in attrs else 4
            elif kind == "Integer":
                lo = float(re.search(r'minimum="([-\d.eE+]+)"', attrs).group(1))
                hi = float(re.search(r'maximum="([-\d.eE+]+)"', attrs).group(1))
                if lo >= 0:
                    rec += 1 if hi <= 255 else (2 if hi <= 65535 else 4)
                else:
                    rec += 1 if (lo >= -128 and hi <= 127) else (
                        2 if (lo >= -32768 and hi <= 32767) else 4)
            else:
                rec += 4
        print(f"  [{k}] 逻辑 {off:,} ver={ver} 段长 {sec_len:,} 记录 {rc:,} "
              f"原型每记录 {rec} B", flush=True)
        if ver != 1:
            print("    !! 段头版本不是 1"); ok = False
        if off + sec_len > len(lg):
            print(f"    !! 段越过逻辑流末尾（越界 {off + sec_len - len(lg):+,}）"); ok = False
        if sec_end > size:
            print(f"    !! 段末越过物理文件末尾（越界 {sec_end - size:+,}）"); ok = False
        # 包链
        pos, pkts, records = off + 32, 0, 0
        while pos < off + sec_len:
            hdr = lg[pos:pos + 6]
            if len(hdr) < 6:
                print("    !! 包头越界"); ok = False
                break
            flags = hdr[0]
            plen = struct.unpack_from("<H", hdr, 2)[0] + 1
            nf = struct.unpack_from("<H", hdr, 4)[0]
            if flags != 1:
                print(f"    !! 第 {pkts} 包 flags={flags}"); ok = False
            if pos + plen > off + sec_len:
                print(f"    !! 第 {pkts} 包越出段末"); ok = False
                break
            w = struct.unpack_from("<" + "H" * nf, lg, pos + 6)
            pos += plen
            pkts += 1
            if pkts > 200000:
                print("    !! 包数异常"); ok = False
                break
        # 用原型每记录字节推算记录数是否与 recordCount 吻合
        print(f"    {pkts} 个包，链尾 {pos:,}（段末 {off + sec_len:,}，差 {pos - off - sec_len:+,}）"
              f"；按原型字节数推算记录 {(pos - off - 32) // rec if rec else 0:,}"
              f"（声明 {rc:,}）", flush=True)
        if pos != off + sec_len:
            print("    !! 包链总长与段头声明段长不一致"); ok = False

    print("结果:", "OK" if ok else "有问题", flush=True)
    return ok


if __name__ == "__main__":
    allok = True
    for p in sys.argv[1:]:
        allok = check(p) and allok
    sys.exit(0 if allok else 1)
