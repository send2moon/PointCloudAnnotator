"""标注数据模型：按扫描 + 点索引记录器官标签。

标注以 numpy 数组形式驻留内存（每个扫描一组），通过 at_exit 或显式保存写回 e57。
标签体系沿用农学惯例：穗（spike）/ 叶（leaf）/ 茎（stem）/ 其他。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# 标签编码：0 = 未标注
LABEL_NAMES = {
    0: "未标注",
    1: "穗",
    2: "叶",
    3: "茎",
    4: "其他",
}

# 界面配色（RGB，深色背景下的高对比度）
LABEL_COLORS = {
    0: (150, 150, 150),
    1: (255, 96, 96),      # 穗 - 红
    2: (96, 220, 120),     # 叶 - 绿
    3: (110, 170, 255),    # 茎 - 蓝
    4: (255, 205, 90),     # 其他 - 黄
}


@dataclass
class AnnotationSet:
    """一个文件全部扫描的标注。

    labels[i] 与 scans[i] 的点顺序一一对应；instances[i] 记录器官实例编号
    （同一次框选/笔刷落笔的连续点共用一个编号，便于统计穗数）。
    """

    n_scans: int = 0
    labels: List[np.ndarray] = field(default_factory=list)
    instances: List[np.ndarray] = field(default_factory=list)
    next_instance: int = 1
    # 已提交的实例：instance_id -> (label, scan_index)
    instance_info: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def empty(cls, counts: List[int]) -> "AnnotationSet":
        a = cls(n_scans=len(counts))
        for n in counts:
            a.labels.append(np.zeros(n, dtype=np.uint8))
            a.instances.append(np.zeros(n, dtype=np.uint16))
        return a

    @classmethod
    def from_scans(cls, scans) -> "AnnotationSet":
        """从已有 e57 扫描恢复标注（读取 annotationLabel/annotationInstance 字段）。"""
        a = cls.empty([s.count for s in scans])
        for i, s in enumerate(scans):
            lab = s.points.get("annotationLabel")
            ins = s.points.get("annotationInstance")
            if lab is not None and len(lab) == a.labels[i].shape[0]:
                a.labels[i] = lab.astype(np.uint8)
            if ins is not None and len(ins) == a.instances[i].shape[0]:
                a.instances[i] = ins.astype(np.uint16)
        # 重建实例编号
        mx = 0
        for ins in a.instances:
            if ins.size:
                mx = max(mx, int(ins.max()))
        a.next_instance = mx + 1
        for i, (lab, ins) in enumerate(zip(a.labels, a.instances)):
            for uid in np.unique(ins):
                if uid == 0:
                    continue
                m = ins == uid
                # 取众数作为该实例标签
                vals, cnts = np.unique(lab[m], return_counts=True)
                lb = int(vals[cnts.argmax()]) if len(vals) else 0
                a.instance_info[int(uid)] = (lb, i)
        return a

    # ------------------------------------------------------------ 统计 --

    def label_counts(self) -> Dict[int, int]:
        out = {k: 0 for k in LABEL_NAMES}
        for lab in self.labels:
            if lab.size == 0:
                continue
            vals, cnts = np.unique(lab, return_counts=True)
            for v, c in zip(vals, cnts):
                out[int(v)] = out.get(int(v), 0) + int(c)
        return out

    def instance_counts(self) -> Dict[int, int]:
        """每种标签下的实例（器官）数量。"""
        out: Dict[int, int] = {k: 0 for k in LABEL_NAMES}
        for lab, ins in zip(self.labels, self.instances):
            for uid in np.unique(ins):
                if uid == 0:
                    continue
                m = ins == uid
                vals, cnts = np.unique(lab[m], return_counts=True)
                if len(vals) == 0:
                    continue
                lb = int(vals[cnts.argmax()])
                out[lb] = out.get(lb, 0) + 1
        return out

    def instance_ids(self, scan_index: int) -> List[int]:
        ins = self.instances[scan_index]
        if ins.size == 0:
            return []
        return [int(v) for v in np.unique(ins) if v != 0]

    def instance_size(self, scan_index: int, uid: int) -> int:
        return int((self.instances[scan_index] == uid).sum())

    def instance_label(self, scan_index: int, uid: int) -> int:
        info = self.instance_info.get(uid)
        if info:
            return info[0]
        lab = self.labels[scan_index]
        ins = self.instances[scan_index]
        m = ins == uid
        if not m.any():
            return 0
        vals, cnts = np.unique(lab[m], return_counts=True)
        return int(vals[cnts.argmax()])

    def total_marked(self) -> int:
        return sum(int((lab > 0).sum()) for lab in self.labels)

    # ------------------------------------------------------------ 编辑 --

    def new_instance(self, scan_index: int, label: int) -> int:
        uid = self.next_instance
        self.next_instance += 1
        self.instance_info[uid] = (label, scan_index)
        return uid

    def apply(self, scan_index: int, mask: np.ndarray, label: int,
              instance: Optional[int] = None, max_points: int = 0) -> int:
        """把 mask 命中的点标成 label/instance，返回实际标注点数。

        ``instance=0`` 表示**语义标注**：只写类别、不带实例编号；
        默认 ``None`` 表示自动分配一个新实例（实例标注）。
        """
        if instance is None:
            instance = self.new_instance(scan_index, label) if label else 0
        lab = self.labels[scan_index]
        ins = self.instances[scan_index]
        n = int(mask.sum())
        if max_points and n > max_points:
            return 0
        lab[mask] = label
        ins[mask] = instance
        if label and instance:
            self.instance_info[instance] = (label, scan_index)
        return n

    def prune_instance_info(self, scan_index: int) -> int:
        """清掉本站已没有点使用的实例登记，返回清理条数。

        只处理登记在本站（``instance_info[uid][1] == scan_index``）的条目，
        避免把别站的同名登记一起抹掉。
        """
        ins = self.instances[scan_index]
        alive = {int(v) for v in np.unique(ins) if v != 0} if ins.size else set()
        dead = [k for k, v in self.instance_info.items()
                if v[1] == scan_index and k not in alive]
        for k in dead:
            del self.instance_info[k]
        return len(dead)

    def ensure_next_instance(self) -> None:
        """保证 ``next_instance`` 大于所有在用编号。

        清空全部 + 撤销后 ``next_instance`` 可能已被重置为 1，而数组里还留着
        旧编号；不修就会在下次发号时撞号。
        """
        mx = 0
        for ins in self.instances:
            if ins.size:
                mx = max(mx, int(ins.max()))
        self.next_instance = max(int(self.next_instance), mx + 1)

    def label_groups(self, scan_index: int) -> List[Tuple[int, int]]:
        """本站「有类别但没实例编号」的点，按类别汇总为 [(label, 点数)]。

        语义标注产生的点会落在这里；用于列表里的「未分实例」一行。
        """
        lab = self.labels[scan_index]
        ins = self.instances[scan_index]
        if lab.size == 0:
            return []
        m = (lab > 0) & (ins == 0)
        if not m.any():
            return []
        vals, cnts = np.unique(lab[m], return_counts=True)
        return [(int(v), int(c)) for v, c in zip(vals, cnts) if v > 0]

    def clear_scan(self, scan_index: int) -> None:
        self.labels[scan_index][:] = 0
        self.instances[scan_index][:] = 0

    def clear_all(self) -> None:
        for lab, ins in zip(self.labels, self.instances):
            lab[:] = 0
            ins[:] = 0
        self.instance_info.clear()
        self.next_instance = 1

    def erase_scan(self, scan_index: int) -> None:
        self.clear_scan(scan_index)

    def rebuild(self, masks: List[Optional[np.ndarray]], counts: List[int]) -> None:
        """裁剪后按掩膜重建标注（``counts`` 为裁剪后各扫描点数）。

        ``masks[i] is None`` 表示该扫描被整体裁掉。
        """
        if len(masks) != len(counts):
            raise ValueError(
                f"重建标注参数错位：掩膜 {len(masks)} 个，扫描 {len(counts)} 个")
        new_labels, new_instances = [], []
        for i, n in enumerate(counts):
            m = masks[i] if i < len(masks) else None
            if m is None or len(m) != len(self.labels[i]) or not m.any():
                new_labels.append(np.zeros(n, dtype=np.uint8))
                new_instances.append(np.zeros(n, dtype=np.uint16))
                continue
            new_labels.append(self.labels[i][m].astype(np.uint8))
            new_instances.append(self.instances[i][m].astype(np.uint16))
        self.labels = new_labels
        self.instances = new_instances
        self.n_scans = len(counts)
        # 清理已不存在的实例编号
        alive = set()
        for ins in self.instances:
            if ins.size:
                alive.update(int(v) for v in np.unique(ins) if v != 0)
        self.instance_info = {k: v for k, v in self.instance_info.items() if k in alive}
        self.next_instance = (max(alive) + 1) if alive else 1

    def undoable_snapshot(self, scan_index: int):
        return self.labels[scan_index].copy(), self.instances[scan_index].copy()

    def restore(self, scan_index: int, snap) -> None:
        lab, ins = snap
        self.labels[scan_index][:] = lab
        self.instances[scan_index][:] = ins
