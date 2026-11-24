"""Transactional annotation review; original point order and geometry stay intact."""
from __future__ import annotations
import copy
import numpy as np
from annotation import AnnotationSet
import cloudio


class ReviewSession:
    def __init__(self, doc, path='', ann=None, dirty=False):
        self.doc = doc
        self.path = path
        self.ann = ann if ann is not None else AnnotationSet.from_scans(doc.scans)
        self.offsets = np.r_[0, np.cumsum([s.count for s in doc.scans])]
        self.history = []
        self.revision = self.counter = 0
        self.saved_revision = -1 if dirty else 0

    @property
    def dirty(self):
        return self.revision != self.saved_revision

    def flat_labels(self):
        return np.concatenate(self.ann.labels)

    def apply(self, mask, label, action='fill', mode='semantic', target=None):
        mask = np.asarray(mask, bool)
        if mask.shape != (self.doc.total_points,):
            raise ValueError('选区与原始点数不一致，请重新生成预览。')
        if action not in ('fill', 'overwrite', 'erase', 'replace_instance') or label not in (1, 2, 3, 4):
            raise ValueError('无效的修正操作或器官类别。')
        if mode not in ('semantic', 'new', 'existing'):
            raise ValueError('无效的实例方式。')
        if action == 'replace_instance':
            mode = 'existing'
        if mode == 'existing' and action != 'erase':
            if target is None:
                raise ValueError('请先在列表里选择要修正的已有实例。')
            sid, uid = target
            if not 0 <= sid < len(self.doc.scans) or uid not in self.ann.instance_ids(sid):
                raise ValueError('目标实例已失效，请重新选择。')
            if self.ann.instance_label(sid, uid) != label:
                raise ValueError('加入已有实例时，器官类别须与该实例一致。')
            outside = mask.copy()
            outside[self.offsets[sid]:self.offsets[sid + 1]] = False
            if outside.any():
                raise ValueError('一个实例只能属于一个扫描；请限制到目标扫描后重新分割。')
        edits = []
        for sid, lab in enumerate(self.ann.labels):
            selected = mask[self.offsets[sid]:self.offsets[sid + 1]].copy()
            if action == 'fill':
                selected &= lab == 0
            if selected.any() or (action == 'replace_instance' and sid == target[0]):
                edits.append((sid, selected))
        if not edits:
            return 0
        self.ann.ensure_next_instance()
        if mode == 'new' and action != 'erase' and self.ann.next_instance + len(edits) - 1 > 65535:
            raise ValueError('实例编号已达到 65535 上限，请改用语义方式或加入已有实例。')
        snaps = [(sid, self.ann.undoable_snapshot(sid)) for sid, _ in edits]
        record = (self.revision, snaps, self.ann.instance_info.copy(), self.ann.next_instance)
        changed = 0
        for sid, selected in edits:
            lab, ins = self.ann.labels[sid], self.ann.instances[sid]
            if action == 'replace_instance':
                removed = (ins == target[1]) & ~selected
                changed += int(removed.sum())
                lab[removed], ins[removed] = 0, 0
            if action == 'erase':
                changed += int(((lab != 0) | (ins != 0))[selected].sum())
                lab[selected], ins[selected] = 0, 0
            else:
                uid = (self.ann.new_instance(sid, label) if mode == 'new'
                       else target[1] if mode == 'existing' else 0)
                changed += int(((lab != label) | (ins != uid))[selected].sum())
                lab[selected], ins[selected] = label, uid
            self.ann.prune_instance_info(sid)
        if not changed:
            self.ann.instance_info = record[2]
            self.ann.next_instance = record[3]
            return 0
        self.history.append(record)
        self.history = self.history[-40:]
        self.counter += 1
        self.revision = self.counter
        return changed

    def undo(self):
        if not self.history:
            return False
        self.revision, snapshots, info, next_id = self.history.pop()
        for sid, snapshot in snapshots:
            self.ann.restore(sid, snapshot)
        self.ann.instance_info = info
        self.ann.next_instance = max(self.ann.next_instance, next_id)
        self.ann.ensure_next_instance()
        return True

    def save(self, path):
        if cloudio.ext_of(path) == '.e57':
            for sid, scan in enumerate(self.doc.scans):
                for field, values, width in (
                    ('annotationLabel', self.ann.labels[sid], 1),
                    ('annotationInstance', self.ann.instances[sid], 2)):
                    fd = scan.ensure_field(field, byte_size=width)
                    fd.minimum, fd.maximum = 0, 255 if width == 1 else 65535
                    fd.byte_size = width
                    scan.points[field] = values.copy()
            self.doc.save(path)
        else:
            cloudio.save(path, self.doc, self.ann)
        self.path = path
        self.saved_revision = self.revision

    @classmethod
    def copy_from(cls, doc, path, ann, dirty=False):
        return cls(copy.deepcopy(doc), path, copy.deepcopy(ann), dirty)
