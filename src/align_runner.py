"""把 ``align_crop_pointcloud.py`` 作为外部进程接入软件。

为什么走外部进程而不是把算法搬进来：这份脚本依赖 open3d / plyfile /
matplotlib 等重包，且本身是一套经过验证的、会逐字段自检的 E57 保真导出流程。
软件只负责挑文件、给参数、显示日志和载入结果，不复制也不改写算法。

脚本约定（见其 ``run()``）：
* ``--reference`` 点云1，**必须是 PLY**（作为裁剪范围/参考）
* ``--source``   点云2，**必须是 E57**（待配准与裁剪）
* ``--output``   必须是 E57，且不能与两个输入相同、不能已存在
* 输出旁会生成 ``<stem>_artifacts/``（配准过程图、report.json、掩膜），也不能已存在
* 配准变换写进 E57 的 ``<pose>``，原始点记录/属性逐字段保留，导出后还会重开自检
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, QProcess, Signal

DEFAULT_SCRIPT = r"W:\otherData\LY\CODE\Segmentation\align_crop_pointcloud.py"

# 脚本需要的第三方包（缺任何一个都跑不起来）
REQUIRED_MODULES = ("numpy", "scipy", "pye57", "plyfile", "open3d", "matplotlib")

# 优先挑已经装好 open3d / plyfile 的解释器；本机 b2l 环境是齐全的
_CANDIDATE_PYTHONS = (
    r"D:\PCProjects\01wheat\Self-supervised\b2i_canopynet_v0.1.0\b2l\python.exe",
    sys.executable,
    r"C:\Users\Dell\.workbuddy\binaries\python\envs\default\Scripts\python.exe",
)


def candidate_pythons() -> List[str]:
    """返回去重后真实存在的解释器路径列表。"""
    out: List[str] = []
    for p in _CANDIDATE_PYTHONS:
        if p and os.path.isfile(p) and os.path.normcase(p) not in {os.path.normcase(q) for q in out}:
            out.append(p)
    return out


def probe_python(exe: str) -> Tuple[bool, List[str]]:
    """同步检查解释器能否 import 全部依赖，返回 (是否齐全, 缺失包列表)。"""
    if not exe or not os.path.isfile(exe):
        return False, ["解释器不存在"]
    code = "import " + ", ".join(REQUIRED_MODULES)
    p = QProcess()
    p.setProgram(exe)
    p.setArguments(["-c", code])
    p.setProcessChannelMode(QProcess.MergedChannels)
    p.start()
    if not p.waitForStarted(8000):
        return False, ["解释器无法启动"]
    p.waitForFinished(30000)
    out = bytes(p.readAllStandardOutput()).decode("utf-8", "replace")
    if p.exitStatus() == QProcess.NormalExit and p.exitCode() == 0:
        return True, []
    missing = [m for m in REQUIRED_MODULES if m in out]
    return False, missing or [out.strip()[-300:] or "依赖检查失败"]


def find_python() -> Tuple[str, List[str]]:
    """自动挑一个能跑脚本的解释器；返回 (路径, 缺失包列表)。"""
    tried: List[str] = []
    for exe in candidate_pythons():
        ok, missing = probe_python(exe)
        if ok:
            return exe, []
        tried.append(f"{exe}（缺 {', '.join(missing)}）")
    return candidate_pythons()[0] if candidate_pythons() else sys.executable, tried


def unique_output(path: str) -> str:
    """脚本拒绝覆盖已存在的输出与 _artifacts 目录，自动换一个不冲突的名字。"""
    p = Path(path)
    if not (p.exists() or (p.parent / (p.stem + "_artifacts")).exists()):
        return str(p)
    for i in range(2, 1000):
        q = p.with_name(f"{p.stem}_{i}{p.suffix}")
        if not (q.exists() or (q.parent / (q.stem + "_artifacts")).exists()):
            return str(q)
    return str(p.with_name(f"{p.stem}_new{p.suffix}"))


def default_output(source: str) -> str:
    p = Path(source)
    return str(p.with_name(p.stem + "_cropped.e57"))


def build_arguments(script: str, reference: str, source: str, output: str,
                    *, crop_distance: Optional[float] = None,
                    voxel_size: Optional[float] = None,
                    registration: str = "auto",
                    min_overlap: float = 0.10,
                    ransac_iterations: int = 100000,
                    chunk_size: int = 200000,
                    visual_points: int = 50000,
                    workers: int = 4,
                    seed: int = 42,
                    refine_color_pose: bool = False) -> List[str]:
    args = ["-u", script,
            "--reference", reference,
            "--source", source,
            "--output", output,
            "--registration", registration,
            "--min-overlap", f"{min_overlap:g}",
            "--ransac-iterations", str(int(ransac_iterations)),
            "--chunk-size", str(int(chunk_size)),
            "--visual-points", str(int(visual_points)),
            "--workers", str(int(workers)),
            "--seed", str(int(seed))]
    if crop_distance and crop_distance > 0:
        args += ["--crop-distance", f"{crop_distance:g}"]
    if voxel_size and voxel_size > 0:
        args += ["--voxel-size", f"{voxel_size:g}"]
    if refine_color_pose:
        args.append("--refine-color-pose")
    return args


def validate_inputs(reference: str, source: str, output: str,
                    script: str) -> Optional[str]:
    """返回错误说明，没问题返回 None。"""
    if not os.path.isfile(script):
        return f"配准脚本不存在：\n{script}"
    if not os.path.isfile(reference):
        return f"点云1（参考）不存在：\n{reference}"
    if not os.path.isfile(source):
        return f"点云2（待裁剪）不存在：\n{source}"
    if Path(reference).suffix.lower() != ".ply":
        return "点云1 必须是 PLY 文件（脚本用 PLY 作为裁剪范围参考）"
    if Path(source).suffix.lower() != ".e57":
        return "点云2 必须是 E57 文件（脚本以 E57 逐字段保真导出）"
    if Path(output).suffix.lower() != ".e57":
        return "输出文件必须是 .e57"
    if os.path.normcase(os.path.abspath(output)) in (
            os.path.normcase(os.path.abspath(reference)),
            os.path.normcase(os.path.abspath(source))):
        return "输出文件不能与任一输入相同"
    return None


class AlignRunner(QObject):
    """QProcess 封装：异步跑脚本，逐行吐日志，结束报告结果。"""

    logLine = Signal(str)
    started = Signal()
    finished = Signal(bool, str)       # (成功?, 说明/结果摘要)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self.output = ""
        self.artifact_dir = ""

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def run(self, python_exe: str, script: str, reference: str, source: str,
            output: str, **params) -> None:
        if self.is_running():
            self.finished.emit(False, "已有配准任务在运行")
            return
        err = validate_inputs(reference, source, output, script)
        if err:
            self.finished.emit(False, err)
            return
        ok, missing = probe_python(python_exe)
        if not ok:
            self.finished.emit(False, "所选 Python 缺少依赖：" + ", ".join(missing))
            return

        output = unique_output(output)
        self.output = output
        self.artifact_dir = str(Path(output).parent / (Path(output).stem + "_artifacts"))
        args = build_arguments(script, reference, source, output, **params)

        self.proc = QProcess(self)
        self.proc.setProgram(python_exe)
        self.proc.setArguments(args)
        self.proc.setWorkingDirectory(str(Path(script).parent))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_ready)
        self.proc.finished.connect(self._on_finished)
        self.logLine.emit(f"命令：{python_exe} {' '.join(args)}\n")
        self.proc.start()
        if not self.proc.waitForStarted(10000):
            self.logLine.emit("进程启动失败\n")
            self.finished.emit(False, "进程启动失败，请检查 Python 路径")
            self.proc = None
            return
        self.started.emit()

    def stop(self) -> None:
        if self.is_running():
            self.proc.kill()
            self.proc.waitForFinished(3000)
            self.logLine.emit("\n[已手动终止]\n")

    def _on_ready(self) -> None:
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        if data:
            self.logLine.emit(data)

    def _on_finished(self, code: int, status) -> None:
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        if data:
            self.logLine.emit(data)
        normal = status == QProcess.NormalExit and code == 0
        report = self._read_report()
        self.proc.deleteLater()
        self.proc = None
        if normal and os.path.isfile(self.output):
            kept = report.get("kept_points") if report else None
            total = report.get("source_points") if report else None
            extra = ""
            if kept is not None and total:
                extra = f"  保留 {kept:,} / {total:,} 点"
            self.finished.emit(True, f"配准裁剪完成：{self.output}{extra}")
        else:
            tail = ""
            rp = Path(self.artifact_dir) / "report.json"
            if report is None and rp.exists():
                pass
            self.finished.emit(False, f"脚本退出码 {code}，未生成输出。请查看下方日志。{tail}")

    def _read_report(self):
        rp = Path(self.artifact_dir) / "report.json"
        if not rp.exists():
            return None
        try:
            return json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
