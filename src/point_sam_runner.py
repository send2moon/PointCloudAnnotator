"""Nonblocking isolated Point-SAM process; one request at a time."""
import json
from pathlib import Path
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

DEFAULT_REPO = r'D:\PCProjects\01wheat\SemanticSegmentationofWheatOrgans\Point-SAM-main'
DEFAULT_PYTHON = r'D:\PCProjects\01wheat\Self-supervised\b2i_canopynet_v0.1.0\b2l\python.exe'
DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent.parent / 'models/point_sam/model.safetensors')


class PointSAMRunner(QObject):
    finished = Signal(bool, object)
    busyChanged = Signal(bool)
    logLine = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None
        self.busy = False
        self.sequence = 0
        self.buffer = b''
        self.executable = ''
        self.source_root = ''
        self.pending = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300000)
        self.timer.timeout.connect(lambda: self.cancel('推理超时，请缩小区域或降低采样点数。'))

    def submit(self, executable, request):
        if self.busy:
            return False
        if not Path(executable).is_file():
            self.finished.emit(False, 'Python 解释器不存在：' + executable)
            return False
        if self.proc is not None and (self.executable != executable or self.source_root != request.get('repo', '')):
            self.shutdown()
        self.sequence += 1
        request = dict(request, id=self.sequence)
        self.pending = json.dumps(request, ensure_ascii=False).encode('utf-8') + b'\n'
        self.busy = True
        self.busyChanged.emit(True)
        self.timer.start()
        if self.proc is None:
            self.proc = QProcess(self)
            self.executable = executable
            self.source_root = request.get('repo', '')
            env = QProcessEnvironment.systemEnvironment()
            env.insert('PYTHONIOENCODING', 'utf-8')
            env.insert('PYTHONDONTWRITEBYTECODE', '1')
            self.proc.setProcessEnvironment(env)
            self.proc.setProgram(executable)
            self.proc.setArguments(['-u', '-B', str(Path(__file__).with_name('point_sam_worker.py'))])
            self.proc.readyReadStandardOutput.connect(self._read)
            self.proc.readyReadStandardError.connect(self._read_log)
            self.proc.started.connect(self._send)
            self.proc.errorOccurred.connect(self._error)
            self.proc.finished.connect(self._exited)
            self.proc.start()
        elif self.proc.state() == QProcess.Running:
            self._send()
        return True

    def _send(self):
        if self.pending is not None and self.proc is not None:
            self.proc.write(self.pending)
            self.pending = None

    def _read_log(self):
        if self.proc:
            self.logLine.emit(bytes(self.proc.readAllStandardError()).decode('utf-8', 'replace'))

    def _read(self):
        if self.proc is None:
            return
        self.buffer += bytes(self.proc.readAllStandardOutput())
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            try:
                message = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                self.logLine.emit(line.decode('utf-8', 'replace'))
                continue
            if self.busy and message.get('id') == self.sequence:
                self.timer.stop()
                self.busy = False
                self.busyChanged.emit(False)
                self.finished.emit(message.get('ok', False), message.get('result') if message.get('ok') else message.get('error'))

    def _error(self, error):
        if self.busy:
            self.cancel('Point-SAM 工作进程启动或通信失败，请检查 Python 环境。')

    def _exited(self, code, status):
        self._read_log()
        if self.busy:
            self.cancel(f'Point-SAM 进程意外退出（{code}），请检查日志或减少采样点数。')
        else:
            self.shutdown()

    def shutdown(self):
        self.timer.stop()
        proc, self.proc = self.proc, None
        if proc is not None:
            proc.blockSignals(True)
            if proc.state() != QProcess.NotRunning:
                proc.kill()
                proc.waitForFinished(3000)
            proc.deleteLater()
        self.buffer = b''

    def cancel(self, message='已取消 Point-SAM 推理。'):
        active = self.busy
        self.sequence += 1
        self.shutdown()
        self.pending = None
        self.busy = False
        self.busyChanged.emit(False)
        if active:
            self.finished.emit(False, message)
