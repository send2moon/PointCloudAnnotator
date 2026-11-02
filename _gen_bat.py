"""生成 run.bat：强制 CRLF 换行 + 纯 ASCII，避免 cmd 解析错误。

Windows 的 cmd.exe 以 ANSI(GBK) 读取 .bat，且要求 CRLF 行尾。
UTF-8 中文注释会被解码成乱码行而被当成命令执行，LF 行尾会让
if/goto 块解析异常 —— 这两个坑都会导致"启动失败"。
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# 项目运行环境：b2l conda 环境
B2L = "D:" + chr(92) + chr(92).join(
    ["PCProjects", "01wheat", "Self-supervised",
     "b2i_canopynet_v0.1.0", "b2l", "python.exe"]
)

LINES = [
    "@echo off",
    "setlocal EnableExtensions",
    "title CropPointCloudAnnotator",
    "rem Chinese app name is set on this console by src/app.py (SetConsoleTitleW).",
    "",
    'set "HERE=%~dp0"',
    'if "%HERE:~-1%"=="' + chr(92) + '" set "HERE=%HERE:~0,-1%"',
    "",
    "rem LAUNCH holds the full command with quotes, so paths with spaces work.",
    "",
    "rem ---- 1) conda env 'b2l' (the project runtime) ----",
    'if exist "' + B2L + '" goto use_b2l',
    "",
    "rem ---- 2) bundled .venv ----",
    'if exist "%HERE%' + chr(92) + '.venv' + chr(92) + 'Scripts' + chr(92) + 'python.exe" goto use_venv',
    "",
    "rem ---- 3) py launcher / python on PATH ----",
    "where py >nul 2>nul",
    "if not errorlevel 1 goto use_py",
    "where python >nul 2>nul",
    "if not errorlevel 1 goto use_python",
    "",
    "echo [ERROR] No Python interpreter found.",
    "echo         Expected the 'b2l' conda env at:",
    "echo             " + B2L,
    "echo.",
    "pause",
    "exit /b 1",
    "",
    ":use_b2l",
    'set LAUNCH="' + B2L + '"',
    "goto check",
    "",
    ":use_venv",
    'set LAUNCH="%HERE%' + chr(92) + '.venv' + chr(92) + 'Scripts' + chr(92) + 'python.exe"',
    "goto check",
    "",
    ":use_py",
    "set LAUNCH=py -3",
    "goto check",
    "",
    ":use_python",
    "set LAUNCH=python",
    "goto check",
    "",
    ":check",
    'if not exist "%HERE%' + chr(92) + 'src' + chr(92) + 'app.py" goto noapp',
    "",
    '%LAUNCH% -c "import PySide6" >nul 2>nul',
    "if errorlevel 1 goto nodeps",
    "",
    'cd /d "%HERE%"',
    '%LAUNCH% "%HERE%' + chr(92) + 'src' + chr(92) + 'app.py" %*',
    "if errorlevel 1 goto runfail",
    "endlocal",
    "exit /b 0",
    "",
    ":nodeps",
    "echo [ERROR] PySide6 / pye57 are not installed for:",
    "%LAUNCH% -c \"import sys; print('        ' + sys.executable)\"",
    "echo.",
    "echo         Fix it by running:",
    "%LAUNCH% -m pip install -r \"%HERE%" + chr(92) + 'requirements.txt"',
    "echo.",
    "pause",
    "exit /b 1",
    "",
    ":noapp",
    "echo [ERROR] src" + chr(92) + "app.py not found under:",
    "echo         %HERE%",
    "pause",
    "exit /b 1",
    "",
    ":runfail",
    "echo.",
    "echo [ERROR] The program exited with an error, see the message above.",
    "pause",
    "exit /b 1",
]


def main() -> None:
    text = "\r\n".join(LINES) + "\r\n"
    bad = [c for c in text if ord(c) > 127]
    if bad:
        raise SystemExit("non-ASCII characters found: %r" % bad)
    path = os.path.join(HERE, "run.bat")
    with open(path, "wb") as fh:
        fh.write(text.encode("ascii"))
    print("wrote", path)
    print("  bytes      :", len(text))
    print("  CRLF lines :", text.count("\r\n"))
    print("  b2l path   :", B2L)


if __name__ == "__main__":
    main()
