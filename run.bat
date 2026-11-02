@echo off
setlocal EnableExtensions
title CropPointCloudAnnotator
rem Chinese app name is set on this console by src/app.py (SetConsoleTitleW).

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"

rem LAUNCH holds the full command with quotes, so paths with spaces work.

rem ---- 1) conda env 'b2l' (the project runtime) ----
if exist "D:\PCProjects\01wheat\Self-supervised\b2i_canopynet_v0.1.0\b2l\python.exe" goto use_b2l

rem ---- 2) bundled .venv ----
if exist "%HERE%\.venv\Scripts\python.exe" goto use_venv

rem ---- 3) py launcher / python on PATH ----
where py >nul 2>nul
if not errorlevel 1 goto use_py
where python >nul 2>nul
if not errorlevel 1 goto use_python

echo [ERROR] No Python interpreter found.
echo         Expected the 'b2l' conda env at:
echo             D:\PCProjects\01wheat\Self-supervised\b2i_canopynet_v0.1.0\b2l\python.exe
echo.
pause
exit /b 1

:use_b2l
set LAUNCH="D:\PCProjects\01wheat\Self-supervised\b2i_canopynet_v0.1.0\b2l\python.exe"
goto check

:use_venv
set LAUNCH="%HERE%\.venv\Scripts\python.exe"
goto check

:use_py
set LAUNCH=py -3
goto check

:use_python
set LAUNCH=python
goto check

:check
if not exist "%HERE%\src\app.py" goto noapp

%LAUNCH% -c "import PySide6" >nul 2>nul
if errorlevel 1 goto nodeps

cd /d "%HERE%"
%LAUNCH% "%HERE%\src\app.py" %*
if errorlevel 1 goto runfail
endlocal
exit /b 0

:nodeps
echo [ERROR] PySide6 / pye57 are not installed for:
%LAUNCH% -c "import sys; print('        ' + sys.executable)"
echo.
echo         Fix it by running:
%LAUNCH% -m pip install -r "%HERE%\requirements.txt"
echo.
pause
exit /b 1

:noapp
echo [ERROR] src\app.py not found under:
echo         %HERE%
pause
exit /b 1

:runfail
echo.
echo [ERROR] The program exited with an error, see the message above.
pause
exit /b 1
