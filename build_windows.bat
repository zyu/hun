@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || exit /b 1

set "SCOREBOARD_BUILD_PY=.venv-windows-build\Scripts\python.exe"
if exist "%SCOREBOARD_BUILD_PY%" goto install

rem Prefer CPython 3.12 when the Windows Python launcher is installed.
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys, tkinter; assert sys.maxsize > 2**32" >nul 2>&1
    if not errorlevel 1 (
        py -3.12 -m venv .venv-windows-build
        if errorlevel 1 goto failed
        goto install
    )
    py -3 -c "import sys, tkinter; assert sys.version_info >= (3, 11) and sys.maxsize > 2**32" >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv-windows-build
        if errorlevel 1 goto failed
        goto install
    )
)

python -c "import sys, tkinter; assert sys.version_info >= (3, 11) and sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 goto no_python
python -m venv .venv-windows-build
if errorlevel 1 goto failed

:install
echo Installing Windows build dependencies...
"%SCOREBOARD_BUILD_PY%" -m pip install -r requirements.txt -r requirements-yolo26.txt "pyinstaller>=6.22,<7" "pyinstaller-hooks-contrib>=2026.9"
if errorlevel 1 goto failed

echo Building ScoreboardDataset.exe...
"%SCOREBOARD_BUILD_PY%" -m PyInstaller --noconfirm --distpath dist --workpath build\windows ScoreboardDataset.spec
if errorlevel 1 goto failed
if not exist "dist\ScoreboardDataset\ScoreboardDataset.exe" goto failed

echo.
echo Build complete: "%CD%\dist\ScoreboardDataset\ScoreboardDataset.exe"
echo Copy the entire dist\ScoreboardDataset folder to run it on another PC.
popd
if /i not "%~1"=="--no-pause" pause
exit /b 0

:no_python
echo 64-bit Python 3.11 or later with Tkinter is required. Python 3.12 is recommended.
goto end_failure

:failed
echo Build failed. Check the command output above.

:end_failure
popd
if /i not "%~1"=="--no-pause" pause
exit /b 1
