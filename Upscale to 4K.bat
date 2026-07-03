@echo off
rem ------------------------------------------------------------------
rem  ANIME 4K UPSCALER - Windows launcher
rem  Drag & drop a video file onto this .bat, or double-click it and
rem  paste the clip's path when asked.
rem ------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist "%~dp0upscale.py" (
    echo upscale.py was not found next to this launcher.
    echo If you downloaded a ZIP, extract the whole folder first,
    echo then run this file from inside the extracted folder.
    pause
    exit /b 1
)

rem "where python" is fooled by the Microsoft Store stub, so actually
rem run Python to check it works.
python -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo Python was not found ^(or only the Microsoft Store stub is installed^).
    echo Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)

if "%~1"=="" (
    python upscale.py
) else (
    python upscale.py %*
)
echo.
pause
