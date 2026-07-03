@echo off
rem ------------------------------------------------------------------
rem  ANIME 4K UPSCALER - Windows launcher
rem  Drag & drop a video file onto this .bat, or double-click it and
rem  paste the clip's path when asked.
rem ------------------------------------------------------------------
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install it from https://www.python.org/downloads/
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
