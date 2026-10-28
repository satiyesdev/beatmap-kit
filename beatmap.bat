@echo off
rem ============================================================
rem  beatmap-kit entry point
rem  usage: beatmap.bat song.mp3
rem         beatmap.bat song.mp3 0:21 0:56 --lyrics song.txt
rem ============================================================
setlocal
chcp 65001 >nul
set "KIT=%~dp0"
set "PYTHONIOENCODING=utf-8"

if not exist "%KIT%venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run setup.bat once first.
    exit /b 1
)

"%KIT%venv\Scripts\python.exe" "%KIT%scripts\shoushu.py" %*
exit /b %ERRORLEVEL%