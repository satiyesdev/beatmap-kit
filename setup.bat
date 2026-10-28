@echo off
rem ============================================================
rem  beatmap-kit setup - creates venv\ + venv_asr\ and installs deps
rem  usage: setup.bat          (CUDA 12.8 build of torch, needs an NVIDIA GPU)
rem         setup.bat --cpu    (CPU-only build, works everywhere)
rem ============================================================
setlocal
chcp 65001 >nul
set "KIT=%~dp0"
set "CPU=0"
if /i "%~1"=="--cpu" set "CPU=1"

echo.
echo == beatmap-kit setup ==
echo    kit: %KIT%

rem ---- pick a Python 3.10-3.13 interpreter --------------------
rem BEATMAP_PYTHON overrides detection (portable / non-PATH installs):
rem     set BEATMAP_PYTHON=C:\path\to\python.exe
set "PY="
if defined BEATMAP_PYTHON (
    "%BEATMAP_PYTHON%" -c "import sys; assert (3,10)<=sys.version_info<(3,14)" >nul 2>nul && set PY="%BEATMAP_PYTHON%"
)
if not defined PY (
    for %%V in (3.12 3.13 3.11 3.10) do (
        if not defined PY (
            py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
        )
    )
)
if not defined PY (
    python -c "import sys; assert (3,10)<=sys.version_info<(3,14)" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo.
    echo [ERROR] Python 3.10-3.13 not found. Install Python 3.12 from python.org
    echo         and make sure the "py" launcher is on PATH, or set BEATMAP_PYTHON
    echo         to the full path of python.exe.
    goto :fail
)
echo    python: %PY%
%PY% -c "import sys; print('    version:', sys.version.split()[0])"

rem ---- main environment (analysis + Demucs) -------------------
if not exist "%KIT%venv\Scripts\python.exe" (
    echo.
    echo == creating venv ...
    %PY% -m venv "%KIT%venv" || goto :fail
)
set "PYM=%KIT%venv\Scripts\python.exe"
"%PYM%" -m pip install --upgrade pip wheel setuptools || goto :fail

echo.
echo == installing torch (this is the big download) ...
if "%CPU%"=="1" (
    "%PYM%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0 || goto :fail
) else (
    "%PYM%" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchaudio==2.8.0+cu128 || goto :fail
)

echo.
echo == installing main requirements ...
"%PYM%" -m pip install -r "%KIT%requirements.txt" || goto :fail

rem ---- ASR environment (faster-whisper + WhisperX, CPU on purpose) ----
if not exist "%KIT%venv_asr\Scripts\python.exe" (
    echo.
    echo == creating venv_asr ...
    %PY% -m venv "%KIT%venv_asr" || goto :fail
)
set "PYA=%KIT%venv_asr\Scripts\python.exe"
"%PYA%" -m pip install --upgrade pip wheel setuptools || goto :fail

echo.
echo == installing ASR torch (CPU build) ...
"%PYA%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0 || goto :fail

echo.
echo == installing ASR requirements ...
"%PYA%" -m pip install -r "%KIT%requirements-asr.txt" || goto :fail

rem whisperx pulls pyannote/torchcodec/triton that alignment does not need;
rem install it without dependencies and rely on requirements-asr.txt instead.
"%PYA%" -m pip install --no-deps whisperx==3.8.6 || goto :fail

echo.
echo == done ==
echo    run:  beatmap.bat "music\song.mp3"
echo    models (~2.7 GB) download automatically on the first run,
echo    or fetch them now with:  venv\Scripts\python.exe scripts\download_models.py
exit /b 0

:fail
echo.
echo [ERROR] setup failed - see the pip output above.
exit /b 1