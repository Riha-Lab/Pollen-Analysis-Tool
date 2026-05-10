@echo off
:: =============================================================================
::  Pollen Analysis Tool — Conda Setup Script  (Windows)
::  https://github.com/Riha-Lab/Pollen-Analysis-Tool
::
::  Usage:
::    Double-click setup-conda.bat            (auto-detects GPU)
::    set POLLEN_CPU_ONLY=1 && setup-conda.bat  (force CPU-only)
::
::  After setup, launch the tool with:
::    conda activate pollen-analysis
::    python pollen_analysis_app.py
:: =============================================================================

setlocal EnableDelayedExpansion
set "ENV_NAME=pollen-analysis"
set "SCRIPT_DIR=%~dp0"
set "ENV_FILE=%SCRIPT_DIR%environment.yml"
set "TMP_ENV=%TEMP%\pollen_environment_cpu.yml"

echo.
echo  +==================================================+
echo  ^|   ^🌸  Pollen Analysis Tool -- Conda Setup        ^|
echo  +==================================================+
echo.

:: ── 0. Check environment.yml exists ─────────────────────────────────────────
if not exist "%ENV_FILE%" (
    echo [ERROR] environment.yml not found in %SCRIPT_DIR%
    echo         Make sure you cloned the full repository:
    echo           git clone https://github.com/Riha-Lab/Pollen-Analysis-Tool.git
    pause
    exit /b 1
)

:: ── 1. Locate conda ──────────────────────────────────────────────────────────
echo [Step 1] Checking for conda...

set "CONDA_EXE="

:: Check if conda is already on PATH
where conda >nul 2>&1
if not errorlevel 1 (
    set "CONDA_EXE=conda"
    goto :found_conda
)

:: Search common install locations
for %%P in (
    "%USERPROFILE%\miniforge3\Scripts\conda.exe"
    "%USERPROFILE%\mambaforge\Scripts\conda.exe"
    "%USERPROFILE%\Anaconda3\Scripts\conda.exe"
    "%USERPROFILE%\miniconda3\Scripts\conda.exe"
    "C:\ProgramData\miniforge3\Scripts\conda.exe"
    "C:\ProgramData\Anaconda3\Scripts\conda.exe"
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
) do (
    if exist %%P (
        set "CONDA_EXE=%%~P"
        goto :found_conda
    )
)

:: conda not found — download and install Miniforge
echo [WARN] conda not found. Installing Miniforge...
echo        Downloading installer — please wait...

set "MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
set "MINIFORGE_INSTALLER=%TEMP%\Miniforge3-Windows-x86_64.exe"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Invoke-WebRequest -Uri '%MINIFORGE_URL%' -OutFile '%MINIFORGE_INSTALLER%' -UseBasicParsing"

if errorlevel 1 (
    echo [ERROR] Failed to download Miniforge.
    echo         Please install it manually from:
    echo           https://github.com/conda-forge/miniforge/releases/latest
    pause
    exit /b 1
)

echo [INFO]  Running Miniforge installer (silent install)...
"%MINIFORGE_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\miniforge3
if errorlevel 1 (
    echo [ERROR] Miniforge installation failed. Please install manually.
    pause
    exit /b 1
)

del "%MINIFORGE_INSTALLER%" >nul 2>&1
set "CONDA_EXE=%USERPROFILE%\miniforge3\Scripts\conda.exe"
echo [OK]    Miniforge installed at %USERPROFILE%\miniforge3

:: Initialise conda for cmd.exe so future sessions work
call "%USERPROFILE%\miniforge3\Scripts\conda.bat" init cmd.exe >nul 2>&1

:found_conda
echo [OK]    conda found: %CONDA_EXE%

:: Bootstrap conda into this cmd session without requiring a restart
if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniforge3\Scripts\activate.bat" >nul 2>&1
) else if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\Anaconda3\Scripts\activate.bat" >nul 2>&1
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" >nul 2>&1
)

:: ── 2. GPU detection ─────────────────────────────────────────────────────────
echo.
echo [Step 2] Detecting hardware...

set "USE_GPU=false"

if "%POLLEN_CPU_ONLY%"=="1" (
    echo [INFO]  POLLEN_CPU_ONLY=1 set -- installing CPU-only PyTorch.
    goto :gpu_done
)

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [OK]    NVIDIA GPU detected.
    echo [INFO]  Will install PyTorch with CUDA 12.1 support.
    echo [INFO]  conda manages the CUDA runtime -- no system CUDA install needed.
    set "USE_GPU=true"
) else (
    echo [WARN]  No NVIDIA GPU detected -- installing CPU-only PyTorch.
    echo [WARN]  If you have a GPU, ensure the NVIDIA driver is installed,
    echo [WARN]  then re-run this script.
)

:gpu_done

:: ── 3. Prepare environment file ──────────────────────────────────────────────
echo.
echo [Step 3] Preparing environment...

if "%USE_GPU%"=="true" (
    set "ACTIVE_ENV=%ENV_FILE%"
) else (
    echo [INFO]  Patching environment.yml for CPU-only PyTorch...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "(Get-Content '%ENV_FILE%') -replace '- pytorch::pytorch-cuda=12\.1', '- pytorch::cpuonly' | Set-Content '%TMP_ENV%'"
    if errorlevel 1 (
        echo [ERROR] Failed to create CPU environment file.
        pause
        exit /b 1
    )
    set "ACTIVE_ENV=%TMP_ENV%"
)

:: ── 4. Create or update the environment ──────────────────────────────────────
echo.
echo [Step 4] Installing packages...
echo          This downloads ~2-4 GB on first run. Please be patient.
echo.

:: Check if environment already exists
%CONDA_EXE% env list | findstr /B "%ENV_NAME% " >nul 2>&1
if not errorlevel 1 (
    echo [INFO]  Environment '%ENV_NAME%' already exists -- updating...
    %CONDA_EXE% env update -n "%ENV_NAME%" -f "%ACTIVE_ENV%" --prune
) else (
    echo [INFO]  Creating new environment '%ENV_NAME%'...
    %CONDA_EXE% env create -n "%ENV_NAME%" -f "%ACTIVE_ENV%"
)

if errorlevel 1 (
    echo.
    echo [ERROR] Environment creation failed.
    echo         Check the output above for details.
    if exist "%TMP_ENV%" del "%TMP_ENV%" >nul 2>&1
    pause
    exit /b 1
)

if exist "%TMP_ENV%" del "%TMP_ENV%" >nul 2>&1

echo.
echo [OK]    Packages installed.

:: ── 5. Verify the install ─────────────────────────────────────────────────────
echo.
echo [Step 5] Verifying installation...

%CONDA_EXE% run -n "%ENV_NAME%" python -c ^
"import sys, torch; ^
pkgs=['PyQt6','numpy','cv2','matplotlib','cellpose','scipy','statsmodels','pandas','reportlab','PIL','huggingface_hub','requests']; ^
missing=[p for p in pkgs if __import__('importlib').util.find_spec(p) is None]; ^
print(f'  Python:         {sys.version.split()[0]}'); ^
print(f'  PyTorch:        {torch.__version__}'); ^
print(f'  CUDA available: {\"YES\" if torch.cuda.is_available() else \"no (CPU mode)\"}'); ^
print(f'  CUDA version:   {torch.version.cuda or \"N/A\"}'); ^
[print(f'  MISSING: {p}') for p in missing]; ^
sys.exit(1 if missing else 0)"

if errorlevel 1 (
    echo.
    echo [ERROR] Some packages failed to install. See above for details.
    pause
    exit /b 1
)

echo [OK]    All packages verified.

:: ── 6. Done ───────────────────────────────────────────────────────────────────
echo.
echo  +==================================================+
echo  ^|   Setup complete!                                ^|
echo  +--------------------------------------------------+
echo  ^|                                                  ^|
echo  ^|   To launch the Pollen Analysis Tool:            ^|
echo  ^|                                                  ^|
echo  ^|     conda activate pollen-analysis               ^|
echo  ^|     python pollen_analysis_app.py                ^|
echo  ^|                                                  ^|
echo  ^|   Or double-click: launch-pollen.bat             ^|
echo  ^|                                                  ^|
echo  ^|   To update later:                               ^|
echo  ^|     git pull                                     ^|
echo  ^|     setup-conda.bat                              ^|
echo  ^|                                                  ^|
echo  +==================================================+
echo.
pause
