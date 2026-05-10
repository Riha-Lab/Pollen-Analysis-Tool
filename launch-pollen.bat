@echo off
:: =============================================================================
::  Pollen Analysis Tool — Quick Launcher  (Windows)
::  Double-click this file to start the tool.
::  Run setup-conda.bat first if you haven't already.
:: =============================================================================

setlocal
set "ENV_NAME=pollen-analysis"
set "SCRIPT_DIR=%~dp0"

:: Find conda
set "CONDA_EXE="
where conda >nul 2>&1
if not errorlevel 1 set "CONDA_EXE=conda" & goto :launch

for %%P in (
    "%USERPROFILE%\miniforge3\Scripts\conda.exe"
    "%USERPROFILE%\mambaforge\Scripts\conda.exe"
    "%USERPROFILE%\Anaconda3\Scripts\conda.exe"
    "%USERPROFILE%\miniconda3\Scripts\conda.exe"
    "C:\ProgramData\miniforge3\Scripts\conda.exe"
) do (
    if exist %%P set "CONDA_EXE=%%~P" & goto :launch
)

echo [ERROR] conda not found. Please run setup-conda.bat first.
pause
exit /b 1

:launch
:: Activate base then the pollen env, then launch the app
if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniforge3\Scripts\activate.bat" >nul 2>&1
)

%CONDA_EXE% run -n "%ENV_NAME%" python "%SCRIPT_DIR%pollen_analysis_app.py"

if errorlevel 1 (
    echo.
    echo [ERROR] The app exited with an error.
    echo         If the environment is missing, run setup-conda.bat first.
    pause
)
