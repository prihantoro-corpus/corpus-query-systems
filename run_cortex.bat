@echo off
SETLOCAL EnableDelayedExpansion

echo ===================================================
echo CORTEX Launcher and Bootstrapper (v130626)
echo ===================================================

:: Detect if we are inside the CORTEX directory
if exist "%~dp0ui_streamlit\cortex.py" (
    set "CORTEX_DIR=%~dp0"
    goto :dir_found
)
if exist "%~dp0cortex\ui_streamlit\cortex.py" (
    set "CORTEX_DIR=%~dp0cortex"
    goto :dir_found
)

echo CORTEX repository not found in current folder.
echo Attempting to download CORTEX from GitHub...

where git >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Cloning repository via Git...
    git clone https://github.com/prihantoro-corpus/cortex.git
    if exist "cortex\ui_streamlit\cortex.py" (
        set "CORTEX_DIR=%cd%\cortex"
        goto :dir_found
    )
    git clone https://github.com/prihantoro-corpus/corpus-query-systems.git cortex
    if exist "cortex\ui_streamlit\cortex.py" (
        set "CORTEX_DIR=%cd%\cortex"
        goto :dir_found
    )
)

echo Git not found. Downloading repository ZIP via PowerShell...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/prihantoro-corpus/cortex/archive/refs/heads/main.zip' -OutFile 'cortex_temp.zip'"
if exist "cortex_temp.zip" (
    echo Extracting CORTEX...
    powershell -Command "Expand-Archive -Path 'cortex_temp.zip' -DestinationPath '.'"
    del cortex_temp.zip
    if exist "cortex-main" (
        rename cortex-main cortex
        set "CORTEX_DIR=%cd%\cortex"
        goto :dir_found
    )
)

powershell -Command "Invoke-WebRequest -Uri 'https://github.com/prihantoro-corpus/corpus-query-systems/archive/refs/heads/main.zip' -OutFile 'cortex_temp.zip'"
if exist "cortex_temp.zip" (
    echo Extracting CORTEX...
    powershell -Command "Expand-Archive -Path 'cortex_temp.zip' -DestinationPath '.'"
    del cortex_temp.zip
    if exist "corpus-query-systems-main" (
        rename corpus-query-systems-main cortex
        set "CORTEX_DIR=%cd%\cortex"
        goto :dir_found
    )
)

echo ERROR: Could not find or download CORTEX repository.
echo Please make sure you are connected to the internet or have Git installed.
pause
exit /b 1

:dir_found
echo CORTEX directory located: !CORTEX_DIR!
cd /d "!CORTEX_DIR!"

:: Create Desktop Shortcut if it doesn't exist
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\CORTEX.lnk"
if not exist "%SHORTCUT_PATH%" (
    echo Creating Desktop Shortcut...
    set "BATCH_PATH=%~f0"
    set "BATCH_DIR=%~dp0"
    powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut($env:SHORTCUT_PATH); $s.TargetPath = $env:BATCH_PATH; $s.WorkingDirectory = $env:BATCH_DIR; $s.Save()"
    if exist "%SHORTCUT_PATH%" (
        echo Shortcut created successfully on Desktop!
    )
)

:: Check if local virtual environment exists
if exist "!CORTEX_DIR!\.venv\Scripts\python.exe" (
    echo Local virtual environment (.venv) found. Using local Python...
    set "PYTHON_CMD=!CORTEX_DIR!\.venv\Scripts\python.exe"
    goto :python_found
)

:: Check Python installation
echo Checking for Python...
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=python"
    goto :python_found
)

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=py"
    goto :python_found
)

echo Python is not installed or not in PATH.
echo Attempting to install Python via winget...
where winget >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo winget is not available. Please install Python 3.11 manually from https://python.org.
    pause
    exit /b 1
)

winget install --id Python.Python.3.11 -e --silent --accept-source-agreements --accept-package-agreements
:: Refresh path check by running python from target location or warning user
echo Python installation initiated. Please restart this script to load the path changes.
pause
exit /b 0

:python_found
for /f "tokens=2" %%v in ('!PYTHON_CMD! --version 2^>^&1') do set "PY_VER=%%v"
echo Python !PY_VER! found.

:: Git update check (only if it's a git repo)
if exist ".git" (
    where git >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [INFO] Offline mode: Skipping updates from GitHub.
        :: git pull
    )
)

:: Install/update dependencies (reuses existing global installations quickly)
echo Checking and installing/updating dependencies...
:: !PYTHON_CMD! -m pip install --upgrade pip
:: !PYTHON_CMD! -m pip install -r requirements.txt

:: Run the streamlit app
echo Launching CORTEX...
cd ui_streamlit
!PYTHON_CMD! -m streamlit run cortex.py

pause
