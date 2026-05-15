@echo off
:: Installs sortai-dashboard as a Windows Service using NSSM.
:: Requires NSSM (https://nssm.cc/) — install via: winget install nssm
:: Run this script as Administrator.

set SERVICE_NAME=sortai-dashboard
set SORTAI_ROOT=%~dp0..\..

:: Resolve absolute path
for %%i in ("%SORTAI_ROOT%") do set SORTAI_ROOT=%%~fi

where nssm >nul 2>&1
if errorlevel 1 (
    echo NSSM not found. Install it first:  winget install nssm
    exit /b 1
)

:: Determine the sortai executable (prefer venv)
if exist "%SORTAI_ROOT%\.venv\Scripts\sortai.exe" (
    set SORTAI_EXE=%SORTAI_ROOT%\.venv\Scripts\sortai.exe
) else (
    where sortai >nul 2>&1
    if errorlevel 1 (
        echo sortai not found. Activate your virtual environment or install sortai first.
        exit /b 1
    )
    for /f "delims=" %%i in ('where sortai') do set SORTAI_EXE=%%i
)

echo Installing service "%SERVICE_NAME%" ...
echo   Executable : %SORTAI_EXE%
echo   Arguments  : dashboard --watch --no-browser
echo   Working dir: %SORTAI_ROOT%
echo.

nssm install "%SERVICE_NAME%" "%SORTAI_EXE%" dashboard --watch --no-browser
nssm set "%SERVICE_NAME%" AppDirectory "%SORTAI_ROOT%"
nssm set "%SERVICE_NAME%" AppStdout "%SORTAI_ROOT%\logs\dashboard.log"
nssm set "%SERVICE_NAME%" AppStderr "%SORTAI_ROOT%\logs\dashboard.log"
nssm set "%SERVICE_NAME%" AppRotateFiles 1
nssm set "%SERVICE_NAME%" AppRotateBytes 10485760
nssm set "%SERVICE_NAME%" Start SERVICE_AUTO_START
nssm set "%SERVICE_NAME%" Description "sortAI dashboard with inbox watcher"

echo.
echo Service installed. Starting it now ...
nssm start "%SERVICE_NAME%"
echo Done. Check logs\dashboard.log for startup output.
