@echo off
:: Wrapper script for running sortai as a background service.
:: Called by the Task Scheduler task or NSSM service.
:: Stdout and stderr are appended to logs\dashboard.log.

set SORTAI_ROOT=%~dp0..\..
cd /d "%SORTAI_ROOT%"

:: Ensure the logs directory exists
if not exist "logs" mkdir logs

:: Activate the virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

sortai dashboard --watch --no-browser >> logs\dashboard.log 2>&1
