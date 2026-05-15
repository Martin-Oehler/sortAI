param()
# Wrapper called by Task Scheduler.
# PowerShell keeps sortai in the same job object, so Task Scheduler's
# "End Task" reliably kills the full process tree.

$root = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
Set-Location $root

if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}

& sortai dashboard --watch --no-browser *>> logs\dashboard.log
