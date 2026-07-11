# Install sortAI tray-app autostart: creates a shortcut in the user's Startup
# folder pointing at the sortai-tray gui-script (no console window).
#
# Usage:  powershell -ExecutionPolicy Bypass -File deploy\windows\install-autostart.ps1 [-StartNow]

param(
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"

# deploy/windows/ -> repo root, no placeholders to edit.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TrayExe = Join-Path $RepoRoot ".venv\Scripts\sortai-tray.exe"

if (-not (Test-Path $TrayExe)) {
    Write-Error @"
sortai-tray.exe not found at:
  $TrayExe

Install the tray extras into the project venv first:
  cd $RepoRoot
  .venv\Scripts\activate
  pip install -e ".[tray]"
"@
    exit 1
}

$StartupDir = [Environment]::GetFolderPath('Startup')
$LnkPath = Join-Path $StartupDir "sortAI Dashboard.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($LnkPath)
$Shortcut.TargetPath = $TrayExe
# Load-bearing: relative config/config.toml and logs/ resolve from here.
$Shortcut.WorkingDirectory = $RepoRoot
$Shortcut.Description = "sortAI Dashboard (system tray)"
$Shortcut.Save()

Write-Host "Created startup shortcut: $LnkPath"
Write-Host "  Target:   $TrayExe"
Write-Host "  Start in: $RepoRoot"

if ($StartNow) {
    Start-Process -FilePath $TrayExe -WorkingDirectory $RepoRoot
    Write-Host "Started sortai-tray - look for the sortAI icon in the system tray."
}
