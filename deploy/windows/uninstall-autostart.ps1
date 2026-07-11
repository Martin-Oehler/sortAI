# Remove the sortAI tray-app autostart shortcut from the user's Startup folder.
#
# Usage:  powershell -ExecutionPolicy Bypass -File deploy\windows\uninstall-autostart.ps1

$ErrorActionPreference = "Stop"

$LnkPath = Join-Path ([Environment]::GetFolderPath('Startup')) "sortAI Dashboard.lnk"

if (Test-Path $LnkPath) {
    Remove-Item $LnkPath
    Write-Host "Removed startup shortcut: $LnkPath"
} else {
    Write-Host "No startup shortcut found at: $LnkPath"
}

Write-Host "Note: a running sortAI instance is not stopped - quit it via the tray menu."
