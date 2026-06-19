$root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$log    = Join-Path $root "logs\dashboard.log"
$sortai = Join-Path $root ".venv\Scripts\sortai.exe"

if (-not (Test-Path (Join-Path $root "logs"))) {
    New-Item -ItemType Directory -Path (Join-Path $root "logs") | Out-Null
}

# Retry: mapped drives (e.g. Z:\) may not be ready immediately at logon.
# Exit code 0 = clean shutdown (Task Scheduler End Task) -> stop.
# Exit code != 0 = crash (bad config, missing drive, etc.) -> wait and retry.
$maxRetries = 5
for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
    & $sortai dashboard --watch --no-browser >> $log 2>&1
    if ($LASTEXITCODE -eq 0) { break }
    if ($attempt -lt $maxRetries) {
        Add-Content $log "[autostart] attempt $attempt failed (exit $LASTEXITCODE), retrying in 30s..."
        Start-Sleep -Seconds 30
    }
}
