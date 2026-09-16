param(
    [ValidateSet("watchlists", "all")]
    [string]$Scope = "watchlists",

    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $PythonPath) {
    $PythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "AlphaLens Python was not found: $PythonPath"
}

$logDirectory = Join-Path $repoRoot "data\logs\ingestion"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDirectory "ingestion-$timestamp.log"

Push-Location $repoRoot

try {
    Write-Output "AlphaLens scheduled ingestion started: $(Get-Date -Format o)"
    Write-Output "Scope: $Scope"
    Write-Output "Log: $logPath"

    & $PythonPath `
        -m pipelines.scheduled_ingestion `
        --scope $Scope `
        --trigger-type scheduled 2>&1 |
        Tee-Object -FilePath $logPath

    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "Scheduled ingestion failed with exit code $exitCode. See $logPath"
}

