param(
    [ValidateSet("watchlists", "all")]
    [string]$Scope = "watchlists",

    [ValidateRange(0, 50)]
    [int]$MaxAutoBriefs = 5,

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
$exitCode = $null

try {
    Write-Output "AlphaLens scheduled ingestion started: $(Get-Date -Format o)"
    Write-Output "Scope: $Scope"
    Write-Output "Automatic brief limit: $MaxAutoBriefs"
    Write-Output "Log: $logPath"

    # Windows PowerShell wraps native stderr as ErrorRecord objects. Python
    # libraries commonly use stderr for progress and warnings, so keep those
    # messages visible and let the process exit code decide success or failure.
    $originalErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"

        & $PythonPath `
            -m pipelines.scheduled_ingestion `
            --scope $Scope `
            --max-auto-briefs $MaxAutoBriefs `
            --trigger-type scheduled 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $logPath

        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $originalErrorActionPreference
    }
} finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "Scheduled ingestion failed with exit code $exitCode. See $logPath"
}
