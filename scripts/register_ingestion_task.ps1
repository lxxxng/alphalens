param(
    [string]$TaskName = "AlphaLens Daily Ingestion",

    [ValidatePattern("^([01]\d|2[0-3]):[0-5]\d$")]
    [string]$DailyAt = "06:30",

    [ValidateSet("watchlists", "all")]
    [string]$Scope = "watchlists",

    [ValidateRange(0, 50)]
    [int]$MaxAutoBriefs = 5,

    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runner = Join-Path $repoRoot "scripts\run_scheduled_ingestion.ps1"

if (-not $PythonPath) {
    $PythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $runner)) {
    throw "Scheduled ingestion runner was not found: $runner"
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "AlphaLens Python was not found: $PythonPath"
}

$actionArguments = @(
    "-NoProfile"
    "-ExecutionPolicy Bypass"
    "-File `"$runner`""
    "-Scope $Scope"
    "-MaxAutoBriefs $MaxAutoBriefs"
    "-PythonPath `"$PythonPath`""
) -join " "

# Run under the current Windows user so the task can access this workspace,
# its .env file, Docker-exposed PostgreSQL, and the local FAISS indexes.
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $actionArguments `
    -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At ([datetime]::ParseExact($DailyAt, "HH:mm", $null))
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Refresh AlphaLens data and generate quality-gated event briefs." `
    -Force | Out-Null

Write-Output "Registered '$TaskName' to run daily at $DailyAt for scope '$Scope'."
Write-Output "Automatic briefs per run: $MaxAutoBriefs"
Write-Output "Test it with: Start-ScheduledTask -TaskName '$TaskName'"
