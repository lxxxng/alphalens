param(
    [switch]$SkipUnitTests,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

# A self-hosted runner checkout does not contain the large local .venv. CI can
# point ALPHALENS_PYTHON at the tested interpreter in the main workspace.
$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$python = if ($env:ALPHALENS_PYTHON) {
    $env:ALPHALENS_PYTHON
} elseif (Test-Path $venvPython) {
    $venvPython
} else {
    "python"
}

if ($python -ne "python" -and -not (Test-Path -LiteralPath $python)) {
    throw "Configured Python interpreter was not found: $python"
}

function Invoke-QualityCommand {
    param([string[]]$Arguments)

    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if (-not $SkipUnitTests) {
    Invoke-QualityCommand @(
        "-m", "unittest", "discover", "-s", "tests", "-v"
    )
}

# Preflight deliberately runs before retrieval so a missing database, index,
# or secret cannot consume OpenAI requests and then fail halfway through.
Invoke-QualityCommand @("-m", "evals.check_environment")

if ($PreflightOnly) {
    exit 0
}

Invoke-QualityCommand @("-m", "evals.run_retrieval")
Invoke-QualityCommand @("-m", "evals.run_responses")
