param(
    [switch]$SkipUnitTests
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

# Prefer the repository virtual environment for local runs. GitHub Actions
# uses the Python selected by actions/setup-python and has no local .venv.
$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

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
Invoke-QualityCommand @("-m", "evals.run_retrieval")
Invoke-QualityCommand @("-m", "evals.run_responses")
