[CmdletBinding()]
param(
    [switch]$RebuildResearch,
    [switch]$SkipYahooTlsVerification,
    [switch]$SkipRates,
    [string]$StartDate = "2018-01-01",
    [string]$EndDate = (Get-Date).AddDays(1).ToString("yyyy-MM-dd"),
    [int]$Months = 24,
    [int]$Workers = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

Push-Location $root
try {
    if (-not (Test-Path $venvPython)) {
        Write-Host "[setup] Creating .venv..."
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -m venv .venv
        }
        elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv .venv
        }
        else {
            throw "Python 3 was not found. Install Python, then rerun setup.ps1."
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Python failed to create .venv (exit code $LASTEXITCODE)."
        }
    }

    Write-Host "[setup] Installing Python requirements..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed (exit code $LASTEXITCODE)."
    }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Requirement installation failed (exit code $LASTEXITCODE)."
    }

    New-Item -ItemType Directory -Force -Path (Join-Path $root "data") | Out-Null
    if (-not $env:QUANTLAB_SECRET_KEY) {
        $env:QUANTLAB_SECRET_KEY = "local-dev"
    }

    Write-Host "[setup] Creating/updating the SQLite schema..."
    & $venvPython manage.py migrate
    if ($LASTEXITCODE -ne 0) {
        throw "Database migration failed (exit code $LASTEXITCODE)."
    }
    & $venvPython manage.py check
    if ($LASTEXITCODE -ne 0) {
        throw "Django system check failed (exit code $LASTEXITCODE)."
    }
    & $venvPython manage.py ensure_local_admin
    if ($LASTEXITCODE -ne 0) {
        throw "Local admin setup failed (exit code $LASTEXITCODE)."
    }

    if ($RebuildResearch) {
        Write-Host "[setup] Rebuilding prices, factor returns, and $Months months of decompositions..."
        $replicationArgs = @(
            "manage.py",
            "replicate_research",
            "--start", $StartDate,
            "--end", $EndDate,
            "--months", "$Months"
        )
        if ($Workers -gt 0) {
            $replicationArgs += @("--workers", "$Workers")
        }
        if ($SkipYahooTlsVerification) {
            $replicationArgs += "--skip-yahoo-tls-verification"
        }
        if ($SkipRates) {
            $replicationArgs += "--skip-rates"
        }
        & $venvPython @replicationArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Research replication failed (exit code $LASTEXITCODE)."
        }
    }

    Write-Host ""
    Write-Host "Setup complete." -ForegroundColor Green
    if (-not $RebuildResearch) {
        Write-Host "The database schema is empty. To reproduce the research database, run:"
        Write-Host "  .\setup.ps1 -RebuildResearch"
        Write-Host "If Yahoo TLS verification fails on this machine, use only on a trusted network:"
        Write-Host "  .\setup.ps1 -RebuildResearch -SkipYahooTlsVerification"
    }
    Write-Host "Start QuantLab with:"
    Write-Host "  .\start_quantlab.ps1"
    Write-Host "Then open http://127.0.0.1:8086"
}
finally {
    Pop-Location
}
