# Cascadia launcher (PowerShell) — usage:  .\run.ps1 [map|train|validate|serve|all]
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[setup] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    Write-Host "[setup] Installing dependencies (one time, ~1-2 min)..." -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    & ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
}

$env:PYTHONIOENCODING = "utf-8"
& ".venv\Scripts\python.exe" run.py $args
