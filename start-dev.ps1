$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"

Write-Host "Iniciando NumériLab..." -ForegroundColor Cyan
Write-Host "1) Backend FastAPI en http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "2) Frontend React + Vite en http://localhost:5173" -ForegroundColor Green

$backendJob = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendDir'; uv run uvicorn app.main:app --reload --port 8000" -PassThru
$frontendJob = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendDir'; pnpm dev" -PassThru

Write-Host ""
Write-Host "Servidores iniciados (Backend PID=$($backendJob.Id), Frontend PID=$($frontendJob.Id))." -ForegroundColor Yellow
Write-Host "Abre http://localhost:5173 en tu navegador."
