$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Push-Location $root
try {
    & $python manage.py migrate
    $worker = Start-Process -FilePath $python -ArgumentList "manage.py", "quant_worker" -WorkingDirectory $root -PassThru
    Write-Host "QuantLab worker started (PID $($worker.Id))."
    Write-Host "Web: http://127.0.0.1:8086"
    & $python manage.py runserver 127.0.0.1:8086
}
finally {
    if ($worker -and -not $worker.HasExited) { Stop-Process -Id $worker.Id }
    Pop-Location
}
