$ErrorActionPreference = "Stop"

$python = "C:\Users\harrison\AppData\Local\Programs\Python\Python314\python.exe"

if (-not (Test-Path $python)) {
    $python = "python"
}

& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -r requirements-ta.txt
& .\.venv\Scripts\python.exe -m pip install -e .

Write-Host ""
Write-Host "setup finished"
Write-Host "use .\.venv\Scripts\python.exe to run the examples"
