# Powershell helper to run process_raw_data.py inside the venv
$venv = Join-Path -Path (Split-Path -Parent $MyInvocation.MyCommand.Path) -ChildPath '..\.venv\Scripts\Activate.ps1'
if (Test-Path $venv) {
    Write-Host "Activating virtual environment..."
    & $venv
}
Write-Host "Running OCR processing for raw data..."
python .\scripts\process_raw_data.py
