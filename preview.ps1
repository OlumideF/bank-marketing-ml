# Launch Streamlit dashboard and open browser (Windows)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Checking dependencies..."
python -m pip install -q -r requirements.txt

if (-not (Test-Path "outputs\artifacts.joblib")) {
    Write-Host "No trained models found. Running pipeline (first time, ~1-2 min)..."
    python src/pipeline.py --skip-eda
}

Write-Host ""
Write-Host "Starting dashboard at http://localhost:8501"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

Start-Job -ScriptBlock { Start-Sleep -Seconds 4; Start-Process "http://localhost:8501" } | Out-Null
# Pipe empty line on first run to skip Streamlit email prompt
"" | python -m streamlit run app.py --server.port 8501 --server.address localhost
