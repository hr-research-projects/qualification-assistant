# PowerShell-Skript zum Starten der Streamlit-Anwendung
# Verwendung: .\start_app.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ESCO Knowledge Graph - Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Prüfe ob virtuelle Umgebung existiert
if (Test-Path ".\.venv\Scripts\python.exe") {
    Write-Host "Virtuelle Umgebung gefunden." -ForegroundColor Green
    
    # Prüfe ob Streamlit installiert ist
    $streamlitCheck = & ".\\.venv\Scripts\python.exe" -m pip show streamlit 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Streamlit nicht gefunden. Installiere Abhängigkeiten..." -ForegroundColor Yellow
        & ".\\.venv\Scripts\python.exe" -m pip install -r requirements.txt
    }
    
    Write-Host "Starte Anwendung..." -ForegroundColor Green
    Write-Host ""
    
    # Wechsle zum Projektverzeichnis
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $scriptPath
    
    # Starte Streamlit direkt mit Python aus der virtuellen Umgebung
    & ".\\.venv\Scripts\python.exe" -m streamlit run esco_kg_streamlit.py --server.headless false --server.port 8501 --server.address localhost
} else {
    Write-Host "FEHLER: Virtuelle Umgebung nicht gefunden!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Bitte führen Sie zuerst das Setup aus:" -ForegroundColor Yellow
    Write-Host "  .\setup.bat" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Oder manuell:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Drücken Sie Enter zum Beenden"
}

