@echo off
echo ========================================
echo ESCO Knowledge Graph - Setup
echo ========================================
echo.

echo Schritt 1: Erstelle virtuelle Umgebung...
if exist .venv (
    echo Virtuelle Umgebung existiert bereits.
) else (
    python -m venv .venv
    echo Virtuelle Umgebung erstellt.
)

echo.
echo Schritt 2: Aktiviere virtuelle Umgebung...
call .venv\Scripts\activate.bat

echo.
echo Schritt 3: Upgrade pip...
python -m pip install --upgrade pip

echo.
echo Schritt 4: Installiere Abhängigkeiten aus requirements.txt...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup abgeschlossen!
echo ========================================
echo.
echo Um die Anwendung zu starten, führen Sie aus:
echo   streamlit run esco_kg_streamlit.py
echo.
echo Oder verwenden Sie das Start-Skript:
echo   start_app.bat
echo.
pause

