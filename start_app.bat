@echo off
echo Starte ESCO Knowledge Graph Anwendung...
echo.

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    streamlit run esco_kg_streamlit.py --server.headless false --server.port 8501 --server.address localhost
) else (
    echo FEHLER: Virtuelle Umgebung nicht gefunden!
    echo Bitte führen Sie zuerst setup.bat aus.
    pause
)
