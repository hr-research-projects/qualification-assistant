@echo off
REM Einfaches Start-Skript für Windows CMD
REM Einfach ausführen: run.bat

call .venv\Scripts\activate.bat
streamlit run esco_kg_streamlit.py --server.headless false --server.port 8501 --server.address localhost

