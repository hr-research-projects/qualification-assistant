# ESCO Knowledge Graph - Installation und Start

## Schnelleinrichtung

### Option 1: Automatisches Setup (Empfohlen)

1. **Einmaliges Setup ausführen:**
   ```powershell
   .\setup.bat
   ```
   Dies erstellt die virtuelle Umgebung und installiert alle Abhängigkeiten.

2. **Anwendung starten:**
   
   **Am einfachsten (empfohlen):**
   ```powershell
   .\run.ps1
   ```
   oder
   ```cmd
   run.bat
   ```
   
   **Alternative:**
   ```powershell
   .\start_app.ps1
   ```
   oder
   ```cmd
   start_app.bat
   ```
   
   **Hinweis:** Sie können NICHT einfach `streamlit run` ausführen, da streamlit nur in der virtuellen Umgebung installiert ist. Verwenden Sie stattdessen eines der obigen Skripte!

### Option 2: Manuelles Setup

1. **Virtuelle Umgebung erstellen:**
   ```powershell
   python -m venv .venv
   ```

2. **Virtuelle Umgebung aktivieren:**
   
   **PowerShell:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **CMD:**
   ```cmd
   .venv\Scripts\activate.bat
   ```

3. **Abhängigkeiten installieren:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Anwendung starten:**
   ```powershell
   streamlit run esco_kg_streamlit.py
   ```

## Abhängigkeiten

Die `requirements.txt` enthält alle benötigten Pakete:
- streamlit
- pandas
- numpy
- scikit-learn
- beautifulsoup4
- requests
- lxml

## Hinweise

- Nach dem einmaligen Setup können Sie die Anwendung jederzeit mit `streamlit run esco_kg_streamlit.py` starten, solange die virtuelle Umgebung aktiviert ist.
- Die Anwendung öffnet sich automatisch im Browser unter `http://localhost:8501`

