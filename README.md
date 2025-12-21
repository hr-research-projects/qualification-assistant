# ESCO Knowledge Graph Application

## 🚀 Anwendung starten

### **Einfachste Methoden:**

#### **1. Batch-Datei (Empfohlen):**
```
Doppelklick auf: start_app.bat
```

#### **2. Python-Skript:**
```
python start_app.py
```

#### **3. Direkt über Python:**
```
python -m streamlit run esco_kg_streamlit.py
```

#### **4. PowerShell:**
```
.\start_app.ps1
```

### **Desktop-Verknüpfung erstellen:**
```
Doppelklick auf: create_shortcut.bat
```

## 📋 Voraussetzungen

Stelle sicher, dass folgende Pakete installiert sind:
```bash
pip install streamlit pandas scikit-learn beautifulsoup4
```

## 🔧 Troubleshooting

### **Problem: "streamlit" nicht gefunden**
**Lösung:** Verwende `python -m streamlit` statt nur `streamlit`

### **Problem: Abhängigkeiten fehlen**
**Lösung:** Führe aus: `pip install -r requirements.txt`

## 📁 Projektstruktur

```
Jahresprojekt/
├── esco_kg_streamlit.py      # Hauptanwendung
├── start_app.bat            # Batch-Starter
├── start_app.ps1            # PowerShell-Starter
├── start_app.py             # Python-Starter
├── create_shortcut.bat      # Desktop-Verknüpfung erstellen
├── README.md                # Diese Datei
└── data/                    # Datenordner
    ├── employees_data.csv
    ├── courses.csv
    └── ...
```

## 🎯 Features

- 👥 Mitarbeiterverwaltung
- 📊 Kompetenzprofile
- 🔍 Berufsabgleich
- 📚 Kursempfehlungen
- 💾 Persistente Datenspeicherung 