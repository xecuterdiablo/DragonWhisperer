# 🐉 Dragon Whisperer

**Die ultimative Lösung für Live‑Stream‑Transkription & Echtzeit‑Übersetzung** – plattformunabhängig, GPU‑beschleunigt, mit moderner Dark‑GUI und umfangreichen Erweiterungsmöglichkeiten.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/xecuterdiablo/DragonWhisperer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🚀 Überblick

Dragon Whisperer verwandelt jeden Audio‑ oder Videostream **live** in lesbaren Text – und übersetzt ihn gleichzeitig in über 50 Sprachen. Egal ob YouTube, Twitch, lokale Dateien oder HLS‑Streams – die intelligente Streaming‑Engine extrahiert zuverlässig die Tonspur und liefert die Transkription direkt in eine moderne, dunkle Benutzeroberfläche.

---

## ✨ Highlights

| Bereich | Features |
|---------|----------|
| **Transkription** | 🎤 Echtzeit‑Transkription mit Whisper AI (faster‑whisper oder openai‑whisper) |
| **Übersetzung** | 🌐 Live‑Übersetzung in **>50 Sprachen** (Google Translate, argos‑translate, Ollama) |
| **Streaming** | 📡 Unterstützt YouTube, Twitch, Kick, Rumble, Facebook, Twitter/X, TikTok, HLS, DASH, lokale Dateien |
| **GUI** | 🎨 Modernes Dark‑Theme, anpassbares Layout (vertikal/horizontal), 10+ Farbthemen |
| **Performance** | ⚡ GPU‑Beschleunigung (CUDA, ROCm, Apple Silicon MPS), intelligente Caches, dynamisches Queue‑Management |
| **Export** | 📝 Untertitel (SRT, VTT, JSON, TXT, DOCX) |
| **KI‑Zusammenfassung** | 🤖 Ollama‑Summarizer für automatische Zusammenfassungen & Transkript‑Korrektur |
| **Vorlesen** | 🔊 Text‑to‑Speech (Piper, pyttsx3, espeak) |
| **Stabilität** | 🚦 Automatische Reconnects, adaptive Chunk‑Dauer, Rate‑Limiting |
| **Plugins** | 🔌 Plugin‑System für eigene Erweiterungen |
| **Linux‑Optimierung** | 🐧 Leistungsoptimierung (nice‑Werte, Dateideskriptor‑Limits) |

---

## 📸 Screenshot

![dragonscreenshot.avif](https://user11029.na.imgto.link/public/20260514/dragonscreenshot-2.avif)

---

## 📦 Systemanforderungen

| Komponente | Minimal | Empfohlen | Optimal |
|------------|---------|-----------|---------|
| **CPU** | 2 Kerne | 4 Kerne | 8 Kerne |
| **RAM** | 4 GB (tiny‑Modell) | 8 GB (base/small) | 16 GB (medium/large) |
| **GPU** | – | NVIDIA CUDA, AMD ROCm oder Apple MPS (optional, aber dringend empfohlen) | ≥6 GB VRAM |
| **Python** | ≥3.8 | ≥3.10 | ≥3.12 |
| **Betriebssystem** | Windows 10+, macOS (Intel/Apple Silicon), Linux (Arch, Ubuntu, Debian) | – | – |

---

## 🔧 Abhängigkeiten

### System‑Tools

| Tool | Installation (Beispiele) |
|------|--------------------------|
| **ffmpeg** | [ffmpeg.org](https://ffmpeg.org/) / `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Debian/Ubuntu) / `sudo pacman -S ffmpeg` (Arch) |
| **yt-dlp** | `pip install yt-dlp` oder Systempaket (`sudo apt install yt-dlp`) |

### Python‑Pakete

**Empfohlenes Backend:**
```bash
pip install faster-whisper

Alternatives Backend:
bash

pip install openai-whisper

Basis‑Abhängigkeiten (einmaliger Befehl):
bash

pip install torch numpy scipy deep-translator psutil requests

Für alle optionalen Funktionen (TTS, Rauschunterdrückung, Word‑Export, etc.):
bash

pip install argostranslate pyttsx3 noisereduce rapidfuzz python-docx dimits langdetect pathvalidate pynvml

GUI (tkinter)

    Linux: sudo apt install python3-tk (Debian/Ubuntu) / sudo pacman -S tk (Arch)

    Windows/macOS: normalerweise enthalten

PyTorch mit GPU‑Unterstützung
Für CUDA 11.8:
bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Für ROCm oder Apple MPS folge der offiziellen PyTorch‑Dokumentation.
⚡ Schnellstart (alle Betriebssysteme)

    Repository klonen oder Skript herunterladen

    git clone https://github.com/xecuterdiablo/DragonWhisperer.git
    cd DragonWhisperer

    Falls du Git nicht nutzt, lade das Skript manuell herunter (siehe Windows‑Anleitung unten).

    Abhängigkeiten installieren (siehe oben)

    Skript starten    

    python Dragon_Whisperer.py

    URL eingeben – z. B. https://www.youtube.com/watch?v=... oder lokale Datei mit file:///pfad/datei.mp4

    START klicken ✨

    Hinweis: Konfigurationen, Profile und Cache‑Daten werden automatisch in plattformspezifischen Benutzerverzeichnissen gespeichert:

        Linux: ~/.config/dragonwhisperer/

        Windows: %APPDATA%\DragonWhisperer\

        macOS: ~/Library/Application Support/DragonWhisperer/

🛠️ Kommandozeilenoptionen
Option	Beschreibung
--debug	Ausführliche Debug‑Ausgaben (optional --debug=2 oder --debug=vad,network)
--quiet, -q	Nur Fehlermeldungen anzeigen
--check	Systemkompatibilität prüfen
--test	Interne Unit‑Tests ausführen
--version, -v	Versionsinformation anzeigen
--help, -h	Hilfe anzeigen
🤖 Whisper‑Modelle (Übersicht)
Modell	RAM (ca.)	Geschwindigkeit	Genauigkeit	Empfehlung
tiny	~1 GB	🚀 extrem schnell	🔴 gering	Echtzeit, low‑resource
base	~1,5 GB	🚀 sehr schnell	🟡 mittel	Alltagstauglich
small	~2,5 GB	🟡 mittel	🟢 gut	Gute Balance
medium	~6 GB	🔴 langsam	🟢 sehr gut	Für anspruchsvolle Audios
large-v3	>10 GB	🐢 sehr langsam	💎 exzellent	Höchste Präzision
large-v3-turbo	~7 GB	🟡 mittel	🟢 sehr gut	Schneller als large
distil-large-v3	~6 GB	🟢 schnell	🟢 gut	Kompromiss

    Hinweis: Bei GPU‑Nutzung verkürzen sich die Ladezeiten. Die Auswahl erfolgt über das Dropdown‑Menü in der GUI.

🎛️ Erweiterte Einstellungen (⚙️)

Klicke auf das Zahnrad‑Symbol in der Statusleiste. Die Advanced Settings umfassen:

    Audio & VAD – Chunk‑Dauer, VAD‑Filter, Schwellwerte

    Modell & Inferenz – Beam Size, Temperatur, Hotwords, GPU‑Beschleunigung

    Transkriptions‑Filter – Min. Konfidenz, Duplikaterkennung, adaptive Chunk‑Größe

    Übersetzung – Engine (Google, Ollama, Argos), Ollama‑Modell & Host, Reflection‑Modus

    GUI & Display – Max. Zeilen, Theme, Auto‑Save

    Erweitert & System – Cache‑Größe, Plugins, Browser‑Cookies, Asian Mode, Precision Mode, Proxy, Blacklist

    Text‑to‑Speech – Engine (Piper/pyttsx3), Stimme, Geschwindigkeit, Satzpause

    Erweiterte Whisper‑Parameter – best_of, patience, no_speech_threshold, log_prob_threshold, compression_ratio_threshold, condition_on_previous_text, suppress_tokens

    Zusammenfassung (Ollama) – Temperatur und Modell

Alle Einstellungen werden automatisch gespeichert.
🐧 Plattform‑Hinweise

    Linux – Entwickelt unter Arch Linux. Der LinuxPerformanceOptimizer (benötigt psutil) reduziert bei Bedarf die GUI‑Last und optimiert Thread‑Prioritäten.

    Windows – Volle Unterstützung, UTF‑8‑Konsolen‑Setup, CREATE_NO_WINDOW‑Flags, automatische Wiederherstellung der Codepage.

    macOS – Apple Silicon (M1/M2/M3) wird via MPS‑Backend erkannt und genutzt.

🆘 Fehlerbehebung
Problem	Lösung
ffmpeg not found	Installiere ffmpeg (siehe Abhängigkeiten) und stelle sicher, dass es im PATH ist.
yt-dlp not found	Installiere yt-dlp: pip install yt-dlp.
Keine Audiowiedergabe bei YouTube‑Streams	Starte mit --debug und aktualisiere yt-dlp: pip install -U yt-dlp.
GUI startet nicht (Linux)	Fehlt tkinter: sudo apt install python3-tk (Debian/Ubuntu) / sudo pacman -S tk (Arch).
GPU wird nicht erkannt	Prüfe PyTorch: python -c "import torch; print(torch.cuda.is_available())".
Übersetzung funktioniert nicht	Stelle sicher, dass deep-translator oder argos-translate installiert ist. Bei Ollama: ollama serve.
Fehler beim Export von DOCX	Installiere python-docx: pip install python-docx (Fallback auf TXT).

Weitere Hilfe: Erstelle ein Issue auf GitHub und füge die Ausgabe von python dragon_whisperer.py --debug bei.
📄 Lizenz

MIT‑Lizenz – siehe LICENSE.
🙏 Danksagung

    OpenAI Whisper

    faster‑whisper

    yt-dlp

    deep‑translator

    argos‑translate

    Ollama

    Piper

    … und allen anderen Open‑Source‑Projekten.

💬 Kontakt & Support

Bei Fragen, Problemen oder Ideen – öffne ein Issue. Beiträge sind willkommen!

Viel Spaß mit Dragon Whisperer – deinem persönlichen Drachen für Transkription und Übersetzung. 🐉
🪟 Windows‑Einsteiger‑Anleitung (Schritt für Schritt)

Diese Anleitung ist für absolute Anfänger geschrieben, die Python nicht kennen. Du musst keine Vorkenntnisse haben – jede Anweisung ist als genauer Befehl formuliert.
📌 Voraussetzungen

    Windows 10 oder 11 (64‑Bit)

    Admin‑Rechte (nur für die Python‑Installation kurzzeitig nötig)

1️⃣ Python installieren

    Lade Python herunter
    Gehe auf python.org/downloads und klicke auf den gelben Button „Download Python 3.13.2“ (oder eine neuere Version).

    Starte die Installation
    Führe die heruntergeladene Datei aus.
    → Wichtig: Setze unten im Installationsfenster unbedingt den Haken bei ✅ Add Python to PATH
    → Klicke dann auf „Install Now“.

    Prüfe die Installation
    Drücke Win + R, tippe cmd ein und drücke Enter.
    Gib im schwarzen Fenster ein:
    cmd

    python --version

    Zeigt es Python 3.13.2 (oder ähnlich) an, ist alles richtig.

2️⃣ Projekt‑Ordner vorbereiten

    Erstelle einen Ordner (z. B. auf dem Desktop oder in Dokumente).
    Wichtig: Verwende keine Leerzeichen im Pfad, z. B. DragonWhisperer.

    Öffne die Eingabeaufforderung in diesem Ordner:
    Gehe in den neuen Ordner, klicke in die Adressleiste, lösche den Inhalt, tippe cmd ein und drücke Enter.
    (Es öffnet sich ein schwarzes Fenster, das bereits auf diesen Ordner zeigt.)

3️⃣ Virtuelle Umgebung (venv) erstellen

Im geöffneten Konsolenfenster tippe:
cmd

python -m venv venv

Nach einigen Sekunden erscheint ein Unterordner venv.
4️⃣ Das große Skript herunterladen (weil es zu groß für direkten Download ist)

    Die wichtigste Datei: Dragon_Whisperer_Full.py
    Um den kompletten Quellcode zu erhalten, öffne den RAW‑Link im Browser:
    🔗 https://raw.githubusercontent.com/xecuterdiablo/DragonWhisperer/refs/heads/main/Dragon_Whisperer_Full.py

Schritte:

    Klicke auf den RAW‑Link (oder kopiere ihn in die Adresszeile deines Browsers).
    Du siehst nun den gesamten Python‑Code – ohne GitHub‑Menü.

    Markiere alles: Drücke Strg + A (oder Ctrl + A).

    Kopiere den Code: Drücke Strg + C.

    Erstelle die Datei auf deinem Rechner:

        Öffne den Editor (Notepad) oder besser Notepad++ / VS Code.

        Achtung: Speichere die Datei bevor du etwas einfügst?
        Nein – zuerst einfügen, dann speichern.
        Aber du musst den Editor öffnen, dann Strg + V drücken, um den kopierten Code einzufügen.

        Kodierung: Stelle sicher, dass die Datei als UTF‑8 ohne BOM gespeichert wird.

            Im normalen Windows‑Editor: Gehe auf Datei → Speichern unter… → wähle als Dateityp „Alle Dateien“ und trage die Endung .py an.
Der Editor speichert standardmäßig UTF‑8 – das ist in Ordnung.

            In Notepad++: Menü Kodierung → UTF‑8 ohne BOM.

    Speichere die Datei im Projektordner DragonWhisperer unter dem Namen Dragon_Whisperer_Full.py.

    Überprüfe: Die Datei muss im selben Ordner liegen wie der Unterordner venv.

5️⃣ Abhängigkeiten installieren

Im Konsolenfenster (immer noch im Projektordner) aktivierst du zuerst die virtuelle Umgebung:
cmd

venv\Scripts\activate

Du siehst nun (venv) am Anfang der Eingabezeile.

Installiere die notwendigen Pakete:
cmd

pip install faster-whisper torch numpy scipy deep-translator psutil requests

Optional für mehr Funktionen (TTS, Rauschunterdrückung, DOCX‑Export):
cmd

pip install argostranslate pyttsx3 noisereduce rapidfuzz python-docx dimits langdetect pathvalidate pynvml

(Dies kann einige Minuten dauern – Geduld.)
6️⃣ Skript starten

Stelle sicher, dass (venv) aktiv ist. Gib ein:
cmd

python Dragon_Whisperer_Full.py

Das GUI‑Fenster öffnet sich. Fertig! 🎉

Zum Beenden schließe das Fenster oder drücke Strg + C in der Konsole.
7️⃣ (Optional) Desktop‑Verknüpfung für schnellen Start

    Rechtsklick auf Desktop → Neu → Verknüpfung.

    Pfad eingeben (passe den Pfad zu deinem Projektordner an!):
    cmd

    C:\Windows\System32\cmd.exe /k "cd /d C:\Users\DEIN_BENUTZERNAME\DragonWhisperer && venv\Scripts\python.exe Dragon_Whisperer_Full.py"

    Ersetze DEIN_BENUTZERNAME durch deinen echten Windows‑Benutzernamen.

    Weiter, Name z. B. Dragon Whisperer, Fertig.

    Doppelklick auf die Verknüpfung – das Skript startet sofort.

❗ Häufige Probleme & Lösungen
Problem	Lösung
python wird nicht erkannt	Python nicht zu PATH hinzugefügt. Deinstalliere und installiere Python mit Haken bei „Add to PATH“.
Fehler „No module named …“	Pakete nicht installiert. Wiederhole Schritt 5.
RAW‑Link zeigt nur Text, kein Download	Das ist korrekt. Markiere alles (Strg + A), kopiere (Strg + C) und füge in eine neue Datei ein.
Desktop‑Verknüpfung startet nicht	Pfad prüfen: Keine Leerzeichen? Ganzer Befehl in einer Zeile?
Skript schließt sofort nach Start	Konsole manuell öffnen (Schritt 6) und dort starten – dann siehst du die Fehlermeldung.
✅ Zusammenfassung der wichtigsten Befehle
cmd

# Projektordner erstellen und wechseln
mkdir C:\Users\DEIN_NAME\DragonWhisperer
cd /d C:\Users\DEIN_NAME\DragonWhisperer

# Virtuelle Umgebung erstellen
python -m venv venv

# Aktivieren
venv\Scripts\activate

# Pakete installieren
pip install faster-whisper torch numpy scipy deep-translator psutil requests

# Skript starten
python Dragon_Whisperer_Full.py

📌 Hinweis zur Textkodierung

Die Skriptdatei muss UTF‑8 kodiert sein. Der normale Windows‑Editor speichert als UTF‑8 mit BOM – das ist akzeptabel.
Für maximale Kompatibilität empfiehlt sich Notepad++ mit UTF‑8 ohne BOM.
🇬🇧 English Version

For non‑German speakers, the same content in English.
🐉 Dragon Whisperer

The ultimate solution for live‑stream transcription & real‑time translation – cross‑platform, GPU‑accelerated, with a modern dark GUI and extensive extensibility.

https://img.shields.io/badge/python-3.8%252B-blue
https://img.shields.io/badge/license-MIT-green
https://img.shields.io/badge/platform-Windows%2520%257C%2520macOS%2520%257C%2520Linux-lightgrey
https://img.shields.io/badge/code%2520style-black-000000.svg
🚀 Overview

Dragon Whisperer turns any audio or video stream live into readable text – and simultaneously translates
it into over 50 languages. Whether YouTube, Twitch, local files or HLS streams –
the intelligent streaming engine reliably extracts the audio track and delivers the transcription directly into a modern, dark user interface.
✨ Highlights
Area	Features
Transcription	🎤 Real‑time transcription with Whisper AI (faster‑whisper or openai‑whisper)
Translation	🌐 Live translation into >50 languages (Google Translate, argos‑translate, Ollama)
Streaming	📡 Supports YouTube, Twitch, Kick, Rumble, Facebook, Twitter/X, TikTok, HLS, DASH, local files
GUI	🎨 Modern dark theme, adjustable layout (vertical/horizontal), 10+ color themes
Performance	⚡ GPU acceleration (CUDA, ROCm, Apple Silicon MPS), intelligent caches, dynamic queue management
Export	📝 Subtitles (SRT, VTT, JSON, TXT, DOCX)
AI summarisation	🤖 Ollama summarizer for automatic summaries & transcript correction
Read aloud	🔊 Text‑to‑Speech (Piper, pyttsx3, espeak)
Stability	🚦 Automatic reconnects, adaptive chunk duration, rate limiting
Plugins	🔌 Plugin system for custom extensions
Linux optimisation	🐧 Performance tuning (nice values, file descriptor limits)
📸 Screenshot

https://user11029.na.imgto.link/public/20260514/dragonscreenshot-2.avif
📦 System requirements
Component	Minimal	Recommended	Optimal
CPU	2 cores	4 cores	8 cores
RAM	4 GB (tiny model)	8 GB (base/small)	16 GB (medium/large)
GPU	–	NVIDIA CUDA, AMD ROCm or Apple MPS (optional, but highly recommended)	≥6 GB VRAM
Python	≥3.8	≥3.10	≥3.12
OS	Windows 10+, macOS (Intel/Apple Silicon), Linux (Arch, Ubuntu, Debian)	–	–
🔧 Dependencies
System tools
Tool	Installation (examples)
ffmpeg	ffmpeg.org / brew install ffmpeg (macOS) / sudo apt install ffmpeg (Debian/Ubuntu) / sudo pacman -S ffmpeg (Arch)
yt-dlp	pip install yt-dlp or system package (sudo apt install yt-dlp)
Python packages

Recommended backend:
bash

pip install faster-whisper

Alternative backend:
bash

pip install openai-whisper

Base dependencies (one‑time command):
bash

pip install torch numpy scipy deep-translator psutil requests

For all optional features (TTS, noise reduction, Word export, etc.):
bash

pip install argostranslate pyttsx3 noisereduce rapidfuzz python-docx dimits langdetect pathvalidate pynvml

GUI (tkinter)

    Linux: sudo apt install python3-tk (Debian/Ubuntu) / sudo pacman -S tk (Arch)

    Windows/macOS: usually included

PyTorch with GPU support
For CUDA 11.8:
bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For ROCm or Apple MPS follow the official PyTorch documentation.
⚡ Quick start (all platforms)

    Clone the repository or download the script
    bash

    git clone https://github.com/xecuterdiablo/DragonWhisperer.git
    cd DragonWhisperer

    If you don‘t use Git, manually download the script (see Windows guide below).

    Install dependencies (see above)

    Run the script
    bash

    python Dragon_Whisperer.py

    Enter a URL – e.g. https://www.youtube.com/watch?v=... or a local file with file:///path/file.mp4

    Click START ✨

    Note: Configurations, profiles and cache data are automatically stored in platform‑specific user directories:

        Linux: ~/.config/dragonwhisperer/

        Windows: %APPDATA%\DragonWhisperer\

        macOS: ~/Library/Application Support/DragonWhisperer/

🛠️ Command line options
Option	Description
--debug	Verbose debug output (optional --debug=2 or --debug=vad,network)
--quiet, -q	Show only error messages
--check	Check system compatibility
--test	Run internal unit tests
--version, -v	Show version information
--help, -h	Show help
🤖 Whisper models (overview)
Model	RAM (approx.)	Speed	Accuracy	Recommendation
tiny	~1 GB	🚀 extremely fast	🔴 low	Real‑time, low‑resource
base	~1.5 GB	🚀 very fast	🟡 medium	Everyday use
small	~2.5 GB	🟡 medium	🟢 good	Good balance
medium	~6 GB	🔴 slow	🟢 very good	Demanding audio
large-v3	>10 GB	🐢 very slow	💎 excellent	Highest precision
large-v3-turbo	~7 GB	🟡 medium	🟢 very good	Faster than large
distil-large-v3	~6 GB	🟢 fast	🟢 good	Compromise

    Note: GPU usage shortens loading times. Selection is done via the dropdown menu in the GUI.

🎛️ Advanced settings (⚙️)

Click the gear icon in the status bar. The Advanced Settings include:

    Audio & VAD – chunk duration, VAD filter, thresholds

    Model & Inference – beam size, temperature, hotwords, GPU acceleration

    Transcription Filters – min. confidence, duplicate detection, adaptive chunk size

    Translation – engine (Google, Ollama, Argos), Ollama model & host, reflection mode

    GUI & Display – max lines, theme, auto‑save

    Advanced & System – cache size, plugins, browser cookies, Asian mode, Precision mode, proxy, blacklist

    Text‑to‑Speech – engine (Piper/pyttsx3), voice, speed, sentence pause

    Advanced Whisper parameters – best_of, patience, no_speech_threshold, log_prob_threshold, compression_ratio_threshold, condition_on_previous_text, suppress_tokens

    Summarisation (Ollama) – temperature and model

All settings are automatically saved.
🐧 Platform notes

    Linux – Developed on Arch Linux. The LinuxPerformanceOptimizer (requires psutil) reduces GUI load and optimises thread priorities.

    Windows – Full support, UTF‑8 console setup, CREATE_NO_WINDOW flags, automatic code page restoration.

    macOS – Apple Silicon (M1/M2/M3) is detected and used via the MPS backend.

🆘 Troubleshooting
Problem	Solution
ffmpeg not found	Install ffmpeg (see dependencies) and ensure it is in your PATH.
yt-dlp not found	Install yt-dlp: pip install yt-dlp.
No audio playback for YouTube streams	Run with --debug and update yt-dlp: pip install -U yt-dlp.
GUI does not start (Linux)	Missing tkinter: sudo apt install python3-tk (Debian/Ubuntu) / sudo pacman -S tk (Arch).
GPU not detected	Check PyTorch: python -c "import torch; print(torch.cuda.is_available())".
Translation does not work	Ensure deep-translator or argos-translate is installed. For Ollama: ollama serve.
Error exporting DOCX	Install python-docx: pip install python-docx (falls back to TXT).

Further help: Open an issue on GitHub and include the output of python dragon_whisperer.py --debug.
📄 License

MIT License – see LICENSE.
🙏 Acknowledgements

    OpenAI Whisper

    faster‑whisper

    yt-dlp

    deep‑translator

    argos‑translate

    Ollama

    Piper

    … and all other open‑source projects.

💬 Contact & Support

For questions, problems or ideas – just open an Issue. Contributions are welcome!

Enjoy Dragon Whisperer – your personal dragon for transcription and translation. 🐉
🪟 Windows Beginner Guide (step by step)

This guide is written for absolute beginners who do not know Python. You need no prior knowledge – every instruction is given as an exact command.
📌 Prerequisites

    Windows 10 or 11 (64‑bit)

    Admin rights (temporarily needed for Python installation)

1️⃣ Install Python

    Download Python
    Go to python.org/downloads and click the yellow button “Download Python 3.13.2” (or a newer version).

    Run the installer
    Execute the downloaded file.
    → Important: At the bottom of the installer window, make sure to check ✅ Add Python to PATH
    → Then click “Install Now”.

    Verify the installation
    Press Win + R, type cmd and press Enter.
    In the black window type:
    cmd

    python --version

    If it shows Python 3.13.2 (or similar), everything is correct.

2️⃣ Prepare the project folder

    Create a folder (e.g. on your Desktop or in Documents).
    Important: Do not use spaces in the path, e.g. DragonWhisperer.

    Open the Command Prompt in this folder:
    Go into the new folder, click in the address bar, delete its content, type cmd and press Enter.
    (A black window opens that already points to this folder.)

3️⃣ Create a virtual environment (venv)

In the opened console window type:
cmd

python -m venv venv

After a few seconds a subfolder venv appears.
4️⃣ Download the large script (because it is too large for direct download)

    The most important file: Dragon_Whisperer_Full.py
    To get the complete source code, open the RAW link in your browser:
    🔗 https://raw.githubusercontent.com/xecuterdiablo/DragonWhisperer/refs/heads/main/Dragon_Whisperer_Full.py

Steps:

    Click the RAW link (or copy it into the browser‘s address bar).
    You will now see the entire Python code – without the GitHub menu.

    Select all: Press Ctrl + A.

    Copy the code: Press Ctrl + C.

    Create the file on your computer:

        Open Notepad (or better Notepad++ / VS Code).

        Important: First paste the code (Ctrl + V), then save.

        Encoding: Make sure the file is saved as UTF‑8 without BOM.

            In the standard Windows Notepad: Go to File → Save as… → choose file type “All files” and add the extension .py. The editor saves as UTF‑8 by default – that‘s fine.

            In Notepad++: Menu Encoding → UTF‑8 without BOM.

    Save the file in the project folder DragonWhisperer with the name Dragon_Whisperer_Full.py.

    Check: The file must be in the same folder as the venv subfolder.

5️⃣ Install dependencies

In the console window (still in the project folder) first activate the virtual environment:
cmd

venv\Scripts\activate

You will now see (venv) at the beginning of the command line.

Install the required packages:
cmd

pip install faster-whisper torch numpy scipy deep-translator psutil requests

Optional for more features (TTS, noise reduction, DOCX export):
cmd

pip install argostranslate pyttsx3 noisereduce rapidfuzz python-docx dimits langdetect pathvalidate pynvml

(This may take a few minutes – be patient.)
6️⃣ Run the script

Make sure (venv) is active. Type:
cmd

python Dragon_Whisperer_Full.py

The GUI window will open. Done! 🎉

To stop, close the window or press Ctrl + C in the console.
7️⃣ (Optional) Desktop shortcut for quick launch

    Right‑click on Desktop → New → Shortcut.

    Enter the location (adjust the path to your project folder!):
    cmd

    C:\Windows\System32\cmd.exe /k "cd /d C:\Users\YOUR_USERNAME\DragonWhisperer && venv\Scripts\python.exe Dragon_Whisperer_Full.py"

    Replace YOUR_USERNAME with your actual Windows user name.

    Next, name e.g. Dragon Whisperer, Finish.

    Double‑click the shortcut – the script starts immediately.

❗ Common problems & solutions
Problem	Solution
python not recognized	Python not added to PATH. Uninstall and reinstall Python with the “Add to PATH” checkbox.
Error “No module named …”	Packages not installed. Repeat step 5.
RAW link shows only text, no download	That‘s correct. Select all (Ctrl+A), copy (Ctrl+C) and paste into a new file.
Desktop shortcut does not start	Check the path: No spaces? Whole command on one line?
Script closes immediately after start	Open the console manually (step 6) and run it there – then you will see the error message.
✅ Summary of essential commands
cmd

# Create project folder and change into it
mkdir C:\Users\YOUR_NAME\DragonWhisperer
cd /d C:\Users\YOUR_NAME\DragonWhisperer

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install packages
pip install faster-whisper torch numpy scipy deep-translator psutil requests

# Run the script
python Dragon_Whisperer_Full.py

📌 Note on text encoding

The script file must be UTF‑8 encoded. The standard Windows Notepad saves as UTF‑8 with BOM – that is acceptable. For maximum compatibility, use Notepad++ with UTF‑8 without BOM.
text


