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
| **GUI** | 🎨 Modernes Dark‑Theme, anpassbares Layout (vertikal/horizontal), 10+ Farbthemen (Dark, Light, Pastel, High Contrast, Dracula, Nord, …) |
| **Performance** | ⚡ GPU‑Beschleunigung (CUDA, ROCm, Apple Silicon MPS), intelligente Caches (TTL/LRU), dynamisches Queue‑Management |
| **Erweiterte Einstellungen** | 🔧 Beam Size, Temperatur, VAD‑Parameter, Blacklist, Hotwords, Proxy, Browser‑Cookies |
| **Export** | 📝 Untertitel (SRT, VTT, JSON, TXT, DOCX) |
| **KI‑Zusammenfassung** | 🤖 Ollama‑Summarizer für automatische Zusammenfassungen & Transkript‑Korrektur |
| **Vorlesen** | 🔊 Text‑to‑Speech (Piper, pyttsx3, espeak) für Transkription und Übersetzung |
| **Stabilität** | 🚦 Dynamisches Queue‑Management mit Rate‑Limiting, automatische Reconnects, adaptive Chunk‑Dauer |
| **Plugins** | 🔌 Plugin‑System für eigene Erweiterungen |
| **Speicherverwaltung** | 🧹 Automatische Begrenzung von Textpuffern, LRU‑Caches, periodische Garbage‑Collection |
| **Linux‑Optimierung** | 🐧 Leistungsoptimierung (nice‑Werte, Dateideskriptor‑Limits, Compositor‑Erkennung) |

---

## 📸 Screenshot

> *(Füge hier einen aktuellen Screenshot des Hauptfensters ein. Ideal wäre ein animiertes GIF oder ein Video, das die Live‑Transkription und Übersetzung zeigt.)*

![Screenshot Platzhalter](screenshot.png)

---

## 📦 Systemanforderungen

| Komponente | Minimal | Empfohlen | Optimal |
|------------|---------|-----------|---------|
| **CPU** | 2 Kerne | 4 Kerne | 8 Kerne |
| **RAM** | 4 GB (tiny‑Modell) | 8 GB (base/small) | 16 GB (medium/large) |
| **GPU** | – | NVIDIA CUDA, AMD ROCm oder Apple MPS (optional, aber dringend empfohlen) | ≥6 GB VRAM |
| **Python** | ≥3.8 | ≥3.10 | ≥3.12 |
| **Betriebssystem** | Windows 10+, macOS (Intel/Apple Silicon), Linux (getestet unter Arch, Ubuntu, Debian) | – | – |

---

## 🔧 Abhängigkeiten

### System‑Tools

| Tool | Installation (Beispiele) |
|------|--------------------------|
| **ffmpeg** | [ffmpeg.org](https://ffmpeg.org/) / `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Debian/Ubuntu) / `sudo pacman -S ffmpeg` (Arch) |
| **yt-dlp** | `pip install yt-dlp` oder Systempaket (`sudo apt install yt-dlp`) |

### Python‑Pakete

Die meisten Pakete werden **automatisch erkannt** – fehlende erzeugen eine klare Fehlermeldung.

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

pip install faster-whisper argostranslate deep-translator pyttsx3 noisereduce rapidfuzz python-docx dimits langdetect pathvalidate psutil pynvml

GUI (tkinter)

    Linux: sudo apt install python3-tk (Debian/Ubuntu) / sudo pacman -S tk (Arch)

    Windows/macOS: normalerweise enthalten

PyTorch mit GPU‑Unterstützung

Für CUDA 11.8:
bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Für ROCm oder Apple MPS folge der offiziellen PyTorch‑Dokumentation.
⚡ Schnellstart

    Repository klonen (oder Skript herunterladen)
    bash

    git clone https://github.com/xecuterdiablo/DragonWhisperer.git
    cd DragonWhisperer

    Abhängigkeiten installieren (siehe oben)

    Skript starten
    bash

    python Dragon_Whisperer.py

    URL eingeben – z. B. https://www.youtube.com/watch?v=... oder eine lokale Datei mit file:///pfad/datei.mp4

    START klicken – und die Magie erleben! ✨

    Wichtig: Lege das Skript in einem eigenen Ordner ab (z. B. DragonWhisperer/), nicht direkt im Downloads‑Ordner. Dragon Whisperer speichert Konfigurationen, Profile und Cache‑Daten automatisch in plattformspezifischen Benutzerverzeichnissen – diese bleiben dadurch sauber getrennt vom Skriptverzeichnis.

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

    Hinweis: Bei GPU‑Nutzung verkürzen sich die Ladezeiten und die Verarbeitung wird flüssiger. Die Auswahl erfolgt über das Dropdown‑Menü in der GUI.

🎛️ Erweiterte Einstellungen (⚙️)

Über das Zahnrad‑Symbol in der Statusleiste gelangst du zu den Advanced Settings mit folgenden Kategorien:

    Audio & VAD – Chunk‑Dauer, VAD‑Filter, Schwellwerte, Sprach‑/Stilledauer

    Modell & Inferenz – Beam Size, Temperatur, Hotwords, GPU‑Beschleunigung, CPU‑Threads

    Transkriptions‑Filter – Min. Konfidenz, Duplikaterkennung, adaptive Chunk‑Größe

    Übersetzung – Engine (Google, Ollama, Argos), Ollama‑Modell & Host, Reflection‑Modus

    GUI & Display – Max. Zeilen in Textfeldern, Theme, Auto‑Save

    Erweitert & System – Cache‑Größe, Plugins, Browser‑Cookies, Asian Mode, Precision Mode, Proxy, Blacklist

    Text‑to‑Speech – Engine (Piper/pyttsx3), Stimme, Geschwindigkeit, Satzpause

    Erweiterte Whisper‑Parameter – best_of, patience, no_speech_threshold, log_prob_threshold, compression_ratio_threshold, condition_on_previous_text, suppress_tokens

    Zusammenfassung (Ollama) – Temperatur und Modell für den Summarizer

Alle Einstellungen werden automatisch gespeichert und beim nächsten Start wiederhergestellt.
🐧 Plattform‑Hinweise

    Linux – Primär unter Arch Linux entwickelt und getestet. Der integrierte LinuxPerformanceOptimizer (benötigt psutil) reduziert bei Bedarf die GUI‑Last und optimiert Thread‑Prioritäten sowie Dateideskriptor‑Limits.

    Windows – Volle Unterstützung, inkl. UTF‑8‑Konsolen‑Setup, CREATE_NO_WINDOW‑Flags für Subprozesse und automatischer Wiederherstellung der Konsolen‑Codepage.

    macOS – Apple Silicon (M1/M2/M3) wird via MPS‑Backend erkannt und genutzt; Intel‑Macs verwenden CPU oder CUDA (sofern verfügbar).

🆘 Fehlerbehebung
Problem	Lösung
ffmpeg not found	Installiere ffmpeg (siehe Abhängigkeiten) und stelle sicher, dass es im PATH ist.
yt-dlp not found	Installiere yt-dlp: pip install yt-dlp (oder Systempaket).
Keine Audiowiedergabe bei YouTube‑Streams	YouTube ändert häufig seine Streaming‑Protokolle. Starte das Skript mit --debug, um die Extraktion zu verfolgen, und aktualisiere ggf. yt-dlp: pip install -U yt-dlp.
GUI startet nicht (Linux)	Fehlt tkinter: sudo apt install python3-tk (Debian/Ubuntu) oder sudo pacman -S tk (Arch).
GPU wird nicht erkannt	Prüfe PyTorch: python -c "import torch; print(torch.cuda.is_available())". Falls False, installiere die passende torch‑Version für deine CUDA/ROCm/MPS.
Übersetzung funktioniert nicht	Stelle sicher, dass deep-translator oder argos-translate installiert ist. Bei Ollama prüfe, ob der Server läuft (ollama serve).
Fehler beim Export von DOCX	Installiere python-docx: pip install python-docx (Fallback auf TXT).

Weitere Hilfe: Erstelle ein Issue auf GitHub und füge die Ausgabe von python dragon_whisperer.py --debug bei.
📄 Lizenz

Dieses Projekt steht unter der MIT‑Lizenz. Siehe LICENSE für Details.
🙏 Danksagung

    OpenAI Whisper

    faster‑whisper

    yt‑dlp

    deep‑translator

    argos‑translate

    Ollama

    Piper

    … und allen anderen Open‑Source‑Projekten, die dies möglich machen.

💬 Kontakt & Support

Bei Fragen, Problemen oder Ideen – öffne einfach ein Issue. Beiträge sind willkommen!

Viel Spaß mit Dragon Whisperer – deinem persönlichen Drachen für Transkription und Übersetzung. 🐉
text


## 📌 Was wurde verbessert?

| Problem | Korrektur |
|---------|-----------|
| Badges als reine URLs | Jetzt als richtige Markdown‑Badges mit Links |
| Fehlende Sprachangaben in Code‑Blöcken | `bash`, `python` ergänzt |
| Inkonsistente Trennlinien | Einheitlich `---` |
| „Magic“ (Anglizismus) | „Magie“ |
| Fehlende Überschriften-Hierarchie | `###` für Unterkapitel korrigiert |
| Fehlender Abschnitt „Kontakt & Support“ | Ergänzt |
| Doppelte Leerzeichen / Zeilenumbrüche | Bereinigt |
| Schreibweise „Pythin“ (nicht vorhanden) | – |
| `screenshot.png` als echter Link? | Platzhalter bleibt, aber mit Hinweis auf GIF/Video |
