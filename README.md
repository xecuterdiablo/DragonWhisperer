🐉 Dragon Whisperer v1.6

Die ultimative Lösung für Live‑Stream‑Transkription und Echtzeit‑Übersetzung
Plattformunabhängig, GPU‑beschleunigt und mit moderner Dark‑GUI

🚀 Überblick

Dragon Whisperer verwandelt jeden Audio‑ oder Videostream in lesbaren Text – live, genau und in über 50 Sprachen übersetzt.
Egal ob YouTube, Twitch, lokale Dateien oder HLS‑Streams – die intelligente Streaming‑Engine extrahiert zuverlässig die Tonspur und liefert dir die Transkription direkt ins moderne, dunkle GUI.

Highlights
    🎤 Echtzeit‑Transkription mit Whisper AI (wahlweise faster‑whisper oder openai‑whisper)
    🌐 Live‑Übersetzung in >50 Sprachen (über deep‑translator)
    📡 Unterstützt alle gängigen Plattformen (YouTube, Twitch, Kick, Rumble, Facebook, Twitter/X, TikTok, HLS, DASH, lokale Dateien)
    🎨 Modernes Dark‑GUI mit anpassbarem Layout (vertikal/horizontal)
    ⚡ GPU‑Beschleunigung (CUDA, ROCm, Apple Silicon MPS)
    🔧 Erweiterte Einstellungen (Beam Size, Temperatur, VAD‑Parameter, Caching)
    📝 Untertitel‑Export (SRT, VTT, JSON, TXT, DOCX)
    🤖 Optionaler Ollama‑Summarizer für Zusammenfassungen

📸 Screenshot

    X

📦 Systemanforderungen
Minimal
    CPU: 2 Kerne
    RAM: 4 GB (für tiny‑Modell)
    Python: ≥ 3.8
    Betriebssystem: Windows 10+, macOS (Intel/Apple Silicon), Linux

Empfohlen
    CPU: 4 Kerne
    RAM: 8 GB (für base/small)
    GPU: NVIDIA CUDA, AMD ROCm oder Apple MPS (optional, aber dringend empfohlen)

Optimal
    CPU: 8 Kerne
    RAM: 16 GB (für medium/large)
    GPU: mit ≥ 6 GB VRAM

🔧 Abhängigkeiten
Systemweit

    ffmpeg (im PATH oder in Standardpfaden)
    yt-dlp (zur Stream‑Extraktion)

Installation unter

    Windows: ffmpeg.org & pip install yt-dlp
    macOS: brew install ffmpeg yt-dlp
    Linux (Debian/Ubuntu): sudo apt install ffmpeg yt-dlp
    Arch Linux: sudo pacman -S ffmpeg yt-dlp

Python‑Pakete

Die folgenden Pakete werden benötigt – die meisten werden automatisch erkannt, fehlende erzeugen eine klare Fehlermeldung.
bash

pip install faster-whisper      # Empfohlen (schneller, weniger RAM)
# oder alternativ:
pip install openai-whisper       # Falls faster-whisper nicht verfügbar

# Basis‑Abhängigkeiten
pip install torch numpy scipy deep-translator psutil requests

# GUI (tkinter ist meist vorinstalliert, falls nicht):
# Linux: sudo apt install python3-tk
# Windows/macOS: normalerweise enthalten

Hinweis zu PyTorch mit GPU
Für CUDA‑Unterstützung installiere torch mit dem passenden Index, z.B. für CUDA 11.8:
bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Für ROCm oder MPS folge der offiziellen PyTorch‑Dokumentation.
⚡ Schnellstart

    Repository klonen (oder Skript direkt herunterladen)
    bash

    git clone https://github.com/xecuterdiablo/DragonWhisperer/Dragon-Whisperer.git
    cd dragon-whisperer

    Abhängigkeiten installieren (siehe oben)

    Skript starten
    bash

    python Dragon_Whisperer.py

    URL eingeben – z.B. https://www.youtube.com/watch?v=... oder eine lokale Datei mit file:///pfad/datei.mp4

    START klicken und den Magic erleben! ✨

Kommandozeilenoptionen

    --debug – Ausführliche Debug‑Ausgaben
    --quiet – Nur Fehlermeldungen anzeigen
    --check – Systemkompatibilität prüfen
    --help – Hilfe anzeigen
    --version – Versionsinfo

🤖 Whisper‑Modelle
Modell	RAM (ca.)	Geschwindigkeit	Genauigkeit	Empfehlung
tiny	~1 GB	🚀 extrem schnell	🔴 gering	Echtzeit, low‑resource
base	~1,5 GB	🚀 sehr schnell	🟡 mittel	Alltagstauglich
small	~2,5 GB	🟡 mittel	🟢 gut	Gute Balance
medium	~6 GB	🔴 langsam	🟢 sehr gut	Für anspruchsvolle Audios
large	>10 GB	🐢 sehr langsam	💎 exzellent	Höchste Präzision

Die Modellauswahl erfolgt über das Dropdown‑Menü in der GUI. Bei GPU‑Nutzung verkürzen sich die Ladezeiten und die Verarbeitung wird flüssiger.
🎛️ Erweiterte Einstellungen

Über das Zahnrad‑Symbol ⚙️ in der Statusleiste gelangst du zu den Advanced Settings:

    Beam Size – Suchbreite des Decoders (höher = genauer, aber langsamer)
    Temperature – Kreativität bei der Textgenerierung (0.0 = deterministisch)
    VAD‑Filter – Voice Activity Detection (Spracherkennung zur Reduzierung von Rauschen)
    VAD‑Parameter – Schwellwert, minimale Sprach‑/Stilledauer
    GPU‑Beschleunigung – Aktivieren/Deaktivieren der GPU
    Plugins – Integrierte Plugins (Sentiment, Keyword‑Extraktion) ein/aus

Alle Einstellungen werden automatisch gespeichert und beim nächsten Start wiederhergestellt.
🐧 Plattform‑Hinweise

    Linux – Das Skript wurde primär unter Arch Linux entwickelt und getestet.
    Der integrierte LinuxPerformanceOptimizer (benötigt psutil) reduziert bei Bedarf die GUI‑Last und optimiert Thread‑Prioritäten.

    Windows – Volle Unterstützung, inklusive UTF‑8‑Konsolen‑Setup und „No Window“‑Flags für Subprozesse.

    macOS – Apple Silicon (M1/M2) wird via MPS‑Backend erkannt und genutzt.

🆘 Fehlerbehebung

Problem – ffmpeg not found
➡️ Installiere ffmpeg (siehe Abhängigkeiten) und stelle sicher, dass es im PATH ist.

Problem – yt-dlp not found
➡️ Installiere yt-dlp: pip install yt-dlp (oder Systempaket).

Problem – Keine Audiowiedergabe bei YouTube‑Streams
➡️ YouTube ändert häufig seine Streaming‑Protokolle. Starte das Skript mit --debug, um die Extraktion zu verfolgen, und aktualisiere ggf. yt-dlp: pip install -U yt-dlp.

Problem – GUI startet nicht (Linux)
➡️ Fehlt tkinter: sudo apt install python3-tk (Debian/Ubuntu) oder sudo pacman -S tk (Arch).

Problem – GPU wird nicht erkannt
➡️ Prüfe die PyTorch‑Installation:
python

python -c "import torch; print(torch.cuda.is_available())"

Falls False, installiere die passende torch‑Version für deine CUDA/ROCm/MPS.

Weitere Hilfe
Erstelle ein Issue auf GitHub und füge die Ausgabe von python dwII.py --debug bei.
📄 Lizenz

Dieses Projekt steht unter der MIT‑Lizenz. Siehe LICENSE für Details.
🙏 Danksagung

    OpenAI Whisper
    faster‑whisper
    yt‑dlp
    deep‑translator

    und allen anderen Open‑Source‑Projekten, die dies möglich machen.

Viel Spaß mit Dragon Whisperer!
Bei Fragen, Ideen oder Problemen – öffne einfach ein Issue.
