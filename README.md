🐉 Dragon Whisperer v4.0

Die ultimative Lösung für Live‑Stream‑Transkription und Echtzeit‑Übersetzung
Plattformunabhängig, GPU‑beschleunigt und mit moderner Dark‑GUI.

🚀 Überblick

Dragon Whisperer verwandelt jeden Audio‑ oder Videostream in lesbaren Text – live, genau und in über 50 Sprachen übersetzt. Egal ob YouTube, Twitch, lokale Dateien oder HLS‑Streams – die intelligente Streaming‑Engine extrahiert zuverlässig die Tonspur und liefert dir die Transkription direkt ins moderne, dunkle GUI.
✨ Highlights

    🎤 Echtzeit‑Transkription mit Whisper AI (wahlweise faster-whisper oder openai-whisper)
    🌐 Live‑Übersetzung in >50 Sprachen (über deep-translator, argos-translate oder lokale LLMs wie Ollama)
    📡 Unterstützt alle gängigen Plattformen: YouTube, Twitch, Kick, Rumble, Facebook, Twitter/X, TikTok, HLS, DASH, lokale Dateien
    🎨 Modernes Dark‑GUI mit anpassbarem Layout (vertikal/horizontal), Themen (Dark, Light, High Contrast)
    ⚡ GPU‑Beschleunigung (CUDA, ROCm, Apple Silicon MPS)
    🔧 Erweiterte Einstellungen (Beam Size, Temperatur, VAD‑Parameter, Caching, Blacklist, Hotwords, Proxy)
    📝 Untertitel‑Export (SRT, VTT, JSON, TXT, DOCX)
    🤖 Ollama‑Summarizer für automatische Zusammenfassungen und Transkript‑Korrekturen
    🔊 Text‑to‑Speech (Piper, pyttsx3) für das Vorlesen von Transkriptionen
    🧠 Intelligentes Caching (TTL‑ und LRU‑Caches) für bessere Performance
    🚦 Rate‑Limiting & Queue‑Management für flüssige GUI‑Updates
    🐧 Linux‑Performance‑Optimierungen (nice‑Werte, Dateideskriptor‑Limits, Compositor‑Erkennung)
    🔌 Plugin‑System für eigene Erweiterungen
    🧹 Automatische Speicherverwaltung mit MemoryManager

📸 Screenshot

(Füge hier einen aktuellen Screenshot ein)
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
    GPU: ≥ 6 GB VRAM

🔧 Abhängigkeiten
Systemweit

    ffmpeg (im PATH oder in Standardpfaden)
    yt-dlp (zur Stream‑Extraktion)

Installation

    Windows: ffmpeg.org & pip install yt-dlp
    macOS: brew install ffmpeg yt-dlp
    Linux (Debian/Ubuntu): sudo apt install ffmpeg yt-dlp
    Arch Linux: sudo pacman -S ffmpeg yt-dlp

Python‑Pakete

Die meisten Pakete werden automatisch erkannt; fehlende erzeugen eine klare Fehlermeldung.
bash

pip install faster-whisper   # Empfohlen (schneller, weniger RAM)
# oder alternativ:
pip install openai-whisper   # Falls faster-whisper nicht verfügbar

Basis‑Abhängigkeiten
bash

pip install torch numpy scipy deep-translator psutil requests

GUI (tkinter)

    Linux: sudo apt install python3-tk
    Windows/macOS: normalerweise enthalten

Hinweis zu PyTorch mit GPU

Für CUDA‑Unterstützung installiere torch mit dem passenden Index, z.B. für CUDA 11.8:
bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Für ROCm oder MPS folge der offiziellen PyTorch‑Dokumentation.
⚡ Schnellstart

    Repository klonen (oder Skript direkt herunterladen)
    bash

    git clone https://github.com/xecuterdiablo/DragonWhisperer.git
    cd DragonWhisperer

    Abhängigkeiten installieren (siehe oben)

    Skript starten
    bash

    python dragon_whisperer.py

    URL eingeben – z.B. https://www.youtube.com/watch?v=... oder eine lokale Datei mit file:///pfad/datei.mp4

    START klicken und den Magic erleben! ✨

🛠️ Kommandozeilenoptionen


Option	Beschreibung

--debug	Ausführliche Debug‑Ausgaben (optional --debug=2 oder --debug=vad,network)

--quiet, -q	Nur Fehlermeldungen anzeigen

--check	Systemkompatibilität prüfen

--test	Interne Unit‑Tests ausführen

--version, -v	Versionsinformation anzeigen

--help, -h	Hilfe anzeigen


🤖 Whisper‑Modelle

Modell	RAM (ca.)	Geschwindigkeit	Genauigkeit	Empfehlung

tiny	~1 GB	🚀 extrem schnell	🔴 gering	Echtzeit, low‑resource

base	~1,5 GB	🚀 sehr schnell	🟡 mittel	Alltagstauglich

small	~2,5 GB	🟡 mittel	🟢 gut	Gute Balance

medium	~6 GB	🔴 langsam	🟢 sehr gut	Für anspruchsvolle Audios

large	>10 GB	🐢 sehr langsam	💎 exzellent	Höchste Präzision

Die Modellauswahl erfolgt über das Dropdown‑Menü in der GUI. Bei GPU‑Nutzung verkürzen sich die Ladezeiten und die Verarbeitung wird flüssiger.
🎛️ Erweiterte Einstellungen

Über das Zahnrad‑Symbol ⚙️ in der Statusleiste gelangst du zu den Advanced Settings. Hier kannst du:

    Audio & VAD
    Chunk‑Dauer, VAD‑Filter, Schwellwerte, Sprach‑/Stilledauer

    Modell & Inferenz
    Beam Size, Temperatur, Hotwords, GPU‑Beschleunigung

    Transkriptions‑Filter
    Min. Konfidenz, Duplikaterkennung, adaptive Chunk‑Größe

    Übersetzung
    Engine (Google, Ollama, Argos), Ollama‑Modell & Host, Proxy

    GUI & Display
    Max. Zeilen in Textfeldern, Theme, Auto‑Save

    Erweitert & System
    Cache‑Größe, Plugins, Browser‑Cookies, Asian Mode, Precision Mode, Proxy

    Blacklist
    Phrasen, die aus der Ausgabe gefiltert werden (Wort‑ oder Substring‑Modus)

    TTS
    Engine (Piper/pyttsx3), Stimme, Geschwindigkeit, Satzpause

    Erweiterte Whisper‑Parameter
    best_of, patience, no_speech_threshold, log_prob_threshold, compression_ratio_threshold, condition_on_previous_text, suppress_tokens

    Zusammenfassung
    Temperatur und Modell für den Ollama‑Summarizer

Alle Einstellungen werden automatisch gespeichert und beim nächsten Start wiederhergestellt.
🐧 Plattform‑Hinweise

    Linux – Das Skript wurde primär unter Arch Linux entwickelt und getestet.
    Der integrierte LinuxPerformanceOptimizer (benötigt psutil) reduziert bei Bedarf die GUI‑Last und optimiert Thread‑Prioritäten sowie Dateideskriptor‑Limits.

    Windows – Volle Unterstützung, inklusive UTF‑8‑Konsolen‑Setup, CREATE_NO_WINDOW‑Flags für Subprozesse und automatischer Wiederherstellung der Konsolen‑Codepage.

    macOS – Apple Silicon (M1/M2) wird via MPS‑Backend erkannt und genutzt; Intel‑Macs verwenden CPU oder CUDA (sofern verfügbar).

🆘 Fehlerbehebung

Problem	Lösung

ffmpeg not found	Installiere ffmpeg (siehe Abhängigkeiten) und stelle sicher, dass es im PATH ist.

yt-dlp not found	Installiere yt-dlp: pip install yt-dlp (oder Systempaket).

Keine Audiowiedergabe bei YouTube‑Streams	YouTube ändert häufig seine Streaming‑Protokolle. Starte das Skript mit --debug, um die Extraktion zu verfolgen, und aktualisiere ggf. yt-dlp: pip install -U yt-dlp.

GUI startet nicht (Linux)	Fehlt tkinter: sudo apt install python3-tk (Debian/Ubuntu) oder sudo pacman -S tk (Arch).

GPU wird nicht erkannt	Prüfe die PyTorch‑Installation:

bash

python -c "import torch; print(torch.cuda.is_available())"

Falls False, installiere die passende torch‑Version für deine CUDA/ROCm/MPS. |
| Übersetzung funktioniert nicht | Stelle sicher, dass deep-translator oder argos-translate installiert ist. Bei Ollama prüfe, ob der Server läuft (ollama serve). |
| Fehler beim Export von DOCX | Installiere python-docx: pip install python-docx (Fallback auf TXT). |

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

und allen anderen Open‑Source‑Projekten, die dies möglich machen.

Viel Spaß mit Dragon Whisperer! Bei Fragen, Ideen oder Problemen – öffne einfach ein Issue. 🐉
