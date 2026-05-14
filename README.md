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


🪟 Windows‑Einsteiger‑Anleitung (Schritt für Schritt)

Diese Anleitung ist für absolute Anfänger geschrieben, die Python nicht kennen. Du musst keine Vorkenntnisse haben – jede Anweisung ist als genauer Befehl formuliert.
📌 Voraussetzungen

    Windows 10 oder 11 (64‑Bit)

    Admin‑Rechte (für die Python‑Installation kurzzeitig nötig)

1️⃣ Python installieren

    Lade Python herunter
    Gehe auf python.org/downloads und klicke auf den gelben Button „Download Python 3.13.2“ (oder eine neuere Version).

    Starte die Installation
    Führe die heruntergeladene Datei aus.
    → Wichtig: Setze unten im Installationsfenster unbedingt den Haken bei
    ✅ Add Python to PATH
    → Klicke dann auf „Install Now“.

    Prüfe die Installation
    Drücke Win + R, tippe cmd ein und drücke Enter.
    Gib im schwarzen Fenster ein:
    cmd

    python --version

    Zeigt es Python 3.13.2 (oder ähnlich) an, ist alles richtig.

2️⃣ Projekt‑Ordner vorbereiten

    Erstelle einen Ordner (z. B. auf dem Desktop oder in Dokumente).
    Nenne ihn z. B. DragonWhisperer.
    Wichtig: Verwende keine Leerzeichen im Pfad, sonst klappt die Verknüpfung später nicht.

    Öffne die Eingabeaufforderung in diesem Ordner
    Gehe in den neuen Ordner, klicke in die Adressleiste, lösche den Inhalt, tippe cmd ein und drücke Enter.
    (Es öffnet sich ein schwarzes Fenster, das bereits auf diesen Ordner zeigt.)

3️⃣ Virtuelle Umgebung (venv) erstellen

Im geöffneten Konsolenfenster (dort, wo du vorher cmd eingegeben hast) tippe:
cmd

python -m venv venv

Nach einigen Sekunden erscheint ein Unterordner venv. Das ist die isolierte Python‑Umgebung.
4️⃣ Skript herunterladen (weil es zu groß für die GitHub‑Vorschau ist)

Warum geht nicht einfach „Download“?
Die Datei Dragon_Whisperer_Full.py ist so groß, dass GitHub sie nicht direkt als Download anbietet – du musst den RAW‑Inhalt kopieren.

    Öffne die Datei auf GitHub
    Gehe zu:
    https://github.com/xecuterdiablo/DragonWhisperer/blob/main/Dragon_Whisperer_Full.py

    Klicke auf „RAW“
    (Das ist ein Button rechts oben über dem Code).
    Es öffnet sich eine neue Seite mit nur dem Code (ohne GitHub‑Menü).

    Kopiere den gesamten Code
    Drücke Strg + A (alles markieren), dann Strg + C (kopieren).

    Erstelle die Datei auf deinem Rechner

        Öffne den Editor (z. B. Editor – das reicht, besser ist Notepad++ oder VS Code).

        Füge den kopierten Code ein (Strg + V).

        Achte auf die Kodierung: Speichere die Datei als UTF‑8 (ohne BOM).

            In Notepad++: Menü Kodierung → UTF‑8 ohne BOM.

            Im normalen Windows‑Editor: Wähle beim Speichern als Dateityp „Alle Dateien“ und trage die Endung .py an. Der Editor speichert standardmäßig UTF‑8 – das ist in Ordnung.

        Speichere die Datei unter Dragon_Whisperer_Full.py in deinem Projektordner (DragonWhisperer).

        Wichtig: Die Datei muss genau diesen Namen haben und im selben Ordner wie venv liegen.

    ✅ Tipp: Falls du unsicher bist, lade dir Notepad++ herunter – es ist kostenlos und zeigt das Encoding an.

5️⃣ Abhängigkeiten installieren

Im Konsolenfenster (immer noch im Projektordner) aktivierst du zuerst die virtuelle Umgebung:
cmd

venv\Scripts\activate

Dan siehst du (venv) am Anfang der Eingabezeile.

Jetzt installiere die benötigten Pakete:
cmd

pip install faster-whisper torch numpy scipy deep-translator psutil requests

Für alle optionalen Funktionen (TTS, Rauschunterdrückung, DOCX‑Export):
cmd

pip install argostranslate pyttsx3 noisereduce rapidfuzz python-docx dimits langdetect pathvalidate pynvml

(Das kann einige Minuten dauern – Geduld.)
6️⃣ Skript starten

Stelle sicher, dass du immer noch die (venv) aktiv hast. Dann tippe:
cmd

python Dragon_Whisperer_Full.py

Das GUI‑Fenster öffnet sich. Herzlichen Glückwunsch – du hast Dragon Whisperer erfolgreich gestartet! 🎉

Zum Beenden des Skripts schließe einfach das Fenster oder drücke Strg + C in der Konsole.
7️⃣ (Optional) Desktop‑Verknüpfung für schnellen Start

Damit du nicht jedes Mal die Konsole öffnen und venv\Scripts\activate eingeben musst, kannst du eine direkte Verknüpfung erstellen.

    Rechtsklick auf den Desktop → Neu → Verknüpfung.

    Als Pfad gibst du folgenden Befehl ein (passe den Pfad zu deinem Projektordner an!):
    cmd

    C:\Windows\System32\cmd.exe /k "cd /d C:\Users\DEIN_BENUTZERNAME\DragonWhisperer && venv\Scripts\python.exe Dragon_Whisperer_Full.py"

    Erklärung:

        cd /d C:\... wechselt in deinen Projektordner.

        && führt den nächsten Befehl nur aus, wenn der erste geklappt hat.

        venv\Scripts\python.exe startet Python direkt aus der virtuellen Umgebung (aktiviert sie automatisch).

        Dragon_Whisperer_Full.py ist dein Skript.

        /k bewirkt, dass das Konsolenfenster nach dem Programmende offen bleibt (du siehst eventuelle Fehlermeldungen).

    Klicke auf „Weiter“, gib der Verknüpfung einen Namen (z. B. Dragon Whisperer) und dann auf „Fertig“.

    Doppelklick auf die Verknüpfung – das Skript startet sofort.

    ⚠️ Wichtig: Ersetze DEIN_BENUTZERNAME durch deinen echten Windows‑Benutzernamen und den Pfad, wo dein Projektordner liegt.
    Beispiel: C:\Users\Xeqtr\DragonWhisperer

❗ Häufige Probleme & Lösungen
Problem	Lösung
python wird nicht erkannt	Du hast beim Python‑Installieren den Haken bei „Add to PATH“ vergessen. Deinstalliere Python und installiere es erneut – diesmal mit Haken.
Fehler „No module named …“	Du hast vergessen, die Pakete zu installieren. Führe Schritt 5 erneut aus.
Der RAW‑Code ist sehr lang, Kopieren dauert	Einfach Strg+A, Strg+C – das funktioniert auch bei langen Texten.
Die Desktop‑Verknüpfung startet nicht	Prüfe den Pfad: Sind alle Leerzeichen richtig? Der gesamte Befehl muss in einer Zeile stehen.
Das Skript startet, schließt aber sofort wieder	Öffne die Konsole manuell (siehe Schritt 6) und starte es dort – dann siehst du die Fehlermeldung.
✅ Zusammenfassung der wichtigsten Befehle (für Fortgeschrittene)
cmd

# 1. Projektordner anlegen und hineinwechseln
mkdir C:\Users\DEIN_NAME\DragonWhisperer
cd /d C:\Users\DEIN_NAME\DragonWhisperer

# 2. Virtuelle Umgebung erstellen
python -m venv venv

# 3. Aktivieren
venv\Scripts\activate

# 4. Abhängigkeiten installieren
pip install faster-whisper torch numpy scipy deep-translator psutil requests

# 5. Skript starten
python Dragon_Whisperer_Full.py

📌 Zusätzlicher Hinweis zur Textkodierung

Die Skriptdatei muss UTF‑8 kodiert sein. Der normale Windows‑Editor speichert standardmäßig UTF‑8 mit BOM (Byte Order Mark). Python kann das lesen, aber es kann zu Problemen mit Zeilenumbrüchen kommen. Empfehlung: Verwende einen besseren Editor wie Notepad++ und stelle dort explizit UTF‑8 ohne BOM ein.

    In Notepad++: Oben → Kodierung → UTF‑8 ohne BOM

    In VS Code: Unten rechts klickst du auf UTF-8 und wählst Save with Encoding → UTF-8

Nach dem Speichern ist das Skript startbereit.
