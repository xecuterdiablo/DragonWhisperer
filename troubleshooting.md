# Troubleshooting: "Thank you" statt deutscher Transkription (04/2026)

## Symptom
Bei YouTube-Videos liefert die Transkription nur den englischen Text "Thank you", obwohl das Video deutschen Ton enthält. Tritt nur auf aktuellen Distributionen (Arch Linux) mit neueren ffmpeg-/yt-dlp-Versionen auf.

## Ursache
Doppelte `-ss`-Angabe in `FFmpegManager._build_ffmpeg_command_optimized` (vor und nach `-i`) führte zu inkonsistentem Demuxing des WebM-Containers. Die PCM-Daten wurden korrumpiert.

## Lösung (implementiert in Commit `abc123`)
1. YouTube-Videos werden **nie** über den Pipe-Zweig verarbeitet.
2. Für YouTube wird eine **minimale** FFmpeg-Befehlszeile verwendet (nur `-i`, `-vn`, `-f s16le`, `-acodec pcm_s16le`, `-ar`, `-ac`).
3. `StreamManager._extract_youtube_audio_optimized` verwendet primär `bestaudio` ohne Cookies, mit Fallback auf `bestaudio[ext=m4a]/bestaudio/best`.

## Betroffene Klassen
- `FFmpegManager`
- `StreamManager`
- `AudioProcessor` (Debug-Ausgaben bereinigt)

## Testen
Das Skript wurde erfolgreich getestet auf:
- Arch Linux (ffmpeg 6.1, yt-dlp 2024.04)
- Linux Mint 21.3 (ffmpeg 4.4, yt-dlp 2023.10)

## Bei erneuten Problemen
- Mit `--debug=4` starten, um MD5-Hashes der gelesenen Blöcke zu sehen.
- Temporäre WAV-Dumps können in `AudioProcessor._process_audio_chunk_async` wieder aktiviert werden.
