#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐉 THE DRAGON WHISPERER - Decoding the Unspoken 🐉

BRIDGING WORLDS THROUGH SILENT UNDERSTANDING
Livestream Transcription & Real-Time Translation

🎯 MISSION: To connect souls across the divides of language
💫 PHILOSOPHY: That the most profound connections begin 
   with a single, understood word
🌍 VISION: A universe where no heart's message is 
   lost in translation
"""

import os, sys, time, json, logging, threading, queue, subprocess, tempfile
import platform, requests, csv, gc, re, warnings, select
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"⚠️  Torch nicht verfügbar: {e}")

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError as e:
    FASTER_WHISPER_AVAILABLE = False
    print(f"⚠️  Faster-Whisper nicht verfügbar: {e}")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    NUMPY_AVAILABLE = False
    print(f"⚠️  Numpy nicht verfügbar: {e}")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError as e:
    TRANSLATOR_AVAILABLE = False
    print(f"⚠️  Deep-Translator nicht verfügbar: {e}")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError as e:
    PSUTIL_AVAILABLE = False
    print(f"⚠️  Psutil nicht verfügbar: {e}")

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog, Menu
    GUI_AVAILABLE = True
except ImportError as e:
    GUI_AVAILABLE = False
    print(f"⚠️  GUI nicht verfügbar: {e}")

SUPPORTED_LANGUAGES = {
    'de': 'Deutsch', 'en': 'Englisch', 'fr': 'Französisch',
    'es': 'Spanisch', 'it': 'Italienisch', 'pt': 'Portugiesisch',
    'ru': 'Russisch', 'zh': 'Chinesisch', 'ja': 'Japanisch',
    'ko': 'Koreanisch', 'ar': 'Arabisch', 'nl': 'Niederländisch',
    'pl': 'Polnisch', 'tr': 'Türkisch', 'sv': 'Schwedisch',
    'da': 'Dänisch', 'fi': 'Finnisch', 'no': 'Norwegisch',
    'hi': 'Hindi', 'th': 'Thailändisch', 'vi': 'Vietnamesisch'
}


class ExportFormat(Enum):
    TXT = "txt"
    SRT = "srt"
    CSV = "csv"
    JSON = "json"


class ColorScheme:
    """Professionelle Farbpalette für optimalen Kontrast"""
    BG_PRIMARY = "#1e1e1e"
    BG_SECONDARY = "#2e2e2e"
    BG_TERTIARY = "#3e3e3e"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_ACCENT = "#58a6ff"
    TEXT_SUCCESS = "#3fb950"
    TEXT_WARNING = "#d29922"
    TEXT_ERROR = "#f85149"
    ACCENT_BLUE = "#1f6feb"
    ACCENT_GREEN = "#238636"
    ACCENT_ORANGE = "#db6d28"
    STATUS_HEALTHY = "#3fb950"
    STATUS_DEGRADED = "#d29922"
    STATUS_CRITICAL = "#f85149"


@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    language: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranslationResult:
    original: str
    translated: str
    source_lang: str
    target_lang: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemMetrics:
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    audio_buffer_health: float = 100.0
    processing_latency: float = 0.0
    chunks_processed: int = 0
    error_count: int = 0
    successful_transcriptions: int = 0
    successful_translations: int = 0
    silent_chunks_skipped: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdvancedExportManager:
    """Erweiterte Export-Funktionen für verschiedene Formate"""

    @staticmethod
    def export_srt(
            transcriptions: List[TranscriptionResult], filename: str) -> str:
        """Exportiert als SRT (Subtitles) Format mit robustem Error Handling"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for i, trans in enumerate(transcriptions, 1):
                    start_str = AdvancedExportManager._format_timestamp(
                        trans.start_time)
                    end_str = AdvancedExportManager._format_timestamp(
                        trans.end_time)

                    f.write(f"{i}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{trans.text}\n\n")

            return filename
        except Exception as e:
            raise Exception(f"SRT Export fehlgeschlagen: {e}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Formatiert Zeitstempel für SRT"""
        try:
            td = timedelta(seconds=seconds)
            hours = int(td.total_seconds() // 3600)
            minutes = int((td.total_seconds() % 3600) // 60)
            seconds = td.total_seconds() % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(
                '.', ',')
        except BaseException:
            return "00:00:00,000"

    @staticmethod
    def export_txt(
            transcriptions: List[TranscriptionResult], filename: str) -> str:
        """Exportiert als reinen Text mit Zeitstempeln"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for trans in transcriptions:
                    timestamp = datetime.fromtimestamp(
                        trans.start_time).strftime("%H:%M:%S")
                    f.write(f"[{timestamp}] {trans.text}\n")

            return filename
        except Exception as e:
            raise Exception(f"TXT Export fehlgeschlagen: {e}")

    @staticmethod
    def export_csv(
            transcriptions: List[TranscriptionResult], filename: str) -> str:
        """Exportiert als CSV für Analysen"""
        try:
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Start_Time', 'End_Time',
                                'Text', 'Language', 'Confidence', 'Speaker'])

                for trans in transcriptions:
                    writer.writerow([
                        trans.start_time,
                        trans.end_time,
                        trans.text,
                        trans.language,
                        trans.confidence,
                        trans.speaker or 'Unknown'
                    ])

            return filename
        except Exception as e:
            raise Exception(f"CSV Export fehlgeschlagen: {e}")

    @staticmethod
    def export_json(
            transcriptions: List[TranscriptionResult], filename: str) -> str:
        """Exportiert als strukturiertes JSON"""
        try:
            data = {
                'export_time': datetime.now().isoformat(),
                'total_segments': len(transcriptions),
                'segments': [trans.to_dict() for trans in transcriptions]
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return filename
        except Exception as e:
            raise Exception(f"JSON Export fehlgeschlagen: {e}")


class IntelligentSystemProfiler:
    """Intelligente Systemanalyse mit Hardware-Erkennung"""

    def __init__(self):
        self.profile = self.analyze_system()

    def analyze_system(self) -> Dict[str, Any]:
        """Analysiert Systemressourcen und Hardware mit robustem Error Handling"""
        profile = {
            'cpu_cores': self._get_cpu_count(),
            'ram_gb': self._get_ram_gb(),
            'has_gpu': False,
            'gpu_type': 'none',
            'recommended_mode': 'cpu',
            'recommended_model': 'small',
            'optimization_suggestions': [],
            'system_platform': platform.system(),
            'python_version': platform.python_version()
        }

        profile.update(self._detect_gpu())

        profile.update(self._get_model_recommendation(profile['ram_gb']))

        profile['optimization_suggestions'] = self._get_optimization_suggestions(
            profile)

        return profile

    def _get_cpu_count(self) -> int:
        """Sichere CPU-Count Ermittlung"""
        try:
            return psutil.cpu_count() if PSUTIL_AVAILABLE else os.cpu_count() or 4
        except BaseException:
            return 4

    def _get_ram_gb(self) -> float:
        """Sichere RAM Ermittlung"""
        try:
            if PSUTIL_AVAILABLE:
                return psutil.virtual_memory().total / (1024**3)
            return 8.0
        except BaseException:
            return 8.0

    def _detect_gpu(self) -> Dict[str, Any]:
        """GPU Detection mit multiplen Fallbacks"""
        gpu_info = {'has_gpu': False, 'gpu_type': 'none'}

        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_info.update({
                        'has_gpu': True,
                        'gpu_type': 'cuda',
                        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'unknown'
                    })
                    return gpu_info
            except Exception:
                pass

        return gpu_info

    def _get_model_recommendation(self, ram_gb: float) -> Dict[str, str]:
        """Modell-Empfehlung basierend auf RAM"""
        if ram_gb >= 16:
            return {'recommended_model': 'large-v2',
                    'recommended_mode': 'cuda' if self._detect_gpu()['has_gpu'] else 'cpu'}
        elif ram_gb >= 8:
            return {'recommended_model': 'medium', 'recommended_mode': 'cuda' if self._detect_gpu()[
                'has_gpu'] else 'cpu'}
        elif ram_gb >= 4:
            return {'recommended_model': 'small', 'recommended_mode': 'cpu'}
        else:
            return {'recommended_model': 'base', 'recommended_mode': 'cpu'}

    def _get_optimization_suggestions(
            self, profile: Dict[str, Any]) -> List[str]:
        """Generiert Optimierungsvorschläge"""
        suggestions = []

        if not profile['has_gpu']:
            suggestions.append(
                "🎯 Verwende CPU-optimierte Modelle für beste Performance")

        if profile['ram_gb'] < 8:
            suggestions.append(
                "💡 Weniger als 8GB RAM - verwende kleinere Modelle (base/small)")

        if profile['system_platform'] == "Windows":
            suggestions.append(
                "🖥️  Windows-System - stelle Administratorrechte für FFmpeg sicher")

        if profile['cpu_cores'] <= 4:
            suggestions.append(
                "⚡ Begrenzte CPU-Kerne - reduziere Parallelverarbeitung")

        return suggestions


class SystemDiagnostics:
    """Umfassende Systemdiagnose mit Auto-Recovery"""

    def __init__(self):
        self.health_checks = {}
        self.last_check = datetime.now()

    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Führt umfassende Systemdiagnose durch"""
        try:
            self.health_checks = {
                'timestamp': datetime.now().isoformat(),
                'python_dependencies': self.check_python_deps(),
                'system_tools': self.check_system_tools(),
                'network_connectivity': self.check_network(),
                'hardware_resources': self.check_hardware(),
                'ai_model_availability': self.check_ai_models()
            }
            self.last_check = datetime.now()
            return self.health_checks
        except Exception as e:
            logging.error(f"Diagnostic failed: {e}")
            return {'error': str(e)}

    def check_python_deps(self) -> Dict[str, Any]:
        """Überprüft Python-Abhängigkeiten mit Versionsinfo"""
        deps = {
            'torch': {'available': TORCH_AVAILABLE, 'version': 'unknown'},
            'faster_whisper': {'available': FASTER_WHISPER_AVAILABLE, 'version': 'unknown'},
            'numpy': {'available': NUMPY_AVAILABLE, 'version': 'unknown'},
            'deep_translator': {'available': TRANSLATOR_AVAILABLE, 'version': 'unknown'},
            'psutil': {'available': PSUTIL_AVAILABLE, 'version': 'unknown'},
            'gui': {'available': GUI_AVAILABLE, 'version': 'unknown'}
        }

        try:
            if TORCH_AVAILABLE:
                deps['torch']['version'] = torch.__version__
            if NUMPY_AVAILABLE:
                deps['numpy']['version'] = np.__version__
            if PSUTIL_AVAILABLE:
                deps['psutil']['version'] = psutil.__version__
        except Exception:
            pass

        return deps

    def check_system_tools(self) -> Dict[str, Any]:
        """Überprüft System-Tools mit Pfad-Erkennung"""
        tools = ['ffmpeg', 'yt-dlp']
        results = {}

        for tool in tools:
            tool_info = {'available': False, 'path': None}
            try:
                if platform.system() == "Windows":
                    check_cmd = ['where', tool]
                else:
                    check_cmd = ['which', tool]

                result = subprocess.run(
                    check_cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    tool_info['available'] = True
                    tool_info['path'] = result.stdout.strip().split('\n')[0]

            except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                tool_info['error'] = str(e)

            results[tool] = tool_info

        return results

    def check_network(self) -> Dict[str, Any]:
        """Überprüft Netzwerkverbindungen mit Timeouts"""
        endpoints = {
            'youtube': 'https://www.youtube.com',
            'google_translate': 'https://translate.google.com',
            'github': 'https://github.com'
        }

        results = {}
        for name, url in endpoints.items():
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = time.time() - start_time

                results[name] = {
                    'available': response.status_code == 200,
                    'response_time': round(response_time, 2),
                    'status_code': response.status_code
                }
            except Exception as e:
                results[name] = {
                    'available': False,
                    'error': str(e)
                }
        return results

    def check_hardware(self) -> Dict[str, Any]:
        """Überprüft Hardware-Ressourcen"""
        return IntelligentSystemProfiler().analyze_system()

    def check_ai_models(self) -> Dict[str, Any]:
        """Überprüft AI-Model Verfügbarkeit"""
        models = {
            'faster_whisper': {
                'available': FASTER_WHISPER_AVAILABLE,
                'models': ["tiny", "base", "small", "medium", "large-v2"] if FASTER_WHISPER_AVAILABLE else []
            },
            'translation': {
                'available': TRANSLATOR_AVAILABLE,
                'languages': list(SUPPORTED_LANGUAGES.keys()) if TRANSLATOR_AVAILABLE else []
            }
        }
        return models

    def get_health_score(self) -> float:
        """Berechnet Gesamt-Health-Score"""
        if not self.health_checks:
            return 0.0

        total_checks = 0
        passed_checks = 0

        for category, checks in self.health_checks.items():
            if isinstance(checks, dict):
                for check_name, check_result in checks.items():
                    if isinstance(check_result,
                                  dict) and 'available' in check_result:
                        total_checks += 1
                        if check_result['available']:
                            passed_checks += 1
                    elif isinstance(check_result, bool):
                        total_checks += 1
                        if check_result:
                            passed_checks += 1

        return passed_checks / total_checks if total_checks > 0 else 0.0


class PerformanceMonitor:
    """Erweitertes Performance Monitoring mit Alerting"""

    def __init__(self):
        self.performance_thresholds = {
            'max_cpu': 85.0,
            'max_memory': 80.0,
            'max_latency': 10.0,
            'min_buffer_health': 20.0
        }
        self.start_time = time.time()
        self.chunk_times = []
        self.performance_warnings = []
        self.performance_alerts = []

    def check_performance_health(self) -> List[str]:
        """Überprüft Performance-Grenzwerte mit Priorisierung"""
        warnings = []

        try:
            if PSUTIL_AVAILABLE:
                cpu = psutil.cpu_percent()
                if cpu > self.performance_thresholds['max_cpu']:
                    warnings.append(f"🚨 CPU Usage hoch: {cpu:.1f}%")

                memory = psutil.virtual_memory().percent
                if memory > self.performance_thresholds['max_memory']:
                    warnings.append(f"🚨 Memory Usage hoch: {memory:.1f}%")

                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
                if process_memory > 1024:  # 1GB
                    warnings.append(
                        f"💾 Prozess-Speicher hoch: {process_memory:.1f}MB")

        except Exception as e:
            logging.debug(f"Performance check error: {e}")

        return warnings

    def record_chunk_processing(self, processing_time: float):
        """Zeichnet Verarbeitungszeiten auf mit Statistiken"""
        self.chunk_times.append(processing_time)

        if len(self.chunk_times) > 100:
            self.chunk_times.pop(0)

        if len(self.chunk_times) >= 10:
            recent_avg = sum(self.chunk_times[-10:]) / 10
            overall_avg = sum(self.chunk_times) / len(self.chunk_times)

            if recent_avg > overall_avg * 1.5:  # 50% slower than average
                self.performance_warnings.append(
                    f"Performance degradation detected: {recent_avg:.2f}s vs {overall_avg:.2f}s"
                )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt detaillierte Performance-Statistiken zurück"""
        if not self.chunk_times:
            return {}

        total_time = time.time() - self.start_time
        chunks_per_minute = len(self.chunk_times) / \
            total_time * 60 if total_time > 0 else 0

        stats = {
            'avg_processing_time': sum(self.chunk_times) / len(self.chunk_times),
            'max_processing_time': max(self.chunk_times) if self.chunk_times else 0.0,
            'min_processing_time': min(self.chunk_times) if self.chunk_times else 0.0,
            'total_uptime': total_time,
            'chunks_per_minute': chunks_per_minute,
            'total_chunks_processed': len(self.chunk_times),
            'performance_warnings': self.performance_warnings[-5:],
            'current_load': len(self.chunk_times) / 100.0  # Normalized load
        }

        return stats


class AIConfigManager:
    """Robuste Konfigurationsverwaltung mit Auto-Recovery"""

    def __init__(self):
        self.config_path = Path.home() / ".config" / "dragon_whisperer" / "config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.default_config = {
            "transcription_model": "small",
            "translation_enabled": True,
            "target_language": "en",
            "chunk_duration": 5.0,
            "memory_cleanup_interval": 50,
            "auto_clear_interval": 1000,
            "enable_silence_detection": True,
            "translation_cache_size": 1000,
            "auto_recovery": True,
            "silence_threshold": 0.005,
            "enable_auto_scroll": True,
            "max_text_length": 50000,
            "export_format": "txt",
            "enable_speaker_detection": True,
            "enable_sentiment_analysis": False,
            "cloud_translation": False,
            "stream_type": "youtube",
            "ffmpeg_timeout": 30,
            "max_retry_attempts": 3
        }

        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Läd Konfiguration von Datei mit Auto-Recovery"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if self.config_path.exists():
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        user_config = json.load(f)

                    config = self.default_config.copy()
                    config.update(user_config)

                    config = self._validate_config(config)
                    return config
                else:
                    return self.default_config.copy()

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logging.warning(
                    f"Config load attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    logging.error(
                        "All config load attempts failed, using defaults")
                    return self.default_config.copy()
                time.sleep(0.1)

        return self.default_config.copy()

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert und korrigiert Konfigurationswerte"""
        valid_models = ["tiny", "base", "small", "medium", "large-v2"]
        if config.get("transcription_model") not in valid_models:
            config["transcription_model"] = "small"

        if config.get("target_language") not in SUPPORTED_LANGUAGES:
            config["target_language"] = "en"

        config["chunk_duration"] = max(
            1.0, min(30.0, float(config.get("chunk_duration", 5.0))))
        config["silence_threshold"] = max(0.001, min(
            1.0, float(config.get("silence_threshold", 0.02))))
        config["max_text_length"] = max(
            1000, int(config.get("max_text_length", 50000)))

        return config

    def save_config(self) -> bool:
        """Speichert Konfiguration in Datei mit Backup"""
        try:
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.json.backup')
                try:
                    self.config_path.rename(backup_path)
                except Exception:
                    pass

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            logging.error(f"Config save failed: {e}")
            return False


class UltimateAudioProcessor:
    """Optimierter Audio Processor mit Memory Management"""

    def __init__(self, config: AIConfigManager):
        self.config = config
        self.audio_queue = queue.Queue(maxsize=20)
        self.processing = False
        self._lock = threading.RLock()
        self._chunk_counter = 0
        self._last_cleanup = 0
        self.processing_thread = None

    def start_processing(self, callback: Callable):
        """Startet Audio-Verarbeitung mit Resource Management"""
        with self._lock:
            if self.processing:
                return
            self.processing = True

        def processing_loop():
            chunk_id = 0
            while self.processing:
                try:
                    audio_data = self.audio_queue.get(timeout=0.5)
                    chunk_id += 1
                    self._chunk_counter += 1

                    callback(audio_data, chunk_id)
                    self.audio_queue.task_done()

                    self._perform_memory_maintenance(chunk_id)

                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Audio processing error: {e}")

        self.processing_thread = threading.Thread(
            target=processing_loop,
            daemon=True,
            name="AudioProcessor"
        )
        self.processing_thread.start()

    def _perform_memory_maintenance(self, chunk_id: int):
        """Führt Memory Maintenance durch"""
        cleanup_interval = self.config.config.get(
            'memory_cleanup_interval', 50)

        if chunk_id % cleanup_interval == 0:
            gc.collect()
            self._last_cleanup = time.time()

        if PSUTIL_AVAILABLE:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80 and time.time() - self._last_cleanup > 30:
                gc.collect()
                self._last_cleanup = time.time()

    def stop_processing(self):
        """Stoppt Audio-Verarbeitung sauber"""
        with self._lock:
            self.processing = False

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break

        gc.collect()


class UltimateStreamManager:
    """Erweiterter Stream Manager mit Multi-Platform Support"""

    def __init__(self):
        self.supported_platforms = {
            'youtube': ['youtube.com', 'youtu.be'],
            'twitch': ['twitch.tv'],
            'facebook': ['facebook.com', 'fb.watch'],
            'rtmp': ['rtmp://', 'rtsp://'],
            'm3u8': ['.m3u8']
        }

        self.extraction_strategies = [
            ['yt-dlp', '-g', '-f', 'bestaudio[ext=m4a]/bestaudio', '--no-warnings'],
            ['yt-dlp', '-g', '-f', 'best', '--no-warnings'],
            ['yt-dlp', '-g', '--no-warnings'],
        ]

    def detect_stream_type(self, url: str) -> str:
        """Erkennt den Stream-Typ automatisch mit Fallback"""
        url_lower = url.lower()

        for platform_type, domains in self.supported_platforms.items():
            for domain in domains:
                if domain in url_lower:
                    return platform_type

        if (url.startswith(('file://', '/', './', '../')) or
                any(url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mkv'])):
            return 'local'

        return 'unknown'

    def extract_stream_url(self, url: str) -> Optional[str]:
        """Extrahiert Stream-URL mit erweiterten YouTube-Strategien"""
        logging.info(f"🎯 Starte Stream-Extraktion für: {url}")

        # NEUE: Spezielle YouTube-Strategien
        youtube_strategies = [
            # Strategie 1: Best Audio direkt
            ['yt-dlp', '-f', 'bestaudio[ext=m4a]', '--get-url', '--no-warnings',
             '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'],
            # Strategie 2: Audio only
            ['yt-dlp', '-f', 'bestaudio', '--get-url', '--no-warnings'],
            # Strategie 3: Fallback auf beste Qualität
            ['yt-dlp', '-f', 'best', '--get-url', '--no-warnings'],
        ]

        for i, strategy in enumerate(youtube_strategies):
            logging.info(f"🔧 YouTube Strategie {i + 1}: {' '.join(strategy[1:])}")

            try:
                cmd = strategy + [url]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False
                )

                if result.returncode == 0 and result.stdout.strip():
                    extracted_url = result.stdout.strip().split('\n')[0]
                    if extracted_url and extracted_url.startswith(('http')):
                        logging.info(f"✅ Erfolg mit YouTube Strategie {i + 1}")
                        return extracted_url

            except Exception as e:
                logging.warning(f"❌ YouTube Strategie {i + 1} fehlgeschlagen: {e}")

        logging.error("❌ Alle YouTube-Extraktionsversuche fehlgeschlagen")
        return None

    def _handle_local_file(self, file_path: str) -> str:
        """Behandelt lokale Dateien"""
        if file_path.startswith('file://'):
            file_path = file_path[7:]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        return file_path

    def _try_extraction_strategy(
            self, strategy: List[str], url: str, attempt: int) -> Optional[str]:
        """Versucht eine Extraktionsstrategie mit Error Handling"""
        try:
            cmd = strategy + [url]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                extracted_url = result.stdout.strip().split('\n')[0]
                if extracted_url and extracted_url.startswith(
                        ('http', 'rtmp')):
                    logging.info(f"✅ Erfolg mit Strategie {attempt + 1}")
                    return extracted_url
                else:
                    logging.warning(
                        f"⚠️ Ungültige URL von Strategie {attempt + 1}")
            else:
                logging.warning(f"❌ Strategie {attempt + 1} fehlgeschlagen")
                if result.stderr:
                    error_output = result.stderr.strip()[:200]
                    logging.debug(f"   Fehler: {error_output}")

        except subprocess.TimeoutExpired:
            logging.warning(f"⏰ Timeout bei Strategie {attempt + 1}")
        except Exception as e:
            logging.error(f"💥 Fehler bei Strategie {attempt + 1}: {e}")

        time.sleep(2)
        return None


class UltimateTranslationEngine:
    """Ultimativer Translation Engine mit Smart Caching"""

    def __init__(self, config: AIConfigManager):
        self.config = config
        self.translator = None
        self.translation_cache: Dict[str, TranslationResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache_lock = threading.Lock()
        self.setup_translator()

    def setup_translator(self):
        """Initialisiert den Übersetzer mit Fallback"""
        target_lang = self.config.config.get('target_language', 'en')
        try:
            if TRANSLATOR_AVAILABLE:
                self.translator = GoogleTranslator(
                    source='auto', target=target_lang)
                logging.info(
                    f"✅ Translator initialisiert für Sprache: {target_lang}")
            else:
                logging.warning("❌ Translator nicht verfügbar")
        except Exception as e:
            logging.error(f"❌ Translator Setup fehlgeschlagen: {e}")

    def translate_text(self, text: str, source_lang: str,
                       target_lang: str) -> TranslationResult:
        """Übersetzt Text mit intelligentem Caching"""
        if not text or not text.strip():
            return self._create_empty_translation(
                text, source_lang, target_lang)

        cache_key = self._generate_cache_key(text, source_lang, target_lang)

        with self._cache_lock:
            if cache_key in self.translation_cache:
                self.cache_hits += 1
                return self.translation_cache[cache_key]

            self.cache_misses += 1

        result = self._perform_translation(
            text, source_lang, target_lang, cache_key)
        return result

    def _generate_cache_key(
            self, text: str, source_lang: str, target_lang: str) -> str:
        """Generiert Cache-Key"""
        text_hash = str(hash(text.strip().lower()))
        return f"{source_lang}_{target_lang}_{text_hash}"

    def _perform_translation(self, text: str, source_lang: str,
                             target_lang: str, cache_key: str) -> TranslationResult:
        """Führt Übersetzung durch mit Error Handling"""
        try:
            if self.translator and text.strip():
                translated = self.translator.translate(text)

                result = TranslationResult(
                    original=text,
                    translated=translated,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    confidence=0.9
                )

                self._manage_cache(cache_key, result)
                return result

        except Exception as e:
            logging.error(f"❌ Übersetzung fehlgeschlagen: {e}")

        return self._create_empty_translation(text, source_lang, target_lang)

    def _manage_cache(self, cache_key: str, result: TranslationResult):
        """Managed Cache mit automatischer Bereinigung"""
        with self._cache_lock:
            cache_size = self.config.config.get('translation_cache_size', 1000)

            if len(self.translation_cache) >= cache_size:
                if self.translation_cache:
                    oldest_key = next(iter(self.translation_cache))
                    del self.translation_cache[oldest_key]

            self.translation_cache[cache_key] = result

    def _create_empty_translation(
            self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Erstellt leere Translation als Fallback"""
        return TranslationResult(
            original=text,
            translated=text,
            source_lang=source_lang,
            target_lang=target_lang,
            confidence=0.0
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Gibt detaillierte Cache-Statistiken zurück"""
        total_requests = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self.translation_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_ratio': hit_ratio,
            'total_requests': total_requests,
            'memory_usage_estimate': len(str(self.translation_cache)) / 1024
        }

    def clear_cache(self):
        """Leert den Translation Cache"""
        with self._cache_lock:
            self.translation_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0


class AdvancedAnalyticsEngine:
    """Erweiterte Analyse-Funktionen mit ML-ready Architecture"""

    def __init__(self):
        self.sentiment_cache = {}
        self.topic_cache = {}

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Einfache Sentiment-Analyse (kann mit ML-Modellen erweitert werden)"""
        if not text or len(text.strip()) < 3:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        text_hash = hash(text.lower())
        if text_hash in self.sentiment_cache:
            return self.sentiment_cache[text_hash]

        positive_words = {
            'de': ['gut', 'great', 'excellent', 'awesome', 'fantastic', 'love', 'wonderful', 'perfekt', 'super'],
            'en': ['good', 'great', 'excellent', 'awesome', 'fantastic', 'love', 'wonderful', 'perfect', 'super']
        }

        negative_words = {
            'de': ['schlecht', 'bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'schrecklich'],
            'en': ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'terrible']
        }

        text_lower = text.lower()

        language = 'en' if any(word in text_lower for word in [
                               'the', 'and', 'is', 'in', 'to']) else 'de'

        positive_count = sum(1 for word in positive_words.get(
            language, []) if word in text_lower)
        negative_count = sum(1 for word in negative_words.get(
            language, []) if word in text_lower)
        total_words = len(text.split())

        if total_words == 0:
            result = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        else:
            positive_score = positive_count / total_words
            negative_score = negative_count / total_words
            neutral_score = max(0.0, 1.0 - positive_score - negative_score)

            result = {
                'positive': min(1.0, positive_score),
                'negative': min(1.0, negative_score),
                'neutral': neutral_score
            }

        if total_words > 3:
            self.sentiment_cache[text_hash] = result

        return result

    def detect_topics(self, text: str) -> List[str]:
        """Einfache Themen-Erkennung (kann erweitert werden)"""
        if not text:
            return []

        text_hash = hash(text.lower())
        if text_hash in self.topic_cache:
            return self.topic_cache[text_hash]

        topics = []
        text_lower = text.lower()

        topic_keywords = {
            'technologie': ['computer', 'software', 'hardware', 'programm', 'code', 'ai', 'ki', 'internet'],
            'sport': ['sport', 'spiel', 'mannschaft', 'training', 'wettkampf', 'fußball', 'tennis'],
            'politik': ['regierung', 'politik', 'wahl', 'gesetz', 'minister', 'präsident'],
            'wirtschaft': ['wirtschaft', 'geld', 'markt', 'unternehmen', 'investition', 'aktien'],
            'gesundheit': ['gesundheit', 'krank', 'arzt', 'medizin', 'krankenhaus', 'behandlung'],
            'bildung': ['schule', 'universität', 'lernen', 'bildung', 'student', 'lehrer']
        }

        for topic, keywords in topic_keywords.items():
            keyword_count = sum(
                1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 1:  # Mindestens ein Keyword gefunden
                topics.append(topic)

        result = topics[:3]

        if len(text.split()) > 5:
            self.topic_cache[text_hash] = result

        return result


class DragonWhispererTranslator:
    """
    🐉 Dragon Whisperer Translator - Enterprise Grade
    VOLLSTÄNDIG REPARIERT mit robustem Error Handling und Performance Optimierungen
    """

    def __init__(self):
        self.config = AIConfigManager()
        self.profiler = IntelligentSystemProfiler()
        self.diagnostics = SystemDiagnostics()
        self.performance_monitor = PerformanceMonitor()
        self.export_manager = AdvancedExportManager()
        self.analytics_engine = AdvancedAnalyticsEngine()

        self.audio_processor = UltimateAudioProcessor(self.config)
        self.stream_manager = UltimateStreamManager()
        self.translation_engine = UltimateTranslationEngine(self.config)

        self.whisper_model = None
        self.is_running = False
        self.current_session: Optional[Dict[str, Any]] = None
        self.metrics = SystemMetrics()
        self._lock = threading.RLock()
        self._callback_lock = threading.Lock()
        self.start_time = time.time()
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.last_activity = time.time()
        self._shutdown_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None

        self.transcription_history: List[TranscriptionResult] = []
        self.speaker_profiles: Dict[str, int] = {}
        self.session_analytics: Dict[str, Any] = {
            'start_time': datetime.now().isoformat(),
            'total_words': 0,
            'sentiment_trend': [],
            'detected_topics': set(),
            'languages_detected': set()
        }

        self.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="DragonWorker")

        self._initialize_components()

    def get_available_models(self) -> List[str]:
        """Gibt verfügbare Whisper-Modelle zurück"""
        if FASTER_WHISPER_AVAILABLE:
            return ["tiny", "base", "small", "medium", "large-v2"]
        return []

    def start_live_translation_advanced(self, source_language=None, target_language="en", **kwargs):
        """Startet Live-Übersetzung mit erweiterten Parametern"""
        try:
            with self._lock:
                if self.is_running:
                    logging.warning("⚠️ Live-Übersetzung läuft bereits")
                    return False

                self.is_running = True
                self.current_session = {
                    'start_time': datetime.now(),
                    'target_language': target_language,
                    'source_language': source_language,
                    'total_chunks': 0,
                    'additional_params': kwargs
                }

                # Starte Stream-Thread
                self._stream_thread = threading.Thread(
                    target=self._stream_processing_loop,
                    daemon=True,
                    name="StreamProcessor"
                )
                self._stream_thread.start()

                logging.info(f"🎯 Live-Übersetzung gestartet: {source_language} -> {target_language}")
                return True

        except Exception as e:
            logging.error(f"❌ Fehler beim Starten der Live-Übersetzung: {e}")
            self.is_running = False
            return False

    def stop_live_translation(self):
        """Stoppt Live-Übersetzung"""
        try:
            with self._lock:
                if not self.is_running:
                    return True

                self.is_running = False
                self._shutdown_event.set()

                # Warte auf Thread-Ende
                if self._stream_thread and self._stream_thread.is_alive():
                    self._stream_thread.join(timeout=5.0)

                # Beende FFmpeg Prozess
                if self.ffmpeg_process:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process = None

                logging.info("⏹️ Live-Übersetzung gestoppt")
                return True

        except Exception as e:
            logging.error(f"❌ Fehler beim Stoppen der Live-Übersetzung: {e}")
            return False

    def _stream_processing_loop(self):
        """Haupt-Verarbeitungsschleife für Live-Stream"""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Hier kommt deine Stream-Verarbeitungslogik
                time.sleep(0.1)  # Platzhalter

            except Exception as e:
                logging.error(f"❌ Fehler in Stream-Schleife: {e}")
                time.sleep(1.0)

    def get_transcription_history(self) -> List[TranscriptionResult]:
        """Gibt Transkriptions-Historie zurück"""
        with self._lock:
            return self.transcription_history.copy()

    def clear_transcription_history(self):
        """Löscht Transkriptions-Historie"""
        with self._lock:
            self.transcription_history.clear()
            logging.info("🗑️ Transkriptions-Historie gelöscht")

    def get_system_metrics(self) -> Dict[str, Any]:
        """Gibt System-Metriken zurück"""
        return {
            'is_running': self.is_running,
            'session_duration': time.time() - self.start_time,
            'transcription_count': len(self.transcription_history),
            'last_activity': self.last_activity,
            'speaker_count': len(self.speaker_profiles)
        }

    def _initialize_components(self):
        """Initialisiert Komponenten mit Error Handling"""
        try:
            self.diagnostics.run_full_diagnostic()
            logging.info("✅ Translator erfolgreich initialisiert")
        except Exception as e:
            logging.error(f"❌ Initialisierung fehlgeschlagen: {e}")

    def safe_callback(self, callback: Optional[Callable], *args):
        """Thread-sichere Callback-Ausführung mit Robustem Error Handling"""
        with self._callback_lock:
            try:
                if callback and callable(callback):
                    # DEBUG
                    print(
                        f"🔔 CALLBACK: {callback.__name__ if hasattr(callback, '__name__') else 'unknown'} mit {len(args)} Argumenten")
                    callback(*args)
                    print("✅ Callback erfolgreich ausgeführt")  # DEBUG
                else:
                    print(f"❌ Callback nicht callable: {callback}")
            except Exception as e:
                logging.error(f"❌ Callback error: {e}")
                print(f"💥 Callback Exception: {e}")  # DEBUG

    def run_health_check(self) -> str:
        """Führt Gesundheits-Check durch mit erweitertem Monitoring"""
        try:
            if not self.diagnostics.health_checks:
                self.diagnostics.health_checks = self.diagnostics.run_full_diagnostic()

            health_score = self.diagnostics.get_health_score()

            if health_score >= 0.9:
                status = "healthy"
            elif health_score >= 0.7:
                status = "degraded"
            elif health_score >= 0.5:
                status = "warning"
            else:
                status = "critical"

            performance_warnings = self.performance_monitor.check_performance_health()
            if performance_warnings and status == "healthy":
                status = "degraded"

            return status

        except Exception as e:
            logging.error(f"❌ Health check failed: {e}")
            return "error"

    def get_detailed_report(self) -> Dict[str, Any]:
        """Generiert umfassenden Diagnostic Report mit erweiterten Metriken"""
        try:
            current_metrics = self.metrics.to_dict()
            current_metrics.update({
                'uptime_seconds': time.time() - self.start_time,
                'is_running': self.is_running,
                'audio_queue_size': self.audio_processor.audio_queue.qsize(),
                'transcription_history_size': len(self.transcription_history)
            })

            performance_stats = self.performance_monitor.get_performance_stats()

            cache_stats = self.translation_engine.get_cache_stats()

            session_analytics = self.session_analytics.copy()
            session_analytics['detected_topics'] = list(
                session_analytics['detected_topics'])
            session_analytics['languages_detected'] = list(
                session_analytics['languages_detected'])

            report = {
                'timestamp': datetime.now().isoformat(),
                'health_status': self.run_health_check(),
                'health_score': self.diagnostics.get_health_score(),
                'system_profile': self.profiler.profile,
                'diagnostics': self.diagnostics.health_checks,
                'current_metrics': current_metrics,
                'performance_stats': performance_stats,
                'session_stats': self.current_session,
                'cache_stats': cache_stats,
                'analytics': session_analytics,
                'speaker_stats': self.speaker_profiles,
                'configuration': {
                    'model': self.config.config.get('transcription_model'),
                    'target_language': self.config.config.get('target_language'),
                    'translation_enabled': self.config.config.get('translation_enabled')
                }
            }

            return report

        except Exception as e:
            logging.error(f"❌ Report generation failed: {e}")
            return {'error': f"Report generation failed: {e}"}

    def initialize_ai_models(self) -> bool:
        """Initialisiert AI-Modelle mit adaptiver Auswahl und Robustem Error Handling"""
        if not FASTER_WHISPER_AVAILABLE:
            logging.error("❌ Faster-Whisper nicht verfügbar")
            return False

        try:
            recommended_model = self.profiler.profile.get(
                'recommended_model', 'small')
            model_size = self.config.config.get(
                'transcription_model', recommended_model)

            device = "cuda" if self.profiler.profile['has_gpu'] else "cpu"
            compute_type = "int8" if device == "cpu" else "float16"

            logging.info(
                f"🚀 Initialisiere Faster-Whisper: {model_size} auf {device} ({compute_type})")

            download_root = str(Path.home() / ".cache" / "whisper")
            Path(download_root).mkdir(parents=True, exist_ok=True)

            self.whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
                num_workers=min(2, self.profiler.profile['cpu_cores'] // 2)
            )

            logging.info(f"✅ Faster-Whisper erfolgreich geladen: {model_size}")
            return True

        except Exception as e:
            logging.error(
                f"❌ Faster-Whisper Initialisierung fehlgeschlagen: {e}")

            fallback_models = ['small', 'base', 'tiny']
            current_model = self.config.config.get(
                'transcription_model', 'small')

            for fallback_model in fallback_models:
                if fallback_model != current_model:
                    logging.info(
                        f"🔄 Versuche Fallback auf {fallback_model} Modell...")
                    self.config.config['transcription_model'] = fallback_model
                    if self.initialize_ai_models():
                        return True

            logging.error("❌ Alle Fallback-Modelle fehlgeschlagen")
            return False

    def start_live_translation(self, url: str, output_callbacks: Dict[str, Callable]) -> bool:
        """Startet Live-Translation - VOLLSTÄNDIG REPARIERT"""
        try:
            with self._lock:
                if self.is_running:
                    logging.warning("⚠️ Translation läuft bereits")
                    self.safe_callback(output_callbacks.get('error'),
                                       "Translation läuft bereits")
                    return False

                if not self.initialize_ai_models():
                    self.safe_callback(output_callbacks.get('error'),
                                       "AI-Modelle konnten nicht geladen werden")
                    return False

                self.is_running = True
                self._shutdown_event.clear()
                self.current_session = {
                    'start_time': datetime.now().isoformat(),
                    'stream_url': url,
                    'stream_type': self.stream_manager.detect_stream_type(url),
                    'chunks_processed': 0,
                    'status': 'starting'
                }
                self.start_time = time.time()
                self.last_activity = time.time()

            # ⬇️ VERBESSERTE CALLBACK-SICHERSTELLUNG:
            safe_callbacks = output_callbacks.copy()

            def dummy_callback(*args, **kwargs):
                pass

            required_callbacks = ['transcription', 'translation', 'error', 'info', 'clear_text']
            for callback_name in required_callbacks:
                if callback_name not in safe_callbacks or safe_callbacks[callback_name] is None:
                    safe_callbacks[callback_name] = dummy_callback
                    print(f"⚠️  Callback '{callback_name}' wurde mit Dummy ersetzt")

            def async_startup():
                try:
                    self.audio_processor.start_processing(
                        lambda audio, chunk_id: self.process_audio_chunk(
                            audio, chunk_id, safe_callbacks)
                    )

                    self._stream_reading_loop(url, safe_callbacks)

                except Exception as e:
                    logging.error(f"Process startup error: {e}")
                    self.safe_callback(safe_callbacks.get('error'),
                                       f"Start fehlgeschlagen: {e}")
                    self.stop()

            self._stream_thread = threading.Thread(
                target=async_startup, daemon=True, name="MainProcessor")
            self._stream_thread.start()

            threading.Thread(target=self._collect_metrics_loop,
                             daemon=True, name="MetricsCollector").start()

            if self.config.config.get('auto_recovery', True):
                threading.Thread(target=self._auto_recovery_loop,
                                 daemon=True, name="AutoRecovery").start()

            self.setup_memory_guard()
            self.current_session['status'] = 'running'
            logging.info("🎯 Live Translation erfolgreich gestartet!")

            threading.Timer(0.1, lambda: self.safe_callback(
                output_callbacks.get('info'), "Live Translation gestartet")).start()

            return True

        except Exception as e:
            logging.error(f"❌ Start live translation failed: {e}")
            self.safe_callback(output_callbacks.get('error'), f"Start fehlgeschlagen: {e}")
            return False

    def _stream_reading_loop(self, url: str, output_callbacks: Dict[str, Callable]):
        """Haupt-Stream-Leseschleife"""
        try:
            stream_type = self.stream_manager.detect_stream_type(url)
            self.safe_callback(output_callbacks.get('info'),
                               f"🔍 Stream-Typ erkannt: {stream_type}")

            if stream_type == 'local':
                self._process_local_file(url, output_callbacks)
                return

            if stream_type == 'youtube':
                self._process_youtube_direct(url, output_callbacks)
                return

            extracted_url = self.stream_manager.extract_stream_url(url)

            if not extracted_url:
                self.safe_callback(output_callbacks.get(
                    'error'), "Stream-URL konnte nicht extrahiert werden")
                self.stop()
                return

            logging.info(f"🎯 Extrahierte URL: {extracted_url[:100]}...")

            if '.m3u8' in extracted_url.lower():
                self._process_hls_stream(extracted_url, output_callbacks)
            else:
                self._process_regular_stream(extracted_url, output_callbacks)

        except Exception as e:
            logging.error(f"❌ Stream reading loop error: {e}")
            self.safe_callback(output_callbacks.get(
                'error'), f"Stream Fehler: {e}")
            self.stop()

    def _process_youtube_direct(self, youtube_url: str, output_callbacks: Dict[str, Callable]):
        """🔧 NEU: Direkte YouTube-Verarbeitung ohne HLS"""
        try:
            logging.info("🎯 Starte direkte YouTube-Audio-Extraktion...")

            ytdlp_cmd = [
                'yt-dlp',
                '-f', 'bestaudio[ext=m4a]/bestaudio/best',
                '--get-url',
                '--no-warnings',
                '--user-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                '--add-header', 'Accept: */*',
                '--add-header', 'Accept-Language: en-US,en;q=0.9',
                youtube_url
            ]

            result = subprocess.run(
                ytdlp_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0 or not result.stdout.strip():
                logging.error(f"❌ yt-dlp failed: {result.stderr}")
                extracted_url = self.stream_manager.extract_stream_url(youtube_url)
                if extracted_url:
                    self._process_hls_stream(extracted_url, output_callbacks)
                return

            audio_url = result.stdout.strip().split('\n')[0]
            logging.info(f"✅ Direkte Audio-URL erhalten: {audio_url[:100]}...")

            chunk_duration = self.config.config.get('chunk_duration', 5.0)
            chunk_bytes = int(16000 * 2 * chunk_duration)

            ffmpeg_cmd = [
                'ffmpeg',
                '-i', audio_url,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-loglevel', 'info',
                '-'
            ]

            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=chunk_bytes
            )

            stderr_thread = threading.Thread(
                target=self._monitor_ffmpeg_stderr,
                args=(self.ffmpeg_process.stderr, output_callbacks),
                daemon=True
            )
            stderr_thread.start()

            chunk_counter = 0
            last_successful_read = time.time()

            while self.is_running and self.ffmpeg_process.poll() is None:
                try:
                    if time.time() - last_successful_read > 20.0:
                        logging.error("⏰ YouTube Direct Timeout")
                        break

                    ready_to_read, _, _ = select.select(
                        [self.ffmpeg_process.stdout], [], [], 1.0)

                    if ready_to_read:
                        audio_data = self.ffmpeg_process.stdout.read(chunk_bytes)

                        if audio_data and len(audio_data) > 0:
                            chunk_counter += 1
                            last_successful_read = time.time()

                            try:
                                self.audio_processor.audio_queue.put(audio_data, timeout=0.1)
                            except queue.Full:
                                continue

                            if chunk_counter == 1:
                                logging.info("✅ Erster YouTube Audio-Chunk empfangen!")
                                self.safe_callback(output_callbacks.get('info'),
                                                   "✅ YouTube Audio empfangen - Transkription startet...")

                        elif not audio_data:
                            time.sleep(0.1)
                    else:
                        if self.ffmpeg_process.poll() is not None:
                            break
                        time.sleep(0.1)

                except Exception as e:
                    logging.error(f"❌ YouTube direct read error: {e}")
                    time.sleep(0.5)

            logging.info(f"🎯 YouTube Direct beendet. Chunks: {chunk_counter}")

        except Exception as e:
            logging.error(f"❌ YouTube direct processing failed: {e}")
            extracted_url = self.stream_manager.extract_stream_url(youtube_url)
            if extracted_url:
                self._process_hls_stream(extracted_url, output_callbacks)

    def _monitor_ffmpeg_stderr(self, stderr_pipe, output_callbacks: Dict[str, Callable]):
        """🔧 NEU: Überwacht FFmpeg Stderr für Debugging"""
        try:
            while self.is_running and hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
                line = stderr_pipe.readline()
                if not line:
                    break

                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    if any(keyword in line_str.lower() for keyword in ['error', 'failed', 'invalid', 'missing']):
                        logging.error(f"🚨 FFmpeg Error: {line_str}")
                        self.safe_callback(output_callbacks.get('warning'), f"FFmpeg: {line_str[:100]}...")
                    elif 'audio:' in line_str.lower() and 'stream' in line_str.lower():
                        logging.info(f"🔊 FFmpeg Audio Info: {line_str}")
                    elif 'time=' in line_str.lower():
                        logging.debug(f"⏱️  FFmpeg Progress: {line_str}")

        except Exception as e:
            logging.debug(f"FFmpeg stderr monitor error: {e}")

    def _process_local_file(self, file_path: str, output_callbacks: Dict[str, Callable]):
        """🔧 REPARIERT: Verarbeitet lokale Audio/Video Dateien mit Progress Tracking"""
        if file_path.startswith('file://'):
            file_path = file_path[7:]

        if not os.path.exists(file_path):
            self.safe_callback(output_callbacks.get('error'), f"Datei nicht gefunden: {file_path}")
            self.stop()
            return

        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.safe_callback(output_callbacks.get('info'),
                           f"🎵 Verarbeite lokale Datei: {os.path.basename(file_path)} ({file_size / 1024 / 1024:.1f} MB)")

        chunk_duration = self.config.config.get('chunk_duration', 5.0)
        chunk_bytes = int(16000 * 2 * chunk_duration)

        ffmpeg_cmd = [
            'ffmpeg', '-i', file_path,
            '-f', 's16le', '-ar', '16000', '-ac', '1',
            '-loglevel', 'quiet', '-'
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=chunk_bytes)

            chunk_counter = 0
            last_successful_read = time.time()
            file_timeout = 60.0

            while self.is_running and self.ffmpeg_process.poll() is None:
                if time.time() - last_successful_read > file_timeout:
                    logging.warning("⏰ Datei-Verarbeitungs-Timeout")
                    break

                audio_data = self.ffmpeg_process.stdout.read(chunk_bytes)
                if audio_data:
                    chunk_counter += 1
                    self.last_activity = time.time()
                    last_successful_read = time.time()

                    try:
                        self.audio_processor.audio_queue.put(audio_data, timeout=0.1)
                    except queue.Full:
                        if chunk_counter % 10 == 0:
                            logging.warning("⚠️ Audio queue full, skipping chunk")
                        continue

                    if chunk_counter % 10 == 0:
                        self.safe_callback(output_callbacks.get('info'), f"📊 Verarbeitet: {chunk_counter} Chunks")
                else:
                    break

            self.safe_callback(output_callbacks.get('info'), "✅ Datei-Verarbeitung abgeschlossen")

        except Exception as e:
            self.safe_callback(output_callbacks.get('error'), f"❌ Datei-Verarbeitungsfehler: {e}")
        finally:
            self.stop()

    def _process_hls_stream(self, hls_url: str, output_callbacks: Dict[str, Callable]):
        """🔧 VOLLSTÄNDIG REPARIERT: HLS mit korrekten FFmpeg Parametern"""
        chunk_duration = self.config.config.get('chunk_duration', 5.0)
        chunk_bytes = int(16000 * 2 * chunk_duration)

        ffmpeg_cmd = [
            'ffmpeg',
            '-i', hls_url,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-fflags', '+nobuffer+flush_packets',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-max_delay', '1000000',
            '-loglevel', 'info',
            '-'
        ]

        logging.info("🎵 Starte HLS-Verarbeitung mit optimierten Parametern...")
        self.safe_callback(output_callbacks.get('info'), "🎵 Live-Stream erkannt - starte Verarbeitung...")

        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=chunk_bytes
            )

            chunk_counter = 0
            consecutive_empty_reads = 0
            max_empty_reads = 5
            stream_timeout = 15.0
            last_successful_read = time.time()

            stderr_monitor = threading.Thread(
                target=self._monitor_ffmpeg_stderr,
                args=(self.ffmpeg_process.stderr, output_callbacks),
                daemon=True
            )
            stderr_monitor.start()

            while self.is_running and self.ffmpeg_process.poll() is None:
                try:
                    if time.time() - last_successful_read > stream_timeout:
                        logging.error("⏰ Stream-Timeout - FFmpeg produziert keine Daten")
                        self.safe_callback(output_callbacks.get('error'),
                                           "Stream-Timeout - FFmpeg produziert keine Audio-Daten")
                        break

                    ready_to_read, _, _ = select.select(
                        [self.ffmpeg_process.stdout], [], [], 0.5)

                    if ready_to_read:
                        audio_data = self.ffmpeg_process.stdout.read(chunk_bytes)

                        if audio_data and len(audio_data) > 0:
                            chunk_counter += 1
                            consecutive_empty_reads = 0
                            self.last_activity = time.time()
                            last_successful_read = time.time()

                            try:
                                self.audio_processor.audio_queue.put(audio_data, timeout=0.1)
                            except queue.Full:
                                if chunk_counter % 10 == 0:
                                    logging.warning(f"⏩ Queue voll - überspringe Chunk {chunk_counter}")
                                continue

                            if chunk_counter == 1:
                                logging.info(f"✅ Erster Audio-Chunk empfangen! Länge: {len(audio_data)} bytes")
                                self.safe_callback(output_callbacks.get('info'),
                                                   "✅ Audio-Daten empfangen - Transkription läuft...")

                            if chunk_counter % 20 == 0:
                                queue_size = self.audio_processor.audio_queue.qsize()
                                logging.info(f"📊 Chunk {chunk_counter} verarbeitet | Queue: {queue_size}")

                        elif not audio_data:
                            consecutive_empty_reads += 1
                            if consecutive_empty_reads >= max_empty_reads:
                                logging.error("🔇 FFmpeg produziert leere Daten - Stream möglicherweise beendet")
                                break
                            time.sleep(0.1)
                    else:
                        if self.ffmpeg_process.poll() is not None:
                            logging.info("🔚 FFmpeg Prozess beendet")
                            break

                        time.sleep(0.1)
                        continue

                except Exception as e:
                    logging.error(f"❌ HLS Read Error: {e}")
                    time.sleep(0.5)
                    consecutive_empty_reads += 1

                    if consecutive_empty_reads > max_empty_reads * 2:
                        logging.error("🔇 Zu viele Fehler - beende Stream")
                        break

            logging.info(f"🎯 HLS-Verarbeitung beendet. Chunks verarbeitet: {chunk_counter}")

        except Exception as e:
            logging.error(f"❌ HLS Stream-Fehler: {e}")
            self.safe_callback(output_callbacks.get('error'), f"HLS Fehler: {e}")
        finally:
            self._safe_ffmpeg_shutdown()

    def _safe_ffmpeg_shutdown(self):
        """🔧 SICHERER FFMPEG SHUTDOWN"""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                self.ffmpeg_process.terminate()

                for _ in range(10):
                    if self.ffmpeg_process.poll() is not None:
                        break
                    time.sleep(0.2)

                if self.ffmpeg_process.poll() is None:
                    logging.warning("⚠️ FFmpeg reagiert nicht - forcing kill")
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait(timeout=2)

            except Exception as e:
                logging.error(f"❌ FFmpeg shutdown error: {e}")
            finally:
                self.ffmpeg_process = None

    def _process_regular_stream(self, stream_url: str, output_callbacks: Dict[str, Callable]):
        """🔧 REPARIERT: Reguläre Streams mit Timeout-Protection"""
        chunk_duration = self.config.config.get('chunk_duration', 5.0)
        chunk_bytes = int(16000 * 2 * chunk_duration)

        ffmpeg_cmd = [
            'ffmpeg',
            '-reconnect', '1',
            '-reconnect_at_eof', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
            '-i', stream_url,
            '-f', 's16le', '-ar', '16000', '-ac', '1',
            '-loglevel', 'warning', '-'
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=chunk_bytes)

            chunk_counter = 0
            last_successful_read = time.time()
            stream_timeout = 30.0  # 🔥 NEU: Stream-Timeout

            while self.is_running and self.ffmpeg_process.poll() is None:
                if time.time() - last_successful_read > stream_timeout:
                    logging.warning("⏰ Regulärer Stream-Timeout")
                    break

                ready_to_read, _, _ = select.select(
                    [self.ffmpeg_process.stdout], [], [], 1.0)

                if ready_to_read:
                    audio_data = self.ffmpeg_process.stdout.read(chunk_bytes)
                    if audio_data:
                        chunk_counter += 1
                        self.last_activity = time.time()
                        last_successful_read = time.time()
                        try:
                            self.audio_processor.audio_queue.put(audio_data, timeout=0.1)
                        except queue.Full:
                            if chunk_counter % 10 == 0:
                                logging.warning("⚠️ Audio queue full, skipping chunk")
                            continue
                    else:
                        time.sleep(0.1)
                else:
                    if self.ffmpeg_process.poll() is not None:
                        break
                    continue

        except Exception as e:
            self.safe_callback(output_callbacks.get('error'), f"❌ Stream-Fehler: {e}")
        finally:
            self.stop()

    def _is_silent_chunk(self, audio_data: bytes) -> bool:
        """Erkennt stille Audio-Chunks mit robustem Error Handling"""
        if not self.config.config.get('enable_silence_detection', True):
            return False

        if not NUMPY_AVAILABLE or not audio_data:
            return False

        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_np) == 0:
                return True

            rms = np.sqrt(np.mean(audio_np**2))
            silence_threshold = self.config.config.get('silence_threshold', 0.02) * 32768
            return rms < silence_threshold
        except Exception as e:
            logging.debug(f"Silence detection error: {e}")
            return False

    def process_audio_chunk(self, audio_data: bytes, chunk_id: int, output_callbacks: Dict[str, Callable]):
        """Verarbeitet Audio-Chunk mit erweiterten Analysen und Robustem Error Handling"""
        start_time = time.time()

        try:
            if self._is_silent_chunk(audio_data):
                self.metrics.silent_chunks_skipped += 1
                self.metrics.chunks_processed += 1

                if self.metrics.silent_chunks_skipped % 100 == 0:
                    self.safe_callback(output_callbacks.get('info'),
                                       f"🔇 {self.metrics.silent_chunks_skipped} stille Chunks übersprungen")
                return

            transcription = self.transcribe_audio(audio_data)

            if transcription and transcription.text.strip():
                if (transcription.confidence > 0.2 and
                    not self._contains_gibberish(transcription.text) and
                        len(transcription.text) > 2):

                    self.session_analytics['languages_detected'].add(transcription.language)

                    self._perform_advanced_analytics(transcription)

                    self.metrics.successful_transcriptions += 1
                    self.transcription_history.append(transcription)
                    self.safe_callback(output_callbacks.get('transcription'), transcription)

                    if self.config.config.get('translation_enabled', True):
                        translation = self.translation_engine.translate_text(
                            transcription.text,
                            transcription.language,
                            self.config.config.get('target_language', 'en')
                        )
                        if translation and translation.confidence > 0:
                            self.metrics.successful_translations += 1
                            self.safe_callback(output_callbacks.get('translation'), translation)

            self.metrics.chunks_processed += 1
            self.metrics.processing_latency = time.time() - start_time

            auto_clear = self.config.config.get('auto_clear_interval', 1000)
            if auto_clear > 0 and chunk_id % auto_clear == 0:
                self.safe_callback(output_callbacks.get('clear_text'))

        except Exception as e:
            self.metrics.error_count += 1
            logging.error(f"❌ Chunk {chunk_id} processing failed: {e}")
        finally:
            processing_time = time.time() - start_time
            self.performance_monitor.record_chunk_processing(processing_time)

    def _perform_advanced_analytics(self, transcription: TranscriptionResult):
        """Führt erweiterte Analysen durch"""
        if self.config.config.get('enable_sentiment_analysis', False):
            sentiment = self.analytics_engine.analyze_sentiment(transcription.text)
            self.session_analytics['sentiment_trend'].append({
                'timestamp': time.time(),
                'sentiment': sentiment,
                'text_sample': transcription.text[:50] + '...' if len(transcription.text) > 50 else transcription.text
            })

        topics = self.analytics_engine.detect_topics(transcription.text)
        for topic in topics:
            self.session_analytics['detected_topics'].add(topic)

        self.session_analytics['total_words'] += len(transcription.text.split())

        if self.config.config.get('enable_speaker_detection', False):
            speaker_id = self._detect_speaker(transcription.text)
            transcription.speaker = speaker_id
            self.speaker_profiles[speaker_id] = self.speaker_profiles.get(speaker_id, 0) + 1

    def _detect_speaker(self, text: str) -> str:
        """Vereinfachte Sprecher-Erkennung (kann mit ML erweitert werden)"""
        word_count = len(text.split())

        if word_count > 25:
            return "Speaker_Long"
        elif word_count > 15:
            return "Speaker_Medium"
        elif word_count > 5:
            return "Speaker_Short"
        else:
            return "Speaker_Brief"

    def _contains_gibberish(self, text: str) -> bool:
        """Erkennt sinnlose Wiederholungen und seltsame Zeichen"""
        if len(text) < 3:
            return True

        if len(set(text)) < 4 and len(text) > 15:
            return True

        weird_chars = ['༼', '༽', 'ව', 'ʕ', 'ʔ', '�', '⠀']
        if any(char in text for char in weird_chars):
            return True

        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:
            return True

        return False

    def transcribe_audio(self, audio_data: bytes) -> Optional[TranscriptionResult]:
        """Transkribiert Audio mit Faster-Whisper und Robustem Error Handling"""
        try:
            if not NUMPY_AVAILABLE or not audio_data:
                return None

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_np) < 1000:
                return None

            if self.whisper_model is not None:
                segments, info = self.whisper_model.transcribe(
                    audio_np,
                    beam_size=5,
                    best_of=2,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                text = " ".join(segment.text.strip() for segment in segments)
                language = info.language if hasattr(info, 'language') else 'unknown'
                confidence = getattr(info, 'language_probability', 0.5)

                if text.strip():
                    return TranscriptionResult(
                        text=text.strip(),
                        confidence=confidence,
                        language=language,
                        start_time=time.time() - self.config.config.get('chunk_duration', 5.0),
                        end_time=time.time()
                    )
        except Exception as e:
            logging.error(f"❌ Transcription failed: {e}")

        return None

    def _collect_metrics_loop(self):
        """Sammelt System-Metriken kontinuierlich"""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                if PSUTIL_AVAILABLE:
                    self.metrics.cpu_usage = psutil.cpu_percent()
                    self.metrics.memory_usage = psutil.virtual_memory().percent

                queue_size = self.audio_processor.audio_queue.qsize()
                max_size = self.audio_processor.audio_queue.maxsize
                self.metrics.audio_buffer_health = max(
                    0.0, 100.0 - (queue_size / max_size * 100)) if max_size > 0 else 100.0

                warnings = self.performance_monitor.check_performance_health()
                if warnings and self.metrics.chunks_processed % 50 == 0:
                    for warning in warnings[-3:]:
                        logging.warning(warning)

            except Exception as e:
                logging.debug(f"Metrics collection error: {e}")

            time.sleep(2)

    def _auto_recovery_loop(self):
        """Auto-Recovery System mit erweiterten Checks"""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                current_time = time.time()

                if (self.is_running and
                    current_time - self.last_activity > 120 and
                        self.audio_processor.audio_queue.qsize() == 0):

                    logging.warning("🔧 Auto-Recovery: System inaktiv, starte Recovery...")

                    if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                        try:
                            self.ffmpeg_process.terminate()
                            time.sleep(3)
                            if self.ffmpeg_process.poll() is None:
                                self.ffmpeg_process.kill()
                        except Exception as e:
                            logging.error(f"❌ FFmpeg termination failed: {e}")

                    self.last_activity = current_time

            except Exception as e:
                logging.error(f"❌ Auto-Recovery error: {e}")

            time.sleep(15)

    def export_transcriptions(self, format_type: ExportFormat, filename: str) -> str:
        """Exportiert Transkriptionen in verschiedenen Formaten"""
        if not self.transcription_history:
            raise Exception("Keine Transkriptionen zum Exportieren verfügbar")

        try:
            if format_type == ExportFormat.SRT:
                return self.export_manager.export_srt(self.transcription_history, filename)
            elif format_type == ExportFormat.TXT:
                return self.export_manager.export_txt(self.transcription_history, filename)
            elif format_type == ExportFormat.CSV:
                return self.export_manager.export_csv(self.transcription_history, filename)
            elif format_type == ExportFormat.JSON:
                return self.export_manager.export_json(self.transcription_history, filename)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            raise Exception(f"Export failed: {e}")

    def get_available_export_formats(self) -> List[Tuple[str, str]]:
        """Gibt verfügbare Export-Formate zurück"""
        return [
            ("Text File (.txt)", "*.txt"),
            ("Subtitle File (.srt)", "*.srt"),
            ("CSV File (.csv)", "*.csv"),
            ("JSON File (.json)", "*.json")
        ]

    def stop(self):
        """Stoppt alle Prozesse sauber mit erweitertem Cleanup - REPARIERT gegen doppelte Shutdowns"""
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False
            self._shutdown_event.set()

        logging.info("🛑 Starte sauberes Shutdown...")

        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logging.warning("⚠️ FFmpeg termination timeout, forcing kill")
                    self.ffmpeg_process.kill()
            except Exception as e:
                logging.error(f"❌ FFmpeg shutdown error: {e}")
            finally:
                self.ffmpeg_process = None

        self.audio_processor.stop_processing()

        self.thread_pool.shutdown(wait=False)

        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)

        if self.whisper_model is not None:
            try:
                del self.whisper_model
                self.whisper_model = None
            except Exception as e:
                logging.error(f"❌ Model cleanup error: {e}")

        gc.collect()

        if self.current_session:
            self.current_session['status'] = 'stopped'
            self.current_session['end_time'] = datetime.now().isoformat()

        logging.info("✅ Translation gestoppt und bereinigt")

    def setup_memory_guard(self):
        """🔧 VERBESSERT: Memory-Guard mit aggressiverem Cleanup"""
        def memory_guard_loop():
            while self.is_running and not self._shutdown_event.is_set():
                try:
                    time.sleep(60)

                    if PSUTIL_AVAILABLE:
                        process = psutil.Process()
                        process_memory = process.memory_info().rss / 1024 / 1024  # MB

                        if process_memory > 800:
                            logging.warning(f"🧹 Memory-Guard: {process_memory:.1f}MB - Starte Cleanup")

                            # Force garbage collection
                            gc.collect()

                            # Cache reduzieren
                            cache_size = len(self.translation_engine.translation_cache)
                            if cache_size > 200:
                                with self.translation_engine._cache_lock:
                                    keys = list(self.translation_engine.translation_cache.keys())
                                    remove_count = len(keys) // 2
                                    for key in keys[:remove_count]:
                                        del self.translation_engine.translation_cache[key]
                                    logging.info(f"✅ Cache von {cache_size} auf {len(self.translation_engine.translation_cache)} reduziert")

                            if hasattr(self, 'whisper_model') and self.whisper_model:
                                try:
                                    # if hasattr(self.whisper_model, 'model'):
                                    #    delattr(self.whisper_model, 'model')
                                    pass
                                except BaseException:
                                    pass

                except Exception as e:
                    logging.debug(f"Memory guard error: {e}")

        guard_thread = threading.Thread(target=memory_guard_loop, daemon=True, name="MemoryGuard")
        guard_thread.start()

    def export_diagnostic_report(self, filename: str) -> str:
        """Exportiert Diagnostic Report"""
        try:
            report = self.get_detailed_report()
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return filename
        except Exception as e:
            raise Exception(f"Diagnostic report export failed: {e}")

    def clear_translation_cache(self):
        """Leert den Translation Cache"""
        self.translation_engine.clear_cache()
        logging.info("✅ Translation Cache geleert")


class DragonWhispererGUI:
    """Ultimative GUI mit Enterprise Features und REPARIERTEM Layout"""

    def __init__(self):
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI nicht verfügbar - tkinter fehlt")

        self.root = tk.Tk()
        self.translator = DragonWhispererTranslator()
        self.is_translating = False
        self.setup_ultimate_gui()

    def setup_ultimate_gui(self):
        """Initialisiert die komplette GUI mit REPARIERTEM Design"""
        try:
            self.root.configure(bg=ColorScheme.BG_PRIMARY)
            self.root.title("🐉 Dragon Whisperer - LiveStream Transkribator")

            self.root.geometry("1000x700")
            self.root.minsize(900, 600)

            self.root.update_idletasks()
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = (screen_width - 1000) // 2
            y = (screen_height - 700) // 2
            self.root.geometry(f"1000x700+{x}+{y}")

            self.root.protocol("WM_DELETE_WINDOW", self.safe_exit)

            self.setup_modern_style()

            notebook = ttk.Notebook(self.root)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)

            self.setup_translation_tab(notebook)
            self.setup_dashboard_tab(notebook)
            self.setup_system_tab(notebook)
            self.setup_export_tab(notebook)

            self.setup_status_bar()

            self.root.after(1000, self.run_full_diagnostic)

            logging.info("✅ GUI erfolgreich initialisiert")

        except Exception as e:
            logging.error(f"❌ GUI Setup failed: {e}")
            raise

    def on_silence_detection_toggled(self):
        """Handler für Silence Detection Toggle"""
        try:
            enabled = self.silence_detection_var.get()
            self.translator.config.config['enable_silence_detection'] = enabled
            self.translator.config.save_config()
            status = "Aktiv" if enabled else "Inaktiv"
            self.status_var.set(f"✅ Silence Detection: {status}")
        except Exception as e:
            self.status_var.set(f"❌ Einstellungs-Änderung fehlgeschlagen: {e}")

    def setup_modern_style(self):
        """Konfiguriert modernes DARK Styling mit besseren Kontrasten"""
        try:
            style = ttk.Style()
            available_themes = style.theme_names()

            preferred_themes = ['clam', 'alt', 'default', 'classic']
            selected_theme = 'clam'

            for theme in preferred_themes:
                if theme in available_themes:
                    selected_theme = theme
                    break

            style.theme_use(selected_theme)

            style.configure('.',
                            background=ColorScheme.BG_PRIMARY,
                            foreground=ColorScheme.TEXT_PRIMARY,
                            fieldbackground=ColorScheme.BG_TERTIARY,
                            selectbackground=ColorScheme.ACCENT_BLUE,
                            selectforeground=ColorScheme.TEXT_PRIMARY)

            style.configure('TFrame', background=ColorScheme.BG_PRIMARY)
            style.configure('TLabel', background=ColorScheme.BG_PRIMARY,
                            foreground=ColorScheme.TEXT_PRIMARY)
            style.configure('TButton',
                            background=ColorScheme.BG_SECONDARY,
                            foreground=ColorScheme.TEXT_PRIMARY,
                            focuscolor='none')
            style.map('TButton',
                      background=[('active', ColorScheme.ACCENT_BLUE),
                                  ('pressed', ColorScheme.ACCENT_BLUE)])

            style.configure('Accent.TButton',
                            background=ColorScheme.ACCENT_GREEN,
                            foreground=ColorScheme.TEXT_PRIMARY)
            style.map('Accent.TButton',
                      background=[('active', ColorScheme.ACCENT_GREEN),
                                  ('pressed', ColorScheme.ACCENT_GREEN)])

            style.configure('TEntry',
                            fieldbackground=ColorScheme.BG_TERTIARY,
                            foreground=ColorScheme.TEXT_PRIMARY,
                            insertcolor=ColorScheme.TEXT_PRIMARY,
                            selectbackground=ColorScheme.ACCENT_BLUE,
                            selectforeground=ColorScheme.TEXT_PRIMARY)

            style.configure('TCombobox',
                            fieldbackground=ColorScheme.BG_TERTIARY,
                            background=ColorScheme.BG_SECONDARY,
                            foreground=ColorScheme.TEXT_PRIMARY,
                            selectbackground=ColorScheme.ACCENT_BLUE,
                            selectforeground=ColorScheme.TEXT_PRIMARY,
                            arrowcolor=ColorScheme.TEXT_PRIMARY)

            style.map('TCombobox',
                      fieldbackground=[('readonly', ColorScheme.BG_TERTIARY)],
                      selectbackground=[('readonly', ColorScheme.ACCENT_BLUE)],
                      selectforeground=[('readonly', ColorScheme.TEXT_PRIMARY)])

            style.configure('TCheckbutton',
                            background=ColorScheme.BG_PRIMARY,
                            foreground=ColorScheme.TEXT_PRIMARY)

            style.configure('TNotebook', background=ColorScheme.BG_PRIMARY)
            style.configure('TNotebook.Tab',
                            background=ColorScheme.BG_SECONDARY,
                            foreground=ColorScheme.TEXT_SECONDARY,
                            padding=[15, 5])
            style.map('TNotebook.Tab',
                      background=[('selected', ColorScheme.ACCENT_BLUE)],
                      foreground=[('selected', ColorScheme.TEXT_PRIMARY)])

        except Exception as e:
            logging.warning(f"Styling setup warning: {e}")

    def setup_translation_tab(self, notebook: ttk.Notebook):
        """Erstellt den Translation Tab mit REPARIERTEM Layout"""
        try:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text="🎯 Live Translation")

            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(3, weight=1)

            url_frame = ttk.LabelFrame(tab, text="🌐 Stream URL", padding=10)
            url_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
            url_frame.grid_columnconfigure(0, weight=1)

            url_input_frame = ttk.Frame(url_frame)
            url_input_frame.pack(fill='x', pady=5)

            ttk.Label(url_input_frame, text="URL:").pack(side='left')

            self.url_entry = tk.Entry(
                url_input_frame,
                font=("Arial", 10),
                bg=ColorScheme.BG_TERTIARY,
                fg=ColorScheme.TEXT_PRIMARY,
                insertbackground=ColorScheme.TEXT_PRIMARY
            )
            self.url_entry.pack(side='left', fill='x', expand=True, padx=10)
            self.url_entry.insert(0, "https://www.youtube.com/watch?v=kyMj1oMuKI0")

            url_actions_frame = ttk.Frame(url_input_frame)
            url_actions_frame.pack(side='left', padx=10)

            ttk.Button(url_actions_frame, text="📋", command=self.paste_to_url, width=3).pack(side='left', padx=2)
            ttk.Button(url_actions_frame, text="📁", command=self.select_local_file, width=3).pack(side='left', padx=2)
            ttk.Button(url_actions_frame, text="🧹", command=lambda: self.url_entry.delete(0, tk.END), width=3).pack(side='left', padx=2)

            settings_frame = ttk.Frame(tab)
            settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

            model_frame = ttk.Frame(settings_frame)
            model_frame.pack(side='left', padx=10)
            ttk.Label(model_frame, text="Modell:").pack(side='left')
            self.model_var = tk.StringVar(value=self.translator.config.config.get('transcription_model', 'small'))
            model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                       values=self.translator.get_available_models(),
                                       width=10, state="readonly")
            model_combo.pack(side='left', padx=5)
            model_combo.bind('<<ComboboxSelected>>', self.on_model_changed)

            lang_frame = ttk.Frame(settings_frame)
            lang_frame.pack(side='left', padx=10)
            ttk.Label(lang_frame, text="Sprache:").pack(side='left')
            self.lang_var = tk.StringVar(value=self.translator.config.config.get('target_language', 'en'))
            lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var,
                                      values=list(SUPPORTED_LANGUAGES.keys()),
                                      width=8, state="readonly")
            lang_combo.pack(side='left', padx=5)
            lang_combo.bind('<<ComboboxSelected>>', self.on_language_changed)

            feature_frame = ttk.Frame(settings_frame)
            feature_frame.pack(side='left', padx=10)

            self.translation_var = tk.BooleanVar(value=self.translator.config.config.get('translation_enabled', True))
            ttk.Checkbutton(feature_frame, text="Übersetzung", variable=self.translation_var,
                            command=self.on_translation_toggled).pack(side='left')

            self.silence_detection_var = tk.BooleanVar(
                value=self.translator.config.config.get('enable_silence_detection', True)
            )
            ttk.Checkbutton(feature_frame, text="Silence Detection", variable=self.silence_detection_var,
                            command=self.on_silence_detection_toggled).pack(side='left', padx=(10, 0))

            self.auto_scroll_var = tk.BooleanVar(
                value=self.translator.config.config.get('enable_auto_scroll', True)
            )
            ttk.Checkbutton(feature_frame, text="Auto-Scroll", variable=self.auto_scroll_var,
                            command=self.on_auto_scroll_toggled).pack(side='left', padx=(10, 0))

            control_frame = ttk.Frame(tab)
            control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

            self.start_button = ttk.Button(
                control_frame,
                text="🚀 Translation Starten",
                command=self.start_live,
                style="Accent.TButton",
                width=15
            )
            self.start_button.pack(side='left', padx=5)

            ttk.Button(control_frame, text="⏹️ Stoppen", command=self.stop, width=12).pack(side='left', padx=5)
            ttk.Button(control_frame, text="🗑️ Text löschen", command=self.clear_text, width=12).pack(side='left', padx=5)

            ttk.Button(control_frame, text="📊 Stats", command=self.show_stats, width=8).pack(side='left', padx=5)
            ttk.Button(control_frame, text="🔄 Diagnose", command=self.run_full_diagnostic, width=10).pack(side='left', padx=5)

            text_container = ttk.Frame(tab)
            text_container.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
            text_container.grid_rowconfigure(0, weight=1)
            text_container.grid_rowconfigure(1, weight=1)
            text_container.grid_columnconfigure(0, weight=1)

            transcript_frame = ttk.LabelFrame(text_container, text="📝 Live Transkription")
            transcript_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
            transcript_frame.grid_rowconfigure(0, weight=1)
            transcript_frame.grid_columnconfigure(0, weight=1)

            self.transcript_area = scrolledtext.ScrolledText(
                transcript_frame,
                height=10,
                wrap=tk.WORD,
                bg=ColorScheme.BG_TERTIARY,
                fg=ColorScheme.TEXT_PRIMARY,
                insertbackground=ColorScheme.TEXT_PRIMARY,
                font=("Consolas", 9)
            )
            self.transcript_area.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            translation_frame = ttk.LabelFrame(text_container, text="🌐 Live Übersetzung")
            translation_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
            translation_frame.grid_rowconfigure(0, weight=1)
            translation_frame.grid_columnconfigure(0, weight=1)

            self.translation_area = scrolledtext.ScrolledText(
                translation_frame,
                height=8,
                wrap=tk.WORD,
                bg=ColorScheme.BG_TERTIARY,
                fg=ColorScheme.TEXT_PRIMARY,
                insertbackground=ColorScheme.TEXT_PRIMARY,
                font=("Consolas", 9)
            )
            self.translation_area.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            self.setup_context_menus()

        except Exception as e:
            logging.error(f"Translation tab setup failed: {e}")
            raise

    def setup_dashboard_tab(self, notebook: ttk.Notebook):
        """Erstellt das Dashboard mit Live-Metriken - REPARIERT"""
        try:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text="🏠 Dashboard")

            header_frame = ttk.Frame(tab)
            header_frame.pack(fill='x', padx=20, pady=15)

            ttk.Label(header_frame,
                      text="🐉 DRAGON WHISPERER",
                      font=("Arial", 18, "bold"),
                      foreground=ColorScheme.ACCENT_BLUE).pack(pady=5)

            ttk.Label(header_frame,
                      text="Enterprise Grade Stream Translation & Analysis - HLS Blockierung behoben",
                      font=("Arial", 10),
                      foreground=ColorScheme.TEXT_SECONDARY).pack(pady=2)

            health_frame = ttk.LabelFrame(tab, text="🔍 System Health Status", padding=15)
            health_frame.pack(fill='x', padx=20, pady=10)

            self.health_var = tk.StringVar(value="🔄 Systemdiagnose läuft...")
            health_label = ttk.Label(health_frame, textvariable=self.health_var,
                                     font=("Arial", 12, "bold"))
            health_label.pack(pady=5)

            health_details_frame = ttk.Frame(health_frame)
            health_details_frame.pack(fill='x', pady=10)

            self.health_details_var = tk.StringVar(value="Lädt Systeminformationen...")
            health_details = ttk.Label(health_details_frame, textvariable=self.health_details_var,
                                       font=("Arial", 9), foreground=ColorScheme.TEXT_SECONDARY)
            health_details.pack()

            stats_frame = ttk.LabelFrame(tab, text="📊 Live Statistics", padding=15)
            stats_frame.pack(fill='x', padx=20, pady=10)

            stats_row1 = ttk.Frame(stats_frame)
            stats_row1.pack(fill='x', pady=5)

            self.cpu_var = tk.StringVar(value="CPU: --%")
            self.memory_var = tk.StringVar(value="RAM: --%")
            self.chunks_var = tk.StringVar(value="Chunks: 0")
            self.errors_var = tk.StringVar(value="Fehler: 0")

            ttk.Label(stats_row1, textvariable=self.cpu_var,
                      foreground=ColorScheme.TEXT_ACCENT,
                      font=("Arial", 10, "bold")).pack(side='left', padx=15)
            ttk.Label(stats_row1, textvariable=self.memory_var,
                      foreground=ColorScheme.TEXT_ACCENT,
                      font=("Arial", 10, "bold")).pack(side='left', padx=15)
            ttk.Label(stats_row1, textvariable=self.chunks_var,
                      foreground=ColorScheme.TEXT_SUCCESS,
                      font=("Arial", 10, "bold")).pack(side='left', padx=15)
            ttk.Label(stats_row1, textvariable=self.errors_var,
                      foreground=ColorScheme.TEXT_WARNING,
                      font=("Arial", 10, "bold")).pack(side='left', padx=15)

            stats_row2 = ttk.Frame(stats_frame)
            stats_row2.pack(fill='x', pady=5)

            self.skipped_var = tk.StringVar(value="Übersprungen: 0")
            self.cache_var = tk.StringVar(value="Cache: --%")
            self.uptime_var = tk.StringVar(value="Laufzeit: --")
            self.sentiment_var = tk.StringVar(value="Sentiment: --")

            ttk.Label(stats_row2, textvariable=self.skipped_var,
                      foreground=ColorScheme.TEXT_SECONDARY,
                      font=("Arial", 10)).pack(side='left', padx=15)
            ttk.Label(stats_row2, textvariable=self.cache_var,
                      foreground=ColorScheme.TEXT_SUCCESS,
                      font=("Arial", 10)).pack(side='left', padx=15)
            ttk.Label(stats_row2, textvariable=self.uptime_var,
                      foreground=ColorScheme.TEXT_ACCENT,
                      font=("Arial", 10)).pack(side='left', padx=15)
            ttk.Label(stats_row2, textvariable=self.sentiment_var,
                      foreground=ColorScheme.TEXT_WARNING,
                      font=("Arial", 10)).pack(side='left', padx=15)

            info_frame = ttk.LabelFrame(tab, text="📋 System Information", padding=10)
            info_frame.pack(fill='both', expand=True, padx=20, pady=10)

            self.dashboard_text = scrolledtext.ScrolledText(
                info_frame,
                bg=ColorScheme.BG_TERTIARY,
                fg=ColorScheme.TEXT_PRIMARY,
                font=("Consolas", 9)
            )
            self.dashboard_text.pack(fill='both', expand=True)

            action_frame = ttk.Frame(tab)
            action_frame.pack(fill='x', padx=20, pady=10)

            ttk.Button(action_frame, text="🔄 Aktualisieren", command=self.run_full_diagnostic).pack(side='left', padx=5)
            ttk.Button(action_frame, text="📈 Detaillierte Statistiken", command=self.show_detailed_stats).pack(side='left', padx=5)
            ttk.Button(action_frame, text="⚙️ Systemoptimierung", command=self.show_optimization_tips).pack(side='left', padx=5)

        except Exception as e:
            logging.error(f"Dashboard tab setup failed: {e}")
            raise

    def setup_system_tab(self, notebook: ttk.Notebook):
        """Erstellt den System Tab mit erweiterten Diagnose-Funktionen"""
        try:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text="⚙️ System")

            header_frame = ttk.Frame(tab)
            header_frame.pack(fill='x', padx=20, pady=15)

            ttk.Label(header_frame, text="⚙️ System Information & Diagnose",
                      font=("Arial", 16, "bold")).pack(pady=5)

            action_frame = ttk.Frame(tab)
            action_frame.pack(fill='x', padx=20, pady=10)

            ttk.Button(action_frame, text="🔄 Vollständige Diagnose",
                       command=self.run_full_diagnostic).pack(side='left', padx=5)
            ttk.Button(action_frame, text="💾 Report exportieren",
                       command=self.export_diagnostic_report).pack(side='left', padx=5)
            ttk.Button(action_frame, text="🧹 Cache leeren",
                       command=self.clear_translation_cache).pack(side='left', padx=5)
            ttk.Button(action_frame, text="🔧 Performance Check",
                       command=self.run_performance_check).pack(side='left', padx=5)

            info_frame = ttk.LabelFrame(tab, text="System Report", padding=10)
            info_frame.pack(fill='both', expand=True, padx=20, pady=10)

            self.system_info_text = scrolledtext.ScrolledText(
                info_frame,
                bg=ColorScheme.BG_TERTIARY,
                fg=ColorScheme.TEXT_PRIMARY,
                font=("Consolas", 9)
            )
            self.system_info_text.pack(fill='both', expand=True)

        except Exception as e:
            logging.error(f"System tab setup failed: {e}")
            raise

    def setup_export_tab(self, notebook: ttk.Notebook):
        """Erstellt den Export Tab mit erweiterten Funktionen"""
        try:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text="💾 Export")

            settings_frame = ttk.LabelFrame(tab, text="📋 Export Einstellungen", padding=15)
            settings_frame.pack(fill='x', padx=20, pady=10)

            format_frame = ttk.Frame(settings_frame)
            format_frame.pack(fill='x', pady=5)

            ttk.Label(format_frame, text="Export Format:").pack(side='left')
            self.export_format_var = tk.StringVar(value="txt")
            formats = [("Text", "txt"), ("Subtitles", "srt"), ("CSV", "csv"), ("JSON", "json")]
            format_combo = ttk.Combobox(format_frame, textvariable=self.export_format_var,
                                        values=[f[1] for f in formats], width=10)
            format_combo.pack(side='left', padx=10)
            format_combo.bind('<<ComboboxSelected>>', lambda e: self.update_preview())

            export_buttons_frame = ttk.Frame(settings_frame)
            export_buttons_frame.pack(fill='x', pady=10)

            ttk.Button(export_buttons_frame, text="💾 Transkriptionen exportieren",
                       command=self.export_transcriptions).pack(side='left', padx=5)

            ttk.Button(export_buttons_frame, text="📊 Diagnose-Report exportieren",
                       command=self.export_diagnostic_report).pack(side='left', padx=5)

            ttk.Button(export_buttons_frame, text="🧹 Transkriptionen löschen",
                       command=self.clear_transcription_history).pack(side='left', padx=5)

            stats_frame = ttk.LabelFrame(settings_frame, text="📈 Statistiken", padding=10)
            stats_frame.pack(fill='x', pady=10)

            stats_text = ttk.Frame(stats_frame)
            stats_text.pack(fill='x')

            self.stats_var = tk.StringVar(value="Transkriptionen: 0 | Wörter: 0 | Sprachen: 0")
            ttk.Label(stats_text, textvariable=self.stats_var).pack(anchor='w')

            preview_frame = ttk.LabelFrame(tab, text="👁️ Vorschau", padding=10)
            preview_frame.pack(fill='both', expand=True, padx=20, pady=10)

            preview_controls = ttk.Frame(preview_frame)
            preview_controls.pack(fill='x', pady=5)

            ttk.Button(preview_controls, text="🔄 Vorschau aktualisieren",
                       command=self.update_preview).pack(side='left', padx=5)

            ttk.Button(preview_controls, text="📋 Vorschau kopieren",
                       command=self.copy_preview).pack(side='left', padx=5)

            self.preview_text = scrolledtext.ScrolledText(
                preview_frame,
                bg=ColorScheme.BG_TERTIARY,
                fg=ColorScheme.TEXT_PRIMARY,
                font=("Consolas", 9),
                wrap=tk.WORD
            )
            self.preview_text.pack(fill='both', expand=True)

            self.update_preview()

        except Exception as e:
            logging.error(f"Export tab setup failed: {e}")
            raise

    def setup_status_bar(self):
        """Status Bar mit Beenden-Button und REPARIERTEM Layout"""
        try:
            status_frame = ttk.Frame(self.root, relief='sunken', borderwidth=1)
            status_frame.pack(fill='x', side='bottom', padx=2, pady=2)

            self.status_var = tk.StringVar(value="🐉 Dragon Whisperer - REPARIERT - Bereit")
            status_label = ttk.Label(status_frame, textvariable=self.status_var)
            status_label.pack(side='left', fill='x', expand=True, padx=5)

            self.session_var = tk.StringVar(value="Session: --")
            session_label = ttk.Label(status_frame, textvariable=self.session_var,
                                      font=("Arial", 8))
            session_label.pack(side='left', padx=10)

            self.metrics_var = tk.StringVar(value="CPU: 0% | RAM: 0% | Chunks: 0")
            metrics_label = ttk.Label(status_frame, textvariable=self.metrics_var,
                                      font=("Arial", 8))
            metrics_label.pack(side='left', padx=10)

            exit_button = ttk.Button(
                status_frame,
                text="🚪 Beenden",
                command=self.safe_exit,
                width=10
            )
            exit_button.pack(side='right', padx=5)

            self.update_metrics_display()

        except Exception as e:
            logging.error(f"Status bar setup failed: {e}")

    def setup_context_menus(self):
        """Erstellt Context Menüs für Textareas"""
        try:
            transcript_menu = Menu(self.root, tearoff=0,
                                   bg=ColorScheme.BG_TERTIARY,
                                   fg=ColorScheme.TEXT_PRIMARY,
                                   activebackground=ColorScheme.ACCENT_BLUE,
                                   activeforeground=ColorScheme.TEXT_PRIMARY)

            transcript_menu.add_command(label="Kopieren",
                                        command=lambda: self.copy_text(self.transcript_area))
            transcript_menu.add_command(label="Alles auswählen",
                                        command=lambda: self.select_all_text(self.transcript_area))
            transcript_menu.add_command(label="Alles löschen",
                                        command=lambda: self.clear_text_widget(self.transcript_area))
            transcript_menu.add_separator()
            transcript_menu.add_command(label="In Zwischenablage exportieren",
                                        command=lambda: self.export_to_clipboard(self.transcript_area))

            translation_menu = Menu(self.root, tearoff=0,
                                    bg=ColorScheme.BG_TERTIARY,
                                    fg=ColorScheme.TEXT_PRIMARY,
                                    activebackground=ColorScheme.ACCENT_BLUE,
                                    activeforeground=ColorScheme.TEXT_PRIMARY)

            translation_menu.add_command(label="Kopieren",
                                         command=lambda: self.copy_text(self.translation_area))
            translation_menu.add_command(label="Alles auswählen",
                                         command=lambda: self.select_all_text(self.translation_area))
            translation_menu.add_command(label="Alles löschen",
                                         command=lambda: self.clear_text_widget(self.translation_area))
            translation_menu.add_separator()
            translation_menu.add_command(label="In Zwischenablage exportieren",
                                         command=lambda: self.export_to_clipboard(self.translation_area))

            self.transcript_area.bind("<Button-3>",
                                      lambda e: self.show_context_menu(e, transcript_menu))
            self.translation_area.bind("<Button-3>",
                                       lambda e: self.show_context_menu(e, translation_menu))

        except Exception as e:
            logging.error(f"Context menu setup failed: {e}")

    def show_context_menu(self, event, menu: Menu):
        """Zeigt Context Menu an Position"""
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def copy_text(self, text_widget: tk.Text):
        """Kopiert markierten Text"""
        try:
            selected_text = text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.status_var.set("📋 Text kopiert")
        except Exception:
            self.status_var.set("❌ Kein Text markiert")

    def select_all_text(self, text_widget: tk.Text):
        """Selektiert gesamten Text"""
        text_widget.tag_add(tk.SEL, "1.0", tk.END)
        text_widget.mark_set(tk.INSERT, "1.0")
        text_widget.see(tk.INSERT)

    def clear_text_widget(self, text_widget: tk.Text):
        """Löscht Text in Widget"""
        text_widget.delete(1.0, tk.END)

    def export_to_clipboard(self, text_widget: tk.Text):
        """Exportiert gesamten Text in Zwischenablage"""
        try:
            all_text = text_widget.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(all_text)
            self.status_var.set("📋 Gesamter Text kopiert")
        except Exception as e:
            self.status_var.set(f"❌ Export fehlgeschlagen: {e}")

    def select_local_file(self):
        """Wählt lokale Datei aus"""
        try:
            filename = filedialog.askopenfilename(
                title="Audio/Video Datei auswählen",
                filetypes=[
                    ("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                    ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"),
                    ("All files", "*.*")
                ]
            )
            if filename:
                self.url_entry.delete(0, tk.END)
                self.url_entry.insert(0, f"file://{filename}")
                file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
                self.status_var.set(f"📁 Datei ausgewählt: {os.path.basename(filename)} ({file_size:.1f} MB)")
        except Exception as e:
            self.status_var.set(f"❌ Dateiauswahl fehlgeschlagen: {e}")

    def paste_to_url(self):
        """Fügt URL aus Zwischenablage ein"""
        try:
            clipboard_content = self.root.clipboard_get()
            if clipboard_content.strip():
                self.url_entry.delete(0, tk.END)
                self.url_entry.insert(0, clipboard_content.strip())
                self.status_var.set("📋 URL eingefügt")
            else:
                self.status_var.set("❌ Zwischenspeicher ist leer")
        except Exception as e:
            self.status_var.set("❌ Einfügen fehlgeschlagen")

    def on_model_changed(self, event=None):
        """Handler für Model-Änderung"""
        try:
            new_model = self.model_var.get()
            self.translator.config.config['transcription_model'] = new_model
            self.translator.config.save_config()
            self.status_var.set(f"✅ Modell geändert: {new_model}")
        except Exception as e:
            self.status_var.set(f"❌ Modell-Änderung fehlgeschlagen: {e}")

    def on_language_changed(self, event=None):
        """Handler für Sprach-Änderung"""
        try:
            new_lang = self.lang_var.get()
            self.translator.config.config['target_language'] = new_lang
            self.translator.config.save_config()
            self.translator.translation_engine.setup_translator()
            lang_name = SUPPORTED_LANGUAGES.get(new_lang, new_lang)
            self.status_var.set(f"✅ Zielsprache: {lang_name}")
        except Exception as e:
            self.status_var.set(f"❌ Sprach-Änderung fehlgeschlagen: {e}")

    def on_translation_toggled(self):
        """Handler für Übersetzungs-Toggle"""
        try:
            enabled = self.translation_var.get()
            self.translator.config.config['translation_enabled'] = enabled
            self.translator.config.save_config()
            status = "Aktiv" if enabled else "Inaktiv"
            self.status_var.set(f"✅ Übersetzung: {status}")
        except Exception as e:
            self.status_var.set(f"❌ Einstellungs-Änderung fehlgeschlagen: {e}")

    def on_auto_scroll_toggled(self):
        """Handler für Auto-Scroll Toggle"""
        try:
            enabled = self.auto_scroll_var.get()
            self.translator.config.config['enable_auto_scroll'] = enabled
            self.translator.config.save_config()
            status = "Aktiv" if enabled else "Inaktiv"
            self.status_var.set(f"✅ Auto-Scroll: {status}")
        except Exception as e:
            self.status_var.set(f"❌ Einstellungs-Änderung fehlgeschlagen: {e}")

    def clear_translation_cache(self):
        """Leert den Translation Cache"""
        try:
            self.translator.clear_translation_cache()
            self.status_var.set("🧹 Translation Cache geleert")
            self.run_full_diagnostic()  # Cache stats aktualisieren
        except Exception as e:
            self.status_var.set(f"❌ Cache-Löschen fehlgeschlagen: {e}")

    def start_live(self):
        """Startet Live-Translation - REPARIERT"""
        try:
            url = self.url_entry.get().strip()
            if not url:
                messagebox.showerror("Fehler", "Bitte geben Sie eine URL oder wählen Sie eine Datei aus!")
                return

            self.status_var.set("🚀 Starte Live-Translation...")
            self.root.update()

            callbacks = {
                'transcription': self.handle_transcription,
                'translation': self.handle_translation,
                'error': self.handle_error,
                'info': self.handle_info,
                'clear_text': self.clear_text
            }

            if self.translator.start_live_translation(url, callbacks):
                self.is_translating = True
                self.start_button.configure(text="🟢 Translation Läuft", style="Accent.TButton")
                self.status_var.set("✅ Live-Translation gestartet")

                stream_type = self.translator.stream_manager.detect_stream_type(url)
                self.session_var.set(f"Session: {stream_type.upper()}")
            else:
                self.status_var.set("❌ Start fehlgeschlagen")

        except Exception as e:
            self.status_var.set(f"❌ Start fehlgeschlagen: {e}")
            logging.error(f"Start live failed: {e}")

    def stop(self):
        """Stoppt Live-Translation"""
        try:
            self.translator.stop()
            self.is_translating = False
            self.start_button.configure(text="🚀 Translation Starten")
            self.status_var.set("⏹️ Translation gestoppt")
            self.session_var.set("Session: --")
        except Exception as e:
            self.status_var.set(f"❌ Stop fehlgeschlagen: {e}")

    def clear_text(self):
        """Leert Textbereiche"""
        try:
            self.transcript_area.delete(1.0, tk.END)
            self.translation_area.delete(1.0, tk.END)
            self.status_var.set("🗑️ Textbereiche geleert")
        except Exception as e:
            self.status_var.set(f"❌ Löschen fehlgeschlagen: {e}")

    def handle_transcription(self, result: TranscriptionResult):
        """Handler für Transkriptions-Resultate"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")

            confidence_text = f" ({result.confidence:.0%})" if result.confidence < 0.5 else ""
            speaker_text = f" [{result.speaker}]" if result.speaker else ""

            text = f"[{timestamp}] [{result.language.upper()}{confidence_text}{speaker_text}] {result.text}\n"

            self.transcript_area.insert(tk.END, text)

            if self.translator.config.config.get('enable_auto_scroll', True):
                self.transcript_area.see(tk.END)

            max_length = self.translator.config.config.get('max_text_length', 50000)
            if len(self.transcript_area.get(1.0, tk.END)) > max_length:
                self.transcript_area.delete(1.0, f"{1.0}+{max_length // 2}c")

            short_text = result.text[:40] + "..." if len(result.text) > 40 else result.text
            self.status_var.set(f"📝 {short_text}")

        except Exception as e:
            logging.error(f"Transcription handler error: {e}")

    def handle_translation(self, result: TranslationResult):
        """Handler für Übersetzungs-Resultate"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            text = f"[{timestamp}] [{result.target_lang.upper()}] {result.translated}\n"

            self.translation_area.insert(tk.END, text)

            if self.translator.config.config.get('enable_auto_scroll', True):
                self.translation_area.see(tk.END)

            max_length = self.translator.config.config.get('max_text_length', 50000)
            if len(self.translation_area.get(1.0, tk.END)) > max_length:
                self.translation_area.delete(1.0, f"{1.0}+{max_length // 2}c")

            short_text = result.translated[:40] + "..." if len(result.translated) > 40 else result.translated
            self.status_var.set(f"🌐 {short_text}")

        except Exception as e:
            logging.error(f"Translation handler error: {e}")

    def handle_error(self, error_msg: str):
        """Handler für Fehler"""
        try:
            self.status_var.set(f"❌ {error_msg}")
            logging.error(f"Error callback: {error_msg}")
        except Exception as e:
            logging.error(f"Error handler error: {e}")

    def handle_info(self, info_msg: str):
        """Handler für Info-Nachrichten"""
        try:
            self.status_var.set(f"ℹ️  {info_msg}")
            logging.info(f"Info callback: {info_msg}")
        except Exception as e:
            logging.error(f"Info handler error: {e}")

    def export_transcriptions(self):
        """Exportiert Transkriptionen"""
        if not self.translator.transcription_history:
            messagebox.showwarning("Export", "Keine Transkriptionen zum Exportieren verfügbar")
            return

        try:
            format_type = ExportFormat(self.export_format_var.get())
            file_ext = f".{format_type.value}"

            default_name = f"transkription_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"

            filename = filedialog.asksaveasfilename(
                defaultextension=file_ext,
                initialfile=default_name,
                filetypes=[(f"{format_type.value.upper()} files", f"*{file_ext}")]
            )

            if filename:
                exported_file = self.translator.export_transcriptions(format_type, filename)
                self.status_var.set(f"💾 Exportiert: {os.path.basename(exported_file)}")
                messagebox.showinfo("Erfolg",
                                    f"Transkriptionen erfolgreich exportiert:\n{exported_file}\n\n"
                                    f"Format: {format_type.value.upper()}\n"
                                    f"Einträge: {len(self.translator.transcription_history)}")

        except Exception as e:
            messagebox.showerror("Export Fehler", f"Fehler beim Export:\n{e}")

    def update_preview(self):
        """Aktualisiert die Export-Vorschau"""
        try:
            if not self.translator.transcription_history:
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, "Keine Transkriptionen für Vorschau verfügbar")
                return

            format_type = ExportFormat(self.export_format_var.get())

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{format_type.value}') as temp_file:
                temp_filename = temp_file.name

            self.translator.export_transcriptions(format_type, temp_filename)

            with open(temp_filename, 'r', encoding='utf-8') as f:
                preview_content = f.read()

            self.preview_text.delete(1.0, tk.END)
            if len(preview_content) > 2000:
                preview_content = preview_content[:2000] + "\n\n... (Vorschau gekürzt)"
            self.preview_text.insert(tk.END, preview_content)

            total_words = sum(len(trans.text.split()) for trans in self.translator.transcription_history)
            languages = len(self.translator.session_analytics.get('languages_detected', set()))
            self.stats_var.set(f"Transkriptionen: {len(self.translator.transcription_history)} | "
                               f"Wörter: {total_words} | Sprachen: {languages}")

            os.unlink(temp_filename)

        except Exception as e:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, f"Vorschau-Fehler: {e}")

    def copy_preview(self):
        """Kopiert Vorschau in Zwischenablage"""
        try:
            preview_content = self.preview_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(preview_content)
            self.status_var.set("📋 Vorschau kopiert")
        except Exception as e:
            self.status_var.set(f"❌ Kopieren fehlgeschlagen: {e}")

    def clear_transcription_history(self):
        """Löscht die Transkriptions-Historie"""
        if not self.translator.transcription_history:
            messagebox.showinfo("Löschen", "Keine Transkriptionen zum Löschen vorhanden")
            return

        if messagebox.askyesno("Löschen bestätigen",
                               f"Möchten Sie wirklich alle {len(self.translator.transcription_history)} "
                               "Transkriptionen löschen?"):
            self.translator.transcription_history.clear()
            self.translator.session_analytics['total_words'] = 0
            self.translator.session_analytics['languages_detected'].clear()
            self.translator.session_analytics['detected_topics'].clear()
            self.update_preview()
            self.status_var.set("🗑️ Transkriptions-Historie geleert")

    def run_full_diagnostic(self):
        """Führt vollständige Diagnose durch"""
        try:
            self.status_var.set("🔍 Führe Systemdiagnose durch...")
            self.root.update()

            health_status = self.translator.run_health_check()
            report = self.translator.get_detailed_report()

            health_icons = {
                'healthy': '✅',
                'degraded': '⚠️',
                'warning': '🔶',
                'critical': '❌',
                'error': '🔧'
            }
            icon = health_icons.get(health_status, '🔍')
            self.health_var.set(f"{icon} System Status: {health_status.upper()}")

            health_score = report.get('health_score', 0)
            suggestions = report.get('system_profile', {}).get('optimization_suggestions', [])
            details = f"Health Score: {health_score:.1%} | "
            details += " | ".join(suggestions[:2]) if suggestions else "Alle Systeme normal"
            self.health_details_var.set(details)

            metrics = report.get('current_metrics', {})
            cache_stats = report.get('cache_stats', {})
            performance_stats = report.get('performance_stats', {})

            self.cpu_var.set(f"CPU: {metrics.get('cpu_usage', 0):.1f}%")
            self.memory_var.set(f"RAM: {metrics.get('memory_usage', 0):.1f}%")
            self.chunks_var.set(f"Chunks: {metrics.get('chunks_processed', 0)}")
            self.errors_var.set(f"Fehler: {metrics.get('error_count', 0)}")
            self.skipped_var.set(f"Übersprungen: {metrics.get('silent_chunks_skipped', 0)}")
            self.cache_var.set(f"Cache: {cache_stats.get('hit_ratio', 0):.1%}")
            self.uptime_var.set(f"Laufzeit: {timedelta(seconds=int(metrics.get('uptime_seconds', 0)))}")

            sentiment_trend = self.translator.session_analytics.get('sentiment_trend', [])
            if sentiment_trend:
                latest_sentiment = sentiment_trend[-1]['sentiment']
                positive = latest_sentiment.get('positive', 0)
                self.sentiment_var.set(f"Sentiment: {positive:.0%} 👍")

            self.dashboard_text.delete(1.0, tk.END)
            self.dashboard_text.insert(tk.END, json.dumps(report, indent=2, ensure_ascii=False))

            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(tk.END, json.dumps(report, indent=2, ensure_ascii=False))

            self.status_var.set(f"✅ Diagnose abgeschlossen: {health_status}")

        except Exception as e:
            self.status_var.set(f"❌ Diagnose fehlgeschlagen: {e}")
            logging.error(f"Diagnostic failed: {e}")

    def export_diagnostic_report(self):
        """Exportiert Diagnostic Report"""
        try:
            default_name = f"diagnose_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=default_name,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                exported_file = self.translator.export_diagnostic_report(filename)
                self.status_var.set(f"💾 Report exportiert: {os.path.basename(exported_file)}")
                messagebox.showinfo("Erfolg", f"Diagnose-Report erfolgreich exportiert:\n{exported_file}")
        except Exception as e:
            messagebox.showerror("Export Fehler", f"Fehler beim Export:\n{e}")

    def show_stats(self):
        """Zeigt detaillierte Statistiken"""
        try:
            stats = self.translator.get_detailed_report()
            self._show_stats_window(stats, "Live Statistics")
        except Exception as e:
            messagebox.showerror("Fehler", f"Statistiken konnten nicht geladen werden:\n{e}")

    def show_detailed_stats(self):
        """Zeigt erweiterte Statistiken"""
        try:
            stats = self.translator.get_detailed_report()
            self._show_stats_window(stats, "Detailed Statistics", width=800, height=600)
        except Exception as e:
            messagebox.showerror("Fehler", f"Detaillierte Statistiken fehlgeschlagen:\n{e}")

    def _show_stats_window(self, stats: Dict[str, Any], title: str, width: int = 600, height: int = 400):
        """Zeigt Statistiken in einem separaten Fenster"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title(f"📊 {title}")
        stats_window.geometry(f"{width}x{height}")
        stats_window.configure(bg=ColorScheme.BG_PRIMARY)
        stats_window.transient(self.root)
        stats_window.grab_set()

        stats_window.update_idletasks()
        x = (stats_window.winfo_screenwidth() // 2) - (width // 2)
        y = (stats_window.winfo_screenheight() // 2) - (height // 2)
        stats_window.geometry(f"+{x}+{y}")

        text = scrolledtext.ScrolledText(
            stats_window,
            bg=ColorScheme.BG_TERTIARY,
            fg=ColorScheme.TEXT_PRIMARY,
            font=("Consolas", 9)
        )
        text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert(tk.END, json.dumps(stats, indent=2, ensure_ascii=False))
        text.config(state=tk.DISABLED)

        ttk.Button(stats_window, text="Schließen", command=stats_window.destroy).pack(pady=10)

    def show_optimization_tips(self):
        """Zeigt Optimierungstipps"""
        try:
            profile = self.translator.profiler.profile
            suggestions = profile.get('optimization_suggestions', [])

            tips_window = tk.Toplevel(self.root)
            tips_window.title("⚡ Optimierungstipps")
            tips_window.geometry("500x300")
            tips_window.configure(bg=ColorScheme.BG_PRIMARY)
            tips_window.transient(self.root)
            tips_window.grab_set()

            tips_window.update_idletasks()
            x = (tips_window.winfo_screenwidth() // 2) - 250
            y = (tips_window.winfo_screenheight() // 2) - 150
            tips_window.geometry(f"+{x}+{y}")

            ttk.Label(tips_window, text="⚡ System Optimierungstipps",
                      font=("Arial", 14, "bold")).pack(pady=10)

            text = scrolledtext.ScrolledText(
                tips_window,
                bg=ColorScheme.BG_TERTIARY,
                fg=ColorScheme.TEXT_PRIMARY,
                font=("Arial", 10),
                height=10
            )
            text.pack(fill='both', expand=True, padx=10, pady=10)

            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    text.insert(tk.END, f"{i}. {suggestion}\n\n")
            else:
                text.insert(tk.END, "✅ Ihr System ist optimal konfiguriert!\n\n")
                text.insert(tk.END, "💡 Tipps für beste Performance:\n")
                text.insert(tk.END, "• Verwende kleinere Modelle für Echtzeit-Transkription\n")
                text.insert(tk.END, "• Aktiviere Silence Detection für weniger CPU-Last\n")
                text.insert(tk.END, "• Schließe andere rechenintensive Anwendungen\n")

            text.config(state=tk.DISABLED)

            ttk.Button(tips_window, text="Schließen", command=tips_window.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Fehler", f"Optimierungstipps konnten nicht geladen werden:\n{e}")

    def run_performance_check(self):
        """Führt Performance-Check durch"""
        try:
            self.status_var.set("🚀 Führe Performance-Check durch...")

            start_time = time.time()

            test_results = []

            cpu_start = time.time()
            for i in range(1000000):
                _ = i * i
            cpu_time = time.time() - cpu_start
            test_results.append(f"CPU: {cpu_time:.3f}s")

            mem_start = time.time()
            test_list = [i for i in range(100000)]
            mem_time = time.time() - mem_start
            test_results.append(f"Memory: {mem_time:.3f}s")

            total_time = time.time() - start_time

            messagebox.showinfo("Performance Check",
                                f"Performance-Test abgeschlossen:\n\n"
                                f"Gesamtzeit: {total_time:.3f}s\n"
                                f"{', '.join(test_results)}\n\n"
                                f"System: {platform.system()} {platform.release()}\n"
                                f"Prozessor: {platform.processor()}")

            self.status_var.set("✅ Performance-Check abgeschlossen")

        except Exception as e:
            self.status_var.set(f"❌ Performance-Check fehlgeschlagen: {e}")

    def update_metrics_display(self):
        """Aktualisiert Metriken-Anzeige kontinuierlich"""
        try:
            if hasattr(self, 'translator'):
                metrics = self.translator.metrics
                cache_stats = self.translator.translation_engine.get_cache_stats()

                self.metrics_var.set(
                    f"CPU: {metrics.cpu_usage:.1f}% | "
                    f"RAM: {metrics.memory_usage:.1f}% | "
                    f"Chunks: {metrics.chunks_processed} | "
                    f"Cache: {cache_stats.get('hit_ratio', 0):.1%}"
                )

        except Exception as e:
            logging.debug(f"Metrics update error: {e}")

        self.root.after(2000, self.update_metrics_display)

    def safe_exit(self):
        """Sauberes Beenden mit Bestätigungsdialog - REPARIERT"""
        try:
            if messagebox.askokcancel(
                "Anwendung beenden",
                "Möchten Sie die Anwendung wirklich beenden?\n\n" +
                "✅ Laufende Translationen werden gestoppt\n" +
                "✅ Einstellungen werden gespeichert\n" +
                "✅ Alle Prozesse werden sauber beendet"
            ):
                self.status_var.set("🛑 Beende Anwendung...")
                self.root.update()

                if hasattr(self, 'is_translating') and self.is_translating:
                    self.stop()
                    time.sleep(1)

                if hasattr(self, 'translator'):
                    self.translator.config.save_config()
                    self.translator.stop()

                self.root.quit()
                self.root.destroy()

        except Exception as e:
            logging.error(f"Exit error: {e}")
            try:
                self.root.quit()
            except BaseException:
                pass

    def run(self):
        """Startet die GUI-Hauptloop mit Exception Handling"""
        try:
            self.root.mainloop()
        except Exception as e:
            logging.critical(f"❌ GUI Hauptloop fehlgeschlagen: {e}")
            messagebox.showerror("Kritischer Fehler", f"Die Anwendung muss beendet werden:\n{e}")


def main():
    """Hauptfunktion mit erweitertem Error Handling und System Checks"""
    print("🐉 Dragon Whisperer - Stream Translator - VOLLSTÄNDIG REPARIERT")
    print("=" * 60)
    print("🔧 REPARATUREN: HLS-Blockierung behoben mit non-blocking I/O")
    print("🎯 FEATURES: Timeout-Protection, Stream Health Monitoring")
    print("=" * 60)

    print(f"📋 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"📁 Arbeitsverzeichnis: {os.getcwd()}")

    log_dir = Path.home() / ".cache" / "dragon_whisperer"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "dragon_whisperer.log", encoding='utf-8', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    print("🔍 Führe System-Checks durch...")

    if not GUI_AVAILABLE:
        print("❌ GUI nicht verfügbar - tkinter fehlt")
        print("\n💡 Installationsanleitung:")
        if platform.system() == "Linux":
            print("   Ubuntu/Debian: sudo apt-get install python3-tk")
            print("   Fedora: sudo dnf install python3-tkinter")
            print("   Arch: sudo pacman -S tk")
        elif platform.system() == "Darwin":
            print("   macOS: brew install python-tk")
        elif platform.system() == "Windows":
            print("   Windows: Wird mit Python standardmäßig installiert")
        print("\n📚 Alternative: Verwende die Kommandozeilen-Version")
        return

    if not FASTER_WHISPER_AVAILABLE:
        print("⚠️  Faster-Whisper nicht verfügbar")
        print("💡 Installiere: pip install faster-whisper")

    if not TRANSLATOR_AVAILABLE:
        print("⚠️  Übersetzer nicht verfügbar")
        print("💡 Installiere: pip install deep-translator")

    required_tools = ['ffmpeg']
    missing_tools = []

    for tool in required_tools:
        try:
            if platform.system() == "Windows":
                result = subprocess.run(['where', tool], capture_output=True, check=False)
            else:
                result = subprocess.run(['which', tool], capture_output=True, check=False)

            if result.returncode != 0:
                missing_tools.append(tool)
        except Exception:
            missing_tools.append(tool)

    if missing_tools:
        print(f"⚠️  Fehlende Tools: {', '.join(missing_tools)}")
        print("💡 Installationsanleitung:")
        if platform.system() == "Linux":
            print("   Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("   Fedora: sudo dnf install ffmpeg")
            print("   Arch: sudo pacman -S ffmpeg")
        elif platform.system() == "Darwin":
            print("   macOS: brew install ffmpeg")
        elif platform.system() == "Windows":
            print("   Windows: Lade von https://ffmpeg.org/download.html herunter")

    print("✅ System-Checks abgeschlossen")
    print("🚀 Starte Anwendung...")

    try:
        gui = DragonWhispererGUI()
        print("✅ GUI erfolgreich geladen")
        print("🎯 Anwendung bereit - Viel Erfolg!")
        print("=" * 60)
        gui.run()

    except Exception as e:
        logging.critical(f"❌ Application failed: {e}")
        print(f"❌ Kritischer Fehler: {e}")

        error_msg = f"""
        🚨 Die Anwendung konnte nicht gestartet werden!

        Fehler: {e}

        Mögliche Lösungen:
        1. Stellen Sie sicher, dass alle Abhängigkeiten installiert sind
        2. Überprüfen Sie die Python-Umgebung
        3. Starten Sie die Anwendung neu

        Detaillierte Informationen finden Sie in der Log-Datei:
        {log_dir / "dragon_whisperer.log"}
        """

        print(error_msg)

        try:
            if GUI_AVAILABLE:
                root = tk.Tk()
                root.withdraw()  # Hide main window
                messagebox.showerror("Kritischer Fehler",
                                     f"Die Anwendung konnte nicht gestartet werden:\n\n{e}\n\n"
                                     f"Bitte überprüfen Sie die Log-Datei:\n{log_dir / 'dragon_whisperer.log'}")
                root.destroy()
        except BaseException:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
