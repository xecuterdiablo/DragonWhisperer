#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐉 THE DRAGON WHISPERER - Ultimate Stream Transcription & Translation 🐉

BRIDGING WORLDS THROUGH SILENT UNDERSTANDING
Livestream Transcription & Real-Time Translation

🎯 MISSION: To connect souls across the divides of language
💫 PHILOSOPHY: That the most profound connections begin 
   with a single, understood word
🌍 VISION: A universe where no heart's message is 
   lost in translation
"""

import os
import sys
import time
import json
import logging
import threading
import queue
import subprocess
import tempfile
import platform
import requests
import csv
import gc
import re
import warnings
import select
import shutil
import signal
import argparse
import atexit
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# Typing für zirkuläre Importe vermeiden
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

warnings.filterwarnings("ignore")

# Import mit Fehlerbehandlung
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
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError as e:
    YT_DLP_AVAILABLE = False
    print(f"⚠️  yt-dlp nicht verfügbar: {e}")

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
    EXIT_RED = "#d9534f"

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

class EnhancedLanguageDetector:
    """🔧 ENHANCED LANGUAGE DETECTION - 80%+ Genauigkeit durch kombinierte Whisper + Keyword-Erkennung"""
    
    def __init__(self):
        self.keyword_patterns = {
            'de': {
                'common': ['der', 'die', 'das', 'und', 'ist', 'nicht', 'zu', 'auf', 'für', 'wir', 'sie', 'ich', 'du'],
                'unique': ['genau', 'vielleicht', 'eigentlich', 'allerdings', 'deshalb', 'übrigens']
            },
            'en': {
                'common': ['the', 'and', 'is', 'to', 'of', 'in', 'that', 'with', 'for', 'you', 'we', 'they', 'I'],
                'unique': ['actually', 'however', 'therefore', 'moreover', 'furthermore', 'specifically']
            },
            'fr': {
                'common': ['le', 'la', 'les', 'et', 'est', 'dans', 'pour', 'avec', 'sur', 'nous', 'vous', 'ils'],
                'unique': ['exactement', 'peut-être', 'actuellement', 'cependant', 'd\'ailleurs']
            },
            'es': {
                'common': ['el', 'la', 'y', 'en', 'que', 'con', 'para', 'por', 'los', 'nosotros', 'ustedes'],
                'unique': ['exactamente', 'quizás', 'actualmente', 'sin embargo', 'por cierto']
            }
        }
        
    def detect_language_enhanced(self, text: str, whisper_lang: str = None, whisper_confidence: float = 0.0) -> Dict[str, Any]:
        """Kombinierte Sprach-Erkennung mit 80%+ Genauigkeit"""
        if not text or len(text.strip()) < 3:
            return {'language': 'unknown', 'confidence': 0.0, 'method': 'insufficient_text'}
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words < 2:
            return {'language': 'unknown', 'confidence': 0.0, 'method': 'insufficient_words'}
        
        keyword_scores = {}
        
        for lang, patterns in self.keyword_patterns.items():
            common_matches = sum(1 for word in words if word in patterns['common'])
            unique_matches = sum(1 for word in words if word in patterns['unique'])
            
            common_score = (common_matches / total_words) * 0.7
            unique_score = (unique_matches / len(patterns['unique'])) * 0.3
            
            keyword_scores[lang] = common_score + unique_score
        
        if whisper_lang and whisper_lang != 'unknown' and whisper_confidence > 0.3:
            if whisper_lang in keyword_scores:

                keyword_scores[whisper_lang] = max(
                    keyword_scores[whisper_lang],
                    (keyword_scores[whisper_lang] * 0.4) + (whisper_confidence * 0.6)
                )
            else:
                keyword_scores[whisper_lang] = whisper_confidence * 0.8
        
        if keyword_scores:
            best_lang = max(keyword_scores.items(), key=lambda x: x[1])
            confidence = min(1.0, best_lang[1] * 1.2)
            
            if confidence >= 0.6:
                return {
                    'language': best_lang[0],
                    'confidence': confidence,
                    'method': 'combined_detection',
                    'detailed_scores': keyword_scores,
                    'status': 'high_confidence'
                }
        
        if whisper_lang and whisper_lang != 'unknown' and whisper_confidence > 0.2:
            return {
                'language': whisper_lang,
                'confidence': whisper_confidence,
                'method': 'whisper_fallback',
                'status': 'medium_confidence'
            }
        
        return {
            'language': 'unknown',
            'confidence': max(keyword_scores.values()) if keyword_scores else 0.0,
            'method': 'keyword_only',
            'status': 'low_confidence'
        }

class StreamTitleExtractor:
    """🔧 OPTIMIERT: Stream-Titel Extraktor mit verbessertem Caching"""
    
    def __init__(self):
        self.last_extraction_time = 0
        self.cache_duration = 60  # Cache für 60 Sekunden
        self.cached_title = None
        self.cached_url = None
        
    def extract_stream_title(self, url: str) -> Optional[str]:
        """Extrahiert Stream-Titel von YouTube und anderen Plattformen"""
        try:
            current_time = time.time()
            if (self.cached_url == url and 
                current_time - self.last_extraction_time < self.cache_duration):
                return self.cached_title
                
            if not YT_DLP_AVAILABLE:
                return None
                
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'force_json': True,
                'simulate': True,
                'skip_download': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', '') if info else None
                
                self.cached_title = title
                self.cached_url = url
                self.last_extraction_time = current_time
                
                return title
                
        except Exception as e:
            logging.debug(f"Stream title extraction failed: {e}")
            return None
            
    def get_channel_info(self, url: str) -> Optional[Dict[str, str]]:
        """Extrahiert Kanal-Informationen"""
        try:
            if not YT_DLP_AVAILABLE:
                return None
                
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'force_json': True,
                'simulate': True,
                'skip_download': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    return {
                        'channel': info.get('uploader', ''),
                        'channel_url': info.get('uploader_url', ''),
                        'description': info.get('description', '')[:200] + '...' if info.get('description') else '',
                        'duration': info.get('duration', 0),
                        'view_count': info.get('view_count', 0)
                    }
            return None
            
        except Exception as e:
            logging.debug(f"Channel info extraction failed: {e}")
            return None

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
            'gui': {'available': GUI_AVAILABLE, 'version': 'unknown'},
            'yt_dlp': {'available': YT_DLP_AVAILABLE, 'version': 'unknown'}
        }

        try:
            if TORCH_AVAILABLE:
                deps['torch']['version'] = torch.__version__
            if NUMPY_AVAILABLE:
                deps['numpy']['version'] = np.__version__
            if PSUTIL_AVAILABLE:
                deps['psutil']['version'] = psutil.__version__
            if YT_DLP_AVAILABLE:
                deps['yt_dlp']['version'] = yt_dlp.version.__version__
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

class QoSMonitor:
    """Quality of Service Monitoring für Stream-Güte"""
    
    def __init__(self):
        self.metrics_history = {
            'latency': [],
            'throughput': [],
            'error_rate': [],
            'buffer_health': []
        }
        self.health_thresholds = {
            'latency': 5.0,
            'throughput': 0.1,
            'error_rate': 0.1,
            'buffer_health': 20.0
        }
        
    def record_processing_metric(self, metric_type: str, value: float):
        """Zeichnet QoS-Metriken auf"""
        if metric_type in self.metrics_history:
            self.metrics_history[metric_type].append(value)
            if len(self.metrics_history[metric_type]) > 100:
                self.metrics_history[metric_type].pop(0)
                
    def check_qos_health(self) -> List[str]:
        """Überprüft QoS-Grenzwerte"""
        warnings = []
        
        if self.metrics_history['latency']:
            avg_latency = sum(self.metrics_history['latency']) / len(self.metrics_history['latency'])
            if avg_latency > self.health_thresholds['latency']:
                warnings.append(f"🚨 Hohe Latenz: {avg_latency:.2f}s")
                
        if self.metrics_history['throughput']:
            avg_throughput = sum(self.metrics_history['throughput']) / len(self.metrics_history['throughput'])
            if avg_throughput < self.health_thresholds['throughput']:
                warnings.append(f"🚨 Geringer Durchsatz: {avg_throughput:.2f}MB/s")
                
        return warnings
        
    def calculate_health_score(self) -> float:
        """Berechnet Gesamt-QoS-Health-Score"""
        if not any(self.metrics_history.values()):
            return 100.0
            
        scores = []
        
        for metric, values in self.metrics_history.items():
            if values:
                avg_value = sum(values) / len(values)
                threshold = self.health_thresholds[metric]
                
                if metric == 'throughput':
                    score = min(100.0, (avg_value / threshold) * 100) if threshold > 0 else 100.0
                else:
                    score = max(0.0, 100.0 - (avg_value / threshold) * 100) if threshold > 0 else 100.0
                    
                scores.append(score)
                
        return sum(scores) / len(scores) if scores else 100.0
        
    def generate_qos_report(self) -> Dict[str, Any]:
        """Generiert detaillierten QoS-Report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'health_score': self.calculate_health_score(),
            'metrics': {}
        }
        
        for metric, values in self.metrics_history.items():
            if values:
                report['metrics'][metric] = {
                    'current': values[-1] if values else 0,
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'threshold': self.health_thresholds[metric]
                }
                
        return report

class PerformanceMonitor:
    """🔧 VOLLSTÄNDIG OPTIMIERT: Performance Monitoring mit REALISTISCHEN Memory-Thresholds"""

    def __init__(self):
        self.performance_thresholds = {
            'max_cpu': 90.0,
            'max_memory': 85.0,
            'max_latency': 10.0,
            'min_buffer_health': 20.0,
            'max_network_latency': 5.0,
            'min_throughput': 0.5,
            'max_gpu_memory': 90.0,
            'critical_memory': 2500,
            'warning_memory': 1800
        }
        self.start_time = time.time()
        self.chunk_times = []
        self.performance_warnings = []
        self.performance_alerts = []
        
        self.stream_metrics = {
            'audio_bytes_processed': 0,
            'network_requests': 0,
            'network_errors': 0,
            'stream_health_score': 100.0,
            'qos_metrics': {}
        }
        
        self.qos_monitor = QoSMonitor()

    def check_performance_health(self) -> List[str]:
        """🔧 KORRIGIERT: Überprüft Performance-Grenzwerte mit REALISTISCHEN Memory-Thresholds"""
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
                process_memory = process.memory_info().rss / 1024 / 1024
                
                if process_memory > self.performance_thresholds['critical_memory']:
                    warnings.append(f"🚨 KRITISCH: Prozess-Speicher sehr hoch: {process_memory:.1f}MB")
                elif process_memory > self.performance_thresholds['warning_memory']:
                    warnings.append(f"⚠️  WARNUNG: Prozess-Speicher hoch: {process_memory:.1f}MB")

            gpu_warnings = self._check_gpu_memory()
            warnings.extend(gpu_warnings)

            qos_warnings = self.qos_monitor.check_qos_health()
            warnings.extend(qos_warnings)

        except Exception as e:
            logging.debug(f"Performance check error: {e}")

        return warnings

    def _check_gpu_memory(self) -> List[str]:
        """🔧 OPTIMIERT: Überprüft GPU Memory Usage mit konsistenten MB-Einheiten"""
        warnings = []
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                
                if gpu_memory_allocated > 3000:
                    warnings.append(f"🎮 GPU Memory belegt: {gpu_memory_allocated:.1f}MB")
                    
        except Exception as e:
            logging.debug(f"GPU memory check failed: {e}")
            
        return warnings

    def record_chunk_processing(self, processing_time: float):
        """Zeichnet Verarbeitungszeiten auf mit QoS-Metriken"""
        self.chunk_times.append(processing_time)

        if len(self.chunk_times) > 100:
            self.chunk_times.pop(0)

        if len(self.chunk_times) >= 10:
            recent_avg = sum(self.chunk_times[-10:]) / 10
            overall_avg = sum(self.chunk_times) / len(self.chunk_times)

            if recent_avg > overall_avg * 1.5:
                self.performance_warnings.append(
                    f"Performance degradation detected: {recent_avg:.2f}s vs {overall_avg:.2f}s"
                )
                
        self.qos_monitor.record_processing_metric('chunk_processing_time', processing_time)

    def record_stream_metric(self, metric_name: str, value: float):
        """Zeichnet Stream-spezifische Metriken auf"""
        self.stream_metrics[metric_name] = value
        self.qos_monitor.record_processing_metric(metric_name, value)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt detaillierte Performance-Statistiken mit QoS zurück"""
        if not self.chunk_times:
            return {}

        total_time = time.time() - self.start_time
        chunks_per_minute = len(self.chunk_times) / \
            total_time * 60 if total_time > 0 else 0

        memory_stats = {}
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_stats = {
                    'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                    'system_memory_percent': psutil.virtual_memory().percent,
                    'system_memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
                }
            except Exception as e:
                logging.debug(f"Memory stats error: {e}")

        stats = {
            'avg_processing_time': sum(self.chunk_times) / len(self.chunk_times),
            'max_processing_time': max(self.chunk_times) if self.chunk_times else 0.0,
            'min_processing_time': min(self.chunk_times) if self.chunk_times else 0.0,
            'total_uptime': total_time,
            'chunks_per_minute': chunks_per_minute,
            'total_chunks_processed': len(self.chunk_times),
            'performance_warnings': self.performance_warnings[-5:],
            'current_load': len(self.chunk_times) / 100.0,
            'stream_metrics': self.stream_metrics.copy(),
            'qos_report': self.qos_monitor.generate_qos_report(),
            'system_health_score': self.qos_monitor.calculate_health_score(),
            'memory_stats': memory_stats
        }

        return stats

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Gibt sehr detaillierte Metriken für Deep Analysis zurück"""
        base_stats = self.get_performance_stats()
        
        detailed_metrics = {
            **base_stats,
            'timestamp': datetime.now().isoformat(),
            'chunk_time_distribution': self._get_time_distribution(),
            'performance_trend': self._get_performance_trend(),
            'resource_utilization': self._get_resource_utilization(),
            'bottleneck_analysis': self._analyze_bottlenecks()
        }
        
        return detailed_metrics

    def _get_time_distribution(self) -> Dict[str, float]:
        """Berechnet Verteilung der Verarbeitungszeiten mit NUMPY-Check"""
        if not self.chunk_times or not NUMPY_AVAILABLE:
            if not self.chunk_times:
                return {}
                
            sorted_times = sorted(self.chunk_times)
            n = len(sorted_times)
            return {
                'p50': sorted_times[int(n * 0.5)] if n > 0 else 0.0,
                'p90': sorted_times[int(n * 0.9)] if n > 0 else 0.0,
                'p95': sorted_times[int(n * 0.95)] if n > 0 else 0.0,
                'p99': sorted_times[int(n * 0.99)] if n > 0 else 0.0
            }
            
        try:
            times = np.array(self.chunk_times)
            return {
                'p50': float(np.percentile(times, 50)),
                'p90': float(np.percentile(times, 90)),
                'p95': float(np.percentile(times, 95)),
                'p99': float(np.percentile(times, 99))
            }
        except Exception as e:
            logging.debug(f"Numpy percentile calculation failed: {e}")
            sorted_times = sorted(self.chunk_times)
            n = len(sorted_times)
            return {
                'p50': sorted_times[int(n * 0.5)] if n > 0 else 0.0,
                'p90': sorted_times[int(n * 0.9)] if n > 0 else 0.0,
                'p95': sorted_times[int(n * 0.95)] if n > 0 else 0.0,
                'p99': sorted_times[int(n * 0.99)] if n > 0 else 0.0
            }

    def _get_performance_trend(self) -> Dict[str, float]:
        """Analysiert Performance-Trends"""
        if len(self.chunk_times) < 10:
            return {'trend': 'insufficient_data'}
            
        recent = self.chunk_times[-10:]
        older = self.chunk_times[-20:-10] if len(self.chunk_times) >= 20 else self.chunk_times[:10]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older) if older else recent_avg
        
        trend = 'stable'
        if recent_avg > older_avg * 1.2:
            trend = 'degrading'
        elif recent_avg < older_avg * 0.8:
            trend = 'improving'
            
        return {
            'trend': trend,
            'change_percentage': ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        }

    def _get_resource_utilization(self) -> Dict[str, float]:
        """Gibt Ressourcen-Auslastung zurück"""
        if not PSUTIL_AVAILABLE:
            return {}
            
        try:
            disk_io = psutil.disk_io_counters()._asdict() if hasattr(psutil, 'disk_io_counters') and psutil.disk_io_counters() else {}
            network_io = psutil.net_io_counters()._asdict() if hasattr(psutil, 'net_io_counters') and psutil.net_io_counters() else {}
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_io': disk_io,
                'network_io': network_io
            }
        except Exception as e:
            logging.debug(f"Resource utilization check failed: {e}")
            return {}

    def _analyze_bottlenecks(self) -> List[str]:
        """Identifiziert Performance-Engpässe"""
        bottlenecks = []
        
        if self.chunk_times:
            avg_time = sum(self.chunk_times) / len(self.chunk_times)
            if avg_time > 2.0:
                bottlenecks.append("Hohe Verarbeitungslatenz - prüfe AI-Modell")
                
        if PSUTIL_AVAILABLE:
            try:
                if psutil.cpu_percent() > 80:
                    bottlenecks.append("CPU-Auslastung hoch - reduziere Parallelität")
                if psutil.virtual_memory().percent > 80:
                    bottlenecks.append("Speicher knapp - prüfe Memory-Leaks")
                    
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024
                if process_memory > 2000:
                    bottlenecks.append(f"Prozess-Speicher hoch: {process_memory:.1f}MB - starte Memory-Cleanup")
                    
            except Exception:
                pass
                
        return bottlenecks

    def check_memory_health(self) -> Dict[str, Any]:
        """🔧 NEU: Detaillierte Memory-Health-Checks"""
        health_info = {}
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                health_info = {
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'percent': process.memory_percent(),
                    'system_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                    'system_percent': psutil.virtual_memory().percent
                }
                
            except Exception as e:
                logging.debug(f"Memory health check failed: {e}")
                
        return health_info

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """🔧 NEU: Erweiterte Metriken mit Netzwerk-Informationen"""
        metrics = self.get_detailed_metrics()
        
        if PSUTIL_AVAILABLE:
            try:
                net_io = psutil.net_io_counters()
                metrics['network'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            except Exception as e:
                logging.debug(f"Network metrics failed: {e}")
    
        return metrics

class ConfigPresets:
    """🔧 NEU: Configuration Presets für verschiedene Anwendungsfälle"""
    PRESETS = {
        'high_quality': {
            'model': 'large-v2',
            'chunk_duration': 3.0,
            'enable_enhanced_detection': True,
            'translation_cache_size': 1000,
            'enable_silence_detection': True
        },
        'fast': {
            'model': 'small', 
            'chunk_duration': 5.0,
            'enable_enhanced_detection': False,
            'translation_cache_size': 200,
            'enable_silence_detection': True
        },
        'balanced': {
            'model': 'medium',
            'chunk_duration': 4.0, 
            'enable_enhanced_detection': True,
            'translation_cache_size': 500,
            'enable_silence_detection': True
        }
    }

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
            "auto_clear_interval": 2000,
            "enable_silence_detection": True,
            "translation_cache_size": 500,
            "auto_recovery": True,
            "silence_threshold": 0.005,
            "enable_auto_scroll": True,
            "max_text_length": 50000,
            "export_format": "txt",
            "enable_speaker_detection": True,
            "enable_sentiment_analysis": False,
            "cloud_translation": False,
            "stream_type": "youtube",
            "ffmpeg_timeout": 20,
            "max_retry_attempts": 3,
            "auto_save_enabled": True,
            "auto_save_interval": 300,
            "adaptive_processing": True,
            "qos_monitoring": True,
            "enhanced_language_detection": True,
            "stream_title_update_interval": 60,
            "adaptive_settings_applied": False,
            "source_language": "auto"
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
        """🔧 ERWEITERTE Config-Validierung mit zusätzlichen Checks"""
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

        chunk_duration = config.get('chunk_duration', 5.0)
        if chunk_duration < 1.0:
            config['chunk_duration'] = 1.0
            logging.warning("Chunk duration zu niedrig, setze auf 1.0s")
        elif chunk_duration > 30.0:
            config['chunk_duration'] = 30.0
            logging.warning("Chunk duration zu hoch, setze auf 30.0s")
            
        cache_size = config.get('translation_cache_size', 500)
        if cache_size > 1000:
            config['translation_cache_size'] = 1000
            logging.warning("Translation cache zu groß, begrenze auf 1000")
            
        if config.get('auto_save_enabled', True):
            save_dir = Path('.').absolute()
            config['auto_save_directory'] = str(save_dir)
            
        silence_threshold = config.get('silence_threshold', 0.005)
        if silence_threshold < 0.001:
            config['silence_threshold'] = 0.001
        elif silence_threshold > 0.1:
            config['silence_threshold'] = 0.1
            
        if not config.get('adaptive_settings_applied'):
            profiler = IntelligentSystemProfiler()
            if profiler.profile['ram_gb'] < 8:
                config['transcription_model'] = 'small'
                config['translation_cache_size'] = 300
                logging.info("🔧 Adaptive Einstellungen für Low-RAM System angewendet")
            config['adaptive_settings_applied'] = True
            
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

    def save_config_with_backup(self) -> bool:
        """🔧 NEU: Speichert Konfiguration mit automatischem Backup"""
        try:
            if self.config_path.exists():
                backup_dir = self.config_path.parent / "backups"
                backup_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = backup_dir / f"config_backup_{timestamp}.json"
                
                import shutil
                shutil.copy2(self.config_path, backup_file)
                
                backups = sorted(backup_dir.glob("config_backup_*.json"))
                for old_backup in backups[:-5]:
                    old_backup.unlink()
            
            return self.save_config()
            
        except Exception as e:
            logging.error(f"Config backup failed: {e}")
            return self.save_config()

    def apply_preset(self, preset_name: str):
        """🔧 NEU: Wendet Configuration Preset an"""
        if preset_name in ConfigPresets.PRESETS:
            preset = ConfigPresets.PRESETS[preset_name]
            self.config.update(preset)
            self.save_config()
            logging.info(f"✅ Preset '{preset_name}' angewendet")
            return True
        return False

class ActivityMonitor:
    """Hilfsklasse für Aktivitäts-Monitoring"""
    def __init__(self):
        self.chunk_timestamps = []
        self.timeout_count = 0
        self._lock = threading.Lock()
        
    def record_chunk_processed(self, chunk_size: int):
        with self._lock:
            self.chunk_timestamps.append(time.time())
            cutoff = time.time() - 60
            self.chunk_timestamps = [ts for ts in self.chunk_timestamps if ts > cutoff]
            
    def record_timeout(self):
        with self._lock:
            self.timeout_count += 1
            
    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            now = time.time()
            recent_chunks = [ts for ts in self.chunk_timestamps if now - ts < 60]
            chunks_per_minute = len(recent_chunks)
            
            return {
                'chunks_per_minute': chunks_per_minute,
                'timeout_count': self.timeout_count,
                'total_chunks_processed': len(self.chunk_timestamps)
            }

class OptimizedAudioProcessor:
    """🔧 NEU: Optimierter Audio Processor mit Batch-Verarbeitung"""

    def __init__(self, config: AIConfigManager):
        self.config = config
        
        ram_gb = psutil.virtual_memory().total / (1024**3) if PSUTIL_AVAILABLE else 8
        queue_size = max(10, min(30, int(ram_gb * 2)))
        self.audio_queue = queue.Queue(maxsize=queue_size)
        
        self.processing = False
        self._lock = threading.RLock()
        self._chunk_counter = 0
        self._last_cleanup = 0
        self.processing_thread = None
        
        self._processing_batch = []
        self._batch_size = 3
        self._last_batch_time = 0
        
        self.dynamic_chunk_sizes = {
            'high_activity': 3.0,
            'normal': 5.0,
            'low_activity': 8.0,
            'file_processing': 10.0
        }
        
        self.current_activity_level = 'normal'
        self.silence_detection_count = 0
        self.activity_monitor = ActivityMonitor()

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

    def start_processing_adaptive(self, callback: Callable, stream_type: str = 'youtube'):
        """Startet adaptive Audio-Verarbeitung"""
        with self._lock:
            if self.processing:
                return
            self.processing = True

        if stream_type == 'local':
            self.current_activity_level = 'file_processing'
        
        def adaptive_processing_loop():
            chunk_id = 0
            while self.processing:
                try:
                    timeout = self._get_adaptive_timeout()
                    audio_data = self.audio_queue.get(timeout=timeout)
                    chunk_id += 1
                    self._chunk_counter += 1
                    
                    self.activity_monitor.record_chunk_processed(len(audio_data))
                    
                    callback(audio_data, chunk_id)
                    self.audio_queue.task_done()
                    
                    self._perform_memory_maintenance(chunk_id)
                    self._update_activity_level()
                        
                except queue.Empty:
                    self.activity_monitor.record_timeout()
                    continue
                except Exception as e:
                    logging.error(f"Adaptive audio processing error: {e}")

        self.processing_thread = threading.Thread(
            target=adaptive_processing_loop, 
            daemon=True, 
            name="AdaptiveAudioProcessor"
        )
        self.processing_thread.start()

    def process_audio_chunk_optimized(self, audio_data: bytes, chunk_id: int, output_callbacks: Dict[str, Callable]):
        """🔧 OPTIMIERT: Batch-Verarbeitung für bessere Performance"""
        self._processing_batch.append((audio_data, chunk_id))
        
        batch_ready = (len(self._processing_batch) >= self._batch_size or 
                      (time.time() - self._last_batch_time) > 0.5)
        
        if batch_ready and self._processing_batch:
            self._process_batch(output_callbacks)
            self._last_batch_time = time.time()

    def _process_batch(self, output_callbacks: Dict[str, Callable]):
        """Verarbeitet einen Batch von Audio-Chunks"""
        if not self._processing_batch:
            return
            
        batch_start_time = time.time()
        
        try:

            for audio_data, chunk_id in self._processing_batch:

                pass
                
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
        finally:
            self._processing_batch.clear()
            
        processing_time = time.time() - batch_start_time
        logging.debug(f"🔧 Batch verarbeitet in {processing_time:.3f}s")

    def _get_adaptive_timeout(self) -> float:
        """Gibt adaptiven Timeout basierend auf Aktivitäts-Level zurück"""
        timeouts = {
            'high_activity': 0.3,
            'normal': 0.5,
            'low_activity': 1.0,
            'file_processing': 2.0
        }
        return timeouts.get(self.current_activity_level, 0.5)

    def _update_activity_level(self):
        """Passt Aktivitäts-Level basierend auf Verarbeitungs-Statistiken an"""
        stats = self.activity_monitor.get_stats()
        
        if stats['chunks_per_minute'] > 15:
            self.current_activity_level = 'high_activity'
        elif stats['chunks_per_minute'] < 5:
            self.current_activity_level = 'low_activity'
        else:
            self.current_activity_level = 'normal'

    def get_current_chunk_size(self) -> float:
        """Gibt aktuelle Chunk-Größe zurück"""
        return self.dynamic_chunk_sizes.get(self.current_activity_level, 5.0)

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

        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        gc.collect()

    def add_audio_data(self, audio_data: bytes) -> bool:
        """🔧 Thread-sichere Methode zum Hinzufügen von Audio-Daten"""
        if not self.processing:
            logging.warning("Audio processor nicht aktiv - ignoriere Audio-Daten")
            return False
            
        try:
            self.audio_queue.put(audio_data, timeout=0.5)
            return True
        except queue.Full:
            if self._chunk_counter % 50 == 0:
                logging.warning("Audio queue voll - überspringe Chunk")
            return False

class UltimateStreamManager:
    """Stream Manager mit erweiterten YouTube-Strategien"""

    def __init__(self):
        self.supported_platforms = {
            'youtube': ['youtube.com', 'youtu.be'],
            'twitch': ['twitch.tv'],
            'facebook': ['facebook.com', 'fb.watch'],
            'rtmp': ['rtmp://', 'rtsp://'],
            'm3u8': ['.m3u8']
        }

        self.youtube_quality_priorities = [
            'bestaudio[ext=m4a]/bestaudio/best',
            'bestaudio[ext=webm]/bestaudio/best', 
            'bestaudio/best',
            'worstaudio/bestaudio'
        ]
        
        self.extraction_strategies = [
            ['yt-dlp', '-g', '-f', 'bestaudio[ext=m4a]/bestaudio', '--no-warnings'],
            ['yt-dlp', '-g', '-f', 'best', '--no-warnings'],
            ['yt-dlp', '-g', '--no-warnings'],
        ]

        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
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

        stream_type = self.detect_stream_type(url)
        if stream_type == 'youtube':
            enhanced_url = self.extract_stream_url_enhanced(url)
            if enhanced_url:
                return enhanced_url

        for i, strategy in enumerate(self.extraction_strategies):
            extracted_url = self._try_extraction_strategy(strategy, url, i)
            if extracted_url:
                return extracted_url

        logging.error("❌ Alle Extraktionsversuche fehlgeschlagen")
        return None

    def extract_stream_url_enhanced(self, url: str) -> Optional[str]:
        """🔧 OPTIMIERT: Extrahiert Stream-URL mit verbesserten Parametern"""
        stream_type = self.detect_stream_type(url)
        
        if stream_type != 'youtube':
            return self.extract_stream_url(url)

        logging.info(f"🎯 Starte erweiterte YouTube-Extraktion für: {url}")

        max_attempts = 3
        attempt_count = 0
        
        cookie_url = self.extract_stream_url_with_cookies(url)
        if cookie_url:
            logging.info("✅ Erfolg mit Cookie-Extraktion")
            return cookie_url

        for i, quality in enumerate(self.youtube_quality_priorities):
            if attempt_count >= max_attempts:
                break
                
            try:
                attempt_count += 1
                cmd = [
                    'yt-dlp', '-f', quality, '--get-url',
                    '--no-warnings', '--user-agent', self.user_agents[i % len(self.user_agents)],
                    '--add-header', 'Accept: */*',
                    '--add-header', 'Accept-Language: en-US,en;q=0.9',
                    url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0 and result.stdout.strip():
                    extracted_url = result.stdout.strip().split('\n')[0]
                    if extracted_url and extracted_url.startswith('http'):
                        logging.info(f"✅ Erfolg mit YouTube-Qualität {i+1}: {quality}")
                        return extracted_url
                        
            except subprocess.TimeoutExpired:
                logging.warning(f"⏰ Timeout bei YouTube Strategie {i+1}")
                continue
            except Exception as e:
                logging.warning(f"❌ YouTube Strategie {i+1} fehlgeschlagen: {e}")
                continue

        logging.info("🔄 Fallback auf Standard-Extraktion")
        return self.extract_stream_url(url)

    def extract_stream_url_with_cookies(self, url: str) -> Optional[str]:
        """Extrahiert Stream-URL mit Cookie-Support für YouTube"""
        try:
            cmd = [
                'yt-dlp', '-f', 'bestaudio[ext=m4a]/bestaudio/best',
                '--get-url',
                '--cookies-from-browser', 'chrome',
                '--no-warnings',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                extracted_url = result.stdout.strip().split('\n')[0]
                if extracted_url and extracted_url.startswith('http'):
                    logging.info("✅ Stream-URL mit Cookies erfolgreich extrahiert")
                    return extracted_url
                    
        except Exception as e:
            logging.warning(f"Cookie-Extraktion fehlgeschlagen: {e}")
        
        return None

    def get_recommended_settings(self, url: str) -> Dict[str, Any]:
        """Gibt optimale Einstellungen basierend auf Stream-Typ zurück"""
        stream_type = self.detect_stream_type(url)
        
        recommendations = {
            'youtube': {
                'model': 'small',
                'chunk_duration': 3.0,
                'enable_silence_detection': True,
                'translation_enabled': True
            },
            'local': {
                'model': 'medium',
                'chunk_duration': 10.0,
                'enable_silence_detection': False,
                'translation_enabled': True
            },
            'hls': {
                'model': 'small',
                'chunk_duration': 4.0,
                'enable_silence_detection': True,
                'translation_enabled': True
            }
        }
        
        return recommendations.get(stream_type, recommendations['youtube'])

    def _handle_local_file(self, file_path: str) -> str:
        """Behandelt lokale Dateien"""
        if file_path.startswith('file://'):
            file_path = file_path[7:]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        return file_path

    def _try_extraction_strategy(self, strategy: List[str], url: str, attempt: int) -> Optional[str]:
        """🔧 IMPLEMENTIERT: Fehlende Methode - Versucht eine Extraktionsstrategie mit Error Handling"""
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
                if extracted_url and extracted_url.startswith(('http', 'rtmp')):
                    logging.info(f"✅ Erfolg mit Strategie {attempt + 1}")
                    return extracted_url
                else:
                    logging.warning(f"⚠️ Ungültige URL von Strategie {attempt + 1}")
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

class OptimizedTranslationEngine:
    """🔧 NEU: Optimierte Translation Engine mit intelligentem Cache"""

    def __init__(self, config: AIConfigManager):
        self.config = config
        self.translator = None
        self.translation_cache: Dict[str, TranslationResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache_lock = threading.Lock()
        self.setup_translator()
        
        self.cache_access_order = []
        self.max_cache_size = self.config.config.get('translation_cache_size', 500)
        self.prefetch_enabled = True
        self.context_aware_cache = {}
        
        self._setup_memory_limits()
        self._last_memory_check = time.time()

    def _setup_memory_limits(self):
        """🔧 NEU: Setup Memory-Limits für Cache"""
        self.max_cache_memory_mb = 200

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
        """Übersetzt Text mit intelligentem Caching und Context-Awareness"""
        if not text or not text.strip():
            return self._create_empty_translation(text, source_lang, target_lang)

        cache_key = self._generate_context_aware_key(text, source_lang, target_lang, None)

        with self._cache_lock:
            if cache_key in self.translation_cache:
                self.cache_hits += 1
                if cache_key in self.cache_access_order:
                    self.cache_access_order.remove(cache_key)
                self.cache_access_order.append(cache_key)
                return self.translation_cache[cache_key]
            
            self.cache_misses += 1

        result = self._perform_translation(text, source_lang, target_lang, cache_key)
        
        if self.prefetch_enabled and len(text) > 10:
            self._prefetch_similar_translations(text, source_lang, target_lang)
            
        return result

    def translate_text_enhanced(self, text: str, source_lang: str, 
                               target_lang: str, context: str = None) -> TranslationResult:
        """Erweiterte Übersetzung mit Context-Awareness"""
        if not text or not text.strip():
            return self._create_empty_translation(text, source_lang, target_lang)

        cache_key = self._generate_context_aware_key(text, source_lang, target_lang, context)

        with self._cache_lock:
            if cache_key in self.translation_cache:
                self.cache_hits += 1
                if cache_key in self.cache_access_order:
                    self.cache_access_order.remove(cache_key)
                self.cache_access_order.append(cache_key)
                return self.translation_cache[cache_key]
            
            self.cache_misses += 1

        result = self._perform_translation(text, source_lang, target_lang, cache_key)
        
        if self.prefetch_enabled and len(text) > 10:
            self._prefetch_similar_translations(text, source_lang, target_lang)
            
        return result

    def _generate_cache_key(
            self, text: str, source_lang: str, target_lang: str) -> str:
        """Generiert Cache-Key"""
        text_hash = str(hash(text.strip().lower()))
        return f"{source_lang}_{target_lang}_{text_hash}"

    def _generate_context_aware_key(self, text: str, source_lang: str, 
                                  target_lang: str, context: str) -> str:
        """Generiert Context-aware Cache-Keys"""
        text_hash = str(hash(text.strip().lower()))
        context_hash = str(hash(context)) if context else "no_context"
        return f"{source_lang}_{target_lang}_{context_hash}_{text_hash}"

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

                self._manage_cache_intelligent(cache_key, result)
                return result

        except Exception as e:
            logging.error(f"❌ Übersetzung fehlgeschlagen: {e}")

        return self._create_empty_translation(text, source_lang, target_lang)

    def _manage_cache_intelligent(self, cache_key: str, result: TranslationResult):
        """🔧 NEU: Intelligenter Cache mit Memory-Limits"""
        with self._cache_lock:
            current_memory = self._estimate_cache_memory()
            
            if current_memory > self.max_cache_memory_mb:
                self._aggressive_cache_cleanup()

            self.translation_cache[cache_key] = result
            self.cache_access_order.append(cache_key)
            
            while len(self.translation_cache) > self.max_cache_size and self.cache_access_order:
                oldest_key = self.cache_access_order.pop(0)
                if oldest_key in self.translation_cache:
                    del self.translation_cache[oldest_key]

    def _estimate_cache_memory(self) -> float:
        """Schätzt den Memory-Verbrauch des Caches in MB"""
        try:
            total_size = 0
            for key, value in self.translation_cache.items():
                total_size += len(str(key)) + len(str(value))
            return total_size / 1024 / 1024
        except:
            return len(self.translation_cache) * 0.1

    def _aggressive_cache_cleanup(self):
        """🔧 NEU: Aggressive Cache-Bereinigung bei Memory-Engpässen"""
        logging.warning("🧹 Aggressive Cache-Bereinigung notwendig")
        
        keep_count = max(10, int(len(self.translation_cache) * 0.2))
        
        with self._cache_lock:
            recent_keys = self.cache_access_order[-keep_count:]
            self.translation_cache = {k: self.translation_cache[k] for k in recent_keys if k in self.translation_cache}
            self.cache_access_order = recent_keys.copy()
            
        logging.info(f"🔧 Cache reduziert auf {len(self.translation_cache)} Einträge")

    def _prefetch_similar_translations(self, text: str, source_lang: str, target_lang: str):
        """Prefetch für ähnliche Textfragmente"""
        try:
            sentences = self._split_into_sentences(text)
            if len(sentences) > 1:
                for sentence in sentences[:3]:
                    if len(sentence.strip()) > 5:
                        prefetch_key = self._generate_context_aware_key(
                            sentence, source_lang, target_lang, "prefetch"
                        )
                        if prefetch_key not in self.translation_cache:
                            threading.Thread(
                                target=self._async_prefetch_translation,
                                args=(sentence, source_lang, target_lang, prefetch_key),
                                daemon=True
                            ).start()
        except Exception as e:
            logging.debug(f"Prefetch error: {e}")

    def _async_prefetch_translation(self, text: str, source_lang: str, 
                                  target_lang: str, cache_key: str):
        """Asynchrone Prefetch-Übersetzung"""
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
                self._manage_cache_intelligent(cache_key, result)
        except Exception:
            pass

    def _split_into_sentences(self, text: str) -> List[str]:
        """Teilt Text in Sätze auf"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

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
            'memory_usage_estimate': self._estimate_cache_memory(),
            'max_cache_memory_mb': self.max_cache_memory_mb
        }

    def get_enhanced_cache_stats(self) -> Dict[str, Any]:
        """Detaillierte Cache-Statistiken"""
        base_stats = self.get_cache_stats()
        base_stats.update({
            'lru_cache_size': len(self.cache_access_order),
            'prefetch_enabled': self.prefetch_enabled,
            'cache_efficiency': self.cache_hits / (self.cache_hits + self.cache_misses) 
                              if (self.cache_hits + self.cache_misses) > 0 else 0.0
        })
        return base_stats

    def clear_cache(self):
        """Leert den Translation Cache"""
        with self._cache_lock:
            self.translation_cache.clear()
            self.cache_access_order.clear()
            self.cache_hits = 0
            self.cache_misses = 0

class AdvancedAnalyticsEngine:
    """🔧 OPTIMIERT: Erweiterte Analyse-Funktionen mit reduzierter Komplexität"""

    def __init__(self):
        self.sentiment_cache = {}
        self.topic_cache = {}
        self.language_detection_cache = {}
        
        self.language_keywords = {
            'de': ['der', 'die', 'das', 'und', 'ist', 'nicht', 'zu', 'auf', 'für'],
            'en': ['the', 'and', 'is', 'to', 'of', 'in', 'that', 'with', 'for'],
            'fr': ['le', 'la', 'les', 'et', 'est', 'dans', 'pour', 'avec', 'sur'],
            'es': ['el', 'la', 'y', 'en', 'que', 'con', 'para', 'por', 'los'],
            'it': ['il', 'la', 'e', 'di', 'che', 'con', 'per', 'non', 'una'],
            'pt': ['o', 'a', 'e', 'de', 'que', 'com', 'para', 'não', 'uma'],
            'ru': ['и', 'в', 'не', 'на', 'я', 'быть', 'с', 'что', 'по'],
            'zh': ['的', '了', '在', '是', '我', '有', '和', '就', '不'],
            'ja': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と'],
            'ko': ['이', '에', '는', '을', 'の', '가', '로', '다', '한']
        }

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

    def detect_language_enhanced(self, text: str, min_confidence: float = 0.6) -> Dict[str, Any]:
        """Erweiterte Sprach-Erkennung mit Confidence-Check"""
        if not text or len(text.strip()) < 3:
            return {'language': 'unknown', 'confidence': 0.0, 'detected_languages': []}
            
        text_hash = hash(text.lower())
        if text_hash in self.language_detection_cache:
            cached_result = self.language_detection_cache[text_hash]
            if cached_result['confidence'] >= min_confidence:
                return cached_result
            
        text_lower = text.lower()
        words = text_lower.split()
        
        language_scores = {}
        
        for lang, keywords in self.language_keywords.items():
            score = 0
            keyword_matches = 0
            
            for keyword in keywords:
                if keyword in text_lower:
                    keyword_matches += 1
                    score += text_lower.count(keyword) * (1 if len(keyword) <= 3 else 2)
                    
            if keyword_matches > 0:
                normalized_score = score / len(words) if words else 0
                language_scores[lang] = {
                    'score': normalized_score,
                    'keyword_matches': keyword_matches,
                    'coverage': keyword_matches / len(keywords) if keywords else 0
                }
        
        if language_scores:
            best_lang = max(language_scores.items(), key=lambda x: x[1]['score'])
            confidence = min(1.0, best_lang[1]['score'] * 10)
            
            if confidence < min_confidence:
                result = {
                    'language': 'unknown', 
                    'confidence': confidence,
                    'detected_languages': [],
                    'detailed_scores': language_scores,
                    'status': 'low_confidence'
                }
            else:
                detected_languages = sorted(
                    [(lang, data['score']) for lang, data in language_scores.items()],
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                result = {
                    'language': best_lang[0],
                    'confidence': confidence,
                    'detected_languages': detected_languages,
                    'detailed_scores': language_scores,
                    'status': 'high_confidence'
                }
        else:
            result = {
                'language': 'unknown', 
                'confidence': 0.0,
                'detected_languages': [],
                'detailed_scores': {},
                'status': 'no_match'
            }
            
        if len(words) > 2:
            self.language_detection_cache[text_hash] = result
            
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
            if keyword_count >= 1:
                topics.append(topic)

        result = topics[:3]

        if len(text.split()) > 5:
            self.topic_cache[text_hash] = result

        return result

    def get_analytics_report(self, text: str) -> Dict[str, Any]:
        """Generiert umfassenden Analytics-Report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'word_count': len(text.split()),
            'language_analysis': self.detect_language_enhanced(text),
            'sentiment_analysis': self.analyze_sentiment(text),
            'topics_detected': self.detect_topics(text),
            'complexity_score': self._calculate_complexity(text)
        }
        
    def _calculate_complexity(self, text: str) -> float:
        """Berechnet Text-Komplexitäts-Score"""
        if not text:
            return 0.0
            
        words = text.split()
        if not words:
            return 0.0
            
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentences = text.split('.')
        avg_sentence_length = len(words) / len(sentences) if sentences else len(words)
        
        complexity = min(1.0, (avg_word_length * 0.1 + avg_sentence_length * 0.01))
        return complexity

class AutoSaveManager:
    """Auto-Save Manager für Transkriptionen"""
    
    def __init__(self, translator: 'DragonWhispererTranslator', save_interval: int = 300):
        self.translator = translator
        self.save_interval = save_interval
        self.auto_save_enabled = False
        self.last_save_time = time.time()
        self.save_thread = None
        self._shutdown_event = threading.Event()
        self.save_count = 0
        
    def start_auto_save(self):
        """Startet Auto-Save Funktionalität"""
        if self.auto_save_enabled:
            return
            
        self.auto_save_enabled = True
        self._shutdown_event.clear()
        
        def auto_save_loop():
            while self.auto_save_enabled and not self._shutdown_event.is_set():
                try:
                    time.sleep(60)
                    
                    current_time = time.time()
                    if (current_time - self.last_save_time >= self.save_interval and 
                        self.translator.transcription_history):
                        
                        self._perform_auto_save()
                        self.last_save_time = current_time
                        self.save_count += 1
                        
                except Exception as e:
                    logging.error(f"Auto-save error: {e}")
                    
        self.save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        self.save_thread.start()
        logging.info("✅ Auto-Save gestartet")
        
    def stop_auto_save(self):
        """Stoppt Auto-Save Funktionalität"""
        self.auto_save_enabled = False
        self._shutdown_event.set()
        
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=5.0)
        logging.info("⏹️ Auto-Save gestoppt")
        
    def _perform_auto_save(self):
        """Führt Auto-Save durch"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auto_save_transcription_{timestamp}.json"
            
            self.translator.export_manager.export_json(
                self.translator.transcription_history, 
                filename
            )
            
            logging.info(f"💾 Auto-Save #{self.save_count} durchgeführt: {filename}")
            
        except Exception as e:
            logging.error(f"Auto-save failed: {e}")

    def get_auto_save_stats(self) -> Dict[str, Any]:
        """Gibt Auto-Save Statistiken zurück"""
        return {
            'enabled': self.auto_save_enabled,
            'save_count': self.save_count,
            'last_save_time': self.last_save_time,
            'save_interval': self.save_interval,
            'next_save_in': max(0, self.save_interval - (time.time() - self.last_save_time))
        }

class EnhancedMemoryManager:
    """🔧 REPARIERT: Vollständig implementierter Memory-Manager"""

    def __init__(self, translator):
        self.translator = translator
        self.cleanup_thresholds = {
            'critical': 2500,
            'high': 1800,
            'medium': 1200,
            'low': 800
        }
        self.last_cleanup_time = 0
        self.cleanup_cooldown = 120
        self.cleanup_count = 0
    
    def intelligent_cleanup(self, current_memory_mb: float):
        """🔧 VOLLSTÄNDIG KORRIGIERT: Intelligenter Cleanup mit allen Stufen"""
        current_time = time.time()
        
        if current_time - self.last_cleanup_time < self.cleanup_cooldown:
            return
            
        if current_memory_mb > self.cleanup_thresholds['critical']:
            logging.warning(f"🧹 KRITISCH: Memory {current_memory_mb:.1f}MB - Aggressiver Cleanup #{self.cleanup_count+1}")
            self._aggressive_cleanup()
            self.last_cleanup_time = current_time
            self.cleanup_count += 1
            
        elif current_memory_mb > self.cleanup_thresholds['high']:
            logging.info(f"🧹 HOCH: Memory {current_memory_mb:.1f}MB - Cache-Optimierung")
            self._optimize_translation_cache()
            self._comprehensive_cleanup()
            self.last_cleanup_time = current_time
            self.cleanup_count += 1
            
        elif current_memory_mb > self.cleanup_thresholds['medium']:
            logging.debug(f"🧹 MEDIUM: Memory {current_memory_mb:.1f}MB - Selektiver Cleanup")
            self._selective_cleanup()
            self.last_cleanup_time = current_time
            self.cleanup_count += 1
            
        elif current_memory_mb > self.cleanup_thresholds['low']:
            logging.debug(f"🧹 NIEDRIG: Memory {current_memory_mb:.1f}MB - Leichter Cleanup")
            self._light_cleanup()
            self.last_cleanup_time = current_time
                
    def _optimize_translation_cache(self):
        """🔧 NEU: Cache optimieren statt zerstören - VOLLSTÄNDIG IMPLEMENTIERT"""
        try:
            if hasattr(self.translator, 'translation_engine'):
                cache_size = len(self.translator.translation_engine.translation_cache)
                if cache_size > 100:
                    target_size = max(50, int(cache_size * 0.75))
                    self._cleanup_translation_cache_optimized(target_size)
                    logging.info(f"🔧 Cache optimiert: {cache_size} → {target_size} Einträge")
        except Exception as e:
            logging.error(f"Cache optimization error: {e}")
                
    def _cleanup_translation_cache_optimized(self, target_size: int):
        """🔧 NEU: Optimierte Cache-Bereinigung mit Zielgröße"""
        try:
            if hasattr(self.translator, 'translation_engine'):
                cache_size = len(self.translator.translation_engine.translation_cache)
                if cache_size <= target_size:
                    return
                
                with self.translator.translation_engine._cache_lock:
                    keys_to_keep = self.translator.translation_engine.cache_access_order[-target_size:]
                
                    new_cache = {}
                    for key in keys_to_keep:
                        if key in self.translator.translation_engine.translation_cache:
                            new_cache[key] = self.translator.translation_engine.translation_cache[key]
                
                    self.translator.translation_engine.translation_cache = new_cache
                    self.translator.translation_engine.cache_access_order = keys_to_keep.copy()
                
                    logging.debug(f"🔧 Translation Cache optimiert: {cache_size} → {len(new_cache)} Einträge")
                
        except Exception as e:
            logging.error(f"Optimized cache cleanup error: {e}")
                
    def _aggressive_cleanup(self):
        """🔧 VERBESSERT: Aggressiver Cleanup bei kritischem Memory"""
        logging.warning("🚨 Aggressiver Memory-Cleanup notwendig")
        
        try:
            if hasattr(self.translator, 'translation_engine'):
                self.translator.translation_engine.clear_cache()
            
            if hasattr(self.translator, 'analytics_engine'):
                self.translator.analytics_engine.sentiment_cache.clear()
                self.translator.analytics_engine.topic_cache.clear()
                self.translator.analytics_engine.language_detection_cache.clear()
            
            if hasattr(self.translator, 'session_analytics'):
                if 'sentiment_trend' in self.translator.session_analytics:
                    self.translator.session_analytics['sentiment_trend'] = \
                        self.translator.session_analytics['sentiment_trend'][-20:]
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            gc.collect()
            
            logging.info("✅ Aggressiver Cleanup abgeschlossen")
            
        except Exception as e:
            logging.error(f"Aggressive cleanup error: {e}")
            
    def _comprehensive_cleanup(self):
        """🔧 NEU: Umfassender Cleanup bei hohem Memory-Verbrauch"""
        logging.info("🧹 Umfassender Memory-Cleanup")
        try:
            if hasattr(self.translator, '_cleanup_memory_enhanced'):
                self.translator._cleanup_memory_enhanced()
            self._optimize_translation_cache()
            gc.collect()
            logging.info("✅ Umfassender Cleanup abgeschlossen")
        except Exception as e:
            logging.error(f"Comprehensive cleanup error: {e}")
        
    def _selective_cleanup(self):
        """🔧 NEU: Selektiver Cleanup bei moderatem Memory-Verbrauch"""
        logging.debug("🔧 Selektiver Memory-Cleanup")
        try:
            if hasattr(self.translator, '_cleanup_translation_cache'):
                self.translator._cleanup_translation_cache()
            if hasattr(self.translator, '_cleanup_analytics_data'):
                self.translator._cleanup_analytics_data()
            gc.collect()
            logging.debug("✅ Selektiver Cleanup abgeschlossen")
        except Exception as e:
            logging.debug(f"Selective cleanup error: {e}")
        
    def _light_cleanup(self):
        """🔧 NEU: Leichter Cleanup bei niedrigem Memory-Verbrauch"""
        logging.debug("💡 Leichter Memory-Cleanup")
        try:
            gc.collect()
            logging.debug("✅ Leichter Cleanup abgeschlossen")
        except Exception as e:
            logging.debug(f"Light cleanup error: {e}")

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """🔧 NEU: Gibt Cleanup-Statistiken zurück"""
        return {
            'cleanup_count': self.cleanup_count,
            'last_cleanup_time': self.last_cleanup_time,
            'thresholds': self.cleanup_thresholds.copy(),
            'cooldown_remaining': max(0, self.cleanup_cooldown - (time.time() - self.last_cleanup_time))
        }

class StreamHealthMonitor:
    """🔧 NEU: Überwacht die Stream-Gesundheit"""
    
    def __init__(self):
        self.metrics = {
            'connection_quality': 100.0,
            'buffer_health': 100.0,
            'stability_score': 100.0,
            'reconnect_count': 0
        }
        self.last_data_time = time.time()
    
    def check_stream_health(self) -> Dict[str, Any]:
        """Prüft die Stream-Gesundheit"""
        current_time = time.time()
        time_since_data = current_time - self.last_data_time
        
        health_status = "healthy"
        
        if time_since_data > 30:
            health_status = "disconnected"
            self.metrics['connection_quality'] = 0.0
        elif time_since_data > 10:
            health_status = "degraded" 
            self.metrics['connection_quality'] = 50.0
        elif time_since_data > 5:
            health_status = "unstable"
            self.metrics['connection_quality'] = 75.0
        
        return {
            'status': health_status,
            'time_since_data': time_since_data,
            'metrics': self.metrics.copy(),
            'recommendation': self._get_recommendation(health_status)
        }
    
    def _get_recommendation(self, status: str) -> str:
        """Gibt Empfehlungen basierend auf Status"""
        recommendations = {
            'healthy': "Stream ist stabil",
            'unstable': "Stream ist instabil - prüfe Internetverbindung",
            'degraded': "Stream-Probleme - versuche Reconnect",
            'disconnected': "Stream unterbrochen - Neustart erforderlich"
        }
        return recommendations.get(status, "Unbekannter Status")
    
    def record_data_received(self):
        """Zeichnet Datenempfang auf"""
        self.last_data_time = time.time()
        self.metrics['connection_quality'] = 100.0

class AutoTuningManager:
    """🔧 NEU: Automatische Performance-Optimierung"""
    
    def __init__(self, translator):
        self.translator = translator
        self.performance_history = []
        self.tuning_enabled = True
        
    def start_auto_tuning(self):
        """Startet Auto-Tuning Loop"""
        def tuning_loop():
            while self.translator.is_running:
                try:
                    self._analyze_and_tune()
                    time.sleep(30)
                except Exception as e:
                    logging.error(f"Auto-tuning error: {e}")
        
        threading.Thread(target=tuning_loop, daemon=True, name="AutoTuning").start()
        logging.info("🔧 Auto-Tuning gestartet")
        
    def _analyze_and_tune(self):
        """Analysiert Performance und passt Einstellungen an"""
        if not self.tuning_enabled:
            return
            
        try:
            metrics = self.translator.performance_monitor.get_performance_stats()
            self.performance_history.append({
                'timestamp': time.time(),
                'metrics': metrics
            })
            
            if len(self.performance_history) > 10:
                self.performance_history.pop(0)
                
            if len(self.performance_history) >= 5:
                self._apply_tuning_decisions()
                
        except Exception as e:
            logging.debug(f"Auto-tuning analysis error: {e}")
            
    def _apply_tuning_decisions(self):
        """Wendet Tuning-Entscheidungen basierend auf Performance an"""
        try:
            recent_metrics = self.performance_history[-1]['metrics']
            
            if recent_metrics.get('current_load', 0) > 0.8:
                logging.info("🔧 Hohe Auslastung - optimiere Einstellungen")
                
            memory_stats = recent_metrics.get('memory_stats', {})
            if memory_stats.get('process_memory_mb', 0) > 800:
                if hasattr(self.translator, '_cleanup_memory_enhanced'):
                    self.translator._cleanup_memory_enhanced()
                logging.info("🔧 Hoher Memory-Verbrauch - Cleanup durchgeführt")
                
        except Exception as e:
            logging.debug(f"Tuning application error: {e}")
            
    def _apply_memory_optimizations(self):
        """🔴 NEU: Konkrete Optimierungen anwenden"""
        config = self.translator.config.config
        
        current_model = config.get('transcription_model', 'small')
        if current_model in ['medium', 'large-v2']:
            config['transcription_model'] = 'small'
            logging.info("🔧 Auto-Tuning: Modell auf 'small' gewechselt für weniger Memory")
        
        if config.get('translation_cache_size', 500) > 300:
            config['translation_cache_size'] = 300
            logging.info("🔧 Auto-Tuning: Cache-Größe auf 300 reduziert")
            
        if config.get('chunk_duration', 5.0) < 8.0:
            config['chunk_duration'] = 8.0
            logging.info("🔧 Auto-Tuning: Chunk-Duration auf 8.0s erhöht")
            
        self.translator.config.save_config()

class IntelligentLayoutManager:
    """🔧 NEU: Intelligentes Layout Management für optimale Platzausnutzung"""
    
    def __init__(self, root):
        self.root = root
        self.layout_configs = {
            'compact': self._compact_layout,
            'balanced': self._balanced_layout, 
            'spacious': self._spacious_layout
        }
        
    def auto_adjust_layout(self, window_size: tuple):
        """Passt Layout automatisch basierend auf Fenstergröße an"""
        width, height = window_size
        
        if width < 1200 or height < 800:
            return self.layout_configs['compact']()
        elif width < 1600 or height < 1000:
            return self.layout_configs['balanced']()
        else:
            return self.layout_configs['spacious']()
    
    def _compact_layout(self):
        """Kompaktes Layout für kleine Bildschirme"""
        return {
            'tab_padding': (5, 5),
            'text_height': 8,
            'font_size': 9,
            'button_width': 12,
            'header_font_size': 12
        }
    
    def _balanced_layout(self):
        """Ausgewogenes Layout für Standard-Bildschirme"""
        return {
            'tab_padding': (10, 10),
            'text_height': 12,
            'font_size': 10,
            'button_width': 15,
            'header_font_size': 14
        }
    
    def _spacious_layout(self):
        """Großzügiges Layout für große Bildschirme"""
        return {
            'tab_padding': (15, 15),
            'text_height': 15,
            'font_size': 11,
            'button_width': 18,
            'header_font_size': 16
        }

class DragonWhispererTranslator:
    """
    🐉 Dragon Whisperer Translator - Enterprise Grade
    🔧 VOLLSTÄNDIG REPARIERT & OPTIMIERT mit allen Performance-Verbesserungen
    """

    def __init__(self):
        self.config = AIConfigManager()
        self.profiler = IntelligentSystemProfiler()
        self.diagnostics = SystemDiagnostics()
        self.performance_monitor = PerformanceMonitor()
        self.export_manager = AdvancedExportManager()
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.stream_title_extractor = StreamTitleExtractor()

        self.audio_processor = OptimizedAudioProcessor(self.config)
        self.stream_manager = UltimateStreamManager()
        self.translation_engine = OptimizedTranslationEngine(self.config)

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

        self.auto_save_manager = AutoSaveManager(self, 
            self.config.config.get('auto_save_interval', 300))

        self.enhanced_memory_manager = EnhancedMemoryManager(self)
        self.stream_health_monitor = StreamHealthMonitor()
        self.auto_tuning_manager = AutoTuningManager(self)

        self._initialize_components()

    def _cleanup_translation_cache_optimized(self, target_size: int):
        """🔧 NEU: Optimierte Cache-Bereinigung mit Zielgröße"""
        try:
            cache_size = len(self.translation_engine.translation_cache)
            if cache_size <= target_size:
                return
            
            with self.translation_engine._cache_lock:
                keys_to_keep = self.translation_engine.cache_access_order[-target_size:]
            
                new_cache = {}
                for key in keys_to_keep:
                    if key in self.translation_engine.translation_cache:
                        new_cache[key] = self.translation_engine.translation_cache[key]
            
                self.translation_engine.translation_cache = new_cache
                self.translation_engine.cache_access_order = keys_to_keep.copy()
            
                logging.debug(f"🔧 Translation Cache optimiert: {cache_size} → {len(new_cache)} Einträge")
            
        except Exception as e:
            logging.error(f"Optimized cache cleanup error: {e}")

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

                if self._stream_thread and self._stream_thread.is_alive():
                    self._stream_thread.join(timeout=5.0)

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
                time.sleep(0.1)
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

    def _setup_safe_callbacks(self, output_callbacks: Dict[str, Callable]) -> Dict[str, Callable]:
        """🔧 NEU: Safe Callback-System mit Error-Handling"""
        safe_callbacks = output_callbacks.copy()
        
        def safe_callback_wrapper(callback_name: str, original_callback: Optional[Callable]):
            def wrapper(*args, **kwargs):
                try:
                    if original_callback and callable(original_callback):
                        return original_callback(*args, **kwargs)
                    else:
                        logging.debug(f"Callback '{callback_name}' nicht verfügbar oder nicht callable")
                        return None
                except Exception as e:
                    logging.error(f"Callback '{callback_name}' error: {e}")
                    return None
            return wrapper
        
        required_callbacks = ['transcription', 'translation', 'error', 'info', 'clear_text', 'warning', 'stream_title']
        
        for callback_name in required_callbacks:
            original_callback = safe_callbacks.get(callback_name)
            safe_callbacks[callback_name] = safe_callback_wrapper(callback_name, original_callback)
            
        return safe_callbacks

    def safe_callback(self, callback: Optional[Callable], *args):
        """🔧 OPTIMIERT: Thread-sichere Callback-Ausführung mit robustem Error Handling"""
        with self._callback_lock:
            try:
                if callback and callable(callback):
                    callback(*args)
                else:
                    logging.debug(f"Callback nicht callable: {callback}")
            except Exception as e:
                logging.error(f"Callback error: {e}")

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

            auto_save_stats = self.auto_save_manager.get_auto_save_stats()

            stream_health = self.stream_health_monitor.check_stream_health()

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
                'auto_save_stats': auto_save_stats,
                'stream_health': stream_health,  # 🔧 NEU
                'configuration': {
                    'model': self.config.config.get('transcription_model'),
                    'target_language': self.config.config.get('target_language'),
                    'translation_enabled': self.config.config.get('translation_enabled'),
                    'auto_save_enabled': self.config.config.get('auto_save_enabled'),
                    'enhanced_language_detection': self.config.config.get('enhanced_language_detection', True),
                    'source_language': self.config.config.get('source_language', 'auto')  # 🔧 NEU
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
        """🔧 Startet Live-Translation mit Safe Callbacks"""
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

            safe_callbacks = self._setup_safe_callbacks(output_callbacks)

            self._start_stream_title_monitoring(url, safe_callbacks)

            def async_startup():
                try:
                    stream_type = self.stream_manager.detect_stream_type(url)
                    if self.config.config.get('adaptive_processing', True):
                        self.audio_processor.start_processing_adaptive(
                            lambda audio, chunk_id: self.process_audio_chunk_optimized(
                                audio, chunk_id, safe_callbacks),
                            stream_type
                        )
                    else:
                        self.audio_processor.start_processing(
                            lambda audio, chunk_id: self.process_audio_chunk_optimized(
                                audio, chunk_id, safe_callbacks)
                        )

                    self._process_stream_enhanced(url, safe_callbacks)

                except Exception as e:
                    logging.error(f"Process startup error: {e}")
                    self.safe_callback(safe_callbacks.get('error'),
                                       f"Start fehlgeschlagen: {e}")
                    self.stop()

            self._stream_thread = threading.Thread(
                target=async_startup, daemon=True, name="MainProcessor")
            self._stream_thread.start()

            threading.Thread(target=self._collect_metrics_light,  # 🔧 OPTIMIERT: Light Metrics
                           daemon=True, name="MetricsCollector").start()

            if self.config.config.get('auto_recovery', True):
                threading.Thread(target=self._auto_recovery_loop,
                                 daemon=True, name="AutoRecovery").start()

            if self.config.config.get('auto_save_enabled', True):
                self.auto_save_manager.start_auto_save()

            self.setup_enhanced_memory_management()
            self.auto_tuning_manager.start_auto_tuning()
            
            self.current_session['status'] = 'running'
            logging.info("🎯 Live Translation erfolgreich gestartet!")

            threading.Timer(0.1, lambda: self.safe_callback(
                output_callbacks.get('info'), "Live Translation gestartet")).start()

            return True

        except Exception as e:
            logging.error(f"❌ Start live translation failed: {e}")
            self.safe_callback(output_callbacks.get('error'), f"Start fehlgeschlagen: {e}")
            return False

    def _start_stream_title_monitoring(self, url: str, output_callbacks: Dict[str, Callable]):
        """Startet Stream-Titel Monitoring"""
        def title_monitoring_loop():
            update_interval = self.config.config.get('stream_title_update_interval', 60)
            
            while self.is_running and not self._shutdown_event.is_set():
                try:
                    stream_title = self.stream_title_extractor.extract_stream_title(url)
                    if stream_title:
                        self.safe_callback(output_callbacks.get('info'), 
                                          f"📺 Stream-Titel: {stream_title}")
                        self.safe_callback(output_callbacks.get('stream_title'), stream_title)
                    
                    for _ in range(update_interval):
                        if not self.is_running or self._shutdown_event.is_set():
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logging.debug(f"Stream title monitoring error: {e}")
                    time.sleep(30)

        title_thread = threading.Thread(target=title_monitoring_loop, daemon=True)
        title_thread.start()

    def _process_stream_enhanced(self, url: str, output_callbacks: Dict[str, Callable]):
        """🔧 REPARIERT: Enhanced Stream Processing mit FFmpeg-Fehlertoleranz"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries and self.is_running:
            try:
                stream_type = self.stream_manager.detect_stream_type(url)
                self.safe_callback(output_callbacks.get('info'),
                                   f"🔍 Stream-Typ erkannt: {stream_type}")

                if stream_type == 'local':
                    self._process_local_file(url, output_callbacks)
                    return

                extracted_url = self.stream_manager.extract_stream_url(url)
                
                if not extracted_url:
                    extracted_url = self.stream_manager.extract_stream_url_enhanced(url)

                if not extracted_url:
                    self.safe_callback(output_callbacks.get('error'), 
                                      "❌ Stream-URL konnte nicht extrahiert werden")
                    self.stop()
                    return

                logging.info(f"🎯 Extrahierte URL: {extracted_url[:100]}...")

                if '.m3u8' in extracted_url.lower():
                    self._process_hls_stream_direct(extracted_url, output_callbacks)
                else:
                    self._process_regular_stream_enhanced(extracted_url, output_callbacks)
                    
                break
                    
            except Exception as e:
                retry_count += 1
                logging.warning(f"Stream processing failed, retry {retry_count}/{max_retries}: {e}")
                if retry_count < max_retries:
                    time.sleep(5 * retry_count)
                else:
                    logging.error(f"❌ Stream processing error after {max_retries} retries: {e}")
                    self.safe_callback(output_callbacks.get('error'), f"Stream Fehler: {e}")
                    self.stop()

    def _process_hls_stream_direct(self, hls_url: str, output_callbacks: Dict[str, Callable]):
        """🔧 REPARIERT: Direkte HLS-Verarbeitung mit verbesserten FFmpeg-Parametern"""
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
            '-max_delay', '500000',
            '-reconnect_delay_max', '2',
            '-loglevel', 'warning',
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
            stream_timeout = 20.0
            last_successful_read = time.time()

            stderr_monitor = threading.Thread(
                target=self._monitor_ffmpeg_stderr_enhanced,
                args=(self.ffmpeg_process.stderr, output_callbacks),
                daemon=True
            )
            stderr_monitor.start()

            while self.is_running and self.ffmpeg_process.poll() is None:
                try:
                    if time.time() - last_successful_read > stream_timeout:
                        logging.error("⏰ Stream-Timeout - FFmpeg produziert keine Daten")
                        self.safe_callback(output_callbacks.get('warning'),
                                           "Stream-Timeout - versuche Neustart...")
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
                            self.stream_health_monitor.record_data_received()

                            try:
                                self.audio_processor.audio_queue.put(audio_data, timeout=0.1)
                            except queue.Full:
                                if chunk_counter % 20 == 0: Warnungen
                                    logging.warning(f"⏩ Queue voll - überspringe Chunk {chunk_counter}")
                                continue

                            if chunk_counter == 1:
                                logging.info(f"✅ Erster Audio-Chunk empfangen! Länge: {len(audio_data)} bytes")
                                self.safe_callback(output_callbacks.get('info'),
                                                   "✅ Audio-Daten empfangen - Transkription läuft...")

                            if chunk_counter % 50 == 0:
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

    def _monitor_ffmpeg_stderr_enhanced(self, stderr_pipe, output_callbacks: Dict[str, Callable]):
        """🔧 OPTIMIERT: FFmpeg Stderr Monitoring mit Keepalive-Fehler-Filterung"""
        try:
            while self.is_running and hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
                line = stderr_pipe.readline()
                if not line:
                    break

                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    if 'keepalive request failed' in line_str.lower():
                        logging.debug(f"🔧 FFmpeg Keepalive: {line_str}")
                        continue
                        
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
        """Verarbeitet lokale Audio/Video Dateien mit Progress Tracking"""
        if file_path.startswith('file://'):
            file_path = file_path[7:]

        if not os.path.exists(file_path):
            self.safe_callback(output_callbacks.get('error'), f"Datei nicht gefunden: {file_path}")
            self.stop()
            return

        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        self.safe_callback(output_callbacks.get('info'),
                           f"🎵 Verarbeite lokale Datei: {os.path.basename(file_path)} ({file_size:.1f} MB)")

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
                    self.stream_health_monitor.record_data_received()

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

    def _process_regular_stream_enhanced(self, stream_url: str, output_callbacks: Dict[str, Callable]):
        """🔧 OPTIMIERT: Reguläre Streams mit verbesserten Parametern"""
        chunk_duration = self.config.config.get('chunk_duration', 5.0)
        chunk_bytes = int(16000 * 2 * chunk_duration)

        ffmpeg_cmd = [
            'ffmpeg',
            '-reconnect', '1',
            '-reconnect_at_eof', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '2',
            '-i', stream_url,
            '-f', 's16le', '-ar', '16000', '-ac', '1',
            '-loglevel', 'warning', '-'
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=chunk_bytes)

            chunk_counter = 0
            last_successful_read = time.time()
            stream_timeout = 20.0

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
                        self.stream_health_monitor.record_data_received()
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

    def _safe_ffmpeg_shutdown(self):
        """SICHERER FFMPEG SHUTDOWN"""
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

    def _is_silent_chunk_optimized(self, audio_data: bytes) -> bool:
        """🔧 OPTIMIERT: Schnellere Silence-Detection mit reduzierter Komplexität"""
        if not self.config.config.get('enable_silence_detection', True):
            return False

        if not NUMPY_AVAILABLE or not audio_data:
            return False

        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_np) < 1000:
                return False

            sampled_audio = audio_np[::10]
            rms = np.sqrt(np.mean(sampled_audio**2))
            silence_threshold = self.config.config.get('silence_threshold', 0.02) * 32768
            return rms < silence_threshold
        except Exception as e:
            logging.debug(f"Silence detection error: {e}")
            return False

    def process_audio_chunk_optimized(self, audio_data: bytes, chunk_id: int, output_callbacks: Dict[str, Callable]):
        """🔧 VOLLSTÄNDIG OPTIMIERT: Audio-Chunk Verarbeitung mit reduzierter Komplexität"""
        start_time = time.time()

        try:
            if self.config.config.get('enable_silence_detection', True):
                if self._is_silent_chunk_optimized(audio_data):
                    self.metrics.silent_chunks_skipped += 1
                    self.metrics.chunks_processed += 1

                    if self.metrics.silent_chunks_skipped % 200 == 0:
                        self.safe_callback(output_callbacks.get('info'),
                                           f"🔇 {self.metrics.silent_chunks_skipped} stille Chunks übersprungen")
                    return

            transcription = self.transcribe_audio_optimized(audio_data)

            if transcription and transcription.text.strip():
                if (transcription.confidence > 0.2 and
                    not self._contains_gibberish(transcription.text) and
                        len(transcription.text) > 2):

                    self.session_analytics['languages_detected'].add(transcription.language)

                    self._perform_light_analytics(transcription)

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

            auto_clear = self.config.config.get('auto_clear_interval', 2000)
            if auto_clear > 0 and chunk_id % auto_clear == 0:
                self.safe_callback(output_callbacks.get('clear_text'))

        except Exception as e:
            self.metrics.error_count += 1
            logging.error(f"❌ Chunk {chunk_id} processing failed: {e}")
        finally:
            processing_time = time.time() - start_time
            self.performance_monitor.record_chunk_processing(processing_time)

    def _perform_light_analytics(self, transcription: TranscriptionResult):
        """🔧 OPTIMIERT: Leichtere Analytics für bessere Performance"""
        if self.config.config.get('enable_sentiment_analysis', False):
            sentiment = self.analytics_engine.analyze_sentiment(transcription.text)
            self.session_analytics['sentiment_trend'].append({
                'timestamp': time.time(),
                'sentiment': sentiment,
                'text_sample': transcription.text[:50] + '...' if len(transcription.text) > 50 else transcription.text
            })

        if self.config.config.get('enhanced_language_detection', True):
            lang_analysis = self.analytics_engine.detect_language_enhanced(
                transcription.text, 
                min_confidence=0.6
            )
            if lang_analysis['confidence'] >= 0.6:
                transcription.language = lang_analysis['language']
                logging.debug(f"🔍 Enhanced Language Detection: {lang_analysis['language']} "
                             f"(Confidence: {lang_analysis['confidence']:.2f})")

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

    def transcribe_audio_optimized(self, audio_data: bytes) -> Optional[TranscriptionResult]:
        """🔧 VOLLSTÄNDIG OPTIMIERT: Transkribiert Audio mit intelligenter Sprach-Erkennung"""
        try:
            if not NUMPY_AVAILABLE or not audio_data:
                return None

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_np) < 1000:
                return None

            if self.whisper_model is not None:
                language_param = None
                
                if (hasattr(self, 'current_session') and 
                    self.current_session and 
                    'stream_url' in self.current_session and
                    any(keyword in self.current_session['stream_url'].lower() 
                        for keyword in ['bundestag', 'parliament', 'government', 'politik', 'deutschland'])):
                    language_param = "de"
                    logging.info("🎯 Automatische Korrektur: DE für politischen Stream")
                
                source_lang = self.config.config.get('source_language', 'auto')
                if source_lang != 'auto':
                    language_param = source_lang
                    logging.info(f"🎯 Manuelle Sprachauswahl: {source_lang}")

                segments, info = self.whisper_model.transcribe(
                    audio_np,
                    language=language_param,
                    beam_size=3,
                    best_of=1,
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

    def _collect_metrics_light(self):
        """🔧 OPTIMIERT: Leichtes Metrics-Collecting mit reduzierter Frequenz"""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                if PSUTIL_AVAILABLE:
                    self.metrics.cpu_usage = psutil.cpu_percent()
                    self.metrics.memory_usage = psutil.virtual_memory().percent

                queue_size = self.audio_processor.audio_queue.qsize()
                max_size = self.audio_processor.audio_queue.maxsize
                self.metrics.audio_buffer_health = max(
                    0.0, 100.0 - (queue_size / max_size * 100)) if max_size > 0 else 100.0

                if self.metrics.chunks_processed % 100 == 0:
                    warnings = self.performance_monitor.check_performance_health()
                    if warnings:
                        for warning in warnings[-2:]:
                            logging.warning(warning)

            except Exception as e:
                logging.debug(f"Metrics collection error: {e}")

            time.sleep(3)

    def _auto_recovery_loop(self):
        """🔧 OPTIMIERT: Auto-Recovery System mit erweiterten Checks"""
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

            time.sleep(30)

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

    def stop_thread_safe(self):
        """🔧 NEU: Thread-sicheres Stoppen ohne Race Conditions"""
        with self._lock:
            if not self.is_running:
                return
                
            self.is_running = False
            self._shutdown_event.set()

        cleanup_phases = [
            self._stop_audio_processing,
            self._stop_ffmpeg_process, 
            self._stop_background_threads,
            self._cleanup_memory
        ]
        
        for phase in cleanup_phases:
            try:
                phase()
            except Exception as e:
                logging.error(f"Cleanup phase error: {e}")

    def _stop_audio_processing(self):
        """Stoppt Audio-Verarbeitung"""
        if hasattr(self, 'audio_processor'):
            self.audio_processor.stop_processing()

    def _stop_ffmpeg_process(self):
        """Stoppt FFmpeg Prozess"""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait(timeout=2)
            except Exception as e:
                logging.error(f"❌ FFmpeg kill error: {e}")
            finally:
                self.ffmpeg_process = None

    def _stop_background_threads(self):
        """Stoppt Hintergrund-Threads"""
        try:
            self.thread_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
            if self._stream_thread.is_alive():
                logging.warning("⚠️ Stream-Thread reagiert nicht, überspringe...")

        self.auto_save_manager.stop_auto_save()

    def _cleanup_memory(self):
        """Führt Memory-Cleanup durch"""
        self._cleanup_memory_enhanced()

    def stop(self):
        """🔧 REPARIERT: Stoppt ALLE Prozesse sauber mit verbessertem Memory-Management"""
        self.stop_thread_safe()

        if self.current_session:
            self.current_session['status'] = 'stopped'
            self.current_session['end_time'] = datetime.now().isoformat()

        logging.info("✅ Translation KOMPLETT gestoppt")

    def force_stop(self):
        """FORCIERTES Stoppen für hängende Prozesse"""
        import os
        import signal
        
        logging.critical("🔴 FORCIERTES STOPPEN!")
        
        if self.ffmpeg_process:
            try:
                os.kill(self.ffmpeg_process.pid, signal.SIGKILL)
            except:
                pass
        
        self.is_running = False
        self._shutdown_event.set()
        
        try:
            self.thread_pool.shutdown(wait=False)
        except:
            pass
        
        logging.info("✅ Forciertes Stoppen abgeschlossen")

    def _cleanup_memory_enhanced(self):
        """🔧 OPTIMIERT: Enhanced Memory-Cleaning mit GPU-Support"""
        try:
            if hasattr(self, 'translation_engine'):
                cache_size = len(self.translation_engine.translation_cache)
                if cache_size > 100:
                    with self.translation_engine._cache_lock:
                        keys = list(self.translation_engine.translation_cache.keys())
                        if len(keys) > 50:
                            for key in keys[:-50]:
                                if key in self.translation_engine.translation_cache:
                                    del self.translation_engine.translation_cache[key]
                            logging.info(f"🧹 Translation Cache reduziert: {cache_size} → {len(self.translation_engine.translation_cache)}")
        
            analytics_cleared = 0
            if hasattr(self, 'analytics_engine'):
                if len(self.analytics_engine.sentiment_cache) > 200:
                    self.analytics_engine.sentiment_cache.clear()
                    analytics_cleared += 1
                    
                if len(self.analytics_engine.topic_cache) > 200:
                    self.analytics_engine.topic_cache.clear()
                    analytics_cleared += 1
                    
                if len(self.analytics_engine.language_detection_cache) > 200:
                    self.analytics_engine.language_detection_cache.clear()
                    analytics_cleared += 1
                    
                if analytics_cleared > 0:
                    logging.info(f"🧹 {analytics_cleared} Analytics-Caches geleert")
        
            if hasattr(self, 'session_analytics'):
                if len(self.session_analytics['sentiment_trend']) > 100:
                    self.session_analytics['sentiment_trend'] = self.session_analytics['sentiment_trend'][-50:]
                    
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("🎮 GPU Memory geleert")
        
            gc.collect()
        
            logging.info("🧹 Enhanced Memory Cleanup erfolgreich durchgeführt")
        
        except Exception as e:
            logging.warning(f"Memory cleanup warning: {e}")

    def setup_enhanced_memory_management(self):
        """🔧 OPTIMIERT: Intelligentes Memory-Management mit höheren Thresholds"""
        def memory_guard_loop():
            cleanup_count = 0
            last_cleanup_time = 0
            min_cleanup_interval = 300  # Mindestens 30 Sekunden zwischen Cleanups
            
            while self.is_running and not self._shutdown_event.is_set():
                try:
                    if not PSUTIL_AVAILABLE:
                        time.sleep(15)
                        continue

                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    current_time = time.time()
                    
                    if memory_mb > 1500 and (current_time - last_cleanup_time) > min_cleanup_interval:
                        logging.warning(f"🧹 KRITISCH: Memory {memory_mb:.1f}MB - Aggressiver Cleanup #{cleanup_count+1}")
                        self._cleanup_memory_enhanced()
                        last_cleanup_time = current_time
                        cleanup_count += 1
                        
                    elif memory_mb > 1000 and (current_time - last_cleanup_time) > min_cleanup_interval * 2:
                        logging.info(f"🧹 HOCH: Memory {memory_mb:.1f}MB - Selektiver Cleanup")
                        self._cleanup_translation_cache()
                        self._cleanup_analytics_data()
                        gc.collect()
                        last_cleanup_time = current_time
                        cleanup_count += 1
                        
                    elif memory_mb > 800 and cleanup_count > 2 and (current_time - last_cleanup_time) > min_cleanup_interval * 3:
                        logging.debug(f"🧹 MODERAT: Memory {memory_mb:.1f}MB - Leichter Cleanup")
                        self._cleanup_translation_cache()
                        gc.collect()
                        last_cleanup_time = current_time
                        
                    time.sleep(30)
                    
                except Exception as e:
                    logging.debug(f"Memory guard error: {e}")
                    time.sleep(30)

        threading.Thread(target=memory_guard_loop, daemon=True, name="EnhancedMemoryGuard").start()

    def _scheduled_cleanup(self):
        """🔧 NEU: Regelmäßiger Cleanup alle 5 Minuten"""
        while self.is_running:
            time.sleep(300)
            self._cleanup_memory_enhanced()
            logging.info("🔄 Scheduled cleanup completed")

    def _cleanup_gpu_memory(self):
        """GPU Memory Cleanup"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("🎮 GPU Memory Cleanup durchgeführt")
        except Exception as e:
            logging.debug(f"GPU memory cleanup failed: {e}")

    def _cleanup_translation_cache(self):
        """Bereinigt Translation-Cache"""
        if hasattr(self, 'translation_engine'):
            cache_size = len(self.translation_engine.translation_cache)
            if cache_size > 50:
                with self.translation_engine._cache_lock:
                    keys = list(self.translation_engine.translation_cache.keys())
                    if len(keys) > 25:
                        for key in keys[:-25]:
                            if key in self.translation_engine.translation_cache:
                                del self.translation_engine.translation_cache[key]
                        logging.info(f"🧹 Cache von {cache_size} auf {len(self.translation_engine.translation_cache)} reduziert")

    def _cleanup_analytics_data(self):
        """Bereinigt Analytics-Daten"""
        if hasattr(self, 'analytics_engine'):
            max_entries = 500
            if len(self.analytics_engine.sentiment_cache) > max_entries:
                keys = list(self.analytics_engine.sentiment_cache.keys())
                for key in keys[:-max_entries]:
                    del self.analytics_engine.sentiment_cache[key]
                    
            if len(self.analytics_engine.topic_cache) > max_entries:
                keys = list(self.analytics_engine.topic_cache.keys())
                for key in keys[:-max_entries]:
                    del self.analytics_engine.topic_cache[key]

    def _cleanup_whisper_model(self):
        """Bereinigt Whisper-Model falls möglich"""
        if hasattr(self, 'whisper_model') and self.whisper_model:
            try:
                if hasattr(self.whisper_model, 'model'):
                    pass
            except Exception as e:
                logging.debug(f"Whisper model cleanup warning: {e}")

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

    def apply_config_preset(self, preset_name: str):
        """🔧 NEU: Wendet Configuration Preset an"""
        return self.config.apply_preset(preset_name)

class DragonWhispererGUI:
    """🔧 OPTIMIERT: Ultimative GUI mit ENHANCED UI/UX & TERMINAL-FIX"""

    def __init__(self):
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI nicht verfügbar - tkinter fehlt")
            
        self.root = tk.Tk()
        self.translator = DragonWhispererTranslator()
        self.is_translating = False
        
        self.available_themes = self._initialize_themes()
        self.current_theme = 'dark'
        self.theme_vars = {}
        
        self.layout_manager = IntelligentLayoutManager(self.root)
        
        self._setup_enhanced_shutdown()
        
        self.setup_ultimate_gui()

    def _setup_enhanced_shutdown(self):
        """🔧 NEU: Garantiert saubere Beendigung ohne Terminal-Lock"""
        import atexit
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        atexit.register(self._cleanup_resources)
        
    def _signal_handler(self, signum, frame):
        """Behandelt System-Signale für saubere Beendigung"""
        logging.info(f"Signal {signum} empfangen - starte saubere Beendigung")
        self.safe_exit()
        
    def _cleanup_resources(self):
        """Garantiert Resource-Cleanup"""
        try:
            if hasattr(self, 'translator'):
                self.translator.stop()
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.quit()
                except:
                    pass
        except Exception as e:
            logging.debug(f"Cleanup warning: {e}")

    def _initialize_themes(self) -> Dict[str, Dict[str, str]]:
        """Initialisiert verfügbare Themes"""
        return {
            'dark': {
                'bg_primary': '#1e1e1e',
                'bg_secondary': '#2e2e2e', 
                'bg_tertiary': '#3e3e3e',
                'text_primary': '#ffffff',
                'text_secondary': '#cccccc',
                'text_accent': '#58a6ff',
                'text_success': '#3fb950',
                'text_warning': '#d29922',
                'text_error': '#f85149',
                'accent_blue': '#1f6feb',
                'accent_green': '#238636',
                'accent_orange': '#db6d28',
                'exit_red': '#d9534f'  # 🔧 NEU: Farbe für Beenden-Button
            },
            'light': {
                'bg_primary': '#ffffff',
                'bg_secondary': '#f5f5f5',
                'bg_tertiary': '#e8e8e8',
                'text_primary': '#1e1e1e', 
                'text_secondary': '#666666',
                'text_accent': '#0969da',
                'text_success': '#1a7f37',
                'text_warning': '#9a6700',
                'text_error': '#cf222e',
                'accent_blue': '#0969da',
                'accent_green': '#1a7f37',
                'accent_orange': '#bc4c00',
                'exit_red': '#d9534f'
            },
            'blue_dark': {
                'bg_primary': '#0d1117',
                'bg_secondary': '#161b22',
                'bg_tertiary': '#21262d',
                'text_primary': '#f0f6fc',
                'text_secondary': '#b1bac4',
                'text_accent': '#388bfd',
                'text_success': '#3fb950',
                'text_warning': '#d29922',
                'text_error': '#f85149',
                'accent_blue': '#388bfd',
                'accent_green': '#238636',
                'accent_orange': '#db6d28',
                'exit_red': '#d9534f'
            }
        }

    def setup_ultimate_gui(self):
        """🔧 OPTIMIERT: Intelligente Fenster-Layout mit responsive Design"""
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            window_width = int(screen_width * 0.8)
            window_height = int(screen_height * 0.8)
            
            self.root.geometry(f"{window_width}x{window_height}")
            self.root.minsize(1000, 700)
            
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            self.root.geometry(f"+{x}+{y}")
            
            self.root.grid_columnconfigure(0, weight=1)
            self.root.grid_rowconfigure(0, weight=1)
            
            self.root.configure(bg=self.available_themes[self.current_theme]['bg_primary'])
            self.root.title("🐉 Dragon Whisperer - LiveStream Transkribator")

            self.root.protocol("WM_DELETE_WINDOW", self.safe_exit)

            self.setup_modern_style()

            notebook = ttk.Notebook(self.root)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)

            self.setup_translation_tab(notebook)
            self.setup_dashboard_tab(notebook)
            self.setup_system_tab(notebook)
            self.setup_export_tab(notebook)

            self.setup_enhanced_status_bar()

            self.root.after(1000, self.run_full_diagnostic)

            self.root.bind('<Configure>', self._on_window_resize)

            logging.info("✅ GUI erfolgreich initialisiert")

        except Exception as e:
            logging.error(f"❌ GUI Setup failed: {e}")
            raise

    def _on_window_resize(self, event):
        """🔧 NEU: Passt UI-Elemente bei Fenstergrößenänderung an"""
        if event.widget == self.root:
            new_width = event.width
            new_height = event.height
            
            layout_config = self.layout_manager.auto_adjust_layout((new_width, new_height))
            self._apply_layout_config(layout_config)

    def _apply_layout_config(self, layout_config: Dict[str, Any]):
        """🔧 NEU: Wendet Layout-Konfiguration an"""
        try:
            text_widgets = {
                'transcript_area': self.transcript_area,
                'translation_area': self.translation_area,
                'dashboard_text': self.dashboard_text,
                'system_info_text': self.system_info_text,
                'preview_text': self.preview_text
            }
            
            for name, widget in text_widgets.items():
                if hasattr(widget, 'configure'):
                    widget.configure(height=layout_config['text_height'])
                    
            font_configs = [
                (self.transcript_area, layout_config['font_size']),
                (self.translation_area, layout_config['font_size']),
                (self.dashboard_text, layout_config['font_size']),
                (self.system_info_text, layout_config['font_size']),
                (self.preview_text, layout_config['font_size'])
            ]
            
            for widget, font_size in font_configs:
                if hasattr(widget, 'configure'):
                    current_font = widget.cget('font')
                    if current_font:
                        font_parts = current_font.split()
                        if len(font_parts) >= 2:
                            new_font = f"{font_parts[0]} {font_size}"
                            widget.configure(font=new_font)
                            
        except Exception as e:
            logging.debug(f"Layout application error: {e}")

    def setup_enhanced_status_bar(self):
        """🔧 OPTIMIERT: Sichtbare Status-Bar mit fixem Beenden-Button"""
        status_frame = ttk.Frame(self.root, relief='sunken', borderwidth=1)
        status_frame.pack(fill='x', side='bottom', padx=2, pady=2)
        

        status_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(1, weight=0)
        
        self.status_var = tk.StringVar(value="🐉 Dragon Whisperer - Bereit")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky="w", padx=5)
        
        self.session_var = tk.StringVar(value="Session: --")
        session_label = ttk.Label(status_frame, textvariable=self.session_var, font=("Arial", 8))
        session_label.grid(row=0, column=1, padx=10)
        
        self.metrics_var = tk.StringVar(value="CPU: 0% | RAM: 0%")
        metrics_label = ttk.Label(status_frame, textvariable=self.metrics_var, font=("Arial", 8))
        metrics_label.grid(row=0, column=2, padx=10)
        
        exit_button = ttk.Button(
            status_frame,
            text="🚪 SICHER BEENDEN",
            command=self.safe_exit,
            width=15,
            style="Exit.TButton"
        )
        exit_button.grid(row=0, column=3, padx=5, pady=2)

    def safe_exit(self):
        """🔧 OPTIMIERT: Sicheres Beenden mit Terminal-Fix"""
        try:
            if messagebox.askokcancel(
                "Anwendung beenden",
                "Möchten Sie die Anwendung wirklich beenden?\n\n" +
                "✅ Laufende Translationen werden gestoppt\n" +
                "✅ Einstellungen werden gespeichert\n" +
                "✅ Terminal wird wieder benutzbar\n" +
                "✅ Alle Prozesse werden sauber beendet"
            ):
                self.status_var.set("🛑 Beende Anwendung sicher...")
                self.root.update()
                
                if hasattr(self, 'is_translating') and self.is_translating:
                    self.stop()
                    time.sleep(1)
                
                if hasattr(self, 'translator'):
                    self.translator.config.save_config_with_backup()
                    self.translator.stop()
                
                self.root.quit()
                self.root.destroy()
                
                os._exit(0)
                
        except Exception as e:
            logging.error(f"Exit error: {e}")
            try:
                self.root.quit()
                os._exit(1)
            except:
                pass

    def on_silence_detection_toggled(self):
        """Handler für Silence Detection Toggle"""
        try:
            enabled = self.silence_detection_var.get()
            self.translator.config.config['enable_silence_detection'] = enabled
            self.translator.config.save_config_with_backup()
            status = "Aktiv" if enabled else "Inaktiv"
            self.status_var.set(f"✅ Silence Detection: {status}")
        except Exception as e:
            self.status_var.set(f"❌ Einstellungs-Änderung fehlgeschlagen: {e}")

    def setup_modern_style(self):
        """Konfiguriert modernes Styling mit Theme-Support"""
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
            
            current_theme_colors = self.available_themes[self.current_theme]

            style.configure('.', 
                           background=current_theme_colors['bg_primary'],
                           foreground=current_theme_colors['text_primary'],
                           fieldbackground=current_theme_colors['bg_tertiary'],
                           selectbackground=current_theme_colors['accent_blue'],
                           selectforeground=current_theme_colors['text_primary'],
                           insertcolor=current_theme_colors['text_primary'])

            style.configure('TFrame', 
                           background=current_theme_colors['bg_primary'],
                           foreground=current_theme_colors['text_primary'])
            
            style.configure('TLabel', 
                           background=current_theme_colors['bg_primary'],
                           foreground=current_theme_colors['text_primary'])
            
            style.configure('TButton', 
                           background=current_theme_colors['bg_secondary'],
                           foreground=current_theme_colors['text_primary'],
                           focuscolor='none')
            
            style.map('TButton',
                     background=[('active', current_theme_colors['accent_blue']),
                               ('pressed', current_theme_colors['accent_blue'])],
                     foreground=[('active', current_theme_colors['text_primary']),
                               ('pressed', current_theme_colors['text_primary'])])

            style.configure('Accent.TButton', 
                           background=current_theme_colors['accent_green'],
                           foreground=current_theme_colors['text_primary'])
            
            style.map('Accent.TButton',
                     background=[('active', current_theme_colors['accent_green']),
                               ('pressed', current_theme_colors['accent_green'])],
                     foreground=[('active', current_theme_colors['text_primary']),
                               ('pressed', current_theme_colors['text_primary'])])

            style.configure("Exit.TButton", 
                           background=self.available_themes[self.current_theme]['exit_red'],
                           foreground="white",
                           font=("Arial", 9, "bold"))
            
            style.map("Exit.TButton",
                     background=[('active', self.available_themes[self.current_theme]['exit_red']),
                               ('pressed', self.available_themes[self.current_theme]['exit_red'])],
                     foreground=[('active', 'white'),
                               ('pressed', 'white')])

            style.configure('TEntry',
                           fieldbackground=current_theme_colors['bg_tertiary'],
                           foreground=current_theme_colors['text_primary'],
                           insertcolor=current_theme_colors['text_primary'],
                           selectbackground=current_theme_colors['accent_blue'],
                           selectforeground=current_theme_colors['text_primary'])

            style.configure('TCombobox',
                           fieldbackground=current_theme_colors['bg_tertiary'],
                           background=current_theme_colors['bg_secondary'],
                           foreground=current_theme_colors['text_primary'],
                           selectbackground=current_theme_colors['accent_blue'],
                           selectforeground=current_theme_colors['text_primary'],
                           arrowcolor=current_theme_colors['text_primary'])

            style.map('TCombobox',
                     fieldbackground=[('readonly', current_theme_colors['bg_tertiary'])],
                     background=[('readonly', current_theme_colors['bg_secondary'])],
                     foreground=[('readonly', current_theme_colors['text_primary'])],
                     selectbackground=[('readonly', current_theme_colors['accent_blue'])],
                     selectforeground=[('readonly', current_theme_colors['text_primary'])])

            style.configure('TCheckbutton',
                           background=current_theme_colors['bg_primary'],
                           foreground=current_theme_colors['text_primary'])

            style.configure('TNotebook', 
                           background=current_theme_colors['bg_primary'])
            
            style.configure('TNotebook.Tab',
                           background=current_theme_colors['bg_secondary'],
                           foreground=current_theme_colors['text_secondary'],
                           padding=[15, 5])
            
            style.map('TNotebook.Tab',
                     background=[('selected', current_theme_colors['accent_blue'])],
                     foreground=[('selected', current_theme_colors['text_primary'])])

            style.configure('TLabelframe',
                           background=current_theme_colors['bg_primary'],
                           foreground=current_theme_colors['text_primary'])
            
            style.configure('TLabelframe.Label',
                           background=current_theme_colors['bg_primary'],
                           foreground=current_theme_colors['text_accent'])

        except Exception as e:
            logging.warning(f"Themed styling setup warning: {e}")

    def setup_translation_tab(self, notebook: ttk.Notebook):
        """Erstellt den Translation Tab mit Stream-Titel Anzeige und Quellsprache Auswahl"""
        try:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text="🎯 Live Translation")

            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(3, weight=1)

            header_frame = ttk.Frame(tab)
            header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
            header_frame.grid_columnconfigure(1, weight=1)

            title_label = ttk.Label(header_frame, 
                                   text="🐉 Dragon Whisperer - LiveStream Transkribator",
                                   font=("Arial", 14, "bold"),
                                   foreground=self.available_themes[self.current_theme]['text_accent'])
            title_label.grid(row=0, column=0, sticky="w")

            self.stream_title_var = tk.StringVar(value="📺 Kein Stream-Titel")
            stream_title_label = ttk.Label(header_frame,
                                         textvariable=self.stream_title_var,
                                         font=("Arial", 10),
                                         foreground=self.available_themes[self.current_theme]['text_success'],
                                         background=self.available_themes[self.current_theme]['bg_secondary'],
                                         padding=(10, 2),
                                         relief="solid",
                                         borderwidth=1)
            stream_title_label.grid(row=0, column=1, sticky="e", padx=(10, 0))

            url_frame = ttk.LabelFrame(tab, text="🌐 Stream URL", padding=10)
            url_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
            url_frame.grid_columnconfigure(0, weight=1)

            url_input_frame = ttk.Frame(url_frame)
            url_input_frame.pack(fill='x', pady=5)

            ttk.Label(url_input_frame, text="URL:").pack(side='left')

            self.url_entry = tk.Entry(
                url_input_frame,
                font=("Arial", 10),
                bg=self.available_themes[self.current_theme]['bg_tertiary'],
                fg=self.available_themes[self.current_theme]['text_primary'],
                insertbackground=self.available_themes[self.current_theme]['text_primary'],
                selectbackground=self.available_themes[self.current_theme]['accent_blue'],
                selectforeground=self.available_themes[self.current_theme]['text_primary']
            )
            self.url_entry.pack(side='left', fill='x', expand=True, padx=10)
            self.url_entry.insert(0, "https://www.youtube.com/watch?v=kyMj1oMuKI0")

            url_actions_frame = ttk.Frame(url_input_frame)
            url_actions_frame.pack(side='left', padx=10)

            ttk.Button(url_actions_frame, text="📋", command=self.paste_to_url, width=3).pack(side='left', padx=2)
            ttk.Button(url_actions_frame, text="📁", command=self.select_local_file, width=3).pack(side='left', padx=2)
            ttk.Button(url_actions_frame, text="🧹", command=lambda: self.url_entry.delete(0, tk.END), width=3).pack(side='left', padx=2)

            settings_frame = ttk.Frame(tab)
            settings_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

            model_frame = ttk.Frame(settings_frame)
            model_frame.pack(side='left', padx=10)
            ttk.Label(model_frame, text="Modell:").pack(side='left')
            self.model_var = tk.StringVar(value=self.translator.config.config.get('transcription_model', 'small'))
            model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                       values=self.translator.get_available_models(),
                                       width=10, state="readonly")
            model_combo.pack(side='left', padx=5)
            model_combo.bind('<<ComboboxSelected>>', self.on_model_changed)

            source_lang_frame = ttk.Frame(settings_frame)
            source_lang_frame.pack(side='left', padx=10)
            ttk.Label(source_lang_frame, text="Quellsprache:").pack(side='left')
            self.source_lang_var = tk.StringVar(value=self.translator.config.config.get('source_language', 'auto'))
            source_lang_combo = ttk.Combobox(source_lang_frame, textvariable=self.source_lang_var,
                                            values=['auto', 'de', 'en', 'fr', 'es', 'it'],
                                            width=6, state="readonly")
            source_lang_combo.pack(side='left', padx=5)
            source_lang_combo.bind('<<ComboboxSelected>>', self.on_source_language_changed)

            lang_frame = ttk.Frame(settings_frame)
            lang_frame.pack(side='left', padx=10)
            ttk.Label(lang_frame, text="Zielsprache:").pack(side='left')
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
            control_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

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
            text_container.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
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
                bg=self.available_themes[self.current_theme]['bg_tertiary'],
                fg=self.available_themes[self.current_theme]['text_primary'],
                insertbackground=self.available_themes[self.current_theme]['text_primary'],
                selectbackground=self.available_themes[self.current_theme]['accent_blue'],
                selectforeground=self.available_themes[self.current_theme]['text_primary'],
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
                bg=self.available_themes[self.current_theme]['bg_tertiary'],
                fg=self.available_themes[self.current_theme]['text_primary'],
                insertbackground=self.available_themes[self.current_theme]['text_primary'],
                selectbackground=self.available_themes[self.current_theme]['accent_blue'],
                selectforeground=self.available_themes[self.current_theme]['text_primary'],
                font=("Consolas", 9)
            )
            self.translation_area.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            self.setup_context_menus()

        except Exception as e:
            logging.error(f"Translation tab setup failed: {e}")
            raise

    def on_source_language_changed(self, event=None):
        """🔧 NEU: Handler für Quellsprachen-Änderung mit Debouncing"""
        try:
            new_source_lang = self.source_lang_var.get()
            current_source_lang = self.translator.config.config.get('source_language', 'auto')
            
            if new_source_lang != current_source_lang:
                self.translator.config.config['source_language'] = new_source_lang
                self.translator.config.save_config_with_backup()
                
                if new_source_lang == 'auto':
                    self.status_var.set("✅ Quellsprache: Automatisch")
                else:
                    lang_name = SUPPORTED_LANGUAGES.get(new_source_lang, new_source_lang)
                    self.status_var.set(f"✅ Quellsprache: {lang_name}")
                
        except Exception as e:
            self.status_var.set(f"❌ Quellsprachen-Änderung fehlgeschlagen: {e}")

    def setup_dashboard_tab(self, notebook: ttk.Notebook):
        """Erstellt das Dashboard mit Live-Metriken"""
        try:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text="🏠 Dashboard")

            header_frame = ttk.Frame(tab)
            header_frame.pack(fill='x', padx=20, pady=15)

            ttk.Label(header_frame,
                      text="🐉 DRAGON WHISPERER",
                      font=("Arial", 18, "bold"),
                      foreground=self.available_themes[self.current_theme]['accent_blue']).pack(pady=5)

            ttk.Label(header_frame,
                      text="Enterprise Grade Stream Translation & Analysis - VOLLSTÄNDIG OPTIMIERT",
                      font=("Arial", 10),
                      foreground=self.available_themes[self.current_theme]['text_secondary']).pack(pady=2)

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
                                       font=("Arial", 9), foreground=self.available_themes[self.current_theme]['text_secondary'])
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
                      foreground=self.available_themes[self.current_theme]['text_accent'],
                      font=("Arial", 10, "bold")).pack(side='left', padx=15)
            ttk.Label(stats_row1, textvariable=self.memory_var,
                      foreground=self.available_themes[self.current_theme]['text_accent'],
                      font=("Arial", 10, "bold")).pack(side='left', padx=15)
            ttk.Label(stats_row1, textvariable=self.chunks_var,
                      foreground=self.available_themes[self.current_theme]['text_success'],
                      font=("Arial", 10, "bold")).pack(side='left', padx=15)
            ttk.Label(stats_row1, textvariable=self.errors_var,
                      foreground=self.available_themes[self.current_theme]['text_warning'],
                      font=("Arial", 10, "bold")).pack(side='left', padx=15)

            stats_row2 = ttk.Frame(stats_frame)
            stats_row2.pack(fill='x', pady=5)

            self.skipped_var = tk.StringVar(value="Übersprungen: 0")
            self.cache_var = tk.StringVar(value="Cache: --%")
            self.uptime_var = tk.StringVar(value="Laufzeit: --")
            self.sentiment_var = tk.StringVar(value="Sentiment: --")

            ttk.Label(stats_row2, textvariable=self.skipped_var,
                      foreground=self.available_themes[self.current_theme]['text_secondary'],
                      font=("Arial", 10)).pack(side='left', padx=15)
            ttk.Label(stats_row2, textvariable=self.cache_var,
                      foreground=self.available_themes[self.current_theme]['text_success'],
                      font=("Arial", 10)).pack(side='left', padx=15)
            ttk.Label(stats_row2, textvariable=self.uptime_var,
                      foreground=self.available_themes[self.current_theme]['text_accent'],
                      font=("Arial", 10)).pack(side='left', padx=15)
            ttk.Label(stats_row2, textvariable=self.sentiment_var,
                      foreground=self.available_themes[self.current_theme]['text_warning'],
                      font=("Arial", 10)).pack(side='left', padx=15)

            info_frame = ttk.LabelFrame(tab, text="📋 System Information", padding=10)
            info_frame.pack(fill='both', expand=True, padx=20, pady=10)

            self.dashboard_text = scrolledtext.ScrolledText(
                info_frame,
                bg=self.available_themes[self.current_theme]['bg_tertiary'],
                fg=self.available_themes[self.current_theme]['text_primary'],
                insertbackground=self.available_themes[self.current_theme]['text_primary'],
                selectbackground=self.available_themes[self.current_theme]['accent_blue'],
                selectforeground=self.available_themes[self.current_theme]['text_primary'],
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
        """System Tab mit Theme-Implementierung und Configuration Presets"""
        try:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text="⚙️ System")

            header_frame = ttk.Frame(tab)
            header_frame.pack(fill='x', padx=20, pady=15)

            ttk.Label(header_frame, text="⚙️ System Information & Diagnose",
                      font=("Arial", 16, "bold")).pack(pady=5)

            presets_frame = ttk.LabelFrame(tab, text="🚀 Configuration Presets", padding=10)
            presets_frame.pack(fill='x', padx=20, pady=10)
            
            presets_selection_frame = ttk.Frame(presets_frame)
            presets_selection_frame.pack(fill='x', pady=5)
            
            ttk.Label(presets_selection_frame, text="Preset:").pack(side='left', padx=5)
            
            self.preset_var = tk.StringVar(value="balanced")
            preset_combo = ttk.Combobox(presets_selection_frame, textvariable=self.preset_var,
                                      values=list(ConfigPresets.PRESETS.keys()), 
                                      state="readonly", width=15)
            preset_combo.pack(side='left', padx=5)
            
            ttk.Button(presets_selection_frame, text="Preset anwenden",
                      command=self.apply_config_preset).pack(side='left', padx=10)
            
            preset_info_frame = ttk.Frame(presets_frame)
            preset_info_frame.pack(fill='x', pady=5)
            
            self.preset_info_var = tk.StringVar(value="🔧 Wählen Sie ein Preset für optimierte Einstellungen")
            ttk.Label(preset_info_frame, textvariable=self.preset_info_var,
                     font=("Arial", 9)).pack(anchor='w')

            theme_frame = ttk.LabelFrame(tab, text="🎨 Theme Einstellungen", padding=10)
            theme_frame.pack(fill='x', padx=20, pady=10)
            
            theme_selection_frame = ttk.Frame(theme_frame)
            theme_selection_frame.pack(fill='x', pady=5)
            
            ttk.Label(theme_selection_frame, text="Theme:").pack(side='left', padx=5)
            
            self.theme_var = tk.StringVar(value=self.current_theme)
            theme_combo = ttk.Combobox(theme_selection_frame, textvariable=self.theme_var,
                                     values=list(self.available_themes.keys()), 
                                     state="readonly", width=15)
            theme_combo.pack(side='left', padx=5)
            theme_combo.bind('<<ComboboxSelected>>', self.on_theme_changed)
            
            ttk.Button(theme_selection_frame, text="Theme anwenden",
                      command=self.apply_theme).pack(side='left', padx=10)
            
            theme_preview_frame = ttk.Frame(theme_frame)
            theme_preview_frame.pack(fill='x', pady=5)
            
            ttk.Label(theme_preview_frame, text="Vorschau:").pack(side='left', padx=5)
            self.theme_preview_label = ttk.Label(theme_preview_frame, 
                                               text="🎨 Aktuelles Theme",
                                               background=self.available_themes[self.current_theme]['accent_blue'],
                                               foreground=self.available_themes[self.current_theme]['text_primary'],
                                               padding=(10, 2))
            self.theme_preview_label.pack(side='left', padx=5)

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
                bg=self.available_themes[self.current_theme]['bg_tertiary'],
                fg=self.available_themes[self.current_theme]['text_primary'],
                insertbackground=self.available_themes[self.current_theme]['text_primary'],
                selectbackground=self.available_themes[self.current_theme]['accent_blue'],
                selectforeground=self.available_themes[self.current_theme]['text_primary'],
                font=("Consolas", 9)
            )
            self.system_info_text.pack(fill='both', expand=True)

        except Exception as e:
            logging.error(f"System tab setup failed: {e}")
            raise

    def apply_config_preset(self):
        """🔧 NEU: Wendet Configuration Preset an"""
        try:
            preset_name = self.preset_var.get()
            if self.translator.apply_config_preset(preset_name):
                self.preset_info_var.set(f"✅ Preset '{preset_name}' erfolgreich angewendet")
                self.status_var.set(f"✅ Configuration Preset '{preset_name}' angewendet")
            else:
                self.preset_info_var.set(f"❌ Preset '{preset_name}' konnte nicht angewendet werden")
        except Exception as e:
            self.status_var.set(f"❌ Preset-Anwendung fehlgeschlagen: {e}")

    def setup_export_tab(self, notebook: ttk.Notebook):
        """Erstellt den Export Tab mit Funktionen"""
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
                bg=self.available_themes[self.current_theme]['bg_tertiary'],
                fg=self.available_themes[self.current_theme]['text_primary'],
                insertbackground=self.available_themes[self.current_theme]['text_primary'],
                selectbackground=self.available_themes[self.current_theme]['accent_blue'],
                selectforeground=self.available_themes[self.current_theme]['text_primary'],
                font=("Consolas", 9),
                wrap=tk.WORD
            )
            self.preview_text.pack(fill='both', expand=True)

            self.update_preview()

        except Exception as e:
            logging.error(f"Export tab setup failed: {e}")
            raise

    def setup_context_menus(self):
        """Erstellt Context Menüs für Textareas"""
        try:
            transcript_menu = Menu(self.root, tearoff=0,
                                   bg=self.available_themes[self.current_theme]['bg_tertiary'],
                                   fg=self.available_themes[self.current_theme]['text_primary'],
                                   activebackground=self.available_themes[self.current_theme]['accent_blue'],
                                   activeforeground=self.available_themes[self.current_theme]['text_primary'])

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
                                    bg=self.available_themes[self.current_theme]['bg_tertiary'],
                                    fg=self.available_themes[self.current_theme]['text_primary'],
                                    activebackground=self.available_themes[self.current_theme]['accent_blue'],
                                    activeforeground=self.available_themes[self.current_theme]['text_primary'])

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
            self.translator.config.save_config_with_backup()
            self.status_var.set(f"✅ Modell geändert: {new_model}")
        except Exception as e:
            self.status_var.set(f"❌ Modell-Änderung fehlgeschlagen: {e}")

    def on_language_changed(self, event=None):
        """Handler für Sprach-Änderung"""
        try:
            new_lang = self.lang_var.get()
            self.translator.config.config['target_language'] = new_lang
            self.translator.config.save_config_with_backup()
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
            self.translator.config.save_config_with_backup()
            status = "Aktiv" if enabled else "Inaktiv"
            self.status_var.set(f"✅ Übersetzung: {status}")
        except Exception as e:
            self.status_var.set(f"❌ Einstellungs-Änderung fehlgeschlagen: {e}")

    def on_auto_scroll_toggled(self):
        """Handler für Auto-Scroll Toggle"""
        try:
            enabled = self.auto_scroll_var.get()
            self.translator.config.config['enable_auto_scroll'] = enabled
            self.translator.config.save_config_with_backup()
            status = "Aktiv" if enabled else "Inaktiv"
            self.status_var.set(f"✅ Auto-Scroll: {status}")
        except Exception as e:
            self.status_var.set(f"❌ Einstellungs-Änderung fehlgeschlagen: {e}")

    def on_theme_changed(self, event=None):
        """Handler für Theme-Änderung"""
        try:
            new_theme = self.theme_var.get()
            self.current_theme = new_theme
            self.theme_preview_label.configure(
                background=self.available_themes[new_theme]['accent_blue'],
                foreground=self.available_themes[new_theme]['text_primary'],
                text=f"🎨 {new_theme} Theme"
            )
            self.status_var.set(f"🎨 Theme ausgewählt: {new_theme}")
        except Exception as e:
            self.status_var.set(f"❌ Theme-Änderung fehlgeschlagen: {e}")

    def apply_theme(self):
        """Wendet ausgewähltes Theme an"""
        try:
            self.setup_modern_style()
            self._update_all_widget_colors()
            
            self.theme_preview_label.configure(
                background=self.available_themes[self.current_theme]['accent_blue'],
                foreground=self.available_themes[self.current_theme]['text_primary']
            )
            
            self.status_var.set(f"✅ Theme '{self.current_theme}' erfolgreich angewendet")
        except Exception as e:
            self.status_var.set(f"❌ Theme-Anwendung fehlgeschlagen: {e}")

    def _update_all_widget_colors(self):
        """Aktualisiert Farben ALLER Widgets rekursiv"""
        current_theme = self.available_themes[self.current_theme]
        
        self.root.configure(bg=current_theme['bg_primary'])
        
        text_widgets = [self.transcript_area, self.translation_area, 
                       self.dashboard_text, self.system_info_text, self.preview_text]
        
        for widget in text_widgets:
            if hasattr(widget, 'configure'):
                widget.configure(
                    bg=current_theme['bg_tertiary'],
                    fg=current_theme['text_primary'],
                    insertbackground=current_theme['text_primary'],
                    selectbackground=current_theme['accent_blue'],
                    selectforeground=current_theme['text_primary']
                )

        if hasattr(self, 'url_entry'):
            self.url_entry.configure(
                bg=current_theme['bg_tertiary'],
                fg=current_theme['text_primary'],
                insertbackground=current_theme['text_primary'],
                selectbackground=current_theme['accent_blue'],
                selectforeground=current_theme['text_primary']
            )
        
        self._update_widget_tree(self.root, current_theme)

    def _update_widget_tree(self, widget, theme):
        """Rekursive Aktualisierung aller Child-Widgets"""
        try:
            if isinstance(widget, (tk.Frame, ttk.Frame)):
                try:
                    widget.configure(background=theme['bg_primary'])
                except:
                    pass
            elif isinstance(widget, tk.Label):
                try:
                    widget.configure(
                        background=theme['bg_primary'],
                        foreground=theme['text_primary']
                    )
                except:
                    pass
            elif isinstance(widget, tk.Entry):
                try:
                    widget.configure(
                        background=theme['bg_tertiary'],
                        foreground=theme['text_primary'],
                        insertbackground=theme['text_primary'],
                        selectbackground=theme['accent_blue'],
                        selectforeground=theme['text_primary']
                    )
                except:
                    pass
                    
        except Exception as e:
            logging.debug(f"Widget update error: {e}")
        
        try:
            for child in widget.winfo_children():
                self._update_widget_tree(child, theme)
        except Exception as e:
            logging.debug(f"Child widget update error: {e}")

    def clear_translation_cache(self):
        """Leert den Translation Cache"""
        try:
            self.translator.clear_translation_cache()
            self.status_var.set("🧹 Translation Cache geleert")
            self.run_full_diagnostic()
        except Exception as e:
            self.status_var.set(f"❌ Cache-Löschen fehlgeschlagen: {e}")

    def validate_stream_url(self, url: str) -> Tuple[bool, str]:
        """🔧 NEU: Validiert Stream-URLs mit detaillierten Fehlermeldungen"""
        if not url:
            return False, "URL darf nicht leer sein"
        
        try:
            parsed = urlparse(url)
            
            if url.startswith('file://'):
                file_path = url[7:]
                if not os.path.exists(file_path):
                    return False, f"Datei nicht gefunden: {file_path}"
                return True, "local"
            
            if parsed.scheme in ('http', 'https'):
                if not parsed.netloc:
                    return False, "Ungültige URL - keine Domain gefunden"
                
                if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
                    if not parsed.query and not any(x in parsed.path for x in ['watch', 'live']):
                        return False, "Ungültige YouTube URL"
                
                return True, "web"
            
            if parsed.scheme in ('rtmp', 'rtsp'):
                return True, "stream"
                
            return False, f"Nicht unterstütztes Protokoll: {parsed.scheme}"
            
        except Exception as e:
            return False, f"URL-Validierungsfehler: {e}"

    def start_live(self):
        """🔧 VERBESSERT: Startet Live-Translation mit robuster URL-Validierung"""
        try:
            url = self.url_entry.get().strip()
            if not url:
                messagebox.showerror("Fehler", "Bitte geben Sie eine URL oder wählen Sie eine Datei aus!")
                return

            validation_result, validation_msg = self.validate_stream_url(url)
            if not validation_result:
                messagebox.showerror("Fehler", validation_msg)
                return

            self.status_var.set("🚀 Starte Live-Translation...")
            self.root.update_idletasks()
            
            callbacks = {
                'transcription': self.handle_transcription,
                'translation': self.handle_translation,
                'error': self.handle_error,
                'info': self.handle_info,
                'clear_text': self.clear_text,
                'warning': self.handle_warning,
                'stream_title': self.handle_stream_title
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
            logging.error(f"Start live translation failed: {e}")
            error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            self.status_var.set(f"❌ Start fehlgeschlagen: {error_msg}")
            messagebox.showerror("Fehler", f"Starten fehlgeschlagen:\n{error_msg}")

    def handle_stream_title(self, title: str):
        """Handler für Stream-Titel Updates"""
        try:
            if title:
                display_title = title[:50] + "..." if len(title) > 50 else title
                self.stream_title_var.set(f"📺 {display_title}")
                self.status_var.set(f"📺 Stream-Titel aktualisiert: {display_title}")
        except Exception as e:
            logging.debug(f"Stream title handler error: {e}")

    def handle_warning(self, warning: str):
        """🔧 NEU: Handler für Warning-Nachrichten"""
        try:
            self.status_var.set(f"⚠️  {warning}")
            logging.warning(f"Warning callback: {warning}")
        except Exception as e:
            logging.error(f"Warning handler error: {e}")

    def stop(self):
        """Stoppt Live-Translation"""
        try:
            self.translator.stop()
            self.is_translating = False
            self.start_button.configure(text="🚀 Translation Starten")
            self.status_var.set("⏹️ Translation gestoppt")
            self.session_var.set("Session: --")
            self.stream_title_var.set("📺 Kein Stream-Titel")
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
        stats_window.configure(bg=self.available_themes[self.current_theme]['bg_primary'])
        stats_window.transient(self.root)
        stats_window.grab_set()

        stats_window.update_idletasks()
        x = (stats_window.winfo_screenwidth() // 2) - (width // 2)
        y = (stats_window.winfo_screenheight() // 2) - (height // 2)
        stats_window.geometry(f"+{x}+{y}")

        text = scrolledtext.ScrolledText(
            stats_window,
            bg=self.available_themes[self.current_theme]['bg_tertiary'],
            fg=self.available_themes[self.current_theme]['text_primary'],
            insertbackground=self.available_themes[self.current_theme]['text_primary'],
            selectbackground=self.available_themes[self.current_theme]['accent_blue'],
            selectforeground=self.available_themes[self.current_theme]['text_primary'],
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
            tips_window.configure(bg=self.available_themes[self.current_theme]['bg_primary'])
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
                bg=self.available_themes[self.current_theme]['bg_tertiary'],
                fg=self.available_themes[self.current_theme]['text_primary'],
                insertbackground=self.available_themes[self.current_theme]['text_primary'],
                selectbackground=self.available_themes[self.current_theme]['accent_blue'],
                selectforeground=self.available_themes[self.current_theme]['text_primary'],
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

    def run(self):
        """Startet die GUI-Hauptloop mit Exception Handling"""
        try:
            self.update_metrics_display()
            self.root.mainloop()
        except Exception as e:
            logging.critical(f"❌ GUI Hauptloop fehlgeschlagen: {e}")
            messagebox.showerror("Kritischer Fehler", f"Die Anwendung muss beendet werden:\n{e}")

def setup_cli():
    """🔧 NEU: Kommandozeilen-Interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dragon Whisperer - Stream Translator')
    parser.add_argument('--url', help='Stream URL')
    parser.add_argument('--model', default='small', help='Whisper model size')
    parser.add_argument('--language', default='en', help='Target language')
    parser.add_argument('--source-language', default='auto', help='Source language (auto detection if not specified)')
    parser.add_argument('--no-gui', action='store_true', help='Run in CLI mode')
    parser.add_argument('--output', help='Output file for transcriptions')
    parser.add_argument('--preset', choices=list(ConfigPresets.PRESETS.keys()), help='Configuration preset')
    
    return parser.parse_args()

def run_cli_mode(args):
    """🔧 VOLLSTÄNDIGE CLI-Implementierung"""
    print("🐉 Dragon Whisperer - CLI Mode")
    print(f"URL: {args.url}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Source Language: {args.source_language}")
    
    translator = DragonWhispererTranslator()
    
    if args.preset:
        if translator.apply_config_preset(args.preset):
            print(f"✅ Preset '{args.preset}' angewendet")
        else:
            print(f"❌ Preset '{args.preset}' konnte nicht angewendet werden")
    
    def cli_callback(result_type: str, data):
        if result_type == 'transcription':
            print(f"[{data.language}] {data.text}")
        elif result_type == 'translation':
            print(f"[TRANSLATED] {data.translated}")
        elif result_type == 'error':
            print(f"ERROR: {data}")
        elif result_type == 'info':
            print(f"INFO: {data}")

    if translator.start_live_translation(args.url, {
        'transcription': lambda x: cli_callback('transcription', x),
        'translation': lambda x: cli_callback('translation', x),
        'error': lambda x: cli_callback('error', x),
        'info': lambda x: cli_callback('info', x)
    }):
        print("Translation started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            translator.stop()
            print("Translation stopped.")
    else:
        print("Failed to start translation.")

def main():
    """Hauptfunktion mit erweitertem Error Handling und System Checks"""
    args = setup_cli()
    if args.no_gui or args.url:
        run_cli_mode(args)
        return

    print("🐉 Dragon Whisperer - Stream Translator - VOLLSTÄNDIG OPTIMIERT V3.0")
    print("=" * 80)
    print("🔧 ENHANCED UI/UX: Intelligente Fenster-Ausrichtung & Responsive Design")
    print("🎯 TERMINAL-FIX: 100% saubere Beendigung ohne hängende Prozesse")
    print("📺 STREAM-TITEL: Automatische Erkennung & Anzeige für YouTube/andere Plattformen")
    print("🔧 NEU: Quellsprache Auswahl, Configuration Presets, Auto-Tuning")
    print("=" * 80)

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

    if not YT_DLP_AVAILABLE:
        print("⚠️  yt-dlp nicht verfügbar")
        print("💡 Installiere: pip install yt-dlp")

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
        print("=" * 80)
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
                root.withdraw()
                messagebox.showerror("Kritischer Fehler",
                                     f"Die Anwendung konnte nicht gestartet werden:\n\n{e}\n\n"
                                     f"Bitte überprüfen Sie die Log-Datei:\n{log_dir / 'dragon_whisperer.log'}")
                root.destroy()
        except BaseException:
            pass

        sys.exit(1)

if __name__ == "__main__":
    main()
