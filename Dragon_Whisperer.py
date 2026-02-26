#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""🐉 THE DRAGON WHISPERER v2.01)"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import atexit
import gc
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import platform
import queue
import re
import shutil
import signal as py_signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import warnings
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (Any, Callable, Deque, Dict, List, Optional, Set, Tuple,
                    Union)

# -----------------------------------------------------------------------------
# FRÜHE KONFIGURATION (muss vor allem anderen stehen)
# -----------------------------------------------------------------------------
DEBUG_LEVEL = 0
DEBUG_COMPONENTS = []

for arg in sys.argv:
    if arg == '--debug':
        DEBUG_LEVEL = max(DEBUG_LEVEL, 1)
    elif arg.startswith('--debug='):
        value = arg.split('=', 1)[1]
        if value.isdigit():
            DEBUG_LEVEL = max(DEBUG_LEVEL, int(value))
        else:
            DEBUG_COMPONENTS.extend(value.split(','))

QUIET_MODE = "--quiet" in sys.argv or "-q" in sys.argv

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dragon")

for lib in ["httpx", "urllib3", "httpcore"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

if DEBUG_LEVEL >= 1:
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
if QUIET_MODE:
    logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*The pynvml package is deprecated.*")

os.environ.update({
    "PYTHONWARNINGS": "default",
    "TORCH_DISABLE_CUDA_WARNINGS": "1",
    "TORCH_CPP_LOG_LEVEL": "0",
    "PYTORCH_JIT": "0",
})

SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

try:
    machine = platform.machine().lower()
    IS_ARM = machine in ("arm64", "aarch64", "armv8l", "arm")
    IS_X86 = machine in ("x86_64", "amd64", "i386", "i686", "x86")
except Exception:
    IS_ARM = False
    IS_X86 = True

logger.info(f"🐉 Dragon Whisperer - Platform: {SYSTEM} "
            f"{'ARM' if IS_ARM else 'x86'} (Debug-Level: {DEBUG_LEVEL})")

os.environ.update({
    "FFMPEG_DISABLE_RKMPP": "1",
    "AV_DISABLE_RKMPP": "1",
    "FFMPEG_DISABLE_VAAPI": "0" if IS_LINUX else "1",
    "FFMPEG_DISABLE_VDPAU": "0" if IS_LINUX else "1",
    "OPENCV_LOG_LEVEL": "ERROR",
    "GST_DEBUG": "0",
})

if IS_WINDOWS:
    os.environ.update({"PYTHONIOENCODING": "utf-8"})


# =============================================================================
# 2. KONSTANTEN & KONFIGURATION
# =============================================================================

class Constants:
    """Zentrale Konstanten für das gesamte Programm."""
    # Audio
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    AUDIO_FORMAT: str = 's16le'
    BYTES_PER_SAMPLE: int = 2

    # Chunking
    BASE_CHUNK_DURATION: int = 5
    CHUNK_OVERLAP: float = 1.0
    MIN_CHUNK_DURATION: int = 2
    MAX_CHUNK_DURATION: int = 30

    # Prozesse & Timeouts
    MAX_SUBPROCESSES: int = 8
    SUBPROCESS_TIMEOUT: int = 60
    GUI_OPERATION_TIMEOUT: float = 10.0
    MEMORY_CHECK_INTERVAL: int = 15
    MAX_GUI_UPDATES_PER_SECOND: int = 30
    MAX_MEMORY_USAGE: int = 8 * 1024 * 1024 * 1024  # 8 GB
    MAX_CACHE_SIZE: int = 500
    MAX_TEXT_LINES: int = 2000
    DEFAULT_BEAM_SIZE: int = 5
    DEFAULT_TEMPERATURE: float = 0.0
    ENABLE_VAD_FILTER: bool = True
    MAX_CONSECUTIVE_ERRORS: int = 5  # neu

    # Stream
    STREAM_TIMEOUT: int = 10
    INITIAL_BUFFER_SECONDS: float = 1.5
    MAX_EMPTY_READS: int = 30
    RECONNECT_DELAY: int = 2
    READ_RETRY_DELAY: float = 0.1
    YOUTUBE_TIMEOUT: int = 10000000
    NORMAL_TIMEOUT: int = 30000000

    # FFmpeg
    FFMPEG_BUFSIZE: str = '2048k'
    FFMPEG_THREADS: int = 1
    FFMPEG_PROBESIZE: str = '32'
    FFMPEG_ANALYZE_DURATION: str = '0'

    # Audio-Filter (Sprachoptimierung)
    AUDIO_FILTER: str = "aresample=16000,volume=1.5,dynaudnorm"
    LANGUAGE_FILTERS: Dict[str, str] = {
        'ko': "aresample=16000,volume=2.0,highpass=f=80,lowpass=f=3800,afftdn=nf=-15",
        'ja': "aresample=16000,volume=2.0,highpass=f=90,lowpass=f=3700,afftdn=nf=-15",
        'zh': "aresample=16000,volume=2.0,highpass=f=100,lowpass=f=3500,afftdn=nf=-20",
        'de': "aresample=16000,volume=1.8,highpass=f=100,lowpass=f=3200,dynaudnorm",
        'en': "aresample=16000,volume=1.8,highpass=f=80,lowpass=f=3400,dynaudnorm",
        'fr': "aresample=16000,volume=2.0,highpass=f=100,lowpass=f=3300,dynaudnorm",
        'es': "aresample=16000,volume=2.0,highpass=f=100,lowpass=f=3400,dynaudnorm",
    }
    FILTER_PROFILES: Dict[str, str] = {
        'transcription': "aresample=16000,volume=1.5,dynaudnorm",
        'translation': "aresample=16000,volume=2.0,highpass=f=100,lowpass=f=3400",
        'realtime': "aresample=16000,volume=1.8,dynaudnorm",
        'noisy': "aresample=16000,volume=2.5,highpass=f=150,lowpass=f=3000,afftdn=nf=-30",
        'music': "aresample=16000,volume=1.5,highpass=f=50,lowpass=f=5000",
        'podcast': "aresample=16000,volume=2.0,highpass=f=80,lowpass=f=3500",
    }

    # Audio-Enhancement
    AUDIO_ENHANCEMENT_ENABLED: bool = True
    MIN_RMS_THRESHOLD: float = 0.002
    TARGET_RMS: float = 0.2
    MAX_GAIN: float = 5.0
    CLIPPING_THRESHOLD: float = 0.9

    # Duplikaterkennung
    DUPLICATE_CHECK_ENABLED: bool = True
    RECENT_TRANSCRIPTIONS_SIZE: int = 10
    MIN_TEXT_LENGTH: int = 3
    MIN_UNIQUE_WORDS_RATIO: float = 0.3

    # Untertitel
    SUBTITLE_BUFFER_SIZE: int = 1000
    ENABLE_TIMED_TRANSCRIPTIONS: bool = True
    ENABLE_TIMED_TRANSLATIONS: bool = True

    # Logging
    ENABLE_DEBUG_LOGGING: bool = True
    LOG_CHUNK_PROCESSING: bool = False
    LOG_AUDIO_STATS: bool = True
    LOG_PERFORMANCE: bool = True
    LOG_STREAM_EVENTS: bool = True
    PERFORMANCE_LOG_INTERVAL: int = 50

    # Cache
    MAX_CACHE_SIZE_MB: int = 100
    CACHE_ENABLED: bool = True

    # Cache-Größen (TTL in Sekunden)
    TRANSCRIPTION_CACHE_SIZE: int = 200
    TRANSCRIPTION_CACHE_TTL: int = 300
    TRANSLATION_CACHE_SIZE: int = 500
    TRANSLATION_CACHE_TTL: int = 3600
    AUDIO_CACHE_SIZE: int = 128
    AUDIO_CACHE_TTL: int = 1800

    # VAD
    VAD_THRESHOLD: float = 0.3
    VAD_MIN_SPEECH_DURATION_MS: int = 200
    VAD_MIN_SILENCE_DURATION_MS: int = 80

    # Pfad-Validierung (Sicherheit)
    ALLOWED_FILE_SCHEME_PREFIX: str = "file://"
    ALLOWED_FILE_BASE_DIRS: List[str] = [
        str(Path.home()),
        os.getcwd(),
    ]

    # URL-Zeichen Whitelist (für subprocess-Aufrufe)
    URL_ALLOWED_CHARS: str = r"a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%"

    # YouTube-Header
    YOUTUBE_HEADERS: Dict[str, str] = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.youtube.com/',
        'Origin': 'https://www.youtube.com',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Plattform-Konfiguration
    PLATFORM_CONFIG: Dict[str, Dict[str, Any]] = {
        'windows': {
            'ffmpeg_flags': ['-reconnect', '1', '-reconnect_streamed', '1'],
            'process_creation_flags': 0x08000000,
        },
        'macos': {
            'ffmpeg_flags': ['-reconnect', '1', '-reconnect_on_network_error', '1'],
            'start_new_session': True,
        },
        'linux': {
            'ffmpeg_flags': ['-reconnect', '1', '-reconnect_streamed', '1'],
            'start_new_session': True,
        }
    }


# -----------------------------------------------------------------------------
# EXCELLENCE CONFIG
# -----------------------------------------------------------------------------
class ExcellenceConfig:
    """Dynamische Konfiguration, die zur Laufzeit angepasst werden kann."""
    SAMPLE_RATE: int = Constants.SAMPLE_RATE
    CHANNELS: int = Constants.CHANNELS
    AUDIO_FORMAT: str = Constants.AUDIO_FORMAT
    BYTES_PER_SAMPLE: int = Constants.BYTES_PER_SAMPLE

    MAX_SUBPROCESSES: int = Constants.MAX_SUBPROCESSES
    SUBPROCESS_TIMEOUT: int = Constants.SUBPROCESS_TIMEOUT
    GUI_OPERATION_TIMEOUT: float = Constants.GUI_OPERATION_TIMEOUT
    MEMORY_CHECK_INTERVAL: int = Constants.MEMORY_CHECK_INTERVAL
    MAX_GUI_UPDATES_PER_SECOND: int = Constants.MAX_GUI_UPDATES_PER_SECOND
    MAX_MEMORY_USAGE: int = Constants.MAX_MEMORY_USAGE
    MAX_CACHE_SIZE: int = Constants.MAX_CACHE_SIZE
    MAX_TEXT_LINES: int = Constants.MAX_TEXT_LINES
    DEFAULT_BEAM_SIZE: int = Constants.DEFAULT_BEAM_SIZE
    DEFAULT_TEMPERATURE: float = Constants.DEFAULT_TEMPERATURE
    ENABLE_VAD_FILTER: bool = Constants.ENABLE_VAD_FILTER
    MAX_CONSECUTIVE_ERRORS: int = Constants.MAX_CONSECUTIVE_ERRORS  # neu

    _base_chunk_duration: int = Constants.BASE_CHUNK_DURATION
    CHUNK_OVERLAP: float = Constants.CHUNK_OVERLAP
    MIN_CHUNK_DURATION: int = Constants.MIN_CHUNK_DURATION
    MAX_CHUNK_DURATION: int = Constants.MAX_CHUNK_DURATION

    @property
    def CHUNK_DURATION(self) -> float:
        return getattr(self, '_actual_chunk_duration', self._base_chunk_duration)

    @CHUNK_DURATION.setter
    def CHUNK_DURATION(self, value: float) -> None:
        if self.MIN_CHUNK_DURATION <= value <= self.MAX_CHUNK_DURATION:
            self._actual_chunk_duration = float(value)
        else:
            logger.warning(f"⚠️ Chunk duration {value}s out of range, using default")
            self._actual_chunk_duration = self._base_chunk_duration

    @property
    def CHUNK_SIZE_BYTES(self) -> int:
        return int(self.CHUNK_DURATION * self.SAMPLE_RATE *
                   self.CHANNELS * self.BYTES_PER_SAMPLE)

    @property
    def OVERLAP_SIZE_BYTES(self) -> int:
        return int(self.CHUNK_OVERLAP * self.SAMPLE_RATE *
                   self.CHANNELS * self.BYTES_PER_SAMPLE)

    @property
    def BYTES_PER_SECOND(self) -> int:
        return self.SAMPLE_RATE * self.CHANNELS * self.BYTES_PER_SAMPLE

    @property
    def MIN_CHUNK_BYTES(self) -> int:
        return int(self.MIN_CHUNK_DURATION * self.BYTES_PER_SECOND)

    @property
    def MAX_CHUNK_BYTES(self) -> int:
        return int(self.MAX_CHUNK_DURATION * self.BYTES_PER_SECOND)

    STREAM_TIMEOUT: int = Constants.STREAM_TIMEOUT
    INITIAL_BUFFER_SECONDS: float = Constants.INITIAL_BUFFER_SECONDS
    MAX_EMPTY_READS: int = Constants.MAX_EMPTY_READS
    RECONNECT_DELAY: int = Constants.RECONNECT_DELAY
    READ_RETRY_DELAY: float = Constants.READ_RETRY_DELAY

    @property
    def INITIAL_BUFFER_BYTES(self) -> int:
        return int(self.INITIAL_BUFFER_SECONDS * self.BYTES_PER_SECOND)

    FFMPEG_BUFSIZE: str = Constants.FFMPEG_BUFSIZE
    FFMPEG_THREADS: int = Constants.FFMPEG_THREADS
    FFMPEG_PROBESIZE: str = Constants.FFMPEG_PROBESIZE
    FFMPEG_ANALYZE_DURATION: str = Constants.FFMPEG_ANALYZE_DURATION

    YOUTUBE_TIMEOUT: int = Constants.YOUTUBE_TIMEOUT
    NORMAL_TIMEOUT: int = Constants.NORMAL_TIMEOUT

    def get_timeout_microseconds(self, is_youtube: bool = False) -> int:
        return self.YOUTUBE_TIMEOUT if is_youtube else self.NORMAL_TIMEOUT

    AUDIO_FILTER: str = Constants.AUDIO_FILTER
    LANGUAGE_FILTERS: Dict[str, str] = Constants.LANGUAGE_FILTERS
    FILTER_PROFILES: Dict[str, str] = Constants.FILTER_PROFILES

    def get_audio_filter(self, language: Optional[str] = None,
                         profile: Optional[str] = None) -> str:
        if profile and profile in self.FILTER_PROFILES:
            return self.FILTER_PROFILES[profile]
        if language:
            lang_code = language[:2].lower() if len(language) >= 2 else None
            if lang_code in self.LANGUAGE_FILTERS:
                return self.LANGUAGE_FILTERS[lang_code]
        return self.AUDIO_FILTER

    YOUTUBE_HEADERS: Dict[str, str] = Constants.YOUTUBE_HEADERS

    def get_youtube_headers(self, is_manifest: bool = False) -> Dict[str, str]:
        headers = self.YOUTUBE_HEADERS.copy()
        if is_manifest:
            headers.update({
                'X-Client-Data': 'CI22yQE=',
                'Content-Type': 'application/x-mpegURL',
            })
        return headers

    PLATFORM_CONFIG: Dict[str, Dict[str, Any]] = Constants.PLATFORM_CONFIG

    def get_platform_config(self, platform: Optional[str] = None) -> Dict[str, Any]:
        if not platform:
            platform = SYSTEM.lower()
        return self.PLATFORM_CONFIG.get(platform, self.PLATFORM_CONFIG['linux'])

    AUDIO_ENHANCEMENT_ENABLED: bool = Constants.AUDIO_ENHANCEMENT_ENABLED
    MIN_RMS_THRESHOLD: float = Constants.MIN_RMS_THRESHOLD
    TARGET_RMS: float = Constants.TARGET_RMS
    MAX_GAIN: float = Constants.MAX_GAIN
    CLIPPING_THRESHOLD: float = Constants.CLIPPING_THRESHOLD

    DUPLICATE_CHECK_ENABLED: bool = Constants.DUPLICATE_CHECK_ENABLED
    RECENT_TRANSCRIPTIONS_SIZE: int = Constants.RECENT_TRANSCRIPTIONS_SIZE
    MIN_TEXT_LENGTH: int = Constants.MIN_TEXT_LENGTH
    MIN_UNIQUE_WORDS_RATIO: float = Constants.MIN_UNIQUE_WORDS_RATIO

    SUBTITLE_BUFFER_SIZE: int = Constants.SUBTITLE_BUFFER_SIZE
    ENABLE_TIMED_TRANSCRIPTIONS: bool = Constants.ENABLE_TIMED_TRANSCRIPTIONS
    ENABLE_TIMED_TRANSLATIONS: bool = Constants.ENABLE_TIMED_TRANSLATIONS

    ENABLE_DEBUG_LOGGING: bool = Constants.ENABLE_DEBUG_LOGGING
    LOG_CHUNK_PROCESSING: bool = Constants.LOG_CHUNK_PROCESSING
    LOG_AUDIO_STATS: bool = Constants.LOG_AUDIO_STATS
    LOG_PERFORMANCE: bool = Constants.LOG_PERFORMANCE
    LOG_STREAM_EVENTS: bool = Constants.LOG_STREAM_EVENTS
    PERFORMANCE_LOG_INTERVAL: int = Constants.PERFORMANCE_LOG_INTERVAL

    MAX_CACHE_SIZE_MB: int = Constants.MAX_CACHE_SIZE_MB
    CACHE_ENABLED: bool = Constants.CACHE_ENABLED

    def __init__(self) -> None:
        self._actual_chunk_duration = self._base_chunk_duration

    def calculate_optimal_chunk_duration(self, model_size: str = 'medium',
                                         is_realtime: bool = False) -> int:
        if is_realtime:
            return self.MIN_CHUNK_DURATION
        model_durations = {
            'tiny': 3, 'tiny.en': 3,
            'base': 4, 'base.en': 4,
            'small': 5, 'small.en': 5,
            'medium': 5, 'medium.en': 5,
            'large': 6, 'large-v2': 6, 'large-v3': 6,
        }
        return model_durations.get(model_size.lower(), self._base_chunk_duration)

    def validate_config(self) -> bool:
        try:
            valid = (
                self.SAMPLE_RATE in [8000, 16000, 22050, 44100, 48000] and
                self.CHANNELS in [1, 2] and
                self.MIN_CHUNK_DURATION <= self.CHUNK_DURATION <= self.MAX_CHUNK_DURATION
            )
            if not valid:
                logger.error("❌ Config validation failed")
                return False
            return True
        except Exception as e:
            logger.error(f"❌ Config validation error: {e}")
            return False

    def print_summary(self) -> None:
        logger.info("\n" + "="*60)
        logger.info("🎵 EXCELLENCE CONFIGURATION")
        logger.info("="*60)
        logger.info(f"📊 Audio: {self.SAMPLE_RATE}Hz, {self.CHANNELS}ch")
        logger.info(f"📦 Chunk: {self.CHUNK_DURATION}s ({self.CHUNK_SIZE_BYTES:,}B)")
        logger.info(f"⚡ Bytes/sec: {self.BYTES_PER_SECOND:,}")
        logger.info(f"🎛️ Filter Profiles: {len(self.FILTER_PROFILES)}")
        logger.info(f"🌍 Language Filters: {len(self.LANGUAGE_FILTERS)}")
        logger.info(f"✅ Valid: {self.validate_config()}")
        logger.info("="*60)

    def __str__(self) -> str:
        return (f"ExcellenceConfig(chunk={self.CHUNK_DURATION}s, "
                f"filter_profiles={len(self.FILTER_PROFILES)})")


class RealtimeConfig(ExcellenceConfig):
    def __init__(self) -> None:
        super().__init__()
        self.CHUNK_DURATION = 5
        self.CHUNK_OVERLAP = 0.3
        self.STREAM_TIMEOUT = 5
        self.AUDIO_FILTER = self.FILTER_PROFILES['realtime']


class HighAccuracyConfig(ExcellenceConfig):
    def __init__(self) -> None:
        super().__init__()
        self.CHUNK_DURATION = 25
        self.CHUNK_OVERLAP = 0.8
        self.AUDIO_FILTER = ("aresample=16000,volume=1.8,highpass=f=80,"
                             "lowpass=f=3800,dynaudnorm=p=0.3:s=3:g=20")


class YouTubeOptimizedConfig(ExcellenceConfig):
    def __init__(self) -> None:
        super().__init__()
        self.FFMPEG_THREADS = 1
        self.FFMPEG_BUFSIZE = '1024k'
        self.YOUTUBE_TIMEOUT = 5000000
        self.RECONNECT_DELAY = 1
        self.AUDIO_FILTER = ("aresample=16000,volume=2.2,highpass=f=120,"
                             "lowpass=f=3200,compand=attacks=0:decays=0.3")


def get_config(config_type: str = 'default') -> ExcellenceConfig:
    configs = {
        'default': ExcellenceConfig,
        'realtime': RealtimeConfig,
        'high_accuracy': HighAccuracyConfig,
        'youtube': YouTubeOptimizedConfig,
    }
    config_class = configs.get(config_type, ExcellenceConfig)
    config = config_class()
    config.validate_config()
    return config


# =============================================================================
# 3. HILFSKLASSEN UND -FUNKTIONEN (UTILS)
# =============================================================================

# -----------------------------------------------------------------------------
# FastLazyLoader
# -----------------------------------------------------------------------------
class FastLazyLoader:
    _loaded_modules: Dict[str, Any] = {}
    _module_locks: Dict[str, threading.RLock] = {}
    _class_lock = threading.RLock()
    MAX_CACHE_SIZE = 10

    @classmethod
    def _get_lock(cls, module_name: str) -> threading.RLock:
        with cls._class_lock:
            if module_name not in cls._module_locks:
                cls._module_locks[module_name] = threading.RLock()
        return cls._module_locks[module_name]

    @classmethod
    def _prune_cache(cls) -> None:
        with cls._class_lock:
            while len(cls._loaded_modules) > cls.MAX_CACHE_SIZE:
                cls._loaded_modules.popitem(last=False)

    @classmethod
    def load(cls, module_name: str, import_path: Optional[str] = None) -> Any:
        if module_name in cls._loaded_modules:
            return cls._loaded_modules[module_name]

        lock = cls._get_lock(module_name)
        with lock:
            if module_name in cls._loaded_modules:
                return cls._loaded_modules[module_name]

            try:
                if module_name == "torch":
                    import torch
                    try:
                        import torch._logging
                        torch._logging.set_logs(all=logging.ERROR)
                    except Exception:
                        pass
                    cls._loaded_modules["torch"] = torch
                elif module_name == "faster_whisper":
                    from faster_whisper import WhisperModel
                    cls._loaded_modules["faster_whisper"] = WhisperModel
                elif module_name == "numpy":
                    import numpy as np
                    cls._loaded_modules["numpy"] = np
                elif module_name == "deep_translator":
                    from deep_translator import GoogleTranslator
                    cls._loaded_modules["deep_translator"] = GoogleTranslator
                elif module_name == "scipy.signal":
                    import scipy.signal
                    cls._loaded_modules["scipy.signal"] = scipy.signal
                else:
                    module = importlib.import_module(import_path or module_name)
                    cls._loaded_modules[module_name] = module

                cls._prune_cache()
                return cls._loaded_modules[module_name]

            except ImportError as e:
                logger.warning(f"⚠️ Module {module_name} not available: {e}")
                class MockModule:
                    _is_mock = True
                    def __init__(self, name: str):
                        self.__name__ = name
                    def __getattr__(self, name: str) -> Any:
                        def mock_method(*args: Any, **kwargs: Any) -> Any:
                            raise ImportError(f"Module {self.__name__} not available")
                        return mock_method
                mock = MockModule(module_name)
                cls._loaded_modules[module_name] = mock
                cls._prune_cache()
                return mock

    @classmethod
    def is_available(cls, module_name: str) -> bool:
        if importlib.util.find_spec(module_name) is None:
            return False
        if module_name in cls._loaded_modules:
            module = cls._loaded_modules[module_name]
            return not getattr(module, "_is_mock", False)
        return True

    @classmethod
    def clear_cache(cls) -> None:
        with cls._class_lock:
            cls._loaded_modules.clear()
            cls._module_locks.clear()


# -----------------------------------------------------------------------------
# Plattform-Stderr-Filter (unverändert)
# -----------------------------------------------------------------------------
class PlatformStderrFilter:
    def __init__(self, original_stderr: Any) -> None:
        self.original_stderr = original_stderr
        self.filter_patterns = [
            "mpp_soc:", "mpp_platform:", "can not found match soc name",
            "/proc/device-tree/compatible", "rockchip", "ffmpeg", "TORCH_NCCL",
        ]
        if IS_WINDOWS:
            self.filter_patterns.extend([
                "Failed to set direct console mode",
                "Console code page",
                "chcp",
                "win32api",
            ])

    def write(self, text: str) -> None:
        if text and any(p in text for p in self.filter_patterns):
            return
        self.original_stderr.write(text)

    def flush(self) -> None:
        self.original_stderr.flush()


sys.stderr = PlatformStderrFilter(sys.stderr)


# -----------------------------------------------------------------------------
# Terminal-Einstellungen (unverändert)
# -----------------------------------------------------------------------------
_original_stty_settings: Optional[str] = None

def _save_terminal_settings() -> None:
    global _original_stty_settings
    if not IS_LINUX:
        return
    try:
        result = subprocess.run(['stty', '-g'], capture_output=True,
                                 text=True, check=False, timeout=2)
        if result.returncode == 0 and result.stdout:
            _original_stty_settings = result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        if DEBUG_LEVEL >= 2:
            logger.debug(f"stty -g fehlgeschlagen: {e}")

def _restore_terminal_settings() -> None:
    if not IS_LINUX or _original_stty_settings is None:
        return
    try:
        subprocess.run(['stty', _original_stty_settings], check=False, timeout=2)
    except Exception:
        pass

_save_terminal_settings()
atexit.register(_restore_terminal_settings)


# -----------------------------------------------------------------------------
# SignalHandler (überarbeitet mit Thread-Sicherheit)
# -----------------------------------------------------------------------------
class ShutdownPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class SignalHandler:
    _instance: Optional['SignalHandler'] = None
    _lock = threading.RLock()
    _shutdown_requested = False
    _shutdown_in_progress = False
    _signal_count = 0
    _setup_complete = False
    _original_handlers: Dict[int, Any] = {}
    _atexit_registered = False
    _atexit_lock = threading.RLock()
    _config = {
        'verbose': False,
        'silent': True,
        'max_cleanup_time': 20.0,
        'emergency_timeout': 2.0,
        'atexit_enabled': True,
        'hybrid_shutdown': True,
    }
    _cleanup_operations: Dict[ShutdownPriority, List['_CleanupOperation']] = {
        ShutdownPriority.CRITICAL: [],
        ShutdownPriority.HIGH: [],
        ShutdownPriority.MEDIUM: [],
        ShutdownPriority.LOW: [],
    }

    class _CleanupOperation:
        def __init__(self, func: Callable[[], Any], name: str,
                     priority: ShutdownPriority = ShutdownPriority.MEDIUM,
                     timeout: float = 3.0, essential: bool = False):
            self.func = func
            self.name = name
            self.priority = priority
            self.timeout = timeout
            self.essential = essential
            self.attempts = 0
            self.last_error: Optional[str] = None

        def execute(self) -> bool:
            self.attempts += 1
            try:
                result: Any = None
                exc: Optional[BaseException] = None
                def target() -> None:
                    nonlocal result, exc
                    try:
                        result = self.func()
                    except Exception as e:
                        exc = e
                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                thread.join(timeout=self.timeout)
                if thread.is_alive():
                    self.last_error = f"Timeout nach {self.timeout}s"
                    return False
                if exc:
                    self.last_error = str(exc)
                    return False
                self.last_error = None
                return True
            except Exception as e:
                self.last_error = str(e)
                return False

    def __new__(cls) -> 'SignalHandler':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        pass

    @classmethod
    def setup(cls, verbose: bool = False, silent: bool = True,
              hybrid_shutdown: bool = True, **kwargs: Any) -> 'SignalHandler':
        with cls._lock:
            if cls._setup_complete:
                return cls._instance

            cls._config.update({
                'verbose': verbose,
                'silent': silent,
                'hybrid_shutdown': hybrid_shutdown,
                'max_cleanup_time': kwargs.get('max_cleanup_time', 20.0),
                'emergency_timeout': kwargs.get('emergency_timeout', 2.0),
                'atexit_enabled': kwargs.get('atexit_enabled', True),
            })

            if cls._config['verbose']:
                print("🚀 SignalHandler initialisieren...")

            cls._save_original_handlers()
            cls._install_signal_handlers()

            if cls._config['atexit_enabled']:
                cls._register_atexit()

            cls._setup_complete = True

            if cls._config['verbose']:
                print("✅ SignalHandler bereit")
            return cls._instance

    @classmethod
    def _save_original_handlers(cls) -> None:
        if IS_WINDOWS:
            return
        try:
            for sig in [py_signal.SIGINT, py_signal.SIGTERM]:
                cls._original_handlers[sig] = py_signal.getsignal(sig)
        except Exception:
            pass

    @classmethod
    def _install_signal_handlers(cls) -> None:
        def signal_handler(signum: int, frame: Any) -> None:
            with cls._lock:
                cls._signal_count += 1
                if cls._signal_count == 1:
                    if not cls._config['silent']:
                        print("\n⚠️ Shutdown angefordert...")
                    cls._shutdown_requested = True
                    cls._initiate_graceful_shutdown()
                elif cls._signal_count >= 2:
                    if not cls._config['silent']:
                        print("\n🛑 Forcierter Shutdown...")
                    cls._force_shutdown()

        if not IS_WINDOWS:
            try:
                py_signal.signal(py_signal.SIGINT, signal_handler)
                py_signal.signal(py_signal.SIGTERM, signal_handler)
            except Exception:
                pass
        else:
            try:
                import win32api
                def win32_handler(ctrl_type: int) -> bool:
                    if ctrl_type in (0, 1, 2, 5):
                        signal_handler(ctrl_type, None)
                        return True
                    return False
                win32api.SetConsoleCtrlHandler(win32_handler, True)
            except ImportError:
                try:
                    py_signal.signal(py_signal.SIGINT, signal_handler)
                except (ValueError, OSError):
                    pass
            except Exception:
                pass

    @classmethod
    def register_cleanup(cls, func: Callable[[], Any], name: Optional[str] = None,
                         priority: ShutdownPriority = ShutdownPriority.MEDIUM,
                         timeout: float = 3.0, essential: bool = False) -> None:
        if name is None:
            name = func.__name__ if hasattr(func, '__name__') else "Anonymous"

        operation = cls._CleanupOperation(
            func=func, name=name, priority=priority,
            timeout=timeout, essential=essential
        )

        with cls._lock:
            for existing in cls._cleanup_operations[priority]:
                if existing.func == func:
                    return
            cls._cleanup_operations[priority].append(operation)

            if cls._config.get('verbose', False):
                print(f"✅ Cleanup: {name} (Priority: {priority.name})")

    @classmethod
    def unregister_cleanup(cls, func: Callable) -> bool:
        with cls._lock:
            for priority in ShutdownPriority:
                for i, op in enumerate(cls._cleanup_operations[priority]):
                    if op.func == func:
                        del cls._cleanup_operations[priority][i]
                        return True
            return False

    @classmethod
    def _initiate_graceful_shutdown(cls) -> None:
        with cls._lock:
            if cls._shutdown_in_progress:
                return
            cls._shutdown_in_progress = True

        if not cls._config['silent']:
            print("🧹 Starte geordneten Shutdown...")

        try:
            success = cls._execute_priority_cleanup()
            cls._restore_original_handlers()

            if cls._config.get('hybrid_shutdown', True):
                current_thread = threading.current_thread()
                main_thread = threading.main_thread()
                if current_thread == main_thread:
                    if cls._config['verbose']:
                        print("💡 Sauberes Exit im Hauptthread")
                    sys.exit(0 if success else 1)
                else:
                    if cls._config['verbose']:
                        print(f"💡 Sofort-Exit in Thread: {current_thread.name}")
                    os._exit(0 if success else 1)
            else:
                os._exit(0 if success else 1)
        except Exception as e:
            if not cls._config['silent']:
                print(f"❌ Shutdown fehlgeschlagen: {e}")
            os._exit(2)

    @classmethod
    def _execute_priority_cleanup(cls) -> bool:
        overall_success = True
        start_time = time.time()
        completed_ops = 0
        failed_ops = 0

        for priority in ShutdownPriority:
            operations = cls._cleanup_operations.get(priority, [])
            if not operations:
                continue

            if time.time() - start_time > cls._config['max_cleanup_time']:
                if cls._config.get('verbose', False):
                    print(f"⏰ Max cleanup time reached ({cls._config['max_cleanup_time']}s)")
                break

            for op in operations:
                try:
                    success = op.execute()
                    completed_ops += 1
                    if not success:
                        failed_ops += 1
                        if op.essential:
                            overall_success = False
                            if cls._config.get('verbose', False):
                                print(f"❌ ESSENTIAL cleanup failed: {op.name}")
                        elif cls._config.get('verbose', False):
                            print(f"⚠️ Cleanup failed (non-essential): {op.name}")
                    elif cls._config.get('verbose', False):
                        print(f"✅ Cleanup: {op.name}")
                except Exception as e:
                    failed_ops += 1
                    print(f"⚠️ Cleanup execution error: {op.name}: {e}")
                    if op.essential:
                        overall_success = False

        if cls._config.get('verbose', False):
            print(f"📊 Cleanup abgeschlossen: {completed_ops} Operationen, "
                  f"{failed_ops} fehlgeschlagen")
        return overall_success

    @classmethod
    def _force_shutdown(cls) -> None:
        try:
            cls._handle_atexit_cleanup()
        except Exception:
            pass
        os._exit(1)

    @classmethod
    def _register_atexit(cls) -> None:
        with cls._atexit_lock:
            if cls._atexit_registered:
                return

            def safe_atexit_handler() -> None:
                try:
                    if (threading.current_thread() == threading.main_thread() and
                            not cls._shutdown_in_progress):
                        cls._handle_atexit_cleanup()
                except Exception:
                    pass

            atexit.register(safe_atexit_handler)
            cls._atexit_registered = True

            if cls._config.get('verbose', False):
                print("✅ AtExit-Handler registriert")

    @classmethod
    def _handle_atexit_cleanup(cls) -> None:
        if cls._config.get('verbose', False):
            print("🔧 AtExit-Cleanup...")

        critical_ops = []
        for op in cls._cleanup_operations.get(ShutdownPriority.CRITICAL, []):
            if op.essential and ("GPU" in op.name or "Memory" in op.name):
                critical_ops.append(op)
                if len(critical_ops) >= 3:
                    break

        start_time = time.time()
        for op in critical_ops:
            if time.time() - start_time > cls._config['emergency_timeout']:
                break
            try:
                if cls._config.get('verbose', False):
                    print(f"  ⚡ Emergency: {op.name}")
                op.func()
            except Exception:
                pass

        gc.collect()

    @classmethod
    def _restore_original_handlers(cls) -> None:
        if IS_WINDOWS:
            return
        try:
            for sig, handler in cls._original_handlers.items():
                if handler is not None:
                    py_signal.signal(sig, handler)
        except Exception:
            pass

    @classmethod
    def should_shutdown(cls) -> bool:
        with cls._lock:
            return cls._shutdown_requested

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        with cls._lock:
            return {
                'shutdown_requested': cls._shutdown_requested,
                'shutdown_in_progress': cls._shutdown_in_progress,
                'signal_count': cls._signal_count,
                'setup_complete': cls._setup_complete,
                'cleanup_operations': {
                    priority.name: len(ops)
                    for priority, ops in cls._cleanup_operations.items()
                },
                'atexit_registered': cls._atexit_registered,
                'hybrid_mode': cls._config.get('hybrid_shutdown', True),
                'config': {k: v for k, v in cls._config.items()
                           if not k.startswith('_')},
            }

    @classmethod
    def emergency_shutdown(cls, reason: str = "Emergency", exit_code: int = 1) -> None:
        if not cls._config['silent']:
            print(f"🚨 NOTFALL-SHUTDOWN: {reason}")

        with cls._lock:
            cls._shutdown_requested = True
            cls._shutdown_in_progress = True

        os._exit(exit_code)

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._shutdown_requested = False
            cls._shutdown_in_progress = False
            cls._signal_count = 0
            cls._setup_complete = False
            cls._original_handlers = {}
            cls._atexit_registered = False
            cls._cleanup_operations = {
                ShutdownPriority.CRITICAL: [],
                ShutdownPriority.HIGH: [],
                ShutdownPriority.MEDIUM: [],
                ShutdownPriority.LOW: [],
            }


# -----------------------------------------------------------------------------
# PlatformUtils (erweitert um URL-Validierung und Pfad-Sicherheit)
# -----------------------------------------------------------------------------
class PlatformUtils:
    _environment_setup_done = False
    _environment_setup_lock = threading.RLock()
    _dependencies_checked = False
    _dependencies_lock = threading.RLock()

    @staticmethod
    def get_platform_config_dir() -> Path:
        try:
            if IS_WINDOWS:
                config_dir = Path(os.environ.get('APPDATA', '')) / "DragonWhisperer"
            elif IS_MACOS:
                config_dir = Path.home() / "Library" / "Application Support" / "DragonWhisperer"
            else:
                config_dir = Path.home() / ".config" / "dragonwhisperer"
            config_dir.mkdir(parents=True, exist_ok=True)
            return config_dir
        except Exception as e:
            logger.warning(f"⚠️ Config directory error: {e}")
            fallback_dir = Path.home() / ".dragonwhisperer"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir

    @staticmethod
    def kill_process_tree(pid: int) -> bool:
        try:
            if IS_WINDOWS:
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                subprocess.run(
                    ['pkill', '-9', '-P', str(pid)],
                    capture_output=True,
                    timeout=5,
                    check=False
                )
            return True
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Timeout beim Beenden des Prozessbaums {pid}")
            return False
        except (OSError, subprocess.CalledProcessError) as e:
            logger.warning(f"⚠️ Error killing process tree {pid}: {e}")
            return False
        except Exception as e:
            logger.error(f"⚠️ Unerwarteter Fehler beim Beenden von Prozessbaum {pid}: {e}")
            return False

    @staticmethod
    def check_platform_dependencies() -> bool:
        with PlatformUtils._dependencies_lock:
            if PlatformUtils._dependencies_checked:
                return True
            missing: List[str] = []
            issues: List[str] = []
            logger.info("🔍 Checking platform dependencies...")

            ffmpeg_found = shutil.which('ffmpeg') is not None
            if not ffmpeg_found:
                missing.append('ffmpeg')
                issues.append("FFmpeg not found in PATH or standard locations")

            ytdlp_found = shutil.which('yt-dlp') is not None
            if not ytdlp_found:
                missing.append('yt-dlp')
                issues.append("yt-dlp not found in PATH")

            psutil_found = FastLazyLoader.is_available('psutil')
            if not psutil_found:
                issues.append("psutil not available (system monitoring)")

            critical_missing = []
            if not ffmpeg_found:
                critical_missing.append('ffmpeg')
            if not ytdlp_found:
                critical_missing.append('yt-dlp')
            if not GUI_AVAILABLE:
                critical_missing.append('tkinter')
                issues.append("Tkinter not available – required for GUI")

            if not WHISPER_AVAILABLE:
                logger.warning("⚠️ Kein Whisper-Backend verfügbar. Starte im Demo-Modus.")
            if not TRANSLATOR_AVAILABLE:
                issues.append("deep-translator not available (translation will be limited)")
            if not TORCH_AVAILABLE:
                issues.append("PyTorch not available (optional for GPU acceleration)")
            if not psutil_found:
                issues.append("psutil not available (system monitoring limited)")

            if critical_missing:
                error_msg = (f"❌ Fehlende kritische Abhängigkeiten: "
                             f"{', '.join(critical_missing)}\n\n")
                error_msg += "\n".join(issues) + "\n"
                if 'ffmpeg' in critical_missing:
                    error_msg += "FFmpeg Installation:\n"
                    if IS_WINDOWS:
                        error_msg += "  • Download from: https://ffmpeg.org/download.html\n"
                    elif IS_MACOS:
                        error_msg += "  • brew install ffmpeg\n"
                    else:
                        error_msg += "  • sudo apt install ffmpeg\n"
                if 'yt-dlp' in critical_missing:
                    error_msg += "yt-dlp Installation:\n"
                    error_msg += "  • pip install yt-dlp\n"
                if 'tkinter' in critical_missing:
                    error_msg += "Tkinter Installation:\n"
                    error_msg += "  • Usually included with Python. On Linux: sudo apt-get install python3-tk\n"
                if issues:
                    error_msg += "\nAdditional issues:\n"
                    for issue in issues:
                        error_msg += f"  • {issue}\n"
                error_msg += "\n💡 After installing, restart Dragon Whisperer."
                PlatformUtils._dependencies_checked = False
                raise RuntimeError(error_msg)

            logger.info("✅ Alle kritischen Abhängigkeiten gefunden")
            PlatformUtils._dependencies_checked = True
            return True

    @staticmethod
    def setup_platform_environment() -> None:
        with PlatformUtils._environment_setup_lock:
            if PlatformUtils._environment_setup_done:
                return
            logger.info("🔧 Setting up platform environment...")
            if IS_WINDOWS:
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                    os.system('chcp 65001 > nul 2>&1')
                    os.system('color')
                except (OSError, AttributeError) as e:
                    logger.warning(f"⚠️ Windows console setup failed: {e}")
            elif IS_MACOS:
                temp_dir = Path(tempfile.gettempdir()) / "dragonwhisperer"
                try:
                    temp_dir.mkdir(exist_ok=True)
                except (OSError, PermissionError) as e:
                    logger.warning(f"⚠️ macOS temp dir creation failed: {e}")
            PlatformUtils._environment_setup_done = True
            logger.info("✅ Platform environment setup complete")

    @staticmethod
    def get_ffmpeg_path() -> Optional[str]:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        if IS_WINDOWS:
            paths = [
                'C:\\ffmpeg\\bin\\ffmpeg.exe',
                'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
                'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe',
            ]
        elif IS_MACOS:
            paths = [
                '/usr/local/bin/ffmpeg',
                '/opt/homebrew/bin/ffmpeg',
                '/usr/bin/ffmpeg',
            ]
        else:
            paths = [
                '/usr/bin/ffmpeg',
                '/usr/local/bin/ffmpeg',
            ]
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        info = {
            'system': SYSTEM,
            'is_windows': IS_WINDOWS,
            'is_macos': IS_MACOS,
            'is_linux': IS_LINUX,
            'is_arm': IS_ARM,
            'is_x86': IS_X86,
            'python_version': sys.version,
            'python_executable': sys.executable,
            'current_directory': os.getcwd(),
            'environment_setup': PlatformUtils._environment_setup_done,
            'dependencies_checked': PlatformUtils._dependencies_checked,
        }
        try:
            import psutil
            info['cpu_count'] = psutil.cpu_count()
            info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            info['cpu_count'] = 'unknown'
            info['memory_total_gb'] = 'unknown'
        except Exception:
            info['cpu_count'] = 'error'
            info['memory_total_gb'] = 'error'
        return info

    @staticmethod
    def print_platform_info() -> None:
        info = PlatformUtils.get_platform_info()
        logger.info("\n" + "="*60)
        logger.info("🐉 PLATFORM INFORMATION")
        logger.info("="*60)
        for key, value in info.items():
            if key not in ['environment_setup', 'dependencies_checked']:
                logger.info(f"{key:25} {value}")
        logger.info("-"*60)
        logger.info(f"{'Environment Setup':25} {'✅' if info['environment_setup'] else '❌'}")
        logger.info(f"{'Dependencies Checked':25} {'✅' if info['dependencies_checked'] else '❌'}")
        logger.info("="*60)

    # ---------- SICHERHEITSRELEVANTE FUNKTIONEN ----------
    @staticmethod
    def sanitize_url(url: str) -> str:
        """
        Bereinigt eine URL: entfernt führende/folgende Whitespace-Zeichen.
        (Keine Zeichenfilterung mehr, da shell=False ausreichend schützt.)
        """
        if not url:
            return ""
        return url.strip()

    @staticmethod
    def validate_file_path(file_url: str) -> Tuple[bool, str]:
        """
        Prüft, ob eine file://-URL auf eine erlaubte Datei verweist.
        Gibt (ok, normalisierter Pfad) zurück.
        """
        if not file_url.startswith(Constants.ALLOWED_FILE_SCHEME_PREFIX):
            return False, "Keine file://-URL"
        path_part = file_url[len(Constants.ALLOWED_FILE_SCHEME_PREFIX):]
        # Pfad normalisieren
        try:
            real_path = os.path.realpath(path_part)
        except Exception as e:
            return False, f"Pfad kann nicht normalisiert werden: {e}"
        # Prüfen, ob die Datei existiert
        if not os.path.exists(real_path):
            return False, f"Datei existiert nicht: {real_path}"
        # Prüfen, ob der Pfad innerhalb eines erlaubten Basisverzeichnisses liegt
        allowed_bases = [os.path.realpath(p) for p in Constants.ALLOWED_FILE_BASE_DIRS]
        for base in allowed_bases:
            if real_path.startswith(base):
                return True, real_path
        # Falls nicht, zusätzlich prüfen, ob es sich um eine temporäre Datei handelt
        temp_dir = os.path.realpath(tempfile.gettempdir())
        if real_path.startswith(temp_dir):
            return True, real_path
        return False, f"Zugriff auf {real_path} nicht erlaubt (außerhalb erlaubter Verzeichnisse)"


PlatformUtils.setup_platform_environment()


# -----------------------------------------------------------------------------
# Verfügbarkeiten (unverändert)
# -----------------------------------------------------------------------------
TORCH_AVAILABLE = FastLazyLoader.is_available("torch")
NUMPY_AVAILABLE = FastLazyLoader.is_available("numpy")
TRANSLATOR_AVAILABLE = FastLazyLoader.is_available("deep_translator")
SCIPY_AVAILABLE = FastLazyLoader.is_available("scipy.signal")
FASTER_WHISPER_AVAILABLE = importlib.util.find_spec("faster_whisper") is not None
OPENAI_WHISPER_AVAILABLE = importlib.util.find_spec("whisper") is not None
WHISPER_AVAILABLE = FASTER_WHISPER_AVAILABLE or OPENAI_WHISPER_AVAILABLE
OLLAMA_AVAILABLE = importlib.util.find_spec("requests") is not None

if FASTER_WHISPER_AVAILABLE:
    logger.info("✅ faster-whisper verfügbar")
else:
    logger.warning("⚠️ faster-whisper nicht verfügbar")

if OPENAI_WHISPER_AVAILABLE:
    logger.info("✅ openai-whisper verfügbar")
else:
    logger.warning("⚠️ openai-whisper nicht verfügbar")

if not WHISPER_AVAILABLE:
    logger.warning("⚠️ KEINE Whisper-Bibliothek verfügbar! Starte im Demo-Modus.")

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, filedialog
    GUI_AVAILABLE = True
    logger.info("✅ GUI verfügbar")
except ImportError:
    GUI_AVAILABLE = False
    tk = None
    ttk = None
    scrolledtext = None
    logger.info("📟 Terminal-Modus (kein GUI)")


# -----------------------------------------------------------------------------
# Datenklassen (unverändert)
# -----------------------------------------------------------------------------
@dataclass
class ExcellenceTranscriptionResult:
    text: str
    confidence: float
    language: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class ExcellenceTranslationResult:
    original: str
    translated: str
    source_lang: str = "auto"
    target_lang: str = "de"
    timestamp: float = field(default_factory=time.time)
    start: Optional[float] = None
    end: Optional[float] = None


# -----------------------------------------------------------------------------
# SimplePerformanceTracker (unverändert)
# -----------------------------------------------------------------------------
class SimplePerformanceTracker:
    def __init__(self) -> None:
        self.transcription_count = 0
        self.translation_count = 0
        self.start_time = time.time()
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.RLock()

    def log_transcription(self) -> None:
        with self._lock:
            self.transcription_count += 1

    def log_translation(self) -> None:
        with self._lock:
            self.translation_count += 1

    def log_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def log_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def get_basic_stats(self) -> Dict[str, Any]:
        with self._lock:
            uptime_minutes = (time.time() - self.start_time) / 60
            total_cache = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0
            return {
                "transcriptions": self.transcription_count,
                "translations": self.translation_count,
                "uptime_minutes": uptime_minutes,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": f"{cache_hit_rate:.1%}",
                "timestamp": datetime.now().isoformat(),
            }


# -----------------------------------------------------------------------------
# Cache-Klassen (überarbeitet mit Konstanten)
# -----------------------------------------------------------------------------
class ExcellenceTTLCache:
    def __init__(self, maxsize: int = 128, ttl: float = 300.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._order: Deque[str] = deque()
        self._lock = threading.RLock()
        self._cleanup_interval = 300
        self._last_cleanup = time.time()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._stats_lock = threading.Lock()
        self._access_counter = 0

    def _perform_cleanup_if_needed(self) -> None:
        self._access_counter += 1
        if self._access_counter % 10 != 0:
            return
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        expired_keys = []
        for key in list(self._cache.keys()):
            value, timestamp = self._cache[key]
            if (current_time - timestamp) > self.ttl:
                expired_keys.append(key)
        for key in expired_keys:
            self._remove_key(key)
        self._last_cleanup = current_time

    def _remove_key(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
        if key in self._order:
            self._order.remove(key)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._perform_cleanup_if_needed()
            if key in self._cache:
                value, timestamp = self._cache[key]
                if (time.time() - timestamp) > self.ttl:
                    self._remove_key(key)
                    with self._stats_lock:
                        self._stats["misses"] += 1
                    return None
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
                self._order.append(key)
                with self._stats_lock:
                    self._stats["hits"] += 1
                return value
            with self._stats_lock:
                self._stats["misses"] += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._perform_cleanup_if_needed()
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self.maxsize:
                oldest = self._order.popleft()
                self._remove_key(oldest)
                with self._stats_lock:
                    self._stats["evictions"] += 1
            self._cache[key] = (value, time.time())
            self._order.append(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._order.clear()
            self._last_cleanup = time.time()

    def clear_expired(self) -> int:
        with self._lock:
            count = 0
            current_time = time.time()
            expired_keys = []
            for key in list(self._cache.keys()):
                value, timestamp = self._cache[key]
                if (current_time - timestamp) > self.ttl:
                    expired_keys.append(key)
            for key in expired_keys:
                self._remove_key(key)
                count += 1
            return count

    def get_stats(self) -> Dict[str, Any]:
        with self._lock, self._stats_lock:
            total_size = len(self._cache)
            expired = self.clear_expired()
            return {
                "total_entries": total_size,
                "expired_entries": expired,
                "max_size": self.maxsize,
                "ttl_seconds": self.ttl,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"]),
            }


# Globale Cache-Instanzen (mit Konstanten)
transcription_cache = ExcellenceTTLCache(
    maxsize=Constants.TRANSCRIPTION_CACHE_SIZE,
    ttl=Constants.TRANSCRIPTION_CACHE_TTL
)
translation_cache = ExcellenceTTLCache(
    maxsize=Constants.TRANSLATION_CACHE_SIZE,
    ttl=Constants.TRANSLATION_CACHE_TTL
)
audio_cache = ExcellenceTTLCache(
    maxsize=Constants.AUDIO_CACHE_SIZE,
    ttl=Constants.AUDIO_CACHE_TTL
)


def clear_expired_cache_entries() -> Dict[str, int]:
    return {
        "transcription_expired": transcription_cache.clear_expired(),
        "translation_expired": translation_cache.clear_expired(),
        "audio_expired": audio_cache.clear_expired(),
    }


def get_cache_stats() -> Dict[str, Any]:
    return {
        "transcription_cache": transcription_cache.get_stats(),
        "translation_cache": translation_cache.get_stats(),
        "audio_cache": audio_cache.get_stats(),
    }


def cache_transcription(result: ExcellenceTranscriptionResult) -> str:
    key = hashlib.sha256(result.text.encode()).hexdigest()
    transcription_cache.put(key, result)
    return key


def get_cached_transcription(text: str) -> Optional[ExcellenceTranscriptionResult]:
    key = hashlib.sha256(text.encode()).hexdigest()
    return transcription_cache.get(key)


def cache_translation(result: ExcellenceTranslationResult) -> str:
    key = hashlib.sha256((result.original + result.target_lang).encode()).hexdigest()
    translation_cache.put(key, result)
    return key


def get_cached_translation(original: str, target_lang: str) -> Optional[ExcellenceTranslationResult]:
    key = hashlib.sha256((original + target_lang).encode()).hexdigest()
    return translation_cache.get(key)


# -----------------------------------------------------------------------------
# ThreadPoolExecutor und Decorator (unverändert)
# -----------------------------------------------------------------------------
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ExcellenceExec")


def excellence_execution(timeout: int = 60, max_retries: int = 3) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    future = _EXECUTOR.submit(func, *args, **kwargs)
                    return future.result(timeout=timeout)
                except FutureTimeout as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"⏰ Timeout attempt {attempt + 1}/{max_retries + 1} for {func.__name__}")
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"⚠️ Exception in {func.__name__}: {str(e)[:100]}")
                if attempt < max_retries:
                    wait_time = min(30, 2**attempt)
                    time.sleep(wait_time)
                    continue
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Decorator logic failed for {func.__name__}.")
        return wrapper
    return decorator


class ExcellenceError(Exception):
    pass


def excellence_gui_operation(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (tk.TclError, RuntimeError):
            return None
        except Exception:
            return None
    return wrapper


# -----------------------------------------------------------------------------
# THEME-KLASSEN (unverändert)
# -----------------------------------------------------------------------------
class DarkTheme:
    BG_PRIMARY = "#0f1419"
    BG_SECONDARY = "#1a2129"
    BG_TERTIARY = "#242d38"
    BG_HOVER = "#2a3645"
    BG_CARD = "#1c252f"
    TEXT_PRIMARY = "#e6edf3"
    TEXT_SECONDARY = "#8b949e"
    TEXT_ACCENT = "#58a6ff"
    TEXT_MUTED = "#6e7681"
    DRAGON_GREEN = "#238636"
    DRAGON_GREEN_LIGHT = "#2ea043"
    DRAGON_BLUE = "#1f6feb"
    DRAGON_PURPLE = "#8957e5"
    SUCCESS = "#238636"
    WARNING = "#d29922"
    ERROR = "#f85149"
    INFO = "#58a6ff"
    BORDER = "#30363d"
    SCROLLBAR = "#3c444d"
    SCROLLBAR_HOVER = "#4c5560"
    INPUT_BG = "#161b22"
    INPUT_BORDER = "#30363d"
    INPUT_FOCUS = "#1f6feb"
    COMBO_BG = "#161b22"
    COMBO_FG = "#e6edf3"
    COMBO_BORDER = "#30363d"
    COMBO_SELECTION = "#1f6feb"
    CHECKBOX_BG = "#0d1117"
    CHECKBOX_FG = "#e6edf3"
    CHECKBOX_SELECTED = "#238636"
    CHECKBOX_ACTIVE = "#000000"
    SUBTITLE_ACTIVE = "#8957e5"
    SUBTITLE_INACTIVE = "#30363d"
    STATUS_BAR_BG = "#0d1117"
    STATUS_BAR_FG = "#8b949e"
    STATUS_BAR_ACCENT = "#58a6ff"


class LightTheme:
    BG_PRIMARY = "#ffffff"
    BG_SECONDARY = "#f0f0f0"
    BG_TERTIARY = "#e5e5e5"
    BG_HOVER = "#d9d9d9"
    BG_CARD = "#f5f5f5"
    TEXT_PRIMARY = "#000000"
    TEXT_SECONDARY = "#4a4a4a"
    TEXT_ACCENT = "#1f6feb"
    TEXT_MUTED = "#666666"
    DRAGON_GREEN = "#2ecc71"
    DRAGON_GREEN_LIGHT = "#27ae60"
    DRAGON_BLUE = "#3498db"
    DRAGON_PURPLE = "#9b59b6"
    SUCCESS = "#2ecc71"
    WARNING = "#f39c12"
    ERROR = "#e74c3c"
    INFO = "#3498db"
    BORDER = "#cccccc"
    SCROLLBAR = "#b3b3b3"
    SCROLLBAR_HOVER = "#999999"
    INPUT_BG = "#ffffff"
    INPUT_BORDER = "#cccccc"
    INPUT_FOCUS = "#1f6feb"
    COMBO_BG = "#ffffff"
    COMBO_FG = "#000000"
    COMBO_BORDER = "#cccccc"
    COMBO_SELECTION = "#1f6feb"
    CHECKBOX_BG = "#f0f0f0"
    CHECKBOX_FG = "#000000"
    CHECKBOX_SELECTED = "#2ecc71"
    CHECKBOX_ACTIVE = "#ffffff"
    SUBTITLE_ACTIVE = "#8957e5"
    SUBTITLE_INACTIVE = "#b3b3b3"
    STATUS_BAR_BG = "#f0f0f0"
    STATUS_BAR_FG = "#4a4a4a"
    STATUS_BAR_ACCENT = "#1f6feb"


CURRENT_THEME = None  # wird später gesetzt


class DragonFonts:
    TITLE = ("Segoe UI", 12, "bold")
    SUBTITLE = ("Segoe UI", 10, "bold")
    PRIMARY = ("Segoe UI", 9)
    SECONDARY = ("Segoe UI", 8)
    MONOSPACE = ("Cascadia Code", 9)
    BUTTON = ("Segoe UI", 9, "bold")
    STATUS = ("Segoe UI", 8)
    SMALL = ("Segoe UI", 7)


class RateLimiter:
    def __init__(self, max_updates_per_second: float = 30) -> None:
        self.min_interval = 1.0 / max_updates_per_second
        self.last_calls: Dict[str, float] = {}
        self._lock = threading.RLock()

    def can_update(self, update_type: str = "default") -> bool:
        with self._lock:
            now = time.time()
            last = self.last_calls.get(update_type, 0)
            if now - last >= self.min_interval:
                self.last_calls[update_type] = now
                return True
            return False

    def reset(self, update_type: Optional[str] = None) -> None:
        with self._lock:
            if update_type is None:
                self.last_calls.clear()
            elif update_type in self.last_calls:
                del self.last_calls[update_type]


# -----------------------------------------------------------------------------
# DARK MESSAGEBOX (unverändert)
# -----------------------------------------------------------------------------
class DarkMessageBox:
    @staticmethod
    def showinfo(title: str, message: str, parent: Optional[tk.Tk] = None) -> Optional[bool]:
        return DarkMessageBox._show_dialog(title, message, "info", parent)

    @staticmethod
    def showwarning(title: str, message: str, parent: Optional[tk.Tk] = None) -> Optional[bool]:
        return DarkMessageBox._show_dialog(title, message, "warning", parent)

    @staticmethod
    def showerror(title: str, message: str, parent: Optional[tk.Tk] = None) -> Optional[bool]:
        return DarkMessageBox._show_dialog(title, message, "error", parent)

    @staticmethod
    def askokcancel(title: str, message: str, parent: Optional[tk.Tk] = None) -> Optional[bool]:
        return DarkMessageBox._show_dialog(title, message, "question", parent, buttons=True)

    @staticmethod
    def askyesno(title: str, message: str, parent: Optional[tk.Tk] = None) -> Optional[bool]:
        return DarkMessageBox._ask_yesno(title, message, parent)

    @staticmethod
    def _show_dialog(title: str, message: str, msg_type: str,
                     parent: Optional[tk.Tk] = None, buttons: bool = False) -> Optional[bool]:
        try:
            if parent is None:
                parent = tk._default_root
            if not parent or not parent.winfo_exists():
                parent = DarkMessageBox._find_available_parent()
                if not parent:
                    return DarkMessageBox._fallback_messagebox(title, message, msg_type, buttons)

            dialog = tk.Toplevel(parent)
            dialog.title(f"🐉 {title}" if not title.startswith("🐉") else title)
            dialog.configure(bg=CURRENT_THEME.BG_PRIMARY)
            dialog.resizable(False, False)
            dialog.transient(parent)
            dialog.grab_set()

            timeout_seconds = 15 if any(word in title.lower()
                                        for word in ["beenden", "exit", "quit", "schließen"]) else 10
            timeout_id: Optional[str] = None

            def auto_close() -> None:
                nonlocal timeout_id
                try:
                    if dialog and dialog.winfo_exists():
                        logger.warning(f"⚠️ Dialog Timeout nach {timeout_seconds}s: '{title}'")
                        dialog.destroy()
                except Exception:
                    pass
            timeout_id = dialog.after(timeout_seconds * 1000, auto_close)

            icons = {
                "info": ("ℹ️", CURRENT_THEME.TEXT_ACCENT),
                "warning": ("⚠️", CURRENT_THEME.WARNING),
                "error": ("❌", CURRENT_THEME.ERROR),
                "question": ("❓", CURRENT_THEME.TEXT_ACCENT),
                "success": ("✅", CURRENT_THEME.SUCCESS),
            }
            icon, icon_color = icons.get(msg_type, ("💬", CURRENT_THEME.TEXT_PRIMARY))

            main_frame = tk.Frame(dialog, bg=CURRENT_THEME.BG_PRIMARY, padx=25, pady=25)
            main_frame.pack(fill="both", expand=True)

            content_frame = tk.Frame(main_frame, bg=CURRENT_THEME.BG_PRIMARY)
            content_frame.pack(fill="both", expand=True, pady=(0, 20))

            icon_frame = tk.Frame(content_frame, bg=CURRENT_THEME.BG_PRIMARY, width=60)
            icon_frame.pack(side="left", fill="y")
            icon_frame.pack_propagate(False)

            icon_label = tk.Label(
                icon_frame,
                text=icon,
                font=("Segoe UI", 24),
                bg=CURRENT_THEME.BG_PRIMARY,
                fg=icon_color,
            )
            icon_label.pack(expand=True)

            message_frame = tk.Frame(content_frame, bg=CURRENT_THEME.BG_PRIMARY)
            message_frame.pack(side="left", fill="both", expand=True, padx=(20, 0))

            if len(title) > 30:
                title_label = tk.Label(
                    message_frame,
                    text=title,
                    font=DragonFonts.SUBTITLE,
                    bg=CURRENT_THEME.BG_PRIMARY,
                    fg=CURRENT_THEME.TEXT_PRIMARY,
                    justify="left",
                    anchor="w",
                )
                title_label.pack(anchor="w", pady=(0, 10))

            message_label = tk.Label(
                message_frame,
                text=message,
                font=DragonFonts.PRIMARY,
                bg=CURRENT_THEME.BG_PRIMARY,
                fg=CURRENT_THEME.TEXT_PRIMARY,
                justify="left",
                wraplength=350,
                anchor="w",
            )
            message_label.pack(fill="x", expand=True, anchor="w")

            button_frame = tk.Frame(main_frame, bg=CURRENT_THEME.BG_PRIMARY)
            button_frame.pack(fill="x")

            result = {"value": None}

            def set_result(value: Optional[bool]) -> None:
                nonlocal timeout_id
                if timeout_id:
                    try:
                        dialog.after_cancel(timeout_id)
                    except Exception:
                        pass
                    timeout_id = None
                result["value"] = value
                try:
                    if dialog.winfo_exists():
                        dialog.destroy()
                except Exception:
                    pass

            if buttons:
                cancel_btn = tk.Button(
                    button_frame,
                    text="Abbrechen",
                    command=lambda: set_result(False),
                    bg=CURRENT_THEME.BG_TERTIARY,
                    fg=CURRENT_THEME.TEXT_PRIMARY,
                    font=DragonFonts.BUTTON,
                    relief="flat",
                    padx=22,
                    pady=8,
                    cursor="hand2",
                    takefocus=True,
                )
                cancel_btn.pack(side="right", padx=(10, 0))

                ok_btn = tk.Button(
                    button_frame,
                    text="OK",
                    command=lambda: set_result(True),
                    bg=CURRENT_THEME.SUCCESS,
                    fg=CURRENT_THEME.TEXT_PRIMARY,
                    font=DragonFonts.BUTTON,
                    relief="flat",
                    padx=25,
                    pady=8,
                    cursor="hand2",
                    takefocus=True,
                )
                ok_btn.pack(side="right")

                dialog.bind("<Return>", lambda e: set_result(True))
                dialog.bind("<Escape>", lambda e: set_result(False))
                dialog.bind("<space>", lambda e: cancel_btn.focus_set())

                is_exit_dialog = any(word in title.lower()
                                     for word in ["beenden", "exit", "quit", "schließen"])
                if is_exit_dialog:
                    cancel_btn.focus_set()
                else:
                    ok_btn.focus_set()
            else:
                ok_btn = tk.Button(
                    button_frame,
                    text="OK",
                    command=lambda: set_result(True),
                    bg=CURRENT_THEME.SUCCESS,
                    fg=CURRENT_THEME.TEXT_PRIMARY,
                    font=DragonFonts.BUTTON,
                    relief="flat",
                    padx=25,
                    pady=8,
                    cursor="hand2",
                    takefocus=True,
                )
                ok_btn.pack(side="right")

                dialog.bind("<Return>", lambda e: set_result(True))
                dialog.bind("<Escape>", lambda e: set_result(True))
                dialog.bind("<space>", lambda e: set_result(True))
                ok_btn.focus_set()

            def on_closing() -> None:
                set_result(False if buttons else True)

            dialog.protocol("WM_DELETE_WINDOW", on_closing)
            DarkMessageBox._center_dialog(dialog, parent)
            parent.wait_window(dialog)
            return result["value"]
        except (tk.TclError, RuntimeError, AttributeError) as e:
            logger.warning(f"⚠️ DarkMessageBox Error: {e}")
            return DarkMessageBox._fallback_messagebox(title, message, msg_type, buttons)

    @staticmethod
    def _ask_yesno(title: str, message: str, parent: Optional[tk.Tk] = None) -> Optional[bool]:
        try:
            if parent is None:
                parent = tk._default_root
            if not parent or not parent.winfo_exists():
                parent = DarkMessageBox._find_available_parent()
                if not parent:
                    import tkinter.messagebox as mb
                    return mb.askyesno(title, message)
            dialog = tk.Toplevel(parent)
            dialog.title(f"🐉 {title}")
            dialog.configure(bg=CURRENT_THEME.BG_PRIMARY)
            dialog.resizable(False, False)
            dialog.transient(parent)
            dialog.grab_set()
            timeout_id = dialog.after(15000, lambda: dialog.destroy() if dialog.winfo_exists() else None)

            main_frame = tk.Frame(dialog, bg=CURRENT_THEME.BG_PRIMARY, padx=30, pady=25)
            main_frame.pack(fill="both", expand=True)

            icon_label = tk.Label(
                main_frame,
                text="❓",
                font=("Segoe UI", 28),
                bg=CURRENT_THEME.BG_PRIMARY,
                fg=CURRENT_THEME.TEXT_ACCENT,
            )
            icon_label.pack(pady=(0, 20))

            message_label = tk.Label(
                main_frame,
                text=message,
                font=DragonFonts.PRIMARY,
                bg=CURRENT_THEME.BG_PRIMARY,
                fg=CURRENT_THEME.TEXT_PRIMARY,
                justify="center",
                wraplength=350,
            )
            message_label.pack(pady=(0, 30))

            button_frame = tk.Frame(main_frame, bg=CURRENT_THEME.BG_PRIMARY)
            button_frame.pack(fill="x")

            result = {"value": None}

            def set_result(value: bool) -> None:
                if timeout_id:
                    try:
                        dialog.after_cancel(timeout_id)
                    except Exception:
                        pass
                result["value"] = value
                if dialog.winfo_exists():
                    dialog.destroy()

            def on_yes() -> None:
                set_result(True)

            def on_no() -> None:
                set_result(False)

            yes_btn = tk.Button(
                button_frame,
                text="  👍 Ja  ",
                command=on_yes,
                bg=CURRENT_THEME.SUCCESS,
                fg=CURRENT_THEME.TEXT_PRIMARY,
                font=("Segoe UI", 10, "bold"),
                relief="flat",
                padx=25,
                pady=10,
                cursor="hand2",
            )
            yes_btn.pack(side="left", expand=True, padx=(0, 10))

            no_btn = tk.Button(
                button_frame,
                text="  👎 Nein  ",
                command=on_no,
                bg=CURRENT_THEME.ERROR,
                fg=CURRENT_THEME.TEXT_PRIMARY,
                font=("Segoe UI", 10, "bold"),
                relief="flat",
                padx=25,
                pady=10,
                cursor="hand2",
            )
            no_btn.pack(side="right", expand=True)

            dialog.bind("<Return>", lambda e: on_yes())
            dialog.bind("<Escape>", lambda e: on_no())
            dialog.bind("y", lambda e: on_yes())
            dialog.bind("n", lambda e: on_no())
            yes_btn.focus_set()
            DarkMessageBox._center_dialog(dialog, parent)
            parent.wait_window(dialog)
            return result["value"]
        except (tk.TclError, RuntimeError, AttributeError):
            import tkinter.messagebox as mb
            return mb.askyesno(title, message)

    @staticmethod
    def _find_available_parent() -> Optional[tk.Tk]:
        try:
            if not tk._default_root:
                return None
            for widget in tk._default_root.winfo_children():
                if widget.winfo_exists():
                    return widget
            return tk._default_root
        except (tk.TclError, AttributeError):
            return None

    @staticmethod
    def _center_dialog(dialog: tk.Toplevel, parent: tk.Tk) -> None:
        try:
            dialog.update_idletasks()
            if parent and parent.winfo_exists():
                parent_x = parent.winfo_x()
                parent_y = parent.winfo_y()
                parent_width = parent.winfo_width()
                parent_height = parent.winfo_height()
                dialog_width = dialog.winfo_width()
                dialog_height = dialog.winfo_height()
                x = parent_x + (parent_width - dialog_width) // 2
                y = parent_y + (parent_height - dialog_height) // 2
                screen_width = parent.winfo_screenwidth()
                screen_height = parent.winfo_screenheight()
                x = max(10, min(x, screen_width - dialog_width - 10))
                y = max(10, min(y, screen_height - dialog_height - 10))
                dialog.geometry(f"+{x}+{y}")
                dialog.lift()
                dialog.focus_force()
        except Exception:
            try:
                screen_width = dialog.winfo_screenwidth()
                screen_height = dialog.winfo_screenheight()
                dialog_width = dialog.winfo_width()
                dialog_height = dialog.winfo_height()
                x = (screen_width - dialog_width) // 2
                y = (screen_height - dialog_height) // 2
                dialog.geometry(f"+{x}+{y}")
            except Exception:
                pass

    @staticmethod
    def _fallback_messagebox(title: str, message: str, msg_type: str,
                             buttons: bool = False) -> Optional[bool]:
        try:
            import tkinter.messagebox as mb
            if buttons:
                return mb.askokcancel(title, message)
            else:
                if msg_type == "error":
                    mb.showerror(title, message)
                elif msg_type == "warning":
                    mb.showwarning(title, message)
                else:
                    mb.showinfo(title, message)
                return None
        except Exception:
            logger.info(f"💬 {title}: {message}")
            if buttons:
                return False
            return None

    @staticmethod
    def show_progress(title: str, message: str, parent: Optional[tk.Tk] = None,
                      indeterminate: bool = True) -> Any:
        dialog = None
        progress = None

        def create_dialog() -> None:
            nonlocal dialog, progress
            try:
                root_window = parent if parent else tk._default_root
                dialog = tk.Toplevel(root_window)
                dialog.title(f"🐉 {title}")
                dialog.configure(bg=CURRENT_THEME.BG_PRIMARY)
                dialog.resizable(False, False)
                dialog.transient(root_window)

                main_frame = tk.Frame(dialog, bg=CURRENT_THEME.BG_PRIMARY, padx=30, pady=25)
                main_frame.pack(fill="both", expand=True)

                message_label = tk.Label(
                    main_frame,
                    text=message,
                    font=DragonFonts.PRIMARY,
                    bg=CURRENT_THEME.BG_PRIMARY,
                    fg=CURRENT_THEME.TEXT_PRIMARY,
                    justify="center",
                )
                message_label.pack(pady=(0, 20))

                progress = ttk.Progressbar(
                    main_frame,
                    mode="indeterminate" if indeterminate else "determinate",
                    length=300,
                )
                progress.pack(pady=(0, 10))

                progress.start(10)
                DarkMessageBox._center_dialog(dialog, root_window)
            except (tk.TclError, RuntimeError) as e:
                logger.warning(f"⚠️ Progress Dialog Error: {e}")

        def close_dialog() -> None:
            try:
                if progress:
                    progress.stop()
                if dialog and dialog.winfo_exists():
                    dialog.destroy()
            except Exception:
                pass

        if parent and parent.winfo_exists():
            parent.after(0, create_dialog)
        else:
            create_dialog()

        class ProgressController:
            def __init__(self) -> None:
                self.dialog = dialog
                self.progress = progress

            def close(self) -> None:
                close_dialog()

            def update_message(self, new_message: str) -> None:
                if dialog and dialog.winfo_exists():
                    try:
                        for widget in dialog.winfo_children():
                            if isinstance(widget, tk.Frame):
                                for child in widget.winfo_children():
                                    if isinstance(child, tk.Label):
                                        child.config(text=new_message)
                                        break
                    except Exception:
                        pass

        return ProgressController()


# -----------------------------------------------------------------------------
# MEMORY MANAGER (unverändert, bis auf kleine Optimierung)
# -----------------------------------------------------------------------------
class ExcellenceMemoryManager:
    def __init__(self) -> None:
        self._buffers: Dict[str, Deque[str]] = {}
        self._buffer_sizes: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._max_memory_per_component = 100 * 1024 * 1024  # 100 MB
        self._last_gc_time = time.time()
        self._gc_interval = 300
        self._ring_buffers: Dict[str, List[Optional[str]]] = {}
        self._ring_buffer_pointers: Dict[str, int] = {}
        self._ring_buffer_sizes: Dict[str, int] = {}
        self._memory_warning_threshold = 0.8
        self._long_term_monitor: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._monitoring_active = True
        self._maintenance_thread: Optional[threading.Thread] = None
        self._maintenance_stop = threading.Event()
        self._start_maintenance()

    def _start_maintenance(self) -> None:
        def maintenance_worker() -> None:
            while not self._maintenance_stop.is_set():
                try:
                    time.sleep(60)
                    self._perform_periodic_maintenance()
                    self._perform_memory_health_check()
                except Exception as e:
                    logger.warning(f"⚠️ Maintenance worker error: {e}")
        self._maintenance_thread = threading.Thread(
            target=maintenance_worker, daemon=True, name="MemoryMaintenance"
        )
        self._maintenance_thread.start()

    def _perform_memory_health_check(self) -> None:
        try:
            import psutil
        except ImportError:
            return

        try:
            system_memory = psutil.virtual_memory()
            system_usage_percent = system_memory.percent / 100.0
            process = psutil.Process()
            process_memory = process.memory_info().rss
            process_usage_percent = process_memory / Constants.MAX_MEMORY_USAGE
            memory_sample = {
                "timestamp": time.time(),
                "system_usage": system_usage_percent,
                "process_usage": process_usage_percent,
                "system_mb": system_memory.used // (1024 * 1024),
                "process_mb": process_memory // (1024 * 1024),
            }
            self._long_term_monitor.append(memory_sample)
            if system_usage_percent > self._memory_warning_threshold:
                logger.warning(f"⚠️ High system memory usage: {system_memory.percent:.1f}%")
            if process_usage_percent > self._memory_warning_threshold:
                logger.warning(f"⚠️ High process memory usage: {process_usage_percent:.1%}")
                self._aggressive_cleanup()
            if len(self._long_term_monitor) >= 10:
                recent_samples = list(self._long_term_monitor)[-10:]
                avg_usage = sum(s["system_usage"] for s in recent_samples) / len(recent_samples)
                if avg_usage > 0.75:
                    logger.warning(f"⚠️ Sustained high memory usage: {avg_usage:.1%}")
        except Exception as e:
            logger.warning(f"⚠️ Memory health check error: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        try:
            import psutil
        except ImportError:
            return {}

        try:
            system_memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss
            with self._lock:
                buffer_count = len(self._buffers)
                ring_buffer_count = len(self._ring_buffers)
                total_buffer_size = sum(self._buffer_sizes.values())
            return {
                "system_usage_percent": system_memory.percent,
                "system_used_mb": system_memory.used // (1024 * 1024),
                "system_total_mb": system_memory.total // (1024 * 1024),
                "process_usage_percent": (process_memory / Constants.MAX_MEMORY_USAGE) * 100,
                "process_used_mb": process_memory // (1024 * 1024),
                "process_peak_mb": self._get_peak_memory() // (1024 * 1024),
                "long_term_samples": len(self._long_term_monitor),
                "buffer_components": buffer_count,
                "ring_buffer_components": ring_buffer_count,
                "total_buffer_size_mb": total_buffer_size // (1024 * 1024),
                "active_monitoring": self._monitoring_active,
            }
        except Exception as e:
            logger.warning(f"⚠️ Memory stats error: {e}")
            return {}

    def _get_peak_memory(self) -> int:
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0

    def _perform_periodic_maintenance(self) -> None:
        with self._lock:
            current_time = time.time()
            if current_time - self._last_gc_time > self._gc_interval:
                gc.collect()
                self._last_gc_time = current_time
            total_memory = sum(self._buffer_sizes.values())
            memory_usage_percent = total_memory / self._max_memory_per_component
            if memory_usage_percent > 0.8:
                logger.warning(f"⚠️ High buffer memory: {memory_usage_percent:.1%}")
                cleanup_thread = threading.Thread(
                    target=self._aggressive_cleanup, daemon=True
                )
                cleanup_thread.start()

    def _aggressive_cleanup(self) -> None:
        logger.info("🧹 Starting aggressive memory cleanup...")
        with self._lock:
            components = list(self._buffers.keys())
        for component in components:
            try:
                with self._lock:
                    if component in self._buffers:
                        buffer_size = len(self._buffers[component])
                        if buffer_size > 100:
                            keep_count = max(50, int(buffer_size * 0.5))
                            current_buffer = self._buffers[component]
                            new_deque = deque(
                                list(current_buffer)[-keep_count:],
                                maxlen=Constants.MAX_TEXT_LINES
                            )
                            self._buffers[component] = new_deque
                            self._buffer_sizes[component] = sum(
                                len(str(text).encode("utf-8")) if text else 0
                                for text in new_deque
                            )
                            logger.info(f"  ↪ {component}: {buffer_size} → {keep_count} entries")
            except Exception as e:
                logger.warning(f"⚠️ Component cleanup error for {component}: {e}")

        def async_gc() -> None:
            gc.collect()
        gc_thread = threading.Thread(target=async_gc, daemon=True)
        gc_thread.start()
        logger.info("✅ Aggressive cleanup completed")

    def add_text(self, component: str, text: str) -> None:
        if not text or not text.strip():
            return
        with self._lock:
            if component in self._ring_buffers:
                self._add_to_ring_buffer(component, text)
                return
            if component not in self._buffers:
                self._buffers[component] = deque(maxlen=Constants.MAX_TEXT_LINES)
                self._buffer_sizes[component] = 0
            text_size = len(text.encode("utf-8"))
            current_size = self._buffer_sizes[component]
            if current_size + text_size > self._max_memory_per_component:
                self._optimize_buffer(component)
            self._buffers[component].append(text)
            self._buffer_sizes[component] += text_size

    def _add_to_ring_buffer(self, component: str, text: str) -> None:
        with self._lock:
            if component not in self._ring_buffers:
                buffer_size = Constants.MAX_TEXT_LINES
                self._ring_buffers[component] = [None] * buffer_size
                self._ring_buffer_pointers[component] = 0
                self._ring_buffer_sizes[component] = 0
                self._buffer_sizes[component] = 0
            ring_buffer = self._ring_buffers[component]
            pointer = self._ring_buffer_pointers[component]
            text_size = len(text.encode("utf-8"))
            old_text = ring_buffer[pointer]
            if old_text is not None:
                old_size = len(old_text.encode("utf-8"))
                self._buffer_sizes[component] -= old_size
            ring_buffer[pointer] = text
            self._buffer_sizes[component] += text_size
            self._ring_buffer_pointers[component] = (pointer + 1) % len(ring_buffer)
            if self._ring_buffer_sizes[component] < len(ring_buffer):
                self._ring_buffer_sizes[component] += 1

    def _optimize_buffer(self, component: str) -> None:
        if component in self._ring_buffers:
            current_size = self._ring_buffer_sizes[component]
            if current_size > Constants.MAX_TEXT_LINES // 2:
                new_size = Constants.MAX_TEXT_LINES // 2
                self._resize_ring_buffer(component, new_size)
            return
        if component in self._buffers:
            keep_count = int(len(self._buffers[component]) * 0.7)
            if keep_count > 0:
                new_deque = deque(
                    list(self._buffers[component])[-keep_count:],
                    maxlen=Constants.MAX_TEXT_LINES
                )
                self._buffers[component] = new_deque
                self._buffer_sizes[component] = sum(
                    len(text.encode("utf-8")) for text in self._buffers[component]
                )
                logger.debug(f"🧹 Buffer {component} optimized: {keep_count} entries kept")

    def _resize_ring_buffer(self, component: str, new_size: int) -> None:
        if component not in self._ring_buffers:
            return
        with self._lock:
            old_buffer = self._ring_buffers[component]
            old_pointer = self._ring_buffer_pointers[component]
            old_size = self._ring_buffer_sizes[component]
            new_buffer = [None] * new_size
            new_pointer = 0
            new_buffer_size = 0
            new_total_size = 0
            start_idx = (old_pointer - min(old_size, new_size)) % len(old_buffer)
            for i in range(min(old_size, new_size)):
                idx = (start_idx + i) % len(old_buffer)
                text = old_buffer[idx]
                if text is not None:
                    new_buffer[new_pointer] = text
                    new_total_size += len(text.encode("utf-8"))
                    new_pointer = (new_pointer + 1) % new_size
                    new_buffer_size += 1
            self._ring_buffers[component] = new_buffer
            self._ring_buffer_pointers[component] = new_pointer
            self._ring_buffer_sizes[component] = new_buffer_size
            self._buffer_sizes[component] = new_total_size
            logger.debug(f"🧹 Ring buffer {component} resized: {old_size} → {new_size}")

    def get_text(self, component: str) -> str:
        with self._lock:
            if component in self._ring_buffers:
                return self._get_from_ring_buffer(component)
            elif component in self._buffers:
                return "\n".join(self._buffers[component])
            return ""

    def _get_from_ring_buffer(self, component: str) -> str:
        if component not in self._ring_buffers:
            return ""
        with self._lock:
            ring_buffer = self._ring_buffers[component]
            pointer = self._ring_buffer_pointers[component]
            buffer_size = self._ring_buffer_sizes[component]
            total_size = len(ring_buffer)
            if buffer_size == 0:
                return ""
            texts: List[str] = []
            for i in range(buffer_size):
                idx = (pointer - buffer_size + i) % total_size
                if idx < 0:
                    idx += total_size
                text = ring_buffer[idx]
                if text is not None:
                    texts.append(text)
            return "\n".join(texts)

    def clear_component(self, component: str) -> None:
        with self._lock:
            if component in self._buffers:
                del self._buffers[component]
            if component in self._buffer_sizes:
                del self._buffer_sizes[component]
            if component in self._ring_buffers:
                del self._ring_buffers[component]
            if component in self._ring_buffer_pointers:
                del self._ring_buffer_pointers[component]
            if component in self._ring_buffer_sizes:
                del self._ring_buffer_sizes[component]
            logger.info(f"🧹 Component {component} cleared")

    def get_buffer_stats(self, component: str) -> Dict[str, Any]:
        with self._lock:
            if component in self._ring_buffers:
                return {
                    "type": "ring_buffer",
                    "size": self._ring_buffer_sizes.get(component, 0),
                    "capacity": len(self._ring_buffers[component]),
                    "memory_bytes": self._buffer_sizes.get(component, 0),
                    "pointer": self._ring_buffer_pointers.get(component, 0),
                }
            elif component in self._buffers:
                return {
                    "type": "deque",
                    "size": len(self._buffers[component]),
                    "capacity": Constants.MAX_TEXT_LINES,
                    "memory_bytes": self._buffer_sizes.get(component, 0),
                    "maxlen": Constants.MAX_TEXT_LINES,
                }
            return {"type": "not_found"}

    def list_components(self) -> List[str]:
        with self._lock:
            all_components = set(self._buffers.keys())
            all_components.update(self._ring_buffers.keys())
            return list(all_components)

    def get_total_memory_usage(self) -> int:
        with self._lock:
            return sum(self._buffer_sizes.values())

    def optimize_all_buffers(self) -> None:
        logger.info("🧹 Optimizing all buffers...")
        with self._lock:
            components = list(self._buffers.keys()) + list(self._ring_buffers.keys())
        for component in components:
            self._optimize_buffer(component)
        gc.collect()
        logger.info("✅ All buffers optimized")

    def dispose(self) -> None:
        self._monitoring_active = False
        self._maintenance_stop.set()
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=1.0)
        with self._lock:
            self._buffers.clear()
            self._buffer_sizes.clear()
            self._ring_buffers.clear()
            self._ring_buffer_pointers.clear()
            self._ring_buffer_sizes.clear()
            self._long_term_monitor.clear()
        gc.collect()

    def print_debug_info(self) -> None:
        stats = self.get_memory_stats()
        logger.info("\n" + "=" * 50)
        logger.info("🧠 MEMORY MANAGER DEBUG INFO")
        logger.info("=" * 50)
        logger.info(f"System Memory: {stats.get('system_used_mb', 0)}MB / "
                    f"{stats.get('system_total_mb', 0)}MB ({stats.get('system_usage_percent', 0):.1f}%)")
        logger.info(f"Process Memory: {stats.get('process_used_mb', 0)}MB "
                    f"({stats.get('process_usage_percent', 0):.1f}%)")
        logger.info(f"Buffer Components: {stats.get('buffer_components', 0)}")
        logger.info(f"Ring Buffer Components: {stats.get('ring_buffer_components', 0)}")
        logger.info(f"Total Buffer Size: {stats.get('total_buffer_size_mb', 0)}MB")
        logger.info(f"Long Term Samples: {stats.get('long_term_samples', 0)}")
        components = self.list_components()
        if components:
            logger.info(f"\nActive Components ({len(components)}):")
            for comp in components[:5]:
                comp_stats = self.get_buffer_stats(comp)
                logger.info(f"  • {comp}: {comp_stats['type']}, "
                            f"size: {comp_stats.get('size', 0)}")
            if len(components) > 5:
                logger.info(f"  ... and {len(components) - 5} more")
        logger.info("=" * 50)


# -----------------------------------------------------------------------------
# SUPPORTED LANGUAGES & SETTINGS (unverändert)
# -----------------------------------------------------------------------------
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "auto": "Automatisch",
    "de": "Deutsch", "en": "Englisch", "fr": "Französisch", "es": "Spanisch",
    "it": "Italienisch", "pt": "Portugiesisch", "nl": "Niederländisch",
    "pl": "Polnisch", "ru": "Russisch", "ja": "Japanisch",
    "zh": "Chinesisch", "ko": "Koreanisch", "ar": "Arabisch",
    "hi": "Hindi", "tr": "Türkisch", "vi": "Vietnamesisch",
    "th": "Thailändisch", "id": "Indonesisch", "ms": "Malaysisch",
    "fa": "Persisch", "he": "Hebräisch", "bn": "Bengalisch",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
    "kn": "Kannada", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu", "sv": "Schwedisch",
    "da": "Dänisch", "no": "Norwegisch", "fi": "Finnisch",
    "cs": "Tschechisch", "hu": "Ungarisch", "ro": "Rumänisch",
    "bg": "Bulgarisch", "el": "Griechisch", "sk": "Slowakisch",
    "hr": "Kroatisch", "sr": "Serbisch", "uk": "Ukrainisch",
    "ca": "Katalanisch", "eu": "Baskisch", "gl": "Galizisch",
}

SORTED_LANGUAGES: List[Tuple[str, str]] = sorted(
    [(name, code) for code, name in SUPPORTED_LANGUAGES.items()],
    key=lambda x: x[0]
)

LANGUAGE_SHORT_CODES: Dict[str, str] = {
    "auto": "Auto", "de": "Deu", "en": "Eng", "fr": "Fra", "es": "Esp",
    "it": "Ita", "pt": "Por", "nl": "Nld", "pl": "Pol", "ru": "Rus",
    "ja": "Jpn", "zh": "Chi", "ko": "Kor", "ar": "Ara", "hi": "Hin",
    "tr": "Tur", "vi": "Vie", "th": "Tha", "id": "Ind", "ms": "Msa",
    "fa": "Per", "he": "Heb", "sv": "Swe", "da": "Dan", "no": "Nor",
    "fi": "Fin", "cs": "Cze", "hu": "Hun", "ro": "Rom", "bg": "Bul",
    "el": "Gre", "sk": "Slo", "hr": "Hrv", "uk": "Ukr",
}

WHISPER_MODELS: List[str] = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v2",
    "large-v3",
]


class AsianLanguageSupport:
    @staticmethod
    def should_use_word_segmentation(language_code: str) -> bool:
        return language_code in ["zh", "ja", "ko", "th"]

    @staticmethod
    def optimize_display_text(text: str, language_code: str) -> str:
        if language_code == "zh":
            return " ".join(text)
        elif language_code == "ja":
            return text.replace("。", ". ").replace("、", ", ")
        return text


# -----------------------------------------------------------------------------
# ADVANCED SETTINGS (überarbeitet mit Konstanten)
# -----------------------------------------------------------------------------
class AdvancedSettings:
    def __init__(self,
                 beam_size: int = Constants.DEFAULT_BEAM_SIZE,
                 temperature: float = Constants.DEFAULT_TEMPERATURE,
                 vad_filter: bool = Constants.ENABLE_VAD_FILTER,
                 max_cache_size: int = 200,
                 auto_save_interval: int = 300,
                 enable_sentiment_analysis: bool = False,
                 enable_speaker_diarization: bool = False,
                 max_memory_mb: int = 1024,
                 gpu_acceleration: bool = True,
                 optimize_translations: bool = False,
                 config_type: str = 'default') -> None:
        self.config = get_config(config_type)
        self.beam_size = beam_size
        self.temperature = temperature
        self.vad_filter = vad_filter
        self.gpu_acceleration = gpu_acceleration
        self.max_cache_size = max_cache_size
        self.auto_save_interval = auto_save_interval
        self.max_memory_mb = max_memory_mb
        self.enable_sentiment_analysis = enable_sentiment_analysis
        self.enable_speaker_diarization = enable_speaker_diarization
        self.optimize_translations = optimize_translations
        self.chunk_duration = self.config.CHUNK_DURATION

        self.vad_threshold: float = Constants.VAD_THRESHOLD
        self.vad_min_speech_duration_ms: int = Constants.VAD_MIN_SPEECH_DURATION_MS
        self.vad_min_silence_duration_ms: int = Constants.VAD_MIN_SILENCE_DURATION_MS

        logger.info("🔊 AdvancedSettings initialized:")
        logger.info(f"   Config Type: {config_type}")
        logger.info(f"   SAMPLE_RATE: {self.config.SAMPLE_RATE}")
        logger.info(f"   CHANNELS: {self.config.CHANNELS}")
        logger.info(f"   CHUNK_DURATION: {self.chunk_duration}s")
        logger.info(f"   CHUNK_SIZE: {self.config.CHUNK_SIZE_BYTES:,} bytes")
        logger.info(f"   BEAM_SIZE: {self.beam_size}")
        logger.info(f"   GPU_ACCELERATION: {self.gpu_acceleration}")
        logger.info(f"   VAD_THRESHOLD: {self.vad_threshold} (optimiert)")
        logger.info(f"   VAD_MIN_SPEECH_MS: {self.vad_min_speech_duration_ms} (optimiert)")
        logger.info(f"   VAD_MIN_SILENCE_MS: {self.vad_min_silence_duration_ms} (optimiert)")

    @classmethod
    def load_from_file(cls, filename: str = "dragon_advanced_settings.json") -> 'AdvancedSettings':
        try:
            config_dir = PlatformUtils.get_platform_config_dir()
            file_path = config_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except (json.JSONDecodeError, PermissionError, OSError) as e:
                    logger.error(f"Fehler beim Laden der Einstellungen: {e}")
                    if DEBUG_LEVEL >= 2:
                        logger.exception("Stacktrace:")
                    return cls()
                import inspect
                signature = inspect.signature(cls.__init__)
                valid_params = list(signature.parameters.keys())
                if 'self' in valid_params:
                    valid_params.remove('self')
                filtered_data = {k: v for k, v in data.items() if k in valid_params}
                config_type = data.get('config_type', 'default')
                if 'config_type' not in filtered_data:
                    filtered_data['config_type'] = config_type
                instance = cls(**filtered_data)

                instance.vad_threshold = data.get('vad_threshold', Constants.VAD_THRESHOLD)
                instance.vad_min_speech_duration_ms = data.get('vad_min_speech_duration_ms', Constants.VAD_MIN_SPEECH_DURATION_MS)
                instance.vad_min_silence_duration_ms = data.get('vad_min_silence_duration_ms', Constants.VAD_MIN_SILENCE_DURATION_MS)

                if 'chunk_duration' in data:
                    try:
                        instance.config.CHUNK_DURATION = float(data['chunk_duration'])
                    except (ValueError, AttributeError):
                        pass
                if 'sample_rate' in data:
                    try:
                        instance.config.SAMPLE_RATE = int(data.get('sample_rate', Constants.SAMPLE_RATE))
                    except (ValueError, AttributeError):
                        pass
                if 'channels' in data:
                    try:
                        instance.config.CHANNELS = int(data.get('channels', Constants.CHANNELS))
                    except (ValueError, AttributeError):
                        pass
                if 'audio_format' in data:
                    try:
                        instance.config.AUDIO_FORMAT = str(data.get('audio_format', Constants.AUDIO_FORMAT))
                    except (ValueError, AttributeError):
                        pass
                logger.info(f"✅ Settings loaded successfully (Config Type: {config_type})")
                return instance
        except FileNotFoundError:
            logger.info("Keine gespeicherten Einstellungen gefunden, verwende Standard")
        except Exception as e:
            logger.error(f"❌ Error loading advanced settings: {e}")
        logger.info("📝 Using default settings")
        return cls()

    def save_to_file(self, filename: str = "dragon_advanced_settings.json") -> None:
        try:
            config_dir = PlatformUtils.get_platform_config_dir()
            file_path = config_dir / filename
            config_type = 'default'
            if isinstance(self.config, RealtimeConfig):
                config_type = 'realtime'
            elif isinstance(self.config, HighAccuracyConfig):
                config_type = 'high_accuracy'
            elif isinstance(self.config, YouTubeOptimizedConfig):
                config_type = 'youtube'
            save_dict = {
                'beam_size': self.beam_size,
                'temperature': self.temperature,
                'vad_filter': self.vad_filter,
                'config_type': config_type,
                'max_cache_size': self.max_cache_size,
                'auto_save_interval': self.auto_save_interval,
                'enable_sentiment_analysis': self.enable_sentiment_analysis,
                'enable_speaker_diarization': self.enable_speaker_diarization,
                'max_memory_mb': self.max_memory_mb,
                'gpu_acceleration': self.gpu_acceleration,
                'optimize_translations': self.optimize_translations,
                'chunk_duration': self.config.CHUNK_DURATION,
                'sample_rate': self.config.SAMPLE_RATE,
                'channels': self.config.CHANNELS,
                'audio_format': self.config.AUDIO_FORMAT,
                'vad_threshold': self.vad_threshold,
                'vad_min_speech_duration_ms': self.vad_min_speech_duration_ms,
                'vad_min_silence_duration_ms': self.vad_min_silence_duration_ms,
            }
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Settings saved to {file_path} (Config Type: {config_type})")
            except (OSError, PermissionError) as e:
                logger.error(f"Fehler beim Schreiben der Einstellungen: {e}")
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Speichern: {e}")
                if DEBUG_LEVEL >= 2:
                    logger.exception("Stacktrace:")
        except Exception as e:
            logger.error(f"❌ Error saving settings: {e}")

    def repair(self) -> List[str]:
        logger.info("🔧 Repairing AdvancedSettings...")
        repairs_made: List[str] = []
        if not hasattr(self, 'config'):
            self.config = ExcellenceConfig()
            repairs_made.append('Added ExcellenceConfig')
        if not hasattr(self, 'chunk_duration'):
            self.chunk_duration = self.config.CHUNK_DURATION
            repairs_made.append(f'Added chunk_duration from config: {self.chunk_duration}s')
        if not self.config.validate_config():
            logger.warning("⚠️ Config validation failed, resetting to default")
            self.config = ExcellenceConfig()
            repairs_made.append('Config reset to default')
        if repairs_made:
            logger.info(f"✅ Repairs made: {', '.join(repairs_made)}")
            self.save_to_file()
        else:
            logger.info("✅ No repairs needed")
        return repairs_made

    def validate(self) -> List[str]:
        issues: List[str] = []
        if not self.config.validate_config():
            issues.append("ExcellenceConfig validation failed")
        if self.beam_size < 1 or self.beam_size > 20:
            issues.append(f"Invalid beam_size: {self.beam_size} (should be 1-20)")
        if not (0.0 <= self.temperature <= 2.0):
            issues.append(f"Invalid temperature: {self.temperature} (should be 0.0-2.0)")
        if self.max_memory_mb < 100 or self.max_memory_mb > 16384:
            issues.append(f"Invalid max_memory_mb: {self.max_memory_mb} (should be 100-16384)")
        if not (self.config.MIN_CHUNK_DURATION <= self.config.CHUNK_DURATION <= self.config.MAX_CHUNK_DURATION):
            issues.append(f"Invalid CHUNK_DURATION: {self.config.CHUNK_DURATION}s "
                          f"(should be {self.config.MIN_CHUNK_DURATION}-{self.config.MAX_CHUNK_DURATION}s)")
        return issues

    def set_config_type(self, config_type: str) -> bool:
        valid_types = ['default', 'realtime', 'high_accuracy', 'youtube']
        if config_type not in valid_types:
            logger.warning(f"⚠️ Invalid config_type: {config_type}. Must be one of: {valid_types}")
            return False
        old_config_type = 'default'
        if isinstance(self.config, RealtimeConfig):
            old_config_type = 'realtime'
        elif isinstance(self.config, HighAccuracyConfig):
            old_config_type = 'high_accuracy'
        elif isinstance(self.config, YouTubeOptimizedConfig):
            old_config_type = 'youtube'
        if old_config_type == config_type:
            return True
        self.config = get_config(config_type)
        self.chunk_duration = self.config.CHUNK_DURATION
        logger.info(f"🔄 Config type changed: {old_config_type} → {config_type}")
        logger.info(f"   New CHUNK_DURATION: {self.chunk_duration}s")
        logger.info(f"   New CHUNK_SIZE: {self.config.CHUNK_SIZE_BYTES:,} bytes")
        return True

    def get_audio_filter(self, language: Optional[str] = None,
                         profile: Optional[str] = None) -> str:
        return self.config.get_audio_filter(language, profile)

    def get_youtube_headers(self, is_manifest: bool = False) -> Dict[str, str]:
        return self.config.get_youtube_headers(is_manifest)

    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        return self.config.get_platform_config(platform)

    def print_config_summary(self) -> None:
        logger.info("\n" + "="*60)
        logger.info("⚙️ ADVANCED SETTINGS CONFIGURATION")
        logger.info("="*60)
        logger.info("\n🤖 AI Model Parameters:")
        logger.info(f"  • Beam Size: {self.beam_size}")
        logger.info(f"  • Temperature: {self.temperature}")
        logger.info(f"  • VAD Filter: {self.vad_filter}")
        logger.info(f"  • VAD Threshold: {self.vad_threshold}")
        logger.info(f"  • VAD Min Speech (ms): {self.vad_min_speech_duration_ms}")
        logger.info(f"  • VAD Min Silence (ms): {self.vad_min_silence_duration_ms}")
        logger.info(f"  • GPU Acceleration: {self.gpu_acceleration}")
        logger.info("\n🎵 Audio Configuration (from ExcellenceConfig):")
        logger.info(f"  • Sample Rate: {self.config.SAMPLE_RATE} Hz")
        logger.info(f"  • Channels: {self.config.CHANNELS} "
                    f"({'Mono' if self.config.CHANNELS == 1 else 'Stereo'})")
        logger.info(f"  • Chunk Duration: {self.config.CHUNK_DURATION}s")
        logger.info(f"  • Chunk Size: {self.config.CHUNK_SIZE_BYTES:,} bytes")
        logger.info(f"  • Bytes/sec: {self.config.BYTES_PER_SECOND:,}")
        logger.info(f"  • Audio Filter Profiles: {len(self.config.FILTER_PROFILES)}")
        logger.info(f"  • Language Filters: {len(self.config.LANGUAGE_FILTERS)} languages")
        logger.info("\n⚡ Performance Settings:")
        logger.info(f"  • Max Cache Size: {self.max_cache_size}")
        logger.info(f"  • Max Memory: {self.max_memory_mb} MB")
        logger.info(f"  • Auto Save Interval: {self.auto_save_interval}s")
        logger.info("\n🔧 Features:")
        logger.info(f"  • Sentiment Analysis: {self.enable_sentiment_analysis}")
        logger.info(f"  • Speaker Diarization: {self.enable_speaker_diarization}")
        logger.info(f"  • Optimize Translations: {self.optimize_translations}")
        config_type = 'default'
        if isinstance(self.config, RealtimeConfig):
            config_type = 'realtime'
        elif isinstance(self.config, HighAccuracyConfig):
            config_type = 'high_accuracy'
        elif isinstance(self.config, YouTubeOptimizedConfig):
            config_type = 'youtube'
        logger.info(f"\n🎯 Config Type: {config_type.upper()}")
        issues = self.validate()
        if issues:
            logger.info("\n⚠️ Validation Issues:")
            for issue in issues:
                logger.info(f"  • {issue}")
        else:
            logger.info("\n✅ All settings valid")
        logger.info("="*60)

    def __repr__(self) -> str:
        config_type = 'default'
        if isinstance(self.config, RealtimeConfig):
            config_type = 'realtime'
        elif isinstance(self.config, HighAccuracyConfig):
            config_type = 'high_accuracy'
        elif isinstance(self.config, YouTubeOptimizedConfig):
            config_type = 'youtube'
        return (f"AdvancedSettings(type={config_type}, "
                f"beam_size={self.beam_size}, "
                f"chunk={self.config.CHUNK_DURATION}s/{self.config.CHUNK_SIZE_BYTES:,}B, "
                f"gpu={self.gpu_acceleration})")


# -----------------------------------------------------------------------------
# PLUGINS (vereinfacht, unverändert)
# -----------------------------------------------------------------------------
class Plugin:
    def __init__(self, name: str, version: str = "1.0.0") -> None:
        self.name = name
        self.version = version
        self.enabled = True

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        return result

    def on_translation(self, result: ExcellenceTranslationResult) -> ExcellenceTranslationResult:
        return result

    def on_startup(self) -> None:
        pass

    def on_shutdown(self) -> None:
        pass


class SentimentAnalysisPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__("Sentiment Analysis", "1.0.0")
        self.sentiment_cache: Dict[str, Any] = {}

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        return result


class KeywordExtractionPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__("Keyword Extraction", "1.0.0")
        self.common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by",
        }

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        return result


class PluginManager:
    def __init__(self) -> None:
        self.plugins: List[Plugin] = []
        self.enabled = True

    def register_plugin(self, plugin: Plugin) -> None:
        self.plugins.append(plugin)

    def load_builtin_plugins(self) -> None:
        self.register_plugin(SentimentAnalysisPlugin())
        self.register_plugin(KeywordExtractionPlugin())

    def process_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        if not self.enabled:
            return result
        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    result = plugin.on_transcription(result)
                except Exception:
                    pass
        return result

    def process_translation(self, result: ExcellenceTranslationResult) -> ExcellenceTranslationResult:
        if not self.enabled:
            return result
        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    result = plugin.on_translation(result)
                except Exception:
                    pass
        return result


# -----------------------------------------------------------------------------
# TRANSLATION ENGINE (überarbeitet mit Thread-Sicherheit und Fehlerbehandlung)
# -----------------------------------------------------------------------------
class ExcellenceTranslationEngine:
    def __init__(self, target_lang: str = "de",
                 advanced_settings: Optional[AdvancedSettings] = None) -> None:
        self.target_lang = target_lang
        self.translator: Optional[Any] = None
        self._cache = ExcellenceTTLCache(maxsize=Constants.TRANSLATION_CACHE_SIZE, ttl=Constants.TRANSLATION_CACHE_TTL)
        self._lock = threading.RLock()
        self.advanced_settings = advanced_settings or AdvancedSettings()
        self._last_translations: Deque[str] = deque(maxlen=15)
        self._setup_translator()
        self.last_detected_language = "auto"

    def _contains_asian(self, text: str) -> bool:
        asian_ranges = [
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs
            (0xAC00, 0xD7AF),   # Korean Hangul Syllables
            (0x3040, 0x309F),   # Japanese Hiragana
            (0x30A0, 0x30FF),   # Japanese Katakana
            (0x1100, 0x11FF),   # Korean Hangul Jamo
            (0x3130, 0x318F),   # Korean Hangul Compatibility Jamo
        ]
        for char in text:
            if not char:
                continue
            code = ord(char)
            for low, high in asian_ranges:
                if low <= code <= high:
                    return True
        return False

    def _is_valid_translation(self, original: str, translated: str) -> bool:
        if not translated or not translated.strip():
            return False
        orig_clean = original.strip()
        trans_clean = translated.strip()
        if len(trans_clean) < 1:
            return False
        if trans_clean.isspace():
            return False

        is_asian = self._contains_asian(orig_clean) or self._contains_asian(trans_clean)

        if is_asian:
            if len(trans_clean) <= 1:
                return True
            orig_len = len(orig_clean)
            trans_len = len(trans_clean)
            if orig_len == 0 or trans_len == 0:
                return False
            ratio = trans_len / max(orig_len, 1)
            return 0.05 <= ratio <= 15.0
        else:
            if len(trans_clean) <= 3:
                if len(set(trans_clean)) == 1 and len(trans_clean) > 1:
                    return False
            else:
                if len(set(trans_clean)) < 3:
                    return False
            orig_len = len(orig_clean)
            trans_len = len(trans_clean)
            if orig_len == 0 or trans_len == 0:
                return False
            ratio = trans_len / max(orig_len, 1)
            return 0.1 <= ratio <= 8.0

    def _setup_translator(self) -> None:
        try:
            if TRANSLATOR_AVAILABLE:
                GoogleTranslator = FastLazyLoader.load("deep_translator")
                self.translator = GoogleTranslator(source="auto", target=self.target_lang)
            else:
                self.translator = None
        except ImportError as e:
            logger.warning(f"deep_translator nicht verfügbar: {e}")
            self.translator = None

    def set_target_language(self, target_lang: str) -> None:
        if target_lang != self.target_lang:
            self.target_lang = target_lang
            with self._lock:
                self._cache.clear()
                self._last_translations.clear()
            self._setup_translator()

    def _clean_common_errors(self, text: str) -> str:
        if "bass communi" in text.lower():
            return "best community"
        return text

    @excellence_execution(timeout=8.0)
    def translate_text(self, text: str, source_lang: str = "auto") -> Optional[ExcellenceTranslationResult]:
        if source_lang != "auto" and source_lang == self.target_lang:
            return None
        if not text or not self.translator:
            return None
        try:
            original_text = text.strip()
            if len(original_text) < 2:
                return None
            clean_text = self._preprocess_text(original_text)
            if not clean_text:
                return None
            text_hash = hashlib.md5(f"{source_lang}_{self.target_lang}_{clean_text}".encode()).hexdigest()[:16]
            cache_key = f"trans_{text_hash}"
            with self._lock:
                if text_hash in self._last_translations:
                    return None
                cached_result = self._cache.get(cache_key)
                if cached_result is not None:
                    return cached_result

            for attempt in range(3):
                try:
                    if not self.translator:
                        self._setup_translator()
                        if not self.translator:
                            time.sleep(1)
                            continue
                    translated_text = self.translator.translate(clean_text)
                    if not translated_text or not translated_text.strip():
                        time.sleep(1)
                        continue
                    if not self._is_valid_translation(clean_text, translated_text):
                        continue
                    final_translation = self._postprocess_translation(translated_text, clean_text)
                    result = ExcellenceTranslationResult(
                        original=original_text,
                        translated=final_translation,
                        source_lang=source_lang,
                        target_lang=self.target_lang,
                    )
                    with self._lock:
                        self._cache.put(cache_key, result)
                        self._last_translations.append(text_hash)
                    return result
                except Exception:
                    if attempt < 2:
                        time.sleep(1.5)
                        self._setup_translator()
                    else:
                        if DEBUG_LEVEL >= 2:
                            logger.exception("Fehler bei Übersetzung (letzter Versuch):")
                        logger.debug("❌ translate_text: Letzter Versuch fehlgeschlagen, gebe auf.")
            return None
        except Exception:
            return None

    def dispose(self) -> None:
        with self._lock:
            self._cache.clear()
            self._last_translations.clear()
            self.translator = None
            gc.collect()

    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        clean_text = text.strip()
        clean_text = re.sub(r"\s+", " ", clean_text)
        clean_text = re.sub(r"[ ]+([.,!?])", r"\1", clean_text)
        clean_text = re.sub(r"([.,!?])[ ]*", r"\1 ", clean_text)
        common_errors = {
            "bass communi": "best community",
            " ,": ",",
            " .": ".",
            "„": '"',
            "“": '"',
        }
        for error, correction in common_errors.items():
            clean_text = clean_text.replace(error, correction)
        if len(clean_text.split()) < 1:
            return ""
        return clean_text.strip()

    def _postprocess_translation(self, translated: str, original: str) -> str:
        if not translated:
            return ""
        result = translated.strip()
        if not result.endswith((".", "!", "?", ":", ";")):
            result += "."
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        result = re.sub(r"\s+", " ", result)
        rules = [
            (r"\s+\.", "."),
            (r"\s+,", ","),
            (r"\s+\?", "?"),
            (r"\s+!", "!"),
            (r" ,", ","),
            (r" \.", "."),
        ]
        for pattern, replacement in rules:
            result = re.sub(pattern, replacement, result)
        return result.strip()


# -----------------------------------------------------------------------------
# DUMMY ENGINES (unverändert)
# -----------------------------------------------------------------------------
class DummyTranscriptionEngine:
    def __init__(self, advanced_settings: Optional[AdvancedSettings] = None):
        self.advanced_settings = advanced_settings or AdvancedSettings()
        self.model = None
        self.model_size = "dummy"
        self.whisper_backend = None
        self.demo_mode = True

    def load_model(self, model_size: str, set_active: bool = False) -> Optional[Tuple[Any, str]]:
        logger.info("Dummy-Modus: Laden eines Modells nicht erforderlich.")
        return (None, "dummy")

    def transcribe_audio(self, audio_data: bytes, include_timestamps: bool = False) -> Any:
        if include_timestamps:
            dummy = ExcellenceTranscriptionResult(
                text="[Whisper nicht verfügbar]",
                confidence=0.5,
                language="de",
                start=0.0,
                end=5.0
            )
            return [dummy]
        else:
            return ExcellenceTranscriptionResult(
                text="[Whisper nicht verfügbar]",
                confidence=0.5,
                language="de"
            )

    def safe_transcribe(self, audio_data: bytes, max_retries: int = 2) -> Optional[ExcellenceTranscriptionResult]:
        return self.transcribe_audio(audio_data)

    def get_current_model(self) -> str:
        return "dummy"

    def is_model_loading(self) -> bool:
        return False

    def reload_model(self, model_size: str) -> bool:
        return False

    def test_model_functionality(self) -> bool:
        return True

    def dispose(self) -> None:
        pass

class DummyTranslationEngine:
    def __init__(self, target_lang: str = "de",
                 advanced_settings: Optional[AdvancedSettings] = None):
        self.target_lang = target_lang
        self.advanced_settings = advanced_settings or AdvancedSettings()

    def set_target_language(self, target_lang: str) -> None:
        self.target_lang = target_lang

    def translate_text(self, text: str, source_lang: str = "auto") -> Optional[ExcellenceTranslationResult]:
        if not text:
            return None
        return ExcellenceTranslationResult(
            original=text,
            translated=f"[Übersetzer nicht verfügbar] {text}",
            source_lang=source_lang,
            target_lang=self.target_lang
        )

    def dispose(self) -> None:
        pass

# -----------------------------------------------------------------------------
# HILFSKLASSEN FÜR TRANSCRIPTION (unverändert)
# -----------------------------------------------------------------------------
class _EmptyInfo:
    language = "unknown"
    duration = 0.0


class _UniversalInfo:
    def __init__(self, result_dict: Dict[str, Any]) -> None:
        self.language = result_dict.get("language", "unknown")
        self.duration = result_dict.get("duration", 0.0)


class _UniversalSegment:
    def __init__(self, seg_dict: Dict[str, Any]) -> None:
        self.text = seg_dict.get("text", "").strip()
        self.start = seg_dict.get("start", 0.0)
        self.end = seg_dict.get("end", 0.0)
        self.confidence = seg_dict.get("confidence", 0.9)


class _EmergencySegment:
    def __init__(self, seg_dict: Dict[str, Any]) -> None:
        self.text = seg_dict.get("text", "")
        self.start = seg_dict.get("start", 0.0)
        self.end = seg_dict.get("end", 0.0)
        self.confidence = 0.5


# -----------------------------------------------------------------------------
# TRANSCRIPTION ENGINE (überarbeitet mit Thread-Sicherheit und Fehlerbehandlung)
# -----------------------------------------------------------------------------
class ExcellenceTranscriptionEngine:
    def __init__(self, advanced_settings: Optional[AdvancedSettings] = None) -> None:
        self.model: Optional[Any] = None
        self.model_size: Optional[str] = None
        self.whisper_backend: Optional[str] = None
        self._lock = threading.RLock()
        self._model_loading = False
        self._max_cached_models = 3
        self.advanced_settings = advanced_settings or AdvancedSettings()
        self._cache = ExcellenceTTLCache(maxsize=self.advanced_settings.max_cache_size)
        self.device, self.compute_type = self._detect_optimal_device()
        self._performance_monitor = SimplePerformanceTracker()
        self._last_transcription_text = ""
        self._active_model_loads: Set[str] = set()
        self._model_loaded_flag = False
        self._disposing = False
        self._model_cache: Dict[Tuple[str, str], Any] = {}

    def _detect_optimal_device(self) -> Tuple[str, str]:
        device = "cpu"
        compute_type = "int8"
        if TORCH_AVAILABLE:
            torch = FastLazyLoader.load('torch')
            if torch.cuda.is_available():
                try:
                    torch.tensor([1.0]).cuda()
                    device = "cuda"
                    compute_type = "float16" if self.advanced_settings.gpu_acceleration else "int8"
                    logger.info(f"✅ NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    if DEBUG_LEVEL >= 1:
                        logger.warning(f"⚠️ CUDA test failed, falling back: {e}")
            if hasattr(torch.version, 'hip') and torch.version.hip:
                try:
                    if torch.cuda.device_count() > 0:
                        device = "cuda"
                        compute_type = "float16" if self.advanced_settings.gpu_acceleration else "int8"
                        logger.info("✅ AMD GPU (ROCm) detected")
                except (AttributeError, RuntimeError) as e:
                    if DEBUG_LEVEL >= 1:
                        logger.warning(f"⚠️ ROCm test failed: {e}")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                compute_type = "float16"
                logger.info("✅ Apple Silicon GPU (MPS) detected")
        logger.info(f"✅ Verwende Device: {device} (compute_type={compute_type})")
        return device, compute_type

    @excellence_execution(timeout=180.0)
    def load_model(self, model_size: str, set_active: bool = False) -> Optional[Tuple[Any, str]]:
        if set_active:
            self._force_model_cleanup()

        if FASTER_WHISPER_AVAILABLE:
            backend = "faster_whisper"
        elif OPENAI_WHISPER_AVAILABLE:
            backend = "openai_whisper"
        else:
            logger.error("❌ Kein Whisper-Backend verfügbar")
            return None

        cache_key = (model_size, backend)

        with self._lock:
            if cache_key in self._model_cache:
                model = self._model_cache[cache_key]
                if set_active:
                    self.model = model
                    self.model_size = model_size
                    self.whisper_backend = backend
                return model, backend

        load_lock_key = f"model_load_{model_size}_{backend}"
        if load_lock_key in self._active_model_loads:
            logger.info("⏳ Model wird bereits geladen...")
            return None
        self._active_model_loads.add(load_lock_key)

        try:
            self._force_model_cleanup()
            config_dir = PlatformUtils.get_platform_config_dir()
            model_dir = config_dir / "models"
            model_dir.mkdir(exist_ok=True)
            logger.info(f"📁 Model-Verzeichnis: {model_dir}")

            if backend == "faster_whisper":
                try:
                    logger.info("  → Versuche faster-whisper...")
                    from faster_whisper import WhisperModel
                    model = WhisperModel(
                        model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=str(model_dir),
                        cpu_threads=4,
                        num_workers=1,
                    )
                    np = FastLazyLoader.load("numpy")
                    test_audio = np.zeros(1600, dtype=np.float32)
                    segments, info = model.transcribe(
                        test_audio, beam_size=1, best_of=1,
                        vad_filter=False, without_timestamps=True
                    )
                    list(segments)
                    logger.info(f"✅ faster-whisper '{model_size}' erfolgreich geladen und getestet")
                except (ImportError, OSError, RuntimeError) as e:
                    logger.warning(f"⚠️ faster-whisper konnte nicht geladen werden: {e}")
                    if DEBUG_LEVEL >= 2:
                        logger.exception("Stacktrace:")
                    if OPENAI_WHISPER_AVAILABLE:
                        backend = "openai_whisper"
                        cache_key = (model_size, backend)
                        if DEBUG_LEVEL >= 2:
                            logger.debug("  → Wechsle zu openai-whisper...")
                        logger.info("  → Wechsle zu openai-whisper...")
                    else:
                        return None

            if backend == "openai_whisper":
                try:
                    logger.info("  → Versuche openai-whisper...")
                    import whisper
                    model = whisper.load_model(
                        model_size,
                        device="cuda" if self.device == "cuda" else "cpu",
                        download_root=str(model_dir) if model_dir else None,
                    )
                    logger.info(f"✅ openai-whisper '{model_size}' erfolgreich geladen")
                except (ImportError, OSError, RuntimeError) as e:
                    logger.error(f"❌ openai-whisper fehlgeschlagen: {e}")
                    if DEBUG_LEVEL >= 2:
                        logger.exception("Stacktrace:")
                    return None

            with self._lock:
                self._model_cache[cache_key] = model
                if len(self._model_cache) > self._max_cached_models:
                    oldest_key, old_model = next(iter(self._model_cache.items()))
                    del self._model_cache[oldest_key]
                    logger.info(f"🧹 Entferne altes Modell '{oldest_key[0]}' ({oldest_key[1]}) aus Cache")
                    if hasattr(old_model, 'unload_model'):
                        try:
                            old_model.unload_model()
                        except Exception:
                            pass
                if set_active:
                    self.model = model
                    self.model_size = model_size
                    self.whisper_backend = backend
            return model, backend

        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler: {e}")
            return None
        finally:
            if load_lock_key in self._active_model_loads:
                self._active_model_loads.remove(load_lock_key)

    def reload_model(self, model_size: str) -> bool:
        with self._lock:
            if self._model_loading:
                logger.warning("⚠️ Model loading already in progress")
                return False
            self._model_loading = True

        def _load_in_background():
            try:
                result = self.load_model(model_size, set_active=False)
                if result is not None:
                    new_model, new_backend = result
                    with self._lock:
                        self.model = new_model
                        self.model_size = model_size
                        self.whisper_backend = new_backend
                    logger.info(f"✅ Model switched to {model_size} ({new_backend})")
                else:
                    logger.error("❌ Background model loading failed")
            except Exception as e:
                logger.error(f"❌ Background model loading error: {e}")
            finally:
                with self._lock:
                    self._model_loading = False

        thread = threading.Thread(target=_load_in_background, daemon=True, name=f"ModelLoader-{model_size}")
        thread.start()
        return True

    def is_model_loading(self) -> bool:
        return self._model_loading

    def _force_model_cleanup(self) -> None:
        import gc
        import time
        with self._lock:
            if self.model is not None:
                try:
                    if hasattr(self.model, "unload_model"):
                        self.model.unload_model()
                    self.model = None
                    self.model_size = None
                    self._model_loaded_flag = False
                except Exception as e:
                    logger.warning(f"⚠️ Model cleanup error (step 1): {e}")
                time.sleep(0.1)
            gc.collect()  # Einmal reicht
            if self.device == "cuda" and TORCH_AVAILABLE:
                try:
                    torch = FastLazyLoader.load("torch")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        logger.info("🧹 GPU Memory freigegeben")
                except Exception as e:
                    logger.warning(f"⚠️ GPU cleanup error: {e}")
            logger.info("✅ Model-Cleanup abgeschlossen (aktives Modell entladen)")

    def _universal_transcribe(self, audio_np: Any, **kwargs: Any) -> Tuple[List[Any], Any]:
        if not self.model:
            raise ValueError("Kein Modell geladen")
        backend = getattr(self, "whisper_backend", "openai_whisper")
        if backend == "faster_whisper":
            try:
                if DEBUG_LEVEL >= 2 and kwargs.get('vad_filter', False):
                    vad_params = kwargs.get('vad_parameters', {})
                    logger.debug(f"VAD params: threshold={vad_params.get('threshold')}, "
                                 f"min_speech_ms={vad_params.get('min_speech_duration_ms')}, "
                                 f"min_silence_ms={vad_params.get('min_silence_duration_ms')}")

                segments, info = self.model.transcribe(audio_np, **kwargs)
                segments_list = list(segments)

                if DEBUG_LEVEL >= 2 and kwargs.get('vad_filter', False):
                    total_speech_ms = 0.0
                    seg_details = []
                    for seg in segments_list:
                        duration_ms = (seg.end - seg.start) * 1000
                        total_speech_ms += duration_ms
                        seg_details.append(f"{duration_ms:.0f}ms")
                    logger.debug(f"VAD result: {len(segments_list)} segments, total speech={total_speech_ms:.0f}ms, "
                                 f"segments: {', '.join(seg_details)}")
                elif DEBUG_LEVEL >= 1 and kwargs.get('vad_filter', False):
                    logger.debug(f"VAD: {len(segments_list)} speech segments detected (threshold={kwargs.get('vad_parameters', {}).get('threshold', '?')})")

                return segments_list, info
            except (TypeError, ValueError) as e:
                logger.warning(f"⚠️ faster-whisper Parameterfehler: {e} – verwende minimale Parameter")
                minimal_kwargs = {}
                for key in ["language", "task", "temperature", "beam_size", "best_of"]:
                    if key in kwargs:
                        minimal_kwargs[key] = kwargs[key]
                try:
                    segments, info = self.model.transcribe(audio_np, **minimal_kwargs)
                    segments_list = list(segments)
                    return segments_list, info
                except Exception as e2:
                    logger.error(f"❌ faster-whisper auch mit minimalen Parametern fehlgeschlagen: {e2}")
                    if DEBUG_LEVEL >= 2:
                        logger.exception("Stacktrace:")
                    return [], _EmptyInfo()
            except RuntimeError as e:
                logger.error(f"❌ faster-whisper Runtime-Error: {e}")
                if DEBUG_LEVEL >= 2:
                    logger.exception("Stacktrace:")
                return [], _EmptyInfo()
            except Exception as e:
                logger.error(f"❌ faster-whisper transcribe error: {e}")
                if DEBUG_LEVEL >= 2:
                    logger.exception("Stacktrace:")
                return [], _EmptyInfo()
        else:
            OPENAI_ALLOWED_PARAMS = {
                "language", "task", "temperature", "best_of", "beam_size", "patience",
                "length_penalty", "repetition_penalty", "no_repeat_ngram_size",
                "initial_prompt", "prefix", "suppress_tokens", "suppress_tokens_whitespace",
                "without_timestamps", "max_initial_timestamp", "word_timestamps",
                "prepend_punctuations", "append_punctuations", "max_new_tokens",
                "clip_timestamps", "hallucination_silence_threshold",
            }
            filtered_kwargs = {}
            removed_params: List[str] = []
            for key, value in kwargs.items():
                if key in OPENAI_ALLOWED_PARAMS:
                    filtered_kwargs[key] = value
                else:
                    removed_params.append(key)
            if removed_params and DEBUG_LEVEL >= 2:
                logger.debug(f"🔍 Removed {len(removed_params)} unsupported params: {removed_params[:5]}")
            defaults = {
                "language": None,
                "task": "transcribe",
                "temperature": 0.0,
            }
            for key, default_value in defaults.items():
                if key not in filtered_kwargs:
                    filtered_kwargs[key] = default_value
            try:
                result = self.model.transcribe(audio_np, **filtered_kwargs)
                segments = result.get("segments", [])
                detected_lang = result.get("language", "unknown")
                logger.debug(f"✅ openai-whisper: {len(segments)} segments, language: {detected_lang}")
                converted_segments: List[Any] = []
                for seg in segments:
                    if seg.get("text", "").strip():
                        converted_segments.append(_UniversalSegment(seg))
                return converted_segments, _UniversalInfo(result)
            except Exception as e:
                logger.error(f"❌ openai-whisper error: {e}")
                if DEBUG_LEVEL >= 2:
                    logger.exception("Stacktrace:")
                try:
                    logger.info("🔄 Emergency fallback with minimal parameters...")
                    minimal_result = self.model.transcribe(audio_np, language=None, task="transcribe", temperature=0.0)
                    emergency_segments = []
                    for seg in minimal_result.get("segments", []):
                        emergency_segments.append(_EmergencySegment(seg))
                    logger.warning(f"⚠️ Emergency fallback: {len(emergency_segments)} segments")
                    return emergency_segments, _UniversalInfo(minimal_result)
                except Exception as fallback_error:
                    logger.error(f"💥 Even emergency fallback failed: {fallback_error}")
                    if DEBUG_LEVEL >= 2:
                        logger.exception("Fallback stacktrace:")
                    return [], _EmptyInfo()

    def validate_audio_data(self, audio_data: bytes) -> Tuple[bool, str]:
        if not isinstance(audio_data, bytes):
            return False, "Audio data must be bytes"
        if len(audio_data) == 0:
            return False, "Audio data is empty"
        if len(audio_data) < 1600:
            return False, f"Audio data too short: {len(audio_data)} bytes"
        try:
            np = FastLazyLoader.load("numpy")
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if np.all(audio_np == 0):
                return False, "Audio data is completely silent"
            if np.var(audio_np) < 100:
                return False, "Audio variance too low (likely silent)"
        except Exception:
            pass
        return True, "Valid"

    def safe_transcribe(self, audio_data: bytes, max_retries: int = 2) -> Optional[ExcellenceTranscriptionResult]:
        is_valid, validation_msg = self.validate_audio_data(audio_data)
        if not is_valid:
            return None
        for attempt in range(max_retries + 1):
            try:
                try:
                    processed_audio = self.enhance_audio_for_transcription(audio_data)
                except (ValueError, ImportError, RuntimeError) as e:
                    logger.warning(f"Audio-Enhancement fehlgeschlagen (Versuch {attempt+1}): {e}")
                    processed_audio = audio_data
                result = self.transcribe_audio(processed_audio)
                if result and result.text and result.text.strip():
                    return result
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Transkriptionsfehler (Versuch {attempt+1}): {e}")
                if attempt < max_retries:
                    wait_time = 0.5 * (attempt + 1)
                    time.sleep(wait_time)
            except Exception:
                if attempt < max_retries:
                    wait_time = 0.5 * (attempt + 1)
                    time.sleep(wait_time)
        return None

    def enhance_audio_for_transcription(self, audio_data: bytes) -> bytes:
        if not audio_data or len(audio_data) == 0:
            return audio_data
        if not NUMPY_AVAILABLE or len(audio_data) < 1600:
            return audio_data
        try:
            np = FastLazyLoader.load("numpy")
            try:
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                return audio_data
            if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                return audio_data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_np = np.where(np.abs(audio_np) < 1e-10, 0, audio_np)
                rms = np.sqrt(np.mean(audio_np**2))
            if rms < 1e-8:
                gain = 2.0
                audio_np = audio_np * gain
            elif rms < 0.005:
                gain = min(5.0, 0.02 / max(rms, 1e-6))
                audio_np = audio_np * gain
            elif rms > 0.5:
                gain = 0.5 / rms
                audio_np = audio_np * gain
            elif rms > 0.3:
                gain = 0.3 / rms
                audio_np = audio_np * gain
            max_val = np.max(np.abs(audio_np))
            if max_val > 1.0:
                audio_np = audio_np / max_val * 0.99
            elif max_val > 0.9:
                audio_np = audio_np * 0.95
            audio_np = audio_np - np.mean(audio_np)
            if len(audio_np) > 100:
                try:
                    if SCIPY_AVAILABLE:
                        scipy_signal = FastLazyLoader.load("scipy.signal")
                        b, a = scipy_signal.butter(2, 80 / (self.advanced_settings.config.SAMPLE_RATE / 2),
                                                    btype="high")
                        audio_np = scipy_signal.filtfilt(b, a, audio_np)
                except ImportError:
                    audio_np = audio_np - np.mean(audio_np)
                except Exception:
                    pass
            audio_np = np.clip(audio_np, -0.99, 0.99)
            enhanced_audio = (audio_np * 32767).astype(np.int16).tobytes()
            if len(enhanced_audio) != len(audio_data):
                return audio_data
            return enhanced_audio
        except (ImportError, ValueError, RuntimeError) as e:
            logger.warning(f"Audio-Enhancement fehlgeschlagen: {e}")
            if DEBUG_LEVEL >= 2:
                logger.exception("Stacktrace:")
            return audio_data

    def _validate_transcription_segment(self, text: str, confidence: float, segment: Any) -> bool:
        if not text or len(text.strip()) < 2:
            return False
        clean_text = text.strip()
        if clean_text.isspace():
            return False
        if len(clean_text) > 500:
            return False
        if not any(char.isalpha() for char in clean_text):
            return False
        return True

    def _calculate_enhanced_confidence(self, segment: Any, text: str) -> float:
        base_confidence = max(getattr(segment, "confidence", 0.0), 0.05)
        word_count = len(text.split())
        text_length = len(text.strip())
        has_punctuation = any(c in text for c in ".!?,;:")
        has_letters = any(c.isalpha() for c in text)
        unique_words = len(set(text.split()))
        length_boost = min(0.2, text_length / 300.0)
        word_boost = min(0.15, word_count * 0.03)
        punctuation_boost = 0.08 if has_punctuation else 0.0
        letters_boost = 0.1 if has_letters else 0.0
        diversity_boost = min(0.1, unique_words * 0.02)
        calculated_confidence = (
            base_confidence
            + length_boost
            + word_boost
            + punctuation_boost
            + letters_boost
            + diversity_boost
        )
        return min(0.95, calculated_confidence)

    @excellence_execution(timeout=60.0)
    def transcribe_audio(self, audio_data: bytes, include_timestamps: bool = False) -> Any:
        if not self.model or not audio_data:
            return None if not include_timestamps else []
        try:
            processed_audio = self.enhance_audio_for_transcription(audio_data)
            np = FastLazyLoader.load("numpy")
            audio_np = np.frombuffer(processed_audio, dtype=np.int16).astype(np.float32) / 32768.0
            if NUMPY_AVAILABLE:
                rms = np.sqrt(np.mean(audio_np**2))
                if rms < 0.005:
                    beam_size = 8
                elif rms < 0.02:
                    beam_size = 6
                else:
                    beam_size = 5
            else:
                beam_size = 5

            vad_params = None
            if self.advanced_settings.vad_filter:
                vad_params = {
                    "threshold": self.advanced_settings.vad_threshold,
                    "min_speech_duration_ms": self.advanced_settings.vad_min_speech_duration_ms,
                    "min_silence_duration_ms": self.advanced_settings.vad_min_silence_duration_ms,
                }

            segments, info = self._universal_transcribe(
                audio_np,
                language=None,
                task="transcribe",
                temperature=self.advanced_settings.temperature,
                best_of=5,
                beam_size=beam_size,
                patience=1.0,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.2,
                compression_ratio_threshold=2.8,
                condition_on_previous_text=True,
                suppress_tokens=[-1],
                without_timestamps=not include_timestamps,
                word_timestamps=include_timestamps,
                vad_filter=self.advanced_settings.vad_filter,
                vad_parameters=vad_params,
            )
            if not segments:
                logger.debug("🔄 No segments with thresholds, trying without...")
                segments, info = self._universal_transcribe(
                    audio_np,
                    language=None,
                    task="transcribe",
                    temperature=0.0,
                    best_of=5,
                    beam_size=beam_size,
                )
            valid_segments: List[Any] = []
            total_confidence = 0.0
            for segment in segments:
                text = segment.text.strip()
                confidence = getattr(segment, "confidence", 0.0)
                is_valid = self._validate_transcription_segment(text, confidence, segment)
                if is_valid:
                    enhanced_confidence = self._calculate_enhanced_confidence(segment, text)
                    segment.confidence = enhanced_confidence
                    valid_segments.append(segment)
                    total_confidence += enhanced_confidence
            if include_timestamps:
                results = [
                    ExcellenceTranscriptionResult(
                        text=seg.text.strip(),
                        confidence=getattr(seg, "confidence", 0.1),
                        language=getattr(info, "language", "unknown"),
                        start=getattr(seg, "start", 0.0),
                        end=getattr(seg, "end", 0.0),
                    )
                    for seg in valid_segments
                ]
                return results if results else []
            else:
                if valid_segments:
                    final_text = " ".join(seg.text.strip() for seg in valid_segments)
                    avg_confidence = total_confidence / len(valid_segments)
                    return ExcellenceTranscriptionResult(
                        text=final_text,
                        confidence=avg_confidence,
                        language=getattr(info, "language", "unknown"),
                    )
                else:
                    return self.emergency_fallback_transcription(audio_data)
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error(f"❌ transcribe_audio exception: {e}")
            if DEBUG_LEVEL >= 2:
                logger.exception("Stacktrace:")
            return None if not include_timestamps else []
        except Exception as e:
            logger.error(f"❌ transcribe_audio exception: {e}")
            if DEBUG_LEVEL >= 2:
                logger.exception("Stacktrace:")
            return None if not include_timestamps else []

    def emergency_fallback_transcription(self, audio_data: bytes) -> Optional[ExcellenceTranscriptionResult]:
        try:
            if not self.model or not audio_data:
                return None
            np = FastLazyLoader.load("numpy")
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = self._universal_transcribe(
                audio_np,
                language=None,
                task="transcribe",
                temperature=0.0,
                best_of=5,
                beam_size=5,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            segments_list = list(segments)
            best_segment = None
            best_confidence = 0.0
            best_text = ""
            for segment in segments_list:
                text = segment.text.strip()
                if text and len(text) > 1:
                    confidence = self._calculate_enhanced_confidence(segment, text)
                    is_valid = (
                        len(text) >= 2
                        and not text.isspace()
                        and any(c.isalnum() for c in text)
                        and confidence >= 0.1
                    )
                    if is_valid and confidence > best_confidence:
                        best_confidence = confidence
                        best_segment = segment
                        best_text = text
            if best_segment and best_text:
                return ExcellenceTranscriptionResult(
                    text=best_text,
                    confidence=best_confidence,
                    language=getattr(info, "language", "unknown"),
                )
            return None
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error(f"❌ Emergency fallback exception: {e}")
            if DEBUG_LEVEL >= 2:
                logger.exception("Stacktrace:")
        except Exception:
            if DEBUG_LEVEL >= 2:
                logger.exception("Stacktrace:")
            return None

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._last_transcription_text = ""
            gc.collect()
            if self.device == "cuda" and TORCH_AVAILABLE:
                try:
                    torch = FastLazyLoader.load("torch")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def get_current_model(self) -> str:
        return self.model_size if self.model_size else "None"

    # is_model_loading ist bereits oben definiert (Zeile ~3408) – hier keine zweite Definition!

    def test_model_functionality(self) -> bool:
        if not self.model:
            if DEBUG_LEVEL >= 2:
                logger.error("❌ Kein Model geladen zum Testen")
            return False
        try:
            logger.info("🔍 Teste Model-Funktionalität...")
            np = FastLazyLoader.load("numpy")
            test_audio = np.random.randn(16000).astype(np.float32) * 0.1
            segments, info = self._universal_transcribe(
                test_audio,
                language=None,
                task="transcribe",
                temperature=0.0,
                best_of=1,
                beam_size=1,
                without_timestamps=True,
            )
            segments_list = list(segments)
            logger.info(f"   Segmente gefunden: {len(segments_list)}")
            if segments_list:
                for i, seg in enumerate(segments_list[:2]):
                    logger.info(f"   Segment {i}: '{seg.text[:50]}...' (conf: {seg.confidence:.2f})")
            if hasattr(info, "language"):
                logger.info(f"   Sprache erkannt: {info.language}")
            logger.info("✅ Model-Test erfolgreich")
            return True
        except Exception as e:
            logger.error(f"❌ Model-Test fehlgeschlagen: {e}")
            return False

    def dispose(self) -> None:
        logger.info("🧹 Transcription Engine Dispose...")
        self._disposing = True
        with self._lock:
            self._cache.clear()
            self._last_transcription_text = ""
            for (size, backend), model in list(self._model_cache.items()):
                if hasattr(model, 'unload_model'):
                    try:
                        model.unload_model()
                    except (RuntimeError, AttributeError):
                        pass
            self._model_cache.clear()
        self._force_model_cleanup()
        gc.collect()
        logger.info("✅ Transcription Engine disposed")

# -----------------------------------------------------------------------------
# STREAM MANAGER (überarbeitet mit Sicherheitsvalidierung der URLs)
# -----------------------------------------------------------------------------
class StreamManager:
    def __init__(self, enable_debug: bool = False, use_browser_cookies: bool = True) -> None:
        self._platform_cache: OrderedDict[str, Tuple[str, str]] = OrderedDict()
        self._audio_url_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._live_status_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._stream_info_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._debug = enable_debug
        self._last_error: Optional[str] = None
        self._last_method: Optional[str] = None
        self._stats = {
            'extraction_attempts': 0,
            'successful_extractions': 0,
            'cache_hits': 0,
            'errors': 0,
            'start_time': time.time()
        }
        self._stats_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._format_priorities = {
            'youtube': ['bestaudio[ext=m4a]/bestaudio/best', 'bestaudio/best', 'ba'],
            'youtube_live': ['bestaudio/best', 'ba'],
            'twitch': ['bestaudio/best', 'audio_only'],
            'tiktok': ['bestaudio/best'],
            'facebook': ['bestaudio/best'],
            'hls': ['bestaudio/best'],
            'dash': ['bestaudio/best'],
            'generic': ['bestaudio/best', 'ba'],
            'kick': ['bestaudio/best', 'ba'],
            'rumble': ['bestaudio/best', 'ba'],
            'dailymotion': ['bestaudio/best', 'ba'],
            'vimeo': ['bestaudio/best', 'ba'],
            'twitter': ['bestaudio/best', 'ba'],
        }
        self._user_agents = {
            'desktop': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'mobile': 'Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        }
        self._browsers = [
            ('firefox', 'Firefox'),
            ('chrome', 'Chrome'),
            ('brave', 'Brave'),
            ('edge', 'Edge'),
            ('chromium', 'Chromium'),
            ('opera', 'Opera'),
            ('vivaldi', 'Vivaldi'),
        ]
        self.use_browser_cookies = use_browser_cookies  # NEW

    def _extract_youtube_video_id(self, url: str) -> Optional[str]:
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def detect_platform(self, url: str) -> Tuple[str, str]:
        if not url:
            return ('unknown', 'Invalid URL')
        # URL bereinigen (nicht erlaubte Zeichen entfernen)
        url = PlatformUtils.sanitize_url(url)
        with self._cache_lock:
            if url in self._platform_cache:
                with self._stats_lock:
                    self._stats['cache_hits'] += 1
                if self._debug or DEBUG_LEVEL >= 2:
                    logger.debug(f"🔍 detect_platform: Cache-Treffer für {url[:50]}...")
                return self._platform_cache[url]
        url_lower = url.lower().strip()
        detection_reason = []
        if url_lower.startswith('file://'):
            # Prüfe, ob der Pfad erlaubt ist
            ok, _ = PlatformUtils.validate_file_path(url)
            if not ok:
                return ('invalid', 'Invalid file path')
            result = ('local', 'Local File')
            detection_reason = ["startswith file://"]
        elif any(url_lower.endswith(ext) for ext in
                 ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.opus', '.webm']):
            result = ('direct_audio', 'Direct Audio')
            detection_reason = ["audio extension"]
        elif any(url_lower.endswith(ext) for ext in
                 ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.wmv', '.flv']):
            result = ('direct_video', 'Direct Video')
            detection_reason = ["video extension"]
        elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            is_live = self._check_youtube_live_status(url)
            if is_live:
                result = ('youtube_live', 'YouTube Live')
                detection_reason = ["youtube domain + live pattern"]
            else:
                result = ('youtube', 'YouTube Video')
                detection_reason = ["youtube domain"]
        elif 'twitch.tv' in url_lower:
            result = ('twitch', 'Twitch')
            detection_reason = ["twitch domain"]
        elif 'kick.com' in url_lower:
            result = ('kick', 'Kick')
            detection_reason = ["kick domain"]
        elif 'rumble.com' in url_lower:
            result = ('rumble', 'Rumble')
            detection_reason = ["rumble domain"]
        elif 'dailymotion.com' in url_lower:
            result = ('dailymotion', 'Dailymotion')
            detection_reason = ["dailymotion domain"]
        elif 'vimeo.com' in url_lower:
            result = ('vimeo', 'Vimeo')
            detection_reason = ["vimeo domain"]
        elif 'twitter.com' in url_lower or 'x.com' in url_lower:
            result = ('twitter', 'Twitter/X')
            detection_reason = ["twitter/x domain"]
        elif 'tiktok.com' in url_lower:
            result = ('tiktok', 'TikTok')
            detection_reason = ["tiktok domain"]
        elif 'facebook.com' in url_lower or 'fb.watch' in url_lower:
            result = ('facebook', 'Facebook')
            detection_reason = ["facebook domain"]
        elif '.m3u8' in url_lower:
            result = ('hls', 'HLS Stream')
            detection_reason = [".m3u8 in URL"]
        elif '.mpd' in url_lower:
            result = ('dash', 'DASH Stream')
            detection_reason = [".mpd in URL"]
        elif url_lower.startswith(('http://', 'https://')):
            result = ('generic', 'Website/Stream')
            detection_reason = ["http(s) fallback"]
        else:
            result = ('unknown', 'Unknown Source')
            detection_reason = ["no pattern matched"]
        with self._cache_lock:
            if len(self._platform_cache) < 50:
                self._platform_cache[url] = result
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug(f"🔍 detect_platform: {url[:50]}... -> {result}, reason: {', '.join(detection_reason)}")
        return result

    def _check_youtube_live_status(self, url: str) -> bool:
        cache_key = f"live_{hashlib.md5(url.encode()).hexdigest()[:16]}"
        current_time = time.time()
        with self._cache_lock:
            if cache_key in self._live_status_cache:
                cached = self._live_status_cache[cache_key]
                if current_time - cached['timestamp'] < 300:
                    return cached['is_live']
        is_live = False
        url_lower = url.lower()
        live_patterns = ['/live', 'live=1', '/stream', 'livestream']
        if any(pattern in url_lower for pattern in live_patterns):
            is_live = True
        with self._cache_lock:
            if len(self._live_status_cache) > 30:
                oldest_keys = sorted(self._live_status_cache.keys(),
                                      key=lambda k: self._live_status_cache[k]['timestamp'])[:10]
                for k in oldest_keys:
                    del self._live_status_cache[k]
            self._live_status_cache[cache_key] = {
                'is_live': is_live,
                'timestamp': current_time
            }
        return is_live

    def extract_audio_url(self, url: str, force_refresh: bool = False) -> Optional[str]:
        with self._stats_lock:
            self._stats['extraction_attempts'] += 1
        if self._debug or DEBUG_LEVEL >= 1:
            logger.debug(f"\n🎵 [DEBUG] EXTRACT_AUDIO_URL START für: {url[:80]}...")
        self._last_error = None
        self._last_method = None
        if not url or not isinstance(url, str):
            self._last_error = "Invalid input"
            with self._stats_lock:
                self._stats['errors'] += 1
            return None
        cleaned_url = PlatformUtils.sanitize_url(url.strip())
        if not cleaned_url:
            self._last_error = "Empty URL"
            with self._stats_lock:
                self._stats['errors'] += 1
            return None
        cache_key = f"audio_{hashlib.md5(cleaned_url.encode()).hexdigest()[:16]}"
        current_time = time.time()
        if not force_refresh:
            with self._cache_lock:
                if cache_key in self._audio_url_cache:
                    cached = self._audio_url_cache[cache_key]
                    cache_age = current_time - cached['timestamp']
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug(f"📦 [DEBUG] Cache found, age: {cache_age:.1f}s, failed: {cached.get('failed', False)}")
                    ttl = cached.get('ttl', 1800)
                    if cache_age < ttl and not cached.get('failed', False):
                        with self._stats_lock:
                            self._stats['cache_hits'] += 1
                        return cached['url']
                    elif cache_age < 300 and cached.get('failed', False):
                        return None
        platform_id, platform_name = self.detect_platform(cleaned_url)
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug(f"🔍 [DEBUG] Platform detected: {platform_id} ({platform_name})")
        result = None
        extraction_method = "unknown"
        try:
            if cleaned_url.startswith('file://'):
                ok, real_path = PlatformUtils.validate_file_path(cleaned_url)
                if ok:
                    result = cleaned_url
                    extraction_method = "local_file"
                else:
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug(f"❌ [DEBUG] File validation failed: {real_path}")
                    self._last_error = real_path
            if not result:
                url_lower = cleaned_url.lower()
                AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.opus', '.webm')
                VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.wmv', '.flv')
                if url_lower.endswith(AUDIO_EXTENSIONS) or url_lower.endswith(VIDEO_EXTENSIONS):
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug("🎵 [DEBUG] Direct audio/video link detected")
                    result = cleaned_url
                    extraction_method = "direct_link"
            if not result and platform_id in ['youtube', 'youtube_live']:
                if self._debug or DEBUG_LEVEL >= 2:
                    logger.debug("🎯 [DEBUG] YouTube detected, using optimized extraction with cookies...")
                result = self._extract_youtube_audio_optimized(cleaned_url, platform_id)
                extraction_method = "youtube_optimized"
            elif not result:
                if self._debug or DEBUG_LEVEL >= 2:
                    logger.debug("🌐 [DEBUG] Non-YouTube platform, using generic extraction...")
                format_list = self._format_priorities.get(platform_id, self._format_priorities['generic'])
                extraction_method = "ytdlp_generic"
                for i, format_str in enumerate(format_list[:2]):
                    try:
                        if self._debug or DEBUG_LEVEL >= 2:
                            logger.debug(f"  🔄 Trying format {i+1}: {format_str}")
                        cmd = [
                            'yt-dlp',
                            '-g',
                            '-f', format_str,
                            '--no-warnings',
                            '--no-check-certificate',
                            '--socket-timeout', '15',
                            cleaned_url
                        ]
                        # URL wird als separates Argument übergeben, keine Shell-Injection möglich
                        process_result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=15,
                            shell=False,
                            encoding='utf-8',
                            errors='ignore'
                        )
                        if self._debug or DEBUG_LEVEL >= 2:
                            logger.debug(f"  📊 yt-dlp result: returncode={process_result.returncode}, stdout={len(process_result.stdout)} chars")
                        if process_result.returncode == 0 and process_result.stdout.strip():
                            lines = process_result.stdout.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and line.startswith(('http://', 'https://')):
                                    result = line
                                    break
                            if result:
                                if self._debug or DEBUG_LEVEL >= 2:
                                    logger.debug(f"  ✅ Erfolg mit Format {format_str}")
                                break
                    except subprocess.TimeoutExpired:
                        if self._debug or DEBUG_LEVEL >= 2:
                            logger.debug(f"  ⏰ Timeout with format {format_str}")
                        continue
                    except Exception as e:
                        if self._debug or DEBUG_LEVEL >= 2:
                            logger.debug(f"  ⚠️ Error with format {format_str}: {str(e)[:50]}")
                        continue
                if not result:
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug("  🔄 Trying JSON fallback...")
                    try:
                        json_result = self._json_extraction_fallback(cleaned_url)
                        if json_result:
                            result = json_result
                            extraction_method = "json_fallback"
                    except Exception as e:
                        if self._debug or DEBUG_LEVEL >= 2:
                            logger.debug(f"  ⚠️ JSON fallback error: {str(e)[:50]}")
        except Exception as e:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug(f"❌ [DEBUG] EXCEPTION in extract_audio_url: {e}")
            self._last_error = f"Exception: {str(e)[:100]}"
            with self._stats_lock:
                self._stats['errors'] += 1
        ttl = 300 if platform_id in ['youtube', 'youtube_live'] else 1800
        cache_entry = {
            'url': result,
            'timestamp': current_time,
            'failed': result is None,
            'method': extraction_method,
            'platform': platform_id,
            'ttl': ttl,
        }
        with self._cache_lock:
            self._audio_url_cache[cache_key] = cache_entry
            if len(self._audio_url_cache) > 50:
                self._audio_url_cache.popitem(last=False)
        self._last_method = extraction_method
        if result:
            with self._stats_lock:
                self._stats['successful_extractions'] += 1
        if self._debug or DEBUG_LEVEL >= 1:
            logger.debug(f"🎵 [DEBUG] EXTRACT_AUDIO_URL END - Result: {'✅ ' + result[:80] + '...' if result else '❌ None'}")
        return result

    def _extract_youtube_audio_optimized(self, url: str, platform_id: str) -> Optional[str]:
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug(f"  🔍 [DEBUG] OPTIMIZED YouTube extraction for: {url[:60]}...")
        video_id = self._extract_youtube_video_id(url)
        if not video_id or len(video_id) != 11:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug("  ❌ Invalid video ID")
            return None
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug(f"  🔍 Video ID: {video_id}")

        # NEW: Nur Cookie-basierte Methoden ausführen, wenn Einstellung aktiv
        if self.use_browser_cookies:
            for browser_cmd, browser_name in self._browsers:
                try:
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug(f"    🧪 Testing with {browser_name} cookies...")
                    cmd = [
                        'yt-dlp',
                        '-g',
                        '-f', 'bestaudio[ext=m4a]/bestaudio/best',
                        '--cookies-from-browser', browser_cmd,
                        '--no-warnings',
                        '--no-check-certificate',
                        '--socket-timeout', '15',
                        url
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=20,
                        shell=False,
                        encoding='utf-8',
                        errors='ignore'
                    )
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug(f"    📊 Result: returncode={result.returncode}, stdout={len(result.stdout)} chars")
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and line.startswith(('http://', 'https://')):
                                if self._debug or DEBUG_LEVEL >= 2:
                                    logger.debug(f"    ✅ Success with {browser_name} cookies")
                                return line
                    if result.stderr and self._debug:
                        logger.debug(f"    📝 Stderr: {result.stderr[:100]}")
                except subprocess.TimeoutExpired:
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug(f"    ⏰ Timeout with {browser_name} cookies")
                    continue
                except Exception as e:
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug(f"    ⚠️ Error with {browser_name} cookies: {str(e)[:50]}")
                    continue
        else:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug("    ⏩ Cookie-based extraction skipped (user setting)")

        methods = [
            {
                'name': 'Standard yt-dlp',
                'cmd': [
                    'yt-dlp',
                    '-g',
                    '-f', 'bestaudio[ext=m4a]/bestaudio/best',
                    '--no-warnings',
                    '--no-check-certificate',
                    '--user-agent', self._user_agents['desktop'],
                    '--referer', 'https://www.youtube.com/',
                    '--socket-timeout', '15',
                    url
                ],
                'timeout': 20
            },
            {
                'name': 'Mobile User-Agent',
                'cmd': [
                    'yt-dlp',
                    '-g',
                    '-f', 'bestaudio/best',
                    '--no-warnings',
                    '--no-check-certificate',
                    '--user-agent', self._user_agents['mobile'],
                    '--referer', 'https://m.youtube.com/',
                    '--socket-timeout', '15',
                    url
                ],
                'timeout': 20
            },
            {
                'name': 'Lowest Quality',
                'cmd': [
                    'yt-dlp',
                    '-g',
                    '-f', 'worstaudio',
                    '--no-warnings',
                    '--no-check-certificate',
                    '--socket-timeout', '15',
                    url
                ],
                'timeout': 20
            },
        ]
        for method in methods:
            try:
                if self._debug or DEBUG_LEVEL >= 2:
                    logger.debug(f"    🧪 Testing: {method['name']}")
                result = subprocess.run(
                    method['cmd'],
                    capture_output=True,
                    text=True,
                    timeout=method['timeout'],
                    shell=False,
                    encoding='utf-8',
                    errors='ignore'
                )
                if self._debug or DEBUG_LEVEL >= 2:
                    logger.debug(f"    📊 Result: returncode={result.returncode}, stdout={len(result.stdout)} chars")
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith(('http://', 'https://')):
                            if self._debug or DEBUG_LEVEL >= 2:
                                logger.debug(f"    ✅ Success with {method['name']}")
                            return line
                else:
                    if result.stderr:
                        error = result.stderr[:100]
                        if self._debug or DEBUG_LEVEL >= 2:
                            logger.debug(f"    📝 Stderr: {error}")
                        if "Requested format is not available" in error:
                            continue
            except subprocess.TimeoutExpired:
                if self._debug or DEBUG_LEVEL >= 2:
                    logger.debug(f"    ⏰ Timeout: {method['name']}")
                continue
            except Exception as e:
                if self._debug or DEBUG_LEVEL >= 2:
                    logger.debug(f"    ⚠️ Error: {str(e)[:50]}")
                continue

        try:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug("    🔄 Trying JSON fallback...")
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-warnings',
                '--no-check-certificate',
                '--user-agent', self._user_agents['desktop'],
                '--socket-timeout', '20',
                url
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=25,
                shell=False,
                encoding='utf-8',
                errors='ignore'
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    data = json.loads(result.stdout)
                    best_audio = None
                    best_bitrate = 0
                    for fmt in data.get('formats', []):
                        if fmt.get('acodec') != 'none' and fmt.get('url'):
                            bitrate = fmt.get('abr', 0) or fmt.get('tbr', 0) or 0
                            if fmt.get('vcodec') == 'none':
                                bitrate += 1000
                            ext = fmt.get('ext', '').lower()
                            if ext in ['m4a', 'mp4']:
                                bitrate += 500
                            elif ext in ['webm', 'opus']:
                                bitrate += 300
                            if bitrate > best_bitrate:
                                best_bitrate = bitrate
                                best_audio = fmt['url']
                    if best_audio:
                        if self._debug or DEBUG_LEVEL >= 2:
                            logger.debug("    ✅ JSON fallback success")
                        return best_audio
                except json.JSONDecodeError as e:
                    if self._debug or DEBUG_LEVEL >= 2:
                        logger.debug(f"    ⚠️ JSON decode error: {e}")
        except Exception as e:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug(f"    ⚠️ JSON fallback error: {str(e)[:50]}")

        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug("    🔄 Generating direct audio URL...")
        direct_url = f"https://manifest.googlevideo.com/api/manifest/dash/id/{video_id}/source/youtube"
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug("    🔧 Generated direct URL")
        return direct_url

    def _json_extraction_fallback(self, url: str) -> Optional[str]:
        try:
            cmd = ['yt-dlp', '--dump-json', '--no-warnings', '--no-check-certificate', url]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                shell=False,
                encoding='utf-8',
                errors='ignore'
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                audio_formats = []
                for fmt in data.get('formats', []):
                    if fmt.get('acodec') != 'none' and fmt.get('url'):
                        audio_formats.append({
                            'url': fmt['url'],
                            'abr': fmt.get('abr', 0) or fmt.get('tbr', 0) or 0,
                            'ext': fmt.get('ext', ''),
                            'vcodec': fmt.get('vcodec', 'none')
                        })
                if audio_formats:
                    audio_only = [f for f in audio_formats if f['vcodec'] == 'none']
                    if audio_only:
                        audio_formats = audio_only
                    audio_formats.sort(key=lambda x: x['abr'], reverse=True)
                    return audio_formats[0]['url']
        except Exception as e:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug(f"  ⚠️ JSON extraction error: {str(e)[:50]}")
        return None

    def extract_stream_info(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug(f"\n🎯 [DEBUG] EXTRACT_STREAM_INFO für: {url[:60]}...")
        try:
            cache_key = f"info_{hashlib.md5(url.encode()).hexdigest()[:16]}"
            current_time = time.time()
            with self._cache_lock:
                if not force_refresh and cache_key in self._stream_info_cache:
                    cached = self._stream_info_cache[cache_key]
                    if current_time - cached['timestamp'] < 600:
                        if self._debug or DEBUG_LEVEL >= 2:
                            logger.debug("📦 [DEBUG] Stream info cache hit")
                        return cached['info']
            platform_id, platform_name = self.detect_platform(url)
            info = {
                'title': platform_name,
                'uploader': 'Unknown',
                'duration': 'Unknown',
                'view_count': 0,
                'is_live': False,
                'live_status': 'not_live',
                'thumbnail': '',
                'description': '',
                'platform': platform_id,
                'extractor': 'direct',
                'webpage_url': url,
                'extraction_time': current_time
            }
            with self._cache_lock:
                self._stream_info_cache[cache_key] = {
                    'info': info,
                    'timestamp': current_time
                }
                if len(self._stream_info_cache) > 30:
                    oldest_key = min(self._stream_info_cache.items(), key=lambda x: x[1]['timestamp'])[0]
                    del self._stream_info_cache[oldest_key]
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug("✅ [DEBUG] Stream info extracted")
            return info
        except (json.JSONDecodeError, KeyError) as e:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug(f"⚠️ Stream info extraction error: {e}")
        except Exception as e:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug(f"❌ [DEBUG] EXCEPTION in extract_stream_info: {e}")
            platform_id, platform_name = self.detect_platform(url)
            return {
                'title': platform_name,
                'uploader': 'Unknown',
                'duration': 'Unknown',
                'view_count': 0,
                'is_live': False,
                'live_status': 'unknown',
                'thumbnail': '',
                'description': '',
                'platform': platform_id,
                'extractor': 'error',
                'webpage_url': url,
                'extraction_time': time.time()
            }

    def get_ffmpeg_params_for_url(self, url: str) -> Dict[str, Any]:
        is_youtube_hls = ('manifest.googlevideo.com' in url and
                         ('/hls_playlist/' in url or '.m3u8' in url))
        is_youtube_dash = ('manifest.googlevideo.com' in url and
                          '/dash/' in url)
        if is_youtube_hls or is_youtube_dash:
            headers = [
                'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Origin: https://www.youtube.com',
                'Referer: https://www.youtube.com/',
                'Accept: */*',
                'Accept-Language: en-US,en;q=0.9',
                'Accept-Encoding: gzip, deflate, br',
                'Connection: keep-alive',
                'Sec-Fetch-Dest: empty',
                'Sec-Fetch-Mode: cors',
                'Sec-Fetch-Site: same-site',
            ]
            hls_params = [
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '5',
                '-reconnect_on_network_error', '1',
                '-timeout', '10000000',
                '-rw_timeout', '30000000',
                '-multiple_requests', '1',
                '-seekable', '0',
                '-fflags', '+discardcorrupt+fastseek+genpts',
                '-headers', '\\r\\n'.join(headers),
            ]
            return {
                'input_params': hls_params,
                'output_params': [
                    '-f', 's16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-vn',
                    '-fflags', '+genpts',
                ],
                'is_live': True,
                'timeout': 90,
                'buffer_size': 16384,
                'platform': 'youtube_hls',
                'reconnect_attempts': 10,
                'requires_headers': True,
                'headers': headers
            }
        elif 'youtube.com' in url or 'youtu.be' in url:
            return {
                'input_params': [
                    '-reconnect', '1',
                    '-reconnect_streamed', '1',
                    '-reconnect_delay_max', '30',
                    '-rw_timeout', '50000000',
                    '-headers', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    '-headers', 'Referer: https://www.youtube.com/',
                    '-headers', 'Origin: https://www.youtube.com',
                ],
                'output_params': [
                    '-f', 's16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-vn',
                    '-fflags', '+genpts+discardcorrupt',
                ],
                'is_live': False,
                'timeout': 60,
                'buffer_size': 4096,
                'platform': 'youtube',
                'reconnect_attempts': 5
            }
        else:
            return {
                'input_params': [
                    '-rw_timeout', '30000000',
                    '-fflags', '+discardcorrupt+genpts',
                ],
                'output_params': [
                    '-f', 's16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-vn',
                    '-avoid_negative_ts', 'make_zero',
                ],
                'is_live': False,
                'timeout': 30,
                'buffer_size': 2048,
                'platform': 'generic'
            }

    def validate_url_for_processing(self, url: str) -> Tuple[bool, str]:
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug(f"\n🔍 [DEBUG] VALIDATE_URL für: {url[:80]}...")
        if not url or not isinstance(url, str):
            return False, "Invalid input"
        cleaned_url = PlatformUtils.sanitize_url(url.strip())
        if not cleaned_url:
            return False, "Empty URL"
        if cleaned_url.startswith('file://'):
            ok, real_path = PlatformUtils.validate_file_path(cleaned_url)
            if not ok:
                return False, real_path
            file_path = real_path
            if not os.path.exists(file_path):
                return False, "File not found"
            if not os.path.isfile(file_path):
                return False, "Not a valid file"
            try:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    return False, "File is empty"
                filename = os.path.basename(file_path)
                return True, f"File: {filename}"
            except OSError:
                return False, "File access error"
        if not cleaned_url.startswith(('http://', 'https://')):
            return False, "Invalid URL format"
        if len(cleaned_url) > 2000:
            return False, "URL too long"
        audio_url = self.extract_audio_url(cleaned_url)
        if not audio_url:
            platform_id, platform_name = self.detect_platform(cleaned_url)
            error_msg = f"No audio URL extractable ({platform_name})"
            if self._last_error:
                error_msg += f" - {self._last_error}"
            if self._last_method:
                error_msg += f" [method: {self._last_method}]"
            return False, error_msg
        platform_id, platform_name = self.detect_platform(cleaned_url)
        status_parts = [platform_name]
        if 'youtube' in platform_id:
            try:
                cmd = ['yt-dlp', '--dump-json', '--playlist-items', '1', '--no-warnings', cleaned_url]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=False,
                    encoding='utf-8',
                    errors='ignore'
                )
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        data = json.loads(result.stdout)
                        if data.get('title'):
                            title = data['title'][:40]
                            status_parts.insert(0, title)
                        if data.get('duration_string'):
                            status_parts.append(f"⏱️ {data['duration_string']}")
                        if data.get('is_live'):
                            status_parts.append("🔴 LIVE")
                    except (json.JSONDecodeError, KeyError):
                        pass
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
                pass
        status = " | ".join(status_parts)
        return True, status

    def get_diagnostics(self) -> Dict[str, Any]:
        current_time = time.time()
        with self._stats_lock:
            uptime = current_time - self._stats['start_time']
            stats = self._stats.copy()
            stats.update({
                'uptime_seconds': uptime,
                'uptime_human': str(timedelta(seconds=int(uptime))),
                'success_rate': (stats['successful_extractions'] / stats['extraction_attempts'] * 100
                                if stats['extraction_attempts'] > 0 else 0),
                'cache_hit_rate': (stats['cache_hits'] / stats['extraction_attempts'] * 100
                                  if stats['extraction_attempts'] > 0 else 0)
            })
        return {
            'last_error': self._last_error,
            'last_method': self._last_method,
            'stats': stats,
            'cache_sizes': {
                'platform': len(self._platform_cache),
                'audio_url': len(self._audio_url_cache),
                'live_status': len(self._live_status_cache),
                'stream_info': len(self._stream_info_cache)
            },
            'debug_mode': self._debug
        }

    def clear_caches(self) -> None:
        with self._cache_lock:
            self._platform_cache.clear()
            self._audio_url_cache.clear()
            self._live_status_cache.clear()
            self._stream_info_cache.clear()
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug("🗑️ [DEBUG] All caches cleared")

    def dispose(self) -> None:
        self.clear_caches()
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug("🔌 [DEBUG] StreamManager disposed")


# -----------------------------------------------------------------------------
# FFMPEG MANAGER (überarbeitet mit Thread-Sicherheit und Ressourcenmanagement)
# -----------------------------------------------------------------------------
class ExcellenceFFmpegManager:
    def __init__(self, config: Optional[ExcellenceConfig] = None,
                 stream_manager: Optional[StreamManager] = None) -> None:
        self._processes: Dict[str, Dict[str, Any]] = {}
        self._process_counter = 0
        self._lock = threading.RLock()
        self._active_count = 0
        self._shutting_down = False
        self.config = config or ExcellenceConfig()
        self.stream_manager = stream_manager or StreamManager()
        self._pid_tracking: Dict[int, Dict[str, Any]] = {}
        self._live_detection_cache: Dict[str, Dict[str, Any]] = {}
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="FFmpegCleanup"
        )
        self._cleanup_thread.start()
        self._stats = {
            'extraction_attempts': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'cache_hits': 0,
            'start_time': time.time()
        }
        logger.info(f"✅ FFmpeg Manager initialized (Platform: {SYSTEM})")

    def set_stream_manager(self, stream_manager: StreamManager) -> 'ExcellenceFFmpegManager':
        if stream_manager:
            self.stream_manager = stream_manager
            logger.info("✅ FFmpegManager: StreamManager linked")
        return self

    def _build_ffmpeg_command_optimized(self, url: str) -> List[str]:
        is_live, platform = self._detect_stream_type(url)
        stream_type = "LIVE" if is_live else "VIDEO"
        logger.info(f"\n🎬 Building FFmpeg command for {platform} ({stream_type})")
        logger.info(f"  📍 URL: {url[:80]}...")
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']

        if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
            logger.info("  🎯 Adding YouTube-specific headers")
            # Header aus der Konfiguration holen (YOUTUBE_HEADERS)
            headers_dict = self.config.get_youtube_headers(is_manifest='manifest.googlevideo.com' in url)
            headers_list = [f"{k}: {v}" for k, v in headers_dict.items()]
            headers_string = '\r\n'.join(headers_list)
            cmd.extend(['-headers', headers_string])

        if is_live:
            logger.info("  📡 LIVE: Using HLS/Live optimization")
            cmd.extend([
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '5',
                '-reconnect_on_network_error', '1',
                '-timeout', '10000000',
                '-rw_timeout', '30000000',
                '-multiple_requests', '1',
                '-seekable', '0',
                '-fflags', '+discardcorrupt+fastseek+genpts',
            ])
        else:
            logger.info("  🎬 VIDEO: Fast access for non-live content")
            cmd.extend([
                '-rw_timeout', '10000000',
                '-accurate_seek',
                '-ss', '0',
                '-fflags', '+genpts+discardcorrupt+fastseek',
            ])

        cmd.extend(['-i', url])
        cmd.extend([
            '-vn',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(Constants.SAMPLE_RATE),
            '-ac', str(Constants.CHANNELS),
            '-af', 'volume=1.5,dynaudnorm',
            '-fflags', '+genpts+discardcorrupt',
            '-avoid_negative_ts', 'make_zero',
            '-max_interleave_delta', '0',
            '-threads', '2',
            '-bufsize', '2048k',
            'pipe:1'
        ])
        return cmd

    def start_stream(self, video_url: str, output_queue: Optional[queue.Queue],
                     process_id: str, force_refresh_audio_url: bool = False) -> Optional[subprocess.Popen]:
        logger.info(f"\n🎬 FFmpegManager: Starting stream for: {video_url[:80]}...")
        with self._lock:
            if self.is_active(process_id):
                logger.warning(f"⚠️ Stream {process_id} already active")
                return None
        logger.info("🎵 Resolving audio URL...")
        audio_url = self.stream_manager.extract_audio_url(video_url, force_refresh=force_refresh_audio_url)
        if not audio_url:
            logger.error("❌ Audio URL resolution failed")
            return None
        logger.info(f"✅ Resolved URL: {audio_url[:100]}...")
        cmd = self._build_ffmpeg_command_optimized(audio_url)
        try:
            process_kwargs = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.PIPE,
                'stdin': subprocess.DEVNULL,
                'bufsize': 10 * 1024 * 1024,
            }
            if IS_WINDOWS:
                process_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                process_kwargs['encoding'] = 'utf-8'
                process_kwargs['errors'] = 'ignore'
            elif IS_MACOS or IS_LINUX:
                process_kwargs['start_new_session'] = True
            logger.info("🚀 Starting FFmpeg process...")
            process = subprocess.Popen(cmd, **process_kwargs)
            logger.info(f"✅ FFmpeg process started (PID: {process.pid})")
            logger.info("⏳ Waiting for stream initialization...")
            time.sleep(3.0 if 'hls' in audio_url.lower() else 1.5)
            poll_result = process.poll()
            if poll_result is not None:
                try:
                    stderr_output = process.stderr.read(1000).decode('utf-8', errors='ignore')
                    logger.error(f"❌ FFmpeg died immediately. Exit code: {poll_result}")
                    if stderr_output:
                        logger.error("📋 FFMPEG STDERR (first 200 chars):")
                        logger.error(stderr_output[:200])
                except Exception as e:
                    logger.warning(f"⚠️ Could not read stderr: {e}")
                return None
            logger.info(f"✅ FFmpeg is running (PID: {process.pid})")
            self._register_process(process_id, process, output_queue, audio_url)
            return process
        except FileNotFoundError:
            logger.error("❌ FFmpeg not found! Please install FFmpeg.")
            return None
        except PermissionError:
            logger.error("❌ Permission denied - cannot execute FFmpeg")
            return None
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout beim Start von FFmpeg")
            return None
        except OSError as e:
            logger.error(f"❌ OS-Fehler beim Start von FFmpeg: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to start FFmpeg: {e}")
            return None

    def _register_process(self, process_id: str, process: subprocess.Popen,
                         output_queue: Optional[queue.Queue], url: str) -> None:
        with self._lock:
            is_live, platform = self._detect_stream_type(url)
            headers_used = False
            try:
                headers_used = ('-headers' in str(self._build_ffmpeg_command_optimized(url)))
            except Exception:
                pass
            self._processes[process_id] = {
                'process': process,
                'output_queue': output_queue,
                'start_time': time.time(),
                'url': url,
                'stopping': False,
                'bytes_read': 0,
                'platform': platform,
                'is_live': is_live,
                'chunks_processed': 0,
                'last_activity': time.time(),
                'headers_used': headers_used
            }
            self._active_count += 1
            self._pid_tracking[process.pid] = {
                'process_id': process_id,
                'start_time': time.time(),
                'url': url[:100],
                'platform': platform,
                'is_live': is_live
            }
            logger.info(f"📊 Process registered: {process_id} (PID: {process.pid})")

    def update_process_activity(self, process_id: str) -> None:
        with self._lock:
            if process_id in self._processes:
                self._processes[process_id]['last_activity'] = time.time()

    def _detect_stream_type(self, url: str) -> Tuple[bool, str]:
        cache_key = hashlib.md5(url.encode()).hexdigest()[:16]
        with self._lock:
            if cache_key in self._live_detection_cache:
                cached = self._live_detection_cache[cache_key]
                if time.time() - cached['timestamp'] < 300:
                    return cached['is_live'], cached['platform']
        is_live = False
        platform = "unknown"
        try:
            url_lower = url.lower()
            if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
                platform = "YouTube"
                is_live = any(indicator in url_lower for indicator in
                            ['/live', 'live=1', '/stream', 'livestream'])
            elif 'twitch.tv' in url_lower:
                platform = "Twitch"
                is_live = True
            elif 'tiktok.com' in url_lower:
                platform = "TikTok"
            elif 'facebook.com' in url_lower or 'fb.watch' in url_lower:
                platform = "Facebook"
            elif 'kick.com' in url_lower:
                platform = "Kick"
            elif 'rumble.com' in url_lower:
                platform = "Rumble"
            elif 'dailymotion.com' in url_lower:
                platform = "Dailymotion"
            elif 'vimeo.com' in url_lower:
                platform = "Vimeo"
            elif 'twitter.com' in url_lower or 'x.com' in url_lower:
                platform = "Twitter/X"
            elif url_lower.startswith('file://'):
                platform = "Local File"
                is_live = False
            elif '.m3u8' in url_lower:
                platform = "HLS Stream"
                is_live = True
            elif '.mpd' in url_lower:
                platform = "DASH Stream"
                is_live = True
            else:
                platform = "HTTP Stream"
            with self._lock:
                self._live_detection_cache[cache_key] = {
                    'is_live': is_live,
                    'platform': platform,
                    'timestamp': time.time(),
                    'url': url[:50]
                }
                if len(self._live_detection_cache) > 100:
                    oldest = min(self._live_detection_cache.items(),
                               key=lambda x: x[1]['timestamp'])[0]
                    del self._live_detection_cache[oldest]
            return is_live, platform
        except (KeyError, AttributeError) as e:
            logger.warning(f"⚠️ Stream type detection error: {e}")
            return False, "unknown"
        except Exception as e:
            logger.warning(f"⚠️ Stream type detection error: {e}")
            return False, "unknown"

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = self._stats.copy()
            stats['uptime_seconds'] = time.time() - stats['start_time']
            if stats['extraction_attempts'] > 0:
                stats['success_rate'] = (
                    stats['successful_extractions'] / stats['extraction_attempts'] * 100
                )
                stats['failure_rate'] = (
                    stats['failed_extractions'] / stats['extraction_attempts'] * 100
                )
            else:
                stats['success_rate'] = 0
                stats['failure_rate'] = 0
            stats['active_processes'] = self._active_count
            stats['total_processes'] = len(self._processes)
            stats['live_detection_cache_size'] = len(self._live_detection_cache)
            return stats

    def read_audio_data(self, process_id: str, size: int) -> Optional[bytes]:
        with self._lock:
            if process_id not in self._processes:
                return None
            process_info = self._processes[process_id]
            if process_info.get('stopping', False):
                return None
            process = process_info['process']
        # Lesen außerhalb des Locks, um Deadlocks zu vermeiden
        try:
            audio_data = process.stdout.read(size)
            if audio_data:
                with self._lock:
                    process_info['bytes_read'] += len(audio_data)
                    process_info['chunks_processed'] += 1
                    process_info['last_activity'] = time.time()
                return audio_data
            else:
                # Prüfen, ob Prozess beendet ist
                if process.poll() is not None:
                    exit_code = process.poll()
                    logger.warning(f"⚠️ Process {process_id} terminated (exit: {exit_code})")
                    try:
                        stderr = process.stderr.read(300).decode('utf-8', errors='ignore')
                        if stderr:
                            logger.info(f"📝 Last error: {stderr[:150]}")
                    except Exception:
                        pass
                    self.stop_stream(process_id)
                    return None
                return None
        except (IOError, OSError, ValueError) as e:
            logger.warning(f"⚠️ Read error for {process_id}: {e}")
            self.stop_stream(process_id)
            return None
        except Exception as e:  # zusätzlicher Schutz
            logger.warning(f"⚠️ Unexpected read error for {process_id}: {e}")
            self.stop_stream(process_id)
            return None

    def stop_stream(self, process_id: str) -> bool:
        with self._lock:
            if process_id not in self._processes:
                return True
            process_info = self._processes[process_id]
            if process_info.get('stopping', False):
                return True
            process = process_info['process']
            termination_success = False
            try:
                if process.poll() is None:
                    logger.info(f"🔄 Stopping process {process_id} ({process.pid})...")
                    try:
                        process.terminate()
                        process.wait(timeout=1.0)
                        termination_success = True
                        logger.info(f"✅ Process {process_id} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                            process.wait(timeout=1.0)
                            termination_success = True
                            logger.info(f"✅ Process {process_id} killed")
                        except subprocess.TimeoutExpired:
                            termination_success = False
                            logger.error(f"❌ Could not terminate {process_id}")
                else:
                    termination_success = True
                    logger.info(f"✅ Process {process_id} already terminated")
            except Exception as e:
                logger.error(f"❌ Error stopping {process_id}: {e}")
                termination_success = False
            finally:
                self._cleanup_process_resources(process_id, process)
            return termination_success

    def _cleanup_process_resources(self, process_id: str, process: subprocess.Popen) -> None:
        if process_id not in self._processes:
            return
        try:
            if process_id in self._processes:
                del self._processes[process_id]
                self._active_count = max(0, self._active_count - 1)
            if process.pid in self._pid_tracking:
                del self._pid_tracking[process.pid]
            pipes_to_close = []
            if hasattr(process, 'stdout'):
                pipes_to_close.append(process.stdout)
            if hasattr(process, 'stderr'):
                pipes_to_close.append(process.stderr)
            if hasattr(process, 'stdin'):
                pipes_to_close.append(process.stdin)
            for pipe in pipes_to_close:
                if pipe and not pipe.closed:
                    try:
                        pipe.close()
                    except Exception:
                        pass
            if process.poll() is None:
                try:
                    process.terminate()
                    time.sleep(0.1)
                except (OSError, PermissionError):
                    pass
                try:
                    process.kill()
                    time.sleep(0.1)
                except (OSError, PermissionError):
                    pass
            try:
                if process.poll() is None:
                    process.kill()
            except (OSError, PermissionError):
                pass
            # gc.collect() hier entfernt, da es in dispose() gemacht wird
            logger.debug(f"🧹 Resources cleaned for: {process_id}")
        except Exception as e:
            logger.warning(f"⚠️ Resource cleanup error for {process_id}: {e}")

    def stop_all_streams(self) -> None:
        logger.info("🛑 Stopping all streams...")
        with self._lock:
            self._shutting_down = True
            process_ids = list(self._processes.keys())
            success_count = 0
            fail_count = 0
            for process_id in process_ids:
                try:
                    if self.stop_stream(process_id):
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error stopping {process_id}: {e}")
                    fail_count += 1
            self._shutting_down = False
            logger.info(f"✅ Streams stopped: {success_count} successful, {fail_count} failed")

    def is_active(self, process_id: str) -> bool:
        with self._lock:
            if process_id not in self._processes:
                return False
            process = self._processes[process_id]['process']
            if process.poll() is not None:
                return False
            last_activity = self._processes[process_id].get('last_activity', 0)
            if time.time() - last_activity > 30:
                return False
            return True

    def _cleanup_worker(self) -> None:
        while self._cleanup_running:
            try:
                time.sleep(30)
                self.cleanup_stale_processes()
            except Exception as e:
                logger.warning(f"⚠️ Cleanup worker error: {e}")

    def cleanup_stale_processes(self) -> None:
        with self._lock:
            stale_processes = []
            for process_id, process_info in self._processes.items():
                process = process_info['process']
                if process.poll() is not None:
                    stale_processes.append(process_id)
            for process_id in stale_processes:
                logger.info(f"🧹 Cleaning terminated process: {process_id}")
                self._cleanup_process_resources(process_id, self._processes[process_id]['process'])

    def dispose(self) -> None:
        logger.info("🧹 Shutting down FFmpeg Manager...")
        self._cleanup_running = False
        self.stop_all_streams()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        self._live_detection_cache.clear()
        self._pid_tracking.clear()
        self._processes.clear()
        gc.collect()
        logger.info("✅ FFmpeg Manager disposed")


# -----------------------------------------------------------------------------
# STREAM INFO EXTRACTOR (unverändert, aber mit Sicherheitsvalidierung)
# -----------------------------------------------------------------------------
@dataclass
class StreamInfo:
    title: str
    uploader: str
    duration: str
    view_count: int
    platform: str
    description: str = ""


class StreamInfoExtractor:
    def __init__(self) -> None:
        self.current_info = StreamInfo(
            title="Unknown Stream",
            uploader="Unknown",
            duration="Live",
            view_count=0,
            platform="Unknown"
        )
        self._lock = threading.RLock()
        self._debug = DEBUG_LEVEL >= 1
        self.use_browser_cookies = True  # wird später von GUI gesetzt

    def extract_stream_info(self, url: str) -> StreamInfo:
        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug(f"🔍 StreamInfoExtractor.extract_stream_info für: {url[:80]}...")
        url = PlatformUtils.sanitize_url(url)

        if url.startswith('file://'):
            ok, real_path = PlatformUtils.validate_file_path(url)
            if not ok:
                return StreamInfo(
                    title="Invalid file",
                    uploader="Error",
                    duration="",
                    view_count=0,
                    platform="invalid"
                )
            file_path = real_path
            return StreamInfo(
                title=os.path.basename(file_path),
                uploader="Local File",
                duration="File",
                view_count=0,
                platform="local"
            )

        if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
            info = self._extract_youtube_info_with_cookies(url)
            if info:
                return info

        try:
            cmd = ['yt-dlp', '--dump-json', '--no-warnings', '--no-check-certificate', url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, shell=False)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                platform = "unknown"
                extractor = info.get('extractor', '').lower()
                if 'youtube' in extractor:
                    platform = "youtube"
                elif 'twitch' in extractor:
                    platform = "twitch"
                elif 'tiktok' in extractor:
                    platform = "tiktok"
                elif 'facebook' in extractor:
                    platform = "facebook"
                elif 'kick' in extractor:
                    platform = "kick"
                elif 'rumble' in extractor:
                    platform = "rumble"
                elif 'dailymotion' in extractor:
                    platform = "dailymotion"
                elif 'vimeo' in extractor:
                    platform = "vimeo"
                elif 'twitter' in extractor or 'x' in extractor:
                    platform = "twitter"
                self.current_info = StreamInfo(
                    title=info.get('title', 'Unknown Title'),
                    uploader=info.get('uploader', 'Unknown'),
                    duration=info.get('duration_string', 'Live'),
                    view_count=info.get('view_count', 0),
                    platform=platform,
                    description=info.get('description', '')
                )
                return self.current_info
        except Exception as e:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug(f"⚠️ StreamInfoExtractor: yt-dlp JSON fehlgeschlagen für {url[:50]}: {e}")

        if self._debug or DEBUG_LEVEL >= 2:
            logger.debug("🔄 StreamInfoExtractor: Fallback – extrahiere Titel aus URL")
        try:
            parsed = urllib.parse.urlparse(url)
            if 'twitch.tv' in parsed.netloc:
                path = parsed.path.strip('/')
                channel = path.split('/')[0] if path else parsed.netloc.replace('www.', '')
                title = f"{channel} (Twitch)"
                uploader = channel
            else:
                domain = parsed.netloc.replace('www.', '')
                path_segments = [s for s in parsed.path.split('/') if s]
                if path_segments:
                    last = path_segments[-1]
                    title = f"{domain} - {last}"
                else:
                    title = domain
                uploader = domain
            return StreamInfo(
                title=title,
                uploader=uploader,
                duration="Live",
                view_count=0,
                platform="unknown"
            )
        except Exception as e:
            if self._debug or DEBUG_LEVEL >= 2:
                logger.debug(f"❌ StreamInfoExtractor: URL-Fallback fehlgeschlagen: {e}")

        return StreamInfo(
            title="Unknown Stream",
            uploader="Unknown",
            duration="Live",
            view_count=0,
            platform="unknown"
        )

    def _extract_youtube_info_with_cookies(self, url: str) -> Optional[StreamInfo]:
        logger.info("  🎯 YouTube detected, trying optimized cookie methods for channel name...")

        if IS_LINUX:
            chrome_config_dir = Path.home() / '.config' / 'google-chrome'
            chromium_config_dir = Path.home() / '.config' / 'chromium'
            if chromium_config_dir.exists() and not chrome_config_dir.exists():
                try:
                    chrome_config_dir.mkdir(parents=True, exist_ok=True)
                    chromium_files = ['Local State', 'Default/Cookies', 'Default/Login Data']
                    for file_path in chromium_files:
                        chromium_file = chromium_config_dir / file_path
                        chrome_file = chrome_config_dir / file_path
                        if chromium_file.exists() and not chrome_file.exists():
                            chrome_file.parent.mkdir(parents=True, exist_ok=True)
                            os.symlink(str(chromium_file), str(chrome_file))
                    logger.info("    🔗 Created Chrome compatibility symlinks for yt-dlp")
                except Exception as e:
                    logger.warning(f"    ⚠️ Chrome symlink setup failed: {e}")

        methods: List[Tuple[List[str], str]] = []
        # NEW: Nur Cookie-Methoden hinzufügen, wenn erlaubt
        if self.use_browser_cookies:
            if IS_LINUX:
                browsers = [
                    ('firefox', 'Firefox'),
                    ('chromium', 'Chromium'),
                    ('brave', 'Brave'),
                    ('chrome', 'Chrome'),
                    ('vivaldi', 'Vivaldi'),
                    ('opera', 'Opera'),
                    ('edge', 'Edge'),
                ]
                for browser_cmd, browser_name in browsers:
                    methods.append(([
                        'yt-dlp', '--cookies-from-browser', browser_cmd, '--dump-json',
                        '--no-warnings', '--no-check-certificate', '--playlist-items', '1', url
                    ], f"{browser_name} Cookies"))
            elif IS_WINDOWS:
                browsers = [
                    ('chrome', 'Chrome'),
                    ('firefox', 'Firefox'),
                    ('edge', 'Edge'),
                    ('brave', 'Brave'),
                    ('opera', 'Opera'),
                ]
                for browser_cmd, browser_name in browsers:
                    methods.append(([
                        'yt-dlp', '--cookies-from-browser', browser_cmd, '--dump-json',
                        '--no-warnings', '--no-check-certificate', '--playlist-items', '1', url
                    ], f"{browser_name} Cookies"))
            else:  # macOS
                browsers = [
                    ('safari', 'Safari'),
                    ('chrome', 'Chrome'),
                    ('firefox', 'Firefox'),
                    ('brave', 'Brave'),
                    ('edge', 'Edge'),
                ]
                for browser_cmd, browser_name in browsers:
                    methods.append(([
                        'yt-dlp', '--cookies-from-browser', browser_cmd, '--dump-json',
                        '--no-warnings', '--no-check-certificate', '--playlist-items', '1', url
                    ], f"{browser_name} Cookies"))
        else:
            if self._debug:
                logger.debug("    ⏩ Cookie-based extraction skipped (user setting)")

        fallback_methods = [
            (['yt-dlp', '--dump-json', '--no-warnings', '--no-check-certificate',
              '--playlist-items', '1', '--quiet', url],
             "No Cookies (Quiet)"),
            (['yt-dlp', '--dump-json', '--no-warnings', '--no-check-certificate',
              '--playlist-items', '1', url],
             "Simple JSON"),
            (['yt-dlp', '--get-title', '--get-description', '--get-duration',
              '--no-warnings', '--no-check-certificate', '--quiet', url],
             "Direct Info"),
        ]
        methods.extend(fallback_methods)

        logger.info(f"    📋 Using {len(methods)} optimized extraction methods")
        max_attempts = min(3, len(methods))
        attempts = 0

        for cmd, method_name in methods:
            if attempts >= max_attempts:
                break
            attempts += 1
            try:
                logger.info(f"    🧪 Attempt {attempts}/{max_attempts}: {method_name}")
                timeout = 12 if 'Cookies' in method_name else 8
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=False,
                    encoding='utf-8',
                    errors='ignore'
                )
                if result.returncode != 0 and result.stderr:
                    error_preview = result.stderr[:80].replace('\n', ' ')
                    if IS_LINUX and 'chrome' in method_name.lower() and 'could not find' in error_preview:
                        logger.info("      ⏩ Skipping Chrome (not available)")
                        continue
                    if self._debug:
                        logger.info(f"      ❌ Error: {error_preview}")

                if result.returncode == 0 and result.stdout.strip():
                    try:
                        output = result.stdout.strip()
                        json_start = output.find('{')
                        json_end = output.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = output[json_start:json_end]
                            info = json.loads(json_str)
                            uploader = info.get('uploader', 'Unknown')
                            channel = info.get('channel', uploader)
                            creator = info.get('creator', uploader)
                            final_uploader = uploader
                            if channel != 'Unknown' and channel != uploader:
                                final_uploader = channel
                            elif creator != 'Unknown' and creator != uploader:
                                final_uploader = creator
                            if final_uploader == 'Unknown':
                                final_uploader = info.get('uploader_id', 'YouTube')

                            self.current_info = StreamInfo(
                                title=info.get('title', 'YouTube Stream'),
                                uploader=final_uploader,
                                duration=info.get('duration_string', 'Live'),
                                view_count=info.get('view_count', 0),
                                platform="youtube",
                                description=info.get('description', '')[:200] + '...' if len(info.get('description', '')) > 200 else info.get('description', '')
                            )
                            logger.info(f"      ✅ Success with {method_name}")
                            logger.info(f"        Title: {self.current_info.title[:60]}...")
                            logger.info(f"        Channel: {self.current_info.uploader}")
                            return self.current_info
                        else:
                            lines = output.split('\n')
                            for line in lines:
                                if line.strip() and not line.startswith('{') and len(line.strip()) > 10:
                                    possible_title = line.strip()
                                    if len(possible_title) > 20 and len(possible_title) < 200:
                                        self.current_info = StreamInfo(
                                            title=possible_title,
                                            uploader="YouTube",
                                            duration="Live",
                                            view_count=0,
                                            platform="youtube",
                                            description=""
                                        )
                                        logger.info("      ✅ Extracted title from output")
                                        return self.current_info
                    except json.JSONDecodeError:
                        logger.info("      ⚠️ JSON parse failed, trying text extraction...")
                        continue
                    except Exception as e:
                        logger.info(f"      ⚠️ Processing error: {str(e)[:50]}")
                        continue
            except subprocess.TimeoutExpired:
                logger.info(f"      ⏰ Timeout after {timeout}s")
                continue
            except OSError as e:
                logger.info(f"      ⚠️ OS-Fehler bei {method_name}: {e}")
                continue
            except Exception as e:
                logger.info(f"      ⚠️ Method error: {str(e)[:50]}")
                continue

        logger.info("    🔄 Ultimate fallback: Direct title extraction...")
        try:
            cmd_title = ['yt-dlp', '--get-title', '--no-warnings',
                         '--no-check-certificate', '--quiet', url]
            cmd_uploader = ['yt-dlp', '--get-filename', '-o', '%(uploader)s',
                           '--no-warnings', '--no-check-certificate', '--quiet', url]
            with ThreadPoolExecutor(max_workers=2) as executor:
                title_future = executor.submit(
                    subprocess.run, cmd_title,
                    capture_output=True, text=True, timeout=8, shell=False
                )
                uploader_future = executor.submit(
                    subprocess.run, cmd_uploader,
                    capture_output=True, text=True, timeout=8, shell=False
                )
                title_result = title_future.result(timeout=10)
                uploader_result = uploader_future.result(timeout=10)
            title = "YouTube Stream"
            uploader = "YouTube"
            if title_result.returncode == 0 and title_result.stdout.strip():
                title = title_result.stdout.strip().split('\n')[0]
            if uploader_result.returncode == 0 and uploader_result.stdout.strip():
                uploader = uploader_result.stdout.strip().split('\n')[0]
            self.current_info = StreamInfo(
                title=title[:100] if len(title) > 100 else title,
                uploader=uploader,
                duration="Live",
                view_count=0,
                platform="youtube",
                description=""
            )
            logger.info("      ✅ Success with direct extraction")
            return self.current_info
        except Exception as e:
            logger.info(f"      ⚠️ Direct extraction failed: {e}")

        logger.info("    ⚠️ Using generic YouTube info as last resort")
        return None


# -----------------------------------------------------------------------------
# LANGUAGE DETECTOR (unverändert)
# -----------------------------------------------------------------------------
class LanguageDetector:
    def __init__(self, transcription_engine: ExcellenceTranscriptionEngine) -> None:
        self.transcription_engine = transcription_engine

    def _get_media_duration(self, file_path: str) -> Optional[float]:
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, OSError):
            pass
        except Exception:
            pass
        return None

    def detect_video_language(self, video_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(video_path):
                return {"error": "File not found"}

            duration = self._get_media_duration(video_path)
            if duration is None:
                try:
                    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    if file_size_mb > 500:
                        return {"info": "Large file - direct processing recommended"}
                except OSError:
                    pass
                sample_duration = 60
            else:
                sample_duration = min(60, max(30, duration // 2))

            temp_audio = self._extract_audio_sample(video_path, duration=sample_duration)

            if not temp_audio:
                temp_audio = self._extract_audio_sample(video_path, duration=None)

            if not temp_audio:
                return {"error": "Could not extract audio"}

            result = self.transcription_engine.transcribe_audio(temp_audio, include_timestamps=False)

            if result and hasattr(result, 'language'):
                language_code = result.language
                language_name = SUPPORTED_LANGUAGES.get(language_code, "Unknown")
                return {
                    "detected_language": language_code,
                    "language_name": language_name,
                    "confidence": getattr(result, "confidence", 0.8),
                    "sample_text": result.text[:100] + "..." if len(result.text) > 100 else result.text,
                }
            else:
                return {"error": "Language could not be detected"}

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _extract_audio_sample(self, video_path: str, duration: Optional[int] = 30) -> Optional[bytes]:
        try:
            config = self.transcription_engine.advanced_settings.config
            # Temporäre Datei vermeiden, direkt in Bytes streamen
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-f', config.AUDIO_FORMAT,
                '-ar', str(config.SAMPLE_RATE),
                '-ac', str(config.CHANNELS),
                '-loglevel', 'quiet',
                '-'
            ]
            if duration is not None:
                cmd.insert(2, '-t')
                cmd.insert(3, str(duration))

            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception:
            pass
        return None


# -----------------------------------------------------------------------------
# PROGRESS DIALOG (unverändert)
# -----------------------------------------------------------------------------
class ProgressDialog:
    def __init__(self, parent: tk.Tk, title: str = "Processing...") -> None:
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x120")
        self.dialog.configure(bg=CURRENT_THEME.BG_PRIMARY)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.cancel)
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")
        content_frame = tk.Frame(self.dialog, bg=CURRENT_THEME.BG_PRIMARY, padx=20, pady=20)
        content_frame.pack(fill="both", expand=True)
        self.message_label = tk.Label(
            content_frame,
            text="Analyzing video...",
            bg=CURRENT_THEME.BG_PRIMARY,
            fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.PRIMARY,
        )
        self.message_label.pack(pady=(0, 10))
        self.progress = ttk.Progressbar(content_frame, mode="indeterminate", length=250)
        self.progress.pack(pady=(0, 15))
        button_frame = tk.Frame(content_frame, bg=CURRENT_THEME.BG_PRIMARY)
        button_frame.pack()
        self.cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel,
            bg=CURRENT_THEME.ERROR,
            fg=CURRENT_THEME.TEXT_PRIMARY,
            relief="flat",
            padx=15,
        )
        self.cancel_button.pack()
        self.is_cancelled = False
        self.progress.start()
        self._update_interval = 100
        self._is_running = True
        self._schedule_updates()

    def _schedule_updates(self) -> None:
        if self._is_running and self.dialog.winfo_exists():
            try:
                self.dialog.update_idletasks()
                self.dialog.after(self._update_interval, self._schedule_updates)
            except tk.TclError:
                self._is_running = False

    def cancel(self) -> None:
        self.is_cancelled = True
        self._is_running = False
        if self.message_label.winfo_exists():
            self.message_label.config(text="Cancelling...", fg=CURRENT_THEME.ERROR)
        if self.cancel_button.winfo_exists():
            self.cancel_button.config(text="Cancelling...", state="disabled")
        self.close()

    def update_message(self, message: str) -> None:
        if self._is_running and self.message_label.winfo_exists():
            try:
                self.message_label.config(text=message)
            except tk.TclError:
                self._is_running = False

    def close(self) -> None:
        self._is_running = False
        try:
            self.progress.stop()
        except Exception:
            pass
        try:
            if self.dialog.winfo_exists():
                self.dialog.destroy()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# AUDIO PROCESSOR (überarbeitet mit Thread-Sicherheit und Fehlerbehandlung)
# -----------------------------------------------------------------------------
class ExcellenceAudioProcessor:
    def __init__(self, controller_ref: Any, ffmpeg_manager: ExcellenceFFmpegManager,
                 advanced_settings: Optional[AdvancedSettings] = None) -> None:
        self.controller_ref = controller_ref
        self.ffmpeg_manager = ffmpeg_manager
        self.advanced_settings = advanced_settings or AdvancedSettings()
        self.config = self.advanced_settings.config
        self.sample_rate = self.config.SAMPLE_RATE
        self.channels = self.config.CHANNELS
        self.audio_format = self.config.AUDIO_FORMAT
        self.chunk_duration = self.config.CHUNK_DURATION
        self.chunk_size = self.config.CHUNK_SIZE_BYTES
        self.overlap_size = self.config.OVERLAP_SIZE_BYTES
        self.transcription_engine: Optional[ExcellenceTranscriptionEngine] = None
        self.translation_engine: Optional[ExcellenceTranslationEngine] = None
        self.plugin_manager: Optional[PluginManager] = None
        self._stop_event = threading.Event()
        self._processing = False
        self._processing_lock = threading.RLock()
        self._current_stream_id: Optional[str] = None
        self._last_successful_read_time = time.time()
        self._consecutive_empty_chunks = 0
        self._cleanup_done = False
        self._resource_lock = threading.RLock()
        self._translation_active = True
        self._last_transcription_text = ""
        self._timed_transcriptions: Deque[ExcellenceTranscriptionResult] = deque(maxlen=self.config.SUBTITLE_BUFFER_SIZE)
        self._timed_translations: Deque[ExcellenceTranslationResult] = deque(maxlen=self.config.SUBTITLE_BUFFER_SIZE)
        self._subtitle_lock = threading.RLock()
        self.subtitle_mode = False
        self._recent_transcriptions: Deque[str] = deque(maxlen=self.config.RECENT_TRANSCRIPTIONS_SIZE)
        self._duplicate_lock = threading.RLock()  # neu
        self._chunk_counter = 0
        self._empty_reads = 0
        self._stream_start_time: Optional[float] = None
        self._total_bytes_processed = 0
        self._network_quality_history: Deque[float] = deque(maxlen=20)
        self._performance_history: Deque[float] = deque(maxlen=50)
        self._chunk_processing_times: Deque[float] = deque(maxlen=10)
        self.stream_manager = StreamManager(enable_debug=(DEBUG_LEVEL >= 1))

        self._read_error_count = 0
        self._max_backoff = 30

        self._process_finished = threading.Event()
        self._process_finished.set()

        self._audio_buffer = bytearray()
        self._max_buffer_size = self.config.MAX_CHUNK_BYTES * 5

        self._finished_callback: Optional[Callable] = None

        self.last_confidence = 1.0
        self._noisereduce_counter = 0

        self._total_file_size: Optional[int] = None
        self._progress_callback: Optional[Callable[[int, Optional[int], int], None]] = None
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        logger.info("✅ AudioProcessor initialized:")
        logger.info(f"   Config Type: {self._get_config_type()}")
        logger.info(f"   Chunk: {self.chunk_duration}s / {self.chunk_size:,} bytes")
        logger.info(f"   Sample Rate: {self.sample_rate} Hz")
        logger.info(f"   Overlap: {self.overlap_size:,} bytes")
        logger.info(f"   Bytes/sec: {self.config.BYTES_PER_SECOND:,}")

    def _get_config_type(self) -> str:
        if isinstance(self.config, RealtimeConfig):
            return 'realtime'
        elif isinstance(self.config, HighAccuracyConfig):
            return 'high_accuracy'
        elif isinstance(self.config, YouTubeOptimizedConfig):
            return 'youtube'
        return 'default'

    def set_progress_callback(self, callback: Callable[[int, Optional[int], int], None]) -> None:
        self._progress_callback = callback

    def start_processing(self, url: str, transcription_callback: Callable,
                         translation_callback: Callable, info_callback: Callable,
                         error_callback: Callable,
                         finished_callback: Optional[Callable] = None) -> None:
        logger.info(f"\n🔊 [START_PROCESSING] URL: {url[:80]}...")
        logger.info(f"   Config Type: {self._get_config_type()}")
        logger.info(f"   Chunk Size: {self.chunk_size:,} bytes")

        # URL validieren
        url = PlatformUtils.sanitize_url(url)

        if url.startswith('file://'):
            ok, real_path = PlatformUtils.validate_file_path(url)
            if not ok:
                error_callback(f"❌ {real_path}")
                return
            file_path = real_path
            try:
                self._total_file_size = os.path.getsize(file_path)
                logger.info(f"📁 Lokale Datei, Größe: {self._total_file_size} bytes")
            except OSError:
                self._total_file_size = None
        else:
            self._total_file_size = None

        health_issues = self._platform_specific_health_check()
        if health_issues:
            for issue in health_issues:
                logger.warning(f"⚠️ {issue}")
                info_callback(f"⚠️ {issue}")

        with self._processing_lock:
            if self._processing:
                logger.warning("⚠️ Vorheriger Prozess läuft noch – stoppe diesen zuerst.")
                if not self.stop_processing(wait=True, timeout=10.0):
                    error_callback("❌ Vorheriger Prozess konnte nicht gestoppt werden")
                    return
            self._processing = True
            self._process_finished.clear()
            self._stop_event.clear()
            self._current_stream_id = f"stream_{int(time.time())}"
            self._stream_start_time = time.time()
            self._chunk_counter = 0
            self._total_bytes_processed = 0
            self._read_error_count = 0
            self._audio_buffer = bytearray()
            self._finished_callback = finished_callback
            logger.info(f"✅ Flags gesetzt: processing=True, ID={self._current_stream_id}")

        thread = threading.Thread(
            target=self._process_loop_enhanced,
            args=(url, transcription_callback, translation_callback,
                  info_callback, error_callback),
            daemon=True,
            name=f"AudioProc_{self._current_stream_id}"
        )
        thread.start()
        logger.info(f"✅ Processing thread gestartet: {thread.name}")

    def _process_loop_enhanced(self, url: str, transcription_callback: Callable,
                               translation_callback: Callable, info_callback: Callable,
                               error_callback: Callable) -> None:
        process: Optional[subprocess.Popen] = None
        detected_language: Optional[str] = None
        error_occurred = False
        try:
            logger.info(f"\n🎬 [PROCESS_LOOP] Start für: {url[:60]}...")
            info_callback("🔍 Extracting audio URL...")
            audio_url = self.stream_manager.extract_audio_url(url)
            if not audio_url:
                error_callback("❌ Could not extract audio URL")
                error_occurred = True
                return
            logger.info(f"✅ Audio URL: {audio_url[:80]}...")
            info_callback("🔍 Testing audio stream...")
            if not self._test_audio_stream(audio_url):
                logger.warning("⚠️ Stream test failed, trying anyway...")
            info_callback("🔧 Setting up FFmpeg...")
            process_kwargs = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.PIPE,
                'stdin': subprocess.DEVNULL,
                'bufsize': 10 * 1024 * 1024,
            }
            platform = 'windows' if IS_WINDOWS else 'macos' if IS_MACOS else 'linux'
            platform_config = self.config.get_platform_config(platform)
            for key, value in platform_config.items():
                if key != 'ffmpeg_flags':
                    process_kwargs[key] = value
            logger.info("🚀 Starting FFmpeg process...")
            try:
                process = self.ffmpeg_manager.start_stream(
                    video_url=url,
                    output_queue=None,
                    process_id=self._current_stream_id
                )
                if process is None:
                    error_callback("❌ FFmpeg konnte nicht gestartet werden")
                    error_occurred = True
                    return
            except FileNotFoundError:
                error_callback("❌ FFmpeg not found - please install")
                error_occurred = True
                return
            except (OSError, PermissionError) as e:
                error_callback(f"❌ FFmpeg konnte nicht gestartet werden: {e}")
                error_occurred = True
                return
            logger.info(f"✅ FFmpeg started (PID: {process.pid})")
            info_callback("⏳ Initializing stream...")
            wait_time = self.config.INITIAL_BUFFER_SECONDS
            if any(keyword in audio_url.lower() for keyword in ['hls', '.m3u8', 'manifest.googlevideo.com']):
                wait_time = 3.0
                logger.info(f"🎯 HLS/Live stream detected, waiting {wait_time}s...")
            time.sleep(wait_time)
            if process.poll() is not None:
                try:
                    stderr = process.stderr.read(1000).decode('utf-8', errors='ignore')
                    error_msg = f"FFmpeg died: {stderr[:200]}"
                    logger.error(f"❌ {error_msg}")
                    error_callback(f"❌ {error_msg}")
                except Exception:
                    error_callback("❌ FFmpeg failed to start")
                error_occurred = True
                return
            info_callback("✅ Stream connected - starting transcription...")
            is_youtube = any(domain in audio_url for domain in ['youtube.com', 'youtu.be', 'googlevideo.com'])
            if is_youtube:
                logger.info("🎯 Using YouTube-optimized streaming loop")
                self._youtube_streaming_loop(
                    process, audio_url, url, detected_language,
                    transcription_callback, translation_callback,
                    info_callback, error_callback
                )
            else:
                logger.info("🎯 Using standard streaming loop")
                self._standard_streaming_loop(
                    process, audio_url, detected_language,
                    transcription_callback, translation_callback,
                    info_callback, error_callback
                )
            logger.info(f"🔚 [LOOP END] Reason: {'Stop requested' if self._stop_event.is_set() else 'Process ended'}")
        except subprocess.TimeoutExpired:
            error_callback("❌ Timeout - stream not reachable")
            error_occurred = True
        except FileNotFoundError:
            error_callback("❌ FFmpeg not found - please install")
            error_occurred = True
        except OSError as e:
            error_msg = f"OS error: {str(e)[:100]}"
            logger.error(f"❌ {error_msg}")
            error_callback(f"❌ {error_msg}")
            error_occurred = True
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)[:100]}"
            logger.error(f"❌ {error_msg}")
            if DEBUG_LEVEL >= 2:
                logger.exception("Stacktrace:")
            error_callback(f"❌ {error_msg}")
            error_occurred = True
        finally:
            self._flush_audio_buffer(transcription_callback, translation_callback)
            if process:
                self.ffmpeg_manager.stop_stream(self._current_stream_id)
            self._log_final_stats()
            self._guaranteed_cleanup()
            self._process_finished.set()

            is_local_file = url.startswith('file://')
            normal_end = not self._stop_event.is_set() and not error_occurred
            if is_local_file and normal_end and self._finished_callback:
                logger.info("✅ Datei normal beendet – rufe finished_callback auf")
                self._finished_callback()
            elif not self._stop_event.is_set() and not error_occurred:
                error_callback("❌ Stream wurde unerwartet beendet – versuche Neuverbindung...")

            logger.info("✅ Processing loop ended")

    def _standard_streaming_loop(self, process: subprocess.Popen, audio_url: str,
                                 detected_language: Optional[str],
                                 transcription_callback: Callable,
                                 translation_callback: Callable,
                                 info_callback: Callable,
                                 error_callback: Callable) -> None:
        last_data_time = time.time()
        consecutive_timeouts = 0
        backoff = 1.0
        max_reconnects = 10
        reconnect_attempts = 0

        while self._processing and not self._stop_event.is_set():
            if process.poll() is not None:
                logger.info("FFmpeg process terminated – finishing loop.")
                break

            current_time = time.time()
            if current_time - last_data_time > self.config.STREAM_TIMEOUT:
                consecutive_timeouts += 1
                if consecutive_timeouts > self.config.MAX_CONSECUTIVE_ERRORS:
                    if reconnect_attempts < max_reconnects:
                        reconnect_attempts += 1
                        wait = min(self._max_backoff, backoff)
                        logger.warning(f"⚠️ Stream timeout - reconnecting attempt {reconnect_attempts}/{max_reconnects}, waiting {wait:.1f}s")
                        info_callback(f"🔄 Reconnecting... ({reconnect_attempts}/{max_reconnects})")
                        self._stop_event.wait(wait)
                        backoff *= 2
                        consecutive_timeouts = 0
                        continue
                    else:
                        logger.warning("⚠️ Stream timeout - max reconnects reached")
                        error_callback("❌ Stream timeout - no data received")
                        break
                else:
                    wait = min(self._max_backoff, backoff)
                    logger.warning(f"⚠️ Temporary timeout ({consecutive_timeouts}/{self.config.MAX_CONSECUTIVE_ERRORS}), waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue
            else:
                consecutive_timeouts = 0
                backoff = 1.0

            if self._chunk_counter > 0 and self._chunk_counter % 100 == 0:
                info_callback(f"📊 {self._chunk_counter} chunks processed...")
            if (self.config.LOG_PERFORMANCE and
                self._chunk_counter > 0 and
                self._chunk_counter % self.config.PERFORMANCE_LOG_INTERVAL == 0):
                logger.info(f"📊 {self._chunk_counter} chunks processed")

            try:
                audio_data = self._read_with_timeout(process, self.chunk_size, timeout=0.5)
            except (IOError, OSError) as e:
                logger.warning(f"⚠️ Read error: {e}")
                self._read_error_count += 1
                wait = min(self._max_backoff, self.config.READ_RETRY_DELAY * (2 ** (self._read_error_count - 1)))
                time.sleep(wait)
                continue
            except Exception as e:
                logger.warning(f"⚠️ Unexpected read error: {e}")
                self._read_error_count += 1
                wait = min(self._max_backoff, self.config.READ_RETRY_DELAY * (2 ** (self._read_error_count - 1)))
                time.sleep(wait)
                continue

            if audio_data is None:
                if process.poll() is not None:
                    logger.debug("Process ended and no more data – finishing loop.")
                    break
                else:
                    self._empty_reads += 1
                    if self._empty_reads > self.config.MAX_EMPTY_READS:
                        logger.warning(f"⚠️ Too many empty reads: {self._empty_reads}")
                        error_callback("❌ No audio data received")
                        break
                    sleep_time = min(1.0, self.config.READ_RETRY_DELAY * self._empty_reads)
                    time.sleep(sleep_time)
                    continue
            else:
                self.ffmpeg_manager.update_process_activity(self._current_stream_id)
                self._read_error_count = 0
                last_data_time = time.time()
                self._empty_reads = 0
                self._chunk_counter += 1
                self._total_bytes_processed += len(audio_data)
                if self._chunk_counter <= 3:
                    logger.debug(f"📦 Chunk #{self._chunk_counter}: {len(audio_data)} bytes")

                if self._progress_callback:
                    now = time.time()
                    if now - self._last_progress_update >= self._progress_update_interval:
                        self._last_progress_update = now
                        self._progress_callback(
                            self._total_bytes_processed,
                            self._total_file_size,
                            self._chunk_counter
                        )

                try:
                    enhanced_audio = self.enhance_audio_quality(audio_data)
                except Exception as e:
                    logger.warning(f"⚠️ Audio enhancement error: {e}")
                    enhanced_audio = audio_data

                if self.transcription_engine:
                    self._process_audio_chunk(
                        enhanced_audio,
                        transcription_callback,
                        translation_callback
                    )

    def _youtube_streaming_loop(self, process: subprocess.Popen, audio_url: str, original_video_url: str,
                                 detected_language: Optional[str],
                                 transcription_callback: Callable,
                                 translation_callback: Callable,
                                 info_callback: Callable,
                                 error_callback: Callable) -> None:
        max_reconnects = 10
        reconnect_attempt = 0
        session_active = True
        backoff = 1.0
        while session_active and self._processing and not self._stop_event.is_set():
            try:
                session_success = self._single_youtube_session(
                    process, audio_url, original_video_url, detected_language,
                    transcription_callback, translation_callback,
                    info_callback, error_callback
                )
                if not session_success and reconnect_attempt < max_reconnects:
                    reconnect_attempt += 1
                    wait = min(30, backoff)
                    logger.info(f"🔄 YouTube reconnect attempt {reconnect_attempt}/{max_reconnects}, waiting {wait:.1f}s")
                    info_callback(f"🔄 Reconnecting... ({reconnect_attempt}/{max_reconnects})")
                    if process and process.poll() is None:
                        self.ffmpeg_manager.stop_stream(self._current_stream_id)
                    time.sleep(wait)
                    backoff *= 2
                    process = self.ffmpeg_manager.start_stream(
                        video_url=original_video_url,
                        output_queue=None,
                        process_id=self._current_stream_id,
                        force_refresh_audio_url=True
                    )
                    if process is None:
                        logger.error("❌ Could not restart FFmpeg")
                        session_active = False
                        break
                    logger.info(f"✅ FFmpeg reconnected (PID: {process.pid})")
                    time.sleep(3.0)
                elif not session_success:
                    logger.error("❌ Max reconnects reached")
                    error_callback("❌ Could not maintain YouTube connection")
                    session_active = False
                else:
                    session_active = False
            except Exception as e:
                logger.warning(f"⚠️ YouTube streaming loop error: {e}")
                reconnect_attempt += 1
                wait = min(30, backoff)
                time.sleep(wait)
                backoff *= 2

    def _single_youtube_session(self, process: subprocess.Popen, audio_url: str, original_video_url: str,
                                 detected_language: Optional[str],
                                 transcription_callback: Callable,
                                 translation_callback: Callable,
                                 info_callback: Callable,
                                 error_callback: Callable) -> bool:
        chunk_read_attempts = 0
        max_chunk_attempts = 50
        last_successful_chunk_time = time.time()
        backoff = 1.0
        refresh_count = 0
        max_refresh_attempts = 3
        current_process = process
        #current_audio_url = audio_url

        while self._processing and not self._stop_event.is_set():
            current_time = time.time()
            if self._chunk_counter > 0:
                if current_time - last_successful_chunk_time > 25:
                    logger.warning("⚠️ YouTube idle timeout")
                    return False
            else:
                if current_time - last_successful_chunk_time > self.config.STREAM_TIMEOUT:
                    logger.warning("⚠️ YouTube initial timeout")
                    return False

            try:
                audio_data = self._read_with_timeout(current_process, self.chunk_size, timeout=1.0)
            except (IOError, OSError) as e:
                logger.warning(f"⚠️ Session read error: {e}")
                if current_process.poll() is not None:
                    stderr = self._read_stderr(current_process)
                    if self._needs_url_refresh(stderr):
                        logger.info(f"🔄 Detected URL refresh needed: {stderr[:200]}")
                        new_url = self._refresh_youtube_url(original_video_url)
                        if new_url:
                            refresh_count += 1
                            if refresh_count <= max_refresh_attempts:
                                new_proc = self._restart_ffmpeg_with_new_url(self._current_stream_id, original_video_url)
                                if new_proc is None:
                                    logger.error("❌ Failed to restart FFmpeg, aborting session.")
                                    return False
                                current_process = new_proc
                                refresh_count = 0
                                last_successful_chunk_time = time.time()
                                continue
                        refresh_count += 1
                        if refresh_count > max_refresh_attempts:
                            return False
                chunk_read_attempts += 1
                wait = min(10, backoff)
                time.sleep(wait)
                backoff *= 1.5
                continue

            if audio_data is None:
                if current_process.poll() is not None:
                    stderr = self._read_stderr(current_process)
                    if self._needs_url_refresh(stderr):
                        logger.info(f"🔄 Detected URL refresh needed: {stderr[:200]}")
                        new_url = self._refresh_youtube_url(original_video_url)
                        if new_url:
                            refresh_count += 1
                            if refresh_count <= max_refresh_attempts:
                                new_proc = self._restart_ffmpeg_with_new_url(self._current_stream_id, original_video_url)
                                if new_proc is None:
                                    logger.error("❌ Failed to restart FFmpeg, aborting session.")
                                    return False
                                current_process = new_proc
                                #current_audio_url = new_url
                                refresh_count = 0
                                last_successful_chunk_time = time.time()
                                continue
                            else:
                                return False
                        else:
                            refresh_count += 1
                            if refresh_count > max_refresh_attempts:
                                return False
                    else:
                        logger.warning(f"FFmpeg terminated: {stderr[:200]}")
                        return False
                else:
                    chunk_read_attempts += 1
                    if chunk_read_attempts > max_chunk_attempts:
                        logger.warning(f"⚠️ Too many failed chunk reads: {chunk_read_attempts}")
                        return False
                    wait = min(10, backoff)
                    time.sleep(wait)
                    backoff *= 1.5
                    continue
            else:
                self.ffmpeg_manager.update_process_activity(self._current_stream_id)
                self._read_error_count = 0
                chunk_read_attempts = 0
                backoff = 1.0
                last_successful_chunk_time = time.time()
                self._empty_reads = 0
                self._chunk_counter += 1
                self._total_bytes_processed += len(audio_data)
                if self._chunk_counter <= 3:
                    logger.debug(f"📦 YouTube Chunk #{self._chunk_counter}: {len(audio_data)} bytes")

                if self._progress_callback:
                    now = time.time()
                    if now - self._last_progress_update >= self._progress_update_interval:
                        self._last_progress_update = now
                        self._progress_callback(
                            self._total_bytes_processed,
                            self._total_file_size,
                            self._chunk_counter
                        )

                try:
                    enhanced_audio = self.enhance_audio_quality(audio_data)
                except Exception:
                    enhanced_audio = audio_data

                self._audio_buffer.extend(enhanced_audio)
                if len(self._audio_buffer) >= self.config.MIN_CHUNK_BYTES:
                    chunk_to_process = bytes(self._audio_buffer)
                    self._audio_buffer.clear()
                    if self.transcription_engine:
                        self._process_audio_chunk(
                            chunk_to_process,
                            transcription_callback,
                            translation_callback
                        )
                if len(self._audio_buffer) > self._max_buffer_size:
                    logger.warning(f"⚠️ Audio buffer too large ({len(self._audio_buffer)} bytes) – forcing flush")
                    chunk_to_process = bytes(self._audio_buffer)
                    self._audio_buffer.clear()
                    if self.transcription_engine:
                        self._process_audio_chunk(
                            chunk_to_process,
                            transcription_callback,
                            translation_callback
                        )

                if self._chunk_counter % 50 == 0:
                    info_callback(f"📊 {self._chunk_counter} chunks processed...")
        return True

    def _read_with_timeout(self, process: subprocess.Popen, size: int, timeout: float = 1.0) -> Optional[bytes]:
        data = b''
        start_time = time.time()
        remaining = size
        while remaining > 0 and (time.time() - start_time) < timeout:
            try:
                to_read = min(remaining, 4096)
                chunk = process.stdout.read(to_read)
                if not chunk:
                    break
                data += chunk
                remaining -= len(chunk)
            except (IOError, OSError) as e:
                logger.warning(f"⚠️ Read error in _read_with_timeout: {e}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Unexpected read error in _read_with_timeout: {e}")
                break
        return data if len(data) > 0 else None

    def _read_stderr(self, process: subprocess.Popen) -> str:
        try:
            if process.stderr:
                return process.stderr.read().decode('utf-8', errors='ignore')
        except Exception:
            pass
        return ""

    def _needs_url_refresh(self, stderr: str) -> bool:
        patterns = [
            "403", "401", "forbidden", "unauthorized", "invalid parameters",
            "http error 403", "http error 401", "access denied", "signature expired",
            "token expired", "url signature expired"
        ]
        stderr_lower = stderr.lower()
        return any(p in stderr_lower for p in patterns)

    def _refresh_youtube_url(self, video_url: str) -> Optional[str]:
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"🔄 Attempt {attempt}/{max_attempts} to refresh YouTube URL...")
                new_url = self.stream_manager.extract_audio_url(video_url, force_refresh=True)
                if new_url:
                    logger.info(f"✅ Successfully obtained new YouTube URL (attempt {attempt})")
                    if DEBUG_LEVEL >= 2:
                        logger.debug(f"   New URL: {new_url[:100]}...")
                    return new_url
            except Exception as e:
                logger.warning(f"⚠️ Refresh attempt {attempt} error: {e}")
            if attempt < max_attempts:
                wait = 2 ** (attempt - 1)
                time.sleep(wait)
        logger.error("❌ All attempts to refresh YouTube URL failed")
        return None

    def _restart_ffmpeg_with_new_url(self, process_id: str, video_url: str) -> Optional[subprocess.Popen]:
        logger.info(f"🔄 Restarting FFmpeg for {process_id} with new URL...")
        if self.ffmpeg_manager:
            self.ffmpeg_manager.stop_stream(process_id)
            time.sleep(0.5)
        new_process = self.ffmpeg_manager.start_stream(
            video_url=video_url,
            output_queue=None,
            process_id=process_id,
            force_refresh_audio_url=True
        )
        if new_process:
            logger.info(f"✅ Successfully restarted FFmpeg (new PID: {new_process.pid})")
            return new_process
        else:
            logger.error("❌ Failed to restart FFmpeg")
            return None

    def emergency_diagnosis(self, url: str) -> bool:
        logger.info(f"🔍 [EMERGENCY_DIAGNOSIS] Testing: {url[:80]}...")
        try:
            audio_url = self.stream_manager.extract_audio_url(url)
            if not audio_url:
                logger.info("  ❌ Could not extract audio URL")
                return False
            logger.info(f"  ✅ Audio URL extracted: {audio_url[:80]}...")
            is_youtube = 'youtube.com' in audio_url.lower() or 'googlevideo.com' in audio_url
            test_cmd = [
                'ffmpeg',
                '-i', audio_url,
                '-t', '3',
                '-f', 'null',
                '-',
                '-loglevel', 'error'
            ]
            timeout = 8 if is_youtube else 5
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False
            )
            if result.returncode == 0:
                logger.info("  ✅ Stream connection successful")
                return True
            else:
                error_msg = result.stderr[:100] if result.stderr else "Unknown error"
                logger.info(f"  ❌ Stream test failed: {error_msg}")
                if audio_url.startswith(('http://', 'https://')):
                    logger.info("  ⚠️  But URL looks valid, trying anyway...")
                    return True
                return False
        except subprocess.TimeoutExpired:
            logger.info("  ⏰ Stream test timeout")
            if 'youtube.com' in url.lower():
                logger.info("  ⚠️  YouTube timeout common, trying anyway...")
                return True
            return False
        except (OSError, PermissionError) as e:
            logger.info(f"  ⚠️  Emergency diagnosis OS error: {e}")
            return False
        except Exception as e:
            logger.info(f"  ⚠️  Emergency diagnosis error: {e}")
            return True

    def _log_final_stats(self) -> None:
        if self._chunk_counter == 0:
            return
        uptime = time.time() - self._stream_start_time if self._stream_start_time else 0
        logger.info("\n📊 FINAL PROCESSING STATS:")
        logger.info(f"   Config Type: {self._get_config_type()}")
        logger.info(f"   Total Chunks: {self._chunk_counter}")
        logger.info(f"   Total Bytes: {self._total_bytes_processed:,}")
        logger.info(f"   Total Time: {uptime:.1f}s")
        logger.info(f"   Avg Chunk Size: {self._total_bytes_processed/self._chunk_counter if self._chunk_counter > 0 else 0:,.0f} bytes")
        logger.info(f"   Processing Rate: {self._chunk_counter/uptime if uptime > 0 else 0:.1f} chunks/sec")
        logger.info(f"   Data Rate: {self._total_bytes_processed/uptime/1024 if uptime > 0 else 0:.1f} KB/sec")
        logger.info(f"   Empty Reads: {self._empty_reads}")

    def _process_audio_chunk(self, audio_data: bytes, transcription_callback: Callable,
                              translation_callback: Callable) -> None:
        if not self.transcription_engine:
            return
        try:
            if DEBUG_LEVEL >= 2:
                start_time = time.perf_counter()
                audio_len = len(audio_data) / (16000 * 2)

            transcription = self.transcription_engine.safe_transcribe(audio_data)

            if transcription and hasattr(transcription, 'confidence'):
                self.last_confidence = transcription.confidence
            else:
                self.last_confidence = 0.0

            if DEBUG_LEVEL >= 2 and transcription:
                elapsed = time.perf_counter() - start_time
                realtime_factor = elapsed / audio_len if audio_len > 0 else 0
                logger.debug(f"Chunk {self._chunk_counter}: {audio_len:.2f}s audio, "
                             f"transcribe {elapsed*1000:.1f}ms ({realtime_factor:.2f}x realtime)")
                if TORCH_AVAILABLE and self.transcription_engine.whisper_backend == "openai_whisper":
                    torch = FastLazyLoader.load('torch')
                    if torch.cuda.is_available():
                        vram = torch.cuda.memory_allocated() / 1024**3
                        logger.debug(f"  VRAM: {vram:.2f}GB")

            if not transcription or not transcription.text:
                return
            clean_text = transcription.text.strip()
            if (self.config.DUPLICATE_CHECK_ENABLED and
                self._is_duplicate_transcription(clean_text)):
                return
            self._last_transcription_text = clean_text
            if self.subtitle_mode and self.config.ENABLE_TIMED_TRANSCRIPTIONS:
                self._add_timed_transcription(transcription)
            transcription_callback(transcription)
            if (self.translation_engine and self._translation_active and
                hasattr(transcription, 'language')):
                detected_lang = transcription.language
                self._translate_and_send(
                    clean_text,
                    detected_lang,
                    translation_callback
                )
        except Exception as e:
            logger.warning(f"⚠️ Audio chunk processing error: {e}")
            if DEBUG_LEVEL >= 2:
                logger.exception("Stacktrace:")

    def _translate_and_send(self, text: str, source_lang: str,
                            translation_callback: Callable) -> None:
        try:
            translation = self.translation_engine.translate_text(text, source_lang)
            if translation:
                if self.subtitle_mode and self.config.ENABLE_TIMED_TRANSLATIONS:
                    self._add_timed_translation(translation)
                translation_callback(translation)
        except Exception as e:
            logger.warning(f"⚠️ Translation error: {e}")
            if DEBUG_LEVEL >= 2:
                logger.exception("Stacktrace:")

    def enhance_audio_quality(self, audio_data: bytes) -> bytes:
        if not self.config.AUDIO_ENHANCEMENT_ENABLED or len(audio_data) < 1600 or not NUMPY_AVAILABLE:
            return audio_data
        try:
            np = FastLazyLoader.load("numpy")
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_np**2))

            self._noisereduce_counter += 1
            if self.last_confidence < 0.6 and self._noisereduce_counter % 10 == 0:
                try:
                    import noisereduce as nr
                    audio_np = nr.reduce_noise(y=audio_np, sr=self.sample_rate, prop_decrease=0.8)
                    logger.debug(f"🔇 noisereduce angewendet (letzte Konfidenz: {self.last_confidence:.2f})")
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning(f"⚠️ noisereduce fehlgeschlagen: {e}")

            if rms < self.config.MIN_RMS_THRESHOLD:
                return audio_data
            if rms < self.config.TARGET_RMS:
                gain = min(self.config.MAX_GAIN, self.config.TARGET_RMS / max(rms, 1e-6))
                audio_np = audio_np * gain
            max_val = np.max(np.abs(audio_np))
            if max_val > self.config.CLIPPING_THRESHOLD:
                audio_np = audio_np * self.config.CLIPPING_THRESHOLD / max_val
            audio_np = audio_np - np.mean(audio_np)
            return (audio_np * 32767).astype(np.int16).tobytes()
        except Exception:
            return audio_data

    def _is_duplicate_transcription(self, current_text: str) -> bool:
        if not self.config.DUPLICATE_CHECK_ENABLED:
            return False
        with self._duplicate_lock:
            if not current_text or not current_text.strip():
                return True
            current_text = current_text.strip()
            if len(current_text) < self.config.MIN_TEXT_LENGTH:
                return True
            if current_text.isspace():
                return True
            if current_text == self._last_transcription_text:
                return True
            if current_text in self._recent_transcriptions:
                return True
            words = current_text.lower().split()
            if len(words) > 3:
                unique_words = len(set(words))
                unique_ratio = unique_words / len(words)
                if unique_ratio < self.config.MIN_UNIQUE_WORDS_RATIO:
                    return True
            self._recent_transcriptions.append(current_text)
        return False

    def _add_timed_transcription(self, result: ExcellenceTranscriptionResult) -> None:
        with self._subtitle_lock:
            if (hasattr(result, 'start') and result.start is not None and
                hasattr(result, 'end') and result.end is not None):
                self._timed_transcriptions.append(result)

    def _add_timed_translation(self, result: ExcellenceTranslationResult) -> None:
        with self._subtitle_lock:
            if (hasattr(result, 'start') and result.start is not None and
                hasattr(result, 'end') and result.end is not None):
                self._timed_translations.append(result)

    def set_engines(self, transcription_engine: ExcellenceTranscriptionEngine,
                    translation_engine: ExcellenceTranslationEngine,
                    plugin_manager: Optional[PluginManager] = None) -> None:
        self.transcription_engine = transcription_engine
        self.translation_engine = translation_engine
        self.plugin_manager = plugin_manager

    def enable_subtitle_mode(self, enabled: bool) -> None:
        self.subtitle_mode = enabled
        logger.info(f"🎬 Subtitle mode: {'ENABLED' if enabled else 'DISABLED'}")

    def get_status(self) -> Dict[str, Any]:
        return {
            '_processing': self._processing,
            '_stop_event_set': self._stop_event.is_set(),
            '_current_stream_id': self._current_stream_id,
            '_consecutive_empty_chunks': self._consecutive_empty_chunks,
            '_cleanup_done': self._cleanup_done,
            'config_type': self._get_config_type(),
            'chunk_size': self.chunk_size,
            'chunks_processed': self._chunk_counter,
            'total_bytes': self._total_bytes_processed,
            'empty_reads': self._empty_reads,
            'subtitle_mode': self.subtitle_mode,
            'translation_active': self._translation_active,
            'active_threads': threading.active_count(),
        }

    def _safe_kill_process(self, process: subprocess.Popen) -> None:
        if not process:
            return
        pid = process.pid
        logger.info(f"🛑 Terminating process {pid}...")
        try:
            if hasattr(self, 'ffmpeg_manager') and self.ffmpeg_manager:
                temp_id = f"kill_{pid}"
                self.ffmpeg_manager._register_process(temp_id, process, None, "terminate")
                self.ffmpeg_manager.stop_stream(temp_id)
                return
        except Exception:
            pass
        termination_steps = [
            (self._terminate_gracefully, 2.0, "graceful"),
            (self._terminate_forcefully, 1.0, "forceful"),
            (self._terminate_nuclear, 0.5, "nuclear")
        ]
        for terminate_method, timeout, method_name in termination_steps:
            if process.poll() is not None:
                break
            try:
                if terminate_method(process, pid, timeout):
                    logger.info(f"✅ Process {pid} terminated ({method_name})")
                    break
            except Exception as e:
                logger.warning(f"⚠️ {method_name} termination failed: {e}")
        self._cleanup_process_resources(process)

    def _terminate_gracefully(self, process: subprocess.Popen, pid: int, timeout: float) -> bool:
        if IS_WINDOWS:
            process.terminate()
        else:
            try:
                os.killpg(os.getpgid(pid), py_signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _terminate_forcefully(self, process: subprocess.Popen, pid: int, timeout: float) -> bool:
        if IS_WINDOWS:
            process.kill()
        else:
            try:
                os.killpg(os.getpgid(pid), py_signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _terminate_nuclear(self, process: subprocess.Popen, pid: int, timeout: float) -> bool:
        try:
            if IS_WINDOWS:
                cmd = ['taskkill', '/F', '/T', '/PID', str(pid)]
            else:
                cmd = ['pkill', '-9', '-P', str(pid)]
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if result.returncode == 0:
                return True
        except Exception:
            pass
        return False

    def _cleanup_process_resources(self, process: subprocess.Popen) -> None:
        for attr in ['stdout', 'stderr', 'stdin']:
            if hasattr(process, attr):
                pipe = getattr(process, attr)
                if pipe and not pipe.closed:
                    try:
                        pipe.close()
                    except Exception:
                        pass
        try:
            del process
            gc.collect()
        except Exception:
            pass

    def emergency_reset(self, force: bool = False) -> bool:
        logger.info(f"\n🚨 [EMERGENCY_RESET] force={force}")
        with self._resource_lock:
            with self._processing_lock:
                old_state = self._processing
                self._processing = False
            self._stop_event.set()
            self._current_stream_id = None
            self._consecutive_empty_chunks = 0
            if force:
                with self._subtitle_lock:
                    self._timed_transcriptions.clear()
                    self._timed_translations.clear()
                with self._duplicate_lock:
                    self._recent_transcriptions.clear()
        logger.info(f"✅ Reset completed: {old_state} -> {self._processing}")
        return True

    def _guaranteed_cleanup(self) -> None:
        logger.info("\n🧹 [GUARANTEED_CLEANUP]")
        with self._resource_lock:
            with self._processing_lock:
                self._processing = False
            self._current_stream_id = None
            self._consecutive_empty_chunks = 0
            self._empty_reads = 0
            self._chunk_counter = 0
            self._total_bytes_processed = 0
            self._cleanup_done = True
        time.sleep(0.05)
        logger.info("✅ Cleanup completed")

    def _platform_specific_health_check(self) -> List[str]:
        issues: List[str] = []
        if IS_WINDOWS:
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 85:
                    issues.append("High memory usage - consider closing other applications")
            except Exception:
                pass
        return issues

    def _flush_audio_buffer(self, transcription_callback: Callable, translation_callback: Callable) -> None:
        if not self._audio_buffer:
            return
        buffer_len = len(self._audio_buffer)
        logger.info(f"🧹 Flushing audio buffer ({buffer_len} bytes) at end of stream")
        if self.transcription_engine:
            self._process_audio_chunk(
                bytes(self._audio_buffer),
                transcription_callback,
                translation_callback
            )
        self._audio_buffer.clear()

    def _build_ffmpeg_command_enhanced(self, audio_url: str, detected_lang: Optional[str] = None) -> List[str]:
        return []

    def _test_audio_stream(self, audio_url: str) -> bool:
        logger.info(f"🔍 Testing audio stream: {audio_url[:60]}...")
        is_youtube = 'youtube.com' in audio_url.lower() or 'googlevideo.com' in audio_url
        is_hls = '.m3u8' in audio_url.lower() or 'manifest.googlevideo.com' in audio_url

        if is_hls:
            logger.info("🎯 HLS stream detected – skipping quick test (often too slow)")
            return True

        try:
            timeout = 8 if is_youtube else self.config.STREAM_TIMEOUT
            test_cmd = [
                'ffmpeg',
                '-i', audio_url,
                '-t', '2',
                '-f', 'null',
                '-',
                '-loglevel', 'error'
            ]
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False
            )
            if result.returncode == 0:
                logger.info("✅ Stream test successful")
                return True
            else:
                error_msg = result.stderr[:100] if result.stderr else "Unknown error"
                logger.error(f"❌ Stream test failed: {error_msg}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning(f"⏰ Stream test timeout after {timeout}s")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Stream test error: {e}")
            return True

    def dispose(self) -> None:
        logger.info("🧹 ExcellenceAudioProcessor: Starting dispose...")
        try:
            self._stop_event.set()
            with self._processing_lock:
                self._processing = False
            self._cleanup_done = True
            if hasattr(self, 'ffmpeg_manager') and self.ffmpeg_manager:
                try:
                    self.ffmpeg_manager.stop_all_streams()
                except Exception:
                    pass
            with self._subtitle_lock:
                self._timed_transcriptions.clear()
                self._timed_translations.clear()
            with self._duplicate_lock:
                self._recent_transcriptions.clear()
                self._last_transcription_text = ""
            gc.collect()
            logger.info("✅ ExcellenceAudioProcessor disposed")
        except Exception as e:
            logger.warning(f"⚠️ ExcellenceAudioProcessor dispose error: {e}")

    def stop_processing(self, wait: bool = False, timeout: float = 5.0) -> bool:
        logger.info("🛑 ExcellenceAudioProcessor: Stopping processing...")
        self._stop_event.set()
        with self._processing_lock:
            self._processing = False
        if self._current_stream_id:
            logger.info(f"📛 Stream {self._current_stream_id} stopped by user")
            if self.ffmpeg_manager:
                self.ffmpeg_manager.stop_stream(self._current_stream_id)
        if wait:
            return self._process_finished.wait(timeout)
        return True


# -----------------------------------------------------------------------------
# DARK CONTEXT MENUS (unverändert)
# -----------------------------------------------------------------------------
class DarkContextMenu:
    def __init__(self, text_widget: tk.Text) -> None:
        self.text_widget = text_widget
        self.menu = tk.Menu(
            text_widget,
            tearoff=0,
            bg=CURRENT_THEME.BG_TERTIARY,
            fg=CURRENT_THEME.TEXT_PRIMARY,
            activebackground=CURRENT_THEME.BG_HOVER,
            activeforeground=CURRENT_THEME.TEXT_ACCENT,
            borderwidth=1,
            relief="solid",
        )
        self.menu.add_command(label="Copy", command=self.copy_text)
        self.menu.add_command(label="Select All", command=self.select_all)
        self.menu.add_separator()
        self.menu.add_command(label="Delete", command=self.clear_text)
        text_widget.bind("<Button-3>", self.show_menu)

    def show_menu(self, event: tk.Event) -> None:
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def copy_text(self) -> None:
        try:
            selected_text = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.text_widget.clipboard_clear()
            self.text_widget.clipboard_append(selected_text)
        except tk.TclError:
            pass

    def select_all(self) -> None:
        self.text_widget.tag_add(tk.SEL, "1.0", tk.END)
        self.text_widget.mark_set(tk.INSERT, "1.0")
        self.text_widget.see(tk.INSERT)

    def clear_text(self) -> None:
        self.text_widget.delete("1.0", tk.END)


class DarkEntryContextMenu:
    def __init__(self, entry_widget: tk.Entry) -> None:
        self.entry_widget = entry_widget
        self.menu = tk.Menu(
            entry_widget,
            tearoff=0,
            bg=CURRENT_THEME.BG_TERTIARY,
            fg=CURRENT_THEME.TEXT_PRIMARY,
            activebackground=CURRENT_THEME.BG_HOVER,
            activeforeground=CURRENT_THEME.TEXT_ACCENT,
            borderwidth=1,
            relief="solid",
        )
        self.menu.add_command(label="Cut", command=self.cut_text)
        self.menu.add_command(label="Copy", command=self.copy_text)
        self.menu.add_command(label="Paste", command=self.paste_text)
        self.menu.add_separator()
        self.menu.add_command(label="Select All", command=self.select_all)
        self.menu.add_command(label="Delete", command=self.delete_text)
        entry_widget.bind("<Button-3>", self.show_menu)

    def show_menu(self, event: tk.Event) -> None:
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def cut_text(self) -> None:
        self.entry_widget.event_generate("<<Cut>>")

    def copy_text(self) -> None:
        self.entry_widget.event_generate("<<Copy>>")

    def paste_text(self) -> None:
        self.entry_widget.event_generate("<<Paste>>")

    def select_all(self) -> None:
        self.entry_widget.select_range(0, "end")
        self.entry_widget.icursor("end")

    def delete_text(self) -> None:
        self.entry_widget.delete(0, "end")


# -----------------------------------------------------------------------------
# EXPORT MANAGER (unverändert)
# -----------------------------------------------------------------------------
class ExportManager:
    def __init__(self) -> None:
        self.supported_formats = ["txt", "srt", "vtt", "json", "docx"]

    def export_subtitles(
        self,
        transcript_data: List[ExcellenceTranscriptionResult],
        translation_data: Optional[List[ExcellenceTranslationResult]] = None,
        format: str = "srt",
        filename: Optional[str] = None,
    ) -> Union[bool, str]:
        try:
            timed_transcripts = [
                t for t in transcript_data
                if hasattr(t, "start") and t.start is not None and hasattr(t, "end") and t.end is not None
            ]
            if not timed_transcripts:
                raise ExcellenceError("No timed transcriptions available")
            if format.lower() == "srt":
                content = self.generate_srt_content(timed_transcripts, translation_data)
            elif format.lower() == "vtt":
                content = self.generate_vtt_content(timed_transcripts, translation_data)
            else:
                raise ExcellenceError(f"Unsupported format: {format}")
            if filename:
                with open(filename, "w", encoding="utf-8-sig") as f:
                    f.write(content)
                return True
            else:
                return content
        except Exception as e:
            raise ExcellenceError(f"Subtitle export failed: {e}")

    def generate_srt_content(
        self,
        transcript_data: List[ExcellenceTranscriptionResult],
        translation_data: Optional[List[ExcellenceTranslationResult]] = None,
    ) -> str:
        srt_content = ""
        for i, segment in enumerate(transcript_data):
            start_time = self._format_timestamp_srt(segment.start or 0.0)
            end_time = self._format_timestamp_srt(segment.end or 0.0)
            display_text = segment.text
            if translation_data and i < len(translation_data):
                display_text = translation_data[i].translated
            srt_content += f"{i + 1}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{display_text}\n\n"
        return srt_content

    def generate_vtt_content(
        self,
        transcript_data: List[ExcellenceTranscriptionResult],
        translation_data: Optional[List[ExcellenceTranslationResult]] = None,
    ) -> str:
        vtt_content = "WEBVTT\n\n"
        for i, segment in enumerate(transcript_data):
            start_time = self._format_timestamp_vtt(segment.start or 0.0)
            end_time = self._format_timestamp_vtt(segment.end or 0.0)
            display_text = segment.text
            if translation_data and i < len(translation_data):
                display_text = translation_data[i].translated
            vtt_content += f"{start_time} --> {end_time}\n"
            vtt_content += f"{display_text}\n\n"
        return vtt_content

    def _format_timestamp_srt(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

    def _format_timestamp_vtt(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{milliseconds:03d}"

    def export_json(
        self,
        transcript_data: List[ExcellenceTranscriptionResult],
        translation_data: List[ExcellenceTranslationResult],
        filename: str,
    ) -> bool:
        try:
            export_data = {
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "total_segments": len(transcript_data),
                    "version": "4.1.3",
                },
                "transcripts": [
                    {
                        "text": segment.text,
                        "confidence": segment.confidence,
                        "language": segment.language,
                        "timestamp": segment.timestamp,
                        "start_time": getattr(segment, "start", None),
                        "end_time": getattr(segment, "end", None),
                    }
                    for segment in transcript_data
                ],
                "translations": [
                    {
                        "original": trans.original,
                        "translated": trans.translated,
                        "source_lang": trans.source_lang,
                        "target_lang": trans.target_lang,
                        "timestamp": trans.timestamp,
                    }
                    for trans in translation_data
                ] if translation_data else [],
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            raise ExcellenceError(f"JSON export failed: {e}")

    def export_docx(self, transcript_data: List[ExcellenceTranscriptionResult], filename: str) -> bool:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("TRANSCRIPT EXPORT\n")
                f.write("================\n\n")
                for i, segment in enumerate(transcript_data, 1):
                    timestamp = datetime.fromtimestamp(segment.timestamp).strftime("%H:%M:%S")
                    f.write(f"[{timestamp}] {segment.text}\n\n")
            return True
        except Exception as e:
            raise ExcellenceError(f"DOCX export failed: {e}")


# -----------------------------------------------------------------------------
# APP SETTINGS (unverändert, ergänzt um Cookie-Einstellungen)
# -----------------------------------------------------------------------------
@dataclass
class AppSettings:
    last_url: str = ""
    default_model: str = "medium"
    default_language: str = "de"
    layout_mode: str = "vertical"
    recent_urls: List[str] = None
    enable_plugins: bool = True
    export_format: str = "txt"
    auto_save_on_completion: bool = False
    theme: str = "dark"
    use_browser_cookies: bool = True      # NEW
    cookies_notice_shown: bool = False    # NEW

    def __post_init__(self) -> None:
        if self.recent_urls is None:
            self.recent_urls = []

    @classmethod
    def load_from_file(cls, filename: str = "dragon_settings.json") -> 'AppSettings':
        try:
            config_dir = PlatformUtils.get_platform_config_dir()
            file_path = config_dir / filename
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return cls(**data)
        except Exception:
            pass
        return cls()

    def save_to_file(self, filename: str = "dragon_settings.json") -> None:
        try:
            config_dir = PlatformUtils.get_platform_config_dir()
            file_path = config_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def add_recent_url(self, url: str) -> None:
        if url in self.recent_urls:
            self.recent_urls.remove(url)
        self.recent_urls.insert(0, url)
        self.recent_urls = self.recent_urls[:10]
        self.save_to_file()


# -----------------------------------------------------------------------------
# RESOURCE MANAGER (unverändert)
# -----------------------------------------------------------------------------
class ResourceManager:
    def __init__(self) -> None:
        self.processes: List[subprocess.Popen] = []
        self.threads: List[threading.Thread] = []
        self.temp_files: List[str] = []
        self.cleanup_done = False
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()

    def register_process(self, process: subprocess.Popen) -> None:
        with self._lock:
            if process and process not in self.processes:
                self.processes.append(process)

    def register_thread(self, thread: threading.Thread) -> None:
        with self._lock:
            if thread and thread not in self.threads and thread.is_alive():
                self.threads.append(thread)

    def register_temp_file(self, file_path: str) -> None:
        with self._lock:
            if file_path and file_path not in self.temp_files:
                self.temp_files.append(file_path)

    def cleanup(self) -> None:
        if self.cleanup_done:
            return
        self._shutdown_event.set()
        with self._lock:
            cleanup_timeout = 5.0
            start_time = time.time()
            for process in self.processes[:]:
                try:
                    if process and hasattr(process, "poll"):
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=1.0)
                            except (subprocess.TimeoutExpired, AttributeError):
                                try:
                                    process.kill()
                                    process.wait(timeout=0.5)
                                except Exception:
                                    pass
                except Exception:
                    pass
                finally:
                    if process in self.processes:
                        self.processes.remove(process)
                if time.time() - start_time > cleanup_timeout:
                    break
            for thread in self.threads[:]:
                try:
                    if thread and thread.is_alive():
                        thread.join(timeout=1.0)
                except Exception:
                    pass
                finally:
                    if thread in self.threads:
                        self.threads.remove(thread)
                if time.time() - start_time > cleanup_timeout:
                    break
            for temp_file in self.temp_files[:]:
                try:
                    if os.path.exists(temp_file):
                        for attempt in range(2):
                            try:
                                os.unlink(temp_file)
                                break
                            except PermissionError:
                                if attempt < 1:
                                    time.sleep(0.1)
                                    continue
                            except Exception:
                                if attempt < 1:
                                    time.sleep(0.1)
                except (OSError, PermissionError):
                    pass
                finally:
                    if temp_file in self.temp_files:
                        self.temp_files.remove(temp_file)
            try:
                if TORCH_AVAILABLE:
                    torch = FastLazyLoader.load("torch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            self.cleanup_done = True

    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()


# -----------------------------------------------------------------------------
# OLLAMA SUMMARIZER (unverändert)
# -----------------------------------------------------------------------------
class OllamaSummarizer:
    def __init__(self, parent: Any, model: str = "llama3", host: str = "http://localhost:11434") -> None:
        self.parent = parent
        self.model = model
        self.host = host
        self.available = OLLAMA_AVAILABLE

    def get_available_models(self) -> List[str]:
        if not self.available:
            return []
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            pass
        return []

    def summarize(self, text: str, prompt: str, temperature: float,
                  callback: Callable[[str], None],
                  error_callback: Callable[[str], None],
                  complete_callback: Optional[Callable[[], None]] = None) -> None:
        if not self.available:
            error_callback("Ollama nicht verfügbar (requests nicht installiert)")
            return
        if not text or not text.strip():
            error_callback("Kein Text zum Zusammenfassen")
            return

        def worker() -> None:
            try:
                import requests
                full_prompt = f"{prompt}\n\n{text}"
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature
                    }
                }
                response = requests.post(f"{self.host}/api/generate", json=payload, stream=True, timeout=120)
                if response.status_code == 200:
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    chunk = data['response']
                                    full_response += chunk
                                    callback(chunk)
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    if not full_response:
                        error_callback("Leere Antwort von Ollama")
                    else:
                        if complete_callback:
                            complete_callback()
                else:
                    error_callback(f"Ollama Fehler {response.status_code}")
            except requests.exceptions.Timeout:
                error_callback("Ollama Timeout – Server nicht erreichbar?")
            except requests.exceptions.ConnectionError:
                error_callback("Ollama nicht erreichbar (läuft der Server?)")
            except Exception as e:
                error_callback(f"Fehler: {str(e)}")

        threading.Thread(target=worker, daemon=True).start()


class SummarizeDialog:
    def __init__(self, parent: Any, text: str, gui_ref: Any) -> None:
        self.parent = parent
        self.text = text
        self.gui = gui_ref
        self.dialog: Optional[tk.Toplevel] = None
        self.summarizer = OllamaSummarizer(parent)
        self.create_dialog()

    def create_dialog(self) -> None:
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("🐉 Zusammenfassung mit Ollama")
        self.dialog.geometry("700x600")
        self.dialog.configure(bg=CURRENT_THEME.BG_PRIMARY)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        main = tk.Frame(self.dialog, bg=CURRENT_THEME.BG_PRIMARY, padx=15, pady=15)
        main.pack(fill="both", expand=True)

        model_frame = tk.Frame(main, bg=CURRENT_THEME.BG_PRIMARY)
        model_frame.pack(fill="x", pady=5)
        tk.Label(model_frame, text="Modell:", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY).pack(side="left")
        self.model_var = tk.StringVar(value="llama3")
        self.model_combo = ttk.Combobox(
            model_frame, textvariable=self.model_var,
            values=self.summarizer.get_available_models() or ["llama3", "mistral", "gemma"],
            width=20, state="readonly"
        )
        self.model_combo.pack(side="left", padx=10)

        tk.Label(model_frame, text="Temperatur:", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY).pack(side="left", padx=(20,5))
        self.temp_var = tk.DoubleVar(value=0.7)
        temp_scale = tk.Scale(
            model_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self.temp_var, length=150, bg=CURRENT_THEME.BG_PRIMARY,
            fg=CURRENT_THEME.TEXT_PRIMARY, highlightbackground=CURRENT_THEME.BG_PRIMARY
        )
        temp_scale.pack(side="left")

        tk.Label(main, text="Prompt (optional):", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY).pack(anchor="w", pady=(10,2))
        self.prompt_text = scrolledtext.ScrolledText(
            main, height=4, bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.MONOSPACE, wrap=tk.WORD
        )
        self.prompt_text.pack(fill="x", pady=(0,10))
        self.prompt_text.insert("1.0", "Fasse den folgenden Text kurz und präzise auf Deutsch zusammen:")

        tk.Label(main, text="Zusammenfassung:", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY).pack(anchor="w")
        self.summary_text = scrolledtext.ScrolledText(
            main, height=12, bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.MONOSPACE, wrap=tk.WORD
        )
        self.summary_text.pack(fill="both", expand=True, pady=10)
        DarkContextMenu(self.summary_text)

        btn_frame = tk.Frame(main, bg=CURRENT_THEME.BG_PRIMARY)
        btn_frame.pack(fill="x")
        self.summarize_btn = tk.Button(
            btn_frame, text="🤖 Zusammenfassen", command=self.start_summarize,
            bg=CURRENT_THEME.DRAGON_GREEN, fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.BUTTON, padx=20
        )
        self.summarize_btn.pack(side="left", padx=5)
        self.copy_btn = tk.Button(
            btn_frame, text="📋 Kopieren", command=self.copy_summary,
            bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.BUTTON, padx=20
        )
        self.copy_btn.pack(side="left", padx=5)
        self.close_btn = tk.Button(
            btn_frame, text="Schließen", command=self.dialog.destroy,
            bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY
        )
        self.close_btn.pack(side="right", padx=5)

        self.status_label = tk.Label(main, text="", bg=CURRENT_THEME.BG_PRIMARY,
                                      fg=CURRENT_THEME.TEXT_SECONDARY)
        self.status_label.pack(pady=5)

        self.full_summary = ""

    def start_summarize(self) -> None:
        model = self.model_var.get().strip()
        if not model:
            model = "llama3"
        self.summarizer.model = model
        self.summarizer.host = "http://localhost:11434"
        prompt = self.prompt_text.get("1.0", "end-1c").strip()
        if not prompt:
            prompt = "Fasse den folgenden Text kurz und präzise auf Deutsch zusammen:"
        temp = self.temp_var.get()

        self.summarize_btn.config(state="disabled", text="⏳ Warte...")
        self.status_label.config(text="Sende Anfrage an Ollama...")
        self.summary_text.delete("1.0", "end")
        self.full_summary = ""

        def on_complete() -> None:
            if self.dialog and self.dialog.winfo_exists():
                self.dialog.after(0, self._reset_ui)

        self.summarizer.summarize(
            self.text, prompt, temp,
            callback=self.on_chunk,
            error_callback=self.on_error,
            complete_callback=on_complete
        )

    def _reset_ui(self) -> None:
        if self.dialog and self.dialog.winfo_exists():
            self.summarize_btn.config(state="normal", text="🤖 Zusammenfassen")
            self.status_label.config(text="✅ Zusammenfassung abgeschlossen")

    def on_chunk(self, chunk: str) -> None:
        def update() -> None:
            if self.dialog and self.dialog.winfo_exists():
                self.summary_text.insert("end", chunk)
                self.summary_text.see("end")
                self.full_summary += chunk
        if self.dialog and self.dialog.winfo_exists():
            self.dialog.after(0, update)

    def on_error(self, error: str) -> None:
        def update() -> None:
            if self.dialog and self.dialog.winfo_exists():
                self.summary_text.insert("1.0", f"Fehler: {error}")
                self.summarize_btn.config(state="normal", text="🤖 Zusammenfassen")
                self.status_label.config(text="❌ Fehler")
        if self.dialog and self.dialog.winfo_exists():
            self.dialog.after(0, update)

    def copy_summary(self) -> None:
        if self.full_summary:
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(self.full_summary)
            self.status_label.config(text="✅ In Zwischenablage kopiert")
        else:
            self.status_label.config(text="⚠️ Keine Zusammenfassung vorhanden")


# -----------------------------------------------------------------------------
# TOOLTIP (unverändert)
# -----------------------------------------------------------------------------
class ToolTip:
    def __init__(self, widget: tk.Widget, text: str, delay: int = 500) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window: Optional[tk.Toplevel] = None
        self.after_id: Optional[str] = None
        widget.bind('<Enter>', self.enter)
        widget.bind('<Leave>', self.leave)
        widget.bind('<ButtonPress>', self.leave)

    def enter(self, event: Optional[tk.Event] = None) -> None:
        self.schedule()

    def leave(self, event: Optional[tk.Event] = None) -> None:
        self.unschedule()
        self.hide_tip()

    def schedule(self) -> None:
        self.unschedule()
        self.after_id = self.widget.after(self.delay, self.show_tip)

    def unschedule(self) -> None:
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def show_tip(self) -> None:
        if self.tip_window or not self.text:
            return
        x, y = self.widget.winfo_pointerxy()
        x += 10
        y += 10
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief="solid",
                         borderwidth=1, font=("Segoe UI", 9))
        label.pack()

    def hide_tip(self) -> None:
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# -----------------------------------------------------------------------------
# LAYOUT MANAGER (unverändert)
# -----------------------------------------------------------------------------
class WhisperLayoutManager:
    def __init__(self, gui_ref: Any) -> None:
        self.gui_ref = gui_ref
        self.root = gui_ref.root
        self._batch_timer_id: Optional[str] = None
        try:
            self.gui_ref._text_update_queue = queue.Queue(maxsize=150)
            self.gui_ref.gui_queue = queue.Queue(maxsize=200)
            logger.info("✅ Queues erfolgreich erstellt")
        except Exception as e:
            logger.warning(f"⚠️ Queue-Erstellung fehlgeschlagen: {e}")
            class DummyQueue:
                def __init__(self, maxsize: int = 0) -> None:
                    self.maxsize = maxsize
                    self._items: List[Any] = []
                    self._lock = threading.Lock()
                    self.Empty = queue.Empty  # neu: echte Empty-Exception
                def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
                    with self._lock:
                        self._items.append(item)
                        if self.maxsize > 0 and len(self._items) > self.maxsize:
                            self._items.pop(0)
                def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
                    with self._lock:
                        if self._items:
                            return self._items.pop(0)
                        raise self.Empty()
                def empty(self) -> bool:
                    with self._lock:
                        return len(self._items) == 0
                def qsize(self) -> int:
                    with self._lock:
                        return len(self._items)
                def task_done(self) -> None:
                    pass
                def get_nowait(self) -> Any:
                    return self.get(block=False)
                def join(self) -> None:
                    pass
            self.gui_ref._text_update_queue = DummyQueue(maxsize=150)
            self.gui_ref.gui_queue = DummyQueue(maxsize=200)
            logger.warning("⚠️ Verwende Dummy-Queues (eingeschränkte Funktionalität)")

    def setup_gui(self) -> None:
        self.root.configure(bg=self.gui_ref.current_theme.BG_PRIMARY)
        self.root.title("🐉 Dragon Whisperer - Plattformunabhängig")
        self.root.geometry("900x700")
        self.root.minsize(850, 600)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=10)
        self.root.grid_rowconfigure(4, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.setup_dark_styles()
        self.center_window()
        self.root.protocol("WM_DELETE_WINDOW", self.gui_ref._safe_exit_dialog)
        self.create_layout()
        self.root.after(100, self.start_batch_updates)

    def setup_dark_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.TCombobox",
            fieldbackground=self.gui_ref.current_theme.COMBO_BG,
            background=self.gui_ref.current_theme.COMBO_BG,
            foreground=self.gui_ref.current_theme.COMBO_FG,
            selectbackground=self.gui_ref.current_theme.COMBO_SELECTION,
            selectforeground=self.gui_ref.current_theme.TEXT_PRIMARY,
            insertcolor=self.gui_ref.current_theme.TEXT_PRIMARY,
            borderwidth=1,
            relief="flat",
            arrowsize=12,
            padding=5,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[
                ("readonly", self.gui_ref.current_theme.COMBO_BG),
                ("active", self.gui_ref.current_theme.BG_HOVER),
            ],
            background=[
                ("readonly", self.gui_ref.current_theme.COMBO_BG),
                ("active", self.gui_ref.current_theme.BG_HOVER),
            ],
            foreground=[
                ("readonly", self.gui_ref.current_theme.COMBO_FG),
                ("active", self.gui_ref.current_theme.TEXT_PRIMARY),
            ],
        )
        style.configure(
            "Dark.Horizontal.TProgressbar",
            background=self.gui_ref.current_theme.SUCCESS,
            troughcolor=self.gui_ref.current_theme.BG_TERTIARY,
            bordercolor=self.gui_ref.current_theme.BORDER,
        )
        self.root.option_add("*TCombobox*Listbox.background", self.gui_ref.current_theme.COMBO_BG)
        self.root.option_add("*TCombobox*Listbox.foreground", self.gui_ref.current_theme.COMBO_FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.gui_ref.current_theme.COMBO_SELECTION)
        self.root.option_add("*TCombobox*Listbox.selectForeground", self.gui_ref.current_theme.TEXT_PRIMARY)

    def center_window(self) -> None:
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")

    def create_layout(self) -> None:
        header_frame = tk.Frame(self.root, bg=self.gui_ref.current_theme.BG_PRIMARY, height=35)
        header_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=8)
        header_frame.grid_propagate(False)
        title_label = tk.Label(
            header_frame,
            text="🐉 Dragon Whisperer - Livestream Transcription & Translation",
            font=DragonFonts.TITLE,
            bg=self.gui_ref.current_theme.BG_PRIMARY,
            fg=self.gui_ref.current_theme.DRAGON_GREEN,
        )
        title_label.pack(side="left")
        self.gui_ref.status_label = tk.Label(
            header_frame,
            text="✅ READY",
            font=DragonFonts.PRIMARY,
            bg=self.gui_ref.current_theme.BG_PRIMARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
        )
        self.gui_ref.status_label.pack(side="right")
        self.create_stream_info_display()
        self.gui_ref.stream_info_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=3)
        input_frame = tk.Frame(self.root, bg=self.gui_ref.current_theme.BG_PRIMARY)
        input_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=3)
        url_frame = tk.Frame(input_frame, bg=self.gui_ref.current_theme.BG_PRIMARY)
        url_frame.pack(fill="x", pady=2)
        tk.Label(
            url_frame,
            text="URL:",
            bg=self.gui_ref.current_theme.BG_PRIMARY,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            font=DragonFonts.PRIMARY,
        ).pack(side="left")
        self.gui_ref.url_entry = tk.Entry(
            url_frame,
            font=DragonFonts.PRIMARY,
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            insertbackground=self.gui_ref.current_theme.TEXT_PRIMARY,
            selectbackground=self.gui_ref.current_theme.COMBO_SELECTION,
            width=60,
        )
        self.gui_ref.url_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        self.gui_ref.url_entry.insert(0, self.gui_ref.settings.last_url if self.gui_ref.settings.last_url else "")
        DarkEntryContextMenu(self.gui_ref.url_entry)
        self.gui_ref.language_info_label = tk.Label(
            url_frame,
            text="",
            font=DragonFonts.PRIMARY,
            bg=self.gui_ref.current_theme.BG_PRIMARY,
            fg=self.gui_ref.current_theme.TEXT_ACCENT,
        )
        self.gui_ref.language_info_label.pack(side="right", padx=(5, 0))
        self.create_compact_control_panel(input_frame)
        self.setup_status_bar()
        self.gui_ref.status_bar_frame.grid(row=4, column=0, sticky="ew", pady=(2, 0))
        self.create_text_areas()
        self.gui_ref.text_container.grid(row=3, column=0, sticky="nsew", padx=12, pady=8)
        self.gui_ref.url_entry.bind("<KeyRelease>", self.gui_ref.on_url_change)
        self.gui_ref.url_entry.bind("<FocusOut>", self.gui_ref.on_url_change)

    def create_stream_info_display(self) -> None:
        self.gui_ref.stream_info_frame = tk.Frame(self.root, bg=self.gui_ref.current_theme.BG_SECONDARY, height=50)
        self.gui_ref.stream_info_frame.grid_propagate(True)
        self.gui_ref.stream_title_label = tk.Label(
            self.gui_ref.stream_info_frame,
            text="📡 No active stream",
            font=DragonFonts.SUBTITLE,
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_ACCENT,
            wraplength=700,
            justify="left",
        )
        self.gui_ref.stream_title_label.pack(fill="x", padx=8, pady=(6, 2))
        self.gui_ref.stream_details_label = tk.Label(
            self.gui_ref.stream_info_frame,
            text="Ready to connect...",
            font=DragonFonts.PRIMARY,
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
            justify="left",
        )
        self.gui_ref.stream_details_label.pack(fill="x", padx=8, pady=(2, 6))

    def create_compact_control_panel(self, parent: tk.Frame) -> None:
        control_frame = tk.Frame(parent, bg=self.gui_ref.current_theme.BG_PRIMARY)
        control_frame.pack(fill="x", pady=8)
        left_controls = tk.Frame(control_frame, bg=self.gui_ref.current_theme.BG_PRIMARY)
        left_controls.pack(side="left")
        action_buttons = [
            ("📁", self.gui_ref.select_file_dark, "Datei auswählen"),
            ("📋", self.gui_ref.paste_url, "URL aus Zwischenablage einfügen"),
        ]
        for icon, command, tooltip in action_buttons:
            btn = tk.Button(
                left_controls,
                text=icon,
                command=command,
                bg=self.gui_ref.current_theme.BG_TERTIARY,
                fg=self.gui_ref.current_theme.TEXT_PRIMARY,
                relief="flat",
                bd=0,
                font=("Segoe UI", 9),
                cursor="hand2",
            )
            btn.pack(side="left", padx=1)
            ToolTip(btn, tooltip)

        self.gui_ref.layout_btn = tk.Button(
            left_controls,
            text="🔄",
            command=self.gui_ref.toggle_layout,
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            relief="flat",
            bd=0,
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        self.gui_ref.layout_btn.pack(side="left", padx=5)
        ToolTip(self.gui_ref.layout_btn, "Layout umschalten (vertikal/horizontal)")

        center_controls = tk.Frame(control_frame, bg=self.gui_ref.current_theme.BG_PRIMARY)
        center_controls.pack(side="left", padx=15)
        model_frame = tk.Frame(center_controls, bg=self.gui_ref.current_theme.BG_PRIMARY)
        model_frame.pack(side="left", padx=5)
        tk.Label(
            model_frame,
            text="Model:",
            bg=self.gui_ref.current_theme.BG_PRIMARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
            font=DragonFonts.PRIMARY,
        ).pack(side="left")
        self.gui_ref.model_var = tk.StringVar(value=self.gui_ref.settings.default_model)
        self.gui_ref.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.gui_ref.model_var,
            values=WHISPER_MODELS,
            width=10,
            style="Dark.TCombobox",
            state="readonly",
        )
        self.gui_ref.model_combo.pack(side="left", padx=3)
        ToolTip(self.gui_ref.model_combo, "Whisper-Modell auswählen (größer = genauer, aber langsamer)")

        if getattr(self.gui_ref, 'demo_mode', False):
            self.gui_ref.model_combo.config(state="disabled")
            self.gui_ref.model_var.set("dummy (Demo)")

        lang_frame = tk.Frame(center_controls, bg=self.gui_ref.current_theme.BG_PRIMARY)
        lang_frame.pack(side="left", padx=5)
        tk.Label(
            lang_frame,
            text="Translate:",
            bg=self.gui_ref.current_theme.BG_PRIMARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
            font=DragonFonts.PRIMARY,
        ).pack(side="left")
        self.gui_ref.lang_var = tk.StringVar()
        language_groups = {
            "Common": ["German", "English", "French", "Spanish", "Italian"],
            "Asian": ["Japanese", "Chinese", "Korean", "Vietnamese", "Thai"],
            "More": [
                name
                for name, code in SORTED_LANGUAGES
                if name
                not in [
                    "German",
                    "English",
                    "French",
                    "Spanish",
                    "Italian",
                    "Japanese",
                    "Chinese",
                    "Korean",
                    "Vietnamese",
                    "Thai",
                ]
            ],
        }
        all_languages: List[str] = []
        for group_name, languages in language_groups.items():
            if languages:
                all_languages.append(f"--- {group_name} ---")
                all_languages.extend(languages)
        self.gui_ref.lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.gui_ref.lang_var,
            values=all_languages,
            width=14,
            style="Dark.TCombobox",
            state="readonly",
        )
        self.gui_ref.lang_combo.pack(side="left", padx=3)
        ToolTip(self.gui_ref.lang_combo, "Zielsprache für Übersetzung")
        default_lang_name = SUPPORTED_LANGUAGES.get(self.gui_ref.settings.default_language, "German")
        self.gui_ref.lang_var.set(default_lang_name)
        self.gui_ref.lang_combo.bind("<<ComboboxSelected>>", self.gui_ref.on_language_change)

        right_controls = tk.Frame(control_frame, bg=self.gui_ref.current_theme.BG_PRIMARY)
        right_controls.pack(side="right")
        self.gui_ref.start_button = tk.Button(
            right_controls,
            text="🚀 START",
            command=self.gui_ref._on_start_click,
            bg=self.gui_ref.current_theme.SUCCESS,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=20,
        )
        self.gui_ref.start_button.pack(side="left", padx=2)
        ToolTip(self.gui_ref.start_button, "Transkription/Übersetzung starten")

        self.gui_ref.stop_button = tk.Button(
            right_controls,
            text="⏹️ STOP",
            command=self.gui_ref.controller.stop_processing,
            bg=self.gui_ref.current_theme.ERROR,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            state="disabled",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=20,
        )
        self.gui_ref.stop_button.pack(side="left", padx=2)
        ToolTip(self.gui_ref.stop_button, "Laufende Verarbeitung stoppen")

        self.gui_ref.translate_toggle = tk.BooleanVar(value=True)
        translate_btn = tk.Checkbutton(
            right_controls,
            text="Translate",
            variable=self.gui_ref.translate_toggle,
            command=self.gui_ref.toggle_translation,
            bg=self.gui_ref.current_theme.BG_PRIMARY,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            selectcolor=self.gui_ref.current_theme.BG_TERTIARY,
            activebackground=self.gui_ref.current_theme.BG_PRIMARY,
            activeforeground=self.gui_ref.current_theme.TEXT_PRIMARY,
            font=DragonFonts.PRIMARY,
        )
        translate_btn.pack(side="left", padx=5)
        ToolTip(translate_btn, "Übersetzung ein-/ausschalten")

        self.gui_ref.subtitle_btn = tk.Button(
            right_controls,
            text="🎬",
            command=self.gui_ref.toggle_subtitle_mode,
            bg=self.gui_ref.current_theme.SUBTITLE_INACTIVE,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            relief="flat",
            bd=0,
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        self.gui_ref.subtitle_btn.pack(side="left", padx=5)
        ToolTip(self.gui_ref.subtitle_btn, "Untertitel-Modus (mit Zeitstempeln)")

        self.gui_ref.model_combo.bind("<<ComboboxSelected>>", self.gui_ref.on_model_change)

    def create_text_areas(self) -> Tuple[Optional[scrolledtext.ScrolledText], Optional[scrolledtext.ScrolledText]]:
        layout_changed = False
        current_layout = getattr(self.gui_ref, '_current_layout', None)
        if current_layout != self.gui_ref.layout_mode:
            layout_changed = True
            logger.info(f"🔄 Layout change detected: {current_layout} → {self.gui_ref.layout_mode}")
        if hasattr(self.gui_ref, 'text_container') and layout_changed:
            try:
                if self.gui_ref.text_container.winfo_exists():
                    logger.info("   🗑️ Destroying old container for layout change")
                    self.gui_ref.text_container.destroy()
                    time.sleep(0.02)
            except tk.TclError:
                pass
            except Exception as e:
                logger.warning(f"   ⚠️ Container destroy warning: {e}")
        if layout_changed or not hasattr(self.gui_ref, 'text_container'):
            self.gui_ref.text_container = tk.Frame(self.root, bg=self.gui_ref.current_theme.BG_PRIMARY)
            self.gui_ref._current_layout = self.gui_ref.layout_mode
            logger.info(f"   ✅ New container created for {self.gui_ref.layout_mode} layout")
        if self.gui_ref.layout_mode == "horizontal":
            self.create_horizontal_layout()
        else:
            self.create_vertical_layout()
        self.gui_ref.text_container.grid(row=3, column=0, sticky="nsew", padx=12, pady=8)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.update_idletasks()
        return (
            getattr(self.gui_ref, 'transcript_text', None),
            getattr(self.gui_ref, 'translation_text', None)
        )

    def create_vertical_layout(self) -> None:
        main_frame = tk.LabelFrame(
            self.gui_ref.text_container,
            text="Live Transkription & Übersetzung",
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            font=DragonFonts.SUBTITLE,
            padx=8,
            pady=8,
        )
        main_frame.pack(fill="both", expand=True)
        trans_frame = tk.Frame(main_frame, bg=self.gui_ref.current_theme.BG_SECONDARY)
        trans_frame.pack(fill="x", pady=(0, 3))
        trans_header = tk.Frame(trans_frame, bg=self.gui_ref.current_theme.BG_SECONDARY)
        trans_header.pack(fill="x")
        tk.Label(
            trans_header,
            text="🎤 Transkription:",
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_ACCENT,
            font=DragonFonts.SUBTITLE,
        ).pack(side="left")
        self.gui_ref.transcript_scroll_var = tk.BooleanVar(value=True)
        scroll_cb = tk.Checkbutton(
            trans_header,
            variable=self.gui_ref.transcript_scroll_var,
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            activebackground=self.gui_ref.current_theme.BG_SECONDARY,
            selectcolor=self.gui_ref.current_theme.CHECKBOX_ACTIVE,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
        )
        scroll_cb.pack(side="right", padx=3)
        tk.Label(
            trans_header,
            text="Auto-Scroll",
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
            font=("Segoe UI", 7),
        ).pack(side="right", padx=1)
        self.gui_ref.transcript_text = self.create_text_widget(main_frame, height=6)
        transla_frame = tk.Frame(main_frame, bg=self.gui_ref.current_theme.BG_SECONDARY)
        transla_frame.pack(fill="x", pady=(8, 0))
        transla_header = tk.Frame(transla_frame, bg=self.gui_ref.current_theme.BG_SECONDARY)
        transla_header.pack(fill="x")
        self.gui_ref.translation_header = tk.Label(
            transla_header,
            text="🌐 Übersetzung:",
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_ACCENT,
            font=DragonFonts.SUBTITLE,
        )
        self.gui_ref.translation_header.pack(side="left")
        self.gui_ref.translation_scroll_var = tk.BooleanVar(value=True)
        scroll_cb = tk.Checkbutton(
            transla_header,
            variable=self.gui_ref.translation_scroll_var,
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            activebackground=self.gui_ref.current_theme.BG_SECONDARY,
            selectcolor=self.gui_ref.current_theme.CHECKBOX_ACTIVE,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
        )
        scroll_cb.pack(side="right", padx=3)
        tk.Label(
            transla_header,
            text="Auto-Scroll",
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
            font=("Segoe UI", 7),
        ).pack(side="right", padx=1)
        self.gui_ref.translation_text = self.create_text_widget(main_frame, height=6)

    def create_horizontal_layout(self) -> None:
        main_frame = tk.LabelFrame(
            self.gui_ref.text_container,
            text="Live Transkription & Übersetzung",
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            font=DragonFonts.SUBTITLE,
            padx=8,
            pady=8,
        )
        main_frame.pack(fill="both", expand=True)
        self.gui_ref.paned_window = tk.PanedWindow(
            main_frame,
            orient=tk.HORIZONTAL,
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            sashrelief="raised",
            sashwidth=4,
            sashpad=0,
        )
        self.gui_ref.paned_window.pack(fill="both", expand=True)
        left_frame = tk.Frame(self.gui_ref.paned_window, bg=self.gui_ref.current_theme.BG_TERTIARY)
        self.gui_ref.paned_window.add(left_frame, stretch="always", width=400)
        trans_header = tk.Frame(left_frame, bg=self.gui_ref.current_theme.BG_TERTIARY)
        trans_header.pack(fill="x", padx=5, pady=2)
        tk.Label(
            trans_header,
            text="🎤 Transkription",
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            fg=self.gui_ref.current_theme.TEXT_ACCENT,
            font=DragonFonts.SUBTITLE,
        ).pack(side="left")
        self.gui_ref.transcript_scroll_var = tk.BooleanVar(value=True)
        scroll_cb = tk.Checkbutton(
            trans_header,
            variable=self.gui_ref.transcript_scroll_var,
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            activebackground=self.gui_ref.current_theme.BG_TERTIARY,
            selectcolor=self.gui_ref.current_theme.CHECKBOX_ACTIVE,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
        )
        scroll_cb.pack(side="right", padx=3)
        tk.Label(
            trans_header,
            text="Auto-Scroll",
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
            font=("Segoe UI", 7),
        ).pack(side="right", padx=1)
        self.gui_ref.transcript_text = self.create_text_widget(left_frame)
        right_frame = tk.Frame(self.gui_ref.paned_window, bg=self.gui_ref.current_theme.BG_TERTIARY)
        self.gui_ref.paned_window.add(right_frame, stretch="always", width=400)
        transla_header = tk.Frame(right_frame, bg=self.gui_ref.current_theme.BG_TERTIARY)
        transla_header.pack(fill="x", padx=5, pady=2)
        self.gui_ref.translation_header = tk.Label(
            transla_header,
            text="🌐 Übersetzung",
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            fg=self.gui_ref.current_theme.TEXT_ACCENT,
            font=DragonFonts.SUBTITLE,
        )
        self.gui_ref.translation_header.pack(side="left")
        self.gui_ref.translation_scroll_var = tk.BooleanVar(value=True)
        scroll_cb = tk.Checkbutton(
            transla_header,
            variable=self.gui_ref.translation_scroll_var,
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            activebackground=self.gui_ref.current_theme.BG_TERTIARY,
            selectcolor=self.gui_ref.current_theme.CHECKBOX_ACTIVE,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
        )
        scroll_cb.pack(side="right", padx=3)
        tk.Label(
            transla_header,
            text="Auto-Scroll",
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
            font=("Segoe UI", 7),
        ).pack(side="right", padx=1)
        self.gui_ref.translation_text = self.create_text_widget(right_frame)
        self.gui_ref.paned_window.paneconfig(left_frame, minsize=250, width=400)
        self.gui_ref.paned_window.paneconfig(right_frame, minsize=250, width=400)

    def create_text_widget(self, parent: tk.Frame, height: Optional[int] = None) -> scrolledtext.ScrolledText:
        text_widget = scrolledtext.ScrolledText(
            parent,
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            font=DragonFonts.MONOSPACE,
            insertbackground=self.gui_ref.current_theme.TEXT_PRIMARY,
            wrap=tk.WORD,
            relief="flat",
            selectbackground=self.gui_ref.current_theme.COMBO_SELECTION,
            selectforeground=self.gui_ref.current_theme.TEXT_PRIMARY,
            maxundo=30,
            undo=True,
        )
        if height:
            text_widget.config(height=height)
        text_widget.pack(fill="both", expand=True, padx=5, pady=5)
        DarkContextMenu(text_widget)

        def safe_text_cleanup(event: Optional[tk.Event] = None) -> None:
            try:
                lines = int(text_widget.index("end-1c").split(".")[0])
                if lines > 400:
                    component = (
                        "transcript"
                        if text_widget == self.gui_ref.transcript_text
                        else "translation"
                    )
                    self.gui_ref.memory_manager.clear_component(component)
                    keep_lines = 250
                    delete_to = f"{lines - keep_lines}.0"
                    text_widget.delete("1.0", delete_to)
                    gc.collect()
            except Exception:
                pass
        text_widget.bind("<KeyRelease>", safe_text_cleanup)
        return text_widget

    def setup_status_bar(self) -> None:
        self.gui_ref.status_bar_frame = tk.Frame(self.root, bg=self.gui_ref.current_theme.BG_SECONDARY, height=50)
        self.gui_ref.status_bar_frame.grid_propagate(True)
        separator = tk.Frame(self.gui_ref.status_bar_frame, height=2, bg=self.gui_ref.current_theme.DRAGON_GREEN)
        separator.pack(fill="x", side="top")
        main_container = tk.Frame(self.gui_ref.status_bar_frame, bg=self.gui_ref.current_theme.BG_SECONDARY)
        main_container.pack(fill="x", expand=True, padx=12, pady=8)

        main_container.columnconfigure(0, weight=0)
        main_container.columnconfigure(1, weight=1)
        main_container.columnconfigure(2, weight=0)

        left_panel = tk.Frame(main_container, bg=self.gui_ref.current_theme.BG_SECONDARY)
        left_panel.grid(row=0, column=0, sticky="w", padx=5)

        quick_actions = [
            ("🗑️", self.gui_ref.clear_all, "Alles löschen"),
            ("💾", self.gui_ref.save_transcript, "Transkription speichern"),
            ("📝", self.gui_ref.export_subtitles, "Untertitel exportieren"),
            ("📊", self.gui_ref.show_simple_stats, "Statistiken anzeigen"),
            ("⚙️", self.gui_ref.show_advanced_settings, "Erweiterte Einstellungen"),
            ("🌐", self.gui_ref.show_translation_dialog, "Text übersetzen"),
            ("🤖", self.gui_ref.show_summarize_dialog, "Mit Ollama zusammenfassen"),
        ]

        if getattr(self.gui_ref, 'demo_mode', False) or not TRANSLATOR_AVAILABLE:
            install_btn = tk.Button(
                left_panel,
                text="📦",
                command=self.gui_ref.show_install_dialog,
                bg=self.gui_ref.current_theme.BG_TERTIARY,
                fg=self.gui_ref.current_theme.TEXT_PRIMARY,
                relief="flat",
                font=("Segoe UI", 9),
                cursor="hand2",
                padx=4, pady=2,
                activebackground=self.gui_ref.current_theme.BG_HOVER,
            )
            install_btn.grid(row=0, column=len(quick_actions)+1, padx=1, sticky="w")
            ToolTip(install_btn, "Fehlende Pakete installieren")

        for i, (icon, command, tooltip) in enumerate(quick_actions):
            btn = tk.Button(
                left_panel,
                text=icon,
                command=command,
                bg=self.gui_ref.current_theme.BG_TERTIARY,
                fg=self.gui_ref.current_theme.TEXT_PRIMARY,
                relief="flat",
                font=("Segoe UI", 9),
                cursor="hand2",
                padx=4,
                pady=2,
                activebackground=self.gui_ref.current_theme.BG_HOVER,
            )
            btn.grid(row=0, column=i, padx=1, sticky="w")
            ToolTip(btn, tooltip)

        center_panel = tk.Frame(main_container, bg=self.gui_ref.current_theme.BG_SECONDARY)
        center_panel.grid(row=0, column=1, sticky="ew", padx=5)

        self.gui_ref.progress_bar = ttk.Progressbar(
            center_panel,
            mode='determinate',
            length=150,
            style="Dark.Horizontal.TProgressbar"
        )
        self.gui_ref.progress_bar.pack(side="left", padx=(10, 10))

        self.gui_ref.progress_label = tk.Label(
            center_panel,
            text="",
            font=("Segoe UI", 8),
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY
        )
        self.gui_ref.progress_label.pack(side="left", padx=(0, 10))

        if IS_WINDOWS:
            default_text = "🪟 Windows | CPU: --% | RAM: --MB | GPU: --% | Model: --"
        elif IS_MACOS:
            if IS_ARM:
                default_text = "🍎 macOS (Apple Silicon) | CPU: --% | RAM: --MB | GPU: --% | Model: --"
            else:
                default_text = "🍎 macOS (Intel) | CPU: --% | RAM: --MB | GPU: --% | Model: --"
        elif IS_LINUX:
            default_text = "🐧 Linux | CPU: --% | RAM: --MB | GPU: --% | Model: --"
        else:
            default_text = "🌐 Unknown OS | CPU: --% | RAM: --MB | GPU: --% | Model: --"
        self.gui_ref.system_info_label = tk.Label(
            center_panel,
            text=default_text,
            font=("Segoe UI", 8, "normal"),
            bg=self.gui_ref.current_theme.BG_SECONDARY,
            fg=self.gui_ref.current_theme.TEXT_SECONDARY,
            padx=5,
        )
        self.gui_ref.system_info_label.pack(side="left", fill="x", expand=True)

        right_panel = tk.Frame(main_container, bg=self.gui_ref.current_theme.BG_SECONDARY)
        right_panel.grid(row=0, column=2, sticky="e", padx=5)

        self.gui_ref.exit_button = tk.Button(
            right_panel,
            text=" ⏻ EXIT ",
            command=self.gui_ref.controller.safe_exit,
            bg="#dc3545",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="raised",
            cursor="hand2",
            padx=12,
            pady=3,
            activebackground="#c82333",
        )
        self.gui_ref.exit_button.pack(side="right")
        ToolTip(self.gui_ref.exit_button, "Programm beenden (Strg+Q / Cmd+Q)")

        help_btn = tk.Button(
            right_panel,
            text="⌨️",
            command=self.gui_ref.show_shortcuts_help,
            bg=self.gui_ref.current_theme.BG_TERTIARY,
            fg=self.gui_ref.current_theme.TEXT_PRIMARY,
            relief="flat",
            font=("Segoe UI", 9),
            cursor="hand2",
            padx=4
        )
        help_btn.pack(side="right", padx=2)
        ToolTip(help_btn, "Tastenkürzel anzeigen (F1)")

    def process_batch_text_updates(self) -> None:
        if (not hasattr(self.gui_ref, '_shutting_down') or
            getattr(self.gui_ref, '_shutting_down', False)):
            return
        if (not hasattr(self, 'root') or
            self.root is None or
            not self.root.winfo_exists()):
            return
        if not hasattr(self.gui_ref, '_text_update_queue'):
            return
        queue_obj = self.gui_ref._text_update_queue
        if queue_obj is None:
            return
        if not hasattr(queue_obj, 'empty') or not callable(queue_obj.empty):
            return
        if queue_obj.empty():
            return
        processed = 0
        max_updates = 5
        while processed < max_updates:
            try:
                if hasattr(queue_obj, 'get_nowait') and callable(queue_obj.get_nowait):
                    item = queue_obj.get_nowait()
                else:
                    item = queue_obj.get(block=False)
                if isinstance(item, tuple) and len(item) == 2:
                    update_type, text_data = item
                    self._process_update(update_type, text_data)
                try:
                    if hasattr(queue_obj, 'task_done') and callable(queue_obj.task_done):
                        queue_obj.task_done()
                except Exception:
                    pass
                processed += 1
            except Exception as e:
                if "Empty" in type(e).__name__ or "empty" in str(e).lower():
                    break
                logger.warning(f"⚠️ Queue processing error: {e}")
                break
        self._schedule_next_update()

    def _process_update(self, update_type: str, text_data: str) -> None:
        try:
            if update_type == 'transcript':
                if (hasattr(self.gui_ref, 'transcript_text') and
                    self.gui_ref.transcript_text is not None and
                    self.gui_ref.transcript_text.winfo_exists()):
                    self.gui_ref.transcript_text.insert('end', text_data)
                    self._auto_scroll('transcript')
                    self._check_text_limit('transcript')
            elif update_type == 'translation':
                if (hasattr(self.gui_ref, 'translation_text') and
                    self.gui_ref.translation_text is not None and
                    self.gui_ref.translation_text.winfo_exists()):
                    self.gui_ref.translation_text.insert('end', text_data)
                    self._auto_scroll('translation')
                    self._check_text_limit('translation')
        except tk.TclError:
            pass
        except Exception as e:
            logger.warning(f"⚠️ GUI update error: {e}")

    def _auto_scroll(self, text_type: str) -> None:
        try:
            if text_type == 'transcript':
                if (hasattr(self.gui_ref, 'transcript_scroll_var') and
                    self.gui_ref.transcript_scroll_var is not None and
                    self.gui_ref.transcript_scroll_var.get()):
                    self.gui_ref.transcript_text.see('end')
            elif text_type == 'translation':
                if (hasattr(self.gui_ref, 'translation_scroll_var') and
                    self.gui_ref.translation_scroll_var is not None and
                    self.gui_ref.translation_scroll_var.get()):
                    self.gui_ref.translation_text.see('end')
        except Exception:
            pass

    def _check_text_limit(self, text_type: str) -> None:
        try:
            if text_type == 'transcript':
                widget = self.gui_ref.transcript_text
                max_lines = 400
                keep_lines = 300
            else:
                widget = self.gui_ref.translation_text
                max_lines = 300
                keep_lines = 200
            lines = int(widget.index('end-1c').split('.')[0])
            if lines > max_lines:
                delete_to = f'{lines-keep_lines}.0'
                widget.delete('1.0', delete_to)
        except Exception:
            pass

    def _schedule_next_update(self) -> None:
        try:
            if (hasattr(self, 'root') and
                self.root is not None and
                self.root.winfo_exists()):
                interval = 150
                if hasattr(self.gui_ref, '_batch_update_interval'):
                    try:
                        interval = self.gui_ref._batch_update_interval
                    except Exception:
                        pass
                try:
                    if hasattr(self, '_batch_timer_id') and self._batch_timer_id:
                        self.root.after_cancel(self._batch_timer_id)
                except Exception:
                    pass
                self._batch_timer_id = self.root.after(interval, self.process_batch_text_updates)
        except Exception as e:
            logger.warning(f"⚠️ Timer scheduling error: {e}")

    def start_batch_updates(self) -> None:
        try:
            if (hasattr(self, 'root') and
                self.root is not None and
                self.root.winfo_exists()):
                if not hasattr(self.gui_ref, '_text_update_queue') or self.gui_ref._text_update_queue is None:
                    try:
                        self.gui_ref._text_update_queue = queue.Queue(maxsize=150)
                    except Exception:
                        class DummyQueue:
                            def __init__(self, maxsize: int = 0) -> None:
                                self.maxsize = maxsize
                                self._items: List[Any] = []
                                self._lock = threading.Lock()
                                class EmptyException(Exception):
                                    pass
                                self.Empty = EmptyException
                            def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
                                with self._lock:
                                    self._items.append(item)
                                    if self.maxsize > 0 and len(self._items) > self.maxsize:
                                        self._items.pop(0)
                            def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
                                with self._lock:
                                    if self._items:
                                        return self._items.pop(0)
                                    raise self.Empty()
                            def empty(self) -> bool:
                                with self._lock:
                                    return len(self._items) == 0
                            def qsize(self) -> int:
                                with self._lock:
                                    return len(self._items)
                            def task_done(self) -> None:
                                pass
                            def get_nowait(self) -> Any:
                                return self.get(block=False)
                            def join(self) -> None:
                                pass
                        self.gui_ref._text_update_queue = DummyQueue(maxsize=150)
                        logger.warning("⚠️ Queue-Fallback in start_batch_updates")
                self.root.after(100, self.process_batch_text_updates)
                logger.info("✅ Batch updates gestartet")
        except Exception as e:
            logger.warning(f"⚠️ Start batch updates error: {e}")


# -----------------------------------------------------------------------------
# WHISPER CONTROLLER (überarbeitet mit Thread-Sicherheit)
# -----------------------------------------------------------------------------
class WhisperController:
    def __init__(self, gui_ref: Any, ui_update_fn: Optional[Callable] = None,
                 status_update_fn: Optional[Callable] = None) -> None:
        self.gui_ref = gui_ref
        self.is_processing = False
        self._processing_lock = threading.Lock()
        self._cleanup_lock = threading.RLock()
        self._last_transcription_text = ""
        self._duplicate_check_cache: Deque[str] = deque(maxlen=20)
        self._processing_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        if ui_update_fn is None:
            self.ui_update_fn = self._create_default_ui_updater()
        else:
            self.ui_update_fn = ui_update_fn
        if status_update_fn is None:
            self.status_update_fn = self._create_default_status_updater()
        else:
            self.status_update_fn = status_update_fn
        self._stop_requested = False
        self._processing_active = threading.Event()
        self._initialized = True

        self._stop_complete = threading.Event()
        self._stop_complete.set()

    def _create_default_ui_updater(self) -> Callable:
        def default_updater(component: str, text: str) -> None:
            try:
                if not text or not text.strip():
                    return
                if component == "transcript":
                    if hasattr(self.gui_ref, "transcript_text"):
                        self._append_to_text_widget(self.gui_ref.transcript_text, text)
                    else:
                        logger.info(f"🎤 {text[:100]}...")
                elif component == "translation":
                    if hasattr(self.gui_ref, "translation_text"):
                        self._append_to_text_widget(self.gui_ref.translation_text, text)
                    else:
                        logger.info(f"🌐 {text[:100]}...")
            except Exception:
                pass
        return default_updater

    def _create_default_status_updater(self) -> Callable:
        def default_updater(state_info: Dict[str, Any]) -> None:
            try:
                if "status" in state_info:
                    status = state_info["status"]
                    if hasattr(self.gui_ref, "status_label"):
                        self._update_status_label(status)
                    else:
                        logger.info(f"📊 {status}")
            except Exception:
                pass
        return default_updater

    def _processing_finished(self) -> None:
        self.is_processing = False
        self.status_update_fn({"processing_state": False, "status": "✅ Processing completed"})

    def _append_to_text_widget(self, widget: tk.Text, text: str) -> None:
        try:
            if hasattr(self.gui_ref, "root") and self.gui_ref.root.winfo_exists():
                self.gui_ref.root.after(0, lambda: self._safe_text_insert(widget, text))
        except (tk.TclError, AttributeError):
            pass

    def _safe_text_insert(self, widget: tk.Text, text: str) -> None:
        try:
            if widget and widget.winfo_exists():
                widget.insert("end", text)
                if widget == getattr(self.gui_ref, "transcript_text", None):
                    if hasattr(self.gui_ref, "transcript_scroll_var"):
                        if self.gui_ref.transcript_scroll_var.get():
                            widget.see("end")
                elif widget == getattr(self.gui_ref, "translation_text", None):
                    if hasattr(self.gui_ref, "translation_scroll_var"):
                        if self.gui_ref.translation_scroll_var.get():
                            widget.see("end")
        except (tk.TclError, AttributeError):
            pass

    def _update_status_label(self, text: str) -> None:
        try:
            if hasattr(self.gui_ref, "root") and self.gui_ref.root.winfo_exists():
                self.gui_ref.root.after(0, lambda: self.gui_ref.status_label.config(text=text[:100] if text else "Ready"))
        except Exception:
            pass

    def _cleanup_resources(self) -> None:
        if hasattr(self, "_stop_requested") and self._stop_requested:
            return
        self._stop_requested = True
        self.is_processing = False
        try:
            if hasattr(self.gui_ref, "audio_processor"):
                self.gui_ref.audio_processor._processing = False
                if hasattr(self.gui_ref.audio_processor, "_stop_event"):
                    self.gui_ref.audio_processor._stop_event.set()
        except Exception:
            pass

    def _stop_processing_sync(self, timeout: float = 10.0) -> bool:
        self._stop_complete.clear()
        logger.info("🛑 WhisperController: Synchrone Stop angefordert")

        self.stop_processing()

        if hasattr(self.gui_ref, 'audio_processor') and self.gui_ref.audio_processor:
            if hasattr(self.gui_ref.audio_processor, '_process_finished'):
                if not self.gui_ref.audio_processor._process_finished.wait(timeout):
                    logger.warning(f"⚠️ Audio-Processor nicht innerhalb von {timeout}s beendet")
                    return False
            else:
                time.sleep(0.5)
        else:
            time.sleep(0.1)

        self._stop_complete.set()
        return True

    def _on_progress(self, processed: int, total: Optional[int], chunks: int) -> None:
        if hasattr(self.gui_ref, 'update_progress'):
            self.gui_ref.root.after(0, self.gui_ref.update_progress, processed, total, chunks)

    def _start_processing(self) -> None:
        with self._processing_lock:
            if self.is_processing:
                self.status_update_fn({"status": "⚠️ Bereits aktiv"})
                return
            url = ""
            try:
                url = self.gui_ref.url_entry.get().strip()
            except Exception:
                self.status_update_fn({"status": "❌ URL Fehler"})
                return
            if not url:
                self.status_update_fn({"status": "❌ Bitte URL eingeben"})
                return
            try:
                # URL validieren
                url = PlatformUtils.sanitize_url(url)
                if url.startswith("file://"):
                    ok, real_path = PlatformUtils.validate_file_path(url)
                    if not ok:
                        self.status_update_fn({"status": f"❌ {real_path}"})
                        return
                    file_path = real_path
                    if not os.path.exists(file_path):
                        self.status_update_fn({"status": "❌ Datei nicht gefunden"})
                        return
                else:
                    if not url.startswith(("http://", "https://")):
                        url = "https://" + url
                        self.gui_ref.url_entry.delete(0, "end")
                        self.gui_ref.url_entry.insert(0, url)
            except Exception:
                self.status_update_fn({"status": "❌ Ungültige URL"})
                return

            self.status_update_fn({"status": "🔍 Analysiere Stream..."})

            if self.is_processing:
                logger.warning("⚠️ Vorheriger Prozess läuft noch – stoppe diesen zuerst synchron.")
                if not self._stop_processing_sync(timeout=10):
                    self.status_update_fn({"status": "❌ Vorheriger Prozess konnte nicht gestoppt werden"})
                    return

            try:
                if hasattr(self.gui_ref, "stream_manager"):
                    platform_type, platform_name = self.gui_ref.stream_manager.detect_platform(url)
                else:
                    platform_type, platform_name = "unknown", "Unknown"
                stream_info = None
                try:
                    stream_info = StreamInfoExtractor().extract_stream_info(url)
                except Exception:
                    stream_info = StreamInfo(
                        title="Live Stream" if "live" in url.lower() else "Stream",
                        uploader=platform_name,
                        duration="Live" if "live" in url.lower() else "Unknown",
                        view_count=0,
                        platform=platform_type,
                    )
                if stream_info:
                    self.status_update_fn({"stream_info": stream_info})
                    logger.info(f"📡 Stream: {stream_info.title[:50]}...")
            except Exception as e:
                logger.warning(f"⚠️ Stream Info Error: {e}")

            self.status_update_fn({"status": "🎵 Teste Audio-Stream..."})
            stream_test_passed = False
            try:
                if hasattr(self.gui_ref, "audio_processor"):
                    stream_test_passed = self.gui_ref.audio_processor.emergency_diagnosis(url)
                    if not stream_test_passed:
                        try:
                            if hasattr(self.gui_ref, "stream_manager"):
                                audio_url = self.gui_ref.stream_manager.extract_audio_url(url)
                                if audio_url:
                                    stream_test_passed = self.gui_ref.audio_processor._test_audio_stream(audio_url)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"⚠️ Stream Test Error: {e}")
            if not stream_test_passed:
                self.status_update_fn({"status": "❌ Stream nicht erreichbar"})
                logger.error("❌ Stream Test fehlgeschlagen")
                return

            self.status_update_fn({"status": "🤖 Lade KI-Modell..."})
            model_loaded = False
            try:
                if hasattr(self.gui_ref, "transcription_engine"):
                    model_name = "medium"
                    if hasattr(self.gui_ref, "model_var"):
                        model_name = self.gui_ref.model_var.get()
                    result = self.gui_ref.transcription_engine.load_model(model_name, set_active=True)
                    if result is not None:
                        model_loaded = True
                    else:
                        logger.info("🔄 Versuche base model...")
                        result = self.gui_ref.transcription_engine.load_model("base", set_active=True)
                        model_loaded = result is not None
            except Exception as e:
                logger.warning(f"⚠️ Model Load Error: {e}")
            if not model_loaded:
                self.status_update_fn({"status": "❌ KI-Modell konnte nicht geladen werden"})
                return

            try:
                if hasattr(self.gui_ref, "translation_engine") and hasattr(self.gui_ref, "lang_var"):
                    selected_name = self.gui_ref.lang_var.get()
                    target_lang = "de"
                    for name, code in SORTED_LANGUAGES:
                        if name == selected_name:
                            target_lang = code
                            break
                    self.gui_ref.translation_engine.set_target_language(target_lang)
                    lang_display = LANGUAGE_SHORT_CODES.get(target_lang, target_lang)
                    if hasattr(self.gui_ref, "translation_header"):
                        self.gui_ref.translation_header.config(text=f"🌐 Übersetzung ({lang_display})")
            except Exception as e:
                logger.warning(f"⚠️ Translation Setup Error: {e}")

            self.is_processing = True
            if hasattr(self.gui_ref, "is_processing"):
                self.gui_ref.is_processing = True

            def update_gui_buttons() -> None:
                try:
                    if hasattr(self.gui_ref, "start_button"):
                        self.gui_ref.start_button.config(state="disabled")
                    if hasattr(self.gui_ref, "stop_button"):
                        self.gui_ref.stop_button.config(state="normal")
                except Exception:
                    pass
            if hasattr(self.gui_ref, "root") and self.gui_ref.root.winfo_exists():
                self.gui_ref.root.after(0, update_gui_buttons)

            if IS_LINUX and hasattr(self.gui_ref, "performance_optimizer"):
                self.gui_ref.performance_optimizer.optimize_for_processing()

            self.status_update_fn({
                "processing_state": True,
                "status": "🚀 Starte Transkription...",
                "buttons": {"start": "disabled", "stop": "normal"},
            })

            def transcription_callback(result: ExcellenceTranscriptionResult) -> None:
                if not result or not hasattr(result, "text"):
                    return
                try:
                    if hasattr(self.gui_ref, "handle_transcription"):
                        self.gui_ref.handle_transcription(result)
                    else:
                        text = f"{result.text}\n"
                        self.ui_update_fn("transcript", text)
                except Exception as e:
                    logger.warning(f"⚠️ Transcription Callback Error: {e}")

            def translation_callback(result: ExcellenceTranslationResult) -> None:
                if not result or not hasattr(result, "translated"):
                    return
                try:
                    if hasattr(self.gui_ref, "handle_translation"):
                        self.gui_ref.handle_translation(result)
                    else:
                        text = f"{result.translated}\n"
                        self.ui_update_fn("translation", text)
                except Exception as e:
                    logger.warning(f"⚠️ Translation Callback Error: {e}")

            def info_callback(message: str) -> None:
                try:
                    if hasattr(self.gui_ref, "handle_info"):
                        self.gui_ref.handle_info(message)
                    else:
                        self.status_update_fn({"status": f"ℹ️ {message}"})
                except Exception:
                    pass

            def error_callback(message: str) -> None:
                try:
                    if hasattr(self.gui_ref, "handle_error"):
                        self.gui_ref.handle_error(message)
                    else:
                        self.status_update_fn({"status": f"❌ {message}"})
                    self._cleanup_resources()
                except Exception:
                    pass

            def file_finished_callback() -> None:
                logger.info("✅ Dateiende erkannt – zeige Speicherdialog")
                self.status_update_fn({"file_finished": True})
                self._processing_finished()

            try:
                if hasattr(self.gui_ref, "audio_processor"):
                    self.gui_ref.audio_processor._stop_event.clear()
                    self.gui_ref.audio_processor.set_progress_callback(self._on_progress)
                    processing_thread = threading.Thread(
                        target=lambda: self.gui_ref.audio_processor.start_processing(
                            url=url,
                            transcription_callback=transcription_callback,
                            translation_callback=translation_callback,
                            info_callback=info_callback,
                            error_callback=error_callback,
                            finished_callback=file_finished_callback,
                        ),
                        daemon=True,
                        name="AudioProcessor",
                    )
                    self._processing_thread = processing_thread
                    processing_thread.start()
                    self.status_update_fn({"status": "✅ Transkription läuft..."})
                else:
                    error_callback("❌ Audio-Processor nicht verfügbar")
                    self.status_update_fn({"processing_state": False, "status": "❌ Audio-Processor nicht verfügbar"})
                    self.is_processing = False
            except (AttributeError, RuntimeError) as e:
                error_msg = f"Start Error: {str(e)[:100]}"
                logger.error(f"❌ Processing Start Error: {e}")
                error_callback(error_msg)
                self.status_update_fn({"processing_state": False, "status": f"❌ {error_msg}"})
                self.is_processing = False

    def start_processing(self) -> None:
        def start_thread() -> None:
            try:
                self._start_processing()
            except Exception as e:
                logger.error(f"❌ Start Processing Error: {e}")
                self.status_update_fn({"status": f"❌ Start fehlgeschlagen: {str(e)[:50]}"})
        thread = threading.Thread(target=start_thread, daemon=True)
        thread.start()

    def stop_processing(self) -> None:
        if hasattr(self, "_stop_requested"):
            self._stop_requested = True
        if IS_LINUX and hasattr(self.gui_ref, "performance_optimizer"):
            self.gui_ref.performance_optimizer.restore_normal_mode()
        if hasattr(self, "_shutdown_event"):
            self._shutdown_event.set()
        if hasattr(self, "_processing_active"):
            self._processing_active.clear()
        if hasattr(self.gui_ref, "is_processing"):
            self.gui_ref.is_processing = False

        def stop_audio_processor() -> None:
            try:
                if hasattr(self.gui_ref, "audio_processor"):
                    ap = self.gui_ref.audio_processor
                    ap._processing = False
                    if hasattr(ap, "_stop_event"):
                        ap._stop_event.set()
                    if hasattr(self.gui_ref, "ffmpeg_manager"):
                        self.gui_ref.ffmpeg_manager.stop_all_streams()
                self._stop_complete.set()
            except Exception as e:
                logger.warning(f"⚠️ Audio Stop Fehler: {e}")
                self._stop_complete.set()

        audio_stop_thread = threading.Thread(target=stop_audio_processor, daemon=True)
        audio_stop_thread.start()

        def update_gui_immediately() -> None:
            try:
                if hasattr(self.gui_ref, "status_label"):
                    self.gui_ref.status_label.config(text="✅ READY for new stream")
                if hasattr(self.gui_ref, "start_button"):
                    self.gui_ref.start_button.config(state="normal")
                if hasattr(self.gui_ref, "stop_button"):
                    self.gui_ref.stop_button.config(state="disabled")
                if hasattr(self.gui_ref, "stream_title_label"):
                    self.gui_ref.stream_title_label.config(text="📡 Kein aktiver Stream")
                if hasattr(self.gui_ref, "stream_details_label"):
                    self.gui_ref.stream_details_label.config(text="Bereit für neue Verbindung")
                self.is_processing = False
                self.gui_ref._reset_progress()
            except Exception as e:
                logger.warning(f"⚠️ GUI Update Fehler: {e}")
        if hasattr(self.gui_ref, "root") and self.gui_ref.root.winfo_exists():
            self.gui_ref.root.after(0, update_gui_immediately)

        def background_cleanup() -> None:
            try:
                if self._processing_thread and self._processing_thread.is_alive():
                    logger.info("🔄 Warte auf Processing Thread...")
                    self._processing_thread.join(timeout=1.0)
                transcription_cache.clear()
                translation_cache.clear()
                audio_cache.clear()
                self._stop_requested = False
                self._shutdown_event.clear()
            except Exception as e:
                logger.warning(f"⚠️ Cleanup Fehler: {e}")
        cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
        cleanup_thread.start()

    def dispose(self) -> None:
        self._shutdown_event.set()
        self.stop_processing()
        logger.info("🧹 Controller disposed")

    def safe_exit(self) -> None:
        try:
            if hasattr(self.gui_ref, "exit_button"):
                try:
                    self.gui_ref.exit_button.config(state="disabled", text="⏳...")
                except Exception:
                    pass
            if hasattr(self.gui_ref, "_safe_exit_dialog"):
                self.gui_ref._safe_exit_dialog()
            else:
                self._cleanup_resources()
                sys.exit(0)
        except Exception:
            sys.exit(0)


# -----------------------------------------------------------------------------
# LINUX PERFORMANCE OPTIMIZER (unverändert)
# -----------------------------------------------------------------------------
PSUTIL_AVAILABLE = importlib.util.find_spec("psutil") is not None
if not PSUTIL_AVAILABLE:
    logger.warning("⚠️ psutil nicht verfügbar – Linux Performance Optimizer läuft im Dummy-Modus")

if IS_LINUX and PSUTIL_AVAILABLE:
    class LinuxPerformanceOptimizer:
        def __init__(self, gui_ref: 'DragonWhispererGUI') -> None:
            self.gui = gui_ref
            self.is_processing = False
            self._original_settings: Dict[str, Any] = {}
            self._optimization_active = False
            self._monitoring_thread: Optional[threading.Thread] = None
            self._shutdown_event = threading.Event()
            self._monitoring_lock = threading.RLock()
            self._last_gui_access_time = 0.0
            self._gui_access_warning_printed = False

        def optimize_for_processing(self) -> None:
            if not IS_LINUX or self._optimization_active:
                return
            with self._monitoring_lock:
                self._shutdown_event.clear()
                if self._monitoring_thread and self._monitoring_thread.is_alive():
                    logger.warning("⚠️ Optimize: Monitoring-Thread läuft bereits – überspringe")
                    return
                logger.info("🔧 Aktiviere Linux-Performance-Optimierungen...")
                self._optimize_text_widget('transcript_text')
                self._optimize_text_widget('translation_text')
                self._schedule_batch_interval_increase()
                self._clean_queue_safe(getattr(self.gui, 'gui_queue', None), 15)
                self._apply_linux_specific_optimizations()
                self._optimization_active = True
                self.is_processing = True
                self._start_performance_monitoring()
                logger.info("✅ Linux-Performance-Optimierungen aktiviert")

        def _schedule_batch_interval_increase(self) -> None:
            if not self._is_gui_available_safe():
                return
            if hasattr(self.gui, '_batch_update_interval'):
                self._original_settings['batch_update_interval'] = self.gui._batch_update_interval
            def task() -> None:
                if hasattr(self.gui, '_batch_update_interval'):
                    self.gui._batch_update_interval = 250
            if hasattr(self.gui, 'root'):
                self.gui.root.after(0, task)

        def _optimize_text_widget(self, attr_name: str) -> None:
            widget = getattr(self.gui, attr_name, None)
            if widget and widget.winfo_exists():
                self._original_settings[attr_name] = {
                    'maxundo': widget.cget('maxundo'),
                    'undo': widget.cget('undo'),
                    'autoseparators': widget.cget('autoseparators')
                }
                widget.configure(maxundo=5, undo=True, autoseparators=True, height=12)

        def _is_gui_available_safe(self) -> bool:
            try:
                return (hasattr(self.gui, 'root') and
                        self.gui.root is not None and
                        self.gui.root.winfo_exists() and
                        not getattr(self.gui, '_shutting_down', False))
            except Exception:
                return False

        def _apply_linux_specific_optimizations(self) -> None:
            if not self._is_gui_available_safe():
                return
            self._detect_compositor()
            self._increase_resource_limits()

        def _detect_compositor(self) -> None:
            try:
                import psutil
                for proc in psutil.process_iter(['name']):
                    try:
                        name = proc.info['name'].lower()
                        if any(c in name for c in ['compton', 'picom', 'compiz', 'kwin']):
                            logger.info(f"  ↪ Compositor erkannt: {name}")
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except (ImportError, AttributeError):
                pass

        def _increase_resource_limits(self) -> None:
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
                new_soft = min(hard, 1024 * 1024 * 1024)
                if new_soft > soft:
                    resource.setrlimit(resource.RLIMIT_DATA, (new_soft, hard))
                    logger.info(f"  ↪ Daten-Limit erhöht: {soft} → {new_soft}")

                soft_fd, hard_fd = resource.getrlimit(resource.RLIMIT_NOFILE)
                new_soft_fd = min(hard_fd, 8192)
                if new_soft_fd > soft_fd:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_fd, hard_fd))
                    logger.info(f"  ↪ Dateideskriptoren-Limit erhöht: {soft_fd} → {new_soft_fd}")
            except Exception as e:
                logger.warning(f"⚠️ Ressourcenlimits konnten nicht erhöht werden: {e}")

        def _start_performance_monitoring(self) -> None:
            with self._monitoring_lock:
                if self._monitoring_thread and self._monitoring_thread.is_alive():
                    return

                def monitor_worker() -> None:
                    if self._shutdown_event.is_set():
                        return
                    logger.info("🔍 Linux-Performance-Monitoring gestartet")
                    check_count = 0
                    max_checks = 240

                    while not self._shutdown_event.is_set() and self._optimization_active:
                        try:
                            time.sleep(30)
                            if self._shutdown_event.is_set():
                                break
                            if not self._is_gui_available_safe():
                                logger.warning("⚠️ GUI nicht mehr verfügbar – stoppe Monitoring")
                                break

                            check_count += 1
                            if check_count > max_checks:
                                logger.info("⏰ Monitoring-Zeitlimit erreicht – beende")
                                break

                            system_load = self._get_system_load()
                            memory_usage = self._get_memory_usage()

                            if system_load > 0.8 or memory_usage > 0.85:
                                self._schedule_adjustments(system_load, memory_usage)

                            if check_count % 12 == 0:
                                self._print_performance_report(system_load, memory_usage)

                        except (OSError, ValueError) as e:
                            logger.warning(f"⚠️ Fehler im Monitoring-Thread: {e}")

                    logger.info("✅ Linux-Performance-Monitoring beendet")
                    with self._monitoring_lock:
                        self._monitoring_thread = None

                self._monitoring_thread = threading.Thread(target=monitor_worker, daemon=True, name="LinuxPerfMon")
                self._monitoring_thread.start()

        def _get_system_load(self) -> float:
            try:
                load_avg = os.getloadavg()
                cpu_count = os.cpu_count() or 1
                return load_avg[0] / cpu_count
            except (OSError, ValueError):
                return 0.0

        def _get_memory_usage(self) -> float:
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent / 100.0
            except Exception:
                return 0.0

        def _schedule_adjustments(self, load: float, memory: float) -> None:
            if not self._is_gui_available_safe():
                return
            def task() -> None:
                if not self._optimization_active or self._shutdown_event.is_set():
                    return
                adjustments: List[str] = []

                if memory > 0.85 and hasattr(self.gui, 'gui_queue'):
                    cleared = self._clean_queue_safe(self.gui.gui_queue, 5)
                    if cleared > 0:
                        adjustments.append(f"Queue: -{cleared}")

                if load > 0.8 and hasattr(self.gui, '_batch_update_interval'):
                    current = self.gui._batch_update_interval
                    if current < 500:
                        self.gui._batch_update_interval = min(500, current + 50)
                        adjustments.append(f"Update: {current}→{self.gui._batch_update_interval}ms")

                if adjustments:
                    logger.info(f"🔧 Anpassungen: {', '.join(adjustments)}")
            if hasattr(self.gui, 'root'):
                self.gui.root.after(0, task)

        def _clean_queue_safe(self, queue_obj: Optional[queue.Queue], target_size: int) -> int:
            if not queue_obj or queue_obj.qsize() <= target_size:
                return 0
            cleared = 0
            try:
                while queue_obj.qsize() > target_size and cleared < 50:
                    queue_obj.get_nowait()
                    queue_obj.task_done()
                    cleared += 1
            except queue.Empty:
                pass
            except Exception as e:
                logger.warning(f"⚠️ Queue-Cleanup-Fehler: {e}")
            return cleared

        def _print_performance_report(self, load: float, memory: float) -> None:
            try:
                stats = {
                    'System-Last': f"{load:.1%}",
                    'RAM-Auslastung': f"{memory:.1%}",
                    'Status': 'Aktiv' if self.is_processing else 'Inaktiv'
                }
                logger.info("🐧 Linux-Performance-Report: " + " | ".join(f"{k}: {v}" for k, v in stats.items()))
            except Exception:
                pass

        def restore_normal_mode(self) -> None:
            if not IS_LINUX:
                return
            logger.info("🔧 Linux-Optimierer: Fahre herunter...")
            self._shutdown_event.set()
            self._optimization_active = False
            self.is_processing = False

            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=1.0)
                self._monitoring_thread = None

            if self._is_gui_available_safe():
                logger.info("  ↪ Stelle GUI-Einstellungen wieder her...")
                self._restore_text_widget('transcript_text')
                self._restore_text_widget('translation_text')

                if 'batch_update_interval' in self._original_settings:
                    saved_interval = self._original_settings['batch_update_interval']
                    def restore_batch_interval(saved=saved_interval):
                        if hasattr(self.gui, 'root') and self.gui.root.winfo_exists():
                            if hasattr(self.gui, '_batch_update_interval'):
                                self.gui._batch_update_interval = saved
                    if hasattr(self.gui, 'root'):
                        self.gui.root.after(0, restore_batch_interval)

            self._original_settings.clear()
            logger.info("✅ Linux-Optimierer heruntergefahren")

        def _restore_text_widget(self, attr_name: str) -> None:
            widget = getattr(self.gui, attr_name, None)
            if widget and widget.winfo_exists() and attr_name in self._original_settings:
                try:
                    widget.configure(**self._original_settings[attr_name])
                    logger.info(f"    ✅ {attr_name} wiederhergestellt")
                except Exception as e:
                    logger.warning(f"    ⚠️ {attr_name} konnte nicht wiederhergestellt werden: {e}")

        def emergency_optimize(self) -> None:
            if self._shutdown_event.is_set():
                return
            logger.info("🚨 Führe Notfall-Optimierungen durch...")
            if not self._is_gui_available_safe():
                return
            def task() -> None:
                self._clean_queue_safe(getattr(self.gui, 'gui_queue', None), 3)
                self._clean_queue_safe(getattr(self.gui, '_text_update_queue', None), 2)

                for attr in ['transcript_text', 'translation_text']:
                    widget = getattr(self.gui, attr, None)
                    if widget and widget.winfo_exists():
                        try:
                            widget.configure(height=6, maxundo=1)
                        except Exception:
                            pass

                if hasattr(self.gui, '_batch_update_interval'):
                    self.gui._batch_update_interval = 500

                gc.collect()
                logger.info("✅ Notfall-Optimierungen abgeschlossen")
            if hasattr(self.gui, 'root'):
                self.gui.root.after(0, task)

        def get_optimization_status(self) -> Dict[str, Any]:
            return {
                'platform': SYSTEM,
                'optimization_active': self._optimization_active,
                'processing_active': self.is_processing,
                'monitoring_active': self._monitoring_thread and self._monitoring_thread.is_alive(),
                'shutdown_event_set': self._shutdown_event.is_set(),
                'original_settings_count': len(self._original_settings),
                'linux_specific': IS_LINUX,
                'gui_available': self._is_gui_available_safe()
            }

        def dispose(self) -> None:
            logger.info("🧹 Linux-Performance-Optimierer wird entsorgt...")
            self._shutdown_event.set()
            try:
                self.restore_normal_mode()
            except Exception as e:
                logger.warning(f"⚠️ restore_normal_mode fehlgeschlagen: {e}")
            self._original_settings.clear()
            gc.collect()
            logger.info("✅ Linux-Performance-Optimierer entsorgt")

        def emergency_shutdown(self) -> None:
            logger.info("🚨 Externer Notfall-Shutdown des Linux-Optimierers")
            self._shutdown_event.set()
            self._optimization_active = False
            self.is_processing = False
            self._monitoring_thread = None
            self._original_settings.clear()

elif IS_LINUX:
    class LinuxPerformanceOptimizer:
        def __init__(self, gui_ref: 'DragonWhispererGUI') -> None:
            self.gui = gui_ref
            self.is_processing = False
            logger.info("🐧 LinuxPerformanceOptimizer: Dummy-Modus (psutil fehlt)")

        def optimize_for_processing(self) -> None:
            logger.debug("LinuxPerformanceOptimizer (Dummy): optimize_for_processing")

        def restore_normal_mode(self) -> None:
            logger.debug("LinuxPerformanceOptimizer (Dummy): restore_normal_mode")

        def emergency_optimize(self) -> None:
            logger.debug("LinuxPerformanceOptimizer (Dummy): emergency_optimize")

        def get_optimization_status(self) -> Dict[str, Any]:
            return {'platform': SYSTEM, 'dummy_mode': True, 'psutil_missing': True}

        def dispose(self) -> None:
            pass

        def emergency_shutdown(self) -> None:
            pass


# -----------------------------------------------------------------------------
# INSTALL DIALOG (unverändert)
# -----------------------------------------------------------------------------
class InstallDependencyDialog:
    def __init__(self, parent, gui_ref):
        self.parent = parent
        self.gui = gui_ref
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🐉 Fehlende Pakete installieren")
        self.dialog.geometry("600x400")
        self.dialog.configure(bg=CURRENT_THEME.BG_PRIMARY)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = tk.Frame(self.dialog, bg=CURRENT_THEME.BG_PRIMARY, padx=15, pady=15)
        main.pack(fill="both", expand=True)

        tk.Label(main, text="Optionale Pakete, die installiert werden können:",
                 bg=CURRENT_THEME.BG_PRIMARY, fg=CURRENT_THEME.TEXT_PRIMARY,
                 font=DragonFonts.PRIMARY).pack(anchor="w", pady=(0,10))

        self.packages = {}
        frame = tk.Frame(main, bg=CURRENT_THEME.BG_PRIMARY)
        frame.pack(fill="x")

        if not WHISPER_AVAILABLE:
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(frame, text="faster-whisper (benötigt für Transkription)",
                                variable=var, bg=CURRENT_THEME.BG_PRIMARY,
                                fg=CURRENT_THEME.TEXT_PRIMARY,
                                selectcolor=CURRENT_THEME.BG_TERTIARY)
            cb.pack(anchor="w")
            self.packages["faster-whisper"] = var

        if not TRANSLATOR_AVAILABLE:
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(frame, text="deep-translator (für Übersetzungen)",
                                variable=var, bg=CURRENT_THEME.BG_PRIMARY,
                                fg=CURRENT_THEME.TEXT_PRIMARY,
                                selectcolor=CURRENT_THEME.BG_TERTIARY)
            cb.pack(anchor="w")
            self.packages["deep-translator"] = var

        if not FastLazyLoader.is_available('psutil'):
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(frame, text="psutil (Systemmonitoring)",
                                variable=var, bg=CURRENT_THEME.BG_PRIMARY,
                                fg=CURRENT_THEME.TEXT_PRIMARY,
                                selectcolor=CURRENT_THEME.BG_TERTIARY)
            cb.pack(anchor="w")
            self.packages["psutil"] = var

        if not self.packages:
            tk.Label(main, text="✅ Alle optionalen Pakete sind bereits installiert.",
                     bg=CURRENT_THEME.BG_PRIMARY, fg=CURRENT_THEME.SUCCESS).pack(pady=20)
            self.dialog.after(2000, self.dialog.destroy)
            return

        tk.Label(main, text="Installationsausgabe:", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY).pack(anchor="w", pady=(10,2))
        self.output_text = scrolledtext.ScrolledText(
            main, height=10, bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.MONOSPACE, wrap=tk.WORD
        )
        self.output_text.pack(fill="both", expand=True, pady=5)

        btn_frame = tk.Frame(main, bg=CURRENT_THEME.BG_PRIMARY)
        btn_frame.pack(fill="x", pady=5)
        install_btn = tk.Button(btn_frame, text="Installieren", command=self.install_selected,
                                bg=CURRENT_THEME.DRAGON_GREEN, fg=CURRENT_THEME.TEXT_PRIMARY,
                                font=DragonFonts.BUTTON, padx=15)
        install_btn.pack(side="left")
        close_btn = tk.Button(btn_frame, text="Schließen", command=self.dialog.destroy,
                              bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY)
        close_btn.pack(side="right")

    def install_selected(self):
        packages = [pkg for pkg, var in self.packages.items() if var.get()]
        if not packages:
            return
        self.output_text.insert("end", f"Starte Installation von: {', '.join(packages)}...\n")
        self.dialog.update()

        python_exe = sys.executable
        cmd = [python_exe, "-m", "pip", "install"] + packages

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='ignore'
            )
            for line in process.stdout:
                self.output_text.insert("end", line)
                self.output_text.see("end")
                self.dialog.update()
            process.wait()
            if process.returncode == 0:
                self.output_text.insert("end", "\n✅ Installation erfolgreich!\n")
                self.output_text.insert("end", "Bitte starten Sie das Programm neu, um die neuen Pakete zu nutzen.\n")
            else:
                self.output_text.insert("end", f"\n❌ Fehler bei der Installation (Rückgabecode {process.returncode})\n")
        except Exception as e:
            self.output_text.insert("end", f"\n❌ Ausnahmefehler: {e}\n")


# -----------------------------------------------------------------------------
# DRAGON WHISPERER GUI (überarbeitet mit Thread-Sicherheit)
# -----------------------------------------------------------------------------
class DragonWhispererGUI:
    class RateLimiter:
        def __init__(self, max_updates_per_second: int = 30) -> None:
            self.min_interval = 1.0 / max_updates_per_second
            self.last_calls: Dict[str, float] = {}
            self._lock = threading.RLock()

        def can_update(self, update_type: str = "default") -> bool:
            with self._lock:
                now = time.time()
                if update_type not in self.last_calls:
                    self.last_calls[update_type] = 0.0
                last = self.last_calls[update_type]
                if now - last >= self.min_interval:
                    self.last_calls[update_type] = now
                    return True
                return False

        def reset(self, update_type: Optional[str] = None) -> None:
            with self._lock:
                if update_type is None:
                    self.last_calls.clear()
                elif update_type in self.last_calls:
                    del self.last_calls[update_type]

    def __init__(self) -> None:
        self._gui_update_limiter = self.RateLimiter(max_updates_per_second=30)
        self._shutting_down = False
        self._exit_dialog_active = False
        self.is_processing = False
        self.subtitle_mode = False
        self.exit_confirmed = False
        self.current_stream_info: Optional[StreamInfo] = None
        self.current_video_language: Optional[str] = None
        self._progress_bar_started = False

        if not GUI_AVAILABLE:
            logger.error("❌ Tkinter nicht verfügbar. Versuche Fallback...")
            self._try_fallback_gui()
            return

        try:
            self.settings = AppSettings.load_from_file()
            if not self.settings.last_url:
                self.settings.last_url = ""
            self.advanced_settings = AdvancedSettings.load_from_file()
            self.advanced_settings.repair()
            validation_issues = self.advanced_settings.validate()
            if validation_issues:
                logger.warning(f"⚠️ Settings validation issues: {validation_issues}")
            logger.info(f"✅ AdvancedSettings ready: SAMPLE_RATE={self.advanced_settings.config.SAMPLE_RATE}, "
                        f"CHANNELS={self.advanced_settings.config.CHANNELS}, "
                        f"CHUNK_SIZE_BYTES={self.advanced_settings.config.CHUNK_SIZE_BYTES}")
        except Exception as e:
            logger.warning(f"⚠️ Settings load failed: {e}, using defaults")
            self.settings = AppSettings()
            self.settings.last_url = ""
            self.advanced_settings = AdvancedSettings()

        # NEW: Cookie-Hinweis anzeigen, wenn noch nicht gezeigt
        if not self.settings.cookies_notice_shown and self.settings.use_browser_cookies:
            # Wir zeigen den Dialog später, wenn das root-Fenster existiert
            self._show_cookie_notice = True
        else:
            self._show_cookie_notice = False

        if self.settings.theme == "light":
            self.current_theme = LightTheme()
        else:
            self.current_theme = DarkTheme()

        global CURRENT_THEME
        CURRENT_THEME = self.current_theme

        self.demo_mode = not WHISPER_AVAILABLE
        self.layout_mode = getattr(self.settings, 'layout_mode', 'vertical')
        self.current_language = getattr(self.settings, 'default_language', 'de')

        self._translation_reset_counter = 0
        self.progress_dialog: Optional[ProgressDialog] = None

        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except (tk.TclError, RuntimeError) as e:
            raise RuntimeError(f"Tkinter Fehler: {e}")

        self._batch_update_interval = 150
        self._last_batch_update = 0.0
        self._last_gui_update_time = 0.0
        self._processing_lock = threading.Lock()

        self.transcript_history: Deque[ExcellenceTranscriptionResult] = deque(maxlen=1000)
        self.translation_history: Deque[ExcellenceTranslationResult] = deque(maxlen=500)
        self._last_transcription_text = ""
        self._last_translation_text = ""

        self.performance_monitor = SimplePerformanceTracker()
        self.gui_queue: queue.Queue = queue.Queue(maxsize=200)
        self._text_update_queue: queue.Queue = queue.Queue(maxsize=150)

        try:
            self.controller = WhisperController(gui_ref=self)
        except Exception as e:
            logger.error(f"❌ Controller Fehler: {e}")
            self._show_error_and_exit(f"Controller Fehler: {e}")
            return

        try:
            self.layout = WhisperLayoutManager(gui_ref=self)
        except Exception as e:
            logger.error(f"❌ Layout Fehler: {e}")
            self._show_error_and_exit(f"Layout Fehler: {e}")
            return

        try:
            # NEW: use_browser_cookies an StreamManager übergeben
            self.stream_manager = StreamManager(enable_debug=(DEBUG_LEVEL >= 1), use_browser_cookies=self.settings.use_browser_cookies)
            self.ffmpeg_manager = ExcellenceFFmpegManager(self.advanced_settings.config, self.stream_manager)
            if WHISPER_AVAILABLE:
                self.transcription_engine = ExcellenceTranscriptionEngine(self.advanced_settings)
            else:
                self.transcription_engine = DummyTranscriptionEngine(self.advanced_settings)

            if TRANSLATOR_AVAILABLE:
                self.translation_engine = ExcellenceTranslationEngine(self.current_language, self.advanced_settings)
            else:
                self.translation_engine = DummyTranslationEngine(self.current_language, self.advanced_settings)

            self.audio_processor = ExcellenceAudioProcessor(
                controller_ref=self.controller,
                ffmpeg_manager=self.ffmpeg_manager,
                advanced_settings=self.advanced_settings
            )
            self.audio_processor.set_engines(self.transcription_engine, self.translation_engine)
            self.export_manager = ExportManager()
            self.language_detector = LanguageDetector(self.transcription_engine)
            self.resource_manager = ResourceManager()
            self.memory_manager = ExcellenceMemoryManager()

            if IS_LINUX:
                self.performance_optimizer = LinuxPerformanceOptimizer(gui_ref=self)

            self._register_signal_handlers()
        except (ImportError, OSError, RuntimeError) as e:
            logger.error(f"❌ Engine Initialisierung Fehler: {e}")
            self._show_error_and_exit(f"Engine Fehler: {e}")
            return

        try:
            self.layout.setup_gui()
            self._setup_callbacks()
            self.root.after(100, self._start_gui_updaters)

            if hasattr(self, 'url_entry') and self.url_entry.winfo_exists():
                self.url_entry.delete(0, 'end')
                self.url_entry.insert(0, self.settings.last_url)
            else:
                def set_initial_url() -> None:
                    if hasattr(self, 'url_entry') and self.url_entry.winfo_exists():
                        self.url_entry.delete(0, 'end')
                        self.url_entry.insert(0, self.settings.last_url)
                self.root.after(200, set_initial_url)

            self.root.deiconify()
            self.root.title("🐉 Dragon Whisperer")
        except Exception as e:
            logger.error(f"❌ GUI Setup Fehler: {e}")
            self._show_error_and_exit(f"GUI konnte nicht erstellt werden: {e}")
            return

        # NEW: Cookie-Hinweis nach GUI-Setup anzeigen
        if self._show_cookie_notice:
            self.root.after(500, self._show_cookie_notice_dialog)

        self._bind_shortcuts()
        self.root.after(1000, self._start_system_monitoring)
        self.root.after(2000, self._final_initialization_check)
        self._schedule_gui_health_check()

    def _show_cookie_notice_dialog(self) -> None:
        """Zeigt den Datenschutzhinweis für Browser-Cookies."""
        result = DarkMessageBox.askyesno(
            "Datenschutzhinweis",
            "Dragon Whisperer kann auf gespeicherte Browser-Cookies zugreifen, "
            "um YouTube-Streams zuverlässiger abzurufen. Dies kann Ihre Privatsphäre "
            "beeinträchtigen.\n\nMöchten Sie die Nutzung von Browser-Cookies erlauben?\n\n"
            "(Sie können diese Einstellung später in den erweiterten Einstellungen ändern.)",
            parent=self.root
        )
        self.settings.use_browser_cookies = result
        self.settings.cookies_notice_shown = True
        self.settings.save_to_file()
        # Einstellung an StreamManager weitergeben
        if hasattr(self, 'stream_manager'):
            self.stream_manager.use_browser_cookies = result

    def _schedule_gui_health_check(self) -> None:
        if not self._shutting_down and hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(30000, self._perform_gui_health_check)

    def _perform_gui_health_check(self) -> None:
        try:
            checks: List[str] = []
            check_start = time.time()
            self.root.update_idletasks()
            responsiveness_time = time.time() - check_start
            if responsiveness_time > 0.5:
                checks.append(f"⚠️ GUI responsiveness slow: {responsiveness_time:.1f}s")
            if hasattr(self, 'memory_manager'):
                mem_stats = self.memory_manager.get_memory_stats()
                if mem_stats.get("process_usage_percent", 0) > 80:
                    checks.append("⚠️ High memory usage")
            if hasattr(self, 'gui_queue'):
                qsize = self.gui_queue.qsize()
                if qsize > 50:
                    checks.append(f"⚠️ GUI queue backlog: {qsize} items")
                    self._cleanup_queue(self.gui_queue, 20)  # hier wird die neue Methode aufgerufen
            active_threads = threading.enumerate()
            if len(active_threads) > 15:
                checks.append(f"⚠️ Many active threads: {len(active_threads)}")
            try:
                cache_stats = get_cache_stats()
                for cache_name, stats in cache_stats.items():
                    if stats.get("total_entries", 0) > stats.get("max_size", 100) * 0.9:
                        checks.append(f"⚠️ {cache_name} nearly full")
            except Exception:
                pass
            if checks:
                if len(checks) > 0:
                    logger.warning(f"🔍 GUI Health Check Issues: {checks[:3]}")
            if "memory usage" in str(checks) and hasattr(self, 'memory_manager'):
                self.memory_manager._aggressive_cleanup()
        except Exception as e:
            logger.warning(f"⚠️ Health check error: {e}")
        finally:
            self._schedule_gui_health_check()

    def _register_signal_handlers(self) -> None:
        try:
            logger.info("🔧 Registering cleanup handlers with SignalHandler...")
            SignalHandler.register_cleanup(
                self._safe_stop_all_processes,
                name="StopAllProcesses",
                priority=ShutdownPriority.CRITICAL,
                timeout=3.0,
                essential=True
            )
            SignalHandler.register_cleanup(
                self.audio_processor.dispose,
                name="AudioProcessorDispose",
                priority=ShutdownPriority.HIGH,
                timeout=2.0
            )
            if hasattr(self, 'ffmpeg_manager') and self.ffmpeg_manager:
                SignalHandler.register_cleanup(
                    self.ffmpeg_manager.dispose,
                    name="FFmpegManagerDispose",
                    priority=ShutdownPriority.HIGH,
                    timeout=2.0
                )
                logger.info("   ✅ Registered FFmpegManager cleanup")
            SignalHandler.register_cleanup(
                self.transcription_engine.dispose,
                name="TranscriptionEngineDispose",
                priority=ShutdownPriority.MEDIUM,
                timeout=1.0
            )
            SignalHandler.register_cleanup(
                self.translation_engine.dispose,
                name="TranslationEngineDispose",
                priority=ShutdownPriority.MEDIUM,
                timeout=1.0
            )
            SignalHandler.register_cleanup(
                self.memory_manager.dispose,
                name="MemoryManagerDispose",
                priority=ShutdownPriority.LOW
            )
            SignalHandler.register_cleanup(
                self.resource_manager.cleanup,
                name="ResourceManagerCleanup",
                priority=ShutdownPriority.LOW
            )
            SignalHandler.register_cleanup(
                self.stream_manager.dispose,
                name="StreamManagerDispose",
                priority=ShutdownPriority.LOW
            )
            if IS_LINUX and hasattr(self, 'performance_optimizer'):
                SignalHandler.register_cleanup(
                    self._safe_linux_optimizer_cleanup,
                    name="LinuxOptimizerCleanup",
                    priority=ShutdownPriority.LOW,
                    timeout=1.0
                )
            SignalHandler.register_cleanup(
                lambda: (transcription_cache.clear(), translation_cache.clear(), audio_cache.clear()),
                name="ClearGlobalCaches",
                priority=ShutdownPriority.LOW
            )
            SignalHandler.register_cleanup(
                self._cleanup_queues,
                name="CleanupQueues",
                priority=ShutdownPriority.LOW
            )
            try:
                import torch
                if torch.cuda.is_available():
                    SignalHandler.register_cleanup(
                        torch.cuda.empty_cache,
                        name="GPUMemoryCleanup",
                        priority=ShutdownPriority.LOW,
                        timeout=1.0
                    )
                    logger.info("   ✅ Registered GPU cleanup")
            except ImportError:
                pass
            count = sum(len(ops) for ops in SignalHandler._cleanup_operations.values())
            logger.info(f"✅ Registered {count} cleanup handlers")
            SignalHandler.setup(verbose=False, silent=True)
        except Exception as e:
            logger.warning(f"⚠️ SignalHandler registration error: {e}")

    def _safe_stop_all_processes(self) -> None:
        logger.info("🛑 Safely stopping all processes...")
        self._shutting_down = True
        self.is_processing = False
        if hasattr(self, 'controller'):
            try:
                self.controller._shutdown_event.set()
                self.controller._stop_requested = True
                if hasattr(self.controller, '_processing_active'):
                    self.controller._processing_active.clear()
            except Exception as e:
                logger.warning(f"⚠️ Controller stop error: {e}")
        if hasattr(self, 'audio_processor'):
            try:
                self.audio_processor._processing = False
                if hasattr(self.audio_processor, '_stop_event'):
                    self.audio_processor._stop_event.set()
            except Exception as e:
                logger.warning(f"⚠️ Audio processor stop error: {e}")
        if hasattr(self, 'ffmpeg_manager'):
            try:
                self.ffmpeg_manager._shutting_down = True
                if hasattr(self.ffmpeg_manager, '_processes'):
                    import os
                    import signal
                    for process_id, process_info in list(self.ffmpeg_manager._processes.items()):
                        try:
                            process = process_info.get('process')
                            if process and hasattr(process, 'pid'):
                                try:
                                    if IS_WINDOWS:
                                        process.terminate()
                                    else:
                                        os.kill(process.pid, signal.SIGTERM)
                                    time.sleep(0.1)
                                except Exception:
                                    try:
                                        if IS_WINDOWS:
                                            process.kill()
                                        else:
                                            os.kill(process.pid, signal.SIGKILL)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"⚠️ FFmpeg stop error: {e}")
        logger.info("✅ All processes stopped")

    def _safe_linux_optimizer_cleanup(self) -> None:
        if not IS_LINUX or not hasattr(self, 'performance_optimizer'):
            return
        logger.info("🐧 Safe Linux optimizer cleanup...")
        try:
            gui_exists = False
            try:
                if hasattr(self, 'root') and self.root.winfo_exists():
                    gui_exists = True
            except Exception:
                gui_exists = False
            if gui_exists:
                try:
                    self.performance_optimizer.restore_normal_mode()
                except Exception as e:
                    logger.warning(f"⚠️ restore_normal_mode failed: {e}")
                    try:
                        self.performance_optimizer.dispose()
                    except Exception:
                        pass
            else:
                try:
                    self.performance_optimizer.dispose()
                except Exception as e:
                    logger.warning(f"⚠️ dispose failed: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Linux optimizer cleanup error: {e}")

    def _cleanup_queues(self) -> None:
        logger.info("🗑️ Cleaning up queues...")
        try:
            if hasattr(self, 'gui_queue'):
                count = 0
                while not self.gui_queue.empty() and count < 100:
                    try:
                        self.gui_queue.get_nowait()
                        count += 1
                    except Exception:
                        break
                if count > 0:
                    logger.info(f"  Cleared GUI queue: {count} items")
        except Exception as e:
            logger.warning(f"⚠️ GUI queue cleanup error: {e}")
        try:
            if hasattr(self, '_text_update_queue'):
                count = 0
                while not self._text_update_queue.empty() and count < 100:
                    try:
                        self._text_update_queue.get_nowait()
                        count += 1
                    except Exception:
                        break
                if count > 0:
                    logger.info(f"  Cleared text queue: {count} items")
        except Exception as e:
            logger.warning(f"⚠️ Text queue cleanup error: {e}")

    def _safe_exit_dialog(self) -> None:
        if self._shutting_down or self._exit_dialog_active:
            return
        self._exit_dialog_active = True
        try:
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                self._direct_shutdown()
                return
            result = DarkMessageBox.askyesno(
                "🐉 Dragon Whisperer - Beenden",
                "Programm wirklich beenden?\n\n"
                "● Laufende Transkriptionen werden gestoppt\n"
                "● Nicht gespeicherte Daten gehen verloren\n\n"
                "Sicher beenden?",
                parent=self.root
            )
            if result:
                logger.info("✅ User confirmed exit - shutting down...")
                self._direct_shutdown()
            else:
                logger.info("↩️ Exit cancelled by user")
                self._exit_dialog_active = False
        except tk.TclError:
            logger.warning("⚠️ GUI destroyed, performing direct shutdown...")
            self._direct_shutdown()
        except Exception as e:
            logger.warning(f"⚠️ Exit dialog error: {e}")
            self._direct_shutdown()
        finally:
            if not self._shutting_down:
                self._exit_dialog_active = False

    def _direct_shutdown(self) -> None:
        if self._shutting_down:
            logger.warning("⚠️ Shutdown already in progress, skipping...")
            return
        logger.info("🔧 Performing confirmed shutdown...")
        self._shutting_down = True
        self._safe_stop_all_processes()
        time.sleep(0.3)
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.title("🐉 Dragon Whisperer - Beendet...")
                self.root.update_idletasks()
                self.root.quit()
        except tk.TclError:
            pass
        except Exception as e:
            logger.warning(f"⚠️ GUI shutdown error: {e}")

    # neu: Methode zum Bereinigen der Queue
    def _cleanup_queue(self, queue_obj: queue.Queue, max_size: int) -> None:
        """Reduziere die Queue auf max_size, behalte wichtige Nachrichten (status, error)."""
        if not queue_obj or queue_obj.qsize() <= max_size:
            return
        try:
            items = []
            # Zuerst alle Nachrichten aus der Queue holen
            while not queue_obj.empty():
                try:
                    items.append(queue_obj.get_nowait())
                except queue.Empty:
                    break
            # Nachrichten sortieren: wichtige zuerst, dann die neuesten
            important = []
            others = []
            for item in items:
                if isinstance(item, tuple) and len(item) == 2 and item[0] in ('status', 'error', 'file_finished'):
                    important.append(item)
                else:
                    others.append(item)
            # Behalte alle wichtigen und die neuesten others bis max_size erreicht ist
            kept = important + others[-(max_size - len(important)):] if len(important) < max_size else important[:max_size]
            for item in kept:
                try:
                    queue_obj.put_nowait(item)
                except queue.Full:
                    break
            logger.debug(f"🧹 Queue cleaned: {len(items)} -> {len(kept)} items")
        except Exception as e:
            logger.warning(f"⚠️ Queue cleanup error: {e}")

    def run(self) -> None:
        try:
            self.root.title("🐉 Dragon Whisperer")
            self._shutting_down = False
            self._exit_dialog_active = False
            self.root.protocol("WM_DELETE_WINDOW", self._safe_exit_dialog)
            if hasattr(self, 'exit_button'):
                self.exit_button.config(command=self._safe_exit_dialog)
            logger.info("🚀 Starting Dragon Whisperer (with exit confirmation)...")
            if not SignalHandler._setup_complete:
                try:
                    SignalHandler.setup(verbose=False, silent=True)
                except Exception:
                    pass
            self.root.mainloop()
            logger.info("✅ Main loop exited normally")
        except KeyboardInterrupt:
            logger.info("\n🛑 Interrupted by user - showing exit dialog...")
            self._safe_exit_dialog()
        except SystemExit as e:
            logger.info("\n🔧 System exit requested")
            raise e
        except Exception as e:
            logger.error(f"💥 Critical error: {type(e).__name__}: {e}")
            self._direct_shutdown()

    def _post_mainloop_cleanup(self) -> None:
        logger.info("🧹 Post-mainloop cleanup (no GUI access)...")
        self._shutting_down = True
        self.is_processing = False
        self._cleanup_queues()
        try:
            transcription_cache.clear()
            translation_cache.clear()
            audio_cache.clear()
        except Exception as e:
            logger.warning(f"⚠️ Cache cleanup error: {e}")
        gc.collect()
        logger.info("✅ Post-mainloop cleanup completed")

    def _minimal_emergency_cleanup(self) -> None:
        logger.info("🆘 MINIMAL emergency cleanup...")
        self._shutting_down = True
        self.is_processing = False
        try:
            if hasattr(self, 'ffmpeg_manager'):
                import os
                import signal
                if hasattr(self.ffmpeg_manager, '_processes'):
                    for pid, info in list(self.ffmpeg_manager._processes.items()):
                        try:
                            process = info.get('process')
                            if process and hasattr(process, 'pid'):
                                try:
                                    if IS_WINDOWS:
                                        process.kill()
                                    else:
                                        os.kill(process.pid, signal.SIGKILL)
                                except Exception:
                                    pass
                        except Exception:
                            pass
        except Exception:
            pass
        gc.collect()

    def safe_controller_stop(self) -> None:
        if self._shutting_down:
            return
        logger.info("🛑 Safe controller stop...")
        self._safe_stop_all_processes()

    def _cleanup_queue(self, queue_obj: queue.Queue, max_size: int) -> None:
        # bereits oben definiert – für den Aufruf in _perform_gui_health_check
        pass  # wird hier nicht benötigt, da die Methode bereits vorhanden

    @excellence_gui_operation
    def select_file_dark(self) -> None:
        try:
            filename = filedialog.askopenfilename(
                title="🎬 Select Audio/Video File - Dragon Whisperer",
                filetypes=[
                    ("Media files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mkv *.mov *.flac"),
                    ("All files", "*.*"),
                ],
            )
            if filename:
                file_url = f"file://{filename}"
                self.url_entry.delete(0, "end")
                self.url_entry.insert(0, file_url)
                self.update_status(f"📁 File selected: {os.path.basename(filename)}")
                def async_language_detection() -> None:
                    try:
                        self.analyze_video_language(filename)
                    except Exception:
                        pass
                detection_thread = threading.Thread(target=async_language_detection, daemon=True)
                if hasattr(self, 'resource_manager'):
                    self.resource_manager.register_thread(detection_thread)
                detection_thread.start()
                info = StreamInfoExtractor().extract_stream_info(file_url)
                self.update_stream_info(info)
        except Exception as e:
            self.update_status(f"❌ File selection failed: {e}")

    @excellence_gui_operation
    def paste_url(self) -> None:
        try:
            clipboard = self.root.clipboard_get().strip()
            if clipboard:
                try:
                    cleaned_url = self.clean_and_validate_url(clipboard)
                except ValueError as e:
                    self.update_status(f"❌ Ungültige URL: {e}")
                    return
                self.url_entry.delete(0, "end")
                self.url_entry.insert(0, cleaned_url)
                self.update_status("📋 URL eingefügt")
                if cleaned_url.startswith("file://"):
                    file_path = cleaned_url[7:]
                    if os.path.exists(file_path):
                        def async_detection() -> None:
                            try:
                                self.analyze_video_language(file_path)
                            except Exception:
                                pass
                        detection_thread = threading.Thread(target=async_detection, daemon=True)
                        if hasattr(self, 'resource_manager'):
                            self.resource_manager.register_thread(detection_thread)
                        detection_thread.start()
            else:
                self.update_status("❌ Zwischenablage ist leer")
        except tk.TclError:
            self.update_status("❌ Konnte nicht auf Zwischenablage zugreifen")
        except Exception as e:
            self.update_status(f"❌ Fehler beim Einfügen: {str(e)[:50]}")

    def clean_and_validate_url(self, url: str) -> str:
        url = url.strip()
        if not url:
            raise ValueError("URL cannot be empty")
        # Zuerst nur bereinigen, dann validieren
        url = PlatformUtils.sanitize_url(url)
        if url.startswith("file://"):
            ok, real_path = PlatformUtils.validate_file_path(url)
            if not ok:
                raise ValueError(real_path)
            file_path = real_path
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            return url
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            raise ValueError("Invalid URL format")
        if len(url) < 10:
            raise ValueError("URL too short")
        if " " in url:
            raise ValueError("URL cannot contain spaces")
        return url

    def analyze_video_language(self, file_path: str) -> None:
        if hasattr(self, 'language_info_label'):
            self.root.after(0, lambda: self.language_info_label.config(text="🔍 Analyzing..."))

        def language_detection_worker() -> None:
            try:
                detection_result = self.language_detector.detect_video_language(file_path)
                logger.debug(f"Language detection result: {detection_result}")
                def update_result() -> None:
                    if hasattr(self, 'language_info_label'):
                        if 'error' in detection_result:
                            self.language_info_label.config(text=f"❌ {detection_result['error']}")
                        elif 'info' in detection_result:
                            self.language_info_label.config(text=f"ℹ️ {detection_result['info']}")
                        else:
                            language_name = detection_result['language_name']
                            confidence = detection_result['confidence']
                            self.current_video_language = detection_result['detected_language']
                            language_icons = {
                                'zh': '㊗️', 'ja': '🗾', 'ko': '₩',
                                'th': '🇹🇭', 'vi': '🇻🇳',
                            }
                            icon = language_icons.get(self.current_video_language, '✅')
                            display_text = f"{icon} {language_name} ({confidence:.0%})"
                            self.language_info_label.config(text=display_text)
                if hasattr(self, 'root'):
                    self.root.after(0, update_result)
            except Exception as e:
                logger.error(f"Language detection exception: {e}")
                def update_error() -> None:
                    if hasattr(self, 'language_info_label'):
                        self.language_info_label.config(text="❌ Analysis failed")
                if hasattr(self, 'root'):
                    self.root.after(0, update_error)
        detection_thread = threading.Thread(target=language_detection_worker, daemon=True)
        if hasattr(self, 'resource_manager'):
            self.resource_manager.register_thread(detection_thread)
        detection_thread.start()

    def on_url_change(self, event: Optional[tk.Event] = None) -> None:
        if not hasattr(self, 'url_entry'):
            return
        url = self.url_entry.get().strip()
        if url.startswith("file://"):
            file_path = url[7:]
            if os.path.exists(file_path):
                def async_detection() -> None:
                    try:
                        self.analyze_video_language(file_path)
                    except Exception:
                        pass
                detection_thread = threading.Thread(target=async_detection, daemon=True)
                if hasattr(self, 'resource_manager'):
                    self.resource_manager.register_thread(detection_thread)
                detection_thread.start()
            else:
                if hasattr(self, 'language_info_label'):
                    self.language_info_label.config(text="❌ File not found")
        else:
            if hasattr(self, 'language_info_label'):
                self.language_info_label.config(text="")
            self.current_video_language = None

    def on_language_change(self, event: Optional[tk.Event] = None) -> None:
        try:
            selected_name = self.lang_var.get()
            lang_code: Optional[str] = None
            for name, code in SORTED_LANGUAGES:
                if name == selected_name:
                    lang_code = code
                    break
            if lang_code and lang_code != self.current_language:
                self.current_language = lang_code
                if hasattr(self, 'translation_engine'):
                    self.translation_engine.set_target_language(lang_code)
                lang_display = LANGUAGE_SHORT_CODES.get(lang_code, lang_code)
                if hasattr(self, 'translation_header'):
                    self.translation_header.config(text=f"🌐 Translation ({lang_display})")
                self.update_status(f"🌍 Target language: {selected_name}")
        except Exception:
            pass

    def on_model_change(self, event: Optional[tk.Event] = None) -> None:
        if not hasattr(self, 'model_var'):
            return
        new_model = self.model_var.get()
        # Validierung
        if new_model not in WHISPER_MODELS:
            logger.warning(f"⚠️ Invalid model selected: {new_model}")
            # Auf aktuelles Modell zurücksetzen
            current = self.transcription_engine.get_current_model()
            self.model_var.set(current)
            return
        if not hasattr(self, 'transcription_engine'):
            return
        current_model = self.transcription_engine.get_current_model()
        if new_model == current_model:
            return
        if self.transcription_engine.is_model_loading():
            self.update_status("🔄 Model already loading...")
            return
        success = self.transcription_engine.reload_model(new_model)
        if success:
            self.update_status(f"🔄 Switching to {new_model}...")
            self._check_model_loading_complete(new_model)
        else:
            self.update_status("❌ Model switch failed")
            self.model_var.set(current_model)

    def _check_model_loading_complete(self, target_model: str) -> None:
        if self.transcription_engine.is_model_loading():
            self.root.after(200, lambda: self._check_model_loading_complete(target_model))
        else:
            current = self.transcription_engine.get_current_model()
            if current == target_model:
                self.update_status(f"✅ Model switched to {target_model}")
            else:
                self.update_status("❌ Model switch failed")
                self.model_var.set(current)

    def toggle_translation(self) -> None:
        if hasattr(self, 'translate_toggle') and self.translate_toggle.get():
            if hasattr(self, 'translation_engine'):
                self.translation_engine.set_target_language(self.current_language)
            self.update_status("✅ Translation active")
        else:
            self.update_status("❌ Translation inactive")

    def toggle_subtitle_mode(self) -> None:
        self.subtitle_mode = not self.subtitle_mode
        if hasattr(self, 'audio_processor'):
            self.audio_processor.enable_subtitle_mode(self.subtitle_mode)
        if hasattr(self, 'subtitle_btn'):
            if self.subtitle_mode:
                self.subtitle_btn.config(bg=self.current_theme.SUBTITLE_ACTIVE, fg=self.current_theme.TEXT_PRIMARY)
                self.update_status("🎬 SUBTITLE MODE: Timestamps activated")
            else:
                self.subtitle_btn.config(bg=self.current_theme.SUBTITLE_INACTIVE, fg=self.current_theme.TEXT_PRIMARY)
                self.update_status("📝 NORMAL MODE: Continuous text")

    def _on_start_click(self):
        if self.is_processing:
            self.update_status("⚠️ Bereits aktiv")
            return

        try:
            self.start_button.config(state="disabled")
        except Exception:
            pass
        self.controller.start_processing()

    def toggle_layout(self) -> None:
        try:
            logger.info(f"🔄 Starting layout toggle from {self.layout_mode}")
            old_transcript = ""
            old_translation = ""
            try:
                if hasattr(self, 'transcript_text') and self.transcript_text:
                    old_transcript = self.transcript_text.get('1.0', 'end-1c')
                    logger.info(f"  📝 Saved transcript: {len(old_transcript)} chars")
            except (tk.TclError, AttributeError) as e:
                logger.warning(f"  ⚠️ Could not save transcript: {e}")
            try:
                if hasattr(self, 'translation_text') and self.translation_text:
                    old_translation = self.translation_text.get('1.0', 'end-1c')
                    logger.info(f"  📝 Saved translation: {len(old_translation)} chars")
            except (tk.TclError, AttributeError) as e:
                logger.warning(f"  ⚠️ Could not save translation: {e}")
            if self.layout_mode == "vertical":
                self.layout_mode = "horizontal"
                new_mode_text = "Horizontal"
            else:
                self.layout_mode = "vertical"
                new_mode_text = "Vertical"
            logger.info(f"  🔄 Switching to: {self.layout_mode}")
            if hasattr(self, 'settings'):
                self.settings.layout_mode = self.layout_mode
                try:
                    self.settings.save_to_file()
                except Exception as e:
                    logger.warning(f"  ⚠️ Settings save error: {e}")
            self.update_status(f"🔄 Switching to {new_mode_text} layout...")
            if hasattr(self, 'layout'):
                new_transcript, new_translation = self.layout.create_text_areas()
                if new_transcript and old_transcript:
                    try:
                        new_transcript.insert('1.0', old_transcript)
                        logger.info("  ✅ Restored transcript to new widget")
                    except Exception as e:
                        logger.warning(f"  ❌ Failed to restore transcript: {e}")
                if new_translation and old_translation:
                    try:
                        new_translation.insert('1.0', old_translation)
                        logger.info("  ✅ Restored translation to new widget")
                    except Exception as e:
                        logger.warning(f"  ❌ Failed to restore translation: {e}")
            self.update_status(f"✅ {new_mode_text} layout active")
            logger.info("✅ Layout toggle completed successfully")
        except Exception as e:
            logger.error(f"❌ CRITICAL Layout toggle error: {e}")
            self.update_status("❌ Layout change failed")
            try:
                self.layout_mode = "vertical"
                if hasattr(self, 'layout'):
                    self.layout.create_text_areas()
            except Exception:
                pass

    def clear_all(self) -> None:
        try:
            if hasattr(self, 'transcript_text'):
                self.transcript_text.delete("1.0", "end")
            if hasattr(self, 'translation_text'):
                self.translation_text.delete("1.0", "end")
        except Exception:
            pass
        self.transcript_history.clear()
        self.translation_history.clear()
        if hasattr(self, 'memory_manager'):
            self.memory_manager.clear_component('transcript')
            self.memory_manager.clear_component('translation')
        self._last_transcription_text = ""
        self._last_translation_text = ""
        self._translation_reset_counter = 0
        self.update_status("🗑️ Cleared & optimizations reset")

    def save_transcript(self) -> None:
        try:
            if not self.transcript_history:
                DarkMessageBox.showinfo("WARNING", "No transcriptions available to save.", self.root)
                return
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("SRT subtitles", "*.srt"),
                    ("WebVTT", "*.vtt"),
                    ("JSON", "*.json"),
                    ("Word document", "*.docx"),
                    ("All files", "*.*"),
                ],
            )
            if not filename:
                return
            file_ext = Path(filename).suffix.lower()
            success = False
            if file_ext == ".srt":
                success = self.export_manager.export_subtitles(
                    list(self.transcript_history), None, "srt", filename
                )
            elif file_ext == ".vtt":
                success = self.export_manager.export_subtitles(
                    list(self.transcript_history), None, "vtt", filename
                )
            elif file_ext == ".json":
                success = self.export_manager.export_json(
                    list(self.transcript_history), list(self.translation_history), filename
                )
            elif file_ext == ".docx":
                success = self.export_manager.export_docx(list(self.transcript_history), filename)
            else:
                with open(filename, "w", encoding="utf-8") as f:
                    if self.current_stream_info:
                        f.write("=== STREAM INFORMATION ===\n")
                        f.write(f"Title: {self.current_stream_info.title}\n")
                        f.write(f"Uploader: {self.current_stream_info.uploader}\n")
                        f.write(f"Duration: {self.current_stream_info.duration}\n")
                        f.write(f"Platform: {self.current_stream_info.platform}\n")
                        f.write(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("=== TRANSCRIPT ===\n")
                    if hasattr(self, 'transcript_text'):
                        f.write(self.transcript_text.get("1.0", "end-1c"))
                    f.write("\n\n=== TRANSLATION ===\n")
                    if hasattr(self, 'translation_text'):
                        f.write(self.translation_text.get("1.0", "end-1c"))
                success = True
            if success:
                self.update_status(f"💾 Saved: {os.path.basename(filename)}")
            else:
                self.update_status("❌ Export failed")
        except Exception as e:
            self.update_status(f"❌ Save failed: {e}")

    def export_subtitles(self) -> None:
        if not hasattr(self, 'audio_processor') or not self.audio_processor._timed_transcriptions:
            DarkMessageBox.showinfo(
                "WARNING",
                "No subtitle data available.\n\n"
                "Tip: First activate '🎬 Subtitle mode' "
                "and start a transcription.",
                self.root,
            )
            return
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".srt",
                filetypes=[
                    ("SRT subtitles", "*.srt"),
                    ("VTT subtitles", "*.vtt"),
                    ("All files", "*.*"),
                ],
                title="Export subtitles",
            )
            if not filename:
                return
            file_ext = Path(filename).suffix.lower()
            format_type = "srt" if file_ext == ".srt" else "vtt"
            with self.audio_processor._subtitle_lock:
                timed_trans = list(self.audio_processor._timed_transcriptions)
                timed_transl = list(self.audio_processor._timed_translations)
            success = self.export_manager.export_subtitles(
                timed_trans,
                timed_transl,
                format=format_type,
                filename=filename,
            )
            if success:
                segment_count = len(timed_trans)
                translation_count = len(timed_transl)
                self.update_status(f"📝 {format_type.upper()} exported: {os.path.basename(filename)}")
                DarkMessageBox.showinfo(
                    "Success",
                    f"Subtitles successfully exported!\n\n"
                    f"• File: {os.path.basename(filename)}\n"
                    f"• Segments: {segment_count}\n"
                    f"• Translations: {translation_count}\n"
                    f"• Format: {format_type.upper()}\n\n"
                    f"Can be directly imported into video editors.",
                    self.root,
                )
            else:
                self.update_status("❌ Subtitle export failed")
        except Exception as e:
            self.update_status(f"❌ Subtitle export failed: {e}")
            DarkMessageBox.showerror("Error", f"Export failed:\n{str(e)}", self.root)

    def show_simple_stats(self) -> None:
        try:
            stats = self.performance_monitor.get_basic_stats()
            try:
                import psutil
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_used = memory.used // (1024**2)
                memory_percent = memory.percent
                health_status = "Healthy" if cpu < 90 and memory_percent < 85 else "Degraded"
            except Exception:
                cpu = 0.0
                memory_used = 0
                memory_percent = 0
                health_status = "Unknown"
            stats_text = f"""📊 STATISTIKEN:

🤖 PERFORMANCE:
⏱️ Runtime: {stats["uptime_minutes"]:.1f} minutes
📝 Transcriptions: {stats["transcriptions"]}
🌐 Translations: {stats["translations"]}
🎯 Cache Hit Rate: {stats["cache_hit_rate"]}

💻 SYSTEM:
🖥️ CPU: {cpu:.1f}%
🧠 RAM: {memory_used}MB ({memory_percent:.1f}%)
⚡ Status: {health_status}

🎬 Subtitle mode: {"Active" if self.subtitle_mode else "Inactive"}
"""
            DarkMessageBox.showinfo("Performance Statistics", stats_text, self.root)
        except Exception as e:
            self.update_status(f"❌ Statistics error: {e}")

    def show_advanced_settings(self) -> None:
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("Advanced Settings")
        settings_dialog.geometry("400x600")
        settings_dialog.configure(bg=self.current_theme.BG_PRIMARY)
        settings_dialog.transient(self.root)
        settings_dialog.grab_set()
        settings_dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - settings_dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - settings_dialog.winfo_height()) // 2
        settings_dialog.geometry(f"+{x}+{y}")
        main_frame = tk.Frame(settings_dialog, bg=self.current_theme.BG_PRIMARY, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        tk.Label(
            main_frame,
            text="Advanced Settings",
            font=DragonFonts.TITLE,
            bg=self.current_theme.BG_PRIMARY,
            fg=self.current_theme.TEXT_PRIMARY,
        ).pack(pady=(0, 20))
        settings_frame = tk.Frame(main_frame, bg=self.current_theme.BG_PRIMARY)
        settings_frame.pack(fill="both", expand=True)

        tk.Label(settings_frame, text="Beam Size:", bg=self.current_theme.BG_PRIMARY,
                 fg=self.current_theme.TEXT_PRIMARY).grid(row=0, column=0, sticky="w", pady=5)
        beam_var = tk.StringVar(value=str(self.advanced_settings.beam_size))
        beam_entry = tk.Entry(settings_frame, textvariable=beam_var,
                              bg=self.current_theme.BG_TERTIARY, fg=self.current_theme.TEXT_PRIMARY)
        beam_entry.grid(row=0, column=1, sticky="ew", pady=5)

        tk.Label(settings_frame, text="Temperature:", bg=self.current_theme.BG_PRIMARY,
                 fg=self.current_theme.TEXT_PRIMARY).grid(row=1, column=0, sticky="w", pady=5)
        temp_var = tk.StringVar(value=str(self.advanced_settings.temperature))
        temp_entry = tk.Entry(settings_frame, textvariable=temp_var,
                              bg=self.current_theme.BG_TERTIARY, fg=self.current_theme.TEXT_PRIMARY)
        temp_entry.grid(row=1, column=1, sticky="ew", pady=5)

        plugin_var = tk.BooleanVar(value=self.settings.enable_plugins)
        plugin_cb = tk.Checkbutton(settings_frame, text="Enable plugins", variable=plugin_var,
                                   bg=self.current_theme.BG_PRIMARY, fg=self.current_theme.TEXT_PRIMARY,
                                   selectcolor=self.current_theme.BG_TERTIARY)
        plugin_cb.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        gpu_var = tk.BooleanVar(value=self.advanced_settings.gpu_acceleration)
        gpu_cb = tk.Checkbutton(settings_frame, text="Enable GPU acceleration", variable=gpu_var,
                                bg=self.current_theme.BG_PRIMARY, fg=self.current_theme.TEXT_PRIMARY,
                                selectcolor=self.current_theme.BG_TERTIARY)
        gpu_cb.grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

        tk.Label(settings_frame, text="VAD Threshold:", bg=self.current_theme.BG_PRIMARY,
                 fg=self.current_theme.TEXT_PRIMARY).grid(row=4, column=0, sticky="w", pady=5)
        vad_threshold_var = tk.DoubleVar(value=self.advanced_settings.vad_threshold)
        vad_threshold_scale = tk.Scale(settings_frame, from_=0.0, to=1.0, resolution=0.05,
                                        orient=tk.HORIZONTAL, variable=vad_threshold_var,
                                        bg=self.current_theme.BG_TERTIARY, fg=self.current_theme.TEXT_PRIMARY,
                                        length=150)
        vad_threshold_scale.grid(row=4, column=1, sticky="ew", pady=5)
        ToolTip(vad_threshold_scale, "Schwellwert für VAD (0.0-1.0). Höher = weniger, aber präzisere Segmente (optimierter Standard: 0.3)")

        tk.Label(settings_frame, text="Min Speech Duration (ms):", bg=self.current_theme.BG_PRIMARY,
                 fg=self.current_theme.TEXT_PRIMARY).grid(row=5, column=0, sticky="w", pady=5)
        vad_min_speech_var = tk.IntVar(value=self.advanced_settings.vad_min_speech_duration_ms)
        vad_min_speech_spin = tk.Spinbox(settings_frame, from_=0, to=2000, increment=50,
                                         textvariable=vad_min_speech_var,
                                         bg=self.current_theme.BG_TERTIARY, fg=self.current_theme.TEXT_PRIMARY)
        vad_min_speech_spin.grid(row=5, column=1, sticky="ew", pady=5)
        ToolTip(vad_min_speech_spin, "Minimale Dauer eines Sprachsegments in ms (optimierter Standard: 200)")

        tk.Label(settings_frame, text="Min Silence Duration (ms):", bg=self.current_theme.BG_PRIMARY,
                 fg=self.current_theme.TEXT_PRIMARY).grid(row=6, column=0, sticky="w", pady=5)
        vad_min_silence_var = tk.IntVar(value=self.advanced_settings.vad_min_silence_duration_ms)
        vad_min_silence_spin = tk.Spinbox(settings_frame, from_=0, to=2000, increment=50,
                                          textvariable=vad_min_silence_var,
                                          bg=self.current_theme.BG_TERTIARY, fg=self.current_theme.TEXT_PRIMARY)
        vad_min_silence_spin.grid(row=6, column=1, sticky="ew", pady=5)
        ToolTip(vad_min_silence_spin, "Minimale Stilledauer zwischen Sprachsegmenten in ms (optimierter Standard: 80)")

        tk.Label(settings_frame, text="Theme:", bg=self.current_theme.BG_PRIMARY,
                 fg=self.current_theme.TEXT_PRIMARY).grid(row=7, column=0, sticky="w", pady=5)
        theme_var = tk.StringVar(value=self.settings.theme)
        theme_combo = ttk.Combobox(settings_frame, textvariable=theme_var,
                                    values=["dark", "light", "system"],
                                    width=10, state="readonly")
        theme_combo.grid(row=7, column=1, sticky="ew", pady=5)

        # NEW: Checkbox für Browser-Cookies
        cookies_var = tk.BooleanVar(value=self.settings.use_browser_cookies)
        cookies_cb = tk.Checkbutton(settings_frame, text="Use browser cookies for YouTube", variable=cookies_var,
                                    bg=self.current_theme.BG_PRIMARY, fg=self.current_theme.TEXT_PRIMARY,
                                    selectcolor=self.current_theme.BG_TERTIARY)
        cookies_cb.grid(row=8, column=0, columnspan=2, sticky="w", pady=5)

        settings_frame.columnconfigure(1, weight=1)

        def save_settings() -> None:
            try:
                self.advanced_settings.beam_size = int(beam_var.get())
                self.advanced_settings.temperature = float(temp_var.get())
                self.settings.enable_plugins = plugin_var.get()
                self.advanced_settings.gpu_acceleration = gpu_var.get()
                self.advanced_settings.vad_threshold = vad_threshold_var.get()
                self.advanced_settings.vad_min_speech_duration_ms = vad_min_speech_var.get()
                self.advanced_settings.vad_min_silence_duration_ms = vad_min_silence_var.get()

                old_theme = self.settings.theme
                self.settings.theme = theme_var.get()
                if old_theme != self.settings.theme:
                    DarkMessageBox.showinfo("Theme geändert",
                                            "Das neue Theme wird nach einem Neustart aktiv.",
                                            self.root)

                # Cookie-Einstellung speichern
                self.settings.use_browser_cookies = cookies_var.get()
                if hasattr(self, 'stream_manager'):
                    self.stream_manager.use_browser_cookies = self.settings.use_browser_cookies

                self.advanced_settings.save_to_file()
                self.settings.save_to_file()
                if not self.advanced_settings.gpu_acceleration:
                    self.transcription_engine.device = "cpu"
                    self.transcription_engine.compute_type = "int8"
                settings_dialog.destroy()
                self.update_status("✅ Settings saved")
            except (ValueError, TypeError) as e:
                DarkMessageBox.showerror("Error", f"Invalid settings: {e}", self.root)

        button_frame = tk.Frame(main_frame, bg=self.current_theme.BG_PRIMARY)
        button_frame.pack(fill="x", pady=(20, 0))
        save_btn = tk.Button(button_frame, text="Save", command=save_settings,
                             bg=self.current_theme.SUCCESS, fg=self.current_theme.TEXT_PRIMARY,
                             relief="flat", padx=15)
        save_btn.pack(side="right", padx=5)
        cancel_btn = tk.Button(button_frame, text="Cancel", command=settings_dialog.destroy,
                               bg=self.current_theme.BG_TERTIARY, fg=self.current_theme.TEXT_PRIMARY,
                               relief="flat", padx=15)
        cancel_btn.pack(side="right", padx=5)

    def _setup_callbacks(self) -> None:
        self.controller.ui_update_fn = self._handle_ui_update
        self.controller.status_update_fn = self._handle_status_update

    def _handle_ui_update(self, component: str, text: str) -> None:
        if not text or not text.strip():
            return
        def update_task() -> None:
            try:
                if component == "transcript" and hasattr(self, "transcript_text"):
                    if self.transcript_text.winfo_exists():
                        self.transcript_text.insert("end", text)
                        if hasattr(self, "transcript_scroll_var") and self.transcript_scroll_var.get():
                            self.transcript_text.see("end")
                        lines = int(self.transcript_text.index("end-1c").split(".")[0])
                        if lines > 400:
                            keep_lines = 300
                            delete_to = f"{lines - keep_lines}.0"
                            self.transcript_text.delete("1.0", delete_to)
                elif component == "translation" and hasattr(self, "translation_text"):
                    if self.translation_text.winfo_exists():
                        self.translation_text.insert("end", text)
                        if hasattr(self, "translation_scroll_var") and self.translation_scroll_var.get():
                            self.translation_text.see("end")
                        lines = int(self.translation_text.index("end-1c").split(".")[0])
                        if lines > 300:
                            keep_lines = 200
                            delete_to = f"{lines - keep_lines}.0"
                            self.translation_text.delete("1.0", delete_to)
            except tk.TclError:
                pass
            except Exception as e:
                logger.warning(f"⚠️ Transcript GUI error: {e}")
        try:
            if hasattr(self, "gui_queue"):
                self.gui_queue.put(("ui_update", update_task))
            else:
                if hasattr(self, "root") and self.root.winfo_exists():
                    self.root.after(0, update_task)
        except Exception as e:
            logger.warning(f"⚠️ Queue put error: {e}")

    def _handle_status_update(self, state_info: Dict[str, Any]) -> None:
        def update_task() -> None:
            try:
                if "status" in state_info and hasattr(self, "status_label"):
                    if self.status_label.winfo_exists():
                        self.status_label.config(text=state_info["status"][:100])
                if "buttons" in state_info:
                    buttons = state_info["buttons"]
                    if hasattr(self, "start_button") and self.start_button.winfo_exists():
                        self.start_button.config(state=buttons.get("start", "normal"))
                    if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                        self.stop_button.config(state=buttons.get("stop", "disabled"))
                elif "processing_state" in state_info:
                    processing = state_info["processing_state"]
                    if hasattr(self, "start_button") and self.start_button.winfo_exists():
                        self.start_button.config(state="disabled" if processing else "normal")
                    if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                        self.stop_button.config(state="normal" if processing else "disabled")
                if "stream_info" in state_info:
                    stream_info = state_info["stream_info"]
                    self.current_stream_info = stream_info
                    if hasattr(self, "stream_title_label") and self.stream_title_label.winfo_exists():
                        title = stream_info.title[:80] + "..." if len(stream_info.title) > 80 else stream_info.title
                        self.stream_title_label.config(text=f"📡 {title}")
                    if hasattr(self, "stream_details_label") and self.stream_details_label.winfo_exists():
                        details = f"👤 {stream_info.uploader}"
                        if stream_info.duration and stream_info.duration != "Live":
                            details += f" | ⏱️ {stream_info.duration}"
                        self.stream_details_label.config(text=details)
                if "processing_state" in state_info:
                    processing = state_info["processing_state"]
                    self.is_processing = processing
                    if hasattr(self, "start_button") and self.start_button.winfo_exists():
                        self.start_button.config(state="disabled" if processing else "normal")
                    if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                        self.stop_button.config(state="normal" if processing else "disabled")

                if state_info.get("file_finished"):
                    logger.info("📂 Dateiende erkannt – öffne Speicherdialog")
                    if self.settings.auto_save_on_completion:
                        self.save_transcript()
                    else:
                        self.update_status("✅ Dateiende – zum Speichern auf 💾 klicken")

            except Exception as e:
                logger.warning(f"⚠️ Status update error: {e}")
        try:
            if hasattr(self, "gui_queue"):
                self.gui_queue.put(("status_update", update_task))
            else:
                if hasattr(self, "root") and self.root.winfo_exists():
                    self.root.after(0, update_task)
        except Exception:
            pass

    def _start_gui_updaters(self) -> None:
        def process_gui_queue() -> None:
            try:
                processed = 0
                max_per_cycle = 10
                while processed < max_per_cycle and hasattr(self, "gui_queue") and not self.gui_queue.empty():
                    try:
                        item = self.gui_queue.get_nowait()
                        if isinstance(item, tuple) and len(item) == 2:
                            msg_type, callback = item
                            if callable(callback):
                                if self._gui_update_limiter.can_update(f"gui_{msg_type}"):
                                    try:
                                        callback()
                                    except Exception as e:
                                        logger.warning(f"⚠️ GUI callback error: {e}")
                        self.gui_queue.task_done()
                        processed += 1
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Queue processing error: {e}")
                # Queue-Größenprüfung und Cleanup
                if hasattr(self, "gui_queue"):
                    qsize = self.gui_queue.qsize()
                    if qsize > 30:
                        self._cleanup_queue(self.gui_queue, 20)
                # Nach dem Verarbeiten erneut planen
                if hasattr(self, "root") and self.root.winfo_exists():
                    self.root.after(100, process_gui_queue)
            except Exception as e:
                logger.error(f"❌ GUI queue processor error: {e}")
                # Trotz Fehler erneut planen, damit die Updates nicht aufhören
                if hasattr(self, "root") and self.root.winfo_exists():
                    self.root.after(100, process_gui_queue)

        def process_text_updates() -> None:
            try:
                if not hasattr(self, "_text_update_queue"):
                    return
                processed = 0
                max_per_cycle = 5
                while processed < max_per_cycle and not self._text_update_queue.empty():
                    try:
                        update_type, text_data = self._text_update_queue.get_nowait()
                        if update_type == "transcript" and hasattr(self, "transcript_text"):
                            if self.transcript_text.winfo_exists():
                                self.transcript_text.insert("end", text_data)
                                if hasattr(self, "transcript_scroll_var") and self.transcript_scroll_var.get():
                                    self.transcript_text.see("end")
                        elif update_type == "translation" and hasattr(self, "translation_text"):
                            if self.translation_text.winfo_exists():
                                self.translation_text.insert("end", text_data)
                                if hasattr(self, "translation_scroll_var") and self.translation_scroll_var.get():
                                    self.translation_text.see("end")
                        self._text_update_queue.task_done()
                        processed += 1
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Text update error: {e}")
                if hasattr(self, "_text_update_queue"):
                    qsize = self._text_update_queue.qsize()
                    if qsize > 25:
                        self._cleanup_queue(self._text_update_queue, 15)
                if hasattr(self, "root") and self.root.winfo_exists():
                    self.root.after(150, process_text_updates)
            except Exception as e:
                logger.error(f"❌ Text update processor error: {e}")
                if hasattr(self, "root") and self.root.winfo_exists():
                    self.root.after(150, process_text_updates)

        if hasattr(self, "root") and self.root.winfo_exists():
            self.root.after(50, process_gui_queue)
            self.root.after(75, process_text_updates)

    def _start_automatic_maintenance(self) -> None:
        def maintenance_worker() -> None:
            if not self._shutting_down and hasattr(self, 'root') and self.root.winfo_exists():
                self._perform_maintenance()
                self.root.after(600000, maintenance_worker)
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(60000, maintenance_worker)

    def _perform_maintenance(self) -> None:
        logger.info("🛠️ Performing automatic maintenance...")
        try:
            expired_counts = clear_expired_cache_entries()
            if any(count > 0 for count in expired_counts.values()):
                logger.info(f"🧹 Cleared expired cache entries: {expired_counts}")
        except Exception:
            pass
        if hasattr(self, 'memory_manager'):
            try:
                self.memory_manager._perform_periodic_maintenance()
            except Exception:
                pass
        if hasattr(self, 'ffmpeg_manager'):
            try:
                self.ffmpeg_manager.cleanup_stale_processes()
            except Exception:
                pass
        gc.collect()
        logger.info("✅ Maintenance completed")

    def handle_transcription(self, result: ExcellenceTranscriptionResult) -> None:
        if not result or not result.text or not result.text.strip():
            return
        current_text = result.text.strip()
        if current_text == self._last_transcription_text:
            return
        self._last_transcription_text = current_text
        self.performance_monitor.log_transcription()
        self.transcript_history.append(result)
        def update_gui() -> None:
            try:
                if hasattr(self, 'transcript_text') and self.transcript_text.winfo_exists():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    detected_lang = getattr(result, 'language', 'unknown')
                    lang_code = LANGUAGE_SHORT_CODES.get(detected_lang, '??')
                    text = f"[{timestamp}] [{lang_code}] {current_text}\n"
                    self.transcript_text.insert('end', text)
                    if hasattr(self, 'transcript_scroll_var') and self.transcript_scroll_var.get():
                        self.transcript_text.see('end')
                    lines = int(self.transcript_text.index('end-1c').split('.')[0])
                    if lines > 400:
                        keep_lines = 300
                        delete_to = f'{lines-keep_lines}.0'
                        self.transcript_text.delete('1.0', delete_to)
            except tk.TclError:
                pass
            except Exception as e:
                logger.warning(f"⚠️ Transcript GUI error: {e}")
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update_gui)
        if (hasattr(self, 'translate_toggle') and self.translate_toggle.get() and
            hasattr(self, 'translation_engine') and self.translation_engine):
            def async_translation() -> None:
                try:
                    source_lang = getattr(result, 'language', 'unknown')
                    if source_lang not in ['unknown', 'auto']:
                        translation = self.translation_engine.translate_text(current_text, source_lang)
                        if translation:
                            self.handle_translation(translation)
                except Exception as e:
                    logger.warning(f"⚠️ Translation error: {e}")
            translation_thread = threading.Thread(target=async_translation, daemon=True)
            translation_thread.start()

    def handle_translation(self, result: ExcellenceTranslationResult) -> None:
        if not result or not result.translated or not result.translated.strip():
            return
        current_text = result.translated.strip()
        if current_text == self._last_translation_text:
            return
        self._last_translation_text = current_text
        self.performance_monitor.log_translation()
        self.translation_history.append(result)
        def update_gui() -> None:
            try:
                if hasattr(self, 'translation_text') and self.translation_text.winfo_exists():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    text = f"[{timestamp}] {current_text}\n"
                    self.translation_text.insert('end', text)
                    if hasattr(self, 'translation_scroll_var') and self.translation_scroll_var.get():
                        self.translation_text.see('end')
                    lines = int(self.translation_text.index('end-1c').split('.')[0])
                    if lines > 300:
                        keep_lines = 200
                        delete_to = f'{lines-keep_lines}.0'
                        self.translation_text.delete('1.0', delete_to)
            except tk.TclError:
                pass
            except Exception as e:
                logger.warning(f"⚠️ Translation GUI error: {e}")
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update_gui)

    def handle_info(self, info_msg: str) -> None:
        def update() -> None:
            self.update_status(f"ℹ️ {info_msg}")
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update)

    def handle_error(self, error_msg: str) -> None:
        def update() -> None:
            self.update_status(f"❌ {error_msg}")
            if self.is_processing:
                self.controller.stop_processing()
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update)

    def update_status(self, message: str) -> None:
        try:
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text=message[:100])
        except Exception:
            pass

    def update_stream_info(self, info: StreamInfo) -> None:
        def update_gui() -> None:
            try:
                self.current_stream_info = info
                if hasattr(self, 'stream_title_label') and self.stream_title_label.winfo_exists():
                    title = info.title[:80] + "..." if len(info.title) > 80 else info.title
                    self.stream_title_label.config(text=f"📡 {title}")
                if hasattr(self, 'stream_details_label') and self.stream_details_label.winfo_exists():
                    details = f"👤 {info.uploader}"
                    if info.duration and info.duration != 'Live':
                        details += f" | ⏱️ {info.duration}"
                    self.stream_details_label.config(text=details)
            except Exception as e:
                logger.warning(f"⚠️ Stream info update error: {e}")
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update_gui)

    def _try_fallback_gui(self) -> None:
        logger.info("ℹ️ Starte im Kommandozeilen-Modus...")
        raise RuntimeError("Bitte installieren Sie Tkinter: pip install tk")

    def _show_error_and_exit(self, message: str) -> None:
        logger.error(f"💥 KRITISCHER FEHLER: {message}")
        try:
            import tkinter.messagebox as mb
            mb.showerror("Dragon Whisperer - Fehler", message)
        except Exception:
            pass
        self._emergency_cleanup()
        sys.exit(1)

    def _show_warning(self, message: str) -> None:
        logger.warning(f"⚠️ WARNUNG: {message}")

    def show_translation_dialog(self) -> None:
        if hasattr(self, 'translation_engine'):
            TranslationDialog(self.root, self.translation_engine)
        else:
            DarkMessageBox.showerror("Error", "Translation engine not available", self.root)

    def show_summarize_dialog(self) -> None:
        if not OLLAMA_AVAILABLE:
            DarkMessageBox.showerror("Fehler", "Ollama nicht verfügbar (requests nicht installiert)", self.root)
            return
        if hasattr(self, 'transcript_text') and self.transcript_text.winfo_exists():
            text = self.transcript_text.get("1.0", "end-1c").strip()
        else:
            text = ""
        if not text:
            DarkMessageBox.showwarning("Kein Text", "Kein Transkriptions-Text zum Zusammenfassen vorhanden.", self.root)
            return
        SummarizeDialog(self.root, text, self)

    def show_install_dialog(self) -> None:
        InstallDependencyDialog(self.root, self)

    def _start_system_monitoring(self) -> None:
        def monitor() -> None:
            try:
                import psutil
            except ImportError:
                if hasattr(self, 'system_info_label'):
                    self.system_info_label.config(text="⚙️ System monitoring unavailable")
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(3000, monitor)
                return
            try:
                cpu = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                ram_used = memory.used // (1024**2)
                ram_total = memory.total // (1024**2)
                gpu_text = ""
                try:
                    if TORCH_AVAILABLE:
                        torch = FastLazyLoader.load("torch")
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            gpu_text = f" | 🎮 VRAM: {vram_used:.1f}/{vram_total:.1f}GB"
                        else:
                            gpu_text = " | 🎮 GPU: ❌"
                    else:
                        gpu_text = " | 🎮 GPU: N/A"
                except Exception as e:
                    if DEBUG_LEVEL >= 2:
                        logger.debug(f"VRAM‑Abfrage fehlgeschlagen: {e}")
                    gpu_text = " | 🎮 GPU: Fehler"
                current_model = "None"
                if hasattr(self, 'transcription_engine'):
                    current_model = self.transcription_engine.get_current_model()
                demo_hint = " | ⚠️ Demo" if getattr(self, 'demo_mode', False) else ""
                if IS_WINDOWS:
                    info = f"🪟 Windows | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}{demo_hint}"
                elif IS_MACOS:
                    if IS_ARM:
                        info = f"🍎 macOS ARM | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}{demo_hint}"
                    else:
                        info = f"🍎 macOS Intel | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}{demo_hint}"
                elif IS_LINUX:
                    info = f"🐧 Linux | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}{demo_hint}"
                else:
                    info = f"🌐 System | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}{demo_hint}"
                if hasattr(self, 'system_info_label'):
                    self.system_info_label.config(text=info)
            except (ImportError, AttributeError):
                if hasattr(self, 'system_info_label'):
                    self.system_info_label.config(text="⚙️ System monitoring unavailable")
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(3000, monitor)
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(1000, monitor)

    def _final_initialization_check(self) -> None:
        logger.info("✅ Dragon Whisperer initialisiert")
        if getattr(self, 'demo_mode', False):
            self.update_status("⚠️ Demo-Modus: Whisper nicht verfügbar – verwende Dummy-Transkriptionen")

    def _emergency_cleanup(self) -> None:
        logger.info("🆘 Emergency cleanup...")
        self._minimal_emergency_cleanup()

    def update_progress(self, processed: int, total: Optional[int], chunks: int) -> None:
        if not hasattr(self, 'progress_bar') or not self.progress_bar.winfo_exists():
            return
        try:
            if total is not None and total > 0:
                if not self.progress_bar.winfo_ismapped():
                    self.progress_bar.pack(side="left", padx=(10, 10))
                percent = (processed / total) * 100
                self.progress_bar.config(mode='determinate', value=percent)
                mb = processed // (1024*1024)
                tb = total // (1024*1024)
                self.progress_label.config(text=f"{mb}MB/{tb}MB")
            else:
                if self.progress_bar.winfo_ismapped():
                    self.progress_bar.pack_forget()
                self.progress_label.config(text=f"Chunks: {chunks}  |  Daten: {processed//1024} KB")
        except tk.TclError:
            pass

    def _reset_progress(self):
        if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)
            self._progress_bar_started = False
        if hasattr(self, 'progress_label'):
            self.progress_label.config(text="")

    def _bind_shortcuts(self):
        mod = "Command" if IS_MACOS else "Control"

        self.root.bind(f'<{mod}-o>', lambda e: self.select_file_dark())
        self.root.bind(f'<{mod}-v>', lambda e: self.paste_url())
        self.root.bind(f'<{mod}-Return>', lambda e: self._on_start_click())
        self.root.bind(f'<{mod}-q>', lambda e: self._safe_exit_dialog())
        self.root.bind(f'<{mod}-s>', lambda e: self.save_transcript())
        self.root.bind(f'<{mod}-l>', lambda e: self.toggle_layout())
        self.root.bind(f'<{mod}-t>', lambda e: self.toggle_translation())
        self.root.bind(f'<{mod}-e>', lambda e: self.export_subtitles())
        self.root.bind(f'<{mod}-Shift-c>', lambda e: self.clear_all())
        self.root.bind(f'<{mod}-h>', lambda e: self.show_shortcuts_help())
        self.root.bind('<F1>', lambda e: self.show_shortcuts_help())

        self.url_entry.bind(f'<{mod}-v>', lambda e: 'break')

    def show_shortcuts_help(self):
        ShortcutsDialog(self.root)


# -----------------------------------------------------------------------------
# TRANSLATION DIALOG (unverändert)
# -----------------------------------------------------------------------------
class TranslationDialog:
    def __init__(self, parent: tk.Widget, translation_engine: ExcellenceTranslationEngine) -> None:
        self.parent = parent
        self.engine = translation_engine
        self.dialog: Optional[tk.Toplevel] = None
        self.create_dialog()

    def create_dialog(self) -> None:
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("🐉 Text Translation")
        self.dialog.geometry("600x500")
        self.dialog.configure(bg=CURRENT_THEME.BG_PRIMARY)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        main = tk.Frame(self.dialog, bg=CURRENT_THEME.BG_PRIMARY, padx=15, pady=15)
        main.pack(fill="both", expand=True)

        tk.Label(main, text="Source text:", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY, font=DragonFonts.PRIMARY).pack(anchor="w")
        self.source_text = scrolledtext.ScrolledText(
            main, height=8, bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.MONOSPACE, wrap=tk.WORD
        )
        self.source_text.pack(fill="x", pady=(0, 10))
        DarkContextMenu(self.source_text)

        lang_frame = tk.Frame(main, bg=CURRENT_THEME.BG_PRIMARY)
        lang_frame.pack(fill="x", pady=5)

        tk.Label(lang_frame, text="From:", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY).pack(side="left", padx=(0, 5))
        self.src_lang_var = tk.StringVar(value="auto")
        src_combo = ttk.Combobox(
            lang_frame, textvariable=self.src_lang_var,
            values=["auto"] + [name for name, code in SORTED_LANGUAGES],
            width=15, state="readonly"
        )
        src_combo.pack(side="left", padx=(0, 20))

        tk.Label(lang_frame, text="To:", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY).pack(side="left", padx=(0, 5))
        self.tgt_lang_var = tk.StringVar(value="German")
        tgt_combo = ttk.Combobox(
            lang_frame, textvariable=self.tgt_lang_var,
            values=[name for name, code in SORTED_LANGUAGES if name != "auto"],
            width=15, state="readonly"
        )
        tgt_combo.pack(side="left")

        btn_frame = tk.Frame(main, bg=CURRENT_THEME.BG_PRIMARY)
        btn_frame.pack(fill="x", pady=10)
        translate_btn = tk.Button(
            btn_frame, text="🌐 Translate", command=self.translate,
            bg=CURRENT_THEME.DRAGON_GREEN, fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.BUTTON, padx=20
        )
        translate_btn.pack(side="left")

        tk.Label(main, text="Translation:", bg=CURRENT_THEME.BG_PRIMARY,
                 fg=CURRENT_THEME.TEXT_PRIMARY, font=DragonFonts.PRIMARY).pack(anchor="w", pady=(10, 0))
        self.target_text = scrolledtext.ScrolledText(
            main, height=8, bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY,
            font=DragonFonts.MONOSPACE, wrap=tk.WORD, state="normal"
        )
        self.target_text.pack(fill="both", expand=True, pady=(5, 0))
        DarkContextMenu(self.target_text)

        close_btn = tk.Button(
            main, text="Close", command=self.dialog.destroy,
            bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY
        )
        close_btn.pack(pady=10)

    def translate(self) -> None:
        source = self.source_text.get("1.0", "end-1c").strip()
        if not source:
            return

        src_name = self.src_lang_var.get().strip()
        tgt_name = self.tgt_lang_var.get().strip()

        valid_language_names = [name for name, code in SORTED_LANGUAGES]

        if src_name not in valid_language_names and src_name != "auto":
            src_name = "auto"
            self.src_lang_var.set("auto")

        if tgt_name not in valid_language_names:
            tgt_name = "German"
            self.tgt_lang_var.set("German")

        src_code = "auto" if src_name == "auto" else next(code for name, code in SORTED_LANGUAGES if name == src_name)
        tgt_code = next(code for name, code in SORTED_LANGUAGES if name == tgt_name)

        try:
            old_target = self.engine.target_lang
            self.engine.set_target_language(tgt_code)
            result = self.engine.translate_text(source, src_code)
            self.engine.set_target_language(old_target)
            if result and result.translated:
                self.target_text.delete("1.0", "end")
                self.target_text.insert("1.0", result.translated)
            else:
                self.target_text.delete("1.0", "end")
                self.target_text.insert("1.0", "(Translation failed)")
        except Exception as e:
            self.target_text.delete("1.0", "end")
            self.target_text.insert("1.0", f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# SHORTCUTS DIALOG (unverändert)
# -----------------------------------------------------------------------------
class ShortcutsDialog:
    def __init__(self, parent):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🐉 Tastenkürzel")
        self.dialog.geometry("500x400")
        self.dialog.configure(bg=CURRENT_THEME.BG_PRIMARY)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = tk.Frame(self.dialog, bg=CURRENT_THEME.BG_PRIMARY, padx=20, pady=20)
        main.pack(fill="both", expand=True)

        tk.Label(main, text="Tastenkürzel", font=("Segoe UI", 14, "bold"),
                 bg=CURRENT_THEME.BG_PRIMARY, fg=CURRENT_THEME.TEXT_PRIMARY).pack(pady=(0,15))

        mod = "Cmd" if IS_MACOS else "Strg"
        shortcuts = [
            (f"{mod}+O", "Datei öffnen"),
            (f"{mod}+V", "URL einfügen"),
            (f"{mod}+Return", "Transkription starten"),
            (f"{mod}+Q", "Programm beenden"),
            (f"{mod}+S", "Transkript speichern"),
            (f"{mod}+L", "Layout umschalten"),
            (f"{mod}+T", "Übersetzung ein/aus"),
            (f"{mod}+E", "Untertitel exportieren"),
            (f"{mod}+Umschalt+C", "Alles löschen"),
            ("F1", "Diese Hilfe anzeigen"),
        ]

        frame = tk.Frame(main, bg=CURRENT_THEME.BG_SECONDARY)
        frame.pack(fill="both", expand=True)

        for i, (key, desc) in enumerate(shortcuts):
            row = tk.Frame(frame, bg=CURRENT_THEME.BG_TERTIARY if i % 2 else CURRENT_THEME.BG_SECONDARY)
            row.pack(fill="x", pady=1)

            key_label = tk.Label(row, text=key, font=("Segoe UI", 10, "bold"),
                                  bg=row['bg'], fg=CURRENT_THEME.TEXT_ACCENT, width=15, anchor="w")
            key_label.pack(side="left", padx=(10,5), pady=5)

            desc_label = tk.Label(row, text=desc, font=("Segoe UI", 10),
                                   bg=row['bg'], fg=CURRENT_THEME.TEXT_PRIMARY, anchor="w")
            desc_label.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        close_btn = tk.Button(main, text="Schließen", command=self.dialog.destroy,
                              bg=CURRENT_THEME.BG_TERTIARY, fg=CURRENT_THEME.TEXT_PRIMARY)
        close_btn.pack(pady=10)

        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")


# -----------------------------------------------------------------------------
# HILFSFUNKTIONEN (Konsolen-Setup, Help, System-Check)
# -----------------------------------------------------------------------------
_original_console_mode: Optional[int] = None
_original_codepage: Optional[int] = None

def _save_console_state() -> None:
    global _original_console_mode, _original_codepage
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            _original_console_mode = mode.value
        _original_codepage = kernel32.GetConsoleOutputCP()
    except Exception:
        pass

def _restore_console_state() -> None:
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        if _original_console_mode is not None:
            handle = kernel32.GetStdHandle(-11)
            kernel32.SetConsoleMode(handle, _original_console_mode)
        if _original_codepage is not None:
            kernel32.SetConsoleOutputCP(_original_codepage)
            kernel32.SetConsoleCP(_original_codepage)
    except Exception:
        pass

_save_console_state()
atexit.register(_restore_console_state)

def _setup_windows_console() -> None:
    if not IS_WINDOWS:
        return
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass

def check_platform_compatibility() -> List[str]:
    issues: List[str] = []
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
    if IS_WINDOWS:
        try:
            import platform
            win_ver = platform.version()
            major_ver = int(win_ver.split(".")[0]) if "." in win_ver else 0
            if major_ver < 10:
                issues.append("Windows 10+ recommended for best experience")
        except Exception:
            pass
    return issues

def debug_script() -> None:
    print("=" * 60)
    print("DRAGON WHISPERER - DEBUG INFORMATION")
    print("=" * 60)
    print(f"\n🔧 Platform: {SYSTEM} {'ARM' if IS_ARM else 'x86'}")
    print(f"🔧 Python: {sys.version.split()[0]}")
    print("\n📦 Dependencies:")
    deps = [
        ("FFmpeg", shutil.which("ffmpeg") is not None),
        ("yt-dlp", shutil.which("yt-dlp") is not None),
        ("faster-whisper", WHISPER_AVAILABLE),
        ("deep-translator", TRANSLATOR_AVAILABLE),
        ("Tkinter", GUI_AVAILABLE),
        ("NumPy", NUMPY_AVAILABLE),
        ("PyTorch", TORCH_AVAILABLE),
        ("SciPy", SCIPY_AVAILABLE),
        ("psutil", "psutil" in sys.modules),
        ("requests", OLLAMA_AVAILABLE),
    ]
    for name, status in deps:
        symbol = "✅" if status else "❌"
        status_text = "Available" if status else "Not available"
        print(f"  {symbol} {name:18} {status_text}")
    print(f"\n📁 Script: {os.path.abspath(__file__)}")
    print(f"📁 CWD: {os.getcwd()}")
    print("=" * 60)

def setup_platform_environment() -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    if IS_WINDOWS:
        env_vars.update({
            "FFMPEG_BINARY": "ffmpeg.exe",
            "YT_DLP_BINARY": "yt-dlp.exe",
            "PYTHONIOENCODING": "utf-8",
        })
    elif IS_MACOS:
        env_vars.update({
            "FFMPEG_BINARY": "ffmpeg",
            "YT_DLP_BINARY": "yt-dlp",
        })
    else:
        env_vars.update({
            "FFMPEG_BINARY": "ffmpeg",
            "YT_DLP_BINARY": "yt-dlp",
        })
    for key, value in env_vars.items():
        os.environ[key] = value
    return env_vars

def _print_help() -> None:
    print("🐉 Dragon Whisperer - Ultimate Stream Transcription & Translation")
    print("=" * 60)
    print("\nUsage:")
    print("  python dragon_whisperer.py [options]")
    print("\nOptions:")
    print("  --quiet, -q    Quiet mode (minimal output)")
    print("  --debug[=N]    Debug mode with level (1-3). Examples: --debug, --debug=2")
    print("  --debug=COMP   Component-specific debug (e.g., --debug=network,vad)")
    print("  --check        System compatibility check")
    print("  --help, -h     Show this help")
    print("  --version, -v  Show version")
    print("\nExamples:")
    print("  Normal use:   python dragon_whisperer.py")
    print("  Quiet mode:   python dragon_whisperer.py --quiet")
    print("  Debug level2: python dragon_whisperer.py --debug=2")
    print("  VAD debug:    python dragon_whisperer.py --debug=vad")
    print("  System check: python dragon_whisperer.py --check")
    print("\nFeatures:")
    print("  • Live stream transcription (YouTube, Twitch, etc.)")
    print("  • Real-time translation to 50+ languages")
    print("  • Subtitle export (SRT, VTT)")
    print("  • Batch processing")
    print("  • Dark mode GUI")

def _run_system_check() -> int:
    issues = check_platform_compatibility()
    print("🔍 Dragon Whisperer - System Compatibility Check")
    print("=" * 50)
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✅ No compatibility issues")
    print("\n📦 Dependency Check:")
    print(f"  FFmpeg:        {'✅' if shutil.which('ffmpeg') else '❌'}")
    print(f"  yt-dlp:        {'✅' if shutil.which('yt-dlp') else '❌'}")
    print(f"  Tkinter:       {'✅' if GUI_AVAILABLE else '❌'}")
    print(f"  faster-whisper:{'✅' if WHISPER_AVAILABLE else '❌'}")
    print(f"  NumPy:         {'✅' if NUMPY_AVAILABLE else '❌'}")
    print(f"  PyTorch:       {'✅' if TORCH_AVAILABLE else '❌'}")
    print(f"  deep-translator:{'✅' if TRANSLATOR_AVAILABLE else '❌'}")
    print(f"  SciPy:         {'✅' if SCIPY_AVAILABLE else '❌'}")
    print(f"  psutil:        {'✅' if 'psutil' in sys.modules else '❌'}")
    print("\n💻 System Info:")
    print(f"  Platform: {SYSTEM}")
    print(f"  Architecture: {'ARM' if IS_ARM else 'x86'}")
    print(f"  Python: {sys.version.split()[0]}")
    if IS_WINDOWS:
        try:
            import platform
            print(f"  Windows: {platform.version()}")
        except Exception:
            pass
    return 0 if not issues else 1

def _show_user_error(message: str) -> None:
    if IS_WINDOWS and not sys.stdin.isatty():
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f"Dragon Whisperer - Setup Required\n\n{message}\n\n"
                "Please install missing components and try again.",
                "Setup Error",
                0x10,
            )
        except Exception:
            pass
    else:
        if "Tkinter" in message:
            print("\n💡 INSTALLATION HELP:")
            print("  pip install tk")
        elif "FFmpeg" in message:
            print("\n💡 INSTALLATION HELP:")
            if IS_WINDOWS:
                print("  Download from: https://ffmpeg.org/download.html")
            elif IS_MACOS:
                print("  brew install ffmpeg")
            else:
                print("  sudo apt install ffmpeg")
        else:
            print(f"\n❌ {message}")

def _show_critical_error(message: str) -> None:
    if IS_WINDOWS and not sys.stdin.isatty():
        try:
            import ctypes
            short_msg = message[:200] + "..." if len(message) > 200 else message
            ctypes.windll.user32.MessageBoxW(
                0,
                f"Dragon Whisperer - Critical Error\n\n{short_msg}\n\n"
                "Please check the console for details.\n"
                "Try running with --debug flag for more information.",
                "Critical Error",
                0x10,
            )
        except Exception:
            pass
    else:
        print(f"\n💥 {message}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore")

    if IS_WINDOWS:
        _setup_windows_console()

    cli_args = {
        "debug": "--debug" in sys.argv,
        "quiet": "--quiet" in sys.argv or "-q" in sys.argv,
        "check": "--check" in sys.argv,
        "help": "--help" in sys.argv or "-h" in sys.argv,
        "version": "--version" in sys.argv or "-v" in sys.argv,
    }

    if cli_args["help"]:
        _print_help()
        return 0
    if cli_args["version"]:
        print("🐉 Dragon Whisperer v2.0 - überarbeitet")
        print(f"Platform: {SYSTEM} {'ARM' if IS_ARM else 'x86'}")
        return 0

    debug_level = DEBUG_LEVEL

    if cli_args["check"]:
        return _run_system_check()

    app: Optional[DragonWhispererGUI] = None
    exit_code = 0

    try:
        logger.info("🐉 Dragon Whisperer starting...")

        logger.debug("🔍 Checking dependencies...")
        try:
            PlatformUtils.check_platform_dependencies()
        except RuntimeError as e:
            _show_user_error(str(e))
            return 1
        logger.debug("✅ Dependencies OK")

        if not GUI_AVAILABLE:
            raise RuntimeError("Tkinter/GUI not available. Install with: pip install tk")

        logger.debug("⚡ Setting up signal handlers...")
        SignalHandler.setup(verbose=False, silent=True, max_cleanup_time=10.0)

        logger.debug("🖥️ Initializing GUI...")
        app = DragonWhispererGUI()

        if debug_level >= 1 and not cli_args["quiet"]:
            print("\n" + "=" * 50)
            print("🐉 DRAGON WHISPERER READY")
            print("=" * 50)
            print(f"Platform: {SYSTEM} {'ARM' if IS_ARM else 'x86'}")
            print(f"Python: {sys.version.split()[0]}")
            print(f"Working Dir: {os.getcwd()}")
            if hasattr(app, "transcription_engine"):
                current_model = app.transcription_engine.get_current_model()
                print(f"Model: {current_model if current_model else 'Not loaded'}")
            print(f"Layout: {getattr(app, 'layout_mode', 'vertical')}")
            print("=" * 50 + "\n")

        logger.info("🚀 Starting main loop...")
        app.run()
        logger.info("✅ Application closed normally")

    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by user")
        exit_code = 0

    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"❌ {error_msg}")
        _show_user_error(error_msg)
        exit_code = 1

    except Exception as e:
        error_msg = str(e)
        logger.error(f"💥 Unexpected error: {error_msg}")
        if debug_level >= 2:
            import traceback
            traceback.print_exc()
        _show_critical_error(error_msg)
        exit_code = 2

    finally:
        logger.debug("🧹 Final minimal cleanup...")
        try:
            transcription_cache.clear()
            translation_cache.clear()
            audio_cache.clear()
        except Exception:
            pass
        gc.collect()
        logger.debug("✅ Shutdown complete")

    return exit_code


if __name__ == "__main__":
    try:
        if "__file__" in globals():
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.getcwd() != script_dir:
                os.chdir(script_dir)
                if DEBUG_LEVEL >= 1:
                    print(f"📁 Working directory set to: {script_dir}")
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user (KeyboardInterrupt)")
        sys.exit(0)
    except SystemExit as e:
        raise e
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n💥 FATAL ERROR in main guard: {error_type}: {error_msg}")
        if DEBUG_LEVEL >= 2:
            import traceback
            traceback.print_exc()
            print("\n🔧 Debug Info:")
            print(f"  Python: {sys.version}")
            print(f"  Platform: {sys.platform}")
            print(f"  Executable: {sys.executable}")
            print(f"  CWD: {os.getcwd()}")
            print(f"  Script: {__file__ if '__file__' in globals() else 'Unknown'}")
        sys.exit(99)
