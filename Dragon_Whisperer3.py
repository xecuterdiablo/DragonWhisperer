#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""🐉 THE DRAGON WHISPERER v1.0 - Ultimate Stream Transcription & Translation + SUBTITLES (Dark-Edition)"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*The pynvml package is deprecated.*")

os.environ.update({
    'PYTHONWARNINGS': 'ignore',
    'TORCH_DISABLE_CUDA_WARNINGS': '1',
    'TORCH_CPP_LOG_LEVEL': '0',
    'PYTORCH_JIT': '0',
})

import logging
import threading
import time
import signal as py_signal
import gc
import statistics
import collections
import json
import queue
import shutil
import subprocess
import tempfile
import hashlib
import re
import psutil
import traceback
import platform
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import wraps
from collections import deque, OrderedDict
from datetime import datetime, timedelta
import datetime as dt 
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import atexit

# Plattformerkennung für optimierte Performance
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

try:
    machine = platform.machine().lower()
    IS_ARM = machine in ('arm64', 'aarch64', 'armv8l', 'arm')
    IS_X86 = machine in ('x86_64', 'amd64', 'i386', 'i686', 'x86')
except:
    IS_ARM = False
    IS_X86 = True

print(f"🐉 Dragon Whisperer - Platform: {SYSTEM} {'ARM' if IS_ARM else 'x86'}")

os.environ.update({
    'FFMPEG_DISABLE_RKMPP': '1',
    'AV_DISABLE_RKMPP': '1',
    'CUDA_VISIBLE_DEVICES': '0',
    'FFMPEG_DISABLE_VAAPI': '0' if IS_LINUX else '1',
    'FFMPEG_DISABLE_VDPAU': '0' if IS_LINUX else '1',
    'OPENCV_LOG_LEVEL': 'ERROR',
    'GST_DEBUG': '0',
    'PYTHONHASHSEED': '0'
})

if IS_WINDOWS:
    os.environ.update({
        'PYTHONIOENCODING': 'utf-8',
        'FFMPEG_DISABLE_VAAPI': '1',
        'FFMPEG_DISABLE_VDPAU': '1'
    })

class FastLazyLoader:
    """Schnelles Lazy-Loading für teure Module - reduziert Startzeit"""

    _loaded_modules = {}
    _module_locks = {}

    @classmethod
    def load(cls, module_name, import_path=None):
        """Lade Modul erst bei Bedarf"""
        if module_name in cls._loaded_modules:
            return cls._loaded_modules[module_name]

        if module_name not in cls._module_locks:
            cls._module_locks[module_name] = threading.RLock()

        with cls._module_locks[module_name]:
            if module_name in cls._loaded_modules:
                return cls._loaded_modules[module_name]

            try:
                if module_name == 'torch':
                    import torch
                    try:
                        import torch._logging
                        torch._logging.set_logs(all=logging.ERROR)
                    except:
                        pass
                    cls._loaded_modules['torch'] = torch

                elif module_name == 'faster_whisper':
                    from faster_whisper import WhisperModel
                    cls._loaded_modules['faster_whisper'] = WhisperModel

                elif module_name == 'numpy':
                    import numpy as np
                    cls._loaded_modules['numpy'] = np

                elif module_name == 'deep_translator':
                    from deep_translator import GoogleTranslator
                    cls._loaded_modules['deep_translator'] = GoogleTranslator

                elif module_name == 'scipy.signal':
                    import scipy.signal
                    cls._loaded_modules['scipy.signal'] = scipy.signal

                else:
                    module = __import__(module_name)
                    cls._loaded_modules[module_name] = module

                return cls._loaded_modules.get(module_name)

            except ImportError as e:
                print(f"⚠️  Module {module_name} not available: {e}")

                class MockModule:
                    def __init__(self, name):
                        self.__name__ = name

                    def __getattr__(self, name):
                        def mock_method(*args, **kwargs):
                            raise ImportError(f"Module {self.__name__} not available")
                        return mock_method

                mock = MockModule(module_name)
                cls._loaded_modules[module_name] = mock
                return mock

    @classmethod
    def is_available(cls, module_name):
        """Prüfe ob Modul verfügbar ist"""
        try:
            module = cls.load(module_name)
            return not isinstance(module, cls._loaded_modules.get('__mock__', type('Mock', (), {})))
        except:
            return False

TORCH_AVAILABLE = FastLazyLoader.is_available('torch')
WHISPER_AVAILABLE = FastLazyLoader.is_available('faster_whisper')
NUMPY_AVAILABLE = FastLazyLoader.is_available('numpy')
TRANSLATOR_AVAILABLE = FastLazyLoader.is_available('deep_translator')
SCIPY_AVAILABLE = FastLazyLoader.is_available('scipy.signal')

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    tk = None
    ttk = None
    scrolledtext = None

class PlatformStderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filter_patterns = [
            'mpp_soc:', 'mpp_platform:', 'can not found match soc name',
            '/proc/device-tree/compatible', 'rockchip', 'ffmpeg', 'deprecated',
            'DEBUG', 'INFO', 'WARNING', 'FutureWarning', 'pynvml', 'deprecated',
            'TORCH_NCCL', 'CUDA_VISIBLE_DEVICES'
        ]

        if IS_WINDOWS:
            self.filter_patterns.extend([
                'Failed to set direct console mode',
                'Console code page',
                'chcp',
                'win32api'
            ])

    def write(self, text):
        if text and any(p in text for p in self.filter_patterns):
            return
        self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

sys.stderr = PlatformStderrFilter(sys.stderr)
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

class ShutdownPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class SignalHandler:
    """
    🚀 OPTIMIERTER Signal Handler - Hybrid-Lösung für maximale Stabilität
    """
    
    class _CleanupOperation:
        def __init__(self, func: Callable, name: str,
                     priority: ShutdownPriority = ShutdownPriority.MEDIUM,
                     timeout: float = 3.0, essential: bool = False):
            self.func = func
            self.name = name
            self.priority = priority
            self.timeout = timeout
            self.essential = essential
            self.attempts = 0
            self.last_error = None

        def execute(self) -> bool:
            """Führt Cleanup-Operation aus - SICHER mit Timeout"""
            self.attempts += 1
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.func)
                    result = future.result(timeout=self.timeout)
                    self.last_error = None
                    return True
            except TimeoutError:
                self.last_error = f"Timeout nach {self.timeout}s"
                return False
            except Exception as e:
                self.last_error = str(e)
                return False

    _instance = None
    _lock = threading.RLock()
    _atexit_lock = threading.RLock()
    _shutdown_requested = False
    _shutdown_in_progress = False
    _signal_count = 0
    _setup_complete = False
    _cleanup_operations = None
    _original_handlers = None
    _atexit_registered = False
    _config = {
        'verbose': False,
        'silent': True,
        'max_cleanup_time': 20.0,
        'emergency_timeout': 2.0,
        'atexit_enabled': True,
        'hybrid_shutdown': True,
    }

    def __new__(cls):
        """Thread-sicherer Singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SignalHandler, cls).__new__(cls)
                    
                    cls._cleanup_operations = {
                        ShutdownPriority.CRITICAL: [],
                        ShutdownPriority.HIGH: [],
                        ShutdownPriority.MEDIUM: [],
                        ShutdownPriority.LOW: [],
                    }
                    cls._instance._original_handlers = {}
                    cls._instance._shutdown_requested = False
                    cls._instance._shutdown_in_progress = False
                    cls._instance._signal_count = 0
                    cls._instance._setup_complete = False
                    cls._instance._atexit_registered = False
                    cls._instance._initialized = False
                    
        return cls._instance

    def __init__(self):
        """Wird nur einmal ausgeführt"""
        if getattr(self, '_initialized', False):
            return
        
        with self.__class__._lock:
            if self._initialized:
                return
            
            if not hasattr(self, '_cleanup_operations') or self._cleanup_operations is None:
                self._cleanup_operations = {
                    ShutdownPriority.CRITICAL: [],
                    ShutdownPriority.HIGH: [],
                    ShutdownPriority.MEDIUM: [],
                    ShutdownPriority.LOW: [],
                }
            
            if not hasattr(self, '_original_handlers') or self._original_handlers is None:
                self._original_handlers = {}
            
            self._shutdown_requested = False
            self._shutdown_in_progress = False
            self._signal_count = 0
            self._setup_complete = False
            self._atexit_registered = False
            self._initialized = True

    @classmethod
    def setup(cls, verbose=False, silent=True, hybrid_shutdown=True, **kwargs):
        """Initialisiert Signal Handler"""
        with cls._lock:
            instance = cls()
            
            if instance._setup_complete:
                return instance
            
            instance._config.update({
                'verbose': verbose, 
                'silent': silent,
                'hybrid_shutdown': hybrid_shutdown
            })
            
            if 'max_cleanup_time' in kwargs:
                instance._config['max_cleanup_time'] = kwargs['max_cleanup_time']
            if 'emergency_timeout' in kwargs:
                instance._config['emergency_timeout'] = kwargs['emergency_timeout']
            if 'atexit_enabled' in kwargs:
                instance._config['atexit_enabled'] = kwargs['atexit_enabled']
            
            if instance._config['verbose']:
                print("🚀 SignalHandler initialisieren...")
            
            instance._save_original_handlers()
            instance._install_signal_handlers()
            
            if instance._config['atexit_enabled']:
                instance._register_atexit()
            
            instance._setup_complete = True
            
            if instance._config['verbose']:
                print("✅ SignalHandler bereit (Hybrid-Modus)" if hybrid_shutdown else "✅ SignalHandler bereit")
            
            return instance

    @classmethod
    def _save_original_handlers(cls):
        """Speichert originale Signal-Handler"""
        instance = cls._instance
        
        if IS_WINDOWS or not instance:
            return
        
        try:
            for sig in [py_signal.SIGINT, py_signal.SIGTERM]:
                if hasattr(py_signal, sig.__name__):
                    instance._original_handlers[sig] = py_signal.getsignal(sig)
        except Exception:
            pass

    @classmethod
    def _install_signal_handlers(cls):
        """Installiert Signal-Handler"""
        instance = cls._instance
        
        def signal_handler(signum, frame):
            """Haupt-Signal-Handler - Intelligente Exit-Strategie"""
            with cls._lock:
                instance._signal_count += 1
                
                if instance._signal_count == 1:
                    if not instance._config['silent']:
                        print(f"\n⚠️  Shutdown angefordert...")
                    instance._shutdown_requested = True
                    instance._initiate_graceful_shutdown()
                elif instance._signal_count >= 2:
                    if not instance._config['silent']:
                        print("\n🛑 Forcierter Shutdown...")
                    instance._force_shutdown()

        # Unix/Linux/macOS
        if not IS_WINDOWS:
            try:
                py_signal.signal(py_signal.SIGINT, signal_handler)
                py_signal.signal(py_signal.SIGTERM, signal_handler)
            except Exception:
                pass
        
        # Windows
        else:
            try:
                import win32api
                
                def win32_handler(ctrl_type):
                    if ctrl_type in (0, 1, 2, 5):
                        signal_handler(ctrl_type, None)
                        return True
                    return False
                
                win32api.SetConsoleCtrlHandler(win32_handler, True)
            except ImportError:
                try:
                    py_signal.signal(py_signal.SIGINT, signal_handler)
                except:
                    pass
            except Exception:
                pass

    @classmethod
    def register_cleanup(cls, func: Callable, name: str = None,
                         priority: ShutdownPriority = ShutdownPriority.MEDIUM,
                         timeout: float = 3.0, essential: bool = False):
        """Registriert Cleanup-Operation"""
        instance = cls()
        
        if name is None:
            name = func.__name__ if hasattr(func, '__name__') else "Anonymous"
        
        operation = cls._CleanupOperation(
            func=func, name=name, priority=priority, 
            timeout=timeout, essential=essential
        )
        
        with cls._lock:
            if priority not in instance._cleanup_operations:
                instance._cleanup_operations[priority] = []
            
            for existing in instance._cleanup_operations[priority]:
                if existing.func == func:
                    return
            
            instance._cleanup_operations[priority].append(operation)
            
            if instance._config.get('verbose', False):
                print(f"✅ Cleanup: {name} (Priority: {priority.name})")

    @classmethod
    def unregister_cleanup(cls, func: Callable):
        """Entfernt Cleanup-Operation"""
        instance = cls()
        
        with cls._lock:
            for priority in instance._cleanup_operations:
                for i, op in enumerate(instance._cleanup_operations[priority]):
                    if op.func == func:
                        del instance._cleanup_operations[priority][i]
                        return True
            return False

    @classmethod
    def _initiate_graceful_shutdown(cls):
        """💡 INTELLIGENTE HYBRID-LÖSUNG - Best of both worlds"""
        instance = cls._instance
        if not instance:
            sys.exit(0)
        with cls._lock:
            if instance._shutdown_in_progress:
                if instance._config['verbose']:
                    print("🔁 Shutdown bereits aktiv - überspringe")
                return
            instance._shutdown_in_progress = True
        
        if not instance._config['silent']:
            print("🧹 Starte geordneten Shutdown...")
        
        try:
            success = instance._execute_priority_cleanup()
            instance._restore_original_handlers()
            
            if instance._config.get('hybrid_shutdown', True):
                current_thread = threading.current_thread()
                main_thread = threading.main_thread()
                
                if current_thread == main_thread:
                    if instance._config['verbose']:
                        print("💡 Sauberes Exit im Hauptthread")
                    sys.exit(0 if success else 1)
                else:
                    if instance._config['verbose']:
                        print(f"💡 Sofort-Exit in Thread: {current_thread.name}")
                    os._exit(0 if success else 1)
            else:
                os._exit(0 if success else 1)
                
        except Exception as e:
            if not instance._config['silent']:
                print(f"❌ Shutdown fehlgeschlagen: {e}")
            
            os._exit(2)

    @classmethod
    def _execute_priority_cleanup(cls):
        """Führt Cleanup-Operationen nach Priorität aus"""
        instance = cls._instance
        
        if not instance or not any(instance._cleanup_operations.values()):
            return True
        
        overall_success = True
        start_time = time.time()
        completed_ops = 0
        failed_ops = 0
        
        if instance._config.get('verbose', False):
            print(f"🔧 Starte Cleanup ({sum(len(ops) for ops in instance._cleanup_operations.values())} Operationen)...")
        
        for priority in ShutdownPriority:
            operations = instance._cleanup_operations.get(priority, [])
            if not operations:
                continue
            
            if time.time() - start_time > instance._config['max_cleanup_time']:
                if instance._config.get('verbose', False):
                    print(f"⏰ Max cleanup time reached ({instance._config['max_cleanup_time']}s)")
                break
            
            for op in operations:
                try:
                    success = op.execute()
                    completed_ops += 1
                    
                    if not success:
                        failed_ops += 1
                        if op.essential:
                            overall_success = False
                            if instance._config.get('verbose', False):
                                print(f"❌ ESSENTIAL cleanup failed: {op.name}")
                        elif instance._config.get('verbose', False):
                            print(f"⚠️ Cleanup failed (non-essential): {op.name}")
                    elif instance._config.get('verbose', False) and instance._config.get('verbose', False):
                        print(f"✅ Cleanup: {op.name}")
                        
                except Exception as e:
                    failed_ops += 1
                    print(f"⚠️ Cleanup execution error: {op.name}: {e}")
                    if op.essential:
                        overall_success = False
        
        if instance._config.get('verbose', False):
            print(f"📊 Cleanup abgeschlossen: {completed_ops} Operationen, {failed_ops} fehlgeschlagen")
        
        return overall_success

    @classmethod
    def _force_shutdown(cls):
        """Erzwungener Notfall-Shutdown - Immer sofort"""
        instance = cls._instance
        
        try:
            if instance:
                instance._handle_atexit_cleanup()
        except:
            pass
        
        os._exit(1)

    @classmethod
    def _register_atexit(cls):
        """Registriert atexit-Handler"""
        instance = cls._instance
        if not instance:
            return
            
        with cls._atexit_lock:
            if instance._atexit_registered:
                return
            
            def safe_atexit_handler():
                """Sicherer atexit-Handler - nur im Hauptthread"""
                try:
                    if (threading.current_thread() == threading.main_thread() and 
                        instance and not instance._shutdown_in_progress):
                        instance._handle_atexit_cleanup()
                except:
                    pass
            
            atexit.register(safe_atexit_handler)
            instance._atexit_registered = True
            
            if instance._config.get('verbose', False):
                print("✅ AtExit-Handler registriert (nur Hauptthread)")

    @classmethod
    def _handle_atexit_cleanup(cls):
        """Führt atexit-Cleanup aus - nur für kritische Ressourcen"""
        instance = cls._instance
        if not instance:
            return
        
        if instance._config.get('verbose', False):
            print("🔧 AtExit-Cleanup...")
        
        critical_ops = []
        for op in instance._cleanup_operations.get(ShutdownPriority.CRITICAL, []):
            if op.essential and "GPU" in op.name or "Memory" in op.name:
                critical_ops.append(op)
                if len(critical_ops) >= 3:
                    break
        
        start_time = time.time()
        for op in critical_ops:
            if time.time() - start_time > instance._config['emergency_timeout']:
                break
            
            try:
                if instance._config.get('verbose', False):
                    print(f"  ⚡ Emergency: {op.name}")
                op.func()
            except:
                pass
        
        gc.collect()

    @classmethod
    def _restore_original_handlers(cls):
        """Stellt originale Signal-Handler wieder her"""
        instance = cls._instance
        if not instance or not instance._original_handlers or IS_WINDOWS:
            return
        
        try:
            for sig, handler in instance._original_handlers.items():
                if handler is not None:
                    py_signal.signal(sig, handler)
        except Exception:
            pass

    @classmethod
    def should_shutdown(cls) -> bool:
        """Prüft ob Shutdown angefordert wurde"""
        instance = cls._instance
        with cls._lock:
            return instance._shutdown_requested if instance else False

    @classmethod
    def get_status(cls) -> Dict:
        """Gibt Status-Informationen zurück"""
        instance = cls._instance
        with cls._lock:
            if not instance:
                return {'error': 'SignalHandler not initialized'}
                
            return {
                'shutdown_requested': instance._shutdown_requested,
                'shutdown_in_progress': instance._shutdown_in_progress,
                'signal_count': instance._signal_count,
                'setup_complete': instance._setup_complete,
                'cleanup_operations': {
                    priority.name: len(ops)
                    for priority, ops in instance._cleanup_operations.items()
                },
                'atexit_registered': instance._atexit_registered,
                'hybrid_mode': instance._config.get('hybrid_shutdown', True),
                'config': {k: v for k, v in instance._config.items() if not k.startswith('_')},
            }

    @classmethod
    def print_debug_info(cls):
        """Gibt Debug-Informationen aus"""
        status = cls.get_status()
        
        print("\n" + "="*50)
        print("🚀 SIGNAL HANDLER STATUS")
        print("="*50)
        
        for key, value in status.items():
            if key == 'cleanup_operations':
                print(f"Cleanup Operations:")
                for priority, count in value.items():
                    print(f"  {priority}: {count}")
            elif key == 'config':
                print(f"Configuration:")
                for config_key, config_value in value.items():
                    print(f"  {config_key}: {config_value}")
            else:
                print(f"{key}: {value}")
        
        print("="*50)

    @classmethod
    def emergency_shutdown(cls, reason: str = "Emergency", exit_code: int = 1):
        """Notfall-Shutdown von außerhalb - Immer sofort"""
        instance = cls._instance
        if not instance:
            print(f"🚨 NOTFALL-SHUTDOWN: {reason}")
            os._exit(exit_code)
            
        if not instance._config['silent']:
            print(f"🚨 NOTFALL-SHUTDOWN: {reason}")
        
        with cls._lock:
            instance._shutdown_requested = True
            instance._shutdown_in_progress = True
        
        os._exit(exit_code)

    @classmethod
    def reset(cls):
        """Setzt Handler zurück (nur für Tests)"""
        with cls._lock:
            cls._instance = None
            SignalHandler._instance = None
            SignalHandler._shutdown_requested = False
            SignalHandler._shutdown_in_progress = False
            SignalHandler._signal_count = 0
            SignalHandler._setup_complete = False
            SignalHandler._original_handlers = {}
            SignalHandler._atexit_registered = False
            SignalHandler._cleanup_operations = None

    def setup_signal_handler_for_audio_processor(audio_processor, transcription_engine=None, 
                                               translation_engine=None, verbose=True):
        """
        Einfache Setup-Funktion für Audio Processing Anwendungen
        """
        SignalHandler.setup(verbose=verbose, silent=not verbose, hybrid_shutdown=True)
    
        SignalHandler.register_cleanup(
            func=lambda: audio_processor.stop_processing(),
            name="AudioProcessor Stop",
            priority=ShutdownPriority.HIGH,
            timeout=3.0,
            essential=True
        )
    
        SignalHandler.register_cleanup(
            func=lambda: audio_processor.emergency_reset(force=True),
            name="AudioProcessor Emergency Reset",
            priority=ShutdownPriority.CRITICAL,
            timeout=2.0,
            essential=True
        )
    
        if transcription_engine:
            SignalHandler.register_cleanup(
                func=lambda: transcription_engine.dispose(),
                name="TranscriptionEngine Dispose",
                priority=ShutdownPriority.MEDIUM,
                timeout=2.0
            )
    
        if translation_engine:
            SignalHandler.register_cleanup(
                func=lambda: translation_engine.dispose(),
                name="TranslationEngine Dispose",
                priority=ShutdownPriority.MEDIUM,
                timeout=2.0
            )
    
        try:
            import torch
            if torch.cuda.is_available():
                SignalHandler.register_cleanup(
                    func=lambda: torch.cuda.empty_cache(),
                    name="GPU Memory Cleanup",
                    priority=ShutdownPriority.LOW,
                    timeout=1.0
                )
        except ImportError:
            pass
    
        print(f"✅ SignalHandler mit {sum(len(ops) for ops in SignalHandler._cleanup_operations.values())} Operationen")
        return SignalHandler

class ExcellenceConfig:
    """
    🎵 Konfiguration
    """

    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    AUDIO_FORMAT: str = 's16le'
    BYTES_PER_SAMPLE: int = 2
    
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
    
    _base_chunk_duration: int = 5
    CHUNK_OVERLAP: float = 0.5
    MIN_CHUNK_DURATION: int = 2
    MAX_CHUNK_DURATION: int = 30
    
    @property
    def CHUNK_DURATION(self) -> float:
        """Flexible Chunk-Dauer (kann überschrieben werden)"""
        return getattr(self, '_actual_chunk_duration', self._base_chunk_duration)
    
    @CHUNK_DURATION.setter
    def CHUNK_DURATION(self, value: float):
        """Setzt Chunk-Dauer mit Validierung"""
        if self.MIN_CHUNK_DURATION <= value <= self.MAX_CHUNK_DURATION:
            self._actual_chunk_duration = float(value)
        else:
            print(f"⚠️ Chunk duration {value}s out of range, using default")
            self._actual_chunk_duration = self._base_chunk_duration
    
    @property
    def CHUNK_SIZE_BYTES(self) -> int:
        """Berechnet Chunk-Größe in Bytes (automatisch)"""
        return int(self.CHUNK_DURATION * self.SAMPLE_RATE * self.CHANNELS * self.BYTES_PER_SAMPLE)
    
    @property
    def OVERLAP_SIZE_BYTES(self) -> int:
        """Berechnet Overlap-Größe in Bytes"""
        return int(self.CHUNK_OVERLAP * self.SAMPLE_RATE * self.CHANNELS * self.BYTES_PER_SAMPLE)
    
    @property
    def BYTES_PER_SECOND(self) -> int:
        """Bytes pro Sekunde"""
        return self.SAMPLE_RATE * self.CHANNELS * self.BYTES_PER_SAMPLE
    
    @property
    def MIN_CHUNK_BYTES(self) -> int:
        """Minimale sinnvolle Chunk-Größe"""
        return int(self.MIN_CHUNK_DURATION * self.BYTES_PER_SECOND)
    
    @property
    def MAX_CHUNK_BYTES(self) -> int:
        """Maximale Chunk-Größe"""
        return int(self.MAX_CHUNK_DURATION * self.BYTES_PER_SECOND)
    
    STREAM_TIMEOUT: int = 10
    INITIAL_BUFFER_SECONDS: float = 1.5
    MAX_EMPTY_READS: int = 15
    RECONNECT_DELAY: int = 2
    READ_RETRY_DELAY: float = 0.1
    
    @property
    def INITIAL_BUFFER_BYTES(self) -> int:
        """Anfangs-Puffer in Bytes"""
        return int(self.INITIAL_BUFFER_SECONDS * self.BYTES_PER_SECOND)
    
    FFMPEG_BUFSIZE: str = '2048k'
    FFMPEG_THREADS: int = 1
    FFMPEG_PROBESIZE: str = '32'
    FFMPEG_ANALYZE_DURATION: str = '0'
    
    YOUTUBE_TIMEOUT: int = 10000000
    NORMAL_TIMEOUT: int = 30000000
    
    def get_timeout_microseconds(self, is_youtube: bool = False) -> int:
        """Gibt Timeout in Mikrosekunden zurück"""
        return self.YOUTUBE_TIMEOUT if is_youtube else self.NORMAL_TIMEOUT
    

    AUDIO_FILTER: str = "aresample=16000,volume=1.5,dynaudnorm"
    
    LANGUAGE_FILTERS: dict = {
        'ko': "aresample=16000,volume=2.0,highpass=f=80,lowpass=f=3800,afftdn=nf=-15",
        'ja': "aresample=16000,volume=2.0,highpass=f=90,lowpass=f=3700,afftdn=nf=-15",
        'zh': "aresample=16000,volume=2.0,highpass=f=100,lowpass=f=3500,afftdn=nf=-20",
        'de': "aresample=16000,volume=1.8,highpass=f=100,lowpass=f=3200,dynaudnorm",
        'en': "aresample=16000,volume=1.8,highpass=f=80,lowpass=f=3400,dynaudnorm",
        'fr': "aresample=16000,volume=2.0,highpass=f=100,lowpass=f=3300,dynaudnorm",
        'es': "aresample=16000,volume=2.0,highpass=f=100,lowpass=f=3400,dynaudnorm",
    }
    
    FILTER_PROFILES: dict = {
        'transcription': "aresample=16000,volume=1.5,dynaudnorm",
        'translation': "aresample=16000,volume=2.0,highpass=f=100,lowpass=f=3400",
        'realtime': "aresample=16000,volume=1.8,dynaudnorm",
        'noisy': "aresample=16000,volume=2.5,highpass=f=150,lowpass=f=3000,afftdn=nf=-30",
        'music': "aresample=16000,volume=1.5,highpass=f=50,lowpass=f=5000",
        'podcast': "aresample=16000,volume=2.0,highpass=f=80,lowpass=f=3500",
    }
    
    def get_audio_filter(self, language: str = None, profile: str = None) -> str:
        """Gibt optimierten Audio-Filter zurück"""
        if profile and profile in self.FILTER_PROFILES:
            return self.FILTER_PROFILES[profile]
        
        if language:
            lang_code = language[:2].lower() if len(language) >= 2 else None
            if lang_code in self.LANGUAGE_FILTERS:
                return self.LANGUAGE_FILTERS[lang_code]
        
        return self.AUDIO_FILTER
    
    YOUTUBE_HEADERS: dict = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.youtube.com/',
        'Origin': 'https://www.youtube.com',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    def get_youtube_headers(self, is_manifest: bool = False) -> dict:
        """Gibt optimierte YouTube-Headers zurück"""
        headers = self.YOUTUBE_HEADERS.copy()
        
        if is_manifest:
            headers.update({
                'X-Client-Data': 'CI22yQE=',
                'Content-Type': 'application/x-mpegURL',
            })
        
        return headers
    
    PLATFORM_CONFIG: dict = {
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
    
    def get_platform_config(self, platform: str = None) -> dict:
        """Gibt plattformspezifische Konfiguration"""
        if not platform:
            import platform
            platform = platform.system().lower()
        
        return self.PLATFORM_CONFIG.get(platform, self.PLATFORM_CONFIG['linux'])
    
    PERFORMANCE_LOG_INTERVAL: int = 50
    STATS_BUFFER_SIZE: int = 100
    MAX_CONSECUTIVE_ERRORS: int = 3

    AUDIO_ENHANCEMENT_ENABLED: bool = True
    MIN_RMS_THRESHOLD: float = 0.002
    TARGET_RMS: float = 0.2
    MAX_GAIN: float = 5.0
    CLIPPING_THRESHOLD: float = 0.9
    
    DUPLICATE_CHECK_ENABLED: bool = True
    RECENT_TRANSCRIPTIONS_SIZE: int = 10
    MIN_TEXT_LENGTH: int = 3
    MIN_UNIQUE_WORDS_RATIO: float = 0.3
    
    SUBTITLE_BUFFER_SIZE: int = 1000
    ENABLE_TIMED_TRANSCRIPTIONS: bool = True
    ENABLE_TIMED_TRANSLATIONS: bool = True

    ENABLE_DEBUG_LOGGING: bool = True
    LOG_CHUNK_PROCESSING: bool = False
    LOG_AUDIO_STATS: bool = True
    LOG_PERFORMANCE: bool = True
    LOG_STREAM_EVENTS: bool = True
    
    MAX_CACHE_SIZE_MB: int = 100
    CACHE_ENABLED: bool = True
    
    def __init__(self):
        """Initialisiert mit flexiblen Standardwerten"""
        self._actual_chunk_duration = self._base_chunk_duration
    
    def calculate_optimal_chunk_duration(self, model_size: str = 'medium', 
                                         is_realtime: bool = False) -> int:
        """Berechnet optimale Chunk-Dauer"""
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
        """Validiert die Konfiguration"""
        try:
            valid = (
                self.SAMPLE_RATE in [8000, 16000, 22050, 44100, 48000] and
                self.CHANNELS in [1, 2] and
                self.MIN_CHUNK_DURATION <= self.CHUNK_DURATION <= self.MAX_CHUNK_DURATION
            )
            
            if not valid:
                print("❌ Config validation failed")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Config validation error: {e}")
            return False
    
    def print_summary(self) -> None:
        """Gibt Konfigurations-Zusammenfassung aus"""
        print("\n" + "="*60)
        print("🎵 EXCELLENCE CONFIGURATION")
        print("="*60)
        print(f"📊 Audio: {self.SAMPLE_RATE}Hz, {self.CHANNELS}ch")
        print(f"📦 Chunk: {self.CHUNK_DURATION}s ({self.CHUNK_SIZE_BYTES:,}B)")
        print(f"⚡ Bytes/sec: {self.BYTES_PER_SECOND:,}")
        print(f"🎛️ Filter Profiles: {len(self.FILTER_PROFILES)}")
        print(f"🌍 Language Filters: {len(self.LANGUAGE_FILTERS)}")
        print(f"✅ Valid: {self.validate_config()}")
        print("="*60)
    
    def __str__(self) -> str:
        return f"ExcellenceConfig(chunk={self.CHUNK_DURATION}s, filter_profiles={len(self.FILTER_PROFILES)})"

class RealtimeConfig(ExcellenceConfig):
    """Konfiguration für Echtzeit-Transkription"""
    def __init__(self):
        super().__init__()
        self.CHUNK_DURATION = 5
        self.CHUNK_OVERLAP = 0.3
        self.STREAM_TIMEOUT = 5
        self.AUDIO_FILTER = self.FILTER_PROFILES['realtime']

class HighAccuracyConfig(ExcellenceConfig):
    """Konfiguration für maximale Genauigkeit"""
    def __init__(self):
        super().__init__()
        self.CHUNK_DURATION = 25
        self.CHUNK_OVERLAP = 0.8
        self.AUDIO_FILTER = "aresample=16000,volume=1.8,highpass=f=80,lowpass=f=3800,dynaudnorm=p=0.3:s=3:g=20"

class YouTubeOptimizedConfig(ExcellenceConfig):
    """Konfiguration optimiert für YouTube"""
    def __init__(self):
        super().__init__()
        self.FFMPEG_THREADS = 1
        self.FFMPEG_BUFSIZE = '1024k'
        self.YOUTUBE_TIMEOUT = 5000000
        self.RECONNECT_DELAY = 1
        self.AUDIO_FILTER = "aresample=16000,volume=2.2,highpass=f=120,lowpass=f=3200,compand=attacks=0:decays=0.3"

def get_config(config_type: str = 'default') -> ExcellenceConfig:
    """
    Factory-Methode für verschiedene Konfigurationen
    """
    configs = {
        'default': ExcellenceConfig,
        'realtime': RealtimeConfig,
        'high_accuracy': HighAccuracyConfig,
        'youtube': YouTubeOptimizedConfig,
        'transcription': lambda: ExcellenceConfig(),
        'translation': lambda: ExcellenceConfig(),
    }
    
    config_class = configs.get(config_type, ExcellenceConfig)
    
    if callable(config_class):
        config = config_class()
    else:
        config = config_class()
    
    if hasattr(config, 'validate_config'):
        config.validate_config()
    
    return config

class PlatformUtils:
    """Plattformunabhängige Utility-Funktionen - REPARIERT: Thread-sichere Einmal-Ausführung"""
    
    _environment_setup_done = False
    _environment_setup_lock = threading.RLock()
    _dependencies_checked = False
    _dependencies_lock = threading.RLock()
    
    @staticmethod
    def get_platform_config_dir():
        """Get platform-specific configuration directory - Thread-sicher"""
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
            print(f"⚠️ Config directory error: {e}")
            fallback_dir = Path.home() / ".dragonwhisperer"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir

    @staticmethod
    def kill_process_tree(pid: int):
        """Plattformunabhängiges Beenden von Prozess-Bäumen - mit Timeout"""
        try:
            if IS_WINDOWS:
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                    capture_output=True, 
                    timeout=5, 
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
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
            print(f"⚠️ Timeout killing process tree {pid}")
            return False
        except Exception as e:
            print(f"⚠️ Error killing process tree {pid}: {e}")
            return False

    @staticmethod
    def check_platform_dependencies():
        """Verbesserte Dependency-Checks mit Thread-Safety und besseren Fehlermeldungen"""
        with PlatformUtils._dependencies_lock:
            if PlatformUtils._dependencies_checked:
                return True
            
            missing = []
            issues = []
            
            print("🔍 Checking platform dependencies...")
            
            ffmpeg_found = False
            ffmpeg_paths = []
            
            ffmpeg_standard = shutil.which('ffmpeg')
            if ffmpeg_standard:
                ffmpeg_found = True
                ffmpeg_paths.append(ffmpeg_standard)
            
            if IS_WINDOWS:
                windows_paths = [
                    'C:\\ffmpeg\\bin\\ffmpeg.exe',
                    'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
                    'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe',
                    os.path.join(os.environ.get('ProgramFiles', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
                    os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
                ]
                ffmpeg_paths.extend(windows_paths)
            
            elif IS_MACOS:
                mac_paths = [
                    '/usr/local/bin/ffmpeg',
                    '/opt/homebrew/bin/ffmpeg',
                    '/usr/bin/ffmpeg',
                    '/opt/local/bin/ffmpeg',
                ]
                ffmpeg_paths.extend(mac_paths)
            
            else:
                linux_paths = [
                    '/usr/bin/ffmpeg',
                    '/usr/local/bin/ffmpeg',
                    '/snap/bin/ffmpeg',
                    '/opt/bin/ffmpeg',
                ]
                ffmpeg_paths.extend(linux_paths)
            
            for path in ffmpeg_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    if not ffmpeg_found:
                        ffmpeg_found = True
                        print(f"✅ FFmpeg found: {path}")

                        if IS_WINDOWS:
                            try:
                                bin_dir = os.path.dirname(path)
                                current_path = os.environ.get('PATH', '')
                                if bin_dir not in current_path:
                                    os.environ['PATH'] = f"{bin_dir};{current_path}"
                                    print(f"   ↪ Added to PATH: {bin_dir}")
                            except Exception as e:
                                print(f"   ⚠️ PATH update failed: {e}")
                    break
            
            if not ffmpeg_found:
                missing.append('ffmpeg')
                issues.append("FFmpeg not found in PATH or standard locations")
            
            ytdlp_found = False
            ytdlp_path = shutil.which('yt-dlp')
            if ytdlp_path:
                ytdlp_found = True
                print(f"✅ yt-dlp found: {ytdlp_path}")
            else:
                if IS_WINDOWS:
                    win_ytdlp = shutil.which('yt-dlp.exe')
                    if win_ytdlp:
                        ytdlp_found = True
                        print(f"✅ yt-dlp found: {win_ytdlp}")
            
            if not ytdlp_found:
                missing.append('yt-dlp')
                issues.append("yt-dlp not found in PATH")
            
            package_issues = []
            
            if not WHISPER_AVAILABLE:
                missing.append('faster-whisper')
                package_issues.append("faster-whisper: pip install faster-whisper")
            
            if not TORCH_AVAILABLE:
                issues.append("PyTorch not available (optional for GPU acceleration)")
            
            if not TRANSLATOR_AVAILABLE:
                issues.append("deep-translator not available (translation will be limited)")
            
            if not GUI_AVAILABLE:
                missing.append('tkinter')
                package_issues.append("tkinter: Usually included with Python. On Linux: sudo apt-get install python3-tk")
            
            if missing:
                error_msg = f"❌ Missing dependencies: {', '.join(missing)}\n\n"
                
                if 'ffmpeg' in missing:
                    error_msg += "FFmpeg Installation:\n"
                    if IS_WINDOWS:
                        error_msg += "  • Download from: https://ffmpeg.org/download.html\n"
                        error_msg += "  • Or use Chocolatey: choco install ffmpeg\n"
                    elif IS_MACOS:
                        error_msg += "  • brew install ffmpeg\n"
                    else:
                        error_msg += "  • sudo apt install ffmpeg  (Ubuntu/Debian)\n"
                        error_msg += "  • sudo dnf install ffmpeg  (Fedora)\n"
                    error_msg += "  • Add to PATH if installed manually\n\n"
                
                if 'yt-dlp' in missing:
                    error_msg += "yt-dlp Installation:\n"
                    error_msg += "  • pip install yt-dlp\n"
                    error_msg += "  • Or download from: https://github.com/yt-dlp/yt-dlp\n\n"
                
                if package_issues:
                    error_msg += "Python Packages:\n"
                    for issue in package_issues:
                        error_msg += f"  • {issue}\n"
                
                if issues:
                    error_msg += "\nAdditional issues:\n"
                    for issue in issues:
                        error_msg += f"  • {issue}\n"
                
                error_msg += "\n💡 After installing, restart Dragon Whisperer."
                
                PlatformUtils._dependencies_checked = False
                raise RuntimeError(error_msg)
            
            print("✅ All dependencies found")
            
            try:
                if ffmpeg_found:
                    result = subprocess.run(
                        ['ffmpeg', '-version'],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        check=False
                    )
                    if result.returncode == 0:
                        first_line = result.stdout.split('\n')[0]
                        version_match = re.search(r'ffmpeg version (\S+)', first_line)
                        if version_match:
                            print(f"   ↪ FFmpeg version: {version_match.group(1)}")
            except Exception:
                pass
            
            PlatformUtils._dependencies_checked = True
            return True

    @staticmethod
    def setup_platform_environment():
        """Plattformspezifische Umgebungs-Setup - NUR EINMAL AUFRUFEN, Thread-sicher"""
        with PlatformUtils._environment_setup_lock:
            if PlatformUtils._environment_setup_done:
                return
            
            print("🔧 Setting up platform environment...")
            
            common_env = {
                'PYTHONWARNINGS': 'ignore',
                'TORCH_DISABLE_CUDA_WARNINGS': '1',
                'TORCH_CPP_LOG_LEVEL': '0',
                'PYTORCH_JIT': '0',
                'PYTHONHASHSEED': '0',
                'OPENCV_LOG_LEVEL': 'ERROR',
                'GST_DEBUG': '0',
            }
            
            if IS_WINDOWS:
                platform_env = {
                    'PYTHONIOENCODING': 'utf-8',
                    'FFMPEG_DISABLE_VAAPI': '1',
                    'FFMPEG_DISABLE_VDPAU': '1',
                    'FFMPEG_DISABLE_RKMPP': '1',
                    'AV_DISABLE_RKMPP': '1',
                }
                
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                    
                    STD_OUTPUT_HANDLE = -11
                    handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
                    mode = ctypes.c_ulong()
                    if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                        ctypes.windll.kernel32.SetConsoleMode(
                            handle, 
                            mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                        )
                    
                    os.system('chcp 65001 > nul 2>&1')
                    os.system('color')
                    
                except Exception as e:
                    print(f"⚠️ Windows console setup failed: {e}")
                
            elif IS_MACOS:
                platform_env = {
                    'FFMPEG_DISABLE_VAAPI': '1',
                    'FFMPEG_DISABLE_VDPAU': '1',
                }
                
                temp_dir = Path(tempfile.gettempdir()) / "dragonwhisperer"
                temp_dir.mkdir(exist_ok=True)
                os.environ['TMPDIR'] = str(temp_dir)
                
            else: 
                platform_env = {
                    'FFMPEG_DISABLE_VAAPI': '0',
                    'FFMPEG_DISABLE_VDPAU': '0',
                }
                
                if 'WAYLAND_DISPLAY' in os.environ:
                    platform_env['GDK_BACKEND'] = 'wayland'
                    print("   ↪ Wayland detected")
            
            all_env = {**common_env, **platform_env}
            for key, value in all_env.items():
                os.environ[key] = value
            
            if TORCH_AVAILABLE:
                try:
                    import torch
                    if torch.cuda.is_available():
                        print("   ↪ CUDA available, limiting to GPU 0")
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    else:
                        print("   ↪ CUDA not available, using CPU")
                except Exception as e:
                    print(f"   ⚠️ CUDA check failed: {e}")
            
            PlatformUtils._environment_setup_done = True
            print("✅ Platform environment setup complete")

    @staticmethod
    def get_ffmpeg_path():
        """Gibt den gefundenen FFmpeg-Pfad zurück"""
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
    def reset_flags():
        """Setzt Flags zurück (nur für Tests)"""
        with PlatformUtils._environment_setup_lock:
            PlatformUtils._environment_setup_done = False
        with PlatformUtils._dependencies_lock:
            PlatformUtils._dependencies_checked = False
        print("🔄 PlatformUtils flags reset")

    @staticmethod
    def get_platform_info():
        """Gibt detaillierte Plattform-Informationen zurück"""
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
        except:
            info['cpu_count'] = 'unknown'
            info['memory_total_gb'] = 'unknown'
        
        return info

    @staticmethod
    def print_platform_info():
        """Gibt Plattform-Informationen aus"""
        info = PlatformUtils.get_platform_info()
        
        print("\n" + "="*60)
        print("🐉 PLATFORM INFORMATION")
        print("="*60)
        
        for key, value in info.items():
            if key not in ['environment_setup', 'dependencies_checked']:
                print(f"{key:25} {value}")
        
        print("-"*60)
        print(f"{'Environment Setup':25} {'✅' if info['environment_setup'] else '❌'}")
        print(f"{'Dependencies Checked':25} {'✅' if info['dependencies_checked'] else '❌'}")
        print("="*60)

@dataclass
class ExcellenceTranscriptionResult:
    """Represents a transcription result with confidence and timing."""
    text: str
    confidence: float
    language: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    start: Optional[float] = None
    end: Optional[float] = None

@dataclass
class ExcellenceTranslationResult:
    """Represents a translation result."""
    original: str
    translated: str
    source_lang: str = "auto"
    target_lang: str = "de"
    timestamp: float = field(default_factory=time.time)
    start: Optional[float] = None
    end: Optional[float] = None

class SimplePerformanceTracker:
    """Vereinfachter Performance Tracker - kein Dashboard Overhead."""

    def __init__(self):
        self.transcription_count = 0
        self.translation_count = 0
        self.start_time = time.time()
        self.cache_hits = 0
        self.cache_misses = 0

    def log_transcription(self):
        self.transcription_count += 1

    def log_translation(self):
        self.translation_count += 1

    def log_cache_hit(self):
        self.cache_hits += 1

    def log_cache_miss(self):
        self.cache_misses += 1

    def get_basic_stats(self):
        """Nur die wirklich wichtigen Stats"""
        uptime_minutes = (time.time() - self.start_time) / 60

        total_cache = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0

        return {
            'transcriptions': self.transcription_count,
            'translations': self.translation_count,
            'uptime_minutes': uptime_minutes,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{cache_hit_rate:.1%}",
            'timestamp': datetime.now().isoformat()
        }

class ExcellenceCacheTTL:
    """LRU-Cache mit TTL und Thread-Safety"""

    def __init__(self, maxsize=128, ttl: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = {}
        self._order = collections.deque()
        self._lock = threading.RLock()
        self._cleanup_interval = 300
        self._last_cleanup = time.time()
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self._stats_lock = threading.Lock()

    def _perform_cleanup_if_needed(self):
        """Automatische Cleanup von abgelaufenen Einträgen"""
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

    def _remove_key(self, key):
        """Entferne Schlüssel aus allen Datenstrukturen"""
        if key in self._cache:
            del self._cache[key]
        if key in self._order:
            self._order.remove(key)

    def get(self, key):
        """Get value from cache with TTL check and LRU update"""
        with self._lock:
            self._perform_cleanup_if_needed()

            if key in self._cache:
                value, timestamp = self._cache[key]

                if (time.time() - timestamp) > self.ttl:
                    self._remove_key(key)
                    with self._stats_lock:
                        self._stats['misses'] += 1
                    return None

                if key in self._order:
                    self._order.remove(key)
                self._order.append(key)

                with self._stats_lock:
                    self._stats['hits'] += 1

                return value

            with self._stats_lock:
                self._stats['misses'] += 1
            return None

    def put(self, key, value):
        """Put value into cache with TTL timestamp"""
        with self._lock:
            self._perform_cleanup_if_needed()

            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self.maxsize:
                oldest = self._order.popleft()
                self._remove_key(oldest)
                with self._stats_lock:
                    self._stats['evictions'] += 1

            self._cache[key] = (value, time.time())
            self._order.append(key)

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._order.clear()
            self._last_cleanup = time.time()

    def clear_expired(self) -> int:
        """Clear expired entries and return count"""
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
        """Get cache statistics"""
        with self._lock, self._stats_lock:
            total_size = len(self._cache)
            expired = self.clear_expired()

            return {
                'total_entries': total_size,
                'expired_entries': expired,
                'max_size': self.maxsize,
                'ttl_seconds': self.ttl,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'hit_rate': self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
            }

transcription_cache = ExcellenceCacheTTL(maxsize=256, ttl=600)
translation_cache = ExcellenceCacheTTL(maxsize=512, ttl=3600)
audio_cache = ExcellenceCacheTTL(maxsize=128, ttl=1800)

def clear_expired_cache_entries() -> Dict[str, int]:
    """Clear all expired cache entries and return counts"""
    return {
        'transcription_expired': transcription_cache.clear_expired(),
        'translation_expired': translation_cache.clear_expired(),
        'audio_expired': audio_cache.clear_expired()
    }

def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    return {
        'transcription_cache': transcription_cache.get_stats(),
        'translation_cache': translation_cache.get_stats(),
        'audio_cache': audio_cache.get_stats()
    }

def cache_transcription(result) -> str:
    """Cache a transcription result and return cache key"""
    key = hashlib.sha256(result.text.encode()).hexdigest()
    transcription_cache.put(key, result)
    return key

def get_cached_transcription(text: str):
    """Retrieve a cached transcription result"""
    key = hashlib.sha256(text.encode()).hexdigest()
    return transcription_cache.get(key)

def cache_translation(result) -> str:
    """Cache a translation result and return cache key"""
    key = hashlib.sha256((result.original + result.target_lang).encode()).hexdigest()
    translation_cache.put(key, result)
    return key

def get_cached_translation(original: str, target_lang: str):
    """Retrieve a cached translation result"""
    key = hashlib.sha256((original + target_lang).encode()).hexdigest()
    return translation_cache.get(key)

def excellence_execution(timeout: int = 60, max_retries: int = 3) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        return future.result(timeout=timeout)

                except FutureTimeout as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"⏰ Timeout attempt {attempt+1}/{max_retries+1} for {func.__name__}")

                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"⚠️ Exception in {func.__name__}: {str(e)[:100]}")

                if attempt < max_retries:
                    wait_time = min(30, 2 ** attempt)
                    time.sleep(wait_time)
                    continue

            if last_exception is not None:
                raise last_exception

            raise RuntimeError(f"Decorator logic failed for {func.__name__}.")

        return wrapper
    return decorator

class ExcellenceError(Exception):
    """Base exception for excellence system."""
    pass

def excellence_gui_operation(func):
    """    Enhanced GUI operation decorator with exception handling.    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (tk.TclError, RuntimeError):
            return None
        except Exception:
            return None
    return wrapper

class DragonColors:
    """Haupt-Farbpalette"""
    CHECKBOX_ACTIVE = "#000000"
    MODERN_SUBTITLE_ACTIVE = "#00ff88"
    MODERN_SUBTITLE_INACTIVE = "#8b949e"
    BG_PRIMARY = '#0f1419'
    BG_SECONDARY = '#1a2129'
    BG_TERTIARY = '#242d38'
    BG_HOVER = '#2a3645'
    BG_CARD = '#1c252f'
    TEXT_PRIMARY = '#e6edf3'
    TEXT_SECONDARY = '#8b949e'
    TEXT_ACCENT = '#58a6ff'
    TEXT_MUTED = '#6e7681'
    DRAGON_GREEN = '#238636'
    DRAGON_GREEN_LIGHT = '#2ea043'
    DRAGON_BLUE = '#1f6feb'
    DRAGON_PURPLE = '#8957e5'
    SUCCESS = '#238636'
    WARNING = '#d29922'
    ERROR = '#f85149'
    INFO = '#58a6ff'
    BORDER = '#30363d'
    SCROLLBAR = '#3c444d'
    SCROLLBAR_HOVER = '#4c5560'
    INPUT_BG = '#161b22'
    INPUT_BORDER = '#30363d'
    INPUT_FOCUS = '#1f6feb'
    COMBO_BG = '#161b22'
    COMBO_FG = '#e6edf3'
    COMBO_BORDER = '#30363d'
    COMBO_SELECTION = '#1f6feb'
    CHECKBOX_BG = '#0d1117'
    CHECKBOX_FG = '#e6edf3'
    CHECKBOX_SELECTED = '#238636'
    SUBTITLE_ACTIVE = '#8957e5'
    SUBTITLE_INACTIVE = '#30363d'
    STATUS_BAR_BG = '#0d1117'
    STATUS_BAR_FG = '#8b949e'
    STATUS_BAR_ACCENT = '#58a6ff'

class DragonFonts:
    """Haupt-Schriftpalette"""
    TITLE = ("Segoe UI", 12, "bold")
    SUBTITLE = ("Segoe UI", 10, "bold")
    PRIMARY = ("Segoe UI", 9)
    SECONDARY = ("Segoe UI", 8)
    MONOSPACE = ("Cascadia Code", 9)
    BUTTON = ("Segoe UI", 9, "bold")
    STATUS = ("Segoe UI", 8)
    SMALL = ("Segoe UI", 7)

ModernColors = DragonColors
ModernFonts = DragonFonts

class RateLimiter:
    """Begrenzt Update-Rate für GUI-Stabilität mit Thread-Sicherheit"""
    def __init__(self, max_updates_per_second=30):
        self.min_interval = 1.0 / max_updates_per_second
        self.last_calls = {}
        self._lock = threading.RLock()
    
    def can_update(self, update_type="default"):
        with self._lock:
            now = time.time()
            if update_type not in self.last_calls:
                self.last_calls[update_type] = 0

            last = self.last_calls[update_type]

            if now - last >= self.min_interval:
                self.last_calls[update_type] = now
                return True
            return False
    
    def reset(self, update_type=None):
        """Setzt Timer zurück"""
        with self._lock:
            if update_type is None:
                self.last_calls.clear()
            elif update_type in self.last_calls:
                del self.last_calls[update_type]

class DarkMessageBox:
    """    🎨 DARK MODE MESSAGEBOX - Stabil, sicher & benutzerfreundlich    """

    @staticmethod
    def showinfo(title, message, parent=None):
        """ℹ️ Zeige Informations-Dialog"""
        return DarkMessageBox._show_dialog(title, message, "info", parent)

    @staticmethod
    def showwarning(title, message, parent=None):
        """⚠️ Zeige Warnungs-Dialog"""
        return DarkMessageBox._show_dialog(title, message, "warning", parent)

    @staticmethod
    def showerror(title, message, parent=None):
        """❌ Zeige Fehler-Dialog"""
        return DarkMessageBox._show_dialog(title, message, "error", parent)

    @staticmethod
    def askokcancel(title, message, parent=None):
        """🤔 Zeige Bestätigungs-Dialog (OK/Abbrechen)"""
        return DarkMessageBox._show_dialog(title, message, "question", parent, buttons=True)

    @staticmethod
    def askyesno(title, message, parent=None):
        """✅❌ Zeige Ja/Nein Dialog"""
        return DarkMessageBox._ask_yesno(title, message, parent)

    @staticmethod
    def _show_dialog(title, message, msg_type, parent=None, buttons=False):
        try:
            if parent is None:
                parent = tk._default_root

            if not parent or not hasattr(parent, 'winfo_exists') or not parent.winfo_exists():
                parent = DarkMessageBox._find_available_parent()

                if not parent:
                    return DarkMessageBox._fallback_messagebox(title, message, msg_type, buttons)

            dialog = tk.Toplevel(parent)
            dialog.title(f"🐉 {title}" if not title.startswith("🐉") else title)
            dialog.configure(bg=ModernColors.BG_PRIMARY)
            dialog.resizable(False, False)
            dialog.transient(parent)
            dialog.grab_set()

            timeout_seconds = 15 if any(word in title.lower() for word in ['beenden', 'exit', 'quit', 'schließen']) else 10
            timeout_id = None

            def auto_close():
                nonlocal timeout_id
                try:
                    if dialog and dialog.winfo_exists():
                        print(f"⚠️ Dialog Timeout nach {timeout_seconds}s: '{title}'")
                        dialog.destroy()
                except Exception:
                    pass

            timeout_id = dialog.after(timeout_seconds * 1000, auto_close)

            icons = {
                "info": ("ℹ️", ModernColors.TEXT_ACCENT),
                "warning": ("⚠️", ModernColors.WARNING),
                "error": ("❌", ModernColors.ERROR),
                "question": ("❓", ModernColors.TEXT_ACCENT),
                "success": ("✅", ModernColors.SUCCESS)
            }
            icon, icon_color = icons.get(msg_type, ("💬", ModernColors.TEXT_PRIMARY))

            main_frame = tk.Frame(dialog, bg=ModernColors.BG_PRIMARY, padx=25, pady=25)
            main_frame.pack(fill='both', expand=True)

            content_frame = tk.Frame(main_frame, bg=ModernColors.BG_PRIMARY)
            content_frame.pack(fill='both', expand=True, pady=(0, 20))

            icon_frame = tk.Frame(content_frame, bg=ModernColors.BG_PRIMARY, width=60)
            icon_frame.pack(side='left', fill='y')
            icon_frame.pack_propagate(False)

            icon_label = tk.Label(icon_frame, text=icon,
                                 font=("Segoe UI", 24),
                                 bg=ModernColors.BG_PRIMARY,
                                 fg=icon_color)
            icon_label.pack(expand=True)

            message_frame = tk.Frame(content_frame, bg=ModernColors.BG_PRIMARY)
            message_frame.pack(side='left', fill='both', expand=True, padx=(20, 0))

            if len(title) > 30:
                title_label = tk.Label(message_frame, text=title,
                                      font=ModernFonts.SUBTITLE,
                                      bg=ModernColors.BG_PRIMARY,
                                      fg=ModernColors.TEXT_PRIMARY,
                                      justify='left', anchor='w')
                title_label.pack(anchor='w', pady=(0, 10))

            message_label = tk.Label(message_frame, text=message,
                                    font=ModernFonts.PRIMARY,
                                    bg=ModernColors.BG_PRIMARY,
                                    fg=ModernColors.TEXT_PRIMARY,
                                    justify='left',
                                    wraplength=350,
                                    anchor='w')
            message_label.pack(fill='x', expand=True, anchor='w')

            button_frame = tk.Frame(main_frame, bg=ModernColors.BG_PRIMARY)
            button_frame.pack(fill='x')

            result = {"value": None}

            def set_result(value):
                nonlocal timeout_id

                if timeout_id:
                    try:
                        dialog.after_cancel(timeout_id)
                    except:
                        pass
                    timeout_id = None

                result["value"] = value

                try:
                    if dialog.winfo_exists():
                        dialog.destroy()
                except:
                    pass

            if buttons:
                cancel_btn = tk.Button(button_frame, text="Abbrechen",
                                      command=lambda: set_result(False),
                                      bg=ModernColors.BG_TERTIARY,
                                      fg=ModernColors.TEXT_PRIMARY,
                                      font=ModernFonts.BUTTON,
                                      relief='flat',
                                      padx=22, pady=8,
                                      cursor='hand2',
                                      takefocus=True)
                cancel_btn.pack(side='right', padx=(10, 0))

                ok_btn = tk.Button(button_frame, text="OK",
                                  command=lambda: set_result(True),
                                  bg=ModernColors.SUCCESS,
                                  fg=ModernColors.TEXT_PRIMARY,
                                  font=ModernFonts.BUTTON,
                                  relief='flat',
                                  padx=25, pady=8,
                                  cursor='hand2',
                                  takefocus=True)
                ok_btn.pack(side='right')

                dialog.bind('<Return>', lambda e: set_result(True))
                dialog.bind('<Escape>', lambda e: set_result(False))
                dialog.bind('<space>', lambda e: cancel_btn.focus_set())

                is_exit_dialog = any(word in title.lower() for word in ['beenden', 'exit', 'quit', 'schließen'])
                if is_exit_dialog:
                    cancel_btn.focus_set()
                else:
                    ok_btn.focus_set()

            else:
                ok_btn = tk.Button(button_frame, text="OK",
                                  command=lambda: set_result(True),
                                  bg=ModernColors.SUCCESS,
                                  fg=ModernColors.TEXT_PRIMARY,
                                  font=ModernFonts.BUTTON,
                                  relief='flat',
                                  padx=25, pady=8,
                                  cursor='hand2',
                                  takefocus=True)
                ok_btn.pack(side='right')

                dialog.bind('<Return>', lambda e: set_result(True))
                dialog.bind('<Escape>', lambda e: set_result(True))
                dialog.bind('<space>', lambda e: set_result(True))

                ok_btn.focus_set()

            def on_closing():
                set_result(False if buttons else True)

            dialog.protocol("WM_DELETE_WINDOW", on_closing)

            DarkMessageBox._center_dialog(dialog, parent)

            parent.wait_window(dialog)

            return result["value"]

        except Exception as e:
            print(f"⚠️ DarkMessageBox Error: {e}")
            import traceback
            traceback.print_exc()

            return DarkMessageBox._fallback_messagebox(title, message, msg_type, buttons)

    @staticmethod
    def _ask_yesno(title, message, parent=None):
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
            dialog.configure(bg=ModernColors.BG_PRIMARY)
            dialog.resizable(False, False)
            dialog.transient(parent)
            dialog.grab_set()

            timeout_id = dialog.after(15000, lambda: dialog.destroy() if dialog.winfo_exists() else None)

            main_frame = tk.Frame(dialog, bg=ModernColors.BG_PRIMARY, padx=30, pady=25)
            main_frame.pack(fill='both', expand=True)

            icon_label = tk.Label(main_frame, text="❓",
                                 font=("Segoe UI", 28),
                                 bg=ModernColors.BG_PRIMARY,
                                 fg=ModernColors.TEXT_ACCENT)
            icon_label.pack(pady=(0, 20))

            message_label = tk.Label(main_frame, text=message,
                                    font=ModernFonts.PRIMARY,
                                    bg=ModernColors.BG_PRIMARY,
                                    fg=ModernColors.TEXT_PRIMARY,
                                    justify='center',
                                    wraplength=350)
            message_label.pack(pady=(0, 30))

            button_frame = tk.Frame(main_frame, bg=ModernColors.BG_PRIMARY)
            button_frame.pack(fill='x')

            result = {"value": None}

            def set_result(value):
                if timeout_id:
                    try:
                        dialog.after_cancel(timeout_id)
                    except:
                        pass

                result["value"] = value
                if dialog.winfo_exists():
                    dialog.destroy()

            def on_yes():
                set_result(True)

            def on_no():
                set_result(False)

            yes_btn = tk.Button(button_frame, text="  👍 Ja  ",
                              command=on_yes,
                              bg=ModernColors.SUCCESS,
                              fg=ModernColors.TEXT_PRIMARY,
                              font=("Segoe UI", 10, "bold"),
                              relief='flat',
                              padx=25, pady=10,
                              cursor='hand2')
            yes_btn.pack(side='left', expand=True, padx=(0, 10))

            no_btn = tk.Button(button_frame, text="  👎 Nein  ",
                             command=on_no,
                             bg=ModernColors.ERROR,
                              fg=ModernColors.TEXT_PRIMARY,
                              font=("Segoe UI", 10, "bold"),
                              relief='flat',
                              padx=25, pady=10,
                              cursor='hand2')
            no_btn.pack(side='right', expand=True)

            dialog.bind('<Return>', lambda e: on_yes())
            dialog.bind('<Escape>', lambda e: on_no())
            dialog.bind('y', lambda e: on_yes())
            dialog.bind('n', lambda e: on_no())

            yes_btn.focus_set()
            DarkMessageBox._center_dialog(dialog, parent)
            parent.wait_window(dialog)

            return result["value"]

        except Exception:
            import tkinter.messagebox as mb
            return mb.askyesno(title, message)

    @staticmethod
    def _find_available_parent():
        try:
            if not tk._default_root:
                return None

            for widget in tk._default_root.winfo_children():
                if hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                    return widget

            return tk._default_root

        except Exception:
            return None

    @staticmethod
    def _center_dialog(dialog, parent):
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
    def _fallback_messagebox(title, message, msg_type, buttons=False):
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
            print(f"💬 {title}: {message}")
            if buttons:
                return False
            return None

    @staticmethod
    def show_progress(title, message, parent=None, indeterminate=True):
        dialog = None
        progress = None

        def create_dialog():
            nonlocal dialog, progress

            try:
                root_window = parent if parent else tk._default_root
                dialog = tk.Toplevel(root_window)
                dialog.title(f"🐉 {title}")
                dialog.configure(bg=ModernColors.BG_PRIMARY)
                dialog.resizable(False, False)
                dialog.transient(root_window)

                main_frame = tk.Frame(dialog, bg=ModernColors.BG_PRIMARY, padx=30, pady=25)
                main_frame.pack(fill='both', expand=True)

                message_label = tk.Label(main_frame, text=message,
                                        font=ModernFonts.PRIMARY,
                                        bg=ModernColors.BG_PRIMARY,
                                        fg=ModernColors.TEXT_PRIMARY,
                                        justify='center')
                message_label.pack(pady=(0, 20))

                progress = ttk.Progressbar(main_frame,
                                          mode='indeterminate' if indeterminate else 'determinate',
                                          length=300)
                progress.pack(pady=(0, 10))

                progress.start(10)

                DarkMessageBox._center_dialog(dialog, root_window)

            except Exception as e:
                print(f"⚠️ Progress Dialog Error: {e}")

        def close_dialog():
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
            def __init__(self):
                self.dialog = dialog
                self.progress = progress

            def close(self):
                close_dialog()

            def update_message(self, new_message):
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

class ExcellenceMemoryManager:

    def __init__(self):
        self._buffers = {}
        self._buffer_sizes = {}
        self._lock = threading.RLock()
        self._max_memory_per_component = 100 * 1024 * 1024
        self._last_gc_time = time.time()
        self._gc_interval = 300
        self._ring_buffers = {}
        self._ring_buffer_pointers = {}
        self._ring_buffer_sizes = {}
        self._memory_warning_threshold = 0.8
        self._long_term_monitor = collections.deque(maxlen=1000)
        self._monitoring_active = True
        self._cleanup_thread = None
        self._health_check_thread = None
        self._start_periodic_cleanup()
        self._start_memory_guardian()

    def _start_memory_guardian(self):
        """Startet Hintergrund-Thread für Memory Health Checks"""
        def memory_guardian_worker():
            while self._monitoring_active:
                try:
                    time.sleep(60)
                    self._perform_memory_health_check()
                except Exception as e:
                    print(f"⚠️ Memory Guardian Error: {e}")
                    time.sleep(30)

        self._health_check_thread = threading.Thread(
            target=memory_guardian_worker,
            daemon=True,
            name="MemoryGuardian"
        )
        self._health_check_thread.start()

    def _perform_memory_health_check(self):
        """Prüft Memory Health und führt Optimierungen durch"""
        try:
            system_memory = psutil.virtual_memory()
            system_usage_percent = system_memory.percent / 100.0

            process = psutil.Process()
            process_memory = process.memory_info().rss
            process_usage_percent = process_memory / ExcellenceConfig.MAX_MEMORY_USAGE

            memory_sample = {
                'timestamp': time.time(),
                'system_usage': system_usage_percent,
                'process_usage': process_usage_percent,
                'system_mb': system_memory.used // (1024 * 1024),
                'process_mb': process_memory // (1024 * 1024)
            }
            self._long_term_monitor.append(memory_sample)

            if system_usage_percent > self._memory_warning_threshold:
                print(f"⚠️ High system memory usage: {system_memory.percent:.1f}%")

            if process_usage_percent > self._memory_warning_threshold:
                print(f"⚠️ High process memory usage: {process_usage_percent:.1%}")
                self._aggressive_cleanup()

            if len(self._long_term_monitor) >= 10:
                recent_samples = list(self._long_term_monitor)[-10:]
                avg_usage = sum(s['system_usage'] for s in recent_samples) / len(recent_samples)
                if avg_usage > 0.75:
                    print(f"⚠️ Sustained high memory usage: {avg_usage:.1%}")

        except Exception as e:
            print(f"⚠️ Memory health check error: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Gibt detaillierte Memory Statistics zurück"""
        try:
            system_memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss

            with self._lock:
                buffer_count = len(self._buffers)
                ring_buffer_count = len(self._ring_buffers)
                total_buffer_size = sum(self._buffer_sizes.values())

            return {
                'system_usage_percent': system_memory.percent,
                'system_used_mb': system_memory.used // (1024 * 1024),
                'system_total_mb': system_memory.total // (1024 * 1024),
                'process_usage_percent': (process_memory / ExcellenceConfig.MAX_MEMORY_USAGE) * 100,
                'process_used_mb': process_memory // (1024 * 1024),
                'process_peak_mb': self._get_peak_memory() // (1024 * 1024),
                'long_term_samples': len(self._long_term_monitor),
                'buffer_components': buffer_count,
                'ring_buffer_components': ring_buffer_count,
                'total_buffer_size_mb': total_buffer_size // (1024 * 1024),
                'active_monitoring': self._monitoring_active
            }
        except Exception as e:
            print(f"⚠️ Memory stats error: {e}")
            return {}

    def _get_peak_memory(self) -> int:
        """Gibt Peak Memory Usage zurück"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0

    def _start_periodic_cleanup(self):
        """Startet periodischen Cleanup Thread"""
        def periodic_cleanup():
            while True:
                time.sleep(60)
                try:
                    self._perform_periodic_maintenance()
                except Exception as e:
                    print(f"⚠️ Periodic cleanup error: {e}")

        self._cleanup_thread = threading.Thread(
            target=periodic_cleanup,
            daemon=True,
            name="PeriodicCleanup"
        )
        self._cleanup_thread.start()

    def _perform_periodic_maintenance(self):
        """Führt periodische Wartung durch"""
        with self._lock:
            current_time = time.time()

            if current_time - self._last_gc_time > self._gc_interval:
                gc.collect()
                self._last_gc_time = current_time

            total_memory = sum(self._buffer_sizes.values())
            memory_usage_percent = total_memory / self._max_memory_per_component

            if memory_usage_percent > 0.8:
                print(f"⚠️ High buffer memory: {memory_usage_percent:.1%}")

                def async_aggressive_cleanup():
                    try:
                        self._aggressive_cleanup()
                    except Exception as e:
                        print(f"⚠️ Aggressive cleanup error: {e}")

                cleanup_thread = threading.Thread(
                    target=async_aggressive_cleanup,
                    daemon=True
                )
                cleanup_thread.start()

    def _aggressive_cleanup(self):
        """Aggressive Cleanup bei Memory Pressure"""
        print("🧹 Starting aggressive memory cleanup...")

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
                            if isinstance(current_buffer, collections.deque):
                                buffer_list = list(current_buffer)
                            else:
                                buffer_list = current_buffer

                            new_deque = collections.deque(
                                buffer_list[-keep_count:],
                                maxlen=ExcellenceConfig.MAX_TEXT_LINES
                            )
                            self._buffers[component] = new_deque
                            self._buffer_sizes[component] = sum(
                                len(str(text).encode('utf-8')) if text else 0
                                for text in new_deque
                            )
                            print(f"  ↪ {component}: {buffer_size} → {keep_count} entries")
            except Exception as e:
                print(f"⚠️ Component cleanup error for {component}: {e}")

        def async_gc():
            for _ in range(3):
                gc.collect()
                time.sleep(0.01)

        gc_thread = threading.Thread(target=async_gc, daemon=True)
        gc_thread.start()

        print("✅ Aggressive cleanup completed")

    def add_text(self, component: str, text: str):
        """Fügt Text zu einem Buffer hinzu mit Memory Management"""
        if not text or not text.strip():
            return

        with self._lock:
            if component in self._ring_buffers:
                self._add_to_ring_buffer(component, text)
                return

            if component not in self._buffers:
                self._buffers[component] = collections.deque(
                    maxlen=ExcellenceConfig.MAX_TEXT_LINES
                )
                self._buffer_sizes[component] = 0

            text_size = len(text.encode('utf-8'))
            current_size = self._buffer_sizes[component]

            if current_size + text_size > self._max_memory_per_component:
                self._optimize_buffer(component)

            self._buffers[component].append(text)
            self._buffer_sizes[component] += text_size

            if len(self._buffers[component]) % 100 == 0:
                def async_gc_collect():
                    gc.collect()

                gc_thread = threading.Thread(target=async_gc_collect, daemon=True)
                gc_thread.start()

    def _add_to_ring_buffer(self, component: str, text: str):
        """Fügt Text zu einem Ring Buffer hinzu"""
        if component not in self._ring_buffers:
            buffer_size = ExcellenceConfig.MAX_TEXT_LINES
            self._ring_buffers[component] = [None] * buffer_size
            self._ring_buffer_pointers[component] = 0
            self._ring_buffer_sizes[component] = 0
            self._buffer_sizes[component] = 0

        ring_buffer = self._ring_buffers[component]
        pointer = self._ring_buffer_pointers[component]
        text_size = len(text.encode('utf-8'))

        old_text = ring_buffer[pointer]
        if old_text is not None:
            old_size = len(old_text.encode('utf-8'))
            self._buffer_sizes[component] -= old_size

        ring_buffer[pointer] = text
        self._buffer_sizes[component] += text_size

        self._ring_buffer_pointers[component] = (pointer + 1) % len(ring_buffer)

        if self._ring_buffer_sizes[component] < len(ring_buffer):
            self._ring_buffer_sizes[component] += 1

    def _optimize_buffer(self, component: str):
        """Optimiert Buffer Größe bei Memory Pressure"""
        if component in self._ring_buffers:
            current_size = self._ring_buffer_sizes[component]
            if current_size > ExcellenceConfig.MAX_TEXT_LINES // 2:
                new_size = ExcellenceConfig.MAX_TEXT_LINES // 2
                self._resize_ring_buffer(component, new_size)
            return

        if component in self._buffers:
            keep_count = int(len(self._buffers[component]) * 0.7)
            if keep_count > 0:
                new_deque = collections.deque(
                    list(self._buffers[component])[-keep_count:],
                    maxlen=ExcellenceConfig.MAX_TEXT_LINES
                )
                self._buffers[component] = new_deque
                self._buffer_sizes[component] = sum(
                    len(text.encode('utf-8')) for text in self._buffers[component]
                )
                print(f"🧹 Buffer {component} optimized: {keep_count} entries kept")

    def _resize_ring_buffer(self, component: str, new_size: int):
        """Ändert Größe eines Ring Buffers"""
        if component not in self._ring_buffers:
            return

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
                new_total_size += len(text.encode('utf-8'))
                new_pointer = (new_pointer + 1) % new_size
                new_buffer_size += 1

        self._ring_buffers[component] = new_buffer
        self._ring_buffer_pointers[component] = new_pointer
        self._ring_buffer_sizes[component] = new_buffer_size
        self._buffer_sizes[component] = new_total_size

        print(f"🧹 Ring buffer {component} resized: {old_size} → {new_size}")

    def get_text(self, component: str) -> str:
        """Holt Text aus einem Buffer"""
        with self._lock:
            if component in self._ring_buffers:
                return self._get_from_ring_buffer(component)
            elif component in self._buffers:
                return '\n'.join(self._buffers[component])
            return ""

    def _get_from_ring_buffer(self, component: str) -> str:
        """Holt Text aus einem Ring Buffer"""
        if component not in self._ring_buffers:
            return ""

        ring_buffer = self._ring_buffers[component]
        pointer = self._ring_buffer_pointers[component]
        buffer_size = self._ring_buffer_sizes[component]
        total_size = len(ring_buffer)

        if buffer_size == 0:
            return ""

        texts = []
        for i in range(buffer_size):
            idx = (pointer - buffer_size + i) % total_size
            if idx < 0:
                idx += total_size
            text = ring_buffer[idx]
            if text is not None:
                texts.append(text)

        return '\n'.join(texts)

    def clear_component(self, component: str):
        """Löscht alle Daten einer Komponente"""
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

            print(f"🧹 Component {component} cleared")

    def get_buffer_stats(self, component: str) -> Dict[str, Any]:
        """Gibt Statistiken für einen Buffer zurück"""
        with self._lock:
            if component in self._ring_buffers:
                return {
                    'type': 'ring_buffer',
                    'size': self._ring_buffer_sizes.get(component, 0),
                    'capacity': len(self._ring_buffers[component]),
                    'memory_bytes': self._buffer_sizes.get(component, 0),
                    'pointer': self._ring_buffer_pointers.get(component, 0)
                }
            elif component in self._buffers:
                return {
                    'type': 'deque',
                    'size': len(self._buffers[component]),
                    'capacity': ExcellenceConfig.MAX_TEXT_LINES,
                    'memory_bytes': self._buffer_sizes.get(component, 0),
                    'maxlen': ExcellenceConfig.MAX_TEXT_LINES
                }
            return {'type': 'not_found'}

    def list_components(self) -> List[str]:
        """Listet alle aktiven Komponenten auf"""
        with self._lock:
            all_components = set(self._buffers.keys())
            all_components.update(self._ring_buffers.keys())
            return list(all_components)

    def get_total_memory_usage(self) -> int:
        """Gibt totale Memory Usage in Bytes zurück"""
        with self._lock:
            return sum(self._buffer_sizes.values())

    def optimize_all_buffers(self):
        """Optimiert alle Buffers"""
        print("🧹 Optimizing all buffers...")
        with self._lock:
            components = list(self._buffers.keys()) + list(self._ring_buffers.keys())

        for component in components:
            self._optimize_buffer(component)

        gc.collect()
        print("✅ All buffers optimized")

    def dispose(self):
        """Gibt alle Ressourcen frei"""
        self._monitoring_active = False

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            time.sleep(0.1)

        if self._health_check_thread and self._health_check_thread.is_alive():
            time.sleep(0.1)

        with self._lock:
            self._buffers.clear()
            self._buffer_sizes.clear()
            self._ring_buffers.clear()
            self._ring_buffer_pointers.clear()
            self._ring_buffer_sizes.clear()
            self._long_term_monitor.clear()

        gc.collect()

        #print("🧹 Memory Manager disposed")

    def __del__(self):
        """Destruktor für zusätzliche Cleanup"""
        try:
            if self._monitoring_active:
                self.dispose()
        except:
            pass

    def print_debug_info(self):
        """Gibt Debug Informationen aus"""
        stats = self.get_memory_stats()
        print("\n" + "="*50)
        print("🧠 MEMORY MANAGER DEBUG INFO")
        print("="*50)
        print(f"System Memory: {stats.get('system_used_mb', 0)}MB / {stats.get('system_total_mb', 0)}MB ({stats.get('system_usage_percent', 0):.1f}%)")
        print(f"Process Memory: {stats.get('process_used_mb', 0)}MB ({stats.get('process_usage_percent', 0):.1f}%)")
        print(f"Buffer Components: {stats.get('buffer_components', 0)}")
        print(f"Ring Buffer Components: {stats.get('ring_buffer_components', 0)}")
        print(f"Total Buffer Size: {stats.get('total_buffer_size_mb', 0)}MB")
        print(f"Long Term Samples: {stats.get('long_term_samples', 0)}")

        components = self.list_components()
        if components:
            print(f"\nActive Components ({len(components)}):")
            for comp in components[:5]:  # Nur erste 5 zeigen
                comp_stats = self.get_buffer_stats(comp)
                print(f"  • {comp}: {comp_stats['type']}, size: {comp_stats.get('size', 0)}")
            if len(components) > 5:
                print(f"  ... and {len(components)-5} more")
        print("="*50)

SUPPORTED_LANGUAGES = {
    'auto': 'Automatisch',
    'de': 'Deutsch', 'en': 'Englisch', 'fr': 'Französisch', 'es': 'Spanisch',
    'it': 'Italienisch', 'pt': 'Portugiesisch', 'nl': 'Niederländisch',
    'pl': 'Polnisch', 'ru': 'Russisch', 'ja': 'Japanisch',
    'zh': 'Chinesisch', 'ko': 'Koreanisch', 'ar': 'Arabisch',
    'hi': 'Hindi', 'tr': 'Türkisch', 'vi': 'Vietnamesisch',
    'th': 'Thailändisch', 'id': 'Indonesisch', 'ms': 'Malaysisch',
    'fa': 'Persisch', 'he': 'Hebräisch', 'bn': 'Bengalisch',
    'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam',
    'kn': 'Kannada', 'mr': 'Marathi', 'gu': 'Gujarati',
    'pa': 'Punjabi', 'ur': 'Urdu', 'sv': 'Schwedisch',
    'da': 'Dänisch', 'no': 'Norwegisch', 'fi': 'Finnisch',
    'cs': 'Tschechisch', 'hu': 'Ungarisch', 'ro': 'Rumänisch',
    'bg': 'Bulgarisch', 'el': 'Griechisch', 'sk': 'Slowakisch',
    'hr': 'Kroatisch', 'sr': 'Serbisch', 'uk': 'Ukrainisch',
    'ca': 'Katalanisch', 'eu': 'Baskisch', 'gl': 'Galizisch'
}

SORTED_LANGUAGES = sorted([(name, code) for code, name in SUPPORTED_LANGUAGES.items()], key=lambda x: x[0])

LANGUAGE_SHORT_CODES = {
    'auto': 'Auto', 'de': 'Deu', 'en': 'Eng', 'fr': 'Fra', 'es': 'Esp',
    'it': 'Ita', 'pt': 'Por', 'nl': 'Nld', 'pl': 'Pol', 'ru': 'Rus',
    'ja': 'Jpn', 'zh': 'Chi', 'ko': 'Kor', 'ar': 'Ara', 'hi': 'Hin',
    'tr': 'Tur', 'vi': 'Vie', 'th': 'Tha', 'id': 'Ind', 'ms': 'Msa',
    'fa': 'Per', 'he': 'Heb', 'sv': 'Swe', 'da': 'Dan', 'no': 'Nor',
    'fi': 'Fin', 'cs': 'Cze', 'hu': 'Hun', 'ro': 'Rom', 'bg': 'Bul',
    'el': 'Gre', 'sk': 'Slo', 'hr': 'Hrv', 'uk': 'Ukr'
}

COUNTRY_FLAGS = {
    'de': '🇩🇪', 'en': '🇬🇧', 'fr': '🇫🇷', 'es': '🇪🇸', 'it': '🇮🇹',
    'pt': '🇵🇹', 'nl': '🇳🇱', 'pl': '🇵🇱', 'ru': '🇷🇺', 'ja': '🇯🇵',
    'zh': '🇨🇳', 'ko': '🇰🇷', 'ar': '🇸🇦', 'hi': '🇮🇳', 'tr': '🇹🇷',
    'vi': '🇻🇳', 'th': '🇹🇭', 'id': '🇮🇩', 'ms': '🇲🇾', 'fa': '🇮🇷',
    'he': '🇮🇱', 'bn': '🇧🇩', 'ta': '🇮🇳', 'te': '🇮🇳', 'ml': '🇮🇳',
    'kn': '🇮🇳', 'mr': '🇮🇳', 'gu': '🇮🇳', 'pa': '🇮🇳', 'ur': '🇵🇰',
    'sv': '🇸🇪', 'da': '🇩🇰', 'no': '🇳🇴', 'fi': '🇫🇮', 'cs': '🇨🇿',
    'hu': '🇭🇺', 'ro': '🇷🇴', 'bg': '🇧🇬', 'el': '🇬🇷', 'sk': '🇸🇰',
    'hr': '🇭🇷', 'sr': '🇷🇸', 'uk': '🇺🇦', 'ca': '🏴', 'eu': '🏴',
    'gl': '🏴', 'auto': '🌐', 'unknown': '🏳️'
}

WHISPER_MODELS = [
    "tiny", "tiny.en", "base", "base.en",
    "small", "small.en", "medium", "medium.en",
    "large-v2", "large-v3"
]

class AsianLanguageSupport:
    """Optimizations for Asian languages including word segmentation."""

    @staticmethod
    def should_use_word_segmentation(language_code):
        return language_code in ['zh', 'ja', 'ko', 'th']

    @staticmethod
    def optimize_display_text(text, language_code):
        if language_code == 'zh':
            return ' '.join(text)
        elif language_code == 'ja':
            return text.replace('。', '. ').replace('、', ', ')
        elif language_code == 'ko':
            return text
        return text

class AdvancedSettings:
    """Advanced settings for optimized performance and AI model configuration."""    
    def __init__(self, 
                 beam_size: int = 5, 
                 temperature: float = 0.0, 
                 vad_filter: bool = True,
                 max_cache_size: int = 200, 
                 auto_save_interval: int = 300,
                 enable_sentiment_analysis: bool = False, 
                 enable_speaker_diarization: bool = False,
                 max_memory_mb: int = 1024, 
                 gpu_acceleration: bool = True,
                 optimize_translations: bool = False,
                 config_type: str = 'default'):
        """        Args:            config_type: 'default', 'realtime', 'high_accuracy', 'youtube'        """
        
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
        
        print(f"🔊 AdvancedSettings initialized:")
        print(f"   Config Type: {config_type}")
        print(f"   SAMPLE_RATE: {self.config.SAMPLE_RATE}")
        print(f"   CHANNELS: {self.config.CHANNELS}")
        print(f"   CHUNK_DURATION: {self.chunk_duration}s")
        print(f"   CHUNK_SIZE: {self.config.CHUNK_SIZE_BYTES:,} bytes")
        print(f"   BEAM_SIZE: {self.beam_size}")
        print(f"   GPU_ACCELERATION: {self.gpu_acceleration}")
    
    @classmethod
    def load_from_file(cls, filename="dragon_advanced_settings.json"):
        """Lädt Einstellungen und verwendet deine ExcellenceConfig"""
        try:
            config_dir = PlatformUtils.get_platform_config_dir()
            file_path = config_dir / filename
        
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
                import inspect
                signature = inspect.signature(cls.__init__)
                valid_params = list(signature.parameters.keys())
            
                if 'self' in valid_params:
                    valid_params.remove('self')
            
                filtered_data = {}
                for key in valid_params:
                    if key in data:
                        filtered_data[key] = data[key]
            
                config_type = data.get('config_type', 'default')
                if 'config_type' not in filtered_data and 'config_type' in data:
                    filtered_data['config_type'] = config_type
            
                audio_params_to_restore = {
                    'chunk_duration': None,
                    'sample_rate': None,
                    'channels': None,
                    'audio_format': None
                }
            
                for audio_param in audio_params_to_restore.keys():
                    if audio_param in data:
                        audio_params_to_restore[audio_param] = data[audio_param]
            
                instance = cls(**filtered_data)
            
                restored_params = []
                if audio_params_to_restore['chunk_duration'] is not None:
                    try:
                        instance.config.CHUNK_DURATION = float(audio_params_to_restore['chunk_duration'])
                        restored_params.append(f"chunk_duration: {audio_params_to_restore['chunk_duration']}s")
                    except (ValueError, AttributeError):
                        pass
            
                if audio_params_to_restore['sample_rate'] is not None:
                    try:
                        instance.config.SAMPLE_RATE = int(audio_params_to_restore['sample_rate'])
                        restored_params.append(f"sample_rate: {audio_params_to_restore['sample_rate']}Hz")
                    except (ValueError, AttributeError):
                        pass
            
                if audio_params_to_restore['channels'] is not None:
                    try:
                        instance.config.CHANNELS = int(audio_params_to_restore['channels'])
                        restored_params.append(f"channels: {audio_params_to_restore['channels']}")
                    except (ValueError, AttributeError):
                        pass
            
                if audio_params_to_restore['audio_format'] is not None:
                    try:
                        instance.config.AUDIO_FORMAT = str(audio_params_to_restore['audio_format'])
                        restored_params.append(f"audio_format: {audio_params_to_restore['audio_format']}")
                    except (ValueError, AttributeError):
                        pass
            
                if restored_params:
                    print(f"   ↪ Restored audio params: {', '.join(restored_params)}")
            
                print(f"✅ Settings loaded successfully (Config Type: {config_type})")
                return instance
            
        except Exception as e:
            print(f"❌ Error loading advanced settings: {e}")
            import traceback
            traceback.print_exc()
            print(f"📝 Using default settings with ExcellenceConfig")
    
        return cls()
    
    def save_to_file(self, filename="dragon_advanced_settings.json"):
        """Speichert Einstellungen inkl. Config Type"""
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
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_dict, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Settings saved to {file_path} (Config Type: {config_type})")
            
        except Exception as e:
            print(f"❌ Error saving settings: {e}")
    
    def repair(self):
        """Repariert/aktualisiert fehlende oder veraltete Attribute"""
        print(f"🔧 Repairing AdvancedSettings...")
        
        repairs_made = []
        
        if not hasattr(self, 'config'):
            self.config = ExcellenceConfig()
            repairs_made.append('Added ExcellenceConfig')
        
        if not hasattr(self, 'SAMPLE_RATE'):
            self.SAMPLE_RATE = self.config.SAMPLE_RATE
            repairs_made.append(f'Added SAMPLE_RATE: {self.SAMPLE_RATE}Hz')
        
        if not hasattr(self, 'CHANNELS'):
            self.CHANNELS = self.config.CHANNELS
            repairs_made.append(f'Added CHANNELS: {self.CHANNELS}')
        
        if not hasattr(self, 'AUDIO_FORMAT'):
            self.AUDIO_FORMAT = self.config.AUDIO_FORMAT
            repairs_made.append(f'Added AUDIO_FORMAT: {self.AUDIO_FORMAT}')
        
        if not hasattr(self, 'CHUNK_SIZE_BYTES'):
            self.CHUNK_SIZE_BYTES = self.config.CHUNK_SIZE_BYTES
            repairs_made.append(f'Added CHUNK_SIZE_BYTES: {self.CHUNK_SIZE_BYTES:,}')
        

        if not hasattr(self, 'chunk_duration'):
            self.chunk_duration = self.config.CHUNK_DURATION
            repairs_made.append(f'Added chunk_duration from config: {self.chunk_duration}s')
        
        old_audio_attrs = ['sample_rate', 'channels', 'audio_format', 'chunk_size_bytes']
        for attr in old_audio_attrs:
            if hasattr(self, attr):
                old_value = getattr(self, attr)
                new_attr = attr.upper()
                if hasattr(self.config, new_attr):
                    setattr(self, attr, getattr(self.config, new_attr))
                    repairs_made.append(f'{attr}: {old_value} → {getattr(self.config, new_attr)}')
        
        if not self.config.validate_config():
            print(f"⚠️ Config validation failed, resetting to default")
            self.config = ExcellenceConfig()
            repairs_made.append('Config reset to default')
        
        if repairs_made:
            print(f"✅ Repairs made: {', '.join(repairs_made)}")
            self.save_to_file()
        else:
            print(f"✅ No repairs needed")
        
        return repairs_made
    
    def validate(self):
        """Validiert alle Einstellungen auf Plausibilität"""
        issues = []
        
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
    
    def set_config_type(self, config_type: str):
        """Ändert den Config-Typ dynamisch"""
        valid_types = ['default', 'realtime', 'high_accuracy', 'youtube']
        
        if config_type not in valid_types:
            print(f"⚠️ Invalid config_type: {config_type}. Must be one of: {valid_types}")
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
        
        print(f"🔄 Config type changed: {old_config_type} → {config_type}")
        print(f"   New CHUNK_DURATION: {self.chunk_duration}s")
        print(f"   New CHUNK_SIZE: {self.config.CHUNK_SIZE_BYTES:,} bytes")
        
        return True
    
    def get_audio_filter(self, language: Optional[str] = None, profile: Optional[str] = None) -> str:
        """Holt optimierten Audio-Filter von deiner Config"""
        return self.config.get_audio_filter(language, profile)
    
    def get_youtube_headers(self, is_manifest: bool = False) -> Dict[str, str]:
        """Holt YouTube-Headers von deiner Config"""
        return self.config.get_youtube_headers(is_manifest)
    
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Holt plattformspezifische Konfiguration"""
        return self.config.get_platform_config(platform)
    
    def print_config_summary(self):
        """Gibt detaillierte Konfigurations-Info aus"""
        print("\n" + "="*60)
        print("⚙️ ADVANCED SETTINGS CONFIGURATION")
        print("="*60)
        
        print(f"\n🤖 AI Model Parameters:")
        print(f"  • Beam Size: {self.beam_size}")
        print(f"  • Temperature: {self.temperature}")
        print(f"  • VAD Filter: {self.vad_filter}")
        print(f"  • GPU Acceleration: {self.gpu_acceleration}")
        
        print(f"\n🎵 Audio Configuration (from ExcellenceConfig):")
        print(f"  • Sample Rate: {self.config.SAMPLE_RATE} Hz")
        print(f"  • Channels: {self.config.CHANNELS} ({'Mono' if self.config.CHANNELS == 1 else 'Stereo'})")
        print(f"  • Chunk Duration: {self.config.CHUNK_DURATION}s")
        print(f"  • Chunk Size: {self.config.CHUNK_SIZE_BYTES:,} bytes")
        print(f"  • Bytes/sec: {self.config.BYTES_PER_SECOND:,}")
        print(f"  • Audio Filter Profiles: {len(self.config.FILTER_PROFILES)}")
        print(f"  • Language Filters: {len(self.config.LANGUAGE_FILTERS)} languages")
        
        print(f"\n⚡ Performance Settings:")
        print(f"  • Max Cache Size: {self.max_cache_size}")
        print(f"  • Max Memory: {self.max_memory_mb} MB")
        print(f"  • Auto Save Interval: {self.auto_save_interval}s")
        
        print(f"\n🔧 Features:")
        print(f"  • Sentiment Analysis: {self.enable_sentiment_analysis}")
        print(f"  • Speaker Diarization: {self.enable_speaker_diarization}")
        print(f"  • Optimize Translations: {self.optimize_translations}")
        
        config_type = 'default'
        if isinstance(self.config, RealtimeConfig):
            config_type = 'realtime'
        elif isinstance(self.config, HighAccuracyConfig):
            config_type = 'high_accuracy'
        elif isinstance(self.config, YouTubeOptimizedConfig):
            config_type = 'youtube'
        
        print(f"\n🎯 Config Type: {config_type.upper()}")
        
        issues = self.validate()
        if issues:
            print(f"\n⚠️ Validation Issues:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print(f"\n✅ All settings valid")
        
        print("="*60)
    
    def __repr__(self):
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

class Plugin:
    """Base class for plugins with lifecycle management."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        return result

    def on_translation(self, result: ExcellenceTranslationResult) -> ExcellenceTranslationResult:
        return result

    def on_startup(self):
        pass

    def on_shutdown(self):
        pass

class SentimentAnalysisPlugin(Plugin):
    """Analyze sentiment of transcribed text."""

    def __init__(self):
        super().__init__("Sentiment Analysis", "1.0.0")
        self.sentiment_cache = {}

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        if not result.text.strip():
            return result

        text = result.text.lower()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst']

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        total = positive_count + negative_count
        if total > 0:
            sentiment_score = (positive_count - negative_count) / total
        else:
            sentiment_score = 0.0

        return ExcellenceTranscriptionResult(
            text=result.text,
            confidence=result.confidence,
            language=result.language,
            timestamp=result.timestamp,
            start=result.start,
            end=result.end
        )

class KeywordExtractionPlugin(Plugin):
    """Extract important keywords from text."""

    def __init__(self):
        super().__init__("Keyword Extraction", "1.0.0")
        self.common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        return result

class PluginManager:
    """Manages plugin loading and execution."""

    def __init__(self):
        self.plugins: List[Plugin] = []
        self.enabled = True

    def register_plugin(self, plugin: Plugin):
        self.plugins.append(plugin)

    def load_builtin_plugins(self):
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

class ExcellenceTTLCache:
    """
    TTL-based cache for better hit rate with automatic expiration.
    """

    def __init__(self, maxsize=128, ttl=3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
        self._order = collections.deque()
        self._lock = threading.RLock()
        self._cleanup_interval = 300
        self._last_cleanup = time.time()

    def _is_expired(self, key):
        value, ts = self._cache[key]
        return (time.time() - ts) > self.ttl

    def _remove_key(self, key):
        if key in self._cache:
            self._order.remove(key)
            del self._cache[key]

    def get(self, key):
        with self._lock:
            self._perform_cleanup_if_needed()

            if key in self._cache:
                if time.time() - self._timestamps[key] > self.ttl:
                    self._remove_key(key)
                    return None

                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            return None

    def put(self, key, value):
        with self._lock:
            self._perform_cleanup_if_needed()

            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self.maxsize:
                oldest = self._order.popleft()
                self._remove_key(oldest)

            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._order.append(key)

    def _perform_cleanup_if_needed(self):
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        expired_keys = []
        for key, timestamp in self._timestamps.items():
            if current_time - timestamp > self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_key(key)

        self._last_cleanup = current_time

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._order.clear()
            self._last_cleanup = time.time()

class ExcellenceTranslationEngine:
    """
    Translation engine with functional cache strategy and error recovery.
    """

    def __init__(self, target_lang: str = "de", advanced_settings: AdvancedSettings = None):
        self.target_lang = target_lang
        self.translator = None
        self._cache = ExcellenceTTLCache(maxsize=500, ttl=3600)
        self._lock = threading.RLock()
        self.advanced_settings = advanced_settings or AdvancedSettings()
        self._last_translations = collections.deque(maxlen=15)
        self._setup_translator()
        self.last_detected_language = 'auto'

    def _is_valid_translation(self, original: str, translated: str) -> bool:
        if not translated or not translated.strip():
            return False

        orig_clean = original.strip()
        trans_clean = translated.strip()

        if len(trans_clean) < 2:
            return False

        if trans_clean.isspace():
            return False

        if len(set(trans_clean)) < 3 and len(trans_clean) > 10:
            return False

        orig_len = len(orig_clean)
        trans_len = len(trans_clean)

        if orig_len == 0 or trans_len == 0:
            return False

        ratio = trans_len / max(orig_len, 1)

        return 0.2 <= ratio <= 5.0

    def _validate_length_ratio(self, original: str, translated: str) -> bool:
        orig_len = len(original)
        trans_len = len(translated)

        if orig_len == 0 or trans_len == 0:
            return False

        ratio = trans_len / orig_len

        return 0.2 <= ratio <= 5.0

    def _setup_translator(self):
        try:
            if TRANSLATOR_AVAILABLE:
                GoogleTranslator = FastLazyLoader.load('deep_translator')
                self.translator = GoogleTranslator(source='auto', target=self.target_lang)
            else:
                self.translator = None
        except Exception:
            self.translator = None

    def set_target_language(self, target_lang: str):
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
        if source_lang != 'auto' and source_lang == self.target_lang:
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
                            continue

                    translated_text = self.translator.translate(clean_text)

                    if not translated_text or not translated_text.strip():
                        continue

                    if not self._is_valid_translation(clean_text, translated_text):
                        continue

                    final_translation = self._postprocess_translation(translated_text, clean_text)

                    result = ExcellenceTranslationResult(
                        original=original_text,
                        translated=final_translation,
                        source_lang=source_lang,
                        target_lang=self.target_lang
                    )

                    with self._lock:
                        self._cache.put(cache_key, result)
                        self._last_translations.append(text_hash)
                        if len(self._last_translations) > 20:
                            self._last_translations.popleft()

                    return result

                except Exception:
                    if attempt < 2:
                        time.sleep(0.5)
                        self._setup_translator()

            return None

        except Exception:
            return None

    def dispose(self):
        """Gibt Ressourcen der Translation Engine frei."""
        with self._lock:
            if hasattr(self, '_cache'):
                self._cache.clear()
            if hasattr(self, '_last_translations'):
                self._last_translations.clear()
            self.translator = None
            gc.collect()
            #print("🧹 Translation Engine disposed")

    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ""

        clean_text = text.strip()

        clean_text = re.sub(r'\s+', ' ', clean_text)

        clean_text = re.sub(r'[ ]+([.,!?])', r'\1', clean_text)
        clean_text = re.sub(r'([.,!?])[ ]*', r'\1 ', clean_text)

        common_errors = {
            "bass communi": "best community",
            " ,": ",",
            " .": ".",
            "„": "\"",
            "“": "\""
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

        if not result.endswith(('.', '!', '?', ':', ';')):
            result += '.'

        if result and result[0].islower():
            result = result[0].upper() + result[1:]

        result = re.sub(r'\s+', ' ', result)

        rules = [
            (r'\s+\.', '.'),
            (r'\s+,', ','),
            (r'\s+\?', '?'),
            (r'\s+!', '!'),
            (r' ,', ','),
            (r' \.', '.')
        ]

        for pattern, replacement in rules:
            result = re.sub(pattern, replacement, result)

        return result.strip()

class ExcellenceTranscriptionEngine:
    """🚀 ULTRA-OPTIMIERT: Speech-to-Text engine mit CACHE & GPU"""
    
    def __init__(self, advanced_settings: AdvancedSettings = None):
        self.model = None
        self.model_size = None
        self._lock = threading.RLock()
        self._model_cache = {}
        self._model_cache_size = 2
        self._model_loading = False
        self.advanced_settings = advanced_settings or AdvancedSettings()
        self.config = self.advanced_settings.config
        self._cache = ExcellenceTTLCache(maxsize=self.advanced_settings.max_cache_size)
        self.device, self.compute_type = self._detect_optimal_device()
        self._performance_monitor = SimplePerformanceTracker()
        self._last_transcription_text = ""
        self._active_model_loads = {}
        self._model_loaded_flag = False
        self._model_load_timeout = 180.0
        self._fallback_attempts = 0
        self._max_fallback_attempts = 3        
        self._model_load_locks = {}
        self._active_fallback_loads = set()
        self._disposing = False
        
        print(f"✅ Transcription Engine initialisiert (Device: {self.device}, Cache: enabled)")

    def _detect_optimal_device(self):
        """🚀 OPTIMIERTE GPU-Erkennung für alle Plattformen"""
        device = "cpu"
        compute_type = "int8"
        
        if TORCH_AVAILABLE:
            try:
                torch = FastLazyLoader.load('torch')

                if torch.cuda.is_available():
                    try:
                        test_size = 1024 * 1024
                        a = torch.randn(test_size, device='cuda')
                        b = torch.randn(test_size, device='cuda')
                        torch.cuda.synchronize()
                        c = a + b
                        torch.cuda.synchronize()
                        
                        device = "cuda"
                        compute_type = "float16" if self.advanced_settings.gpu_acceleration else "int8"
                        
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        print(f"✅ NVIDIA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
                        
                    except Exception as e:
                        print(f"⚠️ CUDA test failed, using CPU: {e}")
                        device = "cpu"
                        compute_type = "int8"
                
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    try:
                        device = "mps"
                        compute_type = "float16"
                        print("✅ Apple Silicon GPU (MPS) detected")
                    except Exception as e:
                        print(f"⚠️ MPS failed: {e}")
                        device = "cpu"
                        compute_type = "int8"
                
                else:
                    device = "cpu"
                    compute_type = "int8"
                    if IS_LINUX:
                        print("🔧 Using CPU with Linux optimizations")
                    elif IS_WINDOWS:
                        print("🔧 Using CPU with Windows optimizations")
                    else:
                        print("🔧 Using CPU")
                        
            except Exception as e:
                print(f"⚠️ GPU detection error, using CPU: {e}")
                device = "cpu"
                compute_type = "int8"
        else:
            print("🔧 PyTorch not available, using CPU")
            device = "cpu"
            compute_type = "int8"

        return device, compute_type

    @excellence_execution(timeout=120.0)
    def load_model(self, model_size: str) -> bool:
        """🚀 ULTRA-OPTIMIERT mit Cache, vereinfachter Logik und CPU-Fallback"""
        print(f"🔍 [PERF] load_model called for {model_size}")
        
        if not WHISPER_AVAILABLE:
            print("❌ faster-whisper nicht verfügbar")
            return False
        
        with self._lock:
            if model_size in self._model_cache:
                print(f"⚡ [PERF] Model '{model_size}' aus Cache geladen")
                self.model = self._model_cache[model_size]
                self.model_size = model_size
                return True
            
            if self.model is not None and self.model_size == model_size:
                print(f"✅ Model '{model_size}' bereits aktiv")
                return True
        
        print(f"🔄 [PERF] Loading model {model_size}...")
        
        if self.device == "cuda":
            print("⚠️ CUDA device selected, but will verify availability...")
        
        try:
            if self.model is not None and self.model_size != model_size:
                self._force_model_cleanup()
            
            WhisperModel = FastLazyLoader.load('faster_whisper')
            
            model_dir = PlatformUtils.get_platform_config_dir() / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📥 Downloading/loading {model_size} to {model_dir}...")
            
            load_success = False
            load_error = None
            
            try:
                model = self._try_load_model_with_device(model_size, model_dir, self.device, self.compute_type)
                if model:
                    load_success = True
                    print(f"✅ Model {model_size} loaded successfully on {self.device}")
            except Exception as e:
                load_error = str(e)
                print(f"⚠️ Initial load failed on {self.device}: {load_error}")
                
                if "CUDA" in load_error.upper() or "CTRANSLATE2" in load_error.upper():
                    print("🔄 CUDA/CTranlsate2 error detected, trying CPU fallback...")
                    
                    self.device = "cpu"
                    self.compute_type = "int8"
                    
                    try:
                        model = self._try_load_model_with_device(model_size, model_dir, "cpu", "int8")
                        if model:
                            load_success = True
                            print(f"✅ CPU fallback successful for {model_size}")
                        else:
                            print("❌ CPU fallback also failed")
                    except Exception as e2:
                        print(f"❌ CPU fallback error: {e2}")
            
            if not load_success:
                if model_size != "tiny":
                    print(f"🔄 Ultimate fallback: trying 'tiny' model on CPU...")
                    try:
                        model = self._try_load_model_with_device("tiny", model_dir, "cpu", "int8")
                        if model:
                            load_success = True
                            print(f"✅ Ultimate fallback: 'tiny' model loaded on CPU")
                    except Exception as e:
                        print(f"❌ Ultimate fallback failed: {e}")
            
            if not load_success:
                print(f"❌ All loading attempts failed for {model_size}")
                return False
            
            print(f"🧪 Testing model {model_size}...")
            try:
                import numpy as np
                test_audio = np.random.randn(16000).astype(np.float32) * 0.01
                segments, info = model.transcribe(test_audio, beam_size=1, without_timestamps=True)
                list(segments)
                print(f"✅ Model test successful")
            except Exception as e:
                print(f"⚠️ Model test warning: {e}")
  
            with self._lock:
                self.model = model
                self.model_size = model_size
                self._model_cache[model_size] = model
                
                if len(self._model_cache) > self._model_cache_size:
                    oldest_key = next(iter(self._model_cache))
                    del self._model_cache[oldest_key]
                    print(f"🧹 Removed from cache: {oldest_key}")
            
            print(f"🎉 [PERF] Model '{model_size}' loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ [PERF] Model load failed: {e}")
            
            fallback_chain = {
                'large-v3': ['large-v2', 'medium', 'small', 'base', 'tiny'],
                'large-v2': ['medium', 'small', 'base', 'tiny'],
                'medium': ['small', 'base', 'tiny'],
                'small': ['base', 'tiny'],
                'base': ['tiny']
            }
            
            if model_size in fallback_chain:
                for fallback_model in fallback_chain[model_size]:
                    print(f"🔄 Trying fallback model on CPU: {fallback_model}")
                    try:
                        WhisperModel = FastLazyLoader.load('faster_whisper')
                        model_dir = PlatformUtils.get_platform_config_dir() / "models"
                        fallback_model_instance = WhisperModel(
                            fallback_model,
                            device="cpu",
                            compute_type="int8",
                            download_root=str(model_dir),
                            cpu_threads=2,
                            num_workers=1,
                            local_files_only=False,
                        )
                        
                        with self._lock:
                            self.model = fallback_model_instance
                            self.model_size = fallback_model
                            self.device = "cpu"
                            self.compute_type = "int8"
                        
                        print(f"✅ Using CPU fallback model: {fallback_model}")
                        return True
                        
                    except Exception as fb_error:
                        print(f"⚠️ Fallback {fallback_model} failed: {fb_error}")
                        continue
            
            return False
    
    def _try_load_model_with_device(self, model_size: str, model_dir: Path, 
                                   device: str, compute_type: str):
        """Versuche Modell mit spezifischem Device zu laden"""
        WhisperModel = FastLazyLoader.load('faster_whisper')
        
        # Optimierte Parameter
        if device == "cuda":
            cpu_threads = 4
            num_workers = 2
        elif device == "mps":
            cpu_threads = 2
            num_workers = 1
        else:  # CPU
            cpu_threads = max(1, os.cpu_count() // 2)
            num_workers = 1
        
        return WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(model_dir),
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            local_files_only=False,
        )

    def clear_model_cache(self):
        """Leert den Modell-Cache"""
        with self._lock:
            cache_size = len(self._model_cache)
            self._model_cache.clear()
            print(f"🧹 Model cache cleared ({cache_size} models)")
    
    def verify_model_loaded(self) -> bool:
        """Verify model is actually loaded and functional"""
        if not self.model:
            print("❌ No model reference")
            return False
    
        try:
            has_model = hasattr(self.model, 'transcribe')
            print(f"🔍 Model verification: has transcribe method = {has_model}")
            return has_model
        except Exception as e:
           print(f"❌ Model verification failed: {e}")
           return False

    def _atomic_model_load_process(self, model_size: str) -> bool:
        """OPTIMIERT: Atomarer Load mit Lock, Fallbacks UND Cleanup"""
        print(f"🔍 [CRITICAL] _atomic_model_load_process START for {model_size}")
        import inspect
        load_lock = self._model_load_locks.setdefault(model_size, threading.RLock())
        print(f"🔍 Acquiring load lock for {model_size}...")
        if not load_lock.acquire(timeout=10.0):
            print(f"⏳ Timeout auf Load-Lock für {model_size}")
            return False
    
        try:
            self._force_model_cleanup()
            original_device = self.device
            original_compute = self.compute_type
            load_success = False
        
            try:
                print(f"🔄 Versuch 1: Isolierter Load für {model_size}...")
                if self._direct_model_load_with_isolation(model_size):
                    load_success = True
                    return True

                print(f"🔄 Versuch 2: Separate Instanz für {model_size}...")
                if self._fallback_with_separate_instance(model_size):
                    load_success = True
                    return True

                print(f"🔄 Versuch 3: CPU-Fallback für {model_size}...")
                if original_device != "cpu":
                    self.device = "cpu"
                    self.compute_type = "int8"
                    if self._safe_cpu_fallback(model_size):
                        load_success = True
                        print(f"✅ Versuch 3 erfolgreich!")
                        return True
            
                print(f"❌ Alle Load-Versuche für {model_size} fehlgeschlagen")
                return False
            
            except Exception as e:
                print(f"❌ Unerwarteter Fehler in Load-Prozess: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            finally:
                if not load_success:
                    self.device = original_device
                    self.compute_type = original_compute
                    print(f"↩️ Device-State restored: {original_device}")
    
        finally:
            load_lock.release()
            print(f"🔍 Load lock released for {model_size}")

    def _direct_model_load_with_isolation(self, model_size: str) -> bool:
        """Direkter Load mit vollständiger Isolation"""
        original_env = os.environ.copy()
        try:
            os.environ.update({
                'CT2_DISABLE_MKL': '1',
                'CT2_FORCE_CPU_ONLY_INFERENCE': '1' if self.device == "cpu" else '0',
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1',
            })
        
            print(f"🔍 Loading faster_whisper module...")
            WhisperModel = FastLazyLoader.load('faster_whisper')
            config_dir = PlatformUtils.get_platform_config_dir()
            model_dir = config_dir / "models"
        
            model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(model_dir),
                cpu_threads=1,
                num_workers=1,
                local_files_only=False,
            )
            
            with self._lock:
                self.model = model
                self.model_size = model_size
            
            return True
        
        except Exception as e:
            print(f"❌ Direct model load failed: {e}")
            return False
        
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def _fallback_with_separate_instance(self, model_size: str) -> bool:
        """Fallback mit SEPARATER Engine-Instanz"""
        print(f"🔄 Starte isolierten Fallback für {model_size}...")
    
        fallback_engine = ExcellenceTranscriptionEngine()
        fallback_engine.device = self.device
        fallback_engine.compute_type = self.compute_type
    
        try:
            success = fallback_engine.load_model(model_size)
        
            if success and fallback_engine.model:
                with self._lock:
                    self.model = fallback_engine.model
                    self.model_size = model_size
                    fallback_engine.model = None
            
                return True
        except Exception as e:
            print(f"⚠️ Isolierter Fallback fehlgeschlagen: {e}")
    
        return False

    def _safe_cpu_fallback(self, model_size: str) -> bool:
        """Sicherer CPU-Fallback ohne Rekursion"""
        temp_engine = ExcellenceTranscriptionEngine()
        temp_engine.device = "cpu"
        temp_engine.compute_type = "int8"
    
        success = temp_engine.load_model(model_size)
    
        if success:
            with self._lock:
                self.model = temp_engine.model
                self.model_size = model_size
                self.device = "cpu"
                self.compute_type = "int8"
                temp_engine.model = None
        
            print(f"✅ CPU-Fallback erfolgreich: {model_size}")
            return True
    
        return False

    def _force_model_cleanup(self):
        """FORCIERTER Modell-Cleanup - SICHER & OHNE DEADLOCKS"""
        import gc
    
        cleanup_start = time.time()
        print("🧹 STARTE MODELL-CLEANUP (sicher)...")
    
        model_refs_cleared = 0
        with self._lock:
            if hasattr(self, 'model') and self.model is not None:
                try:
                    model_size_info = f"Model type: {type(self.model).__name__}"
            
                    if hasattr(self.model, 'unload_model'):
                        try:
                            self.model.unload_model()
                            print("  ↪ Model.unload_model() erfolgreich")
                        except Exception as e:
                            print(f"  ⚠️ unload_model failed: {e}")
            
                    if hasattr(self.model, '_model'):
                        try:
                            del self.model._model
                            print("  ↪ _model reference gelöscht")
                        except:
                            pass
            
                    self.model = None
                    model_refs_cleared += 1
                    print(f"  ↪ Haupt-Model-Reference gelöscht")
            
                except Exception as e:
                    print(f"  ⚠️ Model cleanup error: {e}")

            model_refs_to_clear = []
            for attr_name in dir(self):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(self, attr_name)
                    if attr is not None and hasattr(attr, '__class__'):
                        class_name = attr.__class__.__name__
                        if any(keyword in class_name.lower() for keyword in 
                              ['whisper', 'model', 'transcription', 'ctranslate']):
                            model_refs_to_clear.append(attr_name)
                except:
                    pass
    
            for attr_name in model_refs_to_clear:
                try:
                    setattr(self, attr_name, None)
                    model_refs_cleared += 1
                    print(f"  ↪ Zusätzliche Reference gelöscht: {attr_name}")
                except:
                    pass

            self.model_size = None
            if hasattr(self, '_model_loaded_flag'):
                self._model_loaded_flag = False
            if hasattr(self, '_active_model_loads'):
                self._active_model_loads.clear()
            if hasattr(self, '_active_fallback_loads'):
                self._active_fallback_loads.clear()
    
            print(f"  ↪ {model_refs_cleared} Model-References gelöscht")

        print("  ↪ Starte Garbage Collection...")
        for i in range(2):
            collected = gc.collect(generation=2)
            if i == 0 and collected > 0:
                print(f"    ↪ GC Pass {i+1}: {collected} Objekte gesammelt")
            time.sleep(0.01)

        if hasattr(self, 'device') and self.device == "cuda" and TORCH_AVAILABLE:
            try:
                torch = FastLazyLoader.load('torch')
                if torch.cuda.is_available():
                    print("  ↪ Starte SICHERE GPU Memory Cleanup...")
                
                    torch.cuda.empty_cache()
                    print("    ↪ CUDA Cache geleert")
                
                    try:
                        torch.cuda.synchronize()
                        print("    ↪ CUDA synchronisiert")
                    except Exception:
                        print("    ↪ CUDA sync übersprungen (stability)")

                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"    ↪ Verbleibend: {allocated:.2f}GB allokiert, {reserved:.2f}GB reserviert")
            
            except Exception as e:
                print(f"  ⚠️ GPU cleanup error (non-critical): {e}")

        print("  ↪ Finale Garbage Collection...")
        gc.collect()
        time.sleep(0.005)

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024**2
            print(f"  ↪ Prozess-Speicher: {memory_mb:.1f} MB")
        except:
            pass

        cleanup_time = time.time() - cleanup_start
        print(f"✅ MODELL-CLEANUP ABGESCHLOSSEN ({cleanup_time:.2f}s)")

    def diagnose_memory_leaks(self):
        """Diagnostiziert Memory Leaks speziell für CUDA"""
        if not TORCH_AVAILABLE:
            return "Torch nicht verfügbar"
    
        try:
            torch = FastLazyLoader.load('torch')
            if not torch.cuda.is_available():
                return "CUDA nicht verfügbar"
        
            info = {
                'cuda_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'cuda_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'cuda_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
            }
        
            potential_leaks = []
            if info['cuda_allocated_gb'] > 0.5:
                potential_leaks.append(f"Hoch: {info['cuda_allocated_gb']:.2f}GB allokiert")
        
            return info, potential_leaks
        
        except Exception as e:
            return f"Diagnose fehlgeschlagen: {e}"

    def _safe_model_test(self):
        """SICHERER Model-Test OHNE Memory Leak"""
        if not self.model:
            return False
            
        try:
            np = FastLazyLoader.load('numpy')
            
            test_audio = np.zeros(1600, dtype=np.float32)
            
            segments, info = self.model.transcribe(
                test_audio,
                beam_size=1,
                best_of=1,
                vad_filter=False,
                without_timestamps=True,
                temperature=0.0
            )
            
            try:
                first_segment = next(segments, None)
                
                if first_segment:
                    has_text = bool(first_segment.text and first_segment.text.strip())
                    print(f"🔍 Model test: {'Hat Text' if has_text else 'Kein Text'}")
                    return has_text
                else:
                    print("⚠️ Model test: keine Segmente generiert")
                    return False
                    
            except StopIteration:
                print("⚠️ Model test: Generator erschöpft")
                return False
            except Exception as e:
                print(f"⚠️ Model test error: {e}")
                return False
                
        except Exception as e:
            print(f"⚠️ Model-Test-Warnung: {e}")
            return False
   
    def _safe_gpu_cleanup(self, emergency=False):
        """
        sichere GPU-Bereinigung
        Nur für kritische Systemfehler
        """
        if not TORCH_AVAILABLE:
            return
    
        try:
            torch = FastLazyLoader.load('torch')
            if not torch.cuda.is_available():
                return
        
            print("  ↪ GPU Cleanup...")
        
            torch.cuda.empty_cache()
        
            if not emergency:
                try:
                    torch.cuda.synchronize()
                    print("    ↪ CUDA synchronisiert")
                except Exception:
                    print("    ↪ CUDA sync übersprungen")
        
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
            
                if allocated > 0.5:
                    print(f"    ⚠️ GPU memory: {allocated:.2f}GB allocated")
                else:
                    print(f"    ↪ GPU memory: {allocated:.2f}GB allocated")
            except:
                pass
        
        except Exception as e:
            print(f"  ⚠️ GPU cleanup failed (non-critical): {type(e).__name__}")
                     
    def _handle_model_fallback(self, original_model_size: str, 
                              old_device: str, old_compute_type: str) -> bool:
        """Handle model fallback when test fails - REPARIERT"""
        print(f"🔄 Model test failed, trying alternative models...")
        
        fallback_models = []
        
        if "large" in original_model_size:
            fallback_models = ["medium", "small", "base", "tiny"]
        elif "medium" in original_model_size:
            fallback_models = ["small", "base", "tiny"]
        elif "small" in original_model_size:
            fallback_models = ["base", "tiny"]
        else:
            fallback_models = ["tiny"] if original_model_size != "tiny" else []
        
        print(f"  📋 Fallback-Reihenfolge: {fallback_models}")
        
        for fallback_model in fallback_models:
            if fallback_model == original_model_size:
                continue
            
            print(f"  🔄 Versuche {fallback_model}...")
            
            success = self._try_load_fallback_model(fallback_model)
            if success:
                print(f"  ✅ {fallback_model} erfolgreich geladen als Fallback")
                return True
        
        print(f"⚠️ Alle Fallbacks fehlgeschlagen, stelle Original-Einstellungen wieder her")
        with self._lock:
            self.device = old_device
            self.compute_type = old_compute_type
        
        if old_device != "cpu":
            print(f"  🔄 Versuche {original_model_size} auf CPU...")
            self.device = "cpu"
            self.compute_type = "int8"
            return self._try_load_fallback_model(original_model_size)
        
        return False

    def _try_load_fallback_model(self, model_name: str) -> bool:
        """Versuch ein Fallback-Modell sicher zu laden - OHNE REKURSION"""
        load_key = f"fallback_{model_name}"
    
        if hasattr(self, '_active_fallback_loads'):
            if load_key in self._active_fallback_loads:
                print(f"  ⛔ REKURSION vermieden: {model_name} bereits in Fallback-Ladung")
                return False
        else:
            self._active_fallback_loads = set()
    
        self._active_fallback_loads.add(load_key)
    
        try:
            print(f"  🚀 Starte Fallback-Ladung für {model_name}...")
        
            config_dir = PlatformUtils.get_platform_config_dir()
            model_dir = config_dir / "models"
            model_dir.mkdir(exist_ok=True)
        
            WhisperModel = FastLazyLoader.load('faster_whisper')
        
            temp_model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(model_dir),
                cpu_threads=1,
                num_workers=1,
                local_files_only=True
            )
        
            test_success = self._quick_model_test(temp_model)
        
            if test_success:
                self._safe_model_cleanup()
            
                with self._lock:
                    self.model = temp_model
                    self.model_size = model_name
            
                print(f"  ✅ {model_name} erfolgreich als Fallback geladen")
                return True
            else:
                try:
                    if hasattr(temp_model, 'unload_model'):
                        temp_model.unload_model()
                except:
                    pass
                del temp_model
                return False
            
        except Exception as e:
            print(f"  ❌ Fallback für {model_name} fehlgeschlagen: {e}")
            return False
        finally:
            if load_key in self._active_fallback_loads:
                self._active_fallback_loads.remove(load_key)

    def _handle_cuda_fallback(self, original_model_size: str) -> bool:
        """Behandle CUDA Fallback - MIT REKURSIONSSCHUTZ"""
        print("🔄 CUDA/GPU fehlgeschlagen, wechsle zu CPU...") 
    
        with self._lock:
            self.device = "cpu"
            self.compute_type = "int8"
    
        self._force_model_cleanup()
    
        print(f"🔄 Neuer Versuch: {original_model_size} auf CPU...")
    
        return self._direct_load_with_retry(original_model_size, max_retries=1)

    def _direct_load_with_retry(self, model_name: str, max_retries: int = 2) -> bool:
        """Direkter Load ohne Rekursion"""
        for attempt in range(max_retries):
            try:
                print(f"  🔄 Direkter Load Versuch {attempt+1}/{max_retries}...")
            
                WhisperModel = FastLazyLoader.load('faster_whisper')
                config_dir = PlatformUtils.get_platform_config_dir()
                model_dir = config_dir / "models"
            
                temp_model = WhisperModel(
                    model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(model_dir),
                    cpu_threads=1,
                    num_workers=1,
                    local_files_only=True
                )
            
                import numpy as np
                test_audio = np.zeros(1600, dtype=np.float32)
                segments, _ = temp_model.transcribe(test_audio, beam_size=1)
                list(segments)
            
                with self._lock:
                    self.model = temp_model
                    self.model_size = model_name
            
                return True
            
            except Exception as e:
                print(f"  ⚠️ Direkter Load fehlgeschlagen: {e}")
                time.sleep(0.5 * (attempt + 1))
    
        return False

    def _handle_general_fallback(self, original_model_size: str) -> bool:
        """Behandle allgemeine Fallbacks - REPARIERT"""
        print("🔄 Allgemeiner Fehler, versuche einfachere Modelle...")
        
        fallback_models = ["tiny", "base", "small"]
        
        for fallback_model in fallback_models:
            if fallback_model != original_model_size:
                print(f"  🔄 Versuche {fallback_model}...")
                
                success = self._try_load_fallback_model(fallback_model)
                if success:
                    return True
        
        print("❌ Alle Fallbacks fehlgeschlagen")
        return False

    def validate_audio_data(self, audio_data: bytes) -> Tuple[bool, str]:
        """Validate audio data before transcription"""
        if not isinstance(audio_data, bytes):
            return False, "Audio data must be bytes"

        if len(audio_data) == 0:
            return False, "Audio data is empty"

        if len(audio_data) < 1600:
            return False, f"Audio data too short: {len(audio_data)} bytes"

        try:
            np = FastLazyLoader.load('numpy')
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if np.all(audio_np == 0):
                return False, "Audio data is completely silent"

            if np.var(audio_np) < 100:
                return False, "Audio variance too low (likely silent)"
        except:
            pass

        return True, "Valid"

    def safe_transcribe(self, audio_data: bytes, max_retries: int = 2) -> Optional[ExcellenceTranscriptionResult]:
        """Safe transcription with retries and validation"""
        is_valid, validation_msg = self.validate_audio_data(audio_data)
        if not is_valid:
            print(f"⚠️ Audio validation failed: {validation_msg}")
            return None

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                try:
                    processed_audio = self.enhance_audio_for_transcription(audio_data)
                except Exception:
                    processed_audio = audio_data

                result = self.transcribe_audio(processed_audio)

                if result and result.text and result.text.strip():
                    return result

            except Exception as e:
                last_exception = e
                print(f"⚠️ Transcription attempt {attempt+1} failed: {e}")

                if attempt < max_retries:
                    wait_time = 0.5 * (attempt + 1)
                    time.sleep(wait_time)

        print(f"❌ All transcription attempts failed")
        if last_exception:
            print(f"   Last error: {last_exception}")
            
        return None

    def enhance_audio_for_transcription(self, audio_data: bytes) -> bytes:
        """Enhance audio quality for better transcription"""
        if not audio_data or len(audio_data) == 0:
            return audio_data

        if not NUMPY_AVAILABLE:
            return audio_data

        if len(audio_data) < 1600:
            return audio_data

        try:
            np = FastLazyLoader.load('numpy')
            try:
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                return audio_data

            if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                return audio_data

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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

            if len(audio_np) > 100 and SCIPY_AVAILABLE:
                try:
                    scipy_signal = FastLazyLoader.load('scipy.signal')
                    b, a = scipy_signal.butter(2, 80/(self.config.SAMPLE_RATE/2), btype='high')
                    audio_np = scipy_signal.filtfilt(b, a, audio_np)
                except Exception:
                    audio_np = audio_np - np.mean(audio_np)

            audio_np = np.clip(audio_np, -0.99, 0.99)

            enhanced_audio = (audio_np * 32767).astype(np.int16).tobytes()

            if len(enhanced_audio) != len(audio_data):
                return audio_data

            return enhanced_audio

        except Exception:
            return audio_data

    def _validate_transcription_segment(self, text: str, confidence: float, segment) -> bool:
        """Validate if transcription segment is valid"""
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

    def _contains_gibberish(self, text: str) -> bool:
        """Check if text contains gibberish patterns"""
        if re.search(r'(\w)\1{3,}', text.lower()):
            return True

        if re.search(r'[aeiou]{5,}', text.lower()):
            return True

        words = text.split()
        if len(words) > 3:
            long_words = sum(1 for word in words if len(word) > 15)
            if long_words / len(words) > 0.3:
                return True

        return False

    def _calculate_enhanced_confidence(self, segment, text: str) -> float:
        """Calculate enhanced confidence score"""
        base_confidence = max(getattr(segment, 'confidence', 0.0), 0.05)

        word_count = len(text.split())
        text_length = len(text.strip())
        has_punctuation = any(c in text for c in '.!?,;:')
        has_letters = any(c.isalpha() for c in text)
        unique_words = len(set(text.split()))

        length_boost = min(0.2, text_length / 300.0)
        word_boost = min(0.15, word_count * 0.03)
        punctuation_boost = 0.08 if has_punctuation else 0.0
        letters_boost = 0.1 if has_letters else 0.0
        diversity_boost = min(0.1, unique_words * 0.02)

        calculated_confidence = (
            base_confidence +
            length_boost +
            word_boost +
            punctuation_boost +
            letters_boost +
            diversity_boost
        )

        return min(0.95, calculated_confidence)

    @excellence_execution(timeout=30.0)
    def transcribe_audio(self, audio_data: bytes, include_timestamps: bool = False) -> Any:
        """Transcribe audio with optional timestamps"""
        if not self.model or not audio_data:
            return None if not include_timestamps else []

        try:
            processed_audio = self.enhance_audio_for_transcription(audio_data)
            np = FastLazyLoader.load('numpy')
            audio_np = np.frombuffer(processed_audio, dtype=np.int16).astype(np.float32) / 32768.0

            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                best_of=5,
                patience=1.0,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=800,
                    speech_pad_ms=500,
                    threshold=0.3
                ),
                no_speech_threshold=0.5,
                log_prob_threshold=-1.2,
                compression_ratio_threshold=2.8,
                condition_on_previous_text=True,
                without_timestamps=not include_timestamps,
                word_timestamps=include_timestamps,
                suppress_tokens=[-1]
            )

            segments_list = list(segments)

            valid_segments = []
            total_confidence = 0.0

            for segment in segments_list:
                text = segment.text.strip()
                confidence = getattr(segment, 'confidence', 0.0)

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
                        confidence=getattr(seg, 'confidence', 0.1),
                        language=getattr(info, 'language', 'unknown'),
                        start=getattr(seg, 'start', 0.0),
                        end=getattr(seg, 'end', 0.0)
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
                        language=getattr(info, 'language', 'unknown')
                    )
                else:
                    return self.emergency_fallback_transcription(audio_data)

        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
            return None if not include_timestamps else []

    def emergency_fallback_transcription(self, audio_data: bytes) -> Optional[ExcellenceTranscriptionResult]:
        """Emergency fallback transcription for problematic audio"""
        try:
            if not self.model or not audio_data:
                return None

            np = FastLazyLoader.load('numpy')
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
                without_timestamps=True
            )

            segments_list = list(segments)

            best_segment = None
            best_confidence = 0.0
            best_text = ""

            for i, segment in enumerate(segments_list):
                text = segment.text.strip()
                if text and len(text) > 1:
                    confidence = self._calculate_enhanced_confidence(segment, text)

                    is_valid = (
                        len(text) >= 2 and
                        not text.isspace() and
                        any(c.isalnum() for c in text) and
                        confidence >= 0.1
                    )

                    if is_valid and confidence > best_confidence:
                        best_confidence = confidence
                        best_segment = segment
                        best_text = text

            if best_segment and best_text:
                return ExcellenceTranscriptionResult(
                    text=best_text,
                    confidence=best_confidence,
                    language=getattr(info, 'language', 'unknown')
                )

            return None

        except Exception as e:
            print(f"⚠️ Emergency transcription failed: {e}")
            return None

    def clear_cache(self):
        """Clear transcription cache and free memory"""
        with self._lock:
            if hasattr(self, '_cache'):
                self._cache.clear()
            self._last_transcription_text = ""
            
            import gc
            gc.collect()
            
            if self.device == "cuda" and TORCH_AVAILABLE:
                try:
                    torch = FastLazyLoader.load('torch')
                    torch.cuda.empty_cache()
                except:
                    pass

    def reload_model(self, model_size: str) -> bool:
        """Reload model with given size"""
        return self.load_model(model_size)

    def get_current_model(self) -> str:
        """Get currently loaded model name"""
        return self.model_size if self.model_size else "None"

    def is_model_loading(self) -> bool:
        """Check if model is currently loading"""
        return self._model_loading

    def test_model_functionality(self):
        """Test if loaded model works correctly"""
        if not self.model:
            print("❌ Kein Model geladen zum Testen")
            return False

        try:
            print("🔍 Teste Model-Funktionalität...")

            np = FastLazyLoader.load('numpy')
            test_audio = np.random.randn(16000).astype(np.float32) * 0.1

            segments, info = self.model.transcribe(
                test_audio,
                beam_size=1,
                best_of=1,
                vad_filter=False,
                without_timestamps=True
            )

            segments_list = list(segments)
            print(f"   Segmente gefunden: {len(segments_list)}")

            if segments_list:
                for i, seg in enumerate(segments_list[:2]):
                    print(f"   Segment {i}: '{seg.text[:50]}...' (conf: {seg.confidence:.2f})")

            if hasattr(info, 'language'):
                print(f"   Sprache erkannt: {info.language}")

            print("✅ Model-Test erfolgreich")
            return True

        except Exception as e:
            print(f"❌ Model-Test fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return False

    def dispose(self):
        """Vollständige Freigabe mit Thread-Safety"""
        print("🧹 Transcription Engine Dispose...")
    
        self._disposing = True
    
        with self._transcribe_lock:
            self._active_model_loads.clear()
            self._active_fallback_loads.clear()
        
            self._safe_model_cleanup_with_gpu_fallback()
        
            if hasattr(self, '_cache'):
                self._cache.clear()
    
        print("✅ Transcription Engine disposed")

    def _safe_model_cleanup_with_gpu_fallback(self):
        """GPU-Cleanup auch ohne torch verfügbar"""
        import gc
    
        if hasattr(self, 'model') and self.model is not None:
            try:
                if hasattr(self.model, 'unload_model'):
                    self.model.unload_model()
                elif hasattr(self.model, '_model'):
                    del self.model._model
            except Exception as e:
                print(f"⚠️ Model unload error: {e}")
            finally:
                self.model = None
    
        if hasattr(self, 'device') and self.device == "cuda":
            if TORCH_AVAILABLE:
                try:
                    torch = FastLazyLoader.load('torch')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print("✅ CUDA cache cleared via torch")
                except Exception:
                    pass
        
            try:
                import ctypes
                if IS_WINDOWS:
                    cuda = ctypes.windll.LoadLibrary("nvcuda.dll")
                else:
                    cuda = ctypes.cdll.LoadLibrary("libcuda.so")
            
                print("⚠️ Direct CUDA reset attempted")
            except:
                pass
    
        for _ in range(3):
            gc.collect()

class StreamManager:
    """
    🎯 STREAM MANAGER - Optimiert für YouTube mit funktionierenden URLs
    """

    def __init__(self, enable_debug: bool = False):
        self._platform_cache = OrderedDict()
        self._audio_url_cache = OrderedDict()
        self._live_status_cache = OrderedDict()
        self._stream_info_cache = OrderedDict()
        self._debug = enable_debug
        
        self._last_error = None
        self._last_method = None
        self._stats = {
            'extraction_attempts': 0,
            'successful_extractions': 0,
            'cache_hits': 0,
            'errors': 0,
            'start_time': time.time()
        }

        self._format_priorities = {
            'youtube': ['bestaudio[ext=m4a]/bestaudio/best', 'bestaudio/best', 'ba'],
            'youtube_live': ['bestaudio/best', 'ba'],
            'twitch': ['bestaudio/best', 'audio_only'],
            'tiktok': ['bestaudio/best'],
            'facebook': ['bestaudio/best'],
            'hls': ['bestaudio/best'],
            'dash': ['bestaudio/best'],
            'generic': ['bestaudio/best', 'ba']
        }
        
        self._user_agents = {
            'desktop': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'mobile': 'Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        }

    def detect_platform(self, url: str) -> Tuple[str, str]:
        """
        Platform-Erkennung mit Cache
        """
        if not url:
            return ('unknown', 'Invalid URL')
            
        if url in self._platform_cache:
            self._stats['cache_hits'] += 1
            return self._platform_cache[url]

        url_lower = url.lower().strip()

        if url_lower.startswith('file://'):
            result = ('local', 'Local File')
        elif any(url_lower.endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.opus', '.webm']):
            result = ('direct_audio', 'Direct Audio')
        elif any(url_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.wmv', '.flv']):
            result = ('direct_video', 'Direct Video')

        elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            is_live = self._check_youtube_live_status(url)
            if is_live:
                result = ('youtube_live', 'YouTube Live')
            else:
                result = ('youtube', 'YouTube Video')

        elif 'twitch.tv' in url_lower:
            result = ('twitch', 'Twitch')
        elif 'tiktok.com' in url_lower:
            result = ('tiktok', 'TikTok')
        elif 'facebook.com' in url_lower or 'fb.watch' in url_lower:
            result = ('facebook', 'Facebook')
        elif '.m3u8' in url_lower:
            result = ('hls', 'HLS Stream')
        elif '.mpd' in url_lower:
            result = ('dash', 'DASH Stream')
        elif url_lower.startswith(('http://', 'https://')):
            result = ('generic', 'Website/Stream')
        else:
            result = ('unknown', 'Unknown Source')

        if len(self._platform_cache) < 50:
            self._platform_cache[url] = result

        return result

    def _check_youtube_live_status(self, url: str) -> bool:
        """
        Prüft ob YouTube URL ein Live-Stream ist
        """
        cache_key = f"live_{hashlib.md5(url.encode()).hexdigest()[:16]}"
        current_time = time.time()

        if cache_key in self._live_status_cache:
            cached = self._live_status_cache[cache_key]
            if current_time - cached['timestamp'] < 300:
                return cached['is_live']

        is_live = False
        url_lower = url.lower()
        
        live_patterns = ['/live', 'live=1', '/stream', 'livestream']
        if any(pattern in url_lower for pattern in live_patterns):
            is_live = True

        self._live_status_cache[cache_key] = {
            'is_live': is_live,
            'timestamp': current_time
        }

        if len(self._live_status_cache) > 30:
            oldest_key = min(self._live_status_cache.items(),
                           key=lambda x: x[1]['timestamp'])[0]
            del self._live_status_cache[oldest_key]

        return is_live

    def extract_audio_url(self, url: str, force_refresh: bool = False) -> Optional[str]:
        """
        Audio-URL Extraction - optimiert für YouTube
        """
        self._stats['extraction_attempts'] += 1
        
        if self._debug:
            print(f"\n🎵 [DEBUG] EXTRACT_AUDIO_URL START für: {url[:80]}...")
        
        self._last_error = None
        self._last_method = None
        
        if not url or not isinstance(url, str):
            if self._debug:
                print(f"❌ [DEBUG] Invalid input")
            self._last_error = "Invalid input"
            self._stats['errors'] += 1
            return None
            
        cleaned_url = url.strip()
        if not cleaned_url:
            if self._debug:
                print(f"❌ [DEBUG] Empty URL")
            self._last_error = "Empty URL"
            self._stats['errors'] += 1
            return None
        

        cache_key = f"audio_{hashlib.md5(cleaned_url.encode()).hexdigest()[:16]}"
        current_time = time.time()


        if not force_refresh and cache_key in self._audio_url_cache:
            cached = self._audio_url_cache[cache_key]
            cache_age = current_time - cached['timestamp']
            
            if self._debug:
                print(f"📦 [DEBUG] Cache found, age: {cache_age:.1f}s, failed: {cached.get('failed', False)}")

            if cache_age < 1800 and not cached.get('failed', False):
                self._stats['cache_hits'] += 1
                if self._debug:
                    print(f"✅ [DEBUG] Cache hit, returning cached URL")
                return cached['url']
            elif cache_age < 300 and cached.get('failed', False):
                if self._debug:
                    print(f"⚠️ [DEBUG] Blocked by failed cache entry")
                return None

        platform_id, platform_name = self.detect_platform(cleaned_url)
        if self._debug:
            print(f"🔍 [DEBUG] Platform detected: {platform_id} ({platform_name})")
        
        result = None
        extraction_method = "unknown"

        try:
            if cleaned_url.startswith('file://'):
                if self._debug:
                    print(f"📁 [DEBUG] Local file detected")
                file_path = cleaned_url[7:]
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    result = cleaned_url
                    extraction_method = "local_file"
                else:
                    if self._debug:
                        print(f"❌ [DEBUG] File not found or not accessible")

            if not result:
                url_lower = cleaned_url.lower()
                AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.opus', '.webm')
                VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.wmv', '.flv')

                if url_lower.endswith(AUDIO_EXTENSIONS) or url_lower.endswith(VIDEO_EXTENSIONS):
                    if self._debug:
                        print(f"🎵 [DEBUG] Direct audio/video link detected")
                    result = cleaned_url
                    extraction_method = "direct_link"


            if not result and platform_id in ['youtube', 'youtube_live']:
                if self._debug:
                    print(f"🎯 [DEBUG] YouTube detected, using optimized extraction...")
                result = self._extract_youtube_audio_optimized(cleaned_url, platform_id)
                extraction_method = "youtube_optimized"
            

            elif not result:
                if self._debug:
                    print(f"🌐 [DEBUG] Non-YouTube platform, using generic extraction...")
                if platform_id in self._format_priorities:
                    format_list = self._format_priorities[platform_id]
                else:
                    format_list = self._format_priorities['generic']
                
                extraction_method = "ytdlp_generic"
                
                for i, format_str in enumerate(format_list[:2]):
                    try:
                        if self._debug:
                            print(f"  🔄 Trying format {i+1}: {format_str}")
                        
                        cmd = [
                            'yt-dlp',
                            '-g',
                            '-f', format_str,
                            '--no-warnings',
                            '--no-check-certificate',
                            '--socket-timeout', '15',
                            cleaned_url
                        ]

                        process_result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=15,
                            shell=False,
                            encoding='utf-8',
                            errors='ignore'
                        )

                        if self._debug:
                            print(f"  📊 yt-dlp result: returncode={process_result.returncode}, "
                                  f"stdout={len(process_result.stdout)} chars")

                        if process_result.returncode == 0 and process_result.stdout.strip():
                            if self._debug:
                                print(f"  ✅ yt-dlp succeeded")
                            lines = process_result.stdout.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and line.startswith(('http://', 'https://')):
                                    result = line
                                    if self._debug:
                                        print(f"  📍 Found URL: {line[:80]}...")
                                    break

                            if result:
                                break

                    except subprocess.TimeoutExpired:
                        if self._debug:
                            print(f"  ⏰ Timeout with format {format_str}")
                        continue
                    except Exception as e:
                        if self._debug:
                            print(f"  ⚠️ Error with format {format_str}: {str(e)[:50]}")
                        continue


                if not result:
                    if self._debug:
                        print(f"  🔄 Trying JSON fallback...")
                    try:
                        json_result = self._json_extraction_fallback(cleaned_url)
                        if json_result:
                            result = json_result
                            extraction_method = "json_fallback"
                            if self._debug:
                                print(f"  ✅ JSON fallback succeeded")
                    except Exception as e:
                        if self._debug:
                            print(f"  ⚠️ JSON fallback error: {str(e)[:50]}")

        except Exception as e:
            if self._debug:
                print(f"❌ [DEBUG] EXCEPTION in extract_audio_url: {e}")
                import traceback
                traceback.print_exc()
            self._last_error = f"Exception: {str(e)[:100]}"
            self._stats['errors'] += 1

        cache_entry = {
            'url': result,
            'timestamp': current_time,
            'failed': result is None,
            'method': extraction_method,
            'platform': platform_id
        }
        
        self._audio_url_cache[cache_key] = cache_entry
        

        if len(self._audio_url_cache) > 50:
            self._audio_url_cache.popitem(last=False)


        self._last_method = extraction_method
        
        if result:
            self._stats['successful_extractions'] += 1
        
        if self._debug:
            print(f"🎵 [DEBUG] EXTRACT_AUDIO_URL END - Result: {'✅ ' + result[:80] + '...' if result else '❌ None'}")
            print(f"       Method: {extraction_method}")
            print(f"       Cache size: {len(self._audio_url_cache)}")
        
        return result

    def _extract_youtube_audio_optimized(self, url: str, platform_id: str) -> Optional[str]:
        """
        Optimierte YouTube Audio-Extraktion
        """
        if self._debug:
            print(f"  🔍 [DEBUG] OPTIMIZED YouTube extraction for: {url[:60]}...")
        
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        
        if not video_id or len(video_id) != 11:
            if self._debug:
                print(f"  ❌ Invalid video ID")
            return None
        
        if self._debug:
            print(f"  🔍 Video ID: {video_id}")
        

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
                if self._debug:
                    print(f"    🧪 Testing: {method['name']}")
                
                result = subprocess.run(
                    method['cmd'],
                    capture_output=True,
                    text=True,
                    timeout=method['timeout'],
                    shell=False,
                    encoding='utf-8',
                    errors='ignore'
                )
                
                if self._debug:
                    print(f"    📊 Result: returncode={result.returncode}, "
                          f"stdout={len(result.stdout)} chars")
                
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith(('http://', 'https://')):
                            if self._debug:
                                print(f"    ✅ Success with {method['name']}")
                                print(f"      URL: {line[:80]}...")
                            return line
                else:
                    if result.stderr:
                        error = result.stderr[:100]
                        if self._debug:
                            print(f"    📝 Stderr: {error}")
                        if "Requested format is not available" in error:
                            continue
                            
            except subprocess.TimeoutExpired:
                if self._debug:
                    print(f"    ⏰ Timeout: {method['name']}")
                continue
            except Exception as e:
                if self._debug:
                    print(f"    ⚠️ Error: {str(e)[:50]}")
                continue
        
        try:
            if self._debug:
                print(f"    🔄 Trying JSON fallback...")
            
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
                        if self._debug:
                            print(f"    ✅ JSON fallback success")
                            print(f"      URL: {best_audio[:80]}...")
                        return best_audio
                        
                except json.JSONDecodeError as e:
                    if self._debug:
                        print(f"    ⚠️ JSON decode error: {e}")
                    
        except Exception as e:
            if self._debug:
                print(f"    ⚠️ JSON fallback error: {str(e)[:50]}")
        
        if self._debug:
            print(f"    🔄 Generating direct audio URL...")
        
        direct_url = f"https://manifest.googlevideo.com/api/manifest/dash/id/{video_id}/source/youtube"
        
        if self._debug:
            print(f"    🔧 Generated direct URL")
            print(f"      URL: {direct_url[:80]}...")
            print(f"    💡 Note: FFmpeg needs proper headers for this URL")
        
        return direct_url

    def _json_extraction_fallback(self, url: str) -> Optional[str]:
        """
        JSON-based extraction fallback für generische URLs
        """
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
            if self._debug:
                print(f"  ⚠️ JSON extraction error: {str(e)[:50]}")

        return None

    def extract_stream_info(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Extrahiert Stream-Informationen
        """
        if self._debug:
            print(f"\n🎯 [DEBUG] EXTRACT_STREAM_INFO für: {url[:60]}...")
        
        try:
            cache_key = f"info_{hashlib.md5(url.encode()).hexdigest()[:16]}"
            current_time = time.time()

            if not force_refresh and cache_key in self._stream_info_cache:
                cached = self._stream_info_cache[cache_key]
                if current_time - cached['timestamp'] < 600:
                    if self._debug:
                        print(f"📦 [DEBUG] Stream info cache hit")
                    return cached['info']

            platform_id, platform_name = self.detect_platform(url)
            if self._debug:
                print(f"🔍 [DEBUG] Platform: {platform_id}")


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

            self._stream_info_cache[cache_key] = {
                'info': info,
                'timestamp': current_time
            }

            if len(self._stream_info_cache) > 30:
                oldest_key = min(self._stream_info_cache.items(),
                               key=lambda x: x[1]['timestamp'])[0]
                del self._stream_info_cache[oldest_key]

            if self._debug:
                print(f"✅ [DEBUG] Stream info extracted")
                print(f"    Title: {info['title']}")

            return info

        except Exception as e:
            if self._debug:
                print(f"❌ [DEBUG] EXCEPTION in extract_stream_info: {e}")
            
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
        """
        Gibt optimierte FFmpeg Parameter für die URL zurück
        MIT SPEZIELLEN HEADERS FÜR YOUTUBE HLS
        """

        is_youtube_hls = ('manifest.googlevideo.com' in url and 
                         ('/hls_playlist/' in url or '.m3u8' in url))
    

        is_youtube_dash = ('manifest.googlevideo.com' in url and 
                          '/dash/' in url)
    
        if is_youtube_hls or is_youtube_dash:

            if self._debug:
                print(f"🎯 YOUTUBE HLS/DASH Manifest detected - using specialized parameters")
        
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
        """
        Validiert URL für die Verarbeitung
        """
        if self._debug:
            print(f"\n🔍 [DEBUG] VALIDATE_URL für: {url[:80]}...")
        
        if not url or not isinstance(url, str):
            if self._debug:
                print(f"❌ [DEBUG] Invalid input")
            return False, "Invalid input"

        cleaned_url = url.strip()
        if not cleaned_url:
            if self._debug:
                print(f"❌ [DEBUG] Empty URL")
            return False, "Empty URL"

        if cleaned_url.startswith('file://'):
            if self._debug:
                print(f"📁 [DEBUG] File URL detected")
            file_path = cleaned_url[7:]
            if not os.path.exists(file_path):
                if self._debug:
                    print(f"❌ [DEBUG] File not found: {file_path}")
                return False, "File not found"
            if not os.path.isfile(file_path):
                if self._debug:
                    print(f"❌ [DEBUG] Not a valid file: {file_path}")
                return False, "Not a valid file"

            try:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    if self._debug:
                        print(f"❌ [DEBUG] File is empty")
                    return False, "File is empty"

                filename = os.path.basename(file_path)
                if self._debug:
                    print(f"✅ [DEBUG] File validated: {filename} ({file_size} bytes)")
                return True, f"File: {filename}"

            except OSError as e:
                if self._debug:
                    print(f"❌ [DEBUG] File access error: {e}")
                return False, f"File access error"

        if not cleaned_url.startswith(('http://', 'https://')):
            if self._debug:
                print(f"❌ [DEBUG] Invalid URL format (not http/https)")
            return False, "Invalid URL format"

        if len(cleaned_url) > 2000:
            if self._debug:
                print(f"❌ [DEBUG] URL too long: {len(cleaned_url)} chars")
            return False, "URL too long"

        if self._debug:
            print(f"🎵 [DEBUG] Attempting audio URL extraction for validation...")
        audio_url = self.extract_audio_url(cleaned_url)
        
        if self._debug:
            print(f"🎵 [DEBUG] Extraction result: {'✅ Success' if audio_url else '❌ Failed'}")

        if not audio_url:

            platform_id, platform_name = self.detect_platform(cleaned_url)
            error_msg = f"No audio URL extractable ({platform_name})"
            
            if self._last_error:
                error_msg += f" - {self._last_error}"
            if self._last_method:
                error_msg += f" [method: {self._last_method}]"
                
            if self._debug:
                print(f"❌ [DEBUG] Validation failed: {error_msg}")
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
                    except:
                        pass
            except:
                pass

        status = " | ".join(status_parts)
        
        if self._debug:
            print(f"✅ [DEBUG] Validation successful: {status}")
            print(f"    Audio URL: {audio_url[:80]}...")
        
        return True, status

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Gibt Diagnose-Informationen zurück
        """
        current_time = time.time()
        uptime = current_time - self._stats['start_time']
        
        stats = self._stats.copy()
        stats.update({
            'uptime_seconds': uptime,
            'uptime_human': str(datetime.timedelta(seconds=int(uptime))),
            'success_rate': (
                stats['successful_extractions'] / stats['extraction_attempts'] * 100
                if stats['extraction_attempts'] > 0 else 0
            ),
            'cache_hit_rate': (
                stats['cache_hits'] / stats['extraction_attempts'] * 100
                if stats['extraction_attempts'] > 0 else 0
            )
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

    def clear_caches(self):
        """Leert alle Caches"""
        self._platform_cache.clear()
        self._audio_url_cache.clear()
        self._live_status_cache.clear()
        self._stream_info_cache.clear()
        
        if self._debug:
            print("🗑️ [DEBUG] All caches cleared")

    def dispose(self):
        """Bereinigt alle Ressourcen"""
        self.clear_caches()
        
        if self._debug:
            print("🔌 [DEBUG] StreamManager disposed")

    def __del__(self):
        """Destruktor"""
        try:
            self.dispose()
        except:
            pass

class ExcellenceFFmpegManager:
    """
    🎯 FFMPEG MANAGER
    - Stabilere YouTube URL-Auflösung
    """

    def __init__(self, config=None):
        """Initialize FFmpeg manager with process tracking and StreamManager integration."""
        self._processes = {}
        self._process_counter = 0
        self._lock = threading.RLock()
        self._active_count = 0
        self._shutting_down = False
        self.config = ExcellenceConfig()
        self._stream_manager = None
        self._pid_tracking = {}
        self._live_detection_cache = {}
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

        print(f"✅ FFmpeg Manager initialized (Platform: {SYSTEM})")

    def _resolve_audio_url_enhanced(self, video_url: str) -> Optional[str]:
        """
         Audio-URL-Extraktion
        """
        print(f"\n🎵 [FFMPEG-MANAGER] Resolving audio URL for: {video_url[:80]}...")
        
        self._stats['extraction_attempts'] += 1
        
        if not video_url or not isinstance(video_url, str):
            print(f"  ❌ Invalid URL input")
            self._stats['failed_extractions'] += 1
            return None
        
        url = video_url.strip()
        if not url:
            print(f"  ❌ Empty URL")
            self._stats['failed_extractions'] += 1
            return None
        
        url_lower = url.lower()
        is_youtube = any(domain in url_lower for domain in ['youtube.com', 'youtu.be'])
        
        if not is_youtube:
            print(f"  ℹ️ Not YouTube, returning direct URL")
            return url
        
        print(f"  🎯 YouTube URL detected")
        
        extraction_methods = [
            ("Simple yt-dlp", self._try_simple_extraction),
            ("Audio formats", self._try_audio_formats),
            ("JSON fallback", self._try_json_extraction),
            ("HLS emergency", self._try_hls_generation)
        ]
        
        for method_name, method_func in extraction_methods:
            print(f"  🔄 Trying: {method_name}...")
            try:
                result = method_func(url)
                if result:
                    print(f"  ✅ SUCCESS with {method_name}!")
                    print(f"    URL: {result[:100]}...")
                    self._stats['successful_extractions'] += 1
                    return result
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    print(f"    ⏰ Timeout")
                elif "not found" in error_msg.lower():
                    print(f"    🔍 Not found")
                else:
                    print(f"    ⚠️ Error: {error_msg[:50]}")
                continue
        
        print(f"  🚨 All methods failed, returning original URL")
        print(f"  ⚠️ FFmpeg will attempt with headers")
        self._stats['failed_extractions'] += 1
        return url

    def _try_simple_extraction(self, url: str) -> Optional[str]:
        """📦 Methode 1: Einfache yt-dlp Extraktion"""
        cmd = [
            'yt-dlp',
            '-g',
            '-f', 'bestaudio[ext=m4a]/bestaudio/best',
            '--no-warnings',
            '--no-check-certificate',
            '--quiet',
            '--no-playlist',
            '--socket-timeout', '10',
            url
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=12,
            shell=False,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            if result.stderr:
                error_preview = result.stderr[:100].replace('\n', ' ')
                print(f"    ❌ yt-dlp error: {error_preview}")
            return None
        
        if not result.stdout.strip():
            return None
        
        lines = result.stdout.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and line.startswith(('http://', 'https://')):
                return line
        
        return None

    def _try_audio_formats(self, url: str) -> Optional[str]:
        """🎵 Methode 2: Spezifische Audio-Format IDs"""
        format_priorities = ['140', '139', '251', '250']
        
        for format_id in format_priorities[:2]:
            try:
                cmd = [
                    'yt-dlp',
                    '-g',
                    '-f', format_id,
                    '--no-warnings',
                    '--no-check-certificate',
                    '--quiet',
                    '--socket-timeout', '8',
                    url
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=False
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    audio_url = result.stdout.strip()
                    if audio_url.startswith(('http://', 'https://')):
                        print(f"    ↪ Found format {format_id}")
                        return audio_url
                        
            except subprocess.TimeoutExpired:
                print(f"    ⏰ Format {format_id} timeout")
                continue
            except Exception:
                continue
        
        return None

    def _try_json_extraction(self, url: str) -> Optional[str]:
        """📄 Methode 3: JSON-basierte Extraktion (Fallback)"""
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-warnings',
            '--no-check-certificate',
            '--quiet',
            '--socket-timeout', '15',
            '--playlist-items', '1',
            url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                shell=False,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                return None
            
            output = result.stdout.strip()
            
            start = output.find('{')
            end = output.rfind('}')
            
            if start == -1 or end == -1:
                return None
            
            json_str = output[start:end + 1]
            
            import json
            data = json.loads(json_str)
            
            best_url = None
            best_score = -1
            
            for fmt in data.get('formats', []):
                if fmt.get('acodec') == 'none':
                    continue
                
                url = fmt.get('url')
                if not url or not url.startswith(('http://', 'https://')):
                    continue
                
                score = 0
                
                if fmt.get('vcodec') == 'none':
                    score += 100
                
                bitrate = fmt.get('abr', 0) or fmt.get('tbr', 0) or 0
                score += bitrate
                
                if score > best_score:
                    best_score = score
                    best_url = url
            
            return best_url
            
        except json.JSONDecodeError:
            print(f"    ⚠️ JSON parse failed")
            return None
        except Exception as e:
            print(f"    ⚠️ JSON extraction error: {str(e)[:50]}")
            return None

    def _try_hls_generation(self, url: str) -> Optional[str]:
        """🚨 Methode 4: HLS Notfall-Generierung"""
        video_id = None
        
        if 'youtube.com/watch?v=' in url:
            parts = url.split('v=')
            if len(parts) > 1:
                video_id = parts[1].split('&')[0]
        elif 'youtu.be/' in url:
            parts = url.split('youtu.be/')
            if len(parts) > 1:
                video_id = parts[1].split('?')[0]
        
        if not video_id or len(video_id) != 11:
            print(f"    ⚠️ Could not extract video ID")
            return None
        
        print(f"    🆔 Extracted Video ID: {video_id}")
        
        expiry = int(time.time()) + 7200
        
        hls_patterns = [
            f"https://manifest.googlevideo.com/api/manifest/hls_variant/expire/{expiry}/ei/random/id/{video_id}",
            f"https://manifest.googlevideo.com/api/manifest/hls_playlist/id/{video_id}",
            f"https://manifest.googlevideo.com/api/manifest/hls_variant/id/{video_id}"
        ]
        
        for hls_url in hls_patterns:
            print(f"    🔧 Generated HLS URL: {hls_url[:80]}...")
            return hls_url
        
        return None

    def _build_ffmpeg_command_optimized(self, url: str) -> List[str]:
        """
        🎯 OPTIMIERT: Baut FFmpeg-Befehl mit VOLLSTÄNDIGEN YOUTUBE HEADERS
        """
        is_live, platform = self._detect_stream_type(url)
        stream_type = "LIVE" if is_live else "VIDEO"

        print(f"\n🎬 Building FFmpeg command for {platform} ({stream_type})")
        print(f"  📍 URL: {url[:80]}...")

        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']

        if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
            print("  🎯 Adding YouTube-specific headers")
            
            headers = [
                'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Origin: https://www.youtube.com',
                'Referer: https://www.youtube.com/',
                'Accept: */*',
                'Accept-Language: en-US,en;q=0.9',
                'Accept-Encoding: gzip, deflate',
                'Connection: keep-alive',
            ]
            
            if 'manifest.googlevideo.com' in url:
                headers.extend([
                    'Sec-Fetch-Dest: empty',
                    'Sec-Fetch-Mode: cors',
                    'Sec-Fetch-Site: same-site',
                ])
            
            headers_string = '\r\n'.join(headers)
            cmd.extend(['-headers', headers_string])
            print(f"  📋 Applied {len(headers)} YouTube headers")

        if is_live:
            print("  📡 LIVE: Using HLS/Live optimization")
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
            print("  🎬 VIDEO: Fast access for non-live content")
            cmd.extend([
                '-rw_timeout', '10000000',
                '-seekable', '1',
                '-flags', '+fastseek',
                '-accurate_seek',
                '-ss', '0',
            ])

        cmd.extend(['-i', url])

        cmd.extend([
            '-vn',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(ExcellenceConfig.SAMPLE_RATE),
            '-ac', str(ExcellenceConfig.CHANNELS),
            '-af', 'volume=1.5,dynaudnorm',
            '-fflags', '+genpts+discardcorrupt',
            '-avoid_negative_ts', 'make_zero',
            '-max_interleave_delta', '0',
            '-threads', '2',
            '-bufsize', '2048k',
            'pipe:1'
        ])

        cmd_preview = ' '.join(cmd[:min(15, len(cmd))])
        if len(cmd) > 15:
            cmd_preview += '...'
        print(f"  ⚙️ Command preview: {cmd_preview}")
        print(f"  📊 Total parameters: {len(cmd)}")

        return cmd

    def start_stream(self, video_url: str, output_queue: queue.Queue,
                    process_id: str) -> Optional[subprocess.Popen]:
        """
        🚀 START STREAM - OPTIMIERT & STABIL
        """
        print(f"\n🎬 FFmpegManager: Starting stream for: {video_url[:80]}...")
        
        with self._lock:
            if self.is_active(process_id):
                print(f"⚠️ Stream {process_id} already active")
                return None
        
        print(f"🎵 Resolving audio URL...")
        audio_url = self._resolve_audio_url_enhanced(video_url)
        
        if not audio_url:
            print("❌ Audio URL resolution failed")
            return None
        
        print(f"✅ Resolved URL: {audio_url[:100]}...")
        
        cmd = self._build_ffmpeg_command_optimized(audio_url)
        
        print(f"🔍 FFmpeg Command Analysis:")
        print(f"  Command length: {len(cmd)} parameters")
        print(f"  Has headers: {'-headers' in str(cmd)}")
        print(f"  Is YouTube: {'youtube' in audio_url.lower()}")
        
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
            
            if 'hls' in audio_url.lower() or '.m3u8' in audio_url.lower():
                print(f"🎯 HLS detected, extended startup time")
            
            print(f"🚀 Starting FFmpeg process...")
            process = subprocess.Popen(cmd, **process_kwargs)
            print(f"✅ FFmpeg process started (PID: {process.pid})")
            
            print(f"⏳ Waiting for stream initialization...")
            time.sleep(3.0 if 'hls' in audio_url.lower() else 1.5)
            
            poll_result = process.poll()
            
            if poll_result is not None:
                try:
                    stderr_output = process.stderr.read(1000).decode('utf-8', errors='ignore')
                    print(f"❌ FFmpeg died immediately. Exit code: {poll_result}")
                    
                    if stderr_output:
                        print(f"📋 FFMPEG STDERR (first 200 chars):")
                        print(stderr_output[:200])
                        
                        if "403" in stderr_output or "Forbidden" in stderr_output:
                            print("💡 TIP: 403 Forbidden - Try different URL or headers")
                        elif "404" in stderr_output:
                            print("💡 TIP: 404 Not Found - URL might be expired")
                        elif "Connection refused" in stderr_output:
                            print("💡 TIP: Connection refused - Server not reachable")
                    
                except Exception as e:
                    print(f"⚠️ Could not read stderr: {e}")
                
                return None
            
            print(f"✅ FFmpeg is running (PID: {process.pid})")
            
            self._register_process(process_id, process, output_queue, audio_url)
            
            print(f"🔍 Testing initial data read...")
            test_data = self.read_audio_data(process_id, 4096)
            
            if test_data and len(test_data) > 0:
                print(f"✅ Stream is producing data ({len(test_data)} bytes)")
                if len(test_data) >= 4:
                    print(f"🔊 First 4 bytes (hex): {test_data[:4].hex()}")
            else:
                print("⚠️ Warning: No initial data from stream")
                for attempt in range(2):
                    print(f"  🔄 Attempt {attempt + 1}/2 waiting for HLS data...")
                    time.sleep(2.0)
                    test_data = self.read_audio_data(process_id, 4096)
                    if test_data:
                        print(f"  ✅ Data arrived after {(attempt + 1) * 2} seconds")
                        break
            
            print(f"🎉 Stream started successfully!")
            return process
            
        except FileNotFoundError:
            print(f"❌ FFmpeg not found! Please install FFmpeg.")
            print(f"💡 Download from: https://ffmpeg.org/download.html")
            return None
        except PermissionError:
            print(f"❌ Permission denied - cannot execute FFmpeg")
            return None
        except Exception as e:
            print(f"❌ Failed to start FFmpeg: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _register_process(self, process_id: str, process: subprocess.Popen,
                         output_queue: queue.Queue, url: str):
        """Register process with enhanced tracking."""
        with self._lock:
            is_live, platform = self._detect_stream_type(url)

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
                'headers_used': ('-headers' in str(self._build_ffmpeg_command_optimized(url)))
            }
            self._active_count += 1

            self._pid_tracking[process.pid] = {
                'process_id': process_id,
                'start_time': time.time(),
                'url': url[:100],
                'platform': platform,
                'is_live': is_live
            }

            print(f"📊 Process registered: {process_id}")
            print(f"   PID: {process.pid}, Platform: {platform}, Live: {is_live}")

    def _detect_stream_type(self, url: str) -> Tuple[bool, str]:
        """🔍 INTELLIGENTE STREAM-TYP-ERKENNUNG"""
        cache_key = hashlib.md5(url.encode()).hexdigest()[:16]
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
            elif url_lower.startswith('file://'):
                platform = "Local File"
            elif '.m3u8' in url_lower:
                platform = "HLS Stream"
                is_live = True
            elif '.mpd' in url_lower:
                platform = "DASH Stream"
                is_live = True
            else:
                platform = "HTTP Stream"

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

        except Exception as e:
            print(f"⚠️ Stream type detection error: {e}")
            return False, "unknown"

    def get_stats(self) -> Dict[str, Any]:
        """📊 Gibt detaillierte Statistiken zurück"""
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
        """Read audio data from FFmpeg process with enhanced error handling."""
        with self._lock:
            if process_id not in self._processes:
                return None

            process_info = self._processes[process_id]

            if process_info.get('stopping', False):
                return None

            process = process_info['process']

            try:
                audio_data = process.stdout.read(size)

                if audio_data:
                    process_info['bytes_read'] += len(audio_data)
                    process_info['chunks_processed'] += 1
                    process_info['last_activity'] = time.time()
                    
                    if process_info['chunks_processed'] <= 3:
                        print(f"📦 Chunk #{process_info['chunks_processed']}: {len(audio_data)} bytes")
                    
                    return audio_data
                else:
                    if process.poll() is not None:
                        exit_code = process.poll()
                        print(f"⚠️ Process {process_id} terminated (exit: {exit_code})")
                        
                        try:
                            stderr = process.stderr.read(300).decode('utf-8', errors='ignore')
                            if stderr:
                                print(f"📝 Last error: {stderr[:150]}")
                        except:
                            pass
                        
                        self.stop_stream(process_id)
                        return None

                    if process_info.get('is_live', False) or 'YouTube' in process_info.get('platform', ''):
                        return None
                    else:
                        return None

            except (IOError, OSError) as e:
                print(f"⚠️ Read error for {process_id}: {e}")
                self.stop_stream(process_id)
                return None
            except Exception as e:
                print(f"⚠️ Unexpected read error for {process_id}: {e}")
                self.stop_stream(process_id)
                return None

    def stop_stream(self, process_id: str) -> bool:
        """Stop FFmpeg process gracefully."""
        with self._lock:
            if process_id not in self._processes:
                return True

            process_info = self._processes[process_id]

            if process_info.get('stopping', False):
                return True

            process = process_info['process']
            process_info['stopping'] = True

            termination_success = False

            try:
                if process.poll() is None:
                    print(f"🔄 Stopping process {process_id} ({process.pid})...")

                    try:
                        process.terminate()
                        process.wait(timeout=1.0)
                        termination_success = True
                        print(f"✅ Process {process_id} terminated gracefully")

                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                            process.wait(timeout=1.0)
                            termination_success = True
                            print(f"✅ Process {process_id} killed")

                        except subprocess.TimeoutExpired:
                            termination_success = False
                            print(f"❌ Could not terminate {process_id}")
                else:
                    termination_success = True
                    print(f"✅ Process {process_id} already terminated")

            except Exception as e:
                print(f"❌ Error stopping {process_id}: {e}")
                termination_success = False
            finally:
                self._cleanup_process_resources(process_id, process)

            return termination_success

    def _cleanup_process_resources(self, process_id: str, process: subprocess.Popen):
        """Clean up process resources thoroughly."""
        try:
            if process_id in self._processes:
                del self._processes[process_id]
                self._active_count = max(0, self._active_count - 1)

            if process.pid in self._pid_tracking:
                del self._pid_tracking[process.pid]

            pipes_to_close = []
            if hasattr(process, 'stdout'): pipes_to_close.append(process.stdout)
            if hasattr(process, 'stderr'): pipes_to_close.append(process.stderr)
            if hasattr(process, 'stdin'): pipes_to_close.append(process.stdin)

            for pipe in pipes_to_close:
                if pipe and not pipe.closed:
                    try:
                        pipe.close()
                    except:
                        pass

            if process.poll() is None:
                try:
                    process.terminate()
                    time.sleep(0.1)
                except:
                    pass

                try:
                    process.kill()
                    time.sleep(0.1)
                except:
                    pass

            gc.collect()

            print(f"🧹 Resources cleaned for: {process_id}")

        except Exception as e:
            print(f"⚠️ Resource cleanup error for {process_id}: {e}")
        finally:
            gc.collect()

    def stop_all_streams(self):
        """Stop all active streams gracefully."""
        print("🛑 Stopping all streams...")

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
                    print(f"⚠️ Error stopping {process_id}: {e}")
                    fail_count += 1

            self._shutting_down = False

            print(f"✅ Streams stopped: {success_count} successful, {fail_count} failed")

    def is_active(self, process_id: str) -> bool:
        """Check if process is active with enhanced checking."""
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

    def _cleanup_worker(self):
        """Cleanup worker thread for stale processes."""
        while self._cleanup_running:
            try:
                time.sleep(30)
                self.cleanup_stale_processes()
            except Exception as e:
                print(f"⚠️ Cleanup worker error: {e}")

    def cleanup_stale_processes(self):
        """Clean up stale processes with intelligent detection."""
        with self._lock:
            current_time = time.time()
            stale_processes = []

            for process_id, process_info in self._processes.items():
                process = process_info['process']
                start_time = process_info['start_time']
                last_activity = process_info.get('last_activity', start_time)

                if process.poll() is not None:
                    stale_processes.append(process_id)

                elif current_time - last_activity > 120:
                    print(f"⚠️ Process {process_id} inactive for {current_time - last_activity:.0f}s")
                    stale_processes.append(process_id)

                elif current_time - start_time > 3600:
                    print(f"⚠️ Process {process_id} running for {current_time - start_time:.0f}s")
                    stale_processes.append(process_id)

            for process_id in stale_processes:
                print(f"🧹 Cleaning stale process: {process_id}")
                self._cleanup_process_resources(process_id, self._processes[process_id]['process'])

    def dispose(self):
        """Clean shutdown with thorough resource cleanup."""
        print("🧹 Shutting down FFmpeg Manager...")

        self._cleanup_running = False
        self.stop_all_streams()

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)

        self._live_detection_cache.clear()
        self._pid_tracking.clear()
        self._processes.clear()

        gc.collect()

        print("✅ FFmpeg Manager disposed")

@dataclass
class StreamInfo:
    title: str
    uploader: str
    duration: str
    view_count: int
    platform: str
    description: str = ""

class StreamInfoExtractor:
    """
    Extracts metadata from streams and local files.
    """

    def __init__(self):
        self.current_info = StreamInfo(
            title="Unknown Stream",
            uploader="Unknown",
            duration="Live",
            view_count=0,
            platform="Unknown"
        )
        self._lock = threading.RLock()
        self.stream_manager = StreamManager(enable_debug=True)

    def extract_stream_info(self, url: str) -> StreamInfo:
        """
        Extrahiert Stream-Info MIT COOKIE-SUPPORT für YouTube
        """
        print(f"🔍 StreamInfoExtractor: Getting info for: {url[:50]}...")
        print(f"🎯 [DEBUG] extract_stream_info called in {self.__class__.__name__}")
        print(f"   URL: {url[:50]}...")

        if url.startswith('file://'):
            file_path = url[7:]
            return StreamInfo(
                title=os.path.basename(file_path),
                uploader="Local File",
                duration="File",
                view_count=0,
                platform="local"
            )
        
        if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
            return self._extract_youtube_info_with_cookies(url)
        
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

                self.current_info = StreamInfo(
                    title=info.get('title', 'Unknown Title'),
                    uploader=info.get('uploader', 'Unknown'),
                    duration=info.get('duration_string', 'Live'),
                    view_count=info.get('view_count', 0),
                    platform=platform,
                    description=info.get('description', '')
                )

                print(f"✅ StreamInfoExtractor: Title found: {self.current_info.title[:50]}...")
                return self.current_info

        except Exception as e:
            print(f"⚠️ StreamInfoExtractor error: {e}")
            pass

        print(f"⚠️ StreamInfoExtractor: Using fallback info")
        return StreamInfo(
            title="YouTube Stream",
            uploader="YouTube",
            duration="Live",
            view_count=0,
            platform="youtube"
        )

    def _extract_youtube_info_with_cookies(self, url: str) -> StreamInfo:
        """
        Extrahiert YouTube-Info mit Cookie-Fallbacks - OPTIMIERT für Linux
        """
        print(f"  🎯 YouTube detected, trying optimized cookie methods for channel name...")
        
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
                            import os
                            os.symlink(str(chromium_file), str(chrome_file))
                    
                    print(f"    🔗 Created Chrome compatibility symlinks for yt-dlp")
                except Exception as e:
                    print(f"    ⚠️ Chrome symlink setup failed: {e}")
        
        methods = []
        
        if IS_LINUX:
            linux_browsers = [
                ('firefox', 'Firefox'),
                ('chromium', 'Chromium'),
                ('brave', 'Brave'),
                ('chrome', 'Chrome'),
                ('vivaldi', 'Vivaldi'),
                ('opera', 'Opera'),
                ('edge', 'Edge'),
            ]
            
            for browser_cmd, browser_name in linux_browsers:
                methods.append(([
                    'yt-dlp', '--cookies-from-browser', browser_cmd, '--dump-json',
                    '--no-warnings', '--no-check-certificate', '--playlist-items', '1', url
                ], f"{browser_name} Cookies"))
                
        elif IS_WINDOWS:
            windows_browsers = [
                ('chrome', 'Chrome'),
                ('firefox', 'Firefox'),
                ('edge', 'Edge'),
                ('brave', 'Brave'),
                ('opera', 'Opera'),
            ]
            
            for browser_cmd, browser_name in windows_browsers:
                methods.append(([
                    'yt-dlp', '--cookies-from-browser', browser_cmd, '--dump-json',
                    '--no-warnings', '--no-check-certificate', '--playlist-items', '1', url
                ], f"{browser_name} Cookies"))
                
        else:
            macos_browsers = [
                ('safari', 'Safari'),
                ('chrome', 'Chrome'),
                ('firefox', 'Firefox'),
                ('brave', 'Brave'),
                ('edge', 'Edge'),
            ]
            
            for browser_cmd, browser_name in macos_browsers:
                methods.append(([
                    'yt-dlp', '--cookies-from-browser', browser_cmd, '--dump-json',
                    '--no-warnings', '--no-check-certificate', '--playlist-items', '1', url
                ], f"{browser_name} Cookies"))
        
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
        
        print(f"    📋 Using {len(methods)} optimized extraction methods")
        
        max_attempts = min(3, len(methods))
        attempts = 0
        
        for cmd, method_name in methods:
            if attempts >= max_attempts:
                break
                
            attempts += 1
            
            try:
                print(f"    🧪 Attempt {attempts}/{max_attempts}: {method_name}")
                
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
                
                if result.returncode != 0:
                    if result.stderr:
                        error_preview = result.stderr[:80].replace('\n', ' ')
                        if IS_LINUX and 'chrome' in method_name.lower() and 'could not find' in error_preview:
                            print(f"      ⏩ Skipping Chrome (not available)")
                            continue
                        print(f"      ❌ Error: {error_preview}")
                
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
                            
                            print(f"      ✅ Success with {method_name}")
                            print(f"        Title: {self.current_info.title[:60]}...")
                            print(f"        Channel: {self.current_info.uploader}")
                            if self.current_info.duration != 'Live':
                                print(f"        Duration: {self.current_info.duration}")
                            
                            return self.current_info
                        
                    except json.JSONDecodeError as e:
                        print(f"      ⚠️ JSON parse failed, trying text extraction...")
                        
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
                                    print(f"      ✅ Extracted title from output")
                                    return self.current_info
                        
                        continue
                    except Exception as e:
                        print(f"      ⚠️ Processing error: {str(e)[:50]}")
                        continue
                    
            except subprocess.TimeoutExpired:
                print(f"      ⏰ Timeout after {timeout}s")
                continue
            except Exception as e:
                print(f"      ⚠️ Method error: {str(e)[:50]}")
                continue
        
        print(f"    🔄 Ultimate fallback: Direct title extraction...")
        try:
            cmd_title = ['yt-dlp', '--get-title', '--no-warnings', 
                         '--no-check-certificate', '--quiet', url]
            cmd_uploader = ['yt-dlp', '--get-filename', '-o', '%(uploader)s', 
                           '--no-warnings', '--no-check-certificate', '--quiet', url]
            
            from concurrent.futures import ThreadPoolExecutor
            
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
            
            print(f"      ✅ Success with direct extraction")
            print(f"        Title: {self.current_info.title[:60]}...")
            print(f"        Channel: {self.current_info.uploader}")
            return self.current_info
            
        except Exception as e:
            print(f"      ⚠️ Direct extraction failed: {e}")
        
        print(f"    ⚠️ Using generic YouTube info as last resort")
        return StreamInfo(
            title="YouTube Stream",
            uploader="YouTube",
            duration="Live",
            view_count=0,
            platform="youtube",
            description=""
        )

class LanguageDetector:
    """
    Detects language of video/audio files.
    """

    def __init__(self, transcription_engine):
        self.transcription_engine = transcription_engine

    def detect_video_language(self, video_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(video_path):
                return {'error': 'File not found'}

            try:
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                if file_size > 500:
                    return {'info': 'Large file - direct processing recommended'}
            except:
                pass

            temp_audio = self._extract_audio_sample(video_path, duration=30)
            if not temp_audio:
                return {'error': 'Could not extract audio'}

            result = self.transcription_engine.transcribe_audio(temp_audio)

            if result and hasattr(result, 'language'):
                language_code = result.language
                language_name = SUPPORTED_LANGUAGES.get(language_code, 'Unknown')

                return {
                    'detected_language': language_code,
                    'language_name': language_name,
                    'confidence': getattr(result, 'confidence', 0.8),
                    'sample_text': result.text[:100] + '...' if len(result.text) > 100 else result.text
                }
            else:
                return {'error': 'Language could not be detected'}

        except Exception:
            return {'error': 'Analysis failed'}

    def _extract_audio_sample(self, video_path: str, duration: int = 30) -> Optional[bytes]:
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-t', str(duration),
                '-f', ExcellenceConfig.AUDIO_FORMAT,
                '-ar', str(ExcellenceConfig.SAMPLE_RATE),
                '-ac', str(ExcellenceConfig.CHANNELS),
                '-loglevel', 'quiet',
                '-'
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=30, shell=False)
            if result.returncode == 0 and result.stdout:
                return result.stdout

        except Exception:
            pass

        return None

class ProgressDialog:
    """
    Progress dialog with robust cancel button and non-blocking updates.
    """

    def __init__(self, parent, title="Processing..."):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x120")
        self.dialog.configure(bg=ModernColors.BG_PRIMARY)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.dialog.protocol("WM_DELETE_WINDOW", self.cancel)

        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")

        content_frame = tk.Frame(self.dialog, bg=ModernColors.BG_PRIMARY, padx=20, pady=20)
        content_frame.pack(fill='both', expand=True)

        self.message_label = tk.Label(
            content_frame,
            text="Analyzing video...",
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.TEXT_PRIMARY,
            font=ModernFonts.PRIMARY
        )
        self.message_label.pack(pady=(0, 10))

        self.progress = ttk.Progressbar(
            content_frame,
            mode='indeterminate',
            length=250
        )
        self.progress.pack(pady=(0, 15))

        button_frame = tk.Frame(content_frame, bg=ModernColors.BG_PRIMARY)
        button_frame.pack()

        self.cancel_button = tk.Button(
            button_frame, text="Cancel", command=self.cancel,
            bg=ModernColors.ERROR, fg=ModernColors.TEXT_PRIMARY,
            relief='flat', padx=15)
        self.cancel_button.pack()

        self.is_cancelled = False
        self.progress.start()

        self._update_interval = 100
        self._is_running = True
        self._schedule_updates()

    def _schedule_updates(self):
        if (self._is_running and
                hasattr(self, 'dialog') and
                self.dialog.winfo_exists()):

            try:
                self.dialog.update_idletasks()
                self.dialog.after(self._update_interval, self._schedule_updates)
            except tk.TclError:
                self._is_running = False

    def cancel(self):
        self.is_cancelled = True
        self._is_running = False

        if hasattr(self, 'message_label') and self.message_label.winfo_exists():
            self.message_label.config(text="Cancelling...", fg=ModernColors.ERROR)

        if hasattr(self, 'cancel_button') and self.cancel_button.winfo_exists():
            self.cancel_button.config(text="Cancelling...", state='disabled')

        self.close()

    def update_message(self, message: str):
        if (self._is_running and
                hasattr(self, 'message_label') and
                self.message_label.winfo_exists()):

            try:
                self.message_label.config(text=message)
            except tk.TclError:
                self._is_running = False

    def close(self):
        self._is_running = False

        try:
            if hasattr(self, 'progress'):
                self.progress.stop()
        except Exception:
            pass

        try:
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.destroy()

        except Exception:
            pass

class ExcellenceAudioProcessor:
    """🎵 VOLLSTÄNDIG OPTIMIERTER Audio Processor mit YouTube-Optimierung"""
    
    def __init__(self, controller_ref=None, ffmpeg_manager=None, advanced_settings=None):
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
        self.transcription_engine = None
        self.translation_engine = None
        self.plugin_manager = None       
        self._stop_event = threading.Event()
        self._processing = False
        self._current_stream_id = None
        self._last_successful_read_time = time.time()
        self._consecutive_empty_chunks = 0
        self._cleanup_done = False
        self._resource_lock = threading.RLock()
        self._translation_active = True
        self._last_transcription_text = ""
        self._timed_transcriptions = collections.deque(maxlen=self.config.SUBTITLE_BUFFER_SIZE)
        self._timed_translations = collections.deque(maxlen=self.config.SUBTITLE_BUFFER_SIZE)
        self.subtitle_mode = False
        self._recent_transcriptions = collections.deque(maxlen=self.config.RECENT_TRANSCRIPTIONS_SIZE)
        
        self._chunk_counter = 0
        self._empty_reads = 0
        self._stream_start_time = None
        self._total_bytes_processed = 0
        
        self._adaptive_chunk_size = self.chunk_size
        self._network_quality_history = collections.deque(maxlen=20)
        self._performance_history = collections.deque(maxlen=50)
        self._last_successful_read_time = time.time()
        self._chunk_processing_times = collections.deque(maxlen=10)
        self._optimal_chunk_found = False        
        
        print(f"✅ AudioProcessor initialized:")
        print(f"   Config Type: {self._get_config_type()}")
        print(f"   Chunk: {self.chunk_duration}s / {self.chunk_size:,} bytes")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Overlap: {self.overlap_size:,} bytes")
        print(f"   Bytes/sec: {self.config.BYTES_PER_SECOND:,}")
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Berechnet optimale Chunk-Größe basierend auf Performance"""
        if len(self._chunk_processing_times) < 5:
            return self.chunk_size
        

        avg_process_time = sum(self._chunk_processing_times) / len(self._chunk_processing_times)
        

        target_process_time = 1.5
        
        if avg_process_time > 3.0:

            reduction = min(0.5, target_process_time / avg_process_time)
            new_size = max(32000, int(self._adaptive_chunk_size * reduction))
            
        elif avg_process_time < 0.5 and self._chunk_counter > 20:

            increase = min(2.0, target_process_time / max(avg_process_time, 0.1))
            new_size = min(self.config.MAX_CHUNK_BYTES, int(self._adaptive_chunk_size * increase))
            
        else:
            return self._adaptive_chunk_size
        
        if abs(new_size - self._adaptive_chunk_size) > (0.1 * self._adaptive_chunk_size):
            old_size = self._adaptive_chunk_size
            self._adaptive_chunk_size = new_size
            
            if not self._optimal_chunk_found:
                print(f"🔧 Adaptive chunk size: {old_size:,} → {new_size:,} bytes")
                print(f"   Processing time: {avg_process_time:.2f}s (target: {target_process_time}s)")
            
            if self._chunk_counter > 50 and not self._optimal_chunk_found:
                self._optimal_chunk_found = True
                print(f"✅ Optimal chunk size found: {new_size:,} bytes")
        
        return self._adaptive_chunk_size
    
    def _process_audio_chunk(self, audio_data, transcription_callback, translation_callback):
        """Verarbeitet Audio-Chunk mit Performance-Tracking"""
        if not self.transcription_engine:
            return
            
        try:
            start_time = time.time()
            
            transcription = self.transcription_engine.safe_transcribe(audio_data)
            
            process_time = time.time() - start_time
            self._chunk_processing_times.append(process_time)
            
            if not transcription or not transcription.text:
                return
            
            if self._chunk_counter % 10 == 0:
                self._calculate_optimal_chunk_size()
            
            # Rest des bestehenden Codes...
            # ... existierende Verarbeitungslogik
            
        except Exception as e:
            print(f"⚠️ Audio chunk processing error: {e}")
    
    def _read_with_timeout(self, process, size, timeout=1.0):
        """Optimiertes Lesen mit adaptiver Größe"""
        read_size = self._calculate_optimal_chunk_size()
        
        data = b''
        start_time = time.time()
        
        while len(data) < read_size and (time.time() - start_time) < timeout:
            try:
                chunk = process.stdout.read(min(4096, read_size - len(data)))
                if chunk:
                    data += chunk
                else:
                    time.sleep(0.001)
            except (IOError, OSError):
                time.sleep(0.001)
            except Exception as e:
                print(f"⚠️ Read error in timeout: {e}")
                break
        
        # Netzwerk-Qualität tracken
        read_speed = len(data) / max(0.001, time.time() - start_time)
        self._network_quality_history.append(read_speed)
        
        return data if len(data) > 1000 else None
    
    
    def _get_config_type(self) -> str:
        """Ermittelt den Config-Typ"""
        if isinstance(self.config, RealtimeConfig):
            return 'realtime'
        elif isinstance(self.config, HighAccuracyConfig):
            return 'high_accuracy'
        elif isinstance(self.config, YouTubeOptimizedConfig):
            return 'youtube'
        return 'default'
    
    def _extract_audio_url_robust(self, url: str) -> Optional[str]:
        """🎯 ROBUSTE Audio-URL Extraktion mit Config-Optimierung"""
        print(f"🎵 [URL-EXTRACTION] for: {url[:80]}...")
        
        is_youtube = any(domain in url.lower() for domain in ['youtube.com', 'youtu.be'])
        
        try:
            stream_manager = StreamManager(enable_debug=True)
            audio_url = stream_manager.extract_audio_url(url)
            
            if audio_url:
                print(f"✅ [METHOD 1] StreamManager: {audio_url[:80]}...")
                
                if is_youtube:
                    audio_url = self._enhance_youtube_url(audio_url)
                
                return audio_url
                
        except Exception as e:
            print(f"⚠️ StreamManager failed: {e}")
        
        print(f"🔄 [METHOD 2] Direct yt-dlp extraction...")
        
        if self._get_config_type() == 'youtube':
            format_options = ['bestaudio[ext=m4a]', 'bestaudio/best', '140']
        else:
            format_options = ['bestaudio[ext=m4a]/bestaudio/best', 'bestaudio/best', 'worstaudio']
        
        for fmt in format_options[:2]:
            try:
                cmd = [
                    'yt-dlp',
                    '-g',
                    '-f', fmt,
                    '--no-warnings',
                    '--no-check-certificate',
                    '--quiet',
                    url
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=15,
                    shell=False,
                    encoding='utf-8',
                    errors='ignore'
                )
                
                if result.returncode == 0 and result.stdout:
                    audio_url = result.stdout.strip().split('\n')[0]
                    if audio_url.startswith(('http://', 'https://')):
                        print(f"✅ [METHOD 2] Direct yt-dlp: {audio_url[:80]}...")
                        
                        if is_youtube:
                            audio_url = self._enhance_youtube_url(audio_url)
                        
                        return audio_url
                        
            except subprocess.TimeoutExpired:
                print(f"⏰ Format {fmt} timeout")
                continue
            except Exception as e:
                print(f"⚠️ Format {fmt} error: {e}")
                continue
        
        return self._json_extraction_fallback(url)
    
    def _enhance_youtube_url(self, audio_url: str) -> str:
        """Fügt YouTube-spezifische Optimierungen hinzu"""
        if 'manifest.googlevideo.com' in audio_url:
            print(f"🎯 YouTube HLS manifest detected - using optimized parameters")
            
            if '?' not in audio_url:
                audio_url += '?'
            else:
                audio_url += '&'
            
            audio_url += f"cache_bust={int(time.time())}"
        
        return audio_url
    
    def _json_extraction_fallback(self, url: str) -> Optional[str]:
        """JSON-basierte Fallback Extraction"""
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-warnings',
                '--no-check-certificate',
                '--playlist-items', '1',
                url
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                shell=False
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                
                best_url = None
                best_score = 0
                
                for fmt in data.get('formats', []):
                    if fmt.get('acodec') != 'none' and fmt.get('url'):
                        score = 0
                        
                        if fmt.get('vcodec') == 'none':
                            score += 100
                        
                        bitrate = fmt.get('abr', 0) or fmt.get('tbr', 0) or 0
                        score += bitrate
                        
                        ext = fmt.get('ext', '').lower()
                        if ext in ['m4a', 'mp4']:
                            score += 50
                        
                        if score > best_score:
                            best_score = score
                            best_url = fmt['url']
                
                if best_url:
                    print(f"✅ [METHOD 3] JSON fallback: {best_url[:80]}...")
                    return best_url
                    
        except Exception as e:
            print(f"❌ JSON extraction failed: {e}")
        
        return None
    
    def _build_ffmpeg_command_enhanced(self, audio_url: str, detected_lang: Optional[str] = None) -> List[str]:
        """🎯 VOLLSTÄNDIG OPTIMIERTER FFmpeg Command mit ExcellenceConfig"""
        is_youtube = any(domain in audio_url for domain in ['youtube.com', 'youtu.be', 'googlevideo.com'])
        is_hls = '.m3u8' in audio_url.lower() or 'manifest.googlevideo.com' in audio_url
        
        print(f"🔧 Building FFmpeg command...")
        print(f"   URL Type: {'YouTube' if is_youtube else 'Generic'}")
        print(f"   Protocol: {'HLS' if is_hls else 'Direct'}")
        print(f"   Config Type: {self._get_config_type()}")
        
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']
        
        platform = 'windows' if IS_WINDOWS else 'macos' if IS_MACOS else 'linux'
        platform_config = self.config.get_platform_config(platform)
        
        cmd.extend(platform_config.get('ffmpeg_flags', []))
        
        if is_youtube:
            print(f"🎯 Adding YouTube-specific headers...")
            
            headers = self.config.get_youtube_headers(is_manifest=is_hls)
            headers_string = '\\r\\n'.join([f'{k}: {v}' for k, v in headers.items()])
            cmd.extend(['-headers', headers_string])
            
            timeout = self.config.get_timeout_microseconds(is_youtube=True)
            cmd.extend(['-timeout', str(timeout)])
        
        cmd.extend([
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', str(self.config.RECONNECT_DELAY),
            '-reconnect_on_network_error', '1',
            '-rw_timeout', str(self.config.get_timeout_microseconds(is_youtube)),
            '-multiple_requests', '1',
        ])
        
        if is_hls:
            cmd.extend([
                '-seekable', '0',
                '-fflags', '+discardcorrupt+fastseek+genpts',
            ])
        
        cmd.extend(['-i', audio_url])
        
        filter_profile = 'realtime' if self._get_config_type() == 'realtime' else 'transcription'
        audio_filter = self.advanced_settings.get_audio_filter(
            language=detected_lang,
            profile=filter_profile
        )
        
        cmd.extend([
            '-vn',
            '-f', self.audio_format,
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-af', audio_filter,
            '-fflags', '+genpts+discardcorrupt+fastseek',
            '-avoid_negative_ts', 'make_zero',
            '-max_interleave_delta', '0',
            '-threads', str(self.config.FFMPEG_THREADS),
            '-bufsize', self.config.FFMPEG_BUFSIZE,
            'pipe:1'
        ])
        
        if self.config.ENABLE_DEBUG_LOGGING:
            print(f"⚙️ Command preview: {' '.join(cmd[:min(10, len(cmd))])}...")
            print(f"📊 Total parameters: {len(cmd)}")
            print(f"🎵 Audio Filter: {audio_filter[:80]}...")
            print(f"⏱️ Timeout: {timeout/1000000:.1f}s")
        
        return cmd
    
    def _test_audio_stream(self, audio_url: str) -> bool:
        """Testet Audio-Stream mit Config-optimierten Timeouts"""
        print(f"🔍 Testing audio stream: {audio_url[:60]}...")
        
        is_youtube = 'youtube.com' in audio_url.lower() or 'googlevideo.com' in audio_url
        
        try:
            timeout = 5 if is_youtube else self.config.STREAM_TIMEOUT
            
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
                print(f"✅ Stream test successful")
                return True
            else:
                error_msg = result.stderr[:100] if result.stderr else "Unknown error"
                print(f"❌ Stream test failed: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Stream test timeout after {timeout}s")
            return False
        except Exception as e:
            print(f"⚠️ Stream test error: {e}")
            return True
    
    def start_processing(self, url: str, transcription_callback: Callable, 
                        translation_callback: Callable, info_callback: Callable, 
                        error_callback: Callable):
        """Startet Audio-Verarbeitung mit ExcellenceConfig-Optimierung"""
        print(f"\n🔊 [START_PROCESSING] URL: {url[:80]}...")
        print(f"   Config Type: {self._get_config_type()}")
        print(f"   Chunk Size: {self.chunk_size:,} bytes")
        
        if self._processing:
            print(f"⚠️ Auto-reset für gestuckten Zustand")
            self.emergency_reset()
            time.sleep(0.3)
        
        with self._resource_lock:
            if self._cleanup_done:
                error_callback("❌ Processor bereits disposed")
                return
            
            self._processing = True
            self._stop_event.clear()
            self._current_stream_id = f"stream_{int(time.time())}"
            self._stream_start_time = time.time()
            self._chunk_counter = 0
            self._total_bytes_processed = 0
            
            print(f"✅ Flags gesetzt: processing=True, ID={self._current_stream_id}")
        
        thread = threading.Thread(
            target=self._process_loop_enhanced,
            args=(url, transcription_callback, translation_callback, 
                  info_callback, error_callback),
            daemon=True,
            name=f"AudioProc_{self._current_stream_id}"
        )
        thread.start()
        print(f"✅ Processing thread gestartet: {thread.name}")
    
    def _process_loop_enhanced(self, url: str, transcription_callback: Callable, 
                              translation_callback: Callable, info_callback: Callable,
                              error_callback: Callable):
        """🎯 OPTIMIERTER Processing Loop mit YouTube-Optimierung"""
        process = None
        detected_language = None
        
        try:
            print(f"\n🎬 [PROCESS_LOOP] Start für: {url[:60]}...")
            
            info_callback("🔍 Extracting audio URL...")
            audio_url = self._extract_audio_url_robust(url)
            
            if not audio_url:
                error_callback("❌ Could not extract audio URL")
                return
            
            print(f"✅ Audio URL: {audio_url[:80]}...")
            
            info_callback("🔍 Testing audio stream...")
            if not self._test_audio_stream(audio_url):
                print(f"⚠️ Stream test failed, trying anyway...")
            
            info_callback("🔧 Setting up FFmpeg...")
            
            cmd = self._build_ffmpeg_command_enhanced(audio_url, detected_language)

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
            
            print(f"🚀 Starting FFmpeg process...")
            process = subprocess.Popen(cmd, **process_kwargs)
            print(f"✅ FFmpeg started (PID: {process.pid})")
            
            info_callback("⏳ Initializing stream...")
            
            wait_time = self.config.INITIAL_BUFFER_SECONDS
            if any(keyword in audio_url.lower() for keyword in ['hls', '.m3u8', 'manifest.googlevideo.com']):
                wait_time = 3.0
                print(f"🎯 HLS/Live stream detected, waiting {wait_time}s...")
            
            time.sleep(wait_time)
            
            if process.poll() is not None:
                try:
                    stderr = process.stderr.read(1000).decode('utf-8', errors='ignore')
                    error_msg = f"FFmpeg died: {stderr[:200]}"
                    print(f"❌ {error_msg}")
                    error_callback(f"❌ {error_msg}")
                except:
                    error_callback("❌ FFmpeg failed to start")
                return
            
            info_callback("✅ Stream connected - starting transcription...")
            
            is_youtube = any(domain in audio_url for domain in ['youtube.com', 'youtu.be', 'googlevideo.com'])
            
            if is_youtube:
                print(f"🎯 Using YouTube-optimized streaming loop")
                self._youtube_streaming_loop(
                    process, audio_url, detected_language,
                    transcription_callback, translation_callback,
                    info_callback, error_callback
                )
            else:
                print(f"🎯 Using standard streaming loop")
                self._standard_streaming_loop(
                    process, audio_url, detected_language,
                    transcription_callback, translation_callback,
                    info_callback, error_callback
                )
            
            print(f"🔚 [LOOP END] Reason: {'Stop requested' if self._stop_event.is_set() else 'Process ended'}")
            
        except subprocess.TimeoutExpired:
            error_callback("❌ Timeout - stream not reachable")
        except FileNotFoundError:
            error_callback("❌ FFmpeg not found - please install")
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)[:100]}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            error_callback(f"❌ {error_msg}")
            
        finally:
            if process and process.poll() is None:
                print(f"🛑 Stopping FFmpeg (PID: {process.pid})...")
                self._safe_kill_process(process)

            self._log_final_stats()
            self._guaranteed_cleanup()
            
            print(f"✅ Processing loop ended")
    
    def _standard_streaming_loop(self, process, audio_url, detected_language,
                               transcription_callback, translation_callback,
                               info_callback, error_callback):
        """Standard Streaming Loop für nicht-YouTube Streams"""
        last_data_time = time.time()
        consecutive_timeouts = 0
        
        while (self._processing and not self._stop_event.is_set() 
               and process.poll() is None):
            
            current_time = time.time()
            
            if current_time - last_data_time > self.config.STREAM_TIMEOUT:
                consecutive_timeouts += 1
                
                if consecutive_timeouts > self.config.MAX_CONSECUTIVE_ERRORS:
                    print(f"⚠️ Stream timeout - no data for {current_time - last_data_time:.0f}s")
                    error_callback("❌ Stream timeout - no data received")
                    break
                else:
                    print(f"⚠️ Temporary timeout ({consecutive_timeouts}/{self.config.MAX_CONSECUTIVE_ERRORS})")
                    time.sleep(self.config.RECONNECT_DELAY)
                    continue
            else:
                consecutive_timeouts = 0
            
            if self._chunk_counter > 0 and self._chunk_counter % 100 == 0:
                info_callback(f"📊 {self._chunk_counter} chunks processed...")
            
            if (self.config.LOG_PERFORMANCE and 
                self._chunk_counter > 0 and 
                self._chunk_counter % self.config.PERFORMANCE_LOG_INTERVAL == 0):
                print(f"📊 {self._chunk_counter} chunks processed")
            
            try:
                audio_data = process.stdout.read(self.chunk_size)
            except (IOError, OSError) as e:
                print(f"⚠️ Read error: {e}")
                time.sleep(self.config.READ_RETRY_DELAY)
                continue
            except Exception as e:
                print(f"⚠️ Unexpected read error: {e}")
                time.sleep(self.config.READ_RETRY_DELAY)
                continue
            
            if not audio_data:
                self._empty_reads += 1
                
                if self._empty_reads > self.config.MAX_EMPTY_READS:
                    print(f"⚠️ Too many empty reads: {self._empty_reads}")
                    error_callback("❌ No audio data received")
                    break
                
                sleep_time = min(1.0, self.config.READ_RETRY_DELAY * self._empty_reads)
                time.sleep(sleep_time)
                continue
            
            last_data_time = time.time()
            self._empty_reads = 0
            self._chunk_counter += 1
            self._total_bytes_processed += len(audio_data)
            
            if self._chunk_counter <= 3:
                print(f"📦 Chunk #{self._chunk_counter}: {len(audio_data)} bytes")
            
            try:
                enhanced_audio = self.enhance_audio_quality(audio_data)
            except Exception as e:
                print(f"⚠️ Audio enhancement error: {e}")
                enhanced_audio = audio_data
            
            if self.transcription_engine:
                self._process_audio_chunk(
                    enhanced_audio, 
                    transcription_callback, 
                    translation_callback
                )
    
    def _youtube_streaming_loop(self, process, audio_url, detected_language,
                              transcription_callback, translation_callback,
                              info_callback, error_callback):
        """YouTube-optimierter Streaming Loop mit Auto-Reconnect"""
        max_reconnects = 5
        reconnect_attempt = 0
        session_active = True
        
        while session_active and self._processing and not self._stop_event.is_set():
            try:
                session_success = self._single_youtube_session(
                    process, audio_url, detected_language,
                    transcription_callback, translation_callback,
                    info_callback, error_callback
                )
                
                if not session_success and reconnect_attempt < max_reconnects:
                    reconnect_attempt += 1
                    print(f"🔄 YouTube reconnect attempt {reconnect_attempt}/{max_reconnects}")
                    info_callback(f"🔄 Reconnecting... ({reconnect_attempt}/{max_reconnects})")
                    
                    if process and process.poll() is None:
                        self._safe_kill_process(process)
                    
                    time.sleep(2.0)
                    cmd = self._build_ffmpeg_command_enhanced(audio_url, detected_language)
                    process = subprocess.Popen(cmd, **{
                        'stdout': subprocess.PIPE,
                        'stderr': subprocess.PIPE,
                        'stdin': subprocess.DEVNULL,
                        'bufsize': 10 * 1024 * 1024,
                        'start_new_session': True
                    })
                    print(f"✅ FFmpeg reconnected (PID: {process.pid})")
                    
                    time.sleep(3.0)
                    
                elif not session_success:
                    print(f"❌ Max reconnects reached")
                    error_callback("❌ Could not maintain YouTube connection")
                    session_active = False
                    
                else:
                    session_active = False
                    
            except Exception as e:
                print(f"⚠️ YouTube streaming loop error: {e}")
                reconnect_attempt += 1
                time.sleep(3.0)
    
    def _single_youtube_session(self, process, audio_url, detected_language,
                              transcription_callback, translation_callback,
                              info_callback, error_callback):
        """Einzelne YouTube Streaming-Session"""
        chunk_read_attempts = 0
        max_chunk_attempts = 50
        last_successful_chunk_time = time.time()
        
        while (self._processing and not self._stop_event.is_set() 
               and process.poll() is None):
            
            current_time = time.time()
            
            if self._chunk_counter > 0:
                youtube_timeout = 25
                
                if current_time - last_successful_chunk_time > youtube_timeout:
                    print(f"⚠️ YouTube idle timeout after {self._chunk_counter} chunks")
                    return False
            else:
                if current_time - last_successful_chunk_time > self.config.STREAM_TIMEOUT:
                    print(f"⚠️ YouTube initial timeout")
                    return False
            
            current_chunk_size = self.chunk_size
            if chunk_read_attempts > 10:
                current_chunk_size = int(2.0 * self.config.BYTES_PER_SECOND)
            
            try:
                audio_data = self._read_with_timeout(process, current_chunk_size, timeout=1.0)
                
                if audio_data and len(audio_data) > 1000:
                    chunk_read_attempts = 0
                    last_successful_chunk_time = time.time()
                    
                    self._empty_reads = 0
                    self._chunk_counter += 1
                    self._total_bytes_processed += len(audio_data)
                    
                    if self._chunk_counter <= 3:
                        print(f"📦 YouTube Chunk #{self._chunk_counter}: {len(audio_data)} bytes")
                    
                    try:
                        enhanced_audio = self.enhance_audio_quality(audio_data)
                    except Exception as e:
                        enhanced_audio = audio_data
                    
                    if self.transcription_engine:
                        self._process_audio_chunk(
                            enhanced_audio, 
                            transcription_callback, 
                            translation_callback
                        )
                    
                    if self._chunk_counter % 50 == 0:
                        info_callback(f"📊 {self._chunk_counter} chunks processed...")
                    
                else:
                    chunk_read_attempts += 1
                    
                    if chunk_read_attempts > max_chunk_attempts:
                        print(f"⚠️ Too many failed chunk reads: {chunk_read_attempts}")
                        return False
                    
                    sleep_time = min(2.0, 0.1 * chunk_read_attempts)
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"⚠️ Session read error: {e}")
                chunk_read_attempts += 1
                time.sleep(1.0)
        
        return True
    
    def _read_with_timeout(self, process, size, timeout=1.0):
        """Liest Daten mit Timeout"""
        data = b''
        start_time = time.time()
        
        while len(data) < size and (time.time() - start_time) < timeout:
            try:
                chunk = process.stdout.read(size - len(data))
                if chunk:
                    data += chunk
                else:
                    # Keine Daten im Buffer
                    time.sleep(0.01)
            except (IOError, OSError):
                # Buffer leer
                time.sleep(0.01)
            except Exception as e:
                print(f"⚠️ Read error in timeout: {e}")
                break
        
        return data if len(data) > 0 else None
    
    def emergency_diagnosis(self, url: str) -> bool:
        """Notfall-Diagnose für Stream-Connectivity"""
        print(f"🔍 [EMERGENCY_DIAGNOSIS] Testing: {url[:80]}...")
        
        try:
            audio_url = self._extract_audio_url_robust(url)
            if not audio_url:
                print(f"  ❌ Could not extract audio URL")
                return False
            
            print(f"  ✅ Audio URL extracted: {audio_url[:80]}...")
            
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
                print(f"  ✅ Stream connection successful")
                return True
            else:
                error_msg = result.stderr[:100] if result.stderr else "Unknown error"
                print(f"  ❌ Stream test failed: {error_msg}")
                
                if audio_url.startswith(('http://', 'https://')):
                    print(f"  ⚠️  But URL looks valid, trying anyway...")
                    return True
                return False
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Stream test timeout")
            if 'youtube.com' in url.lower():
                print(f"  ⚠️  YouTube timeout common, trying anyway...")
                return True
            return False
        except Exception as e:
            print(f"  ⚠️  Emergency diagnosis error: {e}")
            return True
    
    def _log_performance_stats(self):
        """Loggt Performance-Statistiken"""
        if not self.config.LOG_PERFORMANCE or self._chunk_counter == 0:
            return
        
        uptime = time.time() - self._stream_start_time
        
        stats = {
            'chunks': self._chunk_counter,
            'bytes': self._total_bytes_processed,
            'uptime': f"{uptime:.1f}s",
            'chunks_per_second': self._chunk_counter / uptime if uptime > 0 else 0,
            'bytes_per_second': self._total_bytes_processed / uptime if uptime > 0 else 0,
            'avg_chunk_size': self._total_bytes_processed / self._chunk_counter if self._chunk_counter > 0 else 0,
            'empty_reads': self._empty_reads,
            'config_type': self._get_config_type(),
        }
        
        print(f"📈 Performance Stats: {stats}")
    
    def _log_final_stats(self):
        """Loggt finale Statistiken"""
        if self._chunk_counter == 0:
            return
        
        uptime = time.time() - self._stream_start_time if self._stream_start_time else 0
        
        print(f"\n📊 FINAL PROCESSING STATS:")
        print(f"   Config Type: {self._get_config_type()}")
        print(f"   Total Chunks: {self._chunk_counter}")
        print(f"   Total Bytes: {self._total_bytes_processed:,}")
        print(f"   Total Time: {uptime:.1f}s")
        print(f"   Avg Chunk Size: {self._total_bytes_processed/self._chunk_counter if self._chunk_counter > 0 else 0:,.0f} bytes")
        print(f"   Processing Rate: {self._chunk_counter/uptime if uptime > 0 else 0:.1f} chunks/sec")
        print(f"   Data Rate: {self._total_bytes_processed/uptime/1024 if uptime > 0 else 0:.1f} KB/sec")
        print(f"   Empty Reads: {self._empty_reads}")
    
    def _process_audio_chunk(self, audio_data, transcription_callback, translation_callback):
        """Verarbeitet einen Audio-Chunk mit Config-Optimierung"""
        if not self.transcription_engine:
            return
            
        try:
            transcription = self.transcription_engine.safe_transcribe(audio_data)
            
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
            print(f"⚠️ Audio chunk processing error: {e}")
    
    def _translate_and_send(self, text, source_lang, translation_callback):
        """Führt Übersetzung durch und sendet Ergebnis"""
        try:
            translation = self.translation_engine.translate_text(text, source_lang)
            if translation:
                if self.subtitle_mode and self.config.ENABLE_TIMED_TRANSLATIONS:
                    self._add_timed_translation(translation)
                translation_callback(translation)
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
    
    def enhance_audio_quality(self, audio_data: bytes) -> bytes:
        """Verbessert Audio-Qualität mit Config-Parametern"""
        if not self.config.AUDIO_ENHANCEMENT_ENABLED or len(audio_data) < 1600 or not NUMPY_AVAILABLE:
            return audio_data
        
        try:
            audio_np = numpy.frombuffer(audio_data, dtype=numpy.int16).astype(numpy.float32) / 32768.0
            
            rms = numpy.sqrt(numpy.mean(audio_np**2))
            
            if rms < self.config.MIN_RMS_THRESHOLD:
                return audio_data
            
            if rms < self.config.TARGET_RMS:
                gain = min(self.config.MAX_GAIN, self.config.TARGET_RMS / max(rms, 1e-6))
                audio_np = audio_np * gain
            
            max_val = numpy.max(numpy.abs(audio_np))
            if max_val > self.config.CLIPPING_THRESHOLD:
                audio_np = audio_np * self.config.CLIPPING_THRESHOLD / max_val
            
            audio_np = audio_np - numpy.mean(audio_np)            
            return (audio_np * 32767).astype(numpy.int16).tobytes()
            
        except Exception:
            return audio_data
    
    def _is_duplicate_transcription(self, current_text: str) -> bool:
        """Prüft auf Duplikate mit Config-Parametern"""
        if not self.config.DUPLICATE_CHECK_ENABLED:
            return False
        
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
    
    def _add_timed_transcription(self, result):
        """Fügt zeitgesteuerte Transkription hinzu"""
        if (hasattr(result, 'start') and result.start is not None and
            hasattr(result, 'end') and result.end is not None):
            self._timed_transcriptions.append(result)
    
    def _add_timed_translation(self, result):
        """Fügt zeitgesteuerte Übersetzung hinzu"""
        if (hasattr(result, 'start') and result.start is not None and
            hasattr(result, 'end') and result.end is not None):
            self._timed_translations.append(result)
    
    def set_engines(self, transcription_engine, translation_engine, plugin_manager=None):
        """Setzt die Processing-Engines"""
        self.transcription_engine = transcription_engine
        self.translation_engine = translation_engine
        self.plugin_manager = plugin_manager
    
    def enable_subtitle_mode(self, enabled: bool):
        """Aktiviert/Deaktiviert Subtitle-Mode"""
        self.subtitle_mode = enabled
        print(f"🎬 Subtitle mode: {'ENABLED' if enabled else 'DISABLED'}")
    
    def get_status(self) -> dict:
        """Gibt Status für Debugging zurück"""
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
    
    def _safe_kill_process(self, process: subprocess.Popen):
        """Ultimative Prozess-Termination mit Process-Group Management"""
        if not process:
            return
        
        pid = process.pid
        print(f"🛑 Terminating process {pid}...")
        
        try:
            if hasattr(self, 'ffmpeg_manager') and self.ffmpeg_manager:
                temp_id = f"kill_{pid}"
                self.ffmpeg_manager._register_process(temp_id, process, None, "terminate")
                self.ffmpeg_manager.stop_stream(temp_id)
                return
        except:
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
                    print(f"✅ Process {pid} terminated ({method_name})")
                    break
            except Exception as e:
                print(f"⚠️ {method_name} termination failed: {e}")
        
        self._cleanup_process_resources(process)
    
    def _terminate_gracefully(self, process, pid, timeout):
        """Graceful termination"""
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

    def _terminate_forcefully(self, process, pid, timeout):
        """Forcierte Termination"""
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

    def _terminate_nuclear(self, process, pid, timeout):
        """Nukleare Option - Systembefehle"""
        try:
            if IS_WINDOWS:
                cmd = ['taskkill', '/F', '/T', '/PID', str(pid)]
            else:
                cmd = ['pkill', '-9', '-P', str(pid)]
            
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if result.returncode == 0:
                return True
        except:
            pass
        
        return False

    def _cleanup_process_resources(self, process):
        """Bereinigt alle Prozess-Ressourcen"""
        for attr in ['stdout', 'stderr', 'stdin']:
            if hasattr(process, attr):
                pipe = getattr(process, attr)
                if pipe and not pipe.closed:
                    try:
                        pipe.close()
                    except:
                        pass
        
        try:
            del process
            import gc
            gc.collect()
        except:
            pass

    def emergency_reset(self, force: bool = False) -> bool:
        """Notfall-Reset für alle Flags"""
        print(f"\n🚨 [EMERGENCY_RESET] force={force}")
        
        with self._resource_lock:
            old_state = self._processing
            self._processing = False
            self._stop_event.set()
            self._current_stream_id = None
            self._consecutive_empty_chunks = 0
            
            if force:
                self._timed_transcriptions.clear()
                self._timed_translations.clear()
                self._recent_transcriptions.clear()
        
        print(f"✅ Reset completed: {old_state} -> {self._processing}")
        return True

    def _guaranteed_cleanup(self):
        """Garantierte Bereinigung nach Beendigung"""
        print(f"\n🧹 [GUARANTEED_CLEANUP]")
        
        with self._resource_lock:
            self._processing = False
            self._stop_event.set()
            self._current_stream_id = None
            self._consecutive_empty_chunks = 0
            self._empty_reads = 0
            self._chunk_counter = 0
            self._total_bytes_processed = 0
            self._cleanup_done = True
        
        time.sleep(0.05)
        
        print(f"✅ Cleanup completed")

    def dispose(self):
        """Gibt Ressourcen des Audio Processors frei"""
        print("🧹 ExcellenceAudioProcessor: Starting dispose...")
        
        try:
            self._stop_event.set()
            self._processing = False
            self._cleanup_done = True
            
            if hasattr(self, 'ffmpeg_manager') and self.ffmpeg_manager:
                try:
                    self.ffmpeg_manager.stop_all_streams()
                except:
                    pass
            
            try:
                self._timed_transcriptions.clear()
                self._timed_translations.clear()
                self._recent_transcriptions.clear()
                self._last_transcription_text = ""
            except:
                pass
            
            try:
                import gc
                gc.collect()
            except:
                pass
            
            print("✅ ExcellenceAudioProcessor disposed")
            
        except Exception as e:
            print(f"⚠️ ExcellenceAudioProcessor dispose error: {e}")

    def stop_processing(self):
        """Stoppt die Audio-Verarbeitung sicher"""
        print("🛑 ExcellenceAudioProcessor: Stopping processing...")
        self._stop_event.set()
        self._processing = False
        
        if hasattr(self, '_current_stream_id') and self._current_stream_id:
            print(f"📛 Stream {self._current_stream_id} stopped by user")

class DarkContextMenu:
    """Dark theme context menu for text widgets with modern styling."""

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.menu = tk.Menu(text_widget, tearoff=0,
                           bg=ModernColors.BG_TERTIARY,
                           fg=ModernColors.TEXT_PRIMARY,
                           activebackground=ModernColors.BG_HOVER,
                           activeforeground=ModernColors.TEXT_ACCENT,
                           borderwidth=1,
                           relief='solid')

        self.menu.add_command(label="Copy", command=self.copy_text)
        self.menu.add_command(label="Select All", command=self.select_all)
        self.menu.add_separator()
        self.menu.add_command(label="Delete", command=self.clear_text)

        text_widget.bind("<Button-3>", self.show_menu)

    def show_menu(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def copy_text(self):
        try:
            selected_text = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.text_widget.clipboard_clear()
            self.text_widget.clipboard_append(selected_text)
        except tk.TclError:
            pass

    def select_all(self):
        self.text_widget.tag_add(tk.SEL, "1.0", tk.END)
        self.text_widget.mark_set(tk.INSERT, "1.0")
        self.text_widget.see(tk.INSERT)

    def clear_text(self):
        self.text_widget.delete("1.0", tk.END)

class DarkEntryContextMenu:
    """Dark theme context menu for entry widgets with modern styling."""

    def __init__(self, entry_widget):
        self.entry_widget = entry_widget
        self.menu = tk.Menu(entry_widget, tearoff=0,
                           bg=ModernColors.BG_TERTIARY,
                           fg=ModernColors.TEXT_PRIMARY,
                           activebackground=ModernColors.BG_HOVER,
                           activeforeground=ModernColors.TEXT_ACCENT,
                           borderwidth=1,
                           relief='solid')

        self.menu.add_command(label="Cut", command=self.cut_text)
        self.menu.add_command(label="Copy", command=self.copy_text)
        self.menu.add_command(label="Paste", command=self.paste_text)
        self.menu.add_separator()
        self.menu.add_command(label="Select All", command=self.select_all)
        self.menu.add_command(label="Delete", command=self.delete_text)

        entry_widget.bind("<Button-3>", self.show_menu)

    def show_menu(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def cut_text(self):
        self.entry_widget.event_generate("<<Cut>>")

    def copy_text(self):
        self.entry_widget.event_generate("<<Copy>>")

    def paste_text(self):
        self.entry_widget.event_generate("<<Paste>>")

    def select_all(self):
        self.entry_widget.select_range(0, 'end')
        self.entry_widget.icursor('end')

    def delete_text(self):
        self.entry_widget.delete(0, 'end')


class ExportManager:
    """
    Supports SRT, VTT and other subtitle formats for export.
    """

    def __init__(self):
        self.supported_formats = ['txt', 'srt', 'vtt', 'json', 'docx']

    def export_subtitles(self, transcript_data: List[ExcellenceTranscriptionResult],
                        translation_data: List[ExcellenceTranslationResult] = None,
                        format: str = 'srt',
                        filename: str = None):
        try:
            timed_transcripts = [t for t in transcript_data
                               if hasattr(t, 'start') and t.start is not None
                               and hasattr(t, 'end') and t.end is not None]

            if not timed_transcripts:
                raise ExcellenceError("No timed transcriptions available")

            if format.lower() == 'srt':
                content = self.generate_srt_content(timed_transcripts, translation_data)
            elif format.lower() == 'vtt':
                content = self.generate_vtt_content(timed_transcripts, translation_data)
            else:
                raise ExcellenceError(f"Unsupported format: {format}")

            if filename:
                with open(filename, 'w', encoding='utf-8-sig') as f:
                    f.write(content)
                return True
            else:
                return content

        except Exception:
            raise ExcellenceError("Subtitle export failed")

    def generate_srt_content(self, transcript_data: List[ExcellenceTranscriptionResult],
                           translation_data: List[ExcellenceTranslationResult] = None) -> str:
        srt_content = ""

        for i, segment in enumerate(transcript_data):
            start_time = self._format_timestamp_srt(segment.start)
            end_time = self._format_timestamp_srt(segment.end)

            display_text = segment.text
            if translation_data and i < len(translation_data):
                display_text = translation_data[i].translated

            srt_content += f"{i+1}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{display_text}\n\n"

        return srt_content

    def generate_vtt_content(self, transcript_data: List[ExcellenceTranscriptionResult],
                           translation_data: List[ExcellenceTranslationResult] = None) -> str:
        vtt_content = "WEBVTT\n\n"

        for i, segment in enumerate(transcript_data):
            start_time = self._format_timestamp_vtt(segment.start)
            end_time = self._format_timestamp_vtt(segment.end)

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

    def export_json(self, transcript_data: List[ExcellenceTranscriptionResult],
                   translation_data: List[ExcellenceTranslationResult], filename: str):
        try:
            export_data = {
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'total_segments': len(transcript_data),
                    'version': '4.1.3'
                },
                'transcripts': [
                    {
                        'text': segment.text,
                        'confidence': segment.confidence,
                        'language': segment.language,
                        'timestamp': segment.timestamp,
                        'start_time': getattr(segment, 'start', None),
                        'end_time': getattr(segment, 'end', None)
                    }
                    for segment in transcript_data
                ],
                'translations': [
                    {
                        'original': trans.original,
                        'translated': trans.translated,
                        'source_lang': trans.source_lang,
                        'target_lang': trans.target_lang,
                        'timestamp': trans.timestamp
                    }
                    for trans in translation_data
                ] if translation_data else []
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception:
            raise ExcellenceError("JSON export failed")

    def export_docx(self, transcript_data: List[ExcellenceTranscriptionResult], filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("TRANSCRIPT EXPORT\n")
                f.write("================\n\n")

                for i, segment in enumerate(transcript_data, 1):
                    timestamp = datetime.fromtimestamp(segment.timestamp).strftime('%H:%M:%S')
                    f.write(f"[{timestamp}] {segment.text}\n\n")

            return True
        except Exception:
            raise ExcellenceError("DOCX export failed")

class BatchProcessor:
    """
    🚀 VOLLSTÄNDIGER BATCH PROCESSOR - Optimiert für Massenverarbeitung
    """

    def __init__(self, transcription_engine=None, translation_engine=None, audio_processor=None):
        self.jobs = {}
        self.current_job = None
        self.is_processing = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._progress_callbacks = []
        self._max_concurrent_jobs = 2
        self._active_jobs = 0
        self.transcription_engine = transcription_engine
        self.translation_engine = translation_engine
        self.audio_processor = audio_processor
        self.stream_manager = StreamManager()

        self._stats = {
            'total_files_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_processing_time': 0,
            'average_time_per_file': 0
        }

        self._result_cache = ExcellenceTTLCache(maxsize=100, ttl=3600)

        print(f"✅ Batch Processor initialized (Max concurrent: {self._max_concurrent_jobs})")

    def create_batch_job(self, urls: List[str], output_dir: str = None,
                         job_name: str = None, options: Dict = None):
        """
        Erstellt einen neuen Batch-Job mit erweiterten Optionen.

        Args:
            urls: Liste von URLs oder Dateipfaden
            output_dir: Ausgabeverzeichnis (default: batch_output_<timestamp>)
            job_name: Optionaler Job-Name
            options: Zusätzliche Optionen

        Returns:
            Job-ID für spätere Referenz
        """
        with self._lock:
            if not output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"batch_output_{timestamp}"


            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)


            job_id = hashlib.md5(f"{str(urls)}{time.time()}".encode()).hexdigest()[:12]

            job = {
                'id': job_id,
                'name': job_name or f"Batch_{job_id}",
                'urls': urls,
                'total_count': len(urls),
                'processed_count': 0,
                'output_dir': str(output_path),
                'status': 'pending',
                'results': [],
                'failed_items': [],
                'options': options or {},
                'created_at': time.time(),
                'start_time': None,
                'end_time': None,
                'progress_percent': 0.0,
                'current_file': None
            }

            self.jobs[job_id] = job

            (output_path / "transcripts").mkdir(exist_ok=True)
            (output_path / "translations").mkdir(exist_ok=True)
            (output_path / "subtitles").mkdir(exist_ok=True)
            (output_path / "logs").mkdir(exist_ok=True)

            print(f"📦 Batch Job created: {job['name']} ({len(urls)} items)")
            return job_id

    def start_job(self, job_id: str):
        """
        Startet die Verarbeitung eines Batch-Jobs.

        Args:
            job_id: ID des zu startenden Jobs

        Returns:
            True wenn erfolgreich gestartet, False bei Fehler
        """
        with self._lock:
            if self.is_processing:
                print("⚠️ Another job is already processing")
                return False

            if job_id not in self.jobs:
                print(f"❌ Job {job_id} not found")
                return False

            job = self.jobs[job_id]

            if job['status'] in ['processing', 'completed']:
                print(f"⚠️ Job {job_id} is already {job['status']}")
                return False

            job['status'] = 'processing'
            job['start_time'] = time.time()
            job['processed_count'] = 0
            job['progress_percent'] = 0.0

            self.current_job = job_id
            self.is_processing = True
            self._stop_event.clear()

            print(f"🚀 Starting batch job: {job['name']}")

            processing_thread = threading.Thread(
                target=self._process_job,
                args=(job_id,),
                daemon=True,
                name=f"BatchJob_{job_id}"
            )
            processing_thread.start()

            return True

    def _process_job(self, job_id: str):
        """
        Hauptverarbeitungsfunktion für einen Job.
        """
        job = self.jobs.get(job_id)
        if not job:
            return

        try:
            urls = job['urls']
            total_count = len(urls)
            options = job['options']

            process_mode = options.get('mode', 'parallel')

            if process_mode == 'parallel':
                self._process_parallel(job_id, urls, options)
            else:
                self._process_sequential(job_id, urls, options)

            with self._lock:
                if self._stop_event.is_set():
                    job['status'] = 'cancelled'
                    print(f"⏹️ Batch job cancelled: {job['name']}")
                else:
                    job['status'] = 'completed'
                    job['end_time'] = time.time()

                    duration = job['end_time'] - job['start_time']
                    job['duration'] = duration

                    successful = len(job['results'])
                    failed = len(job['failed_items'])

                    print(f"✅ Batch job completed: {job['name']}")
                    print(f"   📊 Results: {successful} successful, {failed} failed")
                    print(f"   ⏱️ Duration: {duration:.1f}s")
                    print(f"   🎯 Avg time per item: {duration/max(1, total_count):.1f}s")

                    self._stats['total_files_processed'] += total_count
                    self._stats['successful'] += successful
                    self._stats['failed'] += failed
                    self._stats['total_processing_time'] += duration
                    self._stats['average_time_per_file'] = (
                        self._stats['total_processing_time'] /
                        max(1, self._stats['total_files_processed'])
                    )

                self.is_processing = False
                self.current_job = None

        except Exception as e:
            print(f"❌ Batch job error: {e}")
            import traceback
            traceback.print_exc()

            with self._lock:
                job['status'] = 'failed'
                job['error'] = str(e)
                self.is_processing = False
                self.current_job = None

    def _process_sequential(self, job_id: str, urls: List[str], options: Dict):
        """
        Verarbeitet URLs sequentiell (nacheinander).
        """
        job = self.jobs[job_id]

        for i, url in enumerate(urls):
            if self._stop_event.is_set():
                break

            with self._lock:
                job['current_file'] = url
                job['processed_count'] = i
                job['progress_percent'] = (i / len(urls)) * 100

            self._notify_progress(job_id, i, len(urls), url)

            try:
                print(f"🔍 Processing {i+1}/{len(urls)}: {url[:80]}...")

                result = self._process_single_item(url, options)

                if result:
                    job['results'].append(result)

                    self._save_item_result(job, result, i)

                    print(f"✅ Completed: {result.get('title', 'Unknown')}")
                else:
                    job['failed_items'].append({
                        'url': url,
                        'error': 'Processing failed',
                        'index': i
                    })
                    print(f"❌ Failed: {url}")

            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
                job['failed_items'].append({
                    'url': url,
                    'error': str(e),
                    'index': i
                })

            time.sleep(0.5)

    def _process_parallel(self, job_id: str, urls: List[str], options: Dict):
        """
        Verarbeitet URLs parallel (mehrere gleichzeitig).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        job = self.jobs[job_id]
        max_workers = min(self._max_concurrent_jobs, len(urls))

        print(f"🔄 Parallel processing with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self._process_single_item_with_context, url, options, i):
                (url, i) for i, url in enumerate(urls)
            }

            completed = 0
            for future in as_completed(future_to_url):
                if self._stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                url, index = future_to_url[future]

                try:
                    result = future.result(timeout=300)

                    if result:
                        job['results'].append(result)
                        self._save_item_result(job, result, index)

                        print(f"✅ Completed ({index+1}/{len(urls)}): {result.get('title', 'Unknown')[:50]}")
                    else:
                        job['failed_items'].append({
                            'url': url,
                            'error': 'Processing failed',
                            'index': index
                        })
                        print(f"❌ Failed ({index+1}/{len(urls)}): {url}")

                except Exception as e:
                    print(f"❌ Error processing {url}: {e}")
                    job['failed_items'].append({
                        'url': url,
                        'error': str(e),
                        'index': index
                    })

                completed += 1

                with self._lock:
                    job['processed_count'] = completed
                    job['progress_percent'] = (completed / len(urls)) * 100

                self._notify_progress(job_id, completed, len(urls), url)

    def _process_single_item_with_context(self, url: str, options: Dict, index: int):
        """
        Verarbeitet ein einzelnes Item mit Index-Kontext für bessere Fehlerbehandlung.
        """
        try:
            return self._process_single_item(url, options)
        except Exception as e:
            print(f"❌ Error in item {index}: {e}")
            raise

    def _process_single_item(self, url: str, options: Dict) -> Optional[Dict]:
        """
        Verarbeitet eine einzelne URL/Datei.
        """

        cache_key = hashlib.md5(f"{url}_{str(options)}".encode()).hexdigest()
        cached_result = self._result_cache.get(cache_key)
        if cached_result:
            print(f"   ↪ Using cached result for: {url[:50]}...")
            return cached_result

        start_time = time.time()

        try:
            stream_info = StreamInfoExtractor().extract_stream_info(url)

            audio_url = self.stream_manager.extract_audio_url(url)
            if not audio_url:
                print(f"   ❌ Could not extract audio URL from: {url}")
                return None

            if not self.transcription_engine:
                self.transcription_engine = ExcellenceTranscriptionEngine()
                self.transcription_engine.load_model("medium")

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_path = temp_audio.name

                ffmpeg_cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error',
                    '-i', audio_url,
                    '-t', str(options.get('duration', 300)),
                    '-ar', '16000',
                    '-ac', '1',
                    '-f', 'wav',
                    temp_path
                ]

                import subprocess
                result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=30)

                if result.returncode != 0:
                    print(f"   ❌ FFmpeg error: {result.stderr.decode()[:100]}")
                    return None

                with open(temp_path, 'rb') as f:
                    audio_data = f.read()

                transcription = self.transcription_engine.transcribe_audio(audio_data)

                if not transcription or not transcription.text:
                    print(f"   ❌ Transcription failed for: {url}")
                    return None


                translation = None
                if options.get('translate', False) and self.translation_engine:
                    translation = self.translation_engine.translate_text(
                        transcription.text,
                        transcription.language if hasattr(transcription, 'language') else 'auto'
                    )

                result = {
                    'url': url,
                    'title': stream_info.title,
                    'uploader': stream_info.uploader,
                    'duration': stream_info.duration,
                    'platform': stream_info.platform,
                    'transcription': transcription.text if transcription else '',
                    'translation': translation.translated if translation else '',
                    'source_language': transcription.language if hasattr(transcription, 'language') else 'unknown',
                    'target_language': translation.target_lang if translation else '',
                    'confidence': transcription.confidence if hasattr(transcription, 'confidence') else 0.0,
                    'processing_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                }

                self._result_cache.put(cache_key, result)

                return result

        except Exception as e:
            print(f"   ❌ Processing error for {url}: {e}")
            return None
        finally:

            try:
                import os
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

    def _save_item_result(self, job: Dict, result: Dict, index: int):
        """
        Speichert das Ergebnis eines einzelnen Items.
        """
        output_dir = Path(job['output_dir'])

        safe_title = "".join(c for c in result.get('title', f'item_{index}')
                           if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:50]
        filename_base = f"{index:04d}_{safe_title}"


        transcript_file = output_dir / "transcripts" / f"{filename_base}_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"=== TRANSCRIPT: {result.get('title', 'Unknown')} ===\n\n")
            f.write(f"URL: {result.get('url', '')}\n")
            f.write(f"Source: {result.get('platform', 'unknown')}\n")
            f.write(f"Uploader: {result.get('uploader', 'Unknown')}\n")
            f.write(f"Language: {result.get('source_language', 'unknown')}\n")
            f.write(f"Confidence: {result.get('confidence', 0.0):.1%}\n")
            f.write(f"Processed: {result.get('timestamp', '')}\n")
            f.write("\n" + "="*50 + "\n\n")
            f.write(result.get('transcription', ''))

        if result.get('translation'):
            translation_file = output_dir / "translations" / f"{filename_base}_translation.txt"
            with open(translation_file, 'w', encoding='utf-8') as f:
                f.write(f"=== TRANSLATION: {result.get('title', 'Unknown')} ===\n\n")
                f.write(f"Original language: {result.get('source_language', 'unknown')}\n")
                f.write(f"Target language: {result.get('target_language', '')}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(result.get('translation', ''))

        metadata_file = output_dir / "logs" / f"{filename_base}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def _notify_progress(self, job_id: str, current: int, total: int, current_file: str):
        """
        Benachrichtigt alle registrierten Progress-Callbacks.
        """
        progress_data = {
            'job_id': job_id,
            'current': current,
            'total': total,
            'percent': (current / total) * 100 if total > 0 else 0,
            'current_file': current_file,
            'timestamp': time.time()
        }

        for callback in self._progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                print(f"⚠️ Progress callback error: {e}")

    def register_progress_callback(self, callback: Callable):
        """
        Registriert einen Callback für Fortschrittsupdates.
        """
        if callback not in self._progress_callbacks:
            self._progress_callbacks.append(callback)

    def stop_job(self, job_id: str = None):
        """
        Stoppt die aktuelle Verarbeitung.

        Args:
            job_id: Spezifische Job-ID oder None für aktuellen Job
        """
        self._stop_event.set()

        with self._lock:
            if job_id and job_id in self.jobs:
                self.jobs[job_id]['status'] = 'cancelled'
            elif self.current_job:
                self.jobs[self.current_job]['status'] = 'cancelled'

        print("⏹️ Batch processing stopped")

    def get_job_status(self, job_id: str) -> Dict:
        """
        Gibt den Status eines Jobs zurück.
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return {'error': 'Job not found'}

            # Basis-Status
            status = {
                'id': job['id'],
                'name': job['name'],
                'status': job['status'],
                'total_count': job['total_count'],
                'processed_count': job['processed_count'],
                'progress_percent': job['progress_percent'],
                'current_file': job['current_file'],
                'results_count': len(job['results']),
                'failed_count': len(job['failed_items']),
                'output_dir': job['output_dir']
            }

            if job['start_time']:
                status['start_time'] = datetime.fromtimestamp(job['start_time']).isoformat()
                status['elapsed_seconds'] = time.time() - job['start_time']

            if job['end_time']:
                status['end_time'] = datetime.fromtimestamp(job['end_time']).isoformat()
                status['duration_seconds'] = job.get('duration', 0)

            return status

    def list_jobs(self) -> List[Dict]:
        """
        Listet alle Jobs auf.
        """
        with self._lock:
            jobs_list = []
            for job_id, job in self.jobs.items():
                jobs_list.append({
                    'id': job_id,
                    'name': job['name'],
                    'status': job['status'],
                    'total_count': job['total_count'],
                    'processed_count': job['processed_count'],
                    'created_at': datetime.fromtimestamp(job['created_at']).isoformat(),
                    'output_dir': job['output_dir']
                })

            return sorted(jobs_list, key=lambda x: x['created_at'], reverse=True)

    def export_job_results(self, job_id: str, format: str = 'json') -> Optional[str]:
        """
        Exportiert alle Ergebnisse eines Jobs.

        Args:
            job_id: ID des Jobs
            format: 'json', 'csv', 'txt'

        Returns:
            Pfad zur Export-Datei oder None bei Fehler
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None

            export_file = Path(job['output_dir']) / f"batch_results_{job_id}.{format}"

            try:
                if format == 'json':
                    with open(export_file, 'w', encoding='utf-8') as f:
                        export_data = {
                            'job_info': {
                                'id': job['id'],
                                'name': job['name'],
                                'total_items': job['total_count'],
                                'successful': len(job['results']),
                                'failed': len(job['failed_items']),
                                'start_time': datetime.fromtimestamp(job['start_time']).isoformat()
                                if job['start_time'] else None,
                                'end_time': datetime.fromtimestamp(job['end_time']).isoformat()
                                if job['end_time'] else None
                            },
                            'results': job['results'],
                            'failed_items': job['failed_items']
                        }
                        json.dump(export_data, f, indent=2, ensure_ascii=False)

                elif format == 'csv':
                    import csv
                    with open(export_file, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            'Index', 'Title', 'URL', 'Platform', 'Uploader',
                            'Source Language', 'Target Language', 'Confidence',
                            'Processing Time', 'Timestamp'
                        ])

                        for i, result in enumerate(job['results']):
                            writer.writerow([
                                i,
                                result.get('title', ''),
                                result.get('url', ''),
                                result.get('platform', ''),
                                result.get('uploader', ''),
                                result.get('source_language', ''),
                                result.get('target_language', ''),
                                f"{result.get('confidence', 0):.3f}",
                                f"{result.get('processing_time', 0):.1f}",
                                result.get('timestamp', '')
                            ])

                elif format == 'txt':
                    with open(export_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== BATCH PROCESSING RESULTS ===\n\n")
                        f.write(f"Job: {job['name']}\n")
                        f.write(f"ID: {job['id']}\n")
                        f.write(f"Total items: {job['total_count']}\n")
                        f.write(f"Successful: {len(job['results'])}\n")
                        f.write(f"Failed: {len(job['failed_items'])}\n\n")

                        f.write("SUCCESSFUL ITEMS:\n")
                        f.write("="*50 + "\n")
                        for i, result in enumerate(job['results']):
                            f.write(f"\n{i+1}. {result.get('title', 'Unknown')}\n")
                            f.write(f"   URL: {result.get('url', '')}\n")
                            f.write(f"   Language: {result.get('source_language', 'unknown')}\n")
                            f.write(f"   Confidence: {result.get('confidence', 0):.1%}\n\n")

                print(f"📤 Exported results to: {export_file}")
                return str(export_file)

            except Exception as e:
                print(f"❌ Export failed: {e}")
                return None

    def get_statistics(self) -> Dict:
        """
        Gibt globale Statistiken zurück.
        """
        with self._lock:
            stats = self._stats.copy()
            stats['total_jobs'] = len(self.jobs)
            stats['active_jobs'] = sum(1 for j in self.jobs.values()
                                     if j['status'] == 'processing')
            stats['success_rate'] = (
                stats['successful'] / max(1, stats['total_files_processed']) * 100
            )

            return stats

    def clear_completed_jobs(self, older_than_hours: int = 24):
        """
        Entfernt abgeschlossene Jobs, die älter als X Stunden sind.
        """
        with self._lock:
            current_time = time.time()
            jobs_to_remove = []

            for job_id, job in self.jobs.items():
                if job['status'] in ['completed', 'failed', 'cancelled']:
                    if job['end_time']:
                        age_hours = (current_time - job['end_time']) / 3600
                        if age_hours > older_than_hours:
                            jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]

            if jobs_to_remove:
                print(f"🧹 Cleared {len(jobs_to_remove)} old jobs")

    def dispose(self):
        """
        Gibt alle Ressourcen frei.
        """
        self.stop_job()
        self._stop_event.set()

        with self._lock:
            self.jobs.clear()
            self._progress_callbacks.clear()
            self._result_cache.clear()

        print("🧹 Batch Processor disposed")

class BatchProcessorGUI:
    """
    GUI für den Batch Processor.
    """

    def __init__(self, parent, batch_processor, main_app=None):
        self.parent = parent
        self.batch_processor = batch_processor
        self.main_app = main_app
        self.dialog = None
        self.job_listbox = None
        self.progress_bar = None
        self.status_label = None
        self.batch_processor.register_progress_callback(self._on_progress_update)

    def show_dialog(self):
        """Zeigt das Batch Processor Fenster."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("🐉 Batch Processor")
        self.dialog.geometry("800x600")
        self.dialog.configure(bg=DragonColors.BG_PRIMARY)

        main_frame = tk.Frame(self.dialog, bg=DragonColors.BG_PRIMARY, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        creation_frame = tk.LabelFrame(main_frame, text="➕ New Batch Job",
                                      bg=DragonColors.BG_SECONDARY,
                                      fg=DragonColors.TEXT_PRIMARY,
                                      font=ModernFonts.SUBTITLE)
        creation_frame.pack(fill='x', pady=(0, 20))

        url_frame = tk.Frame(creation_frame, bg=DragonColors.BG_SECONDARY)
        url_frame.pack(fill='x', padx=10, pady=10)

        tk.Label(url_frame, text="URLs (one per line):",
                bg=DragonColors.BG_SECONDARY,
                fg=DragonColors.TEXT_PRIMARY).pack(anchor='w')

        self.url_text = scrolledtext.ScrolledText(url_frame, height=6,
                                                 bg=DragonColors.BG_TERTIARY,
                                                 fg=DragonColors.TEXT_PRIMARY)
        self.url_text.pack(fill='x', pady=(5, 0))

        options_frame = tk.Frame(creation_frame, bg=DragonColors.BG_SECONDARY)
        options_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(options_frame, text="Job Name:",
                bg=DragonColors.BG_SECONDARY,
                fg=DragonColors.TEXT_PRIMARY).pack(side='left')

        self.job_name_var = tk.StringVar(value=f"Batch_{datetime.now().strftime('%Y%m%d_%H%M')}")
        job_name_entry = tk.Entry(options_frame, textvariable=self.job_name_var,
                                 bg=DragonColors.BG_TERTIARY,
                                 fg=DragonColors.TEXT_PRIMARY, width=30)
        job_name_entry.pack(side='left', padx=(5, 20))

        self.translate_var = tk.BooleanVar(value=True)
        translate_cb = tk.Checkbutton(options_frame, text="Translate",
                                     variable=self.translate_var,
                                     bg=DragonColors.BG_SECONDARY,
                                     fg=DragonColors.TEXT_PRIMARY)
        translate_cb.pack(side='left')

        button_frame = tk.Frame(creation_frame, bg=DragonColors.BG_SECONDARY)
        button_frame.pack(fill='x', padx=10, pady=(10, 5))

        tk.Button(button_frame, text="📂 Load from File...",
                 command=self._load_urls_from_file,
                 bg=DragonColors.BG_TERTIARY,
                 fg=DragonColors.TEXT_PRIMARY).pack(side='left', padx=(0, 10))

        tk.Button(button_frame, text="📋 Paste from Clipboard",
                 command=self._paste_urls,
                 bg=DragonColors.BG_TERTIARY,
                 fg=DragonColors.TEXT_PRIMARY).pack(side='left', padx=(0, 10))

        tk.Button(button_frame, text="🚀 Create & Start Job",
                 command=self._create_and_start_job,
                 bg=DragonColors.SUCCESS,
                 fg=DragonColors.TEXT_PRIMARY,
                 font=ModernFonts.BUTTON).pack(side='right')

        active_frame = tk.LabelFrame(main_frame, text="🔄 Active Jobs",
                                    bg=DragonColors.BG_SECONDARY,
                                    fg=DragonColors.TEXT_PRIMARY,
                                    font=ModernFonts.SUBTITLE)
        active_frame.pack(fill='both', expand=True)

        list_frame = tk.Frame(active_frame, bg=DragonColors.BG_SECONDARY)
        list_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.job_listbox = tk.Listbox(list_frame,
                                     bg=DragonColors.BG_TERTIARY,
                                     fg=DragonColors.TEXT_PRIMARY,
                                     selectbackground=DragonColors.COMBO_SELECTION,
                                     height=10)
        self.job_listbox.pack(side='left', fill='both', expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient='vertical',
                                command=self.job_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.job_listbox.config(yscrollcommand=scrollbar.set)

        self.progress_bar = ttk.Progressbar(active_frame, mode='determinate')
        self.progress_bar.pack(fill='x', padx=10, pady=(0, 5))

        self.status_label = tk.Label(active_frame,
                                    text="Ready to process batch jobs",
                                    bg=DragonColors.BG_SECONDARY,
                                    fg=DragonColors.TEXT_SECONDARY)
        self.status_label.pack(fill='x', padx=10, pady=(0, 10))

        control_frame = tk.Frame(active_frame, bg=DragonColors.BG_SECONDARY)
        control_frame.pack(fill='x', padx=10, pady=(0, 10))

        tk.Button(control_frame, text="⏹️ Stop Selected",
                 command=self._stop_selected_job,
                 bg=DragonColors.ERROR,
                 fg=DragonColors.TEXT_PRIMARY).pack(side='left', padx=(0, 10))

        tk.Button(control_frame, text="📊 View Details",
                 command=self._view_job_details,
                 bg=DragonColors.BG_TERTIARY,
                 fg=DragonColors.TEXT_PRIMARY).pack(side='left', padx=(0, 10))

        tk.Button(control_frame, text="📤 Export Results",
                 command=self._export_results,
                 bg=DragonColors.INFO,
                 fg=DragonColors.TEXT_PRIMARY).pack(side='left', padx=(0, 10))

        tk.Button(control_frame, text="🗑️ Clear Completed",
                 command=self._clear_completed,
                 bg=DragonColors.BG_TERTIARY,
                 fg=DragonColors.TEXT_PRIMARY).pack(side='right')

        self._refresh_job_list()
        self._start_auto_refresh()

    def _refresh_job_list(self):
        """Aktualisiert die Job-Liste."""
        if not self.job_listbox:
            return

        self.job_listbox.delete(0, 'end')

        jobs = self.batch_processor.list_jobs()
        for job in jobs:
            status_icons = {
                'pending': '⏳',
                'processing': '🔄',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '⏹️'
            }

            icon = status_icons.get(job['status'], '❓')
            text = f"{icon} {job['name']} - {job['status'].upper()}"

            if job['status'] == 'processing':
                text += f" ({job['processed_count']}/{job['total_count']})"

            self.job_listbox.insert('end', text)

    def _start_auto_refresh(self):
        """Startet automatisches Aktualisieren der Job-Liste."""
        if self.dialog and self.dialog.winfo_exists():
            self._refresh_job_list()
            self.dialog.after(2000, self._start_auto_refresh)

    def _on_progress_update(self, progress_data):
        """Verarbeitet Fortschrittsupdates."""
        if not self.dialog or not self.dialog.winfo_exists():
            return

        def update_gui():
            job_id = progress_data['job_id']
            status = self.batch_processor.get_job_status(job_id)

            if self.progress_bar:
                self.progress_bar['value'] = status['progress_percent']

            if self.status_label:
                current_file = progress_data['current_file']
                if len(current_file) > 50:
                    current_file = current_file[:47] + "..."

                self.status_label.config(
                    text=f"Processing: {current_file} "
                         f"({progress_data['current']}/{progress_data['total']})"
                )

            self._refresh_job_list()

        if self.dialog.winfo_exists():
            self.dialog.after(0, update_gui)

@dataclass
class AppSettings:
    """    Manages application settings and persistence.    """
    last_url: str = ""
    default_model: str = "medium"
    default_language: str = "de"
    layout_mode: str = "vertical"
    recent_urls: List[str] = None
    enable_plugins: bool = True
    export_format: str = "txt"

    def __post_init__(self):
        if self.recent_urls is None:
            self.recent_urls = []

    @classmethod
    def load_from_file(cls, filename="dragon_settings.json"):
        try:
            config_dir = PlatformUtils.get_platform_config_dir()
            file_path = config_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return cls(**data)
        except Exception:
            pass
        return cls()

    def save_to_file(self, filename="dragon_settings.json"):
        try:
            config_dir = PlatformUtils.get_platform_config_dir()
            file_path = config_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def add_recent_url(self, url):
        if url in self.recent_urls:
            self.recent_urls.remove(url)
        self.recent_urls.insert(0, url)
        self.recent_urls = self.recent_urls[:10]
        self.save_to_file()


class ResourceManager:
    """    Central resource management for cleanup and process termination.    """

    def __init__(self):
        self.processes = []
        self.threads = []
        self.temp_files = []
        self.cleanup_done = False
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()

    def register_process(self, process):
        with self._lock:
            if process and process not in self.processes:
                self.processes.append(process)

    def register_thread(self, thread):
        with self._lock:
            if thread and thread not in self.threads and thread.is_alive():
                self.threads.append(thread)

    def register_temp_file(self, file_path):
        with self._lock:
            if file_path and file_path not in self.temp_files:
                self.temp_files.append(file_path)

    def cleanup(self):
        if self.cleanup_done:
            return

        self._shutdown_event.set()

        with self._lock:
            cleanup_timeout = 5.0
            start_time = time.time()

            for process in self.processes[:]:
                try:
                    if process and hasattr(process, 'poll'):
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=1.0)
                            except (subprocess.TimeoutExpired, AttributeError):
                                try:
                                    process.kill()
                                    process.wait(timeout=0.5)
                                except:
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
                except Exception:
                    pass
                finally:
                    if temp_file in self.temp_files:
                        self.temp_files.remove(temp_file)

            try:
                if TORCH_AVAILABLE:
                    torch = FastLazyLoader.load('torch')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except:
                pass

            gc.collect()

            self.cleanup_done = True

    def is_shutting_down(self):
        return self._shutdown_event.is_set()

class WhisperLayoutManager:
    def __init__(self, gui_ref):
        self.gui_ref = gui_ref
        self.root = gui_ref.root
        self._batch_timer_id = None
        
        import queue

        try:
            self.gui_ref._text_update_queue = queue.Queue(maxsize=150)
            self.gui_ref.gui_queue = queue.Queue(maxsize=200)
            print(f"✅ Queues erfolgreich erstellt")
            
        except Exception as e:
            print(f"⚠️ Queue-Erstellung fehlgeschlagen: {e}")
            import queue
            
            class DummyQueue:
                def __init__(self, maxsize=0):
                    self.maxsize = maxsize
                    self._items = []
                    self._lock = threading.Lock()
                    self.Empty = queue.Empty
                
                def put(self, item, block=True, timeout=None):
                    with self._lock:
                        self._items.append(item)
                        if self.maxsize > 0 and len(self._items) > self.maxsize:
                            self._items.pop(0)
                
                def get(self, block=True, timeout=None):
                    with self._lock:
                        if self._items:
                            return self._items.pop(0)
                        raise self.Empty
                
                def empty(self):
                    with self._lock:
                        return len(self._items) == 0
                
                def qsize(self):
                    with self._lock:
                        return len(self._items)
                
                def task_done(self):
                    pass
                
                def get_nowait(self):
                    return self.get(block=False)
                
                def join(self):
                    pass
            
            self.gui_ref._text_update_queue = DummyQueue(maxsize=150)
            self.gui_ref.gui_queue = DummyQueue(maxsize=200)
            print("⚠️ Verwende Dummy-Queues (eingeschränkte Funktionalität)")

    def setup_gui(self):
        self.root.configure(bg=DragonColors.BG_PRIMARY)
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

    def setup_dark_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Dark.TCombobox',
            fieldbackground=ModernColors.COMBO_BG,
            background=ModernColors.COMBO_BG,
            foreground=ModernColors.COMBO_FG,
            selectbackground=ModernColors.COMBO_SELECTION,
            selectforeground=ModernColors.TEXT_PRIMARY,
            insertcolor=ModernColors.TEXT_PRIMARY,
            borderwidth=1,
            relief='flat',
            arrowsize=12,
            padding=5)

        style.map('Dark.TCombobox',
            fieldbackground=[
                ('readonly', ModernColors.COMBO_BG),
                ('active', ModernColors.BG_HOVER)
            ],
            background=[
                ('readonly', ModernColors.COMBO_BG),
                ('active', ModernColors.BG_HOVER)
            ],
            foreground=[
                ('readonly', ModernColors.COMBO_FG),
                ('active', ModernColors.TEXT_PRIMARY)
            ])

        self.root.option_add('*TCombobox*Listbox.background', ModernColors.COMBO_BG)
        self.root.option_add('*TCombobox*Listbox.foreground', ModernColors.COMBO_FG)
        self.root.option_add('*TCombobox*Listbox.selectBackground', ModernColors.COMBO_SELECTION)
        self.root.option_add('*TCombobox*Listbox.selectForeground', ModernColors.TEXT_PRIMARY)

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')

    def create_layout(self):
        header_frame = tk.Frame(self.root, bg=ModernColors.BG_PRIMARY, height=35)
        header_frame.grid(row=0, column=0, sticky='ew', padx=12, pady=8)
        header_frame.grid_propagate(False)

        title_label = tk.Label(header_frame,
                              text="🐉 Dragon Whisperer - Livestream Transcription & Translation",
                              font=ModernFonts.TITLE,
                              bg=ModernColors.BG_PRIMARY,
                              fg=ModernColors.DRAGON_GREEN)
        title_label.pack(side='left')

        self.gui_ref.status_label = tk.Label(header_frame,
                                   text="✅ READY",
                                   font=ModernFonts.PRIMARY,
                                   bg=ModernColors.BG_PRIMARY,
                                   fg=ModernColors.TEXT_SECONDARY)
        self.gui_ref.status_label.pack(side='right')

        self.create_stream_info_display()
        self.gui_ref.stream_info_frame.grid(row=1, column=0, sticky='ew', padx=12, pady=3)

        input_frame = tk.Frame(self.root, bg=ModernColors.BG_PRIMARY)
        input_frame.grid(row=2, column=0, sticky='ew', padx=12, pady=3)

        url_frame = tk.Frame(input_frame, bg=ModernColors.BG_PRIMARY)
        url_frame.pack(fill='x', pady=2)

        tk.Label(url_frame, text="URL:", bg=ModernColors.BG_PRIMARY,
                fg=ModernColors.TEXT_PRIMARY, font=ModernFonts.PRIMARY).pack(side='left')

        self.gui_ref.url_entry = tk.Entry(url_frame, font=ModernFonts.PRIMARY,
                                 bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
                                 insertbackground=ModernColors.TEXT_PRIMARY,
                                 selectbackground=ModernColors.COMBO_SELECTION,
                                 width=60)
        self.gui_ref.url_entry.pack(side='left', fill='x', expand=True, padx=(5, 5))
        self.gui_ref.url_entry.insert(0, self.gui_ref.settings.last_url)
        DarkEntryContextMenu(self.gui_ref.url_entry)

        self.gui_ref.language_info_label = tk.Label(
            url_frame,
            text="",
            font=ModernFonts.PRIMARY,
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.TEXT_ACCENT
        )
        self.gui_ref.language_info_label.pack(side='right', padx=(5, 0))
        self.create_compact_control_panel(input_frame)
        self.setup_status_bar()
        self.gui_ref.status_bar_frame.grid(row=4, column=0, sticky='ew', pady=(2, 0))
        self.create_text_areas()
        self.gui_ref.text_container.grid(row=3, column=0, sticky='nsew', padx=12, pady=8)
        self.gui_ref.url_entry.bind('<KeyRelease>', self.gui_ref.on_url_change)
        self.gui_ref.url_entry.bind('<FocusOut>', self.gui_ref.on_url_change)

    def create_stream_info_display(self):
        self.gui_ref.stream_info_frame = tk.Frame(self.root, bg=ModernColors.BG_SECONDARY, height=50)
        self.gui_ref.stream_info_frame.grid_propagate(True)
        self.gui_ref.stream_title_label = tk.Label(
            self.gui_ref.stream_info_frame,
            text="📡 No active stream",
            font=ModernFonts.SUBTITLE,
            bg=ModernColors.BG_SECONDARY,
            fg=ModernColors.TEXT_ACCENT,
            wraplength=700,
            justify='left'
        )
        self.gui_ref.stream_title_label.pack(fill='x', padx=8, pady=(6, 2))

        self.gui_ref.stream_details_label = tk.Label(
            self.gui_ref.stream_info_frame,
            text="Ready to connect...",
            font=ModernFonts.PRIMARY,
            bg=ModernColors.BG_SECONDARY,
            fg=ModernColors.TEXT_SECONDARY,
            justify='left'
    )
        self.gui_ref.stream_details_label.pack(fill='x', padx=8, pady=(2, 6))

    def create_compact_control_panel(self, parent):
        control_frame = tk.Frame(parent, bg=ModernColors.BG_PRIMARY)
        control_frame.pack(fill='x', pady=8)

        left_controls = tk.Frame(control_frame, bg=ModernColors.BG_PRIMARY)
        left_controls.pack(side='left')

        action_buttons = [
            ("📁", self.gui_ref.select_file_dark, "Select file"),
            ("📋", self.gui_ref.paste_url, "Paste URL"),
        ]

        for icon, command, tooltip in action_buttons:
            btn = tk.Button(left_controls, text=icon, command=command,
                          bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
                          relief='flat', bd=0, font=("Segoe UI", 9),
                          cursor='hand2')
            btn.pack(side='left', padx=1)

        self.gui_ref.layout_btn = tk.Button(left_controls, text="🔄", command=self.gui_ref.toggle_layout,
                               bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
                               relief='flat', bd=0, font=("Segoe UI", 9),
                               cursor='hand2')
        self.gui_ref.layout_btn.pack(side='left', padx=5)

        center_controls = tk.Frame(control_frame, bg=ModernColors.BG_PRIMARY)
        center_controls.pack(side='left', padx=15)

        model_frame = tk.Frame(center_controls, bg=ModernColors.BG_PRIMARY)
        model_frame.pack(side='left', padx=5)

        tk.Label(model_frame, text="Model:", bg=ModernColors.BG_PRIMARY,
                fg=ModernColors.TEXT_SECONDARY, font=ModernFonts.PRIMARY).pack(side='left')

        self.gui_ref.model_var = tk.StringVar(value=self.gui_ref.settings.default_model)
        self.gui_ref.model_combo = ttk.Combobox(model_frame, textvariable=self.gui_ref.model_var,
                                  values=WHISPER_MODELS,
                                  width=10,
                                  style='Dark.TCombobox',
                                  state="readonly")
        self.gui_ref.model_combo.pack(side='left', padx=3)

        lang_frame = tk.Frame(center_controls, bg=ModernColors.BG_PRIMARY)
        lang_frame.pack(side='left', padx=5)

        tk.Label(lang_frame, text="Translate:", bg=ModernColors.BG_PRIMARY,
                fg=ModernColors.TEXT_SECONDARY, font=ModernFonts.PRIMARY).pack(side='left')

        self.gui_ref.lang_var = tk.StringVar()

        language_groups = {
            'Common': ['German', 'English', 'French', 'Spanish', 'Italian'],
            'Asian': ['Japanese', 'Chinese', 'Korean', 'Vietnamese', 'Thai'],
            'More': [name for name, code in SORTED_LANGUAGES
                       if name not in ['German', 'English', 'French', 'Spanish', 'Italian',
                                      'Japanese', 'Chinese', 'Korean', 'Vietnamese', 'Thai']]
        }

        all_languages = []
        for group_name, languages in language_groups.items():
            if languages:
                all_languages.append(f"--- {group_name} ---")
                all_languages.extend(languages)

        self.gui_ref.lang_combo = ttk.Combobox(lang_frame, textvariable=self.gui_ref.lang_var,
                                      values=all_languages,
                                      width=14,
                                      style='Dark.TCombobox',
                                      state="readonly")
        self.gui_ref.lang_combo.pack(side='left', padx=3)

        default_lang_name = SUPPORTED_LANGUAGES.get(self.gui_ref.settings.default_language, "German")
        self.gui_ref.lang_var.set(default_lang_name)

        self.gui_ref.lang_combo.bind('<<ComboboxSelected>>', self.gui_ref.on_language_change)

        right_controls = tk.Frame(control_frame, bg=ModernColors.BG_PRIMARY)
        right_controls.pack(side='right')

        self.gui_ref.start_button = tk.Button(
            right_controls, text="🚀 START", command=self.gui_ref.controller.start_processing,
            bg=ModernColors.SUCCESS, fg=ModernColors.TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"), relief='flat', padx=20
        )
        self.gui_ref.start_button.pack(side='left', padx=2)

        self.gui_ref.stop_button = tk.Button(
            right_controls, text="⏹️ STOP", command=self.gui_ref.controller.stop_processing,
            bg=ModernColors.ERROR, fg=ModernColors.TEXT_PRIMARY, state='disabled',
            font=("Segoe UI", 9, "bold"), relief='flat', padx=20
        )
        self.gui_ref.stop_button.pack(side='left', padx=2)

        self.gui_ref.translate_toggle = tk.BooleanVar(value=True)
        translate_btn = tk.Checkbutton(right_controls,
                                  text="Translate",
                                  variable=self.gui_ref.translate_toggle,
                                  command=self.gui_ref.toggle_translation,
                                  bg=ModernColors.BG_PRIMARY,
                                  fg=ModernColors.TEXT_PRIMARY,
                                  selectcolor=ModernColors.BG_TERTIARY,
                                  activebackground=ModernColors.BG_PRIMARY,
                                  activeforeground=ModernColors.TEXT_PRIMARY,
                                  font=ModernFonts.PRIMARY)
        translate_btn.pack(side='left', padx=5)

        self.gui_ref.subtitle_btn = tk.Button(
            right_controls, text="🎬", command=self.gui_ref.toggle_subtitle_mode,
            bg=ModernColors.SUBTITLE_INACTIVE, fg=ModernColors.TEXT_PRIMARY,
            relief='flat', bd=0, font=("Segoe UI", 9),
            cursor='hand2'
        )
        self.gui_ref.subtitle_btn.pack(side='left', padx=5)
        self.gui_ref.model_combo.bind('<<ComboboxSelected>>', self.gui_ref.on_model_change)

    def create_text_areas(self):
        """Erstellt Text-Bereiche - INTELLIGENT mit Rückgabe"""
    
        layout_changed = False
        current_layout = getattr(self.gui_ref, '_current_layout', None)
    
        if current_layout != self.gui_ref.layout_mode:
            layout_changed = True
            print(f"🔄 Layout change detected: {current_layout} → {self.gui_ref.layout_mode}")
    
        if (hasattr(self.gui_ref, 'text_container') and 
            layout_changed):
        
            try:
                if self.gui_ref.text_container.winfo_exists():
                    print(f"   🗑️ Destroying old container for layout change")
                    self.gui_ref.text_container.destroy()
                    time.sleep(0.02)
            except tk.TclError:
                pass
            except Exception as e:
                print(f"   ⚠️ Container destroy warning: {e}")
    
        if layout_changed or not hasattr(self.gui_ref, 'text_container'):
            self.gui_ref.text_container = tk.Frame(self.root, bg=ModernColors.BG_PRIMARY)
            self.gui_ref._current_layout = self.gui_ref.layout_mode
            print(f"   ✅ New container created for {self.gui_ref.layout_mode} layout")
    
        if self.gui_ref.layout_mode == "horizontal":
            self.create_horizontal_layout()
        else:
            self.create_vertical_layout()
    
        self.gui_ref.text_container.grid(row=3, column=0, sticky='nsew', padx=12, pady=8)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.update_idletasks()
    
        return (
            getattr(self.gui_ref, 'transcript_text', None),
            getattr(self.gui_ref, 'translation_text', None)
        )

    def create_vertical_layout(self):
        main_frame = tk.LabelFrame(self.gui_ref.text_container, text="Live Transkription & Übersetzung",
                                 bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                 font=ModernFonts.SUBTITLE, padx=8, pady=8)
        main_frame.pack(fill='both', expand=True)

        trans_frame = tk.Frame(main_frame, bg=ModernColors.BG_SECONDARY)
        trans_frame.pack(fill='x', pady=(0, 3))

        trans_header = tk.Frame(trans_frame, bg=ModernColors.BG_SECONDARY)
        trans_header.pack(fill='x')

        tk.Label(trans_header, text="🎤 Transkription:",
                bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_ACCENT,
                font=ModernFonts.SUBTITLE).pack(side='left')

        self.gui_ref.transcript_scroll_var = tk.BooleanVar(value=True)
        scroll_cb = tk.Checkbutton(trans_header,
                                  variable=self.gui_ref.transcript_scroll_var,
                                  bg=ModernColors.BG_SECONDARY,
                                  activebackground=ModernColors.BG_SECONDARY,
                                  selectcolor=ModernColors.CHECKBOX_ACTIVE,
                                  fg=ModernColors.TEXT_PRIMARY)
        scroll_cb.pack(side='right', padx=3)

        tk.Label(trans_header, text="Auto-Scroll",
                bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_SECONDARY,
                font=("Segoe UI", 7)).pack(side='right', padx=1)

        self.gui_ref.transcript_text = self.create_text_widget(main_frame, height=6)

        transla_frame = tk.Frame(main_frame, bg=ModernColors.BG_SECONDARY)
        transla_frame.pack(fill='x', pady=(8, 0))

        transla_header = tk.Frame(transla_frame, bg=ModernColors.BG_SECONDARY)
        transla_header.pack(fill='x')

        self.gui_ref.translation_header = tk.Label(transla_header, text="🌐 Übersetzung:",
                                          bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_ACCENT,
                                          font=ModernFonts.SUBTITLE)
        self.gui_ref.translation_header.pack(side='left')

        self.gui_ref.translation_scroll_var = tk.BooleanVar(value=True)
        scroll_cb = tk.Checkbutton(transla_header,
                                  variable=self.gui_ref.translation_scroll_var,
                                  bg=ModernColors.BG_SECONDARY,
                                  activebackground=ModernColors.BG_SECONDARY,
                                  selectcolor=ModernColors.CHECKBOX_ACTIVE,
                                  fg=ModernColors.TEXT_PRIMARY)
        scroll_cb.pack(side='right', padx=3)

        tk.Label(transla_header, text="Auto-Scroll",
                bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_SECONDARY,
                font=("Segoe UI", 7)).pack(side='right', padx=1)

        self.gui_ref.translation_text = self.create_text_widget(main_frame, height=6)

    def create_horizontal_layout(self):
        main_frame = tk.LabelFrame(self.gui_ref.text_container, text="Live Transkription & Übersetzung",
                                 bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                 font=ModernFonts.SUBTITLE, padx=8, pady=8)
        main_frame.pack(fill='both', expand=True)

        self.gui_ref.paned_window = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL,
                                         bg=ModernColors.BG_SECONDARY,
                                         sashrelief='raised', sashwidth=4, sashpad=0)
        self.gui_ref.paned_window.pack(fill='both', expand=True)

        left_frame = tk.Frame(self.gui_ref.paned_window, bg=ModernColors.BG_TERTIARY)
        self.gui_ref.paned_window.add(left_frame, stretch="always", width=400)

        trans_header = tk.Frame(left_frame, bg=ModernColors.BG_TERTIARY)
        trans_header.pack(fill='x', padx=5, pady=2)

        tk.Label(trans_header, text="🎤 Transkription",
                bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_ACCENT,
                font=ModernFonts.SUBTITLE).pack(side='left')

        self.gui_ref.transcript_scroll_var = tk.BooleanVar(value=True)
        scroll_cb = tk.Checkbutton(trans_header,
                                  variable=self.gui_ref.transcript_scroll_var,
                                  bg=ModernColors.BG_TERTIARY,
                                  activebackground=ModernColors.BG_TERTIARY,
                                  selectcolor=ModernColors.CHECKBOX_ACTIVE,
                                  fg=ModernColors.TEXT_PRIMARY)
        scroll_cb.pack(side='right', padx=3)

        tk.Label(trans_header, text="Auto-Scroll",
                bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_SECONDARY,
                font=("Segoe UI", 7)).pack(side='right', padx=1)

        self.gui_ref.transcript_text = self.create_text_widget(left_frame)

        right_frame = tk.Frame(self.gui_ref.paned_window, bg=ModernColors.BG_TERTIARY)
        self.gui_ref.paned_window.add(right_frame, stretch="always", width=400)

        transla_header = tk.Frame(right_frame, bg=ModernColors.BG_TERTIARY)
        transla_header.pack(fill='x', padx=5, pady=2)

        self.gui_ref.translation_header = tk.Label(transla_header, text="🌐 Übersetzung",
                                          bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_ACCENT,
                                          font=ModernFonts.SUBTITLE)
        self.gui_ref.translation_header.pack(side='left')

        self.gui_ref.translation_scroll_var = tk.BooleanVar(value=True)
        scroll_cb = tk.Checkbutton(transla_header,
                                  variable=self.gui_ref.translation_scroll_var,
                                  bg=ModernColors.BG_TERTIARY,
                                  activebackground=ModernColors.BG_TERTIARY,
                                  selectcolor=ModernColors.CHECKBOX_ACTIVE,
                                  fg=ModernColors.TEXT_PRIMARY)
        scroll_cb.pack(side='right', padx=3)

        tk.Label(transla_header, text="Auto-Scroll",
                bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_SECONDARY,
                font=("Segoe UI", 7)).pack(side='right', padx=1)

        self.gui_ref.translation_text = self.create_text_widget(right_frame)
        self.gui_ref.paned_window.paneconfig(left_frame, minsize=250, width=400)
        self.gui_ref.paned_window.paneconfig(right_frame, minsize=250, width=400)
        

    def create_text_widget(self, parent, height=None):
        text_widget = scrolledtext.ScrolledText(
            parent,
            bg=ModernColors.BG_TERTIARY,
            fg=ModernColors.TEXT_PRIMARY,
            font=ModernFonts.MONOSPACE,
            insertbackground=ModernColors.TEXT_PRIMARY,
            wrap=tk.WORD,
            relief='flat',
            selectbackground=ModernColors.COMBO_SELECTION,
            selectforeground=ModernColors.TEXT_PRIMARY,
            maxundo=30,
            undo=True
        )

        if height:
            text_widget.config(height=height)
        text_widget.pack(fill='both', expand=True, padx=5, pady=5)

        DarkContextMenu(text_widget)

        def safe_text_cleanup(event=None):
            try:
                lines = int(text_widget.index('end-1c').split('.')[0])
                if lines > 400:
                    component = 'transcript' if text_widget == self.gui_ref.transcript_text else 'translation'
                    self.gui_ref.memory_manager.clear_component(component)

                    keep_lines = 250
                    delete_to = f'{lines-keep_lines}.0'
                    text_widget.delete('1.0', delete_to)
                    gc.collect()
            except Exception:
                pass

        text_widget.bind('<KeyRelease>', safe_text_cleanup)
        return text_widget

    def setup_status_bar(self):
        """Status-Leiste mit professionellem System-Info Display."""
        self.gui_ref.status_bar_frame = tk.Frame(
            self.root,
            bg=ModernColors.BG_SECONDARY,
            height=50
        )
        self.gui_ref.status_bar_frame.grid_propagate(True)

        separator = tk.Frame(self.gui_ref.status_bar_frame, height=2, bg=ModernColors.DRAGON_GREEN)
        separator.pack(fill='x', side='top')

        main_container = tk.Frame(self.gui_ref.status_bar_frame, bg=ModernColors.BG_SECONDARY)
        main_container.pack(fill='x', expand=True, padx=12, pady=8)

        left_panel = tk.Frame(main_container, bg=ModernColors.BG_SECONDARY)
        left_panel.pack(side='left', fill='x', expand=False)

        quick_actions = [
            ("🗑️", self.gui_ref.clear_all, "Clear all"),
            ("💾", self.gui_ref.save_transcript, "Save"),
            ("📝", self.gui_ref.export_subtitles, "Export"),
            ("📊", self.gui_ref.show_simple_stats, "Stats"),
            ("⚙️", self.gui_ref.show_advanced_settings, "Settings"),
        ]

        for i, (icon, command, tooltip) in enumerate(quick_actions):
            btn = tk.Button(left_panel, text=icon, command=command,
                          bg=ModernColors.BG_TERTIARY,
                          fg=ModernColors.TEXT_PRIMARY,
                          relief='flat',
                          font=("Segoe UI", 9),
                          cursor='hand2',
                          padx=4, pady=2,
                          activebackground=ModernColors.BG_HOVER)
            btn.grid(row=0, column=i, padx=1, sticky='w')

        center_panel = tk.Frame(main_container, bg=ModernColors.BG_SECONDARY)
        center_panel.pack(side='left', fill='x', expand=True, padx=(15, 15))

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
            bg=ModernColors.BG_SECONDARY,
            fg=ModernColors.TEXT_SECONDARY,
            padx=5
        )
        self.gui_ref.system_info_label.pack(fill='x')

        right_panel = tk.Frame(main_container, bg=ModernColors.BG_SECONDARY)
        right_panel.pack(side='right', fill='x', expand=False)

        self.gui_ref.exit_button = tk.Button(
            right_panel,
            text=" ⏻ EXIT ",
            command=self.gui_ref.controller.safe_exit,
            bg="#dc3545",
            fg='white',
            font=("Segoe UI", 9, "bold"),
            relief='raised',
            cursor='hand2',
            padx=12,
            pady=3,
            activebackground='#c82333'
        )
        self.gui_ref.exit_button.pack(side='right')


    def process_batch_text_updates(self):
        """KORRIGIERT: Mit DummyQueue-Kompatibilität"""
        try:

            if (not hasattr(self.gui_ref, '_shutting_down') or 
                getattr(self.gui_ref, '_shutting_down', False)):
                return
    
            if (not hasattr(self, 'root') or 
                self.root is None or 
                not hasattr(self.root, 'winfo_exists') or 
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
                    except:
                        pass
                
                    processed += 1

                except Exception as e:
                    error_name = type(e).__name__
                    error_msg = str(e).lower()
                    if "Empty" in error_name or "empty" in error_msg:
                        break
                    print(f"⚠️ Queue processing error: {e}")
                    break

        except Exception as e:
            print(f"❌ Batch update error: {e}")
        self._schedule_next_update()

    def _process_update(self, update_type, text_data):
        """Verarbeitet einzelnes GUI-Update"""
        try:
            if update_type == 'transcript':
                if (hasattr(self.gui_ref, 'transcript_text') and
                    self.gui_ref.transcript_text is not None and
                    hasattr(self.gui_ref.transcript_text, 'winfo_exists') and
                    self.gui_ref.transcript_text.winfo_exists()):
                    self.gui_ref.transcript_text.insert('end', text_data)
                    self._auto_scroll('transcript')
                    self._check_text_limit('transcript')
        
            elif update_type == 'translation':
                if (hasattr(self.gui_ref, 'translation_text') and
                    self.gui_ref.translation_text is not None and
                    hasattr(self.gui_ref.translation_text, 'winfo_exists') and
                    self.gui_ref.translation_text.winfo_exists()):
                
                    self.gui_ref.translation_text.insert('end', text_data)
                    self._auto_scroll('translation')
                    self._check_text_limit('translation')
                
        except tk.TclError:
            pass
        except Exception as e:
            print(f"⚠️ GUI update error: {e}")

    def _auto_scroll(self, text_type):
        """Automatisches Scrollen wenn aktiviert"""
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
        except:
            pass

    def _check_text_limit(self, text_type):
        """Prüft Text-Limit und kürzt wenn nötig"""
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
        except:
            pass

    def _schedule_next_update(self):
        """Plant nächsten Batch-Update"""
        try:
            if (hasattr(self, 'root') and
                self.root is not None and
                hasattr(self.root, 'winfo_exists') and
                self.root.winfo_exists()):

                interval = 150
                if hasattr(self.gui_ref, '_batch_update_interval'):
                    try:
                        interval = self.gui_ref._batch_update_interval
                    except:
                        pass

                try:
                    if hasattr(self, '_batch_timer_id') and self._batch_timer_id:
                        self.root.after_cancel(self._batch_timer_id)
                except:
                    pass
            
                self._batch_timer_id = self.root.after(interval, self.process_batch_text_updates)
            
        except Exception as e:
            print(f"⚠️ Timer scheduling error: {e}")

    def start_batch_updates(self):
        """Startet die Batch-Update Schleife"""
        try:
            if (hasattr(self, 'root') and
                self.root is not None and
                hasattr(self.root, 'winfo_exists') and
                self.root.winfo_exists()):

                if not hasattr(self.gui_ref, '_text_update_queue') or self.gui_ref._text_update_queue is None:
                    try:
                        self.gui_ref._text_update_queue = queue.Queue(maxsize=150)
                    except:
                        class DummyQueue:
                            def __init__(self, maxsize=0):
                                self.maxsize = maxsize
                                self._items = []
                                self._lock = threading.Lock()
                                class EmptyException(Exception):
                                    pass
                                self.Empty = EmptyException
                            
                            def put(self, item, block=True, timeout=None):
                                with self._lock:
                                    self._items.append(item)
                                    if self.maxsize > 0 and len(self._items) > self.maxsize:
                                        self._items.pop(0)
                            
                            def get(self, block=True, timeout=None):
                                with self._lock:
                                    if self._items:
                                        return self._items.pop(0)
                                    raise self.Empty()
                            
                            def empty(self):
                                with self._lock:
                                    return len(self._items) == 0
                            
                            def qsize(self):
                                with self._lock:
                                    return len(self._items)
                            
                            def task_done(self):
                                pass
                            
                            def get_nowait(self):
                                return self.get(block=False)
                            
                            def join(self):
                                pass
                        
                        self.gui_ref._text_update_queue = DummyQueue(maxsize=150)
                        print("⚠️ Queue-Fallback in start_batch_updates")

                self.root.after(100, self.process_batch_text_updates)
                print("✅ Batch updates gestartet")

        except Exception as e:
            print(f"⚠️ Start batch updates error: {e}")


class WhisperController:
    """
    🚀 ULTIMATE CONTROLLER - GETESTET & FUNKTIONIERT
    """
    def __init__(self, gui_ref, ui_update_fn: Callable = None, status_update_fn: Callable = None):
        self.gui_ref = gui_ref
        self.is_processing = False
        self._cleanup_lock = threading.RLock()
        self._last_transcription_text = ""
        self._duplicate_check_cache = collections.deque(maxlen=20)
        self._processing_thread = None
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

        if not hasattr(self, '_initialized'):
            self._initialized = True

    def _create_default_ui_updater(self):
        def default_updater(component, text):
            try:
                if not text or not text.strip():
                    return

                if component == 'transcript':
                    if hasattr(self.gui_ref, 'transcript_text'):
                        self._append_to_text_widget(self.gui_ref.transcript_text, text)
                    else:
                        print(f"🎤 {text[:100]}...")

                elif component == 'translation':
                    if hasattr(self.gui_ref, 'translation_text'):
                        self._append_to_text_widget(self.gui_ref.translation_text, text)
                    else:
                        print(f"🌐 {text[:100]}...")

            except Exception:
                pass

        return default_updater

    def _create_default_status_updater(self):
        def default_updater(state_info):
            try:
                if 'status' in state_info:
                    status = state_info['status']
                    if hasattr(self.gui_ref, 'status_label'):
                        self._update_status_label(status)
                    else:
                        print(f"📊 {status}")

            except Exception:
                pass

        return default_updater

    def _append_to_text_widget(self, widget, text):
        try:
            if hasattr(self.gui_ref, 'root') and self.gui_ref.root.winfo_exists():
                self.gui_ref.root.after(0, lambda: self._safe_text_insert(widget, text))
        except Exception:
            pass

    def _safe_text_insert(self, widget, text):
        try:
            if widget and widget.winfo_exists():
                widget.insert('end', text)

                if widget == getattr(self.gui_ref, 'transcript_text', None):
                    if hasattr(self.gui_ref, 'transcript_scroll_var'):
                        if self.gui_ref.transcript_scroll_var.get():
                            widget.see('end')
                elif widget == getattr(self.gui_ref, 'translation_text', None):
                    if hasattr(self.gui_ref, 'translation_scroll_var'):
                        if self.gui_ref.translation_scroll_var.get():
                            widget.see('end')

        except (tk.TclError, AttributeError):
            pass

    def _update_status_label(self, text):
        try:
            if hasattr(self.gui_ref, 'root') and self.gui_ref.root.winfo_exists():
                self.gui_ref.root.after(0, lambda: self.gui_ref.status_label.config(
                    text=text[:100] if text else "Ready"
                ))
        except Exception:
            pass

    def _cleanup_resources(self):
        if hasattr(self, '_stop_requested') and self._stop_requested:
            return

        if hasattr(self, '_stop_requested'):
            self._stop_requested = True

        self.is_processing = False

        try:
            if hasattr(self.gui_ref, 'audio_processor'):
                self.gui_ref.audio_processor._processing = False
                if hasattr(self.gui_ref.audio_processor, '_stop_event'):
                    self.gui_ref.audio_processor._stop_event.set()
        except:
            pass

        try:
            if hasattr(self, '_processing_thread') and self._processing_thread:
                if self._processing_thread.is_alive():
                    pass
        except:
            pass

    def _start_processing(self):
        if self.is_processing:
            self.status_update_fn({'status': '⚠️ Bereits aktiv'})
            return

        url = ""
        try:
            url = self.gui_ref.url_entry.get().strip()
        except Exception:
            self.status_update_fn({'status': '❌ URL Fehler'})
            return

        if not url:
            self.status_update_fn({'status': '❌ Bitte URL eingeben'})
            return
        try:
            if url.startswith('file://'):
                file_path = url[7:]
                if not os.path.exists(file_path):
                    self.status_update_fn({'status': '❌ Datei nicht gefunden'})
                    return
            else:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url

                self.gui_ref.url_entry.delete(0, 'end')
                self.gui_ref.url_entry.insert(0, url)

        except Exception:
            self.status_update_fn({'status': '❌ Ungültige URL'})
            return

        self.status_update_fn({'status': '🔍 Analysiere Stream...'})

        try:
            if hasattr(self.gui_ref, 'stream_manager'):
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
                    platform=platform_type
                )

            if stream_info:
                self.status_update_fn({'stream_info': stream_info})
                print(f"📡 Stream: {stream_info.title[:50]}...")

        except Exception as e:
            print(f"⚠️ Stream Info Error: {e}")

        self.status_update_fn({'status': '🎵 Teste Audio-Stream...'})

        stream_test_passed = False
        try:
            if hasattr(self.gui_ref, 'audio_processor'):
                stream_test_passed = self.gui_ref.audio_processor.emergency_diagnosis(url)

                if not stream_test_passed:
                    try:
                        if hasattr(self.gui_ref, 'stream_manager'):
                            audio_url = self.gui_ref.stream_manager.extract_audio_url(url)
                            if audio_url:
                                stream_test_passed = self.gui_ref.audio_processor._test_stream_connection(audio_url)
                    except Exception:
                        pass

        except Exception as e:
            print(f"⚠️ Stream Test Error: {e}")

        if not stream_test_passed:
            self.status_update_fn({'status': '❌ Stream nicht erreichbar'})
            print("❌ Stream Test fehlgeschlagen")
            return

        self.status_update_fn({'status': '🤖 Lade KI-Modell...'})

        model_loaded = False
        try:
            if hasattr(self.gui_ref, 'transcription_engine'):
                model_name = "medium"
                if hasattr(self.gui_ref, 'model_var'):
                    model_name = self.gui_ref.model_var.get()

                model_loaded = self.gui_ref.transcription_engine.load_model(model_name)

                if not model_loaded:
                    print("🔄 Versuche base model...")
                    model_loaded = self.gui_ref.transcription_engine.load_model("base")

        except Exception as e:
            print(f"⚠️ Model Load Error: {e}")

        if not model_loaded:
            self.status_update_fn({'status': '❌ KI-Modell konnte nicht geladen werden'})
            return

        try:
            if hasattr(self.gui_ref, 'translation_engine') and hasattr(self.gui_ref, 'lang_var'):
                selected_name = self.gui_ref.lang_var.get()
                target_lang = "de"

                for name, code in SORTED_LANGUAGES:
                    if name == selected_name:
                        target_lang = code
                        break

                self.gui_ref.translation_engine.set_target_language(target_lang)

                lang_display = LANGUAGE_SHORT_CODES.get(target_lang, target_lang)
                if hasattr(self.gui_ref, 'translation_header'):
                    self.gui_ref.translation_header.config(text=f"🌐 Übersetzung ({lang_display})")

        except Exception as e:
            print(f"⚠️ Translation Setup Error: {e}")
        self.is_processing = True
        if hasattr(self.gui_ref, 'is_processing'):
            self.gui_ref.is_processing = True

        def update_gui_buttons():
            try:
                if hasattr(self.gui_ref, 'start_button'):
                    self.gui_ref.start_button.config(state='disabled')
                if hasattr(self.gui_ref, 'stop_button'):
                    self.gui_ref.stop_button.config(state='normal')
            except Exception:
                pass

        if hasattr(self.gui_ref, 'root') and self.gui_ref.root.winfo_exists():
            self.gui_ref.root.after(0, update_gui_buttons)

        if IS_LINUX and hasattr(self.gui_ref, 'performance_optimizer'):
            self.gui_ref.performance_optimizer.optimize_for_processing()

        self.status_update_fn({
            'processing_state': True,
            'status': '🚀 Starte Transkription...',
            'buttons': {
            'start': 'disabled',
            'stop': 'normal'}
        })

        def transcription_callback(result):
            if not result or not hasattr(result, 'text'):
                return
            try:
                if hasattr(self.gui_ref, 'handle_transcription'):
                    self.gui_ref.handle_transcription(result)
                else:
                    text = f"{result.text}\n"
                    self.ui_update_fn('transcript', text)
            except Exception as e:
                print(f"⚠️ Transcription Callback Error: {e}")

        def translation_callback(result):
            if not result or not hasattr(result, 'translated'):
                return

            try:
                if hasattr(self.gui_ref, 'handle_translation'):
                    self.gui_ref.handle_translation(result)
                else:
                    text = f"{result.translated}\n"
                    self.ui_update_fn('translation', text)

            except Exception as e:
                print(f"⚠️ Translation Callback Error: {e}")

        def info_callback(message):
            try:
                if hasattr(self.gui_ref, 'handle_info'):
                    self.gui_ref.handle_info(message)
                else:
                    self.status_update_fn({'status': f'ℹ️ {message}'})
            except Exception:
                pass

        def error_callback(message):
            try:
                if hasattr(self.gui_ref, 'handle_error'):
                    self.gui_ref.handle_error(message)
                else:
                    self.status_update_fn({'status': f'❌ {message}'})

                self._cleanup_resources()

            except Exception:
                pass

        try:
            if hasattr(self.gui_ref, 'audio_processor'):
                self.gui_ref.audio_processor._processing = True
                self.gui_ref.audio_processor._stop_event.clear()

                processing_thread = threading.Thread(
                    target=lambda: self.gui_ref.audio_processor.start_processing(
                        url=url,
                        transcription_callback=transcription_callback,
                        translation_callback=translation_callback,
                        info_callback=info_callback,
                        error_callback=error_callback
                    ),
                    daemon=True,
                    name="AudioProcessor"
                )

                self._processing_thread = processing_thread
                processing_thread.start()

                self.status_update_fn({'status': '✅ Transkription läuft...'})

            else:
                error_callback("❌ Audio-Processor nicht verfügbar")

        except Exception as e:
            error_msg = f"Start Error: {str(e)[:100]}"
            print(f"❌ Processing Start Error: {e}")
            error_callback(error_msg)

    def start_processing(self):
        def start_thread():
            try:
                self._start_processing()
            except Exception as e:
                print(f"❌ Start Processing Error: {e}")
                self.status_update_fn({'status': f'❌ Start fehlgeschlagen: {str(e)[:50]}'})

        thread = threading.Thread(target=start_thread, daemon=True)
        thread.start()

    def stop_processing(self):
        if hasattr(self, '_stop_requested'):
            self._stop_requested = True

        if IS_LINUX and hasattr(self.gui_ref, 'performance_optimizer'):
            self.gui_ref.performance_optimizer.restore_normal_mode()

        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()
        if hasattr(self, '_processing_active'):
            self._processing_active.clear()
        if hasattr(self.gui_ref, 'is_processing'):
            self.gui_ref.is_processing = False

        def stop_audio_processor():
            try:
                if hasattr(self.gui_ref, 'audio_processor'):
                    ap = self.gui_ref.audio_processor
                    ap._processing = False
                    if hasattr(ap, '_stop_event'):
                        ap._stop_event.set()

                    if hasattr(self.gui_ref, 'ffmpeg_manager'):
                        self.gui_ref.ffmpeg_manager.stop_all_streams()

            except Exception as e:
                print(f"⚠️ Audio Stop Fehler: {e}")

        audio_stop_thread = threading.Thread(target=stop_audio_processor, daemon=True)
        audio_stop_thread.start()

        def update_gui_immediately():
            try:

                if hasattr(self.gui_ref, 'status_label'):
                    self.gui_ref.status_label.config(text="✅ READY for new stream")

                if hasattr(self.gui_ref, 'start_button'):
                    self.gui_ref.start_button.config(state='normal')
                if hasattr(self.gui_ref, 'stop_button'):
                    self.gui_ref.stop_button.config(state='disabled')

                if hasattr(self.gui_ref, 'stream_title_label'):
                    self.gui_ref.stream_title_label.config(text="📡 Kein aktiver Stream")
                if hasattr(self.gui_ref, 'stream_details_label'):
                    self.gui_ref.stream_details_label.config(text="Bereit für neue Verbindung")

                if hasattr(self, 'is_processing'):
                    self.is_processing = False

            except Exception as e:
                print(f"⚠️ GUI Update Fehler: {e}")

        if hasattr(self.gui_ref, 'root') and self.gui_ref.root.winfo_exists():
            self.gui_ref.root.after(0, update_gui_immediately)

        def background_cleanup():
            try:

                if hasattr(self, '_processing_thread') and self._processing_thread:
                    if self._processing_thread.is_alive():
                        print("🔄 Warte auf Processing Thread...")
                        self._processing_thread.join(timeout=1.0)

                try:
                    transcription_cache.clear()
                    translation_cache.clear()
                    audio_cache.clear()
                except:
                    pass

                if hasattr(self, '_stop_requested'):
                    self._stop_requested = False

                if hasattr(self, '_shutdown_event'):
                    self._shutdown_event.clear()

            except Exception as e:
                print(f"⚠️ Cleanup Fehler: {e}")

        cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
        cleanup_thread.start()

    def dispose(self):
        """Gibt Controller-Ressourcen frei."""
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()
        self.stop_processing()
        print("🧹 Controller disposed")

    def safe_exit(self):
        try:
            if hasattr(self.gui_ref, 'exit_button'):
                try:
                    self.gui_ref.exit_button.config(state='disabled', text="⏳...")
                except:
                    pass

            if hasattr(self.gui_ref, '_safe_exit_dialog'):
                self.gui_ref._safe_exit_dialog()
            else:
                self._cleanup_resources()
                import sys
                sys.exit(0)

        except Exception:
            import sys
            sys.exit(0)

class LinuxPerformanceOptimizer:
    """
    🐧 SHUTDOWN
    """

    def __init__(self, gui_ref):
        self.gui = gui_ref
        self.is_processing = False
        self._original_settings = {}
        self._optimization_active = False
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        self._monitoring_lock = threading.RLock()
        self._last_gui_access_time = 0
        self._gui_access_warning_printed = False

    def optimize_for_processing(self):
        """Aktiviert Performance-Optimierungen - SHUTDOWN-SICHER"""
        if not IS_LINUX or self._optimization_active:
            return

        with self._monitoring_lock:
            try:

                if self._shutdown_event.is_set():
                    print("⚠️ Optimize: Shutdown bereits aktiv - überspringe")
                    return

                print("🔧 Activating Linux performance optimizations...")


                gui_available = self._is_gui_available_safe()
                if not gui_available:
                    print("⚠️ Optimize: GUI nicht verfügbar")
                    return

                try:
                    if hasattr(self.gui, 'transcript_text') and self.gui.transcript_text.winfo_exists():
                        self._original_settings['transcript_text'] = {
                            'maxundo': self.gui.transcript_text.cget('maxundo'),
                            'undo': self.gui.transcript_text.cget('undo'),
                            'autoseparators': self.gui.transcript_text.cget('autoseparators')
                        }
                        self.gui.transcript_text.configure(
                            maxundo=5, undo=True, autoseparators=True, height=12
                        )
                        print("  ↪ Transcript text widget optimized")
                except tk.TclError:
                    print("  ⚠️ Transcript widget nicht verfügbar während Optimize")
                except Exception as e:
                    print(f"  ⚠️ Transcript optimization error: {e}")

                try:
                    if hasattr(self.gui, 'translation_text') and self.gui.translation_text.winfo_exists():
                        self._original_settings['translation_text'] = {
                            'maxundo': self.gui.translation_text.cget('maxundo'),
                            'undo': self.gui.translation_text.cget('undo'),
                            'autoseparators': self.gui.translation_text.cget('autoseparators')
                        }
                        self.gui.translation_text.configure(
                            maxundo=5, undo=True, autoseparators=True, height=12
                        )
                        print("  ↪ Translation text widget optimized")
                except tk.TclError:
                    print("  ⚠️ Translation widget nicht verfügbar während Optimize")
                except Exception as e:
                    print(f"  ⚠️ Translation optimization error: {e}")


                if hasattr(self.gui, '_batch_update_interval'):
                    try:
                        self._original_settings['batch_update_interval'] = self.gui._batch_update_interval
                        self.gui._batch_update_interval = 250
                        print(f"  ↪ Batch update interval: {self._original_settings['batch_update_interval']} → 250ms")
                    except (AttributeError, TypeError):
                        pass


                if hasattr(self.gui, 'gui_queue'):
                    try:
                        cleared = 0
                        while self.gui.gui_queue.qsize() > 15 and cleared < 30:
                            try:
                                self.gui.gui_queue.get_nowait()
                                self.gui.gui_queue.task_done()
                                cleared += 1
                            except queue.Empty:
                                break
                        if cleared > 0:
                            print(f"  ↪ GUI queue: {cleared} items cleared")
                    except (AttributeError, RuntimeError):
                        pass

                if hasattr(self.gui, '_last_gui_update_time'):
                    try:
                        self.gui._last_gui_update_time = time.time()
                    except (AttributeError, TypeError):
                        pass

                if hasattr(self.gui, '_min_update_interval'):
                    try:
                        self._original_settings['min_update_interval'] = self.gui._min_update_interval
                        self.gui._min_update_interval = 0.4
                        print(f"  ↪ Min update interval: {self._original_settings.get('min_update_interval', 0.3)} → 0.4s")
                    except (AttributeError, TypeError):
                        pass


                self._apply_linux_specific_optimizations()
                

                if self._is_gui_available_safe() and not self._shutdown_event.is_set():
                    self._start_performance_monitoring()
                
                self._optimization_active = True
                self.is_processing = True

                print("✅ Linux performance optimizations activated")

            except Exception as e:
                print(f"⚠️ Linux optimization failed: {e}")
                self._optimization_active = False

    def _is_gui_available_safe(self):
        """Sichere Prüfung ob GUI verfügbar ist - OHNE Exceptions"""
        try:

            return (
                hasattr(self.gui, 'root') and 
                self.gui.root is not None and 
                hasattr(self.gui.root, 'winfo_exists') and
                self.gui.root.winfo_exists() and
                not getattr(self.gui, '_shutting_down', False)
            )
        except Exception:
            return False

    def _apply_linux_specific_optimizations(self):
        """Linux-spezifische Optimierungen - SHUTDOWN-SICHER"""
        try:
            if not self._is_gui_available_safe():
                return

            if 'DISPLAY' in os.environ:
                display = os.environ['DISPLAY']
                print(f"  ↪ X11 Display: {display}")

                try:
                    if hasattr(self.gui, 'root') and self.gui.root.winfo_exists():
                        self.gui.root.configure(double=0)
                        print("  ↪ X11 double buffering disabled")
                except tk.TclError:
                    pass

            self._optimize_compositor()
            self._optimize_memory_settings()
            
            if self.is_processing and not self._shutdown_event.is_set():
                self._optimize_thread_priorities()

        except Exception as e:
            print(f"⚠️ Linux-specific optimization failed: {e}")

    def _optimize_compositor(self):
        """Compositor-Einstellungen - SICHER"""
        try:
            compositor_detected = False

            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name'].lower()
                    if 'compton' in name or 'picom' in name or 'compiz' in name or 'kwin' in name:
                        compositor_detected = True
                        print(f"  ↪ Compositor detected: {name}")
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if compositor_detected and self._is_gui_available_safe():
                try:
                    self.gui.root.attributes('-type', 'normal')
                    print("  ↪ Window type set to 'normal' for better compositor compatibility")
                except tk.TclError:
                    pass

        except Exception:
            pass

    def _optimize_memory_settings(self):
        """Memory-Einstellungen - SICHER"""
        try:
            import resource

            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_DATA)
            new_soft = min(hard_limit, 1024 * 1024 * 1024)

            if new_soft > soft_limit:
                resource.setrlimit(resource.RLIMIT_DATA, (new_soft, hard_limit))
                print(f"  ↪ Memory soft limit increased: {soft_limit} → {new_soft}")

            soft_fd, hard_fd = resource.getrlimit(resource.RLIMIT_NOFILE)
            new_soft_fd = min(hard_fd, 8192)

            if new_soft_fd > soft_fd:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_fd, hard_fd))
                print(f"  ↪ File descriptor limit increased: {soft_fd} → {new_soft_fd}")

        except Exception as e:
            print(f"⚠️ Memory optimization failed: {e}")

    def _optimize_thread_priorities(self):
        """Thread Prioritäten - MIT SHUTDOWN-PRÜFUNG"""
        if self._shutdown_event.is_set():
            return
            
        try:
            current_thread = threading.current_thread()

            if hasattr(os, 'nice'):
                current_nice = os.nice(0)
                try:
                    os.nice(-5)
                    print(f"  ↪ Thread priority increased: nice {current_nice} → {os.nice(0)}")
                except PermissionError:
                    print("  ↪ Note: Need sudo for thread priority adjustment")

        except Exception as e:
            print(f"⚠️ Thread priority optimization failed: {e}")

    def _start_performance_monitoring(self):
        """Startet Performance Monitoring - SHUTDOWN-SICHER"""
        with self._monitoring_lock:
            if self._shutdown_event.is_set():
                print("⚠️ Monitoring nicht gestartet: shutdown bereits aktiv")
                return
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                print("⚠️ Monitoring thread läuft bereits")
                return

            def monitor_performance():
                """SICHERER Monitoring Thread mit Shutdown-Erkennung"""
                print("🔍 Linux performance monitoring started")
                check_count = 0
                max_checks = 240
                
                while not self._shutdown_event.is_set() and self._optimization_active:
                    try:
                        if self._shutdown_event.is_set():
                            break
                        
                        if not self._is_gui_available_safe():
                            print("⚠️ GUI nicht verfügbar - stoppe monitoring")
                            break
                        
                        if check_count >= max_checks:
                            print("⏰ Monitoring Zeitlimit erreicht - auto-stop")
                            break
                        
                        for i in range(30):
                            if self._shutdown_event.is_set():
                                return
                            time.sleep(1)
                        
                        if self._shutdown_event.is_set():
                            break
                        
                        check_count += 1
                        
                        try:
                            system_load = self._get_system_load()
                            memory_usage = self._get_memory_usage()
                            gui_responsiveness = self._check_gui_responsiveness()
                            
                            if (system_load > 0.8 or memory_usage > 0.85 or 
                                gui_responsiveness > 1.0):
                                self._adjust_optimizations(system_load, memory_usage, 
                                                         gui_responsiveness)
                            
                            if check_count % 12 == 0:
                                self._print_performance_report(system_load, 
                                                              memory_usage, 
                                                              gui_responsiveness)
                                
                        except Exception as e:
                            print(f"⚠️ Performance metrics error: {e}")
                            
                    except Exception as e:
                        print(f"⚠️ Performance monitor loop error: {e}")
                        time.sleep(10)
                
                print("✅ Linux performance monitoring stopped")
                
                with self._monitoring_lock:
                    self._monitoring_thread = None

            self._monitoring_thread = threading.Thread(
                target=monitor_performance,
                daemon=True,
                name="LinuxPerformanceMonitor"
            )
            self._monitoring_thread.start()

    def _get_system_load(self):
        """System Load - SICHER"""
        try:
            load_avg = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return load_avg[0] / cpu_count
        except:
            return 0.0

    def _get_memory_usage(self):
        """Memory Usage - SICHER"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except:
            return 0.0

    def _check_gui_responsiveness(self):
        """GUI Responsiveness - ABSOLUT SHUTDOWN-SICHER"""
        if self._shutdown_event.is_set():
            return 0.0
        
        try:
            if not self._is_gui_available_safe():
                return 0.0
            
            response_time = 0.0
            check_complete = threading.Event()
            
            def safe_gui_check():
                """Sichere GUI-Prüfung in eigenem Thread"""
                nonlocal response_time
                try:
                    if self._shutdown_event.is_set():
                        return
                        
                    start = time.perf_counter()
                    self.gui.root.update_idletasks()
                    response_time = time.perf_counter() - start
                except tk.TclError:

                    response_time = 0.0
                except Exception as e:

                    response_time = 999.0
                finally:
                    check_complete.set()
            
            check_thread = threading.Thread(target=safe_gui_check, daemon=True)
            check_thread.start()
            check_complete.wait(timeout=1.5)
            
            if check_thread.is_alive():
                return 999.0
                
            return response_time
            
        except Exception:
            return 0.0

    def _adjust_optimizations(self, system_load, memory_usage, gui_responsiveness):
        """Optimierungen anpassen - KEINE GUI-ZUGRIFFE BEI SHUTDOWN"""
        if self._shutdown_event.is_set():
            return
        
        adjustments = []
        
        try:
            if not self._is_gui_available_safe():
                return
            
            if memory_usage > 0.85 and hasattr(self.gui, 'gui_queue'):
                try:
                    current_size = self.gui.gui_queue.qsize()
                    if current_size > 20:
                        target = max(5, int(current_size * 0.3))
                        cleared = self._clean_queue_safe(self.gui.gui_queue, target)
                        if cleared > 0:
                            adjustments.append(f"Queue: -{cleared} items")
                except (AttributeError, RuntimeError):
                    pass
            
            if system_load > 0.8 and hasattr(self.gui, '_batch_update_interval'):
                try:
                    current = self.gui._batch_update_interval
                    if current < 500:
                        self.gui._batch_update_interval = min(500, current + 50)
                        adjustments.append(f"Update: {current}→{self.gui._batch_update_interval}ms")
                except (AttributeError, TypeError):
                    pass
            

            
            if adjustments:
                print(f"🔧 Safe adjustments: {', '.join(adjustments)}")
                
        except tk.TclError:
            pass
        except Exception as e:
            print(f"⚠️ Safe optimization adjustment error (non-critical): {e}")

    def _clean_queue_safe(self, queue_obj, target_size):
        """Sichere Queue-Bereinigung - SHUTDOWN-SICHER"""
        if not queue_obj or queue_obj.empty():
            return 0
        
        cleared = 0
        try:
            current_size = queue_obj.qsize()
            if current_size <= target_size:
                return 0
            
            max_remove = min(50, current_size - target_size)
            
            for _ in range(max_remove):
                try:
                    queue_obj.get_nowait()
                    queue_obj.task_done()
                    cleared += 1
                except queue.Empty:
                    break
                except Exception:
                    break
                    
        except Exception:
            pass
            
        return cleared

    def _print_performance_report(self, system_load, memory_usage, gui_responsiveness):
        """Performance Report - SICHER"""
        try:
            stats = {
                'System Load': f"{system_load:.1%}",
                'Memory Usage': f"{memory_usage:.1%}",
                'GUI Response': f"{gui_responsiveness*1000:.0f}ms",
                'Processing': 'Active' if self.is_processing else 'Inactive'
            }

            if hasattr(self.gui, 'gui_queue'):
                try:
                    stats['Queue Size'] = f"{self.gui.gui_queue.qsize()} items"
                except:
                    stats['Queue Size'] = 'N/A'

            report_lines = ["🐧 Linux Performance Report:"]
            for key, value in stats.items():
                report_lines.append(f"  {key:20} {value}")

            print('\n'.join(report_lines))

        except Exception:
            pass

    def restore_normal_mode(self):
        """Stellt normale Einstellungen wieder her - VOLLSTÄNDIG SHUTDOWN-PROOF"""
        if not IS_LINUX:
            return
        
        print("🔧 Linux optimizer: Starting safe shutdown...")
        
        self._shutdown_event.set()
        self._optimization_active = False
        self.is_processing = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            try:
                print(f"  ↪ Stopping monitor thread...")
                self._monitoring_thread.join(timeout=1.0)
            except Exception as e:
                print(f"  ⚠️ Thread join error (OK during shutdown): {e}")
            finally:
                self._monitoring_thread = None
        
        try:
            gui_alive = self._is_gui_available_safe()
            
            if gui_alive:
                print("  ↪ GUI still alive - restoring settings...")
                
                try:
                    if ('transcript_text' in self._original_settings and 
                        hasattr(self.gui, 'transcript_text') and
                        self.gui.transcript_text.winfo_exists()):
                        
                        self.gui.transcript_text.configure(
                            maxundo=self._original_settings['transcript_text']['maxundo'],
                            undo=self._original_settings['transcript_text']['undo'],
                            autoseparators=self._original_settings['transcript_text']['autoseparators'],
                            height=15
                        )
                        print("    ✅ Transcript restored")
                except tk.TclError:
                    print("    ⚠️ Transcript widget destroyed (normal during shutdown)")
                except Exception as e:
                    print(f"    ⚠️ Transcript restore error: {e}")
                
                try:
                    if ('translation_text' in self._original_settings and 
                        hasattr(self.gui, 'translation_text') and
                        self.gui.translation_text.winfo_exists()):
                        
                        self.gui.translation_text.configure(
                            maxundo=self._original_settings['translation_text']['maxundo'],
                            undo=self._original_settings['translation_text']['undo'],
                            autoseparators=self._original_settings['translation_text']['autoseparators'],
                            height=15
                        )
                        print("    ✅ Translation restored")
                except tk.TclError:
                    print("    ⚠️ Translation widget destroyed (normal during shutdown)")
                except Exception as e:
                    print(f"    ⚠️ Translation restore error: {e}")
                
                try:
                    if ('batch_update_interval' in self._original_settings and 
                        hasattr(self.gui, '_batch_update_interval')):
                        
                        self.gui._batch_update_interval = self._original_settings['batch_update_interval']
                        print(f"    ✅ Update interval restored: {self.gui._batch_update_interval}ms")
                except (AttributeError, KeyError):
                    pass
                
            else:
                print("  ↪ GUI already destroyed - skipping GUI restoration")
                
        except Exception as e:
            print(f"  ⚠️ GUI restoration had errors (normal during shutdown): {e}")
        
        try:
            self._original_settings.clear()
            print("  ✅ Settings cleared")
        except:
            pass
        
        try:
            import gc
            gc.collect()
            print("  ✅ Garbage collection done")
        except:
            pass
        
        print("✅ Linux optimizer: Safe shutdown completed")

    def emergency_optimize(self):
        """Notfall-Optimierungen - EXTREM SICHER"""
        if self._shutdown_event.is_set():
            print("⚠️ Emergency optimize: Shutdown bereits aktiv - überspringe")
            return
            
        print("🚨 Applying emergency Linux optimizations...")
        
        try:
            if not self._is_gui_available_safe():
                print("⚠️ Emergency optimize: GUI nicht verfügbar")
                return
            
            if hasattr(self.gui, 'gui_queue'):
                self._clean_queue_safe(self.gui.gui_queue, 3)
            
            if hasattr(self.gui, '_text_update_queue'):
                self._clean_queue_safe(self.gui._text_update_queue, 2)
            
            try:
                if (hasattr(self.gui, 'transcript_text') and 
                    self.gui.transcript_text.winfo_exists()):
                    self.gui.transcript_text.configure(height=6, maxundo=1)
            except tk.TclError:
                pass
            
            try:
                if (hasattr(self.gui, 'translation_text') and 
                    self.gui.translation_text.winfo_exists()):
                    self.gui.translation_text.configure(height=6, maxundo=1)
            except tk.TclError:
                pass
            
            if hasattr(self.gui, '_batch_update_interval'):
                try:
                    self.gui._batch_update_interval = 500
                except (AttributeError, TypeError):
                    pass
            
            import gc
            gc.collect()
            
            print("✅ Emergency optimizations applied")
            
        except Exception as e:
            print(f"⚠️ Emergency optimization failed: {e}")

    def get_optimization_status(self):
        """Optimierungs-Status - SICHER"""
        thread_alive = False
        thread_id = None
        
        if self._monitoring_thread:
            thread_alive = self._monitoring_thread.is_alive()
            thread_id = getattr(self._monitoring_thread, 'ident', None)
        
        return {
            'platform': SYSTEM,
            'optimization_active': self._optimization_active,
            'processing_active': self.is_processing,
            'monitoring_active': thread_alive,
            'monitoring_thread_id': thread_id,
            'shutdown_event_set': self._shutdown_event.is_set(),
            'original_settings_count': len(self._original_settings),
            'linux_specific': IS_LINUX,
            'gui_available': self._is_gui_available_safe()
        }

    def print_optimization_info(self):
        """Optimierungs-Info - SICHER"""
        status = self.get_optimization_status()
        
        print("\n" + "="*60)
        print("🐧 LINUX PERFORMANCE OPTIMIZER - SHUTDOWN-SICHER")
        print("="*60)
        
        for key, value in status.items():
            print(f"{key:35} {value}")
        
        gui_status = "❌ N/A"
        try:
            gui_status = "✅" if self._is_gui_available_safe() else "❌ Destroyed"
        except:
            pass
        print(f"{'GUI Status':35} {gui_status}")
        
        if hasattr(self.gui, '_batch_update_interval'):
            try:
                print(f"{'Batch Update Interval':35} {self.gui._batch_update_interval}ms")
            except:
                print(f"{'Batch Update Interval':35} N/A")
        
        print("="*60)

    def dispose(self):
        """Gibt alle Ressourcen frei - ULTIMATIVE SICHERHEIT"""
        print("🧹 Linux Performance Optimizer dispose...")
        
        self._shutdown_event.set()
        
        try:
            self.restore_normal_mode()
        except Exception as e:
            print(f"⚠️ restore_normal_mode hatte Fehler (OK): {e}")
        
        try:
            self._original_settings.clear()
        except:
            pass
        
        try:
            import gc
            gc.collect()
        except:
            pass
        
        print("✅ Linux Performance Optimizer disposed")

    def emergency_shutdown(self):
        """EXTERNER NOTFALL-SHUTDOWN (für DragonWhispererGUI)"""
        print("🚨 EXTERNAL EMERGENCY SHUTDOWN für Linux Optimizer")
        
        self._shutdown_event.set()
        self._optimization_active = False
        self.is_processing = False
        
        if hasattr(self, '_monitoring_thread'):
            self._monitoring_thread = None
        
        if hasattr(self, '_original_settings'):
            self._original_settings.clear()
        
        print("✅ External emergency shutdown completed")


class DragonWhispererGUI:
    """🐉 ULTIMATIV STABILE VERSION - Komplett optimiert für sauberes Shutdown"""
    
    def __init__(self):
        self._gui_update_limiter = RateLimiter(max_updates_per_second=30)
        self._shutting_down = False
        self._exit_dialog_active = False
        self.is_processing = False
        self.subtitle_mode = False
        self.current_stream_info = None
        self.current_video_language = None
        self.exit_confirmed = False
        
        if not GUI_AVAILABLE:
            print("❌ Tkinter nicht verfügbar. Versuche Fallback...")
            self._try_fallback_gui()
            return

        try:
            self.settings = AppSettings.load_from_file()
            self.advanced_settings = AdvancedSettings.load_from_file()
            self.advanced_settings.repair()
            
            validation_issues = self.advanced_settings.validate()
            if validation_issues:
                print(f"⚠️ Settings validation issues: {validation_issues}")
                
            print(f"✅ AdvancedSettings ready:")
            print(f"   SAMPLE_RATE: {self.advanced_settings.SAMPLE_RATE}")
            print(f"   CHANNELS: {self.advanced_settings.CHANNELS}")
            print(f"   CHUNK_SIZE: {self.advanced_settings.CHUNK_SIZE_BYTES} bytes")
            
            print(f"\n🔍 [DEBUG TRY-BLOCK] AdvancedSettings AFTER load_from_file:")
            print(f"   Type: {type(self.advanced_settings)}")
            print(f"   Has SAMPLE_RATE: {hasattr(self.advanced_settings, 'SAMPLE_RATE')}")
            print(f"   Has sample_rate: {hasattr(self.advanced_settings, 'sample_rate')}")
    
            if not hasattr(self.advanced_settings, 'SAMPLE_RATE'):
                print(f"🚨 QUICK FIX NEEDED: Adding SAMPLE_RATE to loaded settings")
                self.advanced_settings.SAMPLE_RATE = 16000
            
        except Exception:
            self.settings = AppSettings()
            self.advanced_settings = AdvancedSettings()
            print(f"\n🔍 [DEBUG DRAGONWHISPERER] AdvancedSettings Analysis:")
            print(f"   Type: {type(self.advanced_settings)}")
            print(f"   Has SAMPLE_RATE: {hasattr(self.advanced_settings, 'SAMPLE_RATE')}")
            print(f"   Has sample_rate: {hasattr(self.advanced_settings, 'sample_rate')}")
            print(f"   Has audio_sample_rate: {hasattr(self.advanced_settings, 'audio_sample_rate')}")
            
            print(f"\n   Searching for audio-related attributes:")

        self.layout_mode = getattr(self.settings, 'layout_mode', 'vertical')
        self.current_language = getattr(self.settings, 'default_language', 'de')
        self._translation_reset_counter = 0
        self.current_stream_info = None
        self.progress_dialog = None
        self.current_video_language = None
        self.exit_confirmed = False
        self._exit_dialog_active = False
        self._shutting_down = False

        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except Exception as e:
            raise RuntimeError(f"Tkinter Fehler: {e}")

        self._batch_update_interval = 150
        self._last_batch_update = 0
        self.is_processing = False
        self.subtitle_mode = False
        self._processing_lock = threading.Lock()
        self.layout_mode = getattr(self.settings, 'layout_mode', 'vertical')
        self.transcript_history = collections.deque(maxlen=1000)
        self.translation_history = collections.deque(maxlen=500)
        self._last_transcription_text = ""
        self._last_translation_text = ""
        self.performance_monitor = SimplePerformanceTracker()
        self._last_gui_update_time = 0
        self.gui_queue = queue.Queue(maxsize=200)
        self._text_update_queue = queue.Queue(maxsize=150)
        
        try:
            self.controller = WhisperController(gui_ref=self)
        except Exception as e:
            print(f"❌ Controller Fehler: {e}")
            self._show_error_and_exit(f"Controller Fehler: {e}")
            return

        try:
            self.layout = WhisperLayoutManager(gui_ref=self)
        except Exception as e:
            print(f"❌ Layout Fehler: {e}")
            self._show_error_and_exit(f"Layout Fehler: {e}")
            return

        try:
            self.stream_manager = StreamManager()
            self.ffmpeg_manager = ExcellenceFFmpegManager(self.advanced_settings)
            self.transcription_engine = ExcellenceTranscriptionEngine(self.advanced_settings)
            self.translation_engine = ExcellenceTranslationEngine(
                self.current_language,
                self.advanced_settings
            )
            self.audio_processor = ExcellenceAudioProcessor(
                controller_ref=self.controller,
                ffmpeg_manager=None,
                advanced_settings=self.advanced_settings
            )
            self.audio_processor.set_engines(
                self.transcription_engine,
                self.translation_engine
            )
            self.export_manager = ExportManager()
            self.language_detector = LanguageDetector(self.transcription_engine)
            self.resource_manager = ResourceManager()
            self.memory_manager = ExcellenceMemoryManager()

            if IS_LINUX:
                self.performance_optimizer = LinuxPerformanceOptimizer(gui_ref=self)

            self._register_signal_handlers()

        except Exception as e:
            print(f"❌ Engine Initialisierung Fehler: {e}")
            self._show_error_and_exit(f"Engine Fehler: {e}")
            return

        try:
            self.layout.setup_gui()
            self._setup_callbacks()
            self.root.after(100, self._start_gui_updaters)
            self.root.deiconify()
            self.root.title("🐉 Dragon Whisperer")

        except Exception as e:
            print(f"❌ GUI Setup Fehler: {e}")
            self._show_error_and_exit(f"GUI konnte nicht erstellt werden: {e}")
            return

        self.root.after(1000, self._start_system_monitoring)
        self.root.after(2000, self._final_initialization_check)

    def _register_signal_handlers(self):
        """Registriert alle Cleanup-Handler beim SignalHandler - OPTIMIERT UND KORRIGIERT"""
        try:
            print("🔧 Registering cleanup handlers with SignalHandler...")
            
            if not hasattr(SignalHandler, '_cleanup_operations') or SignalHandler._cleanup_operations is None:
                print("⚠️ SignalHandler._cleanup_operations is None - initializing...")
                SignalHandler._instance = SignalHandler()
        
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
                print(f"   ✅ Registered FFmpegManager cleanup")
            else:
                print(f"   ℹ️ No FFmpegManager to register (normal)")
        
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
                    print(f"   ✅ Registered GPU cleanup")
            except ImportError:
                pass
        
            count = sum(len(ops) for ops in SignalHandler._cleanup_operations.values())
            print(f"✅ Registered {count} cleanup handlers")
        
            # 🔥 WICHTIG: SignalHandler selbst initialisieren
            try:
                SignalHandler.setup(verbose=False, silent=True)
            except:
                pass
        
        except Exception as e:
            print(f"⚠️ SignalHandler registration error: {e}")

    def _safe_stop_all_processes(self):
        """Sicher alle Prozesse stoppen - KEINE GUI-Zugriffe!"""
        print("🛑 Safely stopping all processes...")
        
        self._shutting_down = True
        self.is_processing = False
        
        if hasattr(self, 'controller'):
            try:
                self.controller._shutdown_event.set()
                self.controller._stop_requested = True
                if hasattr(self.controller, '_processing_active'):
                    self.controller._processing_active.clear()
            except Exception as e:
                print(f"⚠️ Controller stop error: {e}")
        
        if hasattr(self, 'audio_processor'):
            try:
                self.audio_processor._processing = False
                if hasattr(self.audio_processor, '_stop_event'):
                    self.audio_processor._stop_event.set()
            except Exception as e:
                print(f"⚠️ Audio processor stop error: {e}")
        
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
                                    os.kill(process.pid, signal.SIGTERM)
                                    time.sleep(0.1)
                                except:
                                    try:
                                        os.kill(process.pid, signal.SIGKILL)
                                    except:
                                        pass
                        except:
                            pass
            except Exception as e:
                print(f"⚠️ FFmpeg stop error: {e}")
        
        print("✅ All processes stopped")

    def _safe_linux_optimizer_cleanup(self):
        """Sichere Linux Optimizer Cleanup mit GUI-Prüfung"""
        if not IS_LINUX or not hasattr(self, 'performance_optimizer'):
            return
            
        print("🐧 Safe Linux optimizer cleanup...")
        
        try:
            gui_exists = False
            try:
                if hasattr(self, 'root') and self.root.winfo_exists():
                    gui_exists = True
            except:
                gui_exists = False
            
            if gui_exists:
                try:
                    self.performance_optimizer.restore_normal_mode()
                except Exception as e:
                    print(f"⚠️ restore_normal_mode failed: {e}")
                    # Fallback: dispose()
                    try:
                        self.performance_optimizer.dispose()
                    except:
                        pass
            else:
                try:
                    self.performance_optimizer.dispose()
                except Exception as e:
                    print(f"⚠️ dispose failed: {e}")
                    
        except Exception as e:
            print(f"⚠️ Linux optimizer cleanup error: {e}")

    def _cleanup_queues(self):
        """Queues sicher leeren"""
        print("🗑️ Cleaning up queues...")
        
        try:
            if hasattr(self, 'gui_queue'):
                count = 0
                while not self.gui_queue.empty() and count < 100:
                    try:
                        self.gui_queue.get_nowait()
                        count += 1
                    except:
                        break
                if count > 0:
                    print(f"  Cleared GUI queue: {count} items")
        except Exception as e:
            print(f"⚠️ GUI queue cleanup error: {e}")
        
        try:
            if hasattr(self, '_text_update_queue'):
                count = 0
                while not self._text_update_queue.empty() and count < 100:
                    try:
                        self._text_update_queue.get_nowait()
                        count += 1
                    except:
                        break
                if count > 0:
                    print(f"  Cleared text queue: {count} items")
        except Exception as e:
            print(f"⚠️ Text queue cleanup error: {e}")

    def _safe_exit_dialog(self):
        """Sicherer Exit-Dialog MIT Bestätigung - REPARIERTE Version"""
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
                print("✅ User confirmed exit - shutting down...")
                self._direct_shutdown()
            else:
                print("↩️ Exit cancelled by user")
                self._exit_dialog_active = False
            
        except tk.TclError:
            print("⚠️ GUI destroyed, performing direct shutdown...")
            self._direct_shutdown()
        except Exception as e:
            print(f"⚠️ Exit dialog error: {e}")
            self._direct_shutdown()
        finally:
            if not self._shutting_down:
                self._exit_dialog_active = False

    def _direct_shutdown(self):
        """Direkter, sicherer Shutdown - REPARIERT"""
        if self._shutting_down:
            print("⚠️ Shutdown already in progress, skipping...")
            return
        
        print("🔧 Performing confirmed shutdown...")
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
            print(f"⚠️ GUI shutdown error: {e}")
            
    def run(self):
        """Haupt-GUI-Loop - MIT Dialog-Funktionalität"""
        try:
            self.root.title("🐉 Dragon Whisperer")
            self._shutting_down = False
            self._exit_dialog_active = False
            self.root.protocol("WM_DELETE_WINDOW", self._safe_exit_dialog)
        
            if hasattr(self, 'exit_button'):
                self.exit_button.config(command=self._safe_exit_dialog)
        
            print("🚀 Starting Dragon Whisperer (with exit confirmation)...")
        
            if not SignalHandler._setup_complete:
                try:
                    SignalHandler.setup(verbose=False, silent=True)
                except:
                    pass
        
            self.root.mainloop()
            print("✅ Main loop exited normally")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user - showing exit dialog...")
            self._safe_exit_dialog()
        
        except SystemExit:
            print("\n🔧 System exit requested")
            raise
        
        except Exception as e:
            print(f"💥 Critical error: {type(e).__name__}: {e}")
            self._direct_shutdown()

    def _post_mainloop_cleanup(self):
        """Post-mainloop Cleanup ohne GUI-Zugriffe"""
        print("🧹 Post-mainloop cleanup (no GUI access)...")
        

        self._shutting_down = True
        self.is_processing = False
        
        self._cleanup_queues()
        
        try:
            transcription_cache.clear()
            translation_cache.clear()
            audio_cache.clear()
        except Exception as e:
            print(f"⚠️ Cache cleanup error: {e}")
        
        import gc
        gc.collect()
        
        print("✅ Post-mainloop cleanup completed")

    def _minimal_emergency_cleanup(self):
        """Minimaler Emergency-Cleanup - NUR für kritische Fehler"""
        print("🆘 MINIMAL emergency cleanup...")
        

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
                                    os.kill(process.pid, signal.SIGKILL)
                                except:
                                    pass
                        except:
                            pass
        except:
            pass
        
        import gc
        gc.collect()

    def safe_controller_stop(self):
        """SICHERER Controller-Stop - KEINE GUI-Fehler bei Shutdown"""
        if self._shutting_down:
            return
            
        print("🛑 Safe controller stop...")
        self._safe_stop_all_processes()

    def _cleanup_queue(self, queue_obj, max_size):
        """QUEUE OVERFLOW PROTECTION - REPARIERT"""
        if not queue_obj or queue_obj.empty():
            return

        try:
            current_size = queue_obj.qsize()
            if current_size <= max_size:
                return

            temp_items = []
            priority_items = []

            while not queue_obj.empty():
                try:
                    item = queue_obj.get_nowait()
                    if isinstance(item, tuple) and len(item) == 2:
                        msg_type, _ = item
                        if msg_type in ['status', 'error']:
                            priority_items.append(item)
                        else:
                            temp_items.append(item)
                    else:
                        temp_items.append(item)

                    queue_obj.task_done()
                except queue.Empty:
                    break

            all_items = priority_items + temp_items
            kept_items = all_items[-max_size:] if len(all_items) > max_size else all_items

            for item in kept_items:
                try:
                    queue_obj.put_nowait(item)
                except queue.Full:
                    break

            cleared = current_size - queue_obj.qsize()
            if cleared > 0:
                print(f"🧹 Queue cleaned: removed {cleared} items")

        except Exception as e:
            print(f"⚠️ Queue cleanup error: {e}")

    @excellence_gui_operation
    def select_file_dark(self):
        try:
            filename = filedialog.askopenfilename(
                title="🎬 Select Audio/Video File - Dragon Whisperer",
                filetypes=[
                    ("Media files", "*.mp3 *.wav *.m4a *.mp4 *.avi *.mkv *.mov *.flac"),
                    ("All files", "*.*")
                ]
            )

            if filename:
                file_url = f"file://{filename}"
                self.url_entry.delete(0, 'end')
                self.url_entry.insert(0, file_url)
                self.update_status(f"📁 File selected: {os.path.basename(filename)}")

                def async_language_detection():
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
    def paste_url(self):
        try:
            clipboard = self.root.clipboard_get().strip()
            if clipboard:
                cleaned_url = self.clean_and_validate_url(clipboard)
                self.url_entry.delete(0, 'end')
                self.url_entry.insert(0, cleaned_url)
                self.update_status("📋 URL pasted")

                if cleaned_url.startswith('file://'):
                    file_path = cleaned_url[7:]
                    if os.path.exists(file_path):
                        def async_detection():
                            try:
                                self.analyze_video_language(file_path)
                            except Exception:
                                pass

                        detection_thread = threading.Thread(target=async_detection, daemon=True)
                        if hasattr(self, 'resource_manager'):
                            self.resource_manager.register_thread(detection_thread)
                        detection_thread.start()
        except ValueError as e:
            self.update_status(f"❌ Invalid URL: {e}")
        except Exception:
            self.update_status("❌ No URL in clipboard")

    def clean_and_validate_url(self, url):
        url = url.strip()

        if not url:
            raise ValueError("URL cannot be empty")

        if url.startswith('file://'):
            file_path = url[7:]
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            return url

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        if len(url) < 10:
            raise ValueError("URL too short")

        if ' ' in url:
            raise ValueError("URL cannot contain spaces")

        return url

    def analyze_video_language(self, file_path: str):
        def language_detection_worker():
            try:
                if hasattr(self, 'language_info_label'):
                    self.language_info_label.config(text="🔍 Analyzing...")

                detection_result = self.language_detector.detect_video_language(file_path)

                def update_result():
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
                def update_error():
                    if hasattr(self, 'language_info_label'):
                        self.language_info_label.config(text="❌ Analysis failed")

                if hasattr(self, 'root'):
                    self.root.after(0, update_error)

        detection_thread = threading.Thread(target=language_detection_worker, daemon=True)
        if hasattr(self, 'resource_manager'):
            self.resource_manager.register_thread(detection_thread)
        detection_thread.start()

    def on_url_change(self, event=None):
        if not hasattr(self, 'url_entry'):
            return

        url = self.url_entry.get().strip()
        if url.startswith('file://'):
            file_path = url[7:]
            if os.path.exists(file_path):
                def async_detection():
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

    def on_language_change(self, event=None):
        try:
            selected_name = self.lang_var.get()
            lang_code = None
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

    def on_model_change(self, event=None):
        if not hasattr(self, 'model_var'):
            return

        new_model = self.model_var.get()

        if not hasattr(self, 'transcription_engine'):
            return

        current_model = self.transcription_engine.get_current_model()

        if new_model == current_model:
            return

        if self.transcription_engine.is_model_loading():
            self.update_status("🔄 Model loading...")
            return

        def reload_model_thread():
            self.update_status(f"🔄 Switching model: {new_model}")

            success = self.transcription_engine.reload_model(new_model)

            if success:
                self.update_status(f"✅ Model: {new_model}")
                if hasattr(self, 'settings'):
                    self.settings.default_model = new_model
            else:
                self.update_status("❌ Model switch failed")
                if hasattr(self, 'model_var'):
                    self.model_var.set(current_model)

        reload_thread = threading.Thread(target=reload_model_thread, daemon=True)
        if hasattr(self, 'resource_manager'):
            self.resource_manager.register_thread(reload_thread)
        reload_thread.start()

    def toggle_translation(self):
        if hasattr(self, 'translate_toggle') and self.translate_toggle.get():
            if hasattr(self, 'translation_engine'):
                self.translation_engine.set_target_language(self.current_language)
            self.update_status("✅ Translation active")
        else:
            self.update_status("❌ Translation inactive")

    def toggle_subtitle_mode(self):
        self.subtitle_mode = not self.subtitle_mode

        if hasattr(self, 'audio_processor'):
            self.audio_processor.enable_subtitle_mode(self.subtitle_mode)

        if hasattr(self, 'subtitle_btn'):
            if self.subtitle_mode:
                self.subtitle_btn.config(bg=DragonColors.SUBTITLE_ACTIVE, fg=DragonColors.TEXT_PRIMARY)
                self.update_status("🎬 SUBTITLE MODE: Timestamps activated")
            else:
                self.subtitle_btn.config(bg=DragonColors.SUBTITLE_INACTIVE, fg=DragonColors.TEXT_PRIMARY)
                self.update_status("📝 NORMAL MODE: Continuous text")

    def toggle_layout(self):
        """Umschalten zwischen Layouts - KOMPLETT REPARIERT"""
        try:
            print(f"🔄 Starting layout toggle from {self.layout_mode}")

            old_transcript = ""
            old_translation = ""

            try:
                if hasattr(self, 'transcript_text') and self.transcript_text:
                    old_transcript = self.transcript_text.get('1.0', 'end-1c')
                    print(f"  📝 Saved transcript: {len(old_transcript)} chars")
            except (tk.TclError, AttributeError) as e:
                print(f"  ⚠️ Could not save transcript: {e}")

            try:
                if hasattr(self, 'translation_text') and self.translation_text:
                    old_translation = self.translation_text.get('1.0', 'end-1c')
                    print(f"  📝 Saved translation: {len(old_translation)} chars")
            except (tk.TclError, AttributeError) as e:
                print(f"  ⚠️ Could not save translation: {e}")

            if self.layout_mode == "vertical":
                self.layout_mode = "horizontal"
                new_mode_text = "Horizontal"
            else:
                self.layout_mode = "vertical"
                new_mode_text = "Vertical"

            print(f"  🔄 Switching to: {self.layout_mode}")

            if hasattr(self, 'settings'):
                self.settings.layout_mode = self.layout_mode
                try:
                    self.settings.save_to_file()
                except Exception as e:
                    print(f"  ⚠️ Settings save error: {e}")

            self.update_status(f"🔄 Switching to {new_mode_text} layout...")

            if hasattr(self, 'layout'):
                new_transcript, new_translation = self.layout.create_text_areas()

                if new_transcript and old_transcript:
                    try:
                        new_transcript.insert('1.0', old_transcript)
                        print(f"  ✅ Restored transcript to new widget")
                    except Exception as e:
                        print(f"  ❌ Failed to restore transcript: {e}")

                if new_translation and old_translation:
                    try:
                        new_translation.insert('1.0', old_translation)
                        print(f"  ✅ Restored translation to new widget")
                    except Exception as e:
                        print(f"  ❌ Failed to restore translation: {e}")

            self.update_status(f"✅ {new_mode_text} layout active")
            print(f"✅ Layout toggle completed successfully")

        except Exception as e:
            print(f"❌ CRITICAL Layout toggle error: {e}")
            import traceback
            traceback.print_exc()
            self.update_status("❌ Layout change failed")

            try:
                self.layout_mode = "vertical"
                if hasattr(self, 'layout'):
                    self.layout.create_text_areas()
            except:
                pass

    def clear_all(self):
        try:
            if hasattr(self, 'transcript_text'):
                self.transcript_text.delete('1.0', 'end')
            if hasattr(self, 'translation_text'):
                self.translation_text.delete('1.0', 'end')
        except:
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

    def save_transcript(self):
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
                    ("All files", "*.*")
                ]
            )
            if not filename:
                return

            file_ext = Path(filename).suffix.lower()

            success = False
            if file_ext == '.srt':
                success = self.export_manager.export_subtitles(self.transcript_history, None, 'srt', filename)
            elif file_ext == '.vtt':
                success = self.export_manager.export_subtitles(self.transcript_history, None, 'vtt', filename)
            elif file_ext == '.json':
                success = self.export_manager.export_json(self.transcript_history, self.translation_history, filename)
            elif file_ext == '.docx':
                success = self.export_manager.export_docx(self.transcript_history, filename)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    if self.current_stream_info:
                        f.write(f"=== STREAM INFORMATION ===\n")
                        f.write(f"Title: {self.current_stream_info.title}\n")
                        f.write(f"Uploader: {self.current_stream_info.uploader}\n")
                        f.write(f"Duration: {self.current_stream_info.duration}\n")
                        f.write(f"Platform: {self.current_stream_info.platform}\n")
                        f.write(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    f.write("=== TRANSCRIPT ===\n")
                    if hasattr(self, 'transcript_text'):
                        f.write(self.transcript_text.get('1.0', 'end-1c'))
                    f.write("\n\n=== TRANSLATION ===\n")
                    if hasattr(self, 'translation_text'):
                        f.write(self.translation_text.get('1.0', 'end-1c'))
                success = True

            if success:
                self.update_status(f"💾 Saved: {os.path.basename(filename)}")
            else:
                self.update_status("❌ Export failed")

        except Exception as e:
            self.update_status(f"❌ Save failed: {e}")

    def export_subtitles(self):
        if not hasattr(self, 'audio_processor') or not self.audio_processor.timed_transcriptions:
            DarkMessageBox.showinfo("WARNING",
                "No subtitle data available.\n\n"
                "Tip: First activate '🎬 Subtitle mode' "
                "and start a transcription.", self.root)
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".srt",
                filetypes=[
                    ("SRT subtitles", "*.srt"),
                    ("VTT subtitles", "*.vtt"),
                    ("All files", "*.*")
                ],
                title="Export subtitles"
            )
            if not filename:
                return

            file_ext = Path(filename).suffix.lower()

            format_type = 'srt' if file_ext == '.srt' else 'vtt'

            success = self.export_manager.export_subtitles(
                self.audio_processor.timed_transcriptions,
                self.audio_processor.timed_translations,
                format=format_type,
                filename=filename
            )

            if success:
                segment_count = len(self.audio_processor.timed_transcriptions)
                translation_count = len(self.audio_processor.timed_translations)

                self.update_status(f"📝 {format_type.upper()} exported: {os.path.basename(filename)}")

                DarkMessageBox.showinfo("Success",
                    f"Subtitles successfully exported!\n\n"
                    f"• File: {os.path.basename(filename)}\n"
                    f"• Segments: {segment_count}\n"
                    f"• Translations: {translation_count}\n"
                    f"• Format: {format_type.upper()}\n\n"
                    f"Can be directly imported into video editors.", self.root)
            else:
                self.update_status("❌ Subtitle export failed")

        except Exception as e:
            self.update_status(f"❌ Subtitle export failed: {e}")
            DarkMessageBox.showerror("Error", f"Export failed:\n{str(e)}", self.root)

    def show_simple_stats(self):
        try:
            stats = self.performance_monitor.get_basic_stats()

            try:
                import psutil
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                health_status = "Healthy" if cpu < 90 and memory.percent < 85 else "Degraded"
            except:
                cpu = 0
                import psutil
                memory = psutil.virtual_memory()
                health_status = "Unknown"

            stats_text = f"""📊 STATISTIKEN:

🤖 PERFORMANCE:
⏱️ Runtime: {stats['uptime_minutes']:.1f} minutes
📝 Transcriptions: {stats['transcriptions']}
🌐 Translations: {stats['translations']}
🎯 Cache Hit Rate: {stats['cache_hit_rate']}

💻 SYSTEM:
🖥️ CPU: {cpu:.1f}%
🧠 RAM: {memory.used // (1024**2)}MB ({memory.percent:.1f}%)
⚡ Status: {health_status}

🎬 Subtitle mode: {'Active' if self.subtitle_mode else 'Inactive'}
"""
            DarkMessageBox.showinfo("Performance Statistics", stats_text, self.root)

        except Exception as e:
            self.update_status(f"❌ Statistics error: {e}")

    def show_advanced_settings(self):
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("Advanced Settings")
        settings_dialog.geometry("400x500")
        settings_dialog.configure(bg=DragonColors.BG_PRIMARY)
        settings_dialog.transient(self.root)
        settings_dialog.grab_set()
        settings_dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - settings_dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - settings_dialog.winfo_height()) // 2
        settings_dialog.geometry(f"+{x}+{y}")

        main_frame = tk.Frame(settings_dialog, bg=DragonColors.BG_PRIMARY, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        tk.Label(main_frame, text="Advanced Settings",
                font=DragonFonts.TITLE, bg=DragonColors.BG_PRIMARY, fg=DragonColors.TEXT_PRIMARY).pack(pady=(0, 20))

        settings_frame = tk.Frame(main_frame, bg=DragonColors.BG_PRIMARY)
        settings_frame.pack(fill='both', expand=True)

        tk.Label(settings_frame, text="Beam Size:", bg=DragonColors.BG_PRIMARY,
                fg=DragonColors.TEXT_PRIMARY).grid(row=0, column=0, sticky='w', pady=5)
        beam_var = tk.StringVar(value=str(self.advanced_settings.beam_size))
        beam_entry = tk.Entry(settings_frame, textvariable=beam_var,
                             bg=DragonColors.BG_TERTIARY, fg=DragonColors.TEXT_PRIMARY)
        beam_entry.grid(row=0, column=1, sticky='ew', pady=5)

        tk.Label(settings_frame, text="Temperature:", bg=DragonColors.BG_PRIMARY,
                fg=DragonColors.TEXT_PRIMARY).grid(row=1, column=0, sticky='w', pady=5)
        temp_var = tk.StringVar(value=str(self.advanced_settings.temperature))
        temp_entry = tk.Entry(settings_frame, textvariable=temp_var,
                             bg=DragonColors.BG_TERTIARY, fg=DragonColors.TEXT_PRIMARY)
        temp_entry.grid(row=1, column=1, sticky='ew', pady=5)

        plugin_var = tk.BooleanVar(value=self.settings.enable_plugins)
        plugin_cb = tk.Checkbutton(settings_frame, text="Enable plugins",
                                  variable=plugin_var, bg=DragonColors.BG_PRIMARY,
                                  fg=DragonColors.TEXT_PRIMARY, selectcolor=DragonColors.BG_TERTIARY)
        plugin_cb.grid(row=2, column=0, columnspan=2, sticky='w', pady=5)

        gpu_var = tk.BooleanVar(value=self.advanced_settings.gpu_acceleration)
        gpu_cb = tk.Checkbutton(settings_frame, text="Enable GPU acceleration",
                               variable=gpu_var, bg=DragonColors.BG_PRIMARY,
                               fg=DragonColors.TEXT_PRIMARY, selectcolor=DragonColors.BG_TERTIARY)
        gpu_cb.grid(row=3, column=0, columnspan=2, sticky='w', pady=5)

        settings_frame.columnconfigure(1, weight=1)

        def save_settings():
            try:
                self.advanced_settings.beam_size = int(beam_var.get())
                self.advanced_settings.temperature = float(temp_var.get())
                self.settings.enable_plugins = plugin_var.get()
                self.advanced_settings.gpu_acceleration = gpu_var.get()
                self.advanced_settings.save_to_file()
                self.settings.save_to_file()

                if not self.advanced_settings.gpu_acceleration:
                    self.transcription_engine.device = "cpu"
                    self.transcription_engine.compute_type = "int8"

                settings_dialog.destroy()
                self.update_status("✅ Settings saved")

            except Exception as e:
                DarkMessageBox.showerror("Error", f"Invalid settings: {e}", self.root)

        button_frame = tk.Frame(main_frame, bg=DragonColors.BG_PRIMARY)
        button_frame.pack(fill='x', pady=(20, 0))

        save_btn = tk.Button(
            button_frame, text="Save", command=save_settings,
            bg=DragonColors.SUCCESS, fg=DragonColors.TEXT_PRIMARY,
            relief='flat', padx=15
        )
        save_btn.pack(side='right', padx=5)

        cancel_btn = tk.Button(
            button_frame, text="Cancel", command=settings_dialog.destroy,
            bg=DragonColors.BG_TERTIARY, fg=DragonColors.TEXT_PRIMARY,
            relief='flat', padx=15)
        cancel_btn.pack(side='right', padx=5)

    def _setup_callbacks(self):
        self.controller.ui_update_fn = self._handle_ui_update
        self.controller.status_update_fn = self._handle_status_update

    def _handle_ui_update(self, component, text):
        if not text or not text.strip():
            return

        def update_task():
            try:
                if component == 'transcript' and hasattr(self, 'transcript_text'):
                    if self.transcript_text.winfo_exists():
                        self.transcript_text.insert('end', text)
                        if (hasattr(self, 'transcript_scroll_var') and
                            self.transcript_scroll_var.get()):
                            self.transcript_text.see('end')

                        lines = int(self.transcript_text.index('end-1c').split('.')[0])
                        if lines > 400:
                            keep_lines = 300
                            delete_to = f'{lines-keep_lines}.0'
                            self.transcript_text.delete('1.0', delete_to)

                elif component == 'translation' and hasattr(self, 'translation_text'):
                    if self.translation_text.winfo_exists():
                        self.translation_text.insert('end', text)
                        if (hasattr(self, 'translation_scroll_var') and
                            self.translation_scroll_var.get()):
                            self.translation_text.see('end')

                        lines = int(self.translation_text.index('end-1c').split('.')[0])
                        if lines > 300:
                            keep_lines = 200
                            delete_to = f'{lines-keep_lines}.0'
                            self.translation_text.delete('1.0', delete_to)

            except tk.TclError:
                pass
            except Exception as e:
                print(f"⚠️ Transcript GUI error: {e}")

        try:
            if hasattr(self, 'gui_queue'):
                self.gui_queue.put(('ui_update', update_task))
            else:
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, update_task)
        except Exception as e:
            print(f"⚠️ Queue put error: {e}")

    def _handle_status_update(self, state_info):
        def update_task():
            try:
                if 'status' in state_info and hasattr(self, 'status_label'):
                    if self.status_label.winfo_exists():
                        self.status_label.config(text=state_info['status'][:100])

                if 'buttons' in state_info:
                    buttons = state_info['buttons']
                    if hasattr(self, 'start_button') and self.start_button.winfo_exists():
                        self.start_button.config(state=buttons.get('start', 'normal'))
                    if hasattr(self, 'stop_button') and self.stop_button.winfo_exists():
                        self.stop_button.config(state=buttons.get('stop', 'disabled'))

                elif 'processing_state' in state_info:
                    processing = state_info['processing_state']
                    if hasattr(self, 'start_button') and self.start_button.winfo_exists():
                        self.start_button.config(state='disabled' if processing else 'normal')
                    if hasattr(self, 'stop_button') and self.stop_button.winfo_exists():
                        self.stop_button.config(state='normal' if processing else 'disabled')

                if 'stream_info' in state_info:
                    stream_info = state_info['stream_info']
                    self.current_stream_info = stream_info

                    if hasattr(self, 'stream_title_label') and self.stream_title_label.winfo_exists():
                        title = stream_info.title[:80] + "..." if len(stream_info.title) > 80 else stream_info.title
                        self.stream_title_label.config(text=f"📡 {title}")

                    if hasattr(self, 'stream_details_label') and self.stream_details_label.winfo_exists():
                        details = f"👤 {stream_info.uploader}"
                        if stream_info.duration and stream_info.duration != 'Live':
                            details += f" | ⏱️ {stream_info.duration}"
                        self.stream_details_label.config(text=details)

                if 'processing_state' in state_info:
                    processing = state_info['processing_state']
                    self.is_processing = processing

                    if hasattr(self, 'start_button') and self.start_button.winfo_exists():
                        self.start_button.config(state='disabled' if processing else 'normal')
                    if hasattr(self, 'stop_button') and self.stop_button.winfo_exists():
                        self.stop_button.config(state='normal' if processing else 'disabled')

            except Exception as e:
                print(f"⚠️ Status update error: {e}")

        try:
            if hasattr(self, 'gui_queue'):
                self.gui_queue.put(('status_update', update_task))
            else:
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, update_task)
        except Exception:
            pass

    def _start_gui_updaters(self):
        """Startet GUI Update System - REPARIERT"""
        def process_gui_queue():
            try:
                processed = 0
                max_per_cycle = 10

                while (processed < max_per_cycle and
                       hasattr(self, 'gui_queue') and
                       not self.gui_queue.empty()):

                    try:
                        item = self.gui_queue.get_nowait()

                        if isinstance(item, tuple) and len(item) == 2:
                            msg_type, callback = item

                            if callable(callback):
                                if self._gui_update_limiter.can_update(f'gui_{msg_type}'):
                                    try:
                                        callback()
                                    except Exception as e:
                                        print(f"⚠️ GUI callback error: {e}")

                        self.gui_queue.task_done()
                        processed += 1

                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"⚠️ Queue processing error: {e}")

                if hasattr(self, 'gui_queue'):
                    qsize = self.gui_queue.qsize()
                    if qsize > 30:
                        self._cleanup_queue(self.gui_queue, 20)

            except Exception as e:
                print(f"❌ GUI queue processor error: {e}")

            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(100, process_gui_queue)

        def process_text_updates():
            try:
                if not hasattr(self, '_text_update_queue'):
                    return

                processed = 0
                max_per_cycle = 5

                while (processed < max_per_cycle and
                       not self._text_update_queue.empty()):

                    try:
                        update_type, text_data = self._text_update_queue.get_nowait()

                        if update_type == 'transcript' and hasattr(self, 'transcript_text'):
                            if self.transcript_text.winfo_exists():
                                self.transcript_text.insert('end', text_data)
                                if (hasattr(self, 'transcript_scroll_var') and
                                    self.transcript_scroll_var.get()):
                                    self.transcript_text.see('end')

                        elif update_type == 'translation' and hasattr(self, 'translation_text'):
                            if self.translation_text.winfo_exists():
                                self.translation_text.insert('end', text_data)
                                if (hasattr(self, 'translation_scroll_var') and
                                    self.translation_scroll_var.get()):
                                    self.translation_text.see('end')

                        self._text_update_queue.task_done()
                        processed += 1

                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"⚠️ Text update error: {e}")

                if hasattr(self, '_text_update_queue'):
                    qsize = self._text_update_queue.qsize()
                    if qsize > 25:
                        self._cleanup_queue(self._text_update_queue, 15)

            except Exception as e:
                print(f"❌ Text update processor error: {e}")

            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(150, process_text_updates)

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(50, process_gui_queue)
            self.root.after(75, process_text_updates)

    def _start_gui_health_check(self):
        def health_check_worker():
            while hasattr(self, 'root') and self.root.winfo_exists():
                try:
                    time.sleep(30)
                    self._perform_gui_health_check()
                except Exception:
                    pass

        health_thread = threading.Thread(target=health_check_worker, daemon=True)
        health_thread.start()

    def _perform_gui_health_check(self):
        try:
            checks = []

            check_start = time.time()
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.update_idletasks()
            responsiveness_time = time.time() - check_start

            if responsiveness_time > 0.5:
                checks.append(f"⚠️ GUI responsiveness slow: {responsiveness_time:.1f}s")

            if hasattr(self, 'memory_manager'):
                mem_stats = self.memory_manager.get_memory_stats()
                if mem_stats.get('process_usage_percent', 0) > 80:
                    checks.append("⚠️ High memory usage")

            if hasattr(self, 'gui_queue'):
                qsize = self.gui_queue.qsize()
                if qsize > 50:
                    checks.append(f"⚠️ GUI queue backlog: {qsize} items")

            active_threads = threading.enumerate()
            if len(active_threads) > 15:
                checks.append(f"⚠️ Many active threads: {len(active_threads)}")

            try:
                cache_stats = get_cache_stats()
                for cache_name, stats in cache_stats.items():
                    if stats.get('total_entries', 0) > stats.get('max_size', 100) * 0.9:
                        checks.append(f"⚠️ {cache_name} nearly full")
            except:
                pass

            if checks:
                self._gui_health_status = "degraded"
                if len(checks) > 0:
                    print(f"🔍 GUI Health Check Issues: {checks[:3]}")
            else:
                self._gui_health_status = "healthy"

            if "queue backlog" in str(checks) and hasattr(self, 'gui_queue'):
                self._cleanup_queue(self.gui_queue, 30)

            if "memory usage" in str(checks) and hasattr(self, 'memory_manager'):
                self.memory_manager._aggressive_cleanup()

        except Exception as e:
            print(f"⚠️ Health check error: {e}")

    def _start_automatic_maintenance(self):
        def maintenance_worker():
            while hasattr(self, 'root') and self.root.winfo_exists():
                try:
                    time.sleep(60)
                    current_time = time.time()

                    if hasattr(self, '_last_maintenance'):
                        if current_time - self._last_maintenance > 600:
                            self._perform_maintenance()
                            self._last_maintenance = current_time
                    else:
                        self._last_maintenance = current_time

                except Exception:
                    pass

        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()

    def _perform_maintenance(self):
        print("🛠️ Performing automatic maintenance...")

        try:
            expired_counts = clear_expired_cache_entries()
            if any(count > 0 for count in expired_counts.values()):
                print(f"🧹 Cleared expired cache entries: {expired_counts}")
        except:
            pass

        if hasattr(self, 'memory_manager'):
            try:
                self.memory_manager._perform_periodic_maintenance()
            except:
                pass

        if hasattr(self, 'ffmpeg_manager'):
            try:
                self.ffmpeg_manager.cleanup_stale_processes()
            except:
                pass

        gc.collect()

        print("✅ Maintenance completed")

    def handle_transcription(self, result: ExcellenceTranscriptionResult):
        if not result or not result.text or not result.text.strip():
            return

        current_text = result.text.strip()

        if current_text == self._last_transcription_text:
            return

        self._last_transcription_text = current_text
        self.performance_monitor.log_transcription()
        self.transcript_history.append(result)

        def update_gui():
            try:
                if hasattr(self, 'transcript_text') and self.transcript_text.winfo_exists():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                
                    detected_lang = getattr(result, 'language', 'unknown')
                
                    lang_code = LANGUAGE_SHORT_CODES.get(detected_lang, '??')
                
                    text = f"[{timestamp}] [{lang_code}] {current_text}\n"
                
                    self.transcript_text.insert('end', text)

                    if (hasattr(self, 'transcript_scroll_var') and
                        self.transcript_scroll_var.get()):
                        self.transcript_text.see('end')

                    lines = int(self.transcript_text.index('end-1c').split('.')[0])
                    if lines > 400:
                        keep_lines = 300
                        delete_to = f'{lines-keep_lines}.0'
                        self.transcript_text.delete('1.0', delete_to)

            except tk.TclError:
                pass
            except Exception as e:
                print(f"⚠️ Transcript GUI error: {e}")

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update_gui)

        if (hasattr(self, 'translate_toggle') and self.translate_toggle.get() and
            hasattr(self, 'translation_engine') and self.translation_engine):

            def async_translation():
                try:
                    source_lang = getattr(result, 'language', 'unknown')
                    if source_lang not in ['unknown', 'auto']:
                        translation = self.translation_engine.translate_text(
                            current_text,
                            source_lang
                        )
                        if translation:
                            self.handle_translation(translation)
                except Exception as e:
                    print(f"⚠️ Translation error: {e}")

            translation_thread = threading.Thread(target=async_translation, daemon=True)
            translation_thread.start()

    def handle_translation(self, result: ExcellenceTranslationResult):
        if not result or not result.translated or not result.translated.strip():
            return

        current_text = result.translated.strip()

        if current_text == self._last_translation_text:
            return

        self._last_translation_text = current_text
        self.performance_monitor.log_translation()
        self.translation_history.append(result)

        def update_gui():
            try:
                if hasattr(self, 'translation_text') and self.translation_text.winfo_exists():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    text = f"[{timestamp}] {current_text}\n"

                    self.translation_text.insert('end', text)

                    if (hasattr(self, 'translation_scroll_var') and
                        self.translation_scroll_var.get()):
                        self.translation_text.see('end')

                    lines = int(self.translation_text.index('end-1c').split('.')[0])
                    if lines > 300:
                        keep_lines = 200
                        delete_to = f'{lines-keep_lines}.0'
                        self.translation_text.delete('1.0', delete_to)

            except tk.TclError:
                pass
            except Exception as e:
                print(f"⚠️ Translation GUI error: {e}")

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update_gui)

    def handle_info(self, info_msg: str):
        def update():
            self.update_status(f"ℹ️ {info_msg}")

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update)

    def handle_error(self, error_msg: str):
        def update():
            self.update_status(f"❌ {error_msg}")
            if self.is_processing:
                self.controller.stop_processing()

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update)

    def update_status(self, message: str):
        try:
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text=message[:100])
        except Exception:
            pass

    def update_stream_info(self, info: StreamInfo):
        def update_gui():
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
                print(f"⚠️ Stream info update error: {e}")

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, update_gui)

    def _try_fallback_gui(self):
        print("ℹ️ Starte im Kommandozeilen-Modus...")
        raise RuntimeError("Bitte installieren Sie Tkinter: pip install tk")

    def _show_error_and_exit(self, message):
        print(f"💥 KRITISCHER FEHLER: {message}")
        try:
            import tkinter.messagebox as mb
            mb.showerror("Dragon Whisperer - Fehler", message)
        except:
            pass
        self._emergency_cleanup()
        sys.exit(1)

    def _show_warning(self, message):
        print(f"⚠️ WARNUNG: {message}")

    def _start_system_monitoring(self):
        def monitor():
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                ram_used = memory.used // (1024**2)
                ram_total = memory.total // (1024**2)
                gpu_text = ""
                try:
                    if TORCH_AVAILABLE:
                        torch = FastLazyLoader.load('torch')
                        if torch.cuda.is_available():
                            gpu_count = torch.cuda.device_count()
                            gpu_name = torch.cuda.get_device_name(0)[:15] + "..."
                            gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                            gpu_text = f" | 🎮 GPU{'' if gpu_count == 1 else f'({gpu_count})'}: {gpu_name}"
                        else:
                            gpu_text = " | 🎮 GPU: ❌"
                    else:
                        gpu_text = " | 🎮 GPU: N/A"
                except Exception as e:
                    gpu_text = " | 🎮 GPU: Error"

                current_model = "None"
                if hasattr(self, 'transcription_engine'):
                    current_model = self.transcription_engine.get_current_model()

                if IS_WINDOWS:
                    info = f"🪟 Windows | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}"
                elif IS_MACOS:
                    if IS_ARM:
                        info = f"🍎 macOS ARM | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}"
                    else:
                        info = f"🍎 macOS Intel | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}"
                elif IS_LINUX:
                    info = f"🐧 Linux | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}"
                else:
                    info = f"🌐 System | 🔧 CPU: {cpu:.0f}% | 💾 RAM: {ram_used}/{ram_total}MB{gpu_text} | 🤖 Model: {current_model}"

                if hasattr(self, 'system_info_label'):
                    self.system_info_label.config(text=info)

            except Exception as e:
                if hasattr(self, 'system_info_label'):
                    self.system_info_label.config(text="⚙️ System monitoring unavailable")

            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(3000, monitor)

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(1000, monitor)

    def _final_initialization_check(self):
        print("✅ Dragon Whisperer initialisiert")

    def _emergency_cleanup(self):
        """Emergency Cleanup für kritische Fehler"""
        print("🆘 Emergency cleanup...")
        self._minimal_emergency_cleanup()

def _setup_windows_console():
    """Windows-spezifische Console Setup"""
    if not IS_WINDOWS:
        return

    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    except Exception:
        pass

def check_platform_compatibility():
    """Prüft Plattform-Kompatibilität"""
    issues = []

    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required (you have {sys.version_info.major}.{sys.version_info.minor})")

    if IS_WINDOWS:
        try:
            import platform
            win_ver = platform.version()
            major_ver = int(win_ver.split('.')[0])
            if major_ver < 10:
                issues.append("Windows 10+ recommended for best experience")
        except:
            pass

    return issues

def debug_script():
    """Debug-Informationen ausgeben"""
    print("=" * 60)
    print("DRAGON WHISPERER - DEBUG INFORMATION")
    print("=" * 60)

    print(f"\n🔧 Platform: {SYSTEM} {'ARM' if IS_ARM else 'x86'}")
    print(f"🔧 Python: {sys.version.split()[0]}")

    print("\n📦 Dependencies:")
    deps = [
        ("FFmpeg", shutil.which('ffmpeg')),
        ("yt-dlp", shutil.which('yt-dlp')),
        ("faster-whisper", WHISPER_AVAILABLE),
        ("deep-translator", TRANSLATOR_AVAILABLE),
        ("Tkinter", GUI_AVAILABLE),
        ("NumPy", NUMPY_AVAILABLE),
        ("PyTorch", TORCH_AVAILABLE),
    ]

    for name, status in deps:
        symbol = "✅" if status else "❌"
        status_text = "Available" if status else "Not available"
        print(f"  {symbol} {name:18} {status_text}")

    print(f"\n📁 Script: {os.path.abspath(__file__)}")
    print(f"📁 CWD: {os.getcwd()}")

    print("=" * 60)

def setup_platform_environment():
    """Plattformspezifische Umgebungs-Setup - REPARIERT: WIRD NUR IN MAIN() AUFGERUFEN"""
    env_vars = {}

    if IS_WINDOWS:
        env_vars.update({
            'FFMPEG_BINARY': 'ffmpeg.exe',
            'YT_DLP_BINARY': 'yt-dlp.exe',
            'PYTHONIOENCODING': 'utf-8'
        })
    elif IS_MACOS:
        env_vars.update({
            'FFMPEG_BINARY': 'ffmpeg',
            'YT_DLP_BINARY': 'yt-dlp'
        })
    else:
        env_vars.update({
            'FFMPEG_BINARY': 'ffmpeg',
            'YT_DLP_BINARY': 'yt-dlp'
        })

    for key, value in env_vars.items():
        os.environ[key] = value

    return env_vars

def _print_help():
    """Hilfetext anzeigen"""
    print("🐉 Dragon Whisperer - Ultimate Stream Transcription & Translation")
    print("="*60)
    print("\nUsage:")
    print("  python dragon_whisperer.py [options]")
    print("\nOptions:")
    print("  --quiet, -q    Quiet mode (minimal output)")
    print("  --debug        Debug mode (verbose output)")
    print("  --check        System compatibility check")
    print("  --help, -h     Show this help")
    print("  --version, -v  Show version")
    print("\nExamples:")
    print("  Normal use:   python dragon_whisperer.py")
    print("  Quiet mode:   python dragon_whisperer.py --quiet")
    print("  Debug mode:   python dragon_whisperer.py --debug")
    print("  System check: python dragon_whisperer.py --check")
    print("\nFeatures:")
    print("  • Live stream transcription (YouTube, Twitch, etc.)")
    print("  • Real-time translation to 50+ languages")
    print("  • Subtitle export (SRT, VTT)")
    print("  • Batch processing")
    print("  • Dark mode GUI")

def _run_system_check():
    """System-Check durchführen"""
    issues = []

    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required (you have {sys.version_info.major}.{sys.version_info.minor})")

    if IS_WINDOWS:
        try:
            import platform
            win_ver = platform.version()
            major_ver = int(win_ver.split('.')[0]) if '.' in win_ver else 0
            if major_ver < 10:
                issues.append("Windows 10+ recommended for best experience")
        except:
            pass

    print("🔍 Dragon Whisperer - System Compatibility Check")
    print("="*50)

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

    print("\n💻 System Info:")
    print(f"  Platform: {SYSTEM}")
    print(f"  Architecture: {'ARM' if IS_ARM else 'x86'}")
    print(f"  Python: {sys.version.split()[0]}")

    if IS_WINDOWS:
        try:
            import platform
            print(f"  Windows: {platform.version()}")
        except:
            pass

    return 0 if not issues else 1

def _show_user_error(message):
    """Benutzerfreundliche Fehlermeldung"""
    if IS_WINDOWS and not sys.stdin.isatty():
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f"Dragon Whisperer - Setup Required\n\n{message}\n\n"
                "Please install missing components and try again.",
                "Setup Error",
                0x10
            )
        except:
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

def _show_critical_error(message):
    """Kritische Fehlermeldung"""
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
                0x10
            )
        except:
            pass

def main():
    """Hauptfunktion - Professionelle Version für Produktion - KORRIGIERT"""
    warnings.filterwarnings("ignore")

    if IS_WINDOWS:
        try:
            import io
            import ctypes
            
            if sys.stdout and hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )

            if sys.stderr and hasattr(sys.stderr, 'buffer'):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)

            STD_OUTPUT_HANDLE = -11
            handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                ENABLE_PROCESSED_OUTPUT = 0x0001
                new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT
                kernel32.SetConsoleMode(handle, new_mode)

        except Exception:
            pass

    debug_level = 0
    cli_args = {
        'debug': '--debug' in sys.argv,
        'quiet': '--quiet' in sys.argv or '-q' in sys.argv,
        'check': '--check' in sys.argv,
        'help': '--help' in sys.argv or '-h' in sys.argv,
        'version': '--version' in sys.argv or '-v' in sys.argv
    }

    if cli_args['help']:
        _print_help()
        return 0
    if cli_args['version']:
        print("🐉 Dragon Whisperer v1.0 - Ultimate Stream Transcription & Translation")
        print(f"Platform: {SYSTEM} {'ARM' if IS_ARM else 'x86'}")
        return 0

    debug_level = 0

    def log(level, message):
        if level <= debug_level:
            import time
            timestamp = time.strftime("%H:%M:%S")
            level_names = {0: "ERROR", 1: "INFO", 2: "DEBUG", 3: "TRACE"}
            level_name = level_names.get(level, "INFO")
            if cli_args['quiet']:
                print(message)
            else:
                print(f"[{timestamp}] [{level_name}] {message}")

    if cli_args['check']:
        return _run_system_check()

    app = None
    exit_code = 0

    try:
        log(1, "🐉 Dragon Whisperer starting...")

        if IS_WINDOWS:
            _setup_windows_console()

        log(2, "🔍 Checking dependencies...")
        PlatformUtils.check_platform_dependencies()
        log(2, "✅ Dependencies OK")

        if not GUI_AVAILABLE:
            raise RuntimeError("Tkinter/GUI not available. Install with: pip install tk")

        log(2, "⚡ Setting up signal handlers...")
        SignalHandler.setup(
            verbose=False,
            silent=True,
            max_cleanup_time=10.0
        )

        log(2, "🖥️ Initializing GUI...")
        app = DragonWhispererGUI()

        if debug_level >= 1 and not cli_args['quiet']:
            print("\n" + "="*50)
            print("🐉 DRAGON WHISPERER READY")
            print("="*50)
            print(f"Platform: {SYSTEM} {'ARM' if IS_ARM else 'x86'}")
            print(f"Python: {sys.version.split()[0]}")
            print(f"Working Dir: {os.getcwd()}")
            if hasattr(app, 'transcription_engine'):
                current_model = app.transcription_engine.get_current_model()
                print(f"Model: {current_model if current_model else 'Not loaded'}")
            print(f"Layout: {getattr(app, 'layout_mode', 'vertical')}")
            print("="*50 + "\n")

        log(1, "🚀 Starting main loop...")
        app.run()
        log(1, "✅ Application closed normally")

    except KeyboardInterrupt:
        log(1, "\n🛑 Interrupted by user")
        exit_code = 0

    except RuntimeError as e:
        error_msg = str(e)
        log(0, f"❌ {error_msg}")
        _show_user_error(error_msg)
        exit_code = 1

    except Exception as e:
        error_msg = str(e)
        log(0, f"💥 Unexpected error: {error_msg}")
        if debug_level >= 2:
            import traceback
            traceback.print_exc()
        _show_critical_error(error_msg)
        exit_code = 2

    finally:
        log(2, "🧹 Final minimal cleanup...")
    
        try:
            transcription_cache.clear()
            translation_cache.clear()
            audio_cache.clear()
        except:
            pass
    
        import gc
        gc.collect()
    
        log(2, "✅ Shutdown complete")
    
    return exit_code

if __name__ == "__main__":
    try:
        if '__file__' in globals():
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.getcwd() != script_dir:
                os.chdir(script_dir)
                if '--debug' in sys.argv:
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

        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
            print(f"\n🔧 Debug Info:")
            print(f"  Python: {sys.version}")
            print(f"  Platform: {sys.platform}")
            print(f"  Executable: {sys.executable}")
            print(f"  CWD: {os.getcwd()}")
            print(f"  Script: {__file__ if '__file__' in globals() else 'Unknown'}")

        sys.exit(99)
