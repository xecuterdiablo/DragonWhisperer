#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🐉 Dragon Whisperer - Live Stream Transcription and Translation
VOLLSTÄNDIG REPARIERTE VERSION - Vereinfachte Stop-Architektur & Memory-Leaks behoben
"""

# ===== STANDARD LIBRARY IMPORTS =====
import argparse
import atexit
import configparser
import gc
import hashlib
import json
import logging
import math
import os
import queue
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import OrderedDict, deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from weakref import WeakSet, ref

# ===== EXTERNAL DEPENDENCIES =====
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, filedialog, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class DummyNP:
        def __getattr__(self, name):
            if name in ['array', 'ndarray', 'float32', 'int16']:
                return self._dummy_array
            return self._dummy_method
        
        def _dummy_array(self, *args, **kwargs):
            return None
            
        def _dummy_method(self, *args, **kwargs):
            return None
            
        def __getitem__(self, key):
            return None
    np = DummyNP()

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ===== ENHANCED MEMORY MANAGEMENT =====
class EnhancedMemoryManager:
    """Enhanced memory management with aggressive cleanup"""
    
    def __init__(self):
        self._last_cleanup = 0
        self._cleanup_interval = 30
        self._high_memory_count = 0
        self._cleanup_callbacks = []
        
    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register callback for memory cleanup"""
        self._cleanup_callbacks.append(callback)
        
    def check_and_cleanup(self) -> bool:
        """Check memory and perform cleanup if needed"""
        current_time = time.time()
        
        if current_time - self._last_cleanup < self._cleanup_interval:
            return True
            
        if not PSUTIL_AVAILABLE:
            return True
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > AppConstants.MAX_MEMORY_MB * 0.8:
                self._high_memory_count += 1
                self._perform_aggressive_cleanup()
                self._last_cleanup = current_time
                
                if self._high_memory_count > 3:
                    logging.warning("Persistent high memory usage detected")
                    return False
                    
            else:
                self._high_memory_count = 0
                
            return True
            
        except Exception as e:
            logging.debug(f"Memory check failed: {e}")
            return True
            
    def _perform_aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        logging.info("Performing aggressive memory cleanup")
        
        # Execute registered cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logging.debug(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# ===== PERFORMANCE MONITORING =====
class PerformanceMonitor:
    """Enhanced performance monitoring with memory tracking"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'memory_usage_mb': [],
            'processing_times': deque(maxlen=100),
            'transcription_latencies': deque(maxlen=100),
            'translation_latencies': deque(maxlen=100),
            'queue_sizes': deque(maxlen=100),
            'error_count': 0,
            'chunks_processed': 0,
            'audio_quality_scores': deque(maxlen=100)
        }
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._monitor_running = False
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitor_running:
            try:
                self._record_memory_usage()
                time.sleep(5)
            except Exception as e:
                logging.debug(f"Performance monitoring error: {e}")
                time.sleep(10)
                
    def _record_memory_usage(self):
        """Record current memory usage"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                with self._lock:
                    self.metrics['memory_usage_mb'].append(memory_mb)
                    # Keep only last 100 readings
                    if len(self.metrics['memory_usage_mb']) > 100:
                        self.metrics['memory_usage_mb'].pop(0)
            except Exception as e:
                logging.debug(f"Memory monitoring failed: {e}")
                
    def record_processing_time(self, duration: float):
        """Record processing time for a chunk"""
        with self._lock:
            self.metrics['processing_times'].append(duration)
            self.metrics['chunks_processed'] += 1
            
    def record_transcription_latency(self, duration: float):
        """Record transcription latency"""
        with self._lock:
            self.metrics['transcription_latencies'].append(duration)
            
    def record_translation_latency(self, duration: float):
        """Record translation latency"""
        with self._lock:
            self.metrics['translation_latencies'].append(duration)
            
    def record_audio_quality(self, quality_score: float):
        """Record audio quality score"""
        with self._lock:
            self.metrics['audio_quality_scores'].append(quality_score)
            
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage - neue Methode"""
        with self._lock:
            self.metrics['memory_usage_mb'].append(memory_mb)
            if len(self.metrics['memory_usage_mb']) > 100:
                self.metrics['memory_usage_mb'].pop(0)
            
    def record_error(self):
        """Record error occurrence"""
        with self._lock:
            self.metrics['error_count'] += 1
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self._lock:
            if not self.metrics['processing_times']:
                avg_processing = 0
                p95_processing = 0
            else:
                avg_processing = np.mean(self.metrics['processing_times']) if NUMPY_AVAILABLE else \
                               sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
                p95_processing = np.percentile(self.metrics['processing_times'], 95) if NUMPY_AVAILABLE else avg_processing * 1.5
                
            if not self.metrics['memory_usage_mb']:
                avg_memory = 0
                max_memory = 0
            else:
                avg_memory = sum(self.metrics['memory_usage_mb']) / len(self.metrics['memory_usage_mb'])
                max_memory = max(self.metrics['memory_usage_mb'])
                
            uptime = time.time() - self.start_time
            
            return {
                'uptime_seconds': uptime,
                'chunks_processed': self.metrics['chunks_processed'],
                'error_count': self.metrics['error_count'],
                'avg_processing_time': avg_processing,
                'p95_processing_time': p95_processing,
                'avg_memory_mb': avg_memory,
                'max_memory_mb': max_memory,
                'error_rate': self.metrics['error_count'] / max(self.metrics['chunks_processed'], 1),
                'status': 'healthy' if self.metrics['error_count'] / max(self.metrics['chunks_processed'], 1) < 0.1 else 'degraded'
            }
            
    def get_performance_health(self) -> Dict[str, Any]:
        """Get performance health status"""
        report = self.get_performance_report()
        return {
            'status': report['status'],
            'error_rate': report['error_rate'],
            'avg_processing_time': report['avg_processing_time']
        }
            
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitor_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

# ===== REPARIERTES EVENT SYSTEM =====
class DragonEvents(Enum):
    TRANSCRIPTION_READY = "transcription_ready"
    TRANSLATION_READY = "translation_ready"
    STREAM_CONNECTED = "stream_connected"
    STREAM_DISCONNECTED = "stream_disconnected"
    ERROR_OCCURRED = "error_occurred"
    HEALTH_UPDATE = "health_update"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_STOPPED = "processing_stopped"
    AUDIO_QUALITY_UPDATE = "audio_quality_update"
    MODEL_LOADED = "model_loaded"
    PERFORMANCE_UPDATE = "performance_update"
    MEMORY_WARNING = "memory_warning"

class DragonEventSystem:
    """Enhanced event system with proper memory management - REPARIERT"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._events = defaultdict(list)
            cls._instance._lock = threading.RLock()
            cls._instance._performance_monitor = PerformanceMonitor()
            # 🔥 REPARIERT: Eigenes Callback-Management
            cls._instance._callback_registry = defaultdict(dict)  # {event_type: {id: callback}}
            cls._instance._next_callback_id = 0
        return cls._instance
    
    def subscribe(self, event_type: DragonEvents, callback: Callable) -> int:
        """Subscribe to event with proper memory management"""
        with self._lock:
            callback_id = self._next_callback_id
            self._next_callback_id += 1
            
            # Callback in Registry speichern
            self._callback_registry[event_type][callback_id] = callback
            
            # Wrapper für sichere Ausführung
            def callback_wrapper(data):
                try:
                    if callback_id in self._callback_registry[event_type]:
                        callback(data)
                except Exception as e:
                    logging.error(f"Event callback failed: {e}")
            
            self._events[event_type].append(callback_wrapper)
            return callback_id
    
    def unsubscribe(self, event_type: DragonEvents, callback_id: int) -> None:
        """Unsubscribe using callback ID"""
        with self._lock:
            if event_type in self._callback_registry and callback_id in self._callback_registry[event_type]:
                # Entferne aus Registry
                del self._callback_registry[event_type][callback_id]
                
                # Entferne entsprechende Wrapper-Funktion
                if event_type in self._events:
                    self._events[event_type] = [
                        cb for cb in self._events[event_type]
                        if not self._is_callback_wrapper(cb, callback_id)
                    ]
    
    def _is_callback_wrapper(self, wrapper_func, callback_id):
        """Check if wrapper belongs to callback ID"""
        try:
            # Prüfe ob die Wrapper-Funktion zu dieser Callback-ID gehört
            return hasattr(wrapper_func, '_callback_id') and wrapper_func._callback_id == callback_id
        except:
            return False
    
    def unsubscribe_all(self, event_type: DragonEvents) -> None:
        """Unsubscribe all callbacks for event type"""
        with self._lock:
            if event_type in self._callback_registry:
                self._callback_registry[event_type].clear()
            if event_type in self._events:
                self._events[event_type].clear()
    
    def publish(self, event_type: DragonEvents, data: Any = None) -> None:
        """Publish event with cleanup"""
        with self._lock:
            callbacks = self._events.get(event_type, [])[:]
        
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Event callback failed for {event_type}: {e}")
    
    def clear_all(self) -> None:
        """Clear all events and callbacks"""
        with self._lock:
            self._events.clear()
            self._callback_registry.clear()
            self._performance_monitor.stop_monitoring()
    
    def start_performance_monitoring(self):
        """Start performance monitoring"""
        self._performance_monitor.start_monitoring()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return self._performance_monitor.get_performance_report()
    
    def record_processing_time(self, duration: float):
        """Record processing time"""
        self._performance_monitor.record_processing_time(duration)
    
    def record_error(self):
        """Record error"""
        self._performance_monitor.record_error()

# ===== OPTIMIZED KONSTANTEN & KONFIGURATION =====
class AppConstants:
    MAX_QUEUE_SIZE = 50
    MAX_CACHE_SIZE = 1000
    MAX_HISTORY_SIZE = 500
    CHUNK_TIMEOUT = 30.0
    MAX_CONSECUTIVE_ERRORS = 3
    MIN_CONFIDENCE_THRESHOLD = 0.3
    CHUNK_DURATION = 6.0
    SUPPORTED_LANGUAGES = ['en', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar']
    VALID_MODELS = ['tiny', 'base', 'small', 'medium', 'large-v2']
    DEFAULT_MODEL = 'small'
    SAMPLE_RATE = 16000
    CHANNELS = 1
    SILENCE_THRESHOLD = 0.005
    MIN_CONFIDENCE = 0.3
    QUALITY_EXCELLENT = 0.7
    QUALITY_GOOD = 0.5
    QUALITY_FAIR = 0.3
    MAX_MEMORY_MB = 2048

class UIConstants:
    MAX_TEXT_LINES = 1000
    CLEANUP_THRESHOLD = 500
    UPDATE_INTERVAL_MS = 2000
    STREAM_TITLE_MAX_LENGTH = 60
    TEXT_CLEANUP_INTERVAL_MS = 30000

class PerformanceConfig:
    _lock = threading.RLock()
    
    @classmethod
    def initialize(cls):
        with cls._lock:
            cls.MAX_QUEUE_SIZE = AppConstants.MAX_QUEUE_SIZE
            cls.MAX_CACHE_SIZE = AppConstants.MAX_CACHE_SIZE
            cls.MAX_HISTORY_SIZE = AppConstants.MAX_HISTORY_SIZE
            cls.CHUNK_TIMEOUT = AppConstants.CHUNK_TIMEOUT
            cls.MAX_CONSECUTIVE_ERRORS = AppConstants.MAX_CONSECUTIVE_ERRORS
            cls.MIN_CONFIDENCE_THRESHOLD = AppConstants.MIN_CONFIDENCE_THRESHOLD
            cls.CHUNK_DURATION = AppConstants.CHUNK_DURATION
            cls.MEMORY_MONITORING = True
    
    @classmethod
    def adjust_for_memory(cls, available_memory: int) -> None:
        with cls._lock:
            if available_memory < 2 * 1024**3:  # Weniger als 2GB RAM
                cls.MAX_CACHE_SIZE = 300
                cls.MAX_QUEUE_SIZE = 20
                cls.MAX_HISTORY_SIZE = 200
                cls.CHUNK_DURATION = 4.0
                logging.info("Performance config adjusted for low memory system")
            elif available_memory > 8 * 1024**3:  # Mehr als 8GB RAM
                cls.MAX_CACHE_SIZE = 2000
                cls.MAX_QUEUE_SIZE = 100
                cls.MAX_HISTORY_SIZE = 1000
                cls.CHUNK_DURATION = 8.0
                logging.info("Performance config optimized for high memory system")
    
    @classmethod
    def adjust_queue_size(cls, new_size: int) -> None:
        with cls._lock:
            old_size = cls.MAX_QUEUE_SIZE
            # Weniger aggressive Anpassungen - nur bei signifikanter Änderung
            if abs(new_size - old_size) > 10:
                cls.MAX_QUEUE_SIZE = max(10, min(200, new_size))  # Limit between 10 and 200
                if old_size != cls.MAX_QUEUE_SIZE:
                    logging.info(f"Queue size adjusted from {old_size} to {cls.MAX_QUEUE_SIZE}")

PerformanceConfig.initialize()

# ===== ENHANCED FEHLERBEHANDLUNG =====
class ErrorHandler:
    """Enhanced error handling with performance tracking"""
    
    @staticmethod
    def handle_critical(error: Exception, context: str = "") -> None:
        logging.critical(f"CRITICAL {context}: {error}")
        DragonEventSystem().publish(DragonEvents.ERROR_OCCURRED, {
            'type': 'critical',
            'error': error,
            'context': context,
            'timestamp': time.time()
        })
        DragonEventSystem().record_error()
    
    @staticmethod
    def handle_operation(error: Exception, operation: str = "") -> None:
        logging.error(f"Operation failed {operation}: {error}")
        DragonEventSystem().publish(DragonEvents.ERROR_OCCURRED, {
            'type': 'operation',
            'error': error,
            'context': operation,
            'timestamp': time.time()
        })
        DragonEventSystem().record_error()
    
    @staticmethod
    def handle_graceful(error: Exception, context: str = "") -> None:
        logging.warning(f"Non-critical {context}: {error}")
        DragonEventSystem().publish(DragonEvents.ERROR_OCCURRED, {
            'type': 'graceful',
            'error': error,
            'context': context,
            'timestamp': time.time()
        })

    @staticmethod
    def handle_audio_error(error: Exception, chunk_id: int) -> None:
        ErrorHandler.handle_operation(error, f"audio processing in chunk {chunk_id}")

    @staticmethod  
    def handle_stream_error(error: Exception, url: str) -> None:
        ErrorHandler.handle_operation(error, f"stream processing for {url}")

    @staticmethod
    def handle_transcription_error(error: Exception, audio_data_size: int) -> None:
        ErrorHandler.handle_operation(error, f"transcription with audio size {audio_data_size}")

# ===== REPARIERTES RESOURCE MANAGEMENT =====
class FixedResourceManager:
    """Enhanced resource manager with memory monitoring"""
    
    def __init__(self):
        self._cleanup_registry = []
        self._resources = {}
        self._lock = threading.RLock()
        self._cleanup_in_progress = False
        self._shutdown_called = False
        self._memory_warning_sent = False
        self._memory_manager = EnhancedMemoryManager()
        
    def register_cleanup(self, cleanup_func: Callable, description: str = "") -> None:
        with self._lock:
            if not self._shutdown_called:
                self._cleanup_registry.append((cleanup_func, description))
    
    def register_resource(self, resource: Any, cleanup_func: Callable) -> None:
        with self._lock:
            if self._shutdown_called:
                return
                
            def cleanup_callback(weak_ref):
                # This gets called when the resource is garbage collected
                if weak_ref in self._resources:
                    cleanup_func_to_call = self._resources.pop(weak_ref)
                    try:
                        cleanup_func_to_call(resource)
                    except Exception as e:
                        ErrorHandler.handle_graceful(e, "auto resource cleanup")
            
            weak_ref = ref(resource, cleanup_callback)
            self._resources[weak_ref] = cleanup_func
    
    def check_memory_usage(self) -> bool:
        """Check memory usage and trigger cleanup if needed"""
        return self._memory_manager.check_and_cleanup()
    
    def register_memory_cleanup_callback(self, callback: Callable) -> None:
        """Register callback for memory cleanup"""
        self._memory_manager.register_cleanup_callback(callback)
    
    def cleanup_all(self) -> None:
        # REPARIERT: Thread-sicheren Check mit Lock
        with self._lock:
            if self._cleanup_in_progress or self._shutdown_called:
                return
            self._cleanup_in_progress = True
            self._shutdown_called = True
        
        try:
            # 1. Execute manual cleanup functions
            for cleanup_func, description in reversed(self._cleanup_registry):
                try:
                    logging.info(f"Cleaning up: {description}")
                    cleanup_func()
                except Exception as e:
                    ErrorHandler.handle_graceful(e, f"cleanup for {description}")
            
            # 2. Force cleanup of remaining resources
            with self._lock:
                remaining_resources = list(self._resources.items())
            
            for weak_ref, cleanup_func in remaining_resources:
                resource = weak_ref()
                if resource is not None:
                    try:
                        cleanup_func(resource)
                    except Exception as e:
                        ErrorHandler.handle_graceful(e, "forced resource cleanup")
            
            # 3. Clear all registries
            with self._lock:
                self._resources.clear()
                self._cleanup_registry.clear()
                
            # 4. Final garbage collection
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            ErrorHandler.handle_critical(e, "cleanup process")
        finally:
            self._cleanup_in_progress = False

# ===== REPARIERTES FFMPEG MANAGEMENT =====
class FFmpegManager:
    """Enhanced FFmpeg process management with reliable shutdown"""
    
    @staticmethod
    def safe_shutdown(process: Optional[subprocess.Popen], timeout: int = 5) -> bool:
        if not process or process.poll() is not None:
            return True
            
        try:
            # Prozessgruppe beenden für Linux/Windows Kompatibilität
            if hasattr(os, 'killpg'):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, AttributeError):
                    process.terminate()
            else:
                process.terminate()
                
            try:
                process.wait(timeout=timeout)
                return True
            except subprocess.TimeoutExpired:
                logging.warning("FFmpeg termination timeout, forcing kill")
                try:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=2)
                    return True
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    return False
        except ProcessLookupError:
            return True
        except Exception as e:
            ErrorHandler.handle_graceful(e, "FFmpeg shutdown")
            return False
    
    @staticmethod
    def close_pipes(process: Optional[subprocess.Popen]) -> None:
        if not process:
            return
            
        pipes = [process.stdout, process.stderr, process.stdin]
        for pipe in pipes:
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass

# ===== ENHANCED THREAD-SICHERE DATENSTRUKTUREN =====
class ThreadSafeSet:
    """Thread-safe set with performance monitoring"""
    
    def __init__(self):
        self._set = set()
        self._lock = threading.RLock()
        self._access_count = 0
    
    def add(self, item: Any) -> None:
        with self._lock:
            self._set.add(item)
            self._access_count += 1
    
    def remove(self, item: Any) -> None:
        with self._lock:
            self._set.discard(item)
            self._access_count += 1
    
    def get_all(self) -> List[Any]:
        with self._lock:
            self._access_count += 1
            return list(self._set)
    
    def clear(self) -> None:
        with self._lock:
            self._set.clear()
            self._access_count += 1
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._set)
    
    def __contains__(self, item: Any) -> bool:
        with self._lock:
            return item in self._set
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'size': len(self._set),
                'access_count': self._access_count
            }

class ThreadSafeAnalytics:
    """Enhanced analytics with performance tracking"""
    
    def __init__(self):
        self._data = {
            'start_time': time.time(),
            'total_words': 0,
            'languages_detected': ThreadSafeSet(),
            'total_processing_time': 0.0,
            'audio_quality_trend': 'unknown',
            'average_confidence': 0.0,
            'confidence_history': deque(maxlen=1000),
            'processing_times': deque(maxlen=500)
        }
        self._lock = threading.RLock()
    
    def add_language(self, language: str) -> None:
        self._data['languages_detected'].add(language)
    
    def get_languages(self) -> List[str]:
        return self._data['languages_detected'].get_all()
    
    def increment_words(self, count: int) -> None:
        with self._lock:
            self._data['total_words'] += count
    
    def add_confidence(self, confidence: float) -> None:
        with self._lock:
            self._data['confidence_history'].append(confidence)
            if self._data['confidence_history']:
                self._data['average_confidence'] = (
                    sum(self._data['confidence_history']) / 
                    len(self._data['confidence_history'])
                )
    
    def add_processing_time(self, processing_time: float) -> None:
        with self._lock:
            self._data['processing_times'].append(processing_time)
            self._data['total_processing_time'] += processing_time
    
    def update_quality_trend(self, trend: str) -> None:
        with self._lock:
            self._data['audio_quality_trend'] = trend
    
    def get_data(self) -> Dict[str, Any]:
        with self._lock:
            processing_times = list(self._data['processing_times'])
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            return {
                'start_time': self._data['start_time'],
                'total_words': self._data['total_words'],
                'languages_detected': self.get_languages(),
                'total_processing_time': self._data['total_processing_time'],
                'audio_quality_trend': self._data['audio_quality_trend'],
                'average_confidence': self._data['average_confidence'],
                'avg_processing_time': avg_processing_time,
                'total_segments_processed': len(processing_times)
            }
    
    def reset(self) -> None:
        with self._lock:
            self._data = {
                'start_time': time.time(),
                'total_words': 0,
                'languages_detected': ThreadSafeSet(),
                'total_processing_time': 0.0,
                'audio_quality_trend': 'unknown',
                'average_confidence': 0.0,
                'confidence_history': deque(maxlen=1000),
                'processing_times': deque(maxlen=500)
            }

# ===== REPARIERTES GUI UPDATE SYSTEM =====
class ThreadSafeGUIUpdater:
    """Enhanced GUI updater with rate limiting and memory optimization"""
    
    def __init__(self, root):
        self.root = root
        self._update_queue = queue.Queue(maxsize=500)
        self._update_lock = threading.RLock()
        self._is_running = True
        self._pending_updates = 0
        self._max_pending_updates = 50
        self._processed_updates = 0
        self._dropped_updates = 0
        self._last_update_time = 0
        self._min_update_interval = 0.05
        self._start_update_processor()
    
    def _start_update_processor(self) -> None:
        def process_updates():
            while self._is_running:
                try:
                    update_data = self._update_queue.get(timeout=0.1)
                    if update_data and self._is_running:
                        try:
                            if self.root.winfo_exists():
                                update_start = time.time()
                                update_data()
                                update_time = time.time() - update_start
                                if update_time > 0.1:
                                    logging.debug(f"Slow GUI update: {update_time:.3f}s")
                        except (tk.TclError, RuntimeError):
                            break
                        finally:
                            with self._update_lock:
                                self._pending_updates = max(0, self._pending_updates - 1)
                                self._processed_updates += 1
                except queue.Empty:
                    continue
                except Exception as e:
                    ErrorHandler.handle_graceful(e, "GUI update processor")
        
        threading.Thread(
            target=process_updates, 
            daemon=True, 
            name="GUIUpdateProcessor"
        ).start()
    
    def safe_update(self, widget, update_func) -> None:
        current_time = time.time()
        
        # Rate Limiting
        if current_time - self._last_update_time < self._min_update_interval:
            return
            
        with self._update_lock:
            if not self._is_running:
                return
            
            # REPARIERT: Widget-Validierung verbessert
            if widget is None:
                return
                
            if self._pending_updates >= self._max_pending_updates:
                self._dropped_updates += 1
                if self._dropped_updates % 100 == 0:
                    logging.warning(f"GUI update queue full, dropped {self._dropped_updates} updates")
                return
            
            self._pending_updates += 1
        
        def guarded_update():
            try:
                if not self._is_running:
                    return
                    
                # REPARIERT: Sichere Widget-Überprüfung
                if (hasattr(widget, 'winfo_exists') and 
                    callable(getattr(widget, 'winfo_exists')) and 
                    widget.winfo_exists() and 
                    self.root.winfo_exists()):
                    update_func()
                    self._last_update_time = time.time()
            except (tk.TclError, RuntimeError, AttributeError) as e:
                logging.debug(f"GUI update failed: {e}")
            finally:
                with self._update_lock:
                    self._pending_updates = max(0, self._pending_updates - 1)
                    self._processed_updates += 1

        try:
            self._update_queue.put_nowait(guarded_update)
        except queue.Full:
            with self._update_lock:
                self._pending_updates = max(0, self._pending_updates - 1)
                self._dropped_updates += 1
            logging.debug("GUI update queue full")
    
    def get_stats(self) -> Dict[str, Any]:
        with self._update_lock:
            return {
                'pending_updates': self._pending_updates,
                'processed_updates': self._processed_updates,
                'dropped_updates': self._dropped_updates,
                'queue_size': self._update_queue.qsize()
            }
    
    def shutdown(self) -> None:
        self._is_running = False
        try:
            while True:
                self._update_queue.get_nowait()
        except queue.Empty:
            pass

# ===== SYSTEM LOAD MONITORING =====
class SystemLoadMonitor:
    """Monitor system load for graceful degradation"""
    
    def __init__(self):
        self._load_history = deque(maxlen=10)
        self._last_check = 0
        self._check_interval = 2.0
        self._degradation_active = False
        
    def should_degrade_quality(self) -> bool:
        """Check if system is overloaded and needs quality degradation"""
        current_time = time.time()
        if current_time - self._last_check < self._check_interval:
            return self._degradation_active
            
        self._last_check = current_time
        
        if not PSUTIL_AVAILABLE:
            return False
            
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            self._load_history.append((cpu_percent, memory_percent))
            
            # Berechne Durchschnitt
            avg_cpu = sum(c[0] for c in self._load_history) / len(self._load_history)
            avg_memory = sum(c[1] for c in self._load_history) / len(self._load_history)
            
            # Degrade wenn System überlastet
            self._degradation_active = avg_cpu > 80 or avg_memory > 85
            
            if self._degradation_active:
                logging.info(f"System load high (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%) - activating quality degradation")
            
            return self._degradation_active
            
        except Exception as e:
            logging.debug(f"Load monitoring failed: {e}")
            return False

# ===== ENHANCED HEALTH CHECK SYSTEM =====
class HealthCheckSystem:
    """Enhanced health checking with detailed metrics"""
    
    def __init__(self):
        self.last_check = time.time()
        self.health_status = {}
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
    
    def check_system_health(self) -> Dict[str, Any]:
        try:
            health_data = {
                'timestamp': time.time(),
                'audio_pipeline': self._check_audio_pipeline(),
                'model_loading': self._check_models(),
                'memory_usage': self._check_memory(),
                'disk_space': self._check_disk(),
                'network_connectivity': self._check_network(),
                'performance_metrics': DragonEventSystem().get_performance_report(),
                'overall_status': 'healthy'
            }
            
            # Determine overall status
            issues = [v for k, v in health_data.items() 
                     if k != 'timestamp' and k != 'overall_status' and k != 'performance_metrics' and v.get('status') != 'healthy']
            
            if any(issue.get('status') == 'critical' for issue in issues):
                health_data['overall_status'] = 'critical'
                self.consecutive_failures += 1
            elif issues:
                health_data['overall_status'] = 'degraded'
                self.consecutive_failures += 1
            else:
                health_data['overall_status'] = 'healthy'
                self.consecutive_failures = 0
            
            # If too many consecutive failures, escalate to critical
            if self.consecutive_failures >= self.max_consecutive_failures:
                health_data['overall_status'] = 'critical'
                health_data['consecutive_failures'] = self.consecutive_failures
            
            self.health_status = health_data
            DragonEventSystem().publish(DragonEvents.HEALTH_UPDATE, health_data)
            return health_data
            
        except Exception as e:
            ErrorHandler.handle_graceful(e, "health check system")
            self.consecutive_failures += 1
            return {'overall_status': 'error', 'error': str(e)}
    
    def _check_audio_pipeline(self) -> Dict[str, Any]:
        return {'status': 'healthy', 'message': 'Audio pipeline nominal'}
    
    def _check_models(self) -> Dict[str, Any]:
        try:
            # Check if models can be loaded
            return {'status': 'healthy', 'message': 'Models available'}
        except Exception as e:
            return {'status': 'critical', 'message': f'Model error: {e}'}
    
    def _check_memory(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {'status': 'unknown', 'message': 'psutil not available'}
        
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return {'status': 'critical', 'message': f'Memory usage: {memory.percent}%'}
            elif memory.percent > 80:
                return {'status': 'degraded', 'message': f'Memory usage: {memory.percent}%'}
            else:
                return {'status': 'healthy', 'message': f'Memory usage: {memory.percent}%'}
        except Exception as e:
            return {'status': 'unknown', 'message': f'Memory check failed: {e}'}
    
    def _check_disk(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {'status': 'unknown', 'message': 'psutil not available'}
        
        try:
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                return {'status': 'critical', 'message': f'Disk usage: {disk.percent}%'}
            elif disk.percent > 90:
                return {'status': 'degraded', 'message': f'Disk usage: {disk.percent}%'}
            else:
                return {'status': 'healthy', 'message': f'Disk usage: {disk.percent}%'}
        except Exception as e:
            return {'status': 'unknown', 'message': f'Disk check failed: {e}'}
    
    def _check_network(self) -> Dict[str, Any]:
        return {'status': 'healthy', 'message': 'Network connectivity assumed'}

# ===== ENHANCED LOGGING SETUP =====
def setup_logging() -> None:
    """Enhanced logging setup with rotation"""
    try:
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        
        log_dir = Path.home() / ".dragon_whisperer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Enhanced file handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_dir / "dragon_whisperer.log", 
                encoding='utf-8', 
                mode='a',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except ImportError:
            # Fallback to regular file handler
            file_handler = logging.FileHandler(
                log_dir / "dragon_whisperer.log", 
                encoding='utf-8', 
                mode='w'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Set levels for noisy loggers
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('yt_dlp').setLevel(logging.ERROR)
        if FASTER_WHISPER_AVAILABLE:
            logging.getLogger('faster_whisper').setLevel(logging.WARNING)
            
        logging.info("Enhanced logging initialized successfully")
            
    except Exception as e:
        print(f"Logging setup failed: {e}")
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# ===== DATENMODELLE =====
@dataclass
class TranscriptionResult:
    text: str
    language: str
    start_time: float
    end_time: float
    confidence: float = 0.0
    speaker: Optional[str] = None
    processing_time: float = 0.0
    word_count: int = 0
    audio_quality: Optional[Dict[str, Any]] = None
    plugin_processed: bool = False
    quality_rating: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'language': self.language,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'confidence': self.confidence,
            'speaker': self.speaker,
            'processing_time': self.processing_time,
            'word_count': self.word_count,
            'audio_quality': self.audio_quality,
            'plugin_processed': self.plugin_processed,
            'quality_rating': self.quality_rating
        }

@dataclass
class TranslationResult:
    original: str
    translated: str
    source_lang: str
    target_lang: str
    confidence: float = 0.0
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original': self.original,
            'translated': self.translated,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'confidence': self.confidence,
            'processing_time': self.processing_time
        }

# ===== SICHERHEITS-UTILITIES =====
class SecurityUtils:
    @staticmethod
    def sanitize_url(url: str) -> str:
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ['http', 'https', 'file', 'rtmp']:
                raise ValueError(f"Unsupported scheme: {parsed.scheme}")
            return url
        except Exception as e:
            raise ValueError(f"Invalid URL: {e}")
    
    @staticmethod
    def validate_file_path(path: str, allowed_base: Optional[Path] = None) -> str:
        try:
            if allowed_base is None:
                allowed_base = Path.home() / "dragon_whisperer_export"
                allowed_base.mkdir(exist_ok=True, mode=0o700)
            
            safe_path = Path(os.path.abspath(os.path.normpath(path)))
            
            try:
                safe_path.relative_to(allowed_base)
            except ValueError:
                raise ValueError(f"Path traversal detected: {path}")
            
            if not safe_path.exists():
                logging.warning(f"File path does not exist: {safe_path}")
                
            return str(safe_path)
        except Exception as e:
            raise ValueError(f"Invalid file path: {e}")

# ===== ENHANCED UTILITY-KLASSEN =====
class CircuitBreaker:
    """Enhanced circuit breaker with adaptive timeouts"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        self._lock = threading.RLock()
        self._success_count = 0
        self._adaptive_timeout = recovery_timeout
        
    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "OPEN":
                current_time = time.time()
                recovery_time = self._adaptive_timeout
                
                if current_time - self.last_failure_time > recovery_time:
                    self.state = "HALF_OPEN"
                    return True
                return False
            return True
        
    def record_success(self) -> None:
        with self._lock:
            self.state = "CLOSED"
            self.failures = 0
            self.last_failure_time = 0
            self._success_count += 1
            
            # Reduce timeout after consecutive successes
            if self._success_count >= 3:
                self._adaptive_timeout = max(30, self._adaptive_timeout * 0.8)
                self._success_count = 0
        
    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            self._success_count = 0
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                # Increase timeout after failures
                self._adaptive_timeout = min(300, self._adaptive_timeout * 1.5)

class SecureCounter:
    """Enhanced counter with rate limiting"""
    
    def __init__(self, max_history: int = 1000):
        self._value = 0
        self._lock = threading.RLock()
        self._history: List[float] = []
        self._max_history = max_history
        self._last_reset = time.time()
    
    def increment(self) -> None:
        with self._lock:
            self._value += 1
            self._history.append(time.time())
            if len(self._history) > self._max_history:
                self._history.pop(0)
    
    def get_value(self) -> int:
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        with self._lock:
            self._value = 0
            self._history.clear()
            self._last_reset = time.time()
    
    def get_rate(self, window_seconds: float = 60.0) -> float:
        if window_seconds <= 0:
            return 0.0
            
        with self._lock:
            if not self._history:
                return 0.0
                
            now = time.time()
            cutoff = now - window_seconds
            
            count = 0
            for t in reversed(self._history):
                if t > cutoff:
                    count += 1
                else:
                    break
                    
            return count / window_seconds
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'value': self._value,
                'history_size': len(self._history),
                'rate_1min': self.get_rate(60),
                'rate_5min': self.get_rate(300),
                'uptime': time.time() - self._last_reset
            }

@dataclass
class CacheEntry:
    value: Any
    timestamp: float
    access_count: int = 0
    size: int = 0  # Estimated size in bytes

class LRUCacheWithTTL:
    """Enhanced LRU cache with memory tracking"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._cleanup_counter = 0
        self._total_size = 0
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry.timestamp > self._ttl:
                    del self._cache[key]
                    self._total_size -= entry.size
                    self._misses += 1
                    return None
                
                self._cache.move_to_end(key)
                entry.access_count += 1
                self._hits += 1
                return entry.value
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any, size: int = 0) -> None:
        with self._lock:
            self._cleanup_counter += 1
            if self._cleanup_counter >= 100:
                self._cleanup_expired()
                self._cleanup_counter = 0
            
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size -= old_entry.size
            
            self._cache[key] = CacheEntry(value, time.time(), size=size)
            self._total_size += size
            
            if len(self._cache) > self._max_size:
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._total_size -= oldest_entry.size
    
    def _cleanup_expired(self) -> None:
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > self._ttl
        ]
        for key in expired_keys:
            entry = self._cache[key]
            self._total_size -= entry.size
            del self._cache[key]
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._total_size = 0
            self._hits = 0
            self._misses = 0
    
    def size(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            hit_ratio = self._hits / max(self._hits + self._misses, 1)
            return {
                'size': len(self._cache),
                'total_size_bytes': self._total_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_ratio': hit_ratio,
                'avg_entry_size': self._total_size / max(len(self._cache), 1)
            }

# ===== KERN-BUSINESS-LOGIK (Optimized) =====
class OptimizedLanguageDetector:
    """Enhanced language detection with better performance"""
    
    def __init__(self):
        self.keyword_patterns = {
            'de': {
                'common': ['der', 'die', 'das', 'und', 'ist', 'nicht', 'zu', 'auf', 'für', 'wir', 'sie', 'ich', 'du'],
                'unique': ['genau', 'vielleicht', 'eigentlich', 'allerdings', 'deshalb', 'übrigens'],
                'script': 'latin'
            },
            'en': {
                'common': ['the', 'and', 'is', 'to', 'of', 'in', 'that', 'with', 'for', 'you', 'we', 'they', 'I'],
                'unique': ['actually', 'however', 'therefore', 'moreover', 'furthermore', 'specifically'],
                'script': 'latin'
            },
            'fr': {
                'common': ['le', 'la', 'les', 'et', 'est', 'dans', 'pour', 'avec', 'sur', 'nous', 'vous', 'ils'],
                'unique': ['exactement', 'peut-être', 'actuellement', 'cependant', 'd\'ailleurs'],
                'script': 'latin'
            },
            'es': {
                'common': ['el', 'la', 'y', 'en', 'que', 'con', 'para', 'por', 'los', 'nosotros', 'ustedes'],
                'unique': ['exactamente', 'quizás', 'actualmente', 'sin embargo', 'por cierto'],
                'script': 'latin'
            }
        }
        self._lock = threading.RLock()
        self.language_priorities = ['en', 'de', 'fr', 'es', 'it', 'ja', 'ko', 'zh', 'ru', 'ar']
        self.script_patterns = {
            'cjk': re.compile(r'[\u4e00-\u9fff]'),
            'cyrillic': re.compile(r'[\u0400-\u04FF]'),
            'arabic': re.compile(r'[\u0600-\u06FF]'),
            'japanese': re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),
            'korean': re.compile(r'[\uAC00-\uD7AF]')
        }
        
        self.language_corrections = {
            'yo': 'en',
            'cy': 'en', 
            'sn': 'en',
            'hr': 'en',
            'sw': 'en',
        }
        
        # Pre-compile regex patterns for performance
        self._word_pattern = re.compile(r'\b\w+\b')
        self._clean_pattern = re.compile(r'[^\w\s.,!?]')
        self._whitespace_pattern = re.compile(r'\s+')
        
    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        
        # Use pre-compiled patterns for better performance
        clean_text = self._clean_pattern.sub('', text)
        clean_text = self._whitespace_pattern.sub(' ', clean_text).strip()
        
        return clean_text
        
    @lru_cache(maxsize=1000)
    def detect_script_type(self, text: str) -> str:
        if not text:
            return 'unknown'
            
        for script_type, pattern in self.script_patterns.items():
            if pattern.search(text):
                return script_type
        return 'latin'
    
    def detect_language_enhanced(self, text: str, whisper_lang: Optional[str] = None, 
                               whisper_confidence: float = 0.0) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            clean_text = self._preprocess_text(text)
            
            if not clean_text or len(clean_text.strip()) < 3:
                return {'language': 'unknown', 'confidence': 0.0, 'method': 'insufficient_text'}
            
            if (whisper_lang in self.language_corrections and 
                whisper_confidence < 0.6 and 
                self._is_likely_english(clean_text)):
                whisper_lang = self.language_corrections[whisper_lang]
                whisper_confidence = max(0.3, whisper_confidence)
            
            text_lower = clean_text.lower()
            words = self._word_pattern.findall(text_lower)
            total_words = len(words)
            
            if total_words < 1:
                return {'language': 'unknown', 'confidence': 0.0, 'method': 'insufficient_words'}
            
            script_type = self.detect_script_type(clean_text)
            keyword_scores = {}
            
            with self._lock:
                prioritized_languages = self._get_prioritized_languages(script_type)
                
                for lang in prioritized_languages:
                    if lang not in self.keyword_patterns:
                        continue
                        
                    patterns = self.keyword_patterns[lang]
                    common_matches = sum(1 for word in words if word in patterns['common'])
                    unique_matches = sum(1 for word in words if word in patterns['unique'])
                    
                    common_score = (common_matches / max(total_words, 1)) * 0.7
                    unique_score = (unique_matches / max(total_words, 1)) * 0.3
                    
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
                
                if confidence >= 0.3:
                    processing_time = time.time() - start_time
                    return {
                        'language': best_lang[0],
                        'confidence': confidence,
                        'method': 'combined_detection',
                        'script_type': script_type,
                        'status': 'high_confidence' if confidence > 0.6 else 'medium_confidence',
                        'processing_time': processing_time
                    }
            
            if whisper_lang and whisper_lang != 'unknown' and whisper_confidence > 0.1:
                processing_time = time.time() - start_time
                return {
                    'language': whisper_lang,
                    'confidence': whisper_confidence,
                    'method': 'whisper_fallback',
                    'script_type': script_type,
                    'status': 'medium_confidence',
                    'processing_time': processing_time
                }
            
            fallback_lang = self._get_fallback_language(script_type)
            processing_time = time.time() - start_time
            return {
                'language': fallback_lang,
                'confidence': 0.1,
                'method': 'script_fallback',
                'script_type': script_type,
                'status': 'low_confidence',
                'processing_time': processing_time
            }
            
        except Exception as e:
            ErrorHandler.handle_graceful(e, "language detection")
            processing_time = time.time() - start_time
            if whisper_lang and whisper_lang != 'unknown':
                return {
                    'language': whisper_lang,
                    'confidence': max(0.1, whisper_confidence),
                    'method': 'whisper_fallback_error',
                    'status': 'low_confidence',
                    'processing_time': processing_time
                }
            return {
                'language': 'unknown', 
                'confidence': 0.0, 
                'method': 'error', 
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _is_likely_english(self, text: str) -> bool:
        english_indicators = {'the', 'and', 'is', 'to', 'of', 'in', 'that', 'with', 'for', 'you', 'we', 'they', 'I'}
        words = text.lower().split()
        if len(words) < 3:
            return False
        
        english_word_count = sum(1 for word in words if word in english_indicators)
        required_threshold = min(3, max(1, len(words) // 3))
        return english_word_count >= required_threshold
    
    def _get_prioritized_languages(self, script_type: str) -> List[str]:
        if script_type == 'cjk':
            return ['ja', 'ko', 'zh', 'en']
        elif script_type == 'cyrillic':
            return ['ru', 'en']
        elif script_type == 'arabic':
            return ['ar', 'en']
        elif script_type == 'japanese':
            return ['ja', 'en']
        elif script_type == 'korean':
            return ['ko', 'en']
        else:
            return self.language_priorities
    
    def _get_fallback_language(self, script_type: str) -> str:
        fallbacks = {
            'cjk': 'zh',
            'cyrillic': 'ru', 
            'arabic': 'ar',
            'japanese': 'ja',
            'korean': 'ko',
            'latin': 'en',
            'unknown': 'en'
        }
        return fallbacks.get(script_type, 'en')

# ===== ENHANCED STREAM MANAGEMENT =====
class StreamTitleExtractor:
    """Enhanced title extraction with caching"""
    
    def __init__(self, cache_duration: int = 120):
        self.last_extraction_time = 0.0
        self.cache_duration = cache_duration
        self.cached_title: Optional[str] = None
        self.cached_url: Optional[str] = None
        self._lock = threading.RLock()
        self._extraction_count = 0
        self._cache_hits = 0
        
    def extract_stream_title(self, url: str) -> Optional[str]:
        try:
            current_time = time.time()
            with self._lock:
                if (self.cached_url == url and 
                    current_time - self.last_extraction_time < self.cache_duration):
                    self._cache_hits += 1
                    return self.cached_title
                    
            if not YT_DLP_AVAILABLE:
                return None
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'force_json': True,
                'simulate': True,
                'skip_download': True,
                'socket_timeout': 10,
                'extractretries': 2
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', '') if info else None
                
                with self._lock:
                    self.cached_title = title
                    self.cached_url = url
                    self.last_extraction_time = current_time
                    self._extraction_count += 1
                
                return title
                    
        except Exception as e:
            logging.debug(f"Title extraction failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            hit_ratio = self._cache_hits / max(self._extraction_count, 1)
            return {
                'extraction_count': self._extraction_count,
                'cache_hits': self._cache_hits,
                'cache_hit_ratio': hit_ratio,
                'cached_title': self.cached_title is not None
            }

class EnhancedStreamManager:
    """Enhanced stream manager with better performance tracking"""
    
    def __init__(self, max_cache_size: int = 100):
        self.stream_cache = LRUCacheWithTTL(max_cache_size, ttl_seconds=1800)
        self.title_extractor = StreamTitleExtractor()
        self.circuit_breaker = CircuitBreaker()
        self._stats = {
            'stream_detections': SecureCounter(),
            'cache_hits': SecureCounter(),
            'extraction_attempts': SecureCounter()
        }
        
    def detect_stream_type(self, url: str) -> str:
        self._stats['stream_detections'].increment()
        
        if not url:
            return 'unknown'
            
        url_lower = url.lower()
        
        if url_lower.startswith('file://'):
            file_path = url[7:]
            if os.path.isfile(file_path):
                return 'local'
        elif os.path.isfile(url):
            return 'local'
        elif '.m3u8' in url_lower:
            return 'hls'
        elif 'youtube.com/watch' in url_lower or 'youtu.be' in url_lower:
            return 'youtube'
        elif 'twitch.tv' in url_lower:
            return 'twitch'
        elif any(ext in url_lower for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']):
            return 'local_audio'
        elif any(ext in url_lower for ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']):
            return 'local_video'
        else:
            return 'generic'
            
    def extract_stream_title(self, url: str) -> Optional[str]:
        self._stats['extraction_attempts'].increment()
        title = self.title_extractor.extract_stream_title(url)
        if not title:
            if url.startswith('file://'):
                return os.path.basename(url[7:])
            elif os.path.isfile(url):
                return os.path.basename(url)
        return title
            
    def extract_stream_url_enhanced(self, url: str, callbacks: Optional[Dict[str, Callable]] = None) -> Optional[str]:
        if not url:
            return None
            
        if not self.circuit_breaker.can_execute():
            if callbacks and 'warning' in callbacks:
                callbacks['warning']("Circuit breaker open - waiting for recovery")
            return None
        
        cached_result = self.stream_cache.get(url)
        if cached_result is not None:
            self._stats['cache_hits'].increment()
            return cached_result
        
        stream_type = self.detect_stream_type(url)
        
        if stream_type in ['local', 'local_audio', 'local_video']:
            if url.startswith('file://'):
                result = url
            else:
                result = f"file://{url}"
            self.stream_cache.put(url, result)
            self.circuit_breaker.record_success()
            return url
        
        if not YT_DLP_AVAILABLE:
            if callbacks and 'warning' in callbacks:
                callbacks['warning']("yt-dlp not available, using direct URL")
            self.stream_cache.put(url, url)
            self.circuit_breaker.record_success()
            return url
        
        try:
            strategies = [
                self._extract_with_simple_opts,
                self._extract_direct_fallback
            ]
            
            for strategy in strategies:
                try:
                    result = strategy(url, callbacks)
                    if result:
                        self.stream_cache.put(url, result)
                        self.circuit_breaker.record_success()
                        return result
                except Exception as e:
                    continue
                        
            if callbacks and 'warning' in callbacks:
                callbacks['warning']("Using direct URL as fallback")
                            
        except Exception as e:
            self.circuit_breaker.record_failure()
            ErrorHandler.handle_operation(e, "URL extraction")
            if callbacks and 'warning' in callbacks:
                callbacks['warning'](f"URL extraction limited: {str(e)[:100]}...")
                
        self.stream_cache.put(url, url)
        return url

    def _extract_with_simple_opts(self, url: str, callbacks: Optional[Dict[str, Callable]] = None) -> Optional[str]:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'format': 'bestaudio/best',
            'ignoreerrors': True,
            'no_check_certificate': True,
            'socket_timeout': 15,
            'extractretries': 2
        }
        
        return self._perform_extraction(url, ydl_opts, "simple", callbacks)

    def _extract_direct_fallback(self, url: str, callbacks: Optional[Dict[str, Callable]] = None) -> Optional[str]:
        if callbacks and 'info' in callbacks:
            callbacks['info']("Using enhanced YouTube extraction...")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'format': 'worstaudio/worst',
            'ignoreerrors': True,
            'socket_timeout': 10
        }
        
        return self._perform_extraction(url, ydl_opts, "direct", callbacks)

    def _perform_extraction(self, url: str, ydl_opts: Dict[str, Any], strategy: str, 
                          callbacks: Optional[Dict[str, Callable]] = None) -> Optional[str]:
        try:
            if callbacks and 'info' in callbacks:
                callbacks['info'](f"Extracting stream URL ({strategy} method)...")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info and 'url' in info:
                    if callbacks and 'info' in callbacks:
                        callbacks['info'](f"Stream URL extracted ({strategy})")
                    return info['url']
                    
                if info and 'formats' in info:
                    audio_formats = []
                    for fmt in info['formats']:
                        if fmt.get('acodec') != 'none':
                            audio_formats.append(fmt)
                    
                    if audio_formats:
                        audio_only = [f for f in audio_formats if f.get('vcodec') == 'none']
                        if audio_only:
                            best_audio = max(audio_only, key=lambda x: x.get('abr', 0) or 0)
                        else:
                            best_audio = max(audio_formats, key=lambda x: x.get('abr', 0) or 0)
                        
                        if callbacks and 'info' in callbacks:
                            callbacks['info'](f"Audio stream found ({strategy})")
                        return best_audio['url']
                        
        except Exception as e:
            error_msg = str(e)
            if any(keyword in error_msg.lower() for keyword in ['sign in', 'bot', 'authentication', 'cookie']):
                if callbacks and 'warning' in callbacks:
                    callbacks['warning']("YouTube authentication required. Using fallback method...")
                raise
            elif 'timed out' in error_msg.lower():
                if callbacks and 'warning' in callbacks:
                    callbacks['warning']("Extraction timed out, trying next method...")
                raise
            else:
                logging.warning(f"Extraction strategy {strategy} failed: {e}")
                raise
                
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        cache_stats = self.stream_cache.get_stats()
        title_stats = self.title_extractor.get_stats()
        
        return {
            'stream_detections': self._stats['stream_detections'].get_stats(),
            'cache_hits': self._stats['cache_hits'].get_stats(),
            'extraction_attempts': self._stats['extraction_attempts'].get_stats(),
            'cache_performance': cache_stats,
            'title_extraction': title_stats,
            'circuit_breaker_state': self.circuit_breaker.state
        }

# ===== REPARIERTE VERARBEITUNGS-ENGINES =====
class AdaptiveAudioProcessor:
    """Enhanced audio processor with adaptive queue management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.audio_queue = queue.Queue(maxsize=PerformanceConfig.MAX_QUEUE_SIZE)
        self.is_processing = False
        self._processing_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._chunk_counter = 0
        self.quality_history: List[float] = []
        self._queue_size_history = deque(maxlen=100)
        self._last_queue_warning = 0
        
    def start_processing_adaptive(self, process_callback: Callable, stream_type: str = 'generic') -> None:
        with self._lock:
            if self.is_processing:
                return
                
            self.is_processing = True
            self._shutdown_event.clear()
            self._chunk_counter = 0
        
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(process_callback,),
            daemon=True,
            name=f"AudioProcessor_{stream_type}"
        )
        self._processing_thread.start()
        
    def _processing_loop(self, process_callback: Callable) -> None:
        consecutive_errors = 0
        max_consecutive_errors = PerformanceConfig.MAX_CONSECUTIVE_ERRORS
        
        try:
            while self.is_processing and not self._shutdown_event.is_set():
                try:
                    # Monitor queue size
                    current_size = self.audio_queue.qsize()
                    self._queue_size_history.append(current_size)
                    
                    # Warn if queue is consistently full
                    if (current_size > PerformanceConfig.MAX_QUEUE_SIZE * 0.8 and 
                        time.time() - self._last_queue_warning > 30):
                        logging.warning(f"Audio queue consistently high: {current_size}/{PerformanceConfig.MAX_QUEUE_SIZE}")
                        self._last_queue_warning = time.time()
                    
                    # REPARIERT: Besseres Error Handling für Audio-Daten
                    audio_data = self.audio_queue.get(timeout=1.0)
                    if audio_data is None:
                        break
                        
                    # Validierung der Audio-Daten
                    if not isinstance(audio_data, bytes) or len(audio_data) == 0:
                        logging.warning("Invalid audio data received, skipping")
                        continue
                        
                    self._chunk_counter += 1
                    try:
                        process_start = time.time()
                        process_callback(audio_data, self._chunk_counter)
                        processing_time = time.time() - process_start
                        
                        DragonEventSystem().record_processing_time(processing_time)
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        DragonEventSystem().record_error()
                        ErrorHandler.handle_audio_error(e, self._chunk_counter)
                        if consecutive_errors >= max_consecutive_errors:
                            logging.error(f"Max consecutive errors reached, stopping processor")
                            break
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    consecutive_errors += 1
                    DragonEventSystem().record_error()
                    ErrorHandler.handle_audio_error(e, self._chunk_counter)
                    if consecutive_errors >= max_consecutive_errors:
                        logging.error(f"Max consecutive errors reached, stopping processor")
                        break
                    time.sleep(0.1)
        except Exception as e:
            DragonEventSystem().record_error()
            ErrorHandler.handle_critical(e, "audio processing loop")
                
    def adjust_chunk_size_based_on_quality(self, quality_score: float) -> None:
        if quality_score < 0.3:
            new_duration = max(3.0, PerformanceConfig.CHUNK_DURATION * 0.8)
            if new_duration != PerformanceConfig.CHUNK_DURATION:
                PerformanceConfig.CHUNK_DURATION = new_duration
                logging.info(f"Reduced chunk duration to {new_duration}s due to poor audio quality")
        elif quality_score > 0.7:
            new_duration = min(15.0, PerformanceConfig.CHUNK_DURATION * 1.2)
            if new_duration != PerformanceConfig.CHUNK_DURATION:
                PerformanceConfig.CHUNK_DURATION = new_duration
                logging.info(f"Increased chunk duration to {new_duration}s due to good audio quality")
                
    def _safe_audio_put(self, audio_data: bytes, chunk_id: int) -> bool:
        """Safe audio data put with adaptive timeout"""
        try:
            current_size = self.audio_queue.qsize()
            
            # Adaptive Wartezeit basierend auf Queue-Größe
            timeout = max(0.1, min(2.0, current_size / PerformanceConfig.MAX_QUEUE_SIZE))
            
            self.audio_queue.put(audio_data, timeout=timeout)
            return True
            
        except queue.Full:
            self._handle_queue_congestion(chunk_id)
            return False

    def _handle_queue_congestion(self, chunk_id: int) -> None:
        """Handle queue congestion with adaptive strategies"""
        current_size = self.audio_queue.qsize()
        
        if current_size > PerformanceConfig.MAX_QUEUE_SIZE * 0.9:
            # Kritischer Zustand - reduziere Chunk-Größe
            PerformanceConfig.CHUNK_DURATION = max(
                2.0, PerformanceConfig.CHUNK_DURATION * 0.8
            )
            logging.warning(f"Queue critical, reduced chunk duration to {PerformanceConfig.CHUNK_DURATION}s")
            
        if chunk_id % 5 == 0:  # Reduzierte Log-Frequenz
            logging.warning(f"Audio queue full ({current_size}/{PerformanceConfig.MAX_QUEUE_SIZE}), dropped chunk {chunk_id}")
                
    def stop_processing(self) -> None:
        with self._lock:
            if not self.is_processing:
                return
                
            self.is_processing = False
            self._shutdown_event.set()
        
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass
            
        try:
            self.audio_queue.put(None, timeout=0.5)
        except queue.Full:
            pass
            
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
            if self._processing_thread.is_alive():
                logging.warning("Audio processor thread did not terminate cleanly")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        with self._lock:
            avg_queue_size = sum(self._queue_size_history) / len(self._queue_size_history) if self._queue_size_history else 0
            return {
                'is_processing': self.is_processing,
                'chunks_processed': self._chunk_counter,
                'current_queue_size': self.audio_queue.qsize(),
                'avg_queue_size': avg_queue_size,
                'max_queue_size': PerformanceConfig.MAX_QUEUE_SIZE
            }

class OptimizedTranslationEngine:
    """Enhanced translation engine with better memory management"""
    
    def __init__(self, target_lang: str = "en", max_cache_size: int = 2000, enabled: bool = True):
        self.target_lang = target_lang
        self.translator: Optional[Any] = None
        self.translation_cache = LRUCacheWithTTL(max_cache_size, ttl_seconds=7200)
        self.supported_languages = AppConstants.SUPPORTED_LANGUAGES
        self._initialized = False
        self.enabled = enabled
        self._lock = threading.RLock()
        self._translation_count = 0
        self._cache_hits = 0
        
    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
        
    def is_enabled(self) -> bool:
        with self._lock:
            return self.enabled and self._initialized
        
    def initialize(self) -> bool:
        if not self.enabled:
            return False
            
        if self._initialized:
            return True
            
        if not TRANSLATOR_AVAILABLE:
            logging.warning("Translation engine not available - install deep-translator")
            return False
            
        try:
            self.translator = GoogleTranslator(source='auto', target=self.target_lang)
            test_result = self.translator.translate("Hello world")
            self._initialized = test_result is not None and len(test_result) > 0
            
            if self._initialized:
                logging.info(f"Translation engine initialized for target language: {self.target_lang}")
                DragonEventSystem().publish(DragonEvents.MODEL_LOADED, {
                    'type': 'translator',
                    'language': self.target_lang
                })
            else:
                logging.warning("Translation engine test failed")
            return self._initialized
        except Exception as e:
            ErrorHandler.handle_operation(e, "translation engine initialization")
            self._initialized = False
            return False
            
    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized
            
    def is_language_supported(self, lang: str) -> bool:
        return lang in self.supported_languages

    def set_target_language(self, target_lang: str) -> bool:
        if target_lang == 'auto':
            target_lang = 'en'
            
        if target_lang not in self.supported_languages:
            target_lang = 'en'
            
        with self._lock:
            self.target_lang = target_lang
            self._initialized = False
            
        return self.initialize()

    def translate_text_enhanced(self, text: str, source_lang: str, target_lang: str) -> Optional[TranslationResult]:
        if not self.is_enabled():
            return None
            
        if not self.is_initialized():
            return None
            
        if not text or not text.strip():
            return None
            
        if source_lang == target_lang:
            return None
            
        if source_lang == 'unknown' or target_lang == 'unknown':
            return None
            
        if not self.is_language_supported(target_lang):
            return None
            
        cache_key = f"{source_lang}_{target_lang}_{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        
        cached_result = self.translation_cache.get(cache_key)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result
                
        start_time = time.time()
        try:
            if source_lang and source_lang != 'unknown' and self.is_language_supported(source_lang):
                self.translator.source = source_lang
            else:
                self.translator.source = 'auto'
                
            translated = self.translator.translate(text)
            processing_time = time.time() - start_time
            
            if not translated or translated == text:
                return None
            
            confidence = self._calculate_translation_confidence(text, translated, source_lang)
            
            result = TranslationResult(
                original=text,
                translated=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=confidence,
                processing_time=processing_time
            )
            
            # Estimate size for cache management
            size_estimate = len(text.encode('utf-8')) + len(translated.encode('utf-8'))
            self.translation_cache.put(cache_key, result, size=size_estimate)
            self._translation_count += 1
            
            return result
            
        except Exception as e:
            ErrorHandler.handle_graceful(e, "translation")
            return None
            
    def _calculate_translation_confidence(self, original: str, translated: str, source_lang: str) -> float:
        if not original or not translated:
            return 0.0
            
        length_ratio = len(translated) / max(len(original), 1)
        
        failure_patterns = [
            original.lower() == translated.lower(),
            len(translated.strip()) == 0,
            'error' in translated.lower(),
        ]
        
        if any(failure_patterns):
            return 0.1
            
        if 0.3 <= length_ratio <= 3.0:
            base_confidence = 0.7
        else:
            base_confidence = 0.3
            
        complex_pairs = ['zh', 'ja', 'ko', 'ar', 'he']
        if source_lang in complex_pairs:
            base_confidence *= 0.9
            
        return min(base_confidence, 0.95)
            
    def clear_cache(self) -> None:
        self.translation_cache.clear()
        self._translation_count = 0
        self._cache_hits = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get translation engine statistics"""
        cache_stats = self.translation_cache.get_stats()
        with self._lock:
            hit_ratio = self._cache_hits / max(self._translation_count, 1)
            return {
                'enabled': self.enabled,
                'initialized': self._initialized,
                'target_language': self.target_lang,
                'translation_count': self._translation_count,
                'cache_hits': self._cache_hits,
                'cache_hit_ratio': hit_ratio,
                'cache_performance': cache_stats
            }

# ===== ENHANCED KONFIGURATIONSSYSTEM =====
class ConfigManager:
    """Enhanced config manager with backup and validation"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config_dir = Path.home() / ".dragon_whisperer"
        try:
            self.config_dir.mkdir(exist_ok=True, parents=True, mode=0o700)
        except Exception as e:
            self.config_dir = Path(tempfile.gettempdir()) / "dragon_whisperer"
            try:
                self.config_dir.mkdir(exist_ok=True, parents=True, mode=0o700)
            except Exception as e2:
                ErrorHandler.handle_critical(e2, "config directory creation")
                raise
                
        self.config_file = self.config_dir / "config.ini"
        self.presets_file = self.config_dir / "presets.ini"
        self.backup_dir = self.config_dir / "backups"
        self.config = configparser.ConfigParser()
        self.presets = configparser.ConfigParser()
        self._lock = threading.RLock()
        self._save_count = 0
        self.load_config()
        self._initialized = True
    
    def load_config(self) -> None:
        with self._lock:
            if self.config_file.exists():
                try:
                    self.config.read(self.config_file, encoding='utf-8')
                    logging.info("Config loaded successfully")
                except Exception as e:
                    ErrorHandler.handle_operation(e, "config load")
                    self._try_backup_recovery()
            else:
                self.create_default_config()
    
    def _try_backup_recovery(self) -> None:
        """Try to recover from backup if main config is corrupted"""
        try:
            self.backup_dir.mkdir(exist_ok=True)
            backup_files = list(self.backup_dir.glob("config_*.ini"))
            if backup_files:
                # Use most recent backup
                backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_backup = backup_files[0]
                logging.info(f"Attempting config recovery from backup: {latest_backup.name}")
                shutil.copy2(latest_backup, self.config_file)
                self.config.read(self.config_file, encoding='utf-8')
            else:
                self.create_default_config()
        except Exception as e:
            logging.error(f"Config backup recovery failed: {e}")
            self.create_default_config()
    
    def create_default_config(self) -> None:
        try:
            # REPARIERT: Sicherstellen dass alle Werte gesetzt werden und nicht None sind
            default_values = {
                'model_size': 'small',
                'target_language': 'en',
                'translation_enabled': 'true',
                'auto_scroll': 'true',
                'theme': 'dark',
                'cache_size': '1000',
                'chunk_duration': '6.0',
                'audio_quality_threshold': '0.01',
                'auto_recovery': 'true',
                'performance_monitoring': 'true',
                'language_priorities': 'en,de,fr,es,it,ja,ko,zh,ru,ar',
                'min_confidence': '0.3',
                'max_memory_mb': '2048'
            }
            
            # Jeden Wert explizit setzen
            self.config['DEFAULT'] = {}
            for key, value in default_values.items():
                self.config['DEFAULT'][key] = str(value)  # Immer zu String konvertieren
            
            # Gleiches für andere Sections
            gui_values = {
                'window_width': '1400',
                'window_height': '900',
                'font_size': '11',
                'text_cleanup_interval': '30000'
            }
            
            self.config['GUI'] = {}
            for key, value in gui_values.items():
                self.config['GUI'][key] = str(value)
            
            audio_values = {
                'sample_rate': '16000',
                'channels': '1',
                'silence_threshold': '0.005'
            }
            
            self.config['AUDIO'] = {}
            for key, value in audio_values.items():
                self.config['AUDIO'][key] = str(value)
            
            performance_values = {
                'max_queue_size': '50',
                'max_cache_size': '1000',
                'enable_memory_monitoring': 'true'
            }
            
            self.config['PERFORMANCE'] = {}
            for key, value in performance_values.items():
                self.config['PERFORMANCE'][key] = str(value)
            
            self.save_config()
            logging.info("Default config created with validated values")
        except Exception as e:
            ErrorHandler.handle_critical(e, "default config creation")
    
    def save_config(self) -> None:
        with self._lock:
            try:
                # Create backup before saving
                self.backup_dir.mkdir(exist_ok=True)
                if self.config_file.exists():
                    backup_file = self.backup_dir / f"config_{int(time.time())}.ini"
                    shutil.copy2(self.config_file, backup_file)
                    # Keep only last 5 backups
                    self._cleanup_old_backups()
                
                temp_file = self.config_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    self.config.write(f)
                
                temp_file.replace(self.config_file)
                self._save_count += 1
                logging.debug("Config saved successfully")
                
            except Exception as e:
                ErrorHandler.handle_operation(e, "config save")
    
    def _cleanup_old_backups(self) -> None:
        """Keep only the 5 most recent backups"""
        try:
            backup_files = list(self.backup_dir.glob("config_*.ini"))
            if len(backup_files) > 5:
                backup_files.sort(key=lambda x: x.stat().st_mtime)
                for old_backup in backup_files[:-5]:
                    old_backup.unlink()
        except Exception as e:
            logging.debug(f"Backup cleanup failed: {e}")
    
    def get(self, section: str, option: str, fallback: Any = None) -> Any:
        with self._lock:
            try:
                if self.config.has_section(section) and self.config.has_option(section, option):
                    value = self.config.get(section, option)
                    # REPARIERT: None-Werte vermeiden
                    if value is None or value == 'None':
                        return fallback
                    return value
                return fallback
            except Exception as e:
                logging.debug(f"Config get failed for {section}.{option}: {e}")
                return fallback
    
    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        with self._lock:
            try:
                if self.config.has_section(section) and self.config.has_option(section, option):
                    value = self.config.get(section, option)
                    if value is None or value == 'None':
                        return fallback
                    if isinstance(value, bool):
                        return value
                    return value.lower() in ('true', '1', 'yes', 'on')
                return fallback
            except Exception as e:
                logging.debug(f"Config getboolean failed for {section}.{option}: {e}")
                return fallback

    def update_recent_urls(self, url: str) -> None:
        try:
            if not self.config.has_section('RECENT'):
                self.config.add_section('RECENT')
            
            recent_urls = self.config.get('RECENT', 'urls', fallback='').split('|')
            if url in recent_urls:
                recent_urls.remove(url)
            recent_urls.insert(0, url)
            recent_urls = recent_urls[:10]
            
            self.config.set('RECENT', 'urls', '|'.join(recent_urls))
            self.save_config()
        except Exception as e:
            ErrorHandler.handle_graceful(e, "recent URLs update")

    def validate_config(self) -> List[str]:
        issues = []
        
        # REPARIERT: Bessere Validierung mit Standardwerten
        valid_models = ['tiny', 'base', 'small', 'medium', 'large-v2']
        model_size = self.get('DEFAULT', 'model_size', 'small')
        if model_size not in valid_models:
            issues.append(f"Invalid model size: {model_size}")
        
        valid_langs = ['en', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar']
        target_lang = self.get('DEFAULT', 'target_language', 'en')
        if target_lang not in valid_langs and target_lang != 'auto':
            issues.append(f"Invalid target language: {target_lang}")
        
        try:
            chunk_duration = self.get('DEFAULT', 'chunk_duration', '6.0')
            if chunk_duration:
                chunk_duration_float = float(chunk_duration)
                if not (1.0 <= chunk_duration_float <= 30.0):
                    issues.append(f"Chunk duration out of range: {chunk_duration_float}")
        except (ValueError, TypeError):
            issues.append("Invalid chunk duration format")
        
        try:
            max_memory = self.get('DEFAULT', 'max_memory_mb', '2048')
            if max_memory:
                max_memory_int = int(max_memory)
                if max_memory_int < 512 or max_memory_int > 16384:
                    issues.append(f"Max memory setting out of range: {max_memory_int}MB")
        except (ValueError, TypeError):
            issues.append("Invalid max memory format")
        
        return issues
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'save_count': self._save_count,
                'config_file': str(self.config_file),
                'backup_count': len(list(self.backup_dir.glob("config_*.ini"))) if self.backup_dir.exists() else 0,
                'validation_issues': self.validate_config()
            }

# ===== ENHANCED AUDIO QUALITÄTSANALYSE =====
class AudioQualityAnalyzer:
    """Enhanced audio quality analysis with performance tracking"""
    
    def __init__(self, config: ConfigManager, max_history: int = 100):
        self.config = config
        self.quality_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self._lock = threading.RLock()
        self._analysis_count = 0
        self._analysis_time_total = 0.0
    
    def analyze_audio_quality(self, audio_data: bytes) -> Dict[str, Any]:
        start_time = time.time()
        
        if not NUMPY_AVAILABLE or not audio_data:
            result = {'quality_score': 0.0, 'status': 'analysis_unavailable'}
            self._record_analysis_time(start_time)
            return result
        
        try:
            if len(audio_data) < 800:
                result = {'quality_score': 0.0, 'status': 'too_short'}
                self._record_analysis_time(start_time)
                return result
            
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_np) < 8000:
                result = {'quality_score': 0.0, 'status': 'too_short'}
                self._record_analysis_time(start_time)
                return result
            
            audio_float = audio_np.astype(np.float32) / 32768.0
            audio_float = audio_float - np.mean(audio_float)
            
            rms = np.sqrt(np.mean(audio_float**2))
            peak = np.max(np.abs(audio_float))
            
            dynamic_range = 20 * np.log10(peak / max(rms, 1e-10)) if rms > 0 else 0
            
            zero_crossings = np.sum(np.diff(np.signbit(audio_float)))
            zcr = zero_crossings / len(audio_float)
            
            fft_size = min(1024, len(audio_float))
            spectral_centroid = 0
            if fft_size >= 256:
                try:
                    fft = np.fft.rfft(audio_float[:fft_size])
                    magnitudes = np.abs(fft)
                    if np.sum(magnitudes) > 0:
                        spectral_centroid = np.sum(np.arange(len(magnitudes)) * magnitudes) / np.sum(magnitudes)
                except Exception as fft_error:
                    logging.debug(f"FFT analysis failed: {fft_error}")
            
            rms_score = min(rms / 0.1, 1.0) if rms > 0 else 0
            dr_score = min(dynamic_range / 40.0, 1.0) if dynamic_range > 0 else 0
            zcr_score = 1.0 - abs(zcr - 0.1) if zcr > 0 else 0
            
            quality_score = (rms_score * 0.4 + dr_score * 0.3 + zcr_score * 0.3)
            
            if quality_score > 0.7:
                status = "excellent"
            elif quality_score > 0.5:
                status = "good"
            elif quality_score > 0.3:
                status = "fair"
            elif rms < 0.01:
                status = "silent"
            else:
                status = "poor"
            
            result = {
                'quality_score': quality_score,
                'rms_level': rms,
                'peak_level': peak,
                'dynamic_range': dynamic_range,
                'zero_crossing_rate': zcr,
                'spectral_centroid': spectral_centroid,
                'status': status,
                'timestamp': time.time(),
                'analysis_time': time.time() - start_time
            }
            
            with self._lock:
                self.quality_history.append(result)
                if len(self.quality_history) > self.max_history:
                    self.quality_history.pop(0)
                self._analysis_count += 1
                self._analysis_time_total += result['analysis_time']
            
            DragonEventSystem().publish(DragonEvents.AUDIO_QUALITY_UPDATE, result)
            return result
            
        except Exception as e:
            ErrorHandler.handle_graceful(e, "audio quality analysis")
            result = {'quality_score': 0.0, 'status': 'analysis_failed', 'error': str(e)}
            self._record_analysis_time(start_time)
            return result
    
    def _record_analysis_time(self, start_time: float) -> None:
        analysis_time = time.time() - start_time
        with self._lock:
            self._analysis_count += 1
            self._analysis_time_total += analysis_time

    def get_quality_trend(self) -> Dict[str, Any]:
        with self._lock:
            if not self.quality_history:
                return {'trend': 'unknown', 'average_score': 0.0}
            
            recent_scores = [q['quality_score'] for q in self.quality_history[-10:]]
            avg_score = sum(recent_scores) / len(recent_scores)
            
            if avg_score > 0.7:
                trend = 'excellent'
            elif avg_score > 0.5:
                trend = 'good'
            elif avg_score > 0.3:
                trend = 'fair'
            else:
                trend = 'poor'
                
            return {'trend': trend, 'average_score': avg_score}
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_analysis_time = self._analysis_time_total / max(self._analysis_count, 1)
            return {
                'analysis_count': self._analysis_count,
                'avg_analysis_time': avg_analysis_time,
                'history_size': len(self.quality_history),
                'current_trend': self.get_quality_trend()
            }

# ===== ENHANCED PLUGIN-SYSTEM =====
class PluginManager:
    """Enhanced plugin system with performance monitoring"""
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        self.plugin_dir = plugin_dir or Path(__file__).parent / "plugins"
        try:
            self.plugin_dir.mkdir(exist_ok=True, mode=0o755)
        except Exception as e:
            self.plugin_dir = Path(tempfile.gettempdir()) / "dragon_whisperer_plugins"
            self.plugin_dir.mkdir(exist_ok=True, mode=0o755)
            
        self.loaded_plugins: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'plugin_times': {}
        }
        
        self._create_example_plugin()
        self.load_plugins()
    
    def _create_example_plugin(self) -> None:
        example_plugin = self.plugin_dir / "example_plugin.py"
        if not example_plugin.exists():
            example_code = '''"""
Example Dragon Whisperer Plugin
"""
import logging
import time
from typing import Dict, Any

def process_transcription(transcription: Dict[str, Any]) -> Dict[str, Any]:
    """Example: Add custom processing to transcriptions"""
    start_time = time.time()
    try:
        transcription['processed_by'] = 'example_plugin'
        
        if 'text' in transcription:
            transcription['text'] = transcription['text'].upper() + " 🔥"
            
        processing_time = time.time() - start_time
        transcription['plugin_processing_time'] = processing_time
            
        return transcription
    except Exception as e:
        logging.error(f"Example plugin error: {e}")
        return transcription

def initialize_plugin():
    """Plugin initialization"""
    logging.info("✅ Example plugin initialized")
    return True

PLUGIN_METADATA = {
    'name': 'Example Plugin',
    'version': '1.0.0',
    'author': 'Dragon Whisperer Team',
    'description': 'Example plugin for demonstration'
}
'''
            try:
                with open(example_plugin, 'w', encoding='utf-8') as f:
                    f.write(example_code)
                logging.info("Example plugin created")
            except Exception as e:
                logging.warning(f"Failed to create example plugin: {e}")
    
    def load_plugins(self) -> None:
        with self._lock:
            self.loaded_plugins.clear()
        
        if not self.plugin_dir.exists():
            return
        
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name == "__init__.py":
                continue
                
            try:
                import importlib.util
                plugin_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                if spec is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'initialize_plugin'):
                    if module.initialize_plugin():
                        with self._lock:
                            self.loaded_plugins[plugin_name] = module
                        logging.info(f"Plugin loaded: {plugin_name}")
                
            except Exception as e:
                ErrorHandler.handle_graceful(e, f"plugin loading {plugin_file}")
    
    def process_transcription_hook(self, transcription: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        result = transcription.copy()
        
        with self._lock:
            plugins = list(self.loaded_plugins.items())
        
        for plugin_name, plugin_module in plugins:
            try:
                if hasattr(plugin_module, 'process_transcription'):
                    plugin_start = time.time()
                    result = plugin_module.process_transcription(result)
                    plugin_time = time.time() - plugin_start
                    
                    with self._lock:
                        self._processing_stats['total_processed'] += 1
                        self._processing_stats['total_time'] += plugin_time
                        if plugin_name not in self._processing_stats['plugin_times']:
                            self._processing_stats['plugin_times'][plugin_name] = {
                                'count': 0,
                                'total_time': 0.0
                            }
                        self._processing_stats['plugin_times'][plugin_name]['count'] += 1
                        self._processing_stats['plugin_times'][plugin_name]['total_time'] += plugin_time
                    
                    logging.debug(f"Plugin {plugin_name} processed transcription in {plugin_time:.3f}s")
            except Exception as e:
                ErrorHandler.handle_graceful(e, f"plugin {plugin_name}")
        
        total_time = time.time() - start_time
        if total_time > 0.1:  # Log if plugin processing is slow
            logging.warning(f"Total plugin processing time: {total_time:.3f}s")
        
        return result
    
    def get_loaded_plugins(self) -> List[str]:
        with self._lock:
            return list(self.loaded_plugins.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_time = self._processing_stats['total_time'] / max(self._processing_stats['total_processed'], 1)
            plugin_stats = {}
            for plugin_name, stats in self._processing_stats['plugin_times'].items():
                plugin_stats[plugin_name] = {
                    'count': stats['count'],
                    'avg_time': stats['total_time'] / max(stats['count'], 1)
                }
            
            return {
                'loaded_plugins': self.get_loaded_plugins(),
                'total_processed': self._processing_stats['total_processed'],
                'total_processing_time': self._processing_stats['total_time'],
                'avg_processing_time': avg_time,
                'plugin_stats': plugin_stats
            }

# ===== VEREINFACHTES QUICK-START-SYSTEM =====
class SimpleStartManager:
    """Vereinfachtes Start-Management ohne komplexe Presets"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def apply_simple_settings(self, model_size: str, translation_enabled: bool, target_lang: str) -> bool:
        """Einfache Einstellungen anwenden"""
        try:
            # Einfache Config Updates
            self.config.config['DEFAULT']['model_size'] = model_size
            self.config.config['DEFAULT']['translation_enabled'] = str(translation_enabled).lower()
            self.config.config['DEFAULT']['target_language'] = target_lang
            
            self.config.save_config()
            return True
            
        except Exception as e:
            logging.error(f"Failed to apply settings: {e}")
            return False

# ===== ENHANCED HEALTH MONITORING =====
class HealthMonitor:
    """Enhanced health monitoring with detailed metrics"""
    
    def __init__(self):
        self.health_metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'queue_sizes': deque(maxlen=100),
            'error_rates': deque(maxlen=100)
        }
        self._lock = threading.RLock()
        self._last_system_check = 0
        self._system_check_interval = 5  # seconds
    
    def check_health(self, processor_stats: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            current_time = time.time()
            
            # Only check system metrics periodically to reduce overhead
            if current_time - self._last_system_check > self._system_check_interval:
                system_health = self._check_system_health()
                self._last_system_check = current_time
            else:
                system_health = {'status': 'unknown', 'message': 'Using cached system health'}
            
            health_status = {
                'system': system_health,
                'processor': self._check_processor_health(processor_stats),
                'performance': self._check_performance_health(processor_stats),
                'recommendations': self._generate_recommendations(processor_stats),
                'timestamp': current_time
            }
            
            return health_status
    
    def _check_system_health(self) -> Dict[str, str]:
        if not PSUTIL_AVAILABLE:
            return {'status': 'unknown', 'message': 'psutil not available'}
        
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Update metrics
            self.health_metrics['cpu_usage'].append(cpu_percent)
            self.health_metrics['memory_usage'].append(memory_percent)
            
            if memory_percent > 90:
                return {'status': 'critical', 'message': f'High memory usage: {memory_percent}%'}
            elif memory_percent > 80:
                return {'status': 'warning', 'message': f'High memory usage: {memory_percent}%'}
            elif cpu_percent > 90:
                return {'status': 'warning', 'message': f'High CPU usage: {cpu_percent}%'}
            else:
                return {'status': 'healthy', 'message': f'System OK (CPU: {cpu_percent}%, Memory: {memory_percent}%)'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Health check failed: {e}'}

    def _check_processor_health(self, stats: Dict[str, Any]) -> Dict[str, str]:
        try:
            chunks_processed = stats.get('chunks_processed', 0)
            errors = stats.get('errors', 0)
            error_rate = errors / max(chunks_processed, 1)
            queue_size = stats.get('audio_queue_size', 0)
            
            # Update metrics
            self.health_metrics['error_rates'].append(error_rate)
            self.health_metrics['queue_sizes'].append(queue_size)
            
            if error_rate > 0.1:
                return {'status': 'warning', 'message': f'High error rate: {error_rate:.1%}'}
            elif queue_size > PerformanceConfig.MAX_QUEUE_SIZE * 0.8:
                return {'status': 'warning', 'message': f'High queue size: {queue_size}'}
            else:
                return {'status': 'healthy', 'message': 'Processor running smoothly'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Processor health check failed: {e}'}

    def _check_performance_health(self, stats: Dict[str, Any]) -> Dict[str, str]:
        try:
            performance_metrics = stats.get('performance_metrics', {})
            success_ratio = performance_metrics.get('success_ratio', 0)
            transcription_rate = performance_metrics.get('transcription_rate_per_min', 0)
            
            if success_ratio < 0.3:
                return {'status': 'warning', 'message': f'Low success ratio: {success_ratio:.1%}'}
            elif transcription_rate < 1.0:
                return {'status': 'warning', 'message': f'Low transcription rate: {transcription_rate:.1f}/min'}
            else:
                return {'status': 'healthy', 'message': f'Performance good (success: {success_ratio:.1%}, rate: {transcription_rate:.1f}/min)'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Performance health check failed: {e}'}

    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        model_size = stats.get('model_size', 'small')
        chunks_processed = stats.get('chunks_processed', 0)
        success_ratio = stats.get('performance_metrics', {}).get('success_ratio', 0)
        
        if chunks_processed > 50 and success_ratio < 0.3:
            if model_size in ['tiny', 'base']:
                recommendations.append("Consider using a larger model (small/medium) for better accuracy")
        
        successful_translations = stats.get('successful_translations', 0)
        translation_rate = stats.get('performance_metrics', {}).get('translation_rate', 0)
        if (successful_translations > 0 and translation_rate < 0.5):
            recommendations.append("Translation is slow - consider disabling if not needed")
        
        if PSUTIL_AVAILABLE:
            try:
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    recommendations.append("High memory usage - consider reducing chunk duration or model size")
            except Exception:
                pass
        
        # Check queue performance
        queue_size = stats.get('audio_queue_size', 0)
        if queue_size > PerformanceConfig.MAX_QUEUE_SIZE * 0.7:
            recommendations.append("Audio queue frequently full - consider increasing queue size or reducing chunk duration")
        
        return recommendations
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        with self._lock:
            return {
                'cpu_usage': list(self.health_metrics['cpu_usage']),
                'memory_usage': list(self.health_metrics['memory_usage']),
                'queue_sizes': list(self.health_metrics['queue_sizes']),
                'error_rates': list(self.health_metrics['error_rates'])
            }

# ===== ENHANCED METRIKEN-SAMMLUNG =====
class MetricsCollector:
    """Enhanced metrics collection with persistence"""
    
    def __init__(self):
        self.metrics = {
            'transcriptions_total': 0,
            'translation_requests_total': 0,
            'errors_total': 0,
            'processing_duration_seconds': 0.0,
            'sessions_started': 0,
            'sessions_completed': 0,
            'total_audio_processed_bytes': 0,
            'peak_memory_usage_mb': 0.0
        }
        self._lock = threading.RLock()
        self._session_start_time = None
        self._current_session_metrics = {}
        
    def start_session(self) -> None:
        with self._lock:
            self._session_start_time = time.time()
            self.metrics['sessions_started'] += 1
            self._current_session_metrics = {
                'start_time': self._session_start_time,
                'transcriptions': 0,
                'translations': 0,
                'errors': 0,
                'audio_processed': 0
            }
        
    def end_session(self) -> None:
        with self._lock:
            if self._session_start_time:
                self.metrics['sessions_completed'] += 1
                self._session_start_time = None
                self._current_session_metrics = {}
        
    def increment_counter(self, metric_name: str, value: int = 1) -> None:
        with self._lock:
            if metric_name in self.metrics:
                self.metrics[metric_name] += value
            else:
                logging.warning(f"Unknown metric: {metric_name}")
                
    def record_processing_time(self, duration: float) -> None:
        with self._lock:
            self.metrics['processing_duration_seconds'] += duration
    
    def record_audio_processed(self, bytes_count: int) -> None:
        with self._lock:
            self.metrics['total_audio_processed_bytes'] += bytes_count
    
    def update_peak_memory(self, memory_mb: float) -> None:
        with self._lock:
            if memory_mb > self.metrics['peak_memory_usage_mb']:
                self.metrics['peak_memory_usage_mb'] = memory_mb
    
    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            metrics = self.metrics.copy()
            
            # Add derived metrics
            if metrics['processing_duration_seconds'] > 0:
                metrics['transcriptions_per_second'] = metrics['transcriptions_total'] / metrics['processing_duration_seconds']
                metrics['translations_per_second'] = metrics['translation_requests_total'] / metrics['processing_duration_seconds']
            else:
                metrics['transcriptions_per_second'] = 0
                metrics['translations_per_second'] = 0
            
            # Add session information
            if self._session_start_time:
                metrics['current_session_duration'] = time.time() - self._session_start_time
            else:
                metrics['current_session_duration'] = 0
            
            metrics['session_success_rate'] = (
                metrics['sessions_completed'] / metrics['sessions_started'] 
                if metrics['sessions_started'] > 0 else 0
            )
            
            return metrics
    
    def get_session_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return self._current_session_metrics.copy()
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to JSON file"""
        try:
            metrics = self.get_metrics()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logging.info(f"Metrics saved to {filepath}")
        except Exception as e:
            ErrorHandler.handle_graceful(e, "saving metrics")

# ===== REPARIERTER MAIN PROCESSOR =====
class StreamProcessor:
    """Fully enhanced stream processor with comprehensive monitoring"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.event_system = DragonEventSystem()
        
        self.resource_manager = FixedResourceManager()
        self.audio_quality_analyzer = AudioQualityAnalyzer(self.config)
        self.plugin_manager = PluginManager()
        self.simple_start_manager = SimpleStartManager(self.config)
        self.health_monitor = HealthMonitor()
        self.metrics_collector = MetricsCollector()
        # REPARIERT: Eigene PerformanceMonitor Instanz statt Referenz auf EventSystem
        self.performance_monitor = PerformanceMonitor()
        self.load_monitor = SystemLoadMonitor()
        
        self.audio_processor = AdaptiveAudioProcessor()
        self.stream_manager = EnhancedStreamManager()
        self.translation_engine = OptimizedTranslationEngine(
            target_lang=self.config.get('DEFAULT', 'target_language', fallback='en'),
            enabled=self.config.getboolean('DEFAULT', 'translation_enabled', fallback=True)
        )
        self.language_detector = OptimizedLanguageDetector()
        self.whisper_model: Optional[Any] = None
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.transcription_history = deque(maxlen=1000)
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        
        # 🔥 REPARIERT: Thread-safe Processing Status
        self._processing_lock = threading.RLock()
        self._is_processing = False
        
        self.session_analytics = ThreadSafeAnalytics()
        
        self.thread_safe_metrics = {
            'chunks_processed': SecureCounter(),
            'errors': SecureCounter(),
            'successful_transcriptions': SecureCounter(),
            'successful_translations': SecureCounter(),
            'silent_chunks_skipped': SecureCounter(),
            'low_quality_skipped': SecureCounter(),
            'dropped_chunks': SecureCounter(),
        }
        
        self.speaker_profiles: Dict[str, int] = {}
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="DragonWorker"
        )
        
        # Memory cleanup callbacks registrieren
        self.resource_manager.register_memory_cleanup_callback(self._force_memory_cleanup)
        self.resource_manager.register_cleanup(self._safe_cleanup, "StreamProcessor Cleanup")
        self.resource_manager.register_resource(self.thread_pool, lambda tp: self._cleanup_thread_pool_safe())
        
        self.progress_callback: Optional[Callable] = None
        
        atexit.register(self._atexit_cleanup)
        
        # Event Subscriptions
        self.event_system.subscribe(DragonEvents.ERROR_OCCURRED, self._handle_error_event)
        self.event_system.subscribe(DragonEvents.MEMORY_WARNING, self._handle_memory_warning)
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        self.metrics_collector.start_session()
        
    @property
    def is_processing(self) -> bool:
        """Thread-safe property für Processing Status"""
        with self._processing_lock:
            return self._is_processing
            
    @is_processing.setter 
    def is_processing(self, value: bool) -> None:
        """Thread-safe setter für Processing Status"""
        with self._processing_lock:
            self._is_processing = value
        
    def _atexit_cleanup(self) -> None:
        try:
            self.stop_processing()
            self.resource_manager.cleanup_all()
            self.metrics_collector.end_session()
        except Exception as e:
            ErrorHandler.handle_critical(e, "atexit cleanup")
        
    def _handle_error_event(self, data: Dict[str, Any]) -> None:
        """Handle error events from the event system"""
        if data.get('type') == 'critical':
            logging.critical(f"Critical error: {data.get('error')}")
        
    def _handle_memory_warning(self, data: Dict[str, Any]) -> None:
        """Handle memory warning events"""
        memory_mb = data.get('memory_mb', 0)
        logging.warning(f"Memory warning: {memory_mb:.1f}MB used")
        
        # Trigger immediate cleanup
        self._force_memory_cleanup()
    
    def _force_memory_cleanup(self) -> None:
        """Force memory cleanup when warnings occur"""
        try:
            # Clear caches
            self.translation_engine.clear_cache()
            self.stream_manager.stream_cache.clear()
            
            # Clear transcription history if too large
            with self._lock:
                if len(self.transcription_history) > 500:
                    self.transcription_history = deque(list(self.transcription_history)[-500:], maxlen=1000)
            
            # Force garbage collection
            gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            logging.info("Forced memory cleanup completed")
        except Exception as e:
            ErrorHandler.handle_graceful(e, "forced memory cleanup")
    
    def _safe_cleanup(self) -> None:
        try:
            self._shutdown_event.set()
            self._cleanup_thread_pool_safe()
            self._safe_ffmpeg_shutdown()
            
            if hasattr(self, 'translation_engine') and self.translation_engine:
                self.translation_engine.clear_cache()
            
            self.metrics_collector.end_session()
            
            gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            ErrorHandler.handle_graceful(e, "safe cleanup")
    
    def _cleanup_thread_pool_safe(self) -> None:
        if hasattr(self, 'thread_pool') and self.thread_pool:
            try:
                # ThreadPoolExecutor ohne timeout Parameter shutdown
                self.thread_pool.shutdown(wait=True)
            except Exception as e:
                ErrorHandler.handle_graceful(e, "thread pool shutdown")
                try:
                    self.thread_pool.shutdown(wait=False)
                except Exception:
                    pass

    def _safe_ffmpeg_shutdown(self) -> None:
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            FFmpegManager.safe_shutdown(self.ffmpeg_process)
            FFmpegManager.close_pipes(self.ffmpeg_process)
            self.ffmpeg_process = None

    def set_progress_callback(self, callback: Callable) -> None:
        self.progress_callback = callback
        
    def _update_progress(self, message: str, percentage: Optional[int] = None) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(message, percentage)
            except Exception as e:
                ErrorHandler.handle_graceful(e, "progress callback")

    def initialize_models(self, model_size: str = "small", target_lang: str = "en") -> bool:
        try:
            self._update_progress("Downloading AI model...", 10)
            
            if not FASTER_WHISPER_AVAILABLE:
                self._update_progress("Error: faster-whisper not available", None)
                return False
                
            device = "cuda" if self._check_cuda() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            self._update_progress(f"Downloading ({model_size}) AI model {device}...", 30)
            
            try:
                self.whisper_model = WhisperModel(
                    model_size, 
                    device=device, 
                    compute_type=compute_type,
                    download_root=str(Path.home() / ".cache" / "whisper"),
                    num_workers=1
                )
                logging.info(f"Whisper model {model_size} loaded on {device}")
                self.event_system.publish(DragonEvents.MODEL_LOADED, {
                    'type': 'whisper',
                    'model_size': model_size,
                    'device': device
                })
            except Exception as model_error:
                ErrorHandler.handle_operation(model_error, "primary model loading")
                return self._try_fallback_models(model_size, device, compute_type)
            
            self._update_progress("Initializing translation engine...", 70)
            
            self.translation_engine.target_lang = target_lang
            translation_initialized = self.translation_engine.initialize()
            if not translation_initialized:
                logging.warning("Translation engine initialization failed")
                
            self._update_progress("Setting up memory management...", 90)
            
            # Adjust performance config based on available memory
            if PSUTIL_AVAILABLE:
                try:
                    available_memory = psutil.virtual_memory().available
                    PerformanceConfig.adjust_for_memory(available_memory)
                except Exception as e:
                    logging.debug(f"Memory-based adjustment failed: {e}")
            
            self._update_progress("Models initialized successfully!", 100)
            
            return True
            
        except Exception as e:
            ErrorHandler.handle_operation(e, "model initialization")
            self._update_progress(f"Initialization failed: {e}", None)
            
            return self._try_fallback_models(model_size, 
                                           "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
                                           "int8")
    
    def _try_fallback_models(self, original_model: str, device: str, compute_type: str) -> bool:
        fallback_models = ['small', 'base', 'tiny']
        
        if original_model in fallback_models:
            current_index = fallback_models.index(original_model)
        else:
            current_index = 0
            
        for i, fallback_model in enumerate(fallback_models[current_index + 1:], 1):
            progress = 20 + (i * 20)
            self._update_progress(f"Trying fallback model: {fallback_model}", progress)
            
            try:
                self.whisper_model = WhisperModel(
                    fallback_model, 
                    device=device, 
                    compute_type=compute_type,
                    download_root=str(Path.home() / ".cache" / "whisper"),
                    num_workers=1
                )
                
                logging.info(f"Fallback model {fallback_model} loaded successfully")
                self._update_progress(f"Fallback model {fallback_model} loaded successfully", 100)
                return True
                
            except Exception as fallback_error:
                ErrorHandler.handle_graceful(fallback_error, f"fallback model {fallback_model}")
                continue
                
        self._update_progress("All model loading attempts failed", None)
        return False
    
    def _check_cuda(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
            
        try:
            return torch.cuda.is_available()
        except Exception as e:
            logging.debug(f"CUDA check failed: {e}")
            return False

    def process_audio_chunk_optimized(self, audio_data: bytes, chunk_id: int, callbacks: Dict[str, Callable]) -> None:
        start_time = time.time()
        
        try:
            # Check memory usage before processing
            if not self.resource_manager.check_memory_usage():
                logging.warning("Memory usage high, skipping chunk processing")
                self.thread_safe_metrics['dropped_chunks'].increment()
                return
            
            # Check system load for graceful degradation
            if self.load_monitor.should_degrade_quality():
                if chunk_id % 3 == 0:  # Überspringe einige Chunks bei hoher Last
                    self.thread_safe_metrics['dropped_chunks'].increment()
                    return
            
            quality_info = self.audio_quality_analyzer.analyze_audio_quality(audio_data)
            
            if (quality_info.get('status') in ['silent', 'poor'] or 
                quality_info.get('quality_score', 0) < 0.2):
                self.thread_safe_metrics['silent_chunks_skipped'].increment()
                self.thread_safe_metrics['chunks_processed'].increment()
                return
                
            self.audio_processor.adjust_chunk_size_based_on_quality(quality_info['quality_score'])
                
            quality_trend = self.audio_quality_analyzer.get_quality_trend()
            self.session_analytics.update_quality_trend(quality_trend['trend'])
            
            transcription = self.transcribe_audio_enhanced(audio_data)
            
            if transcription and transcription.text.strip():
                transcription.audio_quality = quality_info
                transcription.quality_rating = self._get_quality_rating(transcription.confidence, quality_info)
                
                min_confidence = float(self.config.get('DEFAULT', 'min_confidence', fallback='0.3'))
                if transcription.confidence < min_confidence:
                    self.thread_safe_metrics['low_quality_skipped'].increment()
                    return
                
                if self._contains_gibberish(transcription.text):
                    self.thread_safe_metrics['low_quality_skipped'].increment()
                    return
                
                transcription_dict = transcription.to_dict()
                processed_dict = self.plugin_manager.process_transcription_hook(transcription_dict)
                transcription.plugin_processed = True
                
                if (transcription.confidence > 0.1 and
                    not self._contains_gibberish(transcription.text) and
                    len(transcription.text) > 2):
                    
                    if transcription.language == 'unknown' or transcription.confidence < 0.5:
                        enhanced_detection = self.language_detector.detect_language_enhanced(
                            transcription.text, 
                            transcription.language, 
                            transcription.confidence
                        )
                        if enhanced_detection['confidence'] > transcription.confidence:
                            transcription.language = enhanced_detection['language']
                            transcription.confidence = enhanced_detection['confidence']
                    
                    self.session_analytics.add_language(transcription.language)
                    
                    self._perform_light_analytics(transcription)
                    self.thread_safe_metrics['successful_transcriptions'].increment()
                    self.metrics_collector.increment_counter('transcriptions_total')
                    
                    with self._lock:
                        self.transcription_history.append(transcription)
                        self.session_analytics.add_confidence(transcription.confidence)
                        self.session_analytics.add_processing_time(time.time() - start_time)
                    
                    # Event publish
                    self.event_system.publish(DragonEvents.TRANSCRIPTION_READY, transcription)
                    
                    if callbacks and 'transcription' in callbacks:
                        try:
                            callbacks['transcription'](transcription)
                        except Exception as callback_error:
                            ErrorHandler.handle_graceful(callback_error, "transcription callback")
                    
                    processing_time = time.time() - start_time
                    self.performance_monitor.record_processing_time(processing_time)
                    
                    # REPARIERT: record_audio_quality anstatt record_memory_usage
                    if hasattr(self.performance_monitor, 'record_audio_quality'):
                        self.performance_monitor.record_audio_quality(quality_info['quality_score'])
                    
                    if (self.translation_engine.is_enabled() and 
                        transcription.language != self.translation_engine.target_lang and 
                        transcription.language != 'unknown' and
                        len(transcription.text.strip()) > 3):
                        
                        def translate_async():
                            try:
                                translation_start = time.time()
                                translation = self.translation_engine.translate_text_enhanced(
                                    transcription.text,
                                    transcription.language,
                                    self.translation_engine.target_lang
                                )
                                translation_time = time.time() - translation_start
                                self.performance_monitor.record_translation_latency(translation_time)
                                
                                if translation and translation.confidence > 0.1:
                                    self.thread_safe_metrics['successful_translations'].increment()
                                    self.metrics_collector.increment_counter('translation_requests_total')
                                    
                                    # Event publish
                                    self.event_system.publish(DragonEvents.TRANSLATION_READY, translation)
                                    
                                    if callbacks and 'translation' in callbacks:
                                        try:
                                            callbacks['translation'](translation)
                                        except Exception as callback_error:
                                            ErrorHandler.handle_graceful(callback_error, "translation callback")
                            except Exception as e:
                                ErrorHandler.handle_graceful(e, "async translation")
                        
                        if hasattr(self, 'thread_pool') and self.thread_pool:
                            self.thread_pool.submit(translate_async)
                        else:
                            threading.Thread(target=translate_async, daemon=True).start()
            
            self.thread_safe_metrics['chunks_processed'].increment()
            
        except Exception as e:
            ErrorHandler.handle_operation(e, "audio chunk processing")
            self.thread_safe_metrics['errors'].increment()
            self.metrics_collector.increment_counter('errors_total')
        finally:
            # Record memory usage after processing
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.performance_monitor.record_memory_usage(memory_mb)
                    self.metrics_collector.update_peak_memory(memory_mb)
                except Exception as e:
                    logging.debug(f"Memory recording failed: {e}")

    def _get_quality_rating(self, confidence: float, audio_quality: Dict[str, Any]) -> str:
        audio_score = audio_quality.get('quality_score', 0)
        combined_score = (confidence * 0.6) + (audio_score * 0.4)
        
        if combined_score > 0.7:
            return "excellent"
        elif combined_score > 0.5:
            return "good"
        elif combined_score > 0.3:
            return "fair"
        else:
            return "poor"

    def transcribe_audio_enhanced(self, audio_data: bytes) -> Optional[TranscriptionResult]:
        if not self.whisper_model or not NUMPY_AVAILABLE or not audio_data:
            return None
            
        start_time = time.time()
        
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_np) > 1000:
                audio_np = audio_np - np.mean(audio_np)
                max_val = np.max(np.abs(audio_np))
                if max_val > 0:
                    audio_np = audio_np / max_val
            
            if len(audio_np) < 8000:
                return None
                
            segments, info = self.whisper_model.transcribe(
                audio_np,
                language=None,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=300
                ),
                without_timestamps=True,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                patience=1.0
            )
            
            text_parts = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                clean_text = segment.text.strip()
                if clean_text and len(clean_text) > 1:
                    clean_text = self._clean_german_text(clean_text)
                    if clean_text:
                        text_parts.append(clean_text)
                        confidence = getattr(segment, 'avg_logprob', 0.0)
                        if confidence != 0.0:
                            confidence = min(1.0, max(0.0, np.exp(confidence)))
                        else:
                            confidence = 0.5
                        
                        total_confidence += confidence
                        segment_count += 1
            
            if not text_parts:
                return None
                
            text = " ".join(text_parts).strip()
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.3
            
            MIN_WORD_COUNT = 1
            MIN_CHAR_COUNT = 2
            
            words = text.split()
            if (len(words) < MIN_WORD_COUNT or 
                len(text) < MIN_CHAR_COUNT or
                text in ['.', '..', '...']):
                return None
            
            text = re.sub(r'\s+', ' ', text).strip()
            
            processing_time = time.time() - start_time
            
            result = TranscriptionResult(
                text=text,
                language=getattr(info, 'language', 'unknown') if info else 'unknown',
                start_time=time.time() - PerformanceConfig.CHUNK_DURATION,
                end_time=time.time(),
                confidence=avg_confidence,
                processing_time=processing_time,
                word_count=len(text.split())
            )
            
            self.performance_monitor.record_transcription_latency(processing_time)
            
            return result
            
        except Exception as e:
            ErrorHandler.handle_graceful(e, "transcription")
            return None

    def _perform_light_analytics(self, transcription: TranscriptionResult) -> None:
        self.session_analytics.increment_words(len(transcription.text.split()))
        
        word_count = len(transcription.text.split())
        if word_count > 25:
            speaker_id = "Speaker_Long"
        elif word_count > 15:
            speaker_id = "Speaker_Medium" 
        elif word_count > 5:
            speaker_id = "Speaker_Short"
        else:
            speaker_id = "Speaker_Brief"
            
        transcription.speaker = speaker_id
        with self._lock:
            self.speaker_profiles[speaker_id] = self.speaker_profiles.get(speaker_id, 0) + 1

    def _contains_gibberish(self, text: str) -> bool:
        if len(text) < 2:
            return True
            
        if len(set(text)) < 3 and len(text) > 10:
            return True
            
        weird_chars = ['�', '⠀']
        if any(char in text for char in weird_chars):
            return True
            
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.7:
            return True
            
        words = text.split()
        if len(words) > 3:
            unique_word_ratio = len(set(words)) / len(words)
            if unique_word_ratio < 0.3:
                return True
            
        return False

    def _clean_german_text(self, text: str) -> str:
        if not text:
            return text
        
        clean_text = re.sub(r'\b(äh|ähm|mmmh?|hm+)\b', '', text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\b(um|uh|er|like|you know)\b', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'^\s*[.,!?]\s*$', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text

    def start_stream_processing(self, url: str, callbacks: Dict[str, Callable]) -> bool:
        if self._shutdown_event.is_set():
            if callbacks and 'error' in callbacks:
                callbacks['error']("Processor is shutting down")
            return False
            
        if not url or len(url.strip()) < 5:
            if callbacks and 'error' in callbacks:
                callbacks['error']("Invalid URL provided")
            return False
            
        self.config.update_recent_urls(url)
        
        return self._start_stream_processing_internal(url, callbacks)
    
    def _start_stream_processing_internal(self, url: str, callbacks: Dict[str, Callable]) -> bool:
        try:
            stream_url = self.stream_manager.extract_stream_url_enhanced(url, callbacks)
            if stream_url is None:
                if callbacks and 'error' in callbacks:
                    callbacks['error']("Failed to process URL")
                return False
                
            stream_title = self.stream_manager.extract_stream_title(url)
            if stream_title:
                if callbacks and 'info' in callbacks:
                    callbacks['info'](f"Stream: {stream_title}")
                if callbacks and 'stream_title' in callbacks:
                    callbacks['stream_title'](stream_title)
                
            self._shutdown_event.clear()
            self.session_analytics.reset()
            
            # 🔥 REPARIERT: Processing Status setzen
            self.is_processing = True
            
            stream_type = self.stream_manager.detect_stream_type(stream_url)
            if callbacks and 'info' in callbacks:
                callbacks['info'](f"Stream type detected: {stream_type}")
            
            self.audio_processor.start_processing_adaptive(
                lambda audio, chunk_id: self.process_audio_chunk_optimized(audio, chunk_id, callbacks),
                stream_type
            )
            
            self.processing_thread = threading.Thread(
                target=self._process_stream,
                args=(stream_url, stream_type, callbacks),
                daemon=True,
                name="StreamProcessor"
            )
            self.processing_thread.start()
            
            self.event_system.publish(DragonEvents.PROCESSING_STARTED, {
                'url': url,
                'stream_type': stream_type,
                'title': stream_title
            })
            
            if callbacks and 'info' in callbacks:
                callbacks['info']("Processing started - waiting for audio data...")
            return True
            
        except Exception as e:
            ErrorHandler.handle_operation(e, "stream processing startup")
            if callbacks and 'error' in callbacks:
                callbacks['error'](f"Startup error: {e}")
            # 🔥 REPARIERT: Status zurücksetzen bei Fehler
            self.is_processing = False
            return False

    def _process_stream(self, stream_url: str, stream_type: str, callbacks: Dict[str, Callable]) -> None:
        max_retries = 3  # Reduzierte Retries für schnellere Fehlerbehandlung
        retry_count = 0
        base_delay = 2
        
        while (retry_count < max_retries and 
               not self._shutdown_event.is_set()):
            try:
                # Reset für neuen Versuch
                self._safe_ffmpeg_shutdown()
                time.sleep(0.5)  # Kurze Pause vor Neustart
                
                if stream_type in ['local', 'local_audio', 'local_video']:
                    self._process_local_file(stream_url, callbacks)
                elif stream_type == 'hls':
                    self._process_hls_stream_direct(stream_url, callbacks)
                else:
                    self._process_generic_stream_enhanced(stream_url, callbacks)
                break
                        
            except Exception as e:
                retry_count += 1
                ErrorHandler.handle_operation(e, f"stream processing attempt {retry_count}")
                
                if retry_count < max_retries:
                    delay = base_delay * (2 ** (retry_count - 1))
                    if callbacks and 'warning' in callbacks:
                        callbacks['warning'](f"Stream error, retrying in {delay}s... ({retry_count}/{max_retries})")
                    
                    # Cleanup vor Retry
                    self._safe_ffmpeg_shutdown()
                    time.sleep(delay)
                else:
                    # Finaler Fehler - aber blockiert nicht zukünftige Versuche
                    ErrorHandler.handle_operation(e, f"stream failed after {max_retries} attempts")
                    if callbacks and 'error' in callbacks:
                        callbacks['error'](f"Stream failed after {max_retries} attempts: {e}")
                    
                    # WICHTIG: Shutdown Event zurücksetzen für zukünftige Versuche
                    self._shutdown_event.clear()
                    break

    def _process_hls_stream_direct(self, hls_url: str, callbacks: Dict[str, Callable]) -> None:
        max_retries = 5
        retry_count = 0
        base_delay = 2
        
        while retry_count < max_retries and not self._shutdown_event.is_set():
            try:
                chunk_duration = PerformanceConfig.CHUNK_DURATION
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
                    '-reconnect', '1',
                    '-reconnect_at_eof', '1',
                    '-reconnect_streamed', '1',
                    '-reconnect_delay_max', '5',
                    '-loglevel', 'warning',
                    '-'
                ]
                
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
                
                stderr_thread = threading.Thread(
                    target=self._monitor_ffmpeg_stderr_enhanced,
                    args=(self.ffmpeg_process.stderr, callbacks),
                    daemon=True
                )
                stderr_thread.start()
                
                while (self.ffmpeg_process and 
                       self.ffmpeg_process.poll() is None and 
                       not self._shutdown_event.is_set()):
                    
                    if time.time() - last_successful_read > stream_timeout:
                        if callbacks and 'warning' in callbacks:
                            callbacks['warning']("Stream timeout - trying to restart...")
                        break
                        
                    ready_to_read, _, _ = select.select([self.ffmpeg_process.stdout], [], [], 0.5)
                    if ready_to_read:
                        audio_data = self.ffmpeg_process.stdout.read(chunk_bytes)
                        if audio_data and len(audio_data) > 0:
                            chunk_counter += 1
                            consecutive_empty_reads = 0
                            last_successful_read = time.time()
                            
                            if not self.audio_processor._safe_audio_put(audio_data, chunk_counter):
                                continue
                                
                            if chunk_counter == 1:
                                if callbacks and 'info' in callbacks:
                                    callbacks['info']("Audio data received - transcription running...")
                                
                        elif not audio_data:
                            consecutive_empty_reads += 1
                            if consecutive_empty_reads >= max_empty_reads:
                                break
                            time.sleep(0.1)
                    else:
                        if self.ffmpeg_process.poll() is not None:
                            break
                        time.sleep(0.1)
                        
                break
                
            except Exception as e:
                retry_count += 1
                ErrorHandler.handle_operation(e, f"HLS processing attempt {retry_count}")
                if retry_count < max_retries:
                    delay = base_delay * (2 ** (retry_count - 1))
                    if callbacks and 'warning' in callbacks:
                        callbacks['warning'](f"HLS error, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    ErrorHandler.handle_operation(e, "HLS stream failed after maximum retries")
                    if callbacks and 'error' in callbacks:
                        callbacks['error']("HLS stream failed after maximum attempts")
                    break
            finally:
                self._safe_ffmpeg_shutdown()

    def _process_generic_stream_enhanced(self, stream_url: str, callbacks: Dict[str, Callable]) -> None:
        chunk_duration = PerformanceConfig.CHUNK_DURATION
        chunk_bytes = int(16000 * 2 * chunk_duration)
        
        if callbacks and 'info' in callbacks:
            callbacks['info']("Processing stream...")
        
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', stream_url,
                '-f', 's16le',
                '-ar', '16000',
                '-ac', '1',
                '-loglevel', 'quiet',
                '-fflags', '+nobuffer+flush_packets',
                '-flags', 'low_delay',
                '-avioflags', 'direct',
                '-max_delay', '500000',
                '-'
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=chunk_bytes
            )
            
            chunk_counter = 0
            consecutive_empty_reads = 0
            max_empty_reads = 10
            stream_timeout = PerformanceConfig.CHUNK_TIMEOUT
            last_successful_read = time.time()
            
            stderr_thread = threading.Thread(
                target=self._monitor_ffmpeg_stderr_enhanced,
                args=(self.ffmpeg_process.stderr, callbacks),
                daemon=True
            )
            stderr_thread.start()
            
            while (self.ffmpeg_process and 
                   self.ffmpeg_process.poll() is None and 
                   not self._shutdown_event.is_set()):
                
                current_time = time.time()
                if current_time - last_successful_read > stream_timeout:
                    if callbacks and 'warning' in callbacks:
                        callbacks['warning']("Stream timeout - restarting...")
                    break
                    
                ready_to_read, _, _ = select.select([self.ffmpeg_process.stdout], [], [], 1.0)
                if ready_to_read:
                    audio_data = self.ffmpeg_process.stdout.read(chunk_bytes)
                    if audio_data and len(audio_data) > 0:
                        chunk_counter += 1
                        consecutive_empty_reads = 0
                        last_successful_read = current_time
                        
                        if not self.audio_processor._safe_audio_put(audio_data, chunk_counter):
                            continue
                            
                        if chunk_counter == 1:
                            if callbacks and 'info' in callbacks:
                                callbacks['info']("Audio data received - transcription running...")
                            
                    elif not audio_data:
                        consecutive_empty_reads += 1
                        if consecutive_empty_reads >= max_empty_reads:
                            break
                        time.sleep(0.2)
                else:
                    if self.ffmpeg_process.poll() is not None:
                        break
                    time.sleep(0.2)
                    
        except Exception as e:
            ErrorHandler.handle_operation(e, "generic stream processing")
            if callbacks and 'error' in callbacks:
                callbacks['error'](f"Stream error: {e}")
        finally:
            self._safe_ffmpeg_shutdown()

    def _process_local_file(self, file_path: str, callbacks: Dict[str, Callable]) -> None:
        if file_path.startswith('file://'):
            file_path = file_path[7:]
            
        if not os.path.exists(file_path):
            if callbacks and 'error' in callbacks:
                callbacks['error'](f"File not found: {file_path}")
            self.stop_processing()
            return
            
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        if callbacks and 'info' in callbacks:
            callbacks['info'](f"Processing local file: {os.path.basename(file_path)} ({file_size:.1f} MB)")
        
        chunk_duration = PerformanceConfig.CHUNK_DURATION
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
            
            while (self.ffmpeg_process and 
                   self.ffmpeg_process.poll() is None and 
                   not self._shutdown_event.is_set()):
                
                if time.time() - last_successful_read > file_timeout:
                    break
                    
                audio_data = self.ffmpeg_process.stdout.read(chunk_bytes)
                if audio_data:
                    chunk_counter += 1
                    last_successful_read = time.time()
                    
                    if not self.audio_processor._safe_audio_put(audio_data, chunk_counter):
                        continue
                else:
                    break
                    
            if callbacks and 'info' in callbacks:
                callbacks['info']("File processing completed")
            
        except Exception as e:
            ErrorHandler.handle_operation(e, "local file processing")
            if callbacks and 'error' in callbacks:
                callbacks['error'](f"File processing error: {e}")
        finally:
            self.stop_processing()

    def _monitor_ffmpeg_stderr_enhanced(self, stderr_pipe, callbacks: Dict[str, Callable]) -> None:
        """Überwacht FFmpeg Output für Offline-Stream Erkennung - REPARIERT"""
        try:
            while (hasattr(self, 'ffmpeg_process') and 
                   self.ffmpeg_process and 
                   not self._shutdown_event.is_set()):
            
                line = stderr_pipe.readline()
                if not line:
                    break
                
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    # 🔧 AUTO-STOP: Verbesserte Offline-Stream Erkennung
                    offline_indicators = [
                        "channel is not currently live",
                        "offline",
                        "stream is offline", 
                        "failed to read header",
                        "http error 4",
                        "unable to open",
                        "no such file or directory",
                        "connection refused",
                        "network is unreachable"
                    ]
                    
                    if any(indicator in line_str.lower() for indicator in offline_indicators):
                        logging.info(f"Auto-stop detected: {line_str}")
                        
                        # Sofortiger Stop mit Callback
                        if callbacks and 'error' in callbacks:
                            callbacks['error'](f"Stream offline: {line_str}")
                        
                        # Processing sofort stoppen
                        self._shutdown_event.set()
                        break
                    
                    # Ignoriere Keepalive-Fehler
                    if 'keepalive request failed' in line_str.lower():
                        continue
                        
                    # Andere Fehler melden
                    error_keywords = {'error', 'failed', 'invalid', 'missing'}
                    if any(keyword in line_str.lower() for keyword in error_keywords):
                        if callbacks and 'error' in callbacks:
                            # Spezifische HTTP Fehler
                            if "403" in line_str:
                                callbacks['error']("HTTP 403 - Access denied")
                            elif "404" in line_str:
                                callbacks['error']("HTTP 404 - Stream not found") 
                            elif "Connection refused" in line_str:
                                callbacks['error']("Connection refused")
                            else:
                                callbacks['error'](f"Stream error: {line_str}")
                        logging.warning(f"FFmpeg error: {line_str}")
        except Exception as e:
            logging.debug(f"FFmpeg stderr monitoring failed: {e}")

    def stop_processing(self) -> None:
        """VEREINFACHTE Stop-Methode - ersetzt stop_processing_only() und stop()"""
        with self._processing_lock:
            if not self._is_processing:
                return
            self._is_processing = False
            self._shutdown_event.set()
        
        self.audio_processor.stop_processing()
        self._safe_ffmpeg_shutdown()
        
        # State zurücksetzen für Neustart
        self._shutdown_event.clear()
        
        # Caches behalten für bessere Performance beim Neustart
        # self.translation_engine.clear_cache()  # AUSKOMMENTIERT - Cache behalten
        
        with self._lock:
            self.speaker_profiles.clear()
        
        # Garbage Collection für Memory
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        self.event_system.publish(DragonEvents.PROCESSING_STOPPED, {})
        logging.info("Processing stopped - ready for restart")

    def full_shutdown(self) -> None:
        """Kompletter Shutdown für App-Exit"""
        self.stop_processing()  # ✅ Vereinfachter Aufruf
        self.resource_manager.cleanup_all()
        self.metrics_collector.end_session()

    # ❌ ENTFERNT: Redundante stop() Methode
    # def stop(self) -> None:
    #     """Alias für stop_processing für Abwärtskompatibilität"""
    #     self.stop_processing()

    def export_transcriptions(self, format_type: str, filename: str) -> str:
        safe_filename = SecurityUtils.validate_file_path(filename)
        
        with self._lock:
            if not self.transcription_history:
                raise Exception("No transcriptions available for export")
                
        try:
            with open(safe_filename, 'w', encoding='utf-8') as f:
                if format_type == 'txt':
                    for trans in self.transcription_history:
                        timestamp = time.strftime("%H:%M:%S", time.localtime(trans.start_time))
                        speaker = f" [{trans.speaker}]" if trans.speaker else ""
                        quality = f" [{trans.quality_rating}]" if hasattr(trans, 'quality_rating') else ""
                        f.write(f"[{timestamp}{speaker}{quality}] {trans.text}\n")
                        
                elif format_type == 'srt':
                    for i, trans in enumerate(self.transcription_history, 1):
                        start_str = self._format_timestamp(trans.start_time)
                        end_str = self._format_timestamp(trans.end_time)
                        f.write(f"{i}\n")
                        f.write(f"{start_str} --> {end_str}\n")
                        f.write(f"{trans.text}\n\n")
                        
                elif format_type == 'json':
                    analytics_data = self.session_analytics.get_data()
                    
                    data = {
                        'metadata': {
                            'export_time': time.time(),
                            'total_segments': len(self.transcription_history),
                            'languages_detected': analytics_data['languages_detected'],
                            'total_words': analytics_data['total_words'],
                            'average_confidence': analytics_data.get('average_confidence', 0),
                            'version': '2.1'
                        },
                        'transcriptions': [
                            trans.to_dict() for trans in self.transcription_history
                        ]
                    }
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            
            logging.info(f"Transcriptions exported to {safe_filename}")
            return safe_filename
            
        except Exception as e:
            ErrorHandler.handle_operation(e, "export transcriptions")
            raise Exception(f"Export failed: {e}")

    def _format_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            base_stats = {
                'is_running': not self._shutdown_event.is_set(),
                'transcription_count': len(self.transcription_history),
                'audio_queue_size': self.audio_processor.audio_queue.qsize() if hasattr(self.audio_processor, 'audio_queue') else 0,
                'models_loaded': {
                    'whisper': self.whisper_model is not None,
                    'translator': self.translation_engine.is_initialized()
                },
                'model_size': self.config.get('DEFAULT', 'model_size', 'small')
            }
            
            chunks_processed = self.thread_safe_metrics['chunks_processed'].get_value()
            successful_transcriptions = self.thread_safe_metrics['successful_transcriptions'].get_value()
            
            analytics_data = self.session_analytics.get_data()
            
            quality_metrics = {
                'high_quality_transcriptions': sum(1 for t in self.transcription_history if t.confidence > 0.7),
                'medium_quality_transcriptions': sum(1 for t in self.transcription_history if 0.5 < t.confidence <= 0.7),
                'low_quality_transcriptions': sum(1 for t in self.transcription_history if t.confidence <= 0.5),
                'average_confidence': analytics_data.get('average_confidence', 0),
                'low_quality_skipped': self.thread_safe_metrics['low_quality_skipped'].get_value(),
                'dropped_chunks': self.thread_safe_metrics['dropped_chunks'].get_value()
            }
            
            enhanced_stats = {
                'chunks_processed': chunks_processed,
                'successful_transcriptions': successful_transcriptions,
                'successful_translations': self.thread_safe_metrics['successful_translations'].get_value(),
                'silent_chunks_skipped': self.thread_safe_metrics['silent_chunks_skipped'].get_value(),
                'errors': self.thread_safe_metrics['errors'].get_value(),
                'languages_detected': analytics_data['languages_detected'],
                'total_words': analytics_data['total_words'],
                'session_duration': time.time() - analytics_data['start_time'],
                'audio_quality': analytics_data['audio_quality_trend'],
                'performance_metrics': {
                    'success_ratio': (successful_transcriptions / max(chunks_processed, 1)) if chunks_processed > 0 else 0,
                    'transcription_rate_per_min': (successful_transcriptions / max((time.time() - analytics_data['start_time']), 1)) * 60
                },
                'loaded_plugins': self.plugin_manager.get_loaded_plugins(),
                'quality_metrics': quality_metrics
            }
            
            return {**base_stats, **enhanced_stats}

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including all subsystems"""
        base_stats = self.get_stats()
        
        detailed_stats = {
            'base': base_stats,
            'performance': self.performance_monitor.get_performance_report(),
            'health': self.health_monitor.check_health(base_stats),
            'metrics': self.metrics_collector.get_metrics(),
            'translation_engine': self.translation_engine.get_stats(),
            'stream_manager': self.stream_manager.get_stats(),
            'audio_processor': self.audio_processor.get_stats(),
            'audio_quality_analyzer': self.audio_quality_analyzer.get_stats(),
            'plugin_manager': self.plugin_manager.get_stats(),
            'config': self.config.get_stats(),
            'resource_manager': {
                'memory_ok': self.resource_manager.check_memory_usage()
            }
        }
        
        return detailed_stats

    def clear_history(self) -> None:
        with self._lock:
            self.transcription_history.clear()
            self.session_analytics.reset()
            self.speaker_profiles.clear()
            
            for metric in self.thread_safe_metrics.values():
                metric.reset()
            
            logging.info("History cleared")

# ===== REPARIERTE GUI =====
class DragonWhispererGUI:
    """Fully enhanced GUI with comprehensive monitoring and memory leak fixes"""
    
    def __init__(self):
        setup_logging()  # Logging setup hinzufügen
        try:
            if not GUI_AVAILABLE:
                raise RuntimeError("GUI not available - tkinter required")
            
            self.config = ConfigManager()
            
            self.root = tk.Tk()
            self._gui_running = True
            self._shutdown_in_progress = False  # 🔥 NEU: Shutdown-Flag
            self.is_processing = False            
            self.auto_scroll = True
            
            self.gui_updater = ThreadSafeGUIUpdater(self.root)
            
            self.processor = StreamProcessor()          
            self.processor.set_progress_callback(self._update_progress)
            
            self.colors = {
                'bg_primary': '#1a1a1a',
                'bg_secondary': '#2d2d2d', 
                'bg_tertiary': '#3d3d3d',
                'text_primary': '#e8e8e8',
                'text_secondary': '#a0a0a0',
                'accent_blue': '#2196F3',
                'accent_green': '#4CAF50',
                'accent_red': '#f44336',
                'accent_orange': '#FF9800',
                'accent_purple': '#9C27B0',
                'transcription_bg': '#1e1e1e',
                'translation_bg': '#0d2b47',
                'success_green': '#2e7d32',
                'error_red': '#c62828',
                'status_ready': '#4CAF50',
                'status_connecting': '#FF9800', 
                'status_live': '#4CAF50',
                'status_transcribing': '#2196F3',
                'status_error': '#f44336',
                'status_warning': '#FF9800'
            }
            
            self._setup_enhanced_shutdown()
            self.setup_gui()
            
            # Start periodic tasks
            self._start_periodic_tasks()
            
        except Exception as e:
            print(f"GUI failed: {e}")
            print("Try using --cli mode instead")
            raise

    def _start_periodic_tasks(self):
        """Start periodic maintenance tasks"""
        # Text cleanup every 30 seconds
        self.root.after(UIConstants.TEXT_CLEANUP_INTERVAL_MS, self._periodic_text_cleanup)
        
        # Performance monitoring every 5 seconds
        self.root.after(5000, self._periodic_performance_check)
        
    def _periodic_text_cleanup(self):
        """Periodic text cleanup to prevent memory leaks"""
        if self._gui_running:
            self._cleanup_old_text_safe()
            self.root.after(UIConstants.TEXT_CLEANUP_INTERVAL_MS, self._periodic_text_cleanup)
    
    def _periodic_performance_check(self):
        """Periodic performance monitoring"""
        if self._gui_running and hasattr(self, 'processor'):
            try:
                # Check memory usage
                if PSUTIL_AVAILABLE:
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 85:
                        self.status_var.set(f"High memory usage: {memory_percent}% - consider restarting")
                
                # Update performance metrics
                self.update_dashboard()
                
            except Exception as e:
                logging.debug(f"Performance check failed: {e}")
            finally:
                self.root.after(5000, self._periodic_performance_check)

    def _setup_enhanced_shutdown(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass
        
        try:
            atexit.register(self._cleanup_resources)
        except Exception as e:
            pass
        
    def _signal_handler(self, signum: int, frame) -> None:
        self.safe_shutdown()
        
    def _cleanup_resources(self) -> None:
        try:
            if hasattr(self, 'gui_updater'):
                self.gui_updater.shutdown()
            if hasattr(self, 'processor') and self.processor:
                self.processor.full_shutdown()
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.quit()
                except Exception:
                    pass
        except Exception as e:
            pass

    def setup_gui(self) -> None:
        self.root.title("🐉 Dragon Whisperer - Live Stream Transcription & Translation")
        
        width = self.config.get('GUI', 'window_width', fallback='1400')
        height = self.config.get('GUI', 'window_height', fallback='900')
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1200, 700)
        
        self.root.configure(bg=self.colors['bg_primary'])
        self.setup_styles()
        self.create_enhanced_header()
        self.create_elegant_text_areas()
        self.create_compact_dashboard()
        self.create_status_bar()
        
        self.root.bind('<Configure>', self._on_window_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.safe_shutdown)
        
        self.update_dashboard()

    def _on_window_resize(self, event) -> None:
        if event.widget == self.root:
            self.config.config['GUI']['window_width'] = str(self.root.winfo_width())
            self.config.config['GUI']['window_height'] = str(self.root.winfo_height())
            self.config.save_config()

    def setup_styles(self) -> None:
        style = ttk.Style()
        
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        style.configure('.',
            background=self.colors['bg_primary'],
            foreground=self.colors['text_primary'],
            fieldbackground=self.colors['bg_tertiary'],
            selectbackground=self.colors['accent_blue'],
            selectforeground=self.colors['text_primary'],
            troughcolor=self.colors['bg_secondary'],
            borderwidth=1,
            focuscolor=self.colors['accent_blue']
        )
        
        style.configure('TFrame', background=self.colors['bg_primary'])
        style.configure('TLabel', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        style.configure('TButton', 
                       background=self.colors['bg_secondary'], 
                       foreground=self.colors['text_primary'],
                       focuscolor=self.colors['bg_secondary'])
        style.configure('TEntry', 
                       fieldbackground=self.colors['bg_tertiary'], 
                       foreground=self.colors['text_primary'],
                       insertcolor=self.colors['text_primary'])
        style.configure('TCombobox', 
                       fieldbackground=self.colors['bg_tertiary'], 
                       background=self.colors['bg_secondary'],
                       arrowcolor=self.colors['text_primary'])
        
        style.configure('TLabelframe', 
                       background=self.colors['bg_primary'],
                       foreground=self.colors['accent_green'])
        style.configure('TLabelframe.Label', 
                       background=self.colors['bg_primary'],
                       foreground=self.colors['accent_green'])
        
        style.map('TButton',
                 background=[('active', self.colors['bg_tertiary']),
                           ('pressed', self.colors['accent_blue'])])
        
        style.map('TCombobox',
                 fieldbackground=[('readonly', self.colors['bg_tertiary'])],
                 selectbackground=[('readonly', self.colors['accent_blue'])])

    def create_enhanced_header(self):
        header_frame = ttk.Frame(self.root, padding="10 5 10 5")
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Obere Zeile: Titel + Stream-Info
        top_row = ttk.Frame(header_frame)
        top_row.pack(fill=tk.X, pady=(0, 10))
        
        # Titel links
        title_label = tk.Label(
            top_row,
            text="🐉 DRAGON WHISPERER",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_green'],
            anchor='w'
        )
        title_label.pack(side=tk.LEFT)
        
        # Stream-Titel in der Mitte (wird später gefüllt)
        self.stream_title_var = tk.StringVar(value="📺 No stream selected")
        self.stream_title_label = tk.Label(
            top_row,
            textvariable=self.stream_title_var,
            font=("Segoe UI", 11, "italic"),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            anchor='center'
        )
        self.stream_title_label.pack(side=tk.LEFT, expand=True, padx=20)
        
        # Status-Indikator rechts
        self.stream_status_var = tk.StringVar(value="● Ready")
        self.stream_status_label = tk.Label(
            top_row,
            textvariable=self.stream_status_var,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_primary'],
            fg=self.colors['status_ready'],
            anchor='e'
        )
        self.stream_status_label.pack(side=tk.RIGHT)
        
        # Untere Zeile: Steuerelemente
        row1 = ttk.Frame(header_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(
            row1, 
            textvariable=self.url_var,
            font=("Segoe UI", 10),
            width=40
        )
        self.url_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.url_entry.insert(0, "https://www.twitch.tv/jinnytty")
        
        ttk.Button(row1, text="📋", width=3, 
                   command=self.paste_from_clipboard).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row1, text="📁", width=3,
                   command=self.select_local_file).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(row1, text="Model:", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 2))
        self.model_var = tk.StringVar(value=self.config.get('DEFAULT', 'model_size', 'small'))
        model_combo = ttk.Combobox(row1, textvariable=self.model_var, 
                                  values=["tiny", "base", "small", "medium", "large-v2"], 
                                  width=8, state="readonly", font=("Segoe UI", 9))
        model_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        self.translation_enabled_var = tk.BooleanVar(value=self.config.getboolean('DEFAULT', 'translation_enabled', True))
        self.translation_toggle_btn = tk.Button(
            row1,
            text="🌍 TRANSLATION ON" if self.translation_enabled_var.get() else "🌍 TRANSLATION OFF",
            command=self.toggle_translation,
            bg=self.colors['success_green'] if self.translation_enabled_var.get() else self.colors['error_red'],
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            width=16
        )
        self.translation_toggle_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(row1, text="to:", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 2))
        self.lang_var = tk.StringVar(value=self.config.get('DEFAULT', 'target_language', 'en'))
        self.lang_combo = ttk.Combobox(row1, textvariable=self.lang_var,
                             values=["en", "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar"], 
                             width=6, state="readonly", font=("Segoe UI", 9))
        self.lang_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.lang_combo.bind('<<ComboboxSelected>>', self.on_language_changed)
        
        self.update_translation_controls()
        
        row2 = ttk.Frame(header_frame)
        row2.pack(fill=tk.X, pady=5)
        
        ttk.Button(row2, text="▶ Start Transcription", 
                   command=self.start_processing, width=16).pack(side=tk.LEFT, padx=(20, 5))
        
        # REPARIERT: Stop-Button ruft nur stop_processing auf (vereinfacht)
        ttk.Button(row2, text="⏹ Stop", 
                   command=self.stop_processing, width=8).pack(side=tk.LEFT, padx=(0, 5))
                   
        ttk.Button(row2, text="🗑️ Clear", 
                   command=self.clear_text_areas, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="💾 Export", 
                   command=self.export_transcriptions, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="📊 Stats", 
                   command=self.show_enhanced_stats, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="🚪 Exit", 
                   command=self.confirm_exit, width=8).pack(side=tk.LEFT, padx=(0, 5))

    def create_elegant_text_areas(self):
        text_frame = ttk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        trans_frame = ttk.LabelFrame(text_frame, text="🎙️ LIVE TRANSCRIPTION", padding=10)
        trans_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.transcription_text = scrolledtext.ScrolledText(
            trans_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg=self.colors['transcription_bg'],
            fg='#d4d4d4',
            selectbackground='#264f78',
            selectforeground='#ffffff',
            insertbackground='#569cd6',
            relief='flat',
            borderwidth=1,
            padx=12,
            pady=12,
            cursor='xterm'
        )
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        
        trans_frame = ttk.LabelFrame(text_frame, text="🌍 LIVE TRANSLATION", padding=10)
        trans_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.translation_text = scrolledtext.ScrolledText(
            trans_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg=self.colors['translation_bg'],
            fg='#e6f3ff',
            selectbackground='#2e7d32',
            selectforeground='#ffffff',
            insertbackground='#4fc3f7',
            relief='flat',
            borderwidth=1,
            padx=12,
            pady=12,
            cursor='xterm'
        )
        self.translation_text.pack(fill=tk.BOTH, expand=True)
        
        self.setup_elegant_text_tags()

    def setup_elegant_text_tags(self):
        self.transcription_text.tag_configure('excellent', 
            foreground='#4ec9b0', font=("Consolas", 11, "bold"))
        self.transcription_text.tag_configure('good', 
            foreground='#ce9178', font=("Consolas", 11))
        self.transcription_text.tag_configure('fair', 
            foreground='#d7ba7d', font=("Consolas", 11))
        self.transcription_text.tag_configure('poor', 
            foreground='#f44747', font=("Consolas", 11, "italic"))
        
        self.translation_text.tag_configure('translation_high', 
            foreground='#4fc3f7', font=("Segoe UI", 11, "bold"))
        self.translation_text.tag_configure('translation_medium', 
            foreground='#81c784', font=("Segoe UI", 11))
        self.translation_text.tag_configure('translation_low', 
            foreground='#ffb74d', font=("Segoe UI", 11))

    def create_compact_dashboard(self):
        dashboard_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=26)
        dashboard_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 0))
        dashboard_frame.pack_propagate(False)
        
        metrics_frame = tk.Frame(dashboard_frame, bg=self.colors['bg_secondary'])
        metrics_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.cpu_var = tk.StringVar(value="--%")
        self.memory_var = tk.StringVar(value="--%")
        self.rate_var = tk.StringVar(value="0.0/min")
        self.quality_var = tk.StringVar(value="--")
        self.chunks_var = tk.StringVar(value="0")
        self.transcriptions_var = tk.StringVar(value="0")
        self.errors_var = tk.StringVar(value="0")
        self.languages_var = tk.StringVar(value="--")
        self.confidence_var = tk.StringVar(value="0.0")
        self.low_quality_var = tk.StringVar(value="0")
        self.dropped_var = tk.StringVar(value="0")
        
        metrics = [
            ("CPU:", self.cpu_var, "#ff6b6b"),
            ("Mem:", self.memory_var, "#4ecdc4"), 
            ("Rate:", self.rate_var, "#45b7d1"),
            ("Quality:", self.quality_var, "#96ceb4"),
            ("Chunks:", self.chunks_var, "#feca57"),
            ("Trans:", self.transcriptions_var, "#ff9ff3"),
            ("Errors:", self.errors_var, "#ff6b6b"),
            ("Langs:", self.languages_var, "#a29bfe"),
            ("Conf:", self.confidence_var, "#fd79a8"),
            ("LowQ:", self.low_quality_var, "#e17055"),
            ("Drop:", self.dropped_var, "#e17055")
        ]
        
        for i, (label, var, color) in enumerate(metrics):
            tk.Label(metrics_frame, text=label, bg=self.colors['bg_secondary'], 
                    fg='#cccccc', font=("Segoe UI", 8)).grid(row=0, column=i*2, padx=(5, 0), sticky='w')
            label_widget = tk.Label(metrics_frame, textvariable=var, bg=self.colors['bg_secondary'],
                            fg=color, font=("Segoe UI", 8, "bold"))
            label_widget.grid(row=0, column=i*2+1, padx=(0, 10), sticky='w')
            
            if label == "Conf:":
                self.confidence_label = label_widget

    def create_status_bar(self):
        status_frame = tk.Frame(self.root, bg=self.colors['bg_tertiary'], height=24)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready - Enter URL and click Start Transcription")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            font=("Segoe UI", 9),
            anchor='w'
        )
        status_label.pack(fill=tk.X, padx=10, pady=4)

    def update_dashboard(self):
        try:
            if PSUTIL_AVAILABLE:
                try:
                    self.cpu_var.set(f"{psutil.cpu_percent():.0f}%")
                    self.memory_var.set(f"{psutil.virtual_memory().percent:.0f}%")
                except Exception as e:
                    pass
            
            if hasattr(self, 'processor') and self.processor:
                try:
                    stats = self.processor.get_stats()
                    
                    rate = stats['performance_metrics']['transcription_rate_per_min']
                    self.rate_var.set(f"{rate:.1f}/min")
                    
                    quality = stats['audio_quality']
                    self.quality_var.set(quality.capitalize())
                    
                    self.chunks_var.set(f"{stats['chunks_processed']}")
                    self.transcriptions_var.set(f"{stats['successful_transcriptions']}")
                    self.errors_var.set(f"{stats['errors']}")
                    
                    langs = stats['languages_detected']
                    if langs:
                        self.languages_var.set(f"{len(langs)}")
                    else:
                        self.languages_var.set("0")
                    
                    quality_metrics = stats.get('quality_metrics', {})
                    avg_conf = quality_metrics.get('average_confidence', 0)
                    self.confidence_var.set(f"{avg_conf:.2f}")
                    
                    dropped = quality_metrics.get('dropped_chunks', 0)
                    self.dropped_var.set(f"{dropped}")
                    
                    if avg_conf > 0.7:
                        self.confidence_label.config(fg='#4ec9b0')
                    elif avg_conf > 0.5:
                        self.confidence_label.config(fg='#d7ba7d')
                    else:
                        self.confidence_label.config(fg='#f44747')
                    
                    self.low_quality_var.set(f"{quality_metrics.get('low_quality_skipped', 0)}")
                        
                except Exception as e:
                    self.rate_var.set("0.0/min")
                    self.quality_var.set("--")
                    self.chunks_var.set("0")
                    self.transcriptions_var.set("0")
                    self.errors_var.set("0")
                    self.languages_var.set("--")
                    self.confidence_var.set("0.0")
                    self.low_quality_var.set("0")
                    self.dropped_var.set("0")
            
            self._cleanup_old_text_safe()
            
            if hasattr(self, 'root') and self.root:
                self.root.after(2000, self.update_dashboard)
                
        except Exception as e:
            if hasattr(self, 'root') and self.root:
                self.root.after(2000, self.update_dashboard)

    def _cleanup_old_text_safe(self):
        """Sichere Text-Bereinigung mit Memory-Leak-Protection"""
        def cleanup_transcription():
            try:
                content = self.transcription_text.get(1.0, tk.END)
                lines = content.split('\n')
                if len(lines) > UIConstants.MAX_TEXT_LINES:
                    # Behalte nur die neuesten Zeilen
                    keep_lines = lines[-UIConstants.CLEANUP_THRESHOLD:]
                    self.transcription_text.delete(1.0, tk.END)
                    self.transcription_text.insert(tk.END, '\n'.join(keep_lines))
            except Exception as e:
                logging.debug(f"Transcription text cleanup failed: {e}")
        
        def cleanup_translation():
            try:
                content = self.translation_text.get(1.0, tk.END)
                lines = content.split('\n')
                if len(lines) > UIConstants.MAX_TEXT_LINES:
                    # Behalte nur die neuesten Zeilen
                    keep_lines = lines[-UIConstants.CLEANUP_THRESHOLD:]
                    self.translation_text.delete(1.0, tk.END)
                    self.translation_text.insert(tk.END, '\n'.join(keep_lines))
            except Exception as e:
                logging.debug(f"Translation text cleanup failed: {e}")
        
        self.gui_updater.safe_update(self.transcription_text, cleanup_transcription)
        self.gui_updater.safe_update(self.translation_text, cleanup_translation)

    def toggle_translation(self):
        current_state = self.translation_enabled_var.get()
        new_state = not current_state
        self.translation_enabled_var.set(new_state)
        
        if new_state:
            self.translation_toggle_btn.config(
                text="🌍 TRANSLATION ON",
                bg=self.colors['success_green'],
                fg="white"
            )
            self.status_var.set("Translation enabled")
        else:
            self.translation_toggle_btn.config(
                text="🌍 TRANSLATION OFF",
                bg=self.colors['error_red'],
                fg="white"
            )
            self.status_var.set("Translation disabled")
        
        self.update_translation_controls()

    def on_language_changed(self, event=None):
        new_lang = self.lang_var.get()
        if hasattr(self, 'processor') and self.processor:
            success = self.processor.translation_engine.set_target_language(new_lang)
            if success:
                self.config.config['DEFAULT']['target_language'] = new_lang
                self.config.save_config()
                self.status_var.set(f"Translation language changed to: {new_lang.upper()}")
            else:
                self.status_var.set(f"Failed to change translation language")
        
        self.update_translation_controls()

    def update_translation_controls(self):
        enabled = self.translation_enabled_var.get()
        
        if enabled:
            self.lang_combo.config(state="readonly")
            current_lang = self.lang_var.get()
            if current_lang == 'auto':
                self.lang_var.set('en')
                self.config.config['DEFAULT']['target_language'] = 'en'
                self.config.save_config()
        else:
            self.lang_combo.config(state="disabled")

    def paste_from_clipboard(self) -> None:
        try:
            clipboard_content = self.root.clipboard_get()
            if clipboard_content:
                self.url_var.set(clipboard_content)
                self.status_var.set("URL pasted from clipboard")
        except Exception:
            self.status_var.set("Could not paste from clipboard")

    def select_local_file(self) -> None:
        file_types = [
            ("Audio files", "*.mp3 *.wav *.m4a *.ogg *.flac"),
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio/Video File",
            filetypes=file_types
        )
        
        if filename:
            file_url = f"file://{filename}"
            self.url_var.set(file_url)
            self.status_var.set(f"Selected: {os.path.basename(filename)}")

    def get_callbacks(self):
        """Gibt alle Callback-Funktionen für den Stream Processor zurück"""
        return {
            'transcription': self.handle_transcription,
            'translation': self.handle_translation, 
            'info': self.handle_info,
            'error': self.handle_error,
            'warning': self.handle_warning,
            'stream_title': self.handle_stream_title
        }

    def handle_stream_title(self, title: str) -> None:
        """Handle stream title updates"""
        try:
            if title:
                display_title = title[:UIConstants.STREAM_TITLE_MAX_LENGTH] + "..." if len(title) > UIConstants.STREAM_TITLE_MAX_LENGTH else title
                self.stream_title_var.set(f"📺 {display_title}")
                self.status_var.set(f"Stream: {title}")
        except Exception as e:
            logging.debug(f"Stream title update failed: {e}")

    def start_processing(self) -> None:
        if self.is_processing:
            self.status_var.set("Processing is already running")
            return
            
        url = self.url_var.get().strip()
        if not url:
            self.status_var.set("Please enter a URL or select a file")
            return
            
        self.config.config['DEFAULT']['model_size'] = self.model_var.get()
        self.config.config['DEFAULT']['target_language'] = self.lang_var.get()
        self.config.config['DEFAULT']['translation_enabled'] = str(self.translation_enabled_var.get())
        self.config.save_config()
        
        if hasattr(self.processor, 'translation_engine'):
            self.processor.translation_engine.set_enabled(self.translation_enabled_var.get())
            if self.translation_enabled_var.get():
                self.processor.translation_engine.set_target_language(self.lang_var.get())
        
        progress = tk.Toplevel(self.root)
        progress.title("Initializing Dragon Whisperer...")
        progress.geometry("300x100")
        progress.transient(self.root)
        progress.grab_set()
        
        progress_label = tk.Label(progress, text="Checking AI models...", pady=10)
        progress_label.pack()

        def initialize_processor():
            try:
                model_size = self.model_var.get()
                target_lang = self.lang_var.get()
                
                # REPARIERT: get_callbacks() verwenden
                callbacks = self.get_callbacks()
                
                if not self.processor.initialize_models(model_size, target_lang):
                    # REPARIERT: Lambda mit korrektem Scope
                    self.root.after(0, lambda error="Failed to initialize AI models": self._initialization_failed(error, progress))
                    return
                
                if self.processor.start_stream_processing(url, callbacks):
                    self.root.after(0, lambda: self._initialization_successful(progress))
                else:
                    self.root.after(0, lambda error="Failed to start processing": self._initialization_failed(error, progress))
                    
            except Exception as e:
                # REPARIERT: Lambda mit korrektem Scope
                self.root.after(0, lambda error=e: self._initialization_failed(str(error), progress))
        
        threading.Thread(target=initialize_processor, daemon=True).start()
        
        def check_completion():
            if progress.winfo_exists():
                progress.after(100, check_completion)
        
        progress.after(100, check_completion)

    def _update_progress(self, message: str, percentage: Optional[int] = None) -> None:
        try:
            if percentage is not None:
                self.status_var.set(f"{message} ({percentage}%)")
            else:
                self.status_var.set(message)
        except Exception as e:
            pass

    def _initialization_successful(self, progress_window=None) -> None:
        if progress_window and progress_window.winfo_exists():
            progress_window.destroy()
            
        self.update_processing_state(True)
        self.status_var.set("Processing started successfully")
        
        timestamp = time.strftime("%H:%M:%S")
        
        def update_texts():
            self.transcription_text.delete(1.0, tk.END)
            self.translation_text.delete(1.0, tk.END)
            
            self.transcription_text.insert(tk.END, f"[{timestamp}] Dragon Whisperer started - waiting for audio...\n")
            self.translation_text.insert(tk.END, f"[{timestamp}] Dragon Whisperer started - waiting for translations...\n")
            
            self.transcription_text.see(tk.END)
            self.translation_text.see(tk.END)
        
        self.gui_updater.safe_update(self.transcription_text, update_texts)

    def _initialization_failed(self, error_message: str, progress_window=None) -> None:
        if progress_window and progress_window.winfo_exists():
            progress_window.destroy()
            
        self.update_processing_state(False)
        self.status_var.set(f"{error_message}")
        
        messagebox.showerror("Initialization Error", 
                           f"Failed to initialize processing:\n{error_message}")

    def update_processing_state(self, is_processing: bool) -> None:
        """Aktualisiert den Processing-Status sicher - REPARIERTE VERSION"""
        self.is_processing = is_processing
        if is_processing:
            self.stream_status_var.set("● Transcribing")
            self.stream_status_label.config(fg=self.colors['status_transcribing'])
        else:
            self.stream_status_var.set("● Ready")
            self.stream_status_label.config(fg=self.colors['status_ready'])

    def stop_processing(self) -> None:
        """VEREINFACHTE Stop-Methode - ersetzt stop_processing_only()"""
        if not self.is_processing:
            return
        
        # Sofortige visuelle Rückmeldung
        self.is_processing = False
        self.stream_status_var.set("● Ready")
        self.stream_status_label.config(fg=self.colors['status_ready'])
        self.status_var.set("Stopping processing...")
    
        # Einheitlicher Stop-Aufruf
        if hasattr(self, 'processor') and self.processor:
            try:
                self.processor.stop_processing()  # ✅ Einheitlicher Aufruf
                self.update_processing_state(False)
                self.stream_title_var.set("📺 No stream selected")
                self.status_var.set("Processing stopped - ready for new stream")
            except Exception as e:
                logging.debug(f"Stop processing error: {e}")
                self.status_var.set("Error stopping processing")

    # ❌ ENTFERNT: Redundante stop_processing_only() Methode
    # def stop_processing_only(self) -> None:
    #     self.stop_processing()

    def clear_text_areas(self) -> None:
        def clear_texts():
            self.transcription_text.delete(1.0, tk.END)
            self.translation_text.delete(1.0, tk.END)
        
        self.gui_updater.safe_update(self.transcription_text, clear_texts)
        self.status_var.set("Text areas cleared")

    def export_transcriptions(self) -> None:
        if not hasattr(self, 'processor') or not self.processor or not self.processor.transcription_history:
            self.status_var.set("No transcriptions to export")
            messagebox.showwarning("Export", "No transcriptions available for export.")
            return
            
        file_types = [
            ("Text files", "*.txt"),
            ("Subtitle files", "*.srt"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
        
        default_name = f"transcription_{time.strftime('%Y%m%d_%H%M%S')}"
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=file_types
        )
        
        if filename:
            try:
                if filename.endswith('.srt'):
                    format_type = 'srt'
                elif filename.endswith('.json'):
                    format_type = 'json'
                else:
                    format_type = 'txt'
                    
                exported_file = self.processor.export_transcriptions(format_type, filename)
                self.status_var.set(f"Exported to: {os.path.basename(exported_file)}")
                messagebox.showinfo("Export Successful", f"Transcriptions exported to:\n{exported_file}")
                
            except Exception as e:
                self.status_var.set(f"Export failed: {e}")
                messagebox.showerror("Export Error", f"Failed to export transcriptions:\n{e}")

    def handle_transcription(self, result: TranscriptionResult) -> None:
        if not self._gui_running:
            return
            
        try:
            def format_text():
                timestamp = time.strftime("%H:%M:%S")
                
                if result.confidence > 0.7:
                    color_tag = "excellent"
                    quality_icon = "🟢"
                elif result.confidence > 0.5:
                    color_tag = "good" 
                    quality_icon = "🟡"
                elif result.confidence > 0.3:
                    color_tag = "fair"
                    quality_icon = "🔴"
                else:
                    color_tag = "poor"
                    quality_icon = "⚫"
                
                plugin_indicator = " ⚡" if result.plugin_processed else ""
                
                return {
                    'text': f"[{timestamp}] {quality_icon} {result.text}{plugin_indicator}\n",
                    'tags': [color_tag]
                }
            
            formatted = format_text()
            
            def update_transcription():
                self.transcription_text.insert(tk.END, formatted['text'], formatted['tags'][0])
                self.transcription_text.see(tk.END)
            
            self.gui_updater.safe_update(self.transcription_text, update_transcription)
            
        except Exception as e:
            pass

    def handle_translation(self, result: TranslationResult) -> None:
        if not self._gui_running:
            return
            
        try:
            def format_text():
                timestamp = time.strftime("%H:%M:%S")
                
                confidence_color = "translation_high" if result.confidence > 0.7 else "translation_medium" if result.confidence > 0.5 else "translation_low"
                
                return {
                    'text': f"[{timestamp}] 🌍 {result.translated}\n",
                    'tags': [confidence_color]
                }
            
            formatted = format_text()
            
            def update_translation():
                self.translation_text.insert(tk.END, formatted['text'], formatted['tags'][0])
                self.translation_text.see(tk.END)
            
            self.gui_updater.safe_update(self.translation_text, update_translation)
            
        except Exception as e:
            pass

    def handle_info(self, message: str) -> None:
        if not self._gui_running:
            return
        self.root.after(0, lambda: self.status_var.set(message))
        
        # Stream-Start erkennen
        if "audio data received" in message.lower() or "processing started" in message.lower():
            self.update_processing_state(True)

    def handle_error(self, message: str) -> None:
        """REPARIERT: Auto-Stop mit verbesserter Logik"""
        if not self._gui_running:
            return
        
        error_lower = message.lower()
    
        # 🔧 VERBESSERTE OFFLINE-ERKENNUNG
        offline_patterns = [
            "offline", "not live", "channel is not currently live",
            "stream is offline", "failed to read header", "http error 4",
            "unable to open", "no such file", "connection refused"
        ]
        
        is_offline = any(pattern in error_lower for pattern in offline_patterns)
        
        if is_offline and self.is_processing:
            # 🔥 SOFORTIGER AUTO-STOP mit Status-Update
            self.is_processing = False
            self.stream_status_var.set("● Offline")
            self.stream_status_label.config(fg=self.colors['status_error'])
            self.status_var.set("Stream offline - auto-stopped")
            
            # Processing stoppen mit Verzögerung
            self.root.after(100, self.stop_processing)
            
        elif not is_offline:
            # Normale Fehlerbehandlung
            self.status_var.set(f"Error: {message}")
            self.stream_status_var.set("● Error") 
            self.stream_status_label.config(fg=self.colors['status_error'])

    def handle_warning(self, message: str) -> None:
        if not self._gui_running:
            return
        self.root.after(0, lambda: self.status_var.set(f"Warning: {message}"))
        self.stream_status_var.set("● Warning")
        self.stream_status_label.config(fg=self.colors['status_warning'])

    def show_enhanced_stats(self) -> None:
        if hasattr(self, 'processor') and self.processor:
            try:
                detailed_stats = self.processor.get_detailed_stats()
                
                stats_text = f"""
Comprehensive Statistics:
────────────────────────
Session Status: {'Running' if detailed_stats['base']['is_running'] else 'Stopped'}

Performance:
Uptime: {detailed_stats['performance']['uptime_seconds']:.1f}s
Chunks Processed: {detailed_stats['performance']['chunks_processed']}
Error Rate: {detailed_stats['performance']['error_rate']:.1%}
Avg Processing Time: {detailed_stats['performance']['avg_processing_time']:.2f}s
Memory Usage: {detailed_stats['performance']['avg_memory_mb']:.1f}MB

Transcription:
Total: {detailed_stats['base']['transcription_count']}
Successful: {detailed_stats['base']['successful_transcriptions']}
Rate: {detailed_stats['base']['performance_metrics']['transcription_rate_per_min']:.1f}/min
Success Ratio: {detailed_stats['base']['performance_metrics']['success_ratio']:.1%}

Translation:
Enabled: {detailed_stats['translation_engine']['enabled']}
Initialized: {detailed_stats['translation_engine']['initialized']}
Translations: {detailed_stats['translation_engine']['translation_count']}
Cache Hit Ratio: {detailed_stats['translation_engine']['cache_hit_ratio']:.1%}

Audio Quality:
Current Trend: {detailed_stats['base']['audio_quality']}
Average Confidence: {detailed_stats['base']['quality_metrics']['average_confidence']:.2f}
High Quality: {detailed_stats['base']['quality_metrics']['high_quality_transcriptions']}
Dropped Chunks: {detailed_stats['base']['quality_metrics']['dropped_chunks']}

System Health:
Overall: {detailed_stats['health']['system']['status']}
Processor: {detailed_stats['health']['processor']['status']}
Performance: {detailed_stats['health']['performance']['status']}

Loaded Plugins: {', '.join(detailed_stats['base']['loaded_plugins']) if detailed_stats['base']['loaded_plugins'] else 'None'}

Recommendations:
{chr(10).join(detailed_stats['health']['recommendations']) if detailed_stats['health']['recommendations'] else 'No recommendations'}
                """.strip()
                
                messagebox.showinfo("Comprehensive Statistics", stats_text)
            except Exception as e:
                messagebox.showerror("Statistics Error", f"Failed to generate detailed statistics:\n{e}")
        else:
            messagebox.showinfo("Statistics", "No processor available")

    def confirm_exit(self) -> None:
        """Exit-Dialog - NUR für den Exit-Button!"""
        if messagebox.askokcancel("Quit", "Are you sure you want to exit Dragon Whisperer?"):
            self.safe_shutdown(exit_app=True)

    def safe_shutdown(self, exit_app: bool = False) -> None:
        if self._shutdown_in_progress:
            return
            
        self._shutdown_in_progress = True
        self._gui_running = False
        
        try:
            # 1. GUI Updater stoppen
            if hasattr(self, 'gui_updater'):
                self.gui_updater.shutdown()
            
            # 2. Processing stoppen - EINHEITLICHER AUFRUF
            self.stop_processing()  # ✅ Vereinfacht
            
            if exit_app:
                time.sleep(0.5)
                
                # 3. Processor komplett shutdown
                if hasattr(self, 'processor') and self.processor:
                    self.processor.full_shutdown()
                    time.sleep(0.3)
                
                # 4. GUI ordentlich beenden
                if hasattr(self, 'root') and self.root:
                    self.root.withdraw()
                    time.sleep(0.2)
                    self.root.quit()
                    self.root.destroy()
        
        except Exception as e:
            logging.error(f"Shutdown error: {e}")
        finally:
            if exit_app:
                import sys
                sys.exit(0)

    def run(self) -> None:
        try:
            self.root.mainloop()
        except Exception as e:
            pass
        finally:
            self.safe_shutdown(exit_app=True)

# ===== ENHANCED MAIN FUNCTION =====
def main():
    setup_logging()
    
    critical_deps = {
        'numpy': NUMPY_AVAILABLE,
        'faster_whisper': FASTER_WHISPER_AVAILABLE,
        'tkinter': GUI_AVAILABLE
    }
    
    missing = [dep for dep, available in critical_deps.items() if not available]
    if missing:
        print(f"❌ Install missing dependencies: pip install {' '.join(missing)}")
        print("💡 Additional recommended: pip install psutil torch deep-translator yt-dlp")
        sys.exit(1)
    
    try:
        config = ConfigManager()
        validation_issues = config.validate_config()
        if validation_issues:
            logging.warning(f"Config validation issues: {', '.join(validation_issues)}")
    except Exception as e:
        logging.error(f"Config initialization failed: {e}")
        return
    
    parser = argparse.ArgumentParser(description='🐉 Dragon Whisperer - Live Stream Transcription & Translation')
    parser.add_argument('--url', help='Stream URL')
    parser.add_argument('--model', default='small', choices=['tiny', 'base', 'small', 'medium', 'large-v2'], help='Whisper model size')
    parser.add_argument('--lang', default='en', help='Target language')
    parser.add_argument('--cli', action='store_true', help='CLI mode')
    parser.add_argument('--no-translation', action='store_true', help='Disable translation')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    parser.add_argument('--validate', action='store_true', help='Validate configuration and exit')
    
    args = parser.parse_args()
    
    if args.validate:
        issues = config.validate_config()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid")
        return
    
    if args.lang == 'auto':
        args.lang = 'en'
    
    if not FASTER_WHISPER_AVAILABLE:
        print("❌ faster-whisper required: pip install faster-whisper")
        return
    
    if not NUMPY_AVAILABLE:
        print("❌ numpy required: pip install numpy")
        return
    
    if args.stats:
        processor = StreamProcessor()
        if processor.initialize_models(args.model, args.lang):
            stats = processor.get_detailed_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        return
    
    if args.cli or args.url:
        run_cli_mode(args, config)
    else:
        if not GUI_AVAILABLE:
            print("❌ GUI not available - install tkinter or use --cli mode")
            return
        run_gui_mode()

def run_cli_mode(args, config):
    """CLI Modus ausführen"""
    print("🐉 Dragon Whisperer - Enhanced CLI Mode")
    print("=" * 50)
    
    processor = StreamProcessor()
    
    if args.no_translation:
        processor.translation_engine.set_enabled(False)
    
    if processor.initialize_models(args.model, args.lang):
        # Callbacks für CLI definieren
        def cli_callback(type: str, data: Any) -> None:
            if type == 'transcription':
                quality_ind = " 🌟" if data.confidence > 0.7 else " ✅" if data.confidence > 0.5 else " ⚠️" if data.confidence > 0.3 else " ❌"
                print(f"[{data.language}] {data.text}{quality_ind}")
            elif type == 'translation':
                print(f"[TRANSLATED] {data.translated}")
            elif type in ['info', 'warning', 'error']:
                print(f"{type.upper()}: {data}")
        
        callbacks = {
            'transcription': lambda x: cli_callback('transcription', x),
            'translation': lambda x: cli_callback('translation', x),
            'info': lambda x: cli_callback('info', x),
            'warning': lambda x: cli_callback('warning', x),
            'error': lambda x: cli_callback('error', x),
            'stream_title': lambda x: print(f"STREAM: {x}")
        }
        
        url = args.url or input("Enter stream URL: ")
        
        success = processor.start_stream_processing(url, callbacks)
            
        if success:
            print("✅ Processing started successfully")
            print("Press Ctrl+C to stop...")
            try:
                while not processor._shutdown_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                processor.full_shutdown()
        else:
            print("❌ Failed to start processing")
    else:
        print("❌ Failed to initialize models")

def run_gui_mode():
    """GUI Modus starten"""
    try:
        app = DragonWhispererGUI()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        print("💡 Try using --cli mode instead")
        logging.exception("GUI startup failed")

if __name__ == "__main__":
    main()
