#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐉 THE DRAGON WHISPERER v1.0 - Ultimate Stream Transcription & Translation + SUBTITLES (Dark-Edition)
"""

import logging
import warnings
import os
import sys
import collections
import gc
import threading
import time
import signal
import json
import queue
import shutil
import subprocess
import tempfile
import hashlib
import re
import psutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError, Future
from functools import wraps

# Suppress irrelevant PyTorch future warnings (GPU-Monitoring)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

# === IMPORT CHECKS ===
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("❌ GUI nicht verfügbar - Headless-Modus nicht unterstützt")

# === OPTIMIZED WARNING HANDLING ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", message=".*cuda.*")
warnings.filterwarnings("ignore", message=".*torch.*")

# Environment optimization for CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTHONHASHSEED'] = '0'

# === EXCELLENCE SIGNAL HANDLER ===
class ExcellenceSignalHandler:
    """
    Central signal management for graceful shutdown with thread-safe operations.
    """

    _instance = None
    _shutdown_requested = False
    _lock = threading.RLock()
    _shutdown_event = threading.Event()
    _cleanup_operations = []
    _cleanup_executed = False
    _shutdown_in_progress = False
    _signal_count = 0
    _signal_count_lock = threading.Lock()

    @classmethod
    def setup(cls):
        """Initialize signal handlers for graceful shutdown."""
        if cls._instance is None:
            cls._instance = cls()

        def signal_handler(signum, frame):
            """Atomic signal handler with thread-safe counting."""
            with cls._lock:
                with cls._signal_count_lock:
                    cls._signal_count += 1
                    current_count = cls._signal_count

                if cls._shutdown_requested or cls._shutdown_in_progress:
                    if current_count >= 3:
                        print(f"\n🔴 FORCED SHUTDOWN after {current_count} signals!")
                        os._exit(1)
                    else:
                        print(f"\n⚠️  Shutdown already in progress... ({current_count}/3)")
                    return

                cls._shutdown_requested = True
                cls._shutdown_in_progress = True
                cls._shutdown_event.set()

            print(f"\n🔒 Excellence Signal {signum} - Graceful excellence shutdown...")
            print("💾 Saving state and cleaning up resources...")

            try:
                cls._execute_immediate_cleanup()
            except Exception as e:
                print(f"⚠️  Immediate cleanup warning: {e}")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, signal_handler)

    @classmethod
    def _execute_immediate_cleanup(cls):
        """Execute immediate cleanup operations including CUDA cache clearing."""
        try:
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except:
            pass

    @classmethod
    def register_cleanup(cls, cleanup_func: Callable):
        """
        Register a cleanup function to be executed during shutdown.
        """
        with cls._lock:
            if cleanup_func not in cls._cleanup_operations:
                cls._cleanup_operations.append(cleanup_func)

    @classmethod
    def execute_cleanup(cls):
        """Execute all registered cleanup functions in reverse order."""
        with cls._lock:
            if cls._cleanup_executed:
                return

            for cleanup_func in reversed(cls._cleanup_operations):
                try:
                    cleanup_func()
                except Exception as e:
                    logging.warning(f"⚠️  Cleanup warning: {e}")

            cls._cleanup_executed = True
            cls._shutdown_in_progress = False

    @classmethod
    def should_shutdown(cls):
        """
        Check if shutdown has been requested.
        """
        with cls._lock:
            return (cls._shutdown_requested or
                    cls._shutdown_event.is_set() or
                    cls._shutdown_in_progress)

    @classmethod
    def is_cleanup_executed(cls):
        """
        Check if cleanup has already been executed.
        """
        with cls._lock:
            return cls._cleanup_executed

    @classmethod
    def reset_for_testing(cls):
        """Reset signal handler state for testing purposes."""
        with cls._lock:
            cls._shutdown_requested = False
            cls._shutdown_in_progress = False
            cls._cleanup_executed = False
            cls._shutdown_event.clear()
            with cls._signal_count_lock:
                cls._signal_count = 0

    @classmethod
    def register_auto_recovery(cls, recovery_func: Callable, priority: int = 1):
        """
        Register auto-recovery handlers with priority.
        """
        with cls._lock:
            if not hasattr(cls, '_recovery_handlers'):
                cls._recovery_handlers = []
            
            cls._recovery_handlers.append({
                'function': recovery_func,
                'priority': priority,
                'name': recovery_func.__name__
            })
            
            cls._recovery_handlers.sort(key=lambda x: x['priority'], reverse=True)
            
            logging.info(f"Auto-recovery handler registered: {recovery_func.__name__} (priority: {priority})")

    @classmethod
    def _execute_auto_recovery(cls):
        """Execute auto-recovery procedures with timeout protection."""
        with cls._lock:
            if not hasattr(cls, '_recovery_handlers') or not cls._recovery_handlers:
                logging.info("No auto-recovery handlers registered")
                return 
                
            recovery_results = {
                'timestamp': time.time(),
                'total_handlers': len(cls._recovery_handlers),
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'details': []
            }
                         
            logging.info(f"Executing auto-recovery with {len(cls._recovery_handlers)} handlers...")
            
            for handler in cls._recovery_handlers:
                handler_name = handler['name']
                handler_func = handler['function']
                priority = handler['priority']
                
                try:
                    logging.info(f"Executing recovery handler: {handler_name} (priority: {priority})")
                    
                    def execute_with_timeout():
                        try:
                            return handler_func()
                        except Exception as e:
                            raise e
                    
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(execute_with_timeout)
                        result = future.result(timeout=30.0)
                    
                    recovery_results['successful_recoveries'] += 1
                    recovery_results['details'].append({
                        'handler': handler_name,
                        'status': 'success',
                        'priority': priority,
                        'result': str(result) if result else 'No result returned'
                    })
                    
                    logging.info(f"Recovery handler {handler_name} completed successfully")
                    
                except TimeoutError:
                    recovery_results['failed_recoveries'] += 1
                    recovery_results['details'].append({
                        'handler': handler_name,
                        'status': 'timeout',
                        'priority': priority,
                        'error': 'Handler execution timed out (30s)'
                    })
                    logging.error(f"Recovery handler {handler_name} timed out")
                    
                except Exception as e:
                    recovery_results['failed_recoveries'] += 1
                    recovery_results['details'].append({
                        'handler': handler_name,
                        'status': 'error',
                        'priority': priority,
                        'error': str(e)
                    })
                    logging.error(f"Recovery handler {handler_name} failed: {e}")
            
            success_rate = (recovery_results['successful_recoveries'] / 
                          recovery_results['total_handlers']) * 100
            logging.info(f"Auto-recovery completed: {success_rate:.1f}% success rate")
            
            return recovery_results

    @classmethod
    def trigger_emergency_recovery(cls):
        """
        Manually trigger emergency recovery procedures.
        """
        with cls._lock:
            if cls._shutdown_in_progress:
                logging.warning("Cannot trigger recovery during shutdown")
                return False
            
            logging.critical("EMERGENCY RECOVERY TRIGGERED MANUALLY")
            
            recovery_results = cls._execute_auto_recovery()
            
            if recovery_results['successful_recoveries'] > 0:
                logging.info(f"Emergency recovery partially successful: "
                           f"{recovery_results['successful_recoveries']}/{recovery_results['total_handlers']} handlers succeeded")
                return True
            else:
                logging.error("Emergency recovery failed - no handlers succeeded")
                return False

# Initialize signal handler
ExcellenceSignalHandler.setup()

# === EXCELLENCE CONFIG ===
class ExcellenceConfig:
    """
    Configuration class for performance and memory optimization settings.
    """

    # Performance Excellence
    MAX_SUBPROCESSES = 8
    SUBPROCESS_TIMEOUT = 60
    GUI_OPERATION_TIMEOUT = 10.0
    MEMORY_CHECK_INTERVAL = 15
    MAX_GUI_UPDATES_PER_SECOND = 30

    # Audio Processing
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 5.0
    CHUNK_SIZE_BYTES = 160000 

    # Memory Excellence
    MAX_MEMORY_USAGE = 8 * 1024 * 1024 * 1024  # 8GB
    MAX_CACHE_SIZE = 500
    MAX_TEXT_LINES = 2000

    # AI Excellence
    DEFAULT_BEAM_SIZE = 5
    DEFAULT_TEMPERATURE = 0.0
    ENABLE_VAD_FILTER = True

# === EXCELLENCE PERFORMANCE MONITOR ===
class ExcellencePerformanceMonitor:
    """
    System-wide performance metric collection with Division-by-Zero protection.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if not self._initialized:
            self._metrics = {
                'operations': collections.deque(maxlen=5000),
                'transcription_count': 0,
                'translation_count': 0,
                'processing_times': collections.deque(maxlen=500),
                'memory_peak': 0,
                'start_time': time.time()
            }
            self._lock = threading.RLock()
            self._initialized = True

    def log_operation(self, operation: str, duration: float, details: str = ""):
        """
        Log an operation with timing and details.
        """
        with self._lock:
            self._metrics['operations'].append({
                'timestamp': time.perf_counter(),
                'operation': operation,
                'duration': duration,
                'details': details,
                'memory_used': self._get_memory_usage()
            })

    def log_transcription(self):
        """Increment transcription count atomically."""
        with self._lock:
            self._metrics['transcription_count'] += 1

    def log_translation(self):
        """Increment translation count atomically."""
        with self._lock:
            self._metrics['translation_count'] += 1

    def _get_memory_usage(self) -> int:
        """
        Get current memory usage in bytes.
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            with self._lock:
                self._metrics['memory_peak'] = max(self._metrics['memory_peak'], memory_info.rss)
            return memory_info.rss
        except:
            return 0

    def check_memory_limits(self) -> bool:
        """
        Check if memory usage is within configured limits.
        """
        current_memory = self._get_memory_usage()
        return current_memory <= ExcellenceConfig.MAX_MEMORY_USAGE

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        """
        with self._lock:
            recent_ops = list(self._metrics['operations'])[-100:]
            durations = [op['duration'] for op in recent_ops if op['duration'] > 0]

            uptime_hours = (time.time() - self._metrics['start_time']) / 3600
            if uptime_hours > 0:
                transcripts_per_hour = self._metrics['transcription_count'] / uptime_hours
            else:
                transcripts_per_hour = 0

            if durations:
                avg_processing_time = sum(durations) / len(durations)
            else:
                avg_processing_time = 0

            if uptime_hours > 0:
                operations_per_minute = len(recent_ops) / (uptime_hours * 60)
            else:
                operations_per_minute = 0

            return {
                'total_uptime': time.time() - self._metrics['start_time'],
                'transcriptions': self._metrics['transcription_count'],
                'translations': self._metrics['translation_count'],
                'transcripts_per_hour': transcripts_per_hour,
                'avg_processing_time': avg_processing_time,
                'memory_peak': self._metrics['memory_peak'],
                'current_memory': self._get_memory_usage(),
                'operations_per_minute': operations_per_minute
            }

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information with Division-by-Zero protection.
        """
        with self._lock:
            try:
                system_memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                disk_usage = psutil.disk_usage('/')
                
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                recent_ops = list(self._metrics['operations'])[-50:]
                durations = [op['duration'] for op in recent_ops if op['duration'] > 0]
                error_count = sum(1 for op in recent_ops if 'error' in op.get('details', '').lower())
                
                recent_ops_count = len(recent_ops)
                if recent_ops_count > 0:
                    error_rate = error_count / recent_ops_count
                else:
                    error_rate = 0.0
                
                uptime_seconds = time.time() - self._metrics['start_time']
                if uptime_seconds > 0:
                    transcripts_per_minute = self._metrics['transcription_count'] / (uptime_seconds / 60)
                else:
                    transcripts_per_minute = 0
                
                if durations:
                    avg_processing_time = sum(durations) / len(durations)
                else:
                    avg_processing_time = 0
                
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    
                    'system': {
                        'cpu_percent': round(cpu_percent, 1),
                        'memory_percent': round(system_memory.percent, 1),
                        'memory_used_mb': system_memory.used // (1024 * 1024),
                        'memory_total_mb': system_memory.total // (1024 * 1024),
                        'disk_usage_percent': round(disk_usage.percent, 1),
                        'disk_free_gb': round(disk_usage.free / (1024**3), 1)
                    },
                    
                    'process': {
                        'memory_rss_mb': process_memory.rss // (1024 * 1024),
                        'memory_peak_mb': self._metrics['memory_peak'] // (1024 * 1024),
                        'cpu_percent': round(process_cpu, 1),
                        'thread_count': threading.active_count(),
                        'uptime_seconds': round(uptime_seconds, 1)
                    },
                    
                    'performance': {
                        'transcriptions_total': self._metrics['transcription_count'],
                        'translations_total': self._metrics['translation_count'],
                        'transcripts_per_minute': round(transcripts_per_minute, 2),
                        'avg_processing_time': round(avg_processing_time, 3),
                        'recent_errors': error_count,
                        'operations_in_last_hour': recent_ops_count
                    },
                    
                    'services': {
                        'transcription_engine': self._check_transcription_engine_health(),
                        'translation_engine': self._check_translation_engine_health(),
                        'ffmpeg_manager': self._check_ffmpeg_health(),
                        'memory_manager': self._check_memory_health()
                    },
                    
                    'health_checks': {
                        'memory_within_limits': self.check_memory_limits(),
                        'cpu_under_threshold': cpu_percent < 90,
                        'disk_space_adequate': disk_usage.percent < 85,
                        'recent_error_rate': error_rate
                    }
                }
                
                health_issues = []
                if not health_status['health_checks']['memory_within_limits']:
                    health_issues.append("Memory limits exceeded")
                if not health_status['health_checks']['cpu_under_threshold']:
                    health_issues.append("High CPU usage")
                if not health_status['health_checks']['disk_space_adequate']:
                    health_issues.append("Low disk space")
                if health_status['health_checks']['recent_error_rate'] > 0.1:
                    health_issues.append("High error rate")
                    
                if health_issues:
                    health_status['status'] = 'degraded'
                    health_status['health_issues'] = health_issues
                elif health_status['performance']['recent_errors'] > 5:
                    health_status['status'] = 'warning'
                else:
                    health_status['status'] = 'healthy'
                    
                return health_status
                
            except Exception as e:
                logging.error(f"System health check error: {e}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

    def _check_transcription_engine_health(self) -> Dict[str, Any]:
        """
        Check transcription engine health status.
        """
        try:
            import __main__
            if hasattr(__main__, 'transcription_engine'):
                engine = __main__.transcription_engine
                return {
                    'status': 'available' if hasattr(engine, 'model') and engine.model else 'unavailable',
                    'model_loaded': hasattr(engine, 'model_size') and engine.model_size is not None,
                    'device': getattr(engine, 'device', 'unknown')
                }
        except:
            pass
        return {'status': 'unavailable', 'error': 'Engine not accessible'}

    def _check_translation_engine_health(self) -> Dict[str, Any]:
        """
        Check translation engine health status.
        """
        try:
            import __main__
            if hasattr(__main__, 'translation_engine'):
                engine = __main__.translation_engine
                return {
                    'status': 'available' if hasattr(engine, 'translator') and engine.translator else 'unavailable',
                    'target_language': getattr(engine, 'target_lang', 'unknown')
                }
        except:
            pass
        return {'status': 'unavailable', 'error': 'Engine not accessible'}

    def _check_ffmpeg_health(self) -> Dict[str, Any]:
        """
        Check FFmpeg manager health status.
        """
        try:
            import __main__
            if hasattr(__main__, 'ffmpeg_manager'):
                manager = __main__.ffmpeg_manager
                stats = manager.get_process_stats() if hasattr(manager, 'get_process_stats') else {}
                return {
                    'status': 'active',
                    'active_processes': stats.get('active_processes', 0),
                    'total_processes': stats.get('total_processes', 0)
                }
        except:
            pass
        return {'status': 'unavailable', 'error': 'Manager not accessible'}

    def _check_memory_health(self) -> Dict[str, Any]:
        """
        Check memory manager health status.
        """
        try:
            import __main__
            if hasattr(__main__, 'memory_manager'):
                manager = __main__.memory_manager
                stats = manager.get_memory_stats() if hasattr(manager, 'get_memory_stats') else {}
                return {
                    'status': 'active',
                    'buffer_components': stats.get('buffer_components', 0),
                    'system_usage_percent': stats.get('system_usage_percent', 0)
                }
        except:
            pass
        return {'status': 'unavailable', 'error': 'Manager not accessible'}

    def _collect_enterprise_metrics(self) -> Dict[str, Any]:
        """
        Collect extended metrics for business analytics.
        """
        with self._lock:
            recent_operations = list(self._metrics['operations'])[-100:]
            
            operation_types = {}
            for op in recent_operations:
                op_type = op.get('operation', 'unknown')
                operation_types[op_type] = operation_types.get(op_type, 0) + 1
            
            successful_ops = [op for op in recent_operations if op.get('duration', 0) > 0]
            if recent_operations:
                success_rate = len(successful_ops) / len(recent_operations)
            else:
                success_rate = 1.0
            
            memory_readings = [op.get('memory_used', 0) for op in recent_operations[-10:]]
            if memory_readings:
                avg_memory = sum(memory_readings) / len(memory_readings)
            else:
                avg_memory = 0
            
            durations = [op.get('duration', 0) for op in successful_ops]
            durations.sort()
            if durations:
                p95_index = int(len(durations) * 0.95)
                p99_index = int(len(durations) * 0.99)
                p95_duration = durations[p95_index] if p95_index < len(durations) else durations[-1]
                p99_duration = durations[p99_index] if p99_index < len(durations) else durations[-1]
            else:
                p95_duration = p99_duration = 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'operation_metrics': {
                    'total_operations': len(recent_operations),
                    'success_rate': round(success_rate, 4),
                    'operation_breakdown': operation_types,
                    'avg_response_time': round(sum(durations) / len(durations), 3) if durations else 0,
                    'p95_response_time': round(p95_duration, 3),
                    'p99_response_time': round(p99_duration, 3)
                },
                'resource_metrics': {
                    'avg_memory_usage_mb': avg_memory // (1024 * 1024),
                    'peak_memory_usage_mb': self._metrics['memory_peak'] // (1024 * 1024),
                    'current_memory_usage_mb': self._get_memory_usage() // (1024 * 1024)
                },
                'business_metrics': {
                    'total_transcriptions': self._metrics['transcription_count'],
                    'total_translations': self._metrics['translation_count'],
                    'uptime_hours': round((time.time() - self._metrics['start_time']) / 3600, 2),
                    'throughput_per_minute': round(self._metrics['transcription_count'] / 
                                                ((time.time() - self._metrics['start_time']) / 60), 2) if (time.time() - self._metrics['start_time']) > 0 else 0
                },
                'quality_metrics': {
                    'consecutive_successful_operations': len(successful_ops),
                    'error_trend': 'improving' if success_rate > 0.9 else 'stable' if success_rate > 0.8 else 'degrading',
                    'system_stability': 'high' if success_rate > 0.95 else 'medium' if success_rate > 0.85 else 'low'
                }
            }

    def get_detailed_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics for business insights.
        """
        with self._lock:
            base_health = self.get_system_health()
            enterprise_metrics = self._collect_enterprise_metrics()
            performance_stats = self.get_performance_stats()
            
            analytics = {
                'analytics_timestamp': datetime.now().isoformat(),
                'version': '4.1.3-enterprise',
                
                'health_summary': {
                    'overall_status': base_health.get('status', 'unknown'),
                    'system_health_score': self._calculate_health_score(base_health),
                    'service_availability': self._calculate_service_availability(base_health),
                    'resource_utilization': self._calculate_resource_utilization(base_health)
                },
                
                'performance_analytics': {
                    'throughput_metrics': {
                        'transcriptions_per_hour': performance_stats.get('transcripts_per_hour', 0),
                        'operations_per_minute': performance_stats.get('operations_per_minute', 0),
                        'avg_processing_latency': performance_stats.get('avg_processing_time', 0)
                    },
                    'efficiency_metrics': {
                        'success_rate': enterprise_metrics.get('operation_metrics', {}).get('success_rate', 0),
                        'resource_efficiency': self._calculate_resource_efficiency(base_health, performance_stats),
                        'throughput_trend': self._analyze_throughput_trend()
                    }
                },
                
                'business_intelligence': {
                    'usage_patterns': self._analyze_usage_patterns(),
                    'peak_usage_times': self._identify_peak_usage(),
                    'resource_cost_analysis': self._calculate_cost_metrics(),
                    'service_reliability': self._calculate_reliability_metrics()
                },
                
                'technical_insights': {
                    'bottleneck_analysis': self._identify_bottlenecks(),
                    'optimization_opportunities': self._suggest_optimizations(),
                    'capacity_planning': self._provide_capacity_recommendations()
                },
                
                'raw_metrics': {
                    'health_data': base_health,
                    'enterprise_metrics': enterprise_metrics,
                    'performance_data': performance_stats
                }
            }
            
            return analytics

    def _calculate_health_score(self, health_data: Dict) -> float:
        """
        Calculate overall health score (0-100).
        """
        score = 100.0
        
        if not health_data.get('health_checks', {}).get('memory_within_limits', True):
            score -= 20
        if not health_data.get('health_checks', {}).get('cpu_under_threshold', True):
            score -= 15
        if not health_data.get('health_checks', {}).get('disk_space_adequate', True):
            score -= 10
            
        error_rate = health_data.get('health_checks', {}).get('recent_error_rate', 0)
        score -= min(30, error_rate * 100)
        
        return max(0, round(score, 1))

    def _calculate_service_availability(self, health_data: Dict) -> Dict[str, Any]:
        """
        Calculate service availability metrics.
        """
        services = health_data.get('services', {})
        available_services = sum(1 for service in services.values() 
                               if service.get('status') == 'available')
        total_services = len(services)
        
        if total_services > 0:
            availability_percent = round((available_services / total_services) * 100, 1)
        else:
            availability_percent = 0
        
        return {
            'availability_percent': availability_percent,
            'available_services': available_services,
            'total_services': total_services,
            'degraded_services': [name for name, service in services.items() 
                                if service.get('status') != 'available']
        }

    def _calculate_resource_utilization(self, health_data: Dict) -> Dict[str, Any]:
        """
        Calculate resource utilization metrics.
        """
        system = health_data.get('system', {})
        process = health_data.get('process', {})
        
        memory_total_mb = system.get('memory_total_mb', 1)
        if memory_total_mb > 0:
            process_memory_utilization = round(
                (process.get('memory_rss_mb', 0) / memory_total_mb) * 100, 1
            )
        else:
            process_memory_utilization = 0
        
        return {
            'cpu_utilization': system.get('cpu_percent', 0),
            'memory_utilization': system.get('memory_percent', 0),
            'disk_utilization': system.get('disk_usage_percent', 0),
            'process_memory_utilization': process_memory_utilization
        }

    def _calculate_resource_efficiency(self, health_data: Dict, performance: Dict) -> Dict[str, Any]:
        """
        Calculate resource efficiency metrics.
        """
        throughput = performance.get('transcripts_per_hour', 0)
        memory_used = health_data.get('process', {}).get('memory_rss_mb', 1)
        cpu_used = health_data.get('system', {}).get('cpu_percent', 1)
        
        if memory_used > 0:
            transcripts_per_gb = round(throughput / (memory_used / 1024), 2)
        else:
            transcripts_per_gb = 0
            
        if cpu_used > 0:
            transcripts_per_cpu_percent = round(throughput / cpu_used, 2)
        else:
            transcripts_per_cpu_percent = 0
            
        efficiency_score = round(min(100, (throughput / max(1, memory_used)) * 1000), 1)
        
        return {
            'transcripts_per_gb': transcripts_per_gb,
            'transcripts_per_cpu_percent': transcripts_per_cpu_percent,
            'efficiency_score': efficiency_score
        }

    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """
        Analyze usage patterns for business insights.
        """
        with self._lock:
            recent_ops = list(self._metrics['operations'])[-200:]
            
            op_types = {}
            for op in recent_ops:
                op_type = op.get('operation', 'unknown')
                op_types[op_type] = op_types.get(op_type, 0) + 1
            
            timestamps = [op.get('timestamp', 0) for op in recent_ops]
            if timestamps and len(timestamps) > 1:
                time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                if time_diffs:
                    avg_interval = sum(time_diffs) / len(time_diffs)
                else:
                    avg_interval = 0
            else:
                avg_interval = 0
            
            return {
                'operation_distribution': op_types,
                'avg_operation_interval': round(avg_interval, 2),
                'busiest_operation': max(op_types.items(), key=lambda x: x[1])[0] if op_types else 'none',
                'usage_intensity': 'high' if avg_interval < 1.0 else 'medium' if avg_interval < 5.0 else 'low'
            }

    def _identify_peak_usage(self) -> Dict[str, Any]:
        """
        Identify peak usage times.
        """
        return {
            'current_peak_hour': datetime.now().hour,
            'recommended_scaling_times': ['09:00-11:00', '14:00-16:00'],
            'off_peak_hours': ['02:00-06:00']
        }

    def _calculate_cost_metrics(self) -> Dict[str, Any]:
        """
        Calculate cost-related metrics.
        """
        uptime_hours = (time.time() - self._metrics['start_time']) / 3600
        transcripts = self._metrics['transcription_count']
        
        if uptime_hours > 0:
            transcripts_per_hour = round(transcripts / uptime_hours, 2)
        else:
            transcripts_per_hour = 0
        
        return {
            'transcripts_per_hour': transcripts_per_hour,
            'cost_per_transcript': 0.001,
            'estimated_monthly_cost': round(transcripts * 0.001 * 30, 2),
            'efficiency_improvement_opportunity': '15% potential savings with GPU optimization'
        }

    def _calculate_reliability_metrics(self) -> Dict[str, Any]:
        """
        Calculate service reliability metrics.
        """
        recent_ops = list(self._metrics['operations'])[-100:]
        successful_ops = len([op for op in recent_ops if op.get('duration', 0) > 0])
        
        if recent_ops:
            reliability_score = round((successful_ops / len(recent_ops)) * 100, 1)
        else:
            reliability_score = 100
        
        return {
            'reliability_score': reliability_score,
            'mean_time_between_failures': 'N/A',
            'service_level_objective': '99.5%',
            'current_slo_status': 'meeting' if reliability_score >= 99.5 else 'below'
        }

    def _identify_bottlenecks(self) -> List[str]:
        """
        Identify system bottlenecks.
        """
        bottlenecks = []
        health_data = self.get_system_health()
        
        if health_data.get('system', {}).get('cpu_percent', 0) > 80:
            bottlenecks.append("High CPU usage may impact transcription speed")
        
        if health_data.get('system', {}).get('memory_percent', 0) > 85:
            bottlenecks.append("High memory usage may cause slowdowns")
            
        if health_data.get('health_checks', {}).get('recent_error_rate', 0) > 0.05:
            bottlenecks.append("Elevated error rate indicates stability issues")
            
        return bottlenecks if bottlenecks else ["No significant bottlenecks detected"]

    def _suggest_optimizations(self) -> List[str]:
        """
        Suggest system optimizations.
        """
        optimizations = []
        health_data = self.get_system_health()
        
        if health_data.get('system', {}).get('memory_percent', 0) > 70:
            optimizations.append("Consider increasing system memory or optimizing cache sizes")
            
        if len(self._metrics['operations']) > 1000:
            optimizations.append("High operation count - consider archiving old metrics")
            
        return optimizations if optimizations else ["System is well optimized"]

    def _provide_capacity_recommendations(self) -> Dict[str, Any]:
        """
        Provide capacity planning recommendations.
        """
        current_throughput = self._metrics['transcription_count'] / ((time.time() - self._metrics['start_time']) / 3600) if (time.time() - self._metrics['start_time']) > 0 else 0
        
        return {
            'current_capacity_utilization': round(min(100, (current_throughput / 100) * 100), 1),
            'recommended_scaling_threshold': '80%',
            'projected_growth_capacity': '150% of current load',
            'infrastructure_recommendations': [
                "Monitor memory usage for potential upgrade",
                "Consider GPU acceleration for higher throughput"
            ]
        }

    def _analyze_throughput_trend(self) -> str:
        """
        Analyze throughput trend.
        """
        recent_ops = list(self._metrics['operations'])[-50:]
        if len(recent_ops) < 10:
            return "insufficient_data"
            
        recent_count = len([op for op in recent_ops[-10:] if 'transcription' in op.get('operation', '')])
        previous_count = len([op for op in recent_ops[-20:-10] if 'transcription' in op.get('operation', '')])
        
        if previous_count > 0:
            if recent_count > previous_count * 1.2:
                return "increasing"
            elif recent_count < previous_count * 0.8:
                return "decreasing"
            else:
                return "stable"
        else:
            return "stable" if recent_count == 0 else "increasing"

# === ENHANCED EXCELLENCE DECORATORS ===
def excellence_execution(timeout: int = 60, max_retries: int = 3) -> Callable:
    """
    Decorator for executing functions with timeout and retry logic.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        return future.result(timeout=timeout)

                except TimeoutError:
                    last_exception = TimeoutError(f"Execution timed out after {timeout}s on attempt {attempt+1}")
                    logging.warning(f"Timeout on attempt {attempt+1} on function {func.__name__}")
                
                except Exception as e:
                    last_exception = e
                    logging.error(f"Execution failed on attempt {attempt+1} on function {func.__name__}: {type(e).__name__}: {e}")
                
                if attempt < max_retries:
                    wait_time = min(30, 2 ** attempt)
                    logging.info(f"Waiting {wait_time}s before retry {attempt+2}/{max_retries + 1}...")
                    time.sleep(wait_time)
                    continue
                
                if last_exception is not None:
                    logging.critical(f"All {max_retries + 1} attempts failed for function {func.__name__}.")
                    raise last_exception
            
            raise RuntimeError(f"Decorator logic failed to return or raise for {func.__name__}.")

        return wrapper
    return decorator

# === ENHANCED ERROR HANDLING HIERARCHY ===
class ExcellenceError(Exception):
    """Base exception for excellence system."""
    pass

class ExcellenceMemoryError(ExcellenceError):
    """Memory excellence error."""
    pass

class ExcellenceModelError(ExcellenceError):
    """AI model excellence error."""
    pass

class ExcellenceStreamError(ExcellenceError):
    """Stream excellence error."""
    pass

# === ENHANCED IMMUTABLE DATA STRUCTURES ===
@dataclass(frozen=True)
class ExcellenceTranscriptionResult:
    """
    Immutable transcription result with timing and confidence information.
    """
    text: str
    confidence: float
    language: str
    timestamp: float = None
    start: float = None
    end: float = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())

@dataclass(frozen=True)
class ExcellenceTranslationResult:
    """
    Immutable translation result with language information.
    """
    original: str
    translated: str
    source_lang: str
    target_lang: str
    timestamp: float = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())

# === TOOLTIP IMPLEMENTATION ===
class ToolTip:
    """
    Optimized, stable tooltip implementation with anti-flicker protection.
    """

    _active_tooltip = None
    _last_hide_time = 0
    
    def __init__(self, widget, text, delay=400, duration=5000):
        """
        Initialize tooltip for a widget.
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.duration = duration
        self.tooltip = None
        self.scheduled_id = None
        self.visible = False
        
        self.widget.bind("<Enter>", self._on_enter, add='+')
        self.widget.bind("<Leave>", self._on_leave, add='+')
        self.widget.bind("<ButtonPress>", self._on_click, add='+')
        self.widget.bind("<Motion>", self._on_motion, add='+')

    def _on_enter(self, event=None):
        """Handle mouse enter with cooldown protection."""
        current_time = time.time() * 1000
        if current_time - ToolTip._last_hide_time < 500:
            return
            
        if self.scheduled_id:
            self.widget.after_cancel(self.scheduled_id)
            
        self.scheduled_id = self.widget.after(self.delay, self._show_tooltip)

    def _on_leave(self, event=None):
        """Handle mouse leave and cancel scheduled display."""
        self._hide_tooltip()
        if self.scheduled_id:
            self.widget.after_cancel(self.scheduled_id)
            self.scheduled_id = None

    def _on_click(self, event=None):
        """Hide tooltip on click."""
        self._hide_tooltip()
        if self.scheduled_id:
            self.widget.after_cancel(self.scheduled_id)
            self.scheduled_id = None

    def _on_motion(self, event=None):
        """Handle mouse motion - reset timer if moving between similar widgets."""
        if self.scheduled_id and not self.visible:
            self.widget.after_cancel(self.scheduled_id)
            self.scheduled_id = self.widget.after(self.delay, self._show_tooltip)

    def _show_tooltip(self):
        """Display the tooltip with position adjustment."""
        if not self.widget.winfo_exists() or ToolTip._active_tooltip:
            return
            
        if ToolTip._active_tooltip and ToolTip._active_tooltip != self:
            ToolTip._active_tooltip._hide_tooltip()
            
        try:
            x = self.widget.winfo_rootx() + 25
            y = self.widget.winfo_rooty() + 25
            
            screen_width = self.widget.winfo_screenwidth()
            screen_height = self.widget.winfo_screenheight()
            
            tooltip_width = 350
            if x + tooltip_width > screen_width:
                x = screen_width - tooltip_width - 10
            if y + 100 > screen_height:
                y = self.widget.winfo_rooty() - 80
            
            self.tooltip = tk.Toplevel(self.widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            self.tooltip.configure(bg="#21262d", relief='solid', borderwidth=1)
            
            self.tooltip.attributes('-topmost', True)
            
            label = tk.Label(self.tooltip, 
                           text=self.text,
                           bg="#21262d",
                           fg="#f0f6fc",
                           font=("Segoe UI", 9),
                           justify='left',
                           wraplength=300,
                           padx=10, 
                           pady=6)
            label.pack()
            
            ToolTip._active_tooltip = self
            self.visible = True
            
            self.widget.after(self.duration, self._hide_tooltip)
            
            self.tooltip.bind("<ButtonPress>", self._hide_tooltip)
            label.bind("<ButtonPress>", self._hide_tooltip)
            
        except tk.TclError:
            self._cleanup()

    def _hide_tooltip(self, event=None):
        """Hide the tooltip and update cooldown timer."""
        ToolTip._last_hide_time = time.time() * 1000
        
        if self.visible:
            self.visible = False
            if ToolTip._active_tooltip == self:
                ToolTip._active_tooltip = None
            self._cleanup()

    def _cleanup(self):
        """Clean up tooltip resources."""
        try:
            if self.tooltip and self.tooltip.winfo_exists():
                self.tooltip.destroy()
            self.tooltip = None
            self.visible = False
        except tk.TclError:
            pass
        finally:
            if ToolTip._active_tooltip == self:
                ToolTip._active_tooltip = None

    def destroy(self):
        """Clean up when widget is destroyed."""
        self._hide_tooltip()
        if self.scheduled_id:
            try:
                self.widget.after_cancel(self.scheduled_id)
            except:
                pass

# === EXCELLENCE CACHE ===
class ExcellenceCache:
    """
    LRU-Cache for performance optimization with thread-safe operations.
    """

    def __init__(self, maxsize=128):
        """
        Initialize LRU cache.
        """
        self.maxsize = maxsize
        self._cache = {}
        self._order = collections.deque()
        self._lock = threading.RLock()
        self._lock_timeout = 5.0

    def get(self, key):
        """
        Get value from cache with LRU update.
        """
        if not self._lock.acquire(timeout=self._lock_timeout):
            logging.warning(f"Cache get lock timeout for key: {key}")
            return None
            
        try:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            return None
        finally:
            self._lock.release()

    def put(self, key, value):
        """
        Put value into cache with LRU eviction if needed.
        """
        if not self._lock.acquire(timeout=self._lock_timeout):
            logging.warning(f"Cache put lock timeout for key: {key}")
            return
            
        try:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self.maxsize:
                oldest = self._order.popleft()
                del self._cache[oldest]

            self._cache[key] = value
            self._order.append(key)
        finally:
            self._lock.release()

    def clear(self):
        """Clear all cache entries."""
        if not self._lock.acquire(timeout=self._lock_timeout):
            logging.warning("Cache clear lock timeout")
            return
            
        try:
            self._cache.clear()
            self._order.clear()
        finally:
            self._lock.release()

# === EXCELLENCE FFMPEG MANAGER ===
class ExcellenceFFmpegManager:
    """
    Manages FFmpeg processes for audio extraction with process safety.
    """

    def __init__(self, config=None):
        """Initialize FFmpeg manager with process tracking."""
        self._processes = {}
        self._process_counter = 0
        self._lock = threading.RLock()
        self._active_count = 0
        self._shutting_down = False
        self.config = config or ExcellenceConfig()
        self._cleanup_thread = None
        self._cleanup_running = True
        
        self._pid_tracking = {}
        self._start_cleanup_thread()

    def cleanup_stale_processes(self):
        """Bereinige alte/verwaiste Prozesse die Reconnects blockieren."""
        with self._lock:
            stale_processes = []
            for process_id, process_info in self._processes.items():
                process = process_info['process']
                if (time.time() - process_info['start_time'] > 120 and 
                    process.poll() is not None):
                    stale_processes.append(process_id)
        
            for process_id in stale_processes:
                logging.info(f"🧹 Cleaning stale process: {process_id}")
                self.stop_stream(process_id)

    def _resolve_audio_url_enhanced(self, video_url: str) -> Optional[str]:
        """
        Enhanced URL resolution with multiple fallback methods.
        """
        methods = [
            # Method 1: Standard yt-dlp - BEST AUDIO
            (['yt-dlp', '-g', '-f', 'bestaudio', '--no-warnings', video_url], "yt-dlp bestaudio"),
            # Method 2: yt-dlp with m4a
            (['yt-dlp', '-g', '-f', 'bestaudio[ext=m4a]', '--no-warnings', video_url], "yt-dlp m4a"),
            # Method 3: Direct audio formats
            (['yt-dlp', '-g', '-f', 'ba', '--no-warnings', video_url], "yt-dlp ba"),
            # Method 4: Fallback to any audio
            (['yt-dlp', '-g', '-f', 'wa', '--no-warnings', video_url], "yt-dlp wa"),
        ]
    
        for cmd, method_name in methods:
            try:
                logging.info(f"🔗 Trying {method_name}...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=False)
            
                if result.returncode == 0 and result.stdout.strip():
                    audio_url = result.stdout.strip().split('\n')[0]
                    if audio_url.startswith('http'):
                        logging.info(f"✅ Audio URL resolved with {method_name}: {audio_url[:80]}...")
                        return audio_url
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                logging.warning(f"❌ {method_name} failed: {e}")
                continue
    
        logging.error("❌ All URL resolution methods failed")
        return None
    
    def _detect_and_clean_zombies(self):
        """Detect and clean zombie processes."""
        try:
            with self._lock:
                current_pids = list(self._pid_tracking.keys())
                zombies_found = 0
                
                for pid, process_info in list(self._pid_tracking.items()):
                    process_id = process_info.get('process_id')
                    
                    try:
                        os.kill(pid, 0)
                        
                        process = psutil.Process(pid)
                        if process.status() == psutil.STATUS_ZOMBIE:
                            logging.warning(f"🚨 PROCESS CONTAINMENT: Zombie process detected: PID {pid}, ProcessID {process_id}")
                            
                            self._cleanup_zombie_process(pid, process_id)
                            zombies_found += 1
                            
                    except (OSError, psutil.NoSuchProcess):
                        if pid in self._pid_tracking:
                            del self._pid_tracking[pid]
                        logging.debug(f"🔄 PROCESS CONTAINMENT: Process PID {pid} no longer exists")
                        
                if zombies_found > 0:
                    logging.warning(f"🚨 PROCESS CONTAINMENT: Cleaned {zombies_found} zombie processes")
                    
                self._check_orphaned_processes()
                
        except Exception as e:
            logging.error(f"🚨 PROCESS CONTAINMENT: Zombie detection error: {e}")

    def _cleanup_zombie_process(self, pid: int, process_id: str):
        """
        Cleanup zombie process with termination attempts.
        """
        try:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
            except:
                pass
                
            try:
                os.kill(pid, signal.SIGKILL)
            except:
                pass
                
            if pid in self._pid_tracking:
                del self._pid_tracking[pid]
                
            if process_id in self._processes:
                del self._processes[process_id]
                self._active_count = max(0, self._active_count - 1)
                
            logging.info(f"✅ PROCESS CONTAINMENT: Zombie process PID {pid} cleaned")
            
        except Exception as e:
            logging.error(f"🚨 PROCESS CONTAINMENT: Zombie cleanup error for PID {pid}: {e}")

    def _check_orphaned_processes(self):
        """Check for orphaned processes not in tracking."""
        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            
            for child in children:
                if 'ffmpeg' in child.name().lower():
                    if child.pid not in self._pid_tracking:
                        logging.warning(f"🚨 PROCESS CONTAINMENT: Orphaned FFmpeg process detected: PID {child.pid}")
                        
                        try:
                            child.terminate()
                            child.wait(timeout=5)
                            logging.info(f"✅ PROCESS CONTAINMENT: Orphaned process PID {child.pid} terminated")
                        except:
                            try:
                                child.kill()
                                logging.info(f"✅ PROCESS CONTAINMENT: Orphaned process PID {child.pid} killed")
                            except:
                                logging.error(f"🚨 PROCESS CONTAINMENT: Failed to clean orphaned process PID {child.pid}")
                                
        except Exception as e:
            logging.debug(f"Orphaned process check error: {e}")

    def _start_cleanup_thread(self):
        """Start regular cleanup thread for memory leak prevention."""
        def cleanup_worker():
            while self._cleanup_running:
                try:
                    time.sleep(30)
                    self._perform_regular_cleanup()
                except Exception as e:
                    logging.debug(f"Cleanup thread error: {e}")
                    
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _perform_regular_cleanup(self):
        """Regular cleanup of terminated processes."""
        with self._lock:
            processes_to_remove = []
            for process_id, process_info in self._processes.items():
                process = process_info['process']
                if process.poll() is None:
                    processes_to_remove.append(process_id)
                    
            for process_id in processes_to_remove:
                self._cleanup_process_resources(process_id, process)
                logging.debug(f"Regular cleanup removed finished process: {process_id}")

    def dispose(self):
        """Explicit cleanup method instead of destructor."""
        self._cleanup_running = False
        self._cleanup_all_processes()
        self.stop_all_streams()
        
    def _cleanup_all_processes(self):
        """Clean up all processes including zombies - ONLY called on shutdown."""
        logging.info("🧹 Performing final process cleanup...")
    
        with self._lock:
            # Aktive Prozesse stoppen
            self.stop_all_streams()
        
            # Zombie-Prozesse finden und killen
            try:
                current_process = psutil.Process()
                children = current_process.children(recursive=True)
            
                for child in children:
                    if 'ffmpeg' in child.name().lower():
                        try:
                            child.terminate()
                            child.wait(timeout=2.0)
                            logging.info(f"✅ Terminated FFmpeg process: PID {child.pid}")
                        except:
                            try:
                                child.kill()
                                logging.info(f"✅ Killed FFmpeg process: PID {child.pid}")
                            except:
                                logging.warning(f"⚠️ Could not terminate FFmpeg process: PID {child.pid}")
            except Exception as e:
                logging.debug(f"Final cleanup error: {e}")

    def _resolve_audio_url(self, video_url: str) -> Optional[str]:
        """
        Resolve YouTube/Twitch URLs to direct audio stream URLs using yt-dlp.
        """
        try:
            cmd = ['yt-dlp', '-g', '-f', 'bestaudio[ext=m4a]/bestaudio/best', '--no-warnings', '--no-check-certificate', video_url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=False)
            
            if result.returncode == 0 and result.stdout.strip():
                audio_url = result.stdout.strip().split('\n')[0]
                logging.info(f"✅ Audio URL resolved: {audio_url[:100]}...")
                return audio_url
                
            cmd = ['youtube-dl', '-g', '-f', 'bestaudio[ext=m4a]/bestaudio/best', '--no-warnings', video_url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=False)
            
            if result.returncode == 0 and result.stdout.strip():
                audio_url = result.stdout.strip().split('\n')[0]
                logging.info(f"✅ Audio URL resolved (youtube-dl fallback): {audio_url[:100]}...")
                return audio_url
                
            logging.error("❌ Could not resolve audio URL with yt-dlp or youtube-dl")
            return None
            
        except subprocess.TimeoutExpired:
            logging.error("❌ Audio URL resolution timeout")
            return None
        except Exception as e:
            logging.error(f"❌ Audio URL resolution error: {e}")
            return None

    @excellence_execution(timeout=ExcellenceConfig.SUBPROCESS_TIMEOUT)
    def start_stream(self, audio_url: str, output_queue: queue.Queue, process_id: str) -> Optional[subprocess.Popen]:
        """
        Startet einen FFmpeg-Subprozess, um Audio von einer URL zu streamen.
        """
        logging.info(f"🎯 DEBUG start_stream: Starting with URL: {audio_url}")
        logging.info(f"🎯 DEBUG start_stream: Process ID: {process_id}")
    
        if self.is_active(process_id):
            logging.warning(f"Stream {process_id} ist bereits aktiv.")
            return None

        logging.info("🔗 DEBUG: Resolving audio URL...")
        resolved_url = self._resolve_audio_url_enhanced(audio_url)
        logging.info(f"🔗 DEBUG: Resolved URL: {resolved_url}")
    
        if not resolved_url:
            logging.error(f"Stream-Start abgebrochen: Audio URL für {audio_url} konnte nicht aufgelöst werden.")
            return None
    
        input_options = [
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '30',
            '-rw_timeout', '60000000',
            '-http_persistent', '1'
        ]

        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning',
            *input_options,
            '-i', resolved_url,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(ExcellenceConfig.SAMPLE_RATE),
            '-ac', '1',
            '-fflags', '+genpts',
            'pipe:1'
        ]
    
        logging.info(f"⚙️ DEBUG: Full FFmpeg command: {' '.join(cmd)}")
    
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
        
            logging.info(f"✅ DEBUG: FFmpeg process started successfully with PID: {process.pid}")
        
            time.sleep(0.5)
            poll_result = process.poll()
            logging.info(f"🔍 DEBUG: Process poll result: {poll_result}")
        
            if poll_result is not None:
                stderr_output = process.stderr.read()
                logging.error(f"❌ DEBUG: Process terminated immediately with code: {poll_result}")
                logging.error(f"📝 DEBUG: FFmpeg stderr: {stderr_output.decode('utf-8', errors='ignore')}")
                return None
        
            self._register_process(process_id, process, output_queue)
            logging.info(f"🔥 Stream {process_id} erfolgreich gestartet. PID: {process.pid}. Warte auf Audio-Daten.")
            return process
            
        except FileNotFoundError:
            logging.critical("❌ FFmpeg nicht gefunden! Bitte stellen Sie sicher, dass FFmpeg im Systempfad (PATH) verfügbar ist.")
            return None
        
        except Exception as e:
            logging.critical(f"❌ Kritischer Fehler beim Starten des FFmpeg-Prozesses: {e}")
            return None

    def _register_process(self, process_id: str, process: subprocess.Popen, output_queue: queue.Queue):
        """
        Register process with tracking and safety mechanisms.
        """
        with self._lock:
            self._processes[process_id] = {
                'process': process,
                'output_queue': output_queue,
                'start_time': time.time(),
                'url': 'stream_url',
                'stopping': False
            }
            self._active_count += 1
            
            self._pid_tracking[process.pid] = {
                'process_id': process_id,
                'start_time': time.time()
            }

    def get_process_stats(self) -> Dict[str, Any]:
        """
        Get process statistics for monitoring.
        """
        with self._lock:
            return {
                'active_processes': self._active_count,
                'total_processes': len(self._processes),
                'tracked_pids': len(self._pid_tracking),
                'process_details': [
                    {
                        'process_id': pid,
                        'system_pid': info['process'].pid if info['process'].poll() is None else None,
                        'active': info['process'].poll() is None,
                        'start_time': info['start_time'],
                        'url': info['url'][:50] + '...' if len(info['url']) > 50 else info['url']
                    }
                    for pid, info in self._processes.items()
                ]
            }

    @excellence_execution(timeout=10.0)
    def stop_stream(self, process_id: str) -> bool:
        """
        Stop FFmpeg process with guaranteed termination and improved timeouts.
        """
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
                    logging.info(f"🔄 Stopping FFmpeg process {process_id}...")

                    try:
                        process.terminate()
                        
                        try:
                            process.wait(timeout=1.0)
                            termination_success = True
                            logging.info(f"✅ FFmpeg process {process_id} terminated gracefully")
                        except subprocess.TimeoutExpired:
                            logging.debug(f"⏳ Graceful termination taking longer, waiting 0.5s more...")
                            try:
                                process.wait(timeout=0.5)
                                termination_success = True
                                logging.info(f"✅ FFmpeg process {process_id} terminated after extended wait")
                            except subprocess.TimeoutExpired:
                                logging.warning(f"⚠️ FFmpeg process {process_id} didn't terminate gracefully, forcing...")
                                raise subprocess.TimeoutExpired(process.args, 1.5)

                    except subprocess.TimeoutExpired:
                        try:
                            logging.warning(f"🔄 Forcefully killing FFmpeg process {process_id}...")
                            process.kill()

                            try:
                                process.wait(timeout=1.0)
                                termination_success = True
                                logging.info(f"✅ FFmpeg process {process_id} killed forcefully")
                            except subprocess.TimeoutExpired:
                                logging.error(f"❌ FFmpeg process {process_id} resistant to kill, terminating process tree...")
                                try:
                                    self._kill_process_tree(process.pid)
                                    try:
                                        process.wait(timeout=0.5)
                                        termination_success = True
                                        logging.info(f"✅ FFmpeg process tree {process_id} terminated")
                                    except subprocess.TimeoutExpired:
                                        logging.error(f"❌ Process tree termination timeout for {process_id}")
                                        termination_success = False
                                except Exception as e:
                                    logging.error(f"❌ Process tree termination failed for {process_id}: {e}")
                                    termination_success = False

                        except Exception as e:
                            logging.error(f"❌ Forceful kill failed for {process_id}: {e}")
                            termination_success = False

                else:
                    termination_success = True
                    logging.debug(f"✅ FFmpeg process {process_id} already terminated")

            except Exception as e:
                logging.error(f"❌ Process termination error for {process_id}: {e}")
                termination_success = False
            finally:
                self._cleanup_process_resources(process_id, process)

            return termination_success

    def _kill_process_tree(self, pid: int):
        """
        Terminate entire process tree.
        """
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                              capture_output=True, timeout=5)
            else:
                subprocess.run(['pkill', '-9', '-P', str(pid)],
                              capture_output=True, timeout=5)
        except Exception as e:
            logging.debug(f"Process tree kill attempt failed: {e}")

    def _cleanup_process_resources(self, process_id: str, process: subprocess.Popen):
        """
        Guaranteed cleanup of process resources with memory leak prevention.
        """
        try:
            if process_id in self._processes:
                del self._processes[process_id]
                self._active_count = max(0, self._active_count - 1)

            if process.pid in self._pid_tracking:
                del self._pid_tracking[process.pid]

            resources_cleaned = False
            for attempt in range(3):
                try:
                    if process.stdout and not process.stdout.closed:
                        process.stdout.close()
                    if process.stderr and not process.stderr.closed:
                        process.stderr.close()
                    if process.poll() is None:
                        try:
                            process.terminate()
                            process.wait(timeout=1.0)
                        except:
                            try:
                                process.kill()
                                process.wait(timeout=0.5)
                            except:
                                pass
                    resources_cleaned = True
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    else:
                        logging.warning(f"⚠️ Could not clean all resources for {process_id}: {e}")

            if not resources_cleaned and process.poll() is None:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except:
                    pass

        except Exception as e:
            logging.error(f"❌ Resource cleanup error for {process_id}: {e}")
        finally:
            gc.collect()

    def stop_all_streams(self):
        """Stop all active FFmpeg streams."""
        with self._lock:
            self._shutting_down = True
            process_ids = list(self._processes.keys())

            successful_stops = 0
            total_processes = len(process_ids)

            for process_id in process_ids:
                try:
                    if self.stop_stream(process_id):
                        successful_stops += 1
                    else:
                        logging.warning(f"⚠️ Failed to stop {process_id}")
                except Exception as e:
                    logging.error(f"❌ Error stopping {process_id}: {e}")

            logging.info(f"🔄 Stopped {successful_stops}/{total_processes} FFmpeg processes")
            self._shutting_down = False

    def is_active(self, process_id: str) -> bool:
        """
        Check if process is still active.
        """
        with self._lock:
            if process_id not in self._processes:
                return False
            process = self._processes[process_id]['process']
            return process.poll() is None

    def read_audio_data(self, process_id: str, size: int) -> Optional[bytes]:
        """
        Read audio data from FFmpeg process.
        """
        with self._lock:
            if process_id not in self._processes:
                return None

            process_info = self._processes[process_id]

            if process_info.get('stopping', False):
                return None

            process = process_info['process']
            try:
                data = process.stdout.read(size)
                if data is None or len(data) == 0:
                    if process.poll() is not None:
                        self.stop_stream(process_id)
                        return None
                return data
            except Exception:
                self.stop_stream(process_id)
                return None

# === GUI DECORATOR ===
def excellence_gui_operation(func):
    """
    Enhanced GUI operation decorator with exception handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (tk.TclError, RuntimeError):
            return None
        except Exception as e:
            ExcellencePerformanceMonitor().log_operation(
                "gui_operation_error", 0.0, f"Function:{func.__name__} Error:{str(e)}"
            )
            return None
    return wrapper

# === MODERN DESIGN ===
class ModernColors:
    """Color scheme for modern dark UI design."""
    BG_PRIMARY = "#0d1117"
    BG_SECONDARY = "#161b22"
    BG_TERTIARY = "#21262d"
    BG_HOVER = "#30363d"
    TEXT_PRIMARY = "#f0f6fc"
    TEXT_SECONDARY = "#8b949e"
    TEXT_ACCENT = "#58a6ff"
    SUCCESS = "#3fb950"
    WARNING = "#d29922"
    ERROR = "#f85149"
    BORDER = "#30363d"
    COMBO_BG = "#1e2328"
    COMBO_FG = "#f0f6fc"
    COMBO_SELECTION = "#58a6ff"
    DRAGON_GREEN = "#00ff88"
    CHECKBOX_ACTIVE = "#000000"
    SUBTITLE_ACTIVE = "#00ff88"
    SUBTITLE_INACTIVE = "#8b949e"

class ModernFonts:
    """Font configuration for modern UI."""
    PRIMARY = ("Segoe UI", 9)
    MONOSPACE = ("Cascadia Code", 8)
    BUTTON = ("Segoe UI", 9, "bold")
    TITLE = ("Segoe UI", 11, "bold")
    SUBTITLE = ("Segoe UI", 9, "bold")

# === DARK MESSAGEBOX ===
class DarkMessageBox:
    """Dark Mode Messagebox replacement with modern styling."""

    @staticmethod
    def showinfo(title, message, parent=None):
        """Show information message dialog."""
        return DarkMessageBox._show_dialog(title, message, "info", parent)

    @staticmethod
    def showwarning(title, message, parent=None):
        """Show warning message dialog."""
        return DarkMessageBox._show_dialog(title, message, "warning", parent)

    @staticmethod
    def showerror(title, message, parent=None):
        """Show error message dialog."""
        return DarkMessageBox._show_dialog(title, message, "error", parent)

    @staticmethod
    def askokcancel(title, message, parent=None):
        """Show confirmation dialog with OK/Cancel buttons."""
        dialog = DarkMessageBox._show_dialog(title, message, "question", parent, buttons=True)
        return dialog.result if hasattr(dialog, 'result') else False

    @staticmethod
    def _show_dialog(title, message, msg_type, parent=None, buttons=False):
        """
        Create and show dark themed dialog.
        """
        dialog = tk.Toplevel(parent if parent else tk._default_root)
        dialog.title(title)
        dialog.configure(bg=ModernColors.BG_PRIMARY)
        dialog.resizable(False, False)
        dialog.transient(parent if parent else tk._default_root)
        dialog.grab_set()

        dialog.wait_visibility()

        dialog.update_idletasks()
        if parent:
            x = parent.winfo_x() + (parent.winfo_width() - dialog.winfo_width()) // 2
            y = parent.winfo_y() + (parent.winfo_height() - dialog.winfo_height()) // 2
        else:
            x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
            y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "question": "❓"
        }
        icon = icons.get(msg_type, "💬")

        main_frame = tk.Frame(dialog, bg=ModernColors.BG_PRIMARY, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        content_frame = tk.Frame(main_frame, bg=ModernColors.BG_PRIMARY)
        content_frame.pack(fill='both', expand=True)

        icon_label = tk.Label(content_frame, text=icon, font=("Segoe UI", 16),
                             bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_ACCENT)
        icon_label.grid(row=0, column=0, sticky='n', padx=(0, 15))

        message_label = tk.Label(content_frame, text=message, font=ModernFonts.PRIMARY,
                                bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                                justify='left', wraplength=400)
        message_label.grid(row=0, column=1, sticky='w')

        button_frame = tk.Frame(main_frame, bg=ModernColors.BG_PRIMARY)
        button_frame.pack(fill='x', pady=(20, 0))

        if buttons:
            ok_btn = tk.Button(button_frame, text="OK",
                              command=lambda: DarkMessageBox._set_result(dialog, True),
                              bg=ModernColors.SUCCESS, fg=ModernColors.TEXT_PRIMARY,
                              relief='flat', padx=15)
            ok_btn.pack(side='right', padx=(10, 0))

            cancel_btn = tk.Button(button_frame, text="Abbrechen",
                                  command=lambda: DarkMessageBox._set_result(dialog, False),
                                  bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
                                  relief='flat', padx=15)
            cancel_btn.pack(side='right')
        else:
            ok_btn = tk.Button(button_frame, text="OK", command=dialog.destroy,
                              bg=ModernColors.SUCCESS, fg=ModernColors.TEXT_PRIMARY,
                              relief='flat', padx=15)
            ok_btn.pack(side='right')

        dialog.result = None
        if buttons:
            dialog.wait_window(dialog)
            return dialog.result
        else:
            return None

    @staticmethod
    def _set_result(dialog, result):
        """
        Set dialog result and close.
        """
        dialog.result = result
        dialog.destroy()

# === EXCELLENCE MEMORY MANAGER ===
class ExcellenceMemoryManager:
    """
    Manages text buffers with automatic cleanup and memory optimization.
    """

    def __init__(self):
        """Initialize memory manager with buffer tracking."""
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
        self._start_periodic_cleanup()
        self._start_memory_guardian()

    def _start_memory_guardian(self):
        """Start memory guardian for long-term monitoring >80% warning."""
        def memory_guardian_worker():
            while self._monitoring_active:
                try:
                    time.sleep(60)
                    self._perform_memory_health_check()
                except Exception as e:
                    logging.debug(f"Memory Guardian error: {e}")

        guardian_thread = threading.Thread(target=memory_guardian_worker, daemon=True)
        guardian_thread.start()

    def _perform_memory_health_check(self):
        """Long-term memory health check with >80% warning."""
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
                logging.warning(
                    f"🚨 MEMORY GUARDIAN: System memory usage {system_usage_percent:.1%} "
                    f"({memory_sample['system_mb']}MB) - Consider reducing workload"
                )
            
            if process_usage_percent > self._memory_warning_threshold:
                logging.warning(
                    f"🚨 MEMORY GUARDIAN: Process memory usage {process_usage_percent:.1%} "
                    f"({memory_sample['process_mb']}MB) - Triggering aggressive cleanup"
                )
                self._aggressive_cleanup()
                
            if len(self._long_term_monitor) >= 10:
                recent_samples = list(self._long_term_monitor)[-10:]
                avg_usage = sum(s['system_usage'] for s in recent_samples) / len(recent_samples)
                if avg_usage > 0.75:
                    logging.warning(
                        f"🚨 MEMORY GUARDIAN: Sustained high memory usage {avg_usage:.1%} "
                        f"over last 10 minutes"
                    )
                    
        except Exception as e:
            logging.debug(f"Memory health check error: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics for monitoring.
        """
        try:
            system_memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss
            
            return {
                'system_usage_percent': system_memory.percent,
                'system_used_mb': system_memory.used // (1024 * 1024),
                'system_total_mb': system_memory.total // (1024 * 1024),
                'process_usage_percent': (process_memory / ExcellenceConfig.MAX_MEMORY_USAGE) * 100,
                'process_used_mb': process_memory // (1024 * 1024),
                'process_peak_mb': self._get_peak_memory() // (1024 * 1024),
                'long_term_samples': len(self._long_term_monitor),
                'buffer_components': len(self._buffers),
                'ring_buffer_components': len(self._ring_buffers)
            }
        except Exception as e:
            logging.debug(f"Memory stats error: {e}")
            return {}

    def _get_peak_memory(self) -> int:
        """
        Get peak memory usage.
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0

    def _start_periodic_cleanup(self):
        """Start periodic memory cleanup in background."""
        def periodic_cleanup():
            while True:
                time.sleep(60)
                try:
                    self._perform_periodic_maintenance()
                except Exception as e:
                    logging.debug(f"Periodic maintenance error: {e}")

        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()

    def _perform_periodic_maintenance(self):
        """Perform periodic maintenance tasks."""
        with self._lock:
            current_time = time.time()

            if current_time - self._last_gc_time > self._gc_interval:
                gc.collect()
                self._last_gc_time = current_time
                logging.debug("Periodic garbage collection triggered")

            total_memory = sum(self._buffer_sizes.values())
            if total_memory > self._max_memory_per_component * 0.8:
                def async_aggressive_cleanup():
                    try:
                        self._aggressive_cleanup()
                    except Exception as e:
                        logging.debug(f"Async aggressive cleanup error: {e}")
                
                cleanup_thread = threading.Thread(target=async_aggressive_cleanup, daemon=True)
                cleanup_thread.start()

    def _aggressive_cleanup(self):
        """Aggressive cleanup when memory limits are reached - now non-blocking."""
        logging.info("Performing aggressive memory cleanup (non-blocking)")

        with self._lock:
            for component in list(self._buffers.keys()):
                if component in self._buffers:
                    buffer_size = len(self._buffers[component])
                    if buffer_size > 100:
                        keep_count = max(50, int(buffer_size * 0.5))

                        old_deque = self._buffers[component]
                        new_deque = collections.deque(
                            list(old_deque)[-keep_count:],
                            maxlen=ExcellenceConfig.MAX_TEXT_LINES
                        )
                        self._buffers[component] = new_deque

                        self._buffer_sizes[component] = sum(
                            len(text.encode('utf-8')) for text in new_deque
                        )

        def async_gc():
            gc.collect()
        
        gc_thread = threading.Thread(target=async_gc, daemon=True)
        gc_thread.start()

    def add_text(self, component: str, text: str):
        """
        Add text to storage with automatic cleanup.
        """
        if not text or not text.strip():
            return

        with self._lock:
            if component in self._ring_buffers:
                self._add_to_ring_buffer(component, text)
                return

            if component not in self._buffers:
                self._buffers[component] = collections.deque(maxlen=ExcellenceConfig.MAX_TEXT_LINES)
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
        """
        Add text to ring buffer for O(1) operations.
        """
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
        """Efficient buffer optimization with ring buffer."""
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

    def _resize_ring_buffer(self, component: str, new_size: int):
        """
        Resize ring buffer.
        """
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

    def get_text(self, component: str) -> str:
        """
        Get text from storage with ring buffer support.
        """
        with self._lock:
            if component in self._ring_buffers:
                return self._get_from_ring_buffer(component)
            elif component in self._buffers:
                return '\n'.join(self._buffers[component])
            return ""

    def _get_from_ring_buffer(self, component: str) -> str:
        """
        Get text from ring buffer in correct chronological order.
        """
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
        """
        Clear all text for a component.
        """
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

    def dispose(self):
        """Explicit cleanup method instead of destructor."""
        self._monitoring_active = False
        self._buffers.clear()
        self._buffer_sizes.clear()
        self._ring_buffers.clear()
        self._ring_buffer_pointers.clear()
        self._ring_buffer_sizes.clear()
        self._long_term_monitor.clear()

# === LANGUAGE SUPPORT ===
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

WHISPER_MODELS = [
    "tiny", "tiny.en", "base", "base.en",
    "small", "small.en", "medium", "medium.en",
    "large-v2", "large-v3"
]

# === ASIAN LANGUAGE OPTIMIZATION ===
class AsianLanguageSupport:
    """Optimizations for Asian languages including word segmentation."""

    @staticmethod
    def should_use_word_segmentation(language_code):
        """
        Detect if language needs word segmentation.
        """
        return language_code in ['zh', 'ja', 'ko', 'th']

    @staticmethod
    def optimize_display_text(text, language_code):
        """
        Optimize text display for Asian languages.
        """
        if language_code == 'zh':
            return ' '.join(text)
        elif language_code == 'ja':
            return text.replace('。', '. ').replace('、', ', ')
        elif language_code == 'ko':
            return text
        return text

# === ADVANCED SETTINGS ===
@dataclass
class AdvancedSettings:
    """
    Advanced settings for optimized performance and AI model configuration.
    """
    beam_size: int = 5
    temperature: float = 0.0
    vad_filter: bool = True
    chunk_duration: float = 5.0
    max_cache_size: int = 200
    auto_save_interval: int = 300
    enable_sentiment_analysis: bool = False
    enable_speaker_diarization: bool = False
    max_memory_mb: int = 1024
    gpu_acceleration: bool = True
    optimize_translations: bool = False

    @classmethod
    def load_from_file(cls, filename="dragon_advanced_settings.json"):
        """
        Load advanced settings from JSON file.
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return cls(**data)
        except Exception as e:
            logging.warning(f"Advanced settings load failed: {e}")
        return cls()

    def save_to_file(self, filename="dragon_advanced_settings.json"):
        """
        Save advanced settings to JSON file.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Advanced settings save failed: {e}")

# === PLUGIN SYSTEM ===
class Plugin:
    """Base class for plugins with lifecycle management."""

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize plugin.
        """
        self.name = name
        self.version = version
        self.enabled = True

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        """
        Process transcription result.
        """
        return result

    def on_translation(self, result: ExcellenceTranslationResult) -> ExcellenceTranslationResult:
        """
        Process translation result.
        """
        return result

    def on_startup(self):
        """Called on plugin startup."""
        pass

    def on_shutdown(self):
        """Called on plugin shutdown."""
        pass

class SentimentAnalysisPlugin(Plugin):
    """Analyze sentiment of transcribed text."""

    def __init__(self):
        """Initialize sentiment analysis plugin."""
        super().__init__("Sentiment Analysis", "1.0.0")
        self.sentiment_cache = {}

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        """
        Analyze sentiment of transcribed text.
        """
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
        """Initialize keyword extraction plugin."""
        super().__init__("Keyword Extraction", "1.0.0")
        self.common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    def on_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        """
        Extract keywords from transcribed text.
        """
        return result

class PluginManager:
    """Manages plugin loading and execution."""

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: List[Plugin] = []
        self.enabled = True

    def register_plugin(self, plugin: Plugin):
        """
        Register a plugin.
        """
        self.plugins.append(plugin)
        logging.info(f"Plugin registered: {plugin.name} v{plugin.version}")

    def load_builtin_plugins(self):
        """Load built-in plugins."""
        self.register_plugin(SentimentAnalysisPlugin())
        self.register_plugin(KeywordExtractionPlugin())

    def process_transcription(self, result: ExcellenceTranscriptionResult) -> ExcellenceTranscriptionResult:
        """
        Process transcription through all enabled plugins.
        """
        if not self.enabled:
            return result

        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    result = plugin.on_transcription(result)
                except Exception as e:
                    logging.error(f"Plugin {plugin.name} error: {e}")

        return result

    def process_translation(self, result: ExcellenceTranslationResult) -> ExcellenceTranslationResult:
        """
        Process translation through all enabled plugins.
        """
        if not self.enabled:
            return result

        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    result = plugin.on_translation(result)
                except Exception as e:
                    logging.error(f"Plugin {plugin.name} error: {e}")

        return result

# === TTL CACHE IMPLEMENTATION ===
class ExcellenceTTLCache:
    """
    TTL-based cache for better hit rate with automatic expiration.
    """

    def __init__(self, maxsize=128, ttl=3600):
        """
        Initialize TTL cache.
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
        self._order = collections.deque()
        self._lock = threading.RLock()
        self._cleanup_interval = 300
        self._last_cleanup = time.time()

    def get(self, key):
        """
        Get value with TTL check.
        """
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
        """
        Put value with timestamp.
        """
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
        """Automatic cleanup of expired entries."""
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

    def _remove_key(self, key):
        """
        Remove key from all data structures.
        """
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        if key in self._order:
            self._order.remove(key)

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._order.clear()
            self._last_cleanup = time.time()

# === EXCELLENCE TRANSLATION ENGINE ===
class ExcellenceTranslationEngine:
    """
    Translation engine with functional cache strategy and error recovery.
    """

    def __init__(self, target_lang: str = "de", advanced_settings: AdvancedSettings = None):
        """
        Initialize translation engine with guaranteed functionality.
        """
        self.target_lang = target_lang
        self.translator = None
        self._cache = ExcellenceTTLCache(maxsize=500, ttl=3600)
        self._lock = threading.RLock()
        self.advanced_settings = advanced_settings or AdvancedSettings()
        self._last_translations = collections.deque(maxlen=15)
        self._setup_translator()

    def _setup_translator(self):
        """Robust translator initialization."""
        try:
            if TRANSLATOR_AVAILABLE:
                self.translator = GoogleTranslator(source='auto', target=self.target_lang)
            else:
                self.translator = None
        except Exception as e:
            logging.error(f"Translator setup failed: {e}")
            self.translator = None

    def set_target_language(self, target_lang: str):
        """
        Set target language with cache clearing.
        """
        if target_lang != self.target_lang:
            self.target_lang = target_lang
            with self._lock:
                self._cache.clear()
                self._last_translations.clear()
            self._setup_translator()

    def _clean_common_errors(self, text: str) -> str:
        """
        Clean common translation errors.
        """
        if "bass communi" in text.lower():
            return "best community"
        return text

    @excellence_execution(timeout=8.0)
    def translate_text(self, text: str, source_lang: str = "auto") -> Optional[ExcellenceTranslationResult]:
        """
        Optimierte Übersetzung ohne Terminal-Ausgaben
        """
        # FRÜHE ABBRUCH-BEDINGUNG: Wenn Source und Target gleich sind
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

    def _preprocess_text(self, text: str) -> str:
        """
        Umfassende Text-Vorverarbeitung für bessere Übersetzungen
        """
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
        
        #words = clean_text.split()
        #if len(words) < 3:
        #    clean_text = f"Sentence: {clean_text}"
    
        return clean_text.strip()

    def _is_valid_translation(self, original: str, translated: str) -> bool:
        """
        Erweiterte Validierung der Übersetzungsqualität
        """
        if not translated or not translated.strip():
            return False
    
        orig_len = len(original.strip())
        trans_len = len(translated.strip())

        if trans_len < orig_len * 0.3:
            return False
    
        if original.strip().lower() == translated.strip().lower():
            return False
    
        if not any(char.isalpha() for char in translated):
            return False
    
        linguistic_score = self._calculate_linguistic_score(translated)
        if linguistic_score < 0.3:
            return False
    
        if self.target_lang == 'de':
            german_chars = set('äöüßÄÖÜ')
            if any(char in german_chars for char in translated):
                return True
        
        elif self.target_lang == 'en':
            if translated[0].isupper() and translated.endswith(('.', '!', '?')):
                return True

        return True

    def _postprocess_translation(self, translated: str, original: str) -> str:
        """
        Nachbearbeitung für natürlich klingende Übersetzungen
        """
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

    def _calculate_linguistic_score(self, text: str) -> float:
        """
        Calculate linguistic quality score for translation.
        """
        if not text:
            return 0.0
            
        score = 0.0
        
        # Basic structure checks
        if text[0].isupper():
            score += 0.2
            
        if text.endswith(('.', '!', '?')):
            score += 0.2
            
        # Word count check
        words = text.split()
        if len(words) >= 2:
            score += 0.2
            
        # Character diversity
        unique_chars = len(set(text))
        if unique_chars / len(text) > 0.5:
            score += 0.2
            
        # Sentence length appropriateness
        if 10 <= len(text) <= 200:
            score += 0.2
            
        return min(1.0, score)

# === EXCELLENCE TRANSCRIPTION ENGINE ===
class ExcellenceTranscriptionEngine:
    """
    Speech-to-Text engine with Whisper integration and GPU/CPU fallback.
    """

    def __init__(self, advanced_settings: AdvancedSettings = None):
        """
        Initialize transcription engine.
        """
        self.model = None
        self.model_size = None
        self._lock = threading.RLock()
        self._model_loading = False
        self.advanced_settings = advanced_settings or AdvancedSettings()
        self._cache = ExcellenceCache(maxsize=self.advanced_settings.max_cache_size)
        self.device, self.compute_type = self._detect_optimal_device()
        self._performance_monitor = ExcellencePerformanceMonitor()
        self._last_transcription_text = ""

    def _detect_optimal_device(self):
        """VOLLSTÄNDIGE Device-Erkennung - SILENT"""
        try:
            device = "cpu"
            compute_type = "int8"

            # 1. NVIDIA/CUDA GPU
            if (self.advanced_settings.gpu_acceleration and 
                TORCH_AVAILABLE and torch.cuda.is_available()):
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    device = "cuda"
                
                    if vram >= 6:
                        compute_type = "float16"
                    else:
                        compute_type = "int8"
                    
                except Exception:
                    device = "cpu"

            # 2. Intel GPU (XPU)
            elif (self.advanced_settings.gpu_acceleration and
                  TORCH_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available()):
                try:
                    device = "xpu"
                    compute_type = "float16" if torch.xpu.get_device_properties(0).total_memory >= 6*1024**3 else "int8"
                except Exception:
                    device = "cpu"

            # 3. Apple Silicon (MPS)
            elif (self.advanced_settings.gpu_acceleration and
                  TORCH_AVAILABLE and hasattr(torch, 'backends') and 
                  hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                try:
                    device = "mps"
                    compute_type = "float16"
                except Exception:
                    device = "cpu"

            # 4. AMD GPU (ROCm)
            elif (self.advanced_settings.gpu_acceleration and
                  TORCH_AVAILABLE and hasattr(torch, 'cuda') and 
                  hasattr(torch.version, 'hip') and torch.cuda.is_available()):
                try:
                    device = "cuda"  # ROCm verwendet CUDA API
                    compute_type = "float16"
                except Exception:
                    device = "cpu"

            return device, compute_type

        except Exception:
            return "cpu", "int8"

    @excellence_execution(timeout=120.0)
    def load_model(self, model_size: str) -> bool:
        """Load Whisper model with excellence decorator."""
        if not WHISPER_AVAILABLE:
            return False

        with self._lock:
            if self.model_size == model_size and self.model is not None:
                return True

            if self._model_loading:
                return False

            self._model_loading = True

            try:
                logging.info(f"Loading model: {model_size} on {self.device} ({self.compute_type})")

                if self.model:
                    try:
                        del self.model
                    except:
                        pass
                    gc.collect()
                    if self.device == "cuda" and TORCH_AVAILABLE:
                        torch.cuda.empty_cache()

                self.model = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root="./models"
                )
                self.model_size = model_size
                logging.info(f"Model {model_size} loaded successfully on {self.device}")
                return True

            except Exception as e:
                logging.error(f"Model loading failed: {e}")

                if self.device != "cpu":
                    logging.info(f"🔄 {self.device} error - trying CPU fallback...")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    return self.load_model(model_size)
                elif model_size != "base":
                    logging.info("Trying fallback to base model...")
                    return self.load_model("base")
                return False
            finally:
                self._model_loading = False

    def enhance_audio_for_transcription(self, audio_data: bytes) -> bytes:
        """STABILISIERTE Audio-Verbesserung für Whisper"""
        if not NUMPY_AVAILABLE or len(audio_data) < 1600:
            return audio_data

        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # RMS Analyse
            rms = np.sqrt(np.mean(audio_np**2))
            max_val = np.max(np.abs(audio_np))

            # INTELLIGENTE GAIN-ANPASSUNG
            target_rms = 0.1
        
            if rms < 0.01:
                gain = min(8.0, target_rms / max(rms, 0.0001))
                audio_np = audio_np * gain
                logging.info(f"🎚️ BOOSTED quiet audio: {rms:.4f} -> {rms*gain:.4f} ({gain:.1f}x)")
            elif rms > 0.3:
                gain = target_rms / rms
                audio_np = audio_np * gain
                logging.info(f"🎚️ Reduced loud audio: {gain:.2f}x")
            elif rms < 0.02:
                gain = min(3.0, target_rms / rms)
                audio_np = audio_np * gain
                logging.info(f"🎚️ Enhanced audio: {gain:.2f}x")

            # ROBUSTE FILTERUNG
            try:
                audio_np = audio_np - np.mean(audio_np)
                
            
                if len(audio_np) > 10:
                    alpha = 0.95
                    filtered = np.zeros_like(audio_np)
                    filtered[0] = audio_np[0]
                    for i in range(1, len(audio_np)):
                        filtered[i] = alpha * (filtered[i-1] + audio_np[i] - audio_np[i-1])
                    audio_np = filtered
            
            except Exception as filter_error:
                audio_np = audio_np - np.mean(audio_np)

            # SANFTES CLIPPING
            audio_np = np.tanh(audio_np * 1.5) * 0.9

            # FINALE NORMALISIERUNG
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val * 0.92

            enhanced_audio = (audio_np * 32767).astype(np.int16).tobytes()
    
            return enhanced_audio

        except Exception as e:
            logging.error(f"❌ Audio enhancement critical error: {e}")
            return audio_data

    def _validate_transcription_segment(self, text: str, confidence: float, segment) -> bool:
        """SEHR lockere Validierung für Testzwecke."""
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
        """Erkennung von unsinnigem Text"""
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
        """Debugging der Confidence-Berechnung"""
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
        """Transkription komplett ohne Terminal-Ausgaben"""
        if not self.model or not audio_data:
            return None if not include_timestamps else []

        try:
            # Audio-Vorverarbeitung
            processed_audio = self.enhance_audio_for_transcription(audio_data)
            audio_np = np.frombuffer(processed_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Whisper Transkription
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

            # Ergebnisse zusammenstellen
            if include_timestamps:
                results = [
                    ExcellenceTranscriptionResult(
                        text=seg.text.strip(),
                        confidence=getattr(seg, 'confidence', 0.1),
                        language=getattr(info, 'language', 'unknown'),
                        start=getattr(seg, 'start', 0.0),
                        end=getattr(seg, 'end', 0.0)
                    ) for seg in valid_segments
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
                    emergency_result = self.emergency_fallback_transcription(audio_data)
                    return emergency_result

        except Exception:
            return None if not include_timestamps else []

    def emergency_fallback_transcription(self, audio_data: bytes) -> Optional[ExcellenceTranscriptionResult]:
        """VERBESSERTE Emergency-Fallback mit erweiterten Parametern."""
        try:
            if not self.model or not audio_data:
                return None

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Erweiterte Parameter für bessere Erkennung
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,  # Erhöht für bessere Qualität
                best_of=5,    # Erhöht für bessere Qualität
                temperature=0.0,
                vad_filter=False,  # VAD deaktivieren für Fallback
                no_speech_threshold=0.6,  # Weniger restriktiv
                log_prob_threshold=-1.0,  # Weniger restriktiv
                compression_ratio_threshold=2.4,  # Weniger restriktiv
                condition_on_previous_text=False,  # Deaktivieren für bessere Einzelerkennung
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
                
                    # Weniger strikte Validierung im Fallback
                    is_valid = (
                        len(text) >= 2 and
                        not text.isspace() and
                        any(c.isalnum() for c in text) and
                        confidence >= 0.1  # Reduziert von 0.15
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
            return None

    def clear_cache(self):
        """Clear transcription cache and free memory."""
        with self._lock:
            if hasattr(self, '_cache'):
                self._cache.clear()
            self._last_transcription_text = ""
            gc.collect()
            if self.device == "cuda" and TORCH_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass

    def reload_model(self, model_size: str) -> bool:
        """Thread-safe model reloading."""
        return self.load_model(model_size)

    def get_current_model(self) -> str:
        """Get current model name."""
        return self.model_size if self.model_size else "None"

    def is_model_loading(self) -> bool:
        """Check if model is currently loading."""
        return self._model_loading

    def dispose(self):
        """Explicit cleanup method."""
        self.clear_cache()
        if self.model:
            try:
                del self.model
            except:
                pass
        gc.collect()

# === STREAM INFO EXTRACTOR ===
@dataclass
class StreamInfo:
    """
    Contains metadata about the current stream.
    """
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
        """Initialize stream info extractor."""
        self.current_info = StreamInfo(
            title="Unknown Stream",
            uploader="Unknown",
            duration="Live",
            view_count=0,
            platform="Unknown"
        )
        self._lock = threading.RLock()

    def extract_stream_info(self, url: str) -> StreamInfo:
        """
        Extract stream information from URL.
        """
        if url.startswith('file://'):
            file_path = url[7:]
            return StreamInfo(
                title=os.path.basename(file_path),
                uploader="Local File",
                duration="File",
                view_count=0,
                platform="local"
            )

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

                return self.current_info

        except Exception as e:
            logging.warning(f"Stream info extraction failed: {e}")

        return StreamInfo(
            title="Stream title not available",
            uploader="Unknown",
            duration="Live",
            view_count=0,
            platform="unknown"
        )

class StreamManager:
    """
    Manages stream detection and audio extraction.
    """

    def detect_platform(self, url: str) -> Tuple[str, str]:
        """
        Detect platform from URL.
        """
        url_lower = url.lower().strip()

        if url_lower.startswith('file://'):
            return 'local', 'Local File'
        elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            return 'youtube', 'YouTube'
        elif 'twitch.tv' in url_lower:
            return 'twitch', 'Twitch'
        elif 'tiktok.com' in url_lower:
            return 'tiktok', 'TikTok'
        elif 'facebook.com' in url_lower or 'fb.watch' in url_lower:
            return 'facebook', 'Facebook'
        elif '.m3u8' in url_lower:
            return 'hls', 'HLS Stream'

        return 'unknown', 'Unknown Source'

    def extract_audio_url(self, url: str) -> Optional[str]:
        """
        Extract audio URL from stream URL.
        """
        if url.startswith('file://'):
            return url[7:]

        try:
            cmd = ['yt-dlp', '-g', '-f', 'bestaudio[ext=m4a]/bestaudio/best', '--no-warnings', '--no-check-certificate', url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=False)

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except Exception as e:
            logging.error(f"URL Extraction failed: {e}")

        return None

# === LANGUAGE DETECTOR ===
class LanguageDetector:
    """
    Detects language of video/audio files.
    """

    def __init__(self, transcription_engine):
        """
        Initialize language detector.
        """
        self.transcription_engine = transcription_engine

    def detect_video_language(self, video_path: str) -> Dict[str, Any]:
        """
        Fully asynchronous language detection for video files.
        """
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

        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}

    def _extract_audio_sample(self, video_path: str, duration: int = 30) -> Optional[bytes]:
        """
        Extract audio sample from video.
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-t', str(duration),
                '-f', 's16le',
                '-ar', str(ExcellenceConfig.SAMPLE_RATE),
                '-ac', '1',
                '-loglevel', 'quiet',
                '-'
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=30, shell=False)
            if result.returncode == 0 and result.stdout:
                return result.stdout

        except Exception as e:
            logging.error(f"Audio extraction failed: {e}")

        return None

# === PROGRESS DIALOG ===
class ProgressDialog:
    """
    Progress dialog with robust cancel button and non-blocking updates.
    """

    def __init__(self, parent, title="Processing..."):
        """
        Initialize progress dialog.
        """
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
            relief='flat', padx=15
        )
        self.cancel_button.pack()

        self.is_cancelled = False
        self.progress.start()

        self._update_interval = 100
        self._is_running = True
        self._schedule_updates()

    def _schedule_updates(self):
        """Schedule non-blocking GUI updates."""
        if (self._is_running and
                hasattr(self, 'dialog') and
                self.dialog.winfo_exists()):

            try:
                self.dialog.update_idletasks()
                self.dialog.after(self._update_interval, self._schedule_updates)
            except tk.TclError:
                self._is_running = False

    def cancel(self):
        """Cancel with immediate feedback."""
        self.is_cancelled = True
        self._is_running = False

        if hasattr(self, 'message_label') and self.message_label.winfo_exists():
            self.message_label.config(text="Cancelling...", fg=ModernColors.ERROR)

        if hasattr(self, 'cancel_button') and self.cancel_button.winfo_exists():
            self.cancel_button.config(text="Cancelling...", state='disabled')

        self.close()

    def update_message(self, message: str):
        """
        Safely update progress message.
        """
        if (self._is_running and
                hasattr(self, 'message_label') and
                self.message_label.winfo_exists()):

            try:
                self.message_label.config(text=message)
            except tk.TclError:
                self._is_running = False

    def close(self):
        """Close progress dialog and clean up resources."""
        self._is_running = False

        try:
            if hasattr(self, 'progress'):
                self.progress.stop()
        except:
            pass

        try:
            if hasattr(self, 'dialog') and self.dialog.winfo_exists():
                self.dialog.destroy()
           
        except:
            pass

# === EXCELLENCE AUDIO PROCESSOR ===
class ExcellenceAudioProcessor:
    """
    Ausgeglichene Audio-Processing Pipeline - robust aber nicht überkompliziert.
    """

    def __init__(self, controller_ref, ffmpeg_manager, config):
        self.controller_ref = controller_ref
        self.ffmpeg_manager = ffmpeg_manager
        self.config = config
        
        self.sample_rate = ExcellenceConfig.SAMPLE_RATE
        self.chunk_duration = ExcellenceConfig.CHUNK_DURATION
        self.chunk_size = ExcellenceConfig.CHUNK_SIZE_BYTES

        self.transcription_engine = None
        self.translation_engine = None
        self.plugin_manager = None

        self._stop_event = threading.Event()
        self._processing = False
        self._current_stream_id = None
        self._last_successful_read_time = time.time()
        self._consecutive_empty_chunks = 0

        self._translation_active = True
        self._last_transcription_text = ""

    def set_engines(self, transcription_engine, translation_engine, plugin_manager=None):
        """Set processing engines."""
        self.transcription_engine = transcription_engine
        self.translation_engine = translation_engine
        self.plugin_manager = plugin_manager

    def enhance_audio_quality(self, audio_data: bytes) -> bytes:
        """Vereinfachte Audio-Verbesserung - nur essentielle Schritte."""
        if not NUMPY_AVAILABLE or len(audio_data) < 1600:
            return audio_data

        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
            rms = np.sqrt(np.mean(audio_np**2))
        
            if rms < 0.005:
                gain = min(2.0, 0.01 / rms)
                audio_np = audio_np * gain
                logging.info(f"🎚️ Leises Audio verstärkt: RMS {rms:.4f} -> {rms*gain:.4f}")
        
            max_val = np.max(np.abs(audio_np))
            if max_val > 0.95:
                audio_np = audio_np * 0.95 / max_val
        
            enhanced_audio = (audio_np * 32767).astype(np.int16).tobytes()
            return enhanced_audio
        
        except Exception as e:
            logging.warning(f"⚠️ Audio enhancement error: {e}")
            return audio_data

    def _is_duplicate_transcription(self, current_text: str) -> bool:
        """Vermeidet Wiederholungen - etwas entspannter."""
        if not current_text or not current_text.strip():
            return True
            
        current_text = current_text.strip()
        
        if current_text == self._last_transcription_text:
            return True
            
        if len(current_text) < 2:
            return True
            
        if current_text.isspace():
            return True
            
        return False

    def _check_stream_health(self, process_id: str) -> bool:
        """Einfacher Health-Check."""
        try:
            if not self.ffmpeg_manager.is_active(process_id):
                return False
                
            test_data = self.ffmpeg_manager.read_audio_data(process_id, 1024)
            return test_data is not None and len(test_data) > 0
            
        except Exception:
            return False

    def emergency_diagnosis(self, url: str) -> bool:
        """Einfache Diagnose für GUI-Kompatibilität."""
        logging.info("🔍 Running emergency diagnosis...")
        
        try:
            stream_manager = StreamManager()
            audio_url = stream_manager.extract_audio_url(url)
            
            if not audio_url:
                return False
                
            success = self._test_stream_connection(audio_url)
            return success
                
        except Exception as e:
            logging.error(f"❌ Diagnosis error: {e}")
            return False

    def emergency_start_test(self, url: str) -> bool:
        """Einfacher Stream-Test - Alias für Kompatibilität."""
        return self.emergency_diagnosis(url)

    def emergency_transcription_test(self) -> bool:
        """Einfacher Transkriptions-Test."""
        
        if not self.transcription_engine:
            return False
            
        try:
            duration = 1
            sample_rate = 16000
            samples = int(sample_rate * duration)
            test_audio = (np.random.normal(0, 0.01, samples) * 32767).astype(np.int16).tobytes()
            
            result = self.transcription_engine.transcribe_audio(test_audio)
            success = result is not None
            return success
            
        except Exception as e:
            return False

    def _test_stream_connection(self, audio_url: str) -> bool:
        """Einfacher Stream-Verbindungstest."""
        try:
            if audio_url.startswith('file://'):
                file_path = audio_url[7:]
                return os.path.exists(file_path) and os.path.getsize(file_path) > 0

            test_cmd = ['ffmpeg', '-i', audio_url, '-t', '0.1', '-f', 'null', '-', '-loglevel', 'error']
            result = subprocess.run(
                test_cmd, 
                capture_output=True, 
                text=True, 
                timeout=3, 
                shell=False
            )
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logging.error("❌ Connection timeout")
            return False
        except Exception as e:
            logging.error(f"❌ Connection test failed: {e}")
            return False

    def start_processing(self, url: str,
                        transcription_callback: Callable,
                        translation_callback: Callable, 
                        info_callback: Callable,
                        error_callback: Callable):
        """Robuste Processing-Loop mit besserem Logging."""
        self._processing = True
        self._stop_event.clear()

        def process_loop():
            try:
                stream_manager = StreamManager()
                audio_url = stream_manager.extract_audio_url(url)
                if not audio_url:
                    error_callback("❌ Could not extract audio stream")
                    return

                info_callback("🎵 Starting audio processing...")
    
                cmd = [
                    'ffmpeg',
                    '-i', audio_url,
                    '-f', 's16le', 
                    '-ar', '16000',
                    '-ac', '1',
                    '-loglevel', 'quiet',
                    '-nostdin',
                    '-'
                ]
    
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._current_stream_id = "direct_stream"
                info_callback("✅ FFmpeg started - beginning transcription...")
    
                chunk_count = 0
                empty_reads = 0
                
                while self._processing and not self._stop_event.is_set() and process.poll() is None:
                    try:
                        audio_data = process.stdout.read(self.chunk_size)
                    except Exception as e:
                        time.sleep(0.1)
                        continue
                
                    if not audio_data:
                        empty_reads += 1
                        if empty_reads > 100:
                            error_callback("❌ No audio data received")
                            break
                        time.sleep(0.1)
                        continue
                    
                    empty_reads = 0
                
                    if len(audio_data) == self.chunk_size:
                        chunk_count += 1
                        
                        enhanced_audio = self.transcription_engine.enhance_audio_for_transcription(audio_data)
                
                        if self.transcription_engine:
                            transcription = self.transcription_engine.transcribe_audio(enhanced_audio)
                    
                            if transcription and transcription.text and transcription.text.strip():
                                clean_text = transcription.text.strip()
                                
                                if not self._is_duplicate_transcription(clean_text):
                                    self._last_transcription_text = clean_text
                                    transcription_callback(transcription)
                                    source_lang = getattr(transcription, 'language', 'unknown')
                                    target_lang = getattr(self.translation_engine, 'target_lang', 'de')
                                    should_translate = (
                                        self.translation_engine and 
                                        self._translation_active and
                                        source_lang != 'unknown' and
                                        source_lang != target_lang
                                    )
                                
                                    if should_translate:
                                        translation = self.translation_engine.translate_text(
                                            clean_text, 
                                            transcription.language
                                        )
                                        if translation:
                                            translation_callback(translation)

                    if chunk_count % 20 == 0:
                        status_msg = f"📊 Processed {chunk_count} chunks"
                        info_callback(status_msg)
                
            except Exception as e:
                error_msg = f"❌ Processing error: {str(e)}"
                error_callback(error_msg)
                
            finally:
                if hasattr(self, '_current_stream_id'):
                    self._current_stream_id = None
                if process and process.poll() is None:
                    process.terminate()
                info_callback("🔄 Processing stopped")
    
        processing_thread = threading.Thread(target=process_loop, daemon=True)
        processing_thread.start()

    def stop_processing(self):
        """Sauberes Stoppen der Verarbeitung."""
        self._processing = False
        self._stop_event.set()
        
        if hasattr(self, '_current_stream_id') and self._current_stream_id:
            self.ffmpeg_manager.stop_stream(self._current_stream_id)
            self._current_stream_id = None
            
        logging.info("🛑 Processing stopped by user")

    def dispose(self):
        """Ressourcen freigeben."""
        self.stop_processing()

# === DARK CONTEXT MENUS ===
class DarkContextMenu:
    """Dark theme context menu for text widgets with modern styling."""

    def __init__(self, text_widget):
        """
        Initialize dark context menu for text widget.
        """
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
        """Show context menu at mouse position."""
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def copy_text(self):
        """Copy selected text to clipboard."""
        try:
            selected_text = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.text_widget.clipboard_clear()
            self.text_widget.clipboard_append(selected_text)
        except tk.TclError:
            pass

    def select_all(self):
        """Select all text in widget."""
        self.text_widget.tag_add(tk.SEL, "1.0", tk.END)
        self.text_widget.mark_set(tk.INSERT, "1.0")
        self.text_widget.see(tk.INSERT)

    def clear_text(self):
        """Clear all text in widget."""
        self.text_widget.delete("1.0", tk.END)

class DarkEntryContextMenu:
    """Dark theme context menu for entry widgets with modern styling."""

    def __init__(self, entry_widget):
        """
        Initialize dark context menu for entry widget.
        """
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
        """Show context menu at mouse position."""
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def cut_text(self):
        """Cut selected text to clipboard."""
        self.entry_widget.event_generate("<<Cut>>")

    def copy_text(self):
        """Copy selected text to clipboard."""
        self.entry_widget.event_generate("<<Copy>>")

    def paste_text(self):
        """Paste text from clipboard."""
        self.entry_widget.event_generate("<<Paste>>")

    def select_all(self):
        """Select all text in entry."""
        self.entry_widget.select_range(0, 'end')
        self.entry_widget.icursor('end')

    def delete_text(self):
        """Delete all text in entry."""
        self.entry_widget.delete(0, 'end')

# === EXPORT MANAGER ===
class ExportManager:
    """
    Supports SRT, VTT and other subtitle formats for export.
    """

    def __init__(self):
        """Initialize export manager with supported formats."""
        self.supported_formats = ['txt', 'srt', 'vtt', 'json', 'docx']

    def export_subtitles(self, transcript_data: List[ExcellenceTranscriptionResult],
                        translation_data: List[ExcellenceTranslationResult] = None,
                        format: str = 'srt',
                        filename: str = None):
        """
        Export subtitles in SRT or VTT format.
        """
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

        except Exception as e:
            raise ExcellenceError(f"Subtitle export failed: {e}")

    def generate_srt_content(self, transcript_data: List[ExcellenceTranscriptionResult],
                           translation_data: List[ExcellenceTranslationResult] = None) -> str:
        """
        Generate SRT content for subtitles.
        """
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
        """
        Generate WebVTT content for subtitles.
        """
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
        """
        Format timestamp for SRT: 00:00:01,500.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

    def _format_timestamp_vtt(self, seconds: float) -> str:
        """
        Format timestamp for VTT: 00:00:01.500.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{milliseconds:03d}"

    def export_json(self, transcript_data: List[ExcellenceTranscriptionResult],
                   translation_data: List[ExcellenceTranslationResult], filename: str):
        """
        Export transcriptions and translations as JSON with timestamps.
        """
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
        except Exception as e:
            raise ExcellenceError(f"JSON export failed: {e}")

    def export_docx(self, transcript_data: List[ExcellenceTranscriptionResult], filename: str):
        """
        Export as Word document (basic text format).
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("TRANSCRIPT EXPORT\n")
                f.write("================\n\n")

                for i, segment in enumerate(transcript_data, 1):
                    timestamp = datetime.fromtimestamp(segment.timestamp).strftime('%H:%M:%S')
                    f.write(f"[{timestamp}] {segment.text}\n\n")

            return True
        except Exception as e:
            raise ExcellenceError(f"DOCX export failed: {e}")

# === BATCH PROCESSOR ===
class BatchProcessor:
    """
    Processes multiple streams/files in batch mode.
    """

    def __init__(self):
        """Initialize batch processor."""
        self.jobs = []
        self.current_job = None
        self.is_processing = False

    def create_batch_job(self, urls: List[str], output_dir: str = "batch_output"):
        """
        Create a new batch processing job.
        """
        job_id = hashlib.md5(str(urls).encode()).hexdigest()[:8]
        job = {
            'id': job_id,
            'urls': urls,
            'output_dir': output_dir,
            'status': 'pending',
            'results': [],
            'start_time': None,
            'end_time': None
        }

        self.jobs.append(job)
        return job_id

# === APP SETTINGS ===
@dataclass
class AppSettings:
    """
    Manages application settings and persistence.
    """
    last_url: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    default_model: str = "medium"
    default_language: str = "de"
    layout_mode: str = "vertical"
    recent_urls: List[str] = None
    enable_plugins: bool = True
    export_format: str = "txt"

    def __post_init__(self):
        """Initialize recent URLs if not provided."""
        if self.recent_urls is None:
            self.recent_urls = []

    @classmethod
    def load_from_file(cls, filename="dragon_settings.json"):
        """
        Load settings from file.
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return cls(**data)
        except Exception as e:
            logging.warning(f"Settings load failed: {e}")
        return cls()

    def save_to_file(self, filename="dragon_settings.json"):
        """
        Save settings to file.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Settings save failed: {e}")

    def add_recent_url(self, url):
        """
        Add URL to recent URLs list.
        """
        if url in self.recent_urls:
            self.recent_urls.remove(url)
        self.recent_urls.insert(0, url)
        self.recent_urls = self.recent_urls[:10]
        self.save_to_file()

# === RESOURCE MANAGER ===
class ResourceManager:
    """
    Central resource management for cleanup and process termination.
    """

    def __init__(self):
        """Initialize resource manager."""
        self.processes = []
        self.threads = []
        self.temp_files = []
        self.cleanup_done = False
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()

    def register_process(self, process):
        """
        Register process for cleanup.
        """
        with self._lock:
            if process and process not in self.processes:
                self.processes.append(process)

    def register_thread(self, thread):
        """
        Register thread for cleanup.
        """
        with self._lock:
            if thread and thread not in self.threads and thread.is_alive():
                self.threads.append(thread)

    def register_temp_file(self, file_path):
        """
        Register temporary file for cleanup.
        """
        with self._lock:
            if file_path and file_path not in self.temp_files:
                self.temp_files.append(file_path)

    def cleanup(self):
        """Perform comprehensive resource cleanup with timeouts."""
        if self.cleanup_done:
            return

        self._shutdown_event.set()

        with self._lock:
            logging.info("Starting enhanced resource cleanup...")

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
                except Exception as e:
                    logging.warning(f"Error terminating process: {e}")
                finally:
                    if process in self.processes:
                        self.processes.remove(process)
                        
                if time.time() - start_time > cleanup_timeout:
                    logging.warning("Cleanup timeout reached - forcing remaining processes")
                    break

            for thread in self.threads[:]:
                try:
                    if thread and thread.is_alive():
                        thread.join(timeout=1.0)
                except Exception as e:
                    logging.warning(f"Error joining thread: {e}")
                finally:
                    if thread in self.threads:
                        self.threads.remove(thread)
                        
                if time.time() - start_time > cleanup_timeout:
                    logging.warning("Cleanup timeout reached - skipping remaining threads")
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
                                else:
                                    logging.warning(f"Could not delete temp file: {temp_file}")
                            except Exception as e:
                                logging.debug(f"Temp file deletion attempt {attempt + 1}: {e}")
                                if attempt < 1:
                                    time.sleep(0.1)
                except Exception as e:
                    logging.debug(f"Temp file cleanup error (non-critical): {e}")
                finally:
                    if temp_file in self.temp_files:
                        self.temp_files.remove(temp_file)

            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

            gc.collect()

            self.cleanup_done = True
            logging.info("Enhanced resource cleanup completed")

    def is_shutting_down(self):
        """
        Check if shutdown is in progress.
        """
        return self._shutdown_event.is_set()

# === WHISPER LAYOUT MANAGER ===
class WhisperLayoutManager:
    """
    Manages design, Tkinter elements, layout, colors, tooltips and all visual updates.
    """

    def __init__(self, gui_ref):
        """
        Initialize layout manager.
        """
        self.gui_ref = gui_ref
        self.root = gui_ref.root
        
    def setup_gui(self):
        """Setup GUI with dark theme and modern styling."""
        self.root.configure(bg=ModernColors.BG_PRIMARY)
        self.root.title("🐉 Dragon Whisperer")
        self.root.geometry("900x650")
        self.root.minsize(800, 550)
        self.setup_dark_styles()
        self.center_window()
        self.root.protocol("WM_DELETE_WINDOW", self.gui_ref.controller.safe_exit)
        self.create_layout()

    def setup_dark_styles(self):
        """Configure dark theme styles for Tkinter widgets."""
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
        """Center window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')

    def create_layout(self):
        """Create main application layout with all UI components."""
        # Header
        header_frame = tk.Frame(self.root, bg=ModernColors.BG_PRIMARY, height=35)
        header_frame.pack(fill='x', padx=12, pady=8)
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame,
                              text="🐉 Dragon Whisperer - Livestream Transcription && Translation",
                              font=ModernFonts.TITLE,
                              bg=ModernColors.BG_PRIMARY,
                              fg=ModernColors.DRAGON_GREEN)
        title_label.pack(side='left')

        self.gui_ref.status_label = tk.Label(header_frame,
                                   text="✅ ACTIVE",
                                   font=ModernFonts.PRIMARY,
                                   bg=ModernColors.BG_PRIMARY,
                                   fg=ModernColors.TEXT_SECONDARY)
        self.gui_ref.status_label.pack(side='right')
        self.create_stream_info_display()

        # Input Section
        input_frame = tk.Frame(self.root, bg=ModernColors.BG_PRIMARY)
        input_frame.pack(fill='x', padx=12, pady=3)

        # URL Row
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
        self.create_text_areas()
        self.setup_status_bar()
        self.gui_ref.url_entry.bind('<KeyRelease>', self.gui_ref.on_url_change)
        self.gui_ref.url_entry.bind('<FocusOut>', self.gui_ref.on_url_change)

    def create_stream_info_display(self):
        """Create stream information display panel."""
        self.gui_ref.stream_info_frame = tk.Frame(self.root, bg=ModernColors.BG_SECONDARY, height=50)
        self.gui_ref.stream_info_frame.pack(fill='x', padx=12, pady=3)
        self.gui_ref.stream_info_frame.pack_propagate(False)
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
        """Create compact control panel with action buttons and settings."""
        control_frame = tk.Frame(parent, bg=ModernColors.BG_PRIMARY)
        control_frame.pack(fill='x', pady=8)

        # Left side - action buttons
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

        # Central controls
        center_controls = tk.Frame(control_frame, bg=ModernColors.BG_PRIMARY)
        center_controls.pack(side='left', padx=15)

        # Model Selection
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

        # Right side - main actions
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
        """Create text areas for transcription and translation output."""
        if hasattr(self.gui_ref, 'text_container'):
            self.gui_ref.text_container.destroy()

        self.gui_ref.text_container = tk.Frame(self.root, bg=ModernColors.BG_PRIMARY)
        self.gui_ref.text_container.pack(fill='both', expand=True, padx=12, pady=8)

        if self.gui_ref.layout_mode == "horizontal":
            self.create_horizontal_layout()
        else:
            self.create_vertical_layout()

    def create_vertical_layout(self):
        """Create vertical layout for text areas."""
        main_frame = tk.LabelFrame(self.gui_ref.text_container, text="Live Transkription & Übersetzung",
                                 bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                 font=ModernFonts.SUBTITLE, padx=8, pady=8)
        main_frame.pack(fill='both', expand=True)

        # Transkription
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

        # Übersetzung
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
        """Create horizontal layout for text areas with paned window."""
        main_frame = tk.LabelFrame(self.gui_ref.text_container, text="Live Transkription & Übersetzung",
                                 bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                 font=ModernFonts.SUBTITLE, padx=8, pady=8)
        main_frame.pack(fill='both', expand=True)

        self.gui_ref.paned_window = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL,
                                         bg=ModernColors.BG_SECONDARY,
                                         sashrelief='raised', sashwidth=4)
        self.gui_ref.paned_window.pack(fill='both', expand=True)

        left_frame = tk.Frame(self.gui_ref.paned_window, bg=ModernColors.BG_TERTIARY)
        self.gui_ref.paned_window.add(left_frame, stretch="always")

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
        self.gui_ref.paned_window.add(right_frame, stretch="always")

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
        self.gui_ref.paned_window.paneconfig(left_frame, minsize=250)
        self.gui_ref.paned_window.paneconfig(right_frame, minsize=250)

    def create_text_widget(self, parent, height=None):
        """
        Create text widget with dark theme and scrolling.
        """
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
            """Text cleanup with Excellence Memory Manager."""
            try:
                lines = int(text_widget.index('end-1c').split('.')[0])
                if lines > 400:
                    component = 'transcript' if text_widget == self.gui_ref.transcript_text else 'translation'
                    self.gui_ref.memory_manager.clear_component(component)

                    keep_lines = 250
                    delete_to = f'{lines-keep_lines}.0'
                    text_widget.delete('1.0', delete_to)
                    gc.collect()
            except Exception as e:
                logging.debug(f"Text cleanup (non-critical): {e}")

        text_widget.bind('<KeyRelease>', safe_text_cleanup)
        return text_widget

    def setup_status_bar(self):
        """Create status bar with action buttons and system info."""
        status_frame = tk.Frame(self.root, bg=ModernColors.BG_SECONDARY, height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)

        left_actions = tk.Frame(status_frame, bg=ModernColors.BG_SECONDARY)
        left_actions.pack(side='left', padx=8, pady=4)

        status_actions = [
            ("🗑️", self.gui_ref.clear_all, "Alles löschen"),
            ("💾", self.gui_ref.save_transcript, "Transkript speichern"),
            ("📝", self.gui_ref.export_subtitles, "Untertitel exportieren"),
            ("📊", self.gui_ref.show_performance_stats, "Statistiken anzeigen"),
            ("⚙️", self.gui_ref.show_advanced_settings, "Einstellungen"),
            ("🚨", self.gui_ref.show_enterprise_dashboard, "Enterprise Dashboard"),
        ]

        for icon, command, tooltip in status_actions:
            btn = tk.Button(left_actions, text=icon, command=command,
                          bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
                          relief='flat', bd=0, font=("Segoe UI", 9),
                          cursor='hand2')
            btn.pack(side='left', padx=1)

        right_section = tk.Frame(status_frame, bg=ModernColors.BG_SECONDARY)
        right_section.pack(side='right', padx=8, pady=4)

        self.gui_ref.system_info_label = tk.Label(
            right_section,
            text="💻 CPU: --% | 🧠 RAM: --MB | ⚡ v1.02",
            font=("Segoe UI", 8),
            bg=ModernColors.BG_SECONDARY,
            fg=ModernColors.TEXT_SECONDARY
        )
        self.gui_ref.system_info_label.pack(side='left', padx=(0, 10))

        self.gui_ref.exit_button = tk.Button(
            right_section, text="❌", command=self.gui_ref.controller.safe_exit,
            bg=ModernColors.ERROR, fg=ModernColors.TEXT_PRIMARY,
            relief='flat', bd=0, font=("Segoe UI", 9),
            cursor='hand2'
        )
        self.gui_ref.exit_button.pack(side='right')

    def add_text(self, component: str, text: str):
        """
        Add text to appropriate component with auto-scroll support.
        """
        if component == 'transcript':
            self.gui_ref.transcript_text.insert('end', text)
            if self.gui_ref.transcript_scroll_var.get():
                self.gui_ref.transcript_text.see('end')
        elif component == 'translation':
            self.gui_ref.translation_text.insert('end', text)
            if self.gui_ref.translation_scroll_var.get():
                self.gui_ref.translation_text.see('end')

    def _update_ui_state(self, state_info: Dict[str, Any]):
        """
        Update UI state based on state information.
        """
        try:
            if 'status' in state_info:
                status_text = state_info['status']
                if hasattr(self.gui_ref, 'status_label') and self.gui_ref.status_label.winfo_exists():
                    self.gui_ref.status_label.config(text=status_text)

            if 'stream_info' in state_info:
                stream_info = state_info['stream_info']
                if hasattr(self.gui_ref, 'stream_title_label') and self.gui_ref.stream_title_label.winfo_exists():
                    title = stream_info.title[:100] + "..." if len(stream_info.title) > 100 else stream_info.title
                    self.gui_ref.stream_title_label.config(text=f"📡 {title}")

                if hasattr(self.gui_ref, 'stream_details_label') and self.gui_ref.stream_details_label.winfo_exists():
                    details = f"👤 {stream_info.uploader}"
                    if stream_info.duration and stream_info.duration != 'Live':
                        details += f" | ⏱️ {stream_info.duration}"
                    self.gui_ref.stream_details_label.config(text=details)

            if 'processing_state' in state_info:
                processing = state_info['processing_state']
                if hasattr(self.gui_ref, 'start_button') and self.gui_ref.start_button.winfo_exists():
                    self.gui_ref.start_button.config(state='disabled' if processing else 'normal')
                if hasattr(self.gui_ref, 'stop_button') and self.gui_ref.stop_button.winfo_exists():
                    self.gui_ref.stop_button.config(state='normal' if processing else 'disabled')

        except Exception as e:
            logging.debug(f"UI state update error: {e}")

    def process_gui_updates(self):
        """Process GUI update queue with exception wrapper for ALL callbacks."""
        try:
            processed = 0
            max_updates = 15

            while processed < max_updates and not self.gui_ref.gui_queue.empty():
                try:
                    item = self.gui_ref.gui_queue.get_nowait()
                    msg_type, callback = item

                    if callable(callback):
                        try:
                            callback()
                        except tk.TclError as e:
                            logging.debug(f"Tkinter GUI callback safe error: {e}")
                        except RuntimeError as e:
                            logging.debug(f"Runtime GUI callback safe error: {e}")
                        except Exception as e:
                            logging.error(f"GUI callback error: {e}")

                    self.gui_ref.gui_queue.task_done()
                    processed += 1

                except queue.Empty:
                    break

            if self.gui_ref.gui_queue.qsize() > 80:
                excess_items = self.gui_ref.gui_queue.qsize() - 50
                for _ in range(min(excess_items, 30)):
                    try:
                        self.gui_ref.gui_queue.get_nowait()
                        self.gui_ref.gui_queue.task_done()
                    except queue.Empty:
                        break
                logging.debug(f"GUI queue overflow protection: cleared {excess_items} items")

        except Exception as e:
            logging.error(f"GUI update processing critical error: {e}")

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(150, self.process_gui_updates)

    def process_batch_text_updates(self):
        """Batch processing for text widget updates with rate limiting."""
        try:
            while not self.gui_ref._text_update_queue.empty():
                try:
                    update_type, text_data = self.gui_ref._text_update_queue.get_nowait()
                    
                    if update_type == 'transcript' and hasattr(self.gui_ref, 'transcript_text'):
                        self._apply_batch_text_update(self.gui_ref.transcript_text, text_data, 
                                                    self.gui_ref.transcript_scroll_var.get())
                    elif update_type == 'translation' and hasattr(self.gui_ref, 'translation_text'):
                        self._apply_batch_text_update(self.gui_ref.translation_text, text_data,
                                                    self.gui_ref.translation_scroll_var.get())
                    
                    self.gui_ref._text_update_queue.task_done()
                except queue.Empty:
                    break

        except Exception as e:
            logging.debug(f"Batch text update processing error: {e}")

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(self.gui_ref._batch_update_interval, self.process_batch_text_updates)

    def _apply_batch_text_update(self, text_widget, text_data, auto_scroll):
        """
        Apply batch update to text widget with memory management.
        """
        try:
            if not text_widget.winfo_exists():
                return

            text_widget.insert('end', text_data)

            if auto_scroll:
                text_widget.see('end')

            current_lines = int(text_widget.index('end-1c').split('.')[0])
            if current_lines > 400:
                keep_lines = 250
                delete_to = f'{current_lines-keep_lines}.0'
                text_widget.delete('1.0', delete_to)

        except Exception as e:
            logging.debug(f"Batch text apply error: {e}")

# === WHISPER CONTROLLER ===
class WhisperController:
    """
    Manages complete program control, all threads, state management, duplicate checking and exit logic.
    """

    def __init__(self, gui_ref, ui_update_fn: Callable = None, status_update_fn: Callable = None):
        """
        Initialize controller with GUI reference and callbacks.
        """
        self.gui_ref = gui_ref
        self.is_processing = False
        self.ui_update_fn = ui_update_fn or (lambda component, text: None)
        self.status_update_fn = status_update_fn or (lambda state_info: None)
        
        self.executor = None
        self.processing_thread = None
        self._cleanup_lock = threading.RLock()
        
    def _cleanup_resources(self):
        """
        Zentralisierte Cleanup-Methode für alle Threads und Ressourcen.
        """
        with self._cleanup_lock:
            logging.info("🔄 Starting centralized resource cleanup...")

            self.is_processing = False

            if self.executor:
                try:
                    self.executor.shutdown(wait=True)
                    logging.info("✅ ThreadPoolExecutor shutdown initiated")
                except Exception as e:
                    logging.warning(f"⚠️ ThreadPoolExecutor shutdown warning: {e}")
                finally:
                    self.executor = None

            if self.processing_thread:
                try:
                    if hasattr(self.processing_thread, 'cancel'):
                        self.processing_thread.cancel()
                except Exception as e:
                    logging.debug(f"Thread cancellation: {e}")

            if hasattr(self.gui_ref, 'audio_processor'):
                self.gui_ref.audio_processor._processing = False
                self.gui_ref.audio_processor._translation_active = False
                self.gui_ref.audio_processor._stop_event.set()

            if hasattr(self.gui_ref, 'memory_manager'):
                self.gui_ref.memory_manager.clear_component('transcript')
                self.gui_ref.memory_manager.clear_component('translation')

            gc.collect()

            logging.info("✅ Centralized resource cleanup completed")

    def _start_processing(self):
        """
        Robuster Stream-Start mit garantierter Cleanup-Ausführung.
        """
        self._cleanup_resources()
        
        url = self.gui_ref.url_entry.get().strip()
        if not url:
            DarkMessageBox.showerror("Error", "Please enter stream URL or file!", self.gui_ref.root)
            return

        self.gui_ref.settings.last_url = url

        try:
            cleaned_url = self.gui_ref.clean_and_validate_url(url)
            self.gui_ref.url_entry.delete(0, 'end')
            self.gui_ref.url_entry.insert(0, cleaned_url)
            self.gui_ref.settings.add_recent_url(cleaned_url)

        except ValueError as e:
            DarkMessageBox.showerror("Invalid URL", f"Invalid URL:\n\n{str(e)}", self.gui_ref.root)
            return

        
        logging.info("🔍 DEBUG: Running pre-start diagnosis...")
        diagnosis_result = self.gui_ref.audio_processor.emergency_diagnosis(url)
        if not diagnosis_result:
            self.gui_ref.handle_error("❌ Pre-start diagnosis failed - check terminal for details")
            return
        logging.info("✅ DEBUG: Pre-start diagnosis passed")
        

        if cleaned_url.startswith('file://'):
            self.gui_ref.progress_dialog = ProgressDialog(self.gui_ref.root, "Analyzing video...")


        def show_immediate_info():
            platform_type, platform_name = self.gui_ref.stream_manager.detect_platform(url)
            immediate_info = StreamInfo(
                title="Preparing connection...",
                uploader=platform_name,
                duration="Initializing...",
                view_count=0,
                platform=platform_type
            )
            self.status_update_fn({'stream_info': immediate_info})

        show_immediate_info()

        def extract_stream_info():
            try:
                self.status_update_fn({'status': "🔍 Connecting and analyzing stream..."})

                stream_info = StreamInfoExtractor().extract_stream_info(url)

                self.status_update_fn({'status': "🔍 Checking stream availability..."})

                stream_manager = StreamManager()
                audio_url = stream_manager.extract_audio_url(url)

                if not audio_url:
                    self.gui_ref.handle_error("❌ Stream not reachable - No audio URL found")
                    return

                self.status_update_fn({'status': "🎵 Testing audio stream..."})
                if not self.gui_ref.audio_processor._test_stream_connection(audio_url):
                    self.gui_ref.handle_error("❌ Stream not reachable - Connection test failed")
                    return

                self.status_update_fn({'stream_info': stream_info, 'status': "✅ Stream connected - Loading AI model..."})

                selected_name = self.gui_ref.lang_var.get()
                lang_code = "de"
                for name, code in SORTED_LANGUAGES:
                    if name == selected_name:
                        lang_code = code
                        break

                self.gui_ref.translation_engine.set_target_language(lang_code)
                lang_display = LANGUAGE_SHORT_CODES.get(lang_code, lang_code)

                def update_gui():
                    if hasattr(self.gui_ref, 'translation_header'):
                        self.gui_ref.translation_header.config(text=f"🌐 Translation ({lang_display})")

                    if hasattr(self.gui_ref, 'progress_dialog') and self.gui_ref.progress_dialog:
                        self.gui_ref.progress_dialog.close()

                self.gui_ref.gui_queue.put(('update_ui', update_gui))

                if not self.gui_ref.transcription_engine.load_model(self.gui_ref.model_var.get()):
                    self.gui_ref.handle_error("AI model could not be loaded!")
                    return

                self.is_processing = True
                self.status_update_fn({'processing_state': True, 'status': f"🚀 Transcription started (Language: {selected_name})"})

                self.gui_ref.audio_processor._translation_active = self.gui_ref.translate_toggle.get()

                self.executor = ThreadPoolExecutor(max_workers=2)
                self.processing_thread = self.executor.submit(
                    self.gui_ref.audio_processor.start_processing,
                    url=url,
                    transcription_callback=self.gui_ref.handle_transcription,
                    translation_callback=self.gui_ref.handle_translation,
                    info_callback=self.gui_ref.handle_info,
                    error_callback=self.gui_ref.handle_error
                )

                if self.processing_thread:
                    self.gui_ref.resource_manager.register_thread(threading.Thread())

            except Exception as e:
                self.gui_ref.handle_error(f"Connection error: {e}")
                if hasattr(self.gui_ref, 'progress_dialog') and self.gui_ref.progress_dialog:
                    self.gui_ref.progress_dialog.close()

        info_thread = threading.Thread(target=extract_stream_info, daemon=True)
        self.gui_ref.resource_manager.register_thread(info_thread)
        info_thread.start()

    def start_processing(self):
        """
        Start audio processing with guaranteed cleanup before start.
        """
        self._start_processing()

    def _stop_processing(self):
        """
        Stoppt die Verarbeitung durch Aufruf der zentralisierten Cleanup-Methode.
        """
        self._cleanup_resources()
        
        self.status_update_fn({
            'processing_state': False,
            'status': 'Stopped'
        })
        
        self.status_update_fn({'stream_info': StreamInfo(
            title="📡 No active stream",
            uploader="--",
            duration="--",
            view_count=0,
            platform="--"
        )})

        if hasattr(self.gui_ref, 'progress_dialog') and self.gui_ref.progress_dialog:
            self.gui_ref.progress_dialog.close()

        logging.info("🔄 Processing stopped via centralized cleanup")

    def stop_processing(self):
        """
        Stop processing using the refactored cleanup method.
        """
        self._stop_processing()

    def _is_duplicate_transcription(self, new_text: str) -> bool:
        """
        Robust duplicate transcription check using string history.
        """
        if not new_text or not new_text.strip():
            return True
            
        current_clean = new_text.strip()
        
        if not hasattr(self, '_last_transcription_text'):
            self._last_transcription_text = ""
            return False
            
        last_clean = self._last_transcription_text.strip()

        if current_clean == last_clean:
            return True
            
        if len(current_clean) < 10 and current_clean in last_clean:
            return True

        return False

    def safe_exit(self):
        """
        Dark Mode Exit V3 (Single-Threaded & Stable).
        """
        import tkinter as tk
        import os
        import time
        import threading
        
        self.gui_ref.exit_confirmed = False

        def show_dark_exit_dialog():
            dialog = tk.Toplevel(self.gui_ref.root)
            dialog.title("Programm beenden")
            dialog.configure(bg=ModernColors.BG_PRIMARY)
            dialog.resizable(False, False)
            dialog.transient(self.gui_ref.root)
            
            def confirm():
                self.gui_ref.exit_confirmed = True
                dialog.destroy()
            
            def cancel():
                self.gui_ref.exit_confirmed = False
                dialog.destroy()
            
            main_frame = tk.Frame(dialog, bg=ModernColors.BG_PRIMARY, padx=20, pady=20)
            main_frame.pack(fill='both', expand=True)

            msg = "Möchten Sie Dragon Whisperer wirklich schließen?"
            if self.is_processing:
                msg = "⚠️ Aufnahme läuft noch! Sind Sie sicher, dass Sie beenden wollen?"

            tk.Label(main_frame, text=msg, font=ModernFonts.PRIMARY,
                     bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY,
                     justify='left', wraplength=350).pack(pady=(0, 20))
            
            btn_frame = tk.Frame(main_frame, bg=ModernColors.BG_PRIMARY)
            btn_frame.pack(fill='x')

            tk.Button(btn_frame, text="Abbrechen", command=cancel,
                      bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
                      relief='flat', padx=15).pack(side='right')

            tk.Button(btn_frame, text="OK", command=confirm,
                      bg=ModernColors.SUCCESS, fg=ModernColors.TEXT_PRIMARY,
                      relief='flat', padx=15).pack(side='right', padx=(10, 0))

            dialog.update_idletasks()
            x = self.gui_ref.root.winfo_x() + (self.gui_ref.root.winfo_width() - dialog.winfo_width()) // 2
            y = self.gui_ref.root.winfo_y() + (self.gui_ref.root.winfo_height() - dialog.winfo_height()) // 2
            dialog.geometry(f"+{x}+{y}")
            
            dialog.focus_set()
            self.gui_ref.root.wait_window(dialog)

        show_dark_exit_dialog()
        
        should_exit = self.gui_ref.exit_confirmed

        if not should_exit:
            logging.info("🛑 SHUTDOWN: Aborted by user.")
            return

        logging.info("🛑 SHUTDOWN: Confirmed. Initiating process termination...")

        try:
            self.gui_ref.root.withdraw()
        except:
            pass

        self._cleanup_resources()

        def force_kill():
            time.sleep(0.5)
            os._exit(0)
        
        threading.Thread(target=force_kill, daemon=True).start()

        os._exit(0)

# === DRAGON WHISPERER GUI ===
class DragonWhispererGUI:
    """
    Main GUI application with Tkinter - REFACTORED.
    """

    def __init__(self):
        """Initialize main GUI application with all components."""
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI not available")

        self.root = tk.Tk()
        self.controller = WhisperController(gui_ref=self)
        self.layout = WhisperLayoutManager(gui_ref=self)
        self.settings = AppSettings.load_from_file()
        self.advanced_settings = AdvancedSettings.load_from_file()
        self.layout_mode = self.settings.layout_mode
        self.current_language = self.settings.default_language
        self.current_stream_info = None
        self.is_processing = False
        self.transcript_history = []
        self.translation_history = []
        self._last_transcription_text = ""
        self._last_translation_text = ""
        self.subtitle_mode = False
        self.current_video_language = None
        self.progress_dialog = None
        self.performance_monitor = ExcellencePerformanceMonitor()
        self.memory_manager = ExcellenceMemoryManager()
        self.gui_queue = queue.Queue(maxsize=100)
        self._text_update_queue = queue.Queue(maxsize=50)
        self._batch_update_interval = 100
        self._pending_text_updates = {'transcript': [], 'translation': []}
        self.stream_manager = StreamManager()
        self._translation_active = True
        self.ffmpeg_manager = ExcellenceFFmpegManager(self.advanced_settings)
        self.transcription_engine = ExcellenceTranscriptionEngine(self.advanced_settings)
        self.translation_engine = ExcellenceTranslationEngine(self.current_language, self.advanced_settings)
        self.audio_processor = ExcellenceAudioProcessor(
            controller_ref=self.controller,
            ffmpeg_manager=self.ffmpeg_manager,
            config=self.advanced_settings
        )
        self.resource_manager = ResourceManager()
        self.plugin_manager = PluginManager()
        self.export_manager = ExportManager()
        self.batch_processor = BatchProcessor()
        self.language_detector = LanguageDetector(self.transcription_engine)

        # Setup callbacks
        self.controller.ui_update_fn = self._actual_ui_update
        self.controller.status_update_fn = self._actual_status_update

        # Register cleanup
        ExcellenceSignalHandler.register_cleanup(self.excellence_cleanup)
        ExcellenceSignalHandler.register_cleanup(self.audio_processor.dispose)
        ExcellenceSignalHandler.register_cleanup(self.transcription_engine.dispose)
        ExcellenceSignalHandler.register_cleanup(self.memory_manager.dispose)
        ExcellenceSignalHandler.register_cleanup(self.ffmpeg_manager.dispose)
        ExcellenceSignalHandler.register_cleanup(self.resource_manager.cleanup)

        if self.settings.enable_plugins:
            self.plugin_manager.load_builtin_plugins()

        self.audio_processor.set_engines(
            self.transcription_engine,
            self.translation_engine,
            self.plugin_manager
        )

        # Setup GUI
        self.layout.setup_gui()
        self.layout.process_gui_updates()
        self.layout.process_batch_text_updates()
        self.start_system_monitoring()
        self.show_gpu_info()
        
        # Delayed tooltip initialization
        self.root.after(1500, self.safe_tooltip_init)

    def _actual_ui_update(self, component, text):
        """Direct GUI update bypassing queue issues"""
        try:
            if component == 'transcript':
                if hasattr(self, 'transcript_text') and self.transcript_text.winfo_exists():
                    self.transcript_text.insert('end', text)
                    if hasattr(self, 'transcript_scroll_var') and self.transcript_scroll_var.get():
                        self.transcript_text.see('end')
            elif component == 'translation':
                if hasattr(self, 'translation_text') and self.translation_text.winfo_exists():
                    self.translation_text.insert('end', text)
                    if hasattr(self, 'translation_scroll_var') and self.translation_scroll_var.get():
                        self.translation_text.see('end')
        except Exception as e:
            logging.debug(f"Direct UI Update error: {e}")

    def _actual_status_update(self, state_info):
        """Direct status update"""
        try:
            if 'status' in state_info:
                status_text = state_info['status']
                if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                    self.status_label.config(text=status_text)

            if 'stream_info' in state_info:
                stream_info = state_info['stream_info']
                if hasattr(self, 'stream_title_label') and self.stream_title_label.winfo_exists():
                    title = stream_info.title[:100] + "..." if len(stream_info.title) > 100 else stream_info.title
                    self.stream_title_label.config(text=f"📡 {title}")

                if hasattr(self, 'stream_details_label') and self.stream_details_label.winfo_exists():
                    details = f"👤 {stream_info.uploader}"
                    if stream_info.duration and stream_info.duration != 'Live':
                        details += f" | ⏱️ {stream_info.duration}"
                    self.stream_details_label.config(text=details)

            if 'processing_state' in state_info:
                processing = state_info['processing_state']
                if hasattr(self, 'start_button') and self.start_button.winfo_exists():
                    self.start_button.config(state='disabled' if processing else 'normal')
                if hasattr(self, 'stop_button') and self.stop_button.winfo_exists():
                    self.stop_button.config(state='normal' if processing else 'disabled')
                    
        except Exception as e:
            logging.debug(f"Status update error: {e}")

    def run(self):
        """Start the GUI with improved tooltip initialization."""
        self.root.protocol("WM_DELETE_WINDOW", self.controller.safe_exit)
        self.root.after(100, self.layout.process_gui_updates)
        self.root.after(100, self.layout.process_batch_text_updates)
        self._tooltips_created = set()
        self.root.after(1500, self.safe_tooltip_init)
        self.root.mainloop()

        if hasattr(self, 'signal_handler') and hasattr(self.signal_handler, 'is_shutdown_requested'):
            if not self.signal_handler.is_shutdown_requested():
                self.signal_handler.request_shutdown()

    def safe_tooltip_init(self):
        """Ultimate tooltip creation with improved widget validation."""
        if not hasattr(self, '_tooltips_created'):
            self._tooltips_created = set()
        elif len(self._tooltips_created) >= 16:
            return

        def setup_all_tooltips():
            if hasattr(self, '_tooltips_created') and len(self._tooltips_created) >= 16:
                return
            
            try:
                self.root.update_idletasks()
                self.root.update()
            
                tooltip_assignments = [
                    (self.start_button, "Start transcription and translation"),
                    (self.stop_button, "Stop processing"),
                    (self.subtitle_btn, "Toggle subtitle mode with timestamps"),
                    (self.layout_btn, "Switch between horizontal and vertical layout"),
                    (self.exit_button, "Exit application safely"),
                    (self.model_combo, "Select AI model for transcription"),
                    (self.lang_combo, "Select target language for translation"),
                    (self.url_entry, "Enter YouTube/Twitch URL or file path"),
                ]
            
                created_count = 0
                for widget, tooltip_text in tooltip_assignments:
                    if widget is None:
                        continue
                    
                    widget_id = str(widget)
                    if self._is_widget_ready(widget) and widget_id not in self._tooltips_created:
                        try:
                            ToolTip(widget, tooltip_text, delay=400)
                            self._tooltips_created.add(widget_id)
                            created_count += 1
                        except Exception:
                            pass
            
                symbol_buttons = self._find_symbol_buttons()
                for button, symbol in symbol_buttons:
                    if button is None:
                        continue
                    
                    button_id = str(button)
                    if self._is_widget_ready(button) and button_id not in self._tooltips_created:
                        tooltip_text = self._get_tooltip_for_symbol(symbol)
                        if tooltip_text:
                            try:
                                ToolTip(button, tooltip_text, delay=100)
                                self._tooltips_created.add(button_id)
                                created_count += 1
                            except Exception:
                                pass
                
            except Exception:
                pass

        if len(self._tooltips_created) < 16:
            if not hasattr(self, '_tooltip_after_ids'):
                self._tooltip_after_ids = []
            after_id1 = self.root.after(1000, setup_all_tooltips)
            after_id2 = self.root.after(3000, setup_all_tooltips)
            self._tooltip_after_ids.extend([after_id1, after_id2])

    def _is_widget_ready(self, widget):
        """Improved check if widget is ready for tooltip."""
        if widget is None:
            return False
        
        try:
            return (hasattr(widget, 'winfo_exists') and 
                    widget.winfo_exists() and 
                    widget.winfo_ismapped() and
                    widget.winfo_width() > 15 and
                    widget.winfo_height() > 15)
        except (tk.TclError, AttributeError):
            return False

    def _find_symbol_buttons(self):
        """Find buttons with symbols."""
        symbol_buttons = []

        def find_buttons(parent):
            try:
                for child in parent.winfo_children():
                    try:
                        if isinstance(child, tk.Button):
                            try:
                                text = child.cget('text')
                                if text and any(char in text for char in ['📁', '📋', '🔄', '🗑️', '💾', '📝', '📊', '⚙️', '🚨']):
                                    symbol_buttons.append((child, text))
                            except tk.TclError:
                                pass
                    
                        if isinstance(child, (tk.Frame, tk.LabelFrame, ttk.Frame, ttk.PanedWindow)):
                            find_buttons(child)
                        
                    except (tk.TclError, AttributeError):
                        continue
                    
            except (tk.TclError, AttributeError):
                pass

        find_buttons(self.root)
        return symbol_buttons

    def _get_tooltip_for_symbol(self, symbol):
        """Tooltip text for symbols."""
        tooltip_map = {
            '📁': "Select audio/video file",
            '📋': "Paste URL from clipboard", 
            '🔄': "Switch layout orientation",
            '🗑️': "Clear all text and reset",
            '💾': "Save transcript to file", 
            '📝': "Export subtitles (SRT/VTT)",
            '📊': "Show performance statistics",
            '⚙️': "Open advanced settings",
            '🚨': "Open enterprise monitoring dashboard"
        }
        return tooltip_map.get(symbol, "")

    def show_gpu_info(self):
        """Display GPU information if available."""
        device = self.transcription_engine.device
        compute_type = self.transcription_engine.compute_type

        if device == "cuda" and TORCH_AVAILABLE:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                info_text = f"🎮 GPU: {gpu_name} ({vram:.1f}GB) - {compute_type}"
            except:
                info_text = "🎮 NVIDIA GPU activated"
        elif device == "xpu":
            info_text = "🎮 Intel GPU activated"
        else:
            info_text = "💻 CPU mode activated"

        print(f"⚡ Hardware: {info_text}")

    def start_system_monitoring(self):
        """Start system monitoring for CPU, memory, and GPU usage."""
        def update_system_info():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_mb = memory.used // (1024 * 1024)

                gpu_info = ""
                try:
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        gpu_info = f" | 🎮 GPU: {gpu_memory:.1f}GB"
                except:
                    pass

                self.system_info_label.config(
                    text=f"💻 CPU: {cpu_percent:.0f}% | 🧠 RAM: {memory_mb}MB{gpu_info} | ⚡ v1.02"
                )

            except Exception:
                pass

            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(2000, update_system_info)

        update_system_info()

    def handle_transcription(self, result: ExcellenceTranscriptionResult):
        """DIRECT GUI UPDATE METHOD without terminal output"""
        if not result or not result.text.strip():
            return

        current_text = result.text.strip()
        if current_text == self._last_transcription_text:
            return

        self._last_transcription_text = current_text
        self.performance_monitor.log_transcription()
        self.transcript_history.append(result)

        timestamp = datetime.now().strftime("%H:%M:%S")
        detected_lang = getattr(result, 'language', 'unknown')
        lang_display = LANGUAGE_SHORT_CODES.get(detected_lang, detected_lang)

        source_lang = detected_lang
        target_lang = self.current_language
    
        should_translate = (
            self.translation_engine is not None and 
            getattr(self, 'translate_toggle', None) is not None and
            self.translate_toggle.get() and
            source_lang not in ['unknown', 'auto', target_lang] and
            len(current_text) >= 2
        )

        time_info = ""
        if self.subtitle_mode and hasattr(result, 'start') and result.start is not None:
            start_str = f"{result.start:.1f}s"
            end_str = f"{result.end:.1f}s"
            time_info = f" [{start_str}→{end_str}]"

        if AsianLanguageSupport.should_use_word_segmentation(detected_lang):
            display_text = AsianLanguageSupport.optimize_display_text(result.text, detected_lang)
        else:
            display_text = result.text

        text = f"[{timestamp}]{time_info} [{lang_display}] {display_text}\n"

        def update_gui_directly():
            try:
                if hasattr(self, 'transcript_text') and self.transcript_text.winfo_exists():
                    self.transcript_text.insert('end', text)
                    if hasattr(self, 'transcript_scroll_var') and self.transcript_scroll_var.get():
                        self.transcript_text.see('end')
                
                    lines = int(self.transcript_text.index('end-1c').split('.')[0])
                    if lines > 400:
                        keep_lines = 250
                        delete_to = f'{lines-keep_lines}.0'
                        self.transcript_text.delete('1.0', delete_to)
                    
            except Exception:
                pass

        self.root.after(0, update_gui_directly)

        if should_translate:
            def translate_async():
                try:
                    time.sleep(0.2)
                    translation = self.translation_engine.translate_text(
                        current_text, 
                        source_lang
                    )
                    if translation:
                        self.handle_translation(translation)
                except Exception:
                    pass
        
            translation_thread = threading.Thread(target=translate_async, daemon=True)
            translation_thread.start()

    def handle_translation(self, result: ExcellenceTranslationResult):
        """DIRECT GUI UPDATE METHOD for translation"""
        if not self.translate_toggle.get() or not result:
            return

        current_text = result.translated.strip()
        if current_text == self._last_translation_text:
            return

        self._last_translation_text = current_text
        self.performance_monitor.log_translation()
        self.translation_history.append(result)

        timestamp = datetime.now().strftime("%H:%M:%S")
        lang_display = LANGUAGE_SHORT_CODES.get(result.target_lang, result.target_lang)

        time_info = ""
        if self.subtitle_mode and hasattr(result, 'start') and result.start is not None:
            start_str = f"{result.start:.1f}s"
            end_str = f"{result.end:.1f}s"
            time_info = f" [{start_str}→{end_str}]"

        text = f"[{timestamp}]{time_info} [{lang_display}] {result.translated}\n"

        def update_gui_directly():
            try:
                if hasattr(self, 'translation_text') and self.translation_text.winfo_exists():
                    self.translation_text.insert('end', text)
                    if hasattr(self, 'translation_scroll_var') and self.translation_scroll_var.get():
                        self.translation_text.see('end')
                    
                    lines = int(self.translation_text.index('end-1c').split('.')[0])
                    if lines > 400:
                        keep_lines = 250
                        delete_to = f'{lines-keep_lines}.0'
                        self.translation_text.delete('1.0', delete_to)
                        
            except Exception as e:
                logging.debug(f"Direct translation GUI update error: {e}")

        self.root.after(0, update_gui_directly)

    def handle_info(self, info_msg: str):
        """Handle information messages."""
        def update():
            self.controller.status_update_fn({'status': f"ℹ️ {info_msg}"})

        try:
            if self.gui_queue.qsize() < 50:
                self.gui_queue.put(('info', update))
        except:
            pass

    def handle_error(self, error_msg: str):
        """Handle error messages."""
        def update():
            self.controller.status_update_fn({'status': f"❌ {error_msg}"})
            if self.is_processing:
                self.controller.stop_processing()

        try:
            if self.gui_queue.qsize() < 50:
                self.gui_queue.put(('error', update))
        except:
            pass

    def safe_update_status(self, message: str):
        """Safely update status message."""
        self.controller.status_update_fn({'status': message})

    def safe_stream_info_update(self, info: StreamInfo):
        """Safely update stream information."""
        self.controller.status_update_fn({'stream_info': info})

    def update_stream_info(self, info: StreamInfo):
        """Compatibility method for stream info updates."""
        self.safe_stream_info_update(info)

    def update_status(self, message: str):
        """Compatibility method for status updates."""
        self.safe_update_status(message)

    def on_language_change(self, event=None):
        """Handle language change selection."""
        try:
            selected_name = self.lang_var.get()
            lang_code = None
            for name, code in SORTED_LANGUAGES:
                if name == selected_name:
                    lang_code = code
                    break

            if lang_code and lang_code != self.current_language:
                self.current_language = lang_code
                self.translation_engine.set_target_language(lang_code)

                lang_display = LANGUAGE_SHORT_CODES.get(lang_code, lang_code)
                if hasattr(self, 'translation_header'):
                    self.translation_header.config(text=f"🌐 Translation ({lang_display})")

                self.safe_update_status(f"🌍 Target language: {selected_name}")

        except Exception as e:
            logging.error(f"Language change error: {e}")

    def on_model_change(self, event=None):
        """Handle AI model change selection."""
        new_model = self.model_var.get()
        current_model = self.transcription_engine.get_current_model()

        if new_model == current_model:
            return

        if self.transcription_engine.is_model_loading():
            self.safe_update_status("🔄 Model loading...")
            return

        def reload_model_thread():
            self.safe_update_status(f"🔄 Switching model: {new_model}")

            success = self.transcription_engine.reload_model(new_model)

            if success:
                self.safe_update_status(f"✅ Model: {new_model}")
                self.settings.default_model = new_model
            else:
                self.safe_update_status("❌ Model switch failed")
                self.model_var.set(current_model)

        reload_thread = threading.Thread(target=reload_model_thread, daemon=True)
        self.resource_manager.register_thread(reload_thread)
        reload_thread.start()

    def toggle_translation(self):
        """Toggle translation on/off."""
        if self.translate_toggle.get():
            self.translation_engine.set_target_language(self.current_language)
            self.safe_update_status("✅ Translation active")
        else:
            self.safe_update_status("❌ Translation inactive")

    def clean_and_validate_url(self, url):
        """Clean and validate URL for processing."""
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

    @excellence_gui_operation
    def paste_url(self):
        """Paste URL from clipboard with validation."""
        try:
            clipboard = self.root.clipboard_get().strip()
            if clipboard:
                cleaned_url = self.clean_and_validate_url(clipboard)
                self.url_entry.delete(0, 'end')
                self.url_entry.insert(0, cleaned_url)
                self.safe_update_status("📋 URL pasted")

                if cleaned_url.startswith('file://'):
                    file_path = cleaned_url[7:]
                    if os.path.exists(file_path):
                        def async_detection():
                            try:
                                self.analyze_video_language(file_path)
                            except Exception as e:
                                logging.debug(f"Paste URL detection failed: {e}")

                        detection_thread = threading.Thread(target=async_detection, daemon=True)
                        self.resource_manager.register_thread(detection_thread)
                        detection_thread.start()
        except ValueError as e:
            self.safe_update_status(f"❌ Invalid URL: {e}")
        except Exception:
            self.safe_update_status("❌ No URL in clipboard")

    @excellence_gui_operation
    def export_subtitles(self):
        """Export subtitles in SRT or VTT format."""
        if not self.audio_processor.timed_transcriptions:
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

                self.safe_update_status(f"📝 {format_type.upper()} exported: {os.path.basename(filename)}")

                DarkMessageBox.showinfo("Success",
                    f"Subtitles successfully exported!\n\n"
                    f"• File: {os.path.basename(filename)}\n"
                    f"• Segments: {segment_count}\n"
                    f"• Translations: {translation_count}\n"
                    f"• Format: {format_type.upper()}\n\n"
                    f"Can be directly imported into video editors.", self.root)
            else:
                self.safe_update_status("❌ Subtitle export failed")

        except Exception as e:
            self.safe_update_status(f"❌ Subtitle export failed: {e}")
            DarkMessageBox.showerror("Error", f"Export failed:\n{str(e)}", self.root)

    @excellence_gui_operation
    def save_transcript(self):
        """Save transcription and translation results to file."""
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
                        f.write(f"Saved at: {datetime.now().strftime('%Y-%m-d %H:%M:%S')}\n\n")

                    f.write("=== TRANSCRIPT ===\n")
                    f.write(self.transcript_text.get('1.0', 'end-1c'))
                    f.write("\n\n=== TRANSLATION ===\n")
                    f.write(self.translation_text.get('1.0', 'end-1c'))
                success = True

            if success:
                self.safe_update_status(f"💾 Saved: {os.path.basename(filename)}")
            else:
                self.safe_update_status("❌ Export failed")

        except Exception as e:
            self.safe_update_status(f"❌ Save failed: {e}")

    @excellence_gui_operation
    def select_file_dark(self):
        """File selection with non-blocking operation."""
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
                self.safe_update_status(f"📁 File selected: {os.path.basename(filename)}")

                def async_language_detection():
                    try:
                        self.analyze_video_language(filename)
                    except Exception as e:
                        logging.debug(f"Language detection failed: {e}")

                detection_thread = threading.Thread(target=async_language_detection, daemon=True)
                self.resource_manager.register_thread(detection_thread)
                detection_thread.start()

                info = StreamInfoExtractor().extract_stream_info(file_url)
                self.safe_stream_info_update(info)

        except Exception as e:
            self.safe_update_status(f"❌ File selection failed: {e}")

    def on_url_change(self, event=None):
        """Called when URL changes - checks for file URLs."""
        url = self.url_entry.get().strip()
        if url.startswith('file://'):
            file_path = url[7:]
            if os.path.exists(file_path):
                def async_detection():
                    try:
                        self.analyze_video_language(file_path)
                    except Exception as e:
                        logging.debug(f"Background language detection failed: {e}")

                detection_thread = threading.Thread(target=async_detection, daemon=True)
                self.resource_manager.register_thread(detection_thread)
                detection_thread.start()
            else:
                self.language_info_label.config(text="❌ File not found")
        else:
            self.language_info_label.config(text="")
            self.current_video_language = None

    def analyze_video_language(self, file_path: str):
        """Fully asynchronous language detection for video files."""
        def create_progress_dialog():
            self.progress_dialog = ProgressDialog(self.root, "Analyzing Video Language...")
            self.progress_dialog.update_message("Loading AI model...")

        self.gui_queue.put(('language_detection', create_progress_dialog))

        def language_detection_worker():
            try:
                def update_loading_progress(phase):
                    def update():
                        if hasattr(self, 'progress_dialog') and self.progress_dialog:
                            self.progress_dialog.update_message(f"{phase}...")
                    self.gui_queue.put(('language_detection', update))

                update_loading_progress("Loading AI model")

                model_loaded = threading.Event()
                model_error = [None]

                def load_model_thread():
                    try:
                        if hasattr(self, 'progress_dialog') and self.progress_dialog and self.progress_dialog.is_cancelled:
                            return

                        if not self.transcription_engine.model:
                            success = self.transcription_engine.load_model(self.model_var.get())
                            if not success:
                                model_error[0] = "Model could not be loaded"
                    except Exception as e:
                        model_error[0] = str(e)
                    finally:
                        model_loaded.set()

                loader_thread = threading.Thread(target=load_model_thread, daemon=True)
                self.resource_manager.register_thread(loader_thread)
                loader_thread.start()

                wait_start = time.time()
                while not model_loaded.is_set() and (time.time() - wait_start) < 30:
                    if hasattr(self, 'progress_dialog') and self.progress_dialog and self.progress_dialog.is_cancelled:
                        def update_cancelled():
                            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                                self.progress_dialog.close()
                            self.language_info_label.config(text="❌ Analysis cancelled")
                        self.gui_queue.put(('language_detection', update_cancelled))
                        return
                    time.sleep(0.1)

                if not model_loaded.is_set():
                    model_error[0] = "Model loading timeout"

                if model_error[0]:
                    def update_ui_error():
                        if hasattr(self, 'progress_dialog') and self.progress_dialog:
                            self.progress_dialog.close()
                        self.language_info_label.config(text=f"❌ {model_error[0]}")
                    self.gui_queue.put(('language_detection', update_ui_error))
                    return

                update_loading_progress("Extracting audio sample")

                if hasattr(self, 'progress_dialog') and self.progress_dialog and self.progress_dialog.is_cancelled:
                    def update_cancelled():
                        if hasattr(self, 'progress_dialog') and self.progress_dialog:
                            self.progress_dialog.close()
                        self.language_info_label.config(text="❌ Analysis cancelled")
                    self.gui_queue.put(('language_detection', update_cancelled))
                    return

                update_loading_progress("Detecting language")

                detection_result = [None]
                detection_error = [None]
                detection_done = threading.Event()

                def detect_language_thread():
                    try:
                        if hasattr(self, 'progress_dialog') and self.progress_dialog and self.progress_dialog.is_cancelled:
                            return
                        detection_result[0] = self.language_detector.detect_video_language(file_path)
                    except Exception as e:
                        detection_error[0] = str(e)
                    finally:
                        detection_done.set()

                detector_thread = threading.Thread(target=detect_language_thread, daemon=True)
                self.resource_manager.register_thread(detector_thread)
                detector_thread.start()

                wait_start = time.time()
                while not detection_done.is_set() and (time.time() - wait_start) < 45:
                    if hasattr(self, 'progress_dialog') and self.progress_dialog and self.progress_dialog.is_cancelled:
                        def update_cancelled():
                            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                                self.progress_dialog.close()
                            self.language_info_label.config(text="❌ Analysis cancelled")
                        self.gui_queue.put(('language_detection', update_cancelled))
                        return
                    time.sleep(0.1)

                def process_final_result():
                    if hasattr(self, 'progress_dialog') and self.progress_dialog:
                        self.progress_dialog.close()

                    if detection_error[0]:
                        self.language_info_label.config(text=f"❌ Detection failed: {detection_error[0]}")
                        return

                    result = detection_result[0]
                    if not result:
                        self.language_info_label.config(text="❌ No detection result")
                        return

                    if 'error' in result:
                        self.language_info_label.config(text=f"❌ {result['error']}")
                    elif 'info' in result:
                        self.language_info_label.config(text=f"ℹ️ {result['info']}")
                    else:
                        language_name = result['language_name']
                        confidence = result['confidence']
                        self.current_video_language = result['detected_language']

                        language_icons = {
                            'zh': '㊗️',  'ja': '🗾',  'ko': '₩',
                            'th': '🇹🇭', 'vi': '🇻🇳',
                        }

                        icon = language_icons.get(self.current_video_language, '✅')
                        display_text = f"{icon} {language_name} ({confidence:.0%})"
                        self.language_info_label.config(text=display_text)

                        logging.info(f"Language detected: {language_name} ({self.current_video_language}) - Confidence: {confidence}")

                self.gui_queue.put(('language_detection', process_final_result))

            except Exception as e:
                def update_ui_failed():
                    if hasattr(self, 'progress_dialog') and self.progress_dialog:
                        self.progress_dialog.close()
                    self.language_info_label.config(text="❌ Analysis failed")
                    logging.debug(f"Language analysis failed: {e}")

                self.gui_queue.put(('language_detection', update_ui_failed))

        detection_thread = threading.Thread(target=language_detection_worker, daemon=True)
        self.resource_manager.register_thread(detection_thread)
        detection_thread.start()

    @excellence_gui_operation
    def toggle_subtitle_mode(self):
        """Toggle subtitle mode with color feedback."""
        self.subtitle_mode = not self.subtitle_mode
        self.audio_processor.enable_subtitle_mode(self.subtitle_mode)

        if self.subtitle_mode:
            self.subtitle_btn.config(bg=ModernColors.SUBTITLE_ACTIVE, fg=ModernColors.TEXT_PRIMARY)
            self.safe_update_status("🎬 SUBTITLE MODE: Timestamps activated")
        else:
            self.subtitle_btn.config(bg=ModernColors.SUBTITLE_INACTIVE, fg=ModernColors.TEXT_PRIMARY)
            self.safe_update_status("📝 NORMAL MODE: Continuous text")
    
    @excellence_gui_operation
    def toggle_layout(self):
        """Toggle between horizontal and vertical layout."""
        transcript_content = self.transcript_text.get('1.0', 'end-1c')
        translation_content = self.translation_text.get('1.0', 'end-1c')

        if self.layout_mode == "vertical":
            self.layout_mode = "horizontal"
        else:
            self.layout_mode = "vertical"

        self.settings.layout_mode = self.layout_mode
        self.layout.create_text_areas()
        self.transcript_text.insert('1.0', transcript_content)
        self.translation_text.insert('1.0', translation_content)
        self.safe_update_status(f"📐 Layout: {self.layout_mode}")

    @excellence_gui_operation
    def clear_all(self):
        """Clear all text areas with excellence memory management."""
        self.transcript_text.delete('1.0', 'end')
        self.translation_text.delete('1.0', 'end')
        self.transcript_history.clear()
        self.translation_history.clear()
        self.memory_manager.clear_component('transcript')
        self.memory_manager.clear_component('translation')
        self._last_transcription_text = ""
        self._last_translation_text = ""
        self.safe_update_status("🗑️ Cleared & optimizations reset")

    @excellence_gui_operation
    def show_performance_stats(self):
        """Show performance statistics with excellence monitor."""
        try:
            stats = self.performance_monitor.get_performance_stats()
            stats['total_plugins'] = len(self.plugin_manager.plugins)

            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            gpu_info = ""
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_memory_max = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_info = f"🎮 GPU memory: {gpu_memory:.1f}GB / {gpu_memory_max:.1f}GB\n"
            except:
                pass

            stats_text = f"""📊 STATISTICS:

🤖 AI-PERFORMANCE:
⏱️ Runtime: {stats['total_uptime']/60:.1f} minutes
📝 Transcriptions: {stats['transcriptions']}
🌐 Translations: {stats['translations']}
⚡ Speed: {stats['transcripts_per_hour']:.1f}/min
📈 Avg latency: {stats['avg_processing_time']:.2f}s
🎯 Operations/min: {stats['operations_per_minute']:.1f}
🧩 Active plugins: {stats['total_plugins']}
🎬 Subtitle mode: {'Active' if self.subtitle_mode else 'Inactive'}
{gpu_info}
💻 SYSTEM:
🖥️ CPU: {cpu_percent:.1f}%
🧠 RAM: {memory.percent:.1f}% ({memory.used//1024**2}MB)
📊 Memory peak: {stats['memory_peak'] // 1024 // 1024}MB
"""
            DarkMessageBox.showinfo("Performance Statistics", stats_text, self.root)

        except Exception as e:
            self.safe_update_status(f"❌ Statistics error: {e}")

    @excellence_gui_operation
    def show_advanced_settings(self):
        """Show advanced settings dialog."""
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("Advanced Settings")
        settings_dialog.geometry("400x500")
        settings_dialog.configure(bg=ModernColors.BG_PRIMARY)
        settings_dialog.transient(self.root)
        settings_dialog.grab_set()
        settings_dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - settings_dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - settings_dialog.winfo_height()) // 2
        settings_dialog.geometry(f"+{x}+{y}")

        main_frame = tk.Frame(settings_dialog, bg=ModernColors.BG_PRIMARY, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        tk.Label(main_frame, text="Advanced Settings",
                font=ModernFonts.TITLE, bg=ModernColors.BG_PRIMARY, fg=ModernColors.TEXT_PRIMARY).pack(pady=(0, 20))

        settings_frame = tk.Frame(main_frame, bg=ModernColors.BG_PRIMARY)
        settings_frame.pack(fill='both', expand=True)

        tk.Label(settings_frame, text="Beam Size:", bg=ModernColors.BG_PRIMARY,
                fg=ModernColors.TEXT_PRIMARY).grid(row=0, column=0, sticky='w', pady=5)
        beam_var = tk.StringVar(value=str(self.advanced_settings.beam_size))
        beam_entry = tk.Entry(settings_frame, textvariable=beam_var,
                             bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY)
        beam_entry.grid(row=0, column=1, sticky='ew', pady=5)

        tk.Label(settings_frame, text="Temperature:", bg=ModernColors.BG_PRIMARY,
                fg=ModernColors.TEXT_PRIMARY).grid(row=1, column=0, sticky='w', pady=5)
        temp_var = tk.StringVar(value=str(self.advanced_settings.temperature))
        temp_entry = tk.Entry(settings_frame, textvariable=temp_var,
                             bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY)
        temp_entry.grid(row=1, column=1, sticky='ew', pady=5)

        tk.Label(settings_frame, text="Max Memory (MB):", bg=ModernColors.BG_PRIMARY,
                fg=ModernColors.TEXT_PRIMARY).grid(row=2, column=0, sticky='w', pady=5)
        memory_var = tk.StringVar(value=str(self.advanced_settings.max_memory_mb))
        memory_entry = tk.Entry(settings_frame, textvariable=memory_var,
                              bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY)
        memory_entry.grid(row=2, column=1, sticky='ew', pady=5)

        plugin_var = tk.BooleanVar(value=self.settings.enable_plugins)
        plugin_cb = tk.Checkbutton(settings_frame, text="Enable plugins",
                                  variable=plugin_var, bg=ModernColors.BG_PRIMARY,
                                  fg=ModernColors.TEXT_PRIMARY, selectcolor=ModernColors.BG_TERTIARY)
        plugin_cb.grid(row=3, column=0, columnspan=2, sticky='w', pady=5)

        gpu_var = tk.BooleanVar(value=self.advanced_settings.gpu_acceleration)
        gpu_cb = tk.Checkbutton(settings_frame, text="Enable GPU acceleration",
                               variable=gpu_var, bg=ModernColors.BG_PRIMARY,
                               fg=ModernColors.TEXT_PRIMARY, selectcolor=ModernColors.BG_TERTIARY)
        gpu_cb.grid(row=4, column=0, columnspan=2, sticky='w', pady=5)

        translation_var = tk.BooleanVar(value=False)
        translation_cb = tk.Checkbutton(settings_frame, text="Place holder ()",
                                       variable=translation_var, bg=ModernColors.BG_PRIMARY,
                                       fg=ModernColors.TEXT_SECONDARY, selectcolor=ModernColors.BG_TERTIARY,
                                       state='disabled')
        translation_cb.grid(row=5, column=0, columnspan=2, sticky='w', pady=5)

        settings_frame.columnconfigure(1, weight=1)

        def save_settings():
            try:
                self.advanced_settings.beam_size = int(beam_var.get())
                self.advanced_settings.temperature = float(temp_var.get())
                self.advanced_settings.max_memory_mb = int(memory_var.get())
                self.settings.enable_plugins = plugin_var.get()
                self.advanced_settings.gpu_acceleration = gpu_var.get()
                self.advanced_settings.optimize_translations = False
                self.advanced_settings.save_to_file()
                self.settings.save_to_file()
                self.plugin_manager.enabled = self.settings.enable_plugins

                if not self.advanced_settings.gpu_acceleration:
                    self.transcription_engine.device = "cpu"
                    self.transcription_engine.compute_type = "int8"

                settings_dialog.destroy()
                self.safe_update_status("✅ Settings saved")

            except Exception as e:
                DarkMessageBox.showerror("Error", f"Invalid settings: {e}", self.root)

        button_frame = tk.Frame(main_frame, bg=ModernColors.BG_PRIMARY)
        button_frame.pack(fill='x', pady=(20, 0))

        save_btn = tk.Button(
            button_frame, text="Save", command=save_settings,
            bg=ModernColors.SUCCESS, fg=ModernColors.TEXT_PRIMARY,
            relief='flat', padx=15
        )
        save_btn.pack(side='right', padx=5)

        cancel_btn = tk.Button(
            button_frame, text="Cancel", command=settings_dialog.destroy,
            bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
            relief='flat', padx=15)
        cancel_btn.pack(side='right', padx=5)

    @excellence_gui_operation
    def show_enterprise_dashboard(self):
        """Comprehensive enterprise monitoring dashboard with safe fallbacks."""
        try:
            dashboard = tk.Toplevel(self.root)
            dashboard.title("🐉 Enterprise Dashboard - Dragon Whisperer v1.0")
            dashboard.geometry("1000x700")
            dashboard.configure(bg=ModernColors.BG_PRIMARY)
            dashboard.transient(self.root)
            dashboard.grab_set()

            dashboard.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - dashboard.winfo_width()) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - dashboard.winfo_height()) // 2
            dashboard.geometry(f"+{x}+{y}")

            close_btn_top = tk.Button(
                dashboard, text="✕", command=dashboard.destroy,
                bg=ModernColors.ERROR, fg=ModernColors.TEXT_PRIMARY,
                font=("Segoe UI", 12, "bold"), relief='flat', width=3,
                cursor='hand2'
            )
            close_btn_top.place(relx=0.98, rely=0.02, anchor='ne')

            notebook = ttk.Notebook(dashboard)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)

            try:
                health_frame = ttk.Frame(notebook, padding=10)
                notebook.add(health_frame, text="🚨 System Health")
                self._create_health_tab(health_frame)
            except Exception as e:
                logging.error(f"Health tab creation failed: {e}")
                error_label = tk.Label(health_frame, text=f"Health tab unavailable: {str(e)}", 
                                     bg=ModernColors.BG_PRIMARY, fg=ModernColors.ERROR)
                error_label.pack(pady=20)

            try:
                analytics_frame = ttk.Frame(notebook, padding=10)
                notebook.add(analytics_frame, text="📊 Performance Analytics")
                self._create_analytics_tab(analytics_frame)
            except Exception as e:
                logging.error(f"Analytics tab creation failed: {e}")
                error_label = tk.Label(analytics_frame, text=f"Analytics tab unavailable: {str(e)}",
                                     bg=ModernColors.BG_PRIMARY, fg=ModernColors.ERROR)
                error_label.pack(pady=20)

            try:
                recovery_frame = ttk.Frame(notebook, padding=10)
                notebook.add(recovery_frame, text="🔧 Auto-Recovery")
                self._create_recovery_tab(recovery_frame)
            except Exception as e:
                logging.error(f"Recovery tab creation failed: {e}")
                error_label = tk.Label(recovery_frame, text=f"Recovery tab unavailable: {str(e)}",
                                     bg=ModernColors.BG_PRIMARY, fg=ModernColors.ERROR)
                error_label.pack(pady=20)

            button_frame = tk.Frame(dashboard, bg=ModernColors.BG_PRIMARY)
            button_frame.pack(fill='x', pady=10)

            refresh_btn = tk.Button(
                button_frame, text="🔄 Refresh Dashboard", 
                command=lambda: self._refresh_dashboard(notebook),
                bg=ModernColors.SUCCESS, fg=ModernColors.TEXT_PRIMARY,
                relief='flat', padx=15
            )
            refresh_btn.pack(side='left', padx=10)

            close_btn_bottom = tk.Button(
                button_frame, text="Schließen", command=dashboard.destroy,
                bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
                relief='flat', padx=20, cursor='hand2'
            )
            close_btn_bottom.pack(side='right', padx=10)

        except Exception as e:
            logging.error(f"Enterprise dashboard error: {e}")
            DarkMessageBox.showerror("Dashboard Error", f"Could not open dashboard:\n{str(e)}", self.root)

    def _create_health_tab(self, parent):
        """Create health monitoring tab with dark theme and exception handling."""
        try:
            status_frame = tk.LabelFrame(parent, text="System Health Status", 
                                       bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                       font=ModernFonts.SUBTITLE, padx=10, pady=10)
            status_frame.pack(fill='x', pady=5)

            self.health_status_label = tk.Label(
                status_frame, text="Loading health data...", 
                font=ModernFonts.TITLE, bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY
            )
            self.health_status_label.pack(pady=5)

            health_tree_frame = tk.Frame(parent, bg=ModernColors.BG_PRIMARY)
            health_tree_frame.pack(fill='both', expand=True, pady=5)

            style = ttk.Style()
            style.configure("Dark.Treeview", 
                           background=ModernColors.BG_TERTIARY,
                           foreground=ModernColors.TEXT_PRIMARY,
                           fieldbackground=ModernColors.BG_TERTIARY)
            style.configure("Dark.Treeview.Heading",
                           background=ModernColors.BG_SECONDARY,
                           foreground=ModernColors.TEXT_ACCENT,
                           relief='flat')
            
            columns = ('Metric', 'Value', 'Status')
            self.health_tree = ttk.Treeview(health_tree_frame, columns=columns, show='headings', 
                                           height=15, style="Dark.Treeview")
            
            for col in columns:
                self.health_tree.heading(col, text=col)
                self.health_tree.column(col, width=200)

            scrollbar = ttk.Scrollbar(health_tree_frame, orient='vertical', command=self.health_tree.yview)
            self.health_tree.configure(yscrollcommand=scrollbar.set)            
            self.health_tree.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
        except Exception as e:
            logging.error(f"Health tab creation error: {e}")
            raise

    def _create_analytics_tab(self, parent):
        """Create analytics tab with dark theme and exception handling."""
        try:
            overview_frame = tk.LabelFrame(parent, text="Performance Overview",
                                         bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                         font=ModernFonts.SUBTITLE, padx=10, pady=10)
            overview_frame.pack(fill='x', pady=5)

            self.analytics_overview_label = tk.Label(
                overview_frame, text="Loading analytics...",
                font=ModernFonts.PRIMARY, bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                justify='left', wraplength=800
            )
            self.analytics_overview_label.pack(pady=5)

            metrics_frame = tk.LabelFrame(parent, text="Detailed Metrics",
                                        bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                        font=ModernFonts.SUBTITLE, padx=10, pady=10)
            metrics_frame.pack(fill='both', expand=True, pady=5)

            self.analytics_text = scrolledtext.ScrolledText(
                metrics_frame,
                bg=ModernColors.BG_TERTIARY,
                fg=ModernColors.TEXT_PRIMARY,
                font=ModernFonts.MONOSPACE,
                wrap=tk.WORD,
                height=15
            )
            self.analytics_text.pack(fill='both', expand=True, padx=5, pady=5)
        except Exception as e:
            logging.error(f"Analytics tab creation error: {e}")
            raise

    def _create_recovery_tab(self, parent):
        """Create recovery tab with dark theme and exception handling."""
        try:
            # Recovery status
            recovery_status_frame = tk.LabelFrame(parent, text="Auto-Recovery Status",
                                               bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                               font=ModernFonts.SUBTITLE, padx=10, pady=10)
            recovery_status_frame.pack(fill='x', pady=5)

            self.recovery_status_label = tk.Label(
                recovery_status_frame, text="Recovery system ready",
                font=ModernFonts.PRIMARY, bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY
            )
            self.recovery_status_label.pack(pady=5)

            # ✅ NEU: System Diagnosis Section
            diagnosis_frame = tk.LabelFrame(parent, text="🔍 System Diagnosis",
                                          bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                          font=ModernFonts.SUBTITLE, padx=10, pady=10)
            diagnosis_frame.pack(fill='x', pady=10)

            # Diagnosis Buttons
            diagnosis_btn_frame = tk.Frame(diagnosis_frame, bg=ModernColors.BG_SECONDARY)
            diagnosis_btn_frame.pack(pady=5)

            self.run_diagnosis_btn = tk.Button(
                diagnosis_btn_frame, text="🔍 Run Comprehensive Diagnosis",
                command=self._run_comprehensive_diagnosis,
                bg=ModernColors.SUCCESS, fg=ModernColors.TEXT_PRIMARY,
                font=ModernFonts.BUTTON, relief='flat', padx=15
            )
            self.run_diagnosis_btn.pack(side='left', padx=5)

            self.quick_check_btn = tk.Button(
                diagnosis_btn_frame, text="⚡ Quick System Check", 
                command=self._run_quick_check,
                bg=ModernColors.BG_TERTIARY, fg=ModernColors.TEXT_PRIMARY,
                font=ModernFonts.BUTTON, relief='flat', padx=15
            )
            self.quick_check_btn.pack(side='left', padx=5)

            # Diagnosis Results
            diagnosis_results_frame = tk.Frame(diagnosis_frame, bg=ModernColors.BG_SECONDARY)
            diagnosis_results_frame.pack(fill='x', pady=5)

            self.diagnosis_results_label = tk.Label(
                diagnosis_results_frame, text="Click a button above to run system checks",
                font=ModernFonts.PRIMARY, bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_SECONDARY,
                wraplength=600
            )
            self.diagnosis_results_label.pack(pady=5)

            # Recovery controls
            controls_frame = tk.Frame(parent, bg=ModernColors.BG_PRIMARY)
            controls_frame.pack(fill='x', pady=10)

            self.trigger_recovery_btn = tk.Button(
                controls_frame, text="🚨 Trigger Emergency Recovery",
                command=self._trigger_dashboard_recovery,
                bg=ModernColors.ERROR, fg=ModernColors.TEXT_PRIMARY,
                font=ModernFonts.BUTTON, relief='flat', padx=20
            )
            self.trigger_recovery_btn.pack(pady=5)

            # Recovery log
            log_frame = tk.LabelFrame(parent, text="Recovery Log",
                                    bg=ModernColors.BG_SECONDARY, fg=ModernColors.TEXT_PRIMARY,
                                    font=ModernFonts.SUBTITLE, padx=10, pady=10)
            log_frame.pack(fill='both', expand=True, pady=5)

            self.recovery_log_text = scrolledtext.ScrolledText(
                log_frame,
                bg=ModernColors.BG_TERTIARY,
                fg=ModernColors.TEXT_PRIMARY,
                font=ModernFonts.MONOSPACE,
                wrap=tk.WORD,
                height=10
            )
            self.recovery_log_text.pack(fill='both', expand=True, padx=5, pady=5)

        except Exception as e:
            logging.error(f"Recovery tab creation error: {e}")
            raise

    def _refresh_dashboard(self, notebook):
        """Refresh all dashboard tabs with safe exception handling."""
        try:
            health_data = self.performance_monitor.get_system_health()
            analytics_data = self.performance_monitor.get_detailed_analytics()

            self._update_health_tab(health_data)
            self._update_analytics_tab(analytics_data)
            self._update_recovery_tab()

        except Exception as e:
            logging.error(f"Dashboard refresh error: {e}")
    
    def _run_comprehensive_diagnosis(self):
        """Comprehensive system diagnosis"""
        def diagnosis_worker():
            try:
                self.diagnosis_results_label.config(
                    text="🔍 Running comprehensive diagnosis...", 
                    fg=ModernColors.WARNING
                )
            
                results = []
                results.append("📊 COMPREHENSIVE SYSTEM DIAGNOSIS")
                results.append("=" * 50)
            
                # Dependencies check
                results.append("\n📦 DEPENDENCIES")
                results.append("-" * 20)
                deps_checks = [
                    ('FFmpeg', shutil.which('ffmpeg')),
                    ('yt-dlp', shutil.which('yt-dlp')),
                    ('Whisper', WHISPER_AVAILABLE),
                    ('Translator', TRANSLATOR_AVAILABLE),
                    ('PyTorch', TORCH_AVAILABLE),
                    ('NumPy', NUMPY_AVAILABLE)
                ]
            
                all_deps_ok = True
                for name, available in deps_checks:
                    status = "✅" if available else "❌"
                    results.append(f"{status} {name}")
                    if not available:
                        all_deps_ok = False
            
                # Hardware check
                results.append("\n💻 HARDWARE")
                results.append("-" * 20)
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    results.append("✅ GPU: Available")
                    try:
                        gpu_name = torch.cuda.get_device_name(0)
                        results.append(f"   Device: {gpu_name}")
                    except:
                        results.append("   Device: Unknown")
                else:
                    results.append("❌ GPU: Not available")
            
                memory = psutil.virtual_memory()
                results.append(f"💾 RAM: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
            
                # Services check
                results.append("\n🌐 SERVICES")
                results.append("-" * 20)
            
                # Internet check
                internet_ok = self._check_internet_connection()
                results.append(f"{'✅' if internet_ok else '❌'} Internet: {'Connected' if internet_ok else 'No connection'}")
            
                # URL resolution check
                url_ok = self._test_url_resolution()
                results.append(f"{'✅' if url_ok else '❌'} URL Resolution: {'Working' if url_ok else 'Failed'}")
            
                # Performance
                results.append("\n⚡ PERFORMANCE")
                results.append("-" * 20)
                try:
                    perf_stats = self.performance_monitor.get_performance_stats()
                    results.append(f"⏱️ Uptime: {perf_stats['total_uptime']/60:.1f} min")
                    results.append(f"📝 Transcriptions: {perf_stats['transcriptions']}")
                    results.append(f"🌐 Translations: {perf_stats['translations']}")
                except:
                    results.append("📊 Performance stats: Unavailable")
            
                # Summary
                results.append("\n🎯 SUMMARY")
                results.append("-" * 20)
                system_ok = all_deps_ok and internet_ok and url_ok
                if system_ok:
                    results.append("✅ SYSTEM STATUS: EXCELLENT")
                    status_color = ModernColors.SUCCESS
                    status_text = "✅ System is fully operational"
                else:
                    results.append("⚠️ SYSTEM STATUS: DEGRADED")
                    status_color = ModernColors.WARNING
                    status_text = "⚠️ Some issues detected - check details above"
            
                final_text = "\n".join(results)
                self.recovery_log_text.delete('1.0', 'end')
                self.recovery_log_text.insert('1.0', final_text)
                self.diagnosis_results_label.config(text=status_text, fg=status_color)
            
            except Exception as e:
                self.diagnosis_results_label.config(
                    text=f"❌ Diagnosis failed: {str(e)}", 
                    fg=ModernColors.ERROR
                )
    
        diagnosis_thread = threading.Thread(target=diagnosis_worker, daemon=True)
        diagnosis_thread.start()

    def _run_quick_check(self):
        """Quick system check"""
        def quick_check_worker():
            try:
                self.diagnosis_results_label.config(
                    text="⚡ Running quick system check...", 
                    fg=ModernColors.WARNING
                )
            
                results = ["⚡ QUICK SYSTEM CHECK", "=" * 30]
            
                critical_checks = [
                    ("FFmpeg", shutil.which('ffmpeg')),
                    ("yt-dlp", shutil.which('yt-dlp')),
                    ("Whisper", WHISPER_AVAILABLE),
                    ("PyTorch", TORCH_AVAILABLE)
                ]
            
                all_ok = True
                for name, available in critical_checks:
                    status = "✅" if available else "❌"
                    results.append(f"{status} {name}")
                    if not available:
                        all_ok = False
            
                results.append("=" * 30)
                if all_ok:
                    results.append("✅ READY - All critical components available")
                    status_color = ModernColors.SUCCESS
                    status_text = "✅ System ready for operation"
                else:
                    results.append("❌ BLOCKED - Missing critical components")
                    status_color = ModernColors.ERROR
                    status_text = "❌ Critical components missing"
            
                final_text = "\n".join(results)
                self.recovery_log_text.delete('1.0', 'end')
                self.recovery_log_text.insert('1.0', final_text)
                self.diagnosis_results_label.config(text=status_text, fg=status_color)
            
            except Exception as e:
                self.diagnosis_results_label.config(
                    text=f"❌ Quick check failed: {str(e)}", 
                    fg=ModernColors.ERROR
                )
    
        quick_thread = threading.Thread(target=quick_check_worker, daemon=True)
        quick_thread.start()

    def _check_internet_connection(self):
        """Check internet connection"""
        try:
            import requests
            requests.get('https://www.google.com', timeout=5)
            return True
        except:
            return False

    def _test_url_resolution(self):
        """Test URL resolution with yt-dlp"""
        try:
            result = subprocess.run(
                ['yt-dlp', '-g', '--no-warnings', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'],
                capture_output=True, timeout=15, text=True
            )
            return result.returncode == 0 and result.stdout.strip()
        except:
            return False
    
    def _update_health_tab(self, health_data):
        """Update health tab with safe exception handling."""
        try:
            status = health_data.get('status', 'unknown').upper()
            status_color = ModernColors.SUCCESS if status == 'HEALTHY' else \
                         ModernColors.WARNING if status == 'DEGRADED' else ModernColors.ERROR
            
            self.health_status_label.config(
                text=f"Status: {status}", 
                fg=status_color
            )

            for item in self.health_tree.get_children():
                self.health_tree.delete(item)

            metrics = [
                ("CPU Usage", f"{health_data.get('system', {}).get('cpu_percent', 0)}%", 
                 "Good" if health_data.get('system', {}).get('cpu_percent', 0) < 80 else "High"),
                ("Memory Usage", f"{health_data.get('system', {}).get('memory_percent', 0)}%",
                 "Good" if health_data.get('system', {}).get('memory_percent', 0) < 85 else "High"),
                ("Disk Usage", f"{health_data.get('system', {}).get('disk_usage_percent', 0)}%",
                 "Good" if health_data.get('system', {}).get('disk_usage_percent', 0) < 90 else "High"),
                ("Transcription Engine", health_data.get('services', {}).get('transcription_engine', {}).get('status', 'unknown'),
                 "Available" if health_data.get('services', {}).get('transcription_engine', {}).get('status') == 'available' else "Degraded"),
                ("Uptime", f"{health_data.get('process', {}).get('uptime_seconds', 0)/3600:.1f} hours", "Stable")
            ]

            for metric, value, status in metrics:
                self.health_tree.insert('', 'end', values=(metric, value, status))

        except Exception as e:
            logging.error(f"Health tab update error: {e}")

    def _update_analytics_tab(self, analytics_data):
        """Update analytics tab with safe exception handling."""
        try:
            health_score = analytics_data.get('health_summary', {}).get('system_health_score', 0)
            overview_text = f"Health Score: {health_score}/100 | "
            overview_text += f"Throughput: {analytics_data.get('performance_analytics', {}).get('throughput_metrics', {}).get('transcriptions_per_hour', 0):.1f} tph | "
            overview_text += f"Reliability: {analytics_data.get('business_intelligence', {}).get('service_reliability', {}).get('reliability_score', 0)}%"
            
            self.analytics_overview_label.config(text=overview_text)

            self.analytics_text.delete('1.0', 'end')
            self.analytics_text.insert('1.0', json.dumps(analytics_data, indent=2, ensure_ascii=False))

        except Exception as e:
            logging.error(f"Analytics tab update error: {e}")

    def _update_recovery_tab(self):
        """Update recovery tab with safe exception handling."""
        try:
            handler_count = len(getattr(ExcellenceSignalHandler, '_recovery_handlers', []))
            self.recovery_status_label.config(
                text=f"Auto-Recovery: {handler_count} handlers registered | System: Ready"
            )

        except Exception as e:
            logging.error(f"Recovery tab update error: {e}")

    def _trigger_dashboard_recovery(self):
        """Trigger recovery from dashboard with safe exception handling."""
        try:
            self.recovery_log_text.insert('end', f"\n[{datetime.now().strftime('%H:%M:%S')}] Manual recovery triggered...\n")
            self.recovery_log_text.see('end')
            
            success = ExcellenceSignalHandler.trigger_emergency_recovery()
            
            if success:
                self.recovery_log_text.insert('end', "✅ Recovery completed successfully\n")
            else:
                self.recovery_log_text.insert('end', "❌ Recovery failed or partially completed\n")
                
            self.recovery_log_text.see('end')

        except Exception as e:
            self.recovery_log_text.insert('end', f"❌ Recovery error: {str(e)}\n")
            self.recovery_log_text.see('end')

    def excellence_cleanup(self):
        """Perform excellence cleanup with resource management."""
        self.is_processing = False
        if hasattr(self, 'audio_processor'):
            self.audio_processor.stop_processing()
        if hasattr(self, 'ffmpeg_manager'):
            self.ffmpeg_manager.dispose()
        if hasattr(self, 'resource_manager'):
            self.resource_manager.cleanup()
        if hasattr(self, 'memory_manager'):
            self.memory_manager.clear_component('transcript')
            self.memory_manager.clear_component('translation')

        self._last_transcription_text = ""
        self._last_translation_text = ""

    def quick_test_fix(self):
        """Quick test of repairs."""
        print("\n🔧 RUNNING QUICK TEST FIX...")
        
        try:
            import numpy as np
            
            duration = 2
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            test_audio = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16).tobytes()
            
            print(f"🎵 Simple test audio: {len(test_audio)} bytes")
            
            result = self.transcription_engine.transcribe_audio(test_audio)
            
            if result and result.text:
                print(f"✅ QUICK TEST PASSED: '{result.text}'")
                return True
            else:
                print("❌ QUICK TEST FAILED - Using debug method...")
                debug_result = self.transcription_engine.debug_transcribe_audio(test_audio)
                if debug_result:
                    print(f"🔍 DEBUG: {debug_result}")
                return False
                
        except Exception as e:
            print(f"❌ QUICK TEST ERROR: {e}")
            return False

# === UTILITY FUNCTIONS ===
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division function to prevent DivisionByZero errors
    """
    if denominator == 0:
        return default
    return numerator / denominator

def main():
    """Dragon Whisperer - Optimized silent version"""
    print("🐉 DRAGON WHISPERER")
    
    logging.basicConfig(
        level=logging.ERROR,
        format='%(levelname)s: %(message)s',        
    )

    noisy_loggers = [
        "faster_whisper", "deep_translator", "torch", "urllib3", 
        "pynvml", "nvidia", "httpx", "httpcore", "asyncio", "psutil"
    ]
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

    if not GUI_AVAILABLE:
        print("❌ GUI not available")
        return

    missing_deps = []
    if not WHISPER_AVAILABLE:
        missing_deps.append("faster-whisper")
    if not TRANSLATOR_AVAILABLE:
        missing_deps.append("deep-translator")
    if not NUMPY_AVAILABLE:
        missing_deps.append("numpy")
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")

    if missing_deps:
        print(f"❌ Missing: {', '.join(missing_deps)}")
        return

    try:
        print("🚀 Starting GUI...")
        
        app = DragonWhispererGUI()
        app.run()

    except Exception as e:
        print(f"❌ {e}")

if __name__ == "__main__":
    main()
