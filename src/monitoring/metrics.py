"""
Metrics collection and monitoring for the sentiment analysis API.
"""

import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)


class MetricsCollector:
    """
    Collects and manages application metrics.
    """

    def __init__(self):
        # Thread-safe counters
        self._lock = threading.Lock()

        # Basic counters
        self.request_count = 0
        self.prediction_count = 0
        self.batch_prediction_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.error_count = 0

        # Response time tracking
        self.request_durations = []
        self.max_request_duration = 0.0
        self.min_request_duration = float('inf')
        self.total_request_duration = 0.0

        # Prometheus metrics
        self.setup_prometheus_metrics()

        # Start time for uptime calculation
        self.start_time = time.time()

    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # Counters
        self.prom_requests_total = Counter(
            'sentiment_api_requests_total',
            'Total number of API requests',
            ['endpoint', 'method', 'status']
        )

        self.prom_predictions_total = Counter(
            'sentiment_api_predictions_total',
            'Total number of sentiment predictions',
            ['type']  # single or batch
        )

        self.prom_cache_operations_total = Counter(
            'sentiment_api_cache_operations_total',
            'Total number of cache operations',
            ['operation']  # hit or miss
        )

        self.prom_errors_total = Counter(
            'sentiment_api_errors_total',
            'Total number of errors',
            ['type']
        )

        # Histograms
        self.prom_request_duration = Histogram(
            'sentiment_api_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

        # Gauges
        self.prom_active_requests = Gauge(
            'sentiment_api_active_requests',
            'Number of active requests'
        )

        self.prom_cache_hit_ratio = Gauge(
            'sentiment_api_cache_hit_ratio',
            'Cache hit ratio'
        )

        self.prom_model_loaded = Gauge(
            'sentiment_api_model_loaded',
            'Whether the model is loaded (1) or not (0)'
        )

    def record_model_latency(self, latency: float):
        """Record model prediction latency in seconds."""
        with self._lock:
            # You could add this to your Prometheus metrics if needed
            # For now we'll just track it internally
            if not hasattr(self, 'model_latencies'):
                self.model_latencies = []
            self.model_latencies.append(latency)

            # Track min/max/avg model latency
            if not hasattr(self, 'max_model_latency'):
                self.max_model_latency = 0.0
                self.min_model_latency = float('inf')
                self.total_model_latency = 0.0

            if latency > self.max_model_latency:
                self.max_model_latency = latency

            if latency < self.min_model_latency:
                self.min_model_latency = latency

            self.total_model_latency += latency

    def record_request_duration(self, duration: float, endpoint: str = "default"):
        """Record request duration."""
        with self._lock:
            self.request_count += 1
            self.request_durations.append(duration)
            self.total_request_duration += duration

            if duration > self.max_request_duration:
                self.max_request_duration = duration

            if duration < self.min_request_duration:
                self.min_request_duration = duration

            # Update Prometheus metrics
            self.prom_request_duration.labels(endpoint=endpoint).observe(duration)

    def increment_prediction_count(self):
        """Increment prediction count."""
        with self._lock:
            self.prediction_count += 1
            self.prom_predictions_total.labels(type="single").inc()

    def increment_batch_prediction_count(self):
        """Increment batch prediction count."""
        with self._lock:
            self.batch_prediction_count += 1
            self.prom_predictions_total.labels(type="batch").inc()

    def increment_cache_hit_count(self, count: int = 1):
        """Increment cache hit count."""
        with self._lock:
            self.cache_hit_count += count
            self.prom_cache_operations_total.labels(operation="hit").inc(count)
            self._update_cache_hit_ratio()

    def increment_cache_miss_count(self, count: int = 1):
        """Increment cache miss count."""
        with self._lock:
            self.cache_miss_count += count
            self.prom_cache_operations_total.labels(operation="miss").inc(count)
            self._update_cache_hit_ratio()

    def increment_error_count(self, error_type: str = "general"):
        """Increment error count."""
        with self._lock:
            self.error_count += 1
            self.prom_errors_total.labels(type=error_type).inc()

    def _update_cache_hit_ratio(self):
        """Update cache hit ratio gauge."""
        total_cache_operations = self.cache_hit_count + self.cache_miss_count
        if total_cache_operations > 0:
            hit_ratio = self.cache_hit_count / total_cache_operations
            self.prom_cache_hit_ratio.set(hit_ratio)

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            # Calculate averages
            avg_request_duration = 0.0
            if self.request_count > 0:
                avg_request_duration = self.total_request_duration / self.request_count

            avg_model_latency = 0.0
            if hasattr(self, 'model_latencies') and self.model_latencies:
                avg_model_latency = self.total_model_latency / len(self.model_latencies)

            # Calculate cache hit ratio
            cache_hit_ratio = 0.0
            total_cache_operations = self.cache_hit_count + self.cache_miss_count
            if total_cache_operations > 0:
                cache_hit_ratio = self.cache_hit_count / total_cache_operations

            # Calculate uptime
            uptime = time.time() - self.start_time

            return {
                "requests": {
                    "total": self.request_count,
                    "predictions": self.prediction_count,
                    "batch_predictions": self.batch_prediction_count,
                    "errors": self.error_count
                },
                "performance": {
                    "avg_request_duration": avg_request_duration,
                    "max_request_duration": self.max_request_duration,
                    "min_request_duration": self.min_request_duration if self.min_request_duration != float('inf') else 0.0,
                    "total_request_duration": self.total_request_duration,
                    "model_latency": {
                        "avg": avg_model_latency,
                        "max": getattr(self, 'max_model_latency', 0.0),
                        "min": getattr(self, 'min_model_latency', 0.0),
                        "total": getattr(self, 'total_model_latency', 0.0),
                        "count": len(getattr(self, 'model_latencies', []))
                    }
                },
                "cache": {
                    "hits": self.cache_hit_count,
                    "misses": self.cache_miss_count,
                    "hit_ratio": cache_hit_ratio,
                    "total_operations": total_cache_operations
                },
                "system": {
                    "uptime": uptime,
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat()
                }
            }

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics."""
        return generate_latest()

    def start_prometheus_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        try:
            start_http_server(port)
            print(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            print(f"Failed to start Prometheus metrics server: {e}")

    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.request_count = 0
            self.prediction_count = 0
            self.batch_prediction_count = 0
            self.cache_hit_count = 0
            self.cache_miss_count = 0
            self.error_count = 0
            self.request_durations = []
            self.max_request_duration = 0.0
            self.min_request_duration = float('inf')
            self.total_request_duration = 0.0

            # Reset model latency metrics
            if hasattr(self, 'model_latencies'):
                self.model_latencies = []
                self.max_model_latency = 0.0
                self.min_model_latency = float('inf')
                self.total_model_latency = 0.0

            self.start_time = time.time()


class DataDriftDetector:
    """
    Simple data drift detection for text inputs.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.text_lengths = []
        self.char_distributions = []
        self.baseline_stats = None
        self._lock = threading.Lock()

    def add_text(self, text: str):
        """Add text for drift detection."""
        with self._lock:
            # Track text length
            self.text_lengths.append(len(text))

            # Track character distribution
            char_counts = defaultdict(int)
            for char in text.lower():
                char_counts[char] += 1

            # Normalize by text length
            total_chars = len(text)
            if total_chars > 0:
                char_freq = {char: count / total_chars for char, count in char_counts.items()}
                self.char_distributions.append(char_freq)

            # Keep only recent data
            if len(self.text_lengths) > self.window_size:
                self.text_lengths.pop(0)
                self.char_distributions.pop(0)

    def calculate_drift_score(self) -> dict[str, float]:
        """Calculate drift score compared to baseline."""
        with self._lock:
            if len(self.text_lengths) < 100:  # Need minimum data
                return {
                    "length_drift": 0.0,
                    "char_drift": 0.0,
                    "overall_drift": 0.0,
                    "sample_size": len(self.text_lengths)
                }

            # Initialize baseline if not set
            if self.baseline_stats is None:
                self.set_baseline()

            # Calculate current statistics
            current_avg_length = sum(self.text_lengths) / len(self.text_lengths)
            current_length_std = 0.0
            if len(self.text_lengths) > 1:
                variance = sum((x - current_avg_length) ** 2 for x in self.text_lengths) / (len(self.text_lengths) - 1)
                current_length_std = variance ** 0.5

            # Length drift score
            length_drift = abs(current_avg_length - self.baseline_stats["avg_length"]) / max(self.baseline_stats["length_std"], 1.0)

            # Character distribution drift (simplified)
            char_drift = 0.0
            if self.char_distributions:
                # Get most recent distribution
                recent_dist = self.char_distributions[-1]
                baseline_dist = self.baseline_stats["char_dist"]

                # Calculate KL divergence (simplified)
                common_chars = set(recent_dist.keys()) & set(baseline_dist.keys())
                if common_chars:
                    kl_div = 0.0
                    for char in common_chars:
                        p = recent_dist[char]
                        q = baseline_dist.get(char, 0.001)  # Small epsilon to avoid division by zero
                        if p > 0 and q > 0:
                            kl_div += p * (p / q)
                    char_drift = kl_div

            # Overall drift score
            overall_drift = (length_drift + char_drift) / 2

            return {
                "length_drift": length_drift,
                "char_drift": char_drift,
                "overall_drift": overall_drift,
                "sample_size": len(self.text_lengths),
                "current_avg_length": current_avg_length,
                "baseline_avg_length": self.baseline_stats["avg_length"]
            }

    def set_baseline(self):
        """Set baseline statistics from current data."""
        with self._lock:
            if not self.text_lengths:
                return

            # Calculate baseline statistics
            avg_length = sum(self.text_lengths) / len(self.text_lengths)
            length_std = 0.0
            if len(self.text_lengths) > 1:
                variance = sum((x - avg_length) ** 2 for x in self.text_lengths) / (len(self.text_lengths) - 1)
                length_std = variance ** 0.5

            # Average character distribution
            char_dist = defaultdict(float)
            for dist in self.char_distributions:
                for char, freq in dist.items():
                    char_dist[char] += freq

            # Normalize by number of samples
            if self.char_distributions:
                for char in char_dist:
                    char_dist[char] /= len(self.char_distributions)

            self.baseline_stats = {
                "avg_length": avg_length,
                "length_std": max(length_std, 1.0),  # Avoid division by zero
                "char_dist": dict(char_dist)
            }

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            return {
                "window_size": self.window_size,
                "current_sample_size": len(self.text_lengths),
                "baseline_set": self.baseline_stats is not None,
                "drift_scores": self.calculate_drift_score()
            }
