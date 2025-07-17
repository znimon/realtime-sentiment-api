"""
Integration tests for the API endpoints.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_components():
    """Mock application components."""
    with (
        patch("src.api.main.sentiment_analyzer") as mock_analyzer,
        patch("src.api.main.cache") as mock_cache,
        patch("src.api.main.metrics_collector") as mock_metrics,
    ):
        # Setup mock analyzer
        mock_analyzer.predict.return_value = {
            "label": "POSITIVE",
            "score": 0.85,
            "confidence": 0.85,
        }
        mock_analyzer.predict_batch.return_value = [
            {"label": "POSITIVE", "score": 0.85, "confidence": 0.85},
            {"label": "NEGATIVE", "score": 0.90, "confidence": 0.90},
        ]
        mock_analyzer.model_name = "test_model"
        mock_analyzer.device = "cpu"

        # Setup mock cache make async methods return coroutines
        mock_cache.connected = True

        async def mock_get(key):
            return None

        async def mock_set(key, value):
            return True

        async def mock_get_batch(keys):
            return {}

        async def mock_set_batch(data):
            return 0

        async def mock_health_check():
            return {"connected": True}

        async def mock_get_stats():
            return {"connected": True, "total_keys": 0, "sentiment_keys": 0}

        mock_cache.get = mock_get
        mock_cache.set = mock_set
        mock_cache.get_batch = mock_get_batch
        mock_cache.set_batch = mock_set_batch
        mock_cache.health_check = mock_health_check
        mock_cache.get_stats = mock_get_stats

        # Setup mock metrics
        mock_metrics.get_metrics.return_value = {
            "requests": {"total": 0, "errors": 0},
            "performance": {"avg_request_duration": 0.0},
            "cache": {"hits": 0, "misses": 0, "hit_ratio": 0.0},
            "system": {"uptime": 0.0},
        }
        mock_metrics.record_request_duration.return_value = None
        mock_metrics.increment_prediction_count.return_value = None
        mock_metrics.increment_batch_prediction_count.return_value = None
        mock_metrics.increment_cache_hit_count.return_value = None
        mock_metrics.increment_cache_miss_count.return_value = None
        mock_metrics.record_model_latency.return_value = None

        yield {"analyzer": mock_analyzer, "cache": mock_cache, "metrics": mock_metrics}


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Real-time Sentiment Analysis API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data
        assert "metrics" in data


class TestHealthEndpoint:
    """Test health endpoint."""

    def test_health_check(self, client, mock_components):
        """Test health check endpoint."""
        with (
            patch("src.api.main.app_start_time", 1000.0),
            patch("time.time", return_value=1100.0),
        ):
            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert data["model"]["loaded"] is True
            assert data["uptime"] == 100.0
            assert data["version"] == "1.0.0"
            assert "model" in data
            assert "cache" in data
            assert "metrics" in data


class TestPredictEndpoint:
    """Test sentiment prediction endpoint."""

    def test_predict_success(self, client, mock_components):
        """Test successful prediction."""
        request_data = {"text": "I love this product!"}

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["text"] == "I love this product!"
        assert data["label"] == "POSITIVE"
        assert data["score"] == 0.85
        assert data["confidence"] == 0.85
        assert "processing_time" in data
        assert "cached" in data

        # Verify mock calls
        mock_components["analyzer"].predict.assert_called_once_with(
            "I love this product!"
        )
        mock_components["metrics"].record_request_duration.assert_called_once()
        mock_components["metrics"].increment_prediction_count.assert_called_once()

    def test_predict_empty_text(self, client, mock_components):
        """Test prediction with empty text."""
        request_data = {"text": ""}

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_text_too_long(self, client, mock_components):
        """Test prediction with text too long."""
        request_data = {"text": "a" * 5001}  # Exceeds max length

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_cached_result(self, client, mock_components):
        """Test prediction with cached result."""
        # Setup cache to return result
        cached_result = {"label": "POSITIVE", "score": 0.85, "confidence": 0.85}

        async def mock_get_cached(key):
            return cached_result

        mock_components["cache"].get = mock_get_cached

        request_data = {"text": "I love this product!"}

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["cached"] is True
        assert data["label"] == "POSITIVE"

        # Verify model wasn't called (cache hit)
        mock_components["analyzer"].predict.assert_not_called()


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint."""

    def test_batch_predict_success(self, client, mock_components):
        """Test successful batch prediction."""
        request_data = {"texts": ["I love this!", "I hate this!"]}

        response = client.post("/batch_predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["results"]) == 2
        assert data["total_count"] == 2
        assert "processing_time" in data
        assert "cached_count" in data

        # Check individual results
        assert data["results"][0]["label"] == "POSITIVE"
        assert data["results"][1]["label"] == "NEGATIVE"

        # Verify mock calls
        mock_components["analyzer"].predict_batch.assert_called_once()
        mock_components["metrics"].increment_batch_prediction_count.assert_called_once()

    def test_batch_predict_empty_list(self, client, mock_components):
        """Test batch prediction with empty list."""
        request_data = {"texts": []}

        response = client.post("/batch_predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_too_many_texts(self, client, mock_components):
        """Test batch prediction with too many texts."""
        request_data = {"texts": ["text"] * 101}  # Exceeds max items

        response = client.post("/batch_predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_with_cache(self, client, mock_components):
        """Test batch prediction with some cached results."""
        # Setup cache to return some results
        cached_results = {
            "I love this!": {"label": "POSITIVE", "score": 0.85, "confidence": 0.85},
            "I hate this!": None,
        }

        async def mock_get_batch_cached(keys):
            return cached_results

        mock_components["cache"].get_batch = mock_get_batch_cached

        request_data = {"texts": ["I love this!", "I hate this!"]}

        response = client.post("/batch_predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["cached_count"] == 1
        assert len(data["results"]) == 2

        # First result should be cached
        assert data["results"][0]["cached"] is True
        # Second result should not be cached
        assert data["results"][1]["cached"] is False


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_get_metrics(self, client, mock_components):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "app_metrics" in data
        assert "cache_stats" in data

        # Verify mock calls
        mock_components["metrics"].get_metrics.assert_called_once()
