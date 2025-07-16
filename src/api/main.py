"""
FastAPI application for real-time sentiment analysis.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from src.api.cache import SentimentCache
from src.api.schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    ErrorResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
)
from src.config import settings
from src.model import ModelManager, SentimentAnalyzer
from src.monitoring.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for app components
model_manager: ModelManager | None = None
sentiment_analyzer: SentimentAnalyzer | None = None
cache: SentimentCache | None = None
metrics_collector: MetricsCollector | None = None
app_start_time: float = 0

# Rate limiting storage
_rate_limit_storage: dict[str, dict[str, Any]] = {}

async def check_rate_limit(request: Request, times: int = 100, minutes: int = 1):
    """Simple in-memory rate limiting implementation"""
    client_ip = request.client.host
    current_window = datetime.now().replace(second=0, microsecond=0)
    window_key = f"{client_ip}:{current_window}"

    if window_key not in _rate_limit_storage:
        _rate_limit_storage[window_key] = {
            "count": 0,
            "window_start": current_window
        }

    _rate_limit_storage[window_key]["count"] += 1

    # Clean up old entries
    for key in list(_rate_limit_storage.keys()):
        if _rate_limit_storage[key]["window_start"] < current_window - timedelta(minutes=5):
            del _rate_limit_storage[key]

    if _rate_limit_storage[window_key]["count"] > times:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {times} requests per {minutes} minute(s)"
        )

async def initialize_components():
    """Initialize all application components."""
    global model_manager, sentiment_analyzer, cache, metrics_collector, app_start_time

    try:
        logger.info("Initializing application components...")
        app_start_time = time.time()

        # Initialize metrics collector first
        metrics_collector = MetricsCollector()
        logger.info("Metrics collector initialized")

        # Initialize model manager and load model
        model_manager = ModelManager()
        sentiment_analyzer = model_manager.load_model(settings.model_name)
        logger.info("Model loaded successfully")

        # Warm up model
        warmup_samples = ["warmup", "test", "prediction"]
        _ = sentiment_analyzer.predict_batch(warmup_samples)
        logger.info("Model warmup completed")

        # Initialize cache
        logger.debug(f"Redis URL from settings: {settings.redis_url}")
        cache = SentimentCache(redis_url=settings.redis_url)
        try:
            logger.debug("About to call cache.connect()")
            connection_success = await cache.connect()
            logger.debug(f"Cache connect returned: {connection_success}")
            if connection_success:
                logger.info("Cache connected successfully")
            else:
                logger.warning("Cache connection failed")
                cache = None
        except Exception as e:
            logger.error(f"Cache connection failed: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            cache = None

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

async def cleanup_components():
    """Cleanup application components."""
    global cache

    try:
        logger.info("Cleaning up application components...")

        if cache:
            await cache.disconnect()
            logger.info("Cache disconnected")

        logger.info("Cleanup completed successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await initialize_components()
    yield
    # Shutdown
    await cleanup_components()

# Create FastAPI app
app = FastAPI(
    title="Real-time Sentiment Analysis API",
    description="Production-ready sentiment analysis API with monitoring and caching",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Dependency functions
async def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get sentiment analyzer instance."""
    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analyzer not initialized"
        )
    return sentiment_analyzer

async def get_cache() -> SentimentCache | None:
    """Get cache instance."""
    return cache

async def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance."""
    if metrics_collector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics collector not initialized"
        )
    return metrics_collector

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")

    if metrics_collector:
        metrics_collector.increment_error_count()

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if app.debug else "An unexpected error occurred"
        ).dict()
    )

# API Routes
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(
    request: Request,
    sentiment_request: SentimentRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer),
    cache_client: SentimentCache | None = Depends(get_cache),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Predict sentiment for a single text.
    
    - **text**: The text to analyze (max 5000 characters)
    
    Returns sentiment prediction with confidence score.
    """
    # Apply rate limiting
    await check_rate_limit(request, times=100, minutes=1)

    start_time = time.time()
    cached = False
    request_id = request.headers.get("X-Request-ID", "")
    logger.info(f"Predict request started - Request ID: {request_id}")

    try:
        # Check cache first
        cached_result = None
        if cache_client and cache_client.connected:
            cached_result = await cache_client.get(sentiment_request.text)

        if cached_result:
            cached = True
            result = cached_result
            logger.debug(f"Cache hit for text: {sentiment_request.text[:50]}...")
        else:
            # Predict sentiment
            result = analyzer.predict(sentiment_request.text)
            metrics.record_model_latency(time.time() - start_time)

            # Cache result if cache is available
            if cache_client and cache_client.connected:
                await cache_client.set(sentiment_request.text, result)

        processing_time = time.time() - start_time

        # Record metrics
        metrics.record_request_duration(processing_time)
        metrics.increment_prediction_count()
        if cached:
            metrics.increment_cache_hit_count()
        else:
            metrics.increment_cache_miss_count()

        # Create response
        response = SentimentResponse(
            text=sentiment_request.text,
            label=result["label"],
            score=result["score"],
            confidence=result.get("confidence", result["score"]),
            processing_time=processing_time,
            cached=cached,
            request_id=request_id
        )

        return response

    except Exception as e:
        logger.error(f"Error in predict_sentiment: {str(e)} - Request ID: {request_id}")
        metrics.increment_error_count()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing sentiment analysis: {str(e)}"
        )

@app.post("/batch_predict", response_model=BatchSentimentResponse)
async def batch_predict_sentiment(
    request: Request,
    batch_request: BatchSentimentRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer),
    cache_client: SentimentCache | None = Depends(get_cache),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Predict sentiment for multiple texts.
    
    - **texts**: List of texts to analyze (max 100 texts, each max 5000 characters)
    
    Returns batch sentiment predictions with confidence scores.
    """
    # Apply rate limiting (more strict for batch)
    await check_rate_limit(request, times=10, minutes=1)

    start_time = time.time()
    cached_count = 0
    request_id = request.headers.get("X-Request-ID", "")
    logger.info(f"Batch predict request started - Request ID: {request_id}")

    try:
        # Validate batch size
        if len(batch_request.texts) > settings.batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum batch size is {settings.batch_size}"
            )

        # Check cache for all texts
        cached_results = {}
        texts_to_predict = []

        if cache_client and cache_client.connected:
            cached_results = await cache_client.get_batch(batch_request.texts)
            texts_to_predict = [text for text in batch_request.texts if cached_results.get(text) is None]
        else:
            texts_to_predict = batch_request.texts

        # Predict sentiment for non-cached texts
        new_results = {}
        if texts_to_predict:
            batch_start = time.time()
            predictions = analyzer.predict_batch(texts_to_predict)
            metrics.record_model_latency(time.time() - batch_start)
            new_results = dict(zip(texts_to_predict, predictions, strict=False))

            # Cache new results
            if cache_client and cache_client.connected:
                await cache_client.set_batch(new_results)

        # Combine cached and new results
        all_results = []
        for text in batch_request.texts:
            if text in cached_results and cached_results[text] is not None:
                result = cached_results[text]
                cached_count += 1
                logger.debug(f"Cache hit for text: {text[:50]}...")
            else:
                result = new_results[text]
                logger.debug(f"Cache miss for text: {text[:50]}...")

            all_results.append(SentimentResponse(
                text=text,
                label=result["label"],
                score=result["score"],
                confidence=result.get("confidence", result["score"]),
                processing_time=0,  # Individual timing not available in batch
                cached=text in cached_results and cached_results[text] is not None,
                request_id=request_id
            ))

        processing_time = time.time() - start_time

        # Record metrics
        metrics.record_request_duration(processing_time)
        metrics.increment_batch_prediction_count()
        metrics.increment_cache_hit_count(cached_count)
        metrics.increment_cache_miss_count(len(batch_request.texts) - cached_count)

        # Create response
        response = BatchSentimentResponse(
            results=all_results,
            total_count=len(batch_request.texts),
            processing_time=processing_time,
            cached_count=cached_count,
            request_id=request_id
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch_predict_sentiment: {str(e)} - Request ID: {request_id}")
        metrics.increment_error_count()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch sentiment analysis: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        # Basic status info
        status_info = {
            "status": "healthy",
            "uptime": time.time() - app_start_time,
            "version": "1.0.0"
        }

        # Model status
        model_status = {
            "loaded": sentiment_analyzer is not None,
            "info": {}
        }
        if sentiment_analyzer:
            model_status["info"] = {
                "model_name": getattr(sentiment_analyzer, "model_name", "unknown"),
                "device": str(getattr(sentiment_analyzer, "device", "unknown"))
            }
        status_info["model"] = model_status

        # Cache status
        cache_status = {"connected": False}
        if cache:
            try:
                logger.debug(f"Cache object exists: {cache}")
                logger.debug(f"Cache connected attribute: {getattr(cache, 'connected', 'NOT_FOUND')}")
                logger.debug("About to call cache.health_check()")
                cache_health = await cache.health_check()
                logger.debug(f"Cache health_check returned: {cache_health}")
                logger.debug(f"Cache health_check type: {type(cache_health)}")
                cache_status = cache_health
                logger.debug(f"Final cache_status: {cache_status}")
            except Exception as e:
                logger.error(f"Exception in cache health check: {str(e)}")
                logger.error(f"Exception type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                cache_status = {
                    "connected": False,
                    "error": str(e)
                }
        else:
            logger.debug("Cache object is None")
        status_info["cache"] = cache_status

        # Metrics status
        status_info["metrics"] = {
            "available": metrics_collector is not None
        }

        return HealthResponse(**status_info)

    except Exception as e:
        logger.error(f"Health check failed completely: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": "Service unavailable",
                "detail": str(e)
            }
        )


@app.get("/metrics")
async def get_metrics(
    metrics: MetricsCollector = Depends(get_metrics_collector),
    cache_client: SentimentCache | None = Depends(get_cache)
):
    """
    Get application metrics.
    
    Returns various application metrics including cache statistics.
    """
    try:
        response = {}

        # Get application metrics
        if metrics:
            response["app_metrics"] = metrics.get_metrics()

        # Get cache statistics
        if cache_client:
            response["cache_stats"] = await cache_client.get_stats()

        return response

    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving metrics: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Real-time Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        server_header=False
    )

if __name__ == "__main__":
    main()
