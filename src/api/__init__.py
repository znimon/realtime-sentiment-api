"""
API package for the sentiment analysis service.
"""

from .cache import SentimentCache
from .main import app
from .schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    ErrorResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
)

__all__ = [
    "app",
    "SentimentCache",
    "SentimentRequest",
    "SentimentResponse",
    "BatchSentimentRequest",
    "BatchSentimentResponse",
    "HealthResponse",
    "ErrorResponse",
]
