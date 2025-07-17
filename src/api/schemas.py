"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SentimentRequest(BaseModel):
    """Request model for single sentiment analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class SentimentResponse(BaseModel):
    """Response model for single sentiment analysis."""

    text: str = Field(..., description="Original text")
    label: str = Field(..., description="Predicted sentiment label")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    cached: bool = Field(default=False, description="Whether result was cached")
    request_id: str = Field(default="", description="Request ID for tracking")


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    texts: list[str] = Field(
        ..., min_length=1, max_length=100, description="List of texts to analyze"
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")

        validated_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(
                    f"Text at index {i} cannot be empty or whitespace only"
                )
            if len(text.strip()) > 5000:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length of 5000 characters"
                )
            validated_texts.append(text.strip())

        return validated_texts


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis."""

    results: list[SentimentResponse] = Field(
        ..., description="List of sentiment analysis results"
    )
    total_count: int = Field(..., description="Total number of texts processed")
    processing_time: float = Field(..., description="Total processing time in seconds")
    cached_count: int = Field(default=0, description="Number of cached results")
    request_id: str = Field(default="", description="Request ID for tracking")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Overall service status")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="Service version")
    model: dict = Field(..., description="Model status information")
    cache: dict = Field(..., description="Cache status information")
    metrics: dict = Field(..., description="Metrics availability")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
    request_id: str | None = Field(None, description="Request ID for tracking")
