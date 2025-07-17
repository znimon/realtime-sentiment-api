"""
Sentiment analysis model using pre-trained BERT model.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using cardiffnlp/twitter-roberta-base-sentiment-latest model.
    """

    def __init__(
        self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
            )

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, text: str) -> dict[str, float]:
        """
        Predict sentiment for a single text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}

        try:
            # Get prediction
            result = self.pipeline(text.strip())

            label = result[0]["label"]
            score = result[0]["score"]

            # Map labels
            label_mapping = {
                "LABEL_0": "NEGATIVE",
                "LABEL_1": "NEUTRAL",
                "LABEL_2": "POSITIVE",
            }

            standardized_label = label_mapping.get(label, label)

            return {
                "label": standardized_label,
                "score": float(score),
                "confidence": float(score),
            }

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.0, "error": str(e)}

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """
        Predict sentiment for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of dictionaries with sentiment scores
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if not texts:
            return []

        try:
            # Filter empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]

            if not valid_texts:
                return [{"label": "NEUTRAL", "score": 0.0} for _ in texts]

            # Get predictions
            results = self.pipeline(valid_texts)

            # Process results
            predictions = []
            for result in results:
                label = result["label"]
                score = result["score"]

                # Map labels to standardized format
                label_mapping = {
                    "LABEL_0": "NEGATIVE",
                    "LABEL_1": "NEUTRAL",
                    "LABEL_2": "POSITIVE",
                }

                standardized_label = label_mapping.get(label, label)

                predictions.append(
                    {
                        "label": standardized_label,
                        "score": float(score),
                        "confidence": float(score),
                    }
                )

            return predictions

        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            return [{"label": "NEUTRAL", "score": 0.0, "error": str(e)} for _ in texts]

    def get_model_info(self) -> dict[str, str]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "tokenizer_vocab_size": len(self.tokenizer.vocab) if self.tokenizer else 0,
            "model_loaded": self.model is not None,
        }


class ModelManager:
    """
    Manages model versions and loading.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.current_model = None
        self.model_registry = {}

    def register_model(self, name: str, version: str, model_path: str) -> None:
        """Register a model version."""
        self.model_registry[f"{name}:{version}"] = {
            "name": name,
            "version": version,
            "path": model_path,
            "registered_at": torch.utils.data.get_worker_info(),
        }

    def load_model(self, name: str, version: str = "latest") -> SentimentAnalyzer:
        """Load a specific model version."""
        if version == "latest":
            analyzer = SentimentAnalyzer()
            analyzer.load_model()
            self.current_model = analyzer
            return analyzer
        else:
            # Load from registry for fine-tuned models
            model_key = f"{name}:{version}"
            if model_key in self.model_registry:
                model_info = self.model_registry[model_key]
                analyzer = SentimentAnalyzer(model_info["path"])
                analyzer.load_model()
                self.current_model = analyzer
                return analyzer
            else:
                raise ValueError(f"Model {model_key} not found in registry")

    def get_current_model(self) -> SentimentAnalyzer | None:
        """Get the currently loaded model."""
        return self.current_model

    def list_models(self) -> dict[str, dict]:
        """List all registered models."""
        return self.model_registry
