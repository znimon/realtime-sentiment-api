"""
Unit tests for sentiment model functionality.
"""

from unittest.mock import Mock, patch

import pytest

from src.model.sentiment_model import ModelManager, SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = SentimentAnalyzer()
        assert analyzer.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert analyzer.tokenizer is None
        assert analyzer.model is None
        assert analyzer.pipeline is None

    def test_init_custom_model(self):
        """Test analyzer initialization with custom model."""
        custom_model = "custom/model"
        analyzer = SentimentAnalyzer(model_name=custom_model)
        assert analyzer.model_name == custom_model

    @patch('src.model.sentiment_model.AutoTokenizer')
    @patch('src.model.sentiment_model.AutoModelForSequenceClassification')
    @patch('src.model.sentiment_model.pipeline')
    def test_load_model_success(self, mock_pipeline, mock_model_class, mock_tokenizer_class):
        """Test successful model loading."""
        # Mock components
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_pipe = Mock()

        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_pipeline.return_value = mock_pipe

        # Test loading
        analyzer = SentimentAnalyzer()
        analyzer.load_model()

        # Verify calls
        mock_tokenizer_class.from_pretrained.assert_called_once_with(analyzer.model_name)
        mock_model_class.from_pretrained.assert_called_once_with(analyzer.model_name)
        mock_model.to.assert_called_once_with(analyzer.device)
        mock_model.eval.assert_called_once()

        # Verify attributes
        assert analyzer.tokenizer == mock_tokenizer
        assert analyzer.model == mock_model
        assert analyzer.pipeline == mock_pipe

    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        analyzer = SentimentAnalyzer()

        with pytest.raises(ValueError, match="Model not loaded"):
            analyzer.predict("test text")

    def test_predict_empty_text(self):
        """Test prediction with empty text."""
        analyzer = SentimentAnalyzer()
        analyzer.pipeline = Mock()  # Mock pipeline

        result = analyzer.predict("")
        expected = {"label": "NEUTRAL", "score": 0.0}
        assert result == expected

        result = analyzer.predict("   ")
        assert result == expected

    def test_predict_success(self):
        """Test successful prediction."""
        analyzer = SentimentAnalyzer()

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "LABEL_2", "score": 0.85}]
        analyzer.pipeline = mock_pipeline

        result = analyzer.predict("I love this!")

        expected = {
            "label": "POSITIVE",
            "score": 0.85,
            "confidence": 0.85
        }
        assert result == expected
        mock_pipeline.assert_called_once_with("I love this!")

    def test_predict_batch_empty(self):
        """Test batch prediction with empty input."""
        analyzer = SentimentAnalyzer()
        analyzer.pipeline = Mock()

        result = analyzer.predict_batch([])
        assert result == []

    def test_predict_batch_success(self):
        """Test successful batch prediction."""
        analyzer = SentimentAnalyzer()

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "LABEL_2", "score": 0.85},
            {"label": "LABEL_0", "score": 0.90}
        ]
        analyzer.pipeline = mock_pipeline

        texts = ["I love this!", "I hate this!"]
        result = analyzer.predict_batch(texts)

        expected = [
            {"label": "POSITIVE", "score": 0.85, "confidence": 0.85},
            {"label": "NEGATIVE", "score": 0.90, "confidence": 0.90}
        ]
        assert result == expected
        mock_pipeline.assert_called_once_with(texts)

    def test_get_model_info(self):
        """Test model info retrieval."""
        analyzer = SentimentAnalyzer()

        # Mock components
        mock_tokenizer = Mock()
        mock_tokenizer.vocab = {"test": 1, "vocab": 2}
        mock_model = Mock()

        analyzer.tokenizer = mock_tokenizer
        analyzer.model = mock_model

        info = analyzer.get_model_info()

        assert info["model_name"] == analyzer.model_name
        assert info["tokenizer_vocab_size"] == 2
        assert info["model_loaded"] is True


class TestModelManager:
    """Test cases for ModelManager."""

    def test_init(self):
        """Test manager initialization."""
        manager = ModelManager()
        assert manager.models_dir.name == "models"
        assert manager.current_model is None
        assert manager.model_registry == {}

    def test_init_custom_dir(self):
        """Test manager initialization with custom directory."""
        manager = ModelManager("custom_models")
        assert manager.models_dir.name == "custom_models"

    def test_register_model(self):
        """Test model registration."""
        manager = ModelManager()

        manager.register_model("test_model", "v1.0", "/path/to/model")

        assert "test_model:v1.0" in manager.model_registry
        model_info = manager.model_registry["test_model:v1.0"]
        assert model_info["name"] == "test_model"
        assert model_info["version"] == "v1.0"
        assert model_info["path"] == "/path/to/model"

    @patch('src.model.sentiment_model.SentimentAnalyzer')
    def test_load_model_latest(self, mock_analyzer_class):
        """Test loading latest model."""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        manager = ModelManager()
        result = manager.load_model("test", "latest")

        mock_analyzer.load_model.assert_called_once()
        assert manager.current_model == mock_analyzer
        assert result == mock_analyzer

    def test_load_model_not_found(self):
        """Test loading non-existent model."""
        manager = ModelManager()

        with pytest.raises(ValueError, match="Model test:v1.0 not found"):
            manager.load_model("test", "v1.0")

    def test_get_current_model(self):
        """Test getting current model."""
        manager = ModelManager()

        # Initially None
        assert manager.get_current_model() is None

        # Set current model
        mock_model = Mock()
        manager.current_model = mock_model
        assert manager.get_current_model() == mock_model

    def test_list_models(self):
        """Test listing models."""
        manager = ModelManager()

        # Initially empty
        assert manager.list_models() == {}

        # Add some models
        manager.register_model("model1", "v1.0", "/path1")
        manager.register_model("model2", "v2.0", "/path2")

        models = manager.list_models()
        assert len(models) == 2
        assert "model1:v1.0" in models
        assert "model2:v2.0" in models
