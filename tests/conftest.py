"""Shared pytest fixtures for the colXLM project."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after test."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_text():
    """Sample text for testing tokenization and text processing."""
    return "This is a sample document for testing tokenization and retrieval."


@pytest.fixture
def sample_query():
    """Sample query for testing retrieval functionality."""
    return "sample query for testing"


@pytest.fixture
def sample_documents():
    """Sample document collection for testing."""
    return [
        "Document about artificial intelligence and machine learning.",
        "Information retrieval systems and search algorithms.",
        "Natural language processing and text analysis.",
        "Cross-lingual retrieval and multilingual systems."
    ]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing without loading actual models."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 2023, 2003, 1037, 3231, 102]  # Sample token IDs
    tokenizer.decode.return_value = "decoded text"
    tokenizer.vocab_size = 30522
    tokenizer.pad_token_id = 0
    tokenizer.cls_token_id = 101
    tokenizer.sep_token_id = 102
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock ColBERT model for testing without loading actual weights."""
    model = MagicMock()
    model.eval.return_value = model
    model.training = False
    
    # Mock forward pass
    def mock_forward(*args, **kwargs):
        batch_size = args[0].shape[0] if args else 1
        seq_len = args[0].shape[1] if args else 10
        return torch.randn(batch_size, seq_len, 128)  # Mock embeddings
    
    model.forward = mock_forward
    model.return_value = torch.randn(1, 10, 128)
    return model


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing retrieval functionality."""
    return torch.randn(5, 128)  # 5 embeddings of dimension 128


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index for testing without actual index operations."""
    index = MagicMock()
    index.ntotal = 100
    index.d = 128
    index.is_trained = True
    
    # Mock search results
    def mock_search(queries, k):
        batch_size = queries.shape[0]
        distances = np.random.random((batch_size, k))
        indices = np.random.randint(0, 100, (batch_size, k))
        return distances, indices
    
    index.search = mock_search
    return index


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        'model_name': 'bert-base-multilingual-cased',
        'max_length': 512,
        'batch_size': 32,
        'learning_rate': 1e-5,
        'num_epochs': 3,
        'embedding_dim': 128,
        'index_path': 'test_index',
        'collection_path': 'test_collection'
    }


@pytest.fixture
def mock_distributed():
    """Mock distributed training setup."""
    with patch('torch.distributed.is_initialized', return_value=False):
        with patch('torch.distributed.get_rank', return_value=0):
            with patch('torch.distributed.get_world_size', return_value=1):
                yield


@pytest.fixture
def mock_cuda():
    """Mock CUDA availability for testing on CPU."""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def sample_batch():
    """Sample batch data for testing training loops."""
    return {
        'input_ids': torch.randint(0, 1000, (2, 10)),
        'attention_mask': torch.ones(2, 10),
        'labels': torch.randint(0, 2, (2,))
    }


@pytest.fixture
def mock_logger():
    """Mock logger for testing logging functionality."""
    logger = MagicMock()
    return logger