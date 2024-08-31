import torch

import pytest

from mvarch.deep_learning import models

BATCH_SIZE = 2
CONTEXT_SIZE = 64
EMBEDDING_SIZE = 6


@pytest.fixture
def batch():
    generator = torch.random.manual_seed(42)
    return {
        "context": torch.randn((BATCH_SIZE, CONTEXT_SIZE), generator=generator),
        "symbol_encoding": torch.randint(0, 2, (BATCH_SIZE,), generator=generator),
        "target": torch.randn((BATCH_SIZE), generator=generator),
    }


def test_simple_timeseries_emedding(batch):
    print(batch)
    model = models.SimpleTimeSeriesEmbedding(embedding_size=EMBEDDING_SIZE)
    embedding = model.forward(**batch)
    assert embedding.shape == (BATCH_SIZE, EMBEDDING_SIZE)


def test_time_series_transformer(batch):
    print(batch)
    model = models.TimeSeriesTransformer(embedding_size=EMBEDDING_SIZE, num_heads=2)
    embedding = model.forward(**batch)
    assert embedding.shape == (BATCH_SIZE, EMBEDDING_SIZE)
