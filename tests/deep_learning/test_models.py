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
        "covariates": torch.randn((BATCH_SIZE, CONTEXT_SIZE), generator=generator),
        "encoded_symbol": torch.randint(0, 2, (BATCH_SIZE,), generator=generator),
        "target": torch.randn((BATCH_SIZE), generator=generator),
    }


def test_simple_timeseries_emedding(batch):
    print(batch)
    model = models.SimpleTimeSeriesEmbedding(embedding_size=EMBEDDING_SIZE)
    embedding = model.forward(**batch)
    assert embedding.shape == (BATCH_SIZE, EMBEDDING_SIZE)


def test_time_series_transformer(batch):
    print(batch)
    model = models.TimeSeriesTransformer(
        sequence_length=64, symbol_count=2, embedding_size=EMBEDDING_SIZE, num_heads=2
    )
    embedding = model.forward(**batch)
    assert embedding.shape == (BATCH_SIZE, EMBEDDING_SIZE)


def test_normal_head():
    latents = torch.randn(BATCH_SIZE, EMBEDDING_SIZE)
    model = models.NormalHead(latent_dim=EMBEDDING_SIZE)
    output = model.forward(latents)
    assert output.shape == (BATCH_SIZE, 2)
    assert all(output[:, 1] > 0)
