import torch

import pytest

from mvarch.deep_learning import models

BATCH_SIZE = 2
CONTEXT_SIZE = 128
EMBEDDING_SIZE = 6


@pytest.fixture
def batch():
    generator = torch.random.manual_seed(42)
    return {
        "window": torch.randn((BATCH_SIZE, CONTEXT_SIZE), generator=generator),
        "encoded_symbol": torch.randint(0, 2, (BATCH_SIZE,), generator=generator),
        "target": torch.randn((BATCH_SIZE), generator=generator),
    }


def test_simple_timeseries_emedding(batch):
    print(batch)
    model = models.SimpleTimeSeriesEmbedding(embedding_size=EMBEDDING_SIZE)
    embedding = model.forward(**batch)
    assert embedding.shape == (BATCH_SIZE, EMBEDDING_SIZE)


def test_time_series_embeddings(batch):
    print(batch)
    for Model in (models.TimeSeriesTransformer,):
        model = Model(sequence_length=64, embedding_size=EMBEDDING_SIZE, num_heads=2)
        embedding = model.forward(**batch)
        assert embedding.shape == (BATCH_SIZE, EMBEDDING_SIZE)


def test_convolutional(batch):
    for Model in (models.Convolutional, models.Convolutional2):
        print(batch)
        model = Model(sequence_length=128, embedding_size=EMBEDDING_SIZE)
        embedding = model.forward(**batch)
        assert embedding.shape == (BATCH_SIZE, EMBEDDING_SIZE)


def test_normal_head(batch):
    latents = torch.randn(BATCH_SIZE, EMBEDDING_SIZE)
    embedding_model = models.TimeSeriesTransformer(
        sequence_length=64, embedding_size=EMBEDDING_SIZE, num_heads=2
    )
    model = models.NormalHead(
        embedding_model=embedding_model, latent_dim=EMBEDDING_SIZE
    )
    output = model.forward(**batch)
    assert output.shape == (BATCH_SIZE, 2)
    assert all(output[:, 1] > 0)
