import torch


class SimpleTimeSeriesEmbedding(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self._embedding_size = embedding_size
        return

    def forward(
        self, *, context: torch.Tensor, symbol_encoding: torch.Tensor, **kwargs
    ):
        batch_size = context.shape[0]
        return torch.randn(batch_size, self._embedding_size)


import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        embedding_size=64,
        sequence_length=64,
        kernel_size=5,
        num_heads=4,
        num_layers=6,
    ):
        super(TimeSeriesTransformer, self).__init__()
        # Convolutional layer to extract local features and increase
        # dimensionality
        self.embedding_size = embedding_size

        self.position_embedding = nn.Embedding(sequence_length, embedding_size)
        self.position_blender = nn.Conv1d(
            2 * embedding_size, embedding_size, kernel_size=1
        )
        self.input_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=embedding_size,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode="zeros",
            ),
            nn.ReLU(),
        )
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=num_heads
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        self.reducer = nn.Conv1d(sequence_length, 1, kernel_size=1)

    def forward(
        self, *, context: torch.Tensor, symbol_encoding: torch.Tensor, **kwargs
    ):
        batch_size, sequence_length = context.shape

        position_embedding = (
            self.position_embedding(torch.tensor(range(sequence_length)))
            .permute(1, 0)
            .expand(batch_size, self.embedding_size, sequence_length)
        )

        x = context.unsqueeze(
            1
        )  # Add channel dimension, assuming input shape (batch_size, time_steps)

        # TODO - Combine with concatenation rather than addition
        x = self.position_blender(
            torch.concat((self.input_layer(x), position_embedding), 1)
        )
        # x = self.input_layer(x) + position_embedding

        x = x.permute(
            2, 0, 1
        )  # Permute to match (seq_length, batch_size, embed_dim) for Transformer
        x = self.transformer(x).permute(
            1, 0, 2
        )  # Permute to (batch_size, seq_length, embed_dim) for reduce
        x = self.reducer(x).squeeze(1)
        return x
