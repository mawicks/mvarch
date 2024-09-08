import torch


class SimpleTimeSeriesEmbedding(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self._embedding_size = embedding_size
        return

    def forward(self, *, covariates: torch.Tensor, **kwargs):
        batch_size = covariates.shape[0]
        return torch.randn(batch_size, self._embedding_size)


class NormalHead(torch.nn.Module):
    def __init__(self, latent_dim=64, sigma_lower_bound=0.0001):
        super().__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, 2),
        )
        self.sigma_transformation = torch.nn.ReLU()
        self.sigma_lower_bound = torch.tensor(sigma_lower_bound)

    def forward(self, latents: torch.Tensor):
        x = self.sequence(latents)
        return torch.stack(
            (x[:, 0], self.sigma_transformation(x[:, 1]) + self.sigma_lower_bound),
            dim=1,
        )


class SimpleLinear(torch.nn.Module):
    def __init__(
        self,
        sequence_length=64,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(sequence_length, 1)
        self.linear2 = torch.nn.Linear(sequence_length, sequence_length)
        self.linear3 = torch.nn.Linear(sequence_length, 1)

    def forward(
        self, *, covariates: torch.Tensor, encoded_symbol: torch.Tensor, **kwargs
    ):
        x = covariates
        mu = self.linear1(x)
        z = self.linear2(x)
        sigma = torch.abs(self.linear3((x - z) ** 2))
        return torch.concat((mu, sigma), 1)

    def hyperparameters(self):
        return {}


class TimeSeriesTransformer(torch.nn.Module):
    def __init__(
        self,
        sequence_length: int,
        symbol_count: int,
        embedding_size=64,
        kernel_size=5,
        num_heads=4,
        num_layers=6,
        symbol_embedding_size=6,
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.position_encoding = torch.tensor(
            range(sequence_length), requires_grad=False
        )
        # Convolutional layer to extract local features and increase
        # dimensionality
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.symbol_embedding_size = symbol_embedding_size

        self.position_embedding = torch.nn.Embedding(sequence_length, embedding_size)
        self.symbol_embedding = torch.nn.Embedding(symbol_count, symbol_embedding_size)
        self.position_blender = torch.nn.Conv1d(
            2 * embedding_size, embedding_size, kernel_size=1
        )
        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=embedding_size,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode="zeros",
            ),
            torch.nn.ReLU(),
        )
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=num_heads
        )
        self.transformer = torch.nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        if embedding_size > 0:
            self.symbol_combiner = torch.nn.Linear(
                embedding_size + symbol_embedding_size, embedding_size
            )

        self.reducer = torch.nn.Conv1d(sequence_length, 1, kernel_size=1)

        self.final = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ReLU(),
        )

    def hyperparameters(self):
        return {
            "embedding_size": self.embedding_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "symbol_embedding_size": self.symbol_embedding_size,
        }

    def forward(
        self, *, covariates: torch.Tensor, encoded_symbol: torch.Tensor, **kwargs
    ):
        batch_size, __sequence_length__ = covariates.shape
        self.position_encoding = self.position_encoding.to(covariates.device)

        position_embedding = (
            self.position_embedding(self.position_encoding)
            .permute(1, 0)
            .expand(batch_size, self.embedding_size, self.sequence_length)
        )

        x = covariates[:, -self.sequence_length :].unsqueeze(
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

        if self.symbol_embedding_size > 0:
            x = self.symbol_combiner(
                torch.concat((x, self.symbol_embedding(encoded_symbol)), dim=1)
            )

        x = self.final(x)
        return x


class Compose(torch.nn.Module):
    def __init__(self, embedding_model, output_model):
        super().__init__()
        self._embedding_model = embedding_model
        self._output_model = output_model

    def forward(self, x):
        return self._output_model(self._embedding_model(**x))

    def hyperparameters(self):
        return self._embedding_model.hyperparameters()
