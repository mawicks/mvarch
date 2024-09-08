# Standard Python
import datetime as dt
import logging
from typing import Callable, Iterable, Iterator, Union, Tuple

# Common packages
import click
import pandas as pd  # type: ignore

import torch
import torch.utils
import torch.utils.data
from lightning import LightningModule  # , LightningDataModule
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.trainer.trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

FAST = False
BATCH_SIZE = 16
TRANSFORMER_LR = 0.00002  # 0.0002  # 1e-4
CONVOLUTIONAL_LR = 0.0001
DEFAULT_LR = CONVOLUTIONAL_LR
TUNE_LR = False
DATASET_SEQUENCE_LENGTH = 128
MODEL_SEQUENCE_LENGTH = 128
SYMBOL_EMBEDDING_SIZE = 0
NUM_LAYERS = 4
NUM_HEADS = 4
# Note: LATENT_DIM must be divisible by NUM_HEADS
LATENT_DIM = 64

# Local modules
from mvarch.data_sources import HugeStockMarketDatasetSource, YFinanceSource
from mvarch.stock_data import (
    FileSystemStore,
    CachingSymbolHistoryLoader,
)
from mvarch.time_series_dataset import MultiSymbolDataset

import mvarch.deep_learning.models as models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    force=True,
)

if torch.cuda.is_available():
    dev = "cuda:0"
elif torch.backends.mps.is_built():
    dev = "mps"
else:
    dev = "cpu"

device = torch.device(dev)

x = torch.nn.GaussianNLLLoss(full=True, eps=1e-12)


def make_loss_function():
    _loss = torch.nn.GaussianNLLLoss(full=True)

    def loss(output, target):
        mu = output[:, 0]
        sigma = output[:, 1]
        return _loss(mu, target, sigma * sigma)

    return loss


class TrainingFixture(LightningModule):
    def __init__(self, model, lr=DEFAULT_LR):
        super().__init__()
        self.model = model
        self.loss_fn = make_loss_function()
        self.lr = lr
        self.save_hyperparameters("lr")
        self.save_hyperparameters(model.hyperparameters())

    def forward(self, x):
        return self.model(x)

    def loss(self, batch):
        output = self.model(**batch)
        target = torch.mean(batch["target"], dim=1)
        return self.loss_fn(output, target)

    def training_step(self, batch, batch_idx):
        return self.loss(batch)

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            threshold_mode="abs",
            threshold=1e-3,
            patience=3,
            cooldown=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }


def prepare_data(
    history_loader: Callable[
        [Union[str, Iterable[str]]], Iterator[Tuple[str, pd.DataFrame]]
    ],
    symbol_list: Iterable[str],
    start_date: Union[dt.date, None] = None,
    end_date: Union[dt.date, None] = None,
    eval_start_date: Union[dt.date, None] = None,
    eval_end_date: Union[dt.date, None] = None,
):
    logging.info("Reading historical data")
    full_history = dict(history_loader(symbol_list))
    logging.info("Done")
    training_data = MultiSymbolDataset(
        full_history,
        context_size=DATASET_SEQUENCE_LENGTH,
        target_dim=25,
        start_date=start_date,
        end_date=end_date,
    )
    evaluation_data = MultiSymbolDataset(
        full_history,
        context_size=DATASET_SEQUENCE_LENGTH,
        target_dim=25,
        start_date=eval_start_date,
        end_date=eval_end_date,
        encoder=training_data.encoder(),
        decoder=training_data.decoder(),
    )
    return training_data, evaluation_data


def check_for_duplicates(symbol_list):
    d = {}
    for s in symbol_list:
        d[s] = d.get(s, 0) + 1
    duplicates = [s for s in d.keys() if d[s] > 1]
    if len(duplicates) > 0:
        raise ValueError(f"Duplicate symbols: {duplicates}")


def run(
    use_hsmd,
    symbols,
    refresh,
    output=None,
    start_date=None,
    end_date=None,
    eval_start_date=None,
    eval_end_date=None,
):
    # Rewrite symbols with deduped, uppercase versions
    symbols = list(map(str.upper, symbols))

    if FAST:
        symbols = symbols[:2]

    # Check for duplicates
    check_for_duplicates(symbols)

    logging.debug(f"device: {device}")
    logging.debug(f"symbols: {symbols}")
    logging.debug(f"refresh: {refresh}")
    logging.debug(f"Start date: {start_date}")
    logging.debug(f"End date: {end_date}")
    logging.debug(f"Evaluation/termination start date: {eval_start_date}")
    logging.debug(f"Evaluation/termination end date: {eval_end_date}")

    data_store = FileSystemStore("training_data")
    if use_hsmd:
        data_source = HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = YFinanceSource()

    history_loader = CachingSymbolHistoryLoader(data_source, data_store, refresh)

    training_data, evaluation_data = prepare_data(
        history_loader,
        symbols,
        start_date=start_date,
        end_date=end_date,
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
    )

    logging.debug(f"length of training_data: {len(training_data)}")
    logging.debug(f"length of evaluation_data: {len(evaluation_data)}")
    logging.debug(f"Symbol count: {evaluation_data.symbol_count()}")

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        evaluation_data, batch_size=100 * BATCH_SIZE, num_workers=7
    )

    convolutional = models.Convolutional2(
        sequence_length=MODEL_SEQUENCE_LENGTH,
        embedding_size=LATENT_DIM,
    )
    transformer = models.TimeSeriesTransformer(
        sequence_length=MODEL_SEQUENCE_LENGTH,
        embedding_size=LATENT_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    )

    embedding_model = transformer

    if SYMBOL_EMBEDDING_SIZE > 0:
        embedding_model = models.SymbolEmbeddings(
            embedding_model,
            symbol_count=len(symbols),
            symbol_embedding_size=SYMBOL_EMBEDDING_SIZE,
            embedding_size=LATENT_DIM,
        )

    model = models.NormalHead(
        embedding_model, latent_dim=LATENT_DIM, sigma_lower_bound=0.001
    )

    # model = models.Compose(models.SimpleLinear(sequence_length=MODEL_SEQUENCE_LENGTH), torch.nn.Identity())

    training_fixture = TrainingFixture(model)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(callbacks=[early_stopping, lr_monitor])

    if TUNE_LR:
        tuner = Tuner(trainer)
        tuner.lr_find(training_fixture, train_loader, eval_loader)

    trainer.fit(training_fixture, train_loader, eval_loader)

    if output is not None:
        output_object = {
            "date": dt.datetime.today(),
            "symbols": symbols,
            "encoder": training_data.encoder(),
            "decoder": training_data.decoder(),
            "model": model,
            "start_date": start_date,
            "end_date": end_date,
            "eval_start_date": eval_start_date,
            "eval_end_date": eval_end_date,
        }

        torch.save(output_object, output.name)

    logging.info(f"Symbols: {symbols}")


@click.command()
@click.option(
    "--symbol",
    "-s",
    multiple=True,
    help="Symbol(s) to include in model (may be specified multiple times)",
)
@click.option(
    "--use-hsmd",
    default=None,
    show_default=True,
    help="Use huge stock market dataset if specified zip file (else use yfinance)",
)
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    show_default=True,
    help="Refresh cached stock data",
)
@click.option(
    "--output",
    "-o",
    type=click.File("wb"),
    help="Output file for trained model and metadata.",
)
@click.option(
    "--start-date",
    default=None,
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date of data used for training",
)
@click.option(
    "--end-date",
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Final date of data used for training",
)
@click.option(
    "--eval-start-date",
    default=None,
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date of data used for evaluation/termination",
)
@click.option(
    "--eval-end-date",
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last date of data used for evaluation/termination",
)
def main_cli(
    symbol,
    use_hsmd,
    refresh,
    output,
    start_date,
    end_date,
    eval_start_date,
    eval_end_date,
):

    if start_date:
        start_date = start_date.date()

    if end_date:
        end_date = end_date.date()

    if eval_start_date:
        eval_start_date = eval_start_date.date()

    if eval_end_date:
        eval_end_date = eval_end_date.date()

    run(
        use_hsmd,
        symbols=symbol,
        refresh=refresh,
        output=output,
        start_date=start_date,
        end_date=end_date,
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
    )


if __name__ == "__main__":
    main_cli()
