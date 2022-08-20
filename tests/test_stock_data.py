import pytest

# Standard Python modules
import os
from unittest.mock import patch

# Third party modules
import pandas as pd

# Local imports
import mvarch.util as util
import mvarch.stock_data as stock_data

SAMPLE_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "open": [1.0, 2.0, 3.0],
        "close": [0.5, 2.5, 3.1],
    }
).set_index("date")

SAMPLE_PATH = "any_path"


@pytest.fixture
def data_source():
    """
    Create an instance of a data source for testing
    """
    mock_data_source = lambda symbols: {s.upper(): SAMPLE_DF for s in symbols}
    return mock_data_source


def test_symbol_history_reader_and_writer(tmp_path):
    filename = os.path.join(tmp_path, "foo.csv")

    # Use writer to write SAMPLE_df

    # But first, intentionally reverse the order of dates.
    sample_copy = SAMPLE_DF.reset_index().sort_values("date", ascending=False)
    assert not util.is_sorted(sample_copy.date)

    # Define a helper to simplify writing slightly different versions of SAMEPLE_DF
    reader = stock_data.SymbolHistoryReader()

    def check(writer):
        with open(filename, "wb") as f:
            writer(f)

        # Use reader to read it back and compare the results.
        with open(filename, "rb") as f:
            loaded_df = reader(f)

        assert loaded_df.index.name == "date"
        assert util.is_sorted(loaded_df.index)
        assert (loaded_df == SAMPLE_DF).all().all()

    for df in [sample_copy, sample_copy.set_index("date")]:
        check(stock_data.SymbolHistoryWriter(df))


def test_file_system_store(tmp_path):
    symbol = "FOO"
    store = stock_data.FileSystemStore(tmp_path)
    assert not store.exists(symbol)

    store.write("FOO", stock_data.SymbolHistoryWriter(SAMPLE_DF))
    assert store.exists("FOO")

    loaded_df = store.read(symbol, stock_data.SymbolHistoryReader())
    assert (loaded_df == SAMPLE_DF).all().all()


def test_check_cache_exists_path(tmp_path):
    """
    Check that the os.path.exists() gets called with the correct path
    and check that exists is not case sensitive.
    """
    tmp_path_store = stock_data.FileSystemStore(tmp_path)
    with patch("mvarch.stock_data.os.path.exists") as os_path_exists:
        tmp_path_store.exists("symbol1")
        os_path_exists.assert_called_with(
            os.path.join(tmp_path_store.cache_dir, "symbol1.csv")
        )

        tmp_path_store.exists("SyMbOL2")
        os_path_exists.assert_called_with(
            os.path.join(tmp_path_store.cache_dir, "symbol2.csv")
        )


def test_history(data_source, tmp_path):
    partial_symbol_set = set(["ABC", "DEF"])
    missing_symbol_set = set(["GHI", "JKL"])
    full_symbol_set = partial_symbol_set.union(missing_symbol_set)

    tmp_path_store = stock_data.FileSystemStore(tmp_path)
    caching_download = stock_data.CachingDownloader(
        data_source,
        tmp_path_store,
        stock_data.SymbolHistoryWriter,
        overwrite_existing=False,
    )

    response = caching_download(partial_symbol_set)
    assert len(response) == len(partial_symbol_set)
    for symbol in partial_symbol_set:
        assert tmp_path_store.exists(symbol)

    for symbol in missing_symbol_set:
        assert not tmp_path_store.exists(symbol)

    response = caching_download(full_symbol_set)
    # Check that only the missing symbols were downloaded
    # This is true if all missing symbols are in the response
    # and if the length of the response is equal to the number
    # of missing symbols
    assert len(response) == len(missing_symbol_set)
    for symbol in missing_symbol_set:
        assert symbol in response

    for symbol in full_symbol_set:
        assert tmp_path_store.exists(symbol)

    # Try downloading again, which should be a no-op
    response = caching_download(full_symbol_set)
    assert len(response) == 0

    # Try loading one of the downloaded files
    loader = stock_data.CachingSymbolHistoryLoader(
        data_source, tmp_path_store, overwrite_existing=False
    )
    # load("pqr")
    combiner = stock_data.PriceHistoryConcatenator()
    combiner(loader("pqr"))
