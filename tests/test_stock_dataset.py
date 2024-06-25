# Standard Python modules
import os
from unittest.mock import patch

# Third party modules

# Local imports
import mvarch.util as util
import mvarch.stock_data as stock_data

SAMPLE_PATH = "any_path"


def test_history(data_source, tmp_path):
    symbol_list = set(["ABC", "DEF"])

    tmp_path_store = stock_data.FileSystemStore(tmp_path)
    caching_download = stock_data.CachingDownloader(
        data_source,
        tmp_path_store,
        stock_data.SymbolHistoryWriter,
        overwrite_existing=False,
    )

    response = caching_download(symbol_list)
    for symbol in symbol_list:
        assert tmp_path_store.exists(symbol)
