import pandas as pd
import math
import pytest

SAMPLE_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
            ]
        ),
        "open": [i + 1 for i in range(8)],
        "close": [round(math.exp(0.1 * (i * (i + 1) / 2)), 2) for i in range(8)],
    }
).set_index("date")


@pytest.fixture
def data_source():
    """
    Create an instance of a data source for testing
    """
    mock_data_source = lambda symbols: {s.upper(): SAMPLE_DF for s in symbols}
    return mock_data_source
