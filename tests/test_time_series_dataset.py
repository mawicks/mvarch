import pytest
from mvarch.time_series_dataset import Dataset, TargetSelection, MultiSymbolDataset


def test_dataset():
    d = Dataset(range(10), 3)
    assert len(d) == 8
    assert tuple(d[0]) == tuple(range(3))
    assert tuple(d[1]) == tuple(range(1, 4))
    assert tuple(d[-1]) == tuple(range(7, 10))
    assert tuple(d[-2]) == tuple(range(6, 9))
    assert tuple(d[-8]) == tuple(range(3))
    with pytest.raises(IndexError):
        d[8]
    with pytest.raises(IndexError):
        d[-9]

    d = Dataset(range(10), 3, stride=2)
    assert len(d) == 4
    assert tuple(d[0]) == tuple(range(3))
    assert tuple(d[1]) == tuple(range(2, 5))
    assert tuple(d[-1]) == tuple(range(6, 9))
    assert tuple(d[-2]) == tuple(range(4, 7))
    assert tuple(d[-4]) == tuple(range(3))
    with pytest.raises(IndexError):
        d[4]
    with pytest.raises(IndexError):
        d[-5]

    with pytest.raises(ValueError):
        Dataset(range(10), 3, stride=0)


def test_target_selection():
    d = Dataset(range(10), 3, stride=2)
    t = TargetSelection(d, target_dim=1)

    i = iter(t)
    d = next(i)
    assert d["target"] == 2
    d = next(i)
    assert d["target"] == 4
    d = next(i)
    assert d["target"] == 6
    d = next(i)
    assert d["target"] == 8

    with pytest.raises(StopIteration):
        next(i)
