import pandas as pd

from pyprideap.core import cached_per_dataset, dataset_cache, numeric_frame, with_dataset_cache


def test_numeric_frame_converts_only_text_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["3.5", "x"]})
    out = numeric_frame(df)
    assert out["a"].tolist() == [1.0, 2.0]
    assert out["b"].iloc[0] == 3.5 and pd.isna(out["b"].iloc[1])
    assert out.equals(df.apply(pd.to_numeric, errors="coerce"))


def test_numeric_frame_returns_a_copy():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = numeric_frame(df)
    out.iloc[0, 0] = 99.0
    assert df.iloc[0, 0] == 1.0


class _Obj:
    pass


def test_cache_only_inside_scope():
    calls = []

    @cached_per_dataset
    def work(ds):
        calls.append(ds)
        return len(calls)

    a, b = _Obj(), _Obj()
    work(a)
    work(a)
    assert len(calls) == 2  # no scope: no caching
    with dataset_cache():
        assert work(a) == work(a) == 3
        assert work(b) == 4  # keyed by object
        assert work(a, "extra") == 5  # extra arguments are never cached
    assert work(a) == 6


def test_with_dataset_cache_decorator():
    calls = []

    @cached_per_dataset
    def work(ds):
        calls.append(ds)

    @with_dataset_cache
    def report(ds):
        work(ds)
        work(ds)

    report(_Obj())
    assert len(calls) == 1
