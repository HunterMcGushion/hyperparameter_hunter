##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.data.data_core import BaseDataChunk, BaseDataset, NullDataChunk

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd
import pytest
from unittest import mock


##################################################
# White-Box/Structural Test Fixtures
##################################################
@pytest.fixture(scope="module")
def null_chunk_fixture():
    """Boring fixture that creates an instance of :class:`data.data_core.NullDataChunk`"""
    return NullDataChunk()


@pytest.fixture(scope="module")
def base_dataset_fixture():
    """Boring fixture that creates an instance of :class:`data.data_core.BaseDataset`"""
    return BaseDataset(None, None)


##################################################
# White-Box/Structural Tests
##################################################
@mock.patch("hyperparameter_hunter.data.data_core.NullDataChunk._on_call_default")
@pytest.mark.parametrize("point", ["start", "end"])
@pytest.mark.parametrize("division", ["experiment", "repetition", "fold", "run"])
def test_callback_method_invocation(mock_on_call_default, point, division, null_chunk_fixture):
    """Test that calling any primary callback methods of :class:`data.data_core.NullDataChunk`
    results in a call to :meth:`data.data_core.BaseDataCore._on_call_default` with the appropriate
    `division` and `point` arguments. Using `on_fold_end` as an example, this function ensures::
        `on_fold_end(...)` call                    ->   `_on_call_default("fold", "end", ...)` call"""
    null_chunk_fixture.__getattribute__(f"on_{division}_{point}")("An arg", k="A kwarg")
    mock_on_call_default.assert_called_once_with(division, point, "An arg", k="A kwarg")


@pytest.mark.parametrize("point", ["start", "end"])
@pytest.mark.parametrize("division", ["experiment", "repetition", "fold", "run"])
def test_do_something_invocation(point, division, null_chunk_fixture):
    """Test that calling :meth:`data.data_core.NullDataChunk._do_something` results in the invocation
    of the proper primary callback method as specified by `division` and `point`. Using
    `on_fold_end` as an example, this function ensures::
        `_do_something("fold", "end", ...)` call   ->   `on_fold_end(...)` call"""
    method_to_mock = f"on_{division}_{point}"
    mock_method_path = f"hyperparameter_hunter.data.data_core.NullDataChunk.{method_to_mock}"

    with mock.patch(mock_method_path) as mock_primary_callback:
        null_chunk_fixture._do_something(division, point, "An arg", k="A kwarg")
        mock_primary_callback.assert_called_once_with("An arg", k="A kwarg")


@pytest.mark.parametrize("point", ["start", "end"])
@pytest.mark.parametrize("division", ["experiment", "repetition", "fold", "run"])
def test_kind_chunk_invocation(point, division, base_dataset_fixture):
    """Test that calling :meth:`data.data_core.BaseDataset._do_something` results in the invocation
    of the proper callback method of :class:`data.data_core.BaseDataChunk` three times (once for
    `input`, `target` and `prediction`). Using `on_fold_end` as an example, this function ensures::
        `_do_something("fold", "end", ...)` `BaseDataset` call   ->
            `on_fold_end(...)` call (`input` chunk)
            `on_fold_end(...)` call (`target` chunk)
            `on_fold_end(...)` call (`prediction` chunk)"""
    method_to_mock = f"on_{division}_{point}"
    mock_method_path = f"hyperparameter_hunter.data.data_core.BaseDataChunk.{method_to_mock}"

    with mock.patch(mock_method_path) as mock_primary_callback:
        base_dataset_fixture._do_something(division, point, "An arg", k="A kwarg")
        mock_primary_callback.assert_has_calls([mock.call("An arg", k="A kwarg")] * 3)


##################################################
# `BaseDataChunk` Equality
##################################################
def _update_data_chunk(updates: dict):
    chunk = BaseDataChunk(None)

    for key, value in updates.items():
        if key.startswith("T."):
            setattr(chunk.T, key[2:], value)
        else:
            setattr(chunk, key, value)
    return chunk


@pytest.fixture()
def data_chunk_fixture(request):
    return _update_data_chunk(getattr(request, "params", dict()))


@pytest.fixture()
def another_data_chunk_fixture(request):
    return _update_data_chunk(getattr(request, "params", dict()))


#################### Test Scenario Data ####################
df_0 = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6]))
df_1 = pd.DataFrame(dict(a=[1, 2, 3], b=[999, 5, 6]))
df_2 = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6]), index=["foo", "bar", "baz"])
df_3 = pd.DataFrame(dict(a=[1, 2, 3], c=[4, 5, 6]), index=["foo", "bar", "baz"])
df_4 = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9]))

chunk_data_0 = dict(d=pd.DataFrame())
chunk_data_1 = dict(d=pd.DataFrame(), fold=df_0)
chunk_data_2 = dict(d=pd.DataFrame(), fold=df_1)
chunk_data_3 = dict(d=pd.DataFrame(), fold=df_2)
chunk_data_4 = {"d": pd.DataFrame(), "fold": df_2, "T.fold": df_3}
chunk_data_5 = {"d": pd.DataFrame(), "fold": df_3, "T.fold": df_2}
chunk_data_6 = {"d": pd.DataFrame(), "fold": df_3, "T.fold": df_2, "T.d": df_4}


@pytest.mark.parametrize(
    ["data_chunk_fixture", "another_data_chunk_fixture"],
    [
        [dict(), dict()],
        [chunk_data_0, chunk_data_0],
        [chunk_data_1, chunk_data_1],
        [chunk_data_2, chunk_data_2],
        [chunk_data_3, chunk_data_3],
        [chunk_data_4, chunk_data_4],
        [chunk_data_5, chunk_data_5],
        [chunk_data_6, chunk_data_6],
    ],
    indirect=True,
)
def test_data_chunk_equality(data_chunk_fixture, another_data_chunk_fixture):
    assert data_chunk_fixture == another_data_chunk_fixture


#################### Inequality Tests ####################
@pytest.mark.parametrize(
    "data_chunk_fixture",
    [chunk_data_1, chunk_data_2, chunk_data_3, chunk_data_4, chunk_data_5, chunk_data_6],
    indirect=True,
)
def test_data_chunk_inequality_0(data_chunk_fixture):
    assert _update_data_chunk(chunk_data_0) != data_chunk_fixture


@pytest.mark.parametrize(
    "data_chunk_fixture",
    [chunk_data_0, chunk_data_2, chunk_data_3, chunk_data_4, chunk_data_5, chunk_data_6],
    indirect=True,
)
def test_data_chunk_inequality_1(data_chunk_fixture):
    assert _update_data_chunk(chunk_data_1) != data_chunk_fixture


@pytest.mark.parametrize(
    "data_chunk_fixture",
    [chunk_data_0, chunk_data_1, chunk_data_3, chunk_data_4, chunk_data_5, chunk_data_6],
    indirect=True,
)
def test_data_chunk_inequality_2(data_chunk_fixture):
    assert _update_data_chunk(chunk_data_2) != data_chunk_fixture


@pytest.mark.parametrize(
    "data_chunk_fixture",
    [chunk_data_0, chunk_data_1, chunk_data_2, chunk_data_4, chunk_data_5, chunk_data_6],
    indirect=True,
)
def test_data_chunk_inequality_3(data_chunk_fixture):
    assert _update_data_chunk(chunk_data_3) != data_chunk_fixture


@pytest.mark.parametrize(
    "data_chunk_fixture",
    [chunk_data_0, chunk_data_1, chunk_data_2, chunk_data_3, chunk_data_5, chunk_data_6],
    indirect=True,
)
def test_data_chunk_inequality_4(data_chunk_fixture):
    assert _update_data_chunk(chunk_data_4) != data_chunk_fixture


@pytest.mark.parametrize(
    "data_chunk_fixture",
    [chunk_data_0, chunk_data_1, chunk_data_2, chunk_data_3, chunk_data_4, chunk_data_6],
    indirect=True,
)
def test_data_chunk_inequality_5(data_chunk_fixture):
    assert _update_data_chunk(chunk_data_5) != data_chunk_fixture


@pytest.mark.parametrize(
    "data_chunk_fixture",
    [chunk_data_0, chunk_data_1, chunk_data_2, chunk_data_3, chunk_data_4, chunk_data_5],
    indirect=True,
)
def test_data_chunk_inequality_6(data_chunk_fixture):
    assert _update_data_chunk(chunk_data_6) != data_chunk_fixture
