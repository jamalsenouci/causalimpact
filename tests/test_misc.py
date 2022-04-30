"""Tests for misc module."""


import mock
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal
import pytest

import causalimpact


standardize = causalimpact.misc.standardize_all_variables
unstandardize = causalimpact.misc.unstandardize
df_print = causalimpact.misc.df_print


def test_basic_standardize():
    pre_period = [0, 2]
    post_period = [3, 4]

    data = {"c1": [1, 4, 8, 9, 10], "c2": [4, 8, 12, 16, 20]}
    data = pd.DataFrame(data)

    result = standardize(data, pre_period, post_period)
    assert_almost_equal(np.zeros((2)), np.mean(result["data_pre"].values, axis=0))

    assert_almost_equal(np.ones((2)), np.std(result["data_pre"].values, axis=0))
    assert len(result["data_pre"]) == pre_period[-1] + 1


def test_standardize_returns_expected_types():
    pre_period = [0, 4]
    post_period = [5, 5]

    data = [-1, 0.1, 1, 2, np.nan, 3]
    data = pd.DataFrame(data)

    result = standardize(data, pre_period, post_period)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(["data_pre", "data_post", "orig_std_params"])

    assert len(result["data_pre"]) == pre_period[-1] + 1
    assert_frame_equal(
        unstandardize(result["data_pre"], result["orig_std_params"]),
        pd.DataFrame(data[:5]),
    )


def test_standardize_w_distinct_inputs():
    test_data = [[1], [1, 1, 1], [1, np.nan, 3], pd.DataFrame([10, 20, 30])]

    test_data = [pd.DataFrame(data, dtype="float") for data in test_data]

    for data in test_data:
        result = standardize(
            data,
            pre_period=[0, len(data) + 1],
            post_period=[len(data) + 1, len(data) + 1],
        )

        pd.util.testing.assert_frame_equal(
            unstandardize(result["data_pre"], result["orig_std_params"]), data
        )


def test_standardize_raises_w_bad_input():
    with pytest.raises(ValueError):
        standardize("text", 1, 2)

    with pytest.raises(ValueError):
        standardize(pd.DataFrame([1, 2]), 1, 2)


def test_unstandardize():
    data = np.array([-1.16247639, -0.11624764, 1.27872403])
    orig_std_params = (4.3333333, 2.8674417556)
    original_data = unstandardize(data, orig_std_params)

    assert_almost_equal(original_data.values, np.array([[1.0, 4.0, 8.0]]).T)


def test_df_print():
    data_mock = mock.Mock()
    df_print(data_mock)
    data_mock.assert_not_called()

    df_print(data_mock, path="path")
    data_mock.to_csv.assert_called_once_with("path")
