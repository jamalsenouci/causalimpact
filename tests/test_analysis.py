"""Unit Tests for analysis module."""


import os
import re
from tempfile import mkdtemp
from unittest.mock import patch
import unittest.mock as mock
import pytest
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal
from causalimpact.inferences import compile_posterior_inferences
from statsmodels.tsa.statespace.structural import UnobservedComponents as UCM
from causalimpact import CausalImpact
from unittest import TestCase


@pytest.fixture()
def data():
    return pd.DataFrame(np.random.randn(200, 3), columns=["y", "x1", "x2"])


@pytest.fixture()
def summary_report_filename(FIXTURES_FOLDER):
    return os.path.join(FIXTURES_FOLDER, "analysis", "summary_report_output.txt")


@pytest.fixture()
def expected_columns():
    return [
        "response",
        "cum_response",
        "point_pred",
        "point_pred_lower",
        "point_pred_upper",
        "cum_pred",
        "cum_pred_lower",
        "cum_pred_upper",
        "point_effect",
        "point_effect_lower",
        "point_effect_upper",
        "cum_effect",
        "cum_effect_lower",
        "cum_effect_upper",
    ]


@pytest.fixture()
def inference_input():
    return {
        "response": np.array([1.0, 2.0, 3.0, 4.0]),
        "point_pred": np.array([1.1, 2.2, 3.1, 4.1]),
        "point_pred_upper": np.array([1.5, 2.6, 3.4, 4.4]),
        "point_pred_lower": np.array([1.0, 2.0, 3.0, 4.0]),
    }


@pytest.fixture()
def pre_period():
    return [0, 100]


@pytest.fixture()
def post_period():
    return [101, 199]


@pytest.fixture()
def ucm_model(data, post_period):
    data_modeling = data.copy()
    data_modeling[post_period[0] : post_period[1] + 1] = np.nan
    return UCM(endog=data_modeling.iloc[:, 0].values, level="llevel")


@pytest.fixture()
def impact_ucm(ucm_model):
    post_period_response = np.random.randn(100)
    return CausalImpact(ucm_model=ucm_model, post_period_response=post_period_response)


@pytest.fixture()
def causal_impact(data, pre_period, post_period):
    model_args = {"niter": 123}
    return CausalImpact(data, pre_period, post_period, model_args)


class TestFormatInput:
    def test_input_default_model(self, causal_impact):
        expected = {
            "data": causal_impact.params["data"],
            "pre_period": causal_impact.params["pre_period"],
            "post_period": causal_impact.params["post_period"],
            "model_args": causal_impact.params["model_args"],
            "ucm_model": None,
            "post_period_response": None,
            "alpha": causal_impact.params["alpha"],
        }

        result = causal_impact._format_input(
            causal_impact.params["data"],
            causal_impact.params["pre_period"],
            causal_impact.params["post_period"],
            causal_impact.params["model_args"],
            None,
            None,
            causal_impact.params["alpha"],
        )

        result_data = result["data"]
        expected_data = expected["data"]
        assert_frame_equal(result_data, expected_data)

        result_model_args = result["model_args"]
        expected_model_args = expected["model_args"]
        assert result_model_args == expected_model_args

        result_other = {
            key: result[key] for key in result if key not in {"model_args", "data"}
        }

        expected_other = {
            key: expected[key] for key in expected if key not in {"model_args", "data"}
        }
        assert result_other == expected_other

    def test_input_raises_w_data_and_ucm_model(self, causal_impact, ucm_model):
        # Test inconsistent input (must not provide both data and ucm_model)
        with pytest.raises(SyntaxError) as excinfo:
            causal_impact._format_input(
                causal_impact.params["data"],
                causal_impact.params["pre_period"],
                causal_impact.params["post_period"],
                causal_impact.params["model_args"],
                ucm_model,
                [1, 2, 3],
                causal_impact.params["alpha"],
            )
        assert str(excinfo.value) == (
            "Must either provide ``data``, "
            "``pre_period`` ,``post_period``, ``model_args`` or "
            "``ucm_modeland ``post_period_response``"
        )

    def test_input_w_ucm_input(self, ucm_model, impact_ucm):
        expected = {
            "data": None,
            "pre_period": None,
            "post_period": None,
            "ucm_model": impact_ucm.params["ucm_model"],
            "post_period_response": impact_ucm.params["post_period_response"],
            "alpha": impact_ucm.params["alpha"],
        }

        result = impact_ucm._format_input(
            None,
            None,
            None,
            None,
            impact_ucm.params["ucm_model"],
            impact_ucm.params["post_period_response"],
            impact_ucm.params["alpha"],
        )
        result.pop("model_args")
        assert result == expected

    def test_format_output_is_df(self, causal_impact):
        # Test that <data> is converted to pandas DataFrame
        expected_data = pd.DataFrame(np.arange(0, 8).reshape(4, 2), index=[0, 1, 2, 3])

        funny_datas = [
            pd.DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]]),
            pd.DataFrame(data=[[0, 1], [2, 3], [4, 5], [6, 7]], index=[0, 1, 2, 3]),
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
        ]

        for funny_data in funny_datas:
            result = causal_impact._format_input(
                funny_data, [0, 2], [3, 3], {}, None, None, 0.05
            )
            assert_array_equal(result["data"].values, expected_data.values)
            assert isinstance(result["data"], pd.DataFrame)

    def test_input_w_bad_data(self, causal_impact):
        text_data = "foo"
        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(text_data, [0, 3], [3, 3], {}, None, None, 0.05)
        assert str(excinfo.value) == (
            "could not convert input data to " "Pandas DataFrame"
        )

    def test_input_w_bad_pre_period(self, data, causal_impact):
        bad_pre_periods = [
            1,
            [],
            [1, 2, 3],
            [np.nan, 2],
            [pd.to_datetime(date) for date in ["2011-01-01", "2011-12-31"]],
        ]

        errors_list = [
            "pre_period and post_period must both be lists",
            "pre_period and post_period must both be of length 2",
            "pre_period and post_period must both be of length 2",
            "pre_period and post period must not contain null values",
            (
                "pre_period (object) and post_period (int64) should have the same"
                " class as the time points in the data (int64)"
            ),
        ]

        for idx, bad_pre_period in enumerate(bad_pre_periods):
            with pytest.raises(ValueError) as excinfo:
                causal_impact._format_input(
                    data, bad_pre_period, [1, 2], None, None, None, 0.05
                )
            assert str(excinfo.value) == errors_list[idx]

    def test_input_w_bad_post_period(self, data, causal_impact):
        bad_post_periods = [
            1,
            [],
            [1, 2, 3],
            [np.nan, 2],
            [pd.to_datetime(date) for date in ["2011-01-01", "2011-12-31"]],
        ]

        errors_list = [
            "pre_period and post_period must both be lists",
            "pre_period and post_period must both be of length 2",
            "pre_period and post_period must both be of length 2",
            "pre_period and post period must not contain null values",
            (
                "pre_period (int64) and post_period (object) should have the same"
                " class as the time points in the data (int64)"
            ),
        ]

        for idx, bad_post_period in enumerate(bad_post_periods):
            print(idx)
            with pytest.raises(ValueError) as excinfo:
                causal_impact._format_input(
                    data, [1, 2], bad_post_period, None, None, None, 0.05
                )
            assert str(excinfo.value) == errors_list[idx]

    def test_input_w_pre_and_post_periods_having_distinct_classes(self, causal_impact):
        # Test what happens when pre_period/post_period has a different class
        # than the timestamps in <data>
        bad_data = pd.DataFrame(
            data=[1, 2, 3, 4],
            index=["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04"],
        )

        bad_pre_period = [0.0, 3.0]  # float
        bad_post_period = [3, 3]

        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                bad_data, bad_pre_period, bad_post_period, None, None, None, 0.05
            )
        assert str(excinfo.value) == (
            "pre_period (float64) and post_period ("
            "int64) should have the same class as the time points in the data"
            " (object)"
        )

        bad_pre_period = [0, 2]  # integer
        bad_post_period = [3, 3]
        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                bad_data, bad_pre_period, bad_post_period, None, None, None, 0.05
            )
        assert str(excinfo.value) == (
            "pre_period (int64) and post_period ("
            "int64) should have the same class as the time points in the data"
            " (object)"
        )

    def test_bad_model_args(self, data, causal_impact):
        with pytest.raises(TypeError) as excinfo:
            causal_impact._format_input(data, [0, 3], [3, 10], 1000, None, None, 0.05)
        assert str(excinfo.value) == "argument of type 'int' is not iterable"
        with pytest.raises(TypeError) as excinfo:
            causal_impact._format_input(
                data, [0, 3], [3, 10], "ninter=1000", None, None, 0.05
            )
        assert str(excinfo.value) == ("'str' object does not support item assignment")

    def test_bad_standardize(self, data, causal_impact):
        bad_standardize_data = [np.nan, 123, "foo", [True, False]]
        for bad_standardize in bad_standardize_data:
            bad_model_args = {"standardize_data": bad_standardize}
            with pytest.raises(ValueError) as excinfo:
                causal_impact._format_input(
                    data, [0, 3], [3, 10], bad_model_args, None, None, 0.05
                )
            assert str(excinfo.value) == (
                "model_args.standardize_data must be a boolean value"
            )

    def test_bad_post_period_response(self, causal_impact, impact_ucm):
        # Note that consistency with ucm.model is not tested in format_input()
        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                None, None, None, None, impact_ucm, pd.to_datetime("2011-01-01"), 0.05
            )
        assert str(excinfo.value) == ("post_period_response must be list-like")

        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(None, None, None, None, impact_ucm, True, 0.05)
        assert str(excinfo.value) == ("post_period_response must be list-like")

        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                None, None, None, None, impact_ucm, [pd.to_datetime("2018-01-01")], 0.05
            )
        assert str(excinfo.value) == (
            "post_period_response should not be datetime values"
        )

        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(None, None, None, None, impact_ucm, [2j], 0.05)
        assert str(excinfo.value) == (
            "post_period_response must contain all real values"
        )

    def test_bad_alpha(self, data, causal_impact):
        bad_alphas = [None, np.nan, -1, 0, 1, [0.8, 0.9], "0.1"]
        for bad_alpha in bad_alphas:
            with pytest.raises(ValueError) as excinfo:
                causal_impact._format_input(
                    data, [0, 3], [3, 10], {}, None, None, bad_alpha
                )
        assert str(excinfo.value) == "alpha must be a real number"

    def test_input_w_date_column(self):
        data = pd.DataFrame(np.random.randn(100, 2), columns=["x1", "x2"])
        data["date"] = pd.date_range(start="2018-01-01", periods=100)
        data = data[["date", "x1", "x2"]]
        pre_period = ["2018-01-01", "2018-02-10"]
        post_period = ["2018-02-11", "2018-4-10"]
        causal_impact = CausalImpact(data, pre_period, post_period, {})
        data = data.set_index("date")
        pre_period = [pd.to_datetime(e) for e in pre_period]
        post_period = [pd.to_datetime(e) for e in post_period]

        expected = {
            "data": data,
            "pre_period": pre_period,
            "post_period": post_period,
            "model_args": causal_impact.params["model_args"],
            "ucm_model": None,
            "post_period_response": None,
            "alpha": causal_impact.params["alpha"],
        }
        result = causal_impact._format_input(
            causal_impact.params["data"],
            causal_impact.params["pre_period"],
            causal_impact.params["post_period"],
            causal_impact.params["model_args"],
            None,
            None,
            causal_impact.params["alpha"],
        )

        result_data = result["data"]
        expected_data = expected["data"]
        assert_frame_equal(result_data, expected_data)

        result_model_args = result["model_args"]
        expected_model_args = expected["model_args"]
        assert result_model_args == expected_model_args

        result_other = {
            key: result[key] for key in result if key not in {"model_args", "data"}
        }

        expected_other = {
            key: expected[key] for key in expected if key not in {"model_args", "data"}
        }
        assert result_other == expected_other

    def test_input_w_time_column(self):
        data = pd.DataFrame(np.random.randn(100, 2), columns=["x1", "x2"])
        data["time"] = pd.date_range(start="2018-01-01", periods=100)
        data = data[["time", "x1", "x2"]]
        pre_period = ["2018-01-01", "2018-02-10"]
        post_period = ["2018-02-11", "2018-4-10"]

        causal_impact = CausalImpact(data, pre_period, post_period, {})

        data = data.set_index("time")
        pre_period = [pd.to_datetime(e) for e in pre_period]
        post_period = [pd.to_datetime(e) for e in post_period]

        expected = {
            "data": data,
            "pre_period": pre_period,
            "post_period": post_period,
            "model_args": causal_impact.params["model_args"],
            "ucm_model": None,
            "post_period_response": None,
            "alpha": causal_impact.params["alpha"],
        }
        result = causal_impact._format_input(
            causal_impact.params["data"],
            causal_impact.params["pre_period"],
            causal_impact.params["post_period"],
            causal_impact.params["model_args"],
            None,
            None,
            causal_impact.params["alpha"],
        )

        result_data = result["data"]
        expected_data = expected["data"]
        assert_frame_equal(result_data, expected_data)

        result_model_args = result["model_args"]
        expected_model_args = expected["model_args"]
        assert result_model_args == expected_model_args

        result_other = {
            key: result[key] for key in result if key not in {"model_args", "data"}
        }

        expected_other = {
            key: expected[key] for key in expected if key not in {"model_args", "data"}
        }
        assert result_other == expected_other

    def test_input_w_just_2_points_raises_exception(self):
        data = pd.DataFrame(np.random.randn(2, 2), columns=["x1", "x2"])
        causal_impact = CausalImpact(data, [0, 0], [1, 1], {})

        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                causal_impact.params["data"],
                causal_impact.params["pre_period"],
                causal_impact.params["post_period"],
                causal_impact.params["model_args"],
                None,
                None,
                causal_impact.params["alpha"],
            )
        assert str(excinfo.value) == "data must have at least 3 time points"

    def test_input_covariates_w_nan_value_raises(self):
        data = np.array([[1, 1, 2], [1, 2, 3], [1, 3, 4], [1, np.nan, 5], [1, 6, 7]])
        data = pd.DataFrame(data, columns=["y", "x1", "x2"])
        causal_impact = CausalImpact(data, [0, 3], [3, 4], {})

        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                causal_impact.params["data"],
                causal_impact.params["pre_period"],
                causal_impact.params["post_period"],
                causal_impact.params["model_args"],
                None,
                None,
                causal_impact.params["alpha"],
            )
        assert str(excinfo.value) == "covariates must not contain null values"

    def test_int_index_pre_period_contains_float(self, causal_impact, pre_period):
        expected = {
            "data": causal_impact.params["data"],
            "pre_period": causal_impact.params["pre_period"],
            "post_period": causal_impact.params["post_period"],
            "model_args": causal_impact.params["model_args"],
            "ucm_model": None,
            "post_period_response": None,
            "alpha": causal_impact.params["alpha"],
        }
        result = causal_impact._format_input(
            causal_impact.params["data"],
            [float(pre_period[0]), pre_period[1]],
            causal_impact.params["post_period"],
            causal_impact.params["model_args"],
            None,
            None,
            causal_impact.params["alpha"],
        )

        result_data = result["data"]
        expected_data = expected["data"]
        assert_frame_equal(result_data, expected_data)

        result_model_args = result["model_args"]
        expected_model_args = expected["model_args"]
        assert result_model_args == expected_model_args

        result_other = {
            key: result[key] for key in result if key not in {"model_args", "data"}
        }

        expected_other = {
            key: expected[key] for key in expected if key not in {"model_args", "data"}
        }
        assert result_other == expected_other

    def test_float_index_pre_period_contains_int(self):
        data = np.random.randn(200, 3)
        data = pd.DataFrame(data, columns=["y", "x1", "x2"])
        data = data.set_index(np.array([float(i) for i in range(200)]))
        causal_impact = CausalImpact(data, [0, 3], [3, 4], {})

        expected = {
            "data": causal_impact.params["data"],
            "pre_period": causal_impact.params["pre_period"],
            "post_period": causal_impact.params["post_period"],
            "model_args": causal_impact.params["model_args"],
            "ucm_model": None,
            "post_period_response": None,
            "alpha": causal_impact.params["alpha"],
        }
        result = causal_impact._format_input(
            causal_impact.params["data"],
            causal_impact.params["pre_period"],
            causal_impact.params["post_period"],
            causal_impact.params["model_args"],
            None,
            None,
            causal_impact.params["alpha"],
        )

        result_data = result["data"]
        expected_data = expected["data"]
        assert_frame_equal(result_data, expected_data)

        result_model_args = result["model_args"]
        expected_model_args = expected["model_args"]
        assert result_model_args == expected_model_args

        result_other = {
            key: result[key] for key in result if key not in {"model_args", "data"}
        }
        expected_other = {
            key: expected[key] for key in expected if key not in {"model_args", "data"}
        }
        assert result_other == expected_other

    def test_pre_period_in_conflict_w_post_period(self):
        data = pd.DataFrame(np.random.randn(20, 2), columns=["x1", "x2"])
        causal_impact = CausalImpact(data, [0, 10], [9, 20], {})

        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                causal_impact.params["data"],
                causal_impact.params["pre_period"],
                causal_impact.params["post_period"],
                causal_impact.params["model_args"],
                None,
                None,
                causal_impact.params["alpha"],
            )
        assert str(excinfo.value) == (
            "post period must start at least 1 observation after the end of "
            "the pre_period"
        )

        causal_impact = CausalImpact(data, [0, 10], [11, 9], {})
        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                causal_impact.params["data"],
                causal_impact.params["pre_period"],
                causal_impact.params["post_period"],
                causal_impact.params["model_args"],
                None,
                None,
                causal_impact.params["alpha"],
            )
        assert str(excinfo.value) == (
            "post_period[1] must not be earlier than post_period[0]"
        )

        causal_impact = CausalImpact(data, [0, 10], [11, 9], {})
        with pytest.raises(ValueError) as excinfo:
            causal_impact._format_input(
                causal_impact.params["data"],
                causal_impact.params["pre_period"],
                causal_impact.params["post_period"],
                causal_impact.params["model_args"],
                None,
                None,
                causal_impact.params["alpha"],
            )
        assert str(excinfo.value) == (
            "post_period[1] must not be earlier than post_period[0]"
        )


class TestRunWithData(object):
    def test_missing_input(self):
        with pytest.raises(SyntaxError):
            impact = CausalImpact()
            impact.run()

    def test_unlabelled_pandas_series(self, expected_columns, pre_period, post_period):
        model_args = {"niter": 123, "standardize_data": False}
        alpha = 0.05
        data = pd.DataFrame(np.random.randn(200, 3))
        causal_impact = CausalImpact(
            data.values, pre_period, post_period, model_args, None, None, alpha, "MLE"
        )

        causal_impact.run()
        actual_columns = list(causal_impact.inferences.columns)
        assert actual_columns == expected_columns

    def test_other_formats(self, expected_columns, pre_period, post_period):
        # Test other data formats
        model_args = {"niter": 100, "standardize_data": True}

        # labelled dataframe
        data = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
        impact = CausalImpact(data, pre_period, post_period, model_args)
        impact.run()
        actual_columns = list(impact.inferences.columns)
        assert actual_columns == expected_columns

        # numpy array
        data = np.random.randn(200, 3)
        impact = CausalImpact(data, pre_period, post_period, model_args)
        impact.run()
        actual_columns = list(impact.inferences.columns)
        assert actual_columns == expected_columns

        # list of lists
        data = np.random.randn(200, 2).tolist()
        impact = CausalImpact(data, pre_period, post_period, model_args)
        impact.run()
        actual_columns = list(impact.inferences.columns)
        assert actual_columns == expected_columns

    def test_frame_w_no_exog(self, pre_period, post_period):
        data = np.random.randn(200)
        impact = CausalImpact(data, pre_period, post_period, {})
        with pytest.raises(ValueError) as excinfo:
            impact.run()
        assert str(excinfo.value) == "data contains no exogenous variables"

    def test_missing_pre_period_data(self, data, pre_period, post_period):
        model_data = data.copy()
        model_data.iloc[3:5, 0] = np.nan
        impact = CausalImpact(model_data, pre_period, post_period)
        impact.run()
        assert len(impact.inferences) == len(model_data)

    def test_pre_period_starts_after_beginning_of_data(self, data):
        pre_period = [3, 100]
        impact = CausalImpact(data, pre_period, [101, 199])
        impact.run()
        np.testing.assert_array_equal(impact.inferences.response.values, data.y.values)
        assert np.all(pd.isnull(impact.inferences.iloc[0 : pre_period[0], 2:]))

    def test_post_period_finishes_before_end_of_data(self, data, pre_period):
        post_period = [101, 197]
        impact = CausalImpact(data, pre_period, post_period)
        impact.run()
        np.testing.assert_array_equal(impact.inferences.response.values, data.y.values)
        assert np.all(pd.isnull(impact.inferences.iloc[-2:, 2:]))

    def test_gap_between_pre_and_post_periods(self, data, pre_period):
        post_period = [120, 199]
        impact = CausalImpact(data, pre_period, post_period)
        impact.run()
        assert np.all(
            pd.isnull(impact.inferences.loc[101:119, impact.inferences.columns[2:]])
        )

    def test_late_start_early_finish_and_gap_between_periods(self, data):
        pre_period = [3, 80]
        post_period = [120, 197]
        impact = CausalImpact(data, pre_period, post_period)
        impact.run()
        assert np.all(
            pd.isnull(impact.inferences.loc[:2, impact.inferences.columns[2:]])
        )
        assert np.all(
            pd.isnull(impact.inferences.loc[81:119, impact.inferences.columns[2:]])
        )
        assert np.all(
            pd.isnull(impact.inferences.loc[198:, impact.inferences.columns[2:]])
        )

    def test_pre_period_lower_than_data_index_min(self, data):
        pre_period = [-1, 100]
        post_period = [101, 199]
        impact = CausalImpact(data, pre_period, post_period)
        impact.run()
        assert impact.params["pre_period"] == [0, 100]

    def test_post_period_bigger_than_data_index_max(self, data):
        pre_period = [0, 100]
        post_period = [101, 300]
        impact = CausalImpact(data, pre_period, post_period)
        impact.run()
        assert impact.params["post_period"] == [101, 199]

    def test_missing_values_in_pre_period_y(self, pre_period, post_period):
        data = pd.DataFrame(np.random.randn(200, 3), columns=["y", "x1", "x2"])
        data.iloc[95:100, 0] = np.nan

        impact = CausalImpact(data, pre_period, post_period)
        impact.run()

        """Test that all columns in the result series except the
        point predictions have missing values at the time points the
        result time series has missing values."""

        predicted_cols = [
            impact.inferences.columns.get_loc(col)
            for col in impact.inferences.columns
            if ("response" not in col and "point_effect" not in col)
        ]

        effect_cols = [
            impact.inferences.columns.get_loc(col)
            for col in impact.inferences.columns
            if "point_effect" in col
        ]

        response_cols = [
            impact.inferences.columns.get_loc(col)
            for col in impact.inferences.columns
            if "response" in col
        ]

        assert np.all(np.isnan(impact.inferences.iloc[95:100, response_cols]))
        assert (np.all(np.isnan(impact.inferences.iloc[95:100, effect_cols])))
        TestCase().assertFalse(np.any(np.isnan(impact.inferences.iloc[95:100, predicted_cols])))
        TestCase().assertFalse(np.any(np.isnan(impact.inferences.iloc[:95, :])))
        TestCase().assertFalse(np.any(np.isnan(impact.inferences.iloc[101:, :])))


class TestRunWithUCM(object):
    def test_regular_run(self, expected_columns, impact_ucm):
        impact_ucm.run()
        actual_columns = list(impact_ucm.inferences.columns)
        assert actual_columns == expected_columns


class TestSummary(object):
    def test_summary(self, inference_input):
        inferences_df = pd.DataFrame(inference_input)
        causal = CausalImpact()

        params = {"alpha": 0.05, "post_period": [2, 4]}

        causal.params = params
        causal.inferences = inferences_df

        expected = [
            [3, 7],
            [3, 7],
            [[3, 3], [7, 7]],
            [" ", " "],
            [0, 0],
            [[0, 0], [0, 0]],
            [" ", " "],
            ["-2.8%", "-2.8%"],
            [["0.0%", "-11.1%"], ["0.0%", "-11.1%"]],
            [" ", " "],
            ["0.0%", " "],
            ["100.0%", " "],
        ]

        expected = pd.DataFrame(
            expected,
            columns=["Average", "Cumulative"],
            index=[
                "Actual",
                "Predicted",
                "95% CI",
                " ",
                "Absolute Effect",
                "95% CI",
                " ",
                "Relative Effect",
                "95% CI",
                " ",
                "P-value",
                "Prob. of Causal Effect",
            ],
        )

        tmpdir = mkdtemp()
        tmp_expected = "tmp_expected"
        tmp_result = "tmp_test_summary"

        result_file = os.path.join(tmpdir, tmp_result)
        expected_file = os.path.join(tmpdir, tmp_expected)

        expected.to_csv(expected_file)
        expected_str = open(expected_file).read()

        causal.summary(path=result_file)

        result = open(result_file).read()
        assert result == expected_str

    def test_summary_w_report_output(
        self, monkeypatch, inference_input, summary_report_filename
    ):
        inferences_df = pd.DataFrame(inference_input)
        causal = CausalImpact()

        params = {"alpha": 0.05, "post_period": [2, 4]}

        causal.params = params
        causal.inferences = inferences_df

        dedent_mock = mock.Mock()

        expected = open(summary_report_filename).read()
        expected = re.sub(r"\s+", " ", expected)
        expected = expected.strip()

        tmpdir = mkdtemp()
        tmp_file = os.path.join(tmpdir, "summary_test")

        def dedent_side_effect(msg):
            with open(tmp_file, "a") as file_obj:
                msg = re.sub(r"\s+", " ", msg)
                msg = msg.strip()
                file_obj.write(msg)
            return msg

        dedent_mock.side_effect = dedent_side_effect
        monkeypatch.setattr("textwrap.dedent", dedent_mock)

        causal.summary(output="report")
        result_str = open(tmp_file, "r").read()
        assert result_str == expected

    def test_summary_wrong_argument_raises(self, inference_input):
        inferences_df = pd.DataFrame(inference_input)
        causal = CausalImpact()

        params = {"alpha": 0.05, "post_period": [2, 4]}

        causal.params = params
        causal.inferences = inferences_df

        with pytest.raises(ValueError):
            causal.summary(output="wrong_argument")


class TestPlot(object):
    # @patch('causalimpact.data')
    def test_plot(self, monkeypatch):

        params = {"alpha": 0.05, "post_period": [2, 4], "pre_period": [0, 1]}
        inferences_mock = mock.MagicMock()
        class EnhancedDict():
            @property
            def index(self):
                return [0, 1]
            @property
            def response(self):
                return "y obs"
            @property
            def point_pred(self):
                return "points predicted"
            @property
            def point_pred_lower(self):
                return "lower predictions"
            @property
            def point_pred_upper(self):
                return "upper predictions"
            @property
            def point_effect(self):
                return "lift"
            @property
            def point_effect_lower(self):
                return "point effect lower"
            @property
            def point_effect_upper(self):
                return "point effect upper"
            @property
            def cum_effect(self):
                return "cum effect"
            @property
            def cum_effect_lower(self):
                return "cum effect lower"
            @property
            def cum_effect_upper(self):
                return "cum effect upper"
        def getitem(name):
            print(name)
            return EnhancedDict()
        inferences_mock.iloc.__getitem__.side_effect = getitem
        class Data(object):
            @property
            def index(self):
                return 'index'

            @property
            def shape(self):
                return [(1, 2)]
        data_mock = Data()

        plot_mock = mock.Mock()
        np_zeros_mock = mock.Mock()
        np_zeros_mock.side_effect = lambda x: [0, 0]
        
        get_lib_mock = mock.Mock(return_value=plot_mock)
        monkeypatch.setattr("causalimpact.analysis.get_matplotlib", get_lib_mock)

        monkeypatch.setattr("numpy.zeros", np_zeros_mock)

        causal = CausalImpact()
        causal.params = params
        causal.inferences = inferences_mock
        causal.data = data_mock

        causal.plot(panels=["original", "pointwise", "cumulative"])
        causal.plot(panels=["pointwise", "cumulative"])

        causal.plot(panels=["original"])
        plot_mock.plot.assert_any_call("y obs", "k", label="endog", linewidth=2)
        plot_mock.plot.assert_any_call(
            "points predicted", "r--", label="model", linewidth=2
        )

        plot_mock.fill_between.assert_any_call(
            [0, 1],
            "lower predictions",
            "upper predictions",
            facecolor="gray",
            interpolate=True,
            alpha=0.25,
        )

        causal.plot(panels=["pointwise"])

        plot_mock.plot.assert_any_call("lift", "r--", linewidth=2)
        plot_mock.plot.assert_any_call("index", [0, 0], "g-", linewidth=2)

        causal.plot(panels=["cumulative"])

        plot_mock.plot.assert_any_call([0, 1], "cum effect", "r--", linewidth=2)
        plot_mock.plot.assert_any_call("index", [0, 0], "g-", linewidth=2)
