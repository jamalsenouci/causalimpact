"""Unit Tests for analysis module."""

from nose.tools import assert_equal
from nose.tools import assert_raises
import pytest

import numpy as np
import pandas as pd
from pandas.core.common import PandasError
from pandas.util.testing import assert_frame_equal
from statsmodels.tsa.statespace.structural import UnobservedComponents as UCM

from causalimpact import CausalImpact
format_input = CausalImpact._format_input


_expected_columns = ["response", "cum_response",
                     "point_pred", "point_pred_upper", "point_pred_lower",
                     "cum_pred", "cum_pred_lower", "cum_pred_upper",
                     "point_effect", "point_effect_lower",
                     "point_effect_upper", "cum_effect",
                     "cum_effect_lower", "cum_effect_upper"]


data = pd.DataFrame(np.random.randn(200, 3), columns=["y", "x1", "x2"])
pre_period = [0, 100]
post_period = [100, 200]
model_args = {"niter": 123}
ucm_model = UCM(endog=data.iloc[:, 0].values, level="llevel")
post_period_response = np.random.randn(100)
alpha = 0.05
impact_data = CausalImpact(data, pre_period, post_period, model_args, None,
                           None, alpha, "MLE")
impact_ucm = CausalImpact(None, None, None, None, ucm_model,
                          post_period_response, alpha,
                          "MLE")


class TestFormatInput(object):
    """Tests for formatting input for CausalImpact."""

    def test_data_input(self):
        # Test data input (usage scenario 1)
        expected = {"data": data, "pre_period": pre_period,
                    "post_period": post_period,
                    "model_args": model_args, "ucm_model": None,
                    "post_period_response": None, "alpha": alpha}
        result = format_input(impact_data,
                              impact_data.params["data"],
                              impact_data.params["pre_period"],
                              impact_data.params["post_period"],
                              impact_data.params["model_args"], None,
                              None, impact_data.params["alpha"])

        result_data = result["data"]
        expected_data = expected["data"]
        result_model_args = result["model_args"]
        expected_model_args = expected["model_args"]
        result_other = {key: result[key] for key in result
                        if key not in ["model_args", "data"]}
        expected_other = {key: expected[key] for key in expected
                          if key not in ["model_args", "data"]}

        assert_frame_equal(result_data, expected_data)
        assert_equal(result_model_args["niter"], expected_model_args["niter"])
        assert_equal(result_other, expected_other)

    def test_ucm_input(self):
        # Test ucm_model input (usage scenario 2)
        expected = {"data": None, "pre_period": None, "post_period": None,
                    "model_args": None, "ucm_model": ucm_model,
                    "post_period_response": post_period_response,
                    "alpha": alpha}
        checked = format_input(impact_ucm, None, None, None, None,
                               impact_ucm.params["ucm_model"],
                               impact_ucm.params["post_period_response"],
                               impact_ucm.params["alpha"])

        checked_other = {key: checked[key] for key in checked
                         if key not in ["model_args"]}
        expected_other = {key: expected[key] for key in expected
                          if key not in ["model_args"]}

        assert_equal(checked_other, expected_other)

    def test_inconsistency_raises_error(self):
        # Test inconsistent input (must not provide both data and ucm_model)
        assert_raises(SyntaxError, format_input, impact_data, data,
                      pre_period, post_period, model_args,
                      ucm_model, post_period_response,
                      alpha)

    def test_format_output_is_df(self):
        # Test that <data> is converted to pandas DataFrame
        expected_data = pd.DataFrame(np.arange(0, 8).reshape(4, 2),
                                     index=[1, 2, 3, 4])
        funny_datas = [pd.DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]]),
                       pd.DataFrame(data=[[0, 1], [2, 3], [4, 5], [6, 7]],
                                    index=[1, 2, 3, 4]),
                       [[0, 1], [2, 3], [4, 5], [6, 7]],
                       np.array([[0, 1], [2, 3], [4, 5], [6, 7]])]

        for funny_data in funny_datas:
            checked = format_input(impact_data, funny_data, [0, 3],
                                   [3, 3], model_args, None, None,
                                   alpha)
            assert(np.all(np.equal(checked["data"].values,
                                   expected_data.values)))

    def test_bad_data(self):
        # Test bad <data>
        text_data = "foo"
        assert_raises(PandasError, format_input, impact_data, text_data,
                      [0, 3], [3, 3], model_args, None, None, alpha)

    def test_bad_pre_period(self):
        # Test bad <pre_period>
        bad_pre_periods = [1, [1, 2, 3], [np.nan, 2],
                           [pd.to_datetime(date) for date in ["2011-01-01",
                                                              "2011-12-31"]]]
        for bad_pre_period in bad_pre_periods:
            assert_raises(ValueError, format_input, impact_data,
                          data, bad_pre_period, post_period,
                          model_args, None, None, alpha)

    def test_bad_post_period(self):
        # Test bad <post_period>
        bad_post_periods = [1, [1, 2, 3], [np.nan, 2],
                            [pd.to_datetime(date) for date in ["2011-01-01",
                                                               "2011-12-31"]]]
        for bad_post_period in bad_post_periods:
            assert_raises(ValueError, format_input, impact_data,
                          data, pre_period, bad_post_period,
                          model_args, None, None, alpha)

    def test_period_diff_class(self):
        # Test what happens when pre_period/post_period has a different class
        # than the timestamps in <data>
        bad_data = pd.DataFrame(data=[1, 2, 3, 4],
                                index=["2014-01-01", "2014-01-02",
                                       "2014-01-03", "2014-01-04"])
        bad_pre_period = [0, 3]  # float
        bad_post_period = [3, 3]

        assert_raises(ValueError, format_input, impact_data, bad_data,
                      bad_pre_period, bad_post_period, model_args, None,
                      None, alpha)

        bad_pre_period = [int(0), int(2)]  # integer
        bad_post_period = [int(3), int(3)]
        assert_raises(ValueError, format_input, impact_data, bad_data,
                      bad_pre_period, bad_post_period, model_args, None,
                      None, alpha)

    def test_bad_model_args(self):
        # Test bad <model.args>
        bad_model_args = [1000, "niter = 1000"]
        for bad_model_arg in bad_model_args:
            assert_raises(TypeError, format_input, impact_data, data,
                          pre_period, post_period,
                          bad_model_arg, None, None, alpha)

    def test_bad_standardize(self):
        bad_standardize_data = [np.nan, 123, "foo", [True, False]]
        for bad_standardize in bad_standardize_data:
            bad_model_args = {"standardize_data": bad_standardize}
            assert_raises(ValueError, format_input, impact_data,
                          data, pre_period, post_period,
                          bad_model_args, None, None, alpha)

    """ Test bad <ucm.model>
    def test_bad_ucm():
        bad_ucm_models = [None, np.nan, 1, [1, 2, 3]]
        for bad_ucm_model in bad_ucm_models:
            assert_raises(ValueError, format_input, None, None,
                          None, None, bad_ucm_model,
                          post_period_response, alpha)
    """

    def test_bad_post_period_response(self):
        # Test bad <post.period.response>
        # Note that consistency with ucm.model is not tested in format_input()
        bad_post_period_response = [pd.to_datetime("2011-01-01"), True]
        for bad_response in bad_post_period_response:
            assert_raises(ValueError, format_input, impact_ucm, None,
                          None, None, None, ucm_model, bad_response,
                          alpha)

    def test_bad_alpha(self):
        # Test bad <alpha>
        bad_alphas = [None, np.nan, -1, 0, 1, [0.8, 0.9], "0.1"]
        for bad_alpha in bad_alphas:
            assert_raises(ValueError, format_input, impact_data,
                          data, pre_period, post_period,
                          model_args, None, None, bad_alpha)


class TestRunWithData(object):
    def test_missing_input(self):
        impact = CausalImpact()
        assert_raises(SyntaxError, impact.run)

    def test_unlabelled_pandas_series(self):
        impact_data.run()
        actual_columns = impact_data.inferences.columns
        assert np.all(np.equal(actual_columns, _expected_columns))

    def test_other_formats(self):
        # Test other data formats
        pre_period = [1, 100]
        post_period = [101, 200]
        model_args = {"niter": 100}

        # labelled dataframe
        data = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
        impact = CausalImpact(data, pre_period, post_period, model_args)
        impact.run()
        actual_columns = impact.inferences.columns
        assert actual_columns[0] == "response"

        # numpy array
        data = np.random.randn(200, 3)
        impact = CausalImpact(data, pre_period, post_period, model_args)
        impact.run()
        actual_columns = impact.inferences.columns
        assert actual_columns[0] == "response"

        # numpy array
        data = np.random.randn(200, 3)
        impact = CausalImpact(data, pre_period, post_period, model_args)
        impact.run()
        actual_columns = impact.inferences.columns
        assert actual_columns[0] == "response"

        # list of lists
        data = [[n, n + 2] for n in range(200)]
        impact = CausalImpact(data, pre_period, post_period, model_args)
        impact.run()
        actual_columns = impact.inferences.columns
        assert actual_columns[0] == "response"

    # Data frame with no exogenous
    def test_frame_no_exog(self):
        data = np.random.randn(200)
        impact = CausalImpact(data, pre_period, post_period, model_args)
        with pytest.raises(ValueError):
            impact.run()
"""
    def test_missing_pre_period_data(self):
        data.iloc[3:5, 0] = np.nan
        impact = CausalImpact(data, pre_period, post_period, model_args)
"""
