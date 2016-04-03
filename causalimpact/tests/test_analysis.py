"""
Unit Tests for analysis module
"""

from nose.tools import assert_raises, assert_equal

import pandas as pd
from pandas.core.common import PandasError
from pandas.util.testing import assert_frame_equal

import numpy as np
from datetime import datetime
from causalimpact.analysis import _format_input as format_input

_expected_columns = ["response", "cum.response",
                     "point.pred", "point.pred.lower", "point.pred.upper",
                     "cum.pred", "cum.pred.lower", "cum.pred.upper",
                     "point.effect", "point.effect.lower",
                     "point.effect.upper", "cum.effect", "cum.effect.lower",
                     "cum.effect.upper"]


class TestFormatInput(object):

    def __init__(self):
        # Specify some healthy input variables
        self.data = pd.DataFrame(np.random.randn(200, 3),
                                 columns=["y", "x1", "x2"])
        self.pre_period = [0, 100]
        self.post_period = [100, 200]
        self.model_args = {"niter": 123}
        self.bsts_model = []
        self.post_period_response = np.random.randn(100)
        self.alpha = 0.05

    # Test data input (usage scenario 1)
    def test_data_input(self):
        expected = {"data": self.data, "pre_period": self.pre_period,
                    "post_period": self.post_period,
                    "model_args": self.model_args, "bsts_model": None,
                    "post_period_response": None, "alpha": self.alpha}

        result = format_input(self.data, self.pre_period, self.post_period,
                              self.model_args, None, None, self.alpha)

        result_data = result["data"]
        expected_data = expected["data"]
        result_model_args = result["model_args"]
        expected_model_args = expected["model_args"]
        result_other = {key: result[key] for key in result if key not in ["model_args", "data"]}
        expected_other = {key: expected[key] for key in expected if key not in ["model_args", "data"]}

        assert_frame_equal(result_data, expected_data)
        assert_equal(result_model_args["niter"], expected_model_args["niter"])
        assert_equal(result_other, expected_other)

    # Test bsts.model input (usage scenario 2)
    def test_bsts_input(self):
        expected = {"data": None, "pre_period": None, "post_period": None, "model_args": None, "bsts_model": self.bsts_model,
                    "post_period_response": self.post_period_response, "alpha": self.alpha}
        checked = format_input(None, None, None, None, self.bsts_model,
                               self.post_period_response, self.alpha)

        checked_other = {key: checked[key] for key in checked if key not in ["model_args"]}
        expected_other = {key: expected[key] for key in expected if key not in ["model_args"]}

        assert_equal(checked_other, expected_other)

    # Test inconsistent input (must not provide both data and bsts.model)
    def test_inconsistency_raises_error(self):
        assert_raises(SyntaxError, format_input, self.data, self.pre_period,
                      self.post_period, self.model_args, self.bsts_model,
                      self.post_period_response, self.alpha)

    # Test that <data> is converted to pandas DataFrame
    def test_format_output_is_df(self):
        expected_data = pd.DataFrame([10, 20, 30, 40], index=[1, 2, 3, 4])
        funny_datas = [pd.DataFrame([10, 20, 30, 40]),
                       pd.DataFrame(data=[10, 20, 30, 40],
                                    index=[1, 2, 3, 4]),
                       [10, 20, 30, 40],
                       [int(float) for float in [10, 20, 30, 40]],
                       np.array([10, 20, 30, 40])
                       ]

        for funny_data in funny_datas:
            checked = format_input(funny_data, [0, 3], [3, 3], self.model_args,
                                   None, None, self.alpha)
            assert(np.all(np.equal(checked["data"].values, expected_data.values)))

    # Test bad <data>
    def test_bad_data(self):
        text_data = "foo"
        assert_raises(PandasError, format_input, text_data, [0, 3], [3, 3],
                      self.model_args, None, None, self.alpha)

    # Test bad <pre_period>
    def test_bad_pre_period(self):
        bad_pre_periods = [1,
                           [1, 2, 3],
                           [np.nan, 2],
                           [datetime.strptime(date, "%Y-%M-%d") for date in
                            ["2011-01-01", "2011-12-31"]
                            ]
                           ]
        for bad_pre_period in bad_pre_periods:
            assert_raises(ValueError, format_input, self.data, bad_pre_period,
                          self.post_period, self.model_args, None, None,
                          self.alpha)

    # Test bad <post.period>
    def test_bad_post_period(self):
        bad_post_periods = [1,
                            [1, 2, 3],
                            [np.nan, 2],
                            [datetime.strptime(date, "%Y-%M-%d") for date in ["2011-01-01", "2011-12-31"]
                             ]
                            ]
        for bad_post_period in bad_post_periods:
            assert_raises(ValueError, format_input, self.data, self.pre_period,
                          bad_post_period, self.model_args, None, None,
                          self.alpha)

    # Test what happens when pre.period/post.period has a different class than
    # the timestamps in <data>

    def test_period_diff_class(self):
        bad_data = pd.DataFrame(data=[1, 2, 3, 4],
                                index=["2014-01-01", "2014-01-02",
                                       "2014-01-03", "2014-01-04"])
        bad_pre_period = [0, 3]  # float
        bad_post_period = [3, 3]

        assert_raises(ValueError, format_input, bad_data, bad_pre_period,
                      bad_post_period, self.model_args, None, None,
                      self.alpha)

        bad_pre_period = [int(0), int(2)]  # integer
        bad_post_period = [int(3), int(3)]
        assert_raises(ValueError, format_input, bad_data, bad_pre_period,
                      bad_post_period, self.model_args, None, None,
                      self.alpha)

    # Test bad <model.args>
    def test_bad_model_args(self):
        bad_model_args = [1000, "niter = 1000"]
        for bad_model_arg in bad_model_args:
            assert_raises(TypeError, format_input, self.data, self.pre_period,
                          self.post_period, bad_model_arg, None, None,
                          self.alpha)

    def test_bad_standardize(self):
        bad_standardize_data = [np.nan, 123, "foo", [True, False]]
        for bad_standardize in bad_standardize_data:
            bad_model_args = {"standardize_data": bad_standardize}
            assert_raises(ValueError, format_input, self.data, self.pre_period,
                          self.post_period, bad_model_args, None, None,
                          self.alpha)

    """ Test bad <bsts.model>
    def test_bad_bsts(self):
        bad_bsts_models = [None, np.nan, 1, [1, 2, 3]]
        for bad_bsts_model in bad_bsts_models:
            assert_raises(ValueError, format_input, None, None, None, None,
                          bad_bsts_model, self.post_period_response, self.alpha)
    """

    # Test bad <post.period.response>
    # Note that consistency with bsts.model is not tested in format_input()
    def test_bad_post_period_response(self):
        bad_post_period_response = [datetime.strptime("2011-01-01", "%Y-%M-%d"), True]
        for bad_response in bad_post_period_response:
            print(bad_response)
            assert_raises(ValueError, format_input, None, None, None, None,
                          self.bsts_model, bad_response, self.alpha)

    # Test bad <alpha>
    def test_bad_alpha(self):
        bad_alphas = [None, np.nan, -1, 0, 1, [0.8, 0.9], "0.1"]
        for bad_alpha in bad_alphas:
            assert_raises(ValueError, format_input, self.data, self.pre_period,
                          self.post_period, self.model_args, None, None,
                          bad_alpha)
