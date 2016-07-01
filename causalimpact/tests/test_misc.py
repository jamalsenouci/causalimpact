"""Tests for misc module."""

import causalimpact
import numpy as np
import pandas as pd

from nose.tools import assert_equal
from nose.tools import assert_raises
from pandas.util.testing import assert_frame_equal
standardize = causalimpact.misc.standardize_all_variables
unstandardize = causalimpact.misc.unstandardize


class test_standardize(object):
    """Tests for standardize function."""
    assert_raises(TypeError, standardize)

    def test_standardize_basic(self):
        """test types produced."""
        data = [-1, 0.1, 1, 2, np.nan, 3]
        data = pd.DataFrame(data)
        result = standardize(data)
        assert_equal(type(result), dict)
        assert_equal(set(result.keys()), set(["data", "orig_std_params"]))
        assert_frame_equal(unstandardize(**result), pd.DataFrame(data))

    def test_standardize_maths(self):
        """test numeric output."""
        data = [1, 2, 3]
        data = pd.DataFrame(data)
        np.testing.assert_array_equal(standardize(data)["data"], pd.DataFrame([-1, 0, 1]))

    def test_standardize_inputs(self):
        """test various input types."""
        test_data = [[1], [1, 1, 1], [1, np.nan, 3], pd.DataFrame([10, 20, 30])]
        test_data = [pd.DataFrame(data, dtype="float") for data in test_data]
        for data in test_data:
            result = standardize(data)
            pd.util.testing.assert_frame_equal(unstandardize(**result), data)

    def test_standardize_bad_input(self):
        """test TypeError with text input."""
        assert_raises(AttributeError, standardize,"text")
