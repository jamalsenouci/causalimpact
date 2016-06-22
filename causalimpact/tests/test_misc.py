"""Tests for misc module."""

import causalimpact
import numpy as np
import pandas as pd

from nose.tools import assert_equal
from nose.tools import assert_raises
standardize = causalimpact.misc.standardize


class test_standardize(object):
    """Tests for standardize function."""
    assert_raises(TypeError, standardize)

    def test_standardize_basic(self):
        """test types produced."""
        data = [-1, 0.1, 1, 2, np.nan, 3]
        result = standardize(data)
        assert_equal(type(result), dict)
        assert_equal(set(result.keys()), set(["y", "unstandardize"]))
        np.testing.assert_array_almost_equal(result["unstandardize"],np.array(data))

    def test_standardize_maths(self):
        """test numeric output."""
        np.testing.assert_array_equal(standardize([1, 2, 3])["y"], [-1, 0, 1])

    def test_standardize_inputs(self):
        """test various input types."""
        test_data = [[1], [1, 1, 1], [1, np.nan, 3], pd.DataFrame([10, 20, 30])]
        for data in test_data:
            result = standardize(data)
            np.testing.assert_array_equal(result["unstandardize"], data)

    def test_standardize_bad_input(self):
        """test TypeError with text input."""
        assert_raises(TypeError, standardize,"text")
