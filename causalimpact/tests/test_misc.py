import numpy as np
import pandas as pd
import causalimpact

from nose.tools import assert_equal
from nose.tools import assert_raises
standardize = causalimpact.misc.standardize

class test_standardize(object):
    
    assert_raises(TypeError, standardize)
    
    def test_standardize_basic(self):
        data = [-1, 0.1, 1, 2, np.nan, 3]
        result = standardize(data)
        assert_equal(type(result), dict)
        assert_equal(result.keys(), dict_keys(["y", "unstandardize"]))
        assert_equal(result["unstandardize"], data)
    
    def test_standardize_maths(self):
        print(standardize([1, 2, 3])["y"])
        assert(np.all(np.equal(standardize([1, 2, 3])["y"], [-1, 0, 1])))
    
    def test_standardize_inputs(self):
        test_data = [[1], [1, 1, 1], np.nan, [1, np.nan, 3], pd.DataFrame([10, 20, 30])]
        for data in test_data:
            result = standardize(data)
            assert_equal(result["unstandardize"], data)
    
    def test_standardize_bad_input(self):
        assert_raises(TypeError, standardize("text"))
