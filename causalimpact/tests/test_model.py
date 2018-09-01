"""Unit Tests for model module"""


import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.arima_process import ArmaProcess

import causalimpact

observations_validate = causalimpact.model.observations_ill_conditioned
construct_model = causalimpact.model.construct_model


def test_raises_when_y_is_None():
    with pytest.raises(ValueError) as excinfo:
        observations_validate(None)
    assert str(excinfo.value) == 'y cannot be None'


def test_raises_when_y_has_len_1():
    with pytest.raises(ValueError) as excinfo:
        observations_validate([1])
    assert str(excinfo.value) == 'y must have len > 1'


def test_raises_when_y_is_all_nan():
    with pytest.raises(ValueError) as excinfo:
        observations_validate([np.nan, np.nan])
    assert str(excinfo.value) == ('Aborting inference due to input series '
        'being all null.')


def test_raises_when_y_have_just_2_values():
    with pytest.raises(ValueError) as excinfo:
        observations_validate(pd.DataFrame([1, 2]))
    assert str(excinfo.value) == ('Aborting inference due to fewer than 3 '
        'nonnull values in input.')


def test_raises_when_y_is_constant():
    with pytest.raises(ValueError) as excinfo:
        observations_validate(pd.Series([1, 1, 1, 1, 1]))
    assert str(excinfo.value) == ('Aborting inference due to input series '
        'being constant')



