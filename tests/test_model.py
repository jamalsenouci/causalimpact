"""Unit Tests for model module"""


import pytest
import mock
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

import causalimpact

observations_validate = causalimpact.model.observations_ill_conditioned
construct_model = causalimpact.model.construct_model
model_fit = causalimpact.model.model_fit


def test_raises_when_y_is_None():
    with pytest.raises(ValueError) as excinfo:
        observations_validate(None)
    assert str(excinfo.value) == "y cannot be None"


def test_raises_when_y_has_len_1():
    with pytest.raises(ValueError) as excinfo:
        observations_validate([1])
    assert str(excinfo.value) == "y must have len > 1"


def test_raises_when_y_is_all_nan():
    with pytest.raises(ValueError) as excinfo:
        observations_validate([np.nan, np.nan])
    assert str(excinfo.value) == (
        "Aborting inference due to input series " "being all null."
    )


def test_raises_when_y_have_just_2_values():
    with pytest.raises(ValueError) as excinfo:
        observations_validate(pd.DataFrame([1, 2]))
    assert str(excinfo.value) == (
        "Aborting inference due to fewer than 3 " "nonnull values in input."
    )


def test_raises_when_y_is_constant():
    with pytest.raises(ValueError) as excinfo:
        observations_validate(pd.Series([1, 1, 1, 1, 1]))
    assert str(excinfo.value) == (
        "Aborting inference due to input series " "being constant"
    )


def test_model_constructor():
    data = pd.DataFrame(np.random.randn(200, 2))
    model = construct_model(data)
    assert_array_equal(model.data.endog, data.iloc[:, 0].values)
    assert model.irregular
    assert model.k_exog == data.shape[1] - 1
    assert model.level
    assert_array_equal(
        model.exog, data.iloc[:, 1].values.reshape(-1, data.shape[1] - 1)
    )


def test_model_constructor_w_just_endog():
    data = pd.DataFrame(np.random.randn(200, 1))
    model = construct_model(data)
    assert_array_equal(model.data.endog, data.iloc[:, 0].values)
    assert model.irregular
    assert model.k_exog == data.shape[1] - 1
    assert model.level
    assert not model.exog


def test_model_fit():
    model_mock = mock.Mock()

    _ = model_fit(model_mock, "MLE", 50)
    model_mock.fit.assert_called_once_with(maxiter=50)
