"""Unit Tests for inferences module"""


import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.arima_process import ArmaProcess

import causalimpact

compile_posterior = causalimpact.inferences.compile_posterior_inferences
np.random.seed(1)


@pytest.fixture
def data():
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)

    X = 1 + arma_process.generate_sample(nsample=100)
    X = X.reshape(-1, 1)
    y = 1.2 * X + np.random.normal(size=(100, 1))
    data = np.concatenate((y, X), axis=1)
    data = pd.DataFrame(data)
    return data


def test_compile_posterior_inferences_w_data(data):
    pre_period = [0, 70]
    post_period = [71, 100]

    df_pre = data.loc[pre_period[0] : pre_period[1], :]
    df_post = data.loc[post_period[0] : post_period[1], :]

    post_period_response = None
    alpha = 0.05
    orig_std_params = (0.0, 1.0)

    model = UnobservedComponents(
        endog=df_pre.iloc[:, 0].values, level="llevel", exog=df_pre.iloc[:, 1:].values
    )

    trained_model = model.fit()

    inferences = compile_posterior(
        trained_model,
        data,
        df_pre,
        df_post,
        post_period_response,
        alpha,
        orig_std_params,
    )

    expected_response = pd.Series(data.iloc[:, 0], name="response")
    assert_series_equal(expected_response, inferences["series"]["response"])

    expected_cumsum = pd.Series(np.cumsum(expected_response), name="cum_response")

    assert_series_equal(expected_cumsum, inferences["series"]["cum_response"])

    predictor = trained_model.get_prediction()
    forecaster = trained_model.get_forecast(
        steps=len(df_post), exog=df_post.iloc[:, 1].values.reshape(-1, 1), alpha=alpha
    )

    pre_pred = predictor.predicted_mean
    post_pred = forecaster.predicted_mean

    point_pred = np.concatenate([pre_pred, post_pred])

    expected_point_pred = pd.Series(point_pred, name="point_pred")
    assert_series_equal(expected_point_pred, inferences["series"]["point_pred"])

    pre_ci = pd.DataFrame(predictor.conf_int(alpha=alpha))
    pre_ci.index = df_pre.index
    post_ci = pd.DataFrame(forecaster.conf_int(alpha=alpha))
    post_ci.index = df_post.index

    ci = pd.concat([pre_ci, post_ci])

    expected_pred_upper = ci.iloc[:, 1]
    expected_pred_upper = expected_pred_upper.rename("point_pred_upper")
    expected_pred_lower = ci.iloc[:, 0]
    expected_pred_lower = expected_pred_lower.rename("point_pred_lower")

    assert_series_equal(expected_pred_upper, inferences["series"]["point_pred_upper"])
    assert_series_equal(expected_pred_lower, inferences["series"]["point_pred_lower"])

    expected_cum_pred = pd.Series(np.cumsum(point_pred), name="cum_pred")
    assert_series_equal(expected_cum_pred, inferences["series"]["cum_pred"])

    expected_cum_pred_lower = pd.Series(
        np.cumsum(expected_pred_lower), name="cum_pred_lower"
    )
    assert_series_equal(expected_cum_pred_lower, inferences["series"]["cum_pred_lower"])

    expected_cum_pred_upper = pd.Series(
        np.cumsum(expected_pred_upper), name="cum_pred_upper"
    )
    assert_series_equal(expected_cum_pred_upper, inferences["series"]["cum_pred_upper"])

    expected_point_effect = pd.Series(
        expected_response - expected_point_pred, name="point_effect"
    )
    assert_series_equal(expected_point_effect, inferences["series"]["point_effect"])

    expected_point_effect_lower = pd.Series(
        expected_response - expected_pred_lower, name="point_effect_lower"
    )
    assert_series_equal(
        expected_point_effect_lower, inferences["series"]["point_effect_lower"]
    )

    expected_point_effect_upper = pd.Series(
        expected_response - expected_pred_upper, name="point_effect_upper"
    )
    assert_series_equal(
        expected_point_effect_upper, inferences["series"]["point_effect_upper"]
    )

    expected_cum_effect = pd.Series(
        np.concatenate(
            (
                np.zeros(len(df_pre)),
                np.cumsum(expected_point_effect.iloc[len(df_pre) :]),
            )
        ),
        name="cum_effect",
    )
    assert_series_equal(expected_cum_effect, inferences["series"]["cum_effect"])

    expected_cum_effect_lower = pd.Series(
        np.concatenate(
            (
                np.zeros(len(df_pre)),
                np.cumsum(expected_point_effect_lower.iloc[len(df_pre) :]),
            )
        ),
        name="cum_effect_lower",
    )
    assert_series_equal(
        expected_cum_effect_lower, inferences["series"]["cum_effect_lower"]
    )

    expected_cum_effect_upper = pd.Series(
        np.concatenate(
            (
                np.zeros(len(df_pre)),
                np.cumsum(expected_point_effect_upper.iloc[len(df_pre) :]),
            )
        ),
        name="cum_effect_upper",
    )
    assert_series_equal(
        expected_cum_effect_upper, inferences["series"]["cum_effect_upper"]
    )


def test_compile_posterior_inferences_w_post_period_response(data):
    pre_period = [0, 70]
    post_period = [71, 100]

    df_pre = data.loc[pre_period[0] : pre_period[1], :]
    df_post = data.loc[post_period[0] : post_period[1], :]

    post_period_response = df_post.loc[post_period[0] : post_period[1]]

    X = df_post.iloc[:, 1:]
    y = X.copy()
    y[:] = np.nan

    df_post = pd.DataFrame(np.concatenate([y, X], axis=1))
    data_index = data.index
    data = pd.concat([df_pre, df_post], axis=0)
    data.index = data_index

    alpha = 0.05
    orig_std_params = (0.0, 1.0)

    model = UnobservedComponents(
        endog=data.iloc[:, 0].values, level="llevel", exog=data.iloc[:, 1:].values
    )

    trained_model = model.fit()

    inferences = compile_posterior(
        trained_model, data, df_pre, None, post_period_response, alpha, orig_std_params
    )

    expected_response = pd.Series(data.iloc[:, 0], name="response")
    assert_series_equal(expected_response, inferences["series"]["response"])

    expected_cumsum = pd.Series(np.cumsum(expected_response), name="cum_response")

    assert_series_equal(expected_cumsum, inferences["series"]["cum_response"])

    predictor = trained_model.get_prediction(end=len(df_pre) - 1)
    forecaster = trained_model.get_prediction(start=len(df_pre))

    pre_pred = predictor.predicted_mean
    post_pred = forecaster.predicted_mean

    point_pred = np.concatenate([pre_pred, post_pred])

    expected_point_pred = pd.Series(point_pred, name="point_pred")
    assert_series_equal(expected_point_pred, inferences["series"]["point_pred"])

    pre_ci = pd.DataFrame(predictor.conf_int(alpha=alpha))
    pre_ci.index = df_pre.index
    post_ci = pd.DataFrame(forecaster.conf_int(alpha=alpha))
    post_ci.index = df_post.index

    ci = pd.concat([pre_ci, post_ci])

    expected_pred_upper = ci.iloc[:, 1]
    expected_pred_upper = expected_pred_upper.rename("point_pred_upper")
    expected_pred_upper.index = data.index

    expected_pred_lower = ci.iloc[:, 0]
    expected_pred_lower = expected_pred_lower.rename("point_pred_lower")
    expected_pred_lower.index = data.index

    assert_series_equal(expected_pred_upper, inferences["series"]["point_pred_upper"])
    assert_series_equal(expected_pred_lower, inferences["series"]["point_pred_lower"])

    expected_cum_pred = pd.Series(np.cumsum(point_pred), name="cum_pred")
    assert_series_equal(expected_cum_pred, inferences["series"]["cum_pred"])

    expected_cum_pred_lower = pd.Series(
        np.cumsum(expected_pred_lower), name="cum_pred_lower"
    )
    assert_series_equal(expected_cum_pred_lower, inferences["series"]["cum_pred_lower"])

    expected_cum_pred_upper = pd.Series(
        np.cumsum(expected_pred_upper), name="cum_pred_upper"
    )
    assert_series_equal(expected_cum_pred_upper, inferences["series"]["cum_pred_upper"])

    expected_point_effect = pd.Series(
        expected_response - expected_point_pred, name="point_effect"
    )
    assert_series_equal(expected_point_effect, inferences["series"]["point_effect"])

    expected_point_effect_lower = pd.Series(
        expected_response - expected_pred_lower, name="point_effect_lower"
    )
    assert_series_equal(
        expected_point_effect_lower, inferences["series"]["point_effect_lower"]
    )

    expected_point_effect_upper = pd.Series(
        expected_response - expected_pred_upper, name="point_effect_upper"
    )
    assert_series_equal(
        expected_point_effect_upper, inferences["series"]["point_effect_upper"]
    )

    expected_cum_effect = pd.Series(
        np.concatenate(
            (
                np.zeros(len(df_pre)),
                np.cumsum(expected_point_effect.iloc[len(df_pre) :]),
            )
        ),
        name="cum_effect",
    )
    assert_series_equal(expected_cum_effect, inferences["series"]["cum_effect"])

    expected_cum_effect_lower = pd.Series(
        np.concatenate(
            (
                np.zeros(len(df_pre)),
                np.cumsum(expected_point_effect_lower.iloc[len(df_pre) :]),
            )
        ),
        name="cum_effect_lower",
    )
    assert_series_equal(
        expected_cum_effect_lower, inferences["series"]["cum_effect_lower"]
    )

    expected_cum_effect_upper = pd.Series(
        np.concatenate(
            (
                np.zeros(len(df_pre)),
                np.cumsum(expected_point_effect_upper.iloc[len(df_pre) :]),
            )
        ),
        name="cum_effect_upper",
    )
    assert_series_equal(
        expected_cum_effect_upper, inferences["series"]["cum_effect_upper"]
    )
