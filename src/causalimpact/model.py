"""Functions for constructing statespace model."""


import numpy as np
import pandas as pd


def observations_ill_conditioned(y):
    """Checks whether the response variable (i.e., the series of observations
    for the dependent variable y) are ill-conditioned. For example, the series
    might contain too few non-NA values. In such cases, inference will be
    aborted.

    Args:
        y: observed series (Pandas Series)

    Returns:
        True if something is wrong with the observations; False otherwise.
    """

    if y is None:
        raise ValueError("y cannot be None")
    if not (len(y) > 1):
        raise ValueError("y must have len > 1")

    # All NA?
    if np.all(pd.isnull(y)):
        raise ValueError("Aborting inference due to input series being all " "null.")
    elif len(y[pd.notnull(y)]) < 3:
        # Fewer than 3 non-NA values?
        raise ValueError(
            "Aborting inference due to fewer than 3 nonnull " "values in input."
        )
    # Constant series?
    elif y.std(skipna=True) == 0:
        raise ValueError("Aborting inference due to input series being " "constant")
    return False


def construct_model(data, model_args={}):
    """Specifies the model and performs inference. Inference means using a
    technique that combines Kalman Filters with Maximum Likelihood Estimators
    methods to fit the parameters that best explain the observed data.

    Args:
      data: time series of response variable and optional covariates
      model_args: optional list of additional model arguments

    Returns:
      An Unobserved Components Model, as returned by UnobservedComponents()
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    y = data.iloc[:, 0]

    observations_ill_conditioned(y)

    # LocalLevel specification of statespace
    ss = {}
    ss["endog"] = y.values
    ss["level"] = "llevel"

    # No regression?
    if len(data.columns) == 1:
        mod = UnobservedComponents(**ss)
        return mod
    else:
        # Static regression
        if not model_args.get("dynamic_regression"):
            ss["exog"] = data.iloc[:, 1:].values
            mod = UnobservedComponents(**ss)
            return mod
        # Dynamic regression
        else:
            raise NotImplementedError()


def model_fit(model, estimation, niter):
    if estimation == "MLE":
        trained_model = model.fit(maxiter=niter)
        return trained_model
    else:
        raise NotImplementedError()
