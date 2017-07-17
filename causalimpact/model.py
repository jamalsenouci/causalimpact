"""Functions for constructing statespace model."""

# import pymc as mc
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

    if (y is None):
        raise ValueError("y cannot be None")
    if not (len(y) > 1):
        raise ValueError("y must have len > 1")

    # All NA?
    if np.all(pd.isnull(y)):
        raise ValueError("""Aborting inference due to input series being all
                         null.""")
    elif len(y[pd.notnull(y)]) < 3:
        # Fewer than 3 non-NA values?
        raise ValueError("""Aborting inference due to fewer than 3 nonnull
                         values in input""")
    # Constant series?
    elif y.std(skipna=True) == 0:
        raise ValueError("""Aborting inference due to input series being
                         constant""")

    return False


def construct_model(self, data, model_args=None):
    """Specifies the model and performs inference. Inference means using the data
    to pass from a prior distribution over parameters and states to a posterior
    distribution. In a Bayesian framework, estimating a model means to obtain
    p(parameters | data) from p(data | parameters) and p(parameters). This
    involves multiplying the prior with the likelihood and normalising the
    resulting distribution using the marginal likelihood or model evidence,
    p(data). Computing the evidence poses a virtually intractable
    high-dimensional integration problem which can be turned into an easier
    optimization problem using, for instance, an approximate stochastic
    inference strategy. Here, we use a Markov chain Monte Carlo algorithm, as
    implemented in the {pymc} package.
    Args:
      data: time series of response variable and optional covariates
      model_args: optional list of additional model arguments

    Returns:
      An Unobserved Components Model, as returned by UnobservedComponents()

    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    # extract y variable
    y = data.iloc[:, 0]

    # If the series is ill-conditioned, abort inference
    observations_ill_conditioned(y)

    # specification params
    ss = {}
    # Local level
    ss["endog"] = y.values
    ss["level"] = "llevel"

    # Add seasonal component?
    if model_args["nseasons"] > 1:
        ss["seasonal_period"] = model_args["season_duration"]

    # No regression?
    if len(data.columns) == 1:
        mod = UnobservedComponents(**ss)
        return mod
    else:
        # Static regression
        if not model_args["dynamic_regression"]:
            ss["exog"] = data.iloc[:, 1:].values
            mod = UnobservedComponents(**ss)
            self.model = mod
            return mod
        # Dynamic regression
        else:
            raise NotImplementedError()


def model_fit(self, model, estimation, niter):
    if estimation == "MLE":
        model = model.fit(maxiter=niter)
        self.model = model
        return self.model
    else:
        raise NotImplementedError()
