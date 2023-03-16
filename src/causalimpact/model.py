"""Constructs and fits the statespace model.

Contains the construct_model and model_fit functions that are called in analysis.py.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
from pytensor.graph.op import Op


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


def construct_model(data, model_args=None):
    """Specifies the model and performs inference. Inference means using a
    technique that combines Kalman Filters with Maximum Likelihood Estimators
    methods to fit the parameters that best explain the observed data.

    Args:
      data: time series of response variable and optional covariates
      model_args: optional list of additional model arguments

    Returns:
      An Unobserved Components Model, as returned by UnobservedComponents()
    """
    if model_args is None:
        model_args = {}
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    y = data.iloc[:, 0]

    observations_ill_conditioned(y)

    # LocalLevel specification of statespace
    ss = {"endog": y.values, 
          "level": model_args.get("level"), 
          "trend": model_args.get("trend"),
          "seasonal": model_args.get("seasonal"), 
          "freq_seasonal": model_args.get("freq_seasonal")
         }
    ss_copy=ss.copy()
    _ = ss_copy.pop('endog');
    print('================================================================================================')
    print('================================ Beginning CausalImpact Analysis ===============================')
    print('================================================================================================')
    print(f"Model inputs are: {ss_copy}")

    # No regression?
    if len(data.columns) > 1:
        # Static regression
        if not model_args.get("dynamic_regression"):
            ss["exog"] = data.iloc[:, 1:].values
        # Dynamic regression
        else:
            raise NotImplementedError()
    mod = UnobservedComponents(**ss)
    print(f"Model endogenous series name: {mod.endog_names}")
    print(f"Model exogenous series name(s): {mod.exog_names}")
    print(f"Model state names are: {mod.state_names}")
    print(f"Model parameter names are: {mod.param_names}")
    print(f"Model start parameters are: {mod.start_params}")
    print(f"Model state names are: {mod.state_names}")
    print('================================================================================================')
    return mod


class Loglike(Op):
    """Theano LogLike wrapper that allow  PyMC3 to compute the likelihood
    and Jacobian in a way that it can make use of."""

    itypes = [at.dvector]  # expects a vector of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, model):
        self.model = model
        self.score = Score(self.model)

    def perform(self, node, inputs, outputs):
        (theta,) = inputs  # contains the vector of parameters
        llf = self.model.loglike(theta)
        outputs[0][0] = np.array(llf)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        out = [g[0] * self.score(theta)]
        return out


class Score(Op):
    """Theano Score wrapper that allow  PyMC3 to compute the likelihood and
    Jacobian in a way that it can make use of."""

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, model):
        self.model = model

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        outputs[0][0] = self.model.score(theta)


class ModelResults:
    """ModelResults class containing everything needed for inference
    intended to allow extension to other models (e.g. tensorflow)

    Parameters
    ----------
    ucm_model : statsmodels.tsa.statespace.structural.UnobservedComponents
        The constructed UCM model being fit
    results :
    estimation : string
        The estimation method. Options are "MLE" or "pymc"

    """

    def __init__(self, ucm_model, results, estimation) -> None:
        self.results = results
        self.estimation = estimation
        self.model_nobs = ucm_model.nobs

    def get_prediction(self, start=None, end=None):
        """
        In-sample prediction and out-of-sample forecasting


        Parameters
        ----------
        start : int
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int
            Zero-indexed observation number at which to end forecasting,
            i.e., the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you want
            out of sample prediction. Default is the last observation in the
            sample.

        Returns
        -------
        ModelPredictions
        """
        predictions = self.results.get_prediction(start=start, end=end)
        return predictions

    def get_forecast(self, df_post, alpha):
        forecast = self.results.get_forecast(
            steps=len(df_post), exog=df_post.iloc[:, 1:], alpha=alpha
        )
        return forecast

    def summary(self):
        return self.results.summary()


def model_fit(model, estimation, model_args):
    """Fits the model and returns a ModelResults object.

    Uses the chosen estimation option to fit the model and
    return a ModelResults object that is agnostic of
    estimation approach.

    Parameters:
    -----------
    model : statsmodels.tsa.statespace.structural.UnobservedComponents
    estimation : str
        Either 'MLE' or 'pymc'.
    model_args : dict
        possible args for MLE are:
        niter: int
        possible args for pymc are:
        ndraws: int
            number of draws from the distribution
        nburn: int
            number of "burn-in points" (which will be discarded)

    """
    if estimation == "MLE":
        trained_model = model.fit(maxiter=model_args["niter"])
        model_results = ModelResults(model, trained_model, estimation)
        return model_results, trained_model.filter_results.loglikelihood_burn # This defines the index after to start plotting due to approximate diffuse start
    
    elif estimation == "pymc":
        loglike = Loglike(model)
        with pm.Model():
            # Priors
            sigma2irregular = pm.InverseGamma("sigma2.irregular", 1, 1)
            sigma2level = pm.InverseGamma("sigma2.level", 1, 1)
            if model.exog is None:
                # convert variables to tensor vectors
                theta = at.as_tensor_variable([sigma2irregular, sigma2level])
            else:
                # prior for regressors
                betax1 = pm.Laplace("beta.x1", mu=0, b=1.0 / 0.7)
                # convert variables to tensor vectors
                theta = at.as_tensor_variable([sigma2irregular, sigma2level, betax1])
            # use a DensityDist (use a lambda function to "call" the Op)
            pm.Potential("likelihood", loglike(theta))

            # Draw samples
            trace = pm.sample(
                model_args["ndraws"],
                tune=model_args["nburn"],
                return_inferencedata=True,
                cores=4,
                compute_convergence_checks=False,
            )
        # Retrieve the posterior means
        params = pm.summary(trace)["mean"].values

        # Construct results using these posterior means as parameter values
        results = model.smooth(params)
        model_results = ModelResults(model, results, estimation)
        
        return model_results, 0
