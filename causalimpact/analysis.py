# from statsmodels.api.statespace import bsts
import numpy as np
import pandas as pd
from pandas.core.common import PandasError
from pandas.util.testing import is_list_like


class CausalImpact(object):
    def __init__(self, data=None, pre_period=None, post_period=None,
                 model_args=None, bsts_model=None, post_period_response=None,
                 alpha=0.05):
        self.impact = None

        kwargs = self._format_input(data, pre_period, post_period, model_args,
                                    bsts_model, post_period_response, alpha)

        # Depending on input, dispatch to the appropriate Run* method()
        if data is not None:
            self._run_with_data(kwargs["data"], kwargs["pre_period"],
                                kwargs["post_period"], kwargs["model_args"],
                                kwargs["alpha"])
        else:
            self._run_with_bsts(kwargs["bsts_model"],
                                kwargs["post_period_response"],
                                kwargs["alpha"])

    def _format_input_data(self, data):
        """Checks and formats the <data> argument provided to CausalImpact().

        Args:
            data: Pandas DataFrame

        Returns:
            correctly formatted Pandas DataFrame
        """
        # If <data> is a Pandas DataFrame and the first column is 'date',
        # try to convert

        if type(data) == pd.DataFrame and type(data.columns[0]) == str:
            if data.columns[0].lower() in ["date", "time"]:
                data = data.set_index(data.columns[0])

        # Try to convert to Pandas DataFrame
        try:
            data = pd.DataFrame(data)
        except PandasError:
            raise PandasError("could not convert input data to Pandas \
                              DataFrame")

        # Must have at least 3 time points
        if len(data.index) < 3:
            raise ValueError("data must have at least 3 time points")

        # Must not have NA in covariates (if any)
        if len(data.columns) >= 2:
            if np.any(pd.isnull(data.ix[:, 1:])):
                raise ValueError("covariates must not contain null values")

        return data

    def _format_input_prepost(self, pre_period, post_period, data):
        """Checks and formats the <pre_period> and <post_period>
           input arguments_

        Args:
            pre_period: two-element list
            post_period: two-element list
            data: already-checked Pandas DataFrame, for reference only
        """
        import numpy as np
        import pandas as pd

        if pre_period is None or post_period is None:
            raise ValueError("pre_period and post period must not contain \
                             null values")
        if type(pre_period) is not list or type(post_period) is not list:
            raise ValueError("pre_period and post_period must bothe be lists")
        if len(pre_period) != 2 or len(post_period) != 2:
            raise ValueError("pre_period and post_period must both be of \
                             length 2")
        if np.any(pd.isnull(pre_period)) or np.any(pd.isnull(post_period)):
            raise ValueError("pre_period and post period must not contain \
                             null values")

        pre_dtype = np.array(pre_period).dtype
        post_dtype = np.array(post_period).dtype

        if data.index.dtype.kind != pre_dtype.kind or \
           data.index.dtype.kind != post_dtype.kind:
            if data.index.dtype == int:
                pre_period = [int(elem) for elem in pre_period]
                post_period = [int(elem) for elem in post_period]
            elif data.index.dtype == float:
                pre_period = [float(elem) for elem in pre_period]
                post_period = [float(elem) for elem in post_period]
            else:
                raise ValueError("pre_period (" + pre_dtype.name +
                                 ") and post_period (" + post_dtype.name +
                                 ") should have the same class as the " +
                                 "time points in the data ("
                                 + data.index.dtype.name + ")")

        if pre_period[0] < data.index.min():
            print("Setting pre_period[1] to start of data: " +
                  str(data.index.min()))
            pre_period[0] = data.index.min()
        if pre_period[1] > data.index.max():
            print("Setting pre_period[1] to end of data: " +
                  str(data.index.max()))
            pre_period[1] = data.index.max()
        if post_period[1] > data.index.max():
            print("Setting post_period[1] to end of data: " +
                  str(data.index.max()))
            post_period[1] = data.index.max()

        if pre_period[1] - pre_period[0] + 1 < 3:
            raise ValueError("pre_period must span at least 3 time points")
        if post_period[1] < post_period[0]:
            raise ValueError("post_period[1] must not be earlier than \
                             post_period[0]")
        if post_period[0] < pre_period[1]:
            raise ValueError("post_period[0] must not be earlier than \
                             pre_period[1]")

        return {"pre_period": pre_period, "post_period": post_period}

    def _format_input(self, data, pre_period, post_period, model_args,
                      bsts_model, post_period_response, alpha):
        """Checks and formats all input arguments supplied to CausalImpact()
           See the documentation of CausalImpact() for details

        Args:
            data:                 Pandas DataFrame or data frame
            pre_period:           beginning and end of pre-period
            post_period:          beginning and end of post-period
            model_args:           list of additional arguments for the model
            bsts_model:           fitted bsts model (instead of data)
            post_period_response: observed response in the post-period
            alpha:                tail-area for posterior intervals

        Returns:
            list of checked (and possibly reformatted) input arguments
"""

        import numpy as np
        import pandas as pd

        # Check that a consistent set of variables has been provided
        args = [data, pre_period, post_period, bsts_model,
                post_period_response]

        data_model_args = [True, True, True, False, False]
        bsts_model_args = [False, False, False, True, True]

        if np.any(pd.isnull(args) != data_model_args) and \
           np.any(pd.isnull(args) != bsts_model_args):
            raise TypeError("must either provide data, pre_period, post_period,\
                            model_args or bsts_model and post_period_response")

        # Check <data> and convert to Pandas DataFrame, with rows
        # representing time points
        if data is not None:
            data = self._format_input_data(data)

        # Check <pre_period> and <post_period>
        if data is not None:
            checked = self._format_input_prepost(pre_period, post_period, data)
            pre_period = checked["pre_period"]
            post_period = checked["post_period"]

        # Parse <model_args>, fill gaps using <_defaults>

        _defaults = {"niter": 1000, "standardize_data": True,
                     "prior_level_sd": 0.01,
                     "nseasons": 1,
                     "season_duration": 1,
                     "dynamic_regression": False}

        if model_args is None:
            model_args = _defaults
        else:
            missing = [key for key in _defaults if key not in model_args]
            for arg in missing:
                model_args[arg] = _defaults[arg]

        """ Check only those parts of <model_args> that are used
            in this file The other fields will be checked in
            FormatInputForConstructModel()"""

        # Check <standardize_data>
        if type(model_args["standardize_data"]) != bool:
            raise ValueError("model_args_standardize_data must be a \
                             boolean value")

        """ Check <bsts_model> TODO
        if bsts_model is not None:
            if type(bsts_model) != bsts:
                raise ValueError("bsts_model must be an object of class \
                                 statsmodels_bsts")
        """

        # Check <post_period_response>
        if bsts_model is not None:
            if not is_list_like(post_period_response):
                raise ValueError("post_period_response must be list-like")
            if np.array(post_period_response).dtype.num == 17:
                raise ValueError("post_period_response should not be \
                                 datetime values")
            if not np.all(np.isreal(post_period_response)):
                raise ValueError("post_period_response must contain all \
                                 real values")

        # Check <alpha>
        if alpha is None:
            raise ValueError("alpha must not be None")
        if not np.isreal(alpha):
            raise ValueError("alpha must be a real number")
        if np.isnan(alpha):
            raise ValueError("alpha must not be NA")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")

        # Return updated arguments
        kwargs = {"data": data, "pre_period": pre_period,
                  "post_period": post_period, "model_args": model_args,
                  "bsts_model": bsts_model,
                  "post_period_response": post_period_response, "alpha": alpha}
        return kwargs

    def _run_with_data(self, data, pre_period, post_period, model_args, alpha):
        # Zoom in on data in modeling range

        first_non_null = pd.isnull(data.ix[:, 1]).nonzero()[0]
        if len(first_non_null) > 0:
            pre_period[0] = max(pre_period[0], data.index[first_non_null[0]])
        data_modeling = data.ix[pre_period[0]:post_period[1], :]

        # Standardize all variables?
        unstandardize = np.identity
        if model_args["standardize_data"]:
            sd_results = standardize_all_variables(data_modeling)
            data_modeling = sd_results["data"]
            unstandardize = sd_results["unstandardize"]

        # Set observed response in post-period to NA
        data_modeling.ix[post_period[0]:, 1] = np.nan

        # Construct model and perform inference
        bsts_model = construct_model(data_modeling, model_args)

        # Compile posterior inferences
        if bsts_model is not None:
            y_post = data.ix[post_period[0]:post_period[1], 1]
            inferences = compile_posterior_inferences(bsts_model, y_post,
                                                      alpha, unstandardize)
        else:
            inferences = compile_na_inferences(data.ix[:, 1])

        # Extend <series> to cover original range
        # (padding with NA as necessary)
        empty = pd.DataFrame(index=data.index)
        inferences["series"] = pd.merge(inferences["series"], empty,
                                        left_index=True, right_index=True,
                                        how="outer")
        if len(inferences["series"]) != len(data):
            raise ValueError("inferences['series'] must have the same number \
                             of rows as 'data'")

        # Replace <y.model> by full original response
        inferences["series"].ix[:, 1] = data[:, 1]

        # Assign response-variable names
        inferences["series"][:, 1].name = "response"
        inferences["series"][:, 2].name = "cum.response"

        # Return 'CausalImpact' object
        model = {"pre_period": pre_period, "post_period": post_period,
                 "model_args": model_args, "bsts_model": bsts_model,
                 "alpha": alpha}

        self.series = inferences["series"]
        self.summary = inferences["summary"]
        self.report = inferences["report"]
        self.model = model

    def _run_with_bsts(self, bsts_model, post_period_response, alpha):
        """ Runs an impact analysis on top of a fitted bsts model.

           Args:
             bsts_model: fitted model, as returned by bsts(), in which the
                         data during the post-period was set to NA
             post_period_response: observed data during the post-intervention
                                   period
             alpha: tail-probabilities of posterior intervals"""
        # Guess <pre_period> and <post_period> from the observation vector
        # These will be needed for plotting period boundaries in plot().
        y = bsts_model["original_series"]
        try:
            indices = infer_period_indices_from_data(y)
        except ValueError:
            raise ValueError("bsts.model must have been fitted on data where \
                             the values in the post-intervention period have \
                             been set to NA")

        # Compile posterior inferences
        inferences = compile_posterior_inferences(bsts_model=bsts_model,
                                                  y_post=post_period_response,
                                                  alpha=alpha)

        # Assign response-variable names
        # N.B. The modeling period comprises everything found in bsts, so the
        # actual observed data is equal to the data in the modeling period
        inferences["series"].columns = ["response", "cum.response"]

        # Return 'CausalImpact' object
        model = {"pre_period": pre_period, "post_period": post_period,
                 "bsts_model": bsts_model, "alpha": alpha}

        self.series = inferences["series"]
        self.summary = inferences["summary"]
        self.report = inferences["report"]
        self.model = model

    def _print_summary(self, digits=2):
        """Prints a summary of the results.

        Args:
            digits: Number of digits to print for all numbers."""
        # TODO finish this task
        # Check input
        # Print title
        print("Posterior inference {CausalImpact}")

        """
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    ('Df Residuals:', None),
                    ('Df Model:', None),
                    ]
        top_right = [('Dep. Variable:', None),
                     ('Model:', None),
                     ('Method:', ['Least Squares']),
                     ('Date:', None),
                     ('Time:', None),
                     ('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ]
        yname = "test"
        xname = "test2"
        title = "title"
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)

        return smry"""
    def _print_report(self):
        if self.impact["report"] is None:
            print("Report empty")
        else:
            print("Analysis report {CausalImpact}")
            print(self.impact["report"])

    def summary(self, output="summary"):
        output = output.lower()
        if output == "summary":
            self._print_summary(self)
        elif output == "report":
            self._print_report
        else:
            raise ValueError("Output argument must be either 'summary' \
                             or 'report'")
