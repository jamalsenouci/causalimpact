import textwrap
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from causalimpact.misc import standardize_all_variables, df_print, get_matplotlib
from causalimpact.model import construct_model, model_fit
from causalimpact.inferences import compile_inferences
import scipy.stats as st


class CausalImpact:
    """CausalImpact() performs causal inference through counterfactual
    predictions using a Bayesian structural time-series model.

    Parameters:
    ----------
        data : pandas dataframe
            the response variable must be in the first column, and any covariates
            in subsequent columns.
        pre_period : list
            A list specifying the first and the last time point of the
            pre-intervention period in the response column. This period can be
            thought of as a training period, used to determine the relationship
            between the response variable and the covariates.
        post_period : list
            A vector specifying the first and the last day of the post-intervention
            period we wish to study. This is the period after the intervention has
            begun whose effect we are interested in. The relationship between
            response variable and covariates, as determined during the pre-period,
            will be used to predict how the response variable should have evolved
            during the post-period had no intervention taken place.
        model_args : dict
            Optional arguments that can be used to adjust the default construction
            of the state-space model used for inference.
            For full control over the model, you can construct your own model using
            the statsmodels package and feed the model into CausalImpact().
        ucm_model : statsmodels.tsa.statespace.structural.UnobservedComponents
            Instead of passing in data and having CausalImpact construct a
            model, it is possible to construct a model yourself using the
            statsmodel package. In this case, omit data, pre_period, and
            post_period. Instead only pass in ucm_model, y_post, alpha (optional).
            The model must have been fitted on data where the response variable was
            set to np.nan during the post-treatment period. The actual observed data
            during this period must then be passed to the function in y_post.
        post_period_response : list | pd.Series | np.Array
            Actual observed data during the post-intervention period. This is required
            if and only if a fitted ucm_model is passed instead of data.
        alpha : float
            Desired tail-area probability for posterior intervals. Defaults to 0.05,
            which will produce central 95% intervals.


        Returns
        -------
        CausalImpact Object

    """

    def __init__(
        self,
        data=None,
        pre_period=None,
        post_period=None,
        model_args=None,
        ucm_model=None,
        post_period_response=None,
        alpha=0.05,
        estimation="MLE",
    ):
        self.series = None
        self.model = {}
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            self.data = data
        self.params = {
            "data": data,
            "pre_period": pre_period,
            "post_period": post_period,
            "model_args": model_args,
            "ucm_model": ucm_model,
            "post_period_response": post_period_response,
            "alpha": alpha,
            "estimation": estimation,
        }
        self.inferences = None
        self.results = None

    def run(self):
        kwargs = self._format_input(
            self.params["data"],
            self.params["pre_period"],
            self.params["post_period"],
            self.params["model_args"],
            self.params["ucm_model"],
            self.params["post_period_response"],
            self.params["alpha"],
        )

        # Depending on input, dispatch to the appropriate Run* method()
        if self.data is not None:
            self._run_with_data(
                kwargs["data"],
                kwargs["pre_period"],
                kwargs["post_period"],
                kwargs["model_args"],
                kwargs["alpha"],
                self.params["estimation"],
            )
        else:
            self._run_with_ucm(
                kwargs["ucm_model"],
                kwargs["post_period_response"],
                kwargs["alpha"],
                kwargs["model_args"],
                self.params["estimation"],
            )

    @staticmethod
    def _format_input_data(data):
        """Check and format the data argument provided to CausalImpact().

        Args:
            data: Pandas DataFrame

        Returns:
            correctly formatted Pandas DataFrame
        """
        # If <data> is a Pandas DataFrame and the first column is 'date',
        # try to convert

        if (
            isinstance(data, pd.DataFrame)
            and isinstance(data.columns[0], str)
            and data.columns[0].lower() in ["date", "time"]
        ):
            data = data.set_index(data.columns[0])

        # Try to convert to Pandas DataFrame
        try:
            data = pd.DataFrame(data)
        except ValueError:
            raise ValueError("could not convert input data to Pandas " + "DataFrame")

        # Must have at least 3 time points
        if len(data.index) < 3:
            raise ValueError("data must have at least 3 time points")

        # Must not have NA in covariates (if any)
        if len(data.columns) >= 2 and pd.isnull(data.iloc[:, 1:]).any(axis=None):
            raise ValueError("covariates must not contain null values")

        return data

    @staticmethod
    def _check_periods_are_valid(pre_period, post_period):
        if not isinstance(pre_period, list) or not isinstance(post_period, list):
            raise ValueError("pre_period and post_period must both be lists")
        if len(pre_period) != 2 or len(post_period) != 2:
            raise ValueError("pre_period and post_period must both be of " + "length 2")
        if pd.isnull(pre_period).any(axis=None) or pd.isnull(post_period).any(
            axis=None
        ):
            raise ValueError(
                "pre_period and post period must not contain " + "null values"
            )

    @staticmethod
    def _align_periods_dtypes(pre_period, post_period, data):
        """align the dtypes of the pre_period and post_period to the data index.

        Args:
            pre_period: two-element list
            post_period: two-element list
            data: already-checked Pandas DataFrame, for reference only
        """
        pre_dtype = np.array(pre_period).dtype
        post_dtype = np.array(post_period).dtype
        # if index is datetime then convert pre and post to datetimes
        if isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex):
            pre_period = [pd.to_datetime(date) for date in pre_period]
            post_period = [pd.to_datetime(date) for date in post_period]
            pd.core.dtypes.common.is_datetime_or_timedelta_dtype(pre_period)
        # if index is not datetime then error if datetime pre and post is passed
        elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
            pd.Series(pre_period)
        ) or pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
            pd.Series(post_period)
        ):
            raise ValueError(
                "pre_period ("
                + pre_dtype.name
                + ") and post_period ("
                + post_dtype.name
                + ") should have the same class as the "
                + "time points in the data ("
                + data.index.dtype.name
                + ")"
            )
        # if index is int
        elif pd.api.types.is_int64_dtype(data.index):
            pre_period = [int(elem) for elem in pre_period]
            post_period = [int(elem) for elem in post_period]
        # if index is int
        elif pd.api.types.is_float_dtype(data.index):
            pre_period = [float(elem) for elem in pre_period]
            post_period = [float(elem) for elem in post_period]
        # if index is string
        elif pd.api.types.is_string_dtype(data.index):
            if pd.api.types.is_numeric_dtype(
                np.array(pre_period)
            ) or pd.api.types.is_numeric_dtype(np.array(post_period)):
                raise ValueError(
                    "pre_period ("
                    + pre_dtype.name
                    + ") and post_period ("
                    + post_dtype.name
                    + ") should have the same class as the "
                    + "time points in the data ("
                    + data.index.dtype.name
                    + ")"
                )
            else:
                pre_period = [str(idx) for idx in pre_period]
                post_period = [str(idx) for idx in post_period]
        else:
            raise ValueError(
                "pre_period ("
                + pre_dtype.name
                + ") and post_period ("
                + post_dtype.name
                + ") should have the same class as the "
                + "time points in the data ("
                + data.index.dtype.name
                + ")"
            )
        return [pre_period, post_period]

    def _format_input_prepost(self, pre_period, post_period, data):
        """Check and format the pre_period and post_period input arguments.

        Args:
            pre_period: two-element list
            post_period: two-element list
            data: already-checked Pandas DataFrame, for reference only
        """
        self._check_periods_are_valid(pre_period, post_period)

        pre_period, post_period = self._align_periods_dtypes(
            pre_period, post_period, data
        )

        if pre_period[1] > post_period[0]:
            raise ValueError(
                "post period must start at least 1 observation"
                + " after the end of the pre_period"
            )

        if isinstance(data.index, pd.RangeIndex):
            loc3 = post_period[0]
            loc4 = post_period[1]
        else:
            loc3 = data.index.get_loc(post_period[0])
            loc4 = data.index.get_loc(post_period[1])

        if loc4 < loc3:
            raise ValueError(
                "post_period[1] must not be earlier than " + "post_period[0]"
            )

        if pre_period[0] < data.index.min():
            pre_period[0] = data.index.min()
        if post_period[1] > data.index.max():
            post_period[1] = data.index.max()
        return {"pre_period": pre_period, "post_period": post_period}

    @staticmethod
    def _check_valid_args_combo(args):
        data_model_args = [True, True, True, False, False]
        ucm_model_args = [False, False, False, True, True]

        if np.any(pd.isnull(args) != data_model_args) and np.any(
            pd.isnull(args) != ucm_model_args
        ):
            raise SyntaxError(
                "Must either provide ``data``, ``pre_period``"
                + " ,``post_period``, ``model_args``"
                " or ``ucm_model" + "and ``post_period_response``"
            )

    @staticmethod
    def _check_valid_alpha(alpha):
        if alpha is None:
            raise ValueError("alpha must not be None")
        if not np.isreal(alpha):
            raise ValueError("alpha must be a real number")
        if np.isnan(alpha):
            raise ValueError("alpha must not be NA")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")

    def _format_input(
        self,
        data,
        pre_period,
        post_period,
        model_args,
        ucm_model,
        post_period_response,
        alpha,
    ):
        """Check and format all input arguments supplied to CausalImpact().
           See the documentation of CausalImpact() for details

        Args:
            data:                 Pandas DataFrame or data frame
            pre_period:           beginning and end of pre-period
            post_period:          beginning and end of post-period
            model_args:           dict of additional arguments for the model
            ucm_model:            UnobservedComponents model (instead of data)
            post_period_response: observed response in the post-period
            alpha:                tail-area for posterior intervals
            estimation:           method of estimation for model fitting

        Returns:
            list of checked (and possibly reformatted) input arguments
        """
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        # Check that a consistent set of variables has been provided
        args = [data, pre_period, post_period, ucm_model, post_period_response]

        self._check_valid_args_combo(args)

        # Check <data> and convert to Pandas DataFrame, with rows
        # representing time points
        if data is not None:
            data = self._format_input_data(data)

        # Check <pre_period> and <post_period>
        if data is not None:
            checked = self._format_input_prepost(pre_period, post_period, data)
            pre_period = checked["pre_period"]
            post_period = checked["post_period"]
            self.params["pre_period"] = pre_period
            self.params["post_period"] = post_period

        # Parse <model_args>, fill gaps using <_defaults>

        _defaults = {
            "ndraws": 1000,
            "nburn": 100,
            "niter": 1000,
            "standardize_data": True,
            "prior_level_sd": 0.01,
            "level": False,
            "trend": False,
            "seasonal": None,
            "freq_seasonal": None,
            "dynamic_regression": False,
        }

        if model_args is None:
            model_args = _defaults
        else:
            missing = [key for key in _defaults if key not in model_args]
            for arg in missing:
                model_args[arg] = _defaults[arg]

        # Check <standardize_data>
        if not isinstance(model_args["standardize_data"], bool):
            raise ValueError("model_args.standardize_data must be a" + " boolean value")

        # Check <ucm_model>
        if ucm_model is not None and not isinstance(ucm_model, UnobservedComponents):
            raise ValueError(
                "ucm_model must be an object of class "
                "statsmodels.tsa.statespace.structural.UnobservedComponents "
                "instead received " + str(type(ucm_model))[8:-2]
            )

        # Check <post_period_response>
        if ucm_model is not None:
            if not is_list_like(post_period_response):
                raise ValueError("post_period_response must be list-like")
            if np.array(post_period_response).dtype.num == 17:
                raise ValueError(
                    "post_period_response should not be" + " datetime values"
                )
            if not np.all(np.isreal(post_period_response)):
                raise ValueError(
                    "post_period_response must contain all" + " real values"
                )

        # Check <alpha>
        self._check_valid_alpha(alpha)

        # Return updated arguments
        kwargs = {
            "data": data,
            "pre_period": pre_period,
            "post_period": post_period,
            "model_args": model_args,
            "ucm_model": ucm_model,
            "post_period_response": post_period_response,
            "alpha": alpha,
        }
        return kwargs

    def _run_with_data(
        self, data, pre_period, post_period, model_args, alpha, estimation
    ):
        # Zoom in on data in modeling range
        if data.shape[1] == 1:  # no exogenous values provided
            raise ValueError("data contains no exogenous variables")
        data_modeling = data.copy()

        df_pre = data_modeling.loc[pre_period[0] : pre_period[1], :]
        df_post = data_modeling.loc[post_period[0] : post_period[1], :]

        # Standardize all variables
        orig_std_params = (0, 1)
        if model_args["standardize_data"]:
            sd_results = standardize_all_variables(
                data_modeling, pre_period, post_period
            )
            df_pre = sd_results["data_pre"]
            df_post = sd_results["data_post"]
            orig_std_params = sd_results["orig_std_params"]

        # Construct model and perform inference
        model = construct_model(df_pre, model_args)
        self.model = model

        model_results, plot_index = model_fit(model, estimation, model_args)

        inferences = compile_inferences(
            model_results,
            data,
            df_pre,
            df_post,
            None,
            alpha,
            orig_std_params,
            estimation,
        )

        # "append" to 'CausalImpact' object
        self.inferences = inferences["series"]
        self.results = model_results
        self.plot_index = plot_index

    def _run_with_ucm(
        self, ucm_model, post_period_response, alpha, model_args, estimation
    ):
        """Runs an impact analysis on top of a ucm model.

        Args:
          ucm_model: Model as returned by UnobservedComponents(),
                     in which the data during the post-period was set to NA
          post_period_response: observed data during the post-intervention
                                period
          alpha: tail-probabilities of posterior intervals"""

        df_pre = ucm_model.data.orig_endog[: -len(post_period_response)]
        df_pre = pd.DataFrame(df_pre)

        post_period_response = pd.DataFrame(post_period_response)

        data = pd.DataFrame(
            np.concatenate([df_pre.values, post_period_response.values])
        )

        orig_std_params = (0, 1)

        model_results, plot_index = model_fit(ucm_model, estimation, model_args)

        # Compile posterior inferences
        inferences = compile_inferences(
            model_results,
            data,
            df_pre,
            None,
            post_period_response,
            alpha,
            orig_std_params,
            estimation,
        )

        obs_inter = model_results.model_nobs - len(post_period_response)

        self.params["pre_period"] = [0, obs_inter - 1]
        self.params["post_period"] = [obs_inter, -1]
        self.data = pd.concat([df_pre, post_period_response])
        self.inferences = inferences["series"]
        self.results = model_results
        self.plot_index = plot_index

    @staticmethod
    def _print_report(
        mean_pred_fmt,
        mean_resp_fmt,
        mean_lower_fmt,
        mean_upper_fmt,
        abs_effect_fmt,
        abs_effect_upper_fmt,
        abs_effect_lower_fmt,
        rel_effect_fmt,
        rel_effect_upper_fmt,
        rel_effect_lower_fmt,
        cum_resp_fmt,
        cum_pred_fmt,
        cum_lower_fmt,
        cum_upper_fmt,
        confidence,
        cum_rel_effect_lower,
        cum_rel_effect_upper,
        cum_rel_effect,
        width,
        p_value,
        alpha,
    ):
        sig = not (cum_rel_effect_lower < 0 < cum_rel_effect_upper)
        pos = cum_rel_effect > 0
        # Summarize averages
        stmt = textwrap.dedent(
            """During the post-intervention period, the response
            variable had an average value of
            approx. {mean_resp}.
                        """.format(
                mean_resp=mean_resp_fmt
            )
        )
        if sig:
            stmt += " By contrast, in "
        else:
            stmt += " In "

        stmt += textwrap.dedent(
            """
                the absence of an intervention, we would have
                expected an average response of {mean_pred}. The
                {confidence} interval of this counterfactual
                prediction is [{mean_lower}, {mean_upper}].
                Subtracting this prediction from the observed
                response yields an estimate of the causal effect
                the intervention had on the response variable.
                This effect is {abs_effect} with a
                {confidence} interval of [{abs_lower},
                {abs_upper}]. For a discussion of the
                significance of this effect,
                see below.
                """.format(
                mean_pred=mean_pred_fmt,
                confidence=confidence,
                mean_lower=mean_lower_fmt,
                mean_upper=mean_upper_fmt,
                abs_effect=abs_effect_fmt,
                abs_upper=abs_effect_upper_fmt,
                abs_lower=abs_effect_lower_fmt,
            )
        )
        # Summarize sums
        stmt2 = textwrap.dedent(
            """
                Summing up the individual data points during the
                post-intervention period (which can only sometimes be
                meaningfully interpreted), the response variable had an
                overall value of {cum_resp}.
                """.format(
                cum_resp=cum_resp_fmt
            )
        )
        if sig:
            stmt2 += " By contrast, had "
        else:
            stmt2 += " Had "

        stmt2 += textwrap.dedent(
            """
                the intervention not taken place, we would have expected
                a sum of {cum_pred}. The {confidence} interval of this
                prediction is [{cum_pred_lower}, {cum_pred_upper}]
                """.format(
                cum_pred=cum_pred_fmt,
                confidence=confidence,
                cum_pred_lower=cum_lower_fmt,
                cum_pred_upper=cum_upper_fmt,
            )
        )

        # Summarize relative numbers (in which case row [1] = row [2])
        stmt3 = textwrap.dedent(
            """
                                The above results are given in terms
                                of absolute numbers. In relative terms, the
                                response variable showed
                                """
        )
        if pos:
            stmt3 += " an increase of "
        else:
            stmt3 += " a decrease of "

        stmt3 += textwrap.dedent(
            """
                        {rel_effect}. The {confidence} interval of this
                        percentage is [{rel_effect_lower},
                        {rel_effect_upper}]
                        """.format(
                confidence=confidence,
                rel_effect=rel_effect_fmt,
                rel_effect_lower=rel_effect_lower_fmt,
                rel_effect_upper=rel_effect_upper_fmt,
            )
        )

        # Comment on significance
        if sig and pos:
            stmt4 = textwrap.dedent(
                """
                        This means that the positive effect observed
                        during the intervention period is statistically
                        significant and unlikely to be due to random
                        fluctuations. It should be noted, however, that
                        the question of whether this increase also bears
                        substantive significance can only be answered by
                        comparing the absolute effect {abs_effect} to
                        the original goal of the underlying
                        intervention.
                        """.format(
                    abs_effect=abs_effect_fmt
                )
            )
        elif sig and not pos:
            stmt4 = textwrap.dedent(
                """
                        This  means that the negative effect observed
                        during the intervention period is statistically
                        significant. If the experimenter had expected a
                        positive effect, it is recommended to double-check
                        whether anomalies in the control variables may have
                        caused an overly optimistic expectation of what
                        should have happened in the response variable in the
                        absence of the intervention.
                        """
            )
        elif not sig and pos:
            stmt4 = textwrap.dedent(
                """
                        This means that, although the intervention
                        appears to have caused a positive effect, this
                        effect is not statistically significant when
                        considering the post-intervention period as a whole.
                        Individual days or shorter stretches within the
                        intervention period may of course still have had a
                        significant effect, as indicated whenever the lower
                        limit of the impact time series (lower plot) was
                        above zero.
                        """
            )
        elif not sig and not pos:
            stmt4 = textwrap.dedent(
                """
                        This means that, although it may look as though
                        the intervention has exerted a negative effect on
                        the response variable when considering the
                        intervention period as a whole, this effect is not
                        statistically significant, and so cannot be
                        meaningfully interpreted.
                        """
            )
        if not sig:
            stmt4 += textwrap.dedent(
                """
                        The apparent effect could be the result of random
                        fluctuations that are unrelated to the intervention.
                        This is often the case when the intervention period
                        is very long and includes much of the time when the
                        effect has already worn off. It can also be the case
                        when the intervention period is too short to
                        distinguish the signal from the noise. Finally,
                        failing to find a significant effect can happen when
                        there are not enough control variables or when these
                        variables do not correlate well with the response
                        variable during the learning period."""
            )
        if p_value < alpha:
            stmt5 = textwrap.dedent(
                """The probability of obtaining this effect by
                chance is very small (Bayesian one-sided tail-area
                probability {p}). This means the
                causal effect can be considered statistically
                significant.""".format(
                    p=np.round(p_value, 3)
                )
            )
        else:
            stmt5 = """The probability of obtaining this effect by
                        chance is p = ", round(p, 3), "). This means the effect may
                        be spurious and would generally not be considered
                        statistically significant.""".format()

        print(textwrap.fill(stmt, width=width))
        print("\n")
        print(textwrap.fill(stmt2, width=width))
        print("\n")
        print(textwrap.fill(stmt3, width=width))
        print("\n")
        print(textwrap.fill(stmt4, width=width))
        print("\n")
        print(textwrap.fill(stmt5, width=width))

    def summary(self, output="summary", width=120, path=None):
        """reports a summary of the results

        Parameters
        ----------
        output: str
            can be summary or report. summary outputs a table.
            report outputs a natural language description of the
            findings
        width : int
            line width of the output. Only relevant if output == report
        path : str
            path to output summary to csv. Only relevant if output == summary

        """
        alpha = self.params["alpha"]
        confidence = "{}%".format(int((1 - alpha) * 100))
        post_period = self.params["post_period"]
        post_inf = self.inferences.loc[post_period[0] : post_period[1], :]
        post_point_resp = post_inf.loc[:, "response"]
        post_point_pred = post_inf.loc[:, "point_pred"]
        post_point_upper = post_inf.loc[:, "point_pred_upper"]
        post_point_lower = post_inf.loc[:, "point_pred_lower"]

        mean_resp = post_point_resp.mean()
        mean_resp_fmt = int(mean_resp)
        cum_resp = post_point_resp.sum()
        cum_resp_fmt = int(cum_resp)
        mean_pred = post_point_pred.mean()
        mean_pred_fmt = int(post_point_pred.mean())
        cum_pred = post_point_pred.sum()
        cum_pred_fmt = int(cum_pred)
        mean_lower = post_point_lower.mean()
        mean_lower_fmt = int(mean_lower)
        mean_upper = post_point_upper.mean()
        mean_upper_fmt = int(mean_upper)
        mean_ci_fmt = [mean_lower_fmt, mean_upper_fmt]
        cum_lower = post_point_lower.sum()
        cum_lower_fmt = int(cum_lower)
        cum_upper = post_point_upper.sum()
        cum_upper_fmt = int(cum_upper)
        cum_ci_fmt = [cum_lower_fmt, cum_upper_fmt]

        abs_effect = (post_point_resp - post_point_pred).mean()
        abs_effect_fmt = int(abs_effect)
        cum_abs_effect = (post_point_resp - post_point_pred).sum()
        cum_abs_effect_fmt = int(cum_abs_effect)
        abs_effect_lower = (post_point_resp - post_point_lower).mean()
        abs_effect_lower_fmt = int(abs_effect_lower)
        abs_effect_upper = (post_point_resp - post_point_upper).mean()
        abs_effect_upper_fmt = int(abs_effect_upper)
        abs_effect_ci_fmt = [abs_effect_lower_fmt, abs_effect_upper_fmt]
        cum_abs_lower = (post_point_resp - post_point_lower).sum()
        cum_abs_lower_fmt = int(cum_abs_lower)
        cum_abs_upper = (post_point_resp - post_point_upper).sum()
        cum_abs_upper_fmt = int(cum_abs_upper)
        cum_abs_effect_ci_fmt = [cum_abs_lower_fmt, cum_abs_upper_fmt]

        rel_effect = abs_effect / mean_pred * 100
        rel_effect_fmt = "{:.1f}%".format(rel_effect)
        cum_rel_effect = cum_abs_effect / cum_pred * 100
        cum_rel_effect_fmt = "{:.1f}%".format(cum_rel_effect)
        rel_effect_lower = abs_effect_lower / mean_pred * 100
        rel_effect_lower_fmt = "{:.1f}%".format(rel_effect_lower)
        rel_effect_upper = abs_effect_upper / mean_pred * 100
        rel_effect_upper_fmt = "{:.1f}%".format(rel_effect_upper)
        rel_effect_ci_fmt = [rel_effect_lower_fmt, rel_effect_upper_fmt]
        cum_rel_effect_lower = cum_abs_lower / cum_pred * 100
        cum_rel_effect_lower_fmt = "{:.1f}%".format(cum_rel_effect_lower)
        cum_rel_effect_upper = cum_abs_upper / cum_pred * 100
        cum_rel_effect_upper_fmt = "{:.1f}%".format(cum_rel_effect_upper)
        cum_rel_effect_ci_fmt = [cum_rel_effect_lower_fmt, cum_rel_effect_upper_fmt]

        # assuming approximately normal distribution
        # calculate standard deviation from the 95% conf interval
        std_pred = (
            mean_upper - mean_pred
        ) / 1.96  # from mean_upper = mean_pred + 1.96 * std
        # calculate z score
        z_score = (0 - mean_pred) / std_pred
        # pvalue from zscore
        p_value = st.norm.cdf(z_score)
        prob_causal = 1 - p_value
        p_value_perc = p_value * 100
        prob_causal_perc = prob_causal * 100

        if output == "summary":
            # Posterior inference {CausalImpact}
            summary = [
                [mean_resp_fmt, cum_resp_fmt],
                [mean_pred_fmt, cum_pred_fmt],
                [mean_ci_fmt, cum_ci_fmt],
                [" ", " "],
                [abs_effect_fmt, cum_abs_effect_fmt],
                [abs_effect_ci_fmt, cum_abs_effect_ci_fmt],
                [" ", " "],
                [rel_effect_fmt, cum_rel_effect_fmt],
                [rel_effect_ci_fmt, cum_rel_effect_ci_fmt],
                [" ", " "],
                ["{:.1f}%".format(p_value_perc), " "],
                ["{:.1f}%".format(prob_causal_perc), " "],
            ]
            summary = pd.DataFrame(
                summary,
                columns=["Average", "Cumulative"],
                index=[
                    "Actual",
                    "Predicted",
                    "95% CI",
                    " ",
                    "Absolute Effect",
                    "95% CI",
                    " ",
                    "Relative Effect",
                    "95% CI",
                    " ",
                    "P-value",
                    "Prob. of Causal Effect",
                ],
            )
            df_print(summary, path)
        elif output == "report":
            self._print_report(
                mean_pred_fmt,
                mean_resp_fmt,
                mean_lower_fmt,
                mean_upper_fmt,
                abs_effect_fmt,
                abs_effect_upper_fmt,
                abs_effect_lower_fmt,
                rel_effect_fmt,
                rel_effect_upper_fmt,
                rel_effect_lower_fmt,
                cum_resp_fmt,
                cum_pred_fmt,
                cum_lower_fmt,
                cum_upper_fmt,
                confidence,
                cum_rel_effect_lower,
                cum_rel_effect_upper,
                cum_rel_effect,
                width,
                p_value,
                alpha,
            )
        else:
            raise ValueError(
                "Output argument must be either 'summary' " + "or 'report'"
            )

    def plot(
        self,
        panels=None,
        figsize=(15, 12),
        fname=None
    ):
        if panels is None:
            panels = ["original", "pointwise", "cumulative"]
        plt = get_matplotlib()
        fig = plt.figure(figsize=figsize)

        data_inter = self.params["pre_period"][1]
        if isinstance(data_inter, pd.DatetimeIndex):
            data_inter = pd.Timestamp(data_inter)

        inferences = self.inferences.iloc[self.plot_index:, :]

        # Observation and regression components
        if "original" in panels:
            ax1 = plt.subplot(3, 1, 1)
            plt.plot(inferences.point_pred, "r--", linewidth=2, label="model")
            plt.plot(inferences.response, "k", linewidth=2, label="endog")

            plt.axvline(data_inter, c="k", linestyle="--")

            plt.fill_between(
                inferences.index,
                inferences.point_pred_lower,
                inferences.point_pred_upper,
                facecolor="gray",
                interpolate=True,
                alpha=0.25,
            )
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.legend(loc="upper left")
            plt.title("Observation vs prediction")

        if "pointwise" in panels:
            # Pointwise difference
            if "ax1" in locals():
                ax2 = plt.subplot(312, sharex=ax1)
            else:
                ax2 = plt.subplot(312)
            lift = inferences.point_effect
            plt.plot(lift, "r--", linewidth=2)
            plt.plot(self.data.index, np.zeros(self.data.shape[0]), "g-", linewidth=2)
            plt.axvline(data_inter, c="k", linestyle="--")

            lift_lower = inferences.point_effect_lower
            lift_upper = inferences.point_effect_upper

            plt.fill_between(
                inferences.index,
                lift_lower,
                lift_upper,
                facecolor="gray",
                interpolate=True,
                alpha=0.25,
            )
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.title("Difference")

        # Cumulative impact
        if "cumulative" in panels:
            if "ax1" in locals():
                plt.subplot(313, sharex=ax1)
            elif "ax2" in locals():
                plt.subplot(313, sharex=ax2)
            else:
                plt.subplot(313)
            plt.plot(
                inferences.index,
                inferences.cum_effect,
                "r--",
                linewidth=2,
            )

            plt.plot(self.data.index, np.zeros(self.data.shape[0]), "g-", linewidth=2)
            plt.axvline(data_inter, c="k", linestyle="--")

            plt.fill_between(
                inferences.index,
                inferences.cum_effect_lower,
                inferences.cum_effect_upper,
                facecolor="gray",
                interpolate=True,
                alpha=0.25,
            )
            plt.axis([inferences.index[0], inferences.index[-1], None, None])

            plt.title("Cumulative Impact")
        plt.xlabel("$T$")
        
        text = ('Note: The first %d observations are not shown, due to'
                    ' approximate diffuse initialization.')
        fig.text(0.1, 0.01, text % self.plot_index, fontsize='large')
        
        if fname is None:
            plt.show()
        else:
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
