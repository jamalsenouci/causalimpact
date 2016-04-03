# from statsmodels import bsts
import pandas as pd
from pandas.core.common import PandasError
import numpy as np

def _format_input_data(data):
    """Checks and formats the <data> argument provided to CausalImpact().
    Args:
        data: Pandas DataFrame

    Returns:
        correctly formatted Pandas DataFrame
    """

    """ If <data> is a Pandas DataFrame and the first column is 'date',
    try to convert
    """

    if type(data) == pd.DataFrame and type(data.columns[0]) == str:
        if data.columns[0].lower() in ["date", "time"]:
            data = data.set_index(data.columns[0])

    # Try to convert to Pandas DataFrame
    try:
        data = pd.DataFrame(data)
    except:
        raise PandasError("could not convert input data to Pandas DataFrame")

    # Must have at least 3 time points
    if len(data.index) < 3:
        raise ValueError("data must have at least 3 time points")

    # Must not have NA in covariates (if any)
    if len(data.columns) >= 2:
        if np.any(pd.isnull(data.ix[:, 1:])):
            raise ValueError("covariates must not contain null values")

    return data


def _format_input_prepost(pre_period, post_period, data):
    """Checks and formats the <pre_period> and <post_period> input arguments.

    Args:
        pre_period: two-element list
        post_period: two-element list
        data: already-checked Pandas DataFrame, for reference only
    """

    import pandas as pd
    import numpy as np

    if pre_period is None or post_period is None:
        raise ValueError("pre_period and post period must not contain null values")
    if type(pre_period) is not list or type(post_period) is not list:
        raise ValueError("pre_period and post_period must bothe be lists")
    if len(pre_period) != 2 or len(post_period) != 2:
        raise ValueError("pre_period and post_period must both be of length 2")
    if np.any(pd.isnull(pre_period)) or np.any(pd.isnull(post_period)):
        raise ValueError("pre_period and post period must not contain null values")

    pre_dtype = np.array(pre_period).dtype
    post_dtype = np.array(post_period).dtype

    if data.index.dtype.kind != pre_dtype.kind or data.index.dtype.kind != post_dtype.kind:
        if data.index.dtype == int:
            pre_period = [int(elem) for elem in pre_period]
            post_period = [int(elem) for elem in post_period]
        elif data.index.dtype == float:
            pre_period = [float(elem) for elem in pre_period]
            post_period = [float(elem) for elem in post_period]
        else:
            raise ValueError("pre_period (" + pre_dtype.name + ") and post_period (" +\
                    post_dtype.name + ") should have the same class as the " +\
                    "time points in the data (" + data.index.dtype.name + ")")

    if pre_period[0] < data.index.min():
        print("Setting pre_period[1] to start of data: " + str(data.index.min()))
        pre_period[0] = data.index.min()
    if pre_period[1] > data.index.max():
        print("Setting pre_period[1] to end of data: " + str(data.index.max()))
        pre_period[1] = data.index.max()
    if post_period[1] > data.index.max():
        print("Setting post_period[1] to end of data: " + str(data.index.max()))
        post_period[1] = data.index.max()

    if pre_period[1] - pre_period[0] + 1 < 3:
        raise ValueError("pre_period must span at least 3 time points")
    if post_period[1] < post_period[0]:
        raise ValueError("post_period[1] must not be earlier than post_period[0]")
    if post_period[0] < pre_period[1]:
        raise ValueError("post_period[0] must not be earlier than pre_period[1]")

    return {"pre_period": pre_period, "post_period": post_period}


def _format_input(data, pre_period, post_period, model_args, bsts_model, post_period_response, alpha):
    """Checks and formats all input arguments supplied to CausalImpact(). See the
       documentation of CausalImpact() for details.

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

    import pandas as pd
    import numpy as np

    # Check that a consistent set of variables has been provided
    args = [data, pre_period, post_period, bsts_model, post_period_response]

    data_model_args = [True, True, True, False, False]
    bsts_model_args = [False, False, False, True, True]
    
    if np.any(pd.isnull(args) != data_model_args) and np.any(pd.isnull(args) != bsts_model_args):
        raise SyntaxError("must either provide data, pre_period, post_period, model_args " +\
                          "or bsts_model and post_period_response")

    # Check <data> and convert to Pandas DataFrame, with rows representing time points
    if data is not None:
        data = _format_input_data(data)

    # Check <pre_period> and <post_period>
    if data is not None:
        checked = _format_input_prepost(pre_period, post_period, data)
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

    """ Check only those parts of <model_args> that are used in this file. The other
     fields will be checked in FormatInputForConstructModel()."""

    # Check <standardize_data>
    if type(model_args["standardize_data"]) != bool:
        raise ValueError("model_args.standardize_data must be a boolean value")

    """ Check <bsts_model> TODO
    if bsts_model is not None:
        if type(bsts_model) != bsts:
            raise "bsts_model must be an object of class statsmodels.bsts"
    """

    # Check <post_period_response>
    if bsts_model is not None:
        if post_period_response is None:
            raise ValueError("post_period_response cannot be None")
        if not np.all(np.isreal(post_period_response)):
            raise ValueError("post_period_response must contain all real values")

    # Check <alpha>
    if not np.isreal(alpha):
        raise ValueError("alpha must be a real number")
    if np.isnan(alpha):
        raise ValueError("alpha must not be NA")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")

    # Return updated arguments
    return {"data": data, "pre_period": pre_period, "post_period": post_period,
            "model_args": model_args, "bsts_model": bsts_model,
            "post_period_response": post_period_response, "alpha": alpha}


def causalimpact():
    pass
