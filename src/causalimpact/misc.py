import pandas as pd


def standardize_all_variables(data, pre_period, post_period):
    """Standardize all columns of a given time series.
    Args:
        data: Pandas DataFrame with one or more columns

    Returns:
        dict:
            data: standardized data
            UnStandardize: function for undoing the transformation of the
            first column in the provided data
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("``data`` must be of type `pandas.DataFrame`")

    if not (pd.api.types.is_list_like(pre_period, list) and pd.api.types.is_list_like(post_period)):
        raise ValueError("``pre_period`` and ``post_period``must be listlike")

    data_mu = data.loc[pre_period[0] : pre_period[1], :].mean(skipna=True)
    data_sd = data.loc[pre_period[0] : pre_period[1], :].std(skipna=True, ddof=0)
    data = data - data_mu
    data_sd = data_sd.fillna(1)

    data[data != 0] = data[data != 0] / data_sd
    y_mu = data_mu[0]
    y_sd = data_sd[0]

    data_pre = data.loc[pre_period[0] : pre_period[1], :]
    data_post = data.loc[post_period[0] : post_period[1], :]

    return {
        "data_pre": data_pre,
        "data_post": data_post,
        "orig_std_params": (y_mu, y_sd),
    }


def unstandardize(data, orig_std_params):
    """Function for reversing the standardization of the first column in the
    provided data.
    """
    data = pd.DataFrame(data)
    y_mu = orig_std_params[0]
    y_sd = orig_std_params[1]
    data = data.mul(y_sd, axis=1)
    data = data.add(y_mu, axis=1)
    return data


def df_print(data, path=None):
    if path:
        data.to_csv(path)
    print(data)


def get_matplotlib():  # pragma: no cover
    """Wrapper function to facilitate unit testing the `plot` tool by removing
    the strong dependencies of matplotlib.

    Returns:
        module matplotlib.pyplot
    """
    import matplotlib.pyplot as plt

    return plt
