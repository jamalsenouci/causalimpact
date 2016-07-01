def standardize_all_variables(data):
    """Standardize all columns of a given time series.
          Args:
              data: Pandas DataFrame with one or more columns

          Returns:
              list of:
                  data: standardized data
                  UnStandardize: function for undoing the transformation of the
                  first column in the provided data
    """
    data_mu = data.mean(skipna=True)
    data_sd = data.std(skipna=True)
    data = (data - data_mu)
    data_sd = data_sd.fillna(1)

    data[data != 0] = data[data != 0] / data_sd

    y_mu = data_mu[0]
    y_sd = data_sd[0]

    return {"data": data, "orig_std_params": (y_mu, y_sd)}


def unstandardize(data, orig_std_params):
    """Function for reversing the standardization of the first column in the
    provided data.
    """
    y_mu = orig_std_params[0]
    y_sd = orig_std_params[1]
    data = data.mul(y_sd, axis=1)
    data = data.add(y_mu, axis=1)
    return data
