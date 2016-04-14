import numpy as np


def standardize(y):
    """ Standardizes a vector {y} (to obtain mean 0 and SD 1). The original
      vector can be restored using {UnStandardize()}, which is a function
      that is supplied as part of the return value.

        Args:
            y: numeric list (may contain {NA} values)

        Returns:
            list of:
                y: standardized version of the input
                UnStandardize: function that restores the original data

        Examples:
            x = [1, 2, 3, 4, 5]
            result = causalimpact.standardize(x)
            y = result["unstandardize"](result["y"])"""
    y_mu = np.nanmean(y)
    y_sd = np.nanstd(y)
    y = (y - y_mu)
    if not np.isnan(y_sd) and y_sd > 0:
        y = y / y_sd

    return [y, unstandardize(y, y_mu, y_sd)]


def unstandardize(y, y_mu, y_sd):
    if not np.isnan(y_sd) and y_sd > 0:
        y = y * y_sd
    y = y + y_mu
    return y
