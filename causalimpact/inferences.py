import numpy as np
import pandas as pd
from causalimpact.misc import unstandardize


def compile_posterior_inferences(results, exog_post, alpha,
                                 orig_std_params, estimation):
    if estimation == "MLE":
        # Compute point predictions of counterfactual (in standardized space)
        predict = results.get_prediction()
        forecast = results.get_forecast(
            steps=len(exog_post), exog=exog_post, alpha=alpha)

        # Compile summary statistics (in original space)
        pre_pred = predict.predicted_mean
        post_pred = forecast.predicted_mean
        point_pred = np.concatenate([pre_pred, post_pred])
        pre_ci = predict.conf_int(alpha=alpha)
        post_ci = forecast.conf_int(alpha=alpha)
        ci = pd.concat([pre_ci, post_ci])

        point_pred_upper = ci["upper y"]
        point_pred_lower = ci["lower y"]
        response = np.concatenate([results.data.orig_endog,
                                   [np.nan] * len(exog_post)])
        cum_response = np.cumsum(response)
        cum_pred = np.cumsum(point_pred)
        cum_pred_upper = np.cumsum(point_pred_upper)
        cum_pred_lower = np.cumsum(point_pred_lower)
        point_effect = response - point_pred
        point_effect_upper = response - point_pred_upper
        point_effect_lower = response - point_pred_lower
        cum_effect = np.cumsum(point_effect)
        cum_effect_upper = np.cumsum(point_effect_upper)
        cum_effect_lower = np.cumsum(point_effect_lower)

        # Create DataFrame of results
        data = np.array([response, cum_response, point_pred, point_pred_upper,
                         point_pred_lower, cum_pred, cum_pred_lower,
                         cum_pred_upper, point_effect, point_effect_lower,
                         point_effect_upper, cum_effect, cum_effect_lower,
                         cum_effect_upper]).T
        stand_series = pd.DataFrame(data=data,
                                    columns=["response", "cum_response",
                                             "point_pred", "point_pred_upper",
                                             "point_pred_lower", "cum_pred",
                                             "cum_pred_lower",
                                             "cum_pred_upper", "point_effect",
                                             "point_effect_lower",
                                             "point_effect_upper",
                                             "cum_effect", "cum_effect_lower",
                                             "cum_effect_upper"])

        # Undo standardization (if any)
        series = unstandardize(stand_series, orig_std_params)
        # summary = compile_summary_table(data_post, predict_mean, alpha)
        # report = interpret_summary_table(summary)

        inferences = {"series": series,
                      # "summary": summary,
                      #  "report": report
                      }
        return inferences
    else:
        raise NotImplementedError()


def compile_na_inferences():
    pass
