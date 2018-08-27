import numpy as np
import pandas as pd
from causalimpact.misc import unstandardize


def compile_posterior_inferences(results, data, df_pre, df_post,
                                 post_period_response, alpha,
                                 orig_std_params, estimation):
    if estimation == "MLE":
        # Compute point predictions of counterfactual (in standardized space)
        if df_post is not None:
            predict = results.get_prediction()
            forecast = results.get_forecast(
                steps=len(df_post),
                exog=df_post.iloc[:, 1:],
                alpha=alpha
            )
        else:
            pre_len = results.model.nobs - len(post_period_response)

            predict = results.get_prediction(end=pre_len - 1)
            forecast = results.get_prediction(start=pre_len)

            df_post = post_period_response
            df_post.index = pd.core.indexes.range.RangeIndex(
                start=pre_len,
                stop=pre_len + len(df_post),
                step=1
            )

        # Compile summary statistics (in original space)
        pre_pred = unstandardize(predict.predicted_mean, orig_std_params)
        pre_pred.index = df_pre.index

        post_pred = unstandardize(forecast.predicted_mean, orig_std_params)
        post_pred.index = df_post.index

        point_pred = pd.concat([pre_pred, post_pred])

        pre_ci = unstandardize(predict.conf_int(alpha=alpha), orig_std_params)
        pre_ci.index = df_pre.index

        post_ci = unstandardize(forecast.conf_int(alpha=alpha),
                                orig_std_params)

        post_ci.index = df_post.index
        ci = pd.concat([pre_ci, post_ci])

        point_pred_upper = ci["upper y"].to_frame()
        point_pred_lower = ci["lower y"].to_frame()

        response = data.iloc[:, 0]
        response_index = data.index

        response = pd.DataFrame(response)

        cum_response = np.cumsum(response)
        cum_pred = np.cumsum(point_pred)
        cum_pred_upper = np.cumsum(point_pred_upper)
        cum_pred_lower = np.cumsum(point_pred_lower)

        data = pd.concat([point_pred, point_pred_upper,
                          point_pred_lower, cum_pred, cum_pred_lower,
                          cum_pred_upper], axis=1)

        data = pd.concat([response,
                          cum_response], axis=1).join(data, lsuffix='l')

        data.columns = ["response", "cum_response", "point_pred",
                        "point_pred_upper", "point_pred_lower", "cum_pred",
                        "cum_pred_lower", "cum_pred_upper"]

        point_effect = (data.response - data.point_pred).to_frame()
        point_effect_upper = (data.response -
                              data.point_pred_upper).to_frame()

        point_effect_lower = (data.response -
                              data.point_pred_lower).to_frame()

        cum_effect = point_effect.copy()
        cum_effect.loc[df_pre.index[0]:df_pre.index[-1]] = 0
        cum_effect = np.cumsum(cum_effect)
        cum_effect_upper = point_effect_upper.copy()
        cum_effect_upper.loc[df_pre.index[0]:df_pre.index[-1]] = 0
        cum_effect_upper = np.cumsum(cum_effect_upper)
        cum_effect_lower = point_effect_lower.copy()
        cum_effect_lower.loc[df_pre.index[0]:df_pre.index[-1]] = 0
        cum_effect_lower = np.cumsum(cum_effect_lower)

        data = pd.concat([data, point_effect, point_effect_upper,
                          point_effect_lower, cum_effect, cum_effect_lower,
                          cum_effect_upper], axis=1)


        # Create DataFrame of results
        data.columns = ["response", "cum_response", "point_pred",
                        "point_pred_upper", "point_pred_lower", "cum_pred",
                        "cum_pred_lower", "cum_pred_upper", "point_effect",
                        "point_effect_lower", "point_effect_upper",
                        "cum_effect", "cum_effect_lower", "cum_effect_upper"]

        data.index = response_index
        # index = np.concatenate([results.data.orig_endog.index,
        #                         df_post.iloc[:, 0].index])
        # data.index = index
        # import pdb
        # pdb.set_trace()
        # Undo standardization (if any)
        series = data
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
