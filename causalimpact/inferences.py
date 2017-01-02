import numpy as np
from causalimpact.misc import unstandardize


def compile_posterior_inferences(model, data_post, alpha=0.05,
                                 orig_std_params=np.identity):

    # Compute point predictions of counterfactual (in standardized space)
    predict_mean = res.forecast(
        steps=len(data_post), exog=data_post, alpha=0.05)

    # Undo standardization (if any)
    predict_res = unstandardize(point_pred, orig_std_params)
    y = unstandardize(model.data.orig_endog, orig_std_params)

    # Compile summary statistics (in original space)
    predict_mean = predict_res.predicted_mean
    predict_ci = predict_res.conf_int(alpha=0.05)

    summary = compile_summary_table(data_post, predict_mean, alpha)
    report = interpret_summary_table(summary)

    inferences = {"series": series, "summary": summary, "report": report}

    # Compute cumulative predictions (in original space)


def compile_na_inferences():
    pass
