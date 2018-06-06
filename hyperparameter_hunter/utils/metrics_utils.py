##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.general_utils import flatten

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import pandas as pd

###############################################
# Import Learning Assets
###############################################
from sklearn import metrics as sk_metrics


def RMSLE(target, prediction):
    return sk_metrics.mean_squared_error(target, prediction) ** 0.5


def RMSLE_binary(target, prediction):
    try:
        _prediction = [sum(float(c) * (2 ** i) for i, c in enumerate(p[::-1])) for p in prediction]
    except ValueError:
        _prediction = [sum(float(c) * (2 ** i) for i, c in enumerate(p[::-1])) for p in prediction.values]

    try:
        _target = [sum(float(c) * (2 ** i) for i, c in enumerate(t[::-1])) for t in target.values]
    except AttributeError:  # target is a list, not a df, so has no 'values' attribute
        _target = [sum(float(c) * (2 ** i) for i, c in enumerate(t[::-1])) for t in target]

    _prediction_log = np.log1p(_prediction)
    _target_log = np.log1p(_target)

    return sk_metrics.mean_squared_error(_target_log, _prediction_log) ** 0.5


def gini_c(target, prediction):
    target = np.asarray(target)
    n = len(target)
    a_s = target[np.argsort(prediction)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n


def gini_normalized_c(target, prediction):
    if prediction.ndim == 2:  # Required for sklearn wrapper
        prediction = prediction[:, 1]
    return gini_c(target, prediction) / gini_c(target, target)


##################################################
# XGBoost Wrapper
##################################################
def wrap_xgboost_metric(metric, metric_name):
    """Create a function to use as the `eval_metric` kwarg for :meth:`xgboost.sklearn.XGBModel.fit`

    Parameters
    ----------
    metric: Function
        The function to calculate the value of metric, with signature: (`target`, `prediction`)
    metric_name: String
        The name of the metric being evaluated

    Returns
    -------
    eval_metric: Function
        The function to pass to XGBoost's :meth:`fit`, with signature: (`prediction`, `target`). It will return a tuple
        of (`metric_name`: str, `metric_value`: float)"""
    def eval_metric(prediction, target):
        """Evaluate a custom metric for use as the `eval_metric` kwarg in :meth:`xgboost.sklearn.XGBModel.fit`

        Parameters
        ----------
        prediction: Array-like
            Predicted values
        target: `xgboost.DMatrix`
            True labels

        Returns
        -------
        Tuple of (`metric_name`: str, `metric_value`: float)"""
        target = target.get_label()
        metric_value = metric(target, prediction)
        # return [(metric_name, metric_value)]
        return (metric_name, metric_value)

    return eval_metric


def gini_xgb(prediction, dtrain):
    target = dtrain.get_label()
    gini_score = gini_normalized_c(target, prediction)
    return [('gini', gini_score)]


if __name__ == '__main__':
    pass
