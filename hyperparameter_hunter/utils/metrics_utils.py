"""This module defines helper functions for assisting in the evaluation of metrics"""


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
        The function to pass to XGBoost's :meth:`fit`, with signature: (`prediction`, `target`). It
        will return a tuple of (`metric_name`: str, `metric_value`: float)"""

    def eval_metric(prediction, target):
        """Evaluate a custom metric for use as the `eval_metric` kwarg in
        :meth:`xgboost.sklearn.XGBModel.fit`

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
