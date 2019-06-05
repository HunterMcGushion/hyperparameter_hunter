"""This module defines :class:`hyperparameter_hunter.metrics.ScoringMixIn` which enables
:class:`hyperparameter_hunter.experiments.BaseExperiment` to score predictions and collect the
results of those evaluations

Related
-------
:mod:`hyperparameter_hunter.experiments`
    This module uses :class:`hyperparameter_hunter.metrics.ScoringMixIn` as the only explicit parent
    class to :class:`hyperparameter_hunter.experiments.BaseExperiment` (that is, the only parent
    class that isn't bestowed upon it by
    :class:`hyperparameter_hunter.experiment_core.ExperimentMeta`)"""
##################################################
# Import Miscellaneous Assets
##################################################
from collections import OrderedDict
from contextlib import suppress
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Tuple, Union

##################################################
# Import Learning Assets
##################################################
from sklearn import metrics as sk_metrics
from sklearn.utils.multiclass import type_of_target, unique_labels

##################################################
# Declare Global Variables
##################################################
data_types = ("__in_fold", "__oof", "__holdout")
ArrayLike = Union[Iterable, pd.DataFrame]


##################################################
# Metric and Metrics Map Helpers
##################################################
class Metric(object):
    def __init__(
        self,
        name: str,
        metric_function: Union[callable, str, None] = None,
        direction: str = "infer",
    ):
        """Class to encapsulate all necessary information for identifying, calculating, and
        evaluating metrics results

        Parameters
        ----------
        name: String
            Identifying name of the metric. Should be unique relative to any other metric names that
            might be provided by the user
        metric_function: Callable, string, None, default=None
            If callable, should expect inputs of form (target, prediction), and return a float. If
            string, will be treated as an attribute in :mod:`sklearn.metrics`. If None, `name`
            will be treated as an attribute in :mod:`sklearn.metrics`, the value of which will be
            retrieved and used as `metric_function`
        direction: {"infer", "max", "min"}, default="infer"
            How to evaluate the result of `metric_function` relative to previous results produced by
            it. "max" signifies that metric values should be maximized, and that higher metric
            values are better than lower values; it should be used for measures of accuracy. "min"
            signifies that metric values should be minimized, and that lower metric values are
            better than higher values; it should be used for measures of error or loss. If "infer",
            `direction` will be set to: 1) "min" if `name` contains one of the following strings:
            ["error", "loss"]; or 2) "max" if `name` contains neither of the aforementioned strings

        Notes
        -----
        Because `direction` = "infer" only looks for "error"/"loss" in `name` , common abbreviations
        for error measures may be ignored, including but not limited to, the following:
        "mae" for "mean_absolute_error"; "rmsle" for "root_mean_squared_logarithmic_error"; or
        simply "hinge", or "cross_entropy" without an "error"/"loss" suffix. In cases such as these,
        provide an explicit `direction` = "min" to avoid backwards optimization and leaderboards

        Examples
        --------
        >>> Metric("roc_auc_score")  # doctest: +ELLIPSIS
        Metric(roc_auc_score, <function roc_auc_score at 0x...>, max)
        >>> Metric("roc_auc_score", sk_metrics.roc_auc_score)  # doctest: +ELLIPSIS
        Metric(roc_auc_score, <function roc_auc_score at 0x...>, max)
        >>> Metric("my_f1_score", "f1_score")  # doctest: +ELLIPSIS
        Metric(my_f1_score, <function f1_score at 0x...>, max)
        >>> Metric("hamming_loss", sk_metrics.hamming_loss)  # doctest: +ELLIPSIS
        Metric(hamming_loss, <function hamming_loss at 0x...>, min)
        >>> Metric("r2_score", sk_metrics.r2_score, direction="min")  # doctest: +ELLIPSIS
        Metric(r2_score, <function r2_score at 0x...>, min)"""
        self.name = name
        self.metric_function = self._set_metric_function(metric_function)
        self.direction = self._set_direction(direction)

    def __str__(self):
        return "Metric({}, {}, {})".format(self.name, self.metric_function.__name__, self.direction)

    def __repr__(self):
        return "Metric({}, {}, {})".format(self.name, self.metric_function, self.direction)

    def __call__(self, target, prediction):
        return self.metric_function(target, prediction)

    def _set_direction(self, direction):
        """Ensure provided `direction` is valid and inferred if necessary

        Parameters
        ----------
        direction: {"infer", "max", "min"}
            See `direction` documentation of :meth:`Metric.__init__`

        Returns
        -------
        String
            One of "min", or "max" depending on explicit `direction`/inference"""
        if direction == "infer":
            return "min" if any(_ in self.name for _ in ["error", "loss"]) else "max"
        elif direction not in ["max", "min"]:
            raise ValueError(f"`direction` must be 'infer', 'max', or 'min', not {direction}")
        return direction

    def _set_metric_function(self, f):
        """Ensure provided `f` is a valid callable

        Parameters
        ----------
        f: Callable, string, None
            See `metric_function` documentation of :meth:`Metric.__init__`

        Returns
        -------
        Callable
            A function derived from `f` if `f` was not already callable. Else `f`"""
        if not callable(f):
            try:
                return sk_metrics.__getattribute__(self.name if f is None else f)
            except AttributeError:
                raise AttributeError(f"`sklearn.metrics` has no attribute: {f or self.name}")
        return f


def format_metrics(metrics: Union[Dict, List]) -> Dict[str, Metric]:
    """Properly format iterable `metrics` to contain instances of :class:`Metric`

    Parameters
    ----------
    metrics: Dict, List
        Iterable describing the metrics to be recorded, along with a means to compute the value of
        each metric. Should be of one of the two following forms:

        List Form:

        * ["<metric name>", "<metric name>", ...]:
          Where each value of the list is a string that names an attribute in :mod:`sklearn.metrics`
        * [`Metric`, `Metric`, ...]:
          Where each value of the list is an instance of :class:`Metric`
        * [(<\*args>), (<\*args>), ...]:
          Where each value of the list is a tuple of arguments that will be used to instantiate a
          :class:`Metric`. Arguments given in tuples must be in order expected by :class:`Metric`

        Dict Form:

        * {"<metric name>": <metric_function>, ...}:
          Where each key is a name for the corresponding metric callable, which is used to compute
          the value of the metric
        * {"<metric name>": (<metric_function>, <direction>), ...}:
          Where each key is a name for the corresponding metric callable and direction, all of which
          are used to instantiate a :class:`Metric`
        * {"<metric name>": "<sklearn metric name>", ...}:
          Where each key is a name for the metric, and each value is the name of the attribute in
          :mod:`sklearn.metrics` for which the corresponding key is an alias
        * {"<metric name>": None, ...}:
          Where each key is the name of the attribute in :mod:`sklearn.metrics`
        * {"<metric name>": `Metric`, ...}:
          Where each key names an instance of :class:`Metric`. This is the internally-used format to
          which all other formats will be converted

        Metric callable functions should expect inputs of form (target, prediction), and should
        return floats. See the documentation of :class:`Metric` for information regarding expected
        parameters and types

    Returns
    -------
    metrics_dict: Dict
        Cast of `metrics` to a dict, in which values are instances of :class:`Metric`

    Examples
    --------
    >>> format_metrics(["roc_auc_score", "f1_score"])  # doctest: +ELLIPSIS
    {'roc_auc_score': Metric(roc_auc_score, <function roc_auc_score at 0x...>, max), 'f1_score': Metric(f1_score, <function f1_score at 0x...>, max)}
    >>> format_metrics([Metric("log_loss"), Metric("r2_score", direction="min")])  # doctest: +ELLIPSIS
    {'log_loss': Metric(log_loss, <function log_loss at 0x...>, min), 'r2_score': Metric(r2_score, <function r2_score at 0x...>, min)}
    >>> format_metrics({"log_loss": Metric("log_loss"), "r2_score": Metric("r2_score", direction="min")})  # doctest: +ELLIPSIS
    {'log_loss': Metric(log_loss, <function log_loss at 0x...>, min), 'r2_score': Metric(r2_score, <function r2_score at 0x...>, min)}
    >>> format_metrics([("log_loss", None), ("my_r2_score", "r2_score", "min")])  # doctest: +ELLIPSIS
    {'log_loss': Metric(log_loss, <function log_loss at 0x...>, min), 'my_r2_score': Metric(my_r2_score, <function r2_score at 0x...>, min)}
    >>> format_metrics({"roc_auc": sk_metrics.roc_auc_score, "f1": sk_metrics.f1_score})  # doctest: +ELLIPSIS
    {'roc_auc': Metric(roc_auc, <function roc_auc_score at 0x...>, max), 'f1': Metric(f1, <function f1_score at 0x...>, max)}
    >>> format_metrics({"log_loss": (None, ), "my_r2_score": ("r2_score", "min")})  # doctest: +ELLIPSIS
    {'log_loss': Metric(log_loss, <function log_loss at 0x...>, min), 'my_r2_score': Metric(my_r2_score, <function r2_score at 0x...>, min)}
    >>> format_metrics({"roc_auc": "roc_auc_score", "f1": "f1_score"})  # doctest: +ELLIPSIS
    {'roc_auc': Metric(roc_auc, <function roc_auc_score at 0x...>, max), 'f1': Metric(f1, <function f1_score at 0x...>, max)}
    >>> format_metrics({"roc_auc_score": None, "f1_score": None})  # doctest: +ELLIPSIS
    {'roc_auc_score': Metric(roc_auc_score, <function roc_auc_score at 0x...>, max), 'f1_score': Metric(f1_score, <function f1_score at 0x...>, max)}
    """
    if metrics and isinstance(metrics, dict):
        if all(isinstance(_, Metric) for _ in metrics.values()):
            return metrics

        metrics = [
            (k,) + (v if isinstance(v, (tuple, Metric)) else (v,)) for k, v in metrics.items()
        ]
    elif not (metrics and isinstance(metrics, list)):
        raise TypeError(f"`metrics` must be a non-empty list or dict. Received: {metrics}")

    metrics_dict = {}

    for value in metrics:
        if not isinstance(value, Metric):
            if not isinstance(value, tuple):
                value = (value,)

            metrics_dict[value[0]] = Metric(*value)
        else:
            metrics_dict[value.name] = value

    if not all(metrics_dict):
        raise TypeError(f"`metrics` keys must all be truthy. Received: {metrics_dict}")

    return metrics_dict


def get_formatted_target_metric(
    target_metric: Union[tuple, str, None], metrics: dict, default_dataset: str = "oof"
) -> Tuple[str, str]:
    """Return a properly formatted target_metric tuple for use with navigating evaluation results

    Parameters
    ----------
    target_metric: Tuple, String, or None
        Path denoting metric to be used. If tuple, the first value should be in ['oof', 'holdout',
        'in_fold'], and the second value should be the name of a metric supplied in `metrics`.
        If str, should be one of the two values from the tuple form. Else, a value will be chosen
    metrics: Dict
        Properly formatted `metrics` as produced by :func:`metrics.format_metrics`, in which
        keys are strings identifying metrics, and values are instances of :class:`metrics.Metric`.
        See the documentation of :func:`metrics.format_metrics` for more information on
        different metrics formats
    default_dataset: {"oof", "holdout", "in_fold"}, default="oof"
        The default dataset type value to use if one is not provided

    Returns
    -------
    target_metric: Tuple
        A formatted target_metric containing two strings: a dataset_type, followed by a metric name

    Examples
    --------
    >>> get_formatted_target_metric(('holdout', 'roc_auc_score'), format_metrics(['roc_auc_score', 'f1_score']))
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric(('holdout',), format_metrics(['roc_auc_score', 'f1_score']))
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric('holdout', format_metrics(['roc_auc_score', 'f1_score']))
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric('holdout', format_metrics({'roc': 'roc_auc_score', 'f1': 'f1_score'}))
    ('holdout', 'roc')
    >>> get_formatted_target_metric('roc_auc_score', format_metrics(['roc_auc_score', 'f1_score']))
    ('oof', 'roc_auc_score')
    >>> get_formatted_target_metric(None, format_metrics(['f1_score', 'roc_auc_score']))
    ('oof', 'f1_score')"""
    ok_datasets = ["oof", "holdout", "in_fold"]

    if isinstance(target_metric, str):
        target_metric = (target_metric,)
    elif target_metric is None:
        target_metric = (default_dataset,)

    if not isinstance(target_metric, tuple):
        raise TypeError(f"`target_metric` should be: tuple, str, or None. Received {target_metric}")
    elif len(target_metric) > 2:
        raise ValueError(f"`target_metric` should be tuple of length 2. Received {target_metric}")
    elif len(target_metric) == 1:
        if target_metric[0] in ok_datasets:
            # Just a dataset was provided - Need metric name
            first_metric_key = list(metrics.keys())[0]
            target_metric = target_metric + (first_metric_key,)
            # TODO: Above will cause problems if `Environment.metrics_params['oof']` is not "all"
        else:
            # Just a metric name was provided - Need dataset type
            target_metric = (default_dataset,) + target_metric

    if not any([_ == target_metric[0] for _ in ok_datasets]):
        raise ValueError(f"`target_metric`[0] must be in {ok_datasets}. Received {target_metric}")
    if not target_metric[1] in metrics.keys():
        raise ValueError(f"target_metric[1]={target_metric[1]} not in metrics={metrics}")

    return target_metric


##################################################
# ScoringMixIn and Helpers
##################################################
class ScoringMixIn(object):
    def __init__(self, metrics, in_fold="all", oof="all", holdout="all", do_score=True):
        """MixIn class to manage metrics to record for each dataset type, and perform evaluations

        Parameters
        ----------
        metrics: Dict, List
            Specifies all metrics to be used by their id keys, along with a means to compute the
            metric. If list, all values must be strings that are attributes in
            :mod:`sklearn.metrics`. If dict, key/value pairs must be of the form:
            (<id>, <callable/None/str sklearn.metrics attribute>), where "id" is a str name for the
            metric. Its corresponding value must be one of: 1) a callable to calculate the metric,
            2) None if the "id" key is an attribute in `sklearn.metrics` and should be used to fetch
            a callable, 3) a string that is an attribute in `sklearn.metrics` and should be used to
            fetch a callable. Metric callable functions should expect inputs of form
            (target, prediction), and should return floats
        in_fold: List of strings, None, default=<all ids in `metrics`>
            Which metrics (from ids in `metrics`) should be recorded for in-fold data
        oof: List of strings, None, default=<all ids in `metrics`>
            Which metrics (from ids in `metrics`) should be recorded for out-of-fold data
        holdout: List of strings, None, default=<all ids in `metrics`>
            Which metrics (from ids in `metrics`) should be recorded for holdout data
        do_score: Boolean, default=True
            This is experimental. If False, scores will be neither calculated nor recorded for the
            duration of the experiment

        Notes
        -----
        For each kwarg in [`in_fold`, `oof`, `holdout`], the following must be true: if the value
        of the kwarg is a list, its contents must be a subset of `metrics`"""
        self.metrics = format_metrics(metrics)
        self.do_score = do_score

        #################### ScoringMixIn-Only Mangled Attributes ####################
        self.__in_fold = in_fold if in_fold else []
        self.__oof = oof if oof else []
        self.__holdout = holdout if holdout else []

        self._validate_metrics_list_parameters()
        self.last_evaluation_results = dict(in_fold=None, oof=None, holdout=None)

    def _validate_metrics_list_parameters(self):
        """Ensure metrics lists input parameters are correct types and compatible with each other"""
        for (_d_type, _m_val) in [(_, getattr(self, f"_ScoringMixIn{_}")) for _ in data_types]:
            if _m_val == "all":
                setattr(self, f"_ScoringMixIn{_d_type}", list(self.metrics.keys()))
            elif not isinstance(_m_val, list):
                raise TypeError(f"{_d_type} must be one of: ['all', None, <list>], not {_m_val}")
            else:
                for _id in _m_val:
                    if not isinstance(_id, str):
                        raise TypeError(f"{_d_type} values must be of type str. Received {_id}")
                    if _id not in self.metrics.keys():
                        raise KeyError(f"{_d_type} values must be in metrics. '{_id}' is not")

    def evaluate(self, data_type, target, prediction, return_list=False, dry_run=False):
        """Apply metric(s) to the given data to calculate the value of the `prediction`

        Parameters
        ----------
        data_type: {"in_fold", "oof", "holdout"}
            The type of dataset for which `target` and `prediction` arguments are being provided
        target: Array-like
            True labels for the data. Should be same shape as `prediction`
        prediction: Array-like
            Predicted labels for the data. Should be same shape as `target`
        return_list: Boolean, default=False
            If True, return list of tuples instead of dict. See "Returns" section below for details
        dry_run: Boolean, default=False
            If True, the value of :attr:`last_evaluation_results` will not be updated to include
            the returned `_result`. The core library callbacks operate under the assumption that
            `last_evaluation_results` will be updated as usual, so restrict usage to debugging or
            :func:`~hyperparameter_hunter.callbacks.bases.lambda_callback` implementations

        Returns
        -------
        _result: OrderedDict, or list
            A dict whose keys are all metric keys supplied for `data_type`, and whose values are the
            results of each metric. If `return_list` is True, returns a list of tuples of:
            (<`data_type` metric str>, <metric result>)

        Notes
        -----
        The required types of `target` and `prediction` are entirely dependent on the metric
        callable's expectations"""
        if self.do_score is False:
            return

        _metric_ids = getattr(self, f"_ScoringMixIn__{data_type}")
        _result = []
        target = np.asarray(target)
        prediction = np.asarray(prediction)

        for _metric_id in _metric_ids:
            try:
                _metric_value = self.metrics[_metric_id](target, prediction)
            except ValueError:
                # Check if target contains integer types, but prediction contains floats
                prediction = get_clean_prediction(target, prediction)
                _metric_value = self.metrics[_metric_id](target, prediction)

            _result.append((_metric_id, _metric_value))

        _result = _result if return_list else OrderedDict(_result)

        if not dry_run:
            self.last_evaluation_results[data_type] = _result

        return _result


def _is_int(a):
    """Determine whether the values of `a` are of type `numpy.int`

    Parameters
    ----------
    a: Array-like
        Array, whose values' types will be checked

    Returns
    -------
    Boolean
        True if the `dtype` of the values of `a` == `numpy.int`. Else, False

    Examples
    --------
    >>> assert _is_int(np.array([0, 1, 2, 3]))
    >>> assert _is_int(pd.DataFrame([0, 1], [2, 3]))
    >>> assert not _is_int(np.array([0.0, 1.1, 2.2, 3.3]))
    >>> assert not _is_int(pd.DataFrame([0.0, 1.1], [2.2, 3.3]))"""
    try:
        return a.values.dtype == np.int
    except AttributeError:
        return a.dtype == np.int


classification_target_types = [
    "binary",
    "multiclass",
    "multiclass-multioutput",
    "multilabel-indicator",
    "multilabel-sequences",
]


def get_clean_prediction(target: ArrayLike, prediction: ArrayLike):
    """Create `prediction` that is of a form comparable to `target`

    Parameters
    ----------
    target: Array-like
        True labels for the data. Should be same shape as `prediction`
    prediction: Array-like
        Predicted labels for the data. Should be same shape as `target`

    Returns
    -------
    prediction: Array-like
        If `target` types are ints, and `prediction` types are not, given predicted labels clipped
        between the min, and max of `target`, then rounded to the nearest integer. Else, original
        predicted labels"""
    target_type = type_of_target(target)
    prediction_type = type_of_target(prediction)
    # ValueError probably: "Classification metrics can't handle a mix of binary and continuous targets"
    if _is_int(target) and not _is_int(prediction):
        #################### Get Minimum/Maximum ####################
        target_min, target_max = target.min(), target.max()

        with suppress(TypeError):  # Bypass one-dimensional arrays, whose min/max should be a scalar
            if (len(target_min) == 1) and (len(target_max) == 1):
                target_min, target_max = target_min[0], target_max[0]

        #################### Clip/Round `prediction` ####################
        try:
            prediction = np.clip(prediction, target_min, target_max)
        except ValueError:
            prediction = prediction.clip(target_min, target_max, axis=1)
        finally:
            prediction = prediction.astype(np.float64)
            prediction = np.rint(prediction)
    elif target_type in classification_target_types and prediction_type.startswith("continuous"):
        prediction = classify_output(target, prediction)

    # TODO: One-hot-encoded outputs will be of type "multiclass-multioutput" - Handle it
    return prediction


def classify_output(target, prediction):
    """Force continuous `prediction` into the discrete, classified space of `target`.
    This is not an output/feature transformer akin to SKLearn's discretization transformers. This
    function is intended for use in the very specific case of having a `target` that is
    classification-like ("binary", "multiclass", etc.), with `prediction` that resembles a
    "continuous" target, despite being made for `target`. The most common reason for this occurrence
    is that `prediction` is actually the division-averaged predictions collected along the course
    of a :class:`~hyperparameter_hunter.experiments.CVExperiment`. In this case, the original model
    predictions should have been classification-like; however, due to disagreement in the division
    predictions, the resulting average predictions appear to be continuous

    Parameters
    ----------
    target: Array-like
        # TODO: ...
    prediction: Array-like
        # TODO: ...

    Returns
    -------
    numpy.array
        # TODO: ...

    Notes
    -----
    Target types used by this function are defined by `sklearn.utils.multiclass.type_of_target`.

    If a `prediction` value is exactly between two `target` values, it will assume the lower of the
    two values. For example, given a single prediction of 1.5 and unique `labels` of [0, 1, 2, 3],
    the value of that prediction will be 1, rather than 2

    Examples
    --------
    >>> import numpy as np
    >>> classify_output(np.array([0, 3, 1, 2]), [0.5, 1.51, 0.66, 4.9])
    array([0, 2, 1, 3])
    >>> classify_output(np.array([0, 1, 2, 3]), [0.5, 1.51, 0.66, 4.9])
    array([0, 2, 1, 3])
    >>> # TODO: ... Add more examples, including binary classification
    """
    # MARK: Might be ignoring 1-dimensional, label encodings, like 2nd case in `test_get_clean_prediction`:
    #   ([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], [1.0, 0.0, 1.0, 0.0])
    labels = unique_labels(target)  # FLAG: ORIGINAL
    # labels = unique_labels(*target)  # FLAG: TEST
    return np.array([labels[(np.abs(labels - _)).argmin()] for _ in prediction])


##################################################
# Miscellaneous Utilities
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


if __name__ == "__main__":
    pass
