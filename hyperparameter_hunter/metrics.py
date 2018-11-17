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
import numpy as np

##################################################
# Import Learning Assets
##################################################
from sklearn import metrics as sk_metrics

##################################################
# Declare Global Variables
##################################################
data_types = ("__in_fold", "__oof", "__holdout")


class Metric(object):
    def __init__(self, name, metric_function=None, direction="infer"):
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
        direction: String in ["infer", "max", "min"], default="infer"
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
        Metric(r2_score, <function r2_score at 0x...>, min)
        """
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

    def get_xgboost_wrapper(self):
        # TODO: Move `utils.metrics_utils.wrap_xgboost_metric` here, and remove `utils.metrics_utils`
        raise NotImplementedError


def format_metrics_map(metrics_map):
    """Properly format iterable `metrics_map` to contain instances of :class:`Metric`

    Parameters
    ----------
    metrics_map: Dict, List
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
    metrics_map_dict: Dict
        Cast of `metrics_map` to a dict, in which values are instances of :class:`Metric`

    Examples
    --------
    >>> format_metrics_map(["roc_auc_score", "f1_score"])  # doctest: +ELLIPSIS
    {'roc_auc_score': Metric(roc_auc_score, <function roc_auc_score at 0x...>, max), 'f1_score': Metric(f1_score, <function f1_score at 0x...>, max)}
    >>> format_metrics_map([Metric("log_loss"), Metric("r2_score", direction="min")])  # doctest: +ELLIPSIS
    {'log_loss': Metric(log_loss, <function log_loss at 0x...>, min), 'r2_score': Metric(r2_score, <function r2_score at 0x...>, min)}
    >>> format_metrics_map([("log_loss", None), ("my_r2_score", "r2_score", "min")])  # doctest: +ELLIPSIS
    {'log_loss': Metric(log_loss, <function log_loss at 0x...>, min), 'my_r2_score': Metric(my_r2_score, <function r2_score at 0x...>, min)}
    >>> format_metrics_map({"roc_auc": sk_metrics.roc_auc_score, "f1": sk_metrics.f1_score})  # doctest: +ELLIPSIS
    {'roc_auc': Metric(roc_auc, <function roc_auc_score at 0x...>, max), 'f1': Metric(f1, <function f1_score at 0x...>, max)}
    >>> format_metrics_map({"log_loss": (None, ), "my_r2_score": ("r2_score", "min")})  # doctest: +ELLIPSIS
    {'log_loss': Metric(log_loss, <function log_loss at 0x...>, min), 'my_r2_score': Metric(my_r2_score, <function r2_score at 0x...>, min)}
    >>> format_metrics_map({"roc_auc": "roc_auc_score", "f1": "f1_score"})  # doctest: +ELLIPSIS
    {'roc_auc': Metric(roc_auc, <function roc_auc_score at 0x...>, max), 'f1': Metric(f1, <function f1_score at 0x...>, max)}
    >>> format_metrics_map({"roc_auc_score": None, "f1_score": None})  # doctest: +ELLIPSIS
    {'roc_auc_score': Metric(roc_auc_score, <function roc_auc_score at 0x...>, max), 'f1_score': Metric(f1_score, <function f1_score at 0x...>, max)}
    """
    if isinstance(metrics_map, dict):
        if all(isinstance(_, Metric) for _ in metrics_map.values()):
            return metrics_map

        metrics_map = [
            (k,) + (v if isinstance(v, (tuple, Metric)) else (v,)) for k, v in metrics_map.items()
        ]

    metrics_map_dict = {}

    for value in metrics_map:
        if not isinstance(value, Metric):
            if not isinstance(value, tuple):
                value = (value,)

            metrics_map_dict[value[0]] = Metric(*value)
        else:
            metrics_map_dict[value.name] = value

    return metrics_map_dict


class ScoringMixIn(object):
    def __init__(self, metrics_map, in_fold="all", oof="all", holdout="all", do_score=True):
        """MixIn class to manage metrics to record for each dataset type, and perform evaluations

        Parameters
        ----------
        metrics_map: Dict, List
            Specifies all metrics to be used by their id keys, along with a means to compute the
            metric. If list, all values must be strings that are attributes in
            :mod:`sklearn.metrics`. If dict, key/value pairs must be of the form:
            (<id>, <callable/None/str sklearn.metrics attribute>), where "id" is a str name for the
            metric. Its corresponding value must be one of: 1) a callable to calculate the metric,
            2) None if the "id" key is an attribute in `sklearn.metrics` and should be used to fetch
            a callable, 3) a string that is an attribute in `sklearn.metrics` and should be used to
            fetch a callable. Metric callable functions should expect inputs of form
            (target, prediction), and should return floats
        in_fold: List of strings, None, default=<all ids in `metrics_map`>
            Which metrics (from ids in `metrics_map`) should be recorded for in-fold data
        oof: List of strings, None, default=<all ids in `metrics_map`>
            Which metrics (from ids in `metrics_map`) should be recorded for out-of-fold data
        holdout: List of strings, None, default=<all ids in `metrics_map`>
            Which metrics (from ids in `metrics_map`) should be recorded for holdout data
        do_score: Boolean, default=True
            This is experimental. If False, scores will be neither calculated nor recorded for the
            duration of the experiment

        Notes
        -----
        For each kwarg in [`in_fold`, `oof`, `holdout`], the following must be true: if the value
        of the kwarg is a list, its contents must be a subset of `metrics_map`"""
        self.metrics_map = format_metrics_map(metrics_map)

        #################### ScoringMixIn-Only Mangled Attributes ####################
        self.__in_fold = in_fold if in_fold else []
        self.__oof = oof if oof else []
        self.__holdout = holdout if holdout else []

        self.do_score = do_score

        #################### Validate Parameters ####################
        self._validate_metrics_list_parameters()
        self._set_default_metrics_parameters()

        self.last_evaluation_results = dict(in_fold=None, oof=None, holdout=None)

    def _validate_metrics_list_parameters(self):
        """Ensure metrics lists input parameters are correct types and compatible with each other"""
        for (_d_type, _m_val) in [(_, getattr(self, f"_ScoringMixIn{_}")) for _ in data_types]:
            if _m_val == "all":
                setattr(self, _d_type, list(self.metrics_map.keys()))
            elif not isinstance(_m_val, list):
                raise TypeError(f"{_d_type} must be one of: ['all', None, <list>], not {_m_val}")
            else:
                for _id in _m_val:
                    if not isinstance(_id, str):
                        raise TypeError(f"{_d_type} values must be of type str. Received {_id}")
                    if _id not in self.metrics_map.keys():
                        raise KeyError(f"{_d_type} values must be in metrics_map. '{_id}' is not")

    def _set_default_metrics_parameters(self):
        """Set default parameters if metrics_map is empty (which implies metrics lists are also
        empty)"""
        if len(self.metrics_map.keys()) == 0:
            self.metrics_map = dict(roc_auc=Metric("roc_auc", sk_metrics.roc_auc_score))
            self.in_fold_metrics = ["roc_auc"]

    def evaluate(self, data_type, target, prediction, return_list=False):
        """Apply metric(s) to the given data to calculate the value of the `prediction`

        Parameters
        ----------
        data_type: String in: ['in_fold', 'oof', 'holdout']
            The type of dataset for which `target` and `prediction` arguments are being provided
        target: Array-like
            True labels for the data. Should be same shape as `prediction`
        prediction: Array-like
            Predicted labels for the data. Should be same shape as `target`
        return_list: Boolean, default=False
            If True, return list of tuples instead of dict. See "Returns" section below for details

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

        if data_type not in ("in_fold", "oof", "holdout"):
            raise ValueError(f"data_type must be in ['in_fold', 'oof', 'holdout'], not {data_type}")

        _metric_ids = getattr(self, "__{}".format(data_type))
        _result = []

        for _metric_id in _metric_ids:
            try:
                _metric_value = self.metrics_map[_metric_id](target, prediction)
            except ValueError:
                # Check if target contains integer types, but prediction contains floats
                prediction = get_clean_prediction(target, prediction)
                _metric_value = self.metrics_map[_metric_id](target, prediction)

            _result.append((_metric_id, _metric_value))

        _result = _result if return_list else OrderedDict(_result)
        self.last_evaluation_results[data_type] = _result

        return _result


def get_clean_prediction(target, prediction):
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
    try:
        target_is_int = target.values.dtype == np.int
    except AttributeError:
        target_is_int = target.dtype == np.int
    try:
        prediction_is_int = prediction.dtype == np.int
    except AttributeError:
        prediction_is_int = prediction.values.dtype == np.int

    if (target_is_int is True) and (prediction_is_int is False):
        # ValueError probably: "Classification metrics can't handle a mix of binary and continuous targets"
        target_min, target_max = target.min(), target.max()

        if (len(target_min) == 1) and (len(target_max) == 1):
            target_min, target_max = target_min[0], target_max[0]
        else:
            # TODO: If len(min/max) > 1: multi-class classification, or other multi-output problem
            # TODO: Then each prediction value must be clipped to its specific min/max
            raise ValueError(
                f"Cannot handle multi-output problems. Received bounds of: {target_min}, {target_max}."
            )

        prediction = np.clip(prediction, target_min, target_max)
        prediction = np.rint(prediction)

    return prediction


def get_formatted_target_metric(target_metric, metrics_map, default_dataset="oof"):
    """Return a properly formatted target_metric tuple for use with navigating evaluation results

    Parameters
    ----------
    target_metric: Tuple, String, or None
        Path denoting metric to be used. If tuple, the first value should be in ['oof', 'holdout',
        'in_fold'], and the second value should be the name of a metric supplied in `metrics_map`.
        If str, should be one of the two values from the tuple form. Else, a value will be chosen
    metrics_map: Dict
        Properly formatted `metrics_map` as produced by :func:`metrics.format_metrics_map`, in which
        keys are strings identifying metrics, and values are instances of :class:`metrics.Metric`.
        See the documentation of :func:`metrics.format_metrics_map` for more information on
        different metrics_map formats
    default_dataset: String in ['oof', 'holdout', 'in_fold'], default='oof'
        The default dataset type value to use if one is not provided

    Returns
    -------
    target_metric: Tuple
        A formatted target_metric containing two strings: a dataset_type, followed by a metric name

    Examples
    --------
    >>> get_formatted_target_metric(('holdout', 'roc_auc_score'), format_metrics_map(['roc_auc_score', 'f1_score']))
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric(('holdout',), format_metrics_map(['roc_auc_score', 'f1_score']))
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric('holdout', format_metrics_map(['roc_auc_score', 'f1_score']))
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric('holdout', format_metrics_map({'roc': 'roc_auc_score', 'f1': 'f1_score'}))
    ('holdout', 'roc')
    >>> get_formatted_target_metric('roc_auc_score', format_metrics_map(['roc_auc_score', 'f1_score']))
    ('oof', 'roc_auc_score')
    >>> get_formatted_target_metric(None, format_metrics_map(['f1_score', 'roc_auc_score']))
    ('oof', 'f1_score')
    """
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
            try:
                first_metric_key = list(metrics_map.keys())[0]
            except AttributeError:
                first_metric_key = metrics_map[0].name
            target_metric = target_metric + (first_metric_key,)
            # TODO: Above will cause problems if `Environment.metrics_params['oof']` is not "all"
        else:
            # Just a metric name was provided - Need dataset type
            target_metric = (default_dataset,) + target_metric

    if not any([_ == target_metric[0] for _ in ok_datasets]):
        raise ValueError(f"`target_metric`[0] must be in {ok_datasets}. Received {target_metric}")
    if not target_metric[1] in metrics_map.keys():
        raise ValueError(f"target_metric[1]={target_metric[1]} not in metrics_map={metrics_map}")

    return target_metric


if __name__ == "__main__":
    pass
