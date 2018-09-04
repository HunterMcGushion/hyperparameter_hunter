"""This module defines :class:`hyperparameter_hunter.metrics.ScoringMixIn` which enables
:class:`hyperparameter_hunter.experiments.BaseExperiment` to score predictions and collect the results of those evaluations

Related
-------
:mod:`hyperparameter_hunter.experiments`
    This module uses :class:`hyperparameter_hunter.metrics.ScoringMixIn` as the only explicit parent class to
    :class:`hyperparameter_hunter.experiments.BaseExperiment` (that is, the only parent class that isn't bestowed upon it by
    :class:`hyperparameter_hunter.experiment_core.ExperimentMeta`)"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.general_utils import type_val

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


class ScoringMixIn(object):
    def __init__(self, metrics_map, in_fold="all", oof="all", holdout="all", do_score=True):
        """MixIn class that manages the metrics to record for each dataset type, and performs evaluations

        Parameters
        ----------
        metrics_map: Dict, List
            Specifies all metrics to be used by their id keys, along with a means to compute the metric. If list, all values must
            be strings that are attributes in :mod:`sklearn.metrics`. If dict, key/value pairs must be of the form:
            (<id>, <callable/None/str sklearn.metrics attribute>), where "id" is a str name for the metric. Its corresponding
            value must be one of: 1) a callable to calculate the metric, 2) None if the "id" key is an attribute in
            `sklearn.metrics` and should be used to fetch a callable, 3) a string that is an attribute in `sklearn.metrics` and
            should be used to fetch a callable. Metric callable functions should expect inputs of form (target, prediction), and
            should return floats
        in_fold: List of strings, None, default=<all ids in `metrics_map`>
            Specifies which metrics (from ids in `metrics_map`) should be recorded for in-fold data
        oof: List of strings, None, default=<all ids in `metrics_map`>
            Specifies which metrics (from ids in `metrics_map`) should be recorded for out-of-fold data
        holdout: List of strings, None, default=<all ids in `metrics_map`>
            Specifies which metrics (from ids in `metrics_map`) should be recorded for holdout data
        do_score: Boolean, default=True
            This is experimental. If False, scores will be neither calculated nor recorded for the duration of the experiment

        Notes
        -----
        For each kwarg in [`in_fold`, `oof`, `holdout`], the following must be true: if the value of the kwarg is a list, its
        contents must be a subset of `metrics_map`"""
        self.metrics_map = metrics_map

        #################### Mangle Below Attributes - Should Only be Used by ScoringMixIn ####################
        self.__in_fold = in_fold if in_fold else []
        self.__oof = oof if oof else []
        self.__holdout = holdout if holdout else []

        self.do_score = do_score

        #################### Validate Parameters ####################
        self._validate_metrics_map()
        self._validate_metrics_list_parameters()
        self._set_default_metrics_parameters()

        self.last_evaluation_results = dict(in_fold=None, oof=None, holdout=None)

    def _validate_metrics_map(self):
        """Ensure `metrics_map` input parameter is properly formatted and yields callable functions for all metrics"""
        if not (isinstance(self.metrics_map, dict) or isinstance(self.metrics_map, list)):
            raise TypeError(
                "metrics_map must be one of: [dict, list]. Received type: {}.".format(
                    type(self.metrics_map)
                )
            )

        #################### If metrics_map is list, convert to dict with None values ####################
        if isinstance(self.metrics_map, list):
            self.metrics_map = {_: None for _ in self.metrics_map}

        for _m_key, _m_val in self.metrics_map.items():
            if not isinstance(_m_key, str):
                raise TypeError(
                    "metrics_map ids must be strings. Received type {}: {}".format(
                        type(_m_key), _m_key
                    )
                )
            if not any([callable(_m_val), isinstance(_m_val, str), _m_val is None]):
                raise TypeError(
                    "metrics_map values must be one of: [callable, str, None]. Received {}".format(
                        type(_m_val)
                    )
                )

            #################### Check sklearn.metrics for: _m_val if str, or _m_key if _m_val is None ####################
            if not callable(_m_val):
                try:
                    self.metrics_map[_m_key] = sk_metrics.__getattribute__(
                        _m_key if _m_val is None else _m_val
                    )
                except AttributeError:
                    raise AttributeError(
                        '"sklearn.metrics" has no attribute "{}".'.format(
                            _m_key if _m_val is None else _m_val
                        )
                    )

    def _validate_metrics_list_parameters(self):
        """Ensure metrics lists input parameters are of correct types and are compatible with each other"""
        for (_d_type, _m_val) in [
            (_, getattr(self, "_ScoringMixIn{}".format(_))) for _ in data_types
        ]:
            if _m_val == "all":
                setattr(self, _d_type, list(self.metrics_map.keys()))
            elif not isinstance(_m_val, list):
                raise TypeError(
                    '{} must be one of: ["all", None, <list>]. Received {}: {}'.format(
                        _d_type, type(_m_val), _m_val
                    )
                )
            else:
                for _id in _m_val:
                    if not isinstance(_id, str):
                        raise TypeError(
                            "{} values must be of type str. Received {}: {}".format(
                                _d_type, type(_id), _id
                            )
                        )
                    if _id not in self.metrics_map.keys():
                        raise KeyError(
                            "{} values must be in metrics_map.keys(). Could not find: {}".format(
                                _d_type, _id
                            )
                        )

    def _set_default_metrics_parameters(self):
        """Set default parameters if metrics_map is empty (which implies metrics lists are also empty)"""
        if len(self.metrics_map.keys()) == 0:
            self.metrics_map = dict(roc_auc=sk_metrics.roc_auc_score)
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
            If True, return type is list of tuples instead of dict. See "Returns" section below for details

        Returns
        -------
        _result: OrderedDict, or list
            A dict whose keys are all metric keys supplied for `data_type`, and whose values are the results of each metric. If
            `return_list` is True, returns a list of tuples of: (<`data_type` metric str>, <metric result>)

        Notes
        -----
        The required types of `target` and `prediction` are entirely dependent on the metric callable's expectations"""
        if self.do_score is False:
            return

        if data_type not in ("in_fold", "oof", "holdout"):
            raise ValueError(
                "data_type must be in: ['in_fold', 'oof', 'holdout']. Received {}: {}".format(
                    *type_val(data_type)
                )
            )

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
        If `target` types are ints, and `prediction` types are not, given predicted labels clipped between the min, and max of
        `target`, then rounded to the nearest integer. Else, original predicted labels"""
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
        A path denoting the metric to be used. If tuple, the first value should be one of ['oof', 'holdout', 'in_fold'], and the
        second value should be the name of a metric supplied in :attr:`environment.Environment.metrics_params`. If str, should be
        one of the two values from the tuple form. Else, a value will be chosen
    metrics_map: Dict, List
        Specifies all metrics to be used by their id keys, along with a means to compute the metric. If list, all values must be
        strings that are attributes in :mod:`sklearn.metrics`. If dict, key/value pairs must be of the form:
        (<id>, <callable/None/str sklearn.metrics attribute>), where "id" is a str name for the metric. Its corresponding value
        must be one of: 1) a callable to calculate the metric, 2) None if the "id" key is an attribute in `sklearn.metrics` and
        should be used to fetch a callable, 3) a string that is an attribute in `sklearn.metrics` and should be used to fetch a
        callable. Metric callable functions should expect inputs of form (target, prediction), and should return floats
    default_dataset: String in ['oof', 'holdout', 'in_fold'], default='oof'
        The default dataset type value to use if one is not provided

    Returns
    -------
    target_metric: Tuple
        A formatted target_metric containing two strings: a dataset_type, followed by a metric name

    Examples
    --------
    >>> get_formatted_target_metric(('holdout', 'roc_auc_score'), ['roc_auc_score', 'f1_score'])
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric(('holdout',), ['roc_auc_score', 'f1_score'])
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric('holdout', ['roc_auc_score', 'f1_score'])
    ('holdout', 'roc_auc_score')
    >>> get_formatted_target_metric('holdout', {'roc': 'roc_auc_score', 'f1': 'f1_score'})
    ('holdout', 'roc')
    >>> get_formatted_target_metric('roc_auc_score', ['roc_auc_score', 'f1_score'])
    ('oof', 'roc_auc_score')
    >>> get_formatted_target_metric(None, ['f1_score', 'roc_auc_score'])
    ('oof', 'f1_score')
    """
    valid_datasets = ["oof", "holdout", "in_fold"]

    if isinstance(target_metric, str):
        target_metric = (target_metric,)
    elif target_metric is None:
        target_metric = (default_dataset,)

    if not isinstance(target_metric, tuple):
        raise TypeError(
            f"Expected `target_metric` to be: tuple, str, or None. Received {type(target_metric)}: {target_metric}"
        )
    elif len(target_metric) > 2:
        raise ValueError(
            f"Expected `target_metric` tuple to be of length 2. Received len={len(target_metric)}: {target_metric}"
        )
    elif len(target_metric) == 1:
        if target_metric[0] in valid_datasets:
            # Just a dataset was provided - Need metric name
            try:
                first_metric_key = list(metrics_map.keys())[0]
            except AttributeError:
                first_metric_key = metrics_map[0]
            target_metric = target_metric + (first_metric_key,)
            # TODO: Above will cause problems if `Environment.metrics_params['oof']` is not "all"
        else:
            # Just a metric name was provided - Need dataset type
            target_metric = (default_dataset,) + target_metric

    if not any([_ == target_metric[0] for _ in valid_datasets]):
        raise ValueError(
            f"The first item of `target_metric` must be in {valid_datasets}. Received: {target_metric}"
        )
    if not target_metric[1] in metrics_map:
        raise ValueError(
            f"The second value of `target_metric` ({target_metric[1]}) must be in `metrics_map`: {metrics_map}"
        )

    return target_metric


if __name__ == "__main__":
    pass
