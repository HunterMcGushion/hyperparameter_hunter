"""This module is the core of all of the experimentation in `hyperparameter_hunter`, hence its name.
It is impossible to understand :mod:`hyperparameter_hunter.experiments` without first having a grasp
on what :class:`hyperparameter_hunter.experiment_core.ExperimentMeta` is doing. This module serves
to bridge the gap between Experiments, and :mod:`hyperparameter_hunter.callbacks` by dynamically
making Experiments inherit various callbacks depending on the inputs given in order to make
Experiments completely functional

Related
-------
:mod:`hyperparameter_hunter.experiments`
    Defines the structure of the experimentation process. While certainly very important,
    :mod:`hyperparameter_hunter.experiments` wouldn't do much at all without
    :mod:`hyperparameter_hunter.callbacks`, or :mod:`hyperparameter_hunter.experiment_core`
:mod:`hyperparameter_hunter.callbacks`
    Defines parent classes to the classes defined in :mod:`hyperparameter_hunter.experiments`. This
    not only makes it very easy to find the entire workflow for a given task, but also ensures that
    each instance of an Experiment inherits exactly the functionality that it needs. For example,
    if no holdout data was given, then :class:`experiment_core.ExperimentMeta` will not add
    :class:`callbacks.evaluators.EvaluatorHoldout` or :class:`callbacks.predictors.PredictorHoldout`
    to the list of callbacks inherited by the Experiment. This means that the Experiment never needs
    to check for the existence of holdout data in order to determine how it should proceed because
    it literally doesn't have the code that deals with holdout data

Notes
-----
Was a metaclass really necessary here? Probably not, but it's being used for two reasons:
1) metaclasses are fun, and programming (especially artificial intelligence) should be fun; and
2) it allowed for a very clean separation between the various functions demanded by Experiments that
are provided by :mod:`hyperparameter_hunter.callbacks`. Having each of the callbacks separated in
their own classes makes it very easy to debug existing functionality, and to add new callbacks in
the future"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.aggregators import AggregatorEvaluations, AggregatorTimes
from hyperparameter_hunter.callbacks.bases import (
    BaseCallback,
    BasePredictorCallback,
    BaseEvaluatorCallback,
)
from hyperparameter_hunter.callbacks.bases import BaseAggregatorCallback, BaseLoggerCallback
from hyperparameter_hunter.callbacks.evaluators import EvaluatorOOF, EvaluatorHoldout
from hyperparameter_hunter.callbacks.loggers import LoggerFitStatus
from hyperparameter_hunter.callbacks.predictors import PredictorOOF, PredictorHoldout, PredictorTest
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
import inspect
import os


class ExperimentMeta(type):
    """Metaclass that determines which callbacks should be inherited by an Experiment in order to
    complete its functionality"""

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        """Prepare the namespace for the Experiment by separating its parent classes according to
        those that were originally provided, those that are provided on a class-wide basis, and
        those that are provided on an instance-wide basis. This is done in order to preserve the
        intended MRO of the original base classes, after adding and sorting new bases"""
        namespace = dict(
            __original_bases=bases, __class_wide_bases=[], __instance_bases=[], source_script=None
        )

        return namespace

    def __new__(mcs, name, bases, namespace, **kwargs):
        """Create a new class object that stores necessary class-wide callbacks to
        :attr:`__class_wide_bases`"""
        class_obj = super().__new__(mcs, name, bases, namespace)

        # Add cross-validation-related bases to inheritance tree
        if name != "NoValidationExperiment":
            namespace["__class_wide_bases"].append(PredictorOOF)
            namespace["__class_wide_bases"].append(EvaluatorOOF)

        # Add Class-Wide Aggregator Bases
        namespace["__class_wide_bases"].append(AggregatorEvaluations)
        namespace["__class_wide_bases"].append(AggregatorTimes)

        # Add Class-Wide Logger Bases
        namespace["__class_wide_bases"].append(LoggerFitStatus)

        return class_obj

    def __call__(cls, *args, **kwargs):
        """Store necessary instance-wide callbacks to :attr:`__instance_bases`, sort all dynamically
        added callback base classes, then add them to the instance"""
        original_bases = getattr(cls, "__original_bases")
        class_wide_bases = getattr(cls, "__class_wide_bases")
        instance_bases = []

        # Get source_script for use by Experiment later
        setattr(
            cls,
            "source_script",
            os.path.abspath(inspect.getframeinfo(inspect.currentframe().f_back)[0]),
        )

        # Add callbacks explicitly supplied on class initialization
        if kwargs.get("callbacks", None) is not None:
            for callback in kwargs["callbacks"]:
                instance_bases.append(callback)

        # Infer necessary callbacks based on class initialization inputs
        if G.Env.holdout_dataset is not None:
            instance_bases.append(PredictorHoldout)
            instance_bases.append(EvaluatorHoldout)

        if G.Env.test_dataset is not None:
            instance_bases.append(PredictorTest)

        # Add callbacks explicitly provided to the Environment
        if len(G.Env.experiment_callbacks) > 0:
            instance_bases.extend(G.Env.experiment_callbacks)

        setattr(cls, "__instance_bases", instance_bases)

        # Sort dynamically added auxiliary base classes
        auxiliary_bases = tuple(base_callback_class_sorter((class_wide_bases + instance_bases)))

        # TODO: If "G.Env.save_full_predictions is True", add callbacks to record full_predictions for the 3 dataset types
        # FLAG: Ensure callbacks to record full_predictions are executed after normal "Predictor..." callbacks
        # FLAG: Add ability to record full_predictions, then provide callback to check on experiment end...
        # FLAG: ... to determine whether full_predictions should actually be saved - Like checking final score/std > threshold

        cls.__bases__ = original_bases + auxiliary_bases

        return super().__call__(*args, **kwargs)


def base_callback_class_sorter(auxiliary_bases, parent_class_order=None):
    """Sort callback classes in order to preserve the intended MRO of their descendant, and to
    enable callbacks that may depend on one another to function properly

    Parameters
    ----------
    auxiliary_bases: List
        The callback classes to be sorted according to the order in which their parent is found in
        `parent_class_order`. For example, if a class (x) in `auxiliary_bases` is the only
        descendant of the last class in `parent_class_order`, then class x will be moved to the last
        position in `sorted_auxiliary_bases`. If multiple classes in `auxiliary_bases` are
        descendants of the same parent in `parent_class_order`, they will be sorted alphabetically
        (from A-Z)
    parent_class_order: List, or None, default=<See description>
        List of base callback classes that define the sort order for `auxiliary_bases`. Note that
        these are not the normal callback classes that add to the functionality of an Experiment,
        but the base classes from which the callback classes are descendants. All the classes in
        `parent_class_order` should be defined in :mod:`hyperparameter_hunter.callbacks.bases`. The
        last class in `parent_class_order` should be
        :class:`hyperparameter_hunter.callbacks.bases.BaseCallback`, which is the parent class for
        all other base classes. This ensures that custom callbacks defined by
        :func:`hyperparameter_hunter.callbacks.bases.lambda_callback` will be recognized as valid
        and executed last

    Returns
    -------
    sorted_auxiliary_bases: List
        The contents of `auxiliary_bases` sorted according to their parents' location in
        `parent_class_order`, then alphabetically

    Raises
    ------
    ValueError
        If `auxiliary_bases` contains a class that is not a descendant of any of the classes in
        `parent_class_order`"""
    if parent_class_order is None:
        parent_class_order = [
            BasePredictorCallback,
            BaseEvaluatorCallback,
            BaseAggregatorCallback,
            BaseLoggerCallback,
            BaseCallback,
        ]

    sorted_auxiliary_bases = []

    for parent_class in parent_class_order:
        callback_holder = [_ for _ in auxiliary_bases if issubclass(_, parent_class)]
        callback_holder = sorted(callback_holder, key=lambda _: _.__name__, reverse=False)

        auxiliary_bases = [_ for _ in auxiliary_bases if not issubclass(_, parent_class)]
        sorted_auxiliary_bases.extend(callback_holder)

    if len(auxiliary_bases) > 0:
        raise ValueError(
            f"Received base class that does not inherit any of the given parent classes: {auxiliary_bases}"
        )

    return sorted_auxiliary_bases


if __name__ == "__main__":
    pass
