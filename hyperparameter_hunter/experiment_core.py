##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.aggregators import AggregatorEvaluations, AggregatorEpochsElapsed, AggregatorTimes
from hyperparameter_hunter.callbacks.bases import BaseCallback, BasePredictorCallback, BaseEvaluatorCallback
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
    # TODO: Add documentation

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        # Separate to preserve MRO of original base classes, after adding and sorting new bases
        namespace = dict(
            __original_bases=bases,
            __class_wide_bases=[],
            __instance_bases=[],
            source_script=None,
        )

        return namespace

    def __new__(mcs, name, bases, namespace, **kwargs):
        class_obj = super().__new__(mcs, name, bases, namespace)

        # Add cross-validation-related bases to inheritance tree
        if name != 'NoValidationExperiment':
            namespace['__class_wide_bases'].append(PredictorOOF)
            namespace['__class_wide_bases'].append(EvaluatorOOF)

        # Add Class-Wide Aggregator Bases
        namespace['__class_wide_bases'].append(AggregatorEvaluations)
        namespace['__class_wide_bases'].append(AggregatorEpochsElapsed)
        namespace['__class_wide_bases'].append(AggregatorTimes)

        # Add Class-Wide Logger Bases
        namespace['__class_wide_bases'].append(LoggerFitStatus)

        return class_obj

    def __call__(cls, *args, **kwargs):
        original_bases = getattr(cls, '__original_bases')
        class_wide_bases = getattr(cls, '__class_wide_bases')
        instance_bases = []

        # Get source_script for use by Experiment later
        setattr(cls, 'source_script', os.path.abspath(inspect.getframeinfo(inspect.currentframe().f_back)[0]))

        # Add callbacks explicitly supplied on class initialization
        if kwargs.get('callbacks', None) is not None:
            for callback in kwargs['callbacks']:
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

        setattr(cls, '__instance_bases', instance_bases)

        # Sort dynamically added auxiliary base classes
        auxiliary_bases = tuple(base_callback_class_sorter((class_wide_bases + instance_bases)))

        # TODO: If "G.Env.save_full_predictions is True", add callbacks to record full_predictions for the 3 dataset types
        # FLAG: Ensure callbacks to record full_predictions are executed after normal "Predictor..." callbacks
        # FLAG: Add ability to record full_predictions, then provide callback to check on experiment end...
        # FLAG: ... to determine whether full_predictions should actually be saved - Like checking final score/std > threshold

        cls.__bases__ = original_bases + auxiliary_bases

        return super().__call__(*args, **kwargs)


def base_callback_class_sorter(auxiliary_bases, parent_class_order=None):
    if parent_class_order is None:
        parent_class_order = [
            BasePredictorCallback, BaseEvaluatorCallback, BaseAggregatorCallback, BaseLoggerCallback, BaseCallback
        ]

    sorted_auxiliary_bases = []

    for parent_class in parent_class_order:
        callback_holder = [_ for _ in auxiliary_bases if issubclass(_, parent_class)]
        callback_holder = sorted(callback_holder, key=lambda _: _.__name__, reverse=False)

        auxiliary_bases = [_ for _ in auxiliary_bases if not issubclass(_, parent_class)]
        sorted_auxiliary_bases.extend(callback_holder)

    if len(auxiliary_bases) > 0:
        raise ValueError(F'Received base class that does not inherit any of the given parent classes: {auxiliary_bases}')

    return sorted_auxiliary_bases


if __name__ == '__main__':
    pass
