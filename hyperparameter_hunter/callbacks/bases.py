"""This module defines the base callback classes, from which all other callback classes in
:mod:`hyperparameter_hunter.callbacks` are descendants. Importantly, the specific base callback
classes contained herein are all descendants of
:class:`hyperparameter_hunter.callbacks.bases.BaseCallback`, ensuring all callbacks descend from the
same base class. This module also defines
:func:`hyperparameter_hunter.callbacks.bases.lambda_callback`, which can be used to define custom
callbacks to be executed during Experiments when passed to
:meth:`hyperparameter_hunter.environment.Environment.__init__` via the `experiment_callbacks`
argument

Related
-------
:mod:`hyperparameter_hunter.callbacks`
    The rest of the submodules within this module should define classes which all descend from the
    base callback classes defined in :mod:`hyperparameter_hunter.callbacks.bases`
:mod:`hyperparameter_hunter.experiment_core`
    This is where callback classes are added as bases inherited by
    :class:`hyperparameter_hunter.experiments.BaseExperiment`. This module is the path that links
    :mod:`hyperparameter_hunter.callbacks` to :mod:`hyperparameter_hunter.experiments`
:mod:`hyperparameter_hunter.environment`
    This module provides the means to use custom callbacks made by
    :func:`hyperparameter_hunter.callbacks.bases.lambda_callback` through the `experiment_callbacks`
    argument of :meth:`hyperparameter_hunter.environment.Environment.__init__`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
from inspect import signature
import numpy as np
from uuid import uuid4 as uuid


class BaseCallback(object):
    """The base class from which all callbacks and all intermediate base callbacks are descendants.
    Callback classes' :meth:`__init__` will not be called, so any tasks that must be performed at
    the onset of an experiment should be placed in :meth:`on_experiment_start`

    Notes
    -----
    __init__(): Some classes that inherit :class:`BaseCallback` may implement :meth:`__init__`;
    however, callback classes are NEVER INITIALIZED. Callback classes should be regarded as
    extensions to the Experiment class that is inheriting them. Because they are inherited, callback
    classes have access to all attributes of the inheriting Experiment class. If any callback class
    does implement :meth:`__init__`, it is simply a convention to clearly declare the attributes
    required by the other methods of that callback class. Again, :meth:`__init__` of classes that
    inherit :class:`BaseCallback` will not be called

    The methods below each call :meth:`settings.G.debug` to signal that the dynamic callback
    inheritance organized in :class:`experiment_core.ExperimentMeta` has proceeded at least
    partially successfully. If all callback methods end by executing the method of the same name in
    their parent classes, then the below debug messages should be visible in the "Heartbeat.log"
    file. Conversely, if any of the below debug messages are not printed to "Heartbeat.log", it is
    likely that a callback class's implementation of the corresponding method does not end with
    "super().<method_name>()". Such cases should be remedied immediately, as the callback stream
    could be skipping any number of other callbacks"""

    # FLAG: Try to implement something like below to ensure other attributes aren't modified (except predictions by Predictors)
    # FLAG: However, since ExperimentMeta makes BaseCallback a superclass of the Experiment classes, they would all pick up...
    # FLAG: ... this method, which would break everything.
    # FLAG: Solution 1: Check name of class that is calling __setattr__. If Callback, apply constraints, else setattr normally
    # FLAG: Solution 2: Try to override __setattr__ method for all classes that require normal behavior; however, ...
    # FLAG: ... uncertain how or where the overridden method might be inherited
    # def __setattr__(self, key, value):
    #     if key == 'stat_aggregates':
    #         self.__dict__[key] = value

    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        print("I should not be printed. Ever.")

    def on_experiment_start(self):
        """Perform tasks when an Experiment is started"""
        G.debug("BaseCallback.on_experiment_start()")

    def on_experiment_end(self):
        """Perform tasks when an Experiment ends"""
        G.debug("BaseCallback.on_experiment_end()")

    def on_repetition_start(self):
        """Perform tasks on repetition start in an Experiment's repeated cross-validation scheme"""
        G.debug("BaseCallback.on_repetition_start()")

    def on_repetition_end(self):
        """Perform tasks on repetition end in an Experiment's repeated cross-validation scheme"""
        G.debug("BaseCallback.on_repetition_end()")

    def on_fold_start(self):
        """Perform tasks on fold start in an Experiment's cross-validation scheme"""
        G.debug("BaseCallback.on_fold_start()")

    def on_fold_end(self):
        """Perform tasks on fold end in an Experiment's cross-validation scheme"""
        G.debug("BaseCallback.on_fold_end()")

    def on_run_start(self):
        """Perform tasks on run start in an Experiment's multiple-run-averaging phase"""
        G.debug("BaseCallback.on_run_start()")

    def on_run_end(self):
        """Perform tasks on run end in an Experiment's multiple-run-averaging phase"""
        G.debug("BaseCallback.on_run_end()")


##################################################
# LambdaCallback
##################################################
def lambda_callback(
    on_experiment_start=None,
    on_experiment_end=None,
    on_repetition_start=None,
    on_repetition_end=None,
    on_fold_start=None,
    on_fold_end=None,
    on_run_start=None,
    on_run_end=None,
    agg_name=None,
):
    """Utility for creating custom callbacks to be declared by :class:`Environment` and used by
    Experiments. The callable "on_<...>_<start/end>" parameters provided will receive as input
    whichever attributes of the Experiment are included in the signature of the given callable. If
    `\*\*kwargs` is given in the callable's signature, a dict of all of the Experiment's attributes
    will be provided. This can be helpful for trying to figure out how to build a custom callback,
    but should not be used unless absolutely necessary. If the Experiment does not have an attribute
    specified in the callable's signature, the following placeholder will be given: "INVALID KWARG"

    Parameters
    ----------
    on_experiment_start: Callable, or None, default=None
        Callable that receives Experiment's values for parameters in the signature at Experiment start
    on_experiment_end: Callable, or None, default=None
        Callable that receives Experiment's values for parameters in the signature at Experiment end
    on_repetition_start: Callable, or None, default=None
        Callable that receives Experiment's values for parameters in the signature at repetition start
    on_repetition_end: Callable, or None, default=None
        Callable that receives Experiment's values for parameters in the signature at repetition end
    on_fold_start: Callable, or None, default=None
        Callable that receives Experiment's values for parameters in the signature at fold start
    on_fold_end: Callable, or None, default=None
        Callable that receives Experiment's values for parameters in the signature at fold end
    on_run_start: Callable, or None, default=None
        Callable that receives Experiment's values for parameters in the signature at run start
    on_run_end: Callable, or None, default=None
        Callable that receives Experiment's values for parameters in the signature at run end
    agg_name: Str, default=uuid.uuid4
        This parameter is only used if the callables are behaving like AggregatorCallbacks by
        returning values (see the "Notes" section below for details on this). If the callables do
        return values, they will be stored under a key named ("_" + `agg_name`) in a dict in
        :attr:`hyperparameter_hunter.experiments.BaseExperiment.stat_aggregates`. The purpose of
        this parameter is to make it easier to understand an Experiment's description file, as
        `agg_name` will default to a UUID if it is not given

    Returns
    -------
    LambdaCallback: :class:`LambdaCallback`
        Uninitialized class, whose methods are the callables of the corresponding "on..." kwarg

    Notes
    -----
    For all of the "on_<...>_<start/end>" callables provided as input to `lambda_callback`, consider
    the following guidelines (for example function "f", which can represent any of the callables):

    - All input parameters in the signature of "f" are attributes of the Experiment being executed

        - If "\*\*kwargs" is a parameter, a dict of all the Experiment's attributes will be provided

    - "f" will be treated as a method of a parent class of the Experiment

        - Take care when modifying attributes, as changes are reflected in the Experiment itself

    - If "f" returns something, it will automatically behave like an AggregatorCallback (see :mod:`hyperparameter_hunter.callbacks.aggregators`). Specifically, the following will occur:

        - A new key (named by `agg_name` if given, else a UUID) with a dict value is added to :attr:`hyperparameter_hunter.experiments.BaseExperiment.stat_aggregates`

            - This new dict can have up to four keys: "runs" (list), "folds" (list), "reps" (list), and "final" (object)

        - If "f" is an "on_run..." function, the returned value is appended to the "runs" list in the new dict
        - Similarly, if "f" is an "on_fold..." or "on_repetition..." function, the returned value is appended to the "folds", or "reps" list, respectively
        - If "f" is an "on_experiment..." function, the "final" key in the new dict is set to the returned value
        - If values were aggregated in the aforementioned manner, the lists of collected values will be reshaped according to runs/folds/reps on Experiment end
        - The aggregated values will be saved in the Experiment's description file

            - This is because :attr:`hyperparameter_hunter.experiments.BaseExperiment.stat_aggregates` is saved in its entirety

    For examples using `lambda_callback` to create custom callbacks, see :mod:`hyperparameter_hunter.callbacks.recipes`

    Examples
    --------
    >>> from hyperparameter_hunter.environment import Environment
    >>> def printer_helper(_rep, _fold, _run, last_evaluation_results):
    ...     print(f"{_rep}.{_fold}.{_run}   {last_evaluation_results}")
    >>> env = Environment(
    ...     train_dataset="i am a dataset",
    ...     root_results_path="path/to/HyperparameterHunterAssets",
    ...     metrics_map=["roc_auc_score"],
    ...     experiment_callbacks=[
    ...         lambda_callback(
    ...             on_experiment_end=printer_helper,
    ...             on_repetition_end=printer_helper,
    ...             on_fold_end=printer_helper,
    ...             on_run_end=printer_helper,
    ...         )
    ...     ]
    ... )
    >>> # ... Now execute an Experiment, or an Optimization Protocol...
    See :mod:`hyperparameter_hunter.examples.lambda_callback_example` for more information"""

    methods = [
        ("on_experiment_start", on_experiment_start, "final"),
        ("on_repetition_start", on_repetition_start, "reps"),
        ("on_fold_start", on_fold_start, "folds"),
        ("on_run_start", on_run_start, "runs"),
        ("on_run_end", on_run_end, "runs"),
        ("on_fold_end", on_fold_end, "folds"),
        ("on_repetition_end", on_repetition_end, "reps"),
        ("on_experiment_end", on_experiment_end, "final"),
    ]

    LambdaCallback = type("LambdaCallback", (BaseCallback,), dict())
    agg_name = "_{}".format(agg_name or str(uuid()))
    does_aggregate = False
    aggregated_shapes = dict(runs=None, folds=None)

    for meth_name, meth_content, agg_key in methods:

        def _method_factory(_meth_name=meth_name, _meth_content=meth_content, _agg_key=agg_key):
            """Provide `_meth_name`, `_meth_content`, and `_agg_key` for :func:`_method`"""

            def _method(self):
                """Execute `_meth_content`, then call parent's method of name `_method_name`"""
                nonlocal does_aggregate

                #################### Execute Custom Callback Method ####################
                try:
                    requested_parameters = {
                        _: getattr(self, _, (self.__dict__ if _ == "kwargs" else "INVALID KWARG"))
                        for _ in dict(signature(_meth_content).parameters)
                    }
                    return_value = _meth_content(**requested_parameters)
                except TypeError:
                    return_value = None

                #################### Handle Return Values ####################
                if return_value is not None:
                    does_aggregate = True
                    self.stat_aggregates.setdefault(agg_name, dict())

                    if _agg_key == "final":
                        self.stat_aggregates[agg_name][_agg_key] = return_value
                    else:
                        self.stat_aggregates[agg_name].setdefault(_agg_key, []).append(return_value)
                        # Record shapes of aggregated return values
                        aggregated_shapes[_agg_key] = np.shape(return_value)

                #################### Reshape Aggregated Values ####################
                if _meth_name == "on_experiment_end" and does_aggregate is True:
                    runs_shape = (self._rep + 1, self._fold + 1, self._run + 1)

                    for (key, shape) in [("runs", runs_shape), ("folds", runs_shape[:-1])]:
                        if key not in self.stat_aggregates[agg_name]:
                            continue
                        self.stat_aggregates[agg_name][key] = np.reshape(
                            self.stat_aggregates[agg_name][key], shape + aggregated_shapes[key]
                        ).tolist()

                #################### Call Next Callback Method in Chain ####################
                getattr(super(LambdaCallback, self), _meth_name)()

            return _method

        setattr(LambdaCallback, meth_name, _method_factory())

    return LambdaCallback


##################################################
# Intermediate Base Callbacks
##################################################
class BasePredictorCallback(BaseCallback):
    """Base class from which all callbacks in :mod:`hyperparameter_hunter.callbacks.predictors` are descendants"""

    pass


class BaseLoggerCallback(BaseCallback):
    """Base class from which all callbacks in :mod:`hyperparameter_hunter.callbacks.loggers` are descendants"""

    pass


class BaseAggregatorCallback(BaseCallback):
    """Base class from which all callbacks in :mod:`hyperparameter_hunter.callbacks.aggregators` are descendants"""

    pass


class BaseEvaluatorCallback(BaseCallback):
    """Base class from which all callbacks in :mod:`hyperparameter_hunter.callbacks.evaluators` are descendants"""

    pass


if __name__ == "__main__":
    pass
