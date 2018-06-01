##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G


class BaseCallback(object):
    """Callback classes' :meth:`__init__` will not be called, so any tasks that must be performed at the onset of an experiment
    should be placed in :meth:`on_experiment_start`

    Notes
    -----
    __init__(): Some classes that inherit :class:`BaseCallback` may implement :meth:`__init__`; however, callback classes are
    NEVER INITIALIZED. Callback classes should be regarded as extensions to the Experiment class that is inheriting them. Because
    they are inherited, callback classes have access to all attributes of the inheriting Experiment class. If any callback class
    does implement :meth:`__init__`, it is simply a convention to clearly declare the attributes required by the other methods of
    that callback class. Again, :meth:`__init__` of classes that inherit :class:`BaseCallback` will not be called

    The methods below each call :meth:`settings.G.debug` to signal that the dynamic callback inheritance organized in
    :meta:`experiment_core.ExperimentMeta` has proceeded at least partially successfully. If all callback methods end by executing
    the method of the same name in their parent classes, then the below debug messages should be visible in the "Heartbeat.log"
    file. Conversely, if any of the below debug messages are not printed to "Heartbeat.log", it is likely that a callback class's
    implementation of the corresponding method does not end with "super().<method_name>()". Such cases should be remedied
    immediately, as the callback stream could be skipping any number of other callbacks
    """

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
        """Uncalled - See the 'Notes' section in the documentation of :class:`callbacks.bases.BaseCallback` for details"""
        print('I should not be printed. Ever.')

    def on_experiment_start(self):
        G.debug('BaseCallback.on_experiment_start()')

    def on_experiment_end(self):
        G.debug('BaseCallback.on_experiment_end()')

    def on_repetition_start(self):
        G.debug('BaseCallback.on_repetition_start()')

    def on_repetition_end(self):
        G.debug('BaseCallback.on_repetition_end()')

    def on_fold_start(self):
        G.debug('BaseCallback.on_fold_start()')

    def on_fold_end(self):
        G.debug('BaseCallback.on_fold_end()')

    def on_run_start(self):
        G.debug('BaseCallback.on_run_start()')

    def on_run_end(self):
        G.debug('BaseCallback.on_run_end()')


def lambda_callback(
        required_attributes,
        on_experiment_start=None, on_experiment_end=None,
        on_repetition_start=None, on_repetition_end=None,
        on_fold_start=None, on_fold_end=None,
        on_run_start=None, on_run_end=None,
):
    """Utility for creating custom callbacks to be declared by :class:`Environment`, and used by Experiments

    Parameters
    ----------
    required_attributes: List of strings
        The names of the Experiment attributes that will be passed to each of the 'on...' callable kwargs. Can contain any
        attributes of the Experiment class that is to be executed
    on_experiment_start: Callable, or None, default=None
        Callable that will receive the Experiment's values of `required_attributes` at the start of the Experiment
    on_experiment_end: Callable, or None, default=None
        Callable that will receive the Experiment's values of `required_attributes` at the end of the Experiment
    on_repetition_start: Callable, or None, default=None
        Callable that will receive the Experiment's values of `required_attributes` at the start of each repetition
    on_repetition_end: Callable, or None, default=None
        Callable that will receive the Experiment's values of `required_attributes` at the end of each repetition
    on_fold_start: Callable, or None, default=None
        Callable that will receive the Experiment's values of `required_attributes` at the start of each fold
    on_fold_end: Callable, or None, default=None
        Callable that will receive the Experiment's values of `required_attributes` at the end of each fold
    on_run_start: Callable, or None, default=None
        Callable that will receive the Experiment's values of `required_attributes` at the start of each run
    on_run_end: Callable, or None, default=None
        Callable that will receive the Experiment's values of `required_attributes` at the end of each run

    Returns
    -------
    LambdaCallback: :class:`LambdaCallback`
        A new, uninitialized class, whose methods contain the callable content of the corresponding "on..." kwarg

    Examples
    --------
    >>> from hyperparameter_hunter.environment import Environment
    >>> env = Environment(
    ...     train_dataset='i am a dataset', root_results_path='path/to/HyperparameterHunterAssets', metrics_map=['roc_auc_score'],
    ...     experiment_callbacks=[lambda_callback(
    ...         required_attributes=['_rep', '_fold', '_run', 'last_evaluation_results'],
    ...         on_experiment_end=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
    ...         on_repetition_end=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
    ...         on_fold_end=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
    ...         on_run_end=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
    ...     )]
    ... )
    >>> # ... Now execute an Experiment, or an Optimization Protocol...
    See :mod:`hyperparameter_hunter.examples.lambda_callback_example` for more"""

    methods = [
        ('on_experiment_start', on_experiment_start), ('on_experiment_end', on_experiment_end),
        ('on_repetition_start', on_repetition_start), ('on_repetition_end', on_repetition_end),
        ('on_fold_start', on_fold_start), ('on_fold_end', on_fold_end),
        ('on_run_start', on_run_start), ('on_run_end', on_run_end),
    ]

    LambdaCallback = type('LambdaCallback', (BaseCallback,), dict())

    for method_name, method_content in methods:
        if callable(method_content):
            def _method_factory(_method_name=method_name, _method_content=method_content):
                def _method(self):
                    _method_content(*[getattr(self, _) for _ in required_attributes])
                    getattr(super(LambdaCallback, self), _method_name)()

                return _method

            setattr(LambdaCallback, method_name, _method_factory())

    return LambdaCallback


class BasePredictorCallback(BaseCallback):
    pass


class BaseLoggerCallback(BaseCallback):
    pass


class BaseAggregatorCallback(BaseCallback):
    pass


class BaseEvaluatorCallback(BaseCallback):
    pass


def execute():
    pass


if __name__ == '__main__':
    execute()
