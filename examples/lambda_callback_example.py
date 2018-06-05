import sys
import os.path

try:
    sys.path.append(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])
except Exception as _ex:
    raise _ex

from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter.callbacks.bases import lambda_callback
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier


def printer_callback():
    """This is a simple callback example that will print out :attr:`CrossValidationExperiment.last_evaluation_results` at all
    available time intervals, along with the repetition, fold, and run number. Of course, printing evaluations at the beginning
    of each of the intervals, as is shown below, is pretty much useless. However, this shows that if you want to, you can do it
    anyways and create your own replacement for the default logger... Or make anything else you might want"""
    return lambda_callback(
        # The contents of `required_attributes` are the Experiment attributes that we want sent to our callables at each interval
        required_attributes=['_rep', '_fold', '_run', 'last_evaluation_results'],
        # Notice that below, each callable expects the four attributes we listed above in `required_attributes`
        on_experiment_start=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
        on_experiment_end=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
        on_repetition_start=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
        on_repetition_end=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
        on_fold_start=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
        on_fold_end=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
        on_run_start=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
        on_run_end=lambda _rep, _fold, _run, evals: print(F'{_rep}.{_fold}.{_run}   {evals}'),
    )


def execute():
    env = Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path='HyperparameterHunterAssets',
        metrics_map=['roc_auc_score'],
        cross_validation_type=RepeatedStratifiedKFold,
        cross_validation_params=dict(n_splits=5, n_repeats=2, random_state=32),
        runs=2,
        # Just instantiate `Environment` with your list of callbacks, and go about business as usual
        experiment_callbacks=[printer_callback()],
    )

    experiment = CrossValidationExperiment(
        model_initializer=XGBClassifier,
        model_init_params={},
        model_extra_params=dict(fit=dict(verbose=False)),
    )

    # TODO: Add confusion matrix example


if __name__ == '__main__':
    execute()
