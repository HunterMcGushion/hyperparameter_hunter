from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter.callbacks.bases import lambda_callback
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from hyperparameter_hunter.callbacks.recipes import confusion_matrix_oof
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier


def printer_callback():
    """This is a simple callback example that will print out :attr:`CVExperiment.last_evaluation_results` at all
    available time intervals, along with the repetition, fold, and run number. Of course, printing evaluations at the beginning
    of each of the intervals, as is shown below, is pretty much useless. However, this shows that if you want to, you can do it
    anyways and create your own replacement for the default logger... Or make anything else you might want"""

    def printer_helper(_rep, _fold, _run, last_evaluation_results):
        print(f"{_rep}.{_fold}.{_run}   {last_evaluation_results}")

    return lambda_callback(
        on_experiment_start=printer_helper,
        on_experiment_end=printer_helper,
        on_repetition_start=printer_helper,
        on_repetition_end=printer_helper,
        on_fold_start=printer_helper,
        on_fold_end=printer_helper,
        on_run_start=printer_helper,
        on_run_end=printer_helper,
    )


def execute():
    env = Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path="HyperparameterHunterAssets",
        metrics_map=["roc_auc_score"],
        cross_validation_type=RepeatedStratifiedKFold,
        cross_validation_params=dict(n_splits=5, n_repeats=2, random_state=32),
        runs=2,
        # Just instantiate `Environment` with your list of callbacks, and go about business as usual
        experiment_callbacks=[printer_callback(), confusion_matrix_oof()],
        # In addition to `printer_callback` made above, we're also adding the `confusion_matrix_oof` callback
        # This, and other callbacks, can be found in `hyperparameter_hunter.callbacks.recipes`
    )

    experiment = CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params={},
        model_extra_params=dict(fit=dict(verbose=False)),
    )


if __name__ == "__main__":
    execute()
