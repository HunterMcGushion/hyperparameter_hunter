##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import lambda_callback

##################################################
# Import Miscellaneous Assets
##################################################
from copy import deepcopy
import pytest


def printer_helper(_rep, _fold, _run, last_evaluation_results):
    print(f"{_rep}.{_fold}.{_run}   {last_evaluation_results}")


good_lambda_callback_kwargs = dict(
    on_exp_start=printer_helper,
    on_rep_start=printer_helper,
    on_fold_start=printer_helper,
    on_run_start=printer_helper,
    on_run_end=printer_helper,
    on_fold_end=printer_helper,
    on_rep_end=printer_helper,
    on_exp_end=printer_helper,
)


@pytest.mark.parametrize(
    ["add_kwarg", "drop_kwarg"],
    [
        ("on_experiment_start", "on_exp_start"),
        ("on_experiment_end", "on_exp_end"),
        ("on_repetition_start", "on_rep_start"),
        ("on_repetition_end", "on_rep_end"),
    ],
)
def test_lambda_callback_deprecations(add_kwarg, drop_kwarg):
    """This function tests the deprecations of the following `lambda_callback` kwargs in 3.0.0:
    * `on_experiment_start` -> `on_exp_start`
    * `on_experiment_end` -> `on_exp_end`
    * `on_repetition_start` -> `on_rep_start`
    * `on_repetition_end` -> `on_rep_end`"""
    bad_lambda_callback_kwargs = deepcopy(good_lambda_callback_kwargs)
    del bad_lambda_callback_kwargs[drop_kwarg]
    bad_lambda_callback_kwargs[add_kwarg] = printer_helper

    with pytest.deprecated_call():
        lambda_callback(**bad_lambda_callback_kwargs)
