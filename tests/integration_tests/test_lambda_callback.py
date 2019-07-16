##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, lambda_callback
from hyperparameter_hunter.callbacks.bases import BaseCallback
from hyperparameter_hunter.callbacks.recipes import confusion_matrix_oof
from hyperparameter_hunter.utils.learning_utils import get_diabetes_data

##################################################
# Import Miscellaneous Assets
##################################################
from copy import deepcopy
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.ensemble import AdaBoostRegressor

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


def env_lambda_cb(lambda_cbs):
    """Return an `Environment` using `lambda_cbs` as `experiment_callbacks`

    Parameters
    ----------
    lambda_cbs: `LambdaCallback`, list of `LambdaCallback`, or None
        LambdaCallback values passed to the `Environment`'s `experiment_callbacks` kwarg"""
    return Environment(
        train_dataset=get_diabetes_data(target="target"),
        results_path=assets_dir,
        metrics=["median_absolute_error"],
        cv_type="KFold",
        cv_params=dict(n_splits=3, random_state=1),
        experiment_callbacks=lambda_cbs,
    )


def exp_lambda_cb(lambda_cbs):
    """Return a `CVExperiment` with `lambda_cbs` as `callbacks`

    Parameters
    ----------
    lambda_cbs: `LambdaCallback`, list of `LambdaCallback`, or None
        LambdaCallback values passed to the `CVExperiment`'s `callbacks` kwarg"""
    return CVExperiment(AdaBoostRegressor, callbacks=lambda_cbs)


##################################################
# Dummy LambdaCallbacks
##################################################
def dummy_lambda_cb_func():
    def _on_fold_start(_rep, _fold, _run):
        print(_rep, _fold, _run)

    return lambda_callback(on_fold_start=_on_fold_start)


class DummyLambdaCallbackClass(BaseCallback):
    def on_run_start(self):
        print("on_run_start", self._rep, self._fold, self._run)

    def on_run_end(self):
        print("on_run_end", self._rep, self._fold, self._run)


##################################################
# Test `lambda_callback` Provision
##################################################
# noinspection PyUnusedLocal
@pytest.mark.parametrize(
    "lambda_cbs",
    [
        dummy_lambda_cb_func(),
        [dummy_lambda_cb_func()],
        [dummy_lambda_cb_func(), confusion_matrix_oof()],
        [dummy_lambda_cb_func(), DummyLambdaCallbackClass, confusion_matrix_oof()],
    ],
)
def test_provide_lambda_callbacks(lambda_cbs):
    """Test that each of the officially-supported methods of providing LambdaCallbacks to an
    Experiment yields the same MRO. Specifically concerned with using the `experiment_callbacks`
    kwarg of :class:`~hyperparameter_hunter.environment.Environment` and using the `callbacks`
    kwarg of :class:`~hyperparameter_hunter.experiments.CVExperiment`. Also sanity check that MROs
    of Experiments with LambdaCallbacks actually differ from the MRO of a basic Experiment

    Parameters
    ----------
    lambda_cbs: `LambdaCallback`, or list of `LambdaCallback`
        LambdaCallback values passed to the different methods of `lambda_callback` provision"""
    #################### Via `Environment`'s `experiment_callbacks` ####################
    env_0 = env_lambda_cb(lambda_cbs)
    exp_0 = exp_lambda_cb(None)
    exp_0_mro = deepcopy(type(exp_0).__mro__)
    # Need to save copy of MRO because it is relative to the CLASS, not the instance, and
    #   `ExperimentMeta` changes the MRO of the `CVExperiment` class

    #################### Via `CVExperiment`'s `callbacks` ####################
    env_1 = env_lambda_cb(None)
    exp_1 = exp_lambda_cb(lambda_cbs)
    exp_1_mro = deepcopy(type(exp_1).__mro__)

    assert exp_0_mro == exp_1_mro
    # Can't compare `type(exp_0).__mro__` == `type(exp_1).__mro__` because they will always be
    #   identical, since (as noted above) `ExperimentMeta`'s changes affect prior `CVExperiment`s

    #################### Baseline Without LambdaCallbacks ####################
    # Test that both of the above MROs actually differ from the MRO of a basic Experiment
    env_2 = env_lambda_cb(None)
    exp_2 = exp_lambda_cb(None)
    exp_2_mro = deepcopy(type(exp_2).__mro__)

    # NOTE: `assert type(exp_2).__mro__ != type(exp_1).__mro__` would FAIL here for the reasons
    #   noted in the comments above. This is why the MRO had to be copied each time
    assert exp_2_mro != exp_0_mro
    assert exp_2_mro != exp_1_mro

    # Baseline MRO should be missing the LambdaCallbacks added to the earlier Experiments
    if isinstance(lambda_cbs, list):
        assert len(exp_2_mro) == (len(exp_0_mro) - len(lambda_cbs))
        assert len(exp_2_mro) == (len(exp_1_mro) - len(lambda_cbs))
    else:
        assert len(exp_2_mro) == (len(exp_0_mro) - 1)
        assert len(exp_2_mro) == (len(exp_1_mro) - 1)
