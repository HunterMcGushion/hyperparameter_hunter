##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, Real, Integer, Categorical, GBRT
from hyperparameter_hunter import lambda_callback
from hyperparameter_hunter.sentinels import DatasetSentinel
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data

##################################################
# Import Miscellaneous Assets
##################################################
import pytest
import sys

##################################################
# Import Learning Assets
##################################################
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except ImportError:
    pass

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


##################################################
# Helpers
##################################################
def expected_sentinels(cv_scheme):
    """Determine expected values of :class`environment.Environment`'s `DatasetSentinel`s given
    train/validation splits created by `cv_scheme`

    Parameters
    ----------
    cv_scheme: Descendant instance of `sklearn.model_selection._split._BaseKFold`
        Cross-validation class instance to produce train/validation data splits via :meth:`split`

    Returns
    -------
    train_sentinels: List
        Tuples of (train_input, train_target) produced by `cv_scheme.split`
    validation_sentinels: List
        Tuples of (validation_input, validation_target) produced by `cv_scheme.split`
    holdout_sentinels: List
        Tuples of (holdout_input, holdout_target) repeated for each period created by `cv_scheme`"""
    train_sentinels, validation_sentinels, holdout_sentinels = [], [], []

    data = get_breast_cancer_data(target="target")
    target_df = data[["target"]]
    input_df = data.drop(["target"], axis=1)

    # TODO: Need to account for feature engineering here - Probably need to have hardcoded test data rather than calculating splits here
    for train_i, validation_i in cv_scheme.split(input_df, target_df):
        train_sentinels.append((input_df.iloc[train_i, :], target_df.iloc[train_i, :]))
        validation_sentinels.append(
            (input_df.iloc[validation_i, :], target_df.iloc[validation_i, :])
        )
        holdout_sentinels.append((input_df, target_df))
        # TODO: Need to account for feature engineering here - Probably need to have hardcoded test data rather than calculating splits here

    return train_sentinels, validation_sentinels, holdout_sentinels


def sentinel_checker():
    """Build :func:`callbacks.bases.lambda_callback` to compare the current `CVExperiment` dataset
    values with the expected values of the dataset (train, validation, and holdout) sentinels

    Returns
    -------
    LambdaCallback
        Result of :func:`callbacks.bases.lambda_callback` to check `DatasetSentinel` values"""

    def on_run_start(
        fold_train_input,
        fold_train_target,
        fold_validation_input,
        fold_validation_target,
        fold_holdout_input,
        fold_holdout_target,
    ):
        #################### Check Train Sentinels ####################
        assert fold_train_input.equals(G.Env.train_input.retrieve_by_sentinel())
        assert fold_train_target.equals(G.Env.train_target.retrieve_by_sentinel())

        #################### Check Validation Sentinels ####################
        assert fold_validation_input.equals(G.Env.validation_input.retrieve_by_sentinel())
        assert fold_validation_target.equals(G.Env.validation_target.retrieve_by_sentinel())

        #################### Check Holdout Sentinels ####################
        assert fold_holdout_input.equals(G.Env.holdout_input.retrieve_by_sentinel())
        assert fold_holdout_target.equals(G.Env.holdout_target.retrieve_by_sentinel())

    return lambda_callback(on_run_start=on_run_start)


##################################################
# Environment Fixtures
##################################################
@pytest.fixture(scope="function", autouse=False)
def env_0():
    """`Environment` fixture that has `holdout_dataset` identical to `train_dataset` and is given
    `experiment_callbacks` consisting of the `lambda_callback` result of :func:`sentinel_checker`"""
    return Environment(
        train_dataset=get_breast_cancer_data(target="target"),
        results_path=assets_dir,
        holdout_dataset=get_breast_cancer_data(target="target"),
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=2, shuffle=True, random_state=32),
        experiment_callbacks=[sentinel_checker()],
    )


##################################################
# Sentinel Workflow Scenarios
##################################################
def get_all_sentinels(env):
    """Get list of all dataset sentinel values in format expected by `XGBClassifier.fit.eval_set`"""
    return [
        (env.train_input, env.train_target),
        (env.validation_input, env.validation_target),
        (env.holdout_input, env.holdout_target),
    ]


@pytest.mark.skipif("xgboost" not in sys.modules, reason="Requires `XGBoost` library")
def test_sentinels_experiment(env_0):
    # noinspection PyUnusedLocal
    experiment = CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(objective="reg:linear", max_depth=3, subsample=0.5),
        model_extra_params=dict(
            fit=dict(eval_set=get_all_sentinels(env_0), early_stopping_rounds=5, eval_metric="mae")
        ),
    )


@pytest.mark.skipif("xgboost" not in sys.modules, reason="Requires `XGBoost` library")
def test_sentinels_optimization(env_0):
    optimizer = GBRT(iterations=2)
    optimizer.set_experiment_guidelines(
        model_initializer=XGBClassifier,
        model_init_params=dict(objective="reg:linear", max_depth=Integer(2, 20), subsample=0.5),
        model_extra_params=dict(
            fit=dict(
                eval_set=get_all_sentinels(env_0),
                early_stopping_rounds=5,
                eval_metric=Categorical(["auc", "mae"]),
            )
        ),
    )
    optimizer.go()


##################################################
# General Sentinel Scenarios
##################################################
@pytest.mark.parametrize(
    ["sentinel_parameters", "error_match"],
    [
        [["foo", "bar"], "Received invalid `dataset_type`: 'foo'"],
        [["train_input", "bar"], "`cv_type`.*"],
        [["train_input", "bar", "CV"], "`global_random_seed`.*"],
    ],
)
def test_dataset_sentinel_validate_parameters(sentinel_parameters, error_match):
    """Ensure appropriate ValueErrors raised by `sentinels.DatasetSentinel._validate_parameters`"""
    with pytest.raises(ValueError, match=error_match):
        DatasetSentinel(*sentinel_parameters)
