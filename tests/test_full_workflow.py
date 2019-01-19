##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, Real, Integer, Categorical, DummySearch
from hyperparameter_hunter import BayesianOptimization, GradientBoostedRegressionTreeOptimization
from hyperparameter_hunter import ExtraTreesOptimization, RandomForestOptimization, lambda_callback
from hyperparameter_hunter.callbacks.recipes import confusion_matrix_oof, confusion_matrix_holdout
from hyperparameter_hunter.recorders import YAMLDescriptionRecorder, UnsortedIDLeaderboardRecorder

# FLAG: Testing will require `yaml` for above
from hyperparameter_hunter.utils.learning_utils import (
    get_toy_classification_data,
    get_breast_cancer_data,
)
from hyperparameter_hunter.utils.test_utils import has_experiment_result_file

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd
from pathlib import Path
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


##################################################
# Environment Fixtures
##################################################
@pytest.fixture(scope="function", autouse=False)
def env_0():
    def do_full_save(experiment_result):
        return experiment_result["final_evaluations"]["oof"]["roc_auc_score"] > 0.75

    return Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path=assets_dir,
        metrics_map=["roc_auc_score"],
        cross_validation_type=RepeatedStratifiedKFold,
        cross_validation_params=dict(n_splits=3, n_repeats=2, random_state=32),
        do_full_save=do_full_save,
    )


@pytest.fixture(scope="function", autouse=False)
def env_1():
    return Environment(
        train_dataset=get_breast_cancer_data(),
        environment_params_path="examples/advanced_examples/environment_params.json",
        root_results_path=assets_dir,
        cross_validation_params=dict(n_splits=3, shuffle=True, random_state=32),
    )


@pytest.fixture(scope="function", autouse=False)
def env_2():
    # noinspection PyUnusedLocal
    def get_holdout_set(train, target_column):
        return train, train.copy()

    return Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path=assets_dir,
        holdout_dataset=get_holdout_set,
        test_dataset=get_toy_classification_data(),
        metrics_map=["roc_auc_score"],
        cross_validation_type=StratifiedKFold,
        cross_validation_params=dict(n_splits=3, shuffle=True, random_state=32),
    )


@pytest.fixture(scope="function", autouse=False)
def env_3():
    def printer_callback():
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

    return Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path=assets_dir,
        metrics_map=["roc_auc_score"],
        holdout_dataset=get_toy_classification_data(),
        cross_validation_type=RepeatedStratifiedKFold,
        cross_validation_params=dict(n_splits=3, n_repeats=2, random_state=32),
        runs=2,
        experiment_callbacks=[
            printer_callback(),
            confusion_matrix_oof(),
            confusion_matrix_holdout(),
        ],
    )


@pytest.fixture(scope="function", autouse=False)
def env_4():
    return Environment(
        train_dataset=get_breast_cancer_data(target="diagnosis"),
        root_results_path=assets_dir,
        target_column="diagnosis",
        metrics_map=dict(
            roc_auc="roc_auc_score",
            f1=f1_score,
            f1_micro=lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
            f1_macro=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
        ),
        cross_validation_type="KFold",
        cross_validation_params=dict(n_splits=2, shuffle=True, random_state=42),
        verbose=1,
    )


@pytest.fixture(
    scope="function",
    autouse=False,
    params=[
        [],
        [(UnsortedIDLeaderboardRecorder, "Leaderboards/UnsortedIDLeaderboard.csv")],
        [
            (UnsortedIDLeaderboardRecorder, "Leaderboards/UnsortedIDLeaderboard.csv"),
            (YAMLDescriptionRecorder, "Experiments/YAMLDescriptions"),
        ],
    ],
)
def env_5(request):
    return Environment(
        train_dataset=get_breast_cancer_data(),
        root_results_path=assets_dir,
        target_column="diagnosis",
        metrics_map=["roc_auc_score"],
        cross_validation_type=StratifiedKFold,
        cross_validation_params=dict(n_splits=3, shuffle=True, random_state=32),
        experiment_recorders=request.param,
    )


##################################################
# Experiment Fixtures
##################################################
#################### XGBoost Experiments ####################
@pytest.fixture(scope="function", autouse=False)
def exp_xgb_0():
    return CVExperiment(model_initializer=XGBClassifier, model_init_params=dict(subsample=0.01))


@pytest.fixture(scope="function", autouse=False)
def exp_xgb_1():
    return CVExperiment(model_initializer=XGBClassifier, model_init_params=dict(subsample=0.5))


@pytest.fixture(scope="function", autouse=False)
def exp_xgb_2():
    return CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params={},
        model_extra_params=dict(fit=dict(verbose=False)),
    )


#################### KNeighborsClassifier Experiments ####################
@pytest.fixture(scope="function", autouse=False)
def exp_knc_0():
    return CVExperiment(model_initializer=KNeighborsClassifier, model_init_params={})


#################### SVC Experiments ####################
@pytest.fixture(scope="function", autouse=False)
def exp_svc_0():
    return CVExperiment(
        model_initializer=SVC, model_init_params=dict(C=1.0, kernel="linear", max_iter=100)
    )


##################################################
# Optimization Protocol Fixtures
##################################################
#################### SVC Optimization Protocols ####################
@pytest.fixture(scope="function", autouse=False, params=[None, "f1_micro", "f1", "f1_macro"])
def opt_svc_0(request):
    optimizer = BayesianOptimization(target_metric=request.param, iterations=2, random_state=32)
    optimizer.set_experiment_guidelines(
        model_initializer=SVC,
        model_init_params=dict(
            C=Real(0.9, 1.1),
            kernel=Categorical(["linear", "poly", "rbf"]),
            max_iter=Integer(50, 125),
            tol=1e-3,
        ),
    )
    optimizer.go()
    yield optimizer

    assert optimizer.target_metric == ("oof", (request.param or "roc_auc"))
    # lb = pd.read_csv(
    #     # Path(assets_dir) / "HyperparameterHunterAssets" / "Leaderboards" / "GlobalLeaderboard.csv",
    #     Path(assets_dir) / "Leaderboards" / "GlobalLeaderboard.csv",
    # )
    # assert lb.columns[0] == f"oof_{request.param}"


#################### XGBClassifier Optimization Protocols ####################
@pytest.fixture(scope="function", autouse=False)
def opt_xgb_0():
    optimizer = BayesianOptimization(iterations=5, random_state=1337)
    optimizer.set_experiment_guidelines(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            objective="reg:linear",
            max_depth=Integer(2, 20),
            learning_rate=Real(0.0001, 0.5),
            subsample=0.5,
            booster=Categorical(["gbtree", "dart"]),
        ),
    )
    optimizer.go()
    yield optimizer


##################################################
# Test Scenarios (Advanced)
##################################################
#################### do_full_save_example ####################
def test_do_full_save(env_0, exp_xgb_0, exp_xgb_1):
    assert has_experiment_result_file(assets_dir, exp_xgb_0, ["Descriptions", "ScriptBackups"])
    assert not has_experiment_result_file(assets_dir, exp_xgb_0, ["PredictionsOOF"])
    assert has_experiment_result_file(assets_dir, exp_xgb_1)


#################### environment_params_path_example ####################
def test_environment_params_path(env_1, exp_knc_0):
    assert env_1.root_results_path.startswith(assets_dir)
    assert env_1.target_column == ["diagnosis"]
    assert env_1.cross_validation_type.__name__ == "StratifiedKFold"
    assert "heartbeat" in env_1.file_blacklist

    assert has_experiment_result_file(
        assets_dir, exp_knc_0, ["Descriptions", "PredictionsOOF", "ScriptBackups"]
    )
    assert not has_experiment_result_file(assets_dir, exp_knc_0, ["Heartbeats"])


#################### holdout_test_datasets_example ####################
def test_holdout_test_datasets(env_2, exp_xgb_1):
    assert has_experiment_result_file(
        assets_dir,
        exp_xgb_1,
        [
            "Descriptions",
            "Heartbeats",
            "PredictionsOOF",
            "PredictionsHoldout",
            "PredictionsTest",
            "ScriptBackups",
        ],
    )


#################### lambda_callback_example ####################
def test_lambda_callback(env_3, exp_xgb_2):
    assert has_experiment_result_file(assets_dir, exp_xgb_2)


#################### multi_metric_example ####################
def test_multi_metric(env_4, exp_svc_0, opt_svc_0):
    assert len(opt_svc_0.similar_experiments) > 0

    for similar_experiment in opt_svc_0.similar_experiments:
        assert has_experiment_result_file(
            assets_dir, similar_experiment[2], ["Descriptions", "Heartbeats", "PredictionsOOF"]
        )


#################### recorder_example ####################
def test_recorder(env_5, opt_xgb_0):
    ...  # TODO: Assert that custom result files have been recorded
