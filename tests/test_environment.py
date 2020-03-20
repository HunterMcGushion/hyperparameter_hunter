##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.recipes import confusion_matrix_oof, confusion_matrix_holdout
from hyperparameter_hunter.environment import (
    Environment,
    define_holdout_set,
    validate_file_blacklist,
)

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import pandas as pd
from pkg_resources import get_distribution
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


##################################################
# Dummy Objects for Testing
##################################################
# TODO: Use `learning_utils.get_breast_cancer_data`. Will need to change expected `env_keys`
def get_breast_cancer_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["diagnosis"] = data.target
    return df


def get_holdout_set(train, target_column):
    # Hello, I am a test comment
    return train, train.copy()


def args_ids_for(scenarios):
    return dict(argvalues=scenarios, ids=[f"{_}" for _ in range(len(scenarios))])


train_dataset = get_breast_cancer_data()
cv_params = dict(n_splits=5, shuffle=True, random_state=32)
repeated_cv_params = dict(n_splits=5, n_repeats=2, random_state=32)

default_env_params = dict(
    train_dataset=train_dataset,
    environment_params_path=None,
    results_path=assets_dir,
    holdout_dataset=get_holdout_set,
    test_dataset=train_dataset.copy(),
    target_column="diagnosis",
    do_predict_proba=False,
    prediction_formatter=None,
    metrics_params=dict(
        metrics_map=dict(roc="roc_auc_score", f1="f1_score", acc="accuracy_score"),
        in_fold="all",
        oof="all",
        holdout="all",
    ),
    random_seeds=None,
    runs=3,
    cross_validation_type="StratifiedKFold",
    verbose=True,
    file_blacklist=None,
    reporting_params=dict(add_frame=False),
    cross_validation_params=repeated_cv_params,
)

##################################################
# Environment Initialization Scenarios
##################################################
if get_distribution("scikit-learn").version >= "0.22.1":
    env_keys = [
        "mq2AbzYPiRbLSIRmfzhChbJU4ZRSyfa_QHpzn2Q91Lw=",
        "3cqeYUuJpzdTqZce_wYJTRnSsJAJhjVe0-ophuYqT-w=",
        "BKd3SkDICQWcmJEF6HgMUpoy6qvzexWcQvoe8Z668RI=",
        "QASd57OkHyewVf6fDtia3QDjUgQJgH2oxJfy0rN8Mk0=",
        "4wfjujp3xIAb2-48TYAnmeTw8pnGfeFeYLJsUUoMmP8=",
        "ypSS2UgcT48_X8YBPdGIwXcqnnzUyjMRnpNla6BM-aU=",
    ]
elif get_distribution("scikit-learn").version >= "0.22":
    env_keys = [
        "CrpaPtZc3pb_iRXx3yOhj9Ia_duy_5YqWVQ9pAvZyYM=",
        "NJEmEitFdS0bTtSHMvb1C9I6vSJ4GlzkIy2JMCidBPo=",
        "fHNr3fZyA-uZjAJWMd_SgMyyb7dex6_psYIXy3PCvSk=",
        "ayj0zhnngjAtuEniXz3yXhqKR71w0eoS13UTUMO171I=",
        "KT732K9_WsVFGeC1bQMgexTqjjtheq0Lge47lSzLWGU=",
        "WQXaRTCYi-OXXnOinTfwgY7KI6JVmxio4MqNI3KTDcc=",
    ]
elif get_distribution("scikit-learn").version >= "0.21.0":
    env_keys = [
        "JmVKziNqJKJu87ZZwtMb4m4nj_aArpSQ_IulDTIbO_E=",
        "a_ykAZaAPL1wvbQXtiMbJx6OpGZdrOlYlnjDQtu8nKI=",
        "3uA1nG3gjPrTVYuqCmDPlAYEMTPFvb5x34MOgPyK9_Q=",
        "EXzJlklxmF6LuWgfeuKym8vuJAowRJ3gXSI-jHUasEk=",
        "GRJ79aqffE333z6grtB56JGxnuPzGPh1Eq8v5WoYYxw=",
        "yKrfVZ-jCdpoxaH1E514QV4nUJGrLEiCH6vVbY58rN0=",
    ]
else:
    env_keys = [
        "2fqFnCq1-qjWDrvv6Gok6aNbUGe2yoklWMuvbg88ncQ=",
        "RGSWQAec5s4YHuotu1nbhUN7qqwULoXJ7sHwI2B1XHI=",
        "GktSneqJVTS-uHH_1JZv6vlaPV3oZhO4EYSNae0hUFQ=",
        "43lJTamcEf1B8rSnEPuDBICPa7iz5Zbenhg20ARhNbs=",
        "borjWHSHy-BMAFX-Zp0YzxXuhAZ7X2J5yzB2T2uF0qo=",
        "skHEmRcUGUCmQC2AGLpMSJYxpRtNdEJ-zEPQKHn_rvQ=",
    ]

scenarios_cv_params = [[repeated_cv_params, env_keys[0]], [cv_params, env_keys[1]]]
scenarios_metrics_map = [
    [dict(roc="roc_auc_score", acc="accuracy_score"), env_keys[2]],
    [
        dict(
            roc="roc_auc_score",
            f1="f1_score",
            acc=lambda _t, _p: accuracy_score(_t, np.clip(np.round(_p), 0, 1)),
        ),
        env_keys[3],
    ],
]
scenarios_cross_experiment_params = [
    [10, "StratifiedKFold", repeated_cv_params, env_keys[4]],
    [3, "KFold", cv_params, env_keys[5]],
]


@pytest.mark.parametrize(["_cv_params", "expected"], **args_ids_for(scenarios_cv_params))
def test_environment_init_cv_params(_cv_params, expected):
    env = Environment(**dict(default_env_params, **dict(cross_validation_params=_cv_params)))
    assert env == expected


@pytest.mark.parametrize(["metrics", "expected"], **args_ids_for(scenarios_metrics_map))
def test_environment_init_metrics(metrics, expected):
    env = Environment(
        **dict(
            default_env_params,
            **dict(
                metrics_params=dict(metrics_map=metrics, in_fold="all", oof="all", holdout="all")
            ),
        )
    )
    assert env == expected


@pytest.mark.parametrize(
    ["runs", "cv_type", "_cv_params", "expected"], **args_ids_for(scenarios_cross_experiment_params)
)
def test_environment_init_cross_experiment_params(runs, cv_type, _cv_params, expected):
    env = Environment(
        **dict(
            default_env_params,
            **dict(runs=runs, cross_validation_type=cv_type, cross_validation_params=_cv_params),
        )
    )
    assert env == expected


# TODO: Refactor horrifying section above - Was this before past-Hunter knew about indirectly
#   parametrized fixtures, or parametrization via meta-function hook?


##################################################
# `Environment.__repr__` Tests
##################################################
@pytest.mark.parametrize("env_params", [default_env_params])
def test_environment_repr(env_params):
    """Test that :meth:`Environment.__repr__` returns the expected value"""
    env = Environment(**env_params)
    assert env.__repr__() == f"Environment(cross_experiment_key={env.cross_experiment_key!s})"


##################################################
# Environment Property Scenarios
##################################################
#################### `results_path` ####################
def test_results_path_setter_type_error(env_fixture_0):
    with pytest.raises(TypeError, match="results_path must be None or str, not .*"):
        env_fixture_0.results_path = lambda _: "`results_path` can't be a callable, dummy"


#################### `cv_type` ####################
@pytest.mark.parametrize("new_value", ["RepeatedStratifiedKFold", RepeatedStratifiedKFold])
def test_cv_type_setter_valid(env_fixture_0, new_value):
    env_fixture_0.cv_type = new_value
    assert env_fixture_0.cv_type == RepeatedStratifiedKFold


def test_cv_type_setter_attribute_error(env_fixture_0):
    with pytest.raises(AttributeError, match="'foo' not in `sklearn.model_selection._split`"):
        env_fixture_0.cv_type = "foo"


#################### `experiment_callbacks` ####################
cm_oof, cm_holdout = confusion_matrix_oof(), confusion_matrix_holdout()


@pytest.mark.parametrize(
    ["new_value", "expected"],
    [
        ([], []),
        ([cm_oof], [cm_oof]),
        (cm_oof, [cm_oof]),
        ([cm_oof, cm_holdout], [cm_oof, cm_holdout]),
    ],
)
def test_experiment_callbacks_setter_valid(env_fixture_0, new_value, expected):
    env_fixture_0.experiment_callbacks = new_value
    assert env_fixture_0.experiment_callbacks == expected


def test_experiment_callbacks_setter_type_error(env_fixture_0):
    with pytest.raises(TypeError, match="experiment_callbacks must be classes, not .*"):
        env_fixture_0.experiment_callbacks = ["foo"]


def test_experiment_callbacks_setter_value_error(env_fixture_0):
    with pytest.raises(ValueError, match="experiment_callbacks must be LambdaCallback instances.*"):
        env_fixture_0.experiment_callbacks = [RepeatedStratifiedKFold]


##################################################
# `define_holdout_set` Scenarios
##################################################
def test_define_holdout_set_str(monkeypatch):
    dummy_df = pd.DataFrame(dict(a=[0, 1], b=[2, 3]))

    # noinspection PyUnusedLocal
    def mock_pandas_read_csv(*args, **kwargs):
        return dummy_df

    monkeypatch.setattr(pd, "read_csv", mock_pandas_read_csv)
    assert define_holdout_set(pd.DataFrame(dict(a=[4], b=[5])), "foo", "x")[1].equals(dummy_df)


@pytest.mark.parametrize("holdout_set", [42, np.array([[0, 1], [2, 3]])])
def test_define_holdout_set_type_error(holdout_set):
    with pytest.raises(TypeError, match="holdout_set must be None, DataFrame, callable, or str,.*"):
        define_holdout_set(pd.DataFrame(), holdout_set, "target")


def test_define_holdout_set_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        define_holdout_set(pd.DataFrame(), "foo", "target")


@pytest.mark.parametrize("holdout_set", [dict(), dict(a=[0], c=[2]), dict(a=[0])])
def test_define_holdout_set_value_error(holdout_set):
    with pytest.raises(ValueError, match="Mismatched columns.*"):
        define_holdout_set(pd.DataFrame(dict(a=[0, 1], b=[2, 3])), pd.DataFrame(holdout_set), "x")


##################################################
# `validate_file_blacklist` Scenarios
##################################################
@pytest.mark.parametrize(
    ["blacklist", "expected"],
    [["ALL", "ALL"], [["current_heartbeat"], ["current_heartbeat", "heartbeat"]]],
)
def test_validate_file_blacklist(blacklist, expected):
    assert validate_file_blacklist(blacklist) == expected


@pytest.mark.parametrize("blacklist", ["foo", 42, dict(a=17, b=18)])
def test_validate_file_blacklist_type_error_list(blacklist):
    with pytest.raises(TypeError, match="Expected blacklist to be a list, not:.*"):
        validate_file_blacklist(blacklist)


@pytest.mark.parametrize("blacklist", [["foo", None], [1], [["foo"]], [dict(a="foo")]])
def test_validate_file_blacklist_type_error_contents(blacklist):
    with pytest.raises(TypeError, match="Expected blacklist contents to be strings, not:.*"):
        validate_file_blacklist(blacklist)


@pytest.mark.parametrize("blacklist", [["description", "foo"], ["foo"]])
def test_validate_file_blacklist_value_error(blacklist):
    with pytest.raises(ValueError, match="Invalid blacklist value: foo.*"):
        validate_file_blacklist(blacklist)
