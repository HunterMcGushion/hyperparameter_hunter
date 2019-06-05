##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, FeatureEngineer, EngineerStep
from hyperparameter_hunter.callbacks.recipes import dataset_recorder

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

##################################################
# Dummy Objects for Testing
##################################################
pima_cols = ["pregnancies", "glucose", "bp", "skin_thickness", "insulin", "bmi", "dpf", "age"]
pima_indians_head = pd.DataFrame(
    {
        "pregnancies": [6, 1, 8, 1, 0],
        "glucose": [148, 85, 183, 89, 137],
        "bp": [72, 66, 64, 66, 40],  # Originally "blood_pressure"
        "skin_thickness": [35, 29, 0, 23, 35],
        "insulin": [0, 0, 0, 94, 168],
        "bmi": [33.6, 26.6, 23.3, 28.1, 43.1],
        "dpf": [0.627, 0.351, 0.672, 0.167, 2.288],  # Originally "diabetes_pedigree_function"
        "age": [50, 31, 32, 21, 33],
        "class": [1, 0, 1, 0, 1],
    }
)


def get_pima_data():
    train_dataset = pima_indians_head.iloc[1:5, :]
    holdout_dataset = pima_indians_head.iloc[[0], :]

    train_targets = train_dataset.loc[:, ["class"]]
    train_inputs = train_dataset.drop(["class"], axis=1)

    holdout_targets = holdout_dataset.loc[:, ["class"]]
    holdout_inputs = holdout_dataset.drop(["class"], axis=1)

    return train_inputs, train_targets, holdout_inputs, holdout_targets


def set_nan_0(train_inputs, holdout_inputs):
    cols = [1, 2, 3, 4, 5]
    train_inputs.iloc[:, cols] = train_inputs.iloc[:, cols].replace(0, np.NaN)
    holdout_inputs.iloc[:, cols] = holdout_inputs.iloc[:, cols].replace(0, np.NaN)
    return train_inputs, holdout_inputs


def impute_negative_one_0(train_inputs, holdout_inputs):
    train_inputs.fillna(-1, inplace=True)
    holdout_inputs.fillna(-1, inplace=True)
    return train_inputs, holdout_inputs


def impute_negative_one_1(all_inputs):
    all_inputs.fillna(-1, inplace=True)
    return all_inputs


def standard_scale_0(train_inputs, holdout_inputs):
    scaler = StandardScaler()
    train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    holdout_inputs[train_inputs.columns] = scaler.transform(holdout_inputs.values)
    return train_inputs, holdout_inputs


def standard_scale_1(train_inputs, validation_inputs):
    scaler = StandardScaler()
    train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    validation_inputs[train_inputs.columns] = scaler.transform(validation_inputs.values)
    return train_inputs, validation_inputs


def standard_scale_2(train_inputs, non_train_inputs):
    scaler = StandardScaler()
    train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    non_train_inputs[train_inputs.columns] = scaler.transform(non_train_inputs.values)
    return train_inputs, non_train_inputs


def test_0():
    train_inputs, train_targets, holdout_inputs, holdout_targets = get_pima_data()

    feature_engineer = FeatureEngineer()
    feature_engineer.add_step(set_nan_0, name="set_nan_0")
    feature_engineer(
        "pre_cv", train_inputs=train_inputs.copy(), holdout_inputs=holdout_inputs.copy()
    )

    expected_train_inputs = [
        [1, 85, 66, 29, np.NaN, 26.6, 0.351, 31],
        [8, 183, 64, np.NaN, np.NaN, 23.3, 0.672, 32],
        [1, 89, 66, 23, 94, 28.1, 0.167, 21],
        [0, 137, 40, 35, 168, 43.1, 2.288, 33],
    ]
    expected_holdout_inputs = [[6, 148, 72, 35, np.NaN, 33.6, 0.627, 50]]

    assert_array_almost_equal(feature_engineer.datasets["train_inputs"], expected_train_inputs)
    assert_array_almost_equal(feature_engineer.datasets["holdout_inputs"], expected_holdout_inputs)


def test_1():
    train_inputs, train_targets, holdout_inputs, holdout_targets = get_pima_data()

    feature_engineer = FeatureEngineer()
    feature_engineer.add_step(set_nan_0)
    assert feature_engineer._steps[-1].name == "set_nan_0"
    feature_engineer.add_step(impute_negative_one_0)
    assert feature_engineer._steps[-1].name == "impute_negative_one_0"
    feature_engineer(
        "pre_cv", train_inputs=train_inputs.copy(), holdout_inputs=holdout_inputs.copy()
    )

    expected_train_inputs = [
        [1, 85, 66, 29, -1, 26.6, 0.351, 31],
        [8, 183, 64, -1, -1, 23.3, 0.672, 32],
        [1, 89, 66, 23, 94, 28.1, 0.167, 21],
        [0, 137, 40, 35, 168, 43.1, 2.288, 33],
    ]
    expected_holdout_inputs = [[6, 148, 72, 35, -1, 33.6, 0.627, 50]]

    assert_array_almost_equal(feature_engineer.datasets["train_inputs"], expected_train_inputs)
    assert_array_almost_equal(feature_engineer.datasets["holdout_inputs"], expected_holdout_inputs)


def test_2():
    train_inputs, train_targets, holdout_inputs, holdout_targets = get_pima_data()

    feature_engineer = FeatureEngineer()
    feature_engineer.add_step(set_nan_0)
    feature_engineer.add_step(impute_negative_one_0)
    feature_engineer.add_step(standard_scale_0)
    feature_engineer(
        "pre_cv", train_inputs=train_inputs.copy(), holdout_inputs=holdout_inputs.copy()
    )

    expected_train_inputs = [
        [-0.468521, -0.962876, 0.636364, 0.548821, -0.929624, -0.48321, -0.618238, 0.363422],
        [1.717911, 1.488081, 0.454545, -1.646464, -0.929624, -0.917113, -0.235491, 0.571092],
        [-0.468521, -0.862837, 0.636364, 0.109764, 0.408471, -0.285982, -0.837632, -1.713275],
        [-0.780869, 0.337632, -1.727273, 0.987878, 1.450776, 1.686305, 1.691360, 0.778761],
    ]
    expected_holdout_inputs = [
        [1.093216, 0.612739, 1.181818, 0.987878, -0.929624, 0.437190, -0.289147, 4.309145]
    ]

    assert_array_almost_equal(feature_engineer.datasets["train_inputs"], expected_train_inputs)
    assert_array_almost_equal(feature_engineer.datasets["holdout_inputs"], expected_holdout_inputs)


##################################################
# `CVExperiment` with `FeatureEngineer` Integration
##################################################
# noinspection PyUnusedLocal
def holdout_first_row(train_dataset, target_column):
    return train_dataset.iloc[1:5, :], train_dataset.iloc[[0], :]


small_toy_dataset = pd.DataFrame(
    data=[[1, -1, 2, 1], [2, 0, 0, 0], [0, 1, -1, 1], [-1, 1, 0, 0]], columns=["a", "b", "c", "t"]
)

initial_fold_train_inputs = [
    pd.DataFrame([[1, -1, 2], [2, 0, 0], [-1, 1, 0]], columns=list("abc"), index=[0, 1, 3]),
    pd.DataFrame([[2, 0, 0], [0, 1, -1], [-1, 1, 0]], columns=list("abc"), index=[1, 2, 3]),
    pd.DataFrame([[1, -1, 2], [0, 1, -1], [-1, 1, 0]], columns=list("abc"), index=[0, 2, 3]),
    pd.DataFrame([[1, -1, 2], [2, 0, 0], [0, 1, -1]], columns=list("abc"), index=[0, 1, 2]),
]
engineered_fold_train_inputs = [
    pd.DataFrame(
        [
            [0.26726124, -1.22474487, 1.41421356],
            [1.06904497, 0.0, -0.70710678],
            [-1.33630621, 1.22474487, -0.70710678],
        ],
        columns=list("abc"),
        index=[0, 1, 3],
    ),
    pd.DataFrame(
        [
            [1.33630621, -1.41421356, 0.70710678],
            [-0.26726124, 0.70710678, -1.41421356],
            [-1.06904497, 0.70710678, 0.70710678],
        ],
        columns=list("abc"),
        index=[1, 2, 3],
    ),
    pd.DataFrame(
        [
            [1.22474487, -1.41421356, 1.33630621],
            [0.0, 0.70710678, -1.06904497],
            [-1.22474487, 0.70710678, -0.26726124],
        ],
        columns=list("abc"),
        index=[0, 2, 3],
    ),
    pd.DataFrame(
        [
            [0.0, -1.22474487, 1.33630621],
            [1.22474487, 0.0, -0.26726124],
            [-1.22474487, 1.22474487, -1.06904497],
        ],
        columns=list("abc"),
        index=[0, 1, 2],
    ),
]
initial_fold_validation_inputs = [
    pd.DataFrame([[0, 1, -1]], columns=list("abc"), index=[2]),
    pd.DataFrame([[1, -1, 2]], columns=list("abc"), index=[0]),
    pd.DataFrame([[2, 0, 0]], columns=list("abc"), index=[1]),
    pd.DataFrame([[-1, 1, 0]], columns=list("abc"), index=[3]),
]
engineered_fold_validation_inputs = [
    pd.DataFrame([[-0.53452248, 1.22474487, -1.76776695]], columns=list("abc"), index=[2]),
    pd.DataFrame([[0.53452248, -3.53553391, 4.94974747]], columns=list("abc"), index=[0]),
    pd.DataFrame([[2.44948974, -0.35355339, -0.26726124]], columns=list("abc"), index=[1]),
    pd.DataFrame([[-2.44948974, 1.22474487, -0.26726124]], columns=list("abc"), index=[3]),
]


@pytest.fixture()
def toy_environment_fixture():
    return Environment(
        train_dataset=pima_indians_head,
        holdout_dataset=holdout_first_row,
        metrics=["roc_auc_score"],
        target_column="class",
        cv_params=dict(n_splits=3, shuffle=True, random_state=32),
    )


@pytest.fixture()
def dataset_recorder_env(request):
    return Environment(
        train_dataset=small_toy_dataset,
        holdout_dataset=getattr(request, "param", None),
        metrics=["accuracy_score"],
        target_column="t",
        cv_params=dict(n_splits=4, shuffle=True, random_state=32),
        experiment_callbacks=[dataset_recorder()],
    )


@pytest.fixture()
def prepped_experiment(request):
    """Build a partially prepared :class:`~hyperparameter_hunter.experiments.CVExperiment` instance

    Specifically, automatic execution is disabled via `auto_start=False`, then the following methods
    are called:

    1. :meth:`~hyperparameter_hunter.experiments.BaseExperiment.preparation_workflow`,
    2. :meth:`~hyperparameter_hunter.experiments.BaseExperiment._initialize_random_seeds`, and
    3. :meth:`~hyperparameter_hunter.experiments.BaseExperiment.on_experiment_start`, which
       initializes the four :mod:`~hyperparameter_hunter.data.datasets` classes, then performs
       pre-CV feature engineering

    Notes
    -----
    Directly calling `on_experiment_start` is ok in this test because after calling
    `_initialize_random_seeds`, `BaseExperiment` calls `execute`, which is implemented by
    `BaseCVExperiment`, and only calls `cross_validation_workflow`, whose first task is to call
    `on_experiment_start`. So nothing gets skipped in between"""
    #################### Build `feature_engineer` ####################
    feature_engineer = FeatureEngineer(steps=request.param)

    #################### Partially Prepare `CVExperiment` ####################
    experiment = CVExperiment(
        model_initializer=AdaBoostClassifier,
        model_init_params=dict(),
        feature_engineer=feature_engineer,
        auto_start=False,
    )
    experiment.preparation_workflow()
    # noinspection PyProtectedMember
    experiment._initialize_random_seeds()
    experiment.on_experiment_start()

    return experiment


@pytest.fixture()
def experiment_fixture(request):
    #################### Build `feature_engineer` ####################
    feature_engineer = FeatureEngineer(steps=request.param)

    #################### Execute `CVExperiment` ####################
    experiment = CVExperiment(
        model_initializer=AdaBoostClassifier,
        model_init_params=dict(),
        feature_engineer=feature_engineer,
    )
    return experiment


#################### Expected End `CVExperiment` Data ####################
end_data_unchanged = get_pima_data()
end_data_sn = (
    pd.DataFrame(
        data=[
            [1, 85, 66, 29, np.NaN, 26.6, 0.351, 31],
            [8, 183, 64, np.NaN, np.NaN, 23.3, 0.672, 32],
            [1, 89, 66, 23, 94, 28.1, 0.167, 21],
            [0, 137, 40, 35, 168, 43.1, 2.288, 33],
        ],
        columns=pima_cols,
        index=[1, 2, 3, 4],
    ),
    end_data_unchanged[1],
    pd.DataFrame([[6, 148, 72, 35, np.NaN, 33.6, 0.627, 50]], columns=pima_cols),
    end_data_unchanged[3],
)
end_data_sn_ino = (
    pd.DataFrame(
        data=[
            [1, 85, 66, 29, -1, 26.6, 0.351, 31],
            [8, 183, 64, -1, -1, 23.3, 0.672, 32],
            [1, 89, 66, 23, 94, 28.1, 0.167, 21],
            [0, 137, 40, 35, 168, 43.1, 2.288, 33],
        ],
        columns=pima_cols,
        index=[1, 2, 3, 4],
    ),
    end_data_unchanged[1],
    pd.DataFrame([[6, 148, 72, 35, -1, 33.6, 0.627, 50]], columns=pima_cols),
    end_data_unchanged[3],
)
end_data_sn_ino_ss = (
    pd.DataFrame(
        data=[
            [-0.468521, -0.962876, 0.636364, 0.548821, -0.929624, -0.48321, -0.618238, 0.363422],
            [1.717911, 1.488081, 0.454545, -1.646464, -0.929624, -0.917113, -0.235491, 0.571092],
            [-0.468521, -0.862837, 0.636364, 0.109764, 0.408471, -0.285982, -0.837632, -1.713275],
            [-0.780869, 0.337632, -1.727273, 0.987878, 1.450776, 1.686305, 1.691360, 0.778761],
        ],
        columns=pima_cols,
        index=[1, 2, 3, 4],
    ),
    end_data_unchanged[1],
    pd.DataFrame(
        [[1.093216, 0.612739, 1.181818, 0.987878, -0.929624, 0.437190, -0.289147, 4.309145]],
        columns=pima_cols,
    ),
    end_data_unchanged[3],
)
end_data_sn_ss = (
    pd.DataFrame(
        data=[
            [-0.468521, -0.962876, 0.636363, 0.0, np.NaN, -0.483210, -0.618237, 0.363421],
            [1.717911, 1.488081, 0.454545, np.NaN, np.NaN, -0.917113, -0.235490, 0.571091],
            [-0.468521, -0.862837, 0.636363, -1.224744, -1.0, -0.285981, -0.837631, -1.713274],
            [-0.780868, 0.337631, -1.727272, 1.224744, 1.0, 1.686305, 1.691360, 0.778761],
        ],
        columns=pima_cols,
        index=[1, 2, 3, 4],
    ),
    end_data_unchanged[1],
    pd.DataFrame(
        [[1.093216, 0.612739, 1.181818, 1.224744, np.NaN, 0.437190, -0.289146, 4.309145]],
        columns=pima_cols,
    ),
    end_data_unchanged[3],
)


@pytest.mark.parametrize(
    ["prepped_experiment", "end_data"],
    [
        (None, end_data_unchanged),
        ([set_nan_0], end_data_sn),
        ([impute_negative_one_0], end_data_unchanged),
        ([set_nan_0, impute_negative_one_0], end_data_sn_ino),
        ([set_nan_0, impute_negative_one_1], end_data_sn_ino),
        ([set_nan_0, impute_negative_one_0, standard_scale_0], end_data_sn_ino_ss),
        ([set_nan_0, standard_scale_0], end_data_sn_ss),
        ([set_nan_0, EngineerStep(standard_scale_0)], end_data_sn_ss),
    ],
    indirect=["prepped_experiment"],
)
def test_feature_engineer_experiment(toy_environment_fixture, prepped_experiment, end_data):
    assert_frame_equal(prepped_experiment.data_train.input.T.d, end_data[0], check_dtype=False)
    assert_frame_equal(prepped_experiment.data_train.target.T.d, end_data[1], check_dtype=False)
    assert_frame_equal(prepped_experiment.data_holdout.input.T.d, end_data[2], check_dtype=False)
    assert_frame_equal(prepped_experiment.data_holdout.target.T.d, end_data[3], check_dtype=False)


@pytest.mark.parametrize(
    "experiment_fixture", [[standard_scale_1], [standard_scale_2]], indirect=True
)
@pytest.mark.parametrize(
    "end_data",
    [
        [
            initial_fold_train_inputs,
            initial_fold_validation_inputs,
            engineered_fold_train_inputs,
            engineered_fold_validation_inputs,
        ]
    ],
)
def test_intra_cv_engineer_experiment(dataset_recorder_env, experiment_fixture, end_data):
    """Tests that original and transformed train/validation inputs for each fold are correct"""
    for f, datasets in enumerate(experiment_fixture.stat_aggregates["_datasets"]["on_fold_start"]):
        assert_frame_equal(datasets["data_train"].input.fold, end_data[0][f])
        assert_frame_equal(datasets["data_oof"].input.fold, end_data[1][f])
        assert_frame_equal(datasets["data_train"].input.T.fold, end_data[2][f])
        assert_frame_equal(datasets["data_oof"].input.T.fold, end_data[3][f])
