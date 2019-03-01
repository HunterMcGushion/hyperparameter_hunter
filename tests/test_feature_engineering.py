##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.feature_engineering import FeatureEngineer

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

##################################################
# Import Learning Assets
##################################################
from sklearn.preprocessing import StandardScaler

##################################################
# Dummy Objects for Testing
##################################################
pima_indians_head = pd.DataFrame(
    {
        "pregnancies": [6, 1, 8, 1, 0],
        "glucose": [148, 85, 183, 89, 137],
        "blood_pressure": [72, 66, 64, 66, 40],
        "skin_thickness": [35, 29, 0, 23, 35],
        "insulin": [0, 0, 0, 94, 168],
        "bmi": [33.6, 26.6, 23.3, 28.1, 43.1],
        "diabetes_pedigree_function": [0.627, 0.351, 0.672, 0.167, 2.288],
        "age": [50, 31, 32, 21, 33],
        "class": [1, 0, 1, 0, 1],
    }
)


def get_pima_data():
    train_dataset = pima_indians_head.iloc[1:5, :]
    holdout_dataset = pima_indians_head.iloc[[0], :]

    train_targets = train_dataset.loc[:, "class"]
    train_inputs = train_dataset.drop(["class"], axis=1)

    holdout_targets = holdout_dataset.loc[:, "class"]
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


def standard_scale_0(train_inputs, holdout_inputs):
    scaler = StandardScaler()
    train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    holdout_inputs[train_inputs.columns] = scaler.transform(holdout_inputs.values)
    return train_inputs, holdout_inputs


def test_0():
    train_inputs, train_targets, holdout_inputs, holdout_targets = get_pima_data()

    feature_engineer = FeatureEngineer()
    feature_engineer.add_step(set_nan_0, "set_nan_0")
    feature_engineer("foo", train_inputs=train_inputs.copy(), holdout_inputs=holdout_inputs.copy())

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
    feature_engineer("foo", train_inputs=train_inputs.copy(), holdout_inputs=holdout_inputs.copy())

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
    feature_engineer("foo", train_inputs=train_inputs.copy(), holdout_inputs=holdout_inputs.copy())

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
