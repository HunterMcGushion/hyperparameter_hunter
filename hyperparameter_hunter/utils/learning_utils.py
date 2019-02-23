"""This module defines simple utilities for making toy datasets to be used in testing/examples"""
##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd

###############################################
# Import Learning Assets
###############################################
from sklearn.datasets import load_breast_cancer, make_classification, load_diabetes


##################################################
# Dataset Utilities
##################################################
def get_breast_cancer_data(target="diagnosis"):
    """Get the Wisconsin Breast Cancer classification dataset, formatted as a DataFrame

    Parameters
    ----------
    target: String, default='diagnosis'
        What to name the column in `df` that contains the target output values

    Returns
    -------
    df: `pandas.DataFrame`
        The breast cancer dataset, with friendly column names"""
    data = load_breast_cancer()
    df = pd.DataFrame(data=data.data, columns=[_.replace(" ", "_") for _ in data.feature_names])
    df[target] = data.target
    return df


def get_diabetes_data(target="progression"):
    """Get the SKLearn Diabetes regression dataset, formatted as a DataFrame

    Parameters
    ----------
    target: String, default='progression'
        What to name the column in `df` that contains the target output values

    Returns
    -------
    df: `pandas.DataFrame`
        The diabetes dataset, with friendly column names"""
    data = load_diabetes()
    df = pd.DataFrame(data=data.data, columns=[_.replace(" ", "_") for _ in data.feature_names])
    df[target] = data.target
    return df


def get_toy_classification_data(
    target="target", n_samples=300, n_classes=2, shuffle=True, random_state=32, **kwargs
):
    """Wrapper around `sklearn.datasets.make_classification` to produce a `pandas.DataFrame`"""
    x, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        shuffle=shuffle,
        random_state=random_state,
        **kwargs
    )
    train_df = pd.DataFrame(data=x, columns=range(x.shape[1]))
    train_df[target] = y
    return train_df
