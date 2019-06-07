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
    target: String, default="diagnosis"
        What to name the column in `df` that contains the target output values

    Returns
    -------
    df: `pandas.DataFrame`
        The breast cancer dataset, with friendly column names"""
    data = load_breast_cancer()
    df = pd.DataFrame(data=data.data, columns=[_.replace(" ", "_") for _ in data.feature_names])
    df[target] = data.target
    return df


def get_pima_indians_data(target="class"):
    """Get the Pima Indians Diabetes binary classification dataset, formatted as a DataFrame

    Parameters
    ----------
    target: String, default="class"
        What to name the column in `df` that contains the target output values

    Returns
    -------
    df: `pandas.DataFrame`
        The Pima Indians dataset, of shape (768, 8 + 1), with column names of:
            "pregnancies",
            "glucose",
            "bp" (shortened from "blood_pressure"),
            "skin_thickness",
            "insulin",
            "bmi",
            "dpf" (shortened from "diabetes_pedigree_function"),
            "age",
            "class" (or given `target` column)

    Notes
    -----
    This dataset is originally from the National Institute of Diabetes and Digestive and Kidney
    Diseases. Thanks to Jason Brownlee (of MachineLearningMastery.com), who has generously made a
    public repository to collect copies of all the datasets he uses in his tutorials

    Examples
    --------
    >>> get_pima_indians_data().head()
       pregnancies  glucose  bp  skin_thickness  insulin   bmi    dpf  age  class
    0            6      148  72              35        0  33.6  0.627   50      1
    1            1       85  66              29        0  26.6  0.351   31      0
    2            8      183  64               0        0  23.3  0.672   32      1
    3            1       89  66              23       94  28.1  0.167   21      0
    4            0      137  40              35      168  43.1  2.288   33      1
    """
    input_cols = ["pregnancies", "glucose", "bp", "skin_thickness", "insulin", "bmi", "dpf", "age"]
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        names=(input_cols + [target]),
    )
    return df


def get_diabetes_data(target="progression"):
    """Get the SKLearn Diabetes regression dataset, formatted as a DataFrame

    Parameters
    ----------
    target: String, default="progression"
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
