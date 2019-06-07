"""This module defines simple utilities for making toy datasets to be used in testing/examples"""
##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd

###############################################
# Import Learning Assets
###############################################
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes, make_classification


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


def get_boston_data():
    """Get SKLearn's Boston House Prices regression dataset

    Returns
    -------
    df: `pandas.DataFrame`
        The Boston House Prices dataset of shape (506, 13 + 1)

    Notes
    -----
    The intended target column in this dataset is "MEDV"; however, the weighted distances
    column "DIS" can also be used as the target column

    Examples
    --------
    >>> get_boston_data().head()
          CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV
    0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98  24.0
    1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14  21.6
    2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03  34.7
    3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94  33.4
    4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33  36.2
    """
    data = load_boston()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["MEDV"] = data.target
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
