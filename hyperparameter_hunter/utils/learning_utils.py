"""This module defines simple utilities for making toy datasets to be used in testing/examples"""
##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import pandas as pd

###############################################
# Import Learning Assets
###############################################
from sklearn.datasets import load_breast_cancer, make_classification

# from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


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


##################################################
# Miscellaneous Unused Utilities
# Pay no attention to these
##################################################
def upsample(input_df, target_df, target_feature, target_value, **kwargs):
    """EXPERIMENTAL"""
    ##################################################
    # Get Samples Matching target_value
    ##################################################
    # FLAG: VERIFY THAT BELOW ACTUALLY WORKS
    try:
        add_samples = pd.Series(target_df[target_feature] == target_value)
    except KeyError:
        add_samples = pd.Series(target_df == target_value)
        add_samples = add_samples[add_samples].index
    # add_samples = pd.Series(target_df[target_feature] == target_value)  # FLAG: ORIGINAL
    # FLAG: VERIFY THAT ABOVE ACTUALLY WORKS

    ##################################################
    # Add Samples
    ##################################################
    input_df = pd.concat([input_df, input_df.loc[add_samples]], axis=0)
    target_df = pd.concat([target_df, target_df.loc[add_samples]], axis=0)

    ##################################################
    # Shuffle Data
    ##################################################
    indexes = np.arange(len(input_df))
    np.random.shuffle(indexes)

    input_df = input_df.iloc[indexes]
    target_df = target_df.iloc[indexes]

    return (input_df, target_df)


# def add_noise(series, noise_level):
#     return series * (1 + noise_level * np.random.randn(len(series)))
#
#
# def high_cardinality_categorical_encode(
#         train_series=None,
#         validation_series=None,
#         test_series=None,
#         target=None,
#         min_samples_leaf=1,
#         smoothing=1,
#         noise_level=0
# ):
#     assert len(train_series) == len(target)
#     assert train_series.name == test_series.name
#
#     temp = pd.concat([train_series, target], axis=1)
#
#     ##################################################
#     # Compute Target Mean
#     ##################################################
#     averages = temp.groupby(by=train_series.name)[target.name].agg(['mean', 'count'])
#
#     ##################################################
#     # Compute Smoothing
#     ##################################################
#     smoothing = (1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing)))
#
#     ##################################################
#     # Apply Average Function to Target Data
#     ##################################################
#     prior = target.mean()
#
#     ##################################################
#     # The bigger the count, the less full_avg is taken into account
#     ##################################################
#     averages[target.name] = (prior * (1 - smoothing) + averages['mean'] * smoothing)
#     averages.drop(['mean', 'count'], axis=1, inplace=True)
#
#     ##################################################
#     # Apply Averages to Train Series
#     ##################################################
#     encoded_train_series = pd.merge(
#         train_series.to_frame(train_series.name),
#         averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
#         on=train_series.name,
#         how='left'
#     )['average'].rename(train_series.name + '_mean').fillna(prior)
#     encoded_train_series.index = train_series.index
#     add_noise(encoded_train_series, noise_level)
#
#     ##################################################
#     # Apply Averages to Test Series
#     ##################################################
#     encoded_test_series = pd.merge(
#         test_series.to_frame(test_series.name),
#         averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
#         on=test_series.name,
#         how='left'
#     )['average'].rename(train_series.name + '_mean').fillna(prior)
#     encoded_test_series.index = test_series.index
#     add_noise(encoded_test_series, noise_level)
#
#     ##################################################
#     # Apply Averages to Validation Series
#     ##################################################
#     if validation_series is not None:
#         encoded_validation_series = pd.merge(
#             validation_series.to_frame(validation_series.name),
#             averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
#             on=validation_series.name,
#             how='left'
#         )['average'].rename(train_series.name + '_mean').fillna(prior)
#         encoded_validation_series.index = validation_series.index
#         add_noise(encoded_validation_series, noise_level)
#
#         return (encoded_train_series, encoded_validation_series, encoded_test_series)
#
#     return (encoded_train_series, encoded_test_series)
#
#     # return (add_noise(
#     #   encoded_train_series, noise_level),
#     #   add_noise(encoded_validation_series, noise_level),
#     #   add_noise(encoded_test_series, noise_level)
#     # )


# def test_iris_upsample():
#     iris_data = pd.read_csv('../data/iris.csv')
#     iris_input = iris_data.drop(['species'], axis=1)
#     iris_target = iris_data['species']
#
#     label_binarizer = LabelBinarizer()
#     binarized_targets = label_binarizer.fit_transform(iris_target)
#
#     target_counts = zip(np.unique(iris_target, return_counts=True))
#
#     # encoder = OneHotEncoder()
#     # one_hot_labels = encoder.fit_transform(iris_target.reshape(-1, 1)).toarray()
#
#     data = pd.DataFrame(
#         data=np.c_[iris_input.values, binarized_targets],
#         columns=iris_input.columns.values.tolist() + ['is_' + _ for _ in target_counts[0][0]]
#     )
#
#     return (
#         data.drop(['is_setosa', 'is_versicolor', 'is_virginica'], axis=1),
#         data[['is_setosa', 'is_versicolor', 'is_virginica']]
#     )
