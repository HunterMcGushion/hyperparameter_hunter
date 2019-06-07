"""This example is a loose HyperparameterHunter adaptation of the SKLearn example on the
"Effect of transforming the targets in regression models"
(https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#real-world-data-set).
Specifically, we'll be looking at the section using the Boston Housing regression dataset, adapting
the target transformations therein to be used with HyperparameterHunter"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, FeatureEngineer
from hyperparameter_hunter import DummySearch, Categorical
from hyperparameter_hunter.utils.learning_utils import get_boston_data

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np

##################################################
# Import Learning Assets
##################################################
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler


# noinspection PyUnusedLocal
def get_holdout_data(train, target_column):
    train_data, holdout_data = train_test_split(train, random_state=1)
    return train_data, holdout_data


##################################################
# Feature Engineering Steps
##################################################
def quantile_transform(train_targets, non_train_targets):
    transformer = QuantileTransformer(output_distribution="normal", n_quantiles=100)
    train_targets[train_targets.columns] = transformer.fit_transform(train_targets.values)
    non_train_targets[train_targets.columns] = transformer.transform(non_train_targets.values)
    return train_targets, non_train_targets, transformer


def log_transform(all_targets):
    all_targets = np.log1p(all_targets)
    return all_targets, np.expm1


def standard_scale(train_inputs, non_train_inputs):
    scaler = StandardScaler()
    train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    non_train_inputs[train_inputs.columns] = scaler.transform(non_train_inputs.values)
    return train_inputs, non_train_inputs


def standard_scale_BAD(all_inputs):
    """If you wanted to standard-scale, by fitting on your entire dataset, rather than only on your
    train dataset (which is not recommended), this is how you could do it"""
    scaler = StandardScaler()
    all_inputs[all_inputs.columns] = scaler.fit_transform(all_inputs.values)
    return all_inputs


def square_sum_feature(all_inputs):
    all_inputs["square_sum"] = all_inputs.agg(
        lambda row: np.sqrt(np.sum([np.square(_) for _ in row])), axis="columns"
    )
    return all_inputs


##################################################
# Execute
##################################################
def execute():
    #################### Environment ####################
    env = Environment(
        train_dataset=get_boston_data(),
        results_path="HyperparameterHunterAssets",
        holdout_dataset=get_holdout_data,
        target_column="DIS",
        metrics=["r2_score", "median_absolute_error"],
        cv_type="KFold",
        cv_params=dict(n_splits=10, random_state=1),
    )

    #################### CVExperiment ####################
    exp_0 = CVExperiment(
        model_initializer=Ridge,
        model_init_params=dict(),
        feature_engineer=FeatureEngineer([quantile_transform]),
    )

    #################### Optimization ####################
    # `opt_0` recognizes `exp_0`'s `feature_engineer` and its results as valid learning material
    # This is because `opt_0` marks the engineer step functions omitted by `exp_0` as `optional=True`
    opt_0 = DummySearch(iterations=10)
    opt_0.set_experiment_guidelines(
        model_initializer=Ridge,
        model_init_params=dict(),
        feature_engineer=FeatureEngineer(
            [
                Categorical([quantile_transform, log_transform], optional=True),
                Categorical([standard_scale, standard_scale_BAD], optional=True),
                Categorical([square_sum_feature], optional=True),
            ]
        ),
    )
    opt_0.go()


if __name__ == "__main__":
    execute()
