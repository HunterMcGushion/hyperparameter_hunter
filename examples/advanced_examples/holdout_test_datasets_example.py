from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def get_holdout_set(train, target_column):
    """This is a sample callable to demonstrate how the Environment's `holdout_dataset` is evaluated. If you do provide a
    callable, it should expect two inputs: the train_dataset (pandas.DataFrame), and the target_column name (string). You should
    return two DataFrames: a modified train_dataset, and a holdout_dataset. What happens in between is up to you, perhaps split
    apart a portion of the rows, but the idea is to remove part of train_dataset, and turn it into holdout_dataset. For this
    example, we'll just copy train_dataset, which is a VERY BAD IDEA in practice. Don't actually do this"""
    return train, train.copy()


def execute():
    env = Environment(
        train_dataset=get_toy_classification_data(),
        results_path="HyperparameterHunterAssets",
        # Both `holdout_dataset`, and `train_dataset` can be any of the following: pandas.DataFrame, filepath, or None
        # If a filepath is provided, it will be passed to :meth:`pandas.read_csv`.
        # In addition to the above types, `holdout_dataset` can also be provided as a callable (see above :func:`get_holdout_set`)
        holdout_dataset=get_holdout_set,
        test_dataset=get_toy_classification_data(),
        # By default, `holdout_dataset` will be scored with the provided metrics_map, just like OOF predictions
        # However, you can provide the additional `metrics_params` kwarg to specify which metrics are calculated for each dataset
        # See the documentation in :class:`environment.Environment` and :class:`metrics.ScoringMixIn` for more information
        metrics_map=["roc_auc_score"],
        cross_validation_type=StratifiedKFold,
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    experiment = CVExperiment(
        model_initializer=XGBClassifier, model_init_params=dict(subsample=0.5)
    )
    # At the end of the Experiment, notice a few differences from the results of an Experiment given only training data:
    # 1) A "PredictionsHoldout" directory is created to house holdout predictions for Experiments given holdout data,
    # 2) A "PredictionsTest" directory is created to house test predictions for Experiments given test data,
    # 3) The Experiment's "Description" file will describe the evaluation of the holdout data, just like the OOF data,
    # 4) Leaderboards are modified to accommodate new holdout metrics evaluations, and
    # 5) New directories are created in "KeyAttributeLookup" for holdout and test datasets
    # The new "KeyAttributeLookup" entries serve to ensure the same datasets are used, and improper comparisons aren't made


if __name__ == "__main__":
    execute()
