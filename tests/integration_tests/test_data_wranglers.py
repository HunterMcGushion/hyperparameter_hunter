##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, FeatureEngineer
from hyperparameter_hunter.callbacks.recipes import lambda_check_train_targets
from hyperparameter_hunter.data.data_chunks.target_chunks import TrainTargetChunk

##################################################
# Import Miscellaneous Assets
##################################################
from copy import deepcopy
import pandas as pd
import pytest

##################################################
# Import Learning Assets
##################################################
# from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split  # , PredefinedSplit
from sklearn.preprocessing import LabelEncoder  # , OneHotEncoder
from sklearn.svm import SVC

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


##################################################
# Define Data
##################################################
def get_iris_sample():
    """Manageable subset of Iris classification dataset, comprising three samples for each of the
    three classes - nine rows in all"""
    sample_df = pd.DataFrame(
        data=[
            [5.1, 3.5, 1.4, 0.2, "setosa"],  # 0
            [7.0, 3.2, 4.7, 1.4, "versicolor"],  # 1
            [6.3, 3.3, 6.0, 2.5, "virginica"],  # 2
            [4.9, 3.0, 1.4, 0.2, "setosa"],  # 3
            [6.4, 3.2, 4.5, 1.5, "versicolor"],  # 4
            [5.8, 2.7, 5.1, 1.9, "virginica"],  # 5
            [4.7, 3.2, 1.3, 0.2, "setosa"],  # 6
            [6.9, 3.1, 4.9, 1.5, "versicolor"],  # 7
            [7.1, 3.0, 5.9, 2.1, "virginica"],  # 8
        ],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    )
    return sample_df


# noinspection PyUnusedLocal
def get_holdout(data, target_column):
    return train_test_split(data, test_size=0.33, random_state=1, shuffle=False)


##################################################
# Feature Engineering Functions
##################################################
def label_encode_targets(all_targets):
    label_encoder = LabelEncoder()
    all_targets.loc[:, "species"] = label_encoder.fit_transform(all_targets.loc[:, "species"])
    # return all_targets, label_encoder  # TODO: Test support
    return all_targets


##################################################
# Fixtures
##################################################
@pytest.fixture
def mini_iris_env(request):
    """`Environment` fixture that supports provision of a `holdout_dataset`/`test_dataset` `request`

    Parameters
    ----------
    request: Object
        If `request` has a "param" attribute, it must be a dict, containing any, or none of the
        keys {"holdout", "test", "callbacks"}, whose values will be passed as the `Environment`'s
        `holdout_dataset`, `test_dataset` and `experiment_callbacks` parameters, respectively"""
    params = getattr(request, "param", dict())
    return Environment(
        train_dataset=get_iris_sample(),
        holdout_dataset=params.get("holdout", None),
        test_dataset=params.get("test", None),
        results_path=assets_dir,
        metrics=dict(f1=lambda t, p: f1_score(t, p, average="micro")),
        target_column="species",
        cv_params=dict(n_splits=3, shuffle=True, random_state=32),
        experiment_callbacks=params.get("callbacks", []),
    )


@pytest.fixture
def engineer_experiment(request):
    """`CVExperiment` fixture that supports provision of a `feature_engineer` through `request`

    Parameters
    ----------
    request: Object
        If `request` has a "param" attribute, it must be a list of feature engineering steps to
        provide to :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`"""
    feature_engineer = FeatureEngineer(steps=getattr(request, "param", None))
    experiment = CVExperiment(
        model_initializer=SVC, model_init_params=dict(), feature_engineer=feature_engineer
    )
    return experiment


##################################################
# Smoke-ish Tests
##################################################
@pytest.mark.parametrize("mini_iris_env", [dict(holdout=get_holdout)], indirect=True)
def test_invalid_targets(mini_iris_env):
    """Verify that the experiment breaks when attempting to deal with untransformed string labels"""
    with pytest.raises(ValueError):
        exp = CVExperiment(model_initializer=SVC, model_init_params=dict(), feature_engineer=None)


@pytest.mark.parametrize(
    ["mini_iris_env", "engineer_experiment"],
    [
        [dict(), [label_encode_targets]],
        [dict(holdout=get_holdout), [label_encode_targets]],
        [dict(test=get_iris_sample()), [label_encode_targets]],
        [dict(holdout=get_holdout, test=get_iris_sample()), [label_encode_targets]],
    ],
    indirect=True,
)
def test_extra_datasets_smoke(mini_iris_env, engineer_experiment):
    """Verify that `engineer_experiment` doesn't catch fire when given holdout/test datasets"""
    ...


def lambda_check_train_targets_fold_start():
    #################### Base Chunk ####################
    initial_target_chunk = TrainTargetChunk(
        pd.DataFrame(
            dict(
                species=[
                    "setosa",
                    "versicolor",
                    "virginica",
                    "setosa",
                    "versicolor",
                    "virginica",
                    "setosa",
                    "versicolor",
                    "virginica",
                ]
            )
        )
    )
    initial_target_chunk.T.d = pd.DataFrame(dict(species=[0, 1, 2, 0, 1, 2, 0, 1, 2]))

    #################### Fold 0 Chunk ####################
    fold_0_chunk = deepcopy(initial_target_chunk)
    fold_0_chunk.fold = pd.DataFrame(
        dict(species=["setosa", "setosa", "versicolor", "virginica", "versicolor", "virginica"]),
        index=[0, 3, 4, 5, 7, 8],  # OOF: 1, 2, 6
    )
    fold_0_chunk.T.fold = pd.DataFrame(dict(species=[0, 0, 1, 2, 1, 2]), index=[0, 3, 4, 5, 7, 8])

    #################### Fold 1 Chunk ####################
    fold_1_chunk = deepcopy(initial_target_chunk)
    fold_1_chunk.fold = pd.DataFrame(
        dict(species=["versicolor", "virginica", "setosa", "virginica", "setosa", "versicolor"]),
        index=[1, 2, 3, 5, 6, 7],  # OOF: 0, 4, 8
    )
    fold_1_chunk.T.fold = pd.DataFrame(dict(species=[1, 2, 0, 2, 0, 1]), index=[1, 2, 3, 5, 6, 7])

    #################### Fold 2 Chunk ####################
    fold_2_chunk = deepcopy(initial_target_chunk)
    fold_2_chunk.fold = pd.DataFrame(
        dict(species=["setosa", "versicolor", "virginica", "versicolor", "setosa", "virginica"]),
        index=[0, 1, 2, 4, 6, 8],  # OOF: 3, 5, 7
    )
    fold_2_chunk.T.fold = pd.DataFrame(dict(species=[0, 1, 2, 1, 0, 2]), index=[0, 1, 2, 4, 6, 8])

    return lambda_check_train_targets(on_fold_start=[fold_0_chunk, fold_1_chunk, fold_2_chunk])


@pytest.mark.parametrize(
    ["mini_iris_env", "engineer_experiment"],
    [[dict(callbacks=[lambda_check_train_targets_fold_start()]), [label_encode_targets]]],
    indirect=True,
)
def test_train_targets(mini_iris_env, engineer_experiment):
    ...
