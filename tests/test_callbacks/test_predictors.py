##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter.callbacks.bases import BaseCallback
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data

##################################################
# Import Miscellaneous Assets
##################################################
from copy import deepcopy
import pandas as pd
import pytest

try:
    xgboost = pytest.importorskip("xgboost")
except Exception:
    raise

##################################################
# Import Learning Assets
##################################################
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


class DummyExperimentPredictorHoldout(BaseCallback):
    def __init__(self, data_holdout, feature_engineer, experiment_params, cv_params):
        """This is never actually called. It's just here to document the attributes being used"""
        self.data_holdout = data_holdout
        self.feature_engineer = feature_engineer
        self.experiment_params = experiment_params
        self.cv_params = cv_params

        self.__assert_prediction_is_none(["final", "rep", "fold", "run"])

    def on_exp_start(self):
        super().on_exp_start()

        self.__assert_prediction_is_zero(["final"])
        self.__assert_prediction_is_none(["rep", "fold", "run"])

    def on_rep_start(self):
        super().on_rep_start()

        self.__assert_about_prediction(["final"], lambda _: _ is not None)
        self.__assert_prediction_is_zero(["rep"])
        self.__assert_prediction_is_none(["fold", "run"])

    def on_fold_start(self):
        super().on_fold_start()

        self.__assert_about_prediction(["final", "rep"], lambda _: _ is not None)
        self.__assert_prediction_is_zero(["fold"])

    def on_run_end(self):
        initial_fold_prediction = deepcopy(self.data_holdout.prediction.fold)

        super().on_run_end()

        self.__assert_about_prediction(["final", "rep"], lambda _: _ is not None)
        self.__assert_prediction_is_df(["fold", "run"])
        expected_fold_prediction = initial_fold_prediction + self.data_holdout.prediction.run
        assert self.data_holdout.prediction.fold.equals(expected_fold_prediction)

    def on_fold_end(self):
        initial_fold_prediction = deepcopy(self.data_holdout.prediction.fold)
        initial_rep_prediction = deepcopy(self.data_holdout.prediction.rep)

        super().on_fold_end()

        self.__assert_about_prediction(["final"], lambda _: _ is not None)
        self.__assert_prediction_is_df(["rep", "fold", "run"])
        expected_fold_prediction = initial_fold_prediction / self.experiment_params["runs"]
        expected_rep_prediction = initial_rep_prediction + self.data_holdout.prediction.fold
        assert self.data_holdout.prediction.fold.equals(expected_fold_prediction)
        assert self.data_holdout.prediction.rep.equals(expected_rep_prediction)

    def on_rep_end(self):
        initial_rep_prediction = deepcopy(self.data_holdout.prediction.rep)
        initial_final_prediction = deepcopy(self.data_holdout.prediction.final)

        super().on_rep_end()

        self.__assert_prediction_is_df(["final", "rep", "fold", "run"])
        expected_rep_prediction = initial_rep_prediction / self.cv_params["n_splits"]
        expected_final_prediction = initial_final_prediction + self.data_holdout.prediction.rep
        assert self.data_holdout.prediction.rep.equals(expected_rep_prediction)
        assert self.data_holdout.prediction.final.equals(expected_final_prediction)

    def on_exp_end(self):
        initial_final_prediction = deepcopy(self.data_holdout.prediction.final)

        super().on_exp_end()

        self.__assert_prediction_is_df(["final", "rep", "fold", "run"])
        expected_final_prediction = initial_final_prediction / self.cv_params.get("n_repeats", 1)
        assert self.data_holdout.prediction.final.equals(expected_final_prediction)

    ##################################################
    # Helper Test Methods
    ##################################################
    def __assert_about_prediction(self, divisions, assertion):
        for division in divisions:
            assert assertion(getattr(self.data_holdout.prediction, division))

    def __assert_prediction_is_none(self, divisions):
        for division in divisions:
            assert getattr(self.data_holdout.prediction, division) is None

    def __assert_prediction_is_zero(self, divisions):
        for division in divisions:
            assert getattr(self.data_holdout.prediction, division) == 0

    def __assert_prediction_is_df(self, divisions):
        for division in divisions:
            assert isinstance(getattr(self.data_holdout.prediction, division), pd.DataFrame)


def get_iris_data(target="species"):
    data = load_iris()
    df = pd.DataFrame(
        data=data.data,
        columns=[_.replace(" ", "_") for _ in data.feature_names],
        index=[f"s_{str(_).rjust(3, '_')}_s" for _ in range(len(data.data))],
    )
    df[target] = data.target
    #################### Label-Encode Targets ####################
    label_encoder = LabelEncoder()
    df.loc[:, target] = label_encoder.fit_transform(df.loc[:, target])
    return df


def get_holdout(train_dataset, target_column):
    return train_test_split(
        train_dataset, test_size=0.33, random_state=1, stratify=train_dataset.loc[:, target_column]
    )


def test_predictor_holdout_iris():
    G.priority_callbacks = (DummyExperimentPredictorHoldout,)

    #################### Set Up Environment ####################
    env = Environment(
        train_dataset=get_iris_data(),
        results_path=assets_dir,
        holdout_dataset=get_holdout,
        target_column="species",
        metrics=dict(f1=lambda t, p: f1_score(t, p, average="micro"), hamming_loss="hamming_loss"),
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    #################### Perform Experiment ####################
    experiment = CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            objective="multi:softprob",
            max_depth=1,
            n_estimators=300,
            learning_rate=0.02,
            min_child_weight=6,
            gamma=0.07,
            colsample_bytree=0.31,
        ),
        model_extra_params=dict(
            fit=dict(
                eval_set=[
                    (env.train_input, env.train_target),
                    (env.validation_input, env.validation_target),
                ],
                early_stopping_rounds=20,
                eval_metric="merror",
            )
        ),
    )

    G.priority_callbacks = tuple()


def test_predictor_holdout_breast_cancer():
    G.priority_callbacks = (DummyExperimentPredictorHoldout,)

    #################### Set Up Environment ####################
    env = Environment(
        train_dataset=get_breast_cancer_data(),
        results_path=assets_dir,
        holdout_dataset=get_holdout,
        target_column="diagnosis",
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    #################### Perform Experiment ####################
    experiment = CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            objective="reg:linear",
            max_depth=3,
            n_estimators=100,
            learning_rate=0.02,
            min_child_weight=6,
            gamma=0.07,
            colsample_bytree=0.31,
        ),
        model_extra_params=dict(
            fit=dict(
                eval_set=[
                    (env.train_input, env.train_target),
                    (env.validation_input, env.validation_target),
                ],
                early_stopping_rounds=5,
                eval_metric="mae",
            )
        ),
    )

    G.priority_callbacks = tuple()
