"""This example is a classic shotgun approach to hyperparameter optimization. As the script's name
suggests, you'll need to install the wonderful `imblearn` library. The only "problem" with
`imblearn` is that they have way too many fascinating and useful re-sampling techniques! How could
we ever choose one, and just call our search over? Let's not do that. Instead, we'll choose 18 and
let HyperparameterHunter figure out the best `imblearn` tool for this problem.

All of the 18 engineer step functions follow the same pattern and use the `_sampler_helper` function
defined at the top, so once you've seen one of them, you've pretty much seen them all. Then just
scroll all the way to the bottom of the script to see the actual optimization part!"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, ET, FeatureEngineer, EngineerStep
from hyperparameter_hunter import Real, Integer, Categorical

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd

##################################################
# Import Learning Assets
##################################################
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import (
    ClusterCentroids,
    RandomUnderSampler,
    TomekLinks,
    NearMiss,
    CondensedNearestNeighbour,
    OneSidedSelection,
    NeighbourhoodCleaningRule,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    InstanceHardnessThreshold,
)
from sklearn.datasets import make_classification
from xgboost import XGBClassifier


##################################################
# Helper Function for `imblearn` `EngineerStep`s
##################################################
def _sampler_helper(sampler, train_inputs, train_targets):
    _train_inputs, _train_targets = sampler.fit_resample(train_inputs, train_targets)
    train_inputs = pd.DataFrame(_train_inputs, columns=train_inputs.columns)
    train_targets = pd.DataFrame(_train_targets, columns=train_targets.columns)
    return train_inputs, train_targets


##################################################
# GROUP 1
##################################################
def resample_smote_tomek(train_inputs, train_targets):
    sampler = SMOTETomek(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def over_sample_random(train_inputs, train_targets):
    sampler = RandomOverSampler(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def over_sample_smote(train_inputs, train_targets):
    sampler = SMOTE(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_random(train_inputs, train_targets):
    sampler = RandomUnderSampler(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_cluster_centroids(train_inputs, train_targets):
    sampler = ClusterCentroids(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_tomek_links(train_inputs, train_targets):
    sampler = TomekLinks(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


##################################################
# GROUP 2 (EXTENDED)
##################################################
def resample_smote_enn(train_inputs, train_targets):
    sampler = SMOTEENN(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def over_sample_ADASYN(train_inputs, train_targets):
    sampler = ADASYN(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def over_sample_BorderlineSMOTE(train_inputs, train_targets):
    sampler = BorderlineSMOTE(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def over_sample_SVMSMOTE(train_inputs, train_targets):
    sampler = SVMSMOTE(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_NearMiss(train_inputs, train_targets):
    sampler = NearMiss(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_CondensedNearestNeighbour(train_inputs, train_targets):
    sampler = CondensedNearestNeighbour(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_OneSidedSelection(train_inputs, train_targets):
    sampler = OneSidedSelection(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_NeighbourhoodCleaningRule(train_inputs, train_targets):
    sampler = NeighbourhoodCleaningRule(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_EditedNearestNeighbours(train_inputs, train_targets):
    sampler = EditedNearestNeighbours(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_RepeatedEditedNearestNeighbour(train_inputs, train_targets):
    sampler = RepeatedEditedNearestNeighbours(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_AllKNN(train_inputs, train_targets):
    sampler = AllKNN(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def under_sample_InstanceHardnessThreshold(train_inputs, train_targets):
    sampler = InstanceHardnessThreshold(random_state=32)
    train_inputs, train_targets = _sampler_helper(sampler, train_inputs, train_targets)
    return train_inputs, train_targets


def get_imbalanced_dataset():
    X, y = make_classification(
        n_classes=2,
        class_sep=1.5,
        weights=[0.9, 0.1],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=1000,
        random_state=10,
    )
    df = pd.DataFrame(X)
    df["target"] = y
    return df


##################################################
# The Actual Optimization Part
##################################################
def execute():
    env = Environment(
        train_dataset=get_imbalanced_dataset(),
        results_path="HyperparameterHunterAssets",
        target_column="target",
        metrics=["roc_auc_score", "accuracy_score"],
        cv_type="KFold",
        cv_params=dict(n_splits=5, random_state=7),
    )

    # Since this is HyperparameterHunter, after all, we'll throw in some classic hyperparameter
    #   optimization just for fun. If you're like most people and you think it's absurd to test
    #   18 different `imblearn` techniques, feel free to comment out some `EngineerStep`s below

    opt_0 = ET(iterations=20, random_state=32)
    opt_0.set_experiment_guidelines(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            max_depth=Integer(2, 20),
            n_estimators=Integer(50, 900),
            learning_rate=Real(0.0001, 0.9),
            subsample=0.5,
            booster=Categorical(["gbtree", "gblinear"]),
        ),
        feature_engineer=FeatureEngineer(
            [
                Categorical(
                    [
                        EngineerStep(resample_smote_tomek, stage="intra_cv"),
                        EngineerStep(over_sample_random, stage="intra_cv"),
                        EngineerStep(over_sample_smote, stage="intra_cv"),
                        EngineerStep(under_sample_random, stage="intra_cv"),
                        EngineerStep(under_sample_cluster_centroids, stage="intra_cv"),
                        EngineerStep(under_sample_tomek_links, stage="intra_cv"),
                        #################### GROUP 2 (EXTENDED) ####################
                        EngineerStep(resample_smote_enn, stage="intra_cv"),
                        EngineerStep(over_sample_ADASYN, stage="intra_cv"),
                        EngineerStep(over_sample_BorderlineSMOTE, stage="intra_cv"),
                        EngineerStep(over_sample_SVMSMOTE, stage="intra_cv"),
                        EngineerStep(under_sample_NearMiss, stage="intra_cv"),
                        EngineerStep(under_sample_CondensedNearestNeighbour, stage="intra_cv"),
                        EngineerStep(under_sample_OneSidedSelection, stage="intra_cv"),
                        EngineerStep(under_sample_NeighbourhoodCleaningRule, stage="intra_cv"),
                        EngineerStep(under_sample_EditedNearestNeighbours, stage="intra_cv"),
                        EngineerStep(under_sample_RepeatedEditedNearestNeighbour, stage="intra_cv"),
                        EngineerStep(under_sample_AllKNN, stage="intra_cv"),
                        EngineerStep(under_sample_InstanceHardnessThreshold, stage="intra_cv"),
                    ],
                    optional=True,
                )
            ]
        ),
    )
    opt_0.go()

    # If you're like me and you think this is just too much fun to do only once, run the script a
    #   few times in a row, and watch HyperparameterHunter pile up the results from previous runs


if __name__ == "__main__":
    execute()
